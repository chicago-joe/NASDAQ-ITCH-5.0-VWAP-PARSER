###########################################################################
# Parsing NASDAQ ITCH-5.0 Trade Data
#
#   Created by Joseph Loss on 09/01/2019
#   MS Financial Engineering
#   University of Illinois, Urbana-Champaign
#
#   contact: loss2@illinois.edu
###########################################################################
import gzip
import shutil
from urllib.request import urlretrieve
import urllib.parse
from clint.textui import progress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
from collections import namedtuple, Counter
from datetime import datetime as dt
from datetime import timedelta, time, datetime
from matplotlib.ticker import FuncFormatter
from math import pi
from bokeh.plotting import figure, show, output_file, output_notebook
from time import time
import seaborn as sns
from pathlib import Path
from collections import Counter
import matplotlib as mpl
from scipy.stats import normaltest
import os


# store data download in a subdirectory and convert the result to hdf format (for faster read/write)
data_path = Path('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//data')
figure_path = Path('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//figures')
itch_store = data_path / 'itch.h5'
order_book_store = data_path / 'order_book.h5'

# give FTP address, filename, and the date of the file you want to download:
FTP_URL = 'ftp://emi.nasdaq.com/ITCH/'
SOURCE_FILE = '10302019.NASDAQ_ITCH50.gz'


# download the data and unzip it
def may_be_download(url):
    """Download & unzip ITCH data if not yet available"""
    filename = data_path / url.split('/')[-1]
    if not data_path.exists():
        print('Creating directory')
        data_path.mkdir()
    if not filename.exists():
        print('Downloading...', url)
        urlretrieve(url, filename)
    unzipped = data_path / (filename.stem + '.bin')
    # if not (data_path / unzipped).exists():
    if not (unzipped).exists():
        print('Unzipping to', unzipped)
        with gzip.open(str(filename), 'rb') as f_in:
            with open(unzipped, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return unzipped


file_name = may_be_download(urllib.parse.urljoin(FTP_URL, SOURCE_FILE))
date = file_name.name.split('.')[0]


# put parser format strings (given in ITCH5.0 pdf) into format dictionaries within class:
class format_strings:
    event_codes = { 'O':'Start of Messages',
                    'S':'Start of System Hours',
                    'Q':'Start of Market Hours',
                    'M':'End of Market Hours',
                    'E':'End of System Hours',
                    'C':'End of Messages' }
    encoding = { 'primary_market_maker':{ 'Y':1, 'N':0 },
                 'printable':           { 'Y':1, 'N':0 },
                 'buy_sell_indicator':  { 'B':1, 'S':-1 },
                 'cross_type':          { 'O':0, 'C':1, 'H':2 },
                 'imbalance_direction': { 'B':0, 'S':1, 'N':0, 'O':-1 } }
    formats = {
            ('integer', 2):'H',
            ('integer', 4):'I',
            ('integer', 6):'6s',
            ('integer', 8):'Q',
            ('alpha', 1):  's',
            ('alpha', 2):  '2s',
            ('alpha', 4):  '4s',
            ('alpha', 8):  '8s',
            ('price_4', 4):'I',
            ('price_8', 8):'Q',
    }


# the message_types.xlxs contains type specifications as laid out by ITCH documentation:
message_data = (pd.read_excel('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//src//message_types.xlsx',
                              sheet_name = 'messages',
                              encoding = 'latin1').sort_values('id').drop('id', axis = 1))


# basic string cleaning:
def clean_message_types(df):
    df.columns = [col.lower().strip() for col in df.columns]
    df.value = df.value.str.strip()
    df.name = (df.name
               .str.strip()  # remove whitespace
               .str.lower()
               .str.replace(' ', '_')
               .str.replace('-', '_')
               .str.replace('/', '_'))
    df.notes = df.notes.str.strip()
    df['message_type'] = df.loc[df.name == 'message_type', 'value']
    return df


# read message types from xlsx and run string cleaning function
message_types = clean_message_types(message_data)

# extract message type codes/names to make results more readable
message_labels = (message_types.loc[:, ['message_type', 'notes']]
                  .dropna()
                  .rename(columns = { 'notes':'name' }))
message_labels.name = (message_labels.name
                       .str.lower()
                       .str.replace('message', '')
                       .str.replace('.', '')
                       .str.strip().str.replace(' ', '_'))

print("\nMessage Types and Description:\n", message_labels.head(), "\n")

# finalize msg specs: offset, length, and value type to be used by struct
message_types.message_type = message_types.message_type.ffill()
message_types = message_types[message_types.name != 'message_type']
message_types.value = (message_types.value
                       .str.lower()
                       .str.replace(' ', '_')
                       .str.replace('(', '')
                       .str.replace(')', ''))

# print('\n')
# print(message_types.info())
print("\nMessage Types Header: \n", message_types.head(3), "\n")

# OPTIMIZE:
# csv_path = Path('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//')
# auto-persist from file:
message_types.to_csv('message_types.csv', index = False)
message_types = pd.read_csv('message_types.csv')

# parser translates the message specs into format strings and tuples that capture msg content
# so we create formatting tuples from ITCH specifications:
message_types.loc[:, 'formats'] = (message_types[['value', 'length']].apply(tuple, axis = 1).map(format_strings.formats))

# and extract formatting details for alphanumeric fields
alpha_fields = message_types[message_types.value == 'alpha'].set_index('name')
alpha_msgs = alpha_fields.groupby('message_type')
alpha_formats = { k:v.to_dict() for k, v in alpha_msgs.formats }  # use formats dictionary for fields marked "alpha"
alpha_length = { k:v.add(5).to_dict() for k, v in alpha_msgs.length }

# generate msg classes as named tuples and format strings
message_fields, fstring = { }, { }
for t, message in message_types.groupby('message_type'):
    message_fields[t] = namedtuple(typename = t, field_names = message.name.tolist())
    fstring[t] = '>' + ''.join(message.formats.tolist())


# post-processing for alphanumeric fields
def format_alphanumeric(mtype, data):
    for col in alpha_formats.get(mtype).keys():
        if mtype != 'R' and col == 'stock':  # save memory, we already have stock locator so we can delete the stock ticker name
            data = data.drop(col, axis = 1)
            continue
        data.loc[:, col] = data.loc[:, col].str.decode("utf-8").str.strip()
        if format_strings.encoding.get(col):
            data.loc[:, col] = data.loc[:, col].map(format_strings.encoding.get(col))
    return data


def store_messages(m):
    """Handle occasional storing of all messages"""
    with pd.HDFStore(itch_store) as store:
        for mtype, data in m.items():
            data = pd.DataFrame(data)  # convert to DataFrame

            data.timestamp = data.timestamp.apply(int.from_bytes, byteorder = 'big')
            data.timestamp = pd.to_timedelta(data.timestamp)

            if mtype in alpha_formats.keys():  # apply alpha formatting
                data = format_alphanumeric(mtype, data)

            s = alpha_length.get(mtype)
            if s:
                s = { c:s.get(c) for c in data.columns }
            dc = ['stock_locate']
            if m == 'R':
                dc.append('stock')
            store.append(mtype, data,
                         format = 't',
                         min_itemsize = s,
                         data_columns = dc)

#
# if os.path.exists(itch_store):
#     pass
# else:
messages = { }
message_count = 0
message_type_counter = Counter()
# simplified version to process the bin file and produce the parsed orders by msg type:
# if itch_store.exists()
start = time()
print("\n=====  INITALIZING PARSER  ===== ")
with file_name.open('rb') as data:
    while True:
        # determine msg size in bytes
        message_size = int.from_bytes(data.read(2), byteorder = 'big', signed = False)

        # get msg type by reading first byte
        message_type = data.read(1).decode('ascii')

        # create data structure to capture result
        if not messages.get(message_type):
            messages[message_type] = []

        message_type_counter.update([message_type])

        # read & store msg
        record = data.read(message_size - 1)
        message = message_fields[message_type]._make(unpack(fstring[message_type], record))
        messages[message_type].append(message)

        # system event handler:
        if message_type == 'S':
            timestamp = int.from_bytes(message.timestamp, byteorder = 'big')
            print(format_strings.event_codes.get(message.event_code.decode('ascii'), 'Error'))
            print('\t{0}\t{1:,.0f}'.format(timedelta(seconds = timestamp * 1e-9), message_count))
            if message.event_code.decode('ascii') == 'C':
                store_messages(messages)
                break

        message_count += 1
        if message_count % 2.5e7 == 0:
            timestamp = int.from_bytes(message.timestamp, byteorder = 'big')
            print('\t{0}\t{1:,.0f}\t{2}'.format(timedelta(seconds = timestamp * 1e-9),
                                                message_count,
                                                timedelta(seconds = time() - start)))
            store_messages(messages)
            messages = { }


print("PARSE FINISHED IN: ", timedelta(seconds = time() - start))

## build
# order book flow for the given day
print("\n----------------------------------------------")
stock = 'GE'
order_dict = { -1:'sell', 1:'buy' }

print("=====  RECONSTRUCTING ORDER FLOW  =====\nDATE:  ", date, "\nTICKER:  ", stock)


# get all messages for the chosen stock
def get_messages(date, stock = stock):
    with pd.HDFStore(itch_store) as store:
        stock_locate = store.select('R', where = 'stock = stock').stock_locate.iloc[0]
        target = 'stock_locate = stock_locate'

        data = { }
        trading_msgs = ['A', 'F', 'E', 'C', 'X', 'D', 'U', 'P', 'Q']
        for msg in trading_msgs:
            data[msg] = store.select(msg, where = target).drop('stock_locate', axis = 1).assign(type = msg)

        # public key records in each type of message (order_ref_number, stock locate code, etc)
        order_cols = ['order_reference_number', 'buy_sell_indicator', 'shares', 'price']

        # 'A' and 'F' message types are for Add Orders (with and without unattributed orders/quotes)
        orders = pd.concat([data['A'], data['F']], sort = False, ignore_index = True).loc[:, order_cols]

        for msg in trading_msgs[2: -3]:
            data[msg] = data[msg].merge(orders, how = 'left')

        # Msg for whenever an order on the book has been cancel-replaced
        data['U'] = data['U'].merge(orders, how = 'left',
                                    right_on = 'order_reference_number',
                                    left_on = 'original_order_reference_number',
                                    suffixes = ['', '_replaced'])

        # Cross Trade messages:
        data['Q'].rename(columns = { 'cross_price':'price' }, inplace = True)

        # Order Cancel Messages:
        data['X']['shares'] = data['X']['cancelled_shares']
        data['X'] = data['X'].dropna(subset = ['price'])

        data = pd.concat([data[msg] for msg in trading_msgs], ignore_index = True, sort = False)
        data['date'] = pd.to_datetime(date, format = '%m%d%Y')
        data.timestamp = data['date'].add(data.timestamp)
        data = data[data.printable != 0]

        drop_cols = ['tracking_number', 'order_reference_number', 'original_order_reference_number',
                     'cross_type', 'new_order_reference_number', 'attribution', 'match_number',
                     'printable', 'date', 'cancelled_shares']
        return data.drop(drop_cols, axis = 1).sort_values('timestamp').reset_index(drop = True)


# combine trade orders (reconstruct successful trades, ie orders that are executed)
def get_trades(msg):
    """Combine C, E, P and Q messages into trading records"""
    trade_dict = { 'executed_shares':'shares', 'execution_price':'price' }
    cols = ['timestamp', 'executed_shares']
    trades = pd.concat([msg.loc[msg.type == 'E', cols + ['price']].rename(columns = trade_dict),
                        msg.loc[msg.type == 'C', cols + ['execution_price']].rename(columns = trade_dict),
                        msg.loc[msg.type == 'P', ['timestamp', 'price', 'shares']],
                        msg.loc[msg.type == 'Q', ['timestamp', 'price', 'shares']].assign(cross = 1),
                        ], sort = False).dropna(subset = ['price']).fillna(0)
    return trades.set_index('timestamp').sort_index().astype(int)


# create orders - accumulate sell orders in ascending and buy orders in desc. order for given timestamps
def add_orders(orders, buysell, nlevels):
    new_order = []
    items = sorted(orders.copy().items())
    if buysell == 1:
        items = reversed(items)
    for i, (p, s) in enumerate(items, 1):
        new_order.append((p, s))
        if i == nlevels:
            break
    return orders, new_order


# save orders
def save_orders(orders, append = False):
    cols = ['price', 'shares']
    for buysell, book in orders.items():
        df = (pd.concat([pd.DataFrame(data = data, columns = cols).assign(timestamp = t)
                         for t, data in book.items()]))
        key = '{}/{}'.format(stock, order_dict[buysell])
        df.loc[:, ['price', 'shares']] = df.loc[:, ['price', 'shares']].astype(int)
        with pd.HDFStore(order_book_store) as store:
            if append:
                store.append(key, df.set_index('timestamp'), format = 't')
            else:
                store.put(key, df.set_index('timestamp'))


## iterate over all ITCH msgs to process orders/replacement orders as specified:

# if os.path.exists(order_book_store):
#     pass
# else:




# print msg info
messages = get_messages(date = date)
print("\n =====  ORDER MESSAGES SUMMARY:  =====\n")
messages.info(null_counts = True)

with pd.HDFStore(order_book_store) as store:
    key = '{}/trading_msgs'.format(stock)
    store.put(key, messages)
    print("\n =====  ORDERBOOK STORAGE FORMAT:  =====\n ", store.info(), "\n")






trades = get_trades(messages)
# print(trades.info())

with pd.HDFStore(order_book_store) as store:
    store.put('{}/trades'.format(stock), trades)




order_book = { -1:{ }, 1:{ } }
current_orders = { -1:Counter(), 1:Counter() }
message_counter = Counter()
nlevels = 100
start = time()
for msg in messages.itertuples():
    i = msg[0]
    if i % 1e5 == 0 and i > 0:
        print('{:,.0f}\t\t{}'.format(i, timedelta(seconds = time() - start)))
        save_orders(order_book, append = True)
        order_book = { -1:{ }, 1:{ } }
        start = time()
    if np.isnan(msg.buy_sell_indicator):
        continue
    message_counter.update(msg.type)

    buysell = msg.buy_sell_indicator
    price, shares = None, None

    if msg.type in ['A', 'F', 'U']:
        price = int(msg.price)
        shares = int(msg.shares)
        current_orders[buysell].update({ price:shares })
        current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
        order_book[buysell][msg.timestamp] = new_order

    if msg.type in ['E', 'C', 'X', 'D', 'U']:
        if msg.type == 'U':
            if not np.isnan(msg.shares_replaced):
                price = int(msg.price_replaced)
                shares = -int(msg.shares_replaced)
        else:
            if not np.isnan(msg.price):
                price = int(msg.price)
                shares = -int(msg.shares)
        if price is not None:
            current_orders[buysell].update({ price:shares })
            if current_orders[buysell][price] <= 0:
                current_orders[buysell].pop(price)
            current_orders[buysell], new_order = add_orders(current_orders[buysell], buysell, nlevels)
            order_book[buysell][msg.timestamp] = new_order

# print count of msgs
message_counter = pd.Series(message_counter)
print("\nCount per Message Type: \n", message_counter)

with pd.HDFStore(order_book_store) as store:
    print("\nOrderBook Storage Format (.h5 file): \n", store.info())

##  order book depth
with pd.HDFStore(order_book_store) as store:
    buy = store['{}/buy'.format(stock)].reset_index().drop_duplicates()
    sell = store['{}/sell'.format(stock)].reset_index().drop_duplicates()

# convert price to decimals
buy.price = buy.price.mul(1e-4)
sell.price = sell.price.mul(1e-4)

# remove outliers from book data
percentiles = [.01, .02, .1, .25, .75, .9, .98, .99]
pd.concat([buy.price.describe(percentiles = percentiles).to_frame('buy'),
           sell.price.describe(percentiles = percentiles).to_frame('sell')], axis = 1)
buy = buy[buy.price > buy.price.quantile(.01)]
sell = sell[sell.price < sell.price.quantile(.99)]

# ## ANALYZE BUY-SELL ORDER DISTRIBUTION
market_open = '0930'
market_close = '1600'

fig, ax = plt.subplots(figsize = (7, 5))
hist_kws = { 'linewidth':1, 'alpha':.5 }
sns.distplot(buy.set_index('timestamp').between_time(market_open, market_close).price, ax = ax, label = 'Buy', kde = False,
             hist_kws = hist_kws)
sns.distplot(sell.set_index('timestamp').between_time(market_open, market_close).price, ax = ax, label = 'Sell', kde = False,
             hist_kws = hist_kws)

# formatting
plt.legend(fontsize = 10)
plt.title('Limit Order Price Distribution', fontsize = 14)
ax.set_yticklabels(['{:,}'.format(int(y / 1000)) for y in ax.get_yticks().tolist()])
ax.set_xticklabels(['${:,}'.format(int(x)) for x in ax.get_xticks().tolist()])
plt.xlabel('Price', fontsize = 12)
plt.ylabel('Shares (\'000)', fontsize = 12)
plt.tight_layout()
plt.savefig(figure_path / 'price_distribution.png', dpi = 300)
plt.show()
plt.close()

## ANALYZE ORDER BOOK DEPTH
# utc_offset = timedelta(hours=4)
depth = 50
buy_per_min = (buy.groupby([pd.Grouper(key = 'timestamp', freq = 'Min'), 'price'])
               .shares
               .sum()
               .apply(np.log)
               .to_frame('shares')
               .reset_index('price')
               .between_time(market_open, market_close)
               .groupby(level = 'timestamp', as_index = False, group_keys = False)
               .apply(lambda x:x.nlargest(columns = 'price', n = depth))
               .reset_index())
buy_per_min.timestamp = buy_per_min.timestamp.values.astype(int)
# buy_per_min.timestamp = buy_per_min.timestamp.add(utc_offset).values.astype(int)
buy_per_min.info()

sell_per_min = (sell.groupby([pd.Grouper(key = 'timestamp', freq = 'Min'), 'price'])
                .shares
                .sum()
                .apply(np.log)
                .to_frame('shares')
                .reset_index('price')
                .between_time(market_open, market_close)
                .groupby(level = 'timestamp', as_index = False, group_keys = False)
                .apply(lambda x:x.nsmallest(columns = 'price', n = depth))
                .reset_index())
# sell_per_min.timestamp = sell_per_min.timestamp.values.astype(int)
sell_per_min.info()

with pd.HDFStore(order_book_store) as store:
    trades = store['{}/trades'.format(stock)]
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close)

trades_per_min = (trades.resample('Min').agg({ 'price':'mean', 'shares':'sum' }))
# trades_per_min.index = trades_per_min.index.to_series().add(utc_offset).values.astype(int)
trades_per_min.info()


######################################################################################################
## ANALYZE ORDER BOOK DATA
pd.set_option('display.float_format', lambda x:'%.2f' % x)
plt.style.use('fivethirtyeight')
title = '{} | {}'.format(stock, pd.to_datetime(date, format = '%m%d%Y').date())

# load system event data
with pd.HDFStore(itch_store) as store:
    sys_events = store['S'].set_index('event_code').drop_duplicates()
    sys_events.timestamp = sys_events.timestamp.add(pd.to_datetime(date, format = '%m%d%Y')).dt.time
    market_open = sys_events.loc['Q', 'timestamp']
    market_close = sys_events.loc['M', 'timestamp']

#  GE Trade Summary
with pd.HDFStore(order_book_store) as store:
    trades = store['{}/trades'.format(stock)]

trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0]
trades = trades.between_time(market_open, market_close).drop('cross', axis = 1)
# trades.info()

# Trade data is in nanosecs, the noise causes price oscillations between bid-ask prices
tick_bars = trades.copy()
tick_bars.index = tick_bars.index.time
tick_bars.price.plot(figsize = (10, 5))
plt.xlabel('')
plt.suptitle('GE Tick Bars')
plt.tight_layout()
plt.savefig(figure_path / 'tick_bars.png', dpi = 300)
plt.show()
plt.close()


# TODO: fix this shit
# fig, axes = plt.plot(nrows = 2, sharex = True, figsize = (15, 8))
# axes[0].plot(df.index, df[price])
# axes[1].bar(df.index, df[vol], width = 1 / (len(df.index)), color = 'r')
#
# # formatting
# xfmt = mpl.dates.DateFormatter('%H:%M')
# axes[1].xaxis.set_major_locator(mpl.dates.HourLocator(interval = 3))
# axes[1].xaxis.set_major_formatter(xfmt)
# axes[1].get_xaxis().set_tick_params(which = 'major', pad = 25)
# axes[0].set_title('Price Chart', fontsize = 14, loc = 'left')
# axes[1].set_title('Volume Chart', fontsize = 14, loc = 'left')
# fig.autofmt_xdate()
# fig.suptitle(suptitle)
# plt.subplots_adjust(top = 0.9)
# plt.show()

# Normality Test for tick returns
print("\n\nNormality Test for Tick Returns:\n ", normaltest(tick_bars.price.pct_change().dropna()))

# regularize tick data by using price-volume relation comparison
def price_volume(df, price = 'vwap', vol = 'vol', suptitle = stock):
    fig, axes = plt.subplots(nrows = 2, sharex = True, figsize = (15, 8))
    axes[0].plot(df.index, df[price])
    axes[1].bar(df.index, df[vol], width = 1 / (len(df.index)), color = 'r')

    # formatting
    xfmt = mpl.dates.DateFormatter('%H:%M')
    axes[1].xaxis.set_major_locator(mpl.dates.HourLocator(interval = 3))
    axes[1].xaxis.set_major_formatter(xfmt)
    axes[1].get_xaxis().set_tick_params(which = 'major', pad = 25)
    axes[0].set_title('Price Chart', fontsize = 14, loc = 'left')
    axes[1].set_title('Volume Chart', fontsize = 14, loc = 'left')
    fig.autofmt_xdate()
    fig.suptitle(suptitle)
    plt.subplots_adjust(top = 0.9)
    plt.show()


######################################################################################################
# resample data to create time bars and compare normality tests with tick data
def get_bar_stats(agg_trades):
    vwap = agg_trades.apply(lambda x:np.average(x.price, weights = x.shares)).to_frame('vwap')
    ohlc = agg_trades.price.ohlc()
    vol = agg_trades.shares.sum().to_frame('vol')
    txn = agg_trades.shares.size().to_frame('txn')
    return pd.concat([ohlc, vwap, vol, txn], axis = 1)


# create 1 min time bars
resampled = trades.resample('1Min')
time_bars = get_bar_stats(resampled)

# norrmality test for tick price rets
print("\n\nNormality Test for % Change in Price (1-min):\n ", normaltest(time_bars.vwap.pct_change().dropna()))

# compare to min rets
price_volume(time_bars, suptitle = 'GE Time Bars (1-Min)')
plt.savefig(figure_path / 'GE_timebars.png', dpi = 300)

# time bars don't always account for fragmentation of orders. Volume bars offer an alternative perspective
with pd.HDFStore(order_book_store) as store:
    trades = store['{}/trades'.format(stock)]

trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0]
trades = trades.between_time(market_open, market_close).drop('cross', axis = 1)
trades.info()
trades_per_min = trades.shares.sum() / (60 * 7.5)  # min per trading day
trades['cumul_vol'] = trades.shares.cumsum()

df = trades.reset_index()
by_vol = df.groupby(df.cumul_vol.div(trades_per_min).round().astype(int))
vol_bars = pd.concat([by_vol.timestamp.last().to_frame('timestamp'), get_bar_stats(by_vol)], axis = 1)
print("\nGROUPBY CUMUL. VOLUME TRADES/MIN: \n", vol_bars.head())

price_volume(vol_bars.set_index('timestamp'), suptitle = 'GE Aggr.Trades by Volume')
plt.savefig(figure_path / 'GE_volbars.png', dpi = 300)

print("\n\nNormality Test for Aggregated Trades According to Volume:\n ", normaltest(time_bars.vwap.pct_change().dropna()))
normaltest(vol_bars.vwap.dropna())  # normality test
