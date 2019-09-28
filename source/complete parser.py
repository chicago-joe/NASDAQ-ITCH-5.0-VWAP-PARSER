import gzip
import struct
import datetime
import pandas as pd
import os
import csv
import math


class parser():

    def __init__(self):
        self.temp = []
        self.flag = None
        self.TradeHour = False

    def readData(self, size):
        read = bin_data.read(size)
        return read

    def convert_time(self, stamp):
        # t1=datetime.time(stamp / 1e9)
        t2=datetime.time(math.floor(stamp / 1e9/3600),math.floor(stamp / 1e9%3600/60),math.floor(stamp / 1e9%3600%60))
        time = datetime.datetime.fromtimestamp(stamp / 1e9/3600)
        time = t2.strftime('%H:%M:%S')
        return time


    # This function calculates VWAP for every trade for every hour.

    def calculate_vwap(self, df):
        df['amount'] = df['price'] * df['volume']
        df['time'] = pd.to_datetime(df['time'])
        df = df.groupby([df['time'].dt.hour, df['symbol']])['amount', 'volume'].sum()
        df['vwap'] = df['amount'] / df['volume']
        df['vwap'] = df['vwap'].round(2)
        df = df.reset_index()
        df['time'] = df.apply(lambda x: str(x['time']) + ':00:00', axis=1)
        df['stock']=df['symbol'].str.decode("utf-8")
        df = df[['time', 'stock', 'vwap']]
        return df

    # This function performs data manipulation before calculating VWAP. Also stores results in txt file.

    def extractData(self, message):
        trades = self.tradeMessage(message)
        parsed_data=[trades[3],trades[7],trades[8],trades[6]]
        hour=trades[3].split(':')[0]
        print(trades[3])
        min=trades[3].split(':')[1]
        if self.flag is None:
            self.flag = hour
        elif self.flag != hour:
            df = pd.DataFrame(self.temp, columns=['time', 'symbol', 'price', 'volume'])
            if len(df)>0:
                result = self.calculate_vwap(df)
                result.to_csv(os.path.join('.', 'output', str(self.flag) + '.csv'), sep=',', index=False)
            self.temp = []
            self.flag = None
        else:
            # if (int(hour)==9 and int(min)>=30) or (int(hour)>9 and int(hour)<16):
            if self.TradeHour:
                self.temp.append(parsed_data)

    # This function deals with trade messages. It decodes the 43 byte trade message and extracts details like trade value, time and price.

    def tradeMessage(self, msg):
        msg_type = b'P'
        value = struct.unpack('>sHH6sQsI8sIQ', msg)
        value = list(value)
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        value[3]=CURR_TIME
        value[3] = self.convert_time(value[3])
        value[7] = value[7].strip()
        value[8] = float(value[8])
        value[8] = value[8] / 10000
        return value

    def system_event_message(self,msg):
        msg_type = 'S'
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val = struct.unpack('>cHH6sc',msg);
        val = list(val);
        val[3]=CURR_TIME
        if (val[4])==b'Q':
            self.TradeHour=True
        if (val[4])==b'M':
            self.TradeHour=False
        if (val[4])==b'C':
            df = pd.DataFrame(self.temp, columns=['time', 'symbol', 'price', 'volume'])
            if len(df)>0:
                result = self.calculate_vwap(df)
                result.to_csv(os.path.join('.', 'output', str(self.flag) + '.csv'), sep=',', index=False)
            self.temp = []
            self.flag = None
        return val;

    def stock_directory_message(self,msg):
        msg_type = 'R';
        val = struct.unpack('>cHH6s8sccIcc2scccccIc',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME

        return val;

    def stock_trading_action(self,msg):
        msg_type = 'H';
        val = struct.unpack('>sHH6s8sss4s',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def short_sale_price_test(self,msg):
        msg_type = 'Y';
        val = struct.unpack('>sHH6s8ss',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def market_participation_position(self,msg):
        msg_type = 'L';
        val = struct.unpack('>sHH6s4s8ssss',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def mwcb_decline_level_message(self,msg):
        msg_type = 'V';
        val = struct.unpack('>sHH6sQQQ',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[4:7] = map(float,val[4:7]);
        val[4:7] = map(lambda x:x/(pow(10,8)),val[4:7]);
        return val;

    def mwcb_status_message(self,msg):
        msg_type = 'W';
        val = struct.unpack('>sHH6ss',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def ipo_quoting_period_update(self,msg):
        msg_type = 'K';
        val = struct.unpack('>sHH6s8sIsI',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[7] = float(val[7]);
        val[7] = val[7]/10000;
        return val;

    def add_order_no_mpid(self,msg):
        msg_type = 'A';
        val = struct.unpack('>sHH6sQsI8sI',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[8] = float(val[8]);
        val[8] = val[8]/10000;
        return val;

    def add_order_with_mpid(self,msg):
        msg_type = 'F';
        val = struct.unpack('>sHH6sQsI8sI4s',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[8] = float(val[8]);
        val[8] = val[8]/10000;
        return val;

    def order_executed_message(self,msg):
        msg_type = 'E';
        val = struct.unpack('>sHH6sQIQ',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def order_executed_price_message(self,msg):
        msg_type = 'C';
        val = struct.unpack('>sHH6sQIQsI',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[8] = float(val[8]);
        val[8] = val[8]/10000;
        return val;

    def order_cancel_message(self,msg):
        msg_type = 'X';
        val = struct.unpack('>sHH6sQI',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def order_delete_message(self,msg):
        msg_type = 'D';
        val = struct.unpack('>sHH6sQ',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def order_replace_message(self,msg):
        msg_type = 'U';
        val = struct.unpack('>sHH6sQQII',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[7] = float(val[7]);
        val[7] = val[7]/10000;
        return val;

    def cross_trade_message(self,msg):
        msg_type = 'Q';
        val = struct.unpack('>sHH6sQ8sIQs',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[6] = float(val[6]);
        val[6] = val[6]/10000;
        return val;

    def broken_trade_execution_message(self,msg):
        msg_type = 'B';
        val = struct.unpack('>sHH6sQ',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;

    def net_order_imbalance_message(self,msg):
        msg_type = 'I';
        val = struct.unpack('>sHH6sQQs8sIIIss',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        val[8:11] = map(float,val[8:11]);
        val[8:11] = map(lambda x:x/10000,val[8:11]);
        return val;

    def retail_price_improvement_indicator(self,msg):
        msg_type = 'N';
        val = struct.unpack('>sHH6s8ss',msg);
        val = list(val);
        CURR_TIME = int.from_bytes(msg[5:11], 'big')
        val[3]=CURR_TIME
        return val;


if __name__ == '__main__':

    bin_data = gzip.open('../../01302019.NASDAQ_ITCH50.gz',
                         'r')
    msg_size = int.from_bytes(bin_data.read(2),'big')

    out_file = open('parsed_data.txt','w');
    writer = csv.writer(out_file);

    parser = parser()

    while msg_size:
        message = parser.readData(msg_size)
        msg_header = chr(message[0])

        if msg_header == 'S':
            #message = parser.readData(11)
            parsed_data = parser.system_event_message(message);
            writer.writerow(parsed_data)

        elif msg_header == 'R':
            #message = parser.readData(38)
            parsed_data = parser.stock_directory_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'H':
            #message = parser.readData(24)
            parsed_data = parser.stock_trading_action(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'Y':
            #message = parser.readData(19)
            parsed_data = parser.short_sale_price_test(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'L':
            #message = parser.readData(25)
            parsed_data = parser.market_participation_position(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'V':
            #message = parser.readData(34)
            parsed_data = parser.mwcb_decline_level_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'W':
            #message = parser.readData(11)
            parsed_data = parser.mwcb_status_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'K':
            #message = parser.readData(27)
            parsed_data = parser.ipo_quoting_period_update(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'A':
            #message = parser.readData(35)
            parsed_data = parser.add_order_no_mpid(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'F':
            #message = parser.readData(39)
            parsed_data = parser.add_order_with_mpid(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'E':
            #message = parser.readData(30)
            parsed_data = parser.order_executed_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'C':
            #message = parser.readData(35)
            parsed_data = parser.order_executed_price_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'X':
            #message = parser.readData(22)
            parsed_data = parser.order_cancel_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'D':
            #message = parser.readData(18)
            parsed_data = parser.order_delete_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'U':
            #message = parser.readData(34)
            parsed_data = parser.order_replace_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'P':
            #message = parser.readData(43)
            parser.extractData(message)
            parsed_data = parser.tradeMessage(message);
            writer.writerow(parsed_data)

        elif msg_header == 'Q':
            #message = parser.readData(39)
            parsed_data = parser.cross_trade_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'B':
            #message = parser.readData(18)
            parsed_data = parser.broken_trade_execution_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'I':
            #message = parser.readData(49)
            parsed_data = parser.net_order_imbalance_message(message);
            # writer.writerow(parsed_data)

        elif msg_header == 'N':
            #message = parser.readData(19)
            parsed_data = parser.retail_price_improvement_indicator(message);
            # writer.writerow(parsed_data)

        msg_size = int.from_bytes(bin_data.read(2),'big')

    bin_data.close()
    out_file.close();
