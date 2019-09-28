#Importing packages ...
import Cython
import pyximport
pyximport.install(language_level = 3)


import gzip
import struct
import datetime
import pandas as pd
import os
import csv

#Created a class which parses through the ITCH 5.0 file and calculates VSAP.

class parser():

    def __init__(self):
        self.temp = []
        self.flag = None

    def readData(self, size):
        read = bin_data.read(size)
        return read

    def convert_time(self, stamp):
        time = datetime.datetime.fromtimestamp(stamp / 1e9)
        time = time.strftime('%H:%M:%S')
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
        if self.flag is None:
            self.flag = hour
        elif self.flag != hour:
            df = pd.DataFrame(self.temp, columns=['time', 'symbol', 'price', 'volume'])
            if len(df)>0:
                result = self.calculate_vwap(df)
                # result.to_csv(os.path.join('.', 'output', str(self.flag) + '.txt'), sep=' ', index=False)
                print(result)
            self.temp = []
            self.flag = None
        else:
            self.temp.append(parsed_data)

    # This function deals with trade messages. It decodes the 43 byte trade message and extracts details like trade value, time and price.
    
    def tradeMessage(self, msg):
        msg_type = b'P'
        temp = struct.unpack('>4s6sQcI8cIQ', msg)
        temp=list(temp)
        new_msg = struct.pack('>s4s2s6sQsI8sIQ', msg_type, temp[0], b'\x00\x00', temp[1], temp[2], temp[3], temp[4],
                              b''.join((temp[5:13])), temp[13], temp[14])
        value = struct.unpack('>sHHQQsI8sIQ', new_msg)
        value = list(value)
        value[3] = self.convert_time(value[3])
        value[7] = value[7].strip()
        value[8] = float(value[8])
        value[8] = value[8] / 10000
        return value
    
    def system_event_message(self,msg):
        msg_type = b'S';
        temp = struct.unpack('>HH6sc',msg);
        new_msg = struct.pack('>sHH2s6sc',msg_type,temp[0],temp[1],b'x\00x\00',temp[2],temp[3]);
        val = struct.unpack('>cHHQc',new_msg);
        val = list(val);
        return val;
    
    def stock_directory_message(self,msg):
        msg_type = b'R';
        temp = struct.unpack('>4s6s10cI9cIc',msg);
        new_msg = struct.pack('>s4s2s6s10sI9sIs',msg_type,temp[0],b'\x00\x00',temp[1],b''.join(list(temp[2:12])),temp[12],b''.join(list(temp[13:22])),temp[22],temp[23]);
        val = struct.unpack('>sHHQ8sssIss2ssssssIs',new_msg);
        val = list(val);
        return val;

    def stock_trading_action(self,msg):
        msg_type = b'H';
        temp = struct.unpack('>4s6s14c',msg);
        new_msg = struct.pack('>s4s2s6s14s',msg_type,temp[0],b'\x00\x00',temp[1],b''.join(list(temp[2:16])));
        val = struct.unpack('>sHHQ8sss4s',new_msg);
        val = list(val);
        return val;

    def short_sale_price_test(self,msg):
        msg_type = b'Y';
        temp = struct.unpack('>4s6s9c',msg);
        new_msg = struct.pack('>s4s2s6s9s',msg_type,temp[0],b'\x00\x00',temp[1],b''.join(list(temp[2:11])));
        val = struct.unpack('>sHHQ8ss',new_msg);
        val = list(val);
        return val;

    def market_participation_position(self,msg):
        msg_type = b'L';
        temp = struct.unpack('>4s6s15c',msg);
        new_msg = struct.pack('>s4s2s6s15s',msg_type,temp[0],b'\x00\x00',temp[1],b''.join(list(temp[2:17])));
        val = struct.unpack('>sHHQ4s8ssss',new_msg);
        val = list(val);
        return val;

    def mwcb_decline_level_message(self,msg):
        msg_type = b'V';
        temp = struct.unpack('>4s6s24s',msg);
        new_msg = struct.pack('>s4s2s6s24s',msg_type,temp[0],b'\x00\x00',temp[1],temp[2]);
        val = struct.unpack('>sHHQQQQ',new_msg);
        val = list(val);
        val[4:7] = map(float,val[4:7]);
        val[4:7] = map(lambda x:x/(pow(10,8)),val[4:7]);
        return val;

    def mwcb_status_message(self,msg):
        msg_type = b'W';
        temp = struct.unpack('>4s6sc',msg);
        new_msg = struct.pack('>s4s2s6ss',msg_type,temp[0],b'\x00\x00',temp[1],temp[2]);
        val = struct.unpack('>sHHQs',new_msg);
        val = list(val);
        return val;

    def ipo_quoting_period_update(self,msg):
        msg_type = b'K';
        temp = struct.unpack('>4s6s8cIcI',msg);
        new_msg = struct.pack('>s4s2s6s8sIsI',msg_type,temp[0],b'\x00\x00',temp[1],b''.join(list(temp[2:10])),temp[10],temp[11],temp[12]);
        val = struct.unpack('>sHHQ8sIsI',new_msg);
        val = list(val);
        val[7] = float(val[7]);
        val[7] = val[7]/10000;
        return val;

    def add_order_no_mpid(self,msg):
        msg_type = b'A';
        temp = struct.unpack('>4s6sQcI8cI',msg);
        new_msg = struct.pack('>s4s2s6sQsI8sI',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],temp[3],temp[4],b''.join(list(temp[5:13])),temp[13]);
        val = struct.unpack('>sHHQQsI8sI',new_msg);
        val = list(val);
        val[8] = float(val[8]);
        val[8] = val[8]/10000;
        return val;

    def add_order_with_mpid(self,msg):
        msg_type = b'F';
        temp = struct.unpack('>4s6sQcI8cI4c',msg);
        new_msg = struct.pack('>s4s2s6sQsI8sI4s',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],temp[3],temp[4],b''.join(list(temp[5:13])),temp[13],b''.join(list(temp[14:18])));
        val = struct.unpack('>sHHQQsI8sI4s',new_msg);
        val = list(val);
        val[8] = float(val[8]);
        val[8] = val[8]/10000; 
        return val;

    def order_executed_message(self,msg):
        msg_type = b'E';
        temp = struct.unpack('>4s6sQIQ',msg);
        new_msg = struct.pack('>s4s2s6sQIQ',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],temp[3],temp[4]);
        val = struct.unpack('>sHHQQIQ',new_msg);
        val = list(val);
        return val;

    def order_executed_price_message(self,msg):
        msg_type = b'C';
        temp = struct.unpack('>4s6sQIQcI',msg);
        new_msg = struct.pack('>s4s2s6sQIQsI',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],temp[3],temp[4],temp[5],temp[6]);
        val = struct.unpack('>sHHQQIQsI',new_msg);
        val = list(val);
        val[8] = float(val[8]);
        val[8] = val[8]/10000; 
        return val;

    def order_cancel_message(self,msg):
        msg_type = b'X';
        temp = struct.unpack('>4s6sQI',msg);
        new_msg = struct.pack('>s4s2s6sQI',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],temp[3]);
        val = struct.unpack('>sHHQQI',new_msg);
        val = list(val);
        return val;

    def order_delete_message(self,msg):
        msg_type = b'D';
        temp = struct.unpack('>4s6sQ',msg);
        new_msg = struct.pack('>s4s2s6sQ',msg_type,temp[0],b'\x00\x00',temp[1],temp[2]);
        val = struct.unpack('>sHHQQ',new_msg);
        val = list(val);
        return val;

    def order_replace_message(self,msg):
        msg_type = b'U';
        temp = struct.unpack('>4s6sQQII',msg);
        new_msg = struct.pack('>s4s2s6sQQII',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],temp[3],temp[4],temp[5]);
        val = struct.unpack('>sHHQQQII',new_msg);
        val = list(val);
        val[7] = float(val[7]);
        val[7] = val[7]/10000;
        return val;

    def cross_trade_message(self,msg):
        msg_type = b'Q';
        temp = struct.unpack('>4s6sQ8cIQc',msg);
        new_msg = struct.pack('>s4s2s6sQ8sIQs',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],b''.join(list(temp[3:11])),temp[11],temp[12],temp[13]);
        val = struct.unpack('>sHHQQ8sIQs',new_msg);
        val = list(val);
        val[6] = float(val[6]);
        val[6] = val[6]/10000;
        return val;

    def broken_trade_execution_message(self,msg):
        msg_type = b'B';
        temp = struct.unpack('>4s6sQ',msg);
        new_msg = struct.pack('>s4s2s6sQ',msg_type,temp[0],b'\x00\x00',temp[1],temp[2]);
        val = struct.unpack('>sHHQQ',new_msg);
        val = list(val);
        return val;

    def net_order_imbalance_message(self,msg):
        msg_type = b'I';
        temp = struct.unpack('>4s6s16s9c12s2c',msg);
        new_msg = struct.pack('>s4s2s6s16s9s12s2s',msg_type,temp[0],b'\x00\x00',temp[1],temp[2],b''.join(list(temp[3:12])),temp[12],b''.join(list(temp[13:15])));
        val = struct.unpack('>sHHQQQs8sIIIss',new_msg);
        val = list(val);
        val[8:11] = map(float,val[8:11]);
        val[8:11] = map(lambda x:x/10000,val[8:11]);
        return val;

    def retail_price_improvement_indicator(self,msg):
        msg_type = b'N';
        temp = struct.unpack('>4s6s9c',msg);
        new_msg = struct.pack('>s4s2s6s9s',msg_type,temp[0],b'\x00\x00',temp[1],b''.join(list(temp[2:11])));
        val = struct.unpack('>sHHQ8ss',new_msg);
        val = list(val);
        return val;




#Reference: https://docs.python.org/3/library/struct.html

#Main function read byte by byte and performs necessary operation based on its type. It also writes parsed data in the csv file

if __name__ == '__main__':

    bin_data = gzip.open('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//01302019.NASDAQ_ITCH50.gz',
                         'r')
    msg_header = bin_data.read(1)
    out_file = open('parsed_data.csv','w');
    writer = csv.writer(out_file);

    parser = parser()

    while msg_header:
        if msg_header == b'S':
            message = parser.readData(11)
            parsed_data = parser.system_event_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'R':
            message = parser.readData(38)
            parsed_data = parser.stock_directory_message(message);
            writer.writerow(parsed_data)
    
        elif msg_header == b'H':
            message = parser.readData(24)
            parsed_data = parser.stock_trading_action(message);
            writer.writerow(parsed_data)

        elif msg_header == b'Y':
            message = parser.readData(19)
            parsed_data = parser.short_sale_price_test(message);
            writer.writerow(parsed_data)

        elif msg_header == b'L':
            message = parser.readData(25)
            parsed_data = parser.market_participation_position(message);
            writer.writerow(parsed_data)

        elif msg_header == b'V':
            message = parser.readData(34)
            parsed_data = parser.mwcb_decline_level_message(message);
            writer.writerow(parsed_data)
            
        elif msg_header == b'W':
            message = parser.readData(11)
            parsed_data = parser.mwcb_status_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'K':
            message = parser.readData(27)
            parsed_data = parser.ipo_quoting_period_update(message);
            writer.writerow(parsed_data)

        elif msg_header == b'A':
            message = parser.readData(35)
            parsed_data = parser.add_order_no_mpid(message);
            writer.writerow(parsed_data)

        elif msg_header == b'F':
            message = parser.readData(39)
            parsed_data = parser.add_order_with_mpid(message);
            writer.writerow(parsed_data)

        elif msg_header == b'E':
            message = parser.readData(30)
            parsed_data = parser.order_executed_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'C':
            message = parser.readData(35)
            parsed_data = parser.order_executed_price_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'X':
            message = parser.readData(22)
            parsed_data = parser.order_cancel_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'D':
            message = parser.readData(18)
            parsed_data = parser.order_delete_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'U':
            message = parser.readData(34)
            parsed_data = parser.order_replace_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'P':
            message = parser.readData(43)
            parser.extractData(message)
            parsed_data = parser.tradeMessage(message);
            writer.writerow(parsed_data)

        elif msg_header == b'Q':
            message = parser.readData(39)
            parsed_data = parser.cross_trade_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'B':
            message = parser.readData(18)
            parsed_data = parser.broken_trade_execution_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'I':
            message = parser.readData(49)
            parsed_data = parser.net_order_imbalance_message(message);
            writer.writerow(parsed_data)

        elif msg_header == b'N':
            message = parser.readData(19)
            parsed_data = parser.retail_price_improvement_indicator(message);
            writer.writerow(parsed_data)

        msg_header = bin_data.read(1)

    bin_data.close()
    out_file.close();

out_file.close();

# It is mentioned in the problem statement that, I need to write a parser which will parse this entire file. Here, I assumed that I need to parse every type of message and write them in a file seperatly. The computation speed could'vebeen decresed if other messages weren't parsed. To calculate VWAP, only trade_message(T) is useful. Also, if I had avoided writing the code to read messages back into csv, the code would have been much faster. 
# This code writes data of every hour int0 a different file
# So, there are 24 files which contains time, stock name and price.
