#Importing packages ...
import Cython
import pyximport
pyximport.install(language_level = 3)
import csv
import gzip
import struct
import datetime
import pandas as pd
import os

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
    



#Reference: https://docs.python.org/3/library/struct.html

#Main function read byte by byte and performs necessary operation based on its type. It also writes parsed data in the csv file

if __name__ == '__main__':

    bin_data = gzip.open('C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//01302019.NASDAQ_ITCH50.gz',
                         'r')
    msg_header = bin_data.read(1)
    # out_file = open('parsed_data.csv','w');
    # writer = csv.writer(out_file);

    parser = parser()

    while msg_header:
        if msg_header == b'S':
            message = parser.readData(11)


        elif msg_header == b'R':
            message = parser.readData(38)
    
        elif msg_header == b'H':
            message = parser.readData(24)

        elif msg_header == b'Y':
            message = parser.readData(19)

        elif msg_header == b'L':
            message = parser.readData(25)

        elif msg_header == b'V':
            message = parser.readData(34)
            
        elif msg_header == b'W':
            message = parser.readData(11)

        elif msg_header == b'K':
            message = parser.readData(27)

        elif msg_header == b'A':
            message = parser.readData(35)

        elif msg_header == b'F':
            message = parser.readData(39)

        elif msg_header == b'E':
            message = parser.readData(30)

        elif msg_header == b'C':
            message = parser.readData(35)

        elif msg_header == b'X':
            message = parser.readData(22)

        elif msg_header == b'D':
            message = parser.readData(18)

        elif msg_header == b'U':
            message = parser.readData(34)

        elif msg_header == b'P':
            message = parser.readData(43)
            parser.extractData(message)
            # print(msg)

        elif msg_header == b'Q':
            message = parser.readData(39)

        elif msg_header == b'B':
            message = parser.readData(18)

        elif msg_header == b'I':
            message = parser.readData(49)

        elif msg_header == b'N':
            message = parser.readData(19)

        msg_header = bin_data.read(1)

    bin_data.close()

																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																					
