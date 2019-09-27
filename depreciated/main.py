###########################################################################
#               Trexquant Technical Interview Project
#
#                Created by Joseph Loss on 09/24/2019
#                          loss2@illinois.edu
#                       M.S. Financial Engineering
#               University of Illinois at Urbana-Champaign
#
# Problem Description:
#   Write an efficient program that parses trades from a NASDAQ ITCH 5.0 tick data file
#   and outputs a running volume-weighted average price (VWAP)
#   for each stock at every trading hour including the market close.
#
#
# Copyright Trexquant Investment LP -- All Rights Reserved
#
# Redistribution of this question in whole or part or any reformulation or derivative
# of such question or the posting of any answer to such question
# without written consent from Trexquant is prohibited
###########################################################################

import numpy as np
import gzip
import struct
import datetime
import pandas as pd
import os

class ITCH():

    def __init__(self):
        self.temp = []
        self.flag = None
        if not os.path.exists(os.path.join('.', 'output')):
            os.makedirs(os.path.join('.', 'output'))

    def get_binary(self, size):
        read = bin_data.read(size)
        return read

    def convert_time(self, stamp):
        time = datetime.datetime.fromtimestamp(stamp / 1e9)
        time = time.strftime('%H:%M:%S')
        return time

    def cal_vwap(self, df):
        df['amount'] = df['price'] * df['volume']
        df['time'] = pd.to_datetime(df['time'])
        df = df.groupby([df['time'].dt.hour, df['symbol']])['amount', 'volume'].sum()
        df['vwap'] = df['amount'] / df['volume']
        df['vwap'] = df['vwap'].round(2)
        df = df.reset_index()
        df['time'] = df.apply(lambda x: str(x['time']) + ':00:00', axis=1)
        df = df[['time', 'symbol', 'vwap']]
        return df

    def get_vwap(self, message):
        parsed_data, hour = self.trade_message(message)
        if self.flag is None:
            self.flag = hour
        if self.flag != hour:
            df = pd.DataFrame(self.temp, columns=['time', 'symbol', 'price', 'volume'])
            result = self.cal_vwap(df)
            result.to_csv(os.path.join('..', 'output', str(self.flag) + '.csv'), sep=',', index=False)
            print(result)
            self.temp = []
            self.flag = hour
        self.temp.append(parsed_data)

    def trade_message(self, msg):
        msg_type = 'P'
        temp = struct.unpack('<4s6sQcI8cIQ', msg)
        new_msg = struct.pack('<s4s2s6sQsI8sIQ', msg_type, temp[0], '\x00\x00', temp[1], temp[2], temp[3], temp[4],
                              ''.join(list(temp[5:13])), temp[13], temp[14])
        value = struct.unpack('<sHHQQsI8sIQ', new_msg)
        value = list(value)
        print(value)
        value[3] = self.convert_time(value[3])
        value[7] = value[7].strip()
        value[8] = float(value[8])
        value[8] = value[8] / 10000
        return [value[3], value[7], value[8], value[6]], value[3].split(':')[0]

    def trade_hour(self):
        nano_per_hour= 3.6*(10^12)



if __name__ == '__main__':

    # bin_data = open('01302019.NASDAQ_ITCH50', 'rb')
    bin_data = gzip.open('01302019.NASDAQ_ITCH50.gz', 'rb')
    # bin_data = 'C://Users//jloss//PyCharmProjects//trexquant//01302019.NASDAQ_ITCH50'

    msg_header = bin_data.read(1)
    # msg_header = bin_data.`
    msg_header = msg_header.decode()
    itch = ITCH()

    message = ''

    while msg_header:

        print(msg_header)

        if msg_header == "S":
            message = itch.get_binary(11)
            # print(message)

        elif msg_header == "R":
            message = itch.get_binary(38)
            # print(message)

        elif msg_header == "H":
            message = itch.get_binary(24)
            # print(message)

        elif msg_header == "Y":
            message = itch.get_binary(19)
            # print(message)

        elif msg_header == "L":
            message = itch.get_binary(25)
            # print(message)

        elif msg_header == "V":
            message = itch.get_binary(34)
            # print(message)

        elif msg_header == "W":
            message = itch.get_binary(11)
            # print(message)

        elif msg_header == "K":
            message = itch.get_binary(27)
            # print(message)

        elif msg_header == "A":
            message = itch.get_binary(35)
            # print(message)

        elif msg_header == "F":
            message = itch.get_binary(39)
            # print(message)

        elif msg_header == "E":
            message = itch.get_binary(30)
            # print(message)

        elif msg_header == "C":
            message = itch.get_binary(35)
            # print(message)

        elif msg_header == "X":
            message = itch.get_binary(22)
            # print(message)

        elif msg_header == "D":
            message = itch.get_binary(18)
            # print(message)

        elif msg_header == "U":
            message = itch.get_binary(34)
            # print(message)

        elif msg_header == "P":

            print(msg_header,message)
            message = itch.get_binary(43)
            # print(message)
            vwap = itch.get_vwap(message)
            # print(vwap)

        elif msg_header == "Q":
            message = itch.get_binary(39)
            # print(message)

        elif msg_header == "B":
            message = itch.get_binary(18)
            # print(message)

        elif msg_header == "I":
            message = itch.get_binary(49)
            # print(message)

        elif msg_header == "N":
            message = itch.get_binary(19)
            # print(message)

        msg_header = bin_data.read(1)
        msg_header = msg_header.decode()

    bin_data.close()
