# import Cython
# import numpy
# import pyximport
# pyximport.install(setup_args={"script_args":["--compiler=mingw32"],
#                               "include_dirs":numpy.get_include()},
#                   reload_support=True,language_level = 3)
# import gzip
# import struct
# import datetime
# import pandas as pd
# import os
# import csv
# import math
#
# from collections import deque
# from datetime import timedelta
# from struct import unpack
# import gzip
#
# # Create dictionary of message types to size of message.
# message_type_size_dict = {
#     b"S" : 11, b"R" : 38, b"H" : 24, b"Y" : 19, b"L" : 25, b"V" : 34, b"W" : 11, b"K" : 27, b"J" : 34, b"h" : 20, b"A" : 35,
#     b"F" : 39, b"E" : 30, b"C" : 35, b"X" : 22, b"D" : 18, b"U" : 34, b"P" : 43, b"Q" : 39, b"B" : 18, b"I" : 49, b"N" : 19,
#     b"\x00" : 1
# }
#
# # return time stamp in nanoseconds from the time_stamp in binary format.
# cpdef long long get_timestamp(bytes binary_value):
#     time_stamp = unpack('>6s', binary_value)[0];
#     return int.from_bytes(time_stamp, byteorder='big')
#
# # Update the stock queue for last 1 hour of prices and volumn
# cpdef void update_the_stock_queues(bytes stock, shares, price, time_stamp, stocks_queue, big_result):
#     # 1 hour  = 3600000000000 nanoseconds (#3.6e12)
#     cdef long long tick_time = 3600000000000
#     # create a queue which will hold one hour prices and shares for the stock
#     if stock not in stocks_queue:
#             stocks_queue[stock] = deque()
#     stock_queue = stocks_queue[stock]
#     # append current message time_stamp, price, and volumn to the queue.
#     stock_queue.append({'time_stamp' : time_stamp,
#                         'price'  : price,
#                         'volumn' : shares})
#     # Only keep messages in 1 hour window
#     while stock_queue[0]['time_stamp'] < (time_stamp - tick_time):
#         stock_queue.popleft()
#     # Number of messages in last 1 hour
#     cdef int N = len(stock_queue)
#     # cumulative shares * volumn
#     cdef double VP = 0
#     # cumulative volumn
#     cdef double V = 0
#     # calculate cummulative P*V and V
#     for i in range(N):
#         VP = VP + stock_queue[i]['price'] * stock_queue[i]['volumn']
#         V = V + stock_queue[i]['volumn']
#
#     #calculate VWAP for last 1 hr for the stock
#     cdef float vwap = (VP // V) * 0.0001 #10e-4
#     converted_time = timedelta(seconds = time_stamp * 1e-9)
#     print (converted_time)
#     vwap_row = ','.join([str(stock), str(vwap), str(time_stamp)])
#     # append the vwap for the stock to final result list
#     big_result.append(vwap_row)
#
# # Handle cross trade message
# cpdef void handle_trade_C_message(bytes msg, stocks_queue, big_result):
#     (shares, stock, price) = unpack('>Q8sI', msg[10:30])
#     if shares > 0:
#         time_stamp = get_timestamp(msg[4:10])
#         update_the_stock_queues(stock, shares, price, time_stamp, stocks_queue, big_result)
#
# # Handle non-cross trade message
# cpdef void handle_trade_NC_message(bytes msg, stocks_queue, big_result):
#     (shares, stock, price) = unpack('>I8sI', msg[19:35])
#     time_stamp = get_timestamp(msg[4:10])
#     update_the_stock_queues(stock, shares, price, time_stamp, stocks_queue, big_result)
#
# # Calculates VWAP for the ITCH File passed and outputs result to output_csv_file passed
# cpdef void calculate_vwap(itch_file_name, output_csv_name = 'stock_vwap_full.csv'):
#     bin_data = gzip.open(itch_file_name, "rb")
#     cdef bytes message_type = bin_data.read(1)
#     cdef long long number_of_messages = 0
#     # stores queue for each stock; The queue will store last 1 hr details
#     stocks_queue = {}
#     cdef bytes msg
#     big_result = []
#     while message_type:
#         msg = bin_data.read(message_type_size_dict[message_type])
#         # Handle non-cross trade message.
#         if message_type == b'P':
#             handle_trade_NC_message(msg, stocks_queue, big_result)
#         # Handle cross trade message.
#         elif message_type == b'Q':
#             handle_trade_C_message(msg, stocks_queue, big_result)
#         # read next message type
#         message_type = bin_data.read(1);
#
#
#         if number_of_messages > 10000: # for all change to 100000000 or remove this if and break!
#             break
#         number_of_messages += 1
#
#
#     textfile = open(output_csv_name, 'w')
#     textfile.write('\n'.join(big_result))
#     textfile.close()
#     print ("The VWAP for each stock are stored in {}".format(output_csv_name))
#
#
# itch_file_name = 'C://Users//jloss//PyCharmProjects//NASDAQ-ITCH-5.0-VWAP-PARSER//01302019.NASDAQ_ITCH50.gz'
# calculate_vwap(itch_file_name)