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

# load_hdf5()names = [name.lstrip(' ') for name in pd.read_csv('{}/SP500.txt'.format(root))['Symbol']]

# ftp://emi.nasdaq.com/ITCH/Stock_Locate_Codes/ndq_stocklocate_20190130.txt

