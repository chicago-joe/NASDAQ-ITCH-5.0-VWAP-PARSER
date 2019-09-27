## VWAP-ITCH-5.0
*Calculating VWAP for every stock per hour by parsing ITCH 5.0 file*


Download NASDAQ ITCH 5.0 data from here:
[ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/01302019.NASDAQ_ITCH50.gz](ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/01302019.NASDAQ_ITCH50.gz)

filename: 
01302019.NASDAQ_ITCH50.gz


#
#### INSTRUCTIONS
1. `cd /source`

2. `python setup.py build_ext --inplace`

3. `python VWAP.pyx`
