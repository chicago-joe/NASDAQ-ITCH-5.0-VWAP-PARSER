## VWAP-ITCH-5.0
*Calculating VWAP for every stock per hour by parsing ITCH 5.0 file*


Download NASDAQ ITCH 5.0 data from here:
[ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/01302019.NASDAQ_ITCH50.gz](ftp://emi.nasdaq.com/ITCH/Nasdaq_ITCH/01302019.NASDAQ_ITCH50.gz)

filename: 
01302019.NASDAQ_ITCH50.gz


#
#### REQUIREMENTS
```
pip install -U Cython, numpy, pandas, datetime
```

#### INSTRUCTIONS
Please make sure to adjust bin_data (gzip file location) in the .py and
.pyx files.

Output files are written in csv format to the /source/output folder.

**Step 1**
```
cd /source
```

**Step 2**
``` 
python setup.py build_ext --inplace
```

**Step 3**
```
python complete_parser.pyx
```
