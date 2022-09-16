###########################################################################
# Parsing NASDAQ ITCH-5.0 Trade Data
### MS Financial Engineering || University of Illinois at Urbana-Champaign
####

####   Created by Joseph Loss on 09/01/2019
####   contact: connect@josephjloss.com
  
> **Sample Output**
> Unzipping to C:\Users\jloss\PyCharmProjects\NASDAQ-ITCH-5.0-VWAP-PARSER\data\10302019.NASDAQ_ITCH50.bin
> 
> Message Types and Description:
>     message_type                                               name
> 0             S                                       system_event
> 5             R                                    stock_directory
> 23            H                               stock_trading_action
> 31            Y  reg_sho_short_sale_price_test_restricted_indic
> 37            L                        market_participant_position 
> 
> Message Types Header: 
>                name  offset                              notes message_type
> 1     stock_locate       1                           Always 0            S
> 2  tracking_number       3    Nasdaq internal tracking number            S
> 3        timestamp       5         Nanoseconds since midnight            S
> 
> [3 rows x 6 columns] 
> 
> 
> =====  INITALIZING PARSER  ===== 
> Start of Messages
> 	3:02:31.647283	0
> Start of System Hours
> 	4:00:00.000203	241,258
> Start of Market Hours
> 	9:30:00.000070	9,559,279
> 	9:44:09.234742	25,000,000	0:01:21.695234
> 	10:07:45.145293	50,000,000	0:04:47.039161
> 	10:39:56.236836	75,000,000	0:07:51.579376
> 	11:18:09.635432	100,000,000	0:10:59.959686
> 	11:58:35.348562	125,000,000	0:14:07.851465
> 	12:44:20.614655	150,000,000	0:17:37.421978
> 	13:41:03.747497	175,000,000	0:21:46.086413
> 	14:18:44.524342	200,000,000	0:26:16.709852
> 	14:49:19.384369	225,000,000	0:30:42.701890
> 	15:19:40.719266	250,000,000	0:35:11.552232
> 	15:50:23.011120	275,000,000	0:39:31.721726
> End of Market Hours
> 	16:00:00.000087	290,920,164
> End of System Hours
> 	20:00:00.000021	293,944,863
> End of Messages
> 	20:05:00.000062	293,989,078
> PARSE FINISHED IN:  0:44:59.213565
> 
> ----------------------------------------------
> =====  RECONSTRUCTING ORDER FLOW  =====
> DATE:   10302019 
> TICKER:   GE
> 
>  =====  ORDER MESSAGES SUMMARY:  =====
> 
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 660998 entries, 0 to 660997
> Data columns (total 9 columns):
> timestamp             660998 non-null datetime64[ns]
> buy_sell_indicator    627120 non-null float64
> shares                634961 non-null float64
> price                 634961 non-null float64
> type                  660998 non-null object
> executed_shares       32252 non-null float64
> execution_price       1258 non-null float64
> shares_replaced       25118 non-null float64
> price_replaced        25118 non-null float64
> dtypes: datetime64[ns](1), float64(7), object(1)
> memory usage: 45.4+ MB
> 
>  =====  ORDERBOOK STORAGE FORMAT:  =====
>   <class 'pandas.io.pytables.HDFStore'>
> File path: C:\Users\jloss\PyCharmProjects\NASDAQ-ITCH-5.0-VWAP-PARSER\data\order_book.h5
> /GE/trading_msgs            frame        (shape->[660998,9]) 
> 
> 100,000		0:00:17.898000
> 200,000		0:00:10.426043
> 300,000		0:00:10.524115
> 400,000		0:00:11.262737
> 500,000		0:00:12.428444
> 600,000		0:00:11.937682
> 
> Count per Message Type: 
>  A    300755
> D    254758
> E     25895
> X      4370
> P     14722
> U     25118
> F       375
> C      1127
> dtype: int64
> 
> OrderBook Storage Format (.h5 file): 
>  <class 'pandas.io.pytables.HDFStore'>
> File path: C:\Users\jloss\PyCharmProjects\NASDAQ-ITCH-5.0-VWAP-PARSER\data\order_book.h5
> /GE/buy                     frame_table  (typ->appendable,nrows->26555553,ncols->2,indexers->[index],dc->[])
> /GE/sell                    frame_table  (typ->appendable,nrows->27060942,ncols->2,indexers->[index],dc->[])
> /GE/trades                  frame        (shape->[41877,3])                                                 
> /GE/trading_msgs            frame        (shape->[660998,9])                                                
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 16150 entries, 0 to 16149
> Data columns (total 3 columns):
> timestamp    16150 non-null int32
> price        16150 non-null float64
> shares       16150 non-null float64
> dtypes: float64(2), int32(1)
> memory usage: 315.6 KB
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: 16150 entries, 0 to 16149
> Data columns (total 3 columns):
> timestamp    16150 non-null datetime64[ns]
> price        16150 non-null float64
> shares       16150 non-null float64
> dtypes: datetime64[ns](1), float64(2)
> memory usage: 378.6 KB
> <class 'pandas.core.frame.DataFrame'>
> DatetimeIndex: 390 entries, 2019-10-30 09:30:00 to 2019-10-30 15:59:00
> Freq: T
> Data columns (total 2 columns):
> price     390 non-null float64
> shares    390 non-null int32
> dtypes: float64(1), int32(1)
> memory usage: 17.6 KB
> 
> 
> Normality Test for Tick Returns:
>   NormaltestResult(statistic=8196.929781884955, pvalue=0.0)
> 
> 
> Normality Test for % Change in Price (1-min):
>   NormaltestResult(statistic=48.44611334056492, pvalue=3.0203700368893266e-11)
> <class 'pandas.core.frame.DataFrame'>
> DatetimeIndex: 35544 entries, 2019-10-30 09:30:00.004986259 to 2019-10-30 15:59:58.026050348
> Data columns (total 2 columns):
> shares    35544 non-null int32
> price     35544 non-null float64
> dtypes: float64(1), int32(1)
> memory usage: 694.2 KB
> 
> GROUPBY CUMUL. VOLUME TRADES/MIN: 
>                                timestamp  open  high    vwap    vol  txn
> cumul_vol                                                              
> 0         2019-10-30 09:30:19.278125824  9.78  9.79    9.77  11449   59
> 1         2019-10-30 09:30:19.975008073  9.75  9.85    9.80  53122  105
> 2         2019-10-30 09:30:31.240346784  9.85  9.85    9.79  40266  120
> 3         2019-10-30 09:30:36.081301941  9.80  9.80    9.79  41921   52
> 4         2019-10-30 09:30:48.788765688  9.78  9.80    9.78  46309   97
> 
> [5 rows x 8 columns]
> 
> 
> Normality Test for Aggregated Trades According to Volume:
>   NormaltestResult(statistic=48.44611334056492, pvalue=3.0203700368893266e-11)