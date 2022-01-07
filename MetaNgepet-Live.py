#Import needed modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from finta import TA

from datetime import datetime
import os, pytz
import MetaTrader5 as mt5

import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle

#Trading Account Parameter
account = config.username   #Account number
password = config.password  #Password number
server = config.mt5_server  #Server name
path = config.mt5_path      #path of Metatrader5 director

mt5.initialize(
   path = path,          # path to the MetaTrader 5 terminal EXE file
   login = account,      # account number
   password = password,  # password
   server = server,      # server name as it is specified in the terminal
   #timeout = TIMEOUT,   # timeout
   portable = False      # portable mode
   )

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__,"\n")

# establish connection to the MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize failed, error code =",mt5.last_error())
    mt5.shutdown()
else:
    print("MetaTrader5 Initialized!")
    account_info_dict = mt5.account_info()._asdict()
    Acc_Info = pd.DataFrame(list(account_info_dict.items()),columns=['property','value'])
    print("\n", "account_info() as dataframe:")
    print(Acc_Info)

#Extract Account info from dataframe
leverage = Acc_Info.loc[2, "value"]
equity = Acc_Info.loc[13, "value"]
margin_free = Acc_Info.loc[15, "value"]

print(leverage)
print(equity)
print(margin_free)

# extract information from pair and timeframe
symbols = ["EURUSD", "AUDUSD", "GBPUSD"]
print ("------------------------------")
for symbol in symbols:
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info!=None:
        # display the terminal data 'as is'    
        #print(symbol_info)
        print("\n","{}:".format(symbol))
        print("spread =",symbol_info.spread,", digits =",symbol_info.digits, "\n")
        print(symbol_info.swap_long, '\n')
        print ("------------------------------")
    else:
        print("There is no such symbol of {}".format(symbol))

# set time zone to UTC
timezone = pytz.timezone("Etc/UTC")
symbol = 'AUDUSD'
timeframe = config.timeframe
bars = 500

rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, bars + 1)

# create DataFrame out of the obtained data
rates_frame = pd.DataFrame(rates, dtype=np.dtype("float"))

# convert time in seconds into the datetime format
rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
rates_frame = rates_frame.rename(columns={'tick_volume': 'volume'})
del rates_frame['real_volume']

rates_frame = rates_frame.set_index('time')                           

# display data
print("\nDisplay dataframe with data")
print(rates_frame)