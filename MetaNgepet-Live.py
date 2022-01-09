#Import needed modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from finta import TA

from datetime import datetime
from time import sleep
import os, pytz
import MetaTrader5 as mt5

import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle

#===================================================================

#Trading Account Parameter
account = config.username   #Account number
password = config.password  #Password number
server = config.mt5_server  #Server name
path = config.mt5_path      #path of Metatrader5 director


timezone = pytz.timezone("Etc/UTC") # set time zone to UTC
symbol = config.symbol
timeframe = config.timeframe
bars = 500

model_filename = 'MetaNgepet\saved_model\MetaNgepet_{}_{}_LinearReg_Model.pkl'.format(symbol, timeframe)
scaler_filename = 'MetaNgepet\saved_scaler\MetaNgepet_{}_{}_LinearReg_Scaler.pkl'.format(symbol, timeframe)

#===================================================================

model = pickle.load(open(model_filename, 'rb'))
scaler = pickle.load(open(scaler_filename, 'rb'))

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

#Extract Account info from dataframe
leverage = Acc_Info.loc[2, "value"]
balance = Acc_Info.loc[10, "value"]
profit = Acc_Info.loc[12, "value"]
equity = Acc_Info.loc[13, "value"]
margin_free = Acc_Info.loc[15, "value"]

print(leverage)
print(balance)
print(profit)
print(equity)
print(margin_free)

# extract info of terminal
terminal_info_dict = mt5.terminal_info()._asdict()
for prop in terminal_info_dict:
    print("  {}={}".format(prop, terminal_info_dict[prop]))

# extract information from pair and timeframe
print ("------------------------------")
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

def get_price(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, bars + 1)

    # create DataFrame out of the obtained data
    rates_frame = pd.DataFrame(rates, dtype=np.dtype("float"))

    # convert time in seconds into the datetime format
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame = rates_frame.rename(columns={'tick_volume': 'volume'})
    del rates_frame['real_volume']

    rates_frame = rates_frame.set_index('time')

    return rates_frame                        

def generate_features(df):
    """ Generate features for a stock/index/currency/commodity based on historical price and performance
    Args:
        df (dataframe with columns "open", "close", "high", "low", "volume")
    Returns:
        dataframe, data set with new features
    Timeframe for calculation:
        6 timeframe for 30 minutes
        48 timeframe for 4 hours
        144 timeframe for 12 hours
        288 timeframe for 1 day
    """
    df_new = pd.DataFrame()
    
    # 6 original features
    df_new['open'] = df['open']
    df_new['open_5m'] = df['open'].shift(1)
    df_new['close_5m'] = df['close'].shift(1)
    df_new['high_5m'] = df['high'].shift(1)
    df_new['low_5m'] = df['low'].shift(1)
    df_new['volume_5m'] = df['volume'].shift(1)
    df_new['spread_5m'] = df['spread'].shift(1)
    
    # 50 original features
    # average price
    df_new['avg_price_30m'] = df['close'].rolling(window=6).mean().shift(1)
    df_new['avg_price_4h'] = df['close'].rolling(window=48).mean().shift(1)
    df_new['avg_price_12h'] = df['close'].rolling(window=144).mean().shift(1)
    df_new['avg_price_1d'] = df['close'].rolling(window=288).mean().shift(1)
    
    # average price ratio
    df_new['ratio_avg_price_30m_4h'] = df_new['avg_price_30m'] / df_new['avg_price_4h']
    df_new['ratio_avg_price_30m_12h'] = df_new['avg_price_30m'] / df_new['avg_price_12h']
    df_new['ratio_avg_price_30m_1d'] = df_new['avg_price_30m'] / df_new['avg_price_1d']
    df_new['ratio_avg_price_4h_12h'] = df_new['avg_price_4h'] / df_new['avg_price_12h']
    df_new['ratio_avg_price_4h_1d'] = df_new['avg_price_4h'] / df_new['avg_price_1d']
    df_new['ratio_avg_price_12h_1d'] = df_new['avg_price_12h'] / df_new['avg_price_1d']                                            
    
    
    # average volume
    df_new['avg_volume_30m'] = df['volume'].rolling(window=6).mean().shift(1)
    df_new['avg_volume_4h'] = df['volume'].rolling(window=48).mean().shift(1)
    df_new['avg_volume_12h'] = df['volume'].rolling(window=144).mean().shift(1)
    df_new['avg_volume_1d'] = df['volume'].rolling(window=288).mean().shift(1)
    
    # average volume ratio
    df_new['ratio_avg_volume_30m_4h'] = df_new['avg_volume_30m'] / df_new['avg_volume_4h']
    df_new['ratio_avg_volumee_30m_12h'] = df_new['avg_volume_30m'] / df_new['avg_volume_12h']                                                   
    df_new['ratio_avg_volume_30m_1d'] = df_new['avg_volume_30m'] / df_new['avg_volume_1d']
    df_new['ratio_avg_volume_4h_12h'] = df_new['avg_volume_4h'] / df_new['avg_volume_12h']
    df_new['ratio_avg_volume_4h_1d'] = df_new['avg_volume_4h'] / df_new['avg_volume_1d']
    df_new['ratio_avg_volume_12h_1d'] = df_new['avg_volume_12h'] / df_new['avg_volume_1d']                                                 
    
    
    # standard deviation of prices
    df_new['std_price_30m'] = df['close'].rolling(window=6).std().shift(1)
    df_new['std_price_4h'] = df['close'].rolling(window=48).std().shift(1)
    df_new['std_price_12h'] = df['close'].rolling(window=144).std().shift(1)                                               
    df_new['std_price_1d'] = df['close'].rolling(window=288).std().shift(1)
    
    # standard deviation ratio of prices 
    df_new['ratio_std_price_30m_4h'] = df_new['std_price_30m'] / df_new['std_price_4h']
    df_new['ratio_std_price_30m_12h'] = df_new['std_price_30m'] / df_new['std_price_12h']
    df_new['ratio_std_price_30m_1d'] = df_new['std_price_30m'] / df_new['std_price_1d']
    df_new['ratio_std_price_4h_12h'] = df_new['std_price_4h'] / df_new['std_price_12h'] 
    df_new['ratio_std_price_3h_1d'] = df_new['std_price_4h'] / df_new['std_price_1d']                                               
    df_new['ratio_std_price_12h_1d'] = df_new['std_price_12h'] / df_new['std_price_1d']                                                
    
    
    # standard deviation of volumes
    df_new['std_volume_30m'] = df['volume'].rolling(window=6).std().shift(1)
    df_new['std_volume_4h'] = df['volume'].rolling(window=48).std().shift(1)
    df_new['std_volume_12h'] = df['volume'].rolling(window=144).std().shift(1)
    df_new['std_volume_1d'] = df['volume'].rolling(window=288).std().shift(1)
    
    # standard deviation ratio of volumes
    df_new['ratio_std_volume_30m_4h'] = df_new['std_volume_30m'] / df_new['std_volume_4h']
    df_new['ratio_std_volume_30m_12h'] = df_new['std_volume_30m'] / df_new['std_volume_12h']
    df_new['ratio_std_volume_30m_1d'] = df_new['std_volume_30m'] / df_new['std_volume_1d']
    df_new['ratio_std_volume_4h_12h'] = df_new['std_volume_4h'] / df_new['std_volume_12h'] 
    df_new['ratio_std_volume_3h_1d'] = df_new['std_volume_4h'] / df_new['std_volume_1d']                                               
    df_new['ratio_std_volume_12h_1d'] = df_new['std_volume_12h'] / df_new['std_volume_1d']                                               
                                                   
    # return
    df_new['return_5m'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)).shift(1)
    df_new['return_30m'] = ((df['close'] - df['close'].shift(6)) / df['close'].shift(6)).shift(1)
    df_new['return_4h'] = ((df['close'] - df['close'].shift(48)) / df['close'].shift(48)).shift(1)
    df_new['return_12h'] = ((df['close'] - df['close'].shift(144)) / df['close'].shift(144)).shift(1)                                                
    df_new['return_1d'] = ((df['close'] - df['close'].shift(288)) / df['close'].shift(288)).shift(1)
    
    # average of return
    df_new['return_avg_30m'] = df_new['return_5m'].rolling(window=6).mean()
    df_new['return_avg_4h'] = df_new['return_5m'].rolling(window=48).mean()
    df_new['return_avg_12h'] = df_new['return_5m'].rolling(window=144).mean()
    df_new['return_avg_1d'] = df_new['return_5m'].rolling(window=288).mean()

    # indicator
    df_new['EMA_4h'] = TA.EMA(df, 48)

    # the target
    # df_new['close'] = df['close']
    df_new = df_new.dropna(axis=0)
    return df_new








while True:
    
    now = datetime.now()
    print(now.strftime("%H:%M:%S"))

    """
    rates_frame = get_price(symbol, timeframe, bars)
    features = generate_features(rates_frame)
    
    features_scaled = scaler.transform(features)

    predictions = model.predict(features_scaled)
    """

    sleep(1)