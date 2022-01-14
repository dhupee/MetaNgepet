#Import needed modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from finta import TA

from datetime import datetime, timezone
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

risk_tolerance = config.risk_tolerance

timezone = pytz.timezone("EET") # set time zone to EET
symbol = config.symbol
timeframe = config.timeframe
bars = 500

trading_day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
trading_minute = ["00", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55"]

model_filename = 'D:/DH/projects/MetaNgepet/used_model-scaler/EURUSD_5/MetaNgepet_EURUSD_5_LinearReg_Model.pkl'
scaler_filename = 'D:/DH/projects/MetaNgepet/used_model-scaler/EURUSD_5/MetaNgepet_EURUSD_5_LinearReg_Scaler.pkl'

trade_check = False

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
    #print("\n", "account_info() as dataframe:")
    #print(Acc_Info)

#Extract Account info from dataframe
leverage = Acc_Info.loc[2, "value"]
balance = Acc_Info.loc[10, "value"]
profit = Acc_Info.loc[12, "value"]
equity = Acc_Info.loc[13, "value"]
margin_free = Acc_Info.loc[15, "value"]

"""
print(leverage)
print(balance)
print(profit)
print(equity)
print(margin_free)
"""
# extract info of terminal
terminal_info_dict = mt5.terminal_info()._asdict()
terminal_info = pd.DataFrame(list(terminal_info_dict.items()),columns=['property','value'])
print("\n", "terminal_info() as dataframe:")
#print(terminal_info)

# extract information from pair and timeframe
print ("------------------------------")
symbol_info = mt5.symbol_info(symbol)
if symbol_info!=None:
    # display the terminal data 'as is'    
    #print(symbol_info)
    print("{}:".format(symbol))
    #print("spread =",symbol_info.spread,", digits =",symbol_info.digits, "\n")
    print ("------------------------------")
else:
    print("There is no such symbol of {}".format(symbol))

def get_price(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars + 1)

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

def generate_signal(df_features, rates_frame, predictions):
    df_signal = pd.DataFrame(index=df_features.index)
    
    df_signal['predictions'] = predictions
    df_signal['predictions_5m'] = df_signal['predictions'].shift(1)
    df_signal['EMA_30m'] = TA.EMA(rates_frame, 6)
    df_signal['EMA_4h'] = TA.EMA(rates_frame, 48)
    df_signal['SAR'] = TA.SAR(rates_frame, 0.02, 0.2)
    df_signal['CCI_30m'] = TA.CCI(rates_frame, 6)

    #df_signal['avg_price_30m'] = rates_frame['close'].rolling(window=6).mean().shift(1)
    #df_signal['avg_price_4h'] = rates_frame['close'].rolling(window=48).mean().shift(1)

    df_signal['open'] = rates_frame['open']
    df_signal['open_5m'] = rates_frame['open'].shift(1)
    df_signal['close_5m'] = rates_frame['close'].shift(1)
    df_signal['high_5m'] = rates_frame['high'].shift(1)
    df_signal['low_5m'] = rates_frame['low'].shift(1)

    df_signal['lowest_30m'] = rates_frame['low'].rolling(window=6).min().shift(1)
    df_signal['highest_30m'] = rates_frame['high'].rolling(window=6).max().shift(1)

    return df_signal

def get_buy_condition(df_signal, now_datetime, digits):
    # get signal values
    prediction_value = round(df_signal.loc[now_datetime]["predictions"], digits)
    prediction_5m_value = round(df_signal.loc[now_datetime]["predictions_5m"], digits)
    
    ema_30m = round(df_signal.loc[now_datetime]["EMA_30m"], digits)
    ema_4h = round(df_signal.loc[now_datetime]["EMA_4h"], digits)
    sar = df_signal.loc[now_datetime]["SAR"]
    cci_30m = df_signal.loc[now_datetime]["CCI_30m"]

    lowest_30m = df_signal.loc[now_datetime]["lowest_30m"]

    open = round(df_signal.loc[now_datetime]["open"], digits)
    open_5m = round(df_signal.loc[now_datetime]["open_5m"], digits)
    close_5m = round(df_signal.loc[now_datetime]["close_5m"], digits)
    high_5m = round(df_signal.loc[now_datetime]["high_5m"], digits)
    #low_5m = round(df_signal.loc[now_datetime]["low_5m"], digits)

    if open > ema_4h and open > sar and ema_30m > ema_4h and open > lowest_30m and close_5m > open_5m and prediction_value > open and cci_30m > 100 :
        buy_condition = True
    else:
        buy_condition = False

    return buy_condition

def get_sell_condition(df_signal, now_datetime, digits):
    # get signal values
    prediction_value = round(df_signal.loc[now_datetime]["predictions"], digits)
    prediction_5m_value = round(df_signal.loc[now_datetime]["predictions_5m"], digits)
    ema_30m = round(df_signal.loc[now_datetime]["EMA_30m"], digits)
    ema_4h = round(df_signal.loc[now_datetime]["EMA_4h"], digits)
    sar = df_signal.loc[now_datetime]["SAR"]
    cci_30m = df_signal.loc[now_datetime]["CCI_30m"]

    highest_30m = df_signal.loc[now_datetime]["highest_30m"]

    open = round(df_signal.loc[now_datetime]["open"], digits)
    open_5m = round(df_signal.loc[now_datetime]["open_5m"], digits)
    close_5m = round(df_signal.loc[now_datetime]["close_5m"], digits)
    high_5m = round(df_signal.loc[now_datetime]["high_5m"], digits)
    low_5m = round(df_signal.loc[now_datetime]["low_5m"], digits)

    if open < ema_4h and open < sar and ema_30m < ema_4h and open < highest_30m and close_5m < open_5m and prediction_value < open and cci_30m < -100 :
        sell_condition = True
    else:
        sell_condition = False

    return sell_condition

def get_margin(symbol, lot, ask):
    action = mt5.ORDER_TYPE_BUY
    symbol = symbol
    margin = mt5.order_calc_margin(action, symbol, lot, ask)
    
    return margin

def get_lot(risk_tolerance, symbol, symbol_info, ask):
    volume_min = symbol_info.volume_min
    volume_max = symbol_info.volume_max
    volume_step = symbol_info.volume_step
    lot = volume_min
    margin_tolerance = margin_free * risk_tolerance

    margin = get_margin(symbol, lot, ask)
    while margin <= margin_tolerance and lot < volume_max:
        lot += volume_step
        margin = get_margin(symbol, lot, ask)
    return lot


#===================Main Loop===================#

while True:
    now = datetime.now(timezone)
    now_minute = now.strftime("%M")
    now_day = now.strftime("%A")
    now_datetime = now.strftime("%Y-%m-%d %H:%M") # Year-Month-Day Hour:Minute

    ask = mt5.symbol_info_tick(symbol).ask
    bid = mt5.symbol_info_tick(symbol).bid

    lot = get_lot(risk_tolerance, symbol, symbol_info, ask)

    if now_day in trading_day:
        if now_minute in trading_minute:
            order_amount = mt5.positions_get(symbol = symbol)
            if len(order_amount) == 0 and trade_check == False :
                print("analyzing.......")

                trade_check = True
                #sleep(5)

                # get neccesarry symbol info
                symbol_info = mt5.symbol_info(symbol)
                spread = symbol_info.spread
                digits = symbol_info.digits
                #deviation = (round((spread * pow(10, digits*(-1))), digits))
                deviation = spread

                # get few hundred past rates
                rates_frame = get_price(symbol, timeframe, bars)
                #print(rates_frame)
                now_index = now_datetime in rates_frame.index
                while now_index == False:
                    sleep(0.5)
                    rates_frame = get_price(symbol, timeframe, bars)
                    now_index = now_datetime in rates_frame.index

                # generate & scale feature
                features = generate_features(rates_frame)
                #print(features)
                features_scaled = scaler.transform(features)

                # make a prediction
                predictions = model.predict(features_scaled)

                # generate signals
                df_signal = generate_signal(features, rates_frame, predictions)
                #print(df_signal)

                prediction_value = round(df_signal.loc[now_datetime]["predictions"], digits)
                

                prediction_5m_value = round(df_signal.loc[now_datetime]["predictions_5m"], digits)
                ema_30m = round(df_signal.loc[now_datetime]["EMA_30m"], digits)
                ema_4h = round(df_signal.loc[now_datetime]["EMA_4h"], digits)
                sar = df_signal.loc[now_datetime]["SAR"]
                cci_30m = df_signal.loc[now_datetime]["CCI_30m"]
                highest_30m = df_signal.loc[now_datetime]["highest_30m"]
                lowest_30m = df_signal.loc[now_datetime]["lowest_30m"]
                open = round(df_signal.loc[now_datetime]["open"], digits)
                open_5m = round(df_signal.loc[now_datetime]["open_5m"], digits)
                close_5m = round(df_signal.loc[now_datetime]["close_5m"], digits)
                high_5m = round(df_signal.loc[now_datetime]["high_5m"], digits)
                low_5m = round(df_signal.loc[now_datetime]["low_5m"], digits)

                print('prediction value =',prediction_value)
                print('open = ', open)
                print('close_5m', close_5m)
                print('open_5m', open_5m)
                print('ema 30m =', ema_30m)
                print('ema_4h =', ema_4h)
                print('sar =', sar)
                print('cci_30m =', cci_30m)
                print('highest_30m =', highest_30m)
                print('lowest_30m =', lowest_30m)


                buy_condition = get_buy_condition(df_signal, now_datetime, digits)
                sell_condition = get_sell_condition(df_signal, now_datetime, digits)

                if buy_condition == True:
                    print("there are buy signal at {} \n".format(now_datetime))
                elif sell_condition == True:
                    print("there are sell signal at {} \n".format(now_datetime))
                else:
                    print("no signal yet, be patience! \n")
            elif len(order_amount) != 0 and trade_check == False:
                trade_check = True
                print("there is an order, no trade on {} just yet \n".format(symbol))       
        elif now_minute not in trading_minute :
            trade_check = False
            sleep(1)
    else:
        print("today is not a trade day, rest")
        sleep(1800)