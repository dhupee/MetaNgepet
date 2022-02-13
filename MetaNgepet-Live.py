#Import needed modules

from matplotlib.pyplot import ticklabel_format
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from finta import TA
import pandas_ta

from datetime import datetime, timezone
from time import sleep
#import os
import pytz
import MetaTrader5 as mt5

import config

"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
"""

#===================================================================

#Trading Account Parameter
account = config.username   #Account number
password = config.password  #Password number
server = config.mt5_server  #Server name
path = config.mt5_path      #path of Metatrader5 director

risk_tolerance = config.risk_tolerance
rr_ratio = config.rr_ratio

timezone = pytz.timezone("EET") # set time zone to EET
symbol = config.symbol
timeframe = config.timeframe
bars = 200

trading_day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
trading_minute = ["00", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55"]

open_orders = [pd.DataFrame(columns=['Symbol', 'Position_ID', 'Lot'])]

#model_filename = 'used_model/Model_MetaNgepet_{}_{}.pkl'.format(symbol, timeframe)
#scaler_filename = 'used_model/Scaler_MetaNgepet_{}_{}.pkl'.format(symbol, timeframe)

trade_check = False

#===================================================================

#model = joblib.load(open(model_filename, 'rb'))
#scaler = joblib.load(open(scaler_filename, 'rb'))

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
    print("MetaTrader5 Initialized! \n")
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
#print("\n", "terminal_info() as dataframe:")
#print(terminal_info)

"""
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
"""

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


def generate_signal(rates_frame):
    """ Generate features for a stock/index/currency/commodity based on historical price and performance
    Args:
        df (dataframe with columns "open", "close", "high", "low", "volume")
    Returns:
        dataframe, data set with new features
    Timeframe for calculation (5 minutes timeframe):
        6 bars for 30 minutes
        48 bars for 4 hours
        144 bars for 12 hours
        288 bars for 1 day
    """

    df_signal = pd.DataFrame(index=rates_frame.index)

    df_signal['open'] = rates_frame['open']

    df_signal['open_1'] = rates_frame['open'].shift(1)
    df_signal['close_1'] = rates_frame['close'].shift(1)
    df_signal['high_1'] = rates_frame['high'].shift(1)
    df_signal['low_1'] = rates_frame['low'].shift(1)

    df_signal['lowest_30m'] = df_signal['low_1'].rolling(window=5).min()
    df_signal['highest_30m'] = df_signal['high_1'].rolling(window=5).max()

    # indicator
    CCI = TA.CCI(rates_frame, 6, 0.015)
    AO = TA.AO(rates_frame, 34, 5)

    df_signal['EMA_30m'] = TA.EMA(rates_frame, 6, 'close').shift(1)
    df_signal['EMA_1h'] = TA.EMA(rates_frame, 12, 'close').shift(1)
    df_signal['EMA_4h'] = TA.EMA(rates_frame, 48, 'close').shift(1)
    df_signal['CCI'] = CCI.shift(1)
    df_signal['CCI_1'] = CCI.shift(2)
    df_signal['AO'] = AO.shift(1)
    df_signal['AO_growth'] = AO.shift(1) - AO.shift(2)


    df_signal = df_signal.dropna(axis=0)

    return df_signal

def get_buy_condition(df_signal, now_datetime, digits):
    # get signal values
    ema_30m = df_signal.loc[now_datetime]["EMA_30m"]
    ema_1h = df_signal.loc[now_datetime]["EMA_1h"]
    ema_4h = df_signal.loc[now_datetime]["EMA_4h"]
    CCI = df_signal.loc[now_datetime]["CCI"]
    CCI_1 = df_signal.loc[now_datetime]["CCI_1"]
    AO = df_signal.loc[now_datetime]["AO"]

    #highest_30m = df_signal.loc[now_datetime]["highest_30m"]
    lowest_30m = df_signal.loc[now_datetime]["lowest_30m"]

    open = round(df_signal.loc[now_datetime]["open"], digits)
    open_5m = round(df_signal.loc[now_datetime]["open_1"], digits)
    close_5m = round(df_signal.loc[now_datetime]["close_1"], digits)

    if close_5m > ema_30m and ema_30m > ema_1h > ema_4h and open > lowest_30m and close_5m >= open_5m and CCI > 0 and CCI_1 <= 0 and AO > 0 :
        buy_condition = True
    else:
        buy_condition = False

    return buy_condition

def get_sell_condition(df_signal, now_datetime, digits):
    # get signal values
    ema_30m = df_signal.loc[now_datetime]["EMA_30m"]
    ema_1h = df_signal.loc[now_datetime]["EMA_1h"]
    ema_4h = df_signal.loc[now_datetime]["EMA_4h"]
    CCI = df_signal.loc[now_datetime]["CCI"]
    CCI_1 = df_signal.loc[now_datetime]["CCI_1"]
    AO = df_signal.loc[now_datetime]["AO"]

    highest_30m = df_signal.loc[now_datetime]["highest_30m"]
    #lowest_30m = df_signal.loc[now_datetime]["lowest_30m"]

    open = round(df_signal.loc[now_datetime]["open"], digits)
    open_5m = round(df_signal.loc[now_datetime]["open_1"], digits)
    close_5m = round(df_signal.loc[now_datetime]["close_1"], digits)

    if close_5m < ema_30m and ema_30m < ema_1h < ema_4h and open < highest_30m and close_5m <= open_5m and CCI < 0 and CCI_1 >=0 and AO < 0 :
        sell_condition = True
    else:
        sell_condition = False

    return sell_condition

def get_buy_reversal_condition(df_signal, now_datetime, digits):
    pass
    
def get_sell_reversal_condition(df_signal, now_datetime, digits):
    pass

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

def get_buy(symbol):
    pass

def get_sell(symbol):
    pass

def close_order(symbol, ticket):
    pass


#===================Main Loop===================#
"""
Note:
Sell uses bid price, while Buy uses ask price
"""


while True:
    now = datetime.now(timezone)
    now_minute = now.strftime("%M")
    now_day = now.strftime("%A")
    now_datetime = now.strftime("%Y-%m-%d %H:%M") # Year-Month-Day Hour:Minute

    if now_day in trading_day:
        if now_minute in trading_minute:
            
            # gonna add for loop here, for multiple symbol
            print("analyzing {}....".format(symbol))
            symbol_info = mt5.symbol_info(symbol)
            order_amount = mt5.positions_get(symbol = symbol)
            
            if len(order_amount) == 0 and trade_check == False :
                # get neccesarry symbol info
                ask = mt5.symbol_info_tick(symbol).ask
                bid = mt5.symbol_info_tick(symbol).bid
                lot = get_lot(risk_tolerance, symbol, symbol_info, ask)

                spread = symbol_info.spread
                digits = symbol_info.digits
                tolerance = (round((spread * pow(10, digits*(-1))), digits))
                deviation = spread

                # get few hundred past rates
                rates_frame = get_price(symbol, timeframe, bars)
                #print(rates_frame)
                now_index = now_datetime in rates_frame.index
                while now_index == False:
                    sleep(0.25)
                    rates_frame = get_price(symbol, timeframe, bars)
                    now_index = now_datetime in rates_frame.index
                df_signal = generate_signal(rates_frame)
                #print(df_signal)

                buy_condition = get_buy_condition(df_signal, now_datetime, digits)
                sell_condition = get_sell_condition(df_signal, now_datetime, digits)
                
                if buy_condition == True:
                    print("there are buy signal for {} at {} \n".format(symbol, now_datetime))
                    
                    open = round(df_signal.loc[now_datetime]["open"], digits)
                    highest_30m = df_signal.loc[now_datetime]["highest_30m"]
                    lowest_30m = df_signal.loc[now_datetime]["lowest_30m"]

                    buy_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": open,
                        "sl": lowest_30m - tolerance,
                        "tp": open + ((open - lowest_30m) * rr_ratio),
                        "deviation": deviation,
                        "magic": 42069666,
                        "comment": "python script open",
                        "type_time": mt5.ORDER_TIME_DAY,
                        "type_filling": mt5.ORDER_FILLING_RETURN,
                    }


                elif sell_condition == True:
                    print("there are sell signal for {} at {} \n".format(symbol, now_datetime))

                    sell_request = {}
                else:
                    print("no signal yet on {} yet, be patience! \n".format(symbol))
            elif len(order_amount) != 0 and trade_check == False:
                # make an if-else for stop trade

                print("there is an order, no trade on {} just yet \n".format(symbol))
            trade_check = True
        elif now_minute not in trading_minute :
            trade_check = False
            sleep(1)
    else:
        print("today is not a trade day, time to rest \n")
        sleep(1800) # 1800 Seconds = 30 Minutes