# Major rebuild ~~coming soon~~ is on the way
## To-do list:
* [x] adding config.py file setting (try to make it proper if i want to make it public)
* [x] using basic regression from sklearn
* [x] modifying the feature function for my need and timeframe
* [x] making signal function for trading bot
* [ ] try make a proper bot by combining it with regression model
* [ ] investigate `one code for all` vs `single ticker single code`
* [x] learn how to make proper Readme (I guess it's good enough)

# How to Use This Bot?
"coming soon hehe"

# Dependencies
Just execute this command or uncomment the command on the first cell of the Training Notebook
```
pip install -U MetaTrader5 matplotlib numpy pandas pandas_ta pickle scikit-learn
```

# Config.py Template
```
coming soon, promise
```

# Small Problem I've encounter
* This problem is a minor yet sometimes annoying for me, config for jupyter isn't running for the `timestamp` part, should be a problem but could be a inconvinince.

* wrong timeframe setting can break the scaler with `ValueError: Input contains infinity or a value too large for dtype('float64')` Error.

* making a simple backtest just to see if it's useful.

* somehow I have to delay my code by 2.5 Sec due to MT5 terminal too late to give new candlestick 

# Reference
* https://www.youtube.com/watch?v=AXBhrLongC8
* https://www.w3schools.com/python/python_datetime.asp
* https://www.mql5.com/en/docs/integration/python_metatrader5
* https://scikit-learn.org/stable/modules/classes.html
* https://github.com/peerchemist/finta
* https://www.kaggle.com/manohar676/forex-eurusd-predictive-model
