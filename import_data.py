# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:24:52 2020

@author: Christophe
"""

# Processing
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from dateutil import parser

import time
import os.path


import warnings
warnings.filterwarnings("ignore")
pd.options.mode.use_inf_as_na = True

# BINANCE
# standard key to access the website
api_key = "fUSPSOSq6XtXF7LhioIDKVAxZFJqfZ3ZkFU3CychR1UbbqSmulx6ZasY5bcANoNR"
api_secret = "d4aHLi7ABVnIwiXEBo86wcmuIJ4fgsn4wrqbdMHRn5PHLICFWInZMMMJeRtCq188"
## Loading the necessary library and setup
import importlib
from binance.client import Client
client = Client(api_key, api_secret)




## Functions

def minutes_of_new_data(symbol, kline_size, data):
    
    # If data is already available
    if len(data) > 0:
        # kline = 1d : we get all new data
        if (kline_size == "1d"): old = parser.parse(data["timestamp"].iloc[-1])
        # kline = 1h : we get all new data except if previous data was saved more than 60 weeks ago
        elif (kline_size == "1h"):
            if datetime.today() - parser.parse(data["timestamp"].iloc[-1]) >= timedelta(weeks = 60):
                old = datetime.now() - timedelta(weeks = 60)
            else: old = parser.parse(data["timestamp"].iloc[-1])
        # kline = 15m : we get all new data except if previous data was saved more than 30 weeks ago
        elif (kline_size == "15m"):
            if datetime.now() - parser.parse(data["timestamp"].iloc[-1]) >= timedelta(weeks = 80):
                old = datetime.now() - timedelta(weeks = 80)
            else: old = parser.parse(data["timestamp"].iloc[-1])
        # kline = 5m : we get all new data except if previous data was saved more than 5 weeks ago
        elif (kline_size == "5m"):
            if datetime.now() - parser.parse(data["timestamp"].iloc[-1]) >= timedelta(weeks = 5):
                old = datetime.now() - timedelta(weeks = 5)
            else: old = parser.parse(data["timestamp"].iloc[-1])
        # kline = 1m : we get all new data except if previous data was saved more than 1 week ago
        elif (kline_size == "1m"):
            if datetime.now() - parser.parse(data["timestamp"].iloc[-1]) >= timedelta(weeks = 1):
                old = datetime.now() - timedelta(weeks = 1)
            else: old = parser.parse(data["timestamp"].iloc[-1]) 
     
    # If no data is available
    else:
        # 1d : all data is collected
        if (kline_size == "1d"): old = datetime.strptime('1 Jan 2017', '%d %b %Y')
        # 1h : all data starting 60 weeks ago is collected    
        elif (kline_size == "1h"): old = datetime.today() - timedelta(weeks = 60)
        # 15m : all data starting 30 weeks ago is collected    
        elif (kline_size == "15m"): old = datetime.today() - timedelta(weeks = 80)
        # 5m : all data starting 5 weeks ago is collected    
        elif (kline_size == "5m"): old = datetime.today() - timedelta(weeks = 5)
        # 1m : all data starting 1 weeks ago is collected    
        elif (kline_size == "1m"): old = datetime.today() - timedelta(weeks = 1)

    # Getting latest timestamp from binance        
    new = pd.to_datetime(client.get_klines(symbol=symbol, interval=kline_size)[-2][0], unit='ms')
    return old, new




def get_all_binance(symbol, kline_size, save = True):
    # Filename in folder
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    
    # Creating new df or creating df for file if present 
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    
    # Getting time bounds (oldest, newest) to collect data
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df)
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
        
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    
    if (available_data == 0): data = pd.DataFrame(columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    else:
        klines = client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
        data = pd.DataFrame(klines).iloc[:,0:6]
        data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
        
    data_df.set_index('timestamp', inplace=True)
    
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df



def clean_and_save(df, pair, kline_size):
    
    df.index = pd.DatetimeIndex(df.index)
    df = df.astype('float')
    
    # Duplicates
    duplicates_idx = df.index.duplicated(keep='last')
    df = df.loc[~duplicates_idx]

    # Build full index (from kline frequency)
    klines = ["1m","5m","15m","1h",'1d']
    klines_freq = ['1min','5min','15min','1H','1D']
    freq = klines_freq[klines.index(kline_size)]
    full_index = pd.date_range(start = df.index[0], end = df.index[-1], freq = freq)
    
    # Missing
    print('{} duplicates cleaned {} missing filled.'.format(duplicates_idx.sum(), df['close'].reindex(index = full_index).isna().sum()))
    df = df.reindex(index = full_index, method = 'nearest')
    df.index.name = 'timestamp'
    
    
    filename = '%s-%s-data.csv' % (pair, kline_size)
    df.to_csv(filename)
    return df



#        ---- Program start -----

## Parameters
binsizes = {"1m": 1, "5m": 5, "15m":15, "1h": 60, "1d": 1440}
batch_size = 750

klines = ["1m","5m","15m","1h",'1d']
klines_freqs = ['1min','5min','15min','1H','1D']

kline_size = '15m'
pairs = ['ETHBTC','BTCUSDT','ETHUSDT']


pairs_collection = {}
for pair in pairs:
    try: pairs_collection[pair] = get_all_binance(pair, kline_size)
    except:
        print('unable to connect to client: loading disk data')
        filename = '%s-%s-data.csv' % (pair, kline_size)
        pairs_collection[pair] = pd.read_csv(filename)


pairs_collection = {pair:clean_and_save(pairs_collection[pair], pair, kline_size) for pair in pairs_collection.keys()}