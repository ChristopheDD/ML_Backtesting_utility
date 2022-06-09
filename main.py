# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:20:01 2020

@author: Christophe
"""
# -- Imports --


# Processing
import pandas as pd
import numpy as np
import scipy
import math
from itertools import product
from datetime import datetime, timedelta
from dateutil import parser

import time
import os.path

# Modeling
import scipy.stats as scs
import scipy.signal as signal

# Machine Learning
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, \
                                    f_classif, mutual_info_classif, VarianceThreshold, SelectFwe
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, precision_score, make_scorer
from eli5.sklearn import PermutationImportance
from sklearn.decomposition import KernelPCA, PCA


# Data Viz & analysis
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
sns.set_context("talk")
sns.set_color_codes()
import pyfolio as pf
import ta #Technical analysis


# functions
from functions import pipe, generate_features, make_target_var, create_train_test_splits, features_filter, fit_results, State, model_returns, fit_transform_pca


##  --- Parameters ---

# Import params
kline_size = '15m'
pair_name = 'BTCUSDT'
horizon = 16

# Train set params
train_start = '2018-11-15 00:00:00'
test_start = '2019-01-01 00:00:00'
test_stop = '2020-05-30 00:00:00'

# Generate features params
n_smooth = 3
derivatives = ['d0','d1','d2']
n_range = np.unique(np.round(np.geomspace(2, 150, num = 35))).astype(int)
n_range_adi = np.unique(np.round(np.geomspace(2, 150, num = 35))).astype(int)
n_range_atr = np.unique(np.round(np.geomspace(2, 150, num = 35))).astype(int)

# Filter and fit params
filter_params = {'correl_thres': 0.80,
                'MI_thres': 0.003,
                }

pca_params = {'pca_type':'kernel',
              'n_comp_pca': 80}

indicators = ['awe_osc','stoch_osc','mfi','chaikin','atr','adx','macd_diff']

fit_params = {'percentiles':[0.25,0.75],
              'horizon':horizon,
               'split_len':8,
              'epoch_len':20,
              'n_features_to_select':90,
              'max_features':0.75
              
              }

# ------------ Main -------------


#1/ -- Import and process data for modeling --

filename = '%s-%s-data.csv' % (pair_name, kline_size)
pair = pd.read_csv(filename)
pair = pair.set_index('timestamp').asfreq('15T')

pair = pipe(pair, horizon = horizon)

#2/ -- Generate features --

train_features_start = pd.to_datetime(train_start)-timedelta(days = 10)
# Select pair
pair = pair.loc[train_features_start:test_stop]

## Momentum Indicators

# Awesome Osc
params = {'len': n_range}
indicator = ta.momentum.ao
indicator_name = 'awe_osc'
pair = generate_features(pair, indicator = indicator, indicator_name = 'awe_osc', columns = ['high','low'],
                        smoothing = n_smooth, derivatives = derivatives, params_grid = params)[0]

# Stoch_osc
params = {'n': n_range}
indicator = ta.momentum.stoch_signal
indicator_name = 'stoch_osc'
pair = generate_features(pair, indicator = indicator, indicator_name = 'stoch_osc', columns = ['high','low','close'],
                        smoothing = n_smooth, derivatives = derivatives, params_grid = params)[0]

## Volume Indicators

#Chaikin Money flow
params = {'n': n_range}
indicator = ta.volume.chaikin_money_flow
indicator_name = 'chaikin'
pair = generate_features(pair, indicator = indicator, indicator_name = 'chaikin', columns = ['high','low','close','volume'],
                        smoothing = n_smooth, derivatives = derivatives, params_grid = params)[0]


# MFI
params = {'n': n_range}
indicator = ta.volume.money_flow_index
indicator_name = 'mfi'
pair = generate_features(pair, indicator = indicator, indicator_name = 'mfi', columns = ['high','low','close','volume'],
                        smoothing = n_smooth, derivatives = derivatives, params_grid = params)[0]


## Volatility Indicators

# ATR
params = {'n': n_range_atr}
indicator = ta.volatility.average_true_range
indicator_name = 'atr'
pair = generate_features(pair, indicator = indicator, indicator_name = 'atr', columns = ['high','low','close'],
                         derivatives = derivatives, smoothing = n_smooth, params_grid = params)[0]


## Trend Indicators

# ADI
params = {'n': n_range_adi}
indicator = ta.trend.adx
indicator_name = 'adx'
pair = generate_features(pair, indicator = indicator, indicator_name = 'adx', columns = ['high','low','close'],
                         derivatives = derivatives, smoothing = n_smooth, params_grid = params)[0]


# MACD
n_range_fast = np.unique(np.round(np.geomspace(5, 30, num=10))).astype(int)
n_range_slow = np.unique(np.round(np.geomspace(35, 100, num=5))).astype(int)
params = {'n_fast': n_range_fast,
          'n_slow': n_range_slow}

indicator = ta.trend.macd_diff
indicator_name = 'macd_diff'
pair = generate_features(pair, indicator = indicator, indicator_name = 'macd_diff', columns = ['close'],
                         derivatives = derivatives, smoothing = n_smooth, params_grid = params)[0]


#2/ -- Run model --

# Instanciate filter
filter1 = features_filter(indicators, correl = True, MI = True, pca = None)

# Choose classifier
clf = SVC(class_weight = 'balanced', kernel = 'sigmoid')

# Run model
y_pred, target, fin_feats = fit_results(pair, clf, filter1, filter_params, pca_params,
                             train_start, test_start, test_stop, **fit_params)[0:3]

idx = pair.loc[test_start:test_stop].index[:-1]
y_pred = pd.Series(y_pred, index = idx)
preds_df = pd.DataFrame(y_pred, columns = ['y_pred'])
preds_df['target'] = target

feats_df = pd.DataFrame(fin_feats)

#Save predictions_df as csv for further plotting and analysis
    
preds_df.to_csv(str(pair_name)+'_'+kline_size+'_'+test_start[0:10]+'_'+test_stop[0:10]+'_.csv')
feats_df.to_csv(str(pair_name)+'_'+kline_size+'_'+test_start[0:10]+'_'+test_stop[0:10]+'_features'+'_.csv')

#%%

# y_pred = preds_df['n_pca110']
# returns = (pair['close']/pair['close'].shift(1))-1
# returns = returns.loc[test_start:test_stop][:-1]
# cum_ret = np.cumprod(1+returns)

# bal = model_returns(y_pred,returns,horizon = 16)[0]
# bal = pd.Series(list(bal), returns.index)

# #plt.plot(model_returns(predictions,returns)[0], 'r-')
# plt.figure(figsize = (16,8))
# plt.plot(pd.Series(cum_ret.values, returns.index), 'b-', label = 'BTCUSDT returns')
# plt.plot(bal, 'r-', label = 'Porfolio returns (1)')
# plt.xticks(rotation=45)
# plt.grid()
# plt.legend()
# plt.title('Porfolio vs BTCUSDT (02-15 -> 05-15)')
