# -*- coding: utf-8 -*-
"""
Created on Mon May 18 10:20:29 2020

@author: Christophe
"""

# Processing
import pandas as pd
import numpy as np
import scipy
import math
from itertools import product, groupby
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
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFromModel, \
                                    f_classif, mutual_info_classif, VarianceThreshold, SelectFwe, RFECV, RFE
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, precision_score, make_scorer
from eli5.sklearn import PermutationImportance
from sklearn.decomposition import KernelPCA, PCA


# Data Viz & analysis
from scipy.stats import skew, kurtosis, gmean
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
sns.set_context("talk")
sns.set_color_codes()
import pyfolio as pf
import ta #Technical analysis


import warnings
warnings.filterwarnings("ignore")
pd.options.mode.use_inf_as_na = True

def pipe(dataframe, horizon = 12):
    """ Calculates returns based on C(t+n)/C(t)-1 where n is the time horizon used for prediction"""
    
    df = dataframe.copy()
    #### Process date to index
    df = df.fillna(df.mean())
    
    #### daily return
    ret = list(np.full(shape = horizon, fill_value = np.nan))
    for i in range(horizon, df.shape[0]):
        ret.append((df['close'][i]/df['close'][i-horizon]-1))
        
    df['return'] = ret
    df['return'] = df['return'].shift(-horizon)
    
    df = df.iloc[:-horizon]
    return df

def generate_features(dataframe, indicator, indicator_name, columns,
                          params_grid, derivatives = ['d0','d1','d2'], smoothing = 3):
    '''From technical indicators and the corresponding parameter grid, this function generates a feature
    returning the indicator value for each combinaison of parameters'''
    
    df = dataframe.copy()
    
    def make_params(params_grid):
        keys, values = zip(*params_grid.items())
        params_sets = [dict(zip(keys, v)) for v in product(*values)]
        return params_sets
    
    params_sets = make_params(params_grid)
    for i in range(0,len(params_sets)):
        parameter_names = list(params_sets[i].keys())
        
        if len(params_sets[0]) == 1:
            parameter_names = list(params_sets[i].keys())
            name = indicator_name+'_{}={}_d0'.format(parameter_names[0],params_sets[i][parameter_names[0]])
        
        elif len(params_sets[0]) == 2:
            name = indicator_name+'_{}={}_{}={}_d0'.format(parameter_names[0],params_sets[i][parameter_names[0]],
                                                   parameter_names[1],params_sets[i][parameter_names[1]])
       
        elif len(params_sets[0]) == 3:
            name = indicator_name+'_{}={}_{}={}_{}={}_d0'.format(parameter_names[0],params_sets[i][parameter_names[0]],
                                                              parameter_names[1],params_sets[i][parameter_names[1]],
                                                              parameter_names[2],params_sets[i][parameter_names[2]])
        
        cols_dict = {}
        for col in columns:
            cols_dict[col] = df[col].reset_index(drop = True)

        ind = indicator(**cols_dict, **params_sets[i])
        df[name] = ind.values
        
        #df[name+'_ewm'] = df[name].ewm(horizon).mean()  #ewm
        if 'd1' in derivatives:
            df[name[:-3]+'_d1'] = (df[name] - df[name].shift(1)).ewm(smoothing).mean() #first derivative
        if 'd2' in derivatives:
            df[name[:-3]+'_d2'] = (df[name[:-3]+'_d1'] - df[name[:-3]+'_d1'].shift(1)).ewm(smoothing).mean() #second derivative
        
    df = df.fillna(method = 'backfill')
    
    return df, indicator

def make_target_var(df, percentiles, start, end):
    min_limit = np.log(1+df['return'].loc[start:end]).describe(percentiles = percentiles)[4]
    max_limit = np.log(1+df['return'].loc[start:end]).describe(percentiles = percentiles)[6]
    limit = gmean([np.abs(min_limit),max_limit])
        
    def classifier_variable(x, limit):
        if (x > limit): class_var = 1
        elif (x < -limit): class_var = -1
        else: class_var = 0
    
        return class_var
    
    clf_target = df['return'].apply(lambda x : classifier_variable(x, limit))
    
    print("min limit : {} - max limit : {}".format(min_limit,max_limit))
    print('limit : {}'.format(limit))
    
    return clf_target

def create_train_test_splits(df, train_start, test_start, test_stop, horizon = 12, split_len = 12):
    """ Creates rolling n-long periods train test splits used for validation of the predictive model
    
    For example, if the test set is 60 days long on daily data, for:
    n = 1 there will be 60/1 = 60 total splits, first split on the first day of the test set, second split on the second day etc ..
    n = 3 there will be 60/3 = 20 total splits, first split on the first 3 days, second split on days 3 to 6, etc
    n = 60, there will be 60/60 = only one split, and the full month is tested at once
    
    Returns iterable object containing train/test splits
    """

    train_start_idx = df.index.get_loc(train_start)
    test_start_idx = df.index.get_loc(test_start)
    test_stop_idx = df.index.get_loc(test_stop)
    n_periods = test_stop_idx - test_start_idx
    if (n_periods % split_len  != 0):
        raise ValueError("Please enter valid split length (totalperiods = {})".format(n_periods))
    
    
    train_test_splits = (([*range(train_start_idx+(i-test_start_idx),i-horizon)],[*range(i,i+split_len)]) for i in range(test_start_idx, test_stop_idx, split_len))
    
    print('Train set : from {0} to {1}, {2} samples'.format(train_start, df.index[test_start_idx-horizon],
                                                            len(range(train_start_idx, test_start_idx-horizon))))
    print('Test set : from {0} to {1}, {2} samples'.format(test_start, test_stop,
                                                           len(range(test_start_idx, test_stop_idx))))
    n_splits = n_periods // split_len
    return train_test_splits, n_splits


class features_filter:
    
    def __init__(self, indicators, derivatives = ['d0','d1','d2'], MI = True, correl = True, pca = None):
        self.MI = MI
        self.correl = correl
        self.indicators = indicators
        self.derivatives = derivatives
    
    def filter_by_correl(self, dataframe, train_start, train_stop, features, correl_thres = 0.95):
        """First selection removing collinear features, for each indicator and each derivative"""
        
        # On the test set, the target is not known a priori, so we select the the train set where we evaluate features correlation with tgt
        df = dataframe.loc[train_start:train_stop]
        
        #calculate correlation matrix and sort features by correlation with trg
        corr = df[features+['clf_target']].corr()
        corr_with_target = np.abs(corr['clf_target']).sort_values(ascending = False)[1:] 
        # Reduce matrix to contain only features with >0.01 corr with trg
        correl_tg_feats = list(corr_with_target[corr_with_target > 0.001].index)
        corr = corr[correl_tg_feats+['clf_target']].loc[correl_tg_feats+['clf_target']]
        
        
        # find features validating corr > thres condition with at least 1 other feature (not itself)
        corr_mat_thres = corr[np.abs(corr) > correl_thres]
        features_highlycorreled = corr_mat_thres[corr_mat_thres.notna().sum() >= 2].index.to_list() 
        corr_mat_thres = corr_mat_thres[features_highlycorreled].loc[features_highlycorreled]
        feat_dropped = []
        
        # Start loop on all corr_mat features 
        for feature in corr_mat_thres.columns:
            if feature not in feat_dropped:
                
                # For each feature, select others feats where corr >0.95, i.e strongly correlated to it
                features_corr = corr_mat_thres[feature].sort_values(ascending = False).dropna()[1:]
                multicorr_features = list(corr_mat_thres[feature].sort_values(ascending = False).dropna()[1:].index)
                corr_tg = np.abs(corr['clf_target'].loc[multicorr_features]) #then calculate corr to tgt
                #print(corr_tg)

                if corr_tg.empty != True:
                    kept = corr_tg.idxmax() #For strongly corr features, only keep the one with strongest corr to tgt
                    #print('to keep: {}'.format(kept))
                    multicorr_features.remove(kept)

                    # Reduce corr_matrix by removing bad features
                    corr_mat_thres = corr_mat_thres.drop(multicorr_features, axis = 1).drop(multicorr_features, axis = 0)
                    #print(len(corr_mat_thres))
                    feat_dropped.extend(multicorr_features) # Add to list of features dropped
            
        # final features to keep, should be perfectly uncorrelated with each other
        features_uncorr = [feat for feat in correl_tg_feats if feat not in feat_dropped]
        
        return features_uncorr
    
    def filter_by_MI(self, dataframe, train_start, train_stop, features, feats_after_correl, MI_thres = 0.005):
        """Second selection using MI (mutual importance) filtering technique for each indicator and each derivative """
        
        # On the test set, the target is not known a priori, so we select the the train set where we evaluate features MI with tgt
        df = dataframe.loc[train_start:train_stop]
        
        # If a MI filter is applied, we use previous features
        if feats_after_correl != []:
            feats = feats_after_correl
        else:
            feats = features
        
        # Calculate MI score between features and target
        MI = mutual_info_classif(df[feats], df['clf_target'], n_neighbors = 3)
        scores_MI = pd.Series(MI, index = feats)
        
        features_MI = scores_MI[scores_MI > MI_thres].index.to_list()
        
        return features_MI  
    
    def selector(self, df, train_start, train_stop, correl_thres = 0.85, MI_thres = 0.005, n_comp_pca = 50):
        """Chains features selection techniques 1) Correlation and 2) MI for each feature and each derivative"""
        
        # Count all feats
        all_features = []
        for ind in self.indicators:
            for der in self.derivatives:
                all_features = all_features + [col for col in df.columns if ind in col and der in col]
        
        print('{} features are selected for filtering'.format(len(all_features)))
        
        # Looping on indicators and derivatives
        features_to_keep = []
        for indicator in self.indicators:
            feats_ind = [feat for feat in df.columns if indicator in feat]
            n_features_removed = 0
            
            for derivative in self.derivatives:
                current_features = [feat for feat in df.columns if indicator in feat and derivative in feat]
                feats_after_correl = []
                feats_after_MI = []
                
                if self.correl == True:    
                    feats_after_correl = self.filter_by_correl(df, train_start, train_stop, current_features, correl_thres = correl_thres)
                else: feats_after_correl = current_features
                
                if self.MI == True:
                    feats_after_MI = self.filter_by_MI(df, train_start, train_stop, current_features, feats_after_correl, MI_thres)
                else: feats_after_MI = feats_after_correl
                    
                n_features_removed += len(current_features)-len(feats_after_MI)
                features_to_keep.extend(feats_after_MI)
                
            print('{}/{} features kept for {} indicator'.format(len(feats_ind)-n_features_removed,len(feats_ind),indicator))

        print('{}/{} features total kept'.format(len(features_to_keep),len(all_features)))
        return features_to_keep

def fit_transform_pca(dataframe, train_start, train_stop, pca_type, n_comp_pca):
        """Principal components analysis (PCA) on all aggregate features after filtering"""
        
        # We select the train set to fit pca
        df = dataframe.loc[train_start:train_stop]
        
        if pca_type == 'linear':
            pca = PCA(n_components = n_comp_pca)
        elif pca_type == 'kernel':
            pca = KernelPCA(n_components = n_comp_pca, kernel = 'rbf')
        
        # Name columns and apply pca to relevant features, then add new features to df
        cols_pca = ['pca_'+str(i+1) for i in range(0,n_comp_pca)]
        pca_transformer = pca.fit(df)
        df_pca = pd.DataFrame(pca_transformer.transform(dataframe), index = dataframe.index, columns = cols_pca)
        
        return df_pca

def fit_results(df, clf, feat_filter, filter_params, pca_params, train_start, test_start, test_stop, n_features_to_select = 95, max_features = 0.8,
                   horizon = 16, split_len = 12, epoch_len = 16, percentiles = [0.25,0.75],
                pca = False, get_importances = False):
    
    print('Fitting model {}...'.format(type(clf).__name__))
    start = time.time()    
    
    predictions = []
    targets = []
    scores = []
    importances = []
    split_number = 0
    fin_feats = []
    
    def custom_scorer(y_true,y_pred):
        """ Custom scorer that penalize the scorer for -1->+1 or +1->-1 missclassifications
        parameters : w : additional sample weight value
        """
        w = 1.5       
        
        def sample_weight(y_true, y_pred):
            if (y_pred == -1 and y_true == 1) or (y_pred == 1 and y_true == -1):
                weight = w
            else: weight = 1
            return weight
    
        weights = [sample_weight(y_true,y_pred) for (y_true,y_pred) in zip(y_true,y_pred)]
        
        accuracy = f1_score(y_true,y_pred, average = 'weighted', sample_weight = weights)
    
        return accuracy
    
    train_test_splits, n_splits = create_train_test_splits(df, train_start, test_start, test_stop,
                                                 horizon, split_len = split_len)
    
    print('Total trainings or number of splits: {}'.format(n_splits))
    n_epochs = n_splits // epoch_len + 1
    print('Total epochs: {}'.format(n_epochs))
    
    for train_index, test_index in train_test_splits:
        
        if split_number % epoch_len == 0:
            # Recalculate classifier target + filter features + pca for each new epoch based on first training set
            print('\n')
            print('---- Epoch {}/{} ----'.format(split_number // epoch_len + 1, n_epochs))
            
            #get limit timestamps for current epoch
            train_start_epoch = df.index[train_index[0]]
            train_stop_epoch = df.index[train_index[-1]]
            test_start_epoch = df.index[test_index[0]]
            
            if split_number // epoch_len + 1 == n_epochs : # for last epoch
                test_stop_epoch = df.index[-1]
            else: # all other epochs end after split_len*epoch_len periods 
                test_stop_epoch = df.index[test_index[-1]+split_len*(epoch_len-1)]
            
            print("Epoch end: {}".format(test_stop_epoch))
            
            df_epoch = df.loc[train_start_epoch:test_stop_epoch] #select current epoch 
            df_epoch['clf_target'] = make_target_var(df, percentiles, train_start_epoch, train_stop_epoch) #calculate target
            
            # Filter features
            features = feat_filter.selector(df_epoch, train_start_epoch, train_stop_epoch, **filter_params)
            x = df_epoch[features]
            scaler = PowerTransformer(method = 'yeo-johnson')
            x = pd.DataFrame(scaler.fit_transform(x), index = x.index, columns = x.columns)
            
            # apply pca
            if pca == True:
                print('Applying_pca...')
                x = fit_transform_pca(x, train_start_epoch, train_stop_epoch, **pca_params)
            
            print('Fitting model...')
    
        #get limit timestamps for current split
        train_start_split = df.index[train_index[0]]
        train_stop_split = df.index[train_index[-1]]
        test_start_split = df.index[test_index[0]]
        test_stop_split = df.index[test_index[-1]]
        
        #RFE
        selector = RFE(SGDClassifier(loss = 'log', n_jobs = -1, class_weight = 'balanced'), step = 5, n_features_to_select = n_features_to_select)
        selector = selector.fit(x.loc[train_start_epoch:train_stop_epoch], df_epoch['clf_target'].loc[train_start_epoch:train_stop_epoch])
        fin_feats_i = [x.columns[i] for i in range(0, len(x.columns)) if selector.support_[i] == True]
        fin_feats.append(fin_feats_i)
        x = pd.DataFrame(data = selector.transform(x), index = df_epoch.index, columns = fin_feats_i)
        
        ## Target prediction (-1,0,1) using all remaining features and classifier
        ensemble = BaggingClassifier(base_estimator = clf, n_estimators = 15, max_samples = 0.8, max_features = max_features, n_jobs = -1)
        clf_results = ensemble.fit(x.loc[train_start_split:train_stop_split], df_epoch['clf_target'].loc[train_start_split:train_stop_split])
        prediction = clf_results.predict(x.loc[test_start_split:test_stop_split])
        targets.extend(df_epoch['clf_target'].loc[test_start_split:test_stop_split])
        predictions.extend(prediction)
        
        if get_importances == True:
            perm = PermutationImportance(clf_results, scoring = make_scorer(custom_scorer), n_iter = 3)\
                                    .fit(x.iloc[test_index],df['clf_target'].iloc[test_index]).feature_importances_
            importances.append(perm)
            
        split_number += 1
    
    conf_mat = confusion_matrix(targets,predictions)
    score = accuracy_score(targets, predictions)
    precision = precision_score(targets,predictions, average = 'weighted')
    f1 = f1_score(targets,predictions, average = 'weighted')
    pen_f1 = custom_scorer(targets,predictions)
    importances = pd.DataFrame(importances, columns = x.columns)
    
    
    #print(sample_weights)
    end = time.time()
    print("Completed in: {:.0f}min {:.2f}sec ".format((end - start)//60, (end- start)%60))
    print(' -Accuracy: {:.5f} -Precision: {:.5f}'.format(score, precision))
    print(' -f1-score: {:.5f} -penalized f1-score: {:.5f}'.format(f1, pen_f1))
    
    print(conf_mat)
    
    return predictions, targets, fin_feats, score, pen_f1, conf_mat, importances


class State:
    """Object containing all properties about current trading position on the asset(-1 short, 0 wait, 1 long).
    Can be activated to enter, leave, or switch positions based on the above model forecast. 
    A maximum loss safety exit is implemented"""
    fees = 0.001
    reverse_counter_max = 12
    max_loss = 0.01
    
    
    def __init__(self, balance, horizon):
        self.balance = balance
        self.horizon = horizon
        self.position = 0
        self.counter = 0
        self.reverse_counter = 0
        self.cum_ret = 1
  
    
    def reset_counters(self): # Resets counters
        self.reverse_counter = 0
        self.counter = 0
        self.cum_ret = 1
        
    
    def leave_pos(self, ret): #leave position and update balance, position
        bal = self.balance
        pos = self.position
        
        self.reset_counters()
        
        self.balance = bal*(1+ret*pos)*(1-self.fees)
        self.position = 0
        
    
    def enter_pos(self, new_pos):
        bal = self.balance
        pos = self.position
        
        self.reset_counters()
        self.counter = 1
        self.balance = bal*(1-self.fees)
        self.position = new_pos
        
    def switch_pos(self, ret):
        bal = self.balance
        pos = self.position

        self.reset_counters()
        
        self.counter = 1
        self.balance = bal*(1+ret*pos)*(1-self.fees)**2
        self.position = -pos
        
    def remain_pos(self, pred, ret):
        bal = self.balance
        pos = self.position
        
        self.cum_ret *= (1+ret*pos)
        
        if (pred == -pos):
            self.reverse_counter +=1
            
        if (self.cum_ret <= 1-self.max_loss): # when losses > max_loss
            self.leave_pos(ret) # leave position and wait for next 
        
        #if more than x predictions indicate opposite trend (i.e. downward trend prediction when during long position)
        elif (self.reverse_counter >= self.reverse_counter_max): #if more than x predictions indicate opposite trend (i.e. oppos)
            self.switch_pos(ret) #Switch positions
        
        else:
            self.counter += 1
            self.balance = bal*(1+ret*pos)
            
def model_returns(pred,returns,horizon = 16, roll_span = 9, long_thres = 0.8, short_thres = -0.8, report = False):
    
    
    weighted_preds = pred.ewm(span = roll_span).mean()
    
    full_period = len(pred)
    positions = pd.Series(index=range(0,len(pred)))
    balance = pd.Series(index=range(0,len(pred)))
    state = State(balance = 1, horizon = horizon)
    
    
    for i in range(0,full_period):
        if i == 0:
            if pred[0] == 1:
                state.enter_pos(1)
                balance[0], positions[0] = state.balance, state.position
            if pred[0] == 0:
                balance[0], positions[0] = 1,0
            if pred[0] == -1:
                state.enter_pos(-1)
                balance[0], positions[0] = state.balance, state.position       
        else:
            if positions[i-1] == 0:
                if weighted_preds[i] >= long_thres:
                    state.enter_pos(1)
                if weighted_preds[i] <= short_thres:
                    state.enter_pos(-1)
                balance[i], positions[i] = state.balance, state.position
                
            if positions[i-1] == -1:
                if state.counter < horizon:
                    state.remain_pos(pred[i],returns[i])
                else:
                    if (weighted_preds[i] >= 0) and (weighted_preds[i] <= long_thres):
                        state.leave_pos(returns[i])
                    if weighted_preds[i] >= long_thres:
                        state.switch_pos(returns[i])
                    if weighted_preds[i] <= 0:
                        state.reset_counters()
                        state.remain_pos(pred[i],returns[i])     
                balance[i], positions[i] = state.balance, state.position
                
            if positions[i-1] == 1:
                if state.counter < horizon:
                    state.remain_pos(pred[i],returns[i])
                else:
                    if (weighted_preds[i] <= 0) and (weighted_preds[i] >= short_thres):
                        state.leave_pos(returns[i])
                    if weighted_preds[i] <= short_thres:
                        state.switch_pos(returns[i])
                    if weighted_preds[i] >= 0:
                        state.reset_counters()
                        state.remain_pos(pred[i],returns[i])     
                balance[i], positions[i] = state.balance, state.position
        
    cum_ret = np.cumprod(1+returns.values)
    print('total periods : {}'.format(len(positions)))
    print('reference +- : {:.2f}%'.format((cum_ret[-1]-1)*100))
    print('model +-: {:.2f}%'.format((balance.iloc[-1]-1)*100))

    if report == True:
        
        counts = positions.value_counts()
        short_prc = np.round(counts[-1]/counts.sum()*100, decimals = 1)
        long_prc = np.round(counts[1]/counts.sum()*100, decimals = 1)
        idle_prc = np.round(counts[0]/counts.sum()*100, decimals = 1)
        r_long_short = np.round(counts[1]/counts[-1], decimals = 2)
            
        pos_groups = [list(group) for key, group in groupby(positions.values.tolist())]
        longs = [grp for grp in pos_groups if 1 in grp]
        shorts = [grp for grp in pos_groups if -1 in grp]
        waits = [grp for grp in pos_groups if -1 in grp]

        longs_taken = len(longs)
        avg_long = np.round(np.mean([len(ser) for ser in longs]),2)

        shorts_taken = len(shorts)
        avg_short = np.round(np.mean([len(ser) for ser in shorts]),2)
        
        n_trades = 2*(longs_taken + shorts_taken)
        
        report_df = pd.DataFrame([long_prc, short_prc, idle_prc, r_long_short, longs_taken, avg_long, shorts_taken, avg_short, n_trades], columns = ['stats'],
                                  index = ['Long %', 'Short %', 'Wait %', 'Long to short ratio', 'Longs taken', 'Average long', 'Shorts taken', 'Average short', 'N trades']).transpose()
        display(report_df)
        
        
        
    return balance, cum_ret, positions, weighted_preds, pred

