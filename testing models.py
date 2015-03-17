# -*- coding: utf-8 -*-
"""
Created on Fri Mar 06 09:14:36 2015

@author: p_cohen
"""

import sys
sys.path.append("C:/Git_repos/telematicsPy0")
#some of these are my standard libraries, may not be used
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model.ridge import Ridge
from sklearn.metrics import auc, roc_curve
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.linear_model.ridge import RidgeCV
#from sklearn.linear_model import RandomizedLasso
#from sklearn import svm
import numpy as np
import time
#from sklearn.feature_selection import SelectKBest, f_regression
#import sklearn as skl
#from sklearn.feature_extraction import DictVectorizer
#import gc
#import random
import os
#import pickle
import prepfuncs as prp
import gc
import varbuildingfuncs as varb
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

########################## Functions ##################################
def analyze_trips(df_triplevel, features):
    """ Creates and saves predictions for all drivers using random forest
    
    Input: trip level statistics for a set of drivers
    
    Output: trip level statistics with driver id predictions added
    """
    # Read name of each driver file (will be numeric) and save to a list
    drivers = df_triplevel['driver'].drop_duplicates()
    # initialize prediction column
    df_triplevel['prds'] = .5
    for driver in drivers:
        if int(driver)%100 == 0:
            print "We are modeling driver " + str(driver)
        # make analysis file with all driver trips & some random other trips
        df_modeldrvr = create_modeldrvr(df_triplevel, driver)
        # create in and out sample
        df_modeldrvr['rand'] = pd.Series(np.random.rand(len(df_modeldrvr)), 
                                         index=df_modeldrvr.index)
        df_drvr_ins = df_modeldrvr[df_modeldrvr['rand'] > .3]
        df_drvr_out = df_modeldrvr[df_modeldrvr['rand'] <= .3]
        # fit models
        # forest
        forst = RandomForestClassifier(n_estimators=4000, n_jobs=8,
                                      max_depth=30)
        forst.fit(df_drvr_ins[features], df_drvr_ins['target'])
        forst2 = RandomForestClassifier(n_estimators=8000, n_jobs=8,
                                      max_depth=15)
        forst2.fit(df_drvr_ins[features], df_drvr_ins['target'])
        # boosting
        boost = GradientBoostingClassifier(learning_rate=0.01,
                                           n_estimators=700,  
                                           max_depth=3,
                                           subsample = .8)
        boost.fit(df_drvr_ins[features], df_drvr_ins['target']) 
        boost2 = GradientBoostingClassifier(learning_rate=0.005,
                                           n_estimators=700,  
                                           max_depth=3,
                                           subsample = .8)
        boost2.fit(df_drvr_ins[features], df_drvr_ins['target']) 
        # measure model success
        model = [forst, forst2, boost, boost2]
        model_nm = ['forst', 'forst2', 'boost', 'boost2'] 
        for x in range(1, len(model)):
            name = model_nm[x] + 'preds'
            val = model[x].predict_proba(df_drvr_out[features])[:, 1]
            df_drvr_out[name] = val
            fpr, tpr, thresholds = roc_curve(df_drvr_out[name],
                                             df_drvr_out['target'], pos_label=2)
            auc_val = auc(fpr, tpr) 
            print "For " + str(model_nm) + " auc is " + str(auc_val)

def create_modeldrvr(df_triplevel, driver):
    """
    This creates a file of all trips from the driver of interest and
    a random selection trips from other drivers
    
    Input: file with all drivers in it, and a driver to select
    
    Output: file containing all trips from chosen driver and a randomly 
    selected set of other trips (between 300-500 random trips)
    """
    # Pull all trips from one driver
    df_driveranalysis = df_triplevel[df_triplevel['driver'] == driver]
    # count number of trips
    num_trips = len(df_triplevel['driver'])
    # Create random vector of same length as data
    df_triplevel['rndm'] = pd.Series(np.random.rand(num_trips), 
                                     index=df_triplevel.index)
    # Choose % of other trips to include in data 
    pct_of_othertrips = .006
    if num_trips < 2000: # this is for tests using small data
        pct_of_othertrips = .5
    # pull selection of other trips
    df_othertrips = df_triplevel[((df_triplevel['driver'] != driver) &
                                  (df_triplevel['rndm'] < pct_of_othertrips))]
    # append driver trips and random trips
    df_driveranalysis = df_driveranalysis.append(df_othertrips)
    # create modeling target using the driver id
    df_driveranalysis['target'] = df_driveranalysis['driver'] == driver
    return df_driveranalysis

def len_dist_tripmatch(df):
    """ 
    uses rounded final distance and trip length to create a basic trip
    matching variable
    """
    # create a separate data set so that we dont re-sort our primary one
    # this may not be neccesary but is a good practice
    df_trpmtch = pd.DataFrame(df[['driver', 'trip', 'amt_trn',
                                     'fin_dist', 'tot_dist']])
    # we want to round the final distance to try to work around the fuzziness
    # created by the tournament organizers
    df_trpmtch['rnd_fin_dist'] = (df_trpmtch['fin_dist'] * .01).round()
    df_trpmtch['rnd_tot_dist'] = (df_trpmtch['tot_dist'] * .1).round()
    # sort the data set by rounded final distance first, then by tot dist
    df_trpmtch.sort(['rnd_fin_dist', 'rnd_tot_dist', 'amt_trn'], inplace=True)
    # create lags of adjacent drivers
    df_trpmtch['driver1'] = df_trpmtch['driver'].shift(periods=1)
    df_trpmtch['driver2'] = df_trpmtch['driver'].shift(periods=2)
    df_trpmtch['driver3'] = df_trpmtch['driver'].shift(periods=3)
    df_trpmtch['driver4'] = df_trpmtch['driver'].shift(periods=4)
    df_trpmtch['driver5'] = df_trpmtch['driver'].shift(periods=-1)
    df_trpmtch['driver6'] = df_trpmtch['driver'].shift(periods=-2)
    df_trpmtch['driver7'] = df_trpmtch['driver'].shift(periods=-3)
    df_trpmtch['driver8'] = df_trpmtch['driver'].shift(periods=-4) 
    # create matching vars
    df_trpmtch['trip_matched'] = 0
    for val in ["1", "2", "3", "4", "5", "6", "7", "8"]:
        nm_out = 'match' + val
        nm_in = 'driver' + val
        df_trpmtch[nm_out] = (df_trpmtch[nm_in] == df_trpmtch['driver']
                              ).astype(int)
        df_trpmtch['trip_matched'] += df_trpmtch[nm_out]
    # merge new variable onto trips master
    df = df.merge(df_trpmtch[['trip_matched', 'driver', 'trip']], 
                  on=['driver', 'trip'])
    print df_trpmtch['trip_matched'].mean()
    return df

def create_subm(df_triplevel, nm):
    """ Creates a kaggle submission """
    # initialize submission
    subm = pd.DataFrame(columns = ['driver_trip', 'prob'])
    # store driver and trip as ints
    drvr = df_triplevel['driver'].astype(int)
    trip = df_triplevel['trip'].astype(int)
    # concatenate to match id style of submission
    df_triplevel['driver_trip'] = drvr.map(str) + "_" + trip.map(str)
    df_triplevel['prob'] = df_triplevel['prds']
    # save submission
    subm = df_triplevel[['driver_trip', 'prob']]
    subm.to_csv( nm + ".csv", index=False)

def telemetrics_master(path, file_name, subm_nm, piece, is_fromstart = 1):
    """ 
    Master controller for building variables and analyzing drivers. 
    It starts with a flat file containing all the trips from
    some set of drivers (at the second level), creates or edits a trip level
    data set, fits random forest (one per driver) to predict false trips,
    and creates a submission.
    
    path - path to drivers
    
    file_name - name of flat appended drivers file to use. currently, this
    should be "TripsMaster" for full data, and "tester" for small data
    
    subm_nm - name of output submission
    
    piece - which section of data file (file_name) will be used
    
    is_fromstart - controls which sections of the code to use. To build all 
    trip level features from scratch, set to 1. To merge a few new 
    trip level features to exsiting, set to 0. To exclusively run random
    forests using only pre-existing trip-level features, set to -1
    
    Output: Kaggle submission for all drivers and trips in file_name
    """
    # load second level trips in a big flat file
    df = pd.read_csv(path + "/../" + file_name + str(piece) + ".csv")
    # Use vector operations to build basic variables (this does not require
    # selecting individual drivers, can be done all using flat file)
    varb.build_raw_vars(df)
    # create all of the original trip level statistics
    if is_fromstart == 1:
        # Collapse second-trip level data into trip level data
        df_triplevel = varb.trip_level_data(df)  
        # export trip level data
        df_triplevel.to_csv(path + "/../" + file_name + "_triplvl" +
                            str(piece) + ".csv", index = False)
    # load previous trip level data
    df_triplevel = pd.read_csv(path + "/../" + file_name + "_triplvl" +
                        str(piece) + ".csv")
    # create new trip level statistics and merge onto original stats
    if is_fromstart == 0:                    
        df_newtripstats = varb.new_trip_stats(df)
        # merge in new trip stats
        df_triplevel = df_triplevel.merge(df_newtripstats, 
                                          on = ['driver', 'trip'])
        # re-save with new vars (remember once a trip level stat has been
        # saved, it is considered an original stat and the code should be
        # adjusted accordingly (moved to trip_level_data from new_trip_stats))
        df_triplevel.to_csv(path + "/../" + file_name + "_triplvl" +
                            str(piece) + ".csv", index = False)                          
   
    # create a basic trip matching var
    #df_triplevel = len_dist_tripmatch(df_triplevel)
    df_triplevel['tot_over_fin'] = (df_triplevel['tot_dist']/
                                   (df_triplevel['fin_dist'] + .01))   
    # fill any missings
    cols = df_triplevel.columns                     
    for col in cols:
        val = df_triplevel[col].mean()
        df_triplevel[col].fillna(value= val, inplace=True) 
        #this needs a test or alert or output 
    # normalize all columns 
    #for col in cols:
    #    if ((col != 'driver') & (col != 'trip')):
    #        mean = df_triplevel[col].mean()
    #        std = df_triplevel[col].std()
    #        df_triplevel[col] = (df_triplevel[col] - mean)/std
                               
    # Choose features 'tot_dist_corr', 'trip_match'
    Xfeatures = [# distance
                 'fin_dist',  'fin_dist_quart1',
                 'fin_dist_quart2',
                 'fin_dist_quart3', 'fin_dist_quart4', 'tot_over_fin',
                 # speed
                 'avg_spd', 'avg_spd_nostopsouts', 'max_reasonable_spd',
                 'max_spd', 'std_spd',
                 'avg_spd_quart1', 'avg_spd_quart2', 'avg_spd_quart3',
                 'avg_spd_quart4',  
                 # accel
                 'max_accel', 'std_accelpos_quart1', 'std_accelpos_quart2',
                 'std_accelpos_quart3', 'std_accelpos_quart4', 'std_accelpos',
                 'max_reasonable_accel', 'max_spdaccel', 'min_spdaccel',
                 'avg_negaccel', 'avg_posaccel',
                 # decel
                 'min_decel_4avg', 'std_accelneg', 'min_reasonable_accel',
                 # stops
                 'max_stopaccel', 'avg_stopaccel',
                 'avg_stopposaccel',
                 'full_stops',
                 'unq_stops', 'pct_stopped', #'unq_stops_quart1',
                 #'unq_stops_quart2', 'unq_stops_quart3', 'unq_stops_quart4',
                 # turns
                 'pct_shrp_trn', 'spd_shrp_turn', 
                 'pct_ofturns_accel', 'amt_trn', 'amt_trn_quart1',
                 'amt_trn_quart2', 'amt_trn_quart3', 'amt_trn_quart4',
                 'num_wierdtrn180', 'num_wierdtrn360', 'std_smallturns',
                 'std_allturns'
                 # trip match
                 
                ]
    # Run random forests to analyze drivers
    analyze_trips(df_triplevel, Xfeatures)
       
########################## Assign locations ##################################
#TELEM_PATH = "S:/03 Internal - Current/Kaggle/Driver Telematics Analysis"
#DRVRS_PATH = "/Py0/Structured Data/01 Raw Datasets/drivers"
#SUBM_PATH = "/Py0/Structured Data/04 Graphics and Output Data/Submissions"
TELEM_PATH = "D:/Kaggle_data/Telematics/"
DRVRS_PATH = "drivers"
SUBM_PATH1 = "S:/03 Internal - Current/Kaggle/Driver Telematics Analysis/"
SUBM_PATH2 = "Py0/Structured Data/03 Final Datasets/Submissions/"
####################### Set run parameters ###########################
FULL_PATH = TELEM_PATH + DRVRS_PATH
SUBM_PATH = SUBM_PATH1 + SUBM_PATH2

if __name__ == '__main__':
    reload(prp)
    reload(varb)
    # name the submission created from this analysis
    subm_nm = "gradient boosting"
    # Conduct analysis on each chunk of drivers
    driver_chunks = [1, 2, 3]
    gc.collect()
    # Build variables and analyze the data sequentially by chunk
    for x in driver_chunks:
        gc.collect()
        print "Running piece " + str(x)
        # this is the primary controller of the whole competition
        telemetrics_master(FULL_PATH, "greater", 
                           SUBM_PATH + "pieces/" + subm_nm,
                           x, is_fromstart = -1) 
                           
 