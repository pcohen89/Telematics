# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 09:43:09 2015

@author: p_cohen
"""

########################## Import libraries ##################################
import sys
sys.path.append("C:/Git_repos/telematicsPy0")
#some of these are my standard libraries, may not be used
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model.ridge import Ridge
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

import tripmatching as match
reload(match)

def checker(driver, trip1, trip2):
    plt.plot(df_small.ix[((df_small['driver']==driver) & (df_small['trip'] == trip1)), 'rot_x'],
                   df_small.ix[((df_small['driver']==driver) & (df_small['trip'] == trip1)), 'rot_y_flipped'])
    plt.plot(df_small.ix[((df_small['driver']==driver) & (df_small['trip'] == trip2)), 'rot_x'],
                   df_small.ix[((df_small['driver']==driver) & (df_small['trip'] == trip2)), 'rot_y_flipped'])

def merge_tripmatching(x):
    df_triplevel = pd.read_csv(FULL_PATH + "/../" + "TripsMaster_triplvl" +
                               str(x) + ".csv")
    # load each single trip matching file and merge on onto trips master
    drivers = df_triplevel['driver'].drop_duplicates()
    file_path = 'D:/Kaggle_data/Telematics/trip_match_corr2/' 
    # initialize a file for appending
    trip_match_output = pd.DataFrame(columns = ['driver', 'matching_trip',	'trip', 'trip_match'])
    # for each driver and trip, append all of the trip matching csvs
    for drvr in drivers:
        print "driver " + str(drvr)
        for trip in range(1, 201):
            # identify correct trip
            conds = ((df_triplevel['driver'] == int(drvr)) &
                     (df_triplevel['trip'] == trip))
            mtch = pd.read_csv(file_path + str(drvr) + ".0_" + str(trip) + ".csv") 
            # append
            trip_match_output = trip_match_output.append(mtch)
    # create rows for the symmetric pairings
    trip_match_output2 = trip_match_output.copy(deep=True)
    trip_match_output2['trip'] = trip_match_output['matching_trip']
    trip_match_output2['matching_trip'] = trip_match_output['trip']
    # append symmetric pairings
    trip_match_output = trip_match_output.append(trip_match_output2)
    # drop unmatched
    is_keep = ((trip_match_output['trip'] > 0) &
               (trip_match_output['matching_trip'] > 0))
    trip_match_output = trip_match_output[is_keep]
    trip_match_output.sort(columns = ['driver' , 'trip', 'trip_match'], inplace=True)        
    # Keep only the lowest value for each trip
    cond = (trip_match_output['trip'] != trip_match_output['trip'].shift())
    final_trip_match = trip_match_output[cond]
    # rename variable
    final_trip_match.rename(columns = {'trip_match' : 'trip_match_corr3', 
                                       'matching_trip' : 'matching_trip3'},
                            inplace = True)
    # merge on to trip level data
    df_triplevel = df_triplevel.merge(final_trip_match, on=['driver', 'trip'],
                                      how='left') 
    df_triplevel['trip_match_corr3'].fillna(value= 25, inplace=True) 
    df_triplevel['matching_trip3'].fillna(value=-1, inplace=True) 

    df_triplevel.to_csv(FULL_PATH + "/../" + "TripsMaster_triplvl" +
                               str(x) + ".csv",index=False)

####################### Set run parameters ###########################
TELEM_PATH = "D:/Kaggle_data/Telematics/"
DRVRS_PATH = "drivers"
SUBM_PATH1 = "S:/03 Internal - Current/Kaggle/Driver Telematics Analysis/"
SUBM_PATH2 = "Py0/Structured Data/03 Final Datasets/Submissions/"
FULL_PATH = TELEM_PATH + DRVRS_PATH
SUBM_PATH = SUBM_PATH1 + SUBM_PATH2

####################### Execute ###########################
if __name__ == '__main__':
    for x in [1, 2, 3]:
        reload(match)
        df = pd.read_csv(FULL_PATH + "/../" + "TripsMaster" + str(x) + ".csv")
        df_rotated = match.create_rotated(df)
        print len(df_rotated[['trip', 'driver']].drop_duplicates())
        df_rotated = match.drop_short(df_rotated, 200)
        print len(df_rotated[['trip', 'driver']].drop_duplicates())
        drvrs = df_rotated['driver'].drop_duplicates()
        for drvr in drvrs:
            # create a driver only file
            df_driver = df_rotated[df_rotated['driver'] == drvr]
            match.trip_match_driver(df_driver, drvr) 
  
# load trip level data set
if __name__ == '__main__':
    Parallel(n_jobs=3)(
        delayed(match.merge_tripmatching)
        (x) for x in [1, 2, 3]
    )
                                 
df = pd.read_csv(FULL_PATH + "/../" + "greater2.csv")
df = match.create_rotated(df)
df = match.drop_short(df, 200)
df_small = df
match.trip_match_trip(df_small, 108 , 21)
checker(11, 183, 152)

