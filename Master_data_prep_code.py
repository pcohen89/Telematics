# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:25:27 2015
For questions of style, refer to https://www.python.org/dev/peps/pep-0008/
@author: p_cohen
"""
########################## Import libraries ##################################
import sys
sys.path.append("C:/Git_repos/telematicsPy0")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
import numpy as np
import time
import prepfuncs as prp
import gc
import varbuildingfuncs as varb

########################## Functions ##################################
def analyze_trips(df_triplevel, features, is_testing):
    """ Creates and saves predictions for all drivers using random forest
    
    Input: trip level statistics for a set of drivers
    
    Output: trip level statistics with driver id predictions added
    """
    # capture initial time
    t0 = time.time()
    # Read name of each driver file (will be numeric) and save to a list
    drivers = df_triplevel['driver'].drop_duplicates()
    # initialize prediction column
    df_triplevel['prds'] = .5
    # overall score (for testing)
    overall = 0
    for driver in drivers:
        if ((int(driver)%100 == 0) | (is_testing)):
            print "We are modeling driver " + str(driver)
        # make analysis file with all driver trips & some random other trips
        df_modeldrvr = create_modeldrvr(df_triplevel, driver)
        # run random forest on analysis file, using features passed to func
        forst = RandomForestClassifier(n_estimators=4500, n_jobs=8,
                                       max_depth=50)
        forst.fit(df_modeldrvr[features], df_modeldrvr['target'])
        if ((int(driver)%100 == 0) | (is_testing) ):
            for x in range(0, len(forst.feature_importances_)):
                text1 = "Feature " + str(features[x])
                text2 = " chosen " + str(forst.feature_importances_[x]) 
                print  text1 + text2 + " times"         
        # choose appropriate trips to predict on to
        trips_to_pred = df_triplevel['driver'] == driver
        # Store predictions
        prds = forst.predict_proba(df_triplevel.ix[trips_to_pred,
                                                   features])[:, 1]
        # add predictions to trip level data for mnodelled driver
        df_triplevel['prds'][trips_to_pred] = prds
        # test accuracy of model
        if ((int(driver)%100 == 0) | (is_testing)): 
            overall = test_accuracy(df_modeldrvr, features, overall)
    print overall
    title = "It took {time} minutes to model this section drivers"
    print title.format(time=(time.time()-t0)/60) 

def test_accuracy(df_modeldrvr, features, overall): 
    """ this calculates the out of sample AUC for the model """
    # create insample and outsmaple
    df_modeldrvr['rand'] = pd.Series(np.random.rand(len(df_modeldrvr)), 
                                 index=df_modeldrvr.index)
    df_drvr_ins = df_modeldrvr[df_modeldrvr['rand'] > .3]
    df_drvr_out = df_modeldrvr[df_modeldrvr['rand'] <= .3]
    # create random forest instance
    forst = RandomForestClassifier(n_estimators=4000, n_jobs=8,
                                   max_depth=20)
    # fit forest
    forst.fit(df_drvr_ins[features], df_drvr_ins['target'])
    # make predictions
    df_drvr_out['preds'] = forst.predict_proba(df_drvr_out[features])[:, 1]
    # evaluate the AUC of the ROC
    fpr, tpr, thresholds = roc_curve(df_drvr_out['target'],
                                     df_drvr_out['preds'])
    
    auc_val = auc(fpr, tpr)
    # store in overall
    overall += auc_val
    return overall    

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
        val = df_triplevel[col].median()
        df_triplevel[col].fillna(value= val, inplace=True)               
    # Choose features
    Xfeatures = [# distance 'tot_dist_corr', 
                 'fin_dist', 'fin_dist_quart1',
                 'fin_dist_quart2',
                 'fin_dist_quart3', 'fin_dist_quart4', 'tot_over_fin',
                 # speed
                 'avg_spd',  'max_reasonable_spd', 'max_spd', 
                 'max_speed_4avg', 'std_spd',
                 'avg_spd_quart1', 'avg_spd_quart2', 'avg_spd_quart3',
                 'avg_spd_quart4',
                 # accel
                 'max_accel_4avg', 'max_accel', 'std_accelpos_quart1',
                 'std_accelpos_quart2',
                 'std_accelpos_quart3', 'std_accelpos_quart4', 'std_accelpos',
                 'max_reasonable_accel', 'max_spdaccel', 'min_spdaccel',
                 'min_spdaccel_4avg', 'max_spdaccel_4avg',
                 'avg_negaccel', 'avg_posaccel',
                 # decel
                 'min_decel_4avg', 'std_accelneg', 'min_reasonable_accel',
                 # stops
                 'avg_spd_nostopsouts', 'max_stopaccel', 'avg_stopaccel',
                 'avg_stopposaccel',
                 'full_stops',
                 'unq_stops', 'pct_stopped', 'pct_stopped_quart1',
                 'pct_stopped_quart2', 'pct_stopped_quart3', 
                 'pct_stopped_quart4',
                 # turns
                 'pct_shrp_trn', 'spd_shrp_turn', 
                 'pct_ofturns_accel', 'amt_trn', 'amt_trn_quart1',
                 'amt_trn_quart2', 'amt_trn_quart3', 'amt_trn_quart4',
                 'num_wierdtrn180', 'num_wierdtrn360', 'std_smallturns',
                 'std_allturns',
                 # trip match 
                 'trip_match_corr3'
                ]
    # Run random forests to analyze drivers
    is_testing = len(df_triplevel) < 2000
    analyze_trips(df_triplevel, Xfeatures, is_testing)
    # create submission
    create_subm(df_triplevel, subm_nm + str(piece))
    # clear memory
    df = "null"
    gc.collect()
       
########################## Assign locations ##################################
TELEM_PATH = "D:/Kaggle_data/Telematics/"
DRVRS_PATH = "drivers"
SUBM_PATH1 = "S:/03 Internal - Current/Kaggle/Driver Telematics Analysis/"
SUBM_PATH2 = "Py0/Structured Data/03 Final Datasets/Submissions/"
####################### Set run parameters ###########################
FULL_PATH = TELEM_PATH + DRVRS_PATH
SUBM_PATH = SUBM_PATH1 + SUBM_PATH2
##################### Run overall controllers #############################
# Load trips into a single large master file
# Note !!! this only needs to be re-done if you have a specific reason
# because THESE FILES ALREADY EXIST
# is_testing = 1 -> use only a few drivers per chunk
#prp.create_trips_master(TELEM_PATH + DRVRS_PATH, "greater", is_testing=1)
if __name__ == '__main__':
    reload(prp)
    reload(varb)
    # name the submission created from this analysis
    subm_nm = "rolling average vars"
    # Conduct analysis on each chunk of drivers
    driver_chunks = [1, 2, 3]
    gc.collect()
    # Build variables and analyze the data sequentially by chunk
    for x in driver_chunks:
        gc.collect()
        print "Running piece " + str(x)
        # this is the primary controller of the whole competition
        telemetrics_master(FULL_PATH, "TripsMaster", 
                           SUBM_PATH + "pieces/" + subm_nm,
                           x, is_fromstart = 0)    
    # append submission chunks
    full_subm = pd.DataFrame(columns=['driver_trip', 'prob'])
    for x in driver_chunks:
        subm_chunk = pd.read_csv(SUBM_PATH + "pieces/" + 
                                 subm_nm + str(x) + ".csv")
        full_subm = full_subm.append(subm_chunk)
    full_subm.to_csv(SUBM_PATH + subm_nm + ".csv", index=False)
