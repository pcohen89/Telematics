# -*- coding: utf-8 -*-
"""
Created on Thu Feb 05 16:54:00 2015

@author: p_cohen
"""
import sys
sys.path.append("C:/Git_repos/telematicsPy0")
#some of these are my standard libraries, may not be used
import pandas as pd
#from sklearn.ensemble import RandomForestRegressor
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
import gc
import math

########################## Assign paths ##################################
TELEM_PATH = "S:/03 Internal - Current/Kaggle/Driver Telematics Analysis"
DRVRS_PATH = "/Py0/Structured Data/01 Raw Datasets/drivers"
SUBM_PATH = "/Py0/Structured Data/04 Graphics and Output Data/Submissions"

#########################################################################
def build_raw_vars(master_data):
    """ 
    This code creates variables in the master data that are useful for when
    it comes to collapsing trips data
    """
   
    # create a position observation offset for comparing position 
    # to position in next second
    master_data['x2'] = master_data['x'].shift()
    master_data['y2'] = master_data['y'].shift()
    # direction travelling in relative to position at beginning of second
    master_data['direction'] = np.arctan2(master_data['y'] - master_data['y2'],
                                          master_data['x'] - master_data['x2'])
    # Create angle offset
    master_data['direction2'] = master_data['direction'].shift() 
    # Create angle change (using abs as turning left or right shouldn't matter)
    master_data['deltdirection'] = abs(master_data['direction'] - 
                                   master_data['direction2'])
    # calculate distance between observations in consecutive seconds
    master_data['step_xdist'] = master_data['x'] - master_data['x2']
    master_data['step_ydist'] = master_data['y'] - master_data['y2']
    # sum total distance travelled
    master_data['step_deltpos'] = np.sqrt(master_data['step_xdist'] ** 2 + 
                                           master_data['step_ydist'] ** 2)
    #create speed offset to capture acceleration
    master_data['step_deltpos2'] = master_data['step_deltpos'].shift() 
    # capture acceleration (change in change in distance per second => 
    # change in speed per sec => acceleration )
    master_data['step_deltdeltpos'] = (master_data['step_deltpos'] - 
                                        master_data['step_deltpos2'])
    
def trip_level_data(master_data):
    """
    This code will read the master file and collapse it into a data frame
    where each trip is represented by a single observation of trip level
    data
    input :  master dataframe at the trip-slice level, path to drivers
    output : master dataframe at the trip level with trip summary vars
    """
    # Read name of each driver file (will be numeric) and save to a list
    drivers = master_data['driver'].drop_duplicates()
    # list of columns to be created in collapsed master
    trip_lvl_vars = ['driver', 'trip']
    # Initialize outcome, collapsed data set
    df_trips_clpsd = pd.DataFrame(columns=trip_lvl_vars)
    # create a trip level entry in df_trips_clpsd representing each trip
    t0 = time.time() # capture start time in order to time function
    for drvr in drivers:
        if int(drvr)%100 == 0:
            print "We are on driver " + str(drvr)
        df_driver = single_driver_data(drvr, master_data)
        for trip_num in range(1, 201):
            df_trip = single_trip_data(drvr, trip_num, df_driver)
            # first component of trip are the ids
            trip_stats = {'trip' : trip_num, 'driver' : drvr}
            # create trip length (dist between start and finish) variable
            trip_length(df_trip, trip_stats)
            # create a measure of the total length of the trip
            total_distance(df_trip, trip_stats)
            # create an average speed variable
            avg_speed(df_trip, trip_stats)
            # create a max speed
            max_speed(df_trip, trip_stats)
            # create acceleration vars
            min_acceleration(df_trip, trip_stats)
            max_acceleration(df_trip, trip_stats)
            # create different versions of speed
            avg_speed_wconds(df_trip, trip_stats)
            # create stoppage stats
            stops(df_trip, trip_stats)
            # create std stats
            stds(df_trip, trip_stats)
            # add new cal of accel
            rollingmin_acceleration_alt(df_trip, trip_stats)
            # add turning stats
            turning_stats(df_trip, trip_stats)
            # first set of subsect vars
            subsect_stats(df_trip, trip_stats)
            # add more accel stats
            avg_accel(df_trip, trip_stats)
            # add new subsect vars
            subsect_stats2(df_trip, trip_stats)
            # add speed times acceleration
            speedaccel(df_trip, trip_stats)
            # weird turns
            weird_turns(df_trip, trip_stats)
            # acceleration from stop
            extra_stop_vars(df_trip, trip_stats)
            # standard deviation of amount of turn
            turn_stds(df_trip, trip_stats)
            # add line to trip level data
            df_trips_clpsd = df_trips_clpsd.append(trip_stats,
                                                   ignore_index=True)
    # print time to completion
    title = "It took {time} minutes for " + str(len(drivers)) + " drivers"
    print title.format(time=(time.time()-t0)/60)     
    return  df_trips_clpsd 

def new_trip_stats(master_data):
    """
    This code functions like trip_level_data but is intended to create
    different variables that can be appended on to the already created
    trip level data set
    
    input :  master dataframe at the trip-slice level, path to drivers
    output : master dataframe at the trip level with trip summary vars
    """
    # Read name of each driver file (will be numeric) and save to a list
    drivers = master_data['driver'].drop_duplicates()
    # list of columns to be created in collapsed master
    trip_lvl_vars = ['driver', 'trip']
    # Initialize outcome, collapsed data set
    df_trips_clpsd = pd.DataFrame(columns=trip_lvl_vars)
    # create a trip level entry in df_trips_clpsd representing each trip
    t0 = time.time() # capture start time in order to time function
    for drvr in drivers:
        if int(drvr)%100 == 0:
            print "We are on driver " + str(drvr)
            gc.collect()
        df_driver = single_driver_data(drvr, master_data)
        for trip_num in range(1, 201):
            df_trip = single_trip_data(drvr, trip_num, df_driver)
            # first component of trip are the ids
            trip_stats = {'trip' : trip_num, 'driver' : drvr}
            # add new vars
            subsect_stats3(df_trip, trip_stats)
            rolling_avg_vars(df_trip, trip_stats)
            # add line to trip level data
            df_trips_clpsd = df_trips_clpsd.append(trip_stats,
                                                   ignore_index=True)
    # print time to completion
    title = "It took {time} minutes for " + str(len(drivers)) + " drivers"
    print title.format(time=(time.time()-t0)/60)     
    return  df_trips_clpsd     

def single_trip_data(drvr, trip_num, master):
    """ Creates a df representing only one trip sliced from trip master """
    # create boolean vectors to match ids with the trip of interest
    trip_ids = ((master['trip'].astype(int) == trip_num) &
                (master['driver'] == drvr))
    # create a dataframe with only the trip of interest
    df_trip = pd.DataFrame(master.ix[trip_ids, :])
    # reset row index to start from 0 rather than adopting the row
    # numbers that rows had in the master data set
    df_trip = df_trip.reset_index()
    return df_trip
    
def single_driver_data(drvr, master):
    """ Creates a df representing only one driver sliced from trip master """
    # create boolean vectors to match ids with the trip of interest
    trip_ids = (master['driver'] == drvr)
    # create a dataframe with only the driver of interest
    df_driver = pd.DataFrame(master.ix[trip_ids, :])
    # reset row index to start from 0 rather than adopting the row
    # numbers that rows had in the master data set
    df_driver = df_driver.reset_index()
    return df_driver

def trip_length(data, trip_stats_output):
    """ Measures distance from start to end of trip """
    # Create feature that measures distance between end and start
    fin_dist = np.sqrt((data.ix[len(data) - 1, 'x'] ** 2 +
                        data.ix[len(data) - 1, 'y'] ** 2))
    # save feature to the row being created for the output table
    trip_stats_output['fin_dist'] = round(fin_dist, 1)
    
def total_distance(data, trip_stats_output):
    """ Measures total distance travelled """
    trip_stats_output['tot_dist'] = round(data['step_deltpos'].sum(), 1)
    
def corrected_tot_distance(data, trip_stats_output):
    """ Measures total distance travelled w/o jumps """
    df_nonjump = data.ix[data['step_deltpos' < 60 ]]
    nm = 'tot_dist_corr'
    trip_stats_output[nm] = round(df_nonjump['step_deltpos'].sum(), 1)
    
def avg_speed(data, trip_stats_output):
    """ 
    This quickly calculates the avg dist between trip rows (speed)
    it is faster than other methods because looping over rows uses python,
    whereas this version uses matrix operations that execute in C 
    (which python is build on top of)
    """
    num_seconds = len(data['driver'])
    tot_dist = data.ix[1:, 'step_deltpos'].sum()
    # store avg speed
    trip_stats_output['avg_spd'] = round(tot_dist/num_seconds, 1)
    
def avg_speed_wconds(data, trip_stats_output):
    """ 
    This calculates average speed dropping outliers and separately dropping
    outliers and stops
    """
    # drop first ob
    df_wo_first = data.ix[1:, :]
    # create no stop condition
    stop_cond = df_wo_first['step_deltpos'] > .35 # .8 MPH
    # measure trip distance after removing stops
    df_for_nostops = df_wo_first[stop_cond]
    tot_dist_nostops = df_for_nostops['step_deltpos'].sum() 
    secs_nostops = len(df_for_nostops.index) + .01 #prevent /0 err.
    # create no outliers condition
    outlier_cond = df_for_nostops['step_deltpos'] < 55 # 123 MPH
    # measure 
    df_for_nostopsouts = df_for_nostops[outlier_cond]
    tot_dist_nostopsouts = df_for_nostopsouts['step_deltpos'].sum() 
    sec_nostopsouts = len(df_for_nostopsouts.index) + .01
    # store avg speeds
    trip_stats_output['avg_spd_nostops'] = round(tot_dist_nostops/
                                                 secs_nostops, 1)
    trip_stats_output['avg_spd_nostopsouts'] = round(tot_dist_nostopsouts/
                                                 sec_nostopsouts, 1)
                                                 
def stops(data, trip_stats_output):
    """
    Analysis of stops
    """
    # count seconds
    tot_secs = np.float(len(data.index))
    # Count number of secs spent stopped
    stop_cond = data['step_deltpos'] < .35 # .8 MPH
    stopped_secs = len(data[stop_cond].index)
    trip_stats_output['pct_stopped'] = np.divide(stopped_secs, tot_secs)
    # count number of unique stops 
    unq_stop = ((data['step_deltpos'] < .35) & (data['step_deltpos2'] > .35))
    trip_stats_output['unq_stops'] = len(data[unq_stop].index)
    
def max_speed(data, trip_stats_output):
    """ 
    This calculates the max change in position per second (max speed)
    """
    # store max speed with outliers (seem to be car jumps)
    trip_stats_output['max_spd'] = round(data.ix[1:, 'step_deltpos'].max(), 1)
    # store max speed without outliers (55 m/s is 125 mph)
    w_reasonable_spds = data['step_deltpos'] < 55
    nm = 'max_reasonable_spd'
    vals = data['step_deltpos'][w_reasonable_spds].max()
    trip_stats_output[nm] = round(vals, 1)
    
def max_acceleration(data, trip_stats_output):
    """ 
    This calculates the max change in change in position per second 
    (max acceleration)
    """
    # store max accel with outliers (seem to be car jumps)
    vals = data.ix[2:, 'step_deltdeltpos'].max()
    trip_stats_output['max_accel'] = round(vals, 1)
    # store max accel without outliers (55 m/s is 125 mph)
    w_reasonable_accels = data['step_deltdeltpos'] < 14
    nm = 'max_reasonable_accel'
    # create max acceleration for seconds with reasonable velocities
    vals = data['step_deltdeltpos'][w_reasonable_accels].max()
    trip_stats_output[nm] = round(vals, 1)
    
def min_acceleration(data, trip_stats_output):
    """ 
    This calculates the min change in change in position per second 
    (max acceleration)(these will be negative)
    """
    # store max speed with outliers (seem to be car jumps)
    vals = data.ix[2:len(data)-2, 'step_deltdeltpos'].min()
    trip_stats_output['min_accel'] = round(vals, 1)
    # store max accel without outliers (55 m/s is 125 mph)
    w_reasonable_accels = data['step_deltdeltpos'] > - 14
    nm = 'min_reasonable_accel'
    # create max acceleration for seconds with reasonable velocities
    vals = data['step_deltdeltpos'][w_reasonable_accels].min() 
    trip_stats_output[nm] = round(vals, 1)
        
def rollingmin_acceleration_alt(data, trip_stats_output):
    """ 
    This calculates the 4 lowest acceleration values and takes their average 
    """
    # take the acceleration values
    srtd_accels = pd.Series(data.ix[2:len(data)-2, 'step_deltdeltpos'])
    # eliminate outlier observations
    srtd_accels = srtd_accels[srtd_accels > -14]
    # sort the acceleration values
    srtd_accels.sort(ascending=True)
    # take the average of the first four observations
    vals = srtd_accels[0:3].mean()
    # store output
    trip_stats_output['min_decel_4avg'] = round(vals, 1)

def stds(data, trip_stats_output):
    """
    This calculates the standard deviations of various trip statistics
    """
    # drop first obs
    data = data.ix[1:, :]
    # Calculate speed std
    std_spd = data['step_deltpos'].std()
    trip_stats_output['std_spd'] = std_spd
    # calculate std of acceleration while positive
    accelpos_cond = ((data['step_deltdeltpos'] > 0) & 
                     (data['step_deltdeltpos'] < 11))
    std_accelpos = data['step_deltdeltpos'][accelpos_cond].std()
    trip_stats_output['std_accelpos'] = std_accelpos
    # calc std of accel while braking
    accelneg_cond = ((data['step_deltdeltpos'] < 0) & 
                     (data['step_deltdeltpos'] > -15))
    std_accelneg = data['step_deltdeltpos'][accelneg_cond].std()
    trip_stats_output['std_accelneg'] = std_accelneg

def turning_stats(data, trip_stats_output):
    """ 
    This will create several turning stats
    (we will not include in this analysis turns sharper than >.8 which is 
    approximately a 45 degree turn)(I am using the assumption that it takes
    two seconds at least to make a full 90 degree turn)
    """
    # Exclude big jumps
    is_nonjump = (data['step_deltpos'] < 55)
    df_nonjump = data[is_nonjump]
    num_secs = np.float(len(df_nonjump['x'].index))
    # Measure amount of turning per second on average
    amt_turn = df_nonjump['deltdirection'].sum()
    trip_stats_output['amt_trn'] = round(amt_turn/num_secs, 4)
    
    # Measure percent of trip spent turning fairly sharply
    is_shrp_turn = ((df_nonjump['deltdirection'] < .8) &
                    (df_nonjump['deltdirection'] > .35))
    df_shrp_turn = df_nonjump[(is_shrp_turn & is_nonjump)]
    num_shrp_turn = len(df_shrp_turn.index)
    trip_stats_output['pct_shrp_trn'] = round(num_shrp_turn/num_secs, 4)
    # Measure average speed during sharp turns
    spd_turn_speed = df_shrp_turn['step_deltpos'].mean()
    trip_stats_output['spd_shrp_turn'] = round(spd_turn_speed, 3)
    # Percent of sharp turns with meaningfully positive acceleration
    pct_trns_accl = len(df_shrp_turn[df_shrp_turn['step_deltdeltpos'] > 1])
    trip_stats_output['pct_ofturns_accel'] = round(pct_trns_accl/num_secs, 4)

def subsect_stats(data, trip_stats_output):
    """ 
    This will duplicate a couple stats for several different
    subsections of the trip 
    """
    # create quarters of the data
    num_secs = len(data.index)
    for x in range(1,5):
        # select 1 quarter of the data
        df_quart = data.ix[np.int((x-1) * round(num_secs/4)) : 
                        np.int(round(x * num_secs/4) - 1)]
        # measure distance between start and finish of quarter piece
        fin_dist = np.sqrt((df_quart.ix[np.int((x-1) * num_secs/4), 'x'] -
                            df_quart.ix[np.int(x * num_secs/4 - 2), 'x']) ** 2 
                            +
                            (df_quart.ix[ np.int((x-1) * num_secs/4), 'y'] -
                            df_quart.ix[np.int(x * num_secs/4 - 1), 'y']) ** 2)
        nm = "fin_dist_quart" + str(x)
        trip_stats_output[nm] = fin_dist 
        # Measure std of positive acceleration (useful for full trip)
        accelpos_cond = ((df_quart['step_deltdeltpos'] > 0) & 
                         (df_quart['step_deltdeltpos'] < 11))
        std_accelpos = df_quart['step_deltdeltpos'][accelpos_cond].std()
        nm = 'std_accelpos_quart' + str(x)
        trip_stats_output[nm] = std_accelpos
        # measure speed
        quart_secs = len(df_quart.index)
        tot_dist = df_quart.ix[1:, 'step_deltpos'].sum()
        nm = "avg_spd_quart" + str(x)
        # store avg speed
        trip_stats_output[nm] = round(tot_dist/quart_secs, 1)
        
def subsect_stats2(data, trip_stats_output):
    """ 
    This will duplicate a couple stats for several different
    sections of the trip 
    """
    # create quarters of the data
    num_secs = len(data.index)
    for x in range(1,5):
        # select 1 quarter of the data
        df_quart = data.ix[np.int((x-1) * round(num_secs/4)) : 
                        np.int(round(x * num_secs/4) - 1)]
        # Exclude big jumps
        is_nonjump = (df_quart['step_deltpos'] < 55)
        df_nonjump = df_quart[is_nonjump]
        num_secs = np.float(len(df_nonjump['x'].index))
        # Measure amount of turning per second on average
        amt_turn = df_nonjump['deltdirection'].sum()
        nm = "amt_trn_quart" + str(x)
        trip_stats_output[nm] = round(amt_turn, 5) 
        # Measure std of positive acceleration (useful for full trip)
        # count number of unique stops 
        unq_stop = ((df_nonjump['step_deltpos'] < .35) & 
                    (df_nonjump['step_deltpos2'] > .35))
        nm = "unq_stops_quart" + str(x)
        # store avg speed
        trip_stats_output[nm] = len(df_nonjump[unq_stop].index)
        
def speedaccel(data, trip_stats_output):
    """ Find maximum product of speed and acceleration """
    is_nonjump = ((data['step_deltpos'] < 55) &
                  (data['step_deltdeltpos'] < 15) &
                  (data['step_deltdeltpos'] > -15))
    df_nonjump = data[is_nonjump]
    # create product of speed and acceleration
    df_nonjump['prod_spdaccel'] = (df_nonjump['step_deltpos'] * 
                                   df_nonjump['step_deltdeltpos'])
    # store the max value
    max_spdaccel = df_nonjump['prod_spdaccel'].max()
    trip_stats_output['max_spdaccel'] = max_spdaccel
    # store the min value
    min_spdaccel = df_nonjump['prod_spdaccel'].min()
    trip_stats_output['min_spdaccel'] = min_spdaccel
    
def avg_accel(data, trip_stats_output):
    """ find average pos and neg acceleration """
    is_nonjump = ((data['step_deltpos'] < 55) &
                  (data['step_deltdeltpos'] < 15) &
                  (data['step_deltdeltpos'] > -15))
    # create data that avoids weirdnesses
    df_nonjump = data[is_nonjump]
    # store avg accel (pos)
    is_accelpos = (df_nonjump['step_deltdeltpos'] > 0)
    avg_posaccel = df_nonjump['step_deltdeltpos'][is_accelpos].mean()
    trip_stats_output['avg_posaccel'] = avg_posaccel
    # store avg accel (neg)
    avg_negaccel = df_nonjump['step_deltdeltpos'][~(is_accelpos)].mean()
    trip_stats_output['avg_negaccel'] = avg_negaccel
    
def weird_turns(data, trip_stats_output):
    """ Measure # of really large turns """
    # Exclude big jumps
    is_nonjump = (data['step_deltpos'] < 55)
    df_nonjump = data[is_nonjump]
    # Measure number of turns between big and 180degrees
    is_shrp_turn = ((df_nonjump['deltdirection'] > .8) &
                    (df_nonjump['deltdirection'] < 3.14))
    df_shrp_turn = df_nonjump[(is_shrp_turn)]
    num_shrp_turn = len(df_shrp_turn.index)
    trip_stats_output['num_wierdtrn180'] = round(num_shrp_turn, 4)
    # Measure number of turns between 180 and 360degrees
    is_shrp_turn = ((df_nonjump['deltdirection'] > 3.14) &
                    (df_nonjump['deltdirection'] < 2*3.14))
    df_shrp_turn = df_nonjump[(is_shrp_turn)]
    num_shrp_turn = len(df_shrp_turn.index)
    trip_stats_output['num_wierdtrn360'] = round(num_shrp_turn, 4)
    
def extra_stop_vars(data, trip_stats_output):
    """ 
    This function measures avg and max acceleration from stop as 
    well as number of seconds of full stop
    """
    # create a new data set (make sure I am not adding weird columns to 
    #the original)
    df_forstops = pd.DataFrame(data[2:])
    # existing accel var says how much did I speed up from the last obs
    # THIS accel variable will be how much am I speeding up to the next ob
    df_forstops['accel_2next_sec'] = df_forstops['step_deltdeltpos'].shift(
                                     periods=-1)    
    # Create a very low speeds data set
    stop_cond = data['step_deltpos'] < .35 
    df_stopsonly = data[stop_cond]
    # measure avg accel from stop
    accel_from_stop = df_stopsonly['step_deltdeltpos'].mean()
    trip_stats_output['avg_stopaccel'] = accel_from_stop
    # measure avg pos accel from stop
    pos_accel = df_stopsonly['step_deltdeltpos'] > 0
    posaccel_from_stop = df_stopsonly[pos_accel]['step_deltdeltpos'].mean()
    trip_stats_output['avg_stopposaccel'] = posaccel_from_stop
    # measure avg accel from stop
    max_accel_from_stop = df_stopsonly['step_deltdeltpos'].max()
    trip_stats_output['max_stopaccel'] = max_accel_from_stop
    # measure num full stops
    fullstop_cond = df_stopsonly['step_deltpos'] == 0  
    full_stops = len(df_stopsonly[fullstop_cond].index)
    trip_stats_output['full_stops'] = full_stops
    
def turn_stds(data, trip_stats_output):
    """ Measure std of turns with and without big turns """
    # Exclude big jumps
    is_nonjump = (data['step_deltpos'] < 55)
    df_nonjump = data[is_nonjump]
    # Measure std of turns in whole trip
    std_allturns = df_nonjump['deltdirection'].std()
    trip_stats_output['std_allturns'] = round(std_allturns, 5)
    # Measure std of turns that aren't greater than 90degrees
    is_small_turn = (df_nonjump['deltdirection'] < 1.7)
    df_small_turn = df_nonjump[(is_small_turn)]
    std_smallturns = df_small_turn['deltdirection'].std()
    trip_stats_output['std_smallturns'] = round(std_smallturns, 5)
    
def subsect_stats3(data, trip_stats_output):
    """ 
    This will duplicate stop amount for several different
    sections of the trip 
    """
    # create quarters of the data
    num_secs = len(data.index)
    for x in range(1,5):
        # select 1 quarter of the data
        df_quart = data.ix[np.int((x-1) * round(num_secs/4)) : 
                        np.int(round(x * num_secs/4) - 1)]
        # Exclude big jumps
        is_nonjump = (df_quart['step_deltpos'] < 55)
        df_nonjump = df_quart[is_nonjump]
        tot_secs = np.float(len(data.index))
        # Count number of secs spent stopped
        stop_cond = data['step_deltpos'] < .35 # .8 MPH
        stop_secs = len(df_nonjump[stop_cond].index)
        # store calculated variable
        nm = 'pct_stopped_quart' + str(x)
        trip_stats_output[nm] = round(stop_secs/(tot_secs+.1), 4)
        
def rolling_avg_vars(data, trip_stats_output):
    """ 
    This function creates versions of variables that use rolling
    averages instead of maxs or mins
    """
    # create no jump data
    is_nonjump = (data['step_deltpos'] < 55)
    is_nonjump_accel = ((data.step_deltdeltpos < 14) &
                        (data.step_deltdeltpos > -14))
    df_nonjump = data[(is_nonjump) & (is_nonjump_accel)]
    # take the speed values
    spds = pd.Series(df_nonjump.step_deltpos)
    # sort the acceleration values
    spds.sort(ascending=False)
    # take the average of the first four observations
    vals = spds[0:3].mean()
    # store output
    trip_stats_output['max_speed_4avg'] = round(vals, 3)
    # take the accel values
    accels = pd.Series(df_nonjump.step_deltdeltpos)
    # sort the acceleration values
    accels.sort(ascending=False)
    # take the average of the first four observations
    vals = accels[0:3].mean()
    # store output
    trip_stats_output['max_accel_4avg'] = round(vals, 3)
    # create product of speed and acceleration
    df_nonjump['prod_spdaccel'] = (df_nonjump['step_deltpos'] * 
                                   df_nonjump['step_deltdeltpos'])
    spdaccel = pd.Series(df_nonjump.prod_spdaccel)
    # sort the acceleration values
    spdaccel.sort(ascending=False)
    # take the average of the first four observations
    vals = spdaccel[0:3].mean()
    # store output
    trip_stats_output['max_spdaccel_4avg'] = round(vals, 3)
    # take the average of the last four observations
    vals = spdaccel[-4:].mean()
    # store output
    trip_stats_output['min_spdaccel_4avg'] = round(vals, 3)
    
    
          

   
    
    
                               
                  
        
        
    
    
    
    
    
    
    
    