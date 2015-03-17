# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 11:30:23 2015

@author: p_cohen
"""
########################## Import libraries ##################################
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

########################## Assign paths ##################################
TELEM_PATH = "S:/03 Internal - Current/Kaggle/Driver Telematics Analysis"
DRVRS_PATH = "/Py0/Structured Data/01 Raw Datasets/drivers"
SUBM_PATH = "/Py0/Structured Data/04 Graphics and Output Data/Submissions"
####################### Set run parameters ###########################
def append_trips_to_master(drvr_num, path_into_drvr, df_trip_master):
    """
    This function takes a driver id, a path to the driver, and a dataframe
    loads all trips of that driver, and then appends them
    to the master data set
    """
    # get list of trip numbers for that driver
    trip_names = os.listdir(path_into_drvr)
    # initialize driver datafile
    df_driver = pd.DataFrame(columns=['x', 'y', 'trip', 'driver'])
    # pull each trip, add ids, and append to master
    for trip_name in trip_names:
        # pull trip
        trip_path = path_into_drvr + "/" + str(trip_name)
        df_trip = pd.read_csv(trip_path)
        # add two identifiers (strip '.csv' from the tripNum)
        df_trip['trip'] = int(trip_name.replace(".csv", ""))
        df_trip['driver'] = int(drvr_num)
        # append to master trip list
        df_driver = df_driver.append(df_trip)
    df_trip_master = df_trip_master.append(df_driver, ignore_index=True)
    return df_trip_master

def create_trips_master(path, name, is_testing=1):
    """ This code will load all trips and append them into one master
    data file. Added to each trip's x, y will be the driver and trip
    number of the trip for idenfitication purposes
    Input: path to driver files
    Output: Dataframe containing all trips' x, y, trip #, driver #
    """
    # test assumption that each driver has 200 trips
    #if is_testing:
        #verify_200trips_per_drvr(path)
    # Read name of each driver file (will be numeric) and save to a list
    drivers = sorted(os.listdir(path), key=float)
    # Initialize outcome data set
    df_trip_master = pd.DataFrame(columns=['x', 'y', 'trip', 'driver'])
    # use sample of drivers when testing code
    if is_testing:
        drivers = [1, 2, 3, 10, 11, 12, 13, 14, 15, 16]
    # for each driver, set path to driver trips then run function to append
    # trips to master
    for drvr in drivers:
        if int(drvr)%20 == 0:
            print "We are on driver " + str(drvr)
        # Create path into a driver's folder of trips
        drvr_folder_path = path + "/" + str(drvr)
        # Call function to load each trip, add ids, and append to master
        df_trip_master = append_trips_to_master(drvr, drvr_folder_path,
                                                df_trip_master)
        # export one chunk and start over with a newly initialize dataframe
        if (int(drvr) == 1500) | ((is_testing == 1) & (drvr == 3)):
            df_trip_master.to_csv(path + "/../" + name + "1.csv", index=False)
            df_trip_master = pd.DataFrame(columns=['x', 'y', 'trip', 
                                                   'driver'])
            gc.collect()
        if (int(drvr) == 2500) | ((is_testing == 1) & (drvr == 12)):
            df_trip_master.to_csv(path + "/../" + name + "2.csv", index=False)
            df_trip_master = pd.DataFrame(columns=['x', 'y', 'trip', 
                                                   'driver'])
            gc.collect()
    df_trip_master.to_csv(path + "/../" + name + "3.csv", index=False)

    
##################### Rejected functions #################################
    
def avg_speed_slow(data, trip_stats_output):
    """ This slowly calculates the avg distance between trip rows (speed) """
    tot_dist = 0
    num_seconds = len(data['x'])
    for row in range(1, num_seconds):
        moment_dist = np.sqrt(
                      (data.ix[row, 'x'] - data.ix[row - 1, 'x']) ** 2 
                      +
                      (data.ix[row, 'y'] - data.ix[row - 1, 'y']) ** 2
                      )
        tot_dist += moment_dist
    trip_stats_output['avg_spd'] = tot_dist/num_seconds

def avg_speed_fast(data, trip_stats_output):
    """ 
    This quickly calculates the avg dist between trip rows (speed)
    it is faster than above because looping over rows uses python,
    whereas this version uses matrix operations that execute in C  
    """
    tot_dist = 0
    num_seconds = len(data['x'])
    # offsets x so that x2 represents the x observation on the next line
    data['x2'] = data['x'].shift()
    # offsets y so that y2 represents the y observation on the next line
    data['y2'] = data['y'].shift()
    # measure distances betweeen seconds of trip
    data['x_dist'] = data['x'] - data['x2']
    data['y_dist'] = data['y'] - data['y2']
    # sum total distance travelled
    data['tot_dist'] = np.sqrt(data['x_dist'] ** 2 + data['y_dist'] ** 2)
    tot_dist = data.ix[1:, 'tot_dist'].sum()
    # store avg speed
    trip_stats_output['avg_spd2'] = tot_dist/num_seconds
    
def speed_test(master_data, path, is_testing=1):
    """ this tests the speed of two different approaches to cal avg speed """
     # Read name of each driver file (will be numeric) and save to a list
    drivers = os.listdir(path)
    # use sample of drivers when testing code
    if is_testing:
        drivers = [1, 2, 3]
    # list of columns to be created in collapsed master
    trip_lvl_vars = ['driver', 'trip', 'fin_dist' 
                     ]# add new vars here
    # Initialize outcome, collapsed data set
    df_trips_clpsd = pd.DataFrame(columns=trip_lvl_vars)
    # create a trip level entry in df_trips_clpsd representing each trip
    t0 = time.time() # capture start time in order to time function
    for drvr in drivers:
        for trip_num in range(1, 201):
            df_trip = single_trip_data(drvr, trip_num, master_data)
            # first component of trip are the ids
            trip_stats = {'trip' : trip_num, 'driver' : drvr}
            avg_speed_slow(df_trip, trip_stats)
            # add line to trip level data
            df_trips_clpsd = df_trips_clpsd.append(trip_stats,
                                                   ignore_index=True)
    title = "Row loop took {time} mins for " + str(len(drivers)) + " drivers"
    print title.format(time=(time.time()-t0)/60)
    t0 = time.time() # capture start time in order to time function
    for drvr in drivers:
        for trip_num in range(1, 201):
            df_trip = single_trip_data(drvr, trip_num, master_data)
            # first component of trip are the ids
            trip_stats = {'trip' : trip_num, 'driver' : drvr}
            avg_speed_fast(df_trip, trip_stats)
            # add line to trip level data
            df_trips_clpsd = df_trips_clpsd.append(trip_stats,
                                                   ignore_index=True)
    title = "Col ops took {time} mins for " + str(len(drivers)) + " drivers"
    print title.format(time=(time.time()-t0)/60) 
    t0 = time.time() # capture start time in order to time function
    for drvr in drivers:
        for trip_num in range(1, 201):
            df_trip = single_trip_data(drvr, trip_num, master_data)
            # first component of trip are the ids
            trip_stats = {'trip' : trip_num, 'driver' : drvr}
            avg_speed_vfast(df_trip, trip_stats)
            # add line to trip level data
            df_trips_clpsd = df_trips_clpsd.append(trip_stats,
                                                   ignore_index=True)
    title = "Col ops took {time} mins for " + str(len(drivers)) + " drivers"
    print title.format(time=(time.time()-t0)/60)
def verify_200trips_per_drvr(path):
    """ this will test assumption that each driver has 200 trips
        should crash if path argument is
        'S:/03 Internal - Current/Kaggle/Driver Telematics Analysis/' +
          'Py0/Structured Data/01 Raw Datasets/' +
        'test data crashing verification of trips per driver'
    """
    # Read name of each driver file (will be numeric) and save to a list
    drivers = os.listdir(path)
    # check trip num for each driver
    for drvr in drivers:
        # Create path into a driver's folder of trips
        drvr_folder_path = path + "/" + str(drvr)
        # create list of all trips in driver folder
        trips = os.listdir(drvr_folder_path)
        # count number of items in list of all trips for driver
        num_trips = len(trips)
        # raise error if not 200 trips
        if num_trips != 200:
            error = "drvr " + str(drvr) + " has " + str(num_trips) + " trips"
            raise Exception(error)
    print "Assumption of 200 trips per driver: verified"
