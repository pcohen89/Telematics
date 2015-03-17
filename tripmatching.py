# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 18:50:19 2015

@author: p_cohen
"""

########################## Import libraries ##################################
import sys
sys.path.append("C:/Git_repos/telematicsPy0")
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
TELEM_PATH = "D:/Kaggle_data/Telematics/"
DRVRS_PATH = "drivers"
SUBM_PATH1 = "S:/03 Internal - Current/Kaggle/Driver Telematics Analysis/"
SUBM_PATH2 = "Py0/Structured Data/03 Final Datasets/Submissions/"
FULL_PATH = TELEM_PATH + DRVRS_PATH
SUBM_PATH = SUBM_PATH1 + SUBM_PATH2

def create_rotated(df):
    """ 
    This function reverses trip orientation anonymization. 
    Admin rotated trips so that identical or very similar trips would be
    less obvious
    """
    # identify ever trip end point
    endpoints = df[((df['trip'] != df['trip'].shift(-1)) | 
                    (df['driver'] != df['driver'].shift(-1)))  ]
    # determine angle between end point and origin (theta)
    endpoints['theta'] = np.arctan2(df['y'], df['x'])
    # merge thetas back to appropriate trips in main dataset
    df = df.merge(endpoints[['trip', 'driver', 'theta']],
                  on=['trip', 'driver'])
    # rotate x and y of every point backwards by theta so that all trips
    # start and end on the same axis
    df['rot_x'] = (df['x']*np.cos(-1*df['theta']) - 
                   df['y']*np.sin(-1*df['theta']))
    df['rot_y'] = (df['x']*np.sin(-1*df['theta']) + 
                   df['y']*np.cos(-1*df['theta']))
    # This ensures that all trip end at a positive X coordinate and zero y-coordinate.
    # It's still possible that trips are flipped. To undo this, we check whether
    # a trip's maximum y position is greater in absolute value than its lowest y position in absolute
    # value. If so, we simply flip the y-coordinate (y becomes -y).
    y_extremes = df.groupby([df.driver, df.trip],
                            as_index=False).rot_y.agg({'y_max': max, 'y_min': min})
    # A dummy for whether we should flip the y-coordinates
    y_extremes['flip'] = y_extremes.y_max.abs() > y_extremes.y_min.abs()
    y_extremes['flip'].mean()

    df = df.merge(y_extremes[['driver', 'trip', 'flip']], on=['driver', 'trip'])

    # Flipping the y-coordinate if needed
    df['rot_y_flipped'] = df.rot_y * (1 - df.flip*2)
    return df
    
def drop_short(df, threshold):
    """ 
    This function reverses trip orientation anonymization. 
    Admin rotated trips so that identical or very similar trips would be
    less obvious
    """
    # identify ever trip end point
    endpoints = df[((df['trip'] != df['trip'].shift(-1)) | 
                    (df['driver'] != df['driver'].shift(-1)))  ]
    # determine distance of end point from origin
    endpoints['fin_dist'] = np.sqrt((endpoints['y']**2 + endpoints['x']**2))
    # merge thetas back to appropriate trips in main dataset
    df = df.merge(endpoints[['trip', 'driver', 'fin_dist']],
                  on=['trip', 'driver'],
                  how='outer')
    df = df[df['fin_dist'] > threshold]
    return df


def trip_match_driver(df, driver):
    """
    This function determines the smallest integral that can be found for each
    trip between that trip and any other trip in the same driver file
    most of the action happens in trip_match_trip, this function proper
    mostly just creates a driver level file and calls trip_match_trip in
    parallel
    """
    # time execution
    t0 = time.time() 
    # evaluate the closeness of neighbor for each trip
    # this uses parallel processing on all eight tower cores
    # very speed, much wow
    Parallel(n_jobs=8)(
        delayed(trip_match_trip)
        (df, trip, driver) for trip in range(1, 201)
    )
    # finish measuring execution time
    title = "It took {time} minutes for driver " + str(driver) 
    print title.format(time=(time.time()-t0)/60)
    
def trip_match_driver_trip1(df, driver):
    """
    This function determines the smallest integral that can be found for each
    trip between that trip and any other trip in the same driver file
    most of the action happens in trip_match_trip, this function proper
    mostly just creates a driver level file and calls trip_match_trip in
    parallel
    """
    # time execution
    t0 = time.time() 
    # evaluate the closeness of neighbor for each trip
    # this uses parallel processing on all eight tower cores
    # very speed, much wow
    print len(df)
    trip_match_trip(df, 1, driver) 
    # finish measuring execution time
    title = "It took {time} minutes for driver " + str(driver) 
    print title.format(time=(time.time()-t0)/60)  
 
def trip_match_trip(df_driver, trip, driver):
    """
    inputs:
    df_driver - all second-level trips for one driver
    
    driver - number of driver
    
    trip - trip to evaluate
    
    output:
    
    csv with trip number, driver number, and the smallest intergral between
    trip and any other trip in that driver level file
    """
    # set a starting "smallest integral with other trip" val
    min_integral = 25
    min_trip = -1
    # create condition for finding the trip of interest
    base_trip = ((df_driver['trip'] == int(trip)) & 
                 (df_driver['driver'] == int(driver)))
    d = {'trip' : trip, 'driver' : driver, 'trip_match' : min_integral,
         'matching_trip' : min_trip}
    output = pd.DataFrame(d, index=[1])
    # create file with just the trip of interest
    df_base = pd.DataFrame(df_driver[base_trip])
    if len(df_base) < 120:
        nm = str(driver) + "_" + str(trip) + ".csv"              
        output.to_csv('D:/Kaggle_data/Telematics/trip_match_corr2/'
                      + nm, index=False)
        return
    # compare trip of interest to all other trips in driver file
    for x in range(trip + 1, 201):
        # create trip file for different trip to compare trip of interest to
        comp = ((df_driver['trip'] == x) & 
                (df_driver['driver'] == int(driver)))
        df_comp = pd.DataFrame(df_driver[comp])
        # check if trip is within 3.3 minutes of comparison trip
        # 3.3 chosen arbitrarily bigger means more run time and more accuracy
        if ((np.abs(len(df_base)-len(df_comp)) < 200) &
            (len(df_comp) > 120)):
            # find length of shorter trip
            obs = min(len(df_base), len(df_comp)) - 2
            # reset indexs
            # clear out any background isues
            df_base = df_base[[u'x', u'y', u'trip', u'driver', u'theta', 
                               u'rot_x', u'rot_y', u'flip',
                               u'rot_y_flipped', u'fin_dist']]
            df_comp = df_comp[[u'x', u'y', u'trip', u'driver', u'theta', 
                               u'rot_x', u'rot_y', u'flip',
                               u'rot_y_flipped', u'fin_dist']]
            df_base.reset_index(inplace=True)
            df_comp.reset_index(inplace=True)
            # determine the distance between trips (componentwise first)
            # this compares the beginning of the trips
            x_dists = (df_base['rot_x'][:obs] - df_comp['rot_x'][:obs])
            y_dists = (df_base['rot_y_flipped'][:obs] - 
                       df_comp['rot_y_flipped'][:obs])
            # sum all of the pointwise distance comparisons
            integral1 = ((np.sqrt(x_dists**2 +
                          y_dists**2)
                         ).sum())/(obs**1.9)
            # determine the distance between trips (componentwise first)
            # this compares the ends of the trips
            # create equal length files that represent the end of the two trips
            df_base_end = df_base[-obs:].reset_index()
            df_comp_end = df_comp[-obs:].reset_index()
            x_dists = (df_base_end['rot_x'] - df_comp_end['rot_x'])
            y_dists = (df_base_end['rot_y_flipped'] - 
                       df_comp_end['rot_y_flipped' ])
            # sum all of the pointwise distance comparisons
            integral2 = ((np.sqrt(x_dists**2 +
                          y_dists**2)
                         ).sum())/(obs**1.9)
            # find average integral
            integral = (integral1+ integral2 )/2
            # divide integral by the 1.5 power of trip length
            avg_integral = integral
             # update minimum when new minimum is achieved
            if avg_integral < 6:
                min_trip = x
                d = {'trip' : trip, 'driver' : driver, 'trip_match' : avg_integral,
                     'matching_trip' : min_trip}
                output = output.append(d, ignore_index=True)
               
    # export outcome
    nm = str(driver) + "_" + str(trip) + ".csv"              
    output.to_csv('D:/Kaggle_data/Telematics/trip_match_corr2/' + nm, index=False)
    
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
    trip_match_output.sort(columns = ['driver' , 'trip',
                                      'trip_match'], inplace=True)        
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