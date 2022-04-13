import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
import pandas as pd
import numpy as np
import tensorflow as tf

def featurize_data(data, target_var="inc_hosp", h=1, feature_parameters = 
    [{
        "fun": 'moving_average',
        "args": {'target_var': "inc_hosp", 'num_days': 7}
        },
        {
        "fun": 'lagged_values',
        "args": {'target_var': "inc_hosp", 'window_size': 2}
        },
    ]):
    """
    Convert data to tensors containing features x and responses y for each location
    and time.

    Parameters
    ----------
    data: data frame 
        It has columns location, date, and a column with the response variable to forecast.
        This data frame needs to be sorted by location and date columns in ascending order.
    target_var: string
        Name of the column in the data frame with the forecast target variable.
        Default to "inc_hosp"
    h: integer
        Forecast horizon. Default to 1.
    feature_parameters: array of dictionaries
        List of features to calculate. Each dictionary should have a `fun` key with feature function name
        and an `args` key with parameter name and values of this function.
        Available feature functions are "moving_average" and "lagged_values".
        
    Returns
    -------
    x_train_val: 3D tensor with shape (L, T, P) 
        L is the number of location l and T is the number of time point t for which
        the full feature vector x_{l,t}, possibly including lagged covariate values,
        and the response y_{l,t}, corresponding to the target variable at time t+h,
        could be calculated. P is number of features.
        Each row is a vector x_{l,t} = [x_{l,t,1},...,x_{l,t,P}] of features for some pair 
        (l, t) in the training set.
    y_train_val: 2D tensor with with shape (L, T) 
        Each value is a forecast target variable value in the training set.
        y_{l, t} = z_{l, 1, t+h}
    x_T: 3D tensor with shape (L, T = 1, P)
        Each value is test set feature for each location at forecast date.
    """
    assert target_var in data.columns

    # data has columns date, location, inc hosp, rate and population
    T = max(data['date'])

    # calculate features based on given parameters
    # and collect a list of feature names
    features_list = list()
    for feature in feature_parameters:
        data, features_list = data.pipe(eval(feature["fun"]), features_list =features_list, **feature["args"])
    
    # create a column for h horizon ahead target for observed values. 
    # for each location, this column has h nans in the end.
    # the last nan is for forecast date.
    data['h_days_ahead_target'] = data.groupby('location')[target_var].shift(-h)

    # create x_T using data with date = forecast_date (T)
    data_T = data.loc[data["date"]== T,:]

    # x_T is (L, P)
    x_T = data_T[features_list].values
    # x_T is (L, 1, P)
    x_T = np.expand_dims(x_T, axis = 1)
    
    # list of indices of rows that have as least one nan
    na_idx, _ = np.where(data.isna())
    
    # take out nans in data
    train_val = data.drop(na_idx)
    
    # reformat selected features
    x_train_val = train_val.pivot(index = "location", columns = "date", values = features_list).to_numpy()
    # shape is (L, T, P)
    x_train_val = x_train_val.reshape((x_train_val.shape[0], x_train_val.shape[1]//len(features_list), len(features_list)),order='F')

    # shape is (L, T, P)
    y_train_val = train_val.pivot(index = "location", columns = "date", values = 'h_days_ahead_target').to_numpy()

    # convert everything to tensor
    x_train_val = tf.constant(x_train_val)
    y_train_val = tf.constant(y_train_val)
    x_T = tf.constant(x_T)
    
    return x_train_val, y_train_val, x_T

def moving_average(data, target_var, features_list, num_days = 7):
    """
    Cacluate moving average of target variable and store result in a new column

    Parameters
    ----------
    data: data frame 
        It has columns location, date, and a column with the response variable to forecast.
        This data frame needs to be sorted by location and date columns in ascending order.
    target_var: string
        Name of the column in the data frame with the forecast target variable.
    features_list: list of strings
        Running list of feature column names
    num_days: integer
        Time window to calculate moving average for
    
    Returns
    -------
    data: data frame 
        Original data frame with additional column for moving average
    features_list: list of strings
        Running list of feature column names
    """
    column_name = str(num_days)+"_moving_avg_"+str(target_var)
    features_list.append(column_name)
    data[column_name] = data.groupby('location').rolling(num_days)[target_var].mean().reset_index(drop=True)
    return data, features_list

def lagged_values(data, target_var, features_list, window_size):
    """
    Cacluate lagged values of target variable and store results in new columns

    Parameters
    ----------
    data: data frame 
        It has columns location, date, and a column with the response variable to forecast.
        This data frame needs to be sorted by location and date columns in ascending order.
    target_var: string
        Name of the column in the data frame with the forecast target variable.
    features_list: list of strings
        Running list of feature column names
    window_size: integer
        Time window to calculate lagged values for
    
    Returns
    -------
    data: data frame 
        Original data frame with additional columns for lagged values.
    features_list: list of strings
        Running list of feature column names
    """
    for lag in range(1, window_size + 1):
        data['lag_' + str(lag)] = data.groupby('location')[target_var].shift(lag)
        features_list.append('lag_' + str(lag))
    return data, features_list

