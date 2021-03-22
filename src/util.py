#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:08:21 2021

@author:Wei Zhao @ Metis

"""
#%%
import pickle
import numpy as np
import dask.dataframe as dd
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
#%%
def save_as_pickle(fn, data):
    """
    Function to save data as a pickled file
    Parameters
    ----------
    fn : str
        File directory.
    data : any type, but recommend dictionary
        data to save.
    Returns
    -------
    None.
    """
    with open(fn, 'wb') as to_write:
        pickle.dump(data, to_write)
    print('Saved data to "' + fn + '"')

#--------------------------------------------------------
def read_from_pickle(fn):
    """
    Function to read data from a pickled file
    Parameters
    ----------
    fn : str
        File directory.
    data : any type, but recommend dictionary
        data to read.
    Returns
    -------
    data : same as data
        Read in this variable.
    """
    with open(fn,'rb') as read_file:
        data = pickle.load(read_file)
    print('Read data from "' + fn + '"')
    return data

#--------------------------------------------------------
def get_date(txt):
    """
    Get date to convert to date name

    Parameters
    ----------
    txt : str
        text describing date and time.

    Returns
    -------
    str
        date.

    """
    return txt.split(' ')[0]
#--------------------------------------------------------
def get_time(txt):
    """
    Get time to convert to day part

    Parameters
    ----------
    txt : str
        text describing date and time.

    Returns
    -------
    str
        time.

    """
    return txt.split(' ')[-1]
#--------------------------------------------------------
def get_day_parts(time):
    """
    Separate time 24h into day parts.
    ref: https://stackoverflow.com/questions/55571311/get-part-of-day-morning-afternoon-evening-night-in-python-dataframe

    Parameters
    ----------
    time: str
          time.

    Returns
    -------
    pandas series
        day parts.

    """

    datetime = dd.to_datetime(time)
    datetime2 = (datetime.dt.hour % 24 + 4) // 4
    datetime2 = datetime2.replace({1: 'late_night',
                                   2: 'early_morning',
                                   3: 'morning',
                                   4: 'afternoon',
                                   5: 'evening',
                                   6: 'night'})

    return datetime2

#--------------------------------------------------------
def get_seasons(time):
    """
    Separate 12 mon into seasons.
    ref: https://stackoverflow.com/questions/55571311/get-part-of-day-morning-afternoon-evening-night-in-python-dataframe

    Parameters
    ----------
    time: str
          time.

    Returns
    -------
    pandas series
        seasons.

    """

    datetime = dd.to_datetime(time)
    datetime2 = (datetime.dt.month % 24 + 2) // 3
    datetime2 = datetime2.replace({1: 'spring',
                                   2: 'summer',
                                   3: 'fall',
                                   4: 'winter',
                                   })

    return datetime2
#--------------------------------------------------------
def my_fillna(df, feature_name):
    """
    Fill nan values with average values when grouping
    by state and season

    Parameters
    ----------
    df : pandas data frame
        the entire data frame.
    feature_name : str
        column name.

    Returns
    -------
    df : pandas data frame
        processed data frame.

    """
    df[feature_name+'_na'] = False
    df.loc[df[feature_name].isnull(),feature_name+'_na'] = True

    state_mean = (df[['state', 'season', feature_name]]
                  .groupby(['state', 'season'], as_index=False)
                  .mean()
                  .reset_index()
                 )
    state = df['state'].unique()
    season = ['spring', 'summer', 'fall', 'winter']
    for s in state:
        for se in season:
            mask = (df['state'] == s) & (df['season'] == se)
            df[feature_name][mask] = (df[feature_name][mask]
                                          .fillna(state_mean[feature_name]\
                                                  [(state_mean['state'] == s) &
                                                   (state_mean['season'] == se)
                                                  ].values[0],inplace=False)
                                     )
    return df



#--------------------------------------------------------

def build_nn(n_layers, input_shape, n_units, n_classes,
             activation, initializer, optimizer,
             metrics, dropout_rate=None):
    """
    Build a neural network.
    Ref: https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404

    Parameters
    ----------
    n_layers : int
        number of layers.
    input_shape : int
        input shape, i.e. number of features.
    n_units : int
        number of units per layer.
    n_classes : int
        number of classes.
    activation : str
        name of activation function.
    initializer : str
        name of initializer.
    optimizer : str
        name of optimizer.
    metrics : list of str
        name of metrics.
    dropout_rate : float
        dropout rate

    Returns
    -------
    mdl : keras neural network
        keras neural network.

    """
    if isinstance(n_units, list):
        assert len(n_units) == n_layers
    else:
        n_units = [n_units] * n_layers

    mdl = keras.Sequential()
    # Adds first hidden layer with input_dim parameter
    mdl.add(layers.InputLayer(input_shape=input_shape,
                              name='input'))

    # Adds remaining hidden layers
    for i in range(1, n_layers + 1):
        mdl.add(layers.Dense(units=n_units[i-1],
                             activation=activation,
                             kernel_initializer=initializer,
                             name='h{}'.format(i)
                             )
                )
        mdl.add(layers.BatchNormalization())
        if dropout_rate is not None:
            mdl.add(layers.Dropout(dropout_rate[i-1]))

    # Adds output layer
    mdl.add(layers.Dense(units=n_classes, activation='softmax',
                         kernel_initializer=initializer, name='output'))
    # Compiles the model
    mdl.compile(loss='categorical_crossentropy',
                optimizer=optimizer, metrics=metrics)

    return mdl
#--------------------------------------------------------
def disp_confusion_matrix(cf_matrix,
                          vmin,vmax,
                          cmap='Blues',
                          annot_kws={"size": 15}
                          ):
    """
    Function to display confusion matrix with details reported
    Parameters
    ----------
    cf_matrix : numpy array
        confusion matrix from sklearn.
    Returns
    -------
    ax : handle
        handle from seaborns.heatmap.
    """

    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]

    cf_matrix_norm = cf_matrix / (np.sum(cf_matrix, axis=1)[:, None])

    group_percentages = ['{0:.2%}'.format(value) for value in
                         (cf_matrix_norm).flatten()]

    labels = [f'{v1}\n{v2}' for v1, v2 in
              zip(group_percentages, group_counts)]
    labels = np.asarray(labels).reshape(cf_matrix.shape)

    ax = sns.heatmap(cf_matrix_norm * 100, annot=labels, annot_kws=annot_kws,
                     fmt='', cmap=cmap, vmin=vmin, vmax=vmax,
                     xticklabels=True, yticklabels=True)

    return ax
