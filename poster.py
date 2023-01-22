# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 04:40:45 2023

@author: harsh
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 00:56:09 2023

@author: harsha
"""

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import itertools as iter
def read_file(file_name):
    """
    Function reads data according to the file name passed and returns
    a dataframe with year as column and country as column

    """
    data = pd.read_csv(file_name)
    data = data[['Country Name', '1990', '1991', '1992', '1993', '1993', '1994',
                 '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002',
                 '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
                 '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018',
                 '2019', '2020']]
    data.dropna(inplace=True)
    year_col = data
    year_col.set_index("Country Name")
    country_col = data.transpose()
    return year_col, country_col


def heat_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns 
    in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='crest')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()
    

def norm(array):
    """
    Returns array normalised to [0,1]
    
    """
    min_val = np.min(array)
    max_val = np.max(array)

    scaled = (array-min_val) / (max_val-min_val)

    return scaled


def norm_df(df, first=0, last=None):
    """
    Returns all columns of the dataframe normalised to [0,1] with the 
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column
    
    """
    # iterate over all numerical columns
    for col in df.columns[first:last]:     # excluding the first column
        df[col] = norm(df[col])

    return df


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

