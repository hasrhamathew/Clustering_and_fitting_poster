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

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper


def exp_growth(t, scale, growth):
    """ 
    Computes exponential function with scale and growth as free parameters

    """

    f = scale * np.exp(growth * (t-1950))

    return f


# Reading the csv files
labor_force, labor_force_transpose = read_file("labor_force.csv")
print(labor_force.describe())
labor_force_female, labor_force_female_transpose = read_file(
    "labor_force_female.csv")
print(labor_force_female.describe())
# Selecting the number of rows needed
labor_force = labor_force.head(70)
labor_force_female = labor_force_female.head(70)

final_df = pd.DataFrame()

final_df['Total labor force'] = labor_force['2020']
final_df['Female Labor force'] = labor_force_female['2020']
# Heatmap
heat_corr(final_df, 10)
# Plotting the scatter plot
pd.plotting.scatter_matrix(final_df, figsize=(9.0, 9.0))
# helps to avoid overlap of labels
plt.tight_layout()
plt.show()

# Normalise dataframe and inspect result
df_fit = final_df[["Total labor force", "Female Labor force"]].copy()

df_fit = norm_df(df_fit)
for ic in range(2, 10):
    # Set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # Extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fit, labels))

# Since silhouette score is highest for 3 , clustering for number = 3
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(9.0, 9.0))
# Plotting scatter plot
plt.scatter(df_fit["Total labor force"], df_fit["Female Labor force"],
            c=labels, cmap="Accent", )

# Plotting cluster centre for 3 clusters
for ic in range(3):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("Total labor force", fontsize=15)
plt.ylabel("Female Labor force", fontsize=15)
plt.title("Cluster Diagram with 3 clusters", fontsize=15)
plt.show()
