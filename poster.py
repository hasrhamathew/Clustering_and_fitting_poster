# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 00:56:09 2023

@author: harsha
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def read_file(file_name):
    """
    Function reads data according to the file name passed and returns
    a dataframe with year as column and country as column

    """
    data = pd.read_csv(file_name)
    data = data[['Country Name', '1990', '1995', '2000', '2005',
                 '2010', '2015', '2019']]
    data.dropna(inplace = True)
    year_col = data
    year_col.set_index("Country Name")
    country_col = data.transpose()
    return year_col, country_col
# Reading the csv files
labor_force, contry_col = read_file("labor_force.csv")
print(labor_force.describe())
labor_force_female, col = read_file("labor_force_female.csv")
print(labor_force_female)
print(labor_force_female.describe())

pd.plotting.scatter_matrix(contry_col, figsize=(10,10)
plt.show()
# countries = np.array(contry_col['Country Name'])
# print(countries)

