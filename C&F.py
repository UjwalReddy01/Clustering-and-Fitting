# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:29:25 2024

@author: HP
"""

#Importing pandas package to read the file,for deriving pandas features and stats properties
import pandas as pd
#Importing numpy package
import numpy as np
#Importing matplotlib package for data plotting and visualization
import matplotlib.pyplot as plt

#defining function to produce two dataframes,one with countries as columns and one with years as coulmns
def readcsv(input_file,countries):
    data = pd.read_csv(input_file)
    #replacing the null values with zeroes using fillna() method
    dropping_values = data.fillna(0) 
    test = dropping_values[dropping_values['Country Name'].isin(countries)]
    df_countries = pd.DataFrame(test)
    print(df_countries)
    #transposing data to derive dataframe with years as columns
    transpose=pd.DataFrame.transpose(test)
    header = transpose.iloc[0].values.tolist()
    transpose.columns = header
    transpose = transpose.iloc[0:]
    df_years = transpose
    print(transpose)
    return df_countries,df_years