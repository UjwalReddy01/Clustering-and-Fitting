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

#calling the function to produce two dataframes by choosing few countries
df_co,df_yr=readcsv('C:/Users/HP/Downloads/ADSASSIGNMENT2/API_19_DS2_en_csv_v2_4700503-Copy.csv',['Afghanistan','Albania','Argentina','Austria','Belgium','Bangladesh','Brazil','Canada','Switzerland','Chile','China','Colombia','Denmark','Dominican Republic','Algeria','Spain','Finland','Fiji','France','United Kingdom','Greece','Greenland','Hungary','Indonesia','India','Ireland','Iraq','Iceland','Israel','Italy','Jamaica','Japan','Lebanon','Luxembourg','Morocco','Mexico','Myanmar','Netherlands','New Zealand','Pakistan','Peru','Poland','Romania','Russian Federation','Sweden','Thailand','Tunisia','Turkiye','Uruguay','United States','Vietnam','South Africa','Zimbabwe'])
#Defining features with few indicator names
features = ['CO2 emissions from liquid fuel consumption (% of total)','Urban population growth (annual %)','CO2 emissions from solid fuel consumption (% of total)']

#Creating a new dataframe with few indicator names
df = df_co.loc[df_co['Indicator Name'].isin(features)]
print(df)

#Using Drop function to drop column names - Country name & Indicator name
df= df.drop(columns=['Country Name','Indicator Name'],axis=1)
print(df)
