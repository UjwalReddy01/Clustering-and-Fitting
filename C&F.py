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

#Importing Label Encoder from sklearn as a part of data preprocessing step to replace categorical values to numerical values
from sklearn.preprocessing import LabelEncoder
label_encode = LabelEncoder()
df['Country Code'] = label_encode.fit_transform(df['Country Code'])
df['Indicator Code'] = label_encode.fit_transform(df['Indicator Code'])

#Normalizing the data values by mean and std deviation 
data_rank = df.rank(method='first')
data_normalized = (data_rank - data_rank.mean())/data_rank.std()
data_normalized.head(10)

#Forming clusters using K -means clustering algorithm by elbow method
from sklearn.cluster import KMeans
sum_of_squares = []
for i in range(1,11):
    k_means = KMeans(n_clusters = i,init = 'k-means++',random_state=42)
    k_means.fit(df[['1970','2010']])
    sum_of_squares.append(k_means.inertia_)
plt.plot(range(1,11),sum_of_squares)
plt.title('Knee of a curve')
plt.xlabel('Cluster values')
plt.ylabel('Values of Inertia')
plt.legend()
plt.show()

#Fitting the data into KMeans Algorithm
kmeans = KMeans(n_clusters=3).fit(data_normalized[['1970','2010']])
clusters = df.copy(deep=True)
clusters['Clusters'] = kmeans.labels_
clusters.head(10)

#Defining centers of the clusters 
kmeans_c = KMeans(n_clusters=3).fit(df[['1970','2010']])

#Plotting Three Different clusters with defined centers using scatter plot
plt.scatter(
clusters.loc[clusters['Clusters']==0]['1970'],
clusters.loc[clusters['Clusters']==0]['2010'],
c='r')
plt.scatter(
clusters.loc[clusters['Clusters']==1]['1970'],
clusters.loc[clusters['Clusters']==1]['2010'],
c='g')
plt.scatter(
clusters.loc[clusters['Clusters']==2]['1970'],
clusters.loc[clusters['Clusters']==2]['2010'],
c='b')

#Plotting the centers of the clusters using scatter plot
cen = kmeans_c.cluster_centers_
plt.scatter(cen[:,0],cen[:,1],c='black',s=200,alpha=0.5);
plt.title('Clusters')
plt.xlabel('X_Range')
plt.ylabel('Y_Range')
plt.legend()
plt.grid()
plt.show()
