# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 18:14:30 2023

@author: sande
"""

#importing pandas module to read the file(data set) and to calculate the statistical property(Describe)
import pandas as pd

#importing numpy module to calculate the statistical properties (mean and standard deviation)
import numpy as np

#importing pyplot from matplotlib module to plot the visualization graphs
import matplotlib.pyplot as plt

#importing KMeans from sklearn object to identify the clusters
from sklearn.cluster import KMeans

#importing LabelEncoder to encode the categories of the data
from sklearn.preprocessing import LabelEncoder

#importing the custom error package 
import err_ranges as error

#importing curve_fit from scipy module to predict the population groth of Australia
from scipy.optimize import curve_fit

#defining a function to read the dataset and to produce original and transposed dataframes
def read_data_file(input_file_name,countries):
    #reading the data set using pandas module
    dataFrame = pd.read_csv(input_file_name)
    #cleaning the dataFrame by filling the NaN values with 0
    cleaned_dataFrame = dataFrame.fillna(0)
    #slicing the data frame by selecting fewe countries of our option
    sliced_dataFrame = cleaned_dataFrame[cleaned_dataFrame['Country Name'].isin(countries)]
    #creating a new data frame with countires as first column using the sliced data frame
    dataFrame_countries = pd.DataFrame(sliced_dataFrame)
    print('Original DataFrame:\n',dataFrame_countries)
    #transposing the sliced data frame
    transposed_dataFrame = pd.DataFrame.transpose(sliced_dataFrame)
    #creating a header
    header = transposed_dataFrame.iloc[0].values.tolist()
    #assigning the header to the transposed data frame
    transposed_dataFrame.columns = header
    #assigning the transposed dataframe with years as first column to a new variable
    dataFrame_years = transposed_dataFrame
    print('Transposed DataFrame:\n',dataFrame_years)
    #returning the 2 dataframes (one dataframe with countries as first column and other dataframe with years as first column)
    return dataFrame_countries,dataFrame_years

#function to calculate the logistic value
def logi(t, n0, g, t0):
    f = n0 / (1+np.exp(-g*(t - t0)))
    return f

#function to calculate the exponential value
def exponen_graph(t, n0, g):
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

#calling the function that produces two dataframes: one with countries as columns and another with years as columns
df_countries, df_years= read_data_file('C:/Users/sande/MS-DS/API_19_DS2_en_csv_v2_4700503-Copy.csv',['Afghanistan','Albania','Argentina','Austria','Belgium','Bangladesh','Brazil','Canada','Switzerland','Chile','China','Colombia','Denmark','Dominican Republic','Algeria','Spain','Finland','Fiji','France','United Kingdom','Greece','Greenland','Hungary','Indonesia','India','Ireland','Iraq','Iceland','Israel','Italy','Jamaica','Japan','Lebanon','Luxembourg','Morocco','Mexico','Myanmar','Netherlands','New Zealand','Pakistan','Peru','Poland','Romania','Russian Federation','Sweden','Thailand','Tunisia','Turkiye','Uruguay','United States','Vietnam','South Africa','Zimbabwe'])

#selecting only few idnicators for clustering and fitting purpose
selected_indicators = ['Urban population growth (annual %)','CO2 emissions from liquid fuel consumption (% of total)','CO2 emissions from solid fuel consumption (% of total)']

#filtering the original dataset with the selected indicators
selected_df = df_countries.loc[df_countries["Indicator Name"].isin(selected_indicators)]

#dropped two columns for clustering purpose
selected_df = selected_df.drop(columns=['Country Name','Indicator Name',],axis=1)

#performing data preproceesing by labelling the column as a categorical column
label_encoder = LabelEncoder()
#converting the Class column into encoding values to compare with ground truth values after clustering predictions
selected_df['Country Code'] = label_encoder.fit_transform(selected_df['Country Code'])
selected_df['Indicator Code'] = label_encoder.fit_transform(selected_df['Indicator Code'])

# normalizing the input data
selectedData_rank=selected_df.rank(method='first')
#normalized is subraction from average main on std
normalized_data=(selectedData_rank-selectedData_rank.mean())/selectedData_rank.std()
print('Normalized Data:',normalized_data)

#using the elbow method to find out the K values to form the clusters
sum_cluster_squares = []
for i in range(1, 11):
    k_means_algo = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    k_means_algo.fit(selected_df[['1970','2010']])
    sum_cluster_squares.append(k_means_algo.inertia_)
plt.plot(range(1, 11), sum_cluster_squares)
plt.title('Elbow Method')
plt.xlabel('Clusters')
plt.ylabel('Inertia Value')
plt.show()

#forming the clusters based on the elbow method result
k_means_algo=KMeans(n_clusters=3).fit(normalized_data[['1970','2010']])
no_of_clusters=selected_df.copy(deep=True)
#adding cluster output to the new column of the data frame for visiualization purpose
no_of_clusters['Cluster']=k_means_algo.labels_ 

#fitting the data to produce the centers
k_means_centers=KMeans(n_clusters=3).fit(selected_df[['1970','2010']])

# drawing scatter plot: Clump Thickness with Bland Chromatin
plt.scatter(
no_of_clusters.loc[no_of_clusters['Cluster']==0]['1970'],
no_of_clusters.loc[no_of_clusters['Cluster']==0]['2010'],c='b')
plt.scatter(
no_of_clusters.loc[no_of_clusters['Cluster']==1]['1970'],
no_of_clusters.loc[no_of_clusters['Cluster']==1]['2010'],c='r')
plt.scatter(
no_of_clusters.loc[no_of_clusters['Cluster']==2]['1970'],
no_of_clusters.loc[no_of_clusters['Cluster']==2]['2010'],c='g')
cluster_center = k_means_centers.cluster_centers_
plt.scatter(cluster_center[:, 0] , cluster_center[:, 1], c='black', s=200, alpha=0.7);
plt.title('Cluster Graph')
plt.xlabel('Normalized Range - X-axis')
plt.ylabel('Normalized Range - Y-axis')
plt.legend()
plt.grid()
plt.show()

"""
Curve Fitting:
"""

#calling the function with the data set and our own selection of countries.
countries_df,years_df = read_data_file('C:/Users/sande/MS-DS/API_19_DS2_en_csv_v2_4700503-Copy.csv',['Australia','Bolivia','Canada','Switzerland','Denmark'])

#selecting the Urban Population values of Austrila as series dataset
selected_data_series = years_df.iloc[4:,3]
print('Type of selected data from dataset:',type(selected_data_series))

#converting the series dataset into pandas dataframe
selected_data_df = selected_data_series.to_frame()
print('Converted Datatype:',type(selected_data_df))

#resetting the indexes of the dataframe to assign new column names
selected_data_df = selected_data_df.reset_index()

#assigning the new column names
selected_data_df.columns = ['Year','Urban population growth (annual %)']

#plotting the data which prodcues exponential graph
selected_data_df.plot('Year','Urban population growth (annual %)')

#coverting the values of Year to numeric type
selected_data_df['Year'] = pd.to_numeric(selected_data_df['Year'])

#calculating the parameter and covariance for the curve fit using the exponential function
parameter, covariance = curve_fit(exponen_graph,selected_data_df['Year'],selected_data_df['Urban population growth (annual %)'], p0=(73233967692.102798, 0.03))

selected_data_df['Fit'] = exponen_graph(selected_data_df['Year'], *parameter)

#plotting the curve fit with calculated values of exponential function
selected_data_df.plot('Year',['Urban population growth (annual %)','Fit'])
plt.title('Exponential Growth Graph')
plt.show()

#calculating the parameter and covariance for the curve fit using the logistic function
parameter, covariance = curve_fit(logi, selected_data_df['Year'], selected_data_df['Urban population growth (annual %)'], p0=(3e12,0.03,2000.0))

#calculating sigma value using the covariance
sigma = np.sqrt(np.diag(covariance))
print('Parameters:', parameter)
print('Standard Deviation:', sigma)
selected_data_df['Fit'] = logi(selected_data_df['Year'], *parameter)

#plotting the curve fit with calculated values of logistic function
selected_data_df.plot('Year', ['Urban population growth (annual %)', 'Fit'])
plt.title('Logistic Functional Graph')
plt.show()

#selecting the range of years taht we used for clustering and fitting
range_year = np.arange(1970, 2010)
co2_liquid_emissions = logi(range_year, *parameter)

#calculating the upper limit and lowe limit of the error range using the custom module function
err_range_low, err_range_up = error.err_ranges(range_year, logi, parameter, sigma)

#plotting the graph to display the error ranges and confidence levels
plt.plot(selected_data_df['Year'], selected_data_df['Urban population growth (annual %)'], label='Urban population growth (annual %)')
plt.plot(range_year, co2_liquid_emissions, label='Co2 Emissions from liquid fuel consumption')
plt.fill_between(range_year, err_range_low, err_range_up, color='Yellow', alpha=0.7)
plt.xlabel('Year')
plt.ylabel('Urban population growth (annual %)')
plt.legend()
plt.show()
