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

