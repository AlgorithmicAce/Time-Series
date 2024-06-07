#Importing all the necessary libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the .csv file using pandas
df = pd.read_csv('jena_climate_2009_2016.csv')

#Filtering the minute datas from the dataframe
df = df[5::6]

#Changes the index column and drops the datetime column
df.index = pd.to_datetime(df['Date Time'], format = "%d.%m.%Y %H:%M:%S")
df = df.drop(['Date Time'], axis = 1)

#We are only gonna predict temperature, so we are only gonna take the temperature column
temp = df['T (deg C)']

#Creating a function to change the dataset into X and Y, X being arrays within x numbers, and Y being the next predicted value
