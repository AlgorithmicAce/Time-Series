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
def df_to_X_Y(df, winsize):
    df_as_np = df.to_numpy()
    X = []
    Y = []
    for i in range(len(df) - winsize):
        x = df[i:i+winsize]
        y = df[[i+winsize]]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

#Creating numpy array from dataframe and chaning the shape
trainx, trainy = df_to_X_Y(temp, 5)
trainx = np.reshape(trainx, (-1, 5, 1))

#Creating model using TensorFlow
model = tf.keras.Sequential()

model.add(tf.keras.layers.InputLayer(input_shape = (5,1)))
model.add(tf.keras.layers.LSTM(64))
model.add(tf.keras.layers.Dense(8, 'relu'))
model.add(tf.keras.layers.Dense(1, 'linear'))

#Compiling the model
model.compile(
    optimizer = tf.kras.optimizers.Adam(),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
)

#Creating train and test array
xtrain, ytrain = trainx[:6000], trainy[:6000]
xtest, ytest = trainx[6000:], trainy[6000:]

#Train/Fitting the model with epoch of 15
model.fit(xtrain, ytrain, epochs = 15)

#Predicting the model using test data
ypred = model.predict(xtest)

#Plotting the predicted data and actual data
plt.plot(ypred, label = 'Predicted Value')
plt.plot(ytest, label = 'Actual Value')
plt.legend()
plt.show()
