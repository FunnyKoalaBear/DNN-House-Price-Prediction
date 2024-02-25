import keras 
import pandas as pd 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt #
from sklearn.model_selection import train_test_split #library that splits dataset into a testing set and a dataset, done to test model on data it hasnt seen before



# dataset = pd.read_csv("/Users/sbdagoat/Desktop/Codine/Python/AI/HousePrice DNN/housing_price_dataset.csv")
# print(dataset)

dataset = pd.read_csv("/Users/sbdagoat/Desktop/Codine/Python/AI/HousePrice DNN/HousingPrices.csv")
print(dataset)



#sets the x axis in the data to be Price
x = dataset.drop(columns = ["SalePrice"])

#sets the y axis as "everythign except Price column"
y = dataset[["SalePrice"]]

#allows to add layers sequentially to the model
model = keras.Sequential()

#input layer
#adding 3 layers of neurons 
model.add(keras.layers.Dense(8, activation = "relu", input_shape = (8, )))
#relu is an algorithm takes all values and normalises on a scale of 0 to 1 
#if closer to 1, 1, if it is closer to 0,0 
#sigmoid works like relu 

#This added a layer to the neural network with function Dense(), we added 8, unit which is power to 2
#if the activation makes a sigmoid! this puts all the values in the neural network in between 0 and.1
#the input shape is set to 5 as there are 5 other columns that need to go in as data against he price in the csv file attached 

#this is the hidden layer 
model.add(keras.layers.Dense(8, activation="relu"))
#I could try to add more layers but that will be done to optimize at a later time 

#the output layer
model.add(keras.layers.Dense(1))


#compiling the model 
#model.compile(optimizer= "Adam", loss = "mean_squared_error")model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam', loss='mean_squared_error')
#model.compile(keras.optimizers.Adam(learning_rate=1.0), 'mean_squared_error')



#fitting the model so that we can start the training 
model.fit(x, y, epochs = 30, callbacks=[keras.callbacks.EarlyStopping(patience=3)])

#model.fit(x, y, epochs = 30, callbacks = [keras.callbacks.EarlyStopping(5)] )

#use early stopping to reduce the overfitting 
#this stops doing epoch if the model is already gettig accurate enough 




#inputing a test data 
test_data = np.array([2003,	854,	1710,	2,	1,	3,	8,	2008])
print(model.predict(test_data.reshape(1,8), batch_size=1))

model.summary()
model.evaluate(x, y, batch_size = 64)


model.save("HousePricing Preiction.h5")

ourSavedModel = keras.models.load_model('HousePricing Preiction.h5')
test_data = np.array([2003,	854,	1710,	2,	1,	3,	8,	2008])
print(ourSavedModel.predict(test_data.reshape(1,8), batch_size=1))

#LETS GOOOO IT PREDICTED THE SAME THING AGAIN MEANING MODEL WAS CALLED
#AND OUR MODEL WORKS! 