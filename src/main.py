import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import *

import data

# Initialise random state
R_STATE = np.random.RandomState(0)
# Define the stocks file we want to read
STOCK_FILE = '../data/stocks/MSFT.2013-12-31.2018-12-31.csv'
# Past to calculate technical indicators
PAST_DAYS = 10
# Future days to predict value
FUTURE_DAYS = 10

# Read the stocks csv into a dataframe
stock = data.Stocks(STOCK_FILE)
stock.calc_patel_TI(PAST_DAYS, FUTURE_DAYS)
# stock.shuffle(R_STATE)

# the input dim is equal to the number of technical indicators we use
input_dimensions = stock.raw_values()['X'].shape[1]
# 1 cells for the output
output_dimensions = stock.raw_values()['y'].shape[1]
# 100 cells for the 1st layer
num_layer_1_cells = 100

# Create the model
model = Sequential()
# Add first hidden layer
model.add(layers.Dense(num_layer_1_cells,
          input_shape=(input_dimensions,),
          activation='sigmoid'))
# Add output layer
model.add(layers.Dense(output_dimensions, activation='linear'))

model.compile(optimizer=tf.train.AdagradOptimizer(0.005),
              loss='mse',
              metrics=['mse'])

callbacks = [callbacks.CSVLogger('../logs/training.csv'),
             #callbacks.EarlyStopping(monitor='val_loss', patience=2),
             callbacks.ModelCheckpoint(filepath='../models/best_model.h5',
                                       monitor='val_loss',
                                       save_best_only=True)]

model.fit(stock.raw_values(dataset='train', norm=True)['X'],
          stock.raw_values(dataset='train', norm=True)['y'],
          epochs=1000, batch_size=32, verbose=2,
          callbacks=callbacks,
          validation_data=(stock.raw_values(dataset='val', norm=True)['X'],
                           stock.raw_values(dataset='val', norm=True)['y']))

norm_y_pred = model.predict(stock.raw_values(dataset='test', norm=True)['X'])
denorm_y_pred = stock.denorm_predictions(norm_y_pred)
# def direction(real,pred):
#     total = real.shape[0]-1
#     a = real[1:] - real[:-1]
#     b = pred[1:] - real[:-1]
#     return np.sum(np.sign(a)  == np.sign(b))/total
#
plt.plot(stock.raw_values('test')['y'])
plt.plot(denorm_y_pred)
# print(direction(unscaled_y_test, unscaled_y_pred))
#plt.savefig(str(DAYS)+'-day-ahead.png')
plt.show()
print('test')
