import numpy as np
import tensorflow as tf
from keras import *

import data

# Initialise random state
R_STATE = np.random.RandomState(0)
# Define the stocks file we want to read
STOCK_FILE = '../data/stocks/MSFT.2013-12-31.2018-12-31.csv'

# Read the stocks csv into a dataframe
stock = data.Stocks(STOCK_FILE)
stock.calc_patel_TI(10, 1)
#stock.normalize()
stock.shuffle(R_STATE)

# df_X holds the technical indicators
df_X = stock.df.drop(['close'], axis=1)
# df_y holds the closing prices
df_y = stock.df[['close']]
# Convert dataframes into np arrays of float32
ar_X = np.asarray(df_X.values, dtype='float32')
ar_y = np.asarray(df_y.values, dtype='float32')
# First 70% training
train_limit = int(0.7 * len(ar_X))
# 70%-85% validation
val_limit = int(0.85 * len(ar_X))

raw_X_train = ar_X[:train_limit]
raw_y_train = ar_y[:train_limit]
raw_X_val = ar_X[train_limit:val_limit]
raw_y_val = ar_y[train_limit:val_limit]
raw_X_test = ar_X[val_limit:]
raw_y_test = ar_y[val_limit:]


# the input dim is equal to the number of technical indicators we use
input_dimensions = ar_X.shape[1]
# 1 cells for the output
output_dimensions = ar_y.shape[1]
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
             callbacks.EarlyStopping(monitor='val_loss', patience=2),
             callbacks.ModelCheckpoint(filepath='../models/best_model.h5',
                                       monitor='val_loss',
                                       save_best_only=True)]

model.fit(raw_X_train, raw_y_train, epochs=3000, batch_size=32, verbose=2,
          callbacks=callbacks,
          validation_data=(raw_X_val, raw_y_val))

print('test')
