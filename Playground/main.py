import pandas as pd
import numpy as np
import datetime

import data

# Initialise random state
R_STATE = np.random.RandomState(0)
# Define the stocks file we want to read
STOCK_FILE = '../Data/Stocks/XU100.IS.1997-01-02.2007-12-31.csv'

# Read the stocks csv into a dataframe
stock = data.Stocks(STOCK_FILE)
# # Convert the closing price to a numpy array
# stock.create_np_array(column='close', dtype='float32')
#
# Technical indicators implemented for n=10
stock.moving_average(10)
stock.weighted_moving_average(10)
stock.momentum(10)
stock.stochastic_oscillator_K(10)
print('test')
