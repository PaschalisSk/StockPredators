import numpy as np
from tensorflow import keras
import tensorflow as tf
import time

import data

# Initialise random state
R_STATE = np.random.RandomState(0)
# Define the stocks file we want to read
STOCK_FILE = '../Data/Stocks/MSFT.2013-12-31.2018-12-31.csv'

# Read the stocks csv into a dataframe
stock = data.Stocks(STOCK_FILE)
stock.calc_patel_TI(10)
#stock.normalize()
stock.shuffle(R_STATE)

# KERAS IMPLEMENTATION
#model = keras.Sequential()
# TF IMPLEMENTATION
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

# We will use these as inputs to the model when it comes time to train it (assign values at run time)
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='y_train')

# We will use these as inputs to the model once it comes time to eval it
X_val_node = tf.constant(raw_X_val, name='X_val')
y_val_node = tf.constant(raw_y_val, name='y_val')

# We will use these as inputs to the model once it comes time to test it
X_test_node = tf.constant(raw_X_test, name='X_test')
y_test_node = tf.constant(raw_y_test, name='y_test')

# First layer takes in input and passes output to 2nd layer
weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name='weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name='biases_1')

weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, output_dimensions]), name='weight_2')
biases_2_node = tf.Variable(tf.zeros([output_dimensions]), name='biases_2')


def network(input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
    layer2 = tf.matmul(layer1, weight_2_node) + biases_2_node
    return layer2


y_train_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)

mean_squared_error = tf.losses.mean_squared_error(y_train_node, y_train_prediction)

# TODO: follow paper
optimizer = tf.train.AdagradOptimizer(0.005).minimize(mean_squared_error)

num_epochs = 100

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):

        start_time = time.time()

        _, cross_entropy_score = session.run([optimizer, mean_squared_error],
                                             feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})

        if epoch % 10 == 0:
            timer = time.time() - start_time

            print('Epoch: {}'.format(epoch), 'Current loss: {0:.4f}'.format(cross_entropy_score),
                  'Elapsed time: {0:.2f} seconds'.format(timer))

print('test')
