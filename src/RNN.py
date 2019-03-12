import keras as K
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.model_selection import ParameterGrid
import numpy as np

import data
import metrics

ex = Experiment('rnn_test')
#ex.observers.append(MongoObserver.create())

config_options = {
    'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
    'days_back': [10],
    'days_forward': [1],
    # 'max_epochs': [5000],
    # 'early_stopping_threshold': [20],
    # 'num_neurons': [50, 100, 150],
    # 'num_hidden_layers': [1],
    # 'seed': [0, 1, 2],
    # 'learning_rate': [0.005, 0.01],
    # 'batch_size': [16],
    # 'activation': ['sigmoid'],
    # 'optimizer': ['adagrad'],
    # 'kernel_init': ['glorot_uniform'],
    # 'regularization': [None],
    # 'loss': ['MSE']
}

config_combinations = list(ParameterGrid(config_options))

@ex.main
def main(_run, stock_file, days_back, days_forward):
    # Read the stocks csv into a dataframe
    stock = data.Stocks(stock_file)
    stock.calc_patel_TI(days_back)
    stock.shift(days_forward)

    # Create the model
    model = K.Sequential()

    timesteps=1
    # The data dim is equal to the number of technical indicators we use
    data_dim = stock.raw_values()['X'].shape[1]
    model.add(K.layers.LSTM(10, input_shape=(timesteps, data_dim)))
    model.add(K.layers.Dense(1))
    model.compile(loss='MSE', optimizer='adagrad')

    callbacks_list = [K.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=30),
                      K.callbacks.ModelCheckpoint(
                          filepath='../models/best_model.h5',
                          monitor='val_loss',
                          save_best_only=True)]

    model.fit(stock.raw_values_lstm_wrapper(dataset='train', norm=True,
                                            timesteps=timesteps)['X'],
              stock.raw_values_lstm_wrapper(dataset='train', norm=True,
                                            timesteps=timesteps)['y'],
              epochs=5000, batch_size=16, verbose=2,
              callbacks=callbacks_list,
              validation_data=(stock.raw_values_lstm_wrapper(dataset='val',
                                                             norm=True,
                                                             timesteps=timesteps)['X'],
                               stock.raw_values_lstm_wrapper(dataset='val',
                                                             norm=True,
                                                             timesteps=timesteps)['y'])
              )

    # Calculate metrics for normalized values
    test_norm_metrics = model.evaluate(stock.raw_values_lstm_wrapper(dataset='test', norm=True,
                                            timesteps=timesteps)['X'],
              stock.raw_values_lstm_wrapper(dataset='test', norm=True,
                                            timesteps=timesteps)['y'])

    # Now calculate and save the unnormalised metrics
    # Predict returns normalised values
    y_pred_norm = model.predict(stock.raw_values_lstm_wrapper(dataset='test', norm=True,
                                            timesteps=timesteps)['X'])
    # Scale the output back to the actual stock price
    y_pred = stock.denorm_predictions(y_pred_norm)

    # Calculate the unnormalized metrics
    y_true = stock.raw_values_lstm_wrapper(dataset='test')['y']

    import matplotlib.pyplot as plt
    plt.plot(y_pred, label='pred')
    plt.plot(y_true, label='true')
    plt.legend()
    plt.show()
    print(metrics.mean_squared_error(
            y_true, y_pred))
