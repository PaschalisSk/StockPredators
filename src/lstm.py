import keras as K
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.model_selection import ParameterGrid

import data
import metrics

ex = Experiment('random_data2')
ex.observers.append(MongoObserver.create())

# # config to check for batch size 'lstm_batch_size'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'days_back': [5],
#     'days_forward': [1],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [100],
#     'num_hidden_layers': [1],
#     'seed': [0, 1, 2],
#     'learning_rate': [0.001],
#     'batch_size': [8, 16, 32],
#     'activation': ['tanh'],
#     'optimizer': ['adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [5]
# }

# #  config to check for batch size 'lstm_model_conf'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'days_back': [5],
#     'days_forward': [1],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [50, 100, 150],
#     'num_hidden_layers': [1, 2],
#     'seed': [0, 1, 2],
#     'learning_rate': [0.01, 0.001, 0.0001],
#     'batch_size': [16],
#     'activation': ['tanh'],
#     'optimizer': ['adagrad', 'adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [2, 5, 10]
# }

# #  config to check for days 'lstm_days_conf'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'days_back': [5, 10],
#     'days_forward': [1, 5, 10],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [150],
#     'num_hidden_layers': [2],
#     'seed': [0, 1, 2],
#     'learning_rate': [0.001],
#     'batch_size': [16],
#     'activation': ['tanh'],
#     'optimizer': ['adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [5]
# }

# # config to check for batch size 'lstm_sent_trends_batch_size'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'use_sent_and_trends': [True],
#     'days_back': [5],
#     'days_forward': [1],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [100],
#     'num_hidden_layers': [1],
#     'seed': [0, 1, 2],
#     'learning_rate': [0.001],
#     'batch_size': [8, 16, 32],
#     'activation': ['tanh'],
#     'optimizer': ['adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [5]
# }

# #  config to check for batch size 'lstm_sent_trends_model_conf'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'use_sent_and_trends': [True],
#     'days_back': [5],
#     'days_forward': [1],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [50, 100, 150],
#     'num_hidden_layers': [1, 2],
#     'seed': [0, 1, 2],
#     'learning_rate': [0.01, 0.001, 0.0001],
#     'batch_size': [16],
#     'activation': ['tanh'],
#     'optimizer': ['adagrad', 'adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [2, 5, 10]
# }

# #  config to check for batch size 'lstm_sent_trends_days_conf'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'use_sent_and_trends': [True],
#     'days_back': [5, 10],
#     'days_forward': [1, 5, 10],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [100],
#     'num_hidden_layers': [1],
#     'seed': [0, 1, 2],
#     'learning_rate': [0.01],
#     'batch_size': [16],
#     'activation': ['tanh'],
#     'optimizer': ['adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [2]
# }

# #  config to export plot data'lstm_sent_plot_run'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'use_sent_and_trends': [True],
#     'days_back': [5],
#     'days_forward': [1],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [100],
#     'num_hidden_layers': [1],
#     'seed': [0],
#     'learning_rate': [0.01],
#     'batch_size': [16],
#     'activation': ['tanh'],
#     'optimizer': ['adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [2]
# }
# #  config to export plot data'lstm_plot_run'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'days_back': [5],
#     'days_forward': [1],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [150],
#     'num_hidden_layers': [2],
#     'seed': [0],
#     'learning_rate': [0.001],
#     'batch_size': [16],
#     'activation': ['tanh'],
#     'optimizer': ['adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [5]
# }
# #  config to export plot data'lstm_sent_trends_days_conf_old_params'
# config_options = {
#     'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
#     'use_sent_and_trends': [True],
#     'days_back': [5,10],
#     'days_forward': [1, 5, 10],
#     'max_epochs': [5000],
#     'early_stopping_threshold': [20],
#     'num_neurons': [150],
#     'num_hidden_layers': [2],
#     'seed': [0, 1, 2],
#     'learning_rate': [0.001],
#     'batch_size': [16],
#     'activation': ['tanh'],
#     'optimizer': ['adam'],
#     'kernel_init': ['glorot_uniform'],
#     'regularization': [None],
#     'loss': ['MSE'],
#     'timesteps': [5]
# }
#  config to export plot data'random_data'
config_options = {
    'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
    'use_sent_and_trends': [True],
    'days_back': [5],
    'days_forward': [1],
    'max_epochs': [5000],
    'early_stopping_threshold': [20],
    'num_neurons': [150],
    'num_hidden_layers': [2],
    'seed': [0, 1, 2],
    'learning_rate': [0.001],
    'batch_size': [16],
    'activation': ['tanh'],
    'optimizer': ['adam'],
    'kernel_init': ['glorot_uniform'],
    'regularization': [None],
    'loss': ['MSE'],
    'timesteps': [5]
}
config_combinations = list(ParameterGrid(config_options))


@ex.main
def main(_run, stock_file, days_back, days_forward, max_epochs,
         early_stopping_threshold, num_neurons, num_hidden_layers,
         seed, learning_rate, batch_size, activation, optimizer,
         kernel_init, regularization, loss, timesteps, use_sent_and_trends=False):
    # Read the stocks csv into a dataframe
    stock = data.Stocks(stock_file)
    stock.calc_patel_TI(days_back)
    if use_sent_and_trends:
        # If we have a sentiment file add it to the stock df
        sentiments = pd.read_csv('../data/nytarticles/microsoft.2013-12-31.2018-12-31.imputed.sent.csv', index_col='date')
        trends = pd.read_csv('../data/trends/msft.2013-12-31.2018-12-31.fixed.dates.csv', index_col='date')
        sent_trends = pd.merge(sentiments, trends, how='left',
                          left_index=True, right_index=True)
        sent_trends['sent_trends'] = sent_trends['sentiment'] * sent_trends['msft']
        import numpy as np
        sent_trends['randNumCol'] = np.random.randint(1, 100, sent_trends.shape[0])
        stock.df = pd.merge(stock.df, sent_trends, how='left',
                         left_index=True, right_index=True)
        stock.df.drop(['sentiment', 'msft', 'sent_trends'], axis='columns', inplace=True)

    stock.shift(days_forward)

    # Create the model
    model = K.Sequential()

    # Create the kernel initializer with the seed
    if kernel_init == 'glorot_uniform':
        kernel_initializer = K.initializers.glorot_uniform(seed)
    else:
        raise NotImplementedError

    # Add the layers
    return_sequences = True
    if num_hidden_layers == 1:
        return_sequences = False
    data_dim = stock.raw_values()['X'].shape[1]
    model.add(K.layers.LSTM(num_neurons,
                            input_shape=(timesteps, data_dim),
                            activation=activation,
                            return_sequences=return_sequences,
                            kernel_initializer=kernel_initializer))

    for i in range(num_hidden_layers - 1):
        # If not in the last layer return sequences
        if i != num_hidden_layers - 2:
            model.add(K.layers.LSTM(num_neurons,
                                    activation=activation,
                                    return_sequences=True,
                                    kernel_initializer=kernel_initializer))
        else:
            model.add(K.layers.LSTM(num_neurons,
                                    activation=activation,
                                    kernel_initializer=kernel_initializer))

    # Add output layer
    model.add(K.layers.Dense(1, activation='linear',
                             kernel_initializer=kernel_initializer))

    # Define Root Mean Squared Relative Error metric
    def root_mean_squared_relative_error(y_true, y_pred):
        squared_relative_error = K.backend.square((y_true - y_pred) /
                                                  K.backend.clip(
                                                      K.backend.abs(y_true),
                                                      K.backend.epsilon(),
                                                      None))
        mean_squared_relative_error = K.backend.mean(squared_relative_error,
                                                     axis=-1)
        return K.backend.sqrt(mean_squared_relative_error)

    # Define Direction Accuracy metric
    def direction_accuracy(y_true, y_pred):
        # sign returns either -1 (if <0), 0 (if ==0), or 1 (if >0)
        true_signs = K.backend.sign(y_true[days_forward:] -
                                    y_true[:-days_forward])
        pred_signs = K.backend.sign(y_pred[days_forward:] -
                                    y_true[:-days_forward])

        equal_signs = K.backend.equal(true_signs, pred_signs)
        return K.backend.mean(equal_signs, axis=-1)

    # Create the optimizer
    if optimizer == 'adagrad':
        optimizer = K.optimizers.Adagrad(learning_rate)
    elif optimizer == 'adam':
        optimizer = K.optimizers.Adam(learning_rate)
    else:
        raise NotImplementedError

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['mean_absolute_percentage_error',
                           'mean_absolute_error',
                           root_mean_squared_relative_error,
                           'mean_squared_error',
                           direction_accuracy])

    # Create the logging callback
    # The metrics are logged in the run's metrics and at heartbeat events
    # every 10 secs they get written to mongodb
    def on_epoch_end_metrics_log(epoch, logs):
        for metric_name, metric_value in logs.items():
            # The validation set keys have val_ prepended to the metric,
            # add train_ to the training set keys
            if 'val' not in metric_name:
                metric_name = 'train_' + metric_name

            _run.log_scalar(metric_name, metric_value, epoch)

    metrics_log_callback = K.callbacks.LambdaCallback(
        on_epoch_end=on_epoch_end_metrics_log)

    callbacks_list = [K.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=early_stopping_threshold),
                      K.callbacks.ModelCheckpoint(
                          filepath='../models/best_model.h5',
                          monitor='val_loss',
                          save_best_only=True),
                      metrics_log_callback]

    model.fit(stock.raw_values_lstm_wrapper(dataset='train', norm=True,
                                            timesteps=timesteps)['X'],
              stock.raw_values_lstm_wrapper(dataset='train', norm=True,
                                            timesteps=timesteps)['y'],
              epochs=max_epochs, batch_size=batch_size, verbose=0,
              callbacks=callbacks_list,
              validation_data=(stock.raw_values_lstm_wrapper(dataset='val',
                                                             norm=True,
                                                             timesteps=timesteps)['X'],
                               stock.raw_values_lstm_wrapper(dataset='val',
                                                             norm=True,
                                                             timesteps=timesteps)['y'])
              )

    # Calculate metrics for normalized values
    test_norm_metrics = model.evaluate(
        stock.raw_values_lstm_wrapper(dataset='test', norm=True,
                                      timesteps=timesteps)['X'],
        stock.raw_values_lstm_wrapper(dataset='test', norm=True,
                                      timesteps=timesteps)['y'],
        verbose=0)

    # Log the metrics from the normalized values
    for metric in zip(model.metrics_names, test_norm_metrics):
        _run.log_scalar('test_norm_' + metric[0], metric[1])

    # Now calculate and save the unnormalised metrics
    # Predict returns normalised values
    y_pred_norm = model.predict(stock.raw_values_lstm_wrapper(
        dataset='test',
        norm=True,
        timesteps=timesteps)['X'])
    # Scale the output back to the actual stock price
    y_pred = stock.denorm_predictions(y_pred_norm)

    # Calculate the unnormalized metrics
    y_true = stock.raw_values_lstm_wrapper(dataset='test',
                                           timesteps=timesteps)['y']

    # df1 = pd.DataFrame({'date': stock.df.index.values[-y_pred.shape[0]:], 'y_pred': y_pred.flatten(), 'y_true': y_true.flatten()})
    # df1.set_index('date', inplace=True)
    # df1.to_csv('plot_data_lstm.csv')
    test_metrics = {
        'test_loss': metrics.mean_squared_error(
            y_true, y_pred),
        'test_mean_absolute_percentage_error': metrics.mean_absolute_percentage_error(
            y_true, y_pred),
        'test_mean_absolute_error': metrics.mean_absolute_error(
            y_true, y_pred),
        'test_root_mean_squared_relative_error': metrics.root_mean_squared_relative_error(
            y_true, y_pred),
        'test_mean_squared_error': metrics.mean_squared_error(
            y_true, y_pred),
        'test_direction_accuracy': metrics.direction_accuracy(
            y_true, y_pred, days_forward)
    }

    # Save the metrics
    for metric_name, metric_value in test_metrics.items():
        _run.log_scalar(metric_name, metric_value)
