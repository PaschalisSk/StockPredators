import keras as K
from sacred import Experiment
from sacred.observers import MongoObserver

import data
import metrics

ex = Experiment('baseline')
ex.observers.append(MongoObserver.create())


@ex.main
def main(_run, stock_file, days_back, days_forward, max_epochs,
         early_stopping_threshold, num_neurons, num_hidden_layers,
         seed, learning_rate, batch_size, activation, optimizer,
         kernel_init, regularization, loss):
    # Read the stocks csv into a dataframe
    stock = data.Stocks(stock_file)
    stock.calc_patel_TI(days_back, days_forward)
    # stock.shuffle(seed)
    # The input dim is equal to the number of technical indicators we use
    input_dimensions = stock.raw_values()['X'].shape[1]
    # The output dim is equal to the number of output numbers, 1 here
    output_dimensions = stock.raw_values()['y'].shape[1]

    # Create the model
    model = K.Sequential()

    # Create the kernel initializer with the seed
    if kernel_init == 'glorot_uniform':
        kernel_initializer = K.initializers.glorot_uniform(seed)
    else:
        raise NotImplementedError

    # Add first hidden layer
    model.add(K.layers.Dense(num_neurons,
                             input_shape=(input_dimensions,),
                             activation=activation,
                             kernel_initializer=kernel_initializer))

    # Create the extra layers
    for _ in range(num_hidden_layers - 1):
        model.add(K.layers.Dense(num_neurons, activation=activation,
                                 kernel_initializer=kernel_initializer))

    # Add output layer
    model.add(K.layers.Dense(output_dimensions, activation='linear',
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
        true_signs = K.backend.sign(y_true[1:] - y_true[:-1])
        pred_signs = K.backend.sign(y_pred[1:] - y_true[:-1])

        equal_signs = K.backend.equal(true_signs, pred_signs)
        return K.backend.mean(equal_signs, axis=-1)

    # Create the optimizer
    if optimizer == 'adagrad':
        optimizer = K.optimizers.Adagrad(learning_rate)
    else:
        raise NotImplementedError

    # Compile the model with the defined optimizers, loss function, and metrics
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

    model.fit(stock.raw_values(dataset='train', norm=True)['X'],
              stock.raw_values(dataset='train', norm=True)['y'],
              epochs=max_epochs, batch_size=batch_size, verbose=0,
              callbacks=callbacks_list,
              validation_data=(
                  stock.raw_values(dataset='val', norm=True)['X'],
                  stock.raw_values(dataset='val', norm=True)['y'])
              )

    # Calculate metrics for normalized values
    test_norm_metrics = model.evaluate(stock.raw_values(dataset='test',
                                                        norm=True)['X'],
                                       stock.raw_values(dataset='test',
                                                        norm=True)['y'],
                                       verbose=0)
    # Log the metrics from the normalized values
    for metric in zip(model.metrics_names, test_norm_metrics):
        _run.log_scalar('test_norm_' + metric[0], metric[1])

    # Now calculate and save the unnormalised metrics
    # Predict returns normalised values
    y_pred_norm = model.predict(stock.raw_values(dataset='test',
                                                 norm=True)['X'])
    # Scale the output back to the actual stock price
    y_pred = stock.denorm_predictions(y_pred_norm)

    # Calculate the unnormalized metrics
    y_true = stock.raw_values(dataset='test')['y']

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
            y_true, y_pred)
    }

    # Save the metrics
    for metric_name, metric_value in test_metrics.items():
        _run.log_scalar(metric_name, metric_value)
