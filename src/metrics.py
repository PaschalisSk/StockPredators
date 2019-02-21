import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    diff = np.abs((y_true - y_pred) / np.clip(np.abs(y_true),
                                              np.finfo(np.float32).eps,
                                              None))
    return 100. * np.mean(diff)


def root_mean_squared_relative_error(y_true, y_pred):
    squared_relative_error = np.square((y_true - y_pred) /
                                       np.clip(np.abs(y_true),
                                               np.finfo(np.float32).eps,
                                               None))
    mean_squared_relative_error = np.mean(squared_relative_error)
    return np.sqrt(mean_squared_relative_error)


def direction_accuracy(y_true, y_pred):
    # sign returns either -1 (if <0), 0 (if ==0), or 1 (if >0)
    true_signs = np.sign(y_true[1:] - y_true[:-1])
    pred_signs = np.sign(y_pred[1:] - y_true[:-1])

    equal_signs = np.equal(true_signs, pred_signs)
    return np.mean(equal_signs)
