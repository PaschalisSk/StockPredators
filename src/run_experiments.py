from baseline import ex
from sklearn.model_selection import ParameterGrid

config_options = {
    'stock_file': ['../data/stocks/MSFT.2013-12-31.2018-12-31.csv'],
    'days_back': [5, 10],
    'days_forward': [1, 5, 10],
    'max_epochs': [10000],
    'early_stopping_threshold': [20],
    'num_neurons': [50, 100, 150],
    'num_hidden_layers': [1],
    'seed': [0, 1, 2],
    'learning_rate': [0.001, 0.005, 0.01],
    'batch_size': [8, 32],
    'activation': ['sigmoid'],
    'optimizer': ['adagrad'],
    'kernel_init': ['glorot_uniform'],
    'regularization': [None],
    'loss': ['MSE']
}

config_combinations = list(ParameterGrid(config_options))
run_id = 1
for config_updates in config_combinations:
    print('Run ' + str(run_id) + ' of ' + str(len(config_combinations)))

    ex.run(config_updates=config_updates)
    run_id += 1

