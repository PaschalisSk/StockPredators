import baseline as bl
#import lstm

run_id = 1
for config_updates in bl.config_combinations:
    print('Run ' + str(run_id) + ' of ' + str(len(bl.config_combinations)))

    if config_updates['optimizer'] == 'adagrad':
        config_updates['learning_rate'] *= 10

    bl.ex.run(config_updates=config_updates)
    run_id += 1

