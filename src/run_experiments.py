import baseline as bl
import RNN

run_id = 1
for config_updates in RNN.config_combinations:
    print('Run ' + str(run_id) + ' of ' + str(len(RNN.config_combinations)))

    RNN.ex.run(config_updates=config_updates)
    run_id += 1

