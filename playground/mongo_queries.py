import pymongo

myclient = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = myclient['sacred']
runs = mydb['runs']
metrics = mydb['metrics']

experiment_name = 'sent_trends_model_conf'

pipeline = [
    {
        '$match': {
            'name': 'val_loss'
        }
    },
    {
        '$addFields': {
            'min_value': {'$min': '$values'},
        }
    },
    {
        '$addFields': {
            'min_index': {'$indexOfArray': ['$values', '$min_value']}
        }
    },
    {
        '$addFields': {
            'min_epoch': {'$add': ['$min_index', 1]},
            'min_timestamp': {'$arrayElemAt': ['$timestamps', '$min_index']}
        }
    },
    {
        '$lookup': {
            'from': 'runs',
            'localField': 'run_id',
            'foreignField': '_id',
            'as': 'run'
        }
    },
    {
        '$unwind': '$run'
    },
    {
        '$match': {
            'run.experiment.name': experiment_name,
            'run.status': 'COMPLETED'
        }
    },
    {
        '$addFields': {
            'min_time': {
                '$divide': [
                    {'$subtract': ['$min_timestamp', '$run.start_time']},
                    1000
                ]
            }
        }
    },
    {
      '$project': {
          'run.config.seed': 0
      }
    },
    {
        '$group': {
            '_id': {
                'metric': '$name',
                'config': '$run.config'
            },
            "count": {"$sum": 1},
            'min_value_avg': {'$avg': '$min_value'},
            'min_value_std': {'$stdDevPop': '$min_value'},
            'min_time_avg': {'$avg': '$min_time'},
            'min_time_std': {'$stdDevPop': '$min_time'},
            'min_epoch_avg': {'$avg': '$min_epoch'},
            'min_epoch_std': {'$stdDevPop': '$min_epoch'}
        }
    },
    {
        '$addFields': {
            'min_value_sem': {
                '$divide': [
                    '$min_value_std',
                    {'$sqrt': '$count'}
                ]
            },
            'min_time_sem': {
                '$divide': [
                    '$min_time_std',
                    {'$sqrt': '$count'}
                ]
            },
            'min_epoch_sem': {
                '$divide': [
                    '$min_epoch_std',
                    {'$sqrt': '$count'}
                ]
            }
        }
    },
    {
        '$sort': {
            'min_value_avg': 1
        }
    }
]

mydoc = metrics.aggregate(pipeline)
#mydoc = runs.find({'experiment.name': 'lstm_sent_trends_batch_size'}).sort([('_id', pymongo.ASCENDING)])
# myquery = {'_id': {'$gt': 747}}
# newvalues = { "$set": { "experiment.name": "lstm_sent_trends_model_conf" } }
# x = runs.update_many(myquery, newvalues)
#
# print(x.modified_count, "documents updated.")
#$$deleteO = metri4cs.delete_many({'run_id': {'$gt': 351}})
#runs.distinct('experiment.name')
#mydoc = runs.find({'_id': 702})
#tags = db.mycoll.find({"category": "movie"}).distinct("tags")
#'lstm_sent_trends_batch_size'
for result in mydoc:
    #print(result)
    print('Batch size: ' + str(result['_id']['config']['batch_size']))
    print('Learning rate: ' + str(result['_id']['config']['learning_rate']))
    print('Neurons: ' + str(result['_id']['config']['num_neurons']))
    print('Layers: ' + str(result['_id']['config']['num_hidden_layers']))
    mse_avg = str('%.4f' % (10000*result['min_value_avg']))
    mse_sem = str('%.4f' % (10000*result['min_value_sem']))
    # mse_avg = str('%.2f' % result['min_value_avg'])
    # mse_sem = str('%.2f' % result['min_value_sem'])
    print('MSE:' + mse_avg + '\pm' + mse_sem)
    time_avg = str('%.2f' % result['min_time_avg'])
    time_sem = str('%.2f' % result['min_time_sem'])
    print('Time: ' + time_avg + '\pm' + time_sem)
    print('t')


