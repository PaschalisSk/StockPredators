import numpy as np
import pandas as pd
import data
import os
import re
import tensorflow as tf
import keras as k
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

train_df = pd.read_json('../data/semeval2017 task5 part2/Headline_Trainingdata_1000.json')
train_X = train_df['title']
train_X = train_X.apply(lambda x: str(x).lower())
train_X = train_X.apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
train_y = train_df['sentiment']

max_fatures = 1500
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_X.values)
train_X = tokenizer.texts_to_sequences(train_X.values)
train_X = pad_sequences(train_X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = train_X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='tanh'))
model.compile(loss = 'mean_squared_error', optimizer='rmsprop',metrics = ['mse'])

X_train, X_test, Y_train, Y_test = train_test_split(train_X,train_y, test_size = 0.1, shuffle = False)

batch_size = 10
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size)

val_156_file = pd.read_json('../data/semeval2017 task5 part2/Headline_Trialdata_156.json')
X_val = val_156_file['title'].map(str)

val_X_pred = X_val.apply(lambda x: x.lower())
val_X_pred = val_X_pred.apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_fatures = 1500
val_X_pred = tokenizer.texts_to_sequences(X_val.values)
val_X_pred = pad_sequences(val_X_pred)

val_X_pred_fill = np.zeros([val_X_pred.shape[0],22])
val_X_pred_fill[:,(22-18):] = val_X_pred

prediction_val = model.predict(val_X_pred_fill)
df_val_prediction = pd.DataFrame(prediction_val)
df_val_prediction.columns = ['sentiment']
val_156_file_title = val_156_file['title']
val_156_file_others = val_156_file.drop(['title'], axis = 1)
val_156_file_others.head(n=1)
LSTM_tagged_val_156 = pd.concat([val_156_file_others, df_val_prediction,val_156_file_title], axis =1)
LSTM_tagged_val_156.to_json('LSTM_tagged_val_156.json',orient='records')

# test_df = pd.read_json('../data/semeval2017 task5 part2/Headline_Trialdata_156.json')
#
# test_df = test_df.apply(lambda x: str(x).lower())
# test_df = test_df.apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
#
# max_fatures = 1500
# test_df = tokenizer.texts_to_sequences(test_df.values)
# test_df = pad_sequences(test_df)
#
# val_X_pred_fill = np.zeros([test_df.shape[0],22])
# val_X_pred_fill[:,(22-18):] = test_df
#
# prediction_val = model.predict(val_X_pred_fill)
# df_val_prediction = pd.DataFrame(prediction_val)
# df_val_prediction.columns = ['sentiment']
#
# val_156_file = pd.read_json('Headline_Trialdata_156.json')
# val_156_file_title = val_156_file['title']
# val_156_file_others = val_156_file.drop(['title'], axis = 1)
#
# LSTM_tagged_val_156 = pd.concat([val_156_file_others, df_val_prediction,val_156_file_title], axis =1)
# LSTM_tagged_val_156.to_json('LSTM_tagged_val_156.json',orient='records')
