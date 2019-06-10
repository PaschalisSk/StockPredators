import pandas as pd
from scipy import spatial
import numpy as np
from textblob import TextBlob

texts = pd.read_json('../data/semeval2017 task5 part2/Headline_Trialdata_156.json')['title']
gold_standard = pd.read_json('../data/semeval2017 task5 part2/Headline_Trialdata_156_gold.json')['sentiment']
SemEval_results = pd.read_json('../SemEval2017Task5/trial_results_156.json')['sentiment score']
LSTM_results = pd.read_json('../haydn/123LSTM_tagged_val_156.json')['sentiment']
SVM_results = pd.read_json('../haydn/SVM_tagged_val_156.json')['sentiment']
SWN_results = pd.read_json('../haydn/SWN_tagged_val_156.json')['sentiment']

textblob_results = np.empty(len(texts))
for i, text in enumerate(texts):
    testimonial = TextBlob(text)
    textblob_results[i] = testimonial.sentiment.polarity

SemEval = 1 - spatial.distance.cosine(gold_standard, SemEval_results)
TextBlob = 1 - spatial.distance.cosine(gold_standard, textblob_results)
SVM = 1 - spatial.distance.cosine(gold_standard, SVM_results)
SWN = 1 - spatial.distance.cosine(gold_standard, SWN_results)
LSTM = 1 - spatial.distance.cosine(gold_standard, LSTM_results)
print('SentEval ' + str(SemEval))
print('TextBlob ' + str(TextBlob))
print('SVM ' + str(SVM))
print('SWN ' + str(SWN))
print('LSTM ' + str(LSTM))
#result2 = np.dot(gold_standard, trial_results)/(np.linalg.norm(gold_standard)*np.linalg.norm(trial_results))
print('test')
