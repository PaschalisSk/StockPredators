from searchtweets import collect_results, gen_rule_payload, load_credentials
import pandas as pd

# tweets = pd.read_csv('../haydn/MSFT_tweets_17_11_2015_4months.csv',
#                      usecols=['DateTime', 'Tweet Text'],
#                      skipinitialspace=True, encoding='ISO-8859-1', nrows=1000)
# import os
# test = os.environ["SEARCHTWEETS_CONSUMER_KEY"]
# premium_search_args = load_credentials()
#
# rule = gen_rule_payload("microsoft", results_per_call=10)
# tweets = collect_results(rule,
#                          max_results=10,
#                          result_stream_args=premium_search_args)
import json
tweets = None
with open('10tweets.txt') as json_file:
    tweets = json.load(json_file)
[print(tweet['text'], end='\n\n') for tweet in tweets[0:10]]
print('test')
