from searchtweets import collect_results, gen_rule_payload, load_credentials
import json

premium_search_args = load_credentials()

rule = gen_rule_payload('#microsoft lang:en', results_per_call=100)
tweets = collect_results(rule,
                         max_results=100,
                         result_stream_args=premium_search_args)

with open('tweets1.txt', 'w') as outfile:
    json.dump(tweets, outfile)

# with open('10tweets.txt') as json_file:
#     tweets = json.load(json_file)
#[print(tweet['text'], end='\n\n') for tweet in tweets]
print('test')
