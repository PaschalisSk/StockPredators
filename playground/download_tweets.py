from searchtweets import collect_results, gen_rule_payload, load_credentials

premium_search_args = load_credentials()

rule = gen_rule_payload("microsoft", results_per_call=10)
tweets = collect_results(rule,
                         max_results=10,
                         result_stream_args=premium_search_args)

[print(tweet['text'], end='\n\n') for tweet in tweets[0:10]]
