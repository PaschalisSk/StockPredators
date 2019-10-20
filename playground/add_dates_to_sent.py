import pandas as pd
# Add dates to the result of Microsoft from SentEval
articles_dates_df = pd.read_json('../data/nytarticles/microsoft.2013-12-31.2018-12-31.json')
articles_sent_df = pd.read_json('../data/nytarticles/microsoft.no.dates.sent.json')

combined_df = pd.merge(articles_dates_df[['id', 'date']], articles_sent_df, left_on='id', right_on='id')
combined_df['headline'] = combined_df['text']
sentiment = combined_df['sentiment score']
combined_df.drop(['id', 'company', 'text', 'sentiment score'], axis=1, inplace=True)
combined_df['sentiment'] = sentiment
combined_df.to_csv('../data/nytarticles/microsoft.2013-12-31.2018-12-31.sent.csv',
                   date_format='%Y-%m-%d', index_label='index')
print('test')
