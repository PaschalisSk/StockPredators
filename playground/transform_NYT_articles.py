import pandas as pd
# Transform NYT articles to be used in SemEval2017Task5
articles_df = pd.read_csv('../data/nytarticles/microsoft.2013-12-31.2018-12-31.csv')
articles_df['id'] = articles_df['index'] + 2000
articles_df['company'] = 'Microsoft'
articles_df['title'] = articles_df['headline']
articles_df.drop(['index', 'snippet', 'headline'], axis=1, inplace=True)
articles_df.to_json('../data/nytarticles/microsoft.2013-12-31.2018-12-31.json',
                    orient='records')
print('test')
