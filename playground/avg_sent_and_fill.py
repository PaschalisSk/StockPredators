import pandas as pd
import math

sents = pd.read_csv('../data/nytarticles/microsoft.2013-12-31.2018-12-31.sent.csv',index_col='index')
avg_sents = sents.groupby('date').mean()
window_date_df = pd.date_range(start='31/12/2013', end='31/12/2018').to_frame()
window_date_df.index = window_date_df.index.strftime('%Y-%m-%d')
window_date_df.drop(columns=[0], inplace=True)
combined_df = pd.merge(window_date_df, avg_sents, how='left', left_index=True,right_index=True)

if math.isnan(combined_df.iloc[0]['sentiment']):
    combined_df.iloc[0]['sentiment'] = 0

if math.isnan(combined_df.iloc[-1]['sentiment']):
    combined_df.iloc[-1]['sentiment'] = 0

i=0
for index, row in combined_df.iterrows():
    non_empty_df = combined_df[combined_df['sentiment'].isnull() == False]
    non_empty_df.reset_index(inplace=True)
    if math.isnan(row['sentiment']):
        combined_df.loc[index]['sentiment'] = (non_empty_df.iloc[i]['sentiment'] + non_empty_df.iloc[i-1]['sentiment'])/2
    i += 1

stocks_df = pd.read_csv('../data/stocks/MSFT.2013-12-31.2018-12-31.csv', index_col='date')
final_df = pd.merge(stocks_df, combined_df, how='left', left_index=True,right_index=True)
final_df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)
final_df.to_csv('../data/nytarticles/microsoft.2013-12-31.2018-12-31.imputed.sent.csv',
                   float_format='%.6f',date_format='%Y-%m-%d', index_label='date')
print('test')
