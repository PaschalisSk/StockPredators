import pandas as pd

trends = pd.read_csv('../data/trends/msft.2013-12-31.2018-12-31.csv',index_col='date')
stocks_df = pd.read_csv('../data/stocks/MSFT.2013-12-31.2018-12-31.csv', index_col='date')

final_df = pd.merge(stocks_df, trends, how='left', left_index=True,right_index=True)
final_df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)
final_df.to_csv('../data/trends/msft.2013-12-31.2018-12-31.fixed.dates.csv',
                   float_format='%.2f',date_format='%Y-%m-%d', index_label='date')
print('test')
