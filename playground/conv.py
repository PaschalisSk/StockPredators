import pandas as pd

stock = pd.read_csv('../data/Stocks/BIST 100 Historical data.csv', thousands=',')
stock['date'] = pd.to_datetime(stock['date'], format='%b %d, %Y')
stock['close'] = stock['price']
stock = stock.drop(['change', 'price'], axis=1)
stock = stock[['date', 'open', 'high', 'low', 'close', 'volume']]
stock.set_index('date', inplace=True)
stock.sort_index(inplace=True)
stock.to_csv('../data/Stocks/XU100.IS.1997-01-02.2007-12-31.csv', index_label='date', date_format='%Y-%m-%d',float_format='%.2f')
print('test')
