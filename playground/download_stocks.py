from data import Stocks
from datetime import datetime

stocks = Stocks()
stocks.download('MSFT', datetime(2013, 12, 31), datetime(2018, 12, 31))
stocks.save_df('../data/stocks/MSFT.2013-12-31.2018-12-31.csv')
