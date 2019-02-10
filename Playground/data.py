import pandas as pd
import numpy as np
import datetime
from pytrends.request import TrendReq
import os
import pandas_datareader.data as web


class Data:
    def __init__(self, source=None):
        """
        Args:
            source (str, optional): The local csv source file
        """
        if source is not None:
            self.df = pd.read_csv(source, index_col='date')
        else:
            self.df = None

        #self.np_array = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    # def create_np_array(self, column, **kwargs):
    #     """ Create a numpy array from one dataframe column.
    #     Args:
    #         column (str): The name of the column we want to turn into an array
    #         **kwargs (): Additional kwargs to pass in to_numpy method
    #     """
    #     self.np_array = self.raw_df[column].to_numpy(kwargs)

    def save_df(self, location, **kwargs):
        """ Save the dataframe.
        Args:
            location (str): The location to save the df
            **kwargs (): Additional kwargs to pass in to_csv
        """
        # '../Data/Trends/' + trend + '.' +
        # START_DATE.strftime('%Y-%m-%d') + '.' +
        # END_DATE.strftime('%Y-%m-%d') + '.csv',
        #                 index_label='date', date_format='%Y-%m-%d',
        #                 float_format='%.2f'
        self.df.to_csv(location, kwargs)

    # def reshape(self, window_size):
    #     """Reshape np array to CxSIZE
    #     Args:
    #         window_size (int): The size of each window.
    #     """
    #     rows = np.floor(self.np_array.shape[0]/window_size)
    #     # Resize array to a size multiple of window_size
    #     self.reshaped_np_array = self.np_array[:window_size*rows]
    #     # Reshape data into Cx(window_size) matrix
    #     self.reshaped_np_array = self.reshaped_np_array.reshape(-1, window_size)
    #
    # def shuffle(self):
    #     """Randomly shuffle the rows of our data"""
    #     self.data = np.random.permutation(self.data)
    #
    # def create_sets(self):
    #     """Creates 6 sets of shuffled data for training(70%),
    #     for validation(15%) and for testing(15%).
    #     """
    #     self.reshape()
    #     self.shuffle()
    #
    #     # Get the indices of the sections we want to split the array
    #     sections = np.array([self.data.shape[0] * 70 // 100,
    #                          self.data.shape[0] * 85 // 100])
    #
    #     # Split the rows in chunks of 70%, 15% and 15%
    #     training, validation, testing = np.vsplit(self.data, sections)
    #
    #     # Separate inputs from targets
    #     self.X_shuffle_train = training[:, :20]
    #     self.y_shuffle_train = training[:, -1:]
    #     self.X_shuffle_val = validation[:, :20]
    #     self.y_shuffle_val = validation[:, -1:]
    #     self.X_shuffle_test = testing[:, :20]
    #     self.y_shuffle_test = testing[:, -1:]


class Trends(Data):

    def download(self, kw, start_date, end_date):
        """
        Args:
            kw (str): the keyword for which to get Google Trends data
            start_date (:obj:'datetime'): The start date for the data
            end_date (:obj:'datetime'): The end date for the data
        """
        # Split dates in 90 days intervals in order to fetch daily data
        # Windows of more than 90 days return weekly data
        # The windows must overlap so that we can equalize the returned values
        # since Google Trends returns relevant values in each window,
        # i.e. a 100 denotes the day with the highest search volume, a 50
        # means that on that day we had 50% of the volume we had on the highest
        # day

        # List of tuples that contain start and end dates
        # the start date of tuple n is the end dates of tuple n-1
        # This list is used later to retrieve the data
        dates_windows = []
        cur_date = start_date
        while cur_date < end_date:
            if cur_date + datetime.timedelta(days=90) <= end_date:
                window = (cur_date, cur_date + datetime.timedelta(days=90))
            else:
                window = (cur_date, end_date)
            dates_windows.append(window)
            cur_date += datetime.timedelta(days=90)

        pytrends = TrendReq(hl='en-US', tz=360)
        # Trends_df_list is a list of dataframes that contains
        # dataframes from all the previously created windows
        trends_df_list = []
        for i in range(len(dates_windows)):
            timeframe = dates_windows[i][0].strftime('%Y-%m-%d') \
                        + ' ' + dates_windows[i][1].strftime('%Y-%m-%d')
            pytrends.build_payload([kw], timeframe=timeframe)
            trends_df_list.append(pytrends.interest_over_time())

        # trends_df is the joined df from all elements in the trends_df_list
        trends_df = trends_df_list[0]
        for i in range(1, len(trends_df_list)):
            # Last value of the 'complete' dataframe
            last_value = trends_df.iloc[-1][kw]
            # First value of the dataframe which we are about to add
            first_value = trends_df_list[i].iloc[0][kw]
            # If the values in the next dataframe are smaller then lower the
            # values of the 'complete' dataframe, else do the opposite
            if last_value > first_value:
                adj_factor = first_value / last_value
                trends_df[kw] = adj_factor * trends_df[kw]
                trends_df = trends_df.append(trends_df_list[i].iloc[1:])
            else:
                adj_factor = last_value / first_value
                trends_df_list[i][kw] = adj_factor * trends_df_list[i][kw]
                trends_df = trends_df.append(trends_df_list[i].iloc[1:])

        self.df = trends_df.drop(columns=['isPartial'])


class Stocks(Data):

    def download(self, kw, start_date, end_date):
        """
        Args:
            kw (str): the keyword for which to get Google Trends data
            start_date (:obj:'datetime'): The start date for the data
            end_date (:obj:'datetime'): The end date for the data
        """
        self.df = web.DataReader(kw, 'av-daily',
                                 start=start_date, end=end_date,
                                 access_key=os.getenv('ALPHAVANTAGE_API_KEY'))

    def moving_average(self, days):
        """ Add column to df with moving average over n days
        Args:
            days (int): Days over which to calculate MA
        Returns:
            The added column
        """
        self.df['MA'] = self.df['close'].rolling(days).mean()
        return self.df['MA']

    def weighted_moving_average(self, days):
        """ Add column to df with weighted moving average
            over n days
        Args:
            days (int): Days over which to calculate weighted MA
        Returns:
            The added column
        """
        # The kwargs for np.average
        kwargs = {'weights': list(range(1, days+1))}
        self.df['WMA'] = self.df['close'].rolling(days).apply(np.average,
                                                              raw=True,
                                                              kwargs=kwargs)
        return self.df['WMA']

    def momentum(self, days):
        """ Add column to df with momentum for closing price over n days
        Args:
            days (int): Days over which to calculate weighted MA
        Returns:
            The added column
        """
        self.df['MOM'] = self.df['close'].rolling(days).apply(
            lambda x: x[days-1] - x[0], raw=True)
        return self.df['MOM']

    def stochastic_oscillator_K(self, days):
        """ Add column to df %K over n days
        Args:
            days (int): Days over which to calculate %K
        Returns:
            The added column
        """
        test = self.df.rolling(days)
        self.df['SO_K'] = self.df.rolling(days).apply(
            lambda x: x[-1]['close']))
        return self.df['SO_K']


