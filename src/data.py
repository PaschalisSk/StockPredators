import pandas as pd
import numpy as np
import datetime
from pytrends.request import TrendReq
import os
import pandas_datareader.data as web
from finta import TA
from sklearn.preprocessing import MinMaxScaler


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

        self.scaler = MinMaxScaler()

    def save_df(self, location):
        """ Save the dataframe.
        Args:
            location (str): The location to save the df
        """
        self.df.to_csv(location, index_label='date',
                       date_format='%Y-%m-%d', float_format='%.2f')

    def shuffle(self, random_state=None):
        """Randomly shuffle the rows of our data
        Args:
            random_state (int or numpy.random.RandomState): Seed for the random
                number generator (if int), or numpy RandomState object
        """
        if random_state is not None:
            self.df = self.df.sample(frac=1, random_state=random_state)
        else:
            self.df = self.df.sample(frac=1)

    def raw_values(self, dataset=None, norm=False):
        """ Returns dictionary with keys X and y and values numpy arrays
        Args:
            dataset (str): One of train, val, test
            norm (bool): Normalised or not
        Returns:
            raw_values (dict): Dictionary of raw values with keys X and y
        """
        raw_values = self.df.values
        if norm is True:
            raw_values = self.scaler.fit_transform(raw_values)

        # First 70% training
        train_limit = int(0.7 * len(raw_values))
        # 70%-85% validation
        val_limit = int(0.85 * len(raw_values))

        if dataset == 'train':
            raw_values = raw_values[:train_limit]
        elif dataset == 'val':
            raw_values = raw_values[train_limit:val_limit]
        elif dataset == 'test':
            raw_values = raw_values[val_limit:]

        y_index = self.df.columns.get_loc('close')

        return {'X': np.delete(raw_values, y_index, axis=1),
                'y': raw_values[:, y_index][:, np.newaxis]}

    def denorm_predictions(self, predictions):
        """ Scales predictions back to normal
        Args:
            predictions (numpy array): 1D numpy array of predictions
        Returns:
            numpy array of normalised predictions
        """
        # Create a numpy array similar to the test set but with the predictions
        # instead of the true y values and then inverse_transform
        norm_values = self.scaler.transform(self.df.values)
        val_limit = int(0.85 * len(norm_values))
        norm_test_values = norm_values[val_limit:]
        y_index = self.df.columns.get_loc('close')
        # Replace the true y values with the predicted ones
        # Had to make (184, 1) shape to (184)
        norm_test_values[:, y_index] = predictions[:, 0]

        denorm_test_values = self.scaler.inverse_transform(norm_test_values)
        return denorm_test_values[:, y_index][:, np.newaxis]


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

    def calc_patel_TI(self, days_back, days_forward):
        """ Calculate the technical indicators from Patel et. al. for n days
        Args:
            days_back (int): Days in the past over which to
                calculate relative strength index
            days_forward (int): Days in the future to predict
        """
        # TA.SMA has min_periods=days-1 for some reason
        # Changed it to min_periods=days
        self.df['MA'] = TA.SMA(self.df, days_back)
        # Equal to my implementation
        self.df['WMA'] = TA.WMA(self.df, days_back)
        # Equal
        self.df['MOM'] = TA.MOM(self.df, days_back)
        # Had highest_high - ohlc['close']
        # After changing they are equal
        self.df['STOCH'] = TA.STOCH(self.df, days_back)
        # They didn't have period when calling STOCH, instead it was used for 3
        # STOCHD is actually the mean over 3 days
        self.df['STOCHD'] = TA.STOCHD(self.df, days_back)
        # They used ewm, changed it to simple rolling
        # They also had ohlc['close'].diff()[1:] which resulted in returning
        # one less row
        # Changed min periods
        self.df['RSI'] = TA.RSI(self.df, days_back)
        # TODO: What do they mean in the paper
        # Changed min periods
        self.df['MACD'] = TA.MACD(self.df, signal=days_back)['MACD']
        self.df['WILLIAMS'] = TA.WILLIAMS(self.df, days_back)
        self.df['ADL'] = TA.ADL(self.df)
        self.df['CCI'] = TA.CCI(self.df, days_back)
        # Drop columns we no longer need
        self.df.drop(['open', 'high', 'low', 'volume'], axis=1, inplace=True)
        # Since to predict close price of day n we need the indicators
        # of day n-1 we move the above columns days_forward to the bottom
        self.df['MA'] = self.df['MA'].shift(days_forward)
        self.df['WMA'] = self.df['WMA'].shift(days_forward)
        self.df['MOM'] = self.df['MOM'].shift(days_forward)
        self.df['STOCH'] = self.df['STOCH'].shift(days_forward)
        self.df['STOCHD'] = self.df['STOCHD'].shift(days_forward)
        self.df['MACD'] = self.df['MACD'].shift(days_forward)
        self.df['WILLIAMS'] = self.df['WILLIAMS'].shift(days_forward)
        self.df['ADL'] = self.df['ADL'].shift(days_forward)
        self.df['CCI'] = self.df['CCI'].shift(days_forward)
        # Drop rows with nan
        self.df.dropna(inplace=True)


