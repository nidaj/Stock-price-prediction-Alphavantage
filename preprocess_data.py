import pandas as pd
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import numpy as np

def get_data():
    '''
    Description: Downloads Dataset for google creates train and test csv file
    '''
    ts = TimeSeries('W9D2IE15NBMMJLTW',output_format='pandas')
    prices,meta = ts.get_daily('GOOG',outputsize='full')
    prices.sort_index(inplace=True)
    prices.reset_index(inplace=True)
    raw_data = prices.to_csv('raw_data.csv',index=False)
    prices.rename(columns={ '1. open':'open', '2. high':'high', '3. low':'low','4. close':'close', '5. volume':'volume'},inplace=True)
    index_70_per = int((prices.shape[0]*0.7))
    train = prices[:index_70_per][['open','high','low','volume','close']]
    test = prices[index_70_per:][['open','high','low','volume']]
    train.to_csv('train_data.csv',index=False)
    test.to_csv('test_data.csv',index=False)

get_data()