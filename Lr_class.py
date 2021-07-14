import pandas as pd
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle


class Linear:

    def get_data(self):
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
        test = prices[index_70_per:][['open','high','low','volume','close']]
        train.to_csv('train_data.csv',index=False)
        test.to_csv('test_data.csv',index=False)


    def train_data(self):
        '''
        Description: Load train.csv file apply linear regression and create lr_model.pkl file
        '''
        train_df = pd.read_csv('train_data.csv')
        linreg = LinearRegression().fit(train_df[['open','high','low','volume']],train_df['close'])
        with open('lr_model.pkl', 'wb') as file: 
            pickle.dump(linreg, file)


    def predict_on_test(self):
        '''
        Description: Reads test_data and run model pkl file to return predictions
        '''
        test_df = pd.read_csv('test_data.csv')
        
        # Load the pickled model
        with open('lr_model.pkl', 'rb') as file: 
            lr_from_pickle = pickle.load(file)
        
        # Use the loaded pickled model to make predictions
        prediction = lr_from_pickle.predict(test_df[['open','high','low','volume']])
        self.accuracy(prediction,test_df['close'].values)
        # return prediction
    
    def accuracy(self,prediction,test_data):
        mse = mean_squared_error(test_data,prediction,squared=False)
        MSE = np.square(np.subtract(test_data,prediction)).mean() 
        print(mse)
        print(MSE)

        


if __name__=='__main__':
    obj = Linear()
    obj.get_data()
    obj.train_data()
    obj.predict_on_test()
