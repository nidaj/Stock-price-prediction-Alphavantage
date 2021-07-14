import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

def train_data():
    '''
    Description: Load train.csv file apply linear regression and create lr_model.pkl file
    '''
    train_df = pd.read_csv('train_data.csv')
    linreg = LinearRegression().fit(train_df[['open','high','low','volume']],train_df['close'])
    with open('lr_model.pkl', 'wb') as file: 
        pickle.dump(linreg, file)

    

train_data()