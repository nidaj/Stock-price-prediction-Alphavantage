import pickle
import pandas as pd
 
def predict_on_test():
    '''
    Description: Reads test_data and run model pkl file to return predictions
    '''
    test_df = pd.read_csv('test_data.csv')
    
    # Load the pickled model
    with open('lr_model.pkl', 'rb') as file: 
        lr_from_pickle = pickle.load(file)
    
    # Use the loaded pickled model to make predictions
    prediction = lr_from_pickle.predict(test_df)
    
    return prediction

print(predict_on_test())
