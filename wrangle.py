import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import env


########## Acquire ##########

def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    This function takes in user credentials from an env.py file and a database name and creates a connection to the Codeup database through a connection string 
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


zillow_sql_query =  '''
                    select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                    taxvaluedollarcnt, yearbuilt, taxamount, fips
                    from properties_2017
                    join propertylandusetype using(propertylandusetypeid)
                    where propertylandusedesc = 'Single Family Residential';
                    '''

def query_zillow_data():
    '''
    This function uses the get_connection function to connect to the zillow database and returns the zillow_sql_query read into a pandas dataframe
    '''
    return pd.read_sql(zillow_sql_query,get_connection('zillow'))


def get_zillow_data():
    '''
    This function checks for a local zillow.csv file and reads it into a pandas dataframe, if it exists. If not, it uses the get_connection & query_zillow_data functions to query the data and write it locally to a csv file
    '''
    # If csv file exists locally, read in data from csv file.
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Query and read data from zillow database
        df = query_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
        
    return df


########## Clean & Split ##########


# Remove outliers
def remove_outliers(df, k, col_list):
    ''' 
    This function remove outliers from a list of columns in a dataframe 
    and returns that dataframe
    '''
    
    # loop through each column
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def prepare_zillow_data(df):
    '''
    This function takes in the zillow data, cleans it, splits it, and returns train, validate & test dataframes
    '''
    
    # Rename some columns for simplicity
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet':'area',
                            'taxvaluedollarcnt':'taxvalue'})
    # Apply a function to remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms','area','taxvalue','taxamount'])
    
    # Drop rows with null values since it is only a small portion of the dataframe 
    df = df.dropna()
    
    # Create list of datatypes I want to change
    int_col_list = ['bedrooms','area','taxvalue']
    obj_col_list = ['yearbuilt','fips']
    
    # Change data types where it makes sense
    for col in df:
        if col in int_col_list:
            df[col] = df[col].astype(int)
        if col in obj_col_list:
            df[col] = df[col].astype(int).astype(object)
    
    # Split the data into train, validate, and test
    train_validate, test = train_test_split(df, test_size=.2, random_state=369)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=369)

    return train, validate, test


########## Wrangle ##########

def wrangle_zillow():
    '''This function acquires, cleans, and splits data from the zillow database for exploration'''
    train, validate, test = prepare_zillow_data(get_zillow_data())
    
    return train, validate, test