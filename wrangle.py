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


def clean_zillow_data(df):
    '''
    This function takes in the zillow data, cleans it, and returns a dataframe
    '''
    
    # Rename some columns for simplicity
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 
                            'calculatedfinishedsquarefeet':'area',
                            'taxvaluedollarcnt':'taxvalue'})
    # Apply a function to remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms','area','taxvalue','taxamount'])
    
    # Remove more of the outliers for area
    df = df[(df.area > 500) & (df.area < 2500)]
    # Remove more of the outliers for taxvalue
    df = df[(df.taxvalue > 500) & (df.taxvalue < 800000)]
    
    # Drop rows with null values since it is only a small portion of the dataframe 
    df = df.dropna()

    # create age column based on yearbuilt
    df['age'] = 2021 - df.yearbuilt
    
    # Create list of datatypes I want to change
    int_col_list = ['bedrooms','area','taxvalue','age']
    obj_col_list = ['yearbuilt','fips']
    
    # Change data types where it makes sense
    for col in df:
        if col in int_col_list:
            df[col] = df[col].astype(int)
        if col in obj_col_list:
            df[col] = df[col].astype(int).astype(object)
    
    # drop taxamount since we will be predicting tax value and tax amount is considered data leakage
    df = df.drop(columns='taxamount')

    # Encode FIPS column and concatenate onto original dataframe
    dummy_df = pd.get_dummies(df['fips'], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


########## Split ##########

# Split the data into train, validate, and test
def split_data(df, random_state=369, stratify=None):
    '''
    This function takes in a dataframe and splits the data into train, validate and test samples. 
    Test, validate, and train are 20%, 24%, & 56% of the original dataset, respectively. 
    The function returns train, validate and test dataframes.
    '''
   
    if stratify == None:
        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state)

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, random_state=random_state)
    else:

        # split dataframe 80/20
        train_validate, test = train_test_split(df, test_size=.2, random_state=random_state, stratify=df[stratify])

        # split larger dataframe from previous split 70/30
        train, validate = train_test_split(train_validate, test_size=.3, 
                            random_state=random_state,stratify=train_validate[stratify])

    # results in 3 dataframes
    return train, validate, test


########## Wrangle ##########

def wrangle_zillow():
    '''This function acquires, cleans, and splits data from the zillow database for exploration'''
    train, validate, test = split_data(clean_zillow_data(get_zillow_data()))
    
    return train, validate, test
