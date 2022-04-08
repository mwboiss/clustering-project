import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

import os
import env

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def acquire_zillow(use_cache=True):
    
    filename = ('zillow.csv')
    if os.path.exists(filename) and use_cache:
        print('Using CSV')
        return pd.read_csv(filename)
    
    print('Acquiring from Database')
    url = env.get_db_url('zillow')

    zillow = pd.read_sql('''
    SELECT *
    FROM properties_2017
    JOIN (SELECT parcelid as parid, MAX(transactiondate) as date_sold FROM predictions_2017 GROUP BY parcelid) as last_date
    ON last_date.parid = parcelid
    LEFT JOIN (SELECT parcelid as pid, transactiondate as maxdate, logerror FROM predictions_2017) as log
    ON last_date.parid = log.pid AND last_date.date_sold = log.maxdate
    LEFT JOIN propertylandusetype
    USING(propertylandusetypeid)
    LEFT JOIN storytype
    USING(storytypeid)
    LEFT JOIN typeconstructiontype
    USING(typeconstructiontypeid)
    LEFT JOIN airconditioningtype
    USING(airconditioningtypeid)
    LEFT JOIN architecturalstyletype
    USING(architecturalstyletypeid)
    LEFT JOIN buildingclasstype
    USING(buildingclasstypeid)
    LEFT JOIN heatingorsystemtype
    USING(heatingorsystemtypeid)
    ''',url)

    print('Saving to CSV')
    zillow.to_csv('zillow.csv',index=False)
    return zillow

def handle_missing_values(df,required_col,required_row):
    required_row = round(df.shape[1] * required_row)
    required_col = round(df.shape[0] * required_col)
    df.dropna(axis=0, thresh=required_row, inplace=True)
    df.dropna(axis=1, thresh=required_col, inplace=True)
    return df

def cleanup_zillow(df):
    '''
    This function drops unused columns, removes null values, and renames columns for use
    '''
    # Drop unusable or duplicate columns
    df.drop(columns=['id','pid','parid','maxdate','heatingorsystemtypeid',\
                     'assessmentyear','roomcnt','fips','regionidcounty',\
                     'censustractandblock','rawcensustractandblock',\
                     'calculatedbathnbr','fullbathcnt','finishedsquarefeet12',\
                     'structuretaxvaluedollarcnt','landtaxvaluedollarcnt',\
                     'propertycountylandusecode'],inplace=True)
    # Create a month sold and age column
    df['month_sold'] = pd.DatetimeIndex(df.date_sold).month
    df['age'] = 2017 - df.yearbuilt 
    # Rename columns for ease of use
    df = df.rename(columns={'bedroomcnt' : 'bedrooms',\
                                'bathroomcnt' : 'bathrooms',\
                                'buildingqualitytypeid' : 'quality_id',\
                                'calculatedfinishedsquarefeet' : 'area',\
                                'lotsizesquarefeet' : 'lot_size',\
                                'propertyzoningdesc' : 'Zoning',\
                                'regionidcity' : 'city',\
                                'regionidzip' : 'zip',\
                                'taxvaluedollarcnt' : 'taxable_value',\
                                'propertylandusedesc' : 'land_use',\
                                'heatingorsystemdesc' : 'heating_desc'})
    # rename land use values
    df.land_use = df.land_use.map({'Single Family Residential' : 'single_family',\
                                       'Planned Unit Development' : 'planned_unit',\
                                    'Condominium' : 'condo'})
    # Drop null lat/long values
    df.dropna(subset=['latitude','longitude'],inplace=True)
    # Drop Rows that are not single family single unit homes
    df = df[df['propertylandusetypeid'].isin([261,266,269])]
    df = df[~df['unitcnt'].isin([2.0,3.0,4.0,6.0])]
    # Drop columns that are not of use anymore
    df = df.drop(columns=['propertylandusetypeid','unitcnt'])
    # handle missing values
    df = handle_missing_values(df,0.4,0.4)
    df.dropna(inplace=True)
    return df

def encode_zillow_cat(df,col_list):
    dummy_name = pd.get_dummies(df[col_list])
    df = pd.concat([df,dummy_name],axis=1) 
    return df

def remove_outliers(df, column_list):
    ''' remove outliers from dataframe 
        then return the new dataframe
    '''
    # Iterate through column_list
    for col in column_list:
        
        # find percentiles
        q_25 = np.percentile(df[col], 25)
        q_75 = np.percentile(df[col], 75)
        
        # Calculate IQR
        iqr = q_75 - q_25
        
        # assign upper bound
        upper_bound = q_75 + 1.7 * iqr   
        
        # assign lower bound 
        lower_bound = q_25 - 1.7 * iqr   

        # assign df without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    # return dataframe without outliers    
    return df

def split_df(df):
    '''
    This funciton splits the dataset for modeling into:
    train - for exploring the data, and fitting the models
    validate - for ensuring the model is not overfit
    test - for testing the model on unseen data
    '''
    # This seperates out the test data from the train and validate data. Test makes up 20 % of the data.
    train_validate, test = train_test_split(df, random_state=1729, test_size=0.2)
    
    # This seperates out the train and validates sets. Train makes up 56 % of the data and Validate makes up 24 %.
    train, validate = train_test_split(train_validate, random_state=1729, test_size=0.3)
    
    # The funciton returns the split sets
    return train, validate, test
                                   
def scale_zillow(df,col_list):

    df_scaled = df[col_list]

    minmax = MinMaxScaler()
    minmax.fit(df[col_list])

    df_scaled[col_list] = minmax.transform(df[col_list])
    
    return df_scaled

def scale_data(train, validate, test, return_scaler=False):
    '''
    This function scales the split data and returns a scaled version of the dataset.
    
    If return_scaler is true, the scaler will be returned as well.
    '''
    
    col = train.columns[train.dtypes == 'float']
    col = col.append(train.columns[train.dtypes == 'int'])

    train_scaled = train[col]
    validate_scaled = validate[col]
    test_scaled = test[col]

    scaler = MinMaxScaler()
    scaler.fit(train[col])
    
    train_scaled[col] = scaler.transform(train[col])
    validate_scaled[col] = scaler.transform(validate[col])
    test_scaled[col] = scaler.transform(test[col])
    
    if return_scaler:
        return train_scaled, validate_scaled, test_scaled, scaler
    else:
        return train_scaled, validate_scaled, test_scaled

def model_split(df):
    
    # Assign x for testing the model, y as target for modeling
    X = df.drop(columns=['logerror'])
    y = df[['logerror']]
    
    return X, y


def wrangle_zillow(df):
        
        df = cleanup_zillow(df)
        col_e = ['land_use']
        df = encode_zillow_cat(df,col_e)
        col = ['bathrooms','bedrooms','area','lot_size','taxable_value','taxamount','logerror','age']
        df = remove_outliers(df, col)
        train, validate, test = split_df(df)
        return train, validate, test
        
        
def initial_look(df):
    print(f'Shape:\n\n{df.shape}\n\n')
    print(f'Describe:\n\n{df.describe(include="all")}\n\n')
    print(f'Info:\n\n{df.info()}\n\n')
    print(f'Histograms:\n\n{df.hist(figsize=(40,20), bins =20), plt.show()}')
    
def missing_rows(df):
    return pd.concat([
           df.isna().sum().rename('count'),
           df.isna().mean().rename('percent')
           ], axis=1)

def missing_columns(df):
    col_missing = pd.concat([
    df.isna().sum(axis=1).rename('num_cols_missing'),
    df.isna().mean(axis=1).rename('pct_cols_missing'),
    ], axis=1).value_counts().sort_index()
    col_missing = pd.DataFrame(col_missing)
    col_missing.rename(columns={0:'num_rows'},inplace=True)
    return col_missing.reset_index()
