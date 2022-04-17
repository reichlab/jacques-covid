import covidcast
import pkg_resources
import pandas as pd
import numpy as np
from datetime import date


def load_locations():
    '''
    Return a dataframe with information about locations

    Contains the following fields:
        abbreviation   human readable location abbreviation
        location       location FIPS code
        location_name  location name
        population     population size
    '''
    stream = pkg_resources.resource_stream(__name__, 'data/locations.csv')
    return pd.read_csv(stream, encoding='latin-1')


def load_data(measure='hospitalizations', as_of = None, end_date = '2021-07-01'):
    '''
    Load data for selected measure from covidcast
    
    Parameters
    ----------
    measure: string 
        Default to 'hospitalizations'
    as_of: string of date in YYYY-MM-DD format. 
        Default to None.
    end_date: string of date in YYYY-MM-DD format. 
        Default to '2021-07-01'
    
    Returns
    -------
    df: data frame
        It has columns location, date, inc_hosp, population and rate. 
        It is sorted by location and date columns in ascending order.
    '''
    if measure == 'hospitalizations':
        data_source = 'hhs'
        signal = 'confirmed_admissions_covid_1d'
    
    df = covidcast.signal(data_source=data_source, 
                          signal=signal, 
                          geo_type='state',
                          as_of=date.fromisoformat(as_of))
    
    # convert 2-letter state abbreviations to upper case
    df['geo_value'] = df['geo_value'].str.upper()
    
    # rename columns
    df.rename(columns={'time_value':'date', 'geo_value':'location', 'value':'inc_hosp'}, inplace=True)
    
    # load in population data
    state_info = load_locations()
    
    # merge in population column and also drop territories
    df = df.merge(state_info[['abbreviation','population']], left_on='location', right_on='abbreviation', how = 'right')
    
    meta_cols = ['lag',
                 'missing_value',
                 'missing_stderr',
                 'missing_sample_size',
                 'stderr',
                 'sample_size',
                 'geo_type',
                 'data_source',
                 'signal', 
                 'issue',
                 'abbreviation']
    meta_cols = [c for c in meta_cols if c in df.columns]
    
    # drop columns
    df = df.drop(columns=meta_cols)
    
    # filter by date 
    df = df[df['date'] >= '2020-10-01'] 
    df = df[df['date'] <= end_date]
    
    # create rate column
    df['rate'] = df['inc_hosp']/df['population'] * 100000
    
    # sort data
    df = df.sort_values(['location', 'date'], ascending=[True, True])
    
    return df
