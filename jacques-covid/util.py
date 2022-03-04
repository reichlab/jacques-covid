import covidcast
import pandas as pd
import numpy as np

# return location, date, value, rate, population
# end_date is T in overleaf
def load_data(measure="hospitalizations", as_of = None, end_date = "2021-07-01"):
    if measure == "hospitalizations":
        data_source = "hhs"
        signal = "confirmed_admissions_covid_1d"
    
    df = covidcast.signal(data_source=data_source, 
                          signal=signal, 
                          geo_type="state",
                          as_of=as_of)

    # convert 2-letter state abbreviations to upper case
    df["geo_value"] = df["geo_value"].str.upper()
    
    # rename columns
    df.rename(columns={"time_value":"date", "geo_value":"location", "value":"inc_hosp"}, inplace=True)
    
    # load in population data
    state_info = pd.read_csv("data/locations.csv")
    
    # merge in population column and also drop territories
    df = df.merge(state_info[['abbreviation','population']], left_on="location", right_on="abbreviation", how = "right")
    
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
    df = df[df['date'] >= "2020-10-01"] 
    df = df[df['date'] <= end_date]

    # create rate column
    df['rate'] = df['inc_hosp']/df['population'] * 100000

    # sort data
    df = df.sort_values(['location', 'date'], ascending=[True, True])

    return df


df = load_data()
df.to_csv("hosp_data_till_July.csv", index=False)