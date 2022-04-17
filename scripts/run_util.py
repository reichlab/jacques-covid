import importlib
import json
import pandas as pd
from jacovid import util

def load_config(filename):
    '''Load a config file'''
    try:
        with open(filename) as json_file:
            config = json.load(json_file)
    except Exception as e:
        print("Could not parse config file. Please check syntax. Exception information with deatils follows\n")
        raise(e)
    
    return config


def instantiate_model(model_class, model_args):
    '''Given a string like foo.bar.baz where baz is a class in the
    module foo.bar, imports foo.bar and returns in instance of the
    class foo.bar.baz"
    '''
    module_name, model_name = model_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    model_instance = getattr(module, model_name)(**model_args)
    return model_instance


def make_preds_df(q_hat, tau, locations, forecast_date, horizon):
    preds_df = pd.DataFrame(q_hat)
    preds_df['forecast_date'] = forecast_date
    preds_df['location'] = locations
    preds_df = pd.melt(preds_df,
                        id_vars=['forecast_date', 'location'],
                        var_name='quantile_index')
    preds_df['quantile'] = tau[preds_df['quantile_index'].values.astype(int)]
    preds_df['target_end_date'] = pd.to_datetime(forecast_date) + \
        pd.to_timedelta(horizon, 'days')
    preds_df['target'] = str(horizon) + ' day ahead inc hosp'
    preds_df['type'] = 'quantile'

    # merge in population column
    state_info = util.load_locations()

    # merge in population column and FIPS codes
    preds_df = preds_df.merge(state_info[['abbreviation','location','population']],
                                    left_on="location",
                                    right_on="abbreviation",
                                    how = "left")

    # update predictions to original units rather than rates per 100k population
    preds_df['pop100k'] = preds_df.population / 100000
    preds_df['value'] = preds_df.value * preds_df.pop100k

    preds_df = preds_df[['location_y', 'forecast_date', 'target', 'target_end_date', 'type', 'quantile', 'value']]
    preds_df.columns = ['location', 'forecast_date', 'target', 'target_end_date', 'type', 'quantile', 'value']

    return preds_df
