import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import date, timedelta

from jacovid import util
from run_util import load_config, instantiate_model, make_preds_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch forecast jobs')
    
    main_args = parser.add_argument_group('main arguments')
    main_args.add_argument('--config_file', help='configuration file (default: config.json)', default='config.json')
    main_args.add_argument('--model_config', help='name of model', default = 'kcqe_rollmean')
    main_args.add_argument('--forecast_date', help='forecast date', default='2021-06-28')
    main_args.add_argument('--horizon', help='forecast horizon', default=1, type=int)
    # config_file = 'config.json'
    # model_config = 'kcqe_rollmean'
    # forecast_date = '2021-06-28'
    # horizon = 1
    
    args = parser.parse_args()
    config_file = args.config_file
    model_config = args.model_config
    forecast_date = args.forecast_date
    horizon = args.horizon
    
    # set up file paths
    preds_dir = Path('forecasts_by_horizon') / model_config
    preds_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    preds_file_path = preds_dir / f'{forecast_date}-{model_config}-{horizon}.csv'
    
    params_dir = Path('estimates_by_horizon') / model_config
    params_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    params_file_path = params_dir / f'{forecast_date}-{model_config}-{horizon}.npz'
    
    trace_dir = Path('loss_trace_by_horizon') / model_config
    trace_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    trace_file_path = params_dir / f'{forecast_date}-{model_config}-{horizon}.npz'
    
    # exit early if predictions already exist
    if preds_file_path.exists():
        print(f'Skipping estimation for {model_config}, {forecast_date} horizon {horizon}')
        sys.exit(0)
    
    # load data
    data = util.load_data(as_of = forecast_date, end_date = forecast_date)
    
    # determine effective horizon relative to last observed data date
    last_obs_date = data.date.max()
    horizon_adjust = (pd.Timestamp(forecast_date) - last_obs_date).days
    
    # quantile levels at which to forecast
    pred_tau = tf.constant(np.array([0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
        0.90, 0.95, 0.975, 0.99]), dtype=np.float32)
    
    # get model instance
    config = load_config(config_file)
    model_settings = config['model_configs'][model_config]
    model_args = model_settings['model_args']
    model_args['h'] = horizon + horizon_adjust
    model_args['tau'] = pred_tau
    model = instantiate_model(model_settings['model'], model_args)
    
    # if there is an existing fit from a previous forecast date,
    # load in previous parameter estimates as initial values for
    # the current estimation task
    prev_fit_files = [f for f in params_dir.glob(f'*-{model_config}-{horizon}.npz')]
    date_start_ind = len(str(params_dir)) + 1
    prev_fit_dates = [str(f)[date_start_ind:(date_start_ind + 10)] \
        for f in prev_fit_files]
    prev_fit_files = [prev_fit_files[i] \
        for i, fit_date in enumerate(prev_fit_dates) \
            if fit_date < forecast_date]
    
    if len(prev_fit_files) > 0:
        prev_fit_file = sorted(prev_fit_files)[-1]
        init_param_vec = tf.constant(np.load(prev_fit_file)['param_vec'])
        num_epochs=model_settings['update_fit_epochs']
    else:
        init_param_vec = None
        num_epochs=model_settings['init_fit_epochs']
    
    # fit model; note that we fit using a smaller array of quantile levels tau
    model.fit(data,
              tau=tf.constant(np.array([0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]), dtype=np.float32),
              num_epochs=num_epochs,
              init_param_vec=init_param_vec)
    
    # get predictions
    q_hat = model.predict()
    
    # assemble predictions data frame
    preds_df = make_preds_df(q_hat.numpy().squeeze(),
                             pred_tau.numpy(),
                             data.location.unique(),
                             forecast_date,
                             horizon)
    
    # save predictions
    preds_df.to_csv(preds_file_path, index = False)
    
    # save parameter estimates
    np.savez_compressed(params_file_path, param_vec=model.param_vec.numpy())
    
    # save loss trace
    np.savez_compressed(trace_file_path, loss_trace=model.kcqe_obj.loss_trace)

