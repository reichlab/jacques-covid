import argparse

import itertools
import pandas as pd
from datetime import date, timedelta

from run_util import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch forecast jobs')

    main_args = parser.add_argument_group('main arguments')
    main_args.add_argument('--config_file', help='configuration file (default: config.json)', default='config.json')
    main_args.add_argument('--model_config', help='name of model', default = 'kcqe_taylor')
    main_args.add_argument('--forecast_date', help='forecast date', default='2021-06-28')
    main_args.add_argument('--horizon', help='forecast horizon', default=1)

    args = parser.parse_args()

    # load config file and process args.model_group to get model_configs
    config = load_config(args.config_file)
