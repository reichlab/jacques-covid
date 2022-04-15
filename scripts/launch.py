import argparse

import os
from pathlib import Path

from multiprocessing import Pool
import itertools
import pandas as pd
from datetime import date, timedelta

from run_util import load_config


def expand_grid(data_dict):
    """Create a dataframe from every combination of given values."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def run_command(command):
    """Run system command"""
    os.system(command)


if __name__ == "__main__":
    # Take user inputs
    def boolean_string(s):
      if s not in {'False', 'True'}:
          raise ValueError('Not a valid boolean string')
      return s == 'True'

    parser = argparse.ArgumentParser(description='Launch forecast jobs')

    main_args = parser.add_argument_group('main arguments')
    main_args.add_argument('--config_file', help='configuration file (default: config.json)', default='config.json')
    main_args.add_argument('--model_group', help='name of model group')
    main_args.add_argument('--fc_date_grp',
                           help='forecast dates to run: \'first_val\', \'val\', \'test\', or date formatted as \'yyyy-mm-dd\'', default = 'first_val')

    run_spec_args = parser.add_argument_group('arguments specifying how to run jobs')
    run_spec_args.add_argument('--run_setting', choices = ['local','mghpcc'], default='local', type = str)
    run_spec_args.add_argument('--cores', default=1, type=int)
    run_spec_args.add_argument('--mem', default='1000', type=str)
    run_spec_args.add_argument('--time', default='2:00', type=str)
    run_spec_args.add_argument('--queue', default='long', type=str)
    run_spec_args.add_argument('--sh_path', default='sh', type=str)
    run_spec_args.add_argument('--log_path', default='log', type=str)

    args = parser.parse_args()

    # load config file and process args.model_group to get model_configs
    config = load_config(args.config_file)
    model_configs = { model: config['model_configs'][model] \
      for model in config['model_groups'][args.model_group]['model_configs'] }

    # forecast dates for analysis
    fc_date_grp = args.fc_date_grp
    if fc_date_grp == 'first_val':
      forecast_dates = [date.fromisoformat('2021-06-28')]
    elif fc_date_grp == 'val':
      first_forecast_date = date.fromisoformat('2021-06-28')
      last_forecast_date = date.fromisoformat('2021-10-25')
      num_forecast_dates = (last_forecast_date - first_forecast_date).days // 7 + 1
      forecast_dates = [str(first_forecast_date + i * timedelta(days=7)) \
          for i in range(num_forecast_dates)]
    elif fc_date_grp == 'test':
      raise ValueError("fc_date_group 'test' is not allowed yet!")
      first_forecast_date = date.fromisoformat('2021-11-01')
      last_forecast_date = date.fromisoformat('2022-04-04') # we can update this later
      num_forecast_dates = (last_forecast_date - first_forecast_date).days // 7 + 1
      forecast_dates = [str(first_forecast_date + i * timedelta(days=7)) \
          for i in range(num_forecast_dates)]
    else:
      forecast_dates = [date.fromisoformat(fc_date_grp)]
      if forecast_dates[0] > date.fromisoformat('2021-10-25'):
        raise ValueError("fc_date_group specifying a date after 2021-10-25 is not allowed yet!")

    # all combinations of forecast date and model config
    variations = expand_grid({
      'model_config': model_configs,
      'forecast_date': forecast_dates,
      'horizon': range(1, 31)
    })

    # list of python commands for each variation
    common_args = ['--config_file ' + args.config_file]
    commands = [
      ' '.join(
        ['python3 scripts/run_one_model_date_horizon.py'] + \
          common_args + \
          ['--' + arg_name + ' ' + str(variations[arg_name][i]) \
            for arg_name in variations.columns]) \
        for i in range(variations.shape[0])
    ]

    # run in parallel on local computer or submit cluster jobs
    if args.run_setting == 'local':
      with Pool(processes=args.cores) as pool:
        pool.map(run_command, commands)
    elif args.run_setting == 'cluster':
      # remove old sh scripts
      os.system('rm ' + args.sh_path + '*.sh')
      # create new sh scripts
      for i in range(len(commands)):
        case_str = '_'.join([str(variations[arg_name][i]) for arg_name in variations.columns])
        sh_filename = args.sh_path + case_str + '.sh'
        
        cluster_logfile = args.log_path + 'lsf_logfile.out'
        job_logfile = args.log_path + case_str + '_logfile.txt'
        
        run_cmd = commands[i] + ' > ' + job_logfile
        
        request_cmds = \
          '#!/bin/bash\n' + \
          '#BSUB -n ' + args.cores_req + '\n' + \
          '#BSUB -R span[hosts=1]\n' + \
          '#BSUB -R rusage[mem=' + args.mem_req + ']\n' + \
          '#BSUB -o ' + cluster_logfile + '\n' + \
          '#BSUB -W ' + args.time_req + '\n' + \
          '#BSUB -q ' + args.queue_req + '\n' + \
          '\n' + \
          'module load singularity/singularity-3.6.2\n' + \
          'singularity exec analysis_scripts/singularity/singularity_tfp_cpu.sif ' + \
          run_cmd + '\n'
        
        with open(sh_filename, "w") as sh_file:
          sh_file.write(request_cmds)
        
        # TODO: actually submit a cluster job
