"""
Basic utility functions.
"""
from datetime import datetime, timedelta
import itertools
import json
import math
import time
import os
import random

import pandas as pd
import mlflow
import numpy as np
import torch



def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def print_dict(d):
    for k, v in d.items():
        print(f"{k}: {v}")


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))


def create_timestamped_subdir(base_dir, exp_name="", run_num=0):

    t = time.localtime()
    timestamp = time.strftime('%b_%d_%Y_%H%M%S', t)
    results_subdir = os.path.join(base_dir, exp_name, f"results_{timestamp}_{run_num}")

    if not os.path.exists(results_subdir):
        os.makedirs(results_subdir)

    return results_subdir


def preprocess_config_dict(config_dict):
    """
    Apply pre-processing steps to config dict. (e.g., read files in split list & set them as sweep params)
    :param config_dict: dict with config settings
    :return: processed dict
    """
    if "split_path" in config_dict.keys():
        assert "sweep_params" in config_dict.keys(), "can't set split path if not sweeping over params"
        # read files in at split path
        split_dir = config_dict["split_path"]
        split_files = sorted(os.listdir(split_dir))
        # add dir to filespath
        split_files = [os.path.join(split_dir, x) for x in split_files]
        config_dict["sweep_params"]["split_file"] = split_files
    return config_dict


def get_contrastive_pt_sweep_params(contrastive_pt_path, sweep_param_dict):
    """
    :param contrastive_pt_path: path to json file with contrastive pt parameters
    :param sweep_param_dict: dictionary over parameters to sweep over (specified in config file)
    Get sweep parameters list for case where there are contrastive pt parameters to sweep over.
    (1) Reads data associated with contrastive pre-trainign could be either:
        -- (A) dict mapping split used to train contrastive encoder to list of contrastive encoders trained w/ the split
        -- (b) list of contrastive encoders trained (not split specific)
    if (A), then does:
        (2) Sets split as split file argument and encoders as encoders to sweep over
        (3) Gets sweep param settings list for these arguments (and other sweep params specified)
        (4) Repeats for all splits
    otherwise, adds list of encoders to sweep param dict directly
    :param contrastive_pt_path: path to json file with parameters
    :return: sweep params list
    """
    with open(contrastive_pt_path, 'r') as f:
        pt_info = json.load(f)
    sweep_params_list = []
    if isinstance(pt_info, dict):
        for split_path, encoder_list in pt_info.items():
            sweep_param_dict["split_file"] = [split_path]
            sweep_param_dict["model_args"] = dict()
            sweep_param_dict["model_args"]["encoder_state_dict"] = encoder_list
            sweep_params_list += param_dict_to_list(sweep_param_dict)
    else:
        assert isinstance(pt_info, list), "pt info must be dict or list"
        sweep_param_dict["model_args"] = dict()
        sweep_param_dict["model_args"]["encoder_state_dict"] = pt_info
        sweep_params_list = param_dict_to_list(sweep_param_dict)
    return sweep_params_list


def param_dict_to_list(param_dict):
    """
    Convert dict that has entries that values to try for each parameter to list of
    dicts, where each dict represents a single setting of parameter values.
    :param param_dict: Dict mapping parameter name to list of values to test.
    :return: param_settings_list: list of dicts, each one is a single parameter setting
    """
    param_settings_list = []
    param_names = []
    param_vals = []
    param_dict_list = list(param_dict.items())
    for k, v in param_dict_list:
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                entry_key = (*k, sub_k) if isinstance(k, tuple) else (k, sub_k)
                param_dict_list.append((entry_key, sub_v))
        else:
            assert isinstance(v, list), "need to specify list of values to sweep over; not single value"
            param_names.append(k)
            param_vals.append(v)
    param_settings = itertools.product(*param_vals)
    for setting in param_settings:
        setting_dict = {name: val for name, val in zip(param_names, setting)}
        param_settings_list.append(setting_dict)
    return param_settings_list


def pull_results_mlflow(experiment_names, mlflow_uri):
    dfs = []
    for experiment_name in experiment_names:
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"Retrieving results for exp {experiment_name}")
        current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
        mlflow_experiment_id = current_experiment['experiment_id']

        # Get the mlflow runs
        experiment_results_df = mlflow.search_runs(experiment_ids=mlflow_experiment_id)
        print(f"Retrieving results for exp {experiment_name}, id {mlflow_experiment_id}")
        print(f"Results shape before filtering: {experiment_results_df.shape}")

        # Filter out any runs that shouldn't be considered
        experiment_results_df = experiment_results_df[experiment_results_df["status"] == "FINISHED"]
        print(f"Results shape after filtering: {experiment_results_df.shape}")
        dfs.append(experiment_results_df)

    all_results_df = pd.concat(dfs)
    return all_results_df


def get_contrastive_encoder_params(contrastive_mlflow_exp_names, sweep_params_dict, mlflow_uri, get_splits=True):
    """
    :param contrastive_mlflow_exp_name: list of names of contrastive pre-training experiments to get encoders from
    :param sweep_params_dict: dict with other params to sweep over
    :param mlflow_uri: mlflow URI to use to look up pre-train exp
    :param get_splits: if True, collect the splits used during pre-training and store in sweep params
        (used if evaluating on same dataset as pre-training, but not in transfer learning case)
    :return: sweep params list, list of parameter settings to sweep over
    """
    experiment_results_df = pull_results_mlflow(contrastive_mlflow_exp_names, mlflow_uri)
    sweep_params_list = []
    for idx, row in experiment_results_df.iterrows():
        output_dir = row["params.output_dir"]
        encoder_dict = os.path.join(output_dir, "best_model_encoder_net.pt")
        assert os.path.exists(encoder_dict), \
            f"found experiment without encoder net for exp {row['experiment_id']} and run_id {row['run_id']};" \
            f"encoder missing from path {encoder_dict}"
        seed = row["params.seed"]
        if get_splits:
            split_file = row["params.split_file"]
            sweep_params_dict["split_file"] = [split_file]
        if "model_args" not in sweep_params_dict.keys():
            sweep_params_dict["model_args"] = dict()
        sweep_params_dict["model_args"]["encoder_state_dict"] = [encoder_dict]
        if "train_args" not in sweep_params_dict.keys():
            sweep_params_dict["train_args"] = dict()
        sweep_params_dict["train_args"]["seed"] = [int(seed)]
        sweep_params_list += param_dict_to_list(sweep_params_dict)
    return sweep_params_list


def log_metrics_to_mlflow(results_dict, test_or_val):
    """ Logs the results metrics """
    updated_results_dict = {f"{test_or_val}_{k}": v for k, v in results_dict.items()}
    updated_results_dict = _unpack_confusion_matrix(updated_results_dict)
    mlflow.log_metrics(updated_results_dict)


def _unpack_confusion_matrix(results_dict):
    for k, v in list(results_dict.items()):
        if "confusion_matrix" in k:
            if isinstance(v, list):
                v = np.array(v)
            num_rows, num_cols = v.shape
            for i in range(num_rows):
                for j in range(num_cols):
                    results_dict[f"{k}_{i}{j}"] = v[i][j]
            del results_dict[k]
    return results_dict


def add_nested_item_to_dict(d, key_list, val):
    if len(key_list) == 1:
        d[key_list[0]] = val
        return
    if key_list[0] not in d:
        d[key_list[0]] = dict()
    add_nested_item_to_dict(d[key_list[0]], key_list[1:], val)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dt_milliseconds(dt_str):
    dt = datetime.fromisoformat(dt_str)
    return abs((dt.replace(tzinfo=None) - datetime(1970, 1, 1)) / timedelta(milliseconds=1))
