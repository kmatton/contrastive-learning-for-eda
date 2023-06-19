"""
Main file to run experiments.
"""

import argparse
import copy
import json
import os
import time

import ray
import mlflow


from run_exp import ExpRunner
from utils import param_dict_to_list, add_nested_item_to_dict, get_contrastive_encoder_params, preprocess_config_dict, \
    get_contrastive_pt_sweep_params


def single_run(config_dict, run_num=0):
    init_time = time.time()
    exp_runner = ExpRunner(config_dict, run_num)
    exp_runner.run()
    print(f"finished run {run_num} in time {time.time() - init_time}")


@ray.remote(num_cpus=2, num_gpus=0.25, max_calls=0, scheduling_strategy="SPREAD")
def single_run_ray(config_dict, run_num):
    single_run(config_dict, run_num)


def run_contrastive_eval(config_dict, use_ray=True, get_splits=True):
    print("setting up supervised learning exp with encoder pretrained with contrastive_learning")
    sweep_param_list = (
        get_contrastive_encoder_params(
            config_dict["contrastive_mlflow_exp_names"],
            config_dict["sweep_params"],
            config_dict["mlflow_uri"],
            get_splits=get_splits
        )
    )
    sweep_params(sweep_param_list, config_dict, use_ray=use_ray)


def sweep_params(sweep_param_list, config_dict, start_run=0, use_ray=True):

    if use_ray:
        ray.init(
            address="auto", 
            runtime_env={
                "working_dir": ".",
                "py_modules": ["."],
                "eager_install": True,
            }
        ) 
    
    increment_seeds = False
    if "increment_seeds" in config_dict:
        increment_seeds = config_dict["increment_seeds"]
    
    # read file with already completed experiments
    base_dir = config_dict["train_args"]["log_args"]["output_dir"]
    exp_dir = os.path.join(base_dir, config_dict["mlflow_experiment_name"])
    params_file = os.path.join(exp_dir, "sweep_params_complete.txt")
    finished_params = set()
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            finished_params = set(f.read().splitlines())
    
    print(f"Sweeping over {len(sweep_param_list)} parameter settings")
    runs = []
    for idx, param_setting in enumerate(sweep_param_list):
        if idx < start_run:
            continue
        # update arguments with this setting
        for k, v in param_setting.items():
            # print(f"{k} = {v}")
            if not isinstance(k, tuple):
                config_dict[k] = v
            else:
                add_nested_item_to_dict(config_dict, list(k), v)
        param_setting_str = str(param_setting)
        if param_setting_str in finished_params:
            print(f"Already finished run {idx} with params {finished_params}, skipping")
            continue
        config_dict["sweep_setting"] = param_setting_str
        if increment_seeds:
            config_dict["train_args"]["seed"] = idx
            print(f"setting seed = {idx}")
        
        if use_ray:
            runs.append(single_run_ray.remote(
                copy.deepcopy(config_dict),
                run_num=idx
            ))
        else:
            single_run(
                copy.deepcopy(config_dict),
                run_num=idx,
            )
    
    if use_ray:
        ray.get(runs)
        ray.shutdown()


def init_mlflow_experiment(mlflow_experiment_name, mlflow_uri):
    """ 
    Creates an mlflow experiment for tracking 
    Note: we create this here to avoid any race-condition like bugs when multiprocessing
    """
    mlflow.set_tracking_uri(mlflow_uri)

    t = time.localtime()
    timestamp = time.strftime('%Y%m%d_%H%M%S', t)
    ts_mlflow_experiment_name = f"{timestamp}_{mlflow_experiment_name}"

    try:
        mlflow.create_experiment(ts_mlflow_experiment_name)
    except:
        print(f"Experiment = {ts_mlflow_experiment_name} already exists. Will log new runs to it.")

    mlflow.set_experiment(ts_mlflow_experiment_name)
    experiment_id = mlflow.get_experiment_by_name(ts_mlflow_experiment_name).experiment_id

    print(f"Starting experiment: {ts_mlflow_experiment_name} with ID {experiment_id}")

    return ts_mlflow_experiment_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to json config file specifying experiment options.')
    parser.add_argument('--no_ray', action="store_false", help='Do not use ray to sweep')
    args = parser.parse_args()
    print("Loading config file at {}".format(args.config_path))
    with open(args.config_path, 'r') as f:
        config_dict = json.load(f)
    config_dict = preprocess_config_dict(config_dict)

    config_dict["ts_mlflow_experiment_name"] = init_mlflow_experiment(config_dict["mlflow_experiment_name"], config_dict["mlflow_uri"])

    if config_dict["exp_type"] == "single_run":
        print("Running experiment with single train & test run")
        single_run(config_dict)
    elif config_dict["exp_type"] == "sweep":
        print("Running sweep params exp")
        assert "sweep_params" in config_dict.keys(), 'need to specify sweep params for sweep experiment'
        start_run = 0  # to start part way through parameter sweep
        if "start_run" in config_dict.keys():
            start_run = config_dict["start_run"]
        # check for contrastive pt file
        if "contrastive_pt_path" in config_dict.keys():
            print("found contrastive pt argument")
            sweep_param_list = get_contrastive_pt_sweep_params(config_dict["contrastive_pt_path"],
                                                               config_dict["sweep_params"])
        else:
            sweep_param_list = param_dict_to_list(config_dict["sweep_params"])
        sweep_params(sweep_param_list, config_dict, start_run=start_run, use_ray=args.no_ray)
    elif config_dict["exp_type"] == "contrastive_eval":
        print("Running contrastive eval exp")
        run_contrastive_eval(config_dict, use_ray=args.no_ray, get_splits=True)
    elif config_dict["exp_type"] == "transfer_eval":
        print("Running transfer learning eval")
        run_contrastive_eval(config_dict, use_ray=args.no_ray, get_splits=False)


if __name__ == '__main__':
    main()
