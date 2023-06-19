"""
Class for running an experiment
"""
import os
import socket
import json
import time

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import torch
import mlflow

from args.multi_process_args import MultiProcessArgs
from datasets.affective_road_dataset import AffectiveRoadEDADataset
from datasets.wesad_dataset import WESADDataset
from datasets.verbio_dataset import VerBioDataset
from datasets.marting_dataset import MartinGDataset
from modeling.training.sklearn_trainer import SKLearnTrainer
from modeling.training.trainer import Trainer
from models.classifier_model import ClassifierModel
from models.contrastive_model import ContrastiveModel
from models.encoder_classifier_model import EncoderClassifierModel
from utils import print_dict, create_timestamped_subdir, log_metrics_to_mlflow, set_seed


class ExpRunner:
    """
    Class for running a single experiment.
    """
    def __init__(self, arguments, run_num=0):
        """
        :param arguments (dict): arguments used for experiment run
        :param run_num: # of experiment run (used to name output directory when sweeping over
                  parameter settings)
        """
        self.init_time = time.time()
        self.arguments = arguments
        self.multi_proc_args = self._get_multi_proc_args()
        self.do_train = self.arguments["do_train"]
        self.do_val = self.arguments["do_val"]
        self.do_test = self.arguments["do_test"]
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.run_num = run_num
        # set seed
        self.seed = self.arguments["train_args"]["seed"]
        set_seed(self.seed)
        if self.do_train:
            self.train_dataset = self.load_data("train")
        if self.do_val:
            self.val_dataset = self.load_data("val")
        if self.do_test:
            self.test_dataset = self.load_data("test")

    def run(self):
        base_output_dir = self.arguments["train_args"]["log_args"]["output_dir"]
        self.arguments["train_args"]["log_args"]["output_dir"] = (
            create_timestamped_subdir(self.arguments["train_args"]["log_args"]["output_dir"],
                                      self.arguments["mlflow_experiment_name"],
                                      self.run_num)
        )
        # start mlflow experiment
        mlflow.set_tracking_uri(self.arguments["mlflow_uri"])
        current_experiment = dict(mlflow.get_experiment_by_name(self.arguments["ts_mlflow_experiment_name"]))
        experiment_id = current_experiment["experiment_id"]
        
        with mlflow.start_run(experiment_id=experiment_id):

            self._log_params_to_mlflow()

            model = self.init_model()
            trainer = self.init_trainer(model)

            # save args to file
            file_path = os.path.join(trainer.logging_args.output_dir, "config.json")
            with open(file_path, 'w') as outfile:
                json.dump(self.arguments, outfile)

            if self.do_train:
                print("starting train")
                trainer.train()
            if self.do_val:
                print("starting validation")
                val_results = trainer.evaluate()
                print("Validation Metrics:")
                print_dict(val_results.metrics)
                log_metrics_to_mlflow(val_results.metrics, "val")
            if self.do_test:
                print("starting testing")
                test_results = trainer.evaluate(self.test_dataset, eval_name="test")
                print("Test Metrics:")
                print_dict(test_results.metrics)
                log_metrics_to_mlflow(test_results.metrics, "test")

            self._log_artifacts_to_mlflow()

        # record sweep params
        if "sweep_setting" in self.arguments.keys():
            exp_dir = os.path.join(base_output_dir, self.arguments["mlflow_experiment_name"])
            sweep_params_file = os.path.join(exp_dir, "sweep_params_complete.txt")
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            with open(sweep_params_file, 'a') as f:
                f.write(self.arguments["sweep_setting"]+"\n")
            param_times_file = os.path.join(exp_dir, "params_runtimes.txt")
            with open(param_times_file, 'a') as f:
                runtime = time.time() - self.init_time
                f.write(self.arguments["sweep_setting"]+f"\t{runtime}"+"\n")
            finish_path_file = os.path.join(exp_dir, "finished_params_paths.txt")
            output_dir = self.arguments["train_args"]["log_args"]["output_dir"]
            with open(finish_path_file, 'a') as f:
                f.write(self.arguments["sweep_setting"]+f"\t{output_dir}"+"\n")

    def load_data(self, data_split="train"):
        split_file = None
        if "split_file" in self.arguments.keys():
            split_file = self.arguments["split_file"]
        if self.arguments[f"{data_split}_dataset_name"] == "AffectiveRoadEDADataset":
            return AffectiveRoadEDADataset(**self.arguments[f"{data_split}_dataset_args"], split_file=split_file)
        elif self.arguments[f"{data_split}_dataset_name"] == "WESADDataset":
            return WESADDataset(**self.arguments[f"{data_split}_dataset_args"], split_file=split_file)
        elif self.arguments[f"{data_split}_dataset_name"] == "VerBioDataset":
            return VerBioDataset(**self.arguments[f"{data_split}_dataset_args"], split_file=split_file)
        elif self.arguments[f"{data_split}_dataset_name"] == "MartinGDataset":
            return MartinGDataset(**self.arguments[f"{data_split}_dataset_args"], split_file=split_file)
        print(f"Unrecognized {data_split} dataset name {self.arguments[f'{data_split}_dataset_name']}")
        print("Exiting...")
        exit(1)

    def _get_multi_proc_args(self):
        return MultiProcessArgs(**self.arguments["multi_process_args"])

    def init_trainer(self, model):
        trainer_args = dict(
            train_args=self.arguments["train_args"],
            multi_proc_args=self.multi_proc_args,
            model=model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset
        )
        if self.arguments["trainer_name"] == "basic":
            return Trainer(**trainer_args)
        elif self.arguments["trainer_name"] == "sklearn":
            return SKLearnTrainer(**trainer_args)
        print(f"unrecognized trainer {self.arguments['trainer_name']}")
        print("Exiting...")
        exit(1)

    def init_model(self):
        if self.arguments["model_name"] == "ContrastiveModel":
            model = ContrastiveModel(**self.arguments["model_args"])
        elif self.arguments["model_name"] == "EncoderClassifierModel":
            model = EncoderClassifierModel(**self.arguments["model_args"])
        elif self.arguments["model_name"] == "ClassifierModel":
            model = ClassifierModel(**self.arguments["model_args"])
        elif self.arguments["model_name"] == "DummyClassifier":
            model = DummyClassifier(**self.arguments["model_args"])
        elif self.arguments["model_name"] == "DummyRegressor":
            model = DummyRegressor(**self.arguments["model_args"])
        elif self.arguments["model_name"] == "RandomForestClassifier":
            model = RandomForestClassifier(**self.arguments["model_args"])
        elif self.arguments["model_name"] == "RandomForestRegressor":
            model = RandomForestRegressor(**self.arguments["model_args"])
        elif self.arguments["model_name"] == "LogisticRegression":
            model = LogisticRegression(**self.arguments["model_args"])
        else:
            print(f"Unrecognized model {self.arguments['model_name']}")
            print("Exiting...")
            exit(1)
        if "model_state_dict" in self.arguments:
            model_state_dict = self.arguments["model_state_dict"]
            model.load_state_dict(torch.load(model_state_dict))
        return model

    def _log_dataset_args_to_mlflow(self):
        for data_split in ["train", "val", "test"]:
            if f"{data_split}_dataset_args" not in self.arguments.keys():
                continue
            dataset_args = self.arguments[f"{data_split}_dataset_args"]
            updated_dataset_args = {f"{data_split}_{k}": v for k, v in dataset_args.items()}
            if f"{data_split}_data_transform_args" in updated_dataset_args.keys():
                if len(str(updated_dataset_args[f"{data_split}_data_transform_args"])) <= 500:
                    mlflow.log_param(
                        f"{data_split}_data_transform_args",
                        updated_dataset_args[f"{data_split}_data_transform_args"],
                    )
                else:
                    n_transforms = len(updated_dataset_args[f"{data_split}_data_transform_args"]) - 2 
                    mlflow.log_param(
                        f"{data_split}_data_transform_args",
                        f"Transform string was >500 characters. Contained {n_transforms} transforms",
                    )
                del updated_dataset_args[f"{data_split}_data_transform_args"]
            mlflow.log_params(updated_dataset_args)

    def _log_params_to_mlflow(self):
        """ Unpack and log the experiment parameters """
        mlflow.log_param("server_name", socket.gethostname())
        mlflow.log_param("model_name", self.arguments["model_name"])
        split_file = None
        if "split_file" in self.arguments:
            split_file = self.arguments["split_file"]
        mlflow.log_param("split_file", split_file)
        self._log_dataset_args_to_mlflow()

        mlflow.log_param("output_dir", self.arguments["train_args"]["log_args"]["output_dir"])
        mlflow.log_param("seed", self.arguments["train_args"]["seed"])

        if "loss_args" in self.arguments["train_args"].keys():
            mlflow.log_params(self.arguments["train_args"]["loss_args"])
        if "opt_args" in self.arguments["train_args"].keys():
            mlflow.log_param("opt_args", self.arguments["train_args"]["opt_args"])
        mlflow.log_param("model_args", self.arguments["model_args"])

    def _log_artifacts_to_mlflow(self):
        """ Log contents of output directory to mlflow """
        mlflow.log_artifacts(self.arguments["train_args"]["log_args"]["output_dir"])
