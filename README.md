# Contrastive Learning from EDA Data

Code from [Contrastive Learning of Electrodermal Activity for Stress Detection](https://drive.google.com/file/d/19zVyHcHshMA4dGPCL_R_bcVAwxNb-QAk/view).

## Getting Started
* TODO: add information on requirements.
* TODO: add information on dataset creation
### Data Augmentations
### Contrastive Pre-training
1. Create config, following template in ``config/contrastive-pretrain-template-config.json``. You need to edit the following entries:
   * `mlflow_experiment_name`: Prefix to use when naming mlflow experiment.
   * `mlflow_uri`: URI for mlflow remote tracking.
   * `train_dataset_args: dataset_path`: Path to dataset for model pretraining.
   * `train_dataset_args: dataset_name`: Name of dataset class to use.
   * `log_args:output_dir`: Path to output directory.
   * By default, all data augmentations are used. If you want to run the pretraining with just a subset of augmentations, edit the list of data augmentations included in the ``train_dataset_args: data_transform_names`` argument.
3. Run `python main.py --config_path="<PATH TO CONFIG>"'
### Model Evaluation
