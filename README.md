# Contrastive Learning from EDA Data

Code from [Contrastive Learning of Electrodermal Activity for Stress Detection](https://drive.google.com/file/d/19zVyHcHshMA4dGPCL_R_bcVAwxNb-QAk/view).

## Getting Started
* TODO: add information on requirements.
* TODO: add information on dataset creation, including creating dataset split file
### Data Augmentations
### Contrastive Pre-training
1. Create config, following template in ``config/contrastive-pretrain-template-config.json``. You need to edit the following entries:
   * `mlflow_experiment_name`: Prefix to use when naming mlflow experiment.
   * `mlflow_uri`: URI for mlflow remote tracking.
   * `split_path`: Path to file specifying train/val/test split.
   * `train_dataset_args: dataset_path`: Path to dataset for model pretraining.
   * `train_dataset_args: dataset_name`: Name of dataset class to use.
   * `log_args:output_dir`: Path to output directory.
   * By default, all data augmentations are used. If you want to run the pretraining with just a subset of augmentations, edit the list of data augmentations included in the ``train_dataset_args: data_transform_names`` argument.
2. Run `python main.py --config_path=<PATH TO CONFIG>`
### Model Evaluation
1. Create config, following template in ``config/eval-template-config.json``. You need to edit the following entries:
   * `mlflow_experiment_name`: Prefix to use when naming mlflow experiment.
   * `mlflow_uri`: URI for mlflow remote tracking.
   * `contrastive_mlflow_exp_names`: List of the mlflow experiment names associated with the contrastive pretraining experiments you would like to evaluate the encoders from. The names will be of the form `<experiment_timestamp>_<mlflow_experiment_name> (will be in the list of mlflow logged runs, which can be found by checking the mlflow UI).
       * Note that during evaluation, we take each pretrained encoder for a contrastive learning experiment and evaluate it using the same seed and the same (train/val/test) dataset split as the pre-training experiment. Using the same splits, ensures that we evaluate on test data that was *not* used during pretraining.
   * `<train/val/test>_dataset_args: dataset_path`: Path to dataset for model training/validation/testing.
   * `<train/val/test>_dataset_args: dataset_name`: Name of dataset class to use for training/validation/testing.
   * `log_args: output_dir`: Path to output directory.
   * By default, we simulate the sparse label setting by sub-sampling a random 1\% of the training data to used for supervised training. If you want to change this parameter, you can do so by adjusting the `train_dataset_args: sub_sample_frac` argument.
   * By default, we evaluate using the fine-tuning scenario. To instead perform a linear evaluation (encoder weights fixed), change the `model_args: freeze_encoder` argument from ``false`` to ``true``.
2. Run `python main.py --config_path=<PATH TO CONFIG>`
