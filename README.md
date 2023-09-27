# Contrastive Learning from EDA Data

Code from [Contrastive Learning of Electrodermal Activity for Stress Detection](https://drive.google.com/file/d/19zVyHcHshMA4dGPCL_R_bcVAwxNb-QAk/view).

### Getting Started
* All required Python libraries are in the ``requirements.txt`` file.

### Dataset Preparation
* TODO: add additional information on dataset pre-processing for WESAD and VERBIO

**Creating train/val/test splits**

In the experiments in the paper above, we split the dataset into 5 folds by subject and evaluate via leave-N-subjects-out (LNSO) cross-validation. The splits we use for the WESAD and VERBIO datasets are in sub-directories within the ``dataset_splits`` directory of this repo. There is a single file associated with each of the five folds. Each file contains two columns: the first lists the subject IDS and the second lists the split (train/val/test) that examples from that subject belong to for the given fold.

#### Using Your Own Data
To apply this code to your own dataset, you need to create a new dataset class, following the examples in the ``datasets`` folder (e.g., ``datasets/wesad_dataset.py``). Then add your new dataset class as an option in the ``load_data`` function of the ``ExpRunner`` class in ``run_exp.py``. Make sure to adjust your config files (see below) to indicate this dataset class. You also will need to specify the train/val/test split(s) that you want to use in your experiments. To do this, create a new sub-directory within the  ``dataset_splits`` directory and include a file for each of the splits you would like to use (e.g., 5 splits if doing 5-fold cross validation). The first line of each file should contain two entries separated by a comma: <SPLIT KEY>,split, where <SPLIT KEY> is the attribute to split based on (e.g., subject_id, example_id). The remaining lines should list each possible value of the split key (e.g., all subject ids) and the split of the dataset (i.e., train, val, or test) that they are assigned to. See the files in ``dataset_splits/WESAD`` for an example.

### Data Augmentations
The implementations of all data augmentations are in the ``data_transforms/transform_data.py`` file.
### Contrastive Pre-training
1. Create config, following template in ``config/contrastive-pretrain-template-config.json``. You need to edit the following entries:
   * `mlflow_experiment_name`: Prefix to use when naming mlflow experiment.
   * `mlflow_uri`: URI for mlflow remote tracking.
   * `split_path`: Path to directory with files that specify the train/val/test split for each fold of the dataset.
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
