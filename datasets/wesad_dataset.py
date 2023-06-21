"""
Class for dataset that consists of EDA data from the WESAD dataset
"""
import os

import numpy as np
import pandas as pd

from datasets.dataset import DatasetWrapper
from datasets.dataset_utils import get_eda_row_id_no_wrist
from utils import get_dt_milliseconds


class WESADDataset(DatasetWrapper):
    def __init__(self, dataset_path, include_labels, split=None, split_file=None,
                 sub_sample_frac=None, sub_sample_count=None, sample_count_file=None,
                 data_views=None, data_transform_names=None, data_transform_args=None):
        """
        :param dataset_path: path to folder containing sub-directories with subject data
        :param include_labels: if True, use labelled data; if false, use unlabelled segments
        :param split: split of data to use (should be key in split_file). If not provided, will use all data.
        :param split_file: path to file that contains information about which split each entry in the dataset is in.
        :param sub_sample_frac: If provided will randomly sub sample data to include only 'sub_sample_frac' frac of exs
        :param sample_count_file: if provided, look at this file for data counts
        :param data_views: If provided, will create multiple views of the data.
                           Dict storing parameters associated with the views to create.
        :param data_transform_names: List of names of transforms to apply to signals if working with the transformed view
        :param data_transform_args: arguments to use when applying data transforms
        """
        self.include_labels = include_labels
        y_dim = 1 if self.include_labels else None
        y_var_type = "binary" if self.include_labels else None
        self.timezone = "utc"
        super().__init__(dataset_path, y_dim, y_var_type, split, split_file, sub_sample_frac, sub_sample_count,
                         sample_count_file, data_views, data_transform_names, data_transform_args)

    def read_data(self):
        data_df = self.get_data_df()
        if self.split is not None:
            data_df = self.filter_by_split(data_df)

        # map y=2 to 1 and y=1 to 0
        def trf_labels(x):
            if x == 2:
                return 1
            return 0
        if "y" in data_df.columns:
            data_df["y"] = data_df["y"].apply(lambda x: trf_labels(x))
        # add column to store IDS
        data_df["ID"] = data_df.apply(lambda row: get_eda_row_id_no_wrist(row, self.timezone), axis=1)
        # convert x entries to numpy arrays
        for col in ['x', 'x_left_buffer', 'x_right_buffer']:
            if col in data_df.columns:
                data_df[col] = data_df[col].apply(lambda x: np.array(x))
        data_df["segment_start_milliseconds"] = data_df["segment_start_datetime_utc"].apply(lambda x: get_dt_milliseconds(x))
        data_df["subject_id_int"] = data_df["subject_id"].apply(lambda x: int(x[1:]))
        return data_df

    def read_subject_data(self, subject_dir):
        if self.include_labels:
            df = pd.read_csv(os.path.join(subject_dir, "EDA_labelled.csv"))
        else:
            df = pd.read_csv(os.path.join(subject_dir, "EDA_unlabelled.csv"))
        # convert each sample column into x, x_left_buffer, and x_right_buffer cols
        left_buffer_cols = [col for col in df.columns if col.startswith('x_left_buffer_')]
        right_buffer_cols = [col for col in df.columns if col.startswith('x_right_buffer')]
        buffer_cols = set(left_buffer_cols + right_buffer_cols)
        x_cols = [col for col in df.columns if col.startswith('x_') and col not in buffer_cols]
        # store left buffer, right buffer, and x as separate vars
        df['x_left_buffer'] = df[left_buffer_cols].values.tolist()
        df['x_right_buffer'] = df[right_buffer_cols].values.tolist()
        df['x'] = df[x_cols].values.tolist()
        drop_cols = ["Unnamed: 0"] + x_cols + left_buffer_cols + right_buffer_cols
        df.drop(columns=drop_cols, inplace=True)  # drop unnecessary columns
        return df

    def _sub_dir_to_sub_id(self, sub_dir):
        return sub_dir
