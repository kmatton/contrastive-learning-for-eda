"""
Base class for custom dataset.
"""

import copy
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data_transforms.transform_data import DataTransformer


class DatasetWrapper(Dataset):
    """
    Class for base dataset that wraps Pytorch dataset.
    """
    def __init__(self, dataset_path, y_dim=None, y_var_type=None, split=None, split_file=None, sub_sample_frac=None,
                 sub_sample_count=None, sample_count_file=None, data_views=None, data_transform_names=None,
                 data_transform_args=None):
        """
        :param dataset_path: path to folder containing sub-directories with subject data
        :param y_dim: the dimension of the target variable; (None for unlabelled data)
        :param y_var_type: type of y variable (categorical, binary, or continuous); (None for unlabelled data)
        :param split: split of data to use (should be key in split_file). If not provided, will use all data.
        :param split_file: path to file that contains information about which split each entry in the dataset is in.
        :param sub_sample_frac: if provided, will randomly sub sample data to include only 'sub_sample_frac' frac of exs
        :param sub_sample_count: if provided will randomly sub sample 'sub_sample_count' exs
                                (note can only provide one of sub_sample_frac and sub_sample_count)
        :param sample_count_file: if provided, look at this file for data counts
        :param data_views: If provided, will create multiple views of the data.
                           Dict storing parameters associated with the views to create.
        :param data_transform_names: List of names of transforms to apply to signals if working with the transformed view
        :param data_transform_args: arguments to use when applying data transforms
        """
        super().__init__()
        # (1) set member variables
        self.dataset_path = dataset_path
        self.y_dim = y_dim
        self.y_var_type = y_var_type
        # (2) set up data splitting + get keys of data in split
        self.split = split
        self.split_file = split_file
        assert (self.split is None or self.split is not None and self.split_file is not None), \
            "if split is not None, split file must be provided"
        self.split_key, self.keep_ids = None, None
        if self.split is not None:
            self.get_data_split()
        # (3) set up sub-sampling
        # potentially read sample counts for file (used for split-dependent sub-sampling counts)
        self.sub_sample_frac = sub_sample_frac
        self.sub_sample_count = sub_sample_count
        self.sample_count = None
        self.sample_count_file = sample_count_file
        assert not(self.sub_sample_frac is not None and self.sub_sample_count is not None), \
            "can't use both sub sample frac and count"
        if self.sample_count_file is not None:
            self.sample_count = self.read_sample_count_file()
        # (4) read data (filtering by split happens when reading data)
        self.data_df = self.read_data()
        print(f"Full dataset size: {len(self.data_df)}")
        # (5) sub sample data
        self.maybe_sub_sample_data()
        # (6) set up data transformer (for applying DAs in contrastive learning pipeline)
        self.data_views = data_views
        self.data_transform_names = data_transform_names
        self.data_transformer = None
        self.data_transform_args = data_transform_args
        if self.data_transform_names is not None:
            self.data_transformer = self._get_data_transformer()
            # some transformations, require additional columns; add these if needed
            self._prep_data_for_transforms()
        # (7) print data stats
        if self.y_dim is not None:
            self.print_label_stats()
        # (8) Make copy of 'x' that is 'x_base'
        self.data_df['x_base'] = copy.deepcopy(self.data_df['x'])
        # (9) finally, create list of data_df rows
        self.data_list = self.data_df.to_dict(orient='records')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        if self.data_views is not None:
            # create views
            for view_name, view_params in self.data_views.items():
                self._create_data_view(item, view_name, **view_params)
        return item

    def get_data_split(self):
        split_df = pd.read_csv(self.split_file)
        # get name of column that splits are based on
        self.split_key = [col for col in split_df.columns if col != "split"]
        # if length is one, convert to value
        if len(self.split_key) == 1:
            self.split_key = self.split_key[0]
        # filter split_df by the selected split to get the entries that belong to the split
        split_df = split_df[split_df["split"] == self.split]
        self.keep_ids = split_df[self.split_key].values

    def read_data(self):
        data_df = self.get_data_df()
        if self.split is not None:
            data_df = self.filter_by_split(data_df)
        return data_df

    def maybe_sub_sample_data(self):
        print(f"sub sample frac is {self.sub_sample_frac}")
        if self.sub_sample_frac is not None:
            if self.sample_count is not None:
                sub_sample_count = int(self.sub_sample_frac * self.sample_count)
                self.data_df = self.data_df.sample(n=sub_sample_count)
            else:
                self.data_df = self.data_df.sample(frac=self.sub_sample_frac)
            print(f"Sampled {len(self.data_df)} examples")
        if self.sub_sample_count is not None:
            self.data_df = self.data_df.sample(n=self.sub_sample_count)
            print(f"Sampled {len(self.data_df)} examples")

    def get_data_df(self):
        df_list = []
        # loop through subfolders
        for sub_dir in sorted(os.listdir(self.dataset_path)):
            if sub_dir.startswith('.'):
                # hidden file
                continue
            subject_id = self._sub_dir_to_sub_id(sub_dir)
            if self.split_key is not None and self.split_key == "subject_id" and subject_id not in self.keep_ids:
                continue
            print(f"reading data for subject {subject_id}")
            subject_dir = os.path.join(self.dataset_path, sub_dir)
            subject_df = self.read_subject_data(subject_dir)
            subject_df["subject_id"] = subject_id
            subject_df = self._add_sub_dir_info(sub_dir, subject_df)
            df_list.append(subject_df)
        data_df = pd.concat(df_list)
        return data_df

    def _add_sub_dir_info(self, sub_dir, subject_df):
        return subject_df

    def read_sample_count_file(self):
        sample_counts_df = pd.read_csv(self.sample_count_file)
        assert self.split_file is not None, "need to specify split file if providing sample count file"
        sample_counts_df = sample_counts_df[sample_counts_df["fold"] == self.split_file]
        assert len(sample_counts_df) > 0, "split file must be entry in sample count df"
        sample_counts_df = sample_counts_df.set_index("splits")
        sample_counts_df = sample_counts_df[["count"]]
        sample_counts_series = sample_counts_df.squeeze()
        if self.split is not None:
            sample_count = sample_counts_series.loc[self.split]
        else:
            sample_count = sample_counts_series.sum()
        return sample_count

    def filter_by_split(self, data_df):
        # filter dataframe to include only the ids of items in the selected data split
        if isinstance(self.split_key, list):
            index_df = data_df.set_index(self.split_key)
            keep_tups = set([tuple(x) for x in self.keep_ids])
            keep_rows = index_df.index.isin(keep_tups)
            data_df = data_df[keep_rows]
        else:
            data_df = data_df[data_df[self.split_key].isin(self.keep_ids)]
        assert len(data_df), f"data df after filtering for split {self.split} is empty"
        return data_df

    def _sub_dir_to_sub_id(self, sub_dir):
        return sub_dir.split('_')[0]

    def print_label_stats(self):
        if self.y_var_type == "categorical" or self.y_var_type == "binary":
            vals, counts = np.unique(self.data_df['y'], return_counts=True)
            for val, count in zip(vals, counts):
                print(f"Label {val} count: {count} = {count * 100 / len(self.data_df)}%")
        else:
            print(f"Label mean {np.mean(self.data_df['y'])}, median {np.median(self.data_df['y'])}, "
                  f"std dev {np.std(self.data_df['y'])}")

    def _create_data_view(self, item, view_name, apply_transform):
        """
        Create alternative "view" of the data
        :param item: sample to create view for.
        :param view_name: name of new view (to use for name in item dict)
        :param apply_transform: whether to apply a randomly sampled data transform
        :return:
        """
        x_trf = copy.deepcopy(item['x_base'])
        if apply_transform:
            x_trf = self.data_transformer(item)
        item[view_name] = x_trf

    def read_subject_data(self, subject_dir):
        raise NotImplementedError

    def _get_data_transformer(self):
        return DataTransformer(self.data_transform_names, self.data_transform_args)

    def _prep_data_for_transforms(self):
        pass
