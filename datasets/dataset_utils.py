"""
Utilty functions for storing and processing datasets.
"""

import os


import numpy as np
import pandas as pd
import torch



def read_subject_eda_data(data_dir, dataset_name, x_file_prefix, read_labels=True, y_file_prefix=None, y_name=None,
                          wrist=None):
    """
    Function for reading EDA from a particular subject.
    :param data_dir: directory with subject's data
    :param dataset_name: name of the dataset that the data is associated with
    :param x_file_prefix: prefix of names of files with X data
    :param read_labels: if True, read y/label data, otherwise don't
    :param y_file_prefix: prefix of names of files with y data
    :param y_name: name of column in label CSV files to use as target
    :param wrist: which wrist to use data from
    :return: dataframe with subject data
    """
    df_list = []
    # left wrist data
    if wrist is None or wrist == "left":
        df = read_subject_wrist_eda_data(data_dir, dataset_name, "left",
                                         x_file_prefix, read_labels, y_file_prefix, y_name)
        df["wrist"] = "left"
        df_list.append(df)
    if wrist is None or wrist == "right":
        df = read_subject_wrist_eda_data(data_dir, dataset_name, "right",
                                         x_file_prefix, read_labels, y_file_prefix, y_name)
        df["wrist"] = "right"
        df_list.append(df)
    full_df = pd.concat(df_list)
    return full_df


def read_subject_wrist_eda_data(data_dir, dataset_name, wrist, x_file_prefix, read_labels, y_file_prefix, y_name):
    file_prefix = f"EDA_{wrist}_"
    timezone = "et" if dataset_name == "Panas" else "utc"
    # get X data
    X_file_name = file_prefix + x_file_prefix + ".csv"
    if "unlabelled" in x_file_prefix:
        df = read_eda_unlabelled(data_dir, dataset_name, X_file_name)
    else:
        df = read_eda_labelled(data_dir, dataset_name, X_file_name)
    # if y_type is not None, get y data
    if read_labels:
        assert y_name is not None, "Need to provide name of target variable"
        assert y_file_prefix is not None, "need to provide prefix of y/label files"
        y_file_name = file_prefix + y_file_prefix + ".csv" if dataset_name == "panas" else y_file_prefix + ".csv"
        y_df = read_eda_labels(data_dir, dataset_name, y_file_name, y_name, y_file_prefix)
        # merge X & y dfs
        X_time_stamp_col = df.columns[1]
        y_time_stamp_col = y_df.columns[0]
        init_len = len(df)
        df = df.merge(y_df, how='inner', left_on=X_time_stamp_col, right_on=y_time_stamp_col)
        if len(df) != init_len:
            print(f"WARNING: dropped {init_len - len(df)} samples after merging with labels")
        # standardize naming of timestamp columns
        df = df.rename(columns={X_time_stamp_col: f"segment_start_datetime_{timezone}"})
        if y_time_stamp_col != f"segment_start_datetime_{timezone}":
            df = df.drop(columns=y_time_stamp_col)
    return df


def read_eda_unlabelled(data_dir, dataset_name, file_name):
    """
    Function for reading a CSV file that contains unlabelled eda data
    :param data_dir: directory that contains the file
    :param dataset_name: name of dataset that data is associated with
    :param file_name: name of the file to read
    :return: dataframe containing EDA data
    """
    f_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(f_path)
    # get timestamps associated with left buffer, right buffer, and x
    sample_cols = df.columns[3:]
    assert int(len(sample_cols)) % 3 == 0, "length of X should be divisible by 3 (left buffer, x, right buffer)"
    seg_len = int(len(sample_cols) / 3)
    left_buffer_cols = df.columns[3:3+seg_len]
    right_buffer_cols = df.columns[3+2*seg_len:]
    x_cols = df.columns[3+seg_len:3+2*seg_len]
    # store left buffer, right buffer, and x as separate vars
    df['x_left_buffer'] = df[left_buffer_cols].values.tolist()
    df['x_right_buffer'] = df[right_buffer_cols].values.tolist()
    df['x'] = df[x_cols].values.tolist()
    # drop columns with individual samples
    df = df.drop(columns=sample_cols)
    return df


def read_eda_labelled(data_dir, dataset_name, file_name):
    """
    Function for reading a CSV file that contains eda data (raw or features) associated with panas labels
    :param data_dir: directory that contains the file
    :param dataset_name: name of dataset that data is associated with
    :param file_name: name of the file to read
    :return: dataframe containing EDA features
    """
    f_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(f_path)
    # convert NaNs to zeros
    df = df.fillna(0)
    # columns with features into single 'x' column
    metadata_col_count = 2
    if dataset_name == "AffectiveRoad":
        metadata_col_count = 3
    x_cols = df.columns[metadata_col_count:]
    df['x'] = df[x_cols].values.tolist()
    df = df.drop(columns=x_cols)
    return df


def read_eda_labels(data_dir, dataset_name, file_name, y_name, y_file_prefix):
    """
    Function for reading a CSV file that contains eda labels
    :param data_dir: directory that contains the file
    :param dataset_name: name of dataset that data is associated with
    :param file_name: name of the file to read
    :param y_name: name of column with y variable
    :param y_file_prefix: prefix of names of files with y data
    :return: dataframe containing EDA labels
    """
    f_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(f_path)
    timestamp_col = 0 if dataset_name == "panas" else 1
    keep_cols = {df.columns[timestamp_col], y_name}  # timestamp and target var columns
    remove_cols = [col for col in df.columns if col not in keep_cols]
    df = df.drop(columns=remove_cols)
    # remove rows that are missing the target variable
    df = df.dropna(subset=y_name)
    df = df.rename(columns={y_name: 'y'})
    if dataset_name == "Panas":
        # convert y column to column of ints
        df = df.astype({"y": 'int'})
        # subtract 1 from values to indicate classes 0-4
        df['y'] = df['y'].apply(lambda x: x-1)
    elif "bucketed" in y_file_prefix:
        # convert y column to column of ints
        df = df.astype({"y": 'int'})
    return df


def convert_entries_to_tensors(sample_dict, keys):
    """
    :param sample_dict: dict containing data to convert to Pytorch tensor
    :param keys: keys in dict of items to convert
    :return: sample_dict with entries converted
    """
    for key in keys:
        if not isinstance(sample_dict[key], torch.Tensor):
            sample_dict[key] = torch.tensor(sample_dict[key])
    return sample_dict


def convert_entries_to_numpy(sample_dict, keys):
    """
    :param sample_dict: dict containing data to convert to numpy array
    :param keys: keys in dict of items to convert
    :return: sample_dict with entries converted
    """
    for key in keys:
        if not isinstance(sample_dict[key], torch.Tensor):
            sample_dict[key] = np.array(sample_dict[key])
    return sample_dict


def add_opposite_wrist_col(df, timezone):
    left_wrist_df = df[df["wrist"] == "left"]
    right_wrist_df = df[df["wrist"] == "right"]
    merge_left_wrist_df = left_wrist_df.merge(right_wrist_df, on=['subject_id', f'segment_start_datetime_{timezone}'],
                                              how="left", suffixes=[None, "_opp_wrist"])
    merge_right_wrist_df = right_wrist_df.merge(left_wrist_df, on=['subject_id', f'segment_start_datetime_{timezone}'],
                                                how="left", suffixes=[None, "_opp_wrist"])
    opp_wrist_df = pd.concat([merge_left_wrist_df, merge_right_wrist_df])
    keep_cols = {"subject_id", "wrist", f"segment_start_datetime_{timezone}", "x_opp_wrist"}
    remove_cols = [col for col in opp_wrist_df.columns if col not in keep_cols]
    opp_wrist_df = opp_wrist_df.drop(columns=remove_cols)
    # merge opp_wrist_df with original df
    df = df.merge(opp_wrist_df, how='inner', on=['subject_id', 'wrist', f'segment_start_datetime_{timezone}'])
    # add column to indicate if opposite wrist data is available
    df["opp_wrist_exists"] = ~df["x_opp_wrist"].isnull()
    df.loc[~df["opp_wrist_exists"], "x_opp_wrist"] = df[~df["opp_wrist_exists"]]["x"]
    return df


def merge_opposite_wrist_col(df, timezone):
    left_wrist_df = df[df["wrist"] == "left"]
    right_wrist_df = df[df["wrist"] == "right"]
    merge_wrist_df = left_wrist_df.merge(right_wrist_df, on=['subject_id', f'segment_start_datetime_{timezone}'],
                                         how="inner", suffixes=[None, "_opp_wrist"])
    # add merged column
    merge_wrist_df['x_merged'] = merge_wrist_df.apply(lambda row: np.concatenate([row['x'], row['x_opp_wrist']]), axis=1)
    keep_cols = [col for col in merge_wrist_df.columns if not col.endswith("_opp_wrist")]
    remove_cols = [col for col in merge_wrist_df.columns if col not in keep_cols] + ["x"]
    merge_wrist_df = merge_wrist_df.drop(columns=remove_cols)
    merge_wrist_df = merge_wrist_df.rename(columns={'x_merged': 'x'})
    return merge_wrist_df


def get_eda_row_id(row, timezone):
    return f"{row['subject_id']}_{row['wrist']}_{row[f'segment_start_datetime_{timezone}']}"


def get_eda_row_id_no_wrist(row, timezone):
    return f"{row['subject_id']}_{row[f'segment_start_datetime_{timezone}']}"


def get_eda_row_id_no_ts(row):
    return f"{row['subject_id']}_{row.name}"
