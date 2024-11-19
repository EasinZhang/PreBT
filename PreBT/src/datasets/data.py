from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.datasets import load_from_tsfile

from datasets import utils

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None, revin=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax", 'revin': normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.revin = revin
    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()

            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax2":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return 2*(df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)-1

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        elif self.norm_type == "revin":
            if self.revin is None:
                raise ValueError("RevIN instance must be provided for 'revin' normalization.")
            tensor_df = torch.tensor(df.values, dtype=torch.float32)
            normalized_tensor = self.revin(tensor_df, mode='norm')
            return pd.DataFrame(normalized_tensor.detach().numpy(), index=df.index, columns=df.columns)
        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def denormalize(self, df):
        """
        Args:
            df: normalized dataframe
        Returns:
            df: denormalized dataframe
        """
        if self.norm_type == "standardization":
            return (df * (self.std + np.finfo(float).eps)) + self.mean


        elif self.norm_type == "minmax":
            return (df * (self.max_val.values - self.min_val.values + np.finfo(float).eps)) + self.min_val.values

        elif self.norm_type == "minmax2":
            return ((df+1) * (self.max_val.values - self.min_val.values+ np.finfo(float).eps)/2) + self.min_val.values
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df * grouped.transform('std')) + grouped.transform('mean')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df * (grouped.transform('max') - min_vals + np.finfo(float).eps)) + min_vals

        else:
            raise NameError(f'Denormalize method "{self.norm_type}" not implemented')
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):

        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())



class ReconstructDataClass(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.max_seq_len = 512
        self.all_df, self.all_IDs = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df.set_index('Sample_index', inplace = True)
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['Voltage', 'Capacity', 'Temperature', 'Current']

        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            print(root_dir)
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(ReconstructDataClass.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(ReconstructDataClass.load_single(path) for path in input_paths)

        # 加载完成后，根据分隔符 10000 分割数据
        all_df = all_df.reset_index(drop=True)  # 重置索引

        # 假设分隔符存在于某一列中，这里我们假设是第一列，你可以根据实际情况调整
        all_df['separator'] = all_df[all_df.columns[0]].apply(lambda x: 1 if x == 10000 else 0)

        # 根据分隔符找到分割点
        split_points = all_df[all_df['separator'] == 1].index.tolist()
        split_points.insert(0, 0)  # 添加起始位置
        split_points.append(len(all_df))  # 添加结束位置

        dfs = []
        current_id = 0

        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            df_slice = all_df.iloc[start:end]
            df_slice = df_slice[df_slice['separator'] == 0]  # 删除包含10000的行

            for j in range(0, len(df_slice), self.max_seq_len):
                sample_df = df_slice.iloc[j:j + self.max_seq_len]
                if len(sample_df) >= 10:  # 如果时间步数大于等于10
                    dfs.append(sample_df.assign(Sample_index=current_id))
                    current_id += 1

        # 合并所有分割后的 DataFrame
        all_df = pd.concat(dfs).reset_index(drop=True)

        return all_df, np.arange(current_id)

    @staticmethod
    def load_single(filepath):
        df = ReconstructDataClass.read_data(filepath)
        df = ReconstructDataClass.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logging.warning(f"{num_nan} nan values in {filepath} will be replaced by 0")
            df = df.fillna(0)
        # 添加分割标识
        # 假设df有4列
        num_columns = len(df.columns)
        separator_values = [10000] * num_columns  # 每一列都是10000
        separator = pd.DataFrame([separator_values], columns=df.columns, dtype=int)  # 创建一个新行作为分割标识
        df = pd.concat([df, separator], ignore_index=True)  # 将新行添加到DataFrame的底部
        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        return df
# class SOHDataClass(BaseData):
#     """
#     Dataset class specifically for SOH task.
#     This class processes the dataset by assigning IDs based on the 10000 separator and keeps only the first 80 rows for each ID.
#     The label for each ID is the SOH value from the first row of that ID.
#     """
#
#     def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
#         self.set_num_processes(n_proc=n_proc)
#
#         self.all_df, self.all_IDs = self.load_all(root_dir, file_list=file_list, pattern=pattern)
#         self.all_df.set_index('Sample_index', inplace=True)
#
#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]
#         self.max_seq_len = 512
#         # 处理SOH任务
#
#         self.feature_names = ['Voltage', 'Capacity', 'Temperature', 'Current']
#         self.feature_df = self.all_df[self.feature_names]
#         self.labels_name = ['SOH']
#         self.labels_df = self.all_df.groupby(self.all_df.index).first()[self.labels_name]
#
#     def load_all(self, root_dir, file_list=None, pattern=None):
#         """
#         Loads datasets from csv files and assigns IDs based on the 10000 separator.
#         Keeps only the first 80 rows for each ID.
#         """
#         if file_list is None:
#             data_paths = glob.glob(os.path.join(root_dir, '*'))
#         else:
#             data_paths = [os.path.join(root_dir, p) for p in file_list]
#
#         if len(data_paths) == 0:
#             print(root_dir)
#             raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))
#
#         if pattern is None:
#             selected_paths = data_paths
#         else:
#             selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))
#
#         input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
#         if len(input_paths) == 0:
#             raise Exception("No .csv files found using pattern: '{}'".format(pattern))
#
#         if self.n_proc > 1:
#             _n_proc = min(self.n_proc, len(input_paths))
#             with Pool(processes=_n_proc) as pool:
#                 all_df = pd.concat(pool.map(SOHDataClass.load_single, input_paths))
#         else:
#             all_df = pd.concat(SOHDataClass.load_single(path) for path in input_paths)
#
#         all_df = all_df.reset_index(drop=True)
#
#         # 根据分隔符10000进行ID划分并排除分隔符
#         current_id = 0
#         ids = []
#
#         for i, row in all_df.iterrows():
#             if row[all_df.columns[0]] == 10000:  # 如果是分隔符，ID增加但不记录
#                 current_id += 1
#             else:
#                 ids.append(current_id)
#
#         # 移除分隔符行
#         all_df = all_df[all_df[all_df.columns[0]] != 10000].reset_index(drop=True)
#
#         # 将 Sample_index 列与 ids 对应
#         all_df['Sample_index'] = ids
#
#         # 保留每个ID的前80行
#         grouped = all_df.groupby('Sample_index')
#         dfs = []
#         for name, group in grouped:
#             if len(group) > 10:  # 过滤掉包含行数不足的组
#                 dfs.append(group.iloc[:80])
#
#         all_df = pd.concat(dfs).reset_index(drop=True)
#
#         return all_df, np.arange(current_id-1)
#
#     @staticmethod
#     def load_single(filepath):
#         df = SOHDataClass.read_data(filepath)
#         df = SOHDataClass.select_columns(df)
#         num_nan = df.isna().sum().sum()
#         if num_nan > 0:
#             logging.warning(f"{num_nan} nan values in {filepath} will be replaced by 0")
#             df = df.fillna(0)
#         num_columns = len(df.columns)
#         separator_values = [10000] * num_columns
#         separator = pd.DataFrame([separator_values], columns=df.columns, dtype=int)
#         df = pd.concat([df, separator], ignore_index=True)
#         return df
#
#     @staticmethod
#     def read_data(filepath):
#         df = pd.read_csv(filepath)
#         return df
#
#     @staticmethod
#     def select_columns(df):
#         return df
class SOTDataClass(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique())
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.max_seq_len = 20
        self.data_window_len =config['data_window_len']
        self.all_df, self.all_IDs = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df.set_index('Sample_index', inplace=True)
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['Voltage', 'Capacity', 'Temperature', 'Current']
        self.feature_df = self.all_df[self.feature_names]
        self.label_name = ['SOC']
        self.labels_df = self.all_df[self.label_name]


        # # 计算归一化
        #self.labels_df = (self.labels_df-25) / 10

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            print(root_dir)
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(SOTDataClass.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(SOTDataClass.load_single(path) for path in input_paths)

        # 加载完成后，根据分隔符 10000 分割数据
        all_df = all_df.reset_index(drop=True)  # 重置索引

        # 假设分隔符存在于某一列中，这里我们假设是第一列，你可以根据实际情况调整
        all_df['separator'] = all_df[all_df.columns[0]].apply(lambda x: 1 if x == 10000 else 0)

        # 根据分隔符找到分割点
        split_points = all_df[all_df['separator'] == 1].index.tolist()
        split_points.insert(0, 0)  # 添加起始位置
        split_points.append(len(all_df))  # 添加结束位置

        dfs = []
        current_id = 0

        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            df_slice = all_df.iloc[start:end]
            df_slice = df_slice[df_slice['separator'] == 0]  # 删除包含10000的行

            for j in range(0, len(df_slice) - self.data_window_len + 1, self.data_window_len):
                dfs.append(df_slice.iloc[j:j + self.data_window_len].assign(Sample_index=current_id))
                current_id += 1


            # # 处理最后不足max_seq_len的部分
            # if len(df_slice) % self.slide_step != 0:
            #     last_slice = df_slice.iloc[-self.max_seq_len:]
            #     if len(last_slice) > 0:
            #         dfs.append(last_slice.assign(Sample_index=current_id))
            #         current_id += 1

        # for i in range(len(split_points) - 1):
        #     start, end = split_points[i], split_points[i + 1]
        #     df_slice = all_df.iloc[start:end]
        #     df_slice = df_slice[df_slice['separator'] == 0]  # 删除包含10000的行
        #
        #     for j in range(0, len(df_slice), self.max_seq_len):
        #         sample_df = df_slice.iloc[j:j + self.max_seq_len]
        #         if len(sample_df) >= 10:  # 如果时间步数大于等于10
        #             dfs.append(sample_df.assign(Sample_index=current_id))
        #             current_id += 1
        # 合并所有分割后的 DataFrame
        all_df = pd.concat(dfs).reset_index(drop=True)

        return all_df, np.arange(current_id)

    @staticmethod
    def load_single(filepath):
        df = SOTDataClass.read_data(filepath)
        df = SOTDataClass.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logging.warning(f"{num_nan} nan values in {filepath} will be replaced by 0")
            df = df.fillna(0)
        # 添加分割标识
        # 假设df有4列
        num_columns = len(df.columns)
        separator_values = [10000] * num_columns  # 每一列都是10000
        separator = pd.DataFrame([separator_values], columns=df.columns, dtype=int)  # 创建一个新行作为分割标识
        df = pd.concat([df, separator], ignore_index=True)  # 将新行添加到DataFrame的底部
        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        return df
class SOCDataClass(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique())
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
        self.set_num_processes(n_proc=n_proc)
        self.max_seq_len = 20
        self.data_window_len = config['data_window_len']
        self.slide_step = 1  # 修改滑动步长为1
        self.all_df, self.all_IDs, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df.set_index('Sample_index', inplace=True)
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['Voltage', 'Capacity', 'Temperature', 'Current']
        self.feature_df = self.all_df[self.feature_names]
        # self.label_name = ['SOC']
        # self.labels_df = self.all_df[self.label_name]

    def load_all(self, root_dir, file_list=None, pattern=None):
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            print(root_dir)
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            _n_proc = min(self.n_proc, len(input_paths))
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(SOCDataClass.load_single, input_paths))
        else:
            all_df = pd.concat(SOCDataClass.load_single(path) for path in input_paths)


        all_df = all_df.reset_index(drop=True)  # 重置索引
        for column in all_df.select_dtypes(include=['float64']).columns:
            all_df[column] = all_df[column].astype('float32')


        all_df['separator'] = all_df[all_df.columns[0]].apply(lambda x: 1 if x == 10000 else 0)

        split_points = all_df[all_df['separator'] == 1].index.tolist()
        split_points.insert(0, 0)  # 添加起始位置
        split_points.append(len(all_df))  # 添加结束位置

        dfs = []
        labels = []
        current_id = 0

        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            df_slice = all_df.iloc[start:end]
            df_slice = df_slice[df_slice['separator'] == 0]  # 删除包含10000的行

            for j in range(0, len(df_slice) - self.data_window_len + 1):
                sample_df = df_slice.iloc[j:j + self.data_window_len]
                dfs.append(sample_df.assign(Sample_index=current_id))
                # 提取最后一个时间步的 SOC 作为标签
                labels.append(sample_df.iloc[-1, -2])

                current_id += 1



        all_df = pd.concat(dfs).reset_index(drop=True)
        labels_df = pd.DataFrame(labels, columns=['SOC'], index=np.arange(current_id))
        for column in all_df.select_dtypes(include=['float64']).columns:
            all_df[column] = all_df[column].astype('float32')
        return all_df, np.arange(current_id), labels_df

    @staticmethod
    def load_single(filepath):
        df = SOCDataClass.read_data(filepath)
        df = SOCDataClass.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logging.warning(f"{num_nan} nan values in {filepath} will be replaced by 0")
            df = df.fillna(0)
        num_columns = len(df.columns)
        separator_values = [10000] * num_columns
        separator = pd.DataFrame([separator_values], columns=df.columns, dtype=int)
        df = pd.concat([df, separator], ignore_index=True)
        return df

    @staticmethod
    def read_data(filepath):
        df = pd.read_csv(filepath, dtype=np.float32)

        return df

    @staticmethod
    def select_columns(df):
        return df
# class SOCDataClass(BaseData):
#     """
#     Dataset class for Machine dataset.
#     Attributes:
#         all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
#             Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
#         feature_df: contains the subset of columns of `all_df` which correspond to selected features
#         feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
#         all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique())
#         max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
#             (Moreover, script argument overrides this attribute)
#     """
#
#     def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
#         self.set_num_processes(n_proc=n_proc)
#         self.max_seq_len = 20
#         self.data_window_len = config['data_window_len']
#         self.slide_step = 1  # 修改滑动步长为1
#         self.all_df, self.all_IDs, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
#         self.all_df.set_index('Sample_index', inplace=True)
#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]
#
#         self.feature_names = ['Voltage', 'Capacity', 'Temperature', 'Current']
#         self.feature_df = self.all_df[self.feature_names]
#         # self.label_name = ['SOC']
#         # self.labels_df = self.all_df[self.label_name]
#
#     def load_all(self, root_dir, file_list=None, pattern=None):
#         if file_list is None:
#             data_paths = glob.glob(os.path.join(root_dir, '*'))
#         else:
#             data_paths = [os.path.join(root_dir, p) for p in file_list]
#         if len(data_paths) == 0:
#             print(root_dir)
#             raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))
#
#         if pattern is None:
#             selected_paths = data_paths
#         else:
#             selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))
#
#         input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
#         if len(input_paths) == 0:
#             raise Exception("No .csv files found using pattern: '{}'".format(pattern))
#
#         if self.n_proc > 1:
#             _n_proc = min(self.n_proc, len(input_paths))
#             logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
#             with Pool(processes=_n_proc) as pool:
#                 all_df = pd.concat(pool.map(SOCDataClass.load_single, input_paths))
#         else:
#             all_df = pd.concat(SOCDataClass.load_single(path) for path in input_paths)
#
#         all_df = all_df.reset_index(drop=True)  # 重置索引
#
#         all_df['separator'] = all_df[all_df.columns[0]].apply(lambda x: 1 if x == 10000 else 0)
#
#         split_points = all_df[all_df['separator'] == 1].index.tolist()
#         split_points.insert(0, 0)  # 添加起始位置
#         split_points.append(len(all_df))  # 添加结束位置
#
#         dfs = []
#         labels = []
#         current_id = 0
#
#         for i in range(len(split_points) - 1):
#             start, end = split_points[i], split_points[i + 1]
#             df_slice = all_df.iloc[start:end]
#             df_slice = df_slice[df_slice['separator'] == 0]  # 删除包含10000的行
#
#             for j in range(0, len(df_slice) - self.data_window_len + 1):
#                 sample_df = df_slice.iloc[j:j + self.data_window_len]
#                 dfs.append(sample_df.assign(Sample_index=current_id))
#                 # 提取最后一个时间步的 SOC 作为标签
#                 labels.append(sample_df.iloc[-1]['SOC'])
#                 current_id += 1
#
#             # if len(df_slice) % self.data_window_len != 0:
#             #     last_slice = df_slice.iloc[-self.data_window_len:]
#             #     if len(last_slice) > 0:
#             #         dfs.append(last_slice.assign(Sample_index=current_id))
#             #         labels.append(last_slice.iloc[-1]['SOC'])
#             #         current_id += 1
#
#         all_df = pd.concat(dfs).reset_index(drop=True)
#         labels_df = pd.DataFrame(labels, columns=['SOC'], index=np.arange(current_id))
#
#         return all_df, np.arange(current_id), labels_df
#
#     @staticmethod
#     def load_single(filepath):
#         df = SOCDataClass.read_data(filepath)
#         df = SOCDataClass.select_columns(df)
#         num_nan = df.isna().sum().sum()
#         if num_nan > 0:
#             logging.warning(f"{num_nan} nan values in {filepath} will be replaced by 0")
#             df = df.fillna(0)
#         num_columns = len(df.columns)
#         separator_values = [10000] * num_columns
#         separator = pd.DataFrame([separator_values], columns=df.columns, dtype=int)
#         df = pd.concat([df, separator], ignore_index=True)
#         return df
#
#     @staticmethod
#     def read_data(filepath):
#         df = pd.read_csv(filepath)
#         return df
#
#     @staticmethod
#     def select_columns(df):
#         return df


class SOHDataClass(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique())
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
        self.set_num_processes(n_proc=n_proc)
        self.max_seq_len = 20
        self.data_window_len = config['data_window_len']
        self.slide_step = 1  # 修改滑动步长为1
        self.all_df, self.all_IDs, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df.set_index('Sample_index', inplace=True)
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['Voltage', 'Capacity', 'Temperature', 'Current']
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            print(root_dir)
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            _n_proc = min(self.n_proc, len(input_paths))
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(SOHDataClass.load_single, input_paths))
        else:
            all_df = pd.concat(SOHDataClass.load_single(path) for path in input_paths)

        all_df = all_df.reset_index(drop=True)  # 重置索引

        all_df['separator'] = all_df[all_df.columns[0]].apply(lambda x: 1 if x == 10000 else 0)

        split_points = all_df[all_df['separator'] == 1].index.tolist()
        split_points.insert(0, 0)  # 添加起始位置
        split_points.append(len(all_df))  # 添加结束位置

        dfs = []
        labels = []
        current_id = 0

        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i + 1]
            df_slice = all_df.iloc[start:end]
            df_slice = df_slice[df_slice['separator'] == 0]  # 删除包含10000的行
            if not df_slice.empty:
                dfs.append(df_slice.assign(Sample_index=current_id))
                labels.append(df_slice['SOH'].iloc[-1])
                current_id += 1


        all_df = pd.concat(dfs).reset_index(drop=True)
        labels_df = pd.DataFrame(labels, columns=['SOH'], index=np.arange(current_id))

        return all_df, np.arange(current_id), labels_df

    @staticmethod
    def load_single(filepath):
        df = SOHDataClass.read_data(filepath)
        df = SOHDataClass.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logging.warning(f"{num_nan} nan values in {filepath} will be replaced by 0")
            df = df.fillna(0)
        num_columns = len(df.columns)
        separator_values = [10000] * num_columns
        separator = pd.DataFrame([separator_values], columns=df.columns, dtype=int)
        df = pd.concat([df, separator], ignore_index=True)
        return df

    @staticmethod
    def read_data(filepath):
        df = pd.read_csv(filepath, dtype=np.float32)

        return df

    @staticmethod
    def select_columns(df):
        return df

data_factory = {

        'reconstructdataset': ReconstructDataClass,
        'sohdataset': SOHDataClass,
        'socdataset': SOCDataClass,
        'sotdataset': SOTDataClass
                }
