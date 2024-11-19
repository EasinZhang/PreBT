import numpy as np
from sklearn import model_selection


def split_dataset(data_indices, validation_method, n_splits, validation_ratio, test_set_ratio=0,
                  test_indices=None, random_seed=629, labels=None):
    """
    Splits dataset (i.e. the global datasets indices) into a test set and a training/validation set.
    The training/validation set is used to produce `n_splits` different configurations/splits of indices.

    Returns:
        test_indices: numpy array containing the global datasets indices corresponding to the test set
            (empty if test_set_ratio is 0 or None)
        train_indices: iterable of `n_splits` (num. of folds) numpy arrays,
            each array containing the global datasets indices corresponding to a fold's training set
        val_indices: iterable of `n_splits` (num. of folds) numpy arrays,
            each array containing the global datasets indices corresponding to a fold's validation set
    """

    # Set aside test set, if explicitly defined
    if test_indices is not None:
        data_indices = np.array([ind for ind in data_indices if ind not in set(test_indices)])  # to keep initial order

    datasplitter = DataSplitter.factory(validation_method, data_indices, labels)  # DataSplitter object

    # Set aside a random partition of all data as a test set
    if test_indices is None:
        if test_set_ratio:  # only if test set not explicitly defined
            datasplitter.split_testset(test_ratio=test_set_ratio, random_state=random_seed)
            test_indices = datasplitter.test_indices
        else:
            test_indices = []
    # Split train / validation sets

    datasplitter.split_validation(n_splits, validation_ratio, random_state=random_seed)

    return datasplitter.train_indices, datasplitter.val_indices, test_indices


def split_dataset_order(data_indices, validation_method, n_splits, validation_ratio, test_set_ratio=0,
                  test_indices=None, random_seed=629, labels=None):
    """
    将数据集（即全局数据集索引）划分为测试集和训练/验证集。
    训练/验证集用于生成`n_splits`不同的配置/划分索引。

    返回:
        test_indices: 包含全局数据集索引的numpy数组，表示测试集
            （如果test_set_ratio为0或None，则为空）
        train_indices: `n_splits`（折数）个numpy数组的可迭代对象，
            每个数组包含一个折的训练集全局数据集索引
        val_indices: `n_splits`（折数）个numpy数组的可迭代对象，
            每个数组包含一个折的验证集全局数据集索引
    """

    # 确定测试集索引，如果明确指定了测试集
    if test_indices is not None:
        data_indices = np.array([ind for ind in data_indices if ind not in set(test_indices)])  # 保持原始顺序

    # 计算数据集的总长度
    total_length = len(data_indices)

    # 如果没有指定测试集，则根据测试集比例进行划分
    if test_indices is None and test_set_ratio > 0:
        test_size = int(total_length * test_set_ratio)
        test_indices = data_indices[:test_size]  # 不打乱顺序
        data_indices = data_indices[test_size:]  # 剩余的作为训练/验证集

    # 根据验证集比例划分训练集和验证集
    val_size = int(len(data_indices) * validation_ratio)
    train_indices = data_indices[:-val_size]  # 不打乱顺序
    val_indices = data_indices[-val_size:]  # 剩余的作为训练集

    # 如果需要多个划分（n_splits），可以进行不同的切片（按需实现）
    if n_splits > 1:
        train_indices = [train_indices[i::n_splits] for i in range(n_splits)]
        val_indices = [val_indices[i::n_splits] for i in range(n_splits)]

    return train_indices, val_indices, test_indices


class DataSplitter(object):
    """Factory class, constructing subclasses based on feature type"""

    def __init__(self, data_indices, data_labels=None):
        """data_indices = train_val_indices | test_indices"""

        self.data_indices = data_indices  # global datasets indices
        self.data_labels = data_labels  # global raw datasets labels
        self.train_val_indices = np.copy(self.data_indices)  # global non-test indices (training and validation)
        self.test_indices = []  # global test indices

        if data_labels is not None:
            self.train_val_labels = np.copy(
                self.data_labels)  # global non-test labels (includes training and validation)
            self.test_labels = []  # global test labels # TODO: maybe not needed

    @staticmethod
    def factory(split_type, *args, **kwargs):
        if split_type == "StratifiedShuffleSplit":
            return StratifiedShuffleSplitter(*args, **kwargs)
        if split_type == "ShuffleSplit":
            return ShuffleSplitter(*args, **kwargs)
        else:
            raise ValueError("DataSplitter for '{}' does not exist".format(split_type))

    def split_testset(self, test_ratio, random_state=629):
        """
        Input:
            test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            test_indices: numpy array containing the global datasets indices corresponding to the test set
            test_labels: numpy array containing the labels corresponding to the test set
        """

        raise NotImplementedError("Please override function in child class")

    def split_validation(self):
        """
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        raise NotImplementedError("Please override function in child class")


class StratifiedShuffleSplitter(DataSplitter):
    """
    Returns randomized shuffled folds, which preserve the class proportions of samples in each fold. Differs from k-fold
    in that not all samples are evaluated, and samples may be shared across validation sets,
    which becomes more probable proportionally to validation_ratio/n_splits.
    """

    def split_testset(self, test_ratio, random_state=629):
        """
        Input:
            test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            test_indices: numpy array containing the global datasets indices corresponding to the test set
            test_labels: numpy array containing the labels corresponding to the test set
        """

        splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
        # get local indices, i.e. indices in [0, len(data_labels))
        train_val_indices, test_indices = next(splitter.split(X=np.zeros(len(self.data_indices)), y=self.data_labels))
        # return global datasets indices and labels
        self.train_val_indices, self.train_val_labels = self.data_indices[train_val_indices], self.data_labels[train_val_indices]
        self.test_indices, self.test_labels = self.data_indices[test_indices], self.data_labels[test_indices]

        return

    def split_validation(self, n_splits, validation_ratio, random_state=629):
        """
        Input:
            n_splits: number of different, randomized and independent from one-another folds
            validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        splitter = model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=validation_ratio,
                                                          random_state=random_state)
        # get local indices, i.e. indices in [0, len(train_val_labels)), per fold
        train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(self.train_val_labels)), y=self.train_val_labels))
        # return global datasets indices per fold
        self.train_indices = [self.train_val_indices[fold_indices] for fold_indices in train_indices]
        self.val_indices = [self.train_val_indices[fold_indices] for fold_indices in val_indices]

        return


class ShuffleSplitter(DataSplitter):
    """
    Returns randomized shuffled folds without requiring or taking into account the sample labels. Differs from k-fold
    in that not all samples are evaluated, and samples may be shared across validation sets,
    which becomes more probable proportionally to validation_ratio/n_splits.
    """

    def split_testset(self, test_ratio, random_state=629):
        """
        Input:
            test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            test_indices: numpy array containing the global datasets indices corresponding to the test set
            test_labels: numpy array containing the labels corresponding to the test set
        """

        splitter = model_selection.ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
        # get local indices, i.e. indices in [0, len(data_indices))
        train_val_indices, test_indices = next(splitter.split(X=np.zeros(len(self.data_indices))))
        # return global datasets indices and labels
        self.train_val_indices = self.data_indices[train_val_indices]
        self.test_indices = self.data_indices[test_indices]
        if self.data_labels is not None:
            self.train_val_labels = self.data_labels[train_val_indices]
            self.test_labels = self.data_labels[test_indices]

        return

    def split_validation(self, n_splits, validation_ratio, random_state=629):
        """
        Input:
            n_splits: number of different, randomized and independent from one-another folds
            validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute number of
                samples which is greater or equal to the number of classes
        Returns:
            train_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's training set
            val_indices: iterable of n_splits (num. of folds) numpy arrays,
                each array containing the global datasets indices corresponding to a fold's validation set
        """

        splitter = model_selection.ShuffleSplit(n_splits=n_splits, test_size=validation_ratio,
                                                random_state=random_state)
        # get local indices, i.e. indices in [0, len(train_val_labels)), per fold
        train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(self.train_val_indices))))
        # return global datasets indices per fold
        self.train_indices = [self.train_val_indices[fold_indices] for fold_indices in train_indices]
        self.val_indices = [self.train_val_indices[fold_indices] for fold_indices in val_indices]

        return
