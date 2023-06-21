"""
Class for handling multi-processing arguments
"""


class MultiProcessArgs:
    """
    Base class for multi-processing arguments.
    """
    def __init__(self, apply_data_parallel=True, num_workers=4):
        """
        :param apply_data_parallel: Whether or not to use Pytorch data parallel module so that batch of data is split
                                    into subgroups that are run through the model via different processes in parallel.
        :param num_workers: Number of processes to use when loading data with data loader.
        """
        self.apply_data_parallel = apply_data_parallel
        self.num_workers = num_workers
        self.n_gpu = -1

    def add_n_gpu(self, n_gpu):
        self.n_gpu = n_gpu
