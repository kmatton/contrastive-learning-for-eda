"""
Class for handling logging arguments (during model training)
"""
import os


class LoggingArgs:
    """
    Base class for logging arguments.
    """
    def __init__(self, verbose=False, output_dir="./output", save_best_model=True, save_last_model=True):
        """
        :param verbose: whether to print out training progress
        :param output_dir: directory to output training progress and results to
        :param save_best_model: if true, save the best model trained across all epochs
        :param save_last_model: if true, save the final model
        """
        self.verbose = verbose
        self.output_dir = output_dir
        self.save_best_model = save_best_model
        self.save_last_model = save_last_model
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
