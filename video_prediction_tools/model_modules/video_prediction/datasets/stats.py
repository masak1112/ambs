from abc import ABC, abstractmethod
import dataclasses as dc
import json
from pathlib import Path

import numpy as np

@dc.dataclass
class DatasetStats:
    mean: np.ndarray
    std: np.ndarray
    max: np.ndarray
    min: np.ndarray
    n: int

    def as_array(self):
        return np.vstack([self.mean, self.std, self.max, self.min])
    
    @staticmethod
    def from_json(path):
        with open(path, "r") as f:
            in_dict = json.read(f)
            
        return DatasetStats(
            **{key: np.array(in_dict[key]) if key != "n" else in_dict[key]
            for key in in_dict})

    def to_json(self, path):
        out_dict = dict(self)
        with open(path, "w") as f:
            json.dump(
                {key: list(out_dict[key]) if key != "n" else out_dict[key]
                for key in out_dict}, f)


class Normalize(ABC):
    """
    Provide normalization and denormalization for different normalization approaches.
        self._v
    """
    def __init__(self, stats: DatasetStats):
        self.stats: DatasetStats = stats

    @staticmethod
    def _apply_over_vars(fun, x, required_stats):
        x = x.copy()  # assure no inplace operation

        # normalize each variable seperatly
        for i in range(x.shape[-1]):
            var_stats = [stat[i] for stat in required_stats]
            x[:, :, :, :, i] = fun(x[:, :, :, :, i], *var_stats)

    def normalize_vars(self, x):
        """
        Normalize each variable seperatly, using normalize_fun.
        """
        Normalize._apply_over_vars(self.normalize_fun, x, self.required_stats())
        return x

    def denormalize_vars(self, x):
        """
        Denormalize each variable seperatly, using denormalize_fun.
        """
        Normalize._apply_over_vars(self.deormalize_fun, x, self.required_stats())
        return x

    @abstractmethod
    def required_stats():
        """
        Select statistics that are required for calculation of specific normalization.
        """
        pass

    @abstractmethod
    def normalize_fun(self, x, *stats):
        """
        Normalization for data of shape (batch_size, sequence_len, lat, lon).
        """
        pass

    @abstractmethod
    def denormalize_fun(self, x, *stats):
        """
        Normalization for data of shape (batch_size, sequence_len, lat, lon).
        """



class MinMax(Normalize):
    def required_stats(self):
        return (self.stats.min, self.stats.max)

    def normalize_fun(self, minimum, maximum):
        return (x - minimum) / (maximum - minimum)

    def denormalize_fun(self, minimum, maximum):
        return x * (maximum - minimum) + minimum


class ZScore(Normalize):
    """
    Implement ZScore (De)Normalization.
    """
    def required_stats(self):
        return (self.stats.mean, self.stats.std)

    def normalize_fun(self, x, mean, std):
        return (x - mean) / std

    def denormalize_fun(self, x, mean, var):
        return x * std + mean