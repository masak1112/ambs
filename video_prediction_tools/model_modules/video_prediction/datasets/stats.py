from abc import ABC, abstractmethod
import dataclasses as dc
import json
from pathlib import Path

import numpy as np

@dc.dataclass
class VarStats:
    mean: float
    std: np.ndarray
    maximum: float
    minimum: float

    
@dc.dataclass
class DatasetStats:
    mean: np.ndarray
    std: np.ndarray
    maximum: np.ndarray
    minimum: np.ndarray
    n: int

    def as_array(self):
        return np.vstack([self.mean, self.std, self.max, self.min])
    
    def var_stats(self, index):
        """extract specific stats for variable at position 'index.'"""
        
        return VarStats(mean[index], std[index], maximum[index], minimum[index])
    
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
    """
    def __init__(self, stats: DatasetStats):
        self.stats: DatasetStats = stats

    @staticmethod
    def _apply_over_vars(fun, x):
        x = x.copy()  # assure no inplace operation

        # normalize each variable seperatly
        for i in range(x.shape[-1]):
            stats = self.stats.var_stats(i)
            x[:, :, :, :, i] = fun(x[:, :, :, :, i], stats)

    def normalize_vars(self, x):
        """
        Normalize each variable seperatly, using normalize_fun.
        """
        Normalize._apply_over_vars(self.normalize_fun, x, self.stats)
        return x

    def denormalize_vars(self, x):
        """
        Denormalize each variable seperatly, using denormalize_fun.
        """
        Normalize._apply_over_vars(self.deormalize_fun, x, self.stats)
        return x

    @abstractmethod
    def normalize_fun(self, x, stats: VarStats):
        """
        Normalization for data of shape (batch_size, sequence_len, lat, lon).
        """
        pass

    @abstractmethod
    def denormalize_fun(self, x, stats: VarStats):
        """
        Normalization for data of shape (batch_size, sequence_len, lat, lon).
        """



class MinMax(Normalize):
    def normalize_fun(self, x, stats: VarStats):
        return (x - stats.minimum) / (stats.maximum - stats.minimum)
    
    def denormalize_fun(self, x, stats: VarStats):
        return x * (stats.maximum - stats.minimum) + stats.minimum


class ZScore(Normalize):
    """
    Implement ZScore (De)Normalization.
    """
    def normalize_fun(self, x, stats: VarStats):
        return (x - stats.mean) / stats.std

    def denormalize_fun(self, x, stats: VarStats):
        return x * stats.std + stats.mean