from abc import ABC, abstractmethod
import dataclasses as dc
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

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
        
        return VarStats(self.mean[index], self.std[index], self.maximum[index], self.minimum[index])
    
    @staticmethod
    def from_json(path):
        with open(path, "r") as f:
            in_dict = json.read(f)
            
        return DatasetStats(
            **{key: np.array(in_dict[key]) if key != "n" else in_dict[key]
            for key in in_dict})

    def to_json(self, path):
        out_dict = dc.asdict(self)
        with open(path, "w") as f:
            json.dump(
                {key: list(out_dict[key].astype(float)) if key != "n" else out_dict[key]
                for key in out_dict}, f)



class Normalize(ABC):
    """
    Provide normalization and denormalization for different normalization approaches.
    """
    def __init__(self, stats: DatasetStats):
        self.stats: DatasetStats = stats

    @staticmethod
    def _apply_over_vars(fun, x, stats):
        # x = x.copy()  # assure no inplace operation
        
        def inner_fun(i):
            var_stats = stats.var_stats(i)
            return fun(x[:, :, :, :, i], var_stats)
            

        print(f"overall shape: {x.shape}")
        # normalize each variable seperatly
        x = tf.stack([fun(x[:, :, :, :, i], stats.var_stats(i)) for i in range(x.shape[-1])], axis=-1)
        
        print(f"normalized shape: {x.shape}")
        return x

    def normalize_vars(self, x):
        """
        Normalize each variable seperatly, using normalize_fun.
        """
        return Normalize._apply_over_vars(self.normalize_fun, x, self.stats)

    def denormalize_vars(self, x):
        """
        Denormalize each variable seperatly, using denormalize_fun.
        """
        return Normalize._apply_over_vars(self.denormalize_fun, x, self.stats)

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