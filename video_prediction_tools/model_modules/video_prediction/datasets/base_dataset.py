__author__ = "Bing Gong"
__date__ = "2022-03-17"
__email__ = "b.gong@fz-juelich.de"

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from hparams_utils import *

@dataclass
class DatasetStats:
    mean: np.ndarray
    std: np.ndarray
    maximum: np.ndarray
    minimum: np.ndarray
    


class BaseDataset(ABC):
    modes = ["train", "val", "test"]
    dims = ["time", "lat", "lon", "variables"]
    default_hparams = [
        "context_frames",
        "max_epochs",
        "batch_size",
        "shuffle_on_val",
        "sequence_length",
    ]

    def __init__(
        self,
        input_dir: str,
        datasplit_config: str,
        hparams_dict_config: str,
        mode: str = "train",
        seed: int = None,
        nsamples_ref: int = None,
    ):
        """
        This class is used for preparing data for training/validation and test models
        :param input_dir: the path of tfrecords files
        :param datasplit_config: the path pointing to the datasplit_config json file
        :param hparams_dict_config: the path to the dict that contains hparameters,
        :param mode: string, "train","val" or "test"
        :param seed: int, the seed for dataset
        :param nsamples_ref: number of reference samples which can be used to control repetition factor for dataset
                             for ensuring adopted size of dataset iterator (used for validation data during training)
                             Example: Let nsamples_ref be 1000 while the current datset consists 100 samples, then
                                      the repetition-factor will be 10 (i.e. nsamples*rep_fac = nsamples_ref)
        """
        self.input_dir = input_dir
        self.datasplit_config = datasplit_config
        self.mode = mode
        self.seed = seed  # used for shuffeling
        self.nsamples_ref = nsamples_ref

        # sanity checks
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir '{self.input_dir}' does not exist")

        # get configuration parameters from datasplit- and model parameters-files
        with open(datasplit_config, "r") as f:  # TODO:maybe sanity check
            self.data_dict = json.loads(f.read())

        with open(hparams_dict_config, "r") as f:  # TODO:maybe sanity check
            hparams = dotdict(json.loads(f.read()))

        self._set_hparams(hparams)

        self._training_stats = None
        self._validation_stats = None
        self._test_stats = None

        self._stats_lookup = {
            "train": self._training_stats,
            "val": self._validation_stats,
            "test": self._test_stats
        }

    def _set_hparams(self, hparams_dict):
        """Set all default and specific hyperparameters as attributes."""
        hparams = []
        for key in (BaseDataset.default_hparams, *self.specific_hparams()):
            try:
                self.__dict__[key] = hparams_dict[key]
            except KeyError as e:
                raise ValueError(f"missing hyperparameter: {key}")

        return hparams

    @abstractmethod
    def specific_hparams(self):
        """List names of expected hyperparameters specific to subclass."""

    @abstractmethod
    def filenames(self, mode):
        """
        Get the filenames for train and val dataset
        Must implement in the Child class
        """
        pass

    @abstractmethod
    def load_data(self, files):
        """
        load DataSet from filenames and transform to DataArray (n_samples, lat, lon, channels)
        """
        pass

    @abstractmethod
    def normalize(self, x, args):
        """
        x: tensor (batch_size, seq_length, lat, lon, nvars)
        return: normalised data (tf.Tensor), the shape is the same as input "x"
        """
        pass

    def get_data(self, mode):
        files = self.filenames(mode)
        if not len(files) > 0:
            raise Exception(
                f"no files for dataset {mode} found, check data_split dictionary"
            )
        da = self.load_data(files)

        stats = DatasetStats(
            da.mean(dim=BaseDataset.dims[:3]).values,
            da.std(dim=BaseDataset.dims[:3]).values,
            da.max(dim=BaseDataset.dims[:3]).values,
            da.min(dim=BaseDataset.dims[:3]).values,
        )

        #if mode == "train":
            #self._training_stats = stats
        #elif mode == "val":
            #self._test_stats = stats
        #else:
            #self._validation_stats = stats
        self._stats_lookup[mode] = stats
        return da

    def make_dataset(self, mode):
        """
        Prepare batch_size dataset fedim
        """
        if mode not in BaseDataset.modes:
            raise ValueError(f"Invalid mode {mode}")

        shuffle = mode == "train" or (mode == "val" and self.shuffle_on_val)

        # get data array
        da = self._get_data(mode)

        def generator(iterable):
            iterator = iter(iterable)
            yield from iterator

        # create tf dataset
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            args = [da],
            output_types=tf.float32,
            output_shapes=da.shape[1:],
        )

        # create training sequences
        dataset = dataset.window(
            self.sequence_length, shift=self.shift, drop_remainder=True
        )  # why not dataset.batch(self.sequcence_length)
        dataset = dataset.flat_map(lambda window: window.batch(self.sequence_length))

        # shuffle
        if shuffle:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(
                    buffer_size=1024, count=self.max_epochs, seed=self.seed
                )
            )  # TODO: check, self.seed
        else:
            dataset = dataset.repeat(self.max_epochs)

        # create batches
        dataset = dataset.batch(self.batch_size)

        # normalize
        dataset = dataset.map(lambda x: self.normalize(x, mode))

        return dataset

    @property
    def num_samples(self):
        """
        obtain the number of samples per each epoch
        :return: int
        """
        # TODO calculate num_samples


    @property
    def sample_shape(self):
        """
        Obtain the shape of one training frame DataArray.
        
        return: (n_lat, n_lon, n_vars)
        """
        # TODO get info somewhere

    @property
    def training(self):
        return self.make_dataset(mode="train")

    @property
    def test(self):
        return self.make_dataset(mode="test")

    @property
    def validation(self):
        return self.make_dataset(mode="val")
    
    def _get_stats(self, mode):
        if self._stats_lookup[mode] is None:
            self.get_data(mode)
        return self._stats_lookup[mode]

    @property
    def training_stats(self):
        self._get_stats("train")
    
    @property
    def validation_stats(self):
        self._get_stats("val")
    
    @property
    def test_stats(self):
        self._get_stats("test")
