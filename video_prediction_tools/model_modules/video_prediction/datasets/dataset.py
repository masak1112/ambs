__author__ = "Bing Gong"
__date__ = "2022-03-17"
__email__ = "b.gong@fz-juelich.de"

import json
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

import xarray as xr
import tensorflow as tf

from hparams_utils import *
from model_modules.video_prediction.datasets.stats import DatasetStats, Normalize


class Dataset:
    modes = ["train", "val", "test"]
    dims = ["time", "lat", "lon", "variables"]
    hparams = [
        "context_frames",
        "max_epochs",
        "batch_size",
        "shuffle_on_val",
        "sequence_length",
        "shift"
    ]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        datasplit_path: str,
        hparams_path: str,
        normalize,
        filename_template: str,
        seed: int = None,
        nsamples_ref: int = None # TODO: implemment ?
    ):
        """
        This class is used for preparing data for training/validation and test models
        :param input_dir: the path of tfrecords files
        :param datasplit_path: the path pointing to the datasplit_config json file
        :param hparams_path: the path to the dict that contains hparameters,
        :param mode: string, "train","val" or "test"
        :param seed: int, the seed for shuffeling the dataset
        :param nsamples_ref: number of reference samples which can be used to control repetition factor for dataset
                             for ensuring adopted size of dataset iterator (used for validation data during training)
                             Example: Let nsamples_ref be 1000 while the current datset consists 100 samples, then
                                      the repetition-factor will be 10 (i.e. nsamples*rep_fac = nsamples_ref)
        :param normalize: class of the desired normalization method
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.seed = seed  # used for shuffeling
        self.nsamples_ref = nsamples_ref
        self.normalize = normalize
        self.filename_template = filename_template

        # sanity checks
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("input_dir '{self.input_dir}' does not exist")

        # get configuration parameters from datasplit- and model parameters-files
        with open(datasplit_path, "r") as f:  # TODO:maybe sanity check
            self.datasplit = json.loads(f.read())

        with open(hparams_path, "r") as f:  # TODO:maybe sanity check
            hparams = dotdict(json.loads(f.read()))

        try:
            self.context_frames = hparams["context_frames"]
            self.max_epochs = hparams["max_epochs"]
            self.batch_size = hparams["batch_size"]
            self.shuffle_on_val = hparams["shuffle_on_val"]
            self.sequence_length = hparams["sequence_length"]
            self.shift = hparams["shift"]
        except KeyError as e:
            raise ValueError(f"missing hyperparameter: {e.args[0]}")

        self._stats_lookup = {
            "train": None,
            "val": None,
            "test": None,
        }

    def load_data(self, files):
        """
        load DataSet from files and transform to DataArray (n_samples, lat, lon, channels).
        """
        
        ds = xr.open_mfdataset(files).load()
        da = ds.to_array(dim="variables").squeeze()
        return da.transpose(*Dataset.dims)
    
    def filenames(self, mode):
        """
        Get the filenames for training, validation and testing dataset.

        :param mode: differentiate datasets, should be "train", "val" or "test"
        """
        time_window = self.datasplit[mode]
        files = []
        # {"2008":[1,2,3,4,...], "2009":[1,2,3,4,...]}
        for year, months in time_window.items():
            for month in months:
                files.append(self.input_dir / self.filename_template.format(year=year, month=month))
        
        return files

    def _get_data(self, mode):
        """
        Load data from files into memory and calculate statistics.

        :param mode: indicator to differentiate between training, validation and test data
        """
        files = self.filenames(mode)
        if not len(files) > 0:
            raise Exception(
                f"no files for dataset {mode} found, check data_split dictionary"
            )
        da = self.load_data(files)

        stats = DatasetStats(
            da.mean(dim=Dataset.dims[:3]).values,
            da.std(dim=Dataset.dims[:3]).values,
            da.max(dim=Dataset.dims[:3]).values,
            da.min(dim=Dataset.dims[:3]).values,
            da.sizes["time"]
        )

        self._stats_lookup[mode] = stats
        return da

    def make_dataset(self, mode, use_training_stats=True):
        """
        Prepare Tensorflow dataset, load data and do all nessecary preprocessing.
        """
        if mode not in Dataset.modes:
            raise ValueError(f"Invalid mode {mode}")

        shuffle = mode == "train" or (mode == "val" and self.shuffle_on_val)

        # get data array
        da = self._get_data(mode).load() # load everything into memory
        print(f"xarray info: {da.shape}")

        def data_generator(iterable):
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
        )
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
        if use_training_stats: # use training stats for normalization
            stats = self.training_stats
            print(f"used training stats: {stats}")
        else: # use corresponding stats for normalization
            stats = self._stats_lookup[mode]

        normalize = self.normalize(stats)
        stats.to_json(self.output_dir / "normalization_stats.json")
        
        dataset = dataset.map(normalize.normalize_vars)

        return dataset

    @property
    def num_training_samples(self):
        """
        obtain the number of samples per each epoch
        :return: int
        """
        stats = self._get_stats("train")
        return int((stats.n + self.shift) / (self.sequence_length + self.shift))

    @property
    def num_test_samples(self):
        """
        obtain the number of samples per each epoch
        :return: int
        """
        stats = self._get_stats("test")
        return int((stats.n + self.shift) / (self.sequence_length + self.shift))
    
    @property
    def num_validation_samples(self):
        """
        obtain the number of samples per each epoch
        :return: int
        """
        stats = self._get_stats("val")
        return int((stats.n + self.shift) / (self.sequence_length + self.shift))

    def make_training(self):
        return self.make_dataset(mode="train")

    def make_test(self):
        return self.make_dataset(mode="test")

    def make_validation(self):
        return self.make_dataset(mode="val")
    
    def _get_stats(self, mode):
        if self._stats_lookup[mode] is None:
            self._get_data(mode)
        return self._stats_lookup[mode]

    @property
    def training_stats(self):
        return self._get_stats("train")
    
    @property
    def validation_stats(self):
        return self._get_stats("val")
    
    @property
    def test_stats(self):
        return self._get_stats("test")
