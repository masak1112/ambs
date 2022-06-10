__author__ = "Bing Gong"
__date__ = "2022-03-17"
__email__ = "b.gong@fz-juelich.de"

from hparams_utils import *
import json, os
from abc import ABC, abstractmethod


class BaseDataset(ABC):
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
        if self.mode not in ("train", "val", "test"):
            raise ValueError("%{0}: Invalid mode {1}".format(method, self.mode))
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(
                "%{0} input_dir '{1}' does not exist".format(method, self.input_dir)
            )

        # get configuration parameters from datasplit- and model parameters-files
        with open(datasplit_config, "r") as f:  # TODO:maybe sanity check
            self.data_dict = json.loads(f.read())

        with open(hparams_dict_config, "r") as f:  # TODO:maybe sanity check
            hparams = dotdict(json.loads(f.read()))

        self.default_hparams = [
            "context_frames",
            "max_epochs",
            "batch_size",
            "shuffle_on_val",
            "sequence_length",
        ]
        self._set_hparams(hparams)

    def _set_hparams(self, hparams_dict):
        """Set all default and specific hyperparameters as attributes."""
        hparams = []
        for key in (*self.default_hparams, *self.specific_hparams()):
            try:
                self.__dict__[key] = hparams_dict[key]
            except KeyError as e:
                raise ValueError(f"missing hyperparameter: {key}")

        return hparams

    @abstractmethod
    def specific_hparams(self):
        """List names of expected hyperparameters specific to subclass."""

    @abstractmethod
    def get_filenames_from_datasplit(self):
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
    def make_dataset(self, mode):
        """
        Prepare batch_size dataset fed into to the models.
        If the data are from training dataset,then the data is shuffled;
        If the data are from val dataset, the shuffle var will be decided by the hparams.shuffled_on_val;
        if the data are from test dataset, the data will not be shuffled

        """
        shuffle = mode == "train" or (self.mode == "val" and self.shuffle_on_val)

        # get data array
        files = self.filenames(mode)
        if not len(files) > 0:
            raise Exception(
                f"no files for dataset {mode} found, check data_split dictionary"
            )
        da = self.load_data(files)

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
        dataset = dataset.map(self.normalize)

        return dataset

    def training(self):
        self.make_dataset(mode="train")

    def test(self):
        self.make_dataset(mode="test")

    def validation(self):
        self.make_dataset(mode="val")

    @abstractmethod
    def filenames(self, mode):
        pass

    @abstractmethod
    def normalize(self, x, args):
        """
        x: tensor (batch_size, seq_length, lat, lon, nvars)
        return: normalised data (tf.Tensor), the shape is the same as input "x"
        """
        pass

    def get_stat(self, stat, mode):
        return self.stats[mode][stat]

    @abstractmethod
    def num_samples(self):
        """
        obtain the number of samples per each epoch
        :return: int
        """
