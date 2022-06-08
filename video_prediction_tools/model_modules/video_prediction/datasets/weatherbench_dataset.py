# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "s.grasse@fz-juelich.de"
__author__ = "Simon Grasse"
__date__ = "2022-05-03"

from .base_dataset import BaseDataset
import xarray as xr
import tensorflow as tf

class WeatherBenchDataset(BaseDataset):
    """Weatherbench dataset as presented in https://arxiv.org/abs/2002.00469."""

    def __init__(self, inputdir: str, datasplit_config: str, hparams_dict_config: str, mode: str="train", seed: int = None, nsamples_ref: int = None):
        super.__init__(input_dir, datasplit_config, hparams_dict_config, mode, seed, nsamples_ref)
        self.extra1, self.extra2 = load_dataset_specific_hparams()

        ds = xr.open_mfdataset(self.filenames, combine="by_coords")
        da = ds.to_array(dim = "variables").squeeze()

        # get statistics
        dims = ["time", "lat", "lon"]
        self.mean = da.mean(dim=dims)
        self.std = da.std(dim=dims)

        # put array in expected format (TODO: check)
        data_arr = np.squeeze(da.values)
        data_arr = np.transpose(data_arr, (1, 2, 3, 0)) #swap positions to [n_samples, lat, lon, channels]

        return data_arr


    def normalize(x):
        # mean or min-max normalization ?
        def normalize_var(x,mean, std):
            return (x-mean) / std

        x = x.copy() # assure no inplace operation

        # normalize each variable seperatly
        for i in range(x.shape[-1]):
            x[:,:,:,:,i] = normalize_var(x[:, :, :, :, i], self.mean[i], self.std[i])
            
        return x

        

    def make_dataset(self):
        shuffle = self.mode == 'train' or (self.mode == 'val' and self.shuffle_on_val) # no shuffle in example ?

        data_arr = self.load_data()

        def data_generator():
            for d in data_arr:
                yield d

        # create tf Dataset
        dataset = tf.data.Dataset.from_generator(data_generator, output_types=tf.float32, output_shapes=[self.nlat, self.nlon, self.n_vars])

        # create training sequences
        dataset = dataset.window(self.sequence_length, shift = self.shift, drop_remainder=True)

        # create batches
        dataset = dataset.flat_map(lambda window: window.batch(self.sequence_length))

        # shuffle the data
        if shuffle:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.max_epochs, seed=self.seed)) #seed ?
        else:
            dataset = dataset.repeat(self.max_epochs)

        # obtain data with batch size
        dataset = dataset.batch(self.batch_size)

        # normalise
        dataset = dataset.map(parse_example)
        return dataset


    def num_examples_per_epoch(self):
        #total number of samples if the shift is one
        return self.n_ts - self.sequence_length + 1
        