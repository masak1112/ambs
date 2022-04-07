__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Yan Ji"
__date__ = "2022-03-17"

from .base_dataset import BaseDataset
import tensorflow as tf
import xarray as xr
import numpy as np
import os

class GzprcpDataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


    def get_hparams(self):
        """
        obtain the hparams from the dict to the class variables
        """
        method = GzprcpDataset.get_hparams.__name__

        try:
            self.context_frames = self.hparams['context_frames']
            self.max_epochs = self.hparams['max_epochs']
            self.batch_size = self.hparams['batch_size']
            self.shuffle_on_val = self.hparams['shuffle_on_val']
            self.k = self.hparams['k']

        except Exception as error:
           print("Method %{}: error: {}".format(method,error))
           raise("Method %{}: the hparameter dictionary must include 'context_frames','max_epochs','batch_size','shuffle_on_val'".format(method))


    def get_filenames_from_datasplit(self):
        """
        Obtain the file names based on the datasplit.json configuration file
        """
        method = GzprcpDataset.get_filenames_from_datasplit.__name__
        self.filenames = []
        self.data_mode = self.data_dict[self.mode]
        for year in self.data_mode:
            files = "GZ_prcp_{}.nc".format(year)
            self.filenames.append(os.path.join(self.input_dir,files))

        if len(self.filenames) == 0:
            raise FileNotFoundError("No file is found for Gzprcp Dataset based on the configuration from data_split.json file!")


    def load_data_from_nc(self):
        """
        Obtain the data and meta information from the netcdf files, including the len of the samples, mim and max values
        :return: None
        """
        ds = xr.open_mfdataset(self.filenames)
        da = ds["prcp"].values
        #times = ds["time"].values #return [time_level, sequence_len, number_samples]

        min_vars, max_vars = ds["prcp"].min().values, ds["prcp"].max().values

        self.min_max_values = np.array([min_vars, max_vars])

        data_arr = np.transpose(da, (3, 2, 1, 0)) #swap n_samples and variables position [number_samples, sequence_length, lon, lat]

        #obtain the meta information
        self.lat = ds["lat"].values
        self.lon = ds["lon"].values
        self.n_samples = data_arr.shape[0]
        self.variables = ["prcp"]
        self.n_vars = len(self.variables)
        ds_time = ds["time"].values
        self.init_time = (ds_time[4]+ds_time[3]*100+ds_time[2]*10000+ds_time[1]*1000000+ds_time[0]*100000000).astype(np.int)

        self.sequence_length = data_arr.shape[1]
        return data_arr


    @staticmethod
    def Scaler(k, array):
        array[np.where(array < 0)] = 0
        return np.log(array + k) - np.log(k)

    def make_dataset(self):
        """
         Prepare batch_size dataset fed into to the models.
         If the data are from training dataset,then the data is shuffled;
         If the data are from val dataset, the shuffle var will be decided by the hparams.shuffled_on_val;
         if the data are from test dataset, the data will not be shuffled
        """
        method = GzprcpDataset.make_dataset.__name__

        shuffle = self.mode == 'train' or (self.mode == 'val' and self.shuffle_on_val)

        filenames = self.filenames

        data_arr = GzprcpDataset.load_data_from_nc(self)

        #log of data for precipitation data
        data_arr = GzprcpDataset.Scaler(self.k, data_arr)

        fixed_range = GzprcpDataset.Scaler(self.k, self.min_max_values)

        def normalize_fn(x, min_value, max_value):

            return tf.divide(tf.subtract(x, min_value), tf.subtract(max_value, min_value))


        def parse_example(line_batch):
            # features = tf.transpose(line_batch)
            features = normalize_fn(line_batch, fixed_range[0], fixed_range[1])
            return features


        if len(filenames) == 0:
            raise ("The filenames list is empty for {} dataset, please make sure your data_split dictionary is configured correctly".format( self.mode))

        else:
            #group the data into sequenceds
            dataset = tf.data.Dataset.from_tensor_slices(data_arr)

            # shuffle the data
            if shuffle:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.max_epochs))
            else:
                dataset = dataset.repeat(self.max_epochs)

            # dataset = dataset.shuffle()
            dataset = dataset.batch(self.batch_size) #obtain data with batch size
            dataset = dataset.map(parse_example) #normalise

            return dataset



'''

input_dir = "/p/largedata/jjsc42/project/deeprain/project_data/10min_AWS_prcp"
datasplit_config = "/p/project/deepacf/deeprain/ji4/ambs/video_prediction_tools/data_split/gzprcp/datasplit.json"
hparams_dict_config = "/p/project/deepacf/deeprain/ji4/ambs/video_prediction_tools/hparams/gzprcp_data/convLSTM_gan/model_hparams_template.json"
sequences_per_file = 10
mode = "val"


if __name__ == '__main__':
    GzprcpDataset(input_dir=input_dir, datasplit_config=datasplit_config, hparams_dict_config=hparams_dict_config,
                 mode="val", seed=1234, nsamples_ref=1000)
    dataset = GzprcpDataset.make_dataset()

    for next_element in dataset.take(2):
        # time_s = time.time()
        print(next_element.shape)
        pass
'''


