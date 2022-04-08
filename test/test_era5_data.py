
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"

from video_prediction.datasets.era5_dataset import *
import pytest
import xarray as xr
import os
import tensorflow as tf

input_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/test_data_roshni"
datasplit_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/data_split/test/cv_test.json"
hparams_dict_config = "/p/project/deepacf/deeprain/bing/ambs/video_prediction_tools/hparams/era5/convLSTM/model_hparams_template.json"
mode = "test"


@pytest.fixture(scope="module")

def era5_dataset_case1():
    return ERA5Dataset(input_dir=input_dir, datasplit_config=datasplit_config, hparams_dict_config=hparams_dict_config,
                 mode="test", seed=1234, nsamples_ref=1000)

def test_init_era5_dataset(era5_dataset_case1):
    era5_dataset_case1.get_hparams()
    assert era5_dataset_case1.max_epochs == 20
    assert era5_dataset_case1.mode == mode
    assert era5_dataset_case1.batch_size == 4


def test_get_filenames_from_datasplit(era5_dataset_case1):
    flname= os.path.join(era5_dataset_case1.input_dir, "era5_vars4ambs_201901.nc")
    n_files = len(era5_dataset_case1.filenames)
    check = flname in era5_dataset_case1.filenames
    assert check == True
    assert n_files == 12

def test_make_dataset(era5_dataset_case1):
    # Get the data from nc files directly
    era5_dataset_case1.load_data_from_nc()
    ds = xr.open_mfdataset(era5_dataset_case1.filenames)
    len_dt = len(ds["time"].values) # count number of images/samples in the test dataset
    da = ds.to_array(dim = "variables").squeeze()
    dims = ["time", "lat", "lon"]
    max_vars, min_vars = da.max(dim=dims).values, da.min(dim=dims).values #three dimension

    #normalise the data for the first variable
    def norm_var(x, min_value, max_value):
        return (x - min_value) / (max_value - min_value)
    #mannualy calculate the normalization of the data
    dt_norm = norm_var(da[0], min_vars[0], max_vars[0])
    s1 = dt_norm[:12] #the first sample
    s2 = dt_norm[12:24] # the second sample


    # Get the data from make_dataset function
    test_dataset = era5_dataset_case1.make_dataset()
    train_iterator = test_dataset.make_one_shot_iterator()
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    train_handle = train_iterator.string_handle()
    iterator = tf.data.Iterator.from_string_handle(train_handle, test_dataset.output_types, test_dataset.output_shapes)
    inputs = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #get two samples from dataset
        s1t = sess.run(inputs)
        s2t = sess.run(inputs)
    assert (s1-s1t) == 0
    assert (s2-s2t) == 0
    #compare the data from nc files and make_dataset



