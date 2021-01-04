__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-12-04"


from main_scripts.main_meta_postprocess import *
import os
import pytest

#Params
analysis_config = "/p/home/jusers/gong1/juwels/ambs/video_prediction_tools/analysis_config/analysis_test.json" 
analysis_dir = "/p/home/jusers/gong1/juwels/video_prediction_shared_folder/analysis/bing_test1"

test_nc_fl = "/p/project/deepacf/deeprain/video_prediction_shared_folder/results/era5-Y2015to2017M01to12-64x64-3930N0000E-T2_MSL_gph500/convLSTM/20201221T181605_gong1_sunny/vfp_date_2017030109_sample_ind_8.nc"
test_dir = "/p/project/deepacf/deeprain/video_prediction_shared_folder/results/era5-Y2015to2017M01to12-64x64-3930N0000E-T2_MSL_gph500/convLSTM/20201221T181605_gong1_sunny"




#setup instance
@pytest.fixture(scope="module")
def analysis_inst():
    return MetaPostprocess(analysis_config=analysis_config,analysis_dir=analysis_dir)


def test_create_analysis_dir(analysis_inst):
    analysis_inst.create_analysis_dir()
    is_path = os.path.isdir(analysis_dir)
    assert is_path == True


def test_copy_analysis_config(analysis_inst):
    analysis_inst.copy_analysis_config()
    file_path = os.path.join(analysis_dir,"analysis_config.json")
    is_file_copied = os.path.exists(file_path)
    assert is_file_copied == True


def test_load_analysis_config(analysis_inst):
    analysis_inst.load_analysis_config()
    metrics_test = analysis_inst.metrics[0]
    test_dir_read = analysis_inst.results_dirs[0]
    assert metrics_test == "mse"
    assert test_dir_read == test_dir


def test_read_values_by_var_from_nc(analysis_inst):
    file_nc = test_nc_fl
    real,persistent,forecast,time_forecast  = analysis_inst.read_values_by_var_from_nc(fl_nc = file_nc)
    assert len(real) == len(persistent) == len(forecast) 
    assert len(time_forecast) == len(forecast)


def test_calculate_metric_one_dir(analysis_inst):
    file_nc = test_nc_fl 
    real,persistent,forecast,time_forecast  = analysis_inst.read_values_by_var_from_nc(fl_nc = file_nc)
    eval_forecast = analysis_inst.calculate_metric_one_img(real,forecast,metric="mse")

def test_load_results_dir_parameters(analysis_inst):
    analysis_inst.load_results_dir_parameters()
    assert len(analysis_inst.compare_by_values) == 2

def test_calculate_metric_all_dirs(analysis_inst):
    analysis_inst.calculate_metric_all_dirs()

#def test_calculate_metric_all_dirs(analysis_inst):
#    analysis_inst.calculate_metrics_all_dirs()
#    assert list(analysis_inst.eval_all.keys())[0] == analysis_inst.results_dirs[0]
#    print(analysis_inst.eval_all[test_dir].keys())
#    assert len(analysis_inst.eval_all[test_dir]["persistent"]["mse"][1]) == 10


#def test_calculate_mean_vars_forecast(analysis_inst):
#    analysis_inst.calculate_metrics_all_dirs()
#    analysis_inst.calculate_mean_vars_forecast()
#    assert len(analysis_inst.results_dict[test_dir]["forecasts"]) == 2


def test_plot_results(analysis_inst):
    analysis_inst.compare_by_values[0] = "savp"
    analysis_inst.plot_results()
