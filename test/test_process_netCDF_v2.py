
#export PYTHONPATH=/p/project/deepacf/deeprain/bing/ambs/workflow_parallel_frame_prediction:$PYTHONPATH
from data_preprocess.process_netCDF_v2 import *
import pytest
import numpy as np
import json
slices = {"lat_s": 74,
          "lat_e": 202,
          "lon_s": 550,
          "lon_e": 710
          }


@pytest.fixture(scope="module")
def preprocessData_case1(src_dir="/p/project/deepacf/deeprain/video_prediction_shared_folder/extractedData/test/",\
                         target_dir="/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test/",\
                         year="2017",job_name="01",slices=slices):
    return PreprocessNcToPkl(src_dir,target_dir,year,job_name,slices)


def test_directory_path(preprocessData_case1):
    assert preprocessData_case1.directory_to_process == "/p/project/deepacf/deeprain/video_prediction_shared_folder/extractedData/test/2017/01" 


def test_get_image_list(preprocessData_case1):
    #check the imageList is proper sorted
    imageList = preprocessData_case1.get_images_list()
    assert imageList[0] == "ecmwf_era5_17010100.nc"
    assert imageList[-1] == "ecmwf_era5_17010323.nc"
    

def test_filter_not_match_pattern_files(preprocessData_case1):
    #check if the files name match the patten and if the non-match ones are removed
    imageList = preprocessData_case1.filter_not_match_pattern_files()
    assert len(imageList) == len(preprocessData_case1.imageList_total) - 1


def test_process_images_to_list_by_month(preprocessData_case1):
    preprocessData_case1.initia_list_and_stat()
    preprocessData_case1.process_images_to_list_by_month()
    #Get the first elemnt of imageList, which is ecmwf_era5_17010100.nc and check if the variables values are equal to the first element of EU_list
    im_path = "/p/project/deepacf/deeprain/video_prediction_shared_folder/extractedData/test/2017/01/ecmwf_era5_17010100.nc" 
    with Dataset(im_path,"r") as data_file:
        times = data_file.variables["time"]
        time = num2date(times[:],units=times.units,calendar=times.calendar)
        temp = data_file.variables["T2"][0,slices["lat_s"]:slices["lat_e"], slices["lon_s"]:slices["lon_e"]] 
    #check the shape of EU_stack_list, len should be the same as the number of iamgeList, each element in the list should have the dimensions:[height, width, channels(inputs)]
    assert np.array(preprocessData_case1.EU_stack_list[0]).shape == (-slices["lat_s"]+slices["lat_e"], -slices["lon_s"]+slices["lon_e"],preprocessData_case1.nvars)
    #np.testing_assert_array_almost_equal(np.array(preprocessData_case1.EU_stack_list[0])[:,:,0],temp)              


def test_save_stat_info(preprocessData_case1):
    # statistic file to be tested
    path_test_dir = '/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/test/pickle'
    fstat2test = os.path.join(path_test_dir,'stat_01.json')
    # if statistic file is not present, create it
    if not os.path.isfile(fstat2test):
        preprocessData_case1()
        
        l_stat_exists  = os.path.isfile(os.path.join(path_test_dir,'stat_01.json'))
        l_pickle_exists= os.path.isfile(os.path.join(path_test_dir,'X_01.pkl')) and \
                         os.path.isfile(os.path.join(path_test_dir,'T_01.pkl'))
        
        assert l_stat_exists  == True
        assert l_pickle_exists== True

    temp_list = np.array(preprocessData_case1.EU_stack_list)[:,:,:,0]
    temp_mean = np.mean(temp_list)
    temp_min = np.min(temp_list)
    temp_max = np.max(temp_list)
    msl_list = np.array(preprocessData_case1.EU_stack_list)[:,:,:,1]
    msl_mean = np.mean(msl_list)
   
    with open(fstat2test) as json_file:
       data = json.load(json_file)
    assert data["T2"][0]["avg"] == pytest.approx(temp_mean,0.001)
    assert data["T2"][0]["min"] == pytest.approx(temp_min,0.001)
    assert data["T2"][0]["max"] == pytest.approx(temp_max,0.001)
    assert data["MSL"][0]["avg"] == pytest.approx(msl_mean,0.001) 
    
    #assert preprocessData_case1.save_stat_info.stat_obj["T2"]["min"] == temp_min
    #assert preprocessData_case1.save_stat_info.stat_obj["T2"]["max"] == temp_max 

