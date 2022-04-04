
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2021-03-03"


from data_preprocess.prepare_era5_data import *
import pytest
import os

year="2007"
job_name="01"
src_dir = "/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib"
target_dir= "/p/project/deepacf/deeprain/video_prediction_shared_folder/extractedData"
varslist_json="/p/home/jusers/gong1/juwels/ambs/video_prediction_tools/data_split/data_extraction_era5.json"


@pytest.fixture(scope="module")
def dataExtraction_case1(year=year,job_name=job_name,src_dir=src_dir,target_dir=target_dir,varslist_json=varslist_json):
    return ERA5DataExtraction(year,job_name,src_dir,target_dir,varslist_json)


def test_init(dataExtraction_case1):
    assert dataExtraction_case1.job_name == 1
    assert dataExtraction_case1.src_dir == src_dir




def test_get_varslist(dataExtraction_case1):
    dataExtraction_case1.get_varslist()
    dataExtraction_case1.varslist_keys[0] == "surface"
    dataExtraction_case1.varslist_keys[1] == "mutil"
    list(dataExtraction_case1.varslist_surface.keys())[0] == "2t"
    list(dataExtraction_case1.varslist_surface.values())[0] == "var167"



def test_prepare_era5_data_one_file(dataExtraction_case1):
    dataExtraction_case1.prepare_era5_data_one_file("2007","01","01","01")
    fd_exit = os.path.exists(os.path.join(target_dir,"2007","01"))
    assert fd_exit == True
    file_exit = os.path.exists(os.path.join(target_dir,"2007","01","ecmwf_era5_07010101.nc"))   
    assert file_exit == True
    


def test_process_era5_in_dir(dataExtraction_case1):
    dataExtraction_case1.process_era5_in_dir()
    
