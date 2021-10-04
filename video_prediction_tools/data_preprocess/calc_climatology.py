# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Yan Ji, Bing Gong"
__date__ = "2021-05-26"
"""
Use the monthly average vaules to calculate the climitological values from the datas sources that donwload from ECMWF : /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib/monthly
"""

import os
import time
import glob
import json


class Calc_climatology(object):
    def __init__(self, input_dir="/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib/monthly", output_clim=None, region:list=[10,20], json_path=None):
        self.input_dir = input_dir
        self.output_clim = output_clim
        self.region = region
        self.lats = None
        self.lons = None
        self.json_path = json_path

    @staticmethod
    def calc_avg_per_year(grib_fl: str=None):
        """
        :param grib_fl: the relative path of the monthly average values
        :return: None
        """
        return None

    def cal_avg_all_files(self):

        """
        Average by time for all the grib files
        :return: None
        """

        method = Calc_climatology.cal_avg_all_files.__name__

        multiyears_path = os.path.join(self.output_clim,"multiyears.grb")
        climatology_path = os.path.join(self.output_clim,"climatology.grb")
        grib_files = glob.glob(os.path.join(self.input_dir,"*t2m.grb"))
        if grib_files:
            print ("{0}: There are {1} monthly years grib files".format(method,len(grib_files)))
            # merge all the files into one grib file
            os.system("cdo mergetime {0} {1}".format(grib_files,multiyears_path))
            # average by time
            os.system("cdo timavg {0} {1}".format(multiyears_path,climatology_path))

        else:
            FileExistsError("%{0}: The monthly years grib files do not exit in the input directory %{1}".format(method,self.input_dir))

        return None


    def get_lat_lon_from_json(self):
        """
        Get lons and lats from json file
        :return: list of lats, and lons
        """
        method = Calc_climatology.get_lat_lon_from_json.__name__

        if not os.path.exists(self.json_path):
            raise FileExistsError("{0}: The json file {1} does not exist".format(method,self.json_path))

        with open(self.json_path) as fl:
            meta_data = json.load(fl)

        if "coordinates" not in list(meta_data.keys()):
            raise KeyError("{0}: 'coordinates' is not one of the keys for metadata,json".format(method))
        else:
            meta_coords = meta_data["coordinates"]
            self.lats = meta_coords["lat"]
            self.lons = meta_coords["lon"]
        return self.lats, self.lons


    def get_region_climate(self):
        """
        Get the climitollgical values from the selected regions
        :return: None
        """
        pass


    def __call__(self, *args, **kwargs):

        if os.path.exists(os.path.join(self.output_clim,"climatology.grb")):
            pass
        else:
            self.cal_avg_all_files()
            self.get_lat_lon_from_json()
            self.get_region_climate()





if __name__ == '__main__':
    exp = Calc_climatology(json_path="/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2007-2019M01to12-92x56-3840N0000E-2t_tcc_t_850/metadata.json")
    exp()


