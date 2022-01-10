"""
Class and functions required for preprocessing ERA5 data (preprocessing substep 2)
"""
__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2021-12-13"

import os, glob
from typing import List, Union
import shutil
import subprocess as sp
import datetime as dt
from general_utils import check_str_in_list, get_path_component
from netcdf_datahandling import NetcdfUtils
from pystager_utils import PyStager

# for typing
str_or_List = Union[List, str]


class Preprocess_ERA5_data(object):

    cls_name = "Preprocess_ERA5_data"

    def __init__(self, dirin, dirout, varnames, vartypes, coord_sw, nyx, years):
        """
        This script performs several sanity checks and sets the class attributes accordingly.
        :param dirin: directory to the ERA5 reanalysis data
        :param dirout: directory where the output data will be saved
        :param varnames: list of variables to be extracted
        :param vartypes: type of variable matching varnames
        :param coord_sw: latitude and longitude of south-western corner defining target domain
        :param nyx: number of grid points in latitude and longitude direction for target domain
        """
        method = Preprocess_ERA5_data.__init__.__name__

        self.dirout = dirout if os.path.isdir(dirout) else None
        if not self.dirout:
            raise NotADirectoryError("%{0}: Output directory does not exist.".format(method))

        self.dirin, self.years = self.check_dirin(dirin, years)
        self.varnames, self.vartypes = self.check_varnames(varnames, vartypes, dirin)
        self.lat_bounds, self.lon_bounds = self.check_coords(coord_sw, nyx)

    def check_coords(self, coords_sw: List, nyx: List):

        method = Preprocess_ERA5_data.check_coords.__name__

        fexample = os.path.join(self.dirin, self.years[0], "01", "{0}010100_sf.grb".format(self.years[0]))
        fexample_nc = fexample.replace(".grb", ".nc")

        if not os.path.isfile(fexample):
            raise FileNotFoundError("%{0}: Could not find example file '{1}' for retrieving coordinate data."
                                    .format(method, fexample))

        cmd = "cdo --eccodes -f nc copy {0} {1}".format(fexample, fexample_nc)
        sp.call(cmd, shell=True)

        dexample = NetcdfUtils(fexample_nc)

        lat, lon = dexample["lon"].values, dexample["lat"].values

    @staticmethod
    def check_dirin(dirin: str, years: str_or_List):
        """
        Checks if data directories for all years exist
        :param dirin: path to basic data directory under which files are located
        :param years: years for which data is requested
        :return: status
        """
        method = Preprocess_ERA5_data.check_dirin.__name__

        years = list(years)
        # basic sanity checks
        assert isinstance(dir, str), "%{0}: Parsed dirin must be a string, but is of type '{1}'".format(method,
                                                                                                        type(dirin))
        if not all(isinstance(e, int) for e in years):
            raise ValueError("%{0}: Passed years must be a list of integers.".format(method))

        if not os.path.isdir(dirin):
            raise NotADirectoryError("%{0}: Input directory for ERA%-data '{1}' does not exist.".format(method, dirin))

        # check if at least one ERA5-datafile is present
        for year in years:
            dirin_yr = os.path.join(dirin, str(year))
            sf_f, ml_f = glob.iglob("{0}/*_sf.grb".format(dirin_yr)), glob.iglob("{0}/*_ml.grb".format(dirin_yr))

            if not os.path.isfile(sf_f):
                raise("%{0}: Could not find any ERA5-surface file for year {1:d} under '{2}'".format(method, year,
                                                                                                     dirin_yr))
            if not os.path.isfile(ml_f):
                raise("%{0}: Could not find any ERA5-surface file for year {1:d} under '{2}'".format(method, year,
                                                                                                     dirin_yr))
        return dirin, years

    @staticmethod
    def check_varnames(varnames: str_or_List, vartypes: str_or_List, datadir: str):
        """
        Check if all variables can be found in an exemplary datafile stored under datadir
        :param varnames: list of variable names
        :param vartypes: matching list of variabe type. Must be one of the following {"sf": surface, "ml": multi-level}
        :param datadir: directory where gribfiles of ERA5 reanalysis are stored
        :return:
        """
        method = Preprocess_ERA5_data.check_varnames.__name__

        allowed_vartypes = ["ml", "sf"]
        varnames, vartypes = list(varnames), list(vartypes)

        assert len(varnames) == len(vartypes), "%{0}: Number of elements in varnames and vartypes do not coincide."\
                                               .format(method)

        if not all(vartype in allowed_vartypes for vartype in vartypes):
            raise ValueError("%{0}: All vartypes must be one of the following: '{1}'"
                             .format(method, ",".join(allowed_vartypes)))

        for vartype in allowed_vartypes:
            inds = [i for i, vtype in enumerate(vartypes) if vtype == vartype]
            vars2check = list(map(varnames.__getitem__, inds))

            # retrieve year and month from path to get exemplary datafile
            yr, mm = get_path_component(datadir, -2), get_path_component(datadir, -1)
            f2check = os.path.join(datadir, "{0}{1}0100_{2}.grb".format(yr, mm, vartype))

            stat = Preprocess_ERA5_data.check_var_in_grib(f2check, vars2check, labort=True)

        return varnames, vartypes

    @staticmethod
    def check_var_in_grib(gribfile: str, varnames: str_or_List, labort: bool = False):
        """
        Checks if the desired varname exists in gribfile. Requires grib_ls.
        :param gribfile: name of gribfile to be checked.
        :param varnames: name of variable or list of variable names (must be the shortName!)
        :param labort: flag if script breaks in case that variable is not found in gribfile
        :return: status of check (True if all variables are found in gribfile)
        """
        method = Preprocess_ERA5_data.check_var_in_grib.__name__

        if not (os.path.isfile(gribfile) and gribfile.endswith("grb")):
            raise FileNotFoundError("%{0}: File '{1}' does not exist or is not a grib-file.".format(method, gribfile))

        if shutil.which("grib_ls") is None: raise NotImplementedError("%{0}: Program 'grib_ls' is not available"
                                                                      .format(method))

        cmd = "grib_ls -p shortName:s {0} | tail -n +3 | head -n -3".format(gribfile)
        varlist = str(sp.check_output(cmd, stderr=sp.STDOUT, shell=True)).lstrip("b'").rstrip("'").remove(" ", "")
        varlist = varlist.split("\\n")

        stat = check_str_in_list(varlist, varnames, labort=labort)

        return stat

    @staticmethod
    def get_ERA5_coords(era5_file):
        """
        Retrieves latitude and longitude coordinates from ERA5-file
        :param era5_file: Path to ERA5-file. Can be either a netCDF- or grib2-file
        :return: numpy-arrays of latitude and longitude coordinates
        """

        method = Preprocess_ERA5_data.get_ERA5_coords.__name__

        if not os.path.isfile(era5_file):
            raise FileNotFoundError("%{0}: Could not find example file '{1}' for retrieving coordinate data."
                                    .format(method, era5_file))

        if era5_file.endswith(".grb"):
            era5_file_nc = era5_file.replace(".grb", ".nc")
            cmd = "cdo --eccodes -f nc copy {0} {1}".format(era5_file, era5_file_nc)
            sp.call(cmd, shell=True)
        elif era5_file.endswith(".nc"):
            era5_file_nc = era5_file
        else:
            raise ValueError("%{0}: '{1}' must be either a grib2 or netCDF-file.".format(method, era5_file))

        data_era5 = NetcdfUtils(era5_file_nc)

        try:
            lat, lon = data_era5.coords["lat"], data_era5["lon"]
        except Exception as err:
            print("%{0}: Failed to retrieve lat and lon from file '{1}'".format(method, era5_file_nc))
            raise err

        return lat.values, lon.values











