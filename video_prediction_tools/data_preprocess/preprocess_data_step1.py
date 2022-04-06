"""
Class and functions required for preprocessing ERA5 data (preprocessing substep 2)
"""
__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2022-04-06"

import os, glob
from typing import List, Union, get_args
import shutil
import subprocess as sp
import numpy as np
import pandas as pd
import logging
from general_utils import check_str_in_list, get_path_component, isw
from pystager_utils import PyStager

# for typing
str_or_List = Union[List, str]


class Preprocess_ERA5_data(object):

    cls_name = "Preprocess_ERA5_data"

    def __init__(self, dirin: str, dirout: str, var_req: dict, coord_sw: List, nyx: List, years: List,
                 months: str_or_List = "all", lon_intv: List = (0., 360.), lat_intv: List = (-90., 90.),
                 dx: float = 0.3):
        """
        This script performs several sanity checks and sets the class attributes accordingly.
        :param dirin: directory to the ERA5 reanalysis data
        :param dirout: directory where the output data will be saved
        :param var_req: controlled dictionary for getting variables from ERA5-dataset, e.g. {"t": {"ml": "p850"}}
        :param coord_sw: latitude [°N] and longitude [°E] of south-western corner defining target domain
        :param nyx: number of grid points in latitude and longitude direction for target domain
        :param lon_intv: Allowed interval for slicing along longitude axis for ERA5 data (adapt if required)
        :param lat_intv: Allowed interval for slicing along latitude axis for ERA5 data (adapt if required)
        :param dx: grid spacing of regular, spherical grid onto which the ERA5 data is provided (adapt if required)
        """
        method = Preprocess_ERA5_data.__init__.__name__

        self.dirout = dirout if os.path.isdir(dirout) else None
        if not self.dirout:
            raise NotADirectoryError("%{0}: Output directory does not exist.".format(method))

        self.months = self.get_months(months)
        self.dirin, self.years = self.check_dirin(dirin, years, months)
        self.var_requests = self.check_varnames(var_req)
        # some basic grid information
        self.lat_intv, self.lon_intv = list(lat_intv), list(lon_intv)
        self.lon_intv[1] -= np.abs(dx)
        self.dx = dx
        # check provided data for regional slicing
        self.lat_bounds, self.lon_bounds = self.check_coords(coord_sw, nyx)

        # initialize PyStager
        self.era5_pystager = PyStager(self.worker_func, "year_month_list")

    def __call__(self):
        """
        Set-up and run Pystager to preprocess the data.
        :return: -
        """
        self.era5_pystager.setup(self.years, self.months)
        self.era5_pystager.run(self.dirin, self.dirout, self.var_requests, self.lat_bounds, self.lon_bounds)

    def check_varnames(self, var_req: dict):
        """
        Check if all variables can be found in an exemplary datafile stored under datadir
        :param var_req: nested dictionary where first-level keys carry request variable name. The values of these keys
                        are dictionaries whose keys denote the type of variable (e.g. "sf" for surface)
                        and whose values control (optional) vertical interpolation
        :return: the approved dictionary (same as var_req)
        """
        method = Preprocess_ERA5_data.check_varnames.__name__

        allowed_vartypes = ["ml", "sf"]

        assert isinstance(var_req, dict), "%{0}: var_req must be a (controlled) dictionary. See doc-string."\
                                          .format(method)

        if not all(isinstance(vardict, dict) for vardict in var_req.values()):
            raise ValueError("%{0}: Values of var_req-dictionary must be dictionary, i.e. pass a nested dictionary."
                             .format(method))

        varnames = list(var_req.keys())
        vartypes = [list(var_req[varname].keys())[0] for varname in varnames]

        # first check for availability of variables in grib-files
        for vartype in allowed_vartypes:
            inds = [i for i, vtype in enumerate(vartypes) if vtype == vartype]
            vars2check = list(map(varnames.__getitem__, inds))
            if not vars2check: continue                   # skip the following if no variable to check

            # construct path to exemplary datafile
            yy_str, mm_str = str(self.years[0]), "{0:02d}".format(self.months[0])
            f2check = os.path.join(self.dirin, yy_str, mm_str, "{0}{1}0100_{2}.grb".format(yy_str, mm_str, vartype))

            _ = Preprocess_ERA5_data.check_var_in_grib(f2check, vars2check, labort=True)

        # second check for content of nested dictionaries
        print("%{0}: Start checking consistency of nested dictionaries for each requested variable.".format(method))
        for varname in varnames:
            # check vartypes
            vartype = list(var_req[varname].keys())[0]
            lvl_info = list(var_req[varname].values())[0]
            if not vartype in allowed_vartypes:
                raise ValueError("%{0}: Key of variable dict for '{1}' must be one of the following types: {2}"
                                 .format(method, vartype, ", ".join(allowed_vartypes)))
            # check level types
            if vartype == "sf" and lvl_info != "":
                print("%{0}: lvl_info for surface variable '{1}' is not empty and thus will be ignored."
                      .format(method, varname))
            elif vartype == "ml" and not lvl_info.startswith("p"):
                raise ValueError("%{0}: Variable '' on model levels requires target pressure level".format(method) +
                                 "for interpolation. Thus, provide 'pX' where X denotes a pressure level in hPa.")

        print("%{0}: Check for consistency of nested dictionaries approved.".format(method))

        return var_req

    def check_coords(self, coords_sw: List, nyx: List):
        """
        Check arguments for defining target domain and return bounding coordinates.
        Note: Currently, crossing of the zero-meridian is not supported!
        :param coords_sw: latitude [°N] and longitude [°E] of south-western corner defining target domain
        :param nyx: number of grid points in latitude and longitude direction for target domain
        :return: two tuples of latitude and longitude boundaries of target domain, respectively
        """
        method = Preprocess_ERA5_data.check_coords.__name__

        if self.dx < 0:
            print("%{0}: Grid spacing should be positive. Change negative value to positive one.".format(method))
            self.dx = np.abs(self.dx)

        # convert type of input data if required
        coords_sw = [float(coord) for coord in coords_sw]
        nyx = [int(n) for n in nyx]

        if not isw(coords_sw[0], self.lat_intv):
            raise ValueError("%{0}: Latitude of south-western domain corner {1:.2f}".format(method, coords_sw[0]) +
                             "is not within expected [{0:.1f}°N, {1:.f}°N]".format(self.lat_intv[0], self.lat_intv[1]))

        if not isw(coords_sw[1], self.lon_intv):
            raise ValueError("%{0}: Longitude of south-western domain corner {1:.2f}".format(method, coords_sw[1]) +
                             "is not within expected [{0:.1f}°E, {1:.f}°E]".format(self.lon_intv[0], self.lon_intv[1]))

        coords_ne = coords_sw[0] + (nyx[0] - 1)*self.dx, coords_sw[1] + (nyx[1] - 1)*self.dx

        if not isw(coords_ne[0], self.lat_intv):
            raise ValueError("%{0}: Latitude of north-eastern domain corner {1:.2f}".format(method, coords_ne[0]) +
                             "is not within expected [{0:.1f}°N, {1:.f}°N]. Adapt nyx-argument."
                             "".format(self.lat_intv[0], self.lat_intv[1]))

        if not isw(coords_ne[1], self.lon_intv):
            raise ValueError("%{0}: Longitude of north-eastern domain corner {1:.2f}".format(method, coords_ne[1]) +
                             "is not within expected [{0:.1f}°E, {1:.f}°E]. Adapt nyx-argument."
                             .format(self.lon_intv[0], self.lon_intv[1]))

        return [coords_sw[0], coords_ne[0]], [coords_sw[1], coords_ne[1]]

    @staticmethod
    def worker_func(year_months: list, dirin: str, dirout: str, var_req: dict, lat_bounds, lon_bounds,
                    logger: logging.Logger, nmax_warn: int = 1):
        """
        Handle grib-files from dirin to extract the variables of interest incl. lateral slicing and optional interpol.
        The resulting netCDF-files contain monthly data.
        :param year_months: List of months of years (format: YYYY-MM) to be processed (list elements: datetime-objects)
        :param dirin: base input directory where grib-files are sorted in yearly and monthly sub-directories
        :param dirout: base output-directory; netCDF-files will be colledted in yearly subdirectories
        :param var_req: dictionary for querying variables and interpolation from grib-files,
                        e.g. {"t" : {"ml": "85000."}} for getting 850hPa temperature
        :param lat_bounds: lateral boundaries of target domain in meridional (latitude) direction
        :param lon_bounds: lateral boundaries of target domain in zonal (longitude) direction
        :param logger: logger instance
        :param nmax_warn: maximum number of allowed warnings/errors during runtime of worker
        :return: netCDF-file with monthly data under <dirout>/YYYY/

        TO-DO: Support interpolation on multiple pressure levels
        """
        method = Preprocess_ERA5_data.worker_func.__name__

        # sanity check
        assert isinstance(logger, logging.Logger), "%{0}: logger-argument must be a logging.Logger instance" \
            .format(method)

        varnames = list(var_req.keys())
        vartypes = [list(var_req[varname].keys())[0] for varname in varnames]

        # initilaize warn counter and start iterating over year_month-list
        nwarns = 0

        for year_month in year_months:
            year, month = int(year_month.strftime("%Y")), int(year_month.strftime("%m"))
            year_str, month_str = str(year), "{0:02d}".format(int(month))

            dirin_now, dirout_now = os.path.join(dirin, year_str, month_str), os.path.join(dirout, year_str)
            dirout_tmp = os.path.join(dirout_now, "{0}_tmp".format(month_str))
            dest_file = os.path.join(dirout, "ambs_era5_{0}{1}.nc".format(year_str, month_str))
            # create output- and temp-directory (store intermediate netCDF-files merged with -mergetime operator later)
            os.makedirs(dirout_now, exist_ok=True)
            os.makedirs(dirout_tmp, exist_ok=True)

            for vartype in np.unique(vartypes):
                logger.info("Start processing variable type '{1}'".format(method, vartype))
                vars4type = [varname for c, varname in enumerate(varnames) if vartypes[c] == vartype]
                # ensure that lnsp (logarithmic surface pressure) and z (geopotential) are in list for p-interpolation
                if vartype == "ml":
                    vars4type_aux = set(vars4type + ["lnsp", "z"])
                    # this only allows handling of a single pressure level for all variables!
                    p_lvl = int(float(var_req[vars4type[0]].get("ml").lstrip("p")))
                    cmd_add, vars4type = Preprocess_ERA5_data.get_cdo_op4ml(p_lvl, vars4type)
                else:
                    cmd_add = None
                    vars4type_aux = vars4type

                search_patt = os.path.join(dirin_now, "{0}{1}*_{2}.grb".format(year_str, month_str, vartype))

                logger.info("%{0}: Serach for grib-files under '{1}' for year {2} and month {3}"
                            .format(method, dirin_now, year_str, month_str))
                grb_files = glob.glob(search_patt)

                nfiles, nfiles_exp = len(grb_files), pd.Period("{0}-{1}".format(year_str, month_str)).days_in_month*24

                if not nfiles == nfiles_exp:
                    err = "%{0}: Found {1:d} grib-files with search pattern '{2}'".format(method, nfiles, search_patt) \
                          + ", but {0:d} files found. Check data directory...".format(nfiles)
                    logger.critical(err)
                    raise FileNotFoundError(err)

                logger.info("%{0}: Start converting and slicing of data from {1:d} files found with pattern {2}..."
                            .format(method, nfiles, search_patt))

                # process each file individually and store resulting netCDF-file in temp-directory
                for grb_file in grb_files:
                    tmp_file = os.path.join(dirout_tmp,
                                            "preprocess_{0}".format(os.path.basename(grb_file).replace("grb", "nc")))

                    cmd = "cdo -v --eccodes -f nc copy -selname,{0} -sellonlatbox,{1},{2},{3},{4} {5} {6}" \
                          .format(",".join(vars4type_aux), *lon_bounds, *lat_bounds, grb_file, tmp_file)

                    if cmd_add is not None:         # append cdo-command if required
                        cmd = cmd.replace("copy", "copy{0}".format(cmd_add))

                    nwarns = Preprocess_ERA5_data.run_cmd(cmd, logger, nwarns)
                    # check if nwarns was exceeded
                    if nwarns >= nmax_warn: return -1

                # Merge all files in temp-directory of current vartype to yield monthly data files
                logger.info("%{0}: Start merging all files from vartype '{1}', i.e. '*{1}.nc'.".format(method, vartype))
                cmd = "cdo -v -mergetime -selname,{0} {1} {2}"\
                      .format(",".join(vars4type), os.path.join(dirout_tmp, "*{0}.nc".format(vartype)),
                              os.path.join(dirout_tmp, "preprocess_tmp_{0}.nc".format(vartype)))

                nwarns = Preprocess_ERA5_data.run_cmd(cmd, logger, nwarns, labort=True)
                # check if previous merging failed
                if nwarns == -1: return nwarns

            # Now, merge the temp-files to get the final output file
            cmd = "cdo -v -merge {0} {1}".format(os.path.join(dirout_tmp, "preprocess_tmp_*.nc"), dest_file)
            logger.info("%{0}: Final merge of temp-files to '{1}'.".format(method, dest_file))
            # clean temp-directory
            logger.info("%{0}: Clean temp-directory '{1}'.".format(method, dirout_tmp))
            shutil.rmtree(dirout_tmp)

            nwarns = Preprocess_ERA5_data.run_cmd(cmd, logger, nwarns, labort=True)

        return nwarns

    @staticmethod
    def get_cdo_op4ml(p_lvl, variables):
        """
        Modify cdo-command for performing pressure interpolation and renaming variables in data file.
        :param p_lvl: pressure level onto which data is interpolated [Pa]
        :param variables: list of variable names subject to interpolation
        :param replace_str: sub-string in original cdo command which can be used for manipulation (must be a substring)
        :return: updated cdo-command and list of variable names
        """

        varrename = ["{0},{0}_{1:d}".format(var, int(p_lvl/100.)) for var in variables]
        cdo_add = " -chname,{0} -ml2pl,{1:d} ".format(",".join(varrename), p_lvl)
        # also adjust variable names incl. vars4type for latter variable selection
        vars4type = ["{0}_{1:d}".format(var, int(p_lvl/100.)) for var in variables]

        return cdo_add, vars4type

    @staticmethod
    def run_cmd(cmd: str, logger: logging.Logger, nwarns: int, labort: bool = False):
        """
        Run command in separated shell
        :param cmd: command to run
        :param logger: logger instance
        :param nwarns: number of current warnings
        :param labort: Boolean if abortion will be triggered (i.e. return nwarns = -1)
        :return: updated nwarns
        """
        method = Preprocess_ERA5_data.run_cmd.__name__

        nwarns_loc = nwarns
        try:
            _ = sp.check_output(cmd, stderr=sp.STDOUT, shell=True)
            logger.info("%{0}: Command '{1}' ran successfully...".format(method, cmd))
        except sp.CalledProcessError as exc:
            logger.critical("%{0}: Failed to run the following command: {1}".format(method, cmd))
            logger.critical("%{0}: Return code: {1}, error message: {2}".format(method, exc.returncode,
                                                                                exc.output))
            if labort:
                nwarns_loc = -1
            else:
                nwarns_loc += 1

        return nwarns_loc


    @staticmethod
    def check_dirin(dirin: str, years: str_or_List, months: List):
        """
        Checks if data directories for all years exist
        :param dirin: path to basic data directory under which files are located
        :param years: years for which data is requested
        :param months: months for which data is requested
        :return: status
        """
        method = Preprocess_ERA5_data.check_dirin.__name__

        years = list(years)
        # basic sanity checks
        assert isinstance(dirin, str), "%{0}: Parsed dirin must be a string, but is of type '{1}'".format(method,
                                                                                                          type(dirin))
        if not all(isinstance(yr, int) for yr in years):
            raise ValueError("%{0}: Passed years must be a list of integers.".format(method))

        if not os.path.isdir(dirin):
            raise NotADirectoryError("%{0}: Input directory for ERA%-data '{1}' does not exist.".format(method, dirin))

        # check if at least one ERA5-datafile is present
        for year in years:
            for month in months:
                yr_str, mm_str = str(year), "{0:02d}".format(int(month))
                dirin_now = os.path.join(dirin, yr_str, mm_str)
                print("dirin_base: {0}, dirin_now: {1}".format(dirin, dirin_now))
                f = glob.iglob(os.path.join(dirin_now, "{0}{1}*.grb".format(yr_str, mm_str)))

                _exhausted = object()
                if next(f, _exhausted) is _exhausted:
                    raise FileNotFoundError("%{0}: Could not find any ERA5 file for {1}/{2} under '{3}'"
                                            .format(method, yr_str, mm_str, dirin_now))
        return dirin, years

    @staticmethod
    def get_months(months):

        method = Preprocess_ERA5_data.get_months.__name__

        assert isinstance(months, get_args(str_or_List)), \
               "%{0}: months must be either a list of months or a (known) string.".format(method)

        if isinstance(months, list):
            month_list = [int(x) for x in months]
            if not np.all([isw(x, [1,12]) for x in month_list]):
                for i, x in enumerate(month_list):
                    print(" Value {0:d}: {1}".format(i, x))
                raise ValueError("%{0}: Not all elements of months can serve as month-integers".format(method) +
                                 "(only values between 1 and 12 can be passed).")
        elif months == "DJF":
            month_list = [1, 2, 12]
        elif months == "MAM":
            month_list = [3, 4, 5]
        elif months == "JJA":
            month_list = [6, 7, 8]
        elif months == "SON":
            month_list = [9, 10, 11]
        elif months == "all":
            month_list = list(np.range(1, 13))
        else:
            raise ValueError("%{0}: months-argument cannot be converted to list of months (see doc-string)"
                             .format(method))

        return month_list

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
        varlist = str(sp.check_output(cmd, stderr=sp.STDOUT, shell=True)).lstrip("b'").rstrip("'").replace(" ", "")
        varlist = varlist.split("\\n")

        stat = check_str_in_list(varlist, varnames, labort=labort)

        return stat












