""" 
Class to retrieve and handle meta-data
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2020-xx-xx"

import os
import sys
import time
import xarray as xr
import numpy as np
import json
from general_utils import is_integer, add_str_to_path, check_str_in_list, isw
from netcdf_datahandling import GeoSubdomain


class MetaData:
    """
     Class for handling, storing and retrieving meta-data
    """

    def __init__(self, json_file=None, suffix_indir=None, exp_id=None, data_filename=None, tar_dom=None,
                 variables=None):
        """
        Initailizes MetaData instance by reading a corresponding json-file or by handling arguments of the
        Preprocessing step (i.e. exemplary input file, slices defining region of interest, input variables)
        :param json_file: already existing json-file with metadata, if present the following arguments are not needed
        :param suffix_indir: suffix of directory where processed data is stored for running the models
        :param exp_id: experiment identifier
        :param data_filename: name of netCDF-file serving as base for metadata retrieval
        :param tar_dom: class instance for defining target domain
        :param variables: predictor variables
        """

        method_name = MetaData.__init__.__name__ + " of Class " + MetaData.__name__

        if not json_file is None:
            print(json_file)
            print(type(json_file))
            MetaData.get_metadata_from_file(self, json_file)

        else:
            # No dictionary from json-file available, all other arguments have to set
            if not suffix_indir:
                raise TypeError(method_name + ": 'suffix_indir'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(suffix_indir, str):
                    raise TypeError(method_name + ": 'suffix_indir'-argument must be a string.")

            if not exp_id:
                raise TypeError(method_name + ": 'exp_id'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(exp_id, str):
                    raise TypeError(method_name + ": 'exp_id'-argument must be a string.")

            if not data_filename:
                raise TypeError(method_name + ": 'data_filename'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(data_filename, str):
                    raise TypeError(method_name + ": 'data_filename'-argument must be a string.")

            if not tar_dom:
                raise TypeError(method_name + ": 'slices'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(tar_dom, GeoSubdomain):
                    raise TypeError(method_name + ": 'slices'-argument must be a dictionary.")

            if not variables:
                raise TypeError(method_name + ": 'variables'-argument is required if 'json_file' is not passed.")
            else:
                if not isinstance(variables, list):
                    raise TypeError(method_name + ": 'variables'-argument must be a list.")

            MetaData.get_and_set_metadata_from_file(self, suffix_indir, exp_id, data_filename, tar_dom, variables)

            MetaData.write_metadata_to_file(self)

    def get_and_set_metadata_from_file(self, suffix_indir, exp_id, datafile_name, tar_dom, variables):
        """
        Retrieves several meta data from an ERA5 netCDF-file and sets corresponding class instance attributes.
        Besides, the name of the experiment directory is constructed following the naming convention (see below)

        Naming convention:
        [model_base]_Y[yyyy]to[yyyy]M[mm]to[mm]-[nx]x[ny]-[nnnn]N[eeee]E-[var1]_[var2]_(...)_[varN]
        ---------------- Given ----------------|---------------- Created dynamically --------------

        Note that the model-base as well as the date-identifiers must already be included in target_dir_in.
        :param suffix_indir: Path to directory where the preprocessed data will be stored
        :param exp_id: Experimental identifier
        :param datafile_name: ERA 5 reanalysis netCDF file
        :param tar_dom: class instance for defining target domain
        :param variables: meteorological variables to be processed during preprocessing
        :return: A class instance with the following attributes set:
                 * varnames       : name of variables to be processed
                 * nx             : number of grid points of sliced region in zonal direction
                 * ny             : same as nx, but in meridional direction
                 * sw_c           : south-west corner [lat,lon] coordinates of region of interest
                 * lat            : latitude coordinates of grid points (on a rectangular grid)
                 * lon            : longitude coordinates of grid points (on a rectangular grid)
                 * expname        : name of target experiment directory following anming convention (see above)
                 * expdir         : basename of experiment diretory
                 * exp_id         : experimental identifier
                 * status         : indicate if new metadata was set up or if it's pre-exsting (left empty here!)
        """

        method_name = MetaData.get_and_set_metadata_from_file.__name__ + " of Class " + MetaData.__name__

        if not suffix_indir:
            raise ValueError(method_name + ": suffix_indir must be a non-empty path.")

        # retrieve required information from file 
        flag_coords = ["N", "E"]

        print("Retrieve metadata based on file: '" + datafile_name + "'")
        aux_data = tar_dom.get_data_dom(datafile_name, variables)
        aux_coords = aux_data.coords

        self.lat, self.lon = aux_coords["lat"].values, aux_coords["lon"].values
        self.nx, self.ny = np.shape(self.lon)[0], np.shape(self.lat)[0]
        sw_c = tar_dom.sw_c
        # switch to [-180..180]-range for convenince before setting attribute
        if sw_c[1] > 180.:
            sw_c[1] -= 360.
        self.sw_c = sw_c
        # switch coordinate-flage for longitudes to avoid negative values appearing in expdir-name
        if sw_c[0] < 0.:
            sw_c[0] = np.abs(sw_c[0])
            flag_coords[0] = "S"
        if sw_c[1] < 0:
            sw_c[1] = np.abs(sw_c[1])
            flag_coords[1] = "W"
        self.varnames = variables
        nvar = len(variables)

        # Now start constructing expdir-string
        # splitting has to be done in order to retrieve the expname-suffix (and the year if required)
        path_parts = os.path.split(suffix_indir.rstrip("/"))

        if is_integer(path_parts[1]):
            _ = path_parts[1]
            path_parts = os.path.split(path_parts[0].rstrip("/"))

        expdir, expname = path_parts[0], path_parts[1]

        # extend expdir_in successively (splitted up for better readability)
        expname += "-" + str(self.nx) + "x" + str(self.ny)
        expname += "-" + (("{0: 05.2f}" + flag_coords[0] + "{1:05.2f}" + flag_coords[1]).format(*sw_c)).strip().replace(
            ".", "") + "-"

        # reduced for-loop length as last variable-name is not followed by an underscore (see above)
        for i in range(nvar - 1):
            expname += variables[i] + "_"
        expname += variables[nvar - 1]

        self.expname = expname
        self.expdir = expdir
        self.exp_id = exp_id
        self.status = ""  # uninitialized (is set when metadata is written/compared to/with json-file,
        # see write_metadata_to_file-method)

    # ML 2020/04/24 E         

    def write_metadata_to_file(self, dest_dir=None):
        """
        Writes meta data stored as attributes in the class instance to metadata.json.
        If dest_dir is None, the destination directory is constructed based on the attributes expdir and expname.
        :param dest_dir: path to directory where to store metadata.json
        :return: -
        """

        method_name = MetaData.write_metadata_to_file.__name__ + " of Class " + MetaData.__name__
        # actual work:
        meta_dict = {"expname": self.expname, "expdir": self.expdir, "exp_id": self.exp_id, "sw_corner_frame": {
            "lat": np.around(self.sw_c[0], decimals=2),
            "lon": np.around(self.sw_c[1], decimals=2)},
                     "coordinates": {
                         "lat": np.around(self.lat, decimals=2).tolist(),
                         "lon": np.around(self.lon, decimals=2).tolist()},
                     "frame_size": {
                         "nx": int(self.nx),
                         "ny": int(self.ny)},
                     "variables": []}

        for i in range(len(self.varnames)):
            # print(self.varnames[i])
            meta_dict["variables"].append({"var" + str(i + 1): self.varnames[i]})
        # create directory if required
        if dest_dir is None:
            dest_dir = os.path.join(self.expdir, self.expname)
        if not os.path.exists(dest_dir):
            print("Created experiment directory: '" + self.expdir + "'")
            os.makedirs(dest_dir, exist_ok=True)

        meta_fname = os.path.join(dest_dir, "metadata.json")

        if os.path.exists(meta_fname):  # check if a metadata-file already exists and check its content
            print(method_name + ": json-file ('" + meta_fname + "' already exists. Its content will be checked...")
            self.status = "old"  # set status to old in order to prevent repeated modification of shell-/Batch-scripts
            with open(meta_fname, 'r') as js_file:
                dict_dupl = json.load(js_file)

                if dict_dupl != meta_dict:
                    meta_fname_dbg = os.path.join(dest_dir, "metadata_debug.json")
                    print("%{0}: Already existing metadata (see '{1}') do not fit data being processed right now " +
                          "(see '{2}'). Ensure a common data base.".format(method_name, meta_fname, meta_fname_dbg))
                    with open(meta_fname_dbg, 'w') as js_file_new:
                        json.dump(meta_dict, js_file_new)
                    raise ValueError
                else:  # do not need to do anything
                    pass
        else:
            # write dictionary to file
            print(method_name + ": Write dictionary to json-file: '" + meta_fname + "'")
            with open(meta_fname, 'w') as js_file:
                json.dump(meta_dict, js_file)
            self.status = "new"  # set status to new in order to trigger modification of shell-/Batch-scripts

    def get_metadata_from_file(self, js_file):
        """
        :param js_file: json file from which to retrieve the meta data
        :return: A class instance with the following attributes set:
                 * varnames       : name of variables to be processed
                 * nx             : number of grid points of sliced region in zonal direction
                 * ny             : same as nx, but in meridional direction
                 * sw_c           : south-west corner [lat,lon] coordinates of region of interest
                 * lat            : latitude coordinates of grid points (on a rectangular grid)
                 * lon            : longitude coordinates of grid points (on a rectangular grid)
                 * expname        : name of target experiment directory following naming convention (see above)
                 * expdir         : basename of experiment directory
                 * exp_id         : experimental identifier (if available!)
                 * status         : status to indicate if a new metadata is set-up or pre-existing (left empty here!)
        """

        with open(js_file) as js_file:
            dict_in = json.load(js_file)

            self.expdir = dict_in["expdir"]
            self.expname = dict_in["expname"]
            # check if exp_id is available (retained for ensuring backward compatilibity with
            # old meta data files without exp_id)
            if "exp_id" in dict_in:
                self.exp_id = dict_in["exp_id"]

            self.sw_c = [dict_in["sw_corner_frame"]["lat"], dict_in["sw_corner_frame"]["lon"]]
            self.lat = dict_in["coordinates"]["lat"]
            self.lon = dict_in["coordinates"]["lon"]

            self.nx = dict_in["frame_size"]["nx"]
            self.ny = dict_in["frame_size"]["ny"]
            # dict_in["variables"] is a list like [{var1: varname1},{var2: varname2},...]
            list_of_dict_aux = dict_in["variables"]
            # iterate through the list with an integer ivar
            # note: the naming of the variables starts with var1, thus add 1 to the iterator
            self.variables = [list_of_dict_aux[ivar]["var" + str(ivar + 1)] for ivar in range(len(list_of_dict_aux))]

    def get_coord_array(self):
        """
        Returns data arrays of latitudes and longitudes
        :return lats_da: data array of latitudes
        :return lons_da: data array of longitudes
        """
        method = MetaData.get_coord_array.__name__

        if not hasattr(self.lat) or not hasattr(self.lon):
            raise AttributeError("%{0}: lat and lon are still not set.".format(method))

        lats_da = xr.DataArray(self.lat, coords={"lat": self.lat}, dims="lat",
                               attrs={"units": "degrees_east"})
        lons_da = xr.DataArray(self.lon, coords={"lon": self.lon}, dims="lon",
                               attrs={"units": "degrees_north"})

        return lats_da, lons_da

    def write_dirs_to_batch_scripts(self, batch_script):
        """
        Method for automatic extension of path variables in Batch scripts by the experiment directory which is saved
        in the expname-attribute of the class instance
        :param batch_script: Batch script whose (known) path variables (defined by paths_to_mod below) will be expanded
                             by the expname-attribute of the class instance at hand
        :return: modified Batch script
        Note ML, 2021-03-22: Deprecated !!!
        """

        paths_to_mod = ["source_dir=", "destination_dir=", "checkpoint_dir=", "results_dir="]

        # For backward compability:
        # Check if exp_id (if present) needs to be added to batch_script in order to access the file
        if hasattr(self, "exp_id"):
            sep_idx = batch_script.index(".sh")
            batch_script = batch_script[:sep_idx] + "_" + self.exp_id + batch_script[sep_idx:]

        with open(batch_script, 'r') as file:
            data = file.readlines()

        nlines = len(data)
        matched_lines = [iline for iline in range(nlines) if any(str_id in data[iline] for str_id in paths_to_mod)]

        for i in matched_lines:
            data[i] = add_str_to_path(data[i], self.expname)

        with open(batch_script, 'w') as file:
            file.writelines(data)

    @staticmethod
    def write_destdir_jsontmp(dest_dir, tmp_dir=None):
        """
        Writes dest_dir to temporary json-file (temp.json) stored in the current working directory.
        To be executed by Master node only in parallel mode.
        :param dest_dir: path to destination directory
        :param tmp_dir: directory where to store temp.json (optional)
        :return: -
        """

        if not tmp_dir:
            tmp_dir = os.getcwd()

        file_tmp = os.path.join(tmp_dir, "temp.json")
        dict_tmp = {"destination_dir": dest_dir}

        with open(file_tmp, "w") as js_file:
            print("Save destination_dir-variable in temporary json-file: '" + file_tmp + "'")
            json.dump(dict_tmp, js_file)

    @staticmethod
    def get_destdir_jsontmp(tmp_dir=None):
        """
        Retrieves path destination directory from temp.json file (to be created by write_destdir_jsontmp-method)
        :param tmp_dir: directory where temp.json is stored (optional). If not provided, the working directory is used.
        :return: string containing the path to the destination directory
        """

        method_name = MetaData.get_destdir_jsontmp.__name__ + " of Class " + MetaData.__name__

        if not tmp_dir:
            tmp_dir = os.getcwd()

        file_tmp = os.path.join(tmp_dir, "temp.json")

        try:
            with open(file_tmp, "r") as js_file:
                dict_tmp = json.load(js_file)
        except Exception as err:
            print("%{0}: Could not open requested json-file '{1}'".format(method_name, file_tmp))
            print(str(err))
            sys.exit(1)

        if not "destination_dir" in dict_tmp.keys():
            raise NotADirectoryError("%{0}: Could not find 'destination_dir' in dictionary obtained from {1}"
                                     .format(method_name, file_tmp))
        else:
            return dict_tmp.get("destination_dir")

    @staticmethod
    def wait_for_jsontmp(tmp_dir=None, waittime=10, delay=0.5):
        """
        Waits until temp.json-file becomes available
        :param tmp_dir: directory where temp.json is stored (optional). If not provided, the working directory is used.
        :param waittime: time to wait in seconds (default: 10 s)
        :param delay: length of checkin intervall (default: 0.5 s)
        :return: -
        """

        method_name = MetaData.wait_for_jsontmp.__name__ + " of Class " + MetaData.__name__

        if not tmp_dir:
            tmp_dir = os.getcwd()

        file_tmp = os.path.join(tmp_dir, "temp.json")

        counter_max = waittime / delay
        counter = 0
        status = "not_ok"

        while counter <= counter_max:
            if os.path.isfile(file_tmp):
                status = "ok"
                break
            else:
                time.sleep(delay)

            counter += 1

        if status != "ok":
            raise FileNotFoundError("%{0}: '{1}' does not exist after waiting for {2:5.2f} sec."
                                    .format(method_name, file_tmp, waittime))

# ----------------------------------- end of class MetaData -----------------------------------
