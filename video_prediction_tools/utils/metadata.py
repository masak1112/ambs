""" 
Class to retrieve and handle meta-data
"""

import os
import sys
import time
import numpy as np
import xarray as xr
import json
from netCDF4 import Dataset
from general_utils import is_integer, add_str_to_path, check_str_in_list, isw


class MetaData:
    """
     Class for handling, storing and retrieving meta-data
    """

    def __init__(self, json_file=None, suffix_indir=None, exp_id=None, data_filename=None, tar_dom=None, variables=None):
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
                if not isinstance(tar_dom, Geo_subdomain):
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

        if not suffix_indir: raise ValueError(method_name + ": suffix_indir must be a non-empty path.")

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
            year = path_parts[1]
            path_parts = os.path.split(path_parts[0].rstrip("/"))
        else:
            year = ""

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
        self.status = ""    # uninitialized (is set when metadata is written/compared to/with json-file,
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
                    with open(meta_fname_dbg, 'w') as js_file:
                        json.dump(meta_dict, js_file)
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
        except:
            print(method_name + ": Could not open requested json-file '" + file_tmp + "'")
            sys.exit(1)

        if not "destination_dir" in dict_tmp.keys():
            raise Exception(method_name + ": Could not find 'destination_dir' in dictionary obtained from " + file_tmp)
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


class Netcdf_utils:
    """
    Class containing some auxiliary functions to check netCDf-files
    """

    def __init__(self, filename, lhold_data=False):
        self.filename = filename
        self.data = self.check_file()

        self.varlist = Netcdf_utils.list_vars(self)
        self.coords = Netcdf_utils.get_coords(self)
        self.attributes = None
        if not lhold_data:
            self.data = None

    def check_file(self):
        method = Netcdf_utils.check_file.__name__

        assert hasattr(self, "filename") is True, "%{0}: Class instance does not have a filename property."\
                                                  .format(method)

        if not isinstance(self.filename, str):
            raise ValueError("%{0}: filename property must be a path-string".format(method))

        if not self.filename.endswith(".nc"):
            raise ValueError("%{0}: Passed filename must be a netCDF-file".format(method))

        if not os.path.isfile(self.filename):
            raise FileNotFoundError("%{0}: Could not find passed filename '{1}'".format(method, self.filename))

        try:
            with xr.open_dataset(self.filename) as dfile:
                data = dfile
        except Exception:
            raise IOError("%{0}: Error while reading data from netCDF-file {1}".format(method, self.filename))

        return data

    def get_coords(self):
        """
        Retrive coordinates from Dataset of netCDF-file
        :return coords: dictionary of coordinates from netCDf-file
        """
        method = Netcdf_utils.get_coords.__name__

        try:
            return self.data.coords
        except Exception:
            raise IOError("%{0}: Could not handle coordinates of netCDF-file '{1}'".format(method, self.filename))

    def list_vars(self):
        """
        Retrieves list all variables of file
        :return: varlist
        """
        method = Netcdf_utils.list_vars.__name__

        try:
            varlist = list(self.data.keys())
        except Exception:
            raise IOError("%{0}: Could not open {1}".format(method, self.filename))

        return varlist

    def var_in_file(self, varnames, labort=True):

        method = Netcdf_utils.var_in_file.__name__

        stat = check_str_in_list(self.varlist, varnames, labort=False)

        if not stat and labort:
            raise ValueError("%{0}: Could not find all varnames in netCDF-file".method(method))

        return stat

# ----------------------------------- end of class NetCDF_utils -----------------------------------

class Geo_subdomain(Netcdf_utils):
    """
    Class in order to define target domains onto geographically, gridded data (e.g. ERA5-data on a regular lat-/lon-grid
    """

    def __init__(self, sw_corner, nyx, filename):

        method = Geo_subdomain.__init__.__name__ + " of Class " + Geo_subdomain.__name__
        # inherit from Netcdf_utils
        super().__init__(filename)

        self.base_file = filename
        self.handle_geocoords()
        self.lat_slices, self.lon_slices, self.sw_c = self.get_dom_indices(sw_corner, nyx)

    def handle_geocoords(self):
        """
        Retrieve geographical coordinates named lat and lon from coords-dictionary and sets some key attributes
        :return: class instance with the following attributes:
                 * lat, lon : latitude and longitude values
                 * nlat, nlon: number of grid points in meridional and zonal direction
                 * dy, dx : grid spacing in meridional and zonal direction
                 * lcyclic: flag if data is cyclic in zonal direction
        """

        method = Geo_subdomain.handle_geocoords.__name__ + " of Class " + Geo_subdomain.__name__

        coords = self.coords
        try:
            self.lat, self.lon = coords["lat"], coords["lon"]
            self.nlat, self.nlon = np.shape(self.lat)[0], np.shape(self.lon)[0]
            self.dy, self.dx = (self.lat[1]- self.lat[0]).values, (self.lon[1]- self.lon[0]).values
            self.lcyclic = (self.nlon == np.around(360./self.dx))

        except Exception as err:
            print("%{0}: Could not retrieve geographical coordinates from datafile '{1}'".format(method, self.filename))
            raise err

    def get_dom_indices(self, sw_c, nyx):
        """
        Get indices to perform spatial slicing on data.
        :param sw_c: (lat,lon)-coordinate pair for south-west corner of target domain
        :param nyx: number of grid points of target domain in meridional and zonal direction
        :return lon_slices: tuple of indices in longitude direction, i.e. [lon_s, lon_e]
        :return lat_slices: tuple of indices in latitude direction, i.e. [lat_s, lat_e]
        :return sw_c: (lat,lon)-coordinate pair of true south-west corner of target domain
        """

        method = Geo_subdomain.get_dom_indices.__name__ + " of Class " + Geo_subdomain.__name__

        # sainty check on the method-arguments
        if np.shape(sw_c)[0] != 2:
            raise ValueError("%{0}: The south-west corner of the domain must be given as a coordinate pair (lat,lon)."
                             .format(method))

        if np.shape(nyx)[0] != 2:
            raise ValueError("%{0}: The number of indices of the target domain must be a (ny,nx)-pair of integers."
                             .format(method))
        else:
            for ni in nyx:
                if not isinstance(ni, int):
                    raise ValueError("%{0}: element '{1}' of -nyw is not an integer.".format(method, str(ni)))
                if not ni > 1:
                    raise ValueError("%{0}: elements of -nyw must be larger than 1.".format(method))
            # change sign for negatively oriented geographical axis
            if self.dy < 0.:
                nyx[0] = -np.abs(nyx[0])
                sw_c[0] += self.dy
            if self.dx < 0:
                nyx[1] = -np.abs(nyx[1])
                sw_c[1] += self.dx

        # check if south-west corner is inside data domain
        lat_intv = [np.amin(self.lat.values), np.amax(self.lat.values)]
        lon_intv = [np.amin(self.lon.values), np.amax(self.lon.values)]
        if not isw(sw_c[0], lat_intv):
            raise ValueError("%{0}: The meridional coordinate of the SW-corner at {1:5.2f}N".format(method, sw_c[0]) +
                             " is not part of the data domain (latitude range between {0:5.2f}N and {1:5.2f}N)."
                             .format(*lat_intv))

        if not isw(sw_c[1], lon_intv):
            raise ValueError("%{0}: The zonal coordinate of the SW-corner at {1:5.2f}E".format(method, sw_c[1]) +
                             " is not part of the data domain (longitude range between {0:5.2f}E and {1:5.2f}E)."
                             .format(*lon_intv))

        sw_c = [self.lat.sel(lat=sw_c[0], method='nearest'), self.lon.sel(lon=sw_c[1], method='nearest')]
        sw_c_ind = [np.where(self.lat == sw_c[0].values)[0][0], np.where(self.lon == sw_c[1].values)[0][0]]

        ne_c_ind = np.asarray(sw_c_ind) + nyx

        # check index range
        if not isw(ne_c_ind[0], [0, self.nlat-1]):
            raise ValueError("%{0}: Desired domain exceeds spatial data coverage in meridional direction."
                             .format(method))
        else:
            lat_slices = [np.minimum(sw_c_ind[0], ne_c_ind[0]), np.maximum(sw_c_ind[0], ne_c_ind[0])]

        if not isw(ne_c_ind[1], [0, self.nlon-1]):
            if not self.lcyclic:
                print("%{0}: Requested end index: {1:d}, Maximum end index: {2:d}".format(method, ne_c_ind[1],
                                                                                          self.nlon - 1))
                raise ValueError("%{0}: Desired domain exceeds spatial data coverage in zonal direction."
                                 .format(method))
            else:
                # readjust indices and switch order to trigger correct slicing in get_data_dom-method
                ne_c_ind[1] = np.abs(ne_c_ind[1] - self.nlon)
                lon_slices = [np.maximum(sw_c_ind[1], ne_c_ind[1]), np.minimum(sw_c_ind[1], ne_c_ind[1])]
        else:
            lon_slices = [np.minimum(sw_c_ind[1], ne_c_ind[1]), np.maximum(sw_c_ind[1], ne_c_ind[1])]

        sw_c = [coord.values for coord in sw_c]

        return lat_slices, lon_slices, sw_c

    def get_data_dom(self, filename, variables):
        """
        Performs slicing on data from datafile and cuts dataset to variables of interest
        :param filename: the netCDF-file to handle
        :param variables: list of variables to retrieve
        :return: sliced dataset with variables of interest
        """

        method = Geo_subdomain.get_data_dom.__name__ + " of Class " + Geo_subdomain.__name__

        if not os.path.isfile(filename):
            raise FileNotFoundError("%{0}: Could not find datafile '{1}'".format(method, filename))

        lat_slices, lon_slices = self.lat_slices, self.lon_slices
        lcross_zonal = lon_slices[0] > lon_slices[1]

        try:
            with xr.open_dataset(filename) as dfile:
                if not lcross_zonal:
                    data_sub = dfile.isel(lat=slice(lat_slices[0], lat_slices[1]),
                                          lon=slice(lon_slices[0], lon_slices[1]))
                else:
                    data_sub = self.handle_data_cross(dfile)
        except Exception as err:
            print("%{0}: Could not slice data from file '{1}'".format(method, filename))
            raise err

        _ = self.var_in_file(variables)

        try:
            data_sub = data_sub[variables]
        except Exception as err:
            print("%{0}: Could not retrieve all of the following variables from '{1}': {2}".format(method, filename,
                                                                                                   ",".join(variables)))
            raise err

        return data_sub

    def handle_data_cross(self, data):
        """
        Handles data on target domain that crosses the cyclic boundary in zonal direction
        :param data: the data-object
        :return: the sliced dataset which has been merged to handle the cyclic boundary
        """

        method = Geo_subdomain.get_data_cross.__name__ + " of Class " + Geo_subdomain.__name__

        lat_slices, lon_slices = self.lat_slices, self.lon_slices
        try:
            data_sub1 = data.isel(lat=slice(lat_slices[0], lat_slices[1]), lon=slice(lon_slices[0], self.nlon))
            data_sub = data_sub1.merge(data.isel(lat=slice(lat_slices[0], lat_slices[1]),
                                                 lon=slice(0, lon_slices[1])))
        except Exception as err:
            print("%{0}: Something went wrong when slicing data across cyclic lateral boundary.".format(method))
            raise err

        return data_sub
