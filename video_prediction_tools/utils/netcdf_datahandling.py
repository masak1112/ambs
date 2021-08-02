""" 
Classes to handle netCDF-data files and to extract gridded data on a subdomain 
(e.g. used for handling ERA5-reanalysis data)

Content: * get_era5_varatts (auxiliary function!)
         * NetcdfUtils
         * GeoSubdomain
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2021-03-26"

# import modules
import os
import numpy as np
import xarray as xr
from general_utils import is_integer, add_str_to_path, check_str_in_list, isw, provide_default

# auxiliary function that is not generic enough to be placed in NetcdfUtils


def get_era5_varatts(data_arr: xr.DataArray, name: str):
    """
    Writes longname and unit to data arrays given their name is known
    :param data_arr: the data array
    :param name: the name of the variable
    :return: data array with added attributes 'longname' and 'unit' if they are known
    """

    era5_varname_map = {"2t": "2m temperature", "t_850": "850 hPa temperature", "tcc": "total cloud cover",
                        "msl": "mean sealevel pressure", "10u": "10m u-wind", "10v": "10m v-wind"}
    era5_varunit_map = {"2t": "K", "t_850": "K", "tcc": "%",
                        "msl": "Pa", "10u": "m/s", "10v": "m/s"}

    name_splitted = name.split("_")
    if "fcst" in name:
        addstr = "from {0} model".format(name_splitted[1])
    elif "ref" in name:
        addstr = "from ERA5 reanalysis"
    else:
        addstr = ""

    longname = provide_default(era5_varname_map, name_splitted[0], -1)
    if longname == -1:
        pass
    else:
        data_arr.attrs["longname"] = "{0} {1}".format(longname, addstr)

    unit = provide_default(era5_varunit_map, name_splitted[0], -1)
    if unit == -1:
        pass
    else:
        data_arr.attrs["unit"] = unit

    return data_arr


class NetcdfUtils:
    """
    Class containing some auxiliary functions to check netCDf-files
    """

    def __init__(self, filename, lhold_data=False):
        self.filename = filename
        self.data = self.check_file()

        self.varlist = NetcdfUtils.list_vars(self)
        self.coords = NetcdfUtils.get_coords(self)
        self.attributes = None
        if not lhold_data:
            self.data = None

    def check_file(self):
        method = NetcdfUtils.check_file.__name__

        assert hasattr(self, "filename") is True, "%{0}: Class instance does not have a filename property." \
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
        method = NetcdfUtils.get_coords.__name__

        try:
            return self.data.coords
        except Exception:
            raise IOError("%{0}: Could not handle coordinates of netCDF-file '{1}'".format(method, self.filename))

    def list_vars(self):
        """
        Retrieves list all variables of file
        :return: varlist
        """
        method = NetcdfUtils.list_vars.__name__

        try:
            varlist = list(self.data.keys())
        except Exception:
            raise IOError("%{0}: Could not open {1}".format(method, self.filename))

        return varlist

    def var_in_file(self, varnames, labort=True):

        method = NetcdfUtils.var_in_file.__name__

        stat = check_str_in_list(self.varlist, varnames, labort=False)

        if not stat and labort:
            raise ValueError("%{0}: Could not find all varnames in netCDF-file".format(method))

        return stat

# ----------------------------------- end of class NetcdfUtils -----------------------------------


class GeoSubdomain(NetcdfUtils):
    """
    Class in order to define target domains onto geographically, gridded data
    (e.g. ERA5-data on a regular lat-/lon-grid). Inherits from class NetcdfUtils
    """

    def __init__(self, sw_corner, nyx, filename):
        """
        Initializes subdomain instance based on data from input file
        :param sw_corner: (lat,lon)-coordinate pair defining south-west corner in degree
        :param nyx: number of grid points in meridional (y) and zonal (x)  direction
        :param filename: the netCDF-file from which the subdomain should be initialized
        """

        method = GeoSubdomain.__init__.__name__ + " of Class " + GeoSubdomain.__name__
        # inherit from NetcdfUtils
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

        method = GeoSubdomain.handle_geocoords.__name__ + " of Class " + GeoSubdomain.__name__

        coords = self.coords
        try:
            self.lat, self.lon = coords["lat"], coords["lon"]
            self.nlat, self.nlon = np.shape(self.lat)[0], np.shape(self.lon)[0]
            self.dy, self.dx = (self.lat[1] - self.lat[0]).values, (self.lon[1] - self.lon[0]).values
            self.lcyclic = (self.nlon == np.around(360. / self.dx))

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

        method = GeoSubdomain.get_dom_indices.__name__ + " of Class " + GeoSubdomain.__name__

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
        if not isw(ne_c_ind[0], [0, self.nlat - 1]):
            raise ValueError("%{0}: Desired domain exceeds spatial data coverage in meridional direction."
                             .format(method))
        else:
            lat_slices = [np.minimum(sw_c_ind[0], ne_c_ind[0]), np.maximum(sw_c_ind[0], ne_c_ind[0])]

        if not isw(ne_c_ind[1], [0, self.nlon - 1]):
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

        # readjust sw_c in case that axis are negatively oriented
        if self.dy < 0.:
            sw_c[0] -= self.dy
        if self.dx < 0:
            sw_c[1] -= self.dx

        sw_c = [coord.values for coord in sw_c]

        return lat_slices, lon_slices, sw_c

    def get_data_dom(self, filename, variables):
        """
        Performs slicing on data from datafile and cuts dataset to variables of interest
        :param filename: the netCDF-file to handle
        :param variables: list of variables to retrieve
        :return: sliced dataset with variables of interest
        """

        method = GeoSubdomain.get_data_dom.__name__ + " of Class " + GeoSubdomain.__name__

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

        method = GeoSubdomain.handle_data_cross.__name__ + " of Class " + GeoSubdomain.__name__

        lat_slices, lon_slices = self.lat_slices, self.lon_slices
        try:
            data_sub1 = data.isel(lat=slice(lat_slices[0], lat_slices[1]), lon=slice(lon_slices[0], self.nlon))
            data_sub = data_sub1.merge(data.isel(lat=slice(lat_slices[0], lat_slices[1]),
                                                 lon=slice(0, lon_slices[1])))
        except Exception as err:
            print("%{0}: Something went wrong when slicing data across cyclic lateral boundary.".format(method))
            raise err

        return data_sub

# ----------------------------------- end of class GeoSubdomain -----------------------------------
