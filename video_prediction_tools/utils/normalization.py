# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
 Class for normalizing data. The statistical data for normalization (minimum, maximum, average, standard deviation etc.) is expected to be available from a statistics-dictionary
 created with the calc_data_stat-class (see 'process_netCDF_v2.py'.
"""

from general_utils import get_unique_vars
from statistics import Calc_data_stat
import numpy as np

class Norm_data:

    ### set known norms and the requested statistics (to be retrieved from statistics.json) here ###
    known_norms = {}
    known_norms["minmax"] = ["min", "max"]
    known_norms["znorm"] = ["avg", "sigma"]

    def __init__(self, varnames):
        """Initialize the instance by setting the variable names to be handled and the status (for sanity checks only) as attributes."""
        varnames_uni, _, nvars = get_unique_vars(varnames)

        self.varnames = varnames_uni
        self.status_ok = False

    def check_and_set_norm(self, stat_dict, norm):
        """
         Checks if the statistics-dictionary provides the required data for selected normalization method and expands the instance's attributes accordingly.
         Example: minmax-normalization requires the minimum and maximum value of a variable named var1.
                 If the requested values are provided by the statistics-dictionary, the instance gets the attributes 'var1min' and 'var1max',respectively.
        """

        # some sanity checks
        if not norm in self.known_norms.keys():  # valid normalization requested?
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '" + norm + "' is unknown.")

        if not all(items in stat_dict for items in self.varnames):  # all variables found in dictionary?
            print("Keys in stat_dict:")
            print(stat_dict.keys())

            print("Requested variables:")
            print(self.varnames)
            raise ValueError("Could not find all requested variables in statistics dictionary.")

            # create all attributes for the instance
        for varname in self.varnames:
            for stat_name in self.known_norms[norm]:
                # setattr(self,varname+stat_name,stat_dict[varname][0][stat_name])
                setattr(self, varname + stat_name, Calc_data_stat.get_stat_vars(stat_dict, stat_name, varname))

        self.status_ok = True  # set status for normalization -> ready

    def norm_var(self, data, varname, norm):
        """
         Performs given normalization on input data (given that the instance is already set up)
        """

        # some sanity checks
        if not self.status_ok: raise ValueError(
            "Norm_data-instance needs to be initialized and checked first.")  # status ready?

        if not norm in self.known_norms.keys():  # valid normalization requested?
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '" + norm + "' is unknown.")

        # do the normalization and return
        if norm == "minmax":
            return ((data[...] - getattr(self, varname + "min")) / (
                        getattr(self, varname + "max") - getattr(self, varname + "min")))
        elif norm == "znorm":
            return ((data[...] - getattr(self, varname + "avg")) / getattr(self, varname + "sigma") ** 2)

    def denorm_var(self, data, varname, norm):
        """
         Performs given denormalization on input data (given that the instance is already set up), i.e. inverse method to norm_var
        """

        # some sanity checks
        if not self.status_ok: raise ValueError(
            "Norm_data-instance needs to be initialized and checked first.")  # status ready?

        if not norm in self.known_norms.keys():  # valid normalization requested?
            print("Please select one of the following known normalizations: ")
            for norm_avail in self.known_norms.keys():
                print(norm_avail)
            raise ValueError("Passed normalization '" + norm + "' is unknown.")

        # do the denormalization and return
        if norm == "minmax":
            return (data[...] * (getattr(self, varname + "max") - getattr(self, varname + "min")) + getattr(self,
                                                                                                            varname + "min"))
        elif norm == "znorm":
            return (data[...] * getattr(self, varname + "sigma") ** 2 + getattr(self, varname + "avg"))