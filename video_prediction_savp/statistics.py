"""
Class to calculate and save statistics in the ambs-workflow parallelized by PyStager.
In addition to savin the statistics in json-files, it also comprises methods to read those files.
"""

import os
import sys
import time
import numpy as np
import json
from helper import get_unique_vars

class Calc_data_stat:

    def __init__(self, nvars):
        """
         Initializes the instance for later use, i.e. initializes attributes with expected shape
        """
        self.stat_dict = {}
        self.varmin = np.full((nvars, 1), np.nan)  # avoid rank one-arrays
        self.varmax = np.full((nvars, 1), np.nan)
        self.varavg = np.zeros((nvars,
                                1))  # second dimension acts as placeholder for averaging on master node collecting json-files from slave nodes
        self.nfiles = [0]  # number of processed files
        self.mode = ""  # mode to distinguish between processing on slave and master nodes (sanity check)
        self.jsfiles = [""]  # list of processed json-files (master-mode only!)

    def acc_stat_loc(self, ivar, data):
        """
         Performs accumulation of all statistics while looping through all data files (i.e. updates the statistics) on slave nodes
        """
        if not self.mode:
            self.mode = "loc"
        elif self.mode == "master":
            raise ValueError("Cannot switch to loc-mode during runtime...")
        else:
            pass

        self.varmin[ivar] = np.fmin(self.varmin[ivar], np.amin(data))
        self.varmax[ivar] = np.fmax(self.varmax[ivar], np.amax(data))
        self.varavg[ivar, 0] += np.average(
            data)  # note that we sum the average -> readjustment required in the final step
        if (ivar == 0): self.nfiles[0] += 1

    def finalize_stat_loc(self, varnames):
        """
         Finalizes computation of statistics after going through all the data on slave nodes.
         Afterwards the statistics dictionary is ready for being written in a json-file.
        """

        if self.mode != "loc":
            raise ValueError("Object is not in loc-mode. Probably some master-method has been called previously.")

        if self.stat_dict: raise ValueError("Statistics dictionary is not empty.")

        vars_uni, varsind = np.unique(varnames, return_index=True)
        nvars = len(vars_uni)

        vars_uni, varsind, nvars = get_unique_vars(varnames)

        varmin, varmax, varavg = self.varmin[varsind], self.varmax[varsind], self.varavg[varsind, 0]

        for i in range(nvars):
            varavg[i] /= self.nfiles  # for adjusting the (summed) average

            self.stat_dict[vars_uni[i]] = []
            self.stat_dict[vars_uni[i]].append({
                'min': varmin[i, 0].tolist(),
                'max': varmax[i, 0].tolist(),
                'avg': varavg[i].tolist()
            })
        self.stat_dict["common_stat"] = [
            {"nfiles": self.nfiles[0]}]

    def acc_stat_master(self, file_dir, file_id):
        """
         Opens statistics-file (created by slave nodes) and accumulates its content.
        """

        if (int(file_id) <= 0): raise ValueError("Non-valid file_id passed.")

        if not self.mode:
            self.mode = "master"
        elif self.mode == "loc":
            raise ValueError("Cannot switch to master-mode during runtime...")
        else:
            pass

        # sanity check: check if dictionary is initialized with unique values only
        if self.stat_dict.keys() > set(self.stat_dict.keys()):
            raise ValueError("Initialized dictionary contains duplicates of variales. Need unique collection instead.")
        else:
            pass

        file_name = os.path.join(file_dir, "stat_{0:0=2d}.json".format(int(file_id)))

        if not file_name in self.jsfiles:
            print("Try to open: '" + file_name + "'")

            try:
                with open(file_name) as js_file:
                    dict_in = json.load(js_file)

                    # sanity check
                    if (len(dict_in.keys()) - 1 != len(self.varmin)):
                        raise ValueError(
                            "Different number of variables found in json-file '" + js_file + "' as expected from statistics object.")

                    self.varmin = np.fmin(self.varmin, Calc_data_stat.get_stat_allvars(dict_in, "min"))
                    self.varmax = np.fmax(self.varmax, Calc_data_stat.get_stat_allvars(dict_in, "max"))

                    if (np.all(self.varavg == 0.) or self.nfiles[0] == 0):
                        self.varavg = Calc_data_stat.get_stat_allvars(dict_in, "avg")
                        self.nfiles[0] = Calc_data_stat.get_common_stat(dict_in, "nfiles")
                        self.jsfiles[0] = file_name
                    else:
                        self.varavg = np.append(self.varavg, Calc_data_stat.get_stat_allvars(dict_in, "avg"), axis=1)
                        self.nfiles.append(Calc_data_stat.get_common_stat(dict_in, "nfiles"))
                        self.jsfiles.append(file_name)
            except IOError:
                print("Cannot handle statistics file '" + file_name + "' to be processed.")
            except ValueError:
                print("Cannot retireve all required statistics from '" + file_name + "'")
        else:
            print("Statistics file '" + file_name + "' has already been processed. Thus, just pass here...")
            pass

    def finalize_stat_master(self, vars_uni):
        """
         Performs final compuattion of statistics after accumulation from slave nodes.
        """
        if self.mode != "master":
            raise ValueError("Object is not in master-mode. Probably some loc-method has been called previously.")

        if len(vars_uni) > len(set(vars_uni)):
            raise ValueError("Input variable names are not unique.")

        nvars = len(vars_uni)
        n_jsfiles = len(self.nfiles)
        nfiles_all = np.sum(self.nfiles)
        avg_wgt = np.array(self.nfiles, dtype=float) / float(nfiles_all)

        varmin, varmax = self.varmin, self.varmax
        varavg = np.sum(np.multiply(self.varavg, avg_wgt), axis=1)  # calculate weighted average

        for i in range(nvars):
            self.stat_dict[vars_uni[i]] = []
            self.stat_dict[vars_uni[i]].append({
                'min': varmin[i, 0].tolist(),
                'max': varmax[i, 0].tolist(),
                'avg': varavg[i].tolist()
            })
        self.stat_dict["common_stat"] = [
            {"nfiles": int(nfiles_all),
             "jsfiles": self.jsfiles
             }]

    @staticmethod
    def get_stat_allvars(stat_dict, stat_name):
        """
         Unpacks statistics dictionary and returns values of stat_name of all variables contained in the dictionary.
        """

        # some sanity checks
        if not stat_dict: raise ValueError("Input dictionary is still empty! Cannot access anything from it.")
        if not "common_stat" in stat_dict.keys(): raise ValueError(
            "Input dictionary does not seem to be a proper statistics dictionary as common_stat-element is missing.")

        stat_dict_filter = (stat_dict).copy()
        stat_dict_filter.pop("common_stat")

        if not stat_dict_filter.keys(): raise ValueError("Input dictionary does not contain any variables.")

        try:
            varstat = np.array([stat_dict_filter[i][0][stat_name] for i in [*stat_dict_filter.keys()]])
            if np.ndim(varstat) == 1:  # avoid returning rank 1-arrays
                return varstat.reshape(-1, 1)
            else:
                return varstat
        except:
            raise ValueError("Could not find " + stat_name + " for all variables of input dictionary.")

    @staticmethod
    def get_stat_vars(stat_dict, stat_name, vars_in):
        """
         Retrieves requested statistics (stat_name) for all unique variables listed in allvars given statistics dictionary.
         If more than one unique variable is processed, this method returns a list, whereas a scalar is returned else.
        """

        if not stat_dict: raise ValueError("Statistics dictionary is still empty! Cannot access anything from it.")
        if not "common_stat" in stat_dict.keys(): raise ValueError(
            "Input dictionary does not seem to be a proper statistics dictionary as common_stat-element is missing.")

        vars_uni, indrev = np.unique(vars_in, return_inverse=True)

        try:
            if len(vars_uni) > 1:
                return ([stat_dict[var][0][stat_name] for var in vars_uni[indrev]])
            else:
                return (stat_dict[vars_uni[0]][0][stat_name])
        except:
            raise ValueError("Could not find " + stat_name + " for all variables of input dictionary.")

    @staticmethod
    def get_common_stat(stat_dict, stat_name):

        if not stat_dict: raise ValueError("Input dictionary is still empty! Cannot access anything from it.")
        if not "common_stat" in stat_dict.keys(): raise ValueError(
            "Input dictionary does not seem to be a proper statistics dictionary as common_stat-element is missing.")

        common_stat_dict = stat_dict["common_stat"][0]

        try:
            return (common_stat_dict[stat_name])
        except:
            raise ValueError("Could not find " + stat_name + " in common_stat of input dictionary.")

    def write_stat_json(self, path_out, file_id=-1):
        """
        Writes statistics-dictionary of slave nodes to json-file (with job_id in the output name)
        If file_id is passed (and greater than 0), parallelized peration on a slave node is assumed.
        Else: method is invoked from master node, i.e. final json-file is created
        """
        if (self.mode == "loc"):
            if int(file_id) <= 0: raise ValueError("Object is in loc-mode, but no valid file_id passed")
            # json-file from slave node
            js_file = os.path.join(path_out, 'stat_{0:0=2d}.json'.format(int(file_id)))
        elif (self.mode == "master"):
            if (int(file_id) > 0): print("Warning: Object is master-mode, but file_id passed which will be ignored.")
            # (final) json-file from master node
            js_file = os.path.join(path_out, 'statistics.json')
        else:
            raise ValueError("Object seems to be initialized only, but no data has been processed so far.")

        try:
            with open(js_file, 'w') as stat_out:
                json.dump(self.stat_dict, stat_out)
        except ValueError:
            print("Something went wrong when writing dictionary to json-file: '" + js_file + "''")
        finally:
            print("Created statistics json-file '" + js_file + "' successfully.")

