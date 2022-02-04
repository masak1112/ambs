# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-12-04"
__update_date__ = "2022-02-02"


import os
from matplotlib.pylab import plt
import json
import numpy as np
import shutil
import glob
from netCDF4 import Dataset
from  model_modules.video_prediction.metrics import *
import xarray as xr


class MetaPostprocess(object):

    def __init__(self, root_dir: str = "/p/project/deepacf/deeprain/video_prediction_shared_folder/",
                 analysis_config: str = None, metric: str = "mse", exp_id=None, enable_skill_scores=False):
        """
        This class is used for calculating the evaluation metric, analyize the models' results and make comparsion
        args:
            root_dir           :str, the root directory for the shared folder
            analysis_config    :str, the path pointing to the analysis_configuration json file
            analysis_dir       :str, the path to save the analysis results
            metric             :str,  based on which evalution metric for comparison, "mse","ssim", "texture"  and "acc"
            exp_id             :str,  the given exp_id which is used as the name of postfix of the folder to store the plot
        """
        self.root_dir = root_dir
        self.analysis_config = analysis_config
        self.analysis_dir = os.path.join(root_dir, "meta_postprocess", exp_id)
        self.metric = metric
        self.exp_id = exp_id
        self.enable_skill_scores = enable_skill_scores
        self.models_type = []

    def __call__(self):
        self.sanity_check()
        self.create_analysis_dir()
        self.copy_analysis_config()
        self.load_analysis_config()
        metric_values = self.get_metrics_values()
        self.plot_scores(metric_values)
        # self.calculate_skill_scores()
        # self.plot_scores()

    def sanity_check(self):

        available_metrics = ["mse", "ssim", "texture", "acc"]
        if self.metric not in available_metrics:
            raise ("The 'metric' must be one of the following:", available_metrics)
        if type(self.exp_id) is not str:
            raise ("'exp_id' must be 'str' type ")

    def create_analysis_dir(self):
        """
        Function to create the analysis directory if it does not exist
        """
        if not os.path.exists(self.analysis_dir): os.makedirs(self.analysis_dir)
        print("1. Create analysis dir successfully: The result will be stored to the folder:", self.analysis_dir)

    def copy_analysis_config(self):
        """
        Copy the analysis configuration json to the analysis directory
        """
        try:
            shutil.copy(self.analysis_config, os.path.join(self.analysis_dir, "meta_config.json"))
            self.analysis_config = os.path.join(self.analysis_dir, "meta_config.json")
            print("2. Copy analysis config successs ")
        except Exception as e:
            print("The meta_config.json is not found in the dictory: ", self.analysis_config)
        return None

    def load_analysis_config(self):
        """
        Read the configuration values from the analysis configuration json file
        """
        with open(self.analysis_config) as f:
            self.f = json.load(f)

        print("*****The following results will be compared and ploted*****")
        [print(i) for i in self.f["results"].values()]
        print("*******************************************************")
        print("3. Loading analysis config success")

        return None

    def get_labels(self):
        labels = list(self.f["results"].keys())
        return labels

    def get_meta_info(self):
        """
        get the model types meta information of the results from the options_checkpoints json file from postprocess stage
        """

        for i in self.f["results"].values():
            option_checkpoints = os.path.join(i, "options_checkpoints.json")
            with open(option_checkpoints) as f:
                m = json.load(f)
                self.models_type.append(m["model"])
        return None

    def get_metrics_values(self):
        self.get_meta_info()
        metric_values = []
        for i, result_dir in enumerate(self.f["results"].values()):
            vals = MetaPostprocess.get_one_metric_values(result_dir, self.metric, self.models_type[i])
            metric_values.append(vals)  # return the shape: [result_id, persi_values,model_values]
        print("4. Get metrics values success")
        return metric_values

    @staticmethod
    def get_one_metric_values(result_dir: str = None, metric: str = "mse", model: str = None):
        """
        obtain the metric values (persistence and DL model) in the "evaluation_metrics.nc" file
        """
        filename = 'evaluation_metrics.nc'
        filepath = os.path.join(result_dir, filename)
        try:
            with xr.open_dataset(filepath) as dfiles:
                persi = np.array(dfiles['2t_persistence_{}_bootstrapped'.format(metric)][:])
                model = np.array(dfiles['2t_{}_{}_bootstrapped'.format(model, metric)][:])
                print("The values for evaluation metric '{}' values are obtained from file {}".format(metric, filepath))
                return [persi, model]
        except Exception as e:
            print("!! The evallution metrics retrive from the {} fails".format(filepath))
            print(e)

    def calculate_skill_scores(self):
        if self.enable_skill_scores:
            pass
            # do sometthing
        else:
            pass

    def get_lead_time_labels(metric_values: list = None):
        leadtimes = metric_values[0][0].shape[1]
        leadtimelist = ["leadhour" + str(i + 1) for i in range(leadtimes)]
        return leadtimelist

    def config_plots(self, metric_values):
        self.leadtimelist = MetaPostprocess.get_lead_time_labels(metric_values)
        self.labels = self.get_labels()
        self.markers = self.f["markers"]
        self.colors = self.f["colors"]

    @staticmethod
    def map_ylabels(metric):
        if metric == "mse":
            ylabel = "MSE"
        elif metric == "acc":
            ylabel = "ACC"
        elif metric == "ssim":
            ylabel = "SSIM"
        elif metric == "texture":
            ylabel = "Ratio of gradient ($r_G$)"
        else:
            raise ("The metric is not correct!")
        return ylabel

    def plot_scores(self, metric_values):

        self.config_plots(metric_values)

        if self.enable_skill_scores:
            self.plot_skill_scores(metric_values)
        else:
            self.plot_abs_scores(metric_values)

    def plot_abs_scores(self, metric_values: list = None):
        n_leadtime = len(self.leadtimelist)

        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for i in range(len(metric_values)):
            score_plot = np.nanquantile(metric_values[i][1], 0.5, axis = 0)
            plt.plot(np.arange(1, 1 + n_leadtime), score_plot, label = self.labels[i], color = self.colors[i],
                     marker = self.markers[i], markeredgecolor = 'k', linewidth = 1.2)
            plt.fill_between(np.arange(1, 1 + n_leadtime),
                             np.nanquantile(metric_values[i][1], 0.95, axis = 0),
                             np.nanquantile(metric_values[i][1], 0.05, axis = 0), color = self.colors[i], alpha = 0.2)

            if self.models_type[i] == "convLSTM":
                score_plot = np.nanquantile(metric_values[i][0], 0.5, axis = 0)
                plt.plot(np.arange(1, 1 + n_leadtime), score_plot, label = "Persi_cv" + str(i), color = self.colors[i],
                         marker = "D", markeredgecolor = 'k', linewidth = 1.2)
                plt.fill_between(np.arange(1, 1 + n_leadtime),
                                 np.nanquantile(metric_values[i][0], 0.95, axis = 0),
                                 np.nanquantile(metric_values[i][0], 0.05, axis = 0), color = "b", alpha = 0.2)

        plt.yticks(fontsize = 16)
        plt.xticks(np.arange(1, 13), np.arange(1, 13, 1), fontsize = 16)
        legend = ax.legend(loc = 'upper right', bbox_to_anchor = (1.46, 0.95),
                           fontsize = 14)  # 'upper right', bbox_to_anchor=(1.38, 0.8),
        ylabel = MetaPostprocess.map_ylabels(self.metric)
        ax.set_xlabel("Lead time (hours)", fontsize = 21)
        ax.set_ylabel(ylabel, fontsize = 21)
        fig_path = os.path.join(self.analysis_dir, self.metric + "abs_values.png")
        # fig_path = os.path.join(prefix,fig_name)
        plt.savefig(fig_path, bbox_inches = "tight")
        plt.show()
        plt.close()
        print("The plot saved to {}".format(fig_path))
          







   
