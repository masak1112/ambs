# coding=utf-8
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong, Yan Ji"
__date__ = "2020-12-04"
__updatedate__ = "2022-02-02"

import argparse
import os
from matplotlib.pylab import plt
import json
import numpy as np
import shutil
import glob
from netCDF4 import Dataset
import xarray as xr


def skill_score(tar_score,ref_score,best_score):
    ss = (tar_score-ref_score) / (best_score-ref_score)
    return ss


class MetaPostprocess(object):

    def __init__(self, root_dir: str = "/p/project/deepacf/deeprain/video_prediction_shared_folder/",
            analysis_config: str = None, metric: str = "mse", exp_id: str=None, enable_skill_scores:bool=False, enable_persit_plot:bool=False):
        """
        This class is used for calculating the evaluation metric, analyize the models' results and make comparsion
        args:
            root_dir           :str, the root directory for the shared folder
            analysis_config    :str, the path pointing to the analysis_configuration json file
            analysis_dir       :str, the path to save the analysis results
            metric             :str,  based on which evalution metric for comparison, "mse","ssim", "texture"  and "acc"
            exp_id             :str,  the given exp_id which is used as the name of postfix of the folder to store the plot
            enable_skill_scores:bool, enable the skill scores plot
            enable_persis_plot: bool, enable the persis prediction in the plot
        """
        self.root_dir = root_dir
        self.analysis_config = analysis_config
        self.analysis_dir = os.path.join(root_dir, "meta_postprocess", exp_id)
        self.metric = metric
        self.exp_id = exp_id
        self.persist = enable_persit_plot
        self.enable_skill_scores = enable_skill_scores
        self.models_type = []
        self.metric_values = []  # return the shape: [num_results, persi_values, model_values]
        self.skill_scores = []  # contain the calculated skill scores [num_results, skill_scores_values]


    def __call__(self):
        self.sanity_check()
        self.create_analysis_dir()
        self.copy_analysis_config()
        self.load_analysis_config()
        self.get_metrics_values()
        if self.enable_skill_scores:
            self.calculate_skill_scores()
            self.plot_skill_scores()
        else:
            self.plot_abs_scores()

    def sanity_check(self):

        self.available_metrics = ["mse", "ssim", "texture", "acc"]
        if self.metric not in self.available_metrics:
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
        """
        get  the evaluation metric values of all the results, return a list  [results,persi, model]
        """
        self.get_meta_info()

        for i, result_dir in enumerate(self.f["results"].values()):
            vals = MetaPostprocess.get_one_metric_values(result_dir, self.metric, self.models_type[i],self.enable_skill_scores)
            self.metric_values.append(vals)
        print("4. Get metrics values success")
        return self.metric_values

    @staticmethod
    def get_one_metric_values(result_dir: str = None, metric: str = "mse", model: str = None, enable_skill_scores:bool = False):

        """
        obtain the metric values (persistence and DL model) in the "evaluation_metrics.nc" file
        return:  list contains the evaluatioin metrics of one result. [persi,model]
        """
        filename = 'evaluation_metrics.nc'
        filepath = os.path.join(result_dir, filename)
        try:
            with xr.open_dataset(filepath) as dfiles:
                if enable_skill_scores:
                   persi = np.array(dfiles['2t_persistence_{}_bootstrapped'.format(metric)][:])
                else:
                    persi = []
                model = np.array(dfiles['2t_{}_{}_bootstrapped'.format(model, metric)][:])
                print("The values for evaluation metric '{}' values are obtained from file {}".format(metric, filepath))
                return [persi, model]
        except Exception as e:
            print("!! The evallution metrics retrive from the {} fails".format(filepath))
            print(e)

    def calculate_skill_scores(self):
        """
        calculate the skill scores
        """
        if self.metric_values is None:
            raise ("metric_values should be a list but None is provided")

        best_score = 0
        if self.metric == "mse":
            pass

        elif self.metric in ["ssim", "acc", "texture"]:
            best_score = 1
        else:
            raise ("The metric should be one of the following available metrics :", self.available_metrics)

        if self.enable_skill_scores:
            for i in range(len(self.metric_values)):
                skill_val = skill_score(self.metric_values[i][1], self.metric_values[i][0], best_score)
                self.skill_scores.append(skill_val)

            return self.skill_scores
        else:
            return None

    def get_lead_time_labels(self):
        assert len(self.metric_values) == 2
        leadtimes = np.array(self.metric_values[0][1]).shape[1]
        leadtimelist = ["leadhour" + str(i + 1) for i in range(leadtimes)]
        return leadtimelist

    def config_plots(self):
        self.leadtimelist = self.get_lead_time_labels()
        self.labels = self.get_labels()
        self.markers = self.f["markers"]
        self.colors = self.f["colors"]
        self.n_leadtime = len(self.leadtimelist)

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

    def plot_abs_scores(self):
        self.config_plots()

        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for i in range(len(self.metric_values)): #loop number of test samples
            assert len(self.metric_values)==2
            score_plot = np.nanquantile(self.metric_values[i][1], 0.5, axis = 0)
           
            assert len(score_plot) == self.n_leadtime
            plt.plot(np.arange(1, 1 + self.n_leadtime), list(score_plot),label = self.labels[i], color = self.colors[i],
                     marker = self.markers[i],   markeredgecolor = 'k', linewidth = 1.2)
            plt.fill_between(np.arange(1, 1 + self.n_leadtime),
                             np.nanquantile(self.metric_values[i][1], 0.95, axis = 0),
                             np.nanquantile(self.metric_values[i][1], 0.05, axis = 0), color = self.colors[i],
                             alpha = 0.2)
            #only plot the persist prediction when the enabled
            if self.persist:
                 
                score_plot = np.nanquantile(self.metric_values[i][0], 0.5, axis = 0)
                plt.plot(np.arange(1, 1 + self.n_leadtime), score_plot, label = "Persi_cv" + str(i),
                         color = self.colors[i], marker = "D", markeredgecolor = 'k', linewidth = 1.2)
                plt.fill_between(np.arange(1, 1 + self.n_leadtime),
                                 np.nanquantile(self.metric_values[i][0], 0.95, axis = 0),
                                 np.nanquantile(self.metric_values[i][0], 0.05, axis = 0), color = "b", alpha = 0.2)

        plt.yticks(fontsize = 16)
        plt.xticks(np.arange(1, self.n_leadtime+1), np.arange(1, self.n_leadtime + 1, 1), fontsize = 16)
        legend = ax.legend(loc = 'upper right', bbox_to_anchor = (1.46, 0.95),
                           fontsize = 14)  # 'upper right', bbox_to_anchor=(1.38, 0.8),
        ylabel = MetaPostprocess.map_ylabels(self.metric)
        ax.set_xlabel("Lead time (hours)", fontsize = 21)
        ax.set_ylabel(ylabel, fontsize = 21)
        fig_path = os.path.join(self.analysis_dir, self.metric + "_abs_values.png")
        # fig_path = os.path.join(prefix,fig_name)
        plt.savefig(fig_path, bbox_inches = "tight")
        plt.show()
        plt.close()
        print("The plot saved to {}".format(fig_path))

    def plot_skill_scores(self):
        """
        Plot the skill scores once the enable_skill is True
        """
        self.config_plots()
        fig = plt.figure(figsize = (8, 6))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for i in range(len(self.skill_scores)):
            if self.models_type[i] == "convLSTM":
                c = "r"
            elif self.models_type[i] == "savp":
                c = "b"
            else:
                raise ("current only support convLSTM and SAVP for plotinig the skil scores")

            plt.boxplot(self.skill_scores[i], positions = np.arange(1, self.n_leadtime + 1), medianprops = {'color': c},
                        capprops = {'color': c}, boxprops = {'color': c}, showfliers = False)
            score_plot = np.nanquantile(self.skill_scores[i], 0.5, axis = 0)
            plt.plot(np.arange(1, 1 + self.n_leadtime), score_plot, color = c, linewidth = 1.2, label = self.labels[i])

        legend = ax.legend(loc = 'upper right', bbox_to_anchor = (1.46, 0.95), fontsize = 14)
        plt.yticks(fontsize = 16)
        plt.xticks(np.arange(1, self.n_leadtime +1), np.arange(1, self.n_leadtime+1, 1), fontsize = 16)
        ax.set_xlabel("Lead time (hours)", fontsize = 21)
        ax.set_ylabel("Skill scores of {}".format(self.metric), fontsize = 21)
        fig_path = os.path.join(self.analysis_dir, self.metric + "_skill_scores.png")
        plt.savefig(fig_path, bbox_inches = "tight")
        plt.show()
        plt.close()
        print("The plot saved to {}".format(fig_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="The root path for output dir")
    parser.add_argument("--analysis_config", type=str, required=True, help="The path points to the  meta_postprocess configuration file.",
                        default="../meta_postprocess_config/meta_config.json")
    parser.add_argument("--metric", help="Based on which the models are compared, the value should be in one of [mse,ssim,acc,texture]",default="mse")
    parser.add_argument("--exp_id", help="The experiment id which will be used as postfix of the output directory",default="exp1")
    parser.add_argument("--enable_skill_scores", help="compared by skill scores or the absolute evaluation values",default=False)
    parser.add_argument("--enable_persit_plot", help="If plot persistent foreasts",default=False)
    args = parser.parse_args()

    meta = MetaPostprocess(root_dir=args.root_dir,analysis_config=args.analysis_config, metric=args.metric, exp_id=args.exp_id,
                           enable_skill_scores=args.enable_skill_scores,enable_persit_plot=args.enable_persit_plot)
    meta()


if __name__ == '__main__':
    main()

