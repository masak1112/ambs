from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-12-04"


import os
from matplotlib.pylab import plt
import json
import numpy as np
import shutil
import glob
from netCDF4 import Dataset
from video_prediction.metrics import *

class Analysis(object):
    def __init__(self, analysis_config=None, analysis_dir=None, stochastic_ind=0,forecast_type="deterministic"):
        """
        This class is used for calculating the evaluation metric, analyize the models' results and make comparsion
        args:
            forecast_type :str, "deterministic" or "stochastic"
        """

        self.analysis_config = analysis_config
        self.analysis_dir = analysis_dir
        self.stochastic_ind=stochastic_ind

    def __call__():
        self.create_analysis_dir()
        self.copy_analysis_config_to_analysis_dir()
        self.load_analysis_config()
        #self.load_results_dir_parameters()
        #self.load_prediction_and_persistent_real()
        self.calculate_evaluation_metrics()
        self.save_metrics_all_dir_to_json()
        self.make_comparsion()


    def create_analysis_dir(self):
        if not os.path.exists(self.analysis_dir):os.makedirs(self.analysis_dir)

    def copy_analysis_config(self):
        shutil.copy(self.analysis_config, os.path.join(self.analysis_dir,"analysis_config.json"))
        self.analysis_config = os.path.join(self.analysis_dir,"analysis_config.json")


    def load_analysis_config(self):
        """
        Get the analysis json configuration file
        """
        with open(self.analysis_config) as f:
            self.f = json.load(f)
        self.metrics = self.f["metric"]
        self.results_dirs = self.f["results_dir"]

  
    @staticmethod
    def load_prediction_and_real_from_one_dir(results_dir,var="T2",stochastic_ind=1):
        """
        Load the reference and prediction from one results directory
        """
        fl_names = glob.glob(os.path.join(results_dir,"*.nc"))
        real_all = []
        persistent_all = []
        forecast_all = []
        for fl_nc in fl_names:
            real,persistent, forecast,time_forecast = Analysis.read_values_by_var_from_nc(fl_nc,var,stochastic_ind)
            real_all.append(real)
            persistent_all.append(persistent)
            forecast_all.append(forecast)
        return real_all,persistent_all,forecast_all, time_forecast


    @staticmethod
    def read_values_by_var_from_nc(fl_nc,var="T2",stochastic_ind=0):
        if not var in ["T2","MSL","GPH500"]: raise ValueError ("var name is not correct, should be 'T2','MSL',or 'GPH500'")
        with Dataset(fl_nc, mode = 'r') as fl:
           #load var prediction, real and persistent values
           real = fl["/analysis/reference/"].variables[var][:]
           persistent = fl["/analysis/persistent/"].variables[var][:]
           forecast = fl["/forecast/"+var+"/stochastic"].variables[str(stochastic_ind)][:]
           time_forecast = fl.variables["time_forecast"][:]
        return real, persistent, forecast, time_forecast   


    @staticmethod
    def calculate_metric_one_dir(real, persistent,forecast,metric="mse"):
        if metric == "mse":
         #compare real and persistent
            eval_persistent = mse_imgs(real,persistent)
            eval_forecast = mse_imgs(real, forecast)
        elif metric == "psnr":
            eval_persistent = psnr_imgs(real,forecast)
            eval_forecast = psnr_imgs(real,forecast)   
        return eval_persistent, eval_forecast
    
    @staticmethod
    def reshape_eval_to_one_dim(values):
        return np.array(values).flatten()

    def calculate_metrics_all_dirs(self):
        """
        Calculate the all the metrics for persistent and forecast results
        eval_all is dictionary,
        eval_all = {
                     <results_dir>: 
                                   {
                                     "persistent":
                                                 {
                                                 <metric_name1> : eval_values,
                                                 <metric_name2> : eval_values
                                                 } 
                                   
                                     "forecast" :
                                                {
                                                 <metric_name1> : eval_values,
                                                 <metric_name2> : eval_values                                                
                                                }
                                   
                                   }
                    }

        """
        self.eval_all = {}
        for results_dir in self.results_dirs:
            real_all,persistent_all,forecast_all, self.time_forecast = Analysis.load_prediction_and_real_from_one_dir(results_dir,var="T2",stochastic_ind=self.stochastic_ind)
            for metric in self.metrics:
                self.eval_persistent_all = []
                self.eval_forecast_all = []
                for idx in range(len(real_all)):
                    eval_persistent_per_sample_over_ts = []
                    eval_forecast_per_sample_over_ts = []
                    for time in range(len(self.time_forecast)):
                        #loop for each sample and each timestamp
                        self.eval_persistent, self.eval_forecast = Analysis.calculate_metric_one_dir(real_all[idx][time],persistent_all[idx][time],forecast_all[idx][time], metric=metric)
                        eval_persistent_per_sample_over_ts.append(self.eval_persistent)
                        eval_forecast_per_sample_over_ts.append(self.eval_forecast)

                    self.eval_persistent_all.append(eval_persistent_per_sample_over_ts)
                    self.eval_forecast_all.append(eval_forecast_per_sample_over_ts)
                    #the shape of self.eval_persistent_all is [samples,time_forecast, eval_metric]
                self.eval_all[results_dir] = {"persistent": {metric:self.eval_persistent_all}}            
                self.eval_all[results_dir]  = {"forecast": {metric: self.eval_forecast_all}}      
        
    def save_metrics_all_dir_to_json(self):
        with open("metrics_results.json","w") as f:
            json.dump(self.eval_all,f)

         
    def load_results_dir_parameters(self):
        pass    
    
  
    def plot_results(self):
        pass
     
