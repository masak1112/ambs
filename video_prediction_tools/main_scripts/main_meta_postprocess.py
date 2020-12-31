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
from  model_modules.video_prediction.metrics import *
import xarray as xr


class MetaPostprocess(object):
    def __init__(self, analysis_config=None, analysis_dir=None, stochastic_ind=0, forecast_type="deterministic"):
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
        self.load_results_dir_parameters()
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
        self.compare_by = self.f["compare_by"]
  
    @staticmethod
    def load_prediction_and_real_from_one_dir(results_dir,var="T2",stochastic_ind=0):
        """
        Load the reference and prediction from one results directory
        """
        fl_names = glob.glob(os.path.join(results_dir,"*.nc"))
        real_all = []
        persistent_all = []
        forecast_all = []
        for fl_nc in fl_names:
            real,persistent, forecast,time_forecast = MetaPostprocess.read_values_by_var_from_nc(fl_nc,var,stochastic_ind)
            real_all.append(real)
            persistent_all.append(persistent)
            forecast_all.append(forecast)
        return real_all,persistent_all,forecast_all, time_forecast


    @staticmethod
    def read_values_by_var_from_nc(fl_nc,var="T2",stochastic_ind=0):
        #if not var in ["T2","MSL","GPH500"]: raise ValueError ("var name is not correct, should be 'T2','MSL',or 'GPH500'")
        with Dataset(fl_nc, mode = 'r') as fl:
           #load var prediction, real and persistent values
           real = fl["/analysis/reference/"].variables[var][:]
           persistent = fl["/analysis/persistent/"].variables[var][:]
           forecast = fl["/forecasts/"+var+"/stochastic"].variables[str(stochastic_ind)][:]
           time_forecast = fl.variables["time_forecast"][:]
        return real, persistent, forecast, time_forecast   


    @staticmethod
    def calculate_metric_one_img(real,forecast,metric="mse"):
        if metric == "mse":
         #compare real and persistent
            eval_forecast = mse_imgs(real, forecast)
        elif metric == "psnr":
            eval_forecast = psnr_imgs(real,forecast)   
        return  eval_forecast
    
    @staticmethod
    def reshape_eval_to_one_dim(values):
        return np.array(values).flatten()

    def calculate_metric_all_dirs(self,is_persistent=False,metric="mse"):
        """
        Return the evaluation metrics for persistent and forecasing model over forecasting timestampls
        
        return:
               eval_forecast: list, the evaluation metric values for persistent  with respect to the dimenisons [results_dir,samples,timestampe]
               
        """
        eval_forecast_all_dirs = []
        for results_dir in self.results_dirs:
            real_all, persistent_all, forecast_all, self.time_forecast = MetaPostprocess.load_prediction_and_real_from_one_dir(results_dir,var="T2",stochastic_ind=self.stochastic_ind)
            
            if is_persistent: forecast_all = persistent_all
            eval_forecast_all = []
            #loop for real data
            for idx in range(len(real_all)):
                eval_forecast_per_sample_over_ts = []
                #loop the forecast time
                for time in range(len(self.time_forecast)):
                    #loop for each sample and each timestamp
                    eval_forecast = MetaPostprocess.calculate_metric_one_img(real_all[idx][time],forecast_all[idx][time], metric=metric)
                    eval_forecast_per_sample_over_ts.append(eval_forecast)

                eval_forecast_all.append(list(eval_forecast_per_sample_over_ts))
            eval_forecast_all_dirs.append(eval_forecast_all)

        times = list(range(len(self.time_forecast)))
        samples = list(range(len(real_all)))
        print("shape of list",np.array(eval_forecast_all_dirs).shape)
        evals_forecast = xr.DataArray(eval_forecast_all_dirs, coords=[self.results_dirs, samples , times], dims=["results_dirs", "samples","time_forecast"])
        return evals_forecast

        
    def save_metrics_all_dir_to_json(self):
        with open("metrics_results.json","w") as f:
            json.dump(self.eval_all,f)

         
    def load_results_dir_parameters(self,compare_by="model"):
        self.compare_by_values = []
        for results_dir in self.results_dirs:
            with open(os.path.join(results_dir, "options_checkpoints.json")) as f:
                self.options = json.loads(f.read())
                print("self.options:",self.options)
                #if self.compare_by == "model":
                self.compare_by_values.append(self.options[compare_by])
  
    
              

    
    def plot_results(self,one_persistent=True):
        """
        Plot the mean and vars for the user-defined metrics
        """
        self.load_results_dir_parameters(compare_by="model")
        evals_forecast = self.calculate_metric_all_dirs(is_persistent=False,metric="mse")
        t = evals_forecast["time_forecast"]
        mean_forecast = evals_forecast.groupby("time_forecast").mean(dim="samples").values
        var_forecast = evals_forecast.groupby("time_forecast").var(dim="samples").values
        print("mean_foreast",mean_forecast)
        x = np.array(self.compare_by_values)
        y = np.array(mean_forecast)
        e = np.array(var_forecast)
       
        plt.errorbar(t,y[0],e[0],linestyle="None",marker='^')
        plt.show()
        plt.savefig(os.path.join(self.analysis_dir,self.metrics[0]+".png"))
        plt.close()
       
          





   
