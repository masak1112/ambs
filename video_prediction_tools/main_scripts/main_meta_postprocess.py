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
        if not var in ["T2","MSL","GPH500"]: raise ValueError ("var name is not correct, should be 'T2','MSL',or 'GPH500'")
        with Dataset(fl_nc, mode = 'r') as fl:
           #load var prediction, real and persistent values
           real = fl["/analysis/reference/"].variables[var][:]
           persistent = fl["/analysis/persistent/"].variables[var][:]
           forecast = fl["/forecast/"+var+"/stochastic"].variables[str(stochastic_ind)][:]
           time_forecast = fl.variables["time_forecast"][:]
        return real, persistent, forecast, time_forecast   


    @staticmethod
    def calculate_metric_one_img(real, persistent,forecast,metric="mse"):
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
            self.eval_all.update({results_dir: {"persistent":None}})
            self.eval_all.update({results_dir: {"forecast":None}})
            real_all, persistent_all, forecast_all, self.time_forecast = MetaPostprocess.load_prediction_and_real_from_one_dir(results_dir,var="T2",stochastic_ind=self.stochastic_ind)
            for metric in self.metrics:
                self.eval_persistent_all = []
                self.eval_forecast_all = []
                #loop for real data
                for idx in range(len(real_all)):
                    eval_persistent_per_sample_over_ts = []
                    eval_forecast_per_sample_over_ts = []
                    
                    #loop the forecast time
                    for time in range(len(self.time_forecast)):
                        #loop for each sample and each timestamp
                        self.eval_persistent, self.eval_forecast = MetaPostprocess.calculate_metric_one_img(real_all[idx][time],persistent_all[idx][time],forecast_all[idx][time], metric=metric)
                        eval_persistent_per_sample_over_ts.append(self.eval_persistent)
                        eval_forecast_per_sample_over_ts.append(self.eval_forecast)
                    
                    self.eval_persistent_all.append(list(eval_persistent_per_sample_over_ts))
                    self.eval_forecast_all.append(list(eval_forecast_per_sample_over_ts))
                    #the shape of self.eval_persistent_all is [samples,time_forecast]
                self.eval_all[results_dir]["persistent"] = {metric: list(self.eval_persistent_all)}           
                self.eval_all[results_dir]["forecast"] = {metric: list(self.eval_forecast_all)}   
        
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
  
    
    def calculate_mean_vars_forecast(self):
        """
        Calculate the mean varations of persistent and forecast evalaution metrics
        """
        is_first_persistent = False
        for results_dir in self.results_dirs:
            evals = self.eval_all[results_dir]
            eval_persistent = evals["persistent"]
            eval_forecast = evals["forecast"]
            self.results_dict = {} 
            for metric in self.metrics:
                err_stat = []
                for time in range(len(self.time_forecast)):
                    forecast_values_all = list(eval_forecast[metric])[:][time]
                    persistent_values_all = list(eval_persistent[metric])[:][time]
                    forecast_mean = np.mean(np.array(forecast_values_all),axis=0)
                    persistent_mean = np.mean(np.array(persistent_values_all),axis=0)
                    forecast_vars = np.var(np.array(forecast_values_all),axis=0)
                    persistent_vars = np.var(np.array(persistent_values_all),axis=0)
                    #[time,mean,vars]
                    self.results_dict[results_dir] = {"persistent":[persistent_mean, persistent_vars]} 
                    self.results_dict[results_dir].update({"forecast":[forecast_mean,forecast_vars]})
               

    def plot_results(self,one_persistent=True):
        """
        Plot the mean and vars for the user-defined metrics
        """
        self.load_results_dir_parameters()
        is_first_persistent=True
        mean_all_persistent = []
        vars_all_persistent = []
        mean_all_model = []
        vars_all_model = []
        for results_dir in self.results_dirs:
            mean_all_persistent.append(self.results_dict[results_dir]["persistent"][0])
            vars_all_persistent.append(self.results_dict[results_dir]["persistent"][1])
            mean_all_model.append(self.results_dict[results_dir]["forecast"][0])
            vars_all_model.append(self.results_dict[results_dir]["forecast"][1]) 
    
        x = np.array(self.compare_by_values)
        y = np.array(mean_all_model)
        e = np.array(vars_all_model)

        plt.errorbar(x,y,e,linestyle="None",marker='^')
        plt.show()
        plt.savefig(os.path.join(self.analysis_dir,self.metrics[0]+".png"))
        plt.close()
       
          





   
