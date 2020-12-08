from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2020-12-04"


import os
from matplotlib.pylab import plt
import json
import shutil
import glob
from netCDF4 import Dataset
from video_prediction.metrics import *

class Analysis(object):
    def __init__(self, analysis_config=None, analysis_dir=None,forecast_type="deterministic"):
        """
        This class is used for calculating the evaluation metric, analyize the models' results and make comparsion
        args:
            forecast_type :str, "deterministic" or "stochastic"
        """

        self.analysis_config = analysis_config
        self.analysis_dir = analysis_dir

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
        self.metrics = self.f["metrics"]
        self.results_dirs = self.f["results_dir"]

  
    @staticmethod
    def load_prediction_and_real_from_one_dir(results_dir,var="T2"):
        """
        Load the reference and prediction from one results directory
        """
        fl_names = glob.glob(reseult_dir)
        real_all = []
        persistent_all = []
        forecast_all = []
        for fl_nc in flnames:
            real,persistent, forecast = Analysis.read_values_by_var_from_nc(fl_nc,var)
            real_all.append(real)
            persistent_all.append(persistent)
            forecast_all.append(forecast)
        return real_all,persistent_all,forecast_all


    @staticmethod
    def read_values_by_var_from_nc(fl_nc,var="T2",stochastic_ind=1):
        if not var in ["T2","MSL","GPH500"]: raise ValueError ("var name is not correct, should be 'T2','MSL',or 'GPH500'")
        with Dataset(fl_nc, mode = 'r') as fl:
           #load var prediction, real and persistent values
           real = fl.variables["/analysis/reference/"+var] 
           persistent = fl.variables["/analysis/persistent/"+var]
           forecast = tf.variables["/forecast/"+var+"/stochastic/{}".format(str(stochastic_ind))]
        return real, persistent, forecast   


    @staticmethod
    def calculate_metric_one_dir(real, persistent,forecast,metric="mse"):
      #Todo check the real_all, persistent_all, forecast_all are the same
        if metric == "mse":
         #compare real and persistent
            eval_persistent = mse(real,persistent)
            eval_forecast = mse(real, forecast)
        elif metric == "psnr":
            eval_persistent = psnr(real,forecast)
            eval_forecast = psnr(real,forecast)   
        return eval_persistent, eval_forecast
    

    def calculate_metrics_all_dirs(self):
        self.eval_all = {}
        for results_dir in self.results_dir:
            real_all,persistent_all,forecast_all = Analysis.load_prediction_and_real_from_one_dir(results_dir,var="T2")
            for metric in self.metrics:
                self.eval_persistent, eval_forecast = Analysis.calcualte_metric_one_dir(real_all,persistent_all,forecast_all, metric=metric)
                self.eval_all[results_dir] = {"persistent": {metric:eval_persistent}}            
                self.eval_all[results_dir]  = {"forecast": {metric: eval_forecast}}      
        
    def save_metrics_all_dir_to_json(self):
        with open("metrics_results.json","w") as f:
            json.dump(self.eval_all,f)

         
    def load_results_dir_parameters(self):
        pass    
    
  
    def plot_results(self):
        pass
     
