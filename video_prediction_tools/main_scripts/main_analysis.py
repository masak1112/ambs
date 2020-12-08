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



   def calculate_metric_one_dir(self):
       pass
  
   @staticmethod
   def load_prediction_and_real_from_one_dir(results_dir,var="T2"):
       """
       Load the reference and prediction from one results directory
       """
       fl_names = glob.glob(reseult_dirs)
       real_all = []
       persistent_all = []
       forecast_all = []
       for fl_nc in flnames:
           real,persistent, forecast = Analysis.read_values_by_var_from_nc(fl_nc,var)
           real_all.append(real)
           persistent_all.append(persistent)
           forecast_all.append(forecast)
       return real_ll,persistent_all,forecast_all


   @staticmethod
   def read_values_by_var_from_nc(fl_nc,var="T2"):
       if not var in ["T2","MSL","GPH500"]: raise ValueError ("var name is not correct, should be 'T2','MSL',or 'GPH500'")
       with Dataset(fl_nc, mode = 'r') as fl:
          #load var prediction, real and persistent values
          real = fl.variables["/analysis/reference/"+var] 
          persistent = fl.variables["/analysis/persistent/"+var]
          forecast = tf.variables["/forecast/"+var+"/stochastic/1"]
       return real, persistent, forecast   


   @staticmethod
   def calculate_metric_one_dir(real_all, persistent_all,forecast_all,metric="mse"):
       if metric == "mse":
            #compare real and persistent
           mse_persistent = persisten



   def load_results_dir_parameters(self):
       pass    
    



