
__author__ = "Bing Gong"
__date__ = "2022-03-17"
__email__ = "b.gong@fz-juelich.de"

from hparams_utils import *
import json, os
from abc import ABC, abstractmethod

class BaseDataset(ABC):

    def __init__(self, input_dir: str, datasplit_config: str, hparams_dict_config: str, mode: str = "train", seed: int = None, nsamples_ref: int = None):
        """
        This class is used for preparing data for training/validation and test models
        :param input_dir: the path of tfrecords files
        :param datasplit_config: the path pointing to the datasplit_config json file
        :param hparams_dict_config: the path to the dict that contains hparameters,
        :param mode: string, "train","val" or "test"
        :param seed: int, the seed for dataset
        :param nsamples_ref: number of reference samples which can be used to control repetition factor for dataset
                             for ensuring adopted size of dataset iterator (used for validation data during training)
                             Example: Let nsamples_ref be 1000 while the current datset consists 100 samples, then
                                      the repetition-factor will be 10 (i.e. nsamples*rep_fac = nsamples_ref)
        """
        method = self.__class__.__name__

        self.input_dir = input_dir
        self.datasplit_config = datasplit_config
        self.mode = mode
        self.seed = seed
        self.sequence_length = None                             # will be set in get_example_info
        self.nsamples_ref = None
        self.shuffled = False                                   # will be set properly in make_dataset-method
        # sanity checks
        if self.mode not in ('train', 'val', 'test'):
            raise ValueError('%{0}: Invalid mode {1}'.format(method, self.mode))
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError("%{0} input_dir '{1}' does not exist".format(method, self.input_dir))
        if nsamples_ref is not None:
            self.nsamples_ref = nsamples_ref
        # get configuration parameters from datasplit- and model parameters-files
        self.datasplit_dict_path = datasplit_config
        self.data_dict = self.get_datasplit()
        self.hparams_dict_config = hparams_dict_config
        self.hparams = self.get_model_hparams_dict()
        #self.hparams_dict = self.get_model_hparams_dict()
        #self.hparams = self.parse_hparams() 
        self.filenames = [] #contain the data filenames


    def get_model_hparams_dict(self):
        """
        Get model_hparams_dict from json file
        """
        if self.hparams_dict_config:
            with open(self.hparams_dict_config,'r') as f:
                hparams_dict = json.loads(f.read())
        else:
            raise FileNotFoundError("hparam directory doesn't exist! please check {}!".format(self.hparams_dict_config)) 
       
        return hparams_dict

    def parse_hparams(self):
        """
        Obtain the parameters from directory
        """

        hparams = dotdict(self.hparams_dict)
        return hparams
  

    def get_datasplit(self):
        """
        Get the datasplit json file
        """
        with open(self.datasplit_dict_path,'r') as f:
            datasplit_dict = json.loads(f.read())
        return datasplit_dict


    @abstractmethod
    def get_hparams(self):
        """
        obtain the hparams from the dict to the class variables
        """
        method = BaseDataset.get_hparams.__name__

        try:
            self.context_frames = self.hparams.context_frames 
            self.max_epochs = self.hparams.max_epochs
            self.batch_size = self.hparams.batch_size
            self.shuffle_on_val = self.hparams.shuffle_on_val

        except Exception as error:
           print("Method %{}: error: {}".format(method,error))
           raise("Method %{}: the hparameter dictionary must include 'context_frames','max_epochs','batch_size','shuffle_on_val'".format(method))


    @abstractmethod
    def get_filenames_from_datasplit(self):
        """
        Get the filenames for train and val dataset
        Must implement in the Child class
        """
        pass

    @abstractmethod
    def make_dataset(self):
        """
        Prepare batch_size dataset fed into to the models.
        If the data are from training dataset,then the data is shuffled;
        If the data are from val dataset, the shuffle var will be decided by the hparams.shuffled_on_val;
        if the data are from test dataset, the data will not be shuffled

        """
        pass


    def make_batch(self):
        dataset = self.make_dataset(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()


