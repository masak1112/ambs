

__author__ = "Bing Gong"
__date__ = "2022-03-17"
__email__ = "b.gong@fz-juelich.de"

from hparams_utils import *

class BaseDataset(object):

    def __init__(self, input_dir: str = None, datasplit_config: str = None, hparams_dict_config: str = None,
                 mode: str = "train", seed: int = None, nsamples_ref: int = None):

        """
        This class is used for preparing data for training/validation and test models
        :param input_dir: the path of tfrecords files
        :param datasplit_config: the path pointing to the datasplit_config json file
        :param hparams_dict_config: the path to the dict that contains hparameters,
        :param mode: string, "train","val" or "test"
        :param seed: int, the seed for dataset
        :param nsamples_ref: number of reference samples whch can be used to control repetition factor for dataset
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
        # get configuration parameters from datasplit- and modelparameters-files
        self.datasplit_dict_path = datasplit_config
        self.data_dict = self.get_datasplit()
        self.hparams_dict_config = hparams_dict_config
        self.hparams = self.parse_hparams() 
        self.get_hparams()
        self.filenames = [] #contain the data filenames
        self.get_filesnames_base_datasplit()

    def parse_hparams(self):
        self.hparams_dict = dotdict(self.hparams_dict)
        return self.hparams_dict
  

   def get_datasplit(self):
        """
        Get the datasplit json file
        """

        with open(self.datasplit_dict_path) as f:
            datasplit_dict = json.load(f)
        return datasplit_dict

    def get_hparams(self):
        """
        obtain the hparams from the dict to the class variables
        """
        method = BaseDataset.__class__.__name__
        try:
            self.context_frames = self.hparams.context_frames 
            self.max_epochs = self.hparams.max_epochs
            self.batch_size = self.hparams.batch_size
            self.shuffle_on_val = self.hparams.shuffle_on_val
       except Exception as error:
           print("Method %{}: error: {}".format(method,error))
           raise("Method %{}: the hparameter dictionary must include 'context_frames','max_epochs','batch_size','shuffle_on_val'".format(method))
   
    def get_filnames_base_datasplit(self):
       """
       Get the filenames for train and val dataset
       Must implement in the Child class
       """ 
       raise NotImplmentedError

   def calc_samples_per_epoch(self):
       """
     
       """
   def make_dataset(self):
       """
        Prepare batch_size dataset fed into to the models.
        If the data are from training dataset,then the data is shuffled;
        If the data are from val dataset, the shuffle var will be decided by the hparams.shuffled_on_val;
        if the data are from test dataset, the data will not be shuffled 

       """
       method = BaseDataset.make_dtaset.__name__
       shuffle = self.mode == 'train' or (self.mode == 'val' and self.hparams.shuffle_on_val)
       
             
