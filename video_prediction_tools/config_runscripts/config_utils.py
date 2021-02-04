"""
Parent class which contains some basic atributes and methods shared by all subclasses which are
used to configure the runscripts of dedicated workflow steps (done by config_runscript.py).
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-25"

# import modules
import sys, os
import subprocess as sp

class Config_runscript_base:

    cls_name = "Config_runscript_base"

    # list of known workflow steps
    known_workflow_steps = ["extract", "preprocess1", "preprocess2", "train", "postprocess"]

    def __init__(self, venv_name, lhpc=False):
        """
        Sets some basic attributes required by all workflow steps
        :param venv_name: name of the virtual environment
        :param lhpc: flag if operation is done on an HPC system
        """
        self.VIRT_ENV_NAME = venv_name
        # runscript related attributes
        if lhpc:
            self.runscript_dir = "../HPC_scripts"
        else:
            self.runscript_dir = "../Zam347_scripts"

        self.long_name_wrk_step = None
        self.rscrpt_tmpl_prefix = None
        self.suffix_template = "_template.sh"
        self.runscript_template = None
        self.runscript_target   = None
        # general to be expected attributes
        self.list_batch_vars = None
        self.dataset = None
        self.source_dir = None
        # attribute storing workflow-step dependant function for keyboard interaction
        self.run_config = None
    #
    # -----------------------------------------------------------------------------------
    #
    def run(self):
        """
        Acts as generic wrapper: Checks if run_config is already set up as a callable
        :return: Executes run_config
        """
        method_name = "run" + " of Class " + Config_runscript_base.cls_name
        if self.run_config is None:
            raise ValueError("%{0}: run-method is still uninitialized.".format(method_name))

        if not callable(self.run_config):
            raise ValueError("%{0}: run-method is not callable".format(method_name))

        # simply execute it
        self.run_config(self)
    #
    # -----------------------------------------------------------------------------------
    #
    def finalize(self):
        """
        Converts runscript template to executable and sets user-defined Batch-script variables from class attributes
        :return: user-defined runscript
        """
        # some sanity checks (note that the file existence is already check during keyboard interaction)
        if self.runscript_template is None:
            raise AttributeError("The attribute runscript_template is still uninitialzed." +
                                 "Run keyboard interaction (self.run) first")

        if self.runscript_target is None:
            raise AttributeError("The attribute runscript_target is still uninitialzed." +
                                 "Run keyboard interaction (self.run) first")
        # generate runscript...
        runscript_temp = os.path.join(self.runscript_dir, self.runscript_template).rstrip("_template.sh")
        runscript_tar = os.path.join(self.runscript_dir, self.runscript_target)
        cmd_gen = "./generate_work_runscripts.sh {0} {1}".format(runscript_temp, runscript_tar)
        os.system(cmd_gen)
        # ...do modificatios stored in attributes of class instance
        Config_runscript_base.write_rscr_vars(self, runscript_tar)
    #
    # -----------------------------------------------------------------------------------
    #
    def write_rscr_vars(self, runscript):
        """
        Writes batch-script variables from self.list_batch_vars into target runscript
        :param runscript: name of the runscript to work on
        :return: modified runscript
        """

        method_name = Config_runscript_base.write_rscr_vars.__name__ + " of Class " + Config_runscript_base.cls_name

        # sanity check if list of batch variables to be written is initialized
        if self.list_batch_vars is None:
            raise AttributeError("The attribute list_batch_vars is still unintialized." +
                                 "Run keyboard interaction (self.run) first!")

        for batch_var in self.list_batch_vars:
            err = None
            if not hasattr(self, batch_var):
                err = AttributeError("%{0}: Cannot find attribute '{1}'".format(method_name, batch_var))
            else:
                batch_var_val = getattr(self, batch_var)
                if batch_var_val is None:
                    err= AttributeError("%{0}: Attribute '{1}' is still None.".format(method_name, batch_var))

            if not err is None:
                raise err

            if isinstance(batch_var_val, list):
                # translate to string generating Bash-array
                batch_var_val = "(\"" + "\"\n\"".join(batch_var_val) + "\")"

            write_cmd = "sed -i \"s/{0}=.*/{0}={1}/g\" {2}".format(batch_var, batch_var_val, runscript)
            stat_batch_var = Config_runscript_base.check_var_in_runscript(batch_var, runscript)

            if stat_batch_var:
                os.system(write_cmd)
            else:
                print("%{0}: Runscript script variable {1} could not be set.".format(method_name, batch_var))
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def path_rec_split(full_path):
        """
        :param full_path: input path to be splitted in its components
        :return: list of all splitted components
        """
        rest, tail = os.path.split(full_path)
        if rest in ('', os.path.sep): return tail,

        return Config_runscript_base.path_rec_split(rest) + (tail,)
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def check_var_in_runscript(scr_file, scr_var):
        '''
        Checks if variable in a Shell script is declared, i.e. if "scr_var=*" is part of the script
        :param scr_file: path to shell script/runscript
        :param scr_var: name of variable whose declaration should be checked
        :return stat: True if variable declaration was detected
        '''

        test = sp.Popen(['grep', scr_var+'=', scr_file], stdout=sp.PIPE).communicate()[0]
        test = str(test).strip("b''")                     # if nothing is found, this will return an empty string

        stat = False
        if test:
            stat = True

        return stat
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def keyboard_interaction(console_str, check_input, err, ntries=1, test_arg="xxx"):
        """
        Function to check if the user has passed a proper input via keyboard interaction
        :param console_str: Request printed to the console
        :param check_input: function returning boolean which needs to be passed by input from keyboard interaction.
                            Must have two arguments with the latter being an optional bool called silent.
        :param ntries: maximum number of tries (default: 1)
        :param test_arg: test argument to check_input-function (default: "xxx")
        :return: The approved input from keyboard interaction
        """
        # sanity checks
        if not callable(check_input):
            raise ValueError("check_input must be a function!")
        else:
            try:
                if not type(check_input(test_arg, silent=True)) is bool:
                    raise TypeError("check_input argument does not return a boolean.")
                else:
                    pass
            except:
                raise Exception("Cannot approve check_input-argument to be proper.")
        if not isinstance(err,BaseException):
            raise ValueError("err_str-argument must be an instance of BaseException!")
        if not isinstance(ntries,int) and ntries <= 1:
            raise ValueError("ntries-argument must be an integer greater equal 1!")

        attempt = 0
        while attempt < ntries:
            input_req = input(console_str)
            if check_input(input_req):
                break
            else:
                attempt += 1
                if attempt < ntries:
                    print(err)
                    console_str = "Retry!\n"
                else:
                    raise err

        return input_req

## some further auxiliary functions

def get_base_prefix_compat():
    """
    Robust check if script is running in virtual env from
    see: https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv/38939054
    :return: Base/real prefix, or sys.prefix if there is none.
    """
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix
#
#--------------------------------------------------------------------------------------------------------
#

def in_virtualenv():
    """
    Checks if a virtual environment is activated
    :return: True if virtual environment is running, else False
    """
    return get_base_prefix_compat() != sys.prefix
#
#--------------------------------------------------------------------------------------------------------
#
def check_virtualenv(labort=False):
    '''
    Checks if current script is running a virtual environment and returns the directory's name
    :param labort: If True, the an Exception is raised. If False, only a Warning is given
    :return: name of virtual environment
    '''
    lvirt = in_virtualenv()

    if not lvirt:
        if labort:
            raise EnvironmentError("config_train.py has to run in an activated virtual environment!")
        else:
            raise Warning("config_train.py is not running in an activated virtual environment!")
            return
    else:
        return os.path.basename(sys.prefix)
