# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

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
    # auxiliary shell script for converting templates to executable
    runscript_converter = "../utils/runscript_generator/convert_runscript.sh"

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
            self.runscript_dir = "../no_HPC_scripts"
        
        self.long_name_wrk_step = None
        self.rscrpt_tmpl_prefix = None
        self.runscript_template = None
        self.runscript_target   = None
        self.user = os.getenv("USER").strip()
        self.suffix_template = "_template_" + self.user + ".sh"
        # general to be expected attributes
        self.list_batch_vars = None
        self.dataset = None
        self.source_dir = None

    #
    # -----------------------------------------------------------------------------------
    #
    def run(self):
        """
        Procces keyboard interaction for step-specific runscript configuration.
        """
        raise NotImplementedError()
    #
    # -----------------------------------------------------------------------------------
    #
    def finalize(self):
        """
        Converts runscript template to executable and sets user-defined Batch-script variables from class attributes
        :return: user-defined runscript
        """
        method_name = Config_runscript_base.finalize.__name__ + " of Class " + Config_runscript_base.cls_name

        # some sanity checks (note that the file existence is already check during keyboard interaction)
        if self.runscript_template is None:
            raise AttributeError("%{0}: The attribute runscript_template is still uninitialzed." +
                                 "Run keyboard interaction (self.run) first".format(method_name))

        if self.runscript_target is None:
            raise AttributeError("%{0}: The attribute runscript_target is still uninitialzed." +
                                 "Run keyboard interaction (self.run) first".format(method_name))

        if not os.path.isfile(Config_runscript_base.runscript_converter):
            raise FileNotFoundError("%{0}: Cannot find '{1}' for converting runscript templates to executables."
                                    .format(method_name, Config_runscript_base.runscript_converter))
        # generate runscript...
        runscript_temp = os.path.join(self.runscript_dir, self.runscript_template)
        runscript_tar = os.path.join(self.runscript_dir, self.runscript_target)
        cmd_gen = "{0} {1} {2}".format(Config_runscript_base.runscript_converter, runscript_temp, runscript_tar)
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
                    err = AttributeError("%{0}: Attribute '{1}' is still None.".format(method_name, batch_var))

            if not err is None:
                raise err

            if isinstance(batch_var_val, list):
                # translate to string generating Bash-array
                batch_var_val = "(\"" + "\" \"".join(batch_var_val) + "\")"

            write_cmd = "sed -i \'s|{0}=.*|{0}={1}|g\' {2}".format(batch_var, batch_var_val, runscript)
            stat_batch_var = Config_runscript_base.check_var_in_runscript(runscript, batch_var)

            if stat_batch_var:
                stat = os.system(write_cmd)
                if stat > 0:
                    print("%{0}: Runscript script variable {1} could not be set properly."
                          .format(method_name, batch_var))
            else:
                print("%{0}: Could not find variable {1} in runscript {2} could not be set.".format(method_name, batch_var, runscript))
    #
    # -----------------------------------------------------------------------------------
    #
    def handle_source_dir(self, subdir_name):
        """
        Retrieves the value of the variable 'source_dir' from the runscript template, appends it by subdir_name and
        returns a list of its subdiretories
        :param subdir_name: name of the subdirectory to be attached
        :return: the resulting path
        """
        method_name = Config_runscript_base.handle_source_dir.__name__ + " of Class " + Config_runscript_base.cls_name

        err = None
        if not hasattr(self, "runscript_template"):
            err = ValueError("%{0}: Could not find the attribute runscript_name.".format(method_name))
        if err is None:
            if self.runscript_template is None:
                raise ValueError("%{0}: Attribute runscript_template is still uninitialized.".format(method_name))
        else:
            raise err

        runscript_file = os.path.join(self.runscript_dir, self.runscript_template)
        base_source_dir = os.path.join(Config_runscript_base.get_var_from_runscript(runscript_file,
                                                                                   "source_dir"), subdir_name)

        if not os.path.isdir(base_source_dir):
            raise NotADirectoryError("%{0}: Cannot find directory '{1}'".format(method_name, base_source_dir))

        list_dirs = Config_runscript_base.get_subdir_list(base_source_dir)

        return base_source_dir
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

        method_name = Config_runscript_base.check_var_in_runscript.__name__

        try:
            test = sp.Popen(['grep', scr_var+'=', scr_file], stdout=sp.PIPE).communicate()[0]
            test = str(test).strip("b''")                     # if nothing is found, this will return an empty string
        except:
            raise RuntimeError("%{0}: Could not execute grep-statement.".format(method_name))

        stat = False
        if test:
            stat = True

        return stat
    #
    # --------------------------------------------------------------------------------------------------------
    #
    @staticmethod
    def get_var_from_runscript(runscript_file, script_variable):
        '''
        Search for the declaration of variable in a Shell script and returns its value.
        :param runscript_file: path to shell script/runscript
        :param script_variable: name of variable which is declared in shell script at hand
        :return: value of script_variable
        '''
        script_variable = script_variable + "="
        found = False

        with open(runscript_file) as runscript:
            # Skips text before the beginning of the interesting block:
            for line in runscript:
                if script_variable in line:
                    var_value = (line.strip(script_variable)).replace("\n", "")
                    print(var_value)
                    found = True
                    break

        if not found:
            raise Exception("Could not find declaration of '" + script_variable + "' in '" + runscript_file + "'.")

        return var_value
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def get_subdir_list(base_dir, lprint=True):
        """
        Retrieves list of subdirectories form base_dir and prints it if desired
        :param base_dir: the base directory which is scanned froo subdiretories
        :param lprint: flag if list should be printed
        :return dirlist: list of subdirectories
        """
        method_name = Config_runscript_base.get_subdir_list.__name__

        if not os.path.isdir(base_dir):
            raise NotADirectoryError("%{0}: Passed directory {1} does not exist.".format(method_name, base_dir))

        dirlist = [f.name for f in os.scandir(base_dir) if f.is_dir()]
        if not dirlist:
            raise ValueError("%{0}: Cannot find any subdirectory in {1}".format(method_name, base_dir))

        if lprint:
            print("%{0}: The following subdiretories are found under {1}".format(method_name, base_dir))
            for subdir in dirlist:
                print("* {0}".format(subdir))

        return dirlist
    #
    # -----------------------------------------------------------------------------------
    #
    @staticmethod
    def keyboard_interaction(console_str, check_input, err, ntries=1, test_arg="xxx", prefix2arg=None):
        """
        Function to check if the user has passed a proper input via keyboard interaction
        :param console_str: Request printed to the console
        :param check_input: function returning boolean which needs to be passed by input from keyboard interaction.
                            Must have two arguments with the latter being an optional bool called silent.
        :param ntries: maximum number of tries (default: 1)
        :param test_arg: test argument to check_input-function (default: "xxx")
        :param prefix2arg: optional prefix that might be added to string from keyboard-interaction before it enters
                           check_input-function
        :return: The approved input from keyboard interaction
        """
        # sanity checks
        method_name = Config_runscript_base.keyboard_interaction.__name__

        # string to emphasize pritn statements of keyboard interaction
        kb_emph = " *** "

        if not callable(check_input):
            raise ValueError("%{0}: check_input must be a function!".format(method_name))
        else:
            try:
                if not type(check_input(test_arg, silent=True)) is bool:
                    raise TypeError("%{0}: check_input argument does not return a boolean.".format(method_name))
                else:
                    pass
            except:
                raise Exception("%{0}: Cannot approve check_input-argument to be proper.".format(method_name))
        if not isinstance(err,BaseException):
            raise ValueError("%{0}: err_str-argument must be an instance of BaseException!".format(method_name))
        if not isinstance(ntries,int) and ntries <= 1:
            raise ValueError("%{0}: ntries-argument must be an integer greater equal 1!".format(method_name))

        attempt = 0
        while attempt < ntries:
            func_print_emph = "%" + check_input.__name__ + ": "
            input_req = input(kb_emph + console_str + kb_emph +"\n")
            if not prefix2arg is None:
                input_req = prefix2arg + input_req
            if check_input(input_req):
                break
            else:
                if input_req == "help":
                    pass
                else:
                    attempt += 1
                if attempt < ntries:
                    print(func_print_emph + str(err))
                    console_str = "Retry!"
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
def in_virtualenv_old():
    """
    Deprecated! -> does not work in conjunction with loaded modules
    Checks if a virtual environment is activated
    :return: True if virtual environment is running, else False
    """
    stat = True
    if get_base_prefix_compat() != sys.prefix:
        print("base prefix: {0} does not match sys prefix {1}".format(get_base_prefix_compat(), sys.prefix)) 
        stat = False

    return stat
#
#--------------------------------------------------------------------------------------------------------
#
def check_virtualenv_old(labort=False):
    """
    Deprecated! -> does not work in conjunction with loaded modules
    Checks if current script is running a virtual environment and returns the directory's name
    :param labort: If True, the an Exception is raised. If False, only a Warning is given
    :return: name of virtual environment
    """
    method_name = check_virtualenv_old.__name__

    lvirt = in_virtualenv()
    if not lvirt:
        if labort:
            raise EnvironmentError("%{0}: generate_runscript.py has to run in an activated virtual environment!"
                                   .format(method_name))
        else:
            print("%{0}: config_train.py is not running in an activated virtual environment!".format(method_name))
            return
    else:
        return os.path.basename(sys.prefix)
#
#--------------------------------------------------------------------------------------------------------
#
def in_virtualenv():
    """
    New version! -> relies on "VIRTUAL_ENV" environmental variable which also works in conjunction with loaded modules
    Checks if a virtual environment is activated
    :return: True if virtual environment is running, else False
    """
    stat = bool(os.environ.get("VIRTUAL_ENV"))

    return stat
#
#--------------------------------------------------------------------------------------------------------
#
def check_virtualenv(lactive: bool = True, venv_path: str = "", labort=False):
    """
    Checks if current script is running a virtual environment and returns the directory's name
    :param lactive: If True, virtual environment must be activated. If False, the existence is required only.
    :param venv_path: Path to virtual environment (required if lactive is set to False)
    :param labort: If True, an Exception is raised. If False, only a Warning is given
    :return: name of virtual environment
    """
    method = check_virtualenv.__name__

    if lactive:
        lvirt = in_virtualenv()
        err_mess = "%{0}: No virtual environment is running.".format(method)
        venv_path = os.environ.get("VIRTUAL_ENV")
    else:
        lvirt = os.path.isfile(os.path.join(venv_path, "bin", "activate"))
        err_mess = "%{0}: Virtual environment is not existing under '{1}'".format(method, venv_path)

    if not lvirt:
        if labort:
            raise EnvironmentError(err_mess)
        else:
            raise Warning(err_mess)
            return
    else:
        return os.path.basename(venv_path)
