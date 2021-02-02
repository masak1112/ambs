"""
Parent class which contains some basic atributes and methods shared by all subclasses which are
used to configure the runscripts of dedicated workflow steps (done by config_runscript.py).
"""
__author__ = "Michael Langguth"
__date__ = "2021-01-25"

# import modules
import os

class Config_runscript_base:

    cls_name = "Config_runscript_base"

    def __init__(self, wrk_flw_step, runscript_base, venv_name, lhpc=False):
        """
        Sets some basic attributes required by all workflow steps
        :param wrk_flw_step: short-name of the workflow step
        :param runscript_base: (relative or absolute) path to directory where runscript templates are stored
        """
        self.VIRT_ENV_NAME = venv_name
        # runscript related attributes
        self.runscript_base = runscript_base
        if lhpc:
            self.runscript_dir = "../HPC_scripts"
        else:
            self.runscript_dir = "../Zam347_scripts"

        self.long_name_wrk_step = None
        self.rscrpt_tmpl_prefix = None
        self.suffix_template = "_template.sh"
        self.runscript_template = None             # will be constructed in child class of the workflow step
        self.runscript_target   = None
        Config_runscript_base.check_and_set_basic(self, wrk_flw_step)
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
        :return:
        """
        # some sanity checks (note that the file existence is already check during keyboard interaction)
        if self.runscript_template is None:
            raise AttributeError("The attribute runscript_template is still uninitialzed." +
                                 "Run keyboard interaction (self.run) first")

        if self.runscript_target is None:
            raise AttributeError("The attribute runscript_target is still uninitialzed." +
                                 "Run keyboard interaction (self.run) first")

        runscript_temp = os.path.join(self.runscript_dir, self.runscript_template)
        runscript_tar = os.path.join(self.runscript_dir, self.runscript_target)
        cmd_gen = "./generate_work_runscripts.sh {0} {1}".format(runscript_temp, runscript_tar)
        os.system(cmd_gen)

        Config_runscript_base.write_rscr_vars(self, runscript_tar)

    def check_and_set_basic(self, wrk_flw_step):
        """
        Set the following basic attributes depending on the workflow step (initialized with None in __init__):
        * long_name_wrk_step: long-name of the workflow step
        * rscrpt_tmpl_suffix: prefix of the corresponding runscript template (used for constructing the name of the
                              runscript template file
        :param wrk_flw_step: short-name of the workflow step
        :return: class instance with the aforementioned attributes set
        """

        method_name = Config_runscript_base.check_and_set_basic.__name__ + " of Class " + Config_runscript_base.cls_name

        if not isinstance(wrk_flw_step, str):
            raise ValueError("%{0}: wrk_flw_step-arument must be string indicating the name of workflow substep."
                             .format(method_name))

        if not os.path.isdir(self.runscript_base):
            raise NotADirectoryError("%{0}: Could not find directory where runscript templates are expected: {1}"
                                     .format(method_name, self.runscript_base))

        if wrk_flw_step == "extract":
            self.long_name_wrk_step = "Data Extraction"
            self.rscrpt_tmpl_prefix = "data_extraction"
        elif wrk_flw_step == "preprocess1":
            self.long_name_wrk_step = "Preprocessing step 1"
            self.rscrpt_tmpl_prefix = "preprocess_data"
        elif wrk_flw_step == "preprocess2":
            self.long_name_wrk_step = "Preproccessing step 2"
            self.rscrpt_tmpl_prefix = "preprocess_data"
        elif wrk_flw_step == "train":
            self.long_name_wrk_step = "Training"
            self.rscrpt_tmpl_prefix = "train_model"
        elif wrk_flw_step == "postprocess":
            self.long_name_wrk_step = "Postprocessing"
            self.rscrpt_tmpl_prefix = "visualize_postprocess"
        else:
            raise ValueError("%{0}: Workflow step {1} is unknown / not implemented.".format(method_name, wrk_flw_step))
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
