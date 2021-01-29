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

    def __init__(self, wrk_flw_step, runscript_base):
        """
        Sets some basic attributes required by all workflow steps
        :param wrk_flw_step: short-name of the workflow step
        :param runscript_base: (relative or absolute) path to directory where runscript templates are stored
        """
        self.runscript_base     = runscript_base
        self.long_name_wrk_step = None
        self.rscrpt_tmpl_prefix = None
        self.suffix_template = "_template.sh"
        self.runscript_template = None             # will be constructed in child class of the workflow step
        self.dataset            = None
        Config_runscript_base.check_and_set_basic(self, wrk_flw_step)

        self.source_dir = None
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
