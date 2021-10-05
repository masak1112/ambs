# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Basic task of the Python-script:

Creates user-defined runscripts for all workflow runscripts via keyboard interaction.
"""

__email__ = "b.gong@fz-juelich.de"
__authors__ = "Michael Langguth"
__date__ = "2021-02-03"

# import modules
import sys, os
import socket
if sys.version_info[0] < 3:
    raise Exception("This script has to be run with Python 3!")
sys.path.append(os.path.dirname(sys.path[0]))
from runscript_generator.config_utils import check_virtualenv
# sanity check (is Python running in a virtual environment)
_ = check_virtualenv(labort=True)

from runscript_generator.config_utils import Config_runscript_base
from runscript_generator.config_extraction import Config_Extraction
from runscript_generator.config_preprocess_step1 import Config_Preprocess1
from runscript_generator.config_preprocess_step2 import Config_Preprocess2
from runscript_generator.config_training import Config_Train
from runscript_generator.config_postprocess import Config_Postprocess

#
# ----------------------------- auxiliary function -----------------------------
#
def get_runscript_cls(target_runscript_name, venv_name, lhpc):

    method_name = get_runscript_cls.__name__

    if target_runscript_name == "extract":
        cls_inst = Config_Extraction(venv_name, lhpc)
    elif target_runscript_name == "preprocess1":
        cls_inst = Config_Preprocess1(venv_name, lhpc)
    elif target_runscript_name == "preprocess2":
        cls_inst = Config_Preprocess2(venv_name, lhpc)
    elif target_runscript_name == "train":
        cls_inst = Config_Train(venv_name, lhpc)
    elif target_runscript_name == "postprocess":
        cls_inst = Config_Postprocess(venv_name, lhpc)
    else:
        raise ValueError("%{0}: Unknown workflow runscript '{1}'. passed.".format(method_name, target_runscript_name))

    return cls_inst
#
# ------------------------------------ main ------------------------------------
#
def main():

    venv_name = check_virtualenv(labort=True)

    # check if we are on a known HPC
    lhpc = False
    if any(map(socket.gethostname().__contains__, ["juwels", "hdfml"])):
        lhpc = True

    config_dummy = Config_runscript_base(venv_name, lhpc)
    known_wrkflw_steps = config_dummy.known_workflow_steps
    keyboard_interaction = config_dummy.keyboard_interaction

    # get workflow step by keyboard interaction
    target_runscript_req = "Please enter the name of the workflow step for which a runscript should be created:"
    target_runscript_err = ValueError("Workflow step is unknown. Please select one of the known listed above")

    def check_target_runscript(runscript_name, silent=False):
        if not runscript_name in known_wrkflw_steps:
            if not silent:
                print("Invalid workflow step '{0}' passed!".format(runscript_name))
                print("Known workflow steps:")
                for step in known_wrkflw_steps:
                    print("* {0}".format(step))
            return False
        else:
            return True

    target_runscript = keyboard_interaction(target_runscript_req, check_target_runscript,
                                            target_runscript_err, ntries=2)

    cls_runscript = get_runscript_cls(target_runscript, venv_name, lhpc)

    cls_runscript.run()

    cls_runscript.finalize()

    print("*** Important note: Remember to open newly created directories for other AMBS-users! ***")

if __name__== '__main__':
    main()









