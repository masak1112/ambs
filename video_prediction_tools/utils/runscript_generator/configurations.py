"""
Auxiliary functions that are used in config_runscript.py.
They are used for facilating the customized conversion of the preprocessing step 2- and training runscript-templates
to executable runscripts
"""

import os, sys

# robust check if script is running in virtual env from
# https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv/38939054
def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return getattr(sys, "base_prefix", None) or getattr(sys, "real_prefix", None) or sys.prefix
#
#--------------------------------------------------------------------------------------------------------
#
def path_rec_split(full_path):
    """
    :param full_path: input path to be splitted in its components
    :return: list of all splitted components
    """
    rest, tail = os.path.split(full_path)
    if rest in ('', os.path.sep): return tail,

    return path_rec_split(rest) + (tail,)
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
def check_virtualenv(lactive: bool= True, venv_path: str = "",labort=False):
    '''
    Checks if current script is running a virtual environment and returns the directory's name
    :param lactive: If True, virtual environment must be activated. If False, the existence is required only.
    :param labort: If True, an Exception is raised. If False, only a Warning is given
    :return: name of virtual environment
    '''
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
#
# --------------------------------------------------------------------------------------------------------
#
def get_variable_from_runscript(runscript_file, script_variable):
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
                found = True
                break

    if not found:
        raise Exception("Could not find declaration of '"+script_variable+"' in '"+runscript_file+"'.")

    return var_value
#
#--------------------------------------------------------------------------------------------------------
#
def keyboard_interaction(console_str,check_input,err,ntries=1):
    """
    Function to check if the user has passed a proper input via keyboard interaction
    :param console_str: Request printed to the console
    :param check_input: function returning boolean which needs to be passed by input from keyboard interaction.
                        Must have two arguments with the latter being an optional bool called silent.
    :param ntries: maximum number of tries (default: 1)
    :return: The approved input from keyboard interaction
    """
    # sanity checks
    if not callable(check_input):
        raise ValueError("check_input must be a function!")
    else:
        try:
            if not type(check_input("xxx",silent=True)) is bool:
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
