# Workflow for Frame Prediction by Parallel Deep Learning



## Workflow for parallel deep learning 

This project implements a workflow for parallel deep learning to predict the 2m temperature based on Severin's master thesis [code link](https://github.com/severin1992/airtemprednet) [thesis link](https://b2drop.eudat.eu/s/RmTd8K3pLsDMFw6) . 


The workflow consists of a sqeuence of steps (Data Extraction, Data Preprocessing, Training and Data Postprocess)to implement video prediction, and In each step try to Parallel for accelerating the whole prediction process.


The wokflow have been tested on the supercomputers from JSC, [JURECA](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JURECA/JURECA_node.html) and [JUWELS](https://www.fz-juelich.de/ias/jsc/EN/Expertise/Supercomputers/JUWELS/JUWELS_node.html)


## Requirement:
* Keras
* Horovod
* Python3.6
* mpi4py



## Usage

1. Clone or download this repository,
2. Install the required modules/packages on JURECA/JUWELS.
<<<<<<< HEAD
    
    ```shell
    source packageInstallation.sh
    ```
    
    Add package directory to 'PYTHONPATH'
    ```shell
    export PYTHONPATH=/p/home/jusers/USERNAME/jureca/.local/lib/python3.6/site-packages:$PYTHONPATH
    ``` 
=======

    ```shell
    source packageInstallation.sh
    ```
    Add the packages directory to the PYTHONPATH
    ```shell
    export PATHONPATH=/p/home/jusers/USERNAME/jureca/.local/lib/python3.6/site-packages:$PYTHONPATH
    ```
>>>>>>> parallel_wf_v1.0

3. Configure your input directory, output and log directory in .dat file for each step.


4. Run .sh file for submitting job


## Workflow example

![Compare all types of models in one leading day](Workflow.png?raw=true )

