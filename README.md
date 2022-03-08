<img src="ambs_logo.jpg" width="1000" height="400">

[Check our video](https://www.youtube.com/watch?v=Tf2BDDlSDeQ)


## Table of Contents  

- [Introduction to Atmospheric Machine Learning Benchmarking System](#introduction-to-atmopsheric-machine-learning-benchmarking-system)
- [Prepare your dataset](#prepare-your-dataset)
    + [Access ERA5 dataset (~TB)](#access-era5-dataset---tb-)
    + [Dry run with small samples (~15 GB)](#dry-run-with-small-samples---15-gb-)
    + [Climatological mean data](#climatological-mean-data)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  * [Preparation with NVIDIA's TF1.15 singularity containers](#preparation-with-nvidia-s-tf115-singularity-containers)
- [Start with AMBS](#start-with-ambs)
  * [Set-up virtual environment](#set-up-virtual-environment)
    + [On Jülich's HPC systems](#on-j-lich-s-hpc-systems)
    + [On other HPC systems](#on-other-hpc-systems)
    + [Other systems](#other-systems)
  * [Run the workflow](#run-the-workflow)
  * [Create specific runscripts](#create-specific-runscripts)
  * [Running the workflow substeps](#running-the-workflow-substeps)
  * [Compare and visualize the results](#compare-and-visualize-the-results)
  * [Input and Output folder structure and naming convention](#input-and-output-folder-structure-and-naming-convention)
- [Benchmarking architectures](#benchmarking-architectures)
- [Contributors and contact](#contributors-and-contact)
- [On-going work](#on-going-work)



## Introduction to Atmopsheric Machine Learning Benchmarking System 

**A**tmopsheric **M**achine Learning **B**enchmarking **S**ystem (AMBS) aims to provide state-of-the-art video prediction methods applied to the meteorological domain. In the scope of the current application, the hourly evolution of the 2m temperature over a used-defined region is focused. 

Different Deep Learning video prediction architectures such as ConvLSTM and SAVP are trained with ERA5 reanalysis to perform a prediction for 12 hours based on the previous 12 hours. In addition to the 2m temperature (2t) itself, other variables can be fed to the video frame prediction models to enhance their capability to learn the complex physical processes driving the diurnal cycle of temperature. Currently, the recommended additional meteorological variables are the 850 hPa temperature (t850) and the total cloud cover (tcc) as described in our preprint GMD paper. 


## Prepare your dataset


#### Access ERA5 dataset (~TB)
The experiment described in the GMD paper relies on the rather large ERA5 dataset with 13 years data.  

- For the users of JSC HPC system: You access the data from the followin path: /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib. If you meet access permission issue please contact: Stein, Olaf <o.stein@fz-juelich.de>

-  For the users of other HPC sytems: You can retrieve the ERA5 data from the ECMWF MARS archive by specifying a resolution of 0.3° in the retrieval script (keyword "GRID",  "https://confluence.ecmwf.int/pages/viewpage.action?pageId=123799065 ").  The variable names and the corresponding paramID can be found in the ECMWF documentaation website [ERA5 documentations](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Howtoacknowledge,citeandrefertoERA5)

We recommend the users to store the data following the input structure of the described [in the following description](#input-and-output-folder-structure-and-naming-convention)

#### Dry run with small samples (~15 GB)

In our application, we are dealing with the large dataset. Nevertheless, we also prepared rather small samples ~ 15 GB (1 month data in 2007,2008,2009 respectively data with few variables) to help the users to be able fast test the workflow.  The data can be downloaded through the following link  [link!!] .  For the users of deepacf project in JSC: You can also access from the following path `cd /p/project/deepacf/deeprain/video_prediction_shared_folder/GMD_samples` 


#### Climatological mean data
climatological mean which is inferred at each grid point from the ERA5 reanalysis data between 1990 and 2019 is used in the postprocess step. The data can be downloaded along with the small samples [link!!] .


## Prerequisites
- Linux or macOS
- Python 3.6
- CPU or NVIDIA GPU + CUDA CuDNN
- MPI
- Tensorflow 1.13.1 or CUDA-enabled NVIDIA TensorFlow 1.15 within a singularity container 
- CDO >= 1.9.5

## Installation 

Clone this repo by typing the following command in your personal target dirctory:

```bash 
git clone https://gitlab.jsc.fz-juelich.de/esde/machine-learning/ambs.git
```

Since the project is continuously developed and make the experiments described in the GMD paper reproducible, we also provide a frozen version:

```bash 
git clone https://gitlab.jsc.fz-juelich.de/esde/machine-learning/ambs_gmd1.git
```

This will create a directory called `ambs` under which this README-file and two subdirectories are placed. The subdirectory `[...]/ambs/test/` contains unittest-scripts for the workflow and is therefore of minor relevance for non-developers. The subdirectory `[...]/ambs/video_prediction_tools` contains everything which is needed in the workflow and is, therefore, called the top-level directory in the following.

Thus, change into this subdirectory after cloning:
```bash 
cd ambs/video_prediction_tools/
```


### Preparation with NVIDIA's TF1.15 singularity containers

Since 2022, JSC HPC does not support TF1.X in the current stack software system. As an intermediate solution before the TF2 version being ready,
a singularity container with a CUDA-enabled NVIDIA TensorFlow v1.15 was made available which has to be reflected when setting up the virtual environment and when submiiting the job.  

 Then, you can either download container image ([Link](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-09.html#rel_21-09)) and place it under the folder`HPC_script`;  Or you can access to the image though the symlink command as below, if you are part of the *deepacf* project (still link to the `HPC_scripts`-directory)

```bash
cd ambs/video_prediction_tools/HPC_scripts
ln -sf /p/project/deepacf/deeprain/video_prediction_shared_folder/containers_juwels_booster/nvidia_tensorflow_21.09-tf1-py3.sif tensorflow_21.09-tf1-py3.sif
```


Note that if you are the user of JSC HPC system, you need to log in [Judoor account] (https://judoor.fz-juelich.de/login) and specifically ask for the request to access to the restricted container software. If your system support TF1.X, you can load the corresponding module or install the package by adding it to requirement.txt


## Start with AMBS

### Set-up virtual environment

AMBS is a tool for the users who develop on HPC systems with Slurm batch systems since the large-scale dataset and architectures would be used.
However, aforementioned we also provide a small dataset and runscripts for the users that can explore the tool on their personal computer systems. 
In such case, we provide three approaches to set up your virtual environment based on systems that the users work on: Jülich HPC system, other HPC systems, or other computer systems. The introduction is described below.

#### On Jülich's HPC systems 

The following commands will setup a customized virtual environment on a known HPC-system at JSC (Juwels, Juwels Booster or HDF-ML). The script `create_env.sh` automatically detects on which machine it is executed and loads/installs all required Python (binary) modules and packages. The virtual environment with the name provide by user is then set up in a subdirectory `[...]/ambs/video_prediction_tools/virtual_envs/<env_name>` the top-level directory (`../ambs/video_prediction_tools`).


```bash
cd ../ambs/video_prediction_tools/env_setup
source create_env.sh <env_name> 
```

This also already sets up the runscript templates with regards to the five steps of the workflow for you under the folder  `[...]/ambs/video_prediction_tools/JSC_scripts`. 

By default, the runscript templates make use of the standard target base directory `/p/project/deepacf/deeprain/video_prediction_shared_folder/`. This directory will serve as your standard top-level direcotry to store the output of each step in the workflow see details in the [folder structure section]( #input-and-output-folder-tructure-and-naming-convention). In case that you want to deviate from this, you may call `create_env.sh`  to custermise a new  root direcotyr as follows:

```bash
source create_env.sh <env_name> -base_dir=<my_target_dir>
```
**Note** that suifficient read-write permissions and a reasonable amount of memory space is mandatory for your alternative standard output directory.

#### On other HPC systems
Setting up the environment on other HPC is different from the ones in JSC since there is quite diversity with regards to the available software stack. The users need to load the modules manually. We prepare the templates for each step of workflow under the `HPC_scripts` . The users can follow the guidance to customise the templates. 

#### Other systems

AMBS also allows the users to test on other non-HPC machines.  You may enter the folder `../ambs/video_prediction_tools/env_setup` and excute:

```bash
source create_env_non_HPC.sh <env_name> 
```
Then the virtual enviornment will be created under `../ambs/video_prediction_tools/virtual_envs`. The required packages (`requirement_non_HPC.txt`) will be installed.


### Run the workflow

Depending on the computing system you are working on, the workflow steps will be invoked by dedicated runscripts either from the directory `JSC_scripts/` (on known HPC-systems, see above) or from the directory `HPC_scripts/`, `other_scripts/`
To help the users conduct different experiments with different configuration (e.g. input variables, hyperparameters etc).  Each runscript can be set up conveniently with the help of the Python-script `generate_runscript.py`. Its usage as well the workflow runscripts are described subsequently. 


### Create specific runscripts

Specific runscripts for each workflow substep (see below) are generated conveniently by keyboard interaction.

The interactive Python script under the folder `../ambs/video_prediction_tools/env_setup` thereby has to be executed in an activated virtual environment with some additional modules! After prompting 

```bash
python generate_runscript.py --venv_path <venv_name>
```

You will be asked first which workflow runscript shall be generated. You can chose one of the workflow step name: 
- extract
- preprocess1
- preprocess2
- train
- postprocess 

The subsequent keyboard interactions then allow the user to make individual settings to the workflow step at hand. By pressing simply Enter, the user may receive some guidance for the keyboard interaction. 

Note that the runscript creation of later workflow substeps depends on the preceding steps (i.e. by checking the arguments from keyboard interaction).
Thus, they should be created sequentially instead of all at once at the beginning. 


**Note:** The step of one step relies on the outcomes from previous step in the workfow. Please run the steps sequentially instead of in parallel. 
**Warning**: the `generate_runscript.py` currently is only for the JSC users. You can skip this step for non-JSC HPC users. If you have different settings for various experiments, you can simply copy the template to a new file where you can customize your setting. 

### Running the workflow substeps 

Having created the runscript by keyboard interaction, the workflow substeps can be run sequentially. Depending on the machine you are working on, change either to `JSC_scripts/` (on Juwels, Juwels Booster or HDF-ML), `HPC_scripts/` or `other_scripts/`  . The respective runscripts for all steps of the workflow are located whose order is as follows. Note that `[sbatch]` only has to precede on one of the HPC systems. Besides data extraction and preprocessing step 1 are only mandatory when ERA5 data is subject to the application.

Note we provide default configurations for each runscripts 
that the users still need to manully configure flags based on which project and HPC systems you work on. Particurly, you must configure the flag `#SBATCH --account =<your computing project name>` with your project name. For partitions `#SBATCH --partition`, we refer the users to the following link [JUWELS/JUWELS Booster](https://apps.fz-juelich.de/jsc/hps/juwels/batchsystem.html#slurm-partitions) for further information. If you are using HDF-ML system, you can simply use `batch` as partition.

Now it is time to run the AMBS workflow
1. Data Extraction: This script retrieves the demanded variables for user-defined years from complete ERA% reanalysis grib-files and stores the data into netCDF-files.

```bash
[sbatch] ./data_extraction_era5.sh
```

2. Data Preprocessing: Crop the ERA 5-data (multiple years possible) to the region of interest (preprocesing step 1). All the year data will be touched once and the statistics are calculated and saved in the output folder. The TFrecord-files which are fed to the trained model (next workflow step) are created afterwards. Thus, two cases exist at this stage:

    ```bash
    [sbatch] ./preprocess_data_era5_step1.sh
    [sbatch] ./preprocess_data_era5_step2.sh
    ```

3. Training: Training of one of the available models with the preprocessed data. 
Note that the `exp_id` is generated automatically when running `generate_runscript.py`.

    ```bash
    [sbatch] ./train_model_era5_<exp_id>.sh
    ```
    
4. Postprocess: Create some plots and calculate the evaluation metrics for test dataset. Note that the `exp_id` is generated automatically when running `generate_runscript.py`.

    ```bash
    [sbatch] ./visualize_postprocess_era5_<exp_id>.sh
    ```

### Compare and visualize the results

AMBS also provide the tool (called met_postprocess) for the users to compare different experiments results and visualize the results as shown in GMD paper through `meta_postprocess` step. The runscript template are also prepared in the `HPC_scripts`,  `JSC_scripts`, and  `other_scripts`. 

### Input and Output folder structure and naming convention
To successfully run the workflow and enable to track the result from each step, inputs and output directories, and the file name convention should be constructed as described below:

We demonstrate an example of inputs structure for ERA5 dataset. In detail, the data is recorded hourly and stored into two grib files. The file with postfix `*_ml.grb` consists of multi-layers of the variables, whereas `_sf.grb` only includes the surface data.

```
├── ERA5 dataset
│   ├── [Year]
│   │   ├── [Month]
│   │   │   ├── *_ml.grb 
│   │   │   ├── *_sf.grb 
│   │   │   ├── ...
│   │   ├── [Month]
│   │   │   ├── *_ml.grb 
│   │   │   ├── *_sf.grb 
│   │   │   ├── ...
```

The root output directory should be set up when you run the workflow at the first time as aformentioned. The output strucutre for each step of the workflow along with the file name convention are described below:
```
├── ExtractedData
│   ├── [Year]
│   │   ├── [Month]
│   │   │   ├── **/*.netCDF
├── PreprocessedData
│   ├── [Data_name_convention]
│   │   ├── pickle
│   │   │   ├── X_<Month>.pkl
│   │   │   ├── T_<Month>.pkl
│   │   │   ├── stat_<Month>.pkl
│   │   ├── tfrecords
│   │   │   ├── sequence_Y_<Year>_M_<Month>.tfrecords
│   │   │── metadata.json
├── Models
│   ├── [Data_name_convention]
│   │   ├── [model_name]
│   │   │   ├── <timestamp>_<user>_<exp_id>
│   │   │   │   ├── checkpoint_<iteration>
│   │   │   │   │   ├── model_*
│   │   │   │   │── timing_per_iteration_time.pkl
│   │   │   │   │── timing_total_time.pkl
│   │   │   │   │── timing_training_time.pkl
│   │   │   │   │── train_losses.pkl
│   │   │   │   │── val_losses.pkl
│   │   │   │   │── *.json 
├── Results
│   ├── [Data_name_convention]
│   │   ├── [training_mode]
│   │   │   ├── [source_data_name_convention]
│   │   │   │   ├── [model_name]
│   │   │   │   │  ├── *.nc
├── meta_postprocoess
│   ├── [experiment ID]

```

- ***Details of file name convention:***
| Arguments	| Value	|
|---	|---	|
| [Year]	| 2005;2006;2007,...,2019|
| [Month]	| 01;02;03 ...,12|
|[Data_name_convention]|Y[yyyy]to[yyyy]M[mm]to[mm]-[nx]_[ny]-[nn.nn]N[ee.ee]E-[var1]_[var2]_[var3]|
|[model_name]| convLSTM, savp, ...|


- ***Data name convention***
`Y[yyyy]to[yyyy]M[mm]to[mm]-[nx]_[ny]-[nn.nn]N[ee.ee]E-[var1]_[var2]_[var3]`
    * Y[yyyy]to[yyyy]M[mm]to[mm]
    * [nx]_[ny]: the size of images,e.g 64_64 means 64*64 pixels 
    * [nn.nn]N[ee.ee]E: the geolocation of selected regions with two decimal points. e.g : 0.00N11.50E
    * [var1]_[var2]_[var3]: the abbrevation of selected variables

Here we give some examples to explain the name conventions:
| Examples	| Name abbrevation 	|
|---	|---	|
|all data from March to June of the years 2005-2015	| Y2005toY2015M03to06 |   
|data from February to May of years 2005-2008 + data from March to June of year 2015| Y2005to2008M02to05_Y2015M03to06 |   
|Data from February to May, and October to December of 2005 |  Y2005M02to05_Y2015M10to12 |   
|operational’ data base: whole year 2016 |  Y2016M01to12 |   
|add new whole year data of 2017 on the operational data base |Y2016to2017M01to12 |  
|Note: Y2016to2017M01to12 = Y2016M01to12_Y2017M01to12  


## Benchmarking architectures
Currently, the workflow include the following ML architectures, and we are working on integrating more into the system.
- ConvLSTM: [paper](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf),[code](https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow)
- Stochastic Adversarial Video Prediction (SAVP): [paper](https://arxiv.org/pdf/1804.01523.pdf),[code](https://github.com/alexlee-gk/video_prediction) 
- Variational Autoencoder:[paper](https://arxiv.org/pdf/1312.6114.pdf)

## Contributors and contact

The project is currently developed by Bing Gong, Michael Langguth, Amirpasha Mozafarri, and Yan Ji. 

- Bing Gong: b.gong@fz-juelich.de
- Michael Langguth: m.langguth@fz-juelich.de
- Amirpash Mozafarri: a.mozafarri@fz-juelich.de
- Yan Ji: y.ji@fz-juelich.de

Former code developers are Scarlet Stadtler and Severin Hussmann.

## On-going work

- Port to PyTorch version
- Parallel training neural network
- Integrate precipitation data and new architecture used in our submitted CVPR paper
- Integrate the ML benchmark datasets such as Moving MNIST 
