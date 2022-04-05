<img src="ambs_logo.jpg" width="1000" height="400">

[Check our video](https://www.youtube.com/watch?v=Tf2BDDlSDeQ)


## Table of Contents  

- [Introduction to Atmospheric Machine Learning Benchmarking System](#introduction-to-atmopsheric-machine-learning-benchmarking-system)
- [Prepare your dataset](#prepare-your-dataset)
    + [Access the ERA5 dataset (~TB)](#access-the-era5-dataset---tb-)
    + [Dry run with small samples (~15 GB)](#dry-run-with-small-samples---15-gb-)
    + [Climatological mean data](#climatological-mean-data)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  * [Get NVIDIA's TF1.15 container](#get-nvidia-s-tf115-container)
- [Start with AMBS](#start-with-ambs)
  * [Set-up the virtual environment](#set-up-the-virtual-environment)
    + [On JSC's HPC-system](#on-jsc-s-hpc-system)
    + [On other HPC systems](#on-other-hpc-systems)
      - [Case I - Usage of singularity TF1.15 container](#case-i---usage-of-singularity-tf115-container)
      - [Case II - Usage of singularity TF1.15 container](#case-ii---usage-of-singularity-tf115-container)
      - [Further details on the arguments](#further-details-on-the-arguments)
    + [Other systems](#other-systems)
      - [Case I - Usage of singularity TF1.15 container](#case-i---usage-of-singularity-tf115-container-1)
      - [Case II - Usage of singularity TF1.15 container](#case-ii---usage-of-singularity-tf115-container-1)
      - [Further details](#further-details)
  * [Run the workflow](#run-the-workflow)
    + [Create specific runscripts](#create-specific-runscripts)
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


#### Access the ERA5 dataset (~TB)
The experiments described in the GMD paper rely on the ERA5 dataset from which 13 years are used for the dataset of the video prediction models (training, validation and test datasets).

- For users of JSC's HPC system: Access to the ERA5 dataset is possible via the data repository [meteocloud](https://datapub.fz-juelich.de/slcs/meteocloud/). The corresponding path the grib-data files (used for data extraction, see below) is: `/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib`. If you meet access permission issues, please contact: Stein, Olaf <o.stein@fz-juelich.de>

-  For other users (also on other HPC-systems): You can retrieve the ERA5 data from the [ECMWF MARS archive](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-DataorganisationandhowtodownloadERA5). Once you have access to the archive, the data can be downloaded by specifying a resolution of 0.3° in the retrieval script (keyword "GRID", see [here](https://confluence.ecmwf.int/pages/viewpage.action?pageId=123799065)).  The variable names and the corresponding paramID can be found in the ECMWF documentaation website [ERA5 documentations](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Howtoacknowledge,citeandrefertoERA5). For further informations on the ERA5 dataset, please consult the [documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation) provided by ECMWF.

We recommend the users to store the data following the directory structure for the input data described [below](#Input-and-Output-folder-structure-and-naming-convention).

#### Dry run with small samples (~15 GB)

In our application, the typical use-case is to work on a large dataset. Nevertheless, we also prepared an example dataset (1 month data in 2007, 2008, 2009 respectively data with few variables) to help users to run tests on their own machine or to do some quick tests. The data can be downloaded by requesting from Bing Gong <b.gong@fz-juelich.de>. Users of the deepacf-project at JSC can also access the files from `/p/project/deepacf/deeprain/video_prediction_shared_folder/GMD_samples`.


#### Climatological mean data

To compute anomaly correlations in the postprocessing step (see below), climatological mean data is required. This data constitutes the climatological mean for each daytime hour and for each month for the period 1990-2019. 
For convenince, the data is also provided with our frozon version of code and can be downloaded from [zenodo-link!!]().


## Prerequisites
- Linux or macOS
- Python>=3.6
- NVIDIA GPU + CUDA CuDNN or CPU (small dataset only)
- MPI
- Tensorflow 1.13.1 or [CUDA-enabled NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/overview.html#overview) TensorFlow 1.15 within a (singularity)[https://sylabs.io/guides/3.5/user-guide/quick_start.html] container  
- [CDO](https://code.mpimet.mpg.de/projects/cdo/embedded/index.html) >= 1.9.5

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

### Get NVIDIA's TF1.15 container

In case, your HPC-system allows for the usage of singularity containers (such as JSC's HPC-system does) or if you have a NVIDIA GPU available, you can run the workflow with the help of NVIDIA's TensorFlow 1.15-containers. Note that this is the recommended approach!
To get the correct container version, check your NVIDIA driver with the help of `nvidia-smi`. Then search [here](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/index.html) for a suitable container version (try to get the latest possible container ) and download the singularity image via

```
singularity pull <path_to_image>/nvidia_tensorflow_<version>-tf1-py3.sif docker://nvcr.io/nvidia/tensorflow:<version>-tf1-py3
```
where `<version>` is set accordingly. Ensure that your current target directory (`<path_to_image>`) offers enough memory. The respective images are about 3-5 GB large. 
Then create a symbolic link of the singularity container into the `HPC_scripts` and `no_HPC_scripts`-directory, respectively:
```
ln -s <path_to_image>/nvidia_tensorflow_<version>-tf1-py3.sif HPC_scripts/tensorflow_<version>-tf1-py3.sif
ln -s <path_to_image>/nvidia_tensorflow_<version>-tf1-py3.sif no_HPC_scripts/tensorflow_<version>-tf1-py3.sif
```
Note the slightly different name used for the symbolic link which is recommended to easily distinguish between the original file and the symbolic link.

For users with access to JSC's HPC-system: The required singularity image is available from `ambs/video_prediction_tools/HPC_scripts`. Thus, simply set `<path_to_image>` accordingly in the commands above.
Note that you need to log in [Judoor account]https://judoor.fz-juelich.de/login) and specifically request access to restricted container software beforehand!

In case, your operating system supports TF1.13 (or TF1.15) with GPU-support and does not allow for usage of NVIDIA's singularity containers, you can set your environment up as described below.


## Start with AMBS

### Set-up the virtual environment

The workflow can be set-up on different operating systems. The related virtual environment can be set up with the help of the `create_env.sh`-script under the `env_setup`-directory. 
This script will place all virtual environments under the `virtual_envs`-directory.
Depending on your system, you may do the following:

#### On JSC's HPC-system
After linking the TF1.15 singularity container in the directories for the runscript (see previous step), simply run
```
source create_env.sh <my_virtual_env>
```
where `<my_virtual_env>` corresponds to a user-defined name of the virtual environment.

By default, the script assumes that all data (input and preprocessed data as well as trained models and data from postprocessing) will be stored in the shared directory `/p/project/deepacf/deeprain/video_prediction_shared_folder/`. This directory is called 'base-directory' in the following.

In case that you (need to) deviate from this, you can set a customized base-directory. For this, add the `-base_dir`-flag to the call of `create_env.sh`, i.e.:
```
source create_env.sh <my_virtual_env> -base_dir=<my_target_dir>
```
**Note:** Suifficient read-write permissions and a reasonable amount of memory space are mandatory for alternative base-directories.


#### On other HPC systems
On other HPC-systems, the AMBS workflow can also be run. The runscripts under `HPC_scripts` can still be used provided that your HPC-system uses SLURM for managing jobs. Otherwise, you may try to use the runscripts under `no_HPC_scripts` or set-up own runscripts based on your operating system.

##### Case I - Usage of singularity TF1.15 container

After retrieving a singlualrity container that fits your operating HPC-system (see [above](#get-nVIDIA's-tF1.15-container)), create a virtual environment as follows:
```
source create_env.sh <my_virtual_env> -base_dir=<my_target_dir> -tf_container=<used_container>
```
Further details on the arguments are given after Case II.

##### Case II - Usage of singularity TF1.15 container
In case that running singularity containers is not possible for you, but your operating HPC-system provides the usage of TF 1.13 (or later) via modules, the source-code can still be run.
However, this requires you to populate `modules_train.sh` where all modules are listed. Note that you also need to load modules for opening and reading h5- and netCDF-files as well . Afterwards, the virtual environment can be created by
```
source create_env.sh <my_virtual_env> -base_dir=<my_target_dir> -l_nocontainer 
```

##### Further details on the arguments
In the set-up commands for the virtual environment mentioned above, `<my_virual_env>` corresponds to the user-defined name of the virtual environment.`<my_target_dir>` points to an (existing) directory which offers enough memory to store large amounts of data (>>100 GB)
This directory should also already hold the ERA5-data as described [above](#Access-the-ERA5-dataset-(~TB)). Besides, the basic directory tree for the output of the workflow steps should follow the description provided [here]((#Input-and-Output-folder-structure-and-naming-convention)).
The argument `-tf_container=<used_container>` allows you to specify the used singularity container (in Case I only!). Thus, `used_container` should correspond to `tensorflow_<version>-tf1-py3.sif` as described in this [section](#Get-NVIDIA's-TF1.15-container) above.

#### Other systems
On other systems with access to a NVIDIA GPU, the virtual environment can be run as follows.
In case that you don't have access to a NVIDIA GPU, you can still run TensorFlow on your CPU. However, training becomes very slow then and thus, we recommend to just test with the small dataset mentioned [above](#dry-run-with- small-samples-(~15-GB)). 

Again, we describe the step to set-up the virtual environment separately in the following.

##### Case I - Usage of singularity TF1.15 container

After retrieving a singlualrity container that fits your operating machine (see [above](#Get-NVIDIA's-TF1.15-container)), create a virtual environment as follows:
```
source create_env.sh <my_virtual_env> -base_dir=<my_target_dir> -l_nohpc
```
Further details on the arguments are given after Case II.

##### Case II - Usage of singularity TF1.15 container

Without using a singularity container (and using your CPU instead), please run 
```
source create_env.sh <my_virtual_env> -base_dir=<my_target_dir> -l_nocontainer -l_nohpc
```
**Note:** To reproduce the results of GMD paper, we recommend to use the case II. 

##### Further details 
Futher details on the used arguments are provided [above](#Further-details-on-the-arguments). The only exception holds for the `l_nohpc`-flag that is used to indicate that you are not running on a HPC-system.


### Run the workflow

Depending on the computing system you are working on, the workflow steps will be invoked by dedicated runscripts either from the directory `HPC_scripts/` or from `no_HPC_scripts`. The used directory names are self-explanatory.

To help the users conduct different experiments with varying configurations (e.g. input variables, hyperparameters etc), each runscript can be set up conveniently with the help of the Python-script `generate_runscript.py`. Its usage as well the workflow runscripts are described subsequently. 


#### Create specific runscripts

Specific runscripts for each workflow substep (see below) are generated conveniently by keyboard interaction.

The interactive Python script under the folder `generate_runscript.py` thereby has to be executed after running `create_env.sh`. Note that this script only creates a new virtual environment if `<env_name>` has not been used before. If the corresponding virtual environment is already existing, it is simply activated. 

After prompting 

```bash
python generate_runscript.py --venv_path <env_name>
```
you will be asked first which workflow runscript shall be generated. You can choose one of the following workflow step names: 
- extract
- preprocess1
- preprocess2
- train
- postprocess 

The subsequent keyboard interaction then allows the user to make individual settings to the workflow step at hand. By pressing simply Enter, the user may receive some guidance for the keyboard interaction. 

Note that the runscript creation of later workflow substeps depends on the preceding steps (i.e. by checking the arguments from keyboard interaction).
Thus, they should be created sequentially instead of all at once at the beginning! 


**NoteI**:  The runscript creation depends on the preceding steps (i.e. by checking the arguments from keyboard interaction).
Thus, they should be created sequentially instead of all at once at the beginning! Note that running the workflow step is also mandatory, before the runscript for the next workflow step can be created.

**Note II**: Remember to enable your virtual environment before running `generate_runscripts.py`. For this, you can simply run 
```
source create_env.sh <env_name>
```
where `<env_name>` corresponds to

### Running the workflow substeps 

Having created the runscript by keyboard interaction, the workflow substeps can be run sequentially. 

Note that you have to adapt the `account`, the `partition` as well as the e-mail address in case you running on a HPC-system other than JSC's HPC-systems (HDF-ML, Juwels Cluster and Juwels Booster).

Now,  it is time to run the AMBS workflow
1. **Data Extraction**:<br> This script retrieves the demanded variables for user-defined years from complete ERA% reanalysis grib-files and stores the data into netCDF-files.
```bash
[sbatch] ./data_extraction_era5.sh
```

2. **Data Preprocessing**:<br> Crop the ERA 5-data (multiple years possible) to the region of interest (preprocesing step 1). All the year data will be touched once and the statistics are calculated and saved in the output folder. The TFrecord-files which are fed to the trained model (next workflow step) are created afterwards. Thus, two cases exist at this stage:

    ```bash
    [sbatch] ./preprocess_data_era5_step1.sh
    [sbatch] ./preprocess_data_era5_step2.sh
    ```

3. **Training**:<br> Training of one of the available models with the preprocessed data. Note that the `exp_id` is generated automatically when running `generate_runscript.py`.

    ```bash
    [sbatch] ./train_model_era5_<exp_id>.sh
    ```
    
4. **Postprocessing**:<br> Create some plots and calculate the evaluation metrics for test dataset. Note that the `exp_id` is generated automatically when running `generate_runscript.py`.

    ```bash
    [sbatch] ./visualize_postprocess_era5_<exp_id>.sh
    ```

### Compare and visualize the results

AMBS also provides the tool (called meta-postprocessing) for the users to compare different experiments results and visualize the results as shown in GMD paper through the`meta_postprocess`-step. The runscript template are also prepared in the `HPC_scripts`, `no_HPC_scripts`. 

### Input and Output folder structure and naming convention
To successfully run the workflow and enable tracking the results from each workflow step, inputs and output directories, and the file name convention should be constructed as described below:

Below, we show at first the input data structure for the ERA5 dataset. In detail, the data is recorded hourly and stored into two different kind of grib files. The file with suffix `*_ml.grb` consists of multi-layer data, whereas `*_sf.grb` only includes the surface data.

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

The root output directory should be set up when you run the workflow at the first time as aformentioned. 

The output structure for each step of the workflow along with the file name convention are described below:
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
Currently, the workflow includes the following ML architectures, and we are working on integrating more into the system.
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

