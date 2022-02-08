
<img src="ambs_logo.jpg" width="1000" height="400">

[Check our video](https://www.youtube.com/watch?v=Tf2BDDlSDeQ)


## Table of Contents  

- [AMBS](#ambs)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Set-up environment on Jülich's HPC systems or other computing systems](#set-up-environment-on-j-lich-s-hpc-systems-or-other-computing-systems)
  * [Run the workflow](#run-the-workflow)
    + [Preparation](#preparation)
    + [Create specific runscripts](#create-specific-runscripts)
    + [Running the workflow substeps](#running-the-workflow-substeps)
  * [Running the training in NVIDIA's TF1.15 singularity containers](#running-the-training-in-nvidia-s-tf115-singularity-containers)
  * [Output folder structure and naming convention](#output-folder-structure-and-naming-convention)
  * [Benchmarking architectures:](#benchmarking-architectures-)
  * [Contact](#contact)


 
## AMBS 
**A**tmopsheric **M**achine learning **B**enchmarking **S**ystem (AMBS)
 aims to provide state-of-the-art video prediction methods applied to the meteorological domain.
In the scope of the current application, the hourly evolution of the 2m temperature
over a used-defined region is focused. <br>
Different Deep Learning video prediction architectures such as convLSTM and SAVP
are trained with ERA5 reanalysis to perform a prediction for 12 hours based on the previous 12 hours.
In addition to the 2m temperature (t2m) itself, other variables can be fed to the video frame prediction models to enhance their capability to learn the complex physical processes driving the diurnal cycle of temperature.
Currently, the recommended additional meteorological variables are the 850 hPa temperature (t850) and the total cloud cover (tcc).
Besides, training on other standard video frame prediction datasets (such as MovingMNIST) can be prerformed.

The project is currently developed by Bing Gong, Michael Langguth, Amirpasha Mozafarri, Yan Ji and Karim Mache.<br>
Former code developers are Scarlet Stadtler and Severin Hussmann.


### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- MPI
- Tensorflow 1.13.1 or CUDA-enabled NVIDIA TensorFlow 1.15 within a singularity container 
- CDO >= 1.9.5

### Installation 

Clone this repo by typing the following command in your personal target dirctory:
```bash 
git clone https://gitlab.jsc.fz-juelich.de/esde/machine-learning/ambs.git
```
This will create a directory called `ambs` under which this README-file and 
two subdirectories are placed. The subdirectory `[...]/ambs/test/` contains unittest-scripts for
the workflow and is therefore of minor relevance for non-developers.
The subdirectory `[...]/ambs/video_prediction_tools` contains everything which is needed in the workflow and is
therefore called the top-level directory in the following.

Thus, change into this subdirectory after cloning:
```bash 
cd ambs/video_preditcion_tools/
```

### Set-up environment on Jülich's HPC systems or other computing systems

The following commands will setup a customized virtual environment
either on a known HPC-system at JSC (Juwels, Juwels Booster or HDF-ML). Setting up a virtual environment on other computing systems, e.g. the personal computer, is currently not supported, but targeted for the future. 
The script `create_env.sh` automatically detects on which machine it is executed and loads/installs
all required Python (binary) modules and packages.
The virtual environment is set up in the top-level directory (`[...]/ambs/video_prediction_tools`)
under a subdirectory which gets the name of the virtual environment.

```bash
cd env_setup
source create_env.sh <env_name> 
```
This also already sets up the runscript templates for you. By default, the runscript templates make use of the standard target base directory `/p/project/deepacf/deeprain/video_prediction_shared_folder/`. In case that you want to deviate from this, you may call `create_env.sh` as follows:
```bash
source create_env.sh <env_name> -base_dir=<my_target_dir>
```
Note that suifficient read-write permissions and a reasonable amount of memory space is mandatory for your alternative standard output directory.

### Run the workflow

Depending on the computing system you are working on, the workflow steps will be invoked
by dedicated runscripts either from the directory `HPC_scripts/` (on known HPC-systems, see above) or from
the directory `nonHPC_scripts/` (else, but not implemented yet).<br>
Each runscript can be set up conveniently with the help of the Python-script `generate_runscript.py`.
Its usage as well the workflow runscripts are described subsequently.

#### Preparation 

In case that a virtual environment has already been set up beforehand, the script `create_env.sh` can also be used for activating it. For this, simply do 
```bash
cd env_setup
source create_env.sh <env_name> 
```
similarly to the creation process of the virtual environment (see above). Note that this also loads some modules on the HPC-systems that are required for the runscript generator (see below). <br>
**Remark:** In case you want to change the target base directory _after_ having created the virtual environment, 
you may run `setup_runscript_templates.sh` manually (the `script create_env.sh` does this internally):
```bash
source <top_level_dir>/utils/runscript_generator/setup_runscript_templates.sh <my_target_dir>
```

#### Create specific runscripts

Specific runscripts for each workfow substep (see below) are generated conventiently by keyboard interaction.
The interactive Python-script thereby has to be executed in an activated virtual environment with some addiional modules!
After prompting 
```bash
python generate_runscript.py
```
you will be asked first which workflow runscript shall be generated. You can chose one of the workflow step name: 
- extract
- preprocess1
- preprocess2
- train
- postprocess 

The subsequent keyboard interactions then allow  the user to make individual settings to the workflow step at hand. 
By pressing simply Enter, the user may receive some guidance for the keyboard interaction. <br>
Note that the runscript creation of later workflow substeps depends on the preceding steps (i.e. by checking the arguments from keyboard interaction).
Thus, they should be created sequentially instead of all at once at the beginning. 

#### Running the workflow substeps 
Having created the runscript by keyboard interaction, the workflow substeps can be run sequentially.
Depending on the machine you are working on, change either to `HPC_scripts/` (on Juwels, Juwels Booster or HDF-ML) or to 
`nonHPC347_scripts/` (not implemented yet).
There, the respective runscripts for all steps of the workflow are located 
whose order is as follows. Note that `[sbatch]` only has to precede on one of the HPC systems.
Besides data extraction and preprocessing step 1 are onyl mandatory when ERA5 data is subject to the application.

1. Data Extraction: This script retrieves the demanded variables for user-defined years from complete ERA% reanalysis grib-files and stors the data into netCDF-files.
```bash
[sbatch] ./data_extraction_era5.sh
```
2. Data Preprocessing: Crop the ERA 5-data (multiple years possible) to the region of interest (preprocesing step 1),
The TFrecord-files which are fed to the trained model (next workflow step) are created afterwards.
This is also the place where other datasets such as the MovingMNIST (link?) can be prepared.  
Thus, two cases exist at this stage:
    * **ERA 5 data**
    ```bash
    [sbatch] ./preprocess_data_era5_step1.sh
    [sbatch] ./preprocess_data_era5_step2.sh
    ```
    * **MovingMNIST data**
    ```bash
    [sbatch] ./preprocess_data_moving_mnist.sh
    ```
3. Training: Training of one of the available models with the preprocessed data. <br>
Note that the `exp_id` is generated automatically when running `generate_runscript.py`.
    * **ERA 5 data**
    ```bash
    [sbatch] ./train_model_era5_<exp_id>.sh
    ```
    * **MovingMNIST data**
    ```bash
    [sbatch] ./train_model_moving_mnist_<exp_id>.sh
    ```
4. Postprocess: Create some plots and calculate the evaluation metrics for test dataset. <br>
Note that the `exp_id` is generated automatically when running `generate_runscript.py`.
    * **ERA 5 data**
    ```bash
    [sbatch] ./visualize_postprocess_era5_<exp_id>.sh
    ```
    * **MovingMNIST data**
    ```bash
    [sbatch] ./visualize_postprocess_moving_mnist_<exp_id>.sh
    ```

### Running the training in NVIDIA's TF1.15 singularity containers ###

**The following documentation is deprecated and must be updated.** <br><br>

The computionally expensive training of the Deep Learning video prediction architectures is supposed to benefit from the Booster module which is installed at JSC in autumn 2020. In order to test the potential speed-up on this state-of-the-art HPC system, optimized for massively parallel workloads, we selected the convLSTM-architecture as a test candidate in the scope of the Juwels Booster Early Access program.

To run the training on the Booster, change to the recent Booster related working branch of this git repository.

```bash 
git checkout scarlet_issue#031_booster
```

Currently, there is no system installation of TensorFlow available on Juwels Booster. As an intermediate solution,
a singularity container with a CUDA-enabled NVIDIA TensorFlow v1.15 was made available which has to be reflected when setting up the virtual environment and when submiiting the job. 
The corresponding singularity image file is shared under `/p/project/deepacf/deeprain/video_prediction_shared_folder/containers_juwels_booster/` and needs to be linked to the `HPC_scripts`-directory. If you are not part of the *deepacf*-project, the singularity file can be accessed alternatively from 

```bash
cd video_prediction_tools/HPC_scripts
ln -sf /p/project/deepacf/deeprain/video_prediction_shared_folder/containers_juwels_booster/tf-1.15-ofed-venv.sif tf-1.15-ofed-venv.sif 
```

Before setting up the virtual environment for the workflow, some required modules have to be loaded and an interactive shell has to be spawn within the container.

```bash
cd ../env_setup
source modules_train.sh 
cd ../HPC_scripts/
singularity shell tf-1.15-ofed-venv.sif
```

Now, the virtual environment can be created as usual, i.e. by doing the following steps within the container shell:

```bash
cd ../env_setup
source create_env.sh <env_name> [<exp_id>]
```

This creates a corresponding directory `video_prediction_tools/<env_name>`.

Note that `create_env.sh` is *system-aware* and therefore only creates an executable runscript for the training step,
e.g. `video_prediction_tools/HPC_scripts/train_model_era5_booster_exp1.sh`. 
Since all other substeps of the workflow are supposed to be performed on other systems so far, the data in-and output directories cannot be determined automatically during the workflow.
Thus, the user has to modify the corresponding runscript by adapting manually the script varibales `source_dir` and `destination_dir` in line 19+20 of `train_model_era5_booster_<exp_id>.sh`. 

Finally, the training job can be submitted to Juwels Booster after exiting from the cotainer instance by 

``` bash
exit 
cd ../HPC_scripts
salloc --partition=booster --nodes=1 --account=ea_deepacf --time=00:10:00 --reservation  earlyaccess srun  singularity exec --nv tf-1.15-ofed-venv.sif  ./train_model_era5_booster_<exp_id>.sh
```


### Output folder structure and naming convention
The details can be found [name_convention](docs/structure_name_convention.md)

```
├── ExtractedData
│   ├── [Year]
│   │   ├── [Month]
│   │   │   ├── **/*.netCDF
├── PreprocessedData
│   ├── [Data_name_convention]
│   │   ├── pickle
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── tfrecords
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
├── Models
│   ├── [Data_name_convention]
│   │   ├── [model_name]
│   │   ├── [model_name]
├── Results
│   ├── [Data_name_convention]
│   │   ├── [training_mode]
│   │   │   ├── [source_data_name_convention]
│   │   │   │   ├── [model_name]

```

### Benchmarking architectures:

- convLSTM: [paper](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf),[code](https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow)
- Variational Autoencoder:[paper](https://arxiv.org/pdf/1312.6114.pdf)
- Stochastic Adversarial Video Prediction (SAVP): [paper](https://arxiv.org/pdf/1804.01523.pdf),[code](https://github.com/alexlee-gk/video_prediction) 
- Motion and Content Network (MCnet): [paper](https://arxiv.org/pdf/1706.08033.pdf), [code](https://github.com/rubenvillegas/iclr2017mcnet)



### Contact

- Amirpash Mozafarri: a.mozafarri@fz-juelich.de
- Michael Langguth: m.langguth@fz-juelich.de
- Bing Gong: b.gong@fz-juelich.de
- Scarlet Stadtler: s.stadtler@fz-juelich.de 
