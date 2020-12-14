# AMBS

Atmopsheric Machine learning Benchmarking Systems (AMBS) aims to privde state-of-the-art benchmarking machine learning architectures for video prediction on HPC in the context of atmospheric domain, which is developed by Amirpasha, Michael, Bing, and Scarlet


### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- MPI
- Tensorflow 1.13.1

### Installation 

- Clone this repo:
```bash 
git clone https://gitlab.version.fz-juelich.de/toar/ambs.git
```

### Set-up env on Jülich's HPC systems and zam347

The following commands will setup a user-specific virtual environment
either on Juwels, HDF-ML (HPC clusters) or on zam347 for you.
The script `create_env.sh` automatically detects on which machine it is executed and loads/installs
all required Python (binary) modules and packages.
The virtual environment is set up under the subfolder `video_prediction_savp/<env_name>`.
Besides, user-specific runscripts for each step of the workflow may be created,
e.g. `train_era5_exp1.sh` where `exp1` denotes the default experiment identifier.
The name of this identifier can be controlled by the optional second argument `<exp_id>`.

```bash
cd video_prediction_savp/env_setup
source create_env.sh <env_name> [<exp_id>]
```

### Run workflow the workflow

Depending on the machine you are workin on, change either to 
`video_prediction_savp/HPC_scripts` (on Juwels and HDF-ML) or to 
`video_prediction_savp/Zam347_scripts`.
There, the respective runscripts for all steps of the workflow are located
whose order is the following:


1. Data Extraction: Retrieve ERA5 reanalysis data for one year. For multiple year, execute the runscript sequentially.  
```bash
./DataExtraction_<exp_id>.sh
```

2. Data Preprocessing: Crop all data (multiple years possible) to the region of interest and perform normalization
```bash
./DataPreprocess_<exp_id>.sh
./DataPreprocess2tf_<exp_id>.sh
```

3. Training: Training of one of the available models (see bewlow) with the preprocessed data. 
```bash
./train_era5_<exp_id>.sh
```

4. Postprocess: Create some plots and calculate evaluation metrics for test dataset.
```bash
./generate_era5_<exp_id>.sh
```

### Create additional runscripts ###
In case that you want to perform experiments with varying configuration (e.g. another set of hyperparameters, but still the same input dataset for training), it is convenient to create individual runscripts from the templates. 
This can be done with the help of `generate_workflow_runscripts.sh`. 

The first argument `<runscript_name>` defines the **relative path** to the template runscript
which should be converted to an executable one. Note that only the suffix of the 
template's name must be passed, e.g. `../HPC_scripts/train_era5` in order to create 
a runscript for the training substep.
The second argument `<venv_name>` denotes the name of the virtual environment which has to be set up in advanceand which should be used by the runscript.

Additional optional arguments can be passed to control the experimental identifier and to set manually the realtive path to the 
directory where the preprocessed data is stored (used for the training and postprocessing substep). These optional arguments have to follow a naming convention in order to be identified by `generate_workflow_runscripts.sh`.
The experimental identifer can be passed by adding `-exp_id=<id>`while the path to the preprocessed data requires passing of `-exp_dir=<relative_path_to_dir>`. Note, that the default value `exp1` is used as experimental identifier if the `-exp_id=<id>` is omitted.

``` bash
./generate_workflow_runscripts.sh <runscript_name> <venv_name> [-exp_id=<id>] [-exp_dir=<relative_dir_to_path>]
```

*Specific example:*
``` bash
./generate_workflow_runscripts.sh ../HPC_scripts/train_era5 venv_juwels -exp_id=exp_test -exp_dir=era5-Y2010toY2222M01to12-160x128-2970N1500W-T2_MSL_gph500
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
- Bing Gong (PI): b.gong@fz-juelich.de
- Scarlet Stadtler: s.stadtler@fz-juelich.de 
