# Video Prediction by GAN

This project aims to adopt the GAN-based architectures,  which original proposed by [[Project Page]](https://alexlee-gk.github.io/video_prediction/) [[Paper]](https://arxiv.org/abs/1804.01523), to predict temperature based on ERA5 data
 
## Getting Started ###
### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation 
This project need to work with [Workflow_parallel_frame_prediction project](https://gitlab.version.fz-juelich.de/gong1/workflow_parallel_frame_prediction)
- Clone this repo:
```bash
git clone master https://gitlab.version.fz-juelich.de/gong1/video_prediction_savp.git
git clone master https://gitlab.version.fz-juelich.de/gong1/workflow_parallel_frame_prediction.git
```

### Set-up env on JUWELS

- Set up env and install packages

```bash
cd video_prediction_savp
source env_setup/create_env.sh <dir_name> <env_name>
```

## Workflow by steps

### Data Extraction

```python
python3 ../workflow_video_prediction/DataExtraction/mpi_stager_v2.py  --source_dir <input_dir1> --destination_dir <output_dir1>
```

e.g. 
```python
python3 ../workflow_video_prediction/DataExtraction/mpi_stager_v2.py  --source_dir /p/fastdata/slmet/slmet111/met_data/ecmwf/era5/nc/2017/ --destination_dir /p/scratch/deepacf/bing/extractedData
```

### Data Preprocessing
```python
python3 ../workflow_video_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py --source_dir <output_dir1> --destination_dir <output_dir2> 

python3 video_prediction/datasets/era5_dataset_v2.py  --source_dir   <output_dir2> --destination_dir <.data/exp_name>
```

Example
```python
python3 ../workflow_video_prediction/DataPreprocess/mpi_stager_v2_process_netCDF.py --source_dir /p/scratch/deepacf/bing/extractedData --destination_dir /p/scratch/deepacf/bing/preprocessedData

python3 video_prediction/datasets/era5_dataset_v2.py /p/scratch/deepacf/bing/preprocessedData  ./data/era5_64_64_3_3t_norm
 ```
 
### Trarining

```python
python3 scripts/train_v2.py --input_dir <./data/exp_name> --dataset era5  --model <savp> --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir <./logs/{exp_name}/{mode}/>
```

Example
```python
python3 scripts/train_v2.py --input_dir ./data/era5_size_64_64_3_3t_norm --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5_64_64_3_3t_norm/end_to_end
```
### Postprocessing

Generating prediction frames, model evaluation, and visulization
You can trained your own model from the training step , or you can copy the Bing's trained model

```python
python3 scripts/generate_transfer_learning_finetune.py --input_dir <./data/exp_name>  --dataset_hparams sequence_length=20 --checkpoint <./logs/{exp_name}/{mode}/{model}> --mode test --results_dir <./results/{exp_name}/{mode}>  --batch_size <batch_size> --dataset era5
```

- example: use end_to_end training model from bing for exp_name:era5_size_64_64_3_3t_norm
```python
python3 scripts/generate_transfer_learning_finetune.py --input_dir data/era5_size_64_64_3_3t_norm --dataset_hparams sequence_length=20 --checkpoint /p/project/deepacf/deeprain/bing/video_prediction_savp/logs/era5_size_64_64_3_3t_norm/end_to_end/ours_savp --mode test --results_dir results_test_samples/era5_size_64_64_3_3t_norm/end_to_end  --batch_size 4 --dataset era5
```

![Groud Truth](/results_test_samples/era5_size_64_64_3_norm_dup/ours_savp/Sample_Batch_id_0_Sample_1.mp4)
# End-to-End run the entire workflow

```bash
./bash/workflow_era5.sh <model>  <train_mode>  <exp_name>
```

example:
```bash
./bash/workflow_era5.sh savp end_to_end  era5_size_64_64_3_3t_norm
```



### Recomendation of output folder structure and name convention

```
├── ExtractedData
│   ├── [Year]
│   │   ├── [Month]
│   │   │   ├── **/*.netCDF
├── PreprocessedData
│   ├── [Data_name_convention]
│   │   ├── hickle
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