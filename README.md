# Video Prediction by GAN

This project aims to adopt the GAN-based architectures,  which original proposed by [[Project Page]](https://alexlee-gk.github.io/video_prediction/) [[Paper]](https://arxiv.org/abs/1804.01523), to predict temperature based on ERA5 data
 
## Getting Started ###
### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation 
- Clone this repo:
```bash
git clone -b master https://gitlab.version.fz-juelich.de/gong1/video_prediction_savp.git
cd video_prediction_savp
```
- Install TensorFlow >= 1.9 and dependencies from http://tensorflow.org/
- Install other dependencies

```bash
pip install -r requirements.txt
```

### Miscellaneous installation considerations
- In python >= 3.6, make sure to add the root directory to the PYTHONPATH`, e.g. `export PYTHONPATH=path/to/video_prediction_savp`.
- For the best speed and experimental results, we recommend using cudnn version 7.3.0.29 and any tensorflow version >= 1.9 and <= 1.12. The final training loss is worse when using cudnn versions 7.3.1.20 or 7.4.1.5, compared to when using versions 7.3.0.29 and below.
- Add the directories lpips-tensorflow and hickle (get from [Workflow project](https://gitlab.version.fz-juelich.de/gong1/workflow_parallel_frame_prediction) to the  `PATHONPATH `, e.g export PYTHONPATH=path/to/lpips-tensorflow


### Set-up on JUWELS

- Set up env and install packages

```bash
cd env_setup
./create_env.sh <USER_FOLDER>
```

## Workflow by steps


### Data Extraction

[Workflow project](https://gitlab.version.fz-juelich.de/gong1/workflow_parallel_frame_prediction)

```bash
cd ../workflow_video_prediction/DataExtraction 
python mpi_stager_v2.py  --source_dir <input_dir1> --destination_dir <output_dir1>
```

### Data Preprocessing
```bash
cd  ../workflow_video_prediction/DataPreprocess
python mpi_stager_v2_process_netCDF.py --source_dir <output_dir1> --destination_dir <output_dir2> 
```

```python
video_prediction/datasets/era5_dataset_v2.py <output_dir2/splits>  <output_dir3>
```

```python
python scripts/train_v2.py --input_dir <output_dir3> --dataset era5  --model <savp> --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir <logs/era5/ours_savp>
```
### Postprocessing


### Model Evaluation

![Groud Truth](/results_test_samples/era5_size_64_64_3_norm_dup/ours_savp/Sample_Batch_id_0_Sample_1.mp4)
# End-to-End run the entire workflow

```bash
cd bash
./workflow_era5.sh
```


