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
cd Video_Prediction_SAVP
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
- You may need install packages by pip on JUWELS/JURECA, followed the installation instruction from [Workflow project](https://gitlab.version.fz-juelich.de/gong1/workflow_parallel_frame_prediction) 
### Download data

- Download the ERA5 data (.hkl) from the output of DataPreprocess in the [Workflow project](https://gitlab.version.fz-juelich.de/gong1/workflow_parallel_frame_prediction)
```bash
bash data/download_and_preprocess_dataset_era5.sh --data era5 --input_dir /splits --output_dir  data/era5 
```

### Model Training
```python
python scripts/train.py --input_dir data/era5 --dataset era5  --model savp --model_hparams_dict hparams/kth/ours_savp/model_hparams.json --output_dir logs/era5/ours_savp
```

### Model Evaluation

![Groud Truth](./results_test_samples/era5_size_64_64_1_v2/our_savp/groud_true_images_0.mp4)

