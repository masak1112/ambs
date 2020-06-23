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

### Set-up env on JUWELS and ZAM347

- Set up env and install packages

```bash
cd video_prediction_savp
source env_setup/create_env.sh <dir_name> <env_name>
```

### Run on ZAM347


### Recomendation for output folder structure and name convention
The details can be found [name_convention](docs/structure_name_convention.md)

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

### Benchmarking architectures:

- convLSTM: [paper](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf),[code])https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow)
- Variational Autoencoder:[paper](https://arxiv.org/pdf/1312.6114.pdf)
- Stochastic Adversarial Video Prediction (SAVP): [paper](https://arxiv.org/pdf/1804.01523.pdf),[code](https://github.com/alexlee-gk/video_prediction) 
- Motion and Content Network (MCnet): [paper](https://arxiv.org/pdf/1706.08033.pdf), [code](https://github.com/rubenvillegas/iclr2017mcnet)



### Contact

- Amirpash Mozafarri: a.mozafarri@fz-juelich.de
- Michael Langguth: m.langguth@fz-juelich.de
- Bing Gong: b.gong@fz-juelich.de
- Scarlet Stadtler: s.stadtler@fz-juelich.de 
