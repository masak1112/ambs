#!/bin/bash -x


python ../video_prediction/datasets/era5_dataset_v2.py /home/${USER}/preprocessedData/era5-Y2015toY2017M01to12-64x64-74d00N71d00E-T_MSL_gph500/hickle/splits/ /home/${USER}/preprocessedData/era5-Y2015toY2017M01to12-64x64-74d00N71d00E-T_MSL_gph500/tfrecords/ -vars T2 MSL gph500 -height 64 -width 64 -seq_length 20 
