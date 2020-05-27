#!/bin/bash -x


python ../video_prediction/datasets/era5_dataset_v2.py /home/${USER}/preprocessedData/era5-Y2017M01to02-128x160-74d00N71d00E-T_MSL_gph500/hickle/splits/ /home/${USER}/preprocessedData/era5-Y2017M01to02-128x160-74d00N71d00E-T_MSL_gph500/tfrecords/ -vars T2 MSL gph500 -height 128 -width 160 -seq_length 20 
