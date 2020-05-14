#!/bin/bash -x


python ../video_prediction/datasets/era5_dataset_v2.py /home/${USER}/preprocessedData/era5-Y2015toY2017M01to12-128x160-74d00N71d00E-T_MSL_gph500/hickle/splits/ /home/${USER}/preprocessedData/era5-Y2015toY2017M01to12-128x160-74d00N71d00E-T_MSL_gph500/tfrecords/ -vars T2 T2 T2 
