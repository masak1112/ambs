
##Shared folder structure

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

### Example


```
├── ExtractedData
│   ├── 2016
│   │   ├── 01
│   │   │   ├── *.netCDF
│   │   ├── 02
│   │   ├── 03
│   │   ├── …
│   ├── 2017
│   │   ├── 01
│   │   ├── …
├── PreprocessedData
│   ├── 2016to2017M01to12-64_64-50.00N11.50E-T_T_T
│   │   ├── hickle
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── tfrecords
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
├── Models
│   ├── 2016to2017M01to12-64_64-50.00N11.50E-T_T_T
│   │   ├── outs_savp
│   │   ├── outs_gan
├── Results
│   ├── 2016to2017M01to12-64_64-50.00N11.50E-T_T_T
│   │   ├── end_to_end
│   │   │   ├── ours_savp
│   │   │   ├── ours_gan
│   │   ├── transfer_learning
│   │   │   ├── 2018M01to12-64_64-50.00N11.50E-T_T_T
│   │   │   │   ├── ours_savp
```

