This is the output folder structure and name convention

## Shared folder structure

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

| Arguments	| Value	|
|---	|---	|
| [Year]	| 2005;2005;2007 ...|
| [Month]	| 01;02;03 ...,12|
|[Data_name_convention]|Y[yyyy]to[yyyy]M[mm]to[mm]-[nx]_[ny]-[nn.nn]N[ee.ee]E-[var1]_[var2]_[var3]|
|[model_name]| Ours_savp;  ours_gan;  ours_vae; prednet|
|[training_mode]|end_to_end; transfer_learning|


## Data name convention

`Y[yyyy]to[yyyy]M[mm]to[mm]-[nx]_[ny]-[nn.nn]N[ee.ee]E-[var1]_[var2]_[var3]`


### `Y[yyyy]to[yyyy]M[mm]to[mm]`

| Examples	| Name abbrevation 	|
|---	|---	|
|all data from March to June of the years 2005-2015	| Y2005toY2015M03to06 |   
|data from February to May of years 2005-2008 + data from March to June of year 2015| Y2005to2008M02to05_Y2015M03to06 |   
|Data from February to May, and October to December of 2005 |  Y2005M02to05_Y2015M10to12 |   
|operational’ data base: whole year 2016 |  Y2016M01to12 |   
|add new whole year data of 2017 on the operational data base |Y2016to2017M01to12 |  
| Note: Y2016to2017M01to12 = Y2016M01to12_Y2017M01to12|  



### variable abbrevaition and the corresponding full names

| var	| full  names 	|
|---	|---	|
|T|2m temperature|   
|gph|500 hPa geopotential|   
|msl|meansealevelpressure|   



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

