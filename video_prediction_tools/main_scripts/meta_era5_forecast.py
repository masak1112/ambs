##########################################################
import os,sys
import datetime as dt
import xarray as xr
import time
import numpy as np
import glob
import json

##########################################################
# forecast_path = "/p/project/deepacf/deeprain/video_prediction_shared_folder/results/era5-Y2007-2019M01to12-80x48-3960N0180E-2t_tcc_t_850_langguth1/savp/20210505T131220_mache1_karim_savp_smreg_cv3_3"
# forecast_path = "/p/scratch/deepacf/ji4/results/era5-Y2007-2019M01to12-92x56-3840N0000E-tp_tcc_clwc_925/savp/20210520T233729_ji4_prcp"
forecast_path = "/p/project/deepacf/deeprain/video_prediction_shared_folder/results/gzprcp_data/convLSTM/20210515T105520_ji4_ji0"
with xr.open_dataset(os.path.join(forecast_path,'vfp_sampleind_1499.nc')) as dfiles:
    data = dfiles.load()

mask_path = "/p/scratch/deepacf/video_prediction_shared_folder/extractedData/guizhou_prcp"
with xr.open_dataset(os.path.join(mask_path,'mask.nc')) as dfiles:
    mask = dfiles.load()

##########################################################
# forecast_path = "/p/project/deepacf/deeprain/video_prediction_shared_folder/results/era5-Y2007-2019M01to12-80x48-3960N0180E-2t_tcc_t_850_langguth1/savp/20210505T131220_mache1_karim_savp_smreg_cv3_3"
# forecast_path = "/p/scratch/deepacf/ji4/results/era5-Y2007-2019M01to12-92x56-3840N0000E-tp_tcc_clwc_925/savp/20210520T233729_ji4_prcp"
forecast_path = "/p/project/deepacf/deeprain/video_prediction_shared_folder/results/gzprcp_data/convLSTM/20210515T105520_ji4_ji0"
# file_name = 'vfp_date_*_sample_ind_10[0-9][0-9].nc'
file_name = 'vfp_sampleind_*.nc'
file_path = os.path.join(forecast_path,file_name)

def non_intetst_vars(ds):
    return [v for v in ds.data_vars
            if v not in vars2proc]

lead_hour = 0
def get_relevant_vars(ds,lead_hour=lead_hour):
    return ds.drop(non_intetst_vars(ds)).isel(time_forecast=lead_hour)
    #return ds.drop(non_intetst_vars(ds)).isel(fcst_hour=lead_hour)

# vars2proc = ['2t_ref','2t_savp_fcst']
# vars2proc = ['tp_ref','tp_savp_fcst']
vars2proc = ['2t_ref','2t_fcst']
time0 = time.time()
# with xr.open_mfdataset(file_path,decode_cf=True,concat_dim=["init_time"],compat="broadcast_equals",preprocess=get_relevant_vars) as dfiles:
with xr.open_mfdataset(file_path,decode_cf=True,concat_dim=["time_forecast"],compat="broadcast_equals",preprocess=get_relevant_vars) as dfiles:
    data = dfiles.load()
    print("Redistering forecast data took {:.2f} seconds".format(time.time()-time0))

# ref,savp_fcst = data['2t_ref'], data['2t_savp_fcst']
# ref,savp_fcst = data['tp_ref'], data['tp_savp_fcst']
ref,fcst = data['2t_ref'], data['2t_fcst']

##########################################################
### simpliy compute forecast time (lead_time<24)
#fcst_time = np.array(data.init_time.dt.hour+lead_hour)
#for i in range(len(fcst_time)):
#    if fcst_time[i] > 23:
#        fcst_time[i] = fcst_time[i]-24

### load global climate t2m data, year1980~2019
#t2m_monthly_dir = '/p/scratch/deepacf/video_prediction_shared_folder/preprocessedData/T2monthly'
#t2m_monthly_filename = '*_t2m.nc'
#t2m_month_path = os.path.join(t2m_monthly_dir,t2m_monthly_filename)
#nyear = len(glob.glob(t2m_month_path))
#time0 = time.time()
#with xr.open_mfdataset(t2m_month_path,decode_cf=True,concat_dim=["time"],compat="broadcast_equals") as mfiles:
#    t2m_months = mfiles.load()
#    print("Redistering climate data took {:.2f} seconds".format(time.time()-time0))

#t2m_lon = t2m_months['lon']
#t2m_lat = t2m_months['lat']

### extract targeted area data
#meta_js_dir = '/p/project/deepacf/deeprain/video_prediction_shared_folder/preprocessedData/era5-Y2007-2019M01to12-80x48-3960N0180E-2t_tcc_t_850'
#meta_js_filename = 'metadata.json'
#meta_js_path = os.path.join(meta_js_dir,meta_js_filename)
#with open(meta_js_path) as meta_js_file:
#            meta_data = json.load(meta_js_file)
#meta_coords = meta_data['coordinates']
#meta_lat = meta_coords['lat']
#meta_lon = meta_coords['lon']
#nx,ny = len(meta_lon),len(meta_lat)

#meta_lon_loc = np.zeros((len(t2m_lon)), dtype=bool)
#for i in range(len(t2m_lon)):
#    if np.round(t2m_lon[i]*10)/10 in meta_lon:
#        meta_lon_loc[i] = True

#meta_lat_loc = np.zeros((len(t2m_lat)), dtype=bool)
#for i in range(len(t2m_lat)):
#    if np.round(t2m_lat[i]*10)/10 in meta_lat:
#        meta_lat_loc[i] = True

#t2m = t2m_months['var167']
#t2m_time = t2m_months['time']
#t2m_target = t2m[dict(lon=meta_lon_loc,lat=meta_lat_loc)]

#t2m_data = np.array(t2m_target)
#t2m_monthly = np.reshape(t2m_data,[12*nyear,24,t2m_data.shape[1],t2m_data.shape[2]])
#t2m_climate_hourly = np.mean(t2m_monthly,0)
#t2m_climate = np.mean(t2m_climate_hourly,0)

##########################################################
def compute_rmse(ref,fcst):
    rmse = np.sqrt(np.mean((ref-fcst)**2))
    return rmse

rmse = compute_rmse(ref,fcst)
print('RMSE of model is: ', rmse)

def compute_rmse_metric(ref,fcst):
    '''
    compute rmse metric for plotting
    the shape of input should be: [num_sample,height,width]
    '''
    nx,ny = ref.shape[1],ref.shape[2]
    rmse = np.ones([nx,ny])*np.nan
    for i in range(nx):
        for j in range(ny):
            rmse[i,j] = np.sqrt(np.mean((ref[:,i,j]-fcst[:,i,j])**2))

RMSE = compute_rmse_metric(ref,fcst)

##########################################################
def compute_acc(ref,fcst,clim):
    ref_ = ref - clim
    fcst_ = fcst - clim
    acc = np.sum(ref_*fcst_)/np.sqrt(np.sum(ref_**2) * np.sum(fcst_**2))
    return acc

#acc_savp = compute_acc(ref,savp_fcst,t2m_climate)
#print('ACC of savp model is: ', acc_savp)

##########################################################
def compute_csi(ref,fcst,th0,th1):
    ref = np.reshape(np.array(ref),[-1,1])
    fcst = np.reshape(np.array(fcst),[-1,1])
    hits,misses,false_alarms,correct_negatives = 0,0,0,0
    for i in range(len(ref)):
        if ((ref[i]>th0) & (ref[i]<=th1)) & ((fcst[i]>th0) & (fcst[i]<=th1)):
            hits = hits + 1
        elif ((fcst[i]<=th0) | (fcst[i]>th1)) & ((ref[i]>th0) & (ref[i]<=th1)):
            misses = misses + 1
        elif ((ref[i]<=th0) | (ref[i]>th1)) & ((fcst[i]>th0) & (fcst[i]<=th1)):
            false_alarms = false_alarms + 1
        elif ((ref[i]<=th0) | (ref[i]>th1)) & ((fcst[i]<=th0) | (fcst[i]>th1)):
            correct_negatives = correct_negatives + 1
    csi = hits/(hits+false_alarms+misses)
    pod = misses/(hits+misses)
    far = false_alarms/(hits+false_alarms)
    hits_random = (hits+misses)*(hits+false_alarms)/(hits+correct_negatives+misses+false_alarms)
    ets = (hits-hits_random)/(hits+misses+false_alarms-hits_random)
    print('grid num:', hits+misses+false_alarms+correct_negatives)
    print('grid num:', len(ref))
    return csi,pod,far,ets

th0,th1 = 0,1000
csi_savp,pod_savp,far_savp,ets_savp = compute_csi(ref,savp_fcst,th0,th1)
print('CSI of savp model is: ', csi_savp)
print('POD of savp model is: ', pod_savp)
print('FAR of savp model is: ', far_savp)
print('ETS of savp model is: ', ets_savp)

