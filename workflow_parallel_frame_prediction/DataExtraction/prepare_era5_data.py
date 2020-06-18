
import numpy as np
from datetime import datetime
from netCDF4 import Dataset, date2num
from shiftgrid import shiftgrid
import os

AUTHOR = 'Severin Hussmann (s.hussmann@fz-juelich.de)'
# specify source and target directories

def source_file_name(year, month, day, hour):
    #src_file = '{:04d}/{:02d}/ecmwf_era5_{:02d}{:02d}{:02d}{:02d}.nc'.format(year, month, year % 100, month, day, hour)
    src_file = 'ecmwf_era5_{:02d}{:02d}{:02d}{:02d}.nc'.format(year % 100, month, day, hour)
    return src_file


def prepare_era5_data_one_file(src_file,directory_to_process,target_dir, target="test.nc"):
    try:
        out_file = target
        print(src_file, ' --> ', os.path.join(target_dir,out_file))
        fh = Dataset(os.path.join(directory_to_process, src_file), mode = 'r')

        lons = fh.variables['lon'][:]
        lats = fh.variables['lat'][:]

        # load 2 metre temperature
        t2 = fh.variables['T2'][0, :, :]
        t2_shift, lons_shift = shiftgrid(180., t2, lons, start = False)

        # load mean sea level pressure
        msl = fh.variables['MSL'][0, :, :]
        msl_shift, lons_shift = shiftgrid(180., msl, lons, start = False)

        # transform geopotential to geopotential at 500hpa
        gph = fh.variables['GPH'][0, :, :, :]
        a = fh.variables['a'][:]  # convert netCDF to numpy arrays
        b = fh.variables['b'][:]  # otherwise cannot iterate over netcdf4 variable
        ps = fh.variables['ps'][0, :, :]
        fh.close()

        # obtain dimensions
        z_len, y_len, x_len = gph.shape

        # z_len = 137
        # y_len = 601
        # x_len = 1200
        # Function to calculate the Pressure in hPa at point x/y at level k
        # p(k,j,i) = a(k) + b(k)*ps(j,i)
        def calcP(z, x, y, a=a, b=b, ps=ps):
            p = (a[z] + b[z] * ps[x, y]) / 100
            return p

        # pressure3d
        p3d = np.fromfunction(calcP, (z_len, y_len, x_len), a=a, b=b, ps=ps, dtype = int)

        # level2d
        yindices, xindices = np.indices((y_len, x_len))
        # calculate lowest level index where pressure is below 500 hPa
        # beware of Himalaya, where surface pressure may be below 500 hPa
        # - that region should actually contain missing values; here we cheat a little
        l2d = np.argmax((p3d - 500) < 0., axis = 0)
        l2d[l2d == 0] = 1
        # next lower level should have pressure above 500 hPa
        # pressure levels in Gebhard Guenther's netcdf files are from surface to top of atmosphere
        l2dm1 = l2d - 1
        # calculate interpolation measure
        levfrac = (p3d[l2dm1[:], yindices, xindices] - 500.) / (
                    p3d[l2dm1[:], yindices, xindices] - p3d[l2d[:], yindices, xindices])
        levfrac[levfrac < 0.] = 0.  # Himalaya correction
        print("l2d: ", np.min(l2d), np.max(l2d))
        print("levfrac: ", np.min(levfrac), np.max(levfrac))
        # gp500below: geopotential height below 500 hPa level (i.e. pressure > 500 hPa)
        gp500below = gph[l2dm1[:], yindices, xindices]
        gp500above = gph[l2d[:], yindices, xindices]
        gp500 = gp500below + levfrac * (gp500above - gp500below)
        print("gp500below: ", np.min(gp500below), np.max(gp500below))
        print("gp500above: ", np.min(gp500above), np.max(gp500above))

        # convert values in array from geopotential to geopotential height
        divider = lambda t: t / 9.8
        vfunc = np.vectorize(divider)
        gph500 = vfunc(gp500)

        gph500_shift, lons_shift = shiftgrid(180., gph500, lons, start = False)

        os.chdir(target_dir)
        test = Dataset(out_file, 'w', format = 'NETCDF4', clobber = True)

        # test.createDimension("channel", 3)
        latD = test.createDimension('lat', y_len)
        lonD = test.createDimension('lon', x_len)
        timeD = test.createDimension('time', None)
        # for debugging
        levD = test.createDimension('lev', z_len)

        # print(test.dimensions)
        t2_new = test.createVariable('T2', float, ('time', 'lat', 'lon'), zlib = True)
        t2_new.units = 'K'
        msl_new = test.createVariable('MSL', float, ('time', 'lat', 'lon'), zlib = True)
        msl_new.units = 'Pa'
        gph500_new = test.createVariable('gph500', float, ('time', 'lat', 'lon'), zlib = True)
        gph500_new.units = 'm'
        lat_new = test.createVariable('lat', float, ('lat',), zlib = True)
        lat_new.units = 'degrees_north'
        lon_new = test.createVariable('lon', float, ('lon',), zlib = True)
        lon_new.units = 'degrees_east'
        time_new = test.createVariable('time', 'f8', ('time',), zlib = True)
        time_new.units = "hours since 2000-01-01 00:00:00"
        time_new.calendar = "gregorian"
        p3d_new = test.createVariable('p3d', float, ('lev', 'lat', 'lon'), zlib = True)

        lat_new[:] = lats
        lon_new[:] = lons_shift
        year, month, day, hour = extract_time_from_file_name(src_file)
        dates = np.array([datetime(int(year), int(month), int(day), int(hour), 0, 0)])
        time_new[:] = date2num(dates, units = time_new.units, calendar = time_new.calendar)

        t2_new[:] = t2_shift.reshape(1, y_len, x_len)
        msl_new[:] = msl_shift.reshape(1, y_len, x_len)
        gph500_new[:] = gph500_shift.reshape(1, y_len, x_len)
        p3d_new[:] = p3d
        test.source_file = src_file
        test.title = 'ECMWF ERA5 data sample for Deep Learning'
        test.author = AUTHOR
        test.close()
    except Exception as exc:
        print (exc)
        pass


def extract_time_from_file_name(src_file):
    year = int("20" + src_file[11:13])
    month = int(src_file[13:15])
    day = int(src_file[15:17])
    hour = int(src_file[17:19])
    return year, month, day, hour

def process_era5_in_dir(job_name,src_dir,target_dir):
    print ("job_name",job_name)
    directory_to_process = os.path.join(src_dir, job_name)
    print("Going to process file in directory {}".format(directory_to_process))
    files = os.listdir(directory_to_process)
    os.chdir(directory_to_process)
    #create a subdirectory based on months
    target_dir2 = os.path.join(target_dir,job_name)
    print("The processed files are going to be saved to directory {}".format(target_dir2))
    if not os.path.exists(target_dir2): os.mkdir(target_dir2)
    for src_file in files:
        if src_file.endswith(".nc"):
            if os.path.exists(os.path.join(target_dir2, src_file)):
                print(src_file," file has been processed in directory ", target_dir2)
            else:
                print ("==========Processing file {} =============== ".format(src_file))
                prepare_era5_data_one_file(src_file=src_file,directory_to_process=directory_to_process, target=src_file, target_dir=target_dir2)
