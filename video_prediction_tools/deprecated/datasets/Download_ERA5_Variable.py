#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

import cdsapi
import argparse

'''
code example:
python /mnt/jiyan/ERA5/Download_ERA5_geopotential.py --year_select 2003 --month_select 01 02 03 
--date_select 01 02 31 --hour_select 00:00 01:00 23:00 --lon_start 70 --lon_end 140 --lat_start 15 
--lat_end 60 --data_format 'netcdf' --variable 'temperature' --pressure_level 500 850 --output_path /tmp/
'''

'''
copy the .cdsapirc to ~
then pip install cdsapi
'''

'''
year_select should be in 
    ['1979', '1980', '1981',
    '1982', '1983', '1984',
    '1985', '1986', '1987',
    '1988', '1989', '1990',
    '1991', '1992', '1993',
    '1994', '1995', '1996',
    '1997', '1998', '1999',
    '2000', '2001', '2002',
    '2003', '2004', '2005',
    '2006', '2007', '2008',
    '2009', '2010', '2011',
    '2012', '2013', '2014',
    '2015', '2016', '2017',
    '2018', '2019', '2020',]
'''
year_select = [ '2010', '2011', '2012', '2013', '2014']

'''
month_select should be in 
	'01', '02', '03',
	'04', '05', '06',
	'07', '08', '09',
	'10', '11', '12',
'''
month_select = ['01', '02', '03']

'''
date_select should be in 
	'01', '02', '03',
	'04', '05', '06',
	'07', '08', '09',
	'10', '11', '12',
	'13', '14', '15',
	'16', '17', '18',
	'19', '20', '21',
	'22', '23', '24',
	'25', '26', '27',
	'28', '29', '30',
	'31',
'''
date_select = [ '01', '02', '03']

'''
hour_select should be in 
	'00:00', '01:00', '02:00',
	'03:00', '04:00', '05:00',
	'06:00', '07:00', '08:00',
	'09:00', '10:00', '11:00',
	'12:00', '13:00', '14:00',
	'15:00', '16:00', '17:00',
	'18:00', '19:00', '20:00',
	'21:00', '22:00', '23:00',
'''
hour_select = [ '00:00', '01:00', '02:00'] 

'''
[north(0~90), west（-180～0）,south(0～-90), east(0~180)]
'''
area_select = [60, 70, 15, 140,] 

# 'grib' or 'netcdf'
data_format = 'netcdf' 

'''
variable： should be a single variable in 
    'divergence', 'fraction_of_cloud_cover', 'geopotential',
    'ozone_mass_mixing_ratio', 'potential_vorticity', 'relative_humidity',
    'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content', 'specific_humidity',
    'specific_rain_water_content', 'specific_snow_water_content', 'temperature',
    'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
    'vorticity'
'''
variable = 'temperature' # single varibale

'''
pressure_level should be a single pressure level in 
            '1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000',
'''
pressure_level = '500'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year_select", nargs='+', type=int, default=['1979', '1980', '1981','1982', '1983', '1984',
    '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
    '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
    '2015', '2016', '2017', '2018', '2019', '2020'], help="the year list to be downloaded")
    parser.add_argument("--month_select", nargs='+', type=int, default=['01', '02', '03', '04', '05', '06',
    	'07', '08', '09', '10', '11', '12'], help="the month list to be downloaded")
    parser.add_argument("--date_select", nargs='+', type=int, default=['01', '02', '03', '04', '05', '06',
	'07', '08', '09', '10', '11', '12',	'13', '14', '15', '16', '17', '18',	'19', '20', '21', '22', '23', '24',
	'25', '26', '27', '28', '29', '30',	'31'], help="the date list to be downloaded")
    parser.add_argument("--hour_select", nargs='+', type=str, default=['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
	'06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
	'18:00', '19:00', '20:00', '21:00', '22:00', '23:00'], help="the hour list to be downloaded") 
    parser.add_argument("--lon_start", type=float, default=-180, help="the minimum longitude of the area") 
    parser.add_argument("--lon_end", type=float, default=180, help="the minimum longitude of the area") 
    parser.add_argument("--lat_start", type=float, default=-90, help="the maximum latitude of the area") 
    parser.add_argument("--lat_end", type=float, default=90, help="the minimum latitude of the area") 
    parser.add_argument("--output_path", type=str, required=True, help="the path to be saved") 
    parser.add_argument("--data_format", type=str, default='netcdf', help="the data format")                                                             
    parser.add_argument("--variable", type=str, required=True, help="the variable to be downloaded") 
    parser.add_argument("--pressure_level", nargs='+', type=int, required=True, help="the variable to be downloaded") 
    args = parser.parse_args()
    download_hourly_reanalysis_era5_pl_variables(year_select=args.year_select,month_select=args.month_select,
    	date_select=args.date_select,hour_select=args.hour_select,lon_start=args.lon_start,lon_end=args.lon_end,
    	lat_start=args.lat_start,lat_end=args.lat_end,data_format=args.data_format,variable=args.variable,
    	pressure_level=args.pressure_level,output_path=args.output_path)


def download_hourly_reanalysis_era5_pl_variables(year_select,month_select,date_select,hour_select,
	lon_start,lon_end,lat_start,lat_end,data_format,variable,pressure_level,output_path):
	if data_format=='netcdf':
		fp = '.nc'
	elif data_format=='grib':
		fp = '.grib'
	c = cdsapi.Client()
	for iyear in year_select:
	    c.retrieve(
	    'reanalysis-era5-pressure-levels',
	    {
	        'product_type': 'reanalysis',
	        'format': data_format,
	        'variable': variable,
	        'pressure_level': pressure_level,
	        'year': str(iyear),
	        'month': month_select,
	        'day': date_select,
	        'time': hour_select,
	        'area': [lat_end, lon_start, lat_start, lon_end,],
	    },
		output_path+'ERA5_'+variable+'_'+str(iyear)+fp)

if __name__ == '__main__':
    main()
