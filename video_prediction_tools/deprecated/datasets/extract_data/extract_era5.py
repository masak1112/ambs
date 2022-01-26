#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: YanJI
Email: y.ji@fz-juelich.de
Date: 25 Feb 2021
"""

import os

os.system('module load GCC/9.3.0; module load ParaStationMPI/5.4.7-1; module load CDO/1.9.8')

def mergetime_oneday(input_path,year,month,date,variable,pressure_level,output_path):
    temp_path = os.path.join(output_path,year)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    temp_path = os.path.join(output_path,year,month)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)    
    if variable == 'T2':
        infilelist = year+month+date+'*_sf.grb'
        outfile = year+month+date+'_'+variable+'.grb'
        ori_path = os.path.join(input_path,year,month,infilelist)
        out_path = os.path.join(output_path,year,month,outfile)
        os.system('cdo expr,"var167" -mergetime %s %s' % (ori_path,out_path))
    if variable == 'MSL':
        infilelist = year+month+date+'*_sf.grb'
        outfile = year+month+date+'_'+variable+'.grb'
        ori_path = os.path.join(input_path,year,month,infilelist)
        out_path = os.path.join(output_path,year,month,outfile)
        os.system('cdo expr,"var151" -mergetime %s %s' % (ori_path,out_path))
    if variable == 'gph500':
        infilelist = year+month+date+'*_ml.grb'
        outfile = year+month+date+'_'+variable+'.grb'
        ori_path = os.path.join(input_path,year,month,infilelist)
        out_path = os.path.join(output_path,year,month,outfile)
        os.system('cdo expr,"z" -sellevel,%d -mergetime %s %s' % (pressure_level,ori_path,out_path)) # something wrong -- the variable 'z' only has one level


def mergevars_oneday(input_path,year,month,date,var1,var2,var3):
    ori_path = os.path.join(input_path,year,month)
    varfile1 = os.path.join(ori_path,year+month+date+'_'+var1+'.grb')
    varfile2 = os.path.join(ori_path,year+month+date+'_'+var2+'.grb')
    varfile3 = os.path.join(ori_path,year+month+date+'_'+var3+'.grb')
    varfile1_nc = os.path.join(ori_path,year+month+date+'_'+var1+'.nc')
    varfile2_nc = os.path.join(ori_path,year+month+date+'_'+var2+'.nc')
    varfile3_nc = os.path.join(ori_path,year+month+date+'_'+var3+'.nc')
    outfile = os.path.join(ori_path,year+month+date+'_'+var1+'_'+var2+'_'+var3+'.nc')
    os.system('cdo -f nc copy %s %s' % (varfile1,varfile1_nc))
    os.system('cdo -f nc copy %s %s' % (varfile2,varfile2_nc))
    os.system('cdo -f nc copy %s %s' % (varfile3,varfile3_nc))
    os.system('cdo merge %s %s %s %s' % (varfile1_nc,varfile2_nc,varfile3_nc,outfile))


def mergevars_onehour(input_path,year,month,date,hour,output_path): #extract t2,tcc,msl,t850,u10,v10
    temp_path = os.path.join(output_path,year)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    temp_path = os.path.join(output_path,year,month)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    # surface variables
    infile = os.path.join(input_path,year,month,year+month+date+hour+'_sf.grb')
    outfile = os.path.join(output_path,year,month,year+month+date+hour+'_sfvar.grb')
    outfile_sf = os.path.join(output_path,year,month,year+month+date+hour+'_sfvar.nc')
    os.system('cdo -merge -selname,"var167","var151" %s %s' % (infile,outfile)) # change the select vatiables
    os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
    os.system('rm %s' % outfile)
    # multi-level variables
    infile = os.path.join(input_path,year,month,year+month+date+hour+'_ml.grb')
    outfile = os.path.join(output_path,year,month,year+month+date+hour+'_mlvar.grb')
    outfile_ml = os.path.join(output_path,year,month,year+month+date+hour+'_mlvar.nc')
    pl1 = 95; pl2 = 96 # seclect the levels
    os.system('cdo -merge -selname,"t" -sellevel,%d,%d %s %s' % (pl1,pl2,infile,outfile)) # change the select vatiables and levels
    os.system('cdo -f nc copy %s %s' % (outfile,outfile_ml))
    os.system('rm %s' % outfile)
    # merge both variables
    outfile = os.path.join(output_path,year,month,year+month+date+hour+'_era5.nc') # change the output file name
    os.system('cdo merge %s %s %s' % (outfile_sf,outfile_ml,outfile))
    os.system('rm %s %s' % (outfile_sf,outfile_ml))

input_path='/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib'
output_path='/p/home/jusers/ji4/juwels/ambs/era5_extra'

#### test for one day
#y = '2009'
#m = '01'
#d = '01'
#mergetime_oneday(input_path,y,m,d,'T2',1,output_path)
#mergetime_oneday(input_path,y,m,d,'MSL',1,output_path)
#mergetime_oneday(input_path,y,m,d,'gph500',1,output_path) # something wrong -- the variable 'z' only has one level
#merge_oneday(output_path,y,m,d,'T2','MSL','gph500')

#### looping
for y in ['2007', '2008']:#, '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']:
    for m in ['01', '02']:#, '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:        
        for d in ['01', '02']:#, '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']:
            for h in ['01', '02']:#, '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']:
                mergevars_onehour(input_path,y,m,d,h,output_path)


