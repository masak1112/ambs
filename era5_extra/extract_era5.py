#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: YanJI
Email: y.ji@fz-juelich.de
Date: 25 Feb 2021
"""

import os
import argparse

def mergevars_onehour(input_path,year,month,date,hour,varlist,output_path): #extract 2t,tcc,msl,t850,10u,10v
    temp_path = os.path.join(output_path,year)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    temp_path = os.path.join(output_path,year,month)
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    for var in varlist:
        # surface variables
        infile = os.path.join(input_path,year,month,year+month+date+hour+'_sf.grb')
        outfile = os.path.join(output_path,year,month,year+month+date+hour+'_sfvar.grb')
        outfile_sf = os.path.join(output_path,year,month,year+month+date+hour+'_'+var+'.nc')
        if var == '2t':
            os.system('cdo selname,"var167" %s %s' % (infile,outfile))
            os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
            os.system('rm %s' % outfile)
        if var == 'tcc':
            os.system('cdo selname,"var164" %s %s' % (infile,outfile))
            os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
            os.system('rm %s' % outfile)
        if var == 'msl':
            os.system('cdo selname,"var151" %s %s' % (infile,outfile))
            os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
            os.system('rm %s' % outfile)
        if var == '10u':
            os.system('cdo selname,"var165" %s %s' % (infile,outfile))
            os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
            os.system('rm %s' % outfile)
        if var == '10v':
            os.system('cdo selname,"var166" %s %s' % (infile,outfile))
            os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
            os.system('rm %s' % outfile)
        # multi-level variables
        infile = os.path.join(input_path,year,month,year+month+date+hour+'_ml.grb')
        outfile = os.path.join(output_path,year,month,year+month+date+hour+'_mlvar.grb')
        outfile_sf = os.path.join(output_path,year,month,year+month+date+hour+'_'+var+'.nc')
        if var == 't850':
            pl = int(var[1:])
            os.system('cdo -selname,"t" -ml2pl,%d %s %s' % (pl*100,infile,outfile)) 
            #os.system('cdo -merge -selname,"t" -sellevel,%d %s %s' % (pl,infile,outfile)) 
            os.system('cdo -f nc copy %s %s' % (outfile,outfile_sf))
            os.system('rm %s' % outfile)
    # merge both variables
    infile = os.path.join(output_path,year,month,year+month+date+hour+'*.nc')
    outfile = os.path.join(output_path,year,month,'ecmwf_era5_'+year[2:]+month+date+hour+'.nc') # change the output file name
    os.system('cdo merge %s %s' % (infile,outfile))
    os.system('rm %s' % (infile))

def main():
    current_path = os.getcwd()

    parser=argparse.ArgumentParser()
    parser.add_argument("--source_dir",type=str,default="/p/fastdata/slmet/slmet111/met_data/ecmwf/era5/grib")
    parser.add_argument("--destination_dir",type=str,default="/p/home/jusers/ji4/juwels/ambs/era5_extra")
    parser.add_argument("--year",type=str,nargs="+",default=['2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'])
    parser.add_argument("--variables",type=str,nargs="+",default=['2t','10u','10v','tcc','msl','t850'])
    #parser.add_argument("--logs_path",type=str,default=current_path)
    args = parser.parse_args()
    year = args.year
    var = args.variables
    input_path = args.source_dir
    output_path = args.destination_dir
    print(year)
    print(var)
    for y in year:
        for m in ['01','02','03','04','05','06','07','08','09','10','11','12']:        
            for d in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']:
                for h in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']:
                    mergevars_onehour(input_path,y,m,d,h,var,output_path)

if __name__ == "__main__":
    main()

