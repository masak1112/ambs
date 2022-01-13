"""
Driver for preprocessing step 1 which parses the input arguments from the runscript
and performs parallelization with PyStager.
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"

from mpi4py import MPI
import os, sys, glob
import logging
import time
import argparse
from data_preprocess.process_netCDF_v2 import *
from metadata import MetaData
from netcdf_datahandling import GeoSubdomain
import json


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", "-src_dir", dest="source_dir", type=str,
                        help="Top-level directory where ERA5 grib-files are located under <year>/<month>.")
    parser.add_argument("--destination_dir", "-dest_dir", dest="destination_dir", type=str,
                        help="Destination directory where the netCDF-files will be stored")
    parser.add_argument("--years", "-y", nargs="+", dest="years", help="Years of data to be processed.")
    parser.add_argument("--years", "-m", nargs="+", dest="months",
                        help="Months of data. Can also be 'all' or season-strings, e.g. 'DJF'.")
    parser.add_argument("--variables", "-v",  nargs="+", default=["2t"], help="Variables to be processed.")
    parser.add_argument("--sw_corner", "-swc", dest="sw_corner", nargs="+",
                        help="Defines south-west corner of target domain (lat, lon)=(-90..90, 0..360)")
    parser.add_argument("--nyx", "-nyx", dest="nyx", nargs="+",
                        help="Number of grid points in zonal and meridional direction.")

    args = parser.parse_args()