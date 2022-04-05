"""
Driver for preprocessing step 1 which parses the input arguments from the runscript
and performs parallelization with PyStager.
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"

import json as js
import argparse
from data_preprocess.preprocess_data_step1 import Preprocess_ERA5_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", "-src_dir", dest="source_dir", type=str, required=True,
                        help="Top-level directory where ERA5 grib-files are located under <year>/<month>.")
    parser.add_argument("--destination_dir", "-dest_dir", dest="destination_dir", type=str, required=True,
                        help="Destination directory where the netCDF-files will be stored")
    parser.add_argument("--years", "-y", nargs="+", dest="years", type=int, required=True,
                        help="Years of data to be processed.")
    parser.add_argument("--months", "-m", nargs="+", dest="months", default="all",
                        help="Months of data. Can also be 'all' or season-strings, e.g. 'DJF'.")
    parser.add_argument("--variables", "-v", dest="vars_dict", type=js.loads, default='{"2t": "sfc"}',
                        help="Dictionary-like string to parse variable names (keys) together with " +
                             "variable types (values).")
    parser.add_argument("--sw_corner", "-swc", dest="sw_corner", nargs="+", default=(38.4, 0.),
                        help="Defines south-west corner of target domain (lat, lon)=(-90..90, 0..360)")
    parser.add_argument("--nyx", "-nyx", dest="nyx", nargs="+", type=int, default=(56, 92),
                        help="Number of grid points in zonal and meridional direction.")

    args = parser.parse_args()

    # initialize preprocessing instance...
    era5_preprocess = Preprocess_ERA5_data(args.source_dir, args.destination_dir, args.vars_dict, args.sw_corner,
                                           args.nyx, args.years, args.months)

    # ...and run it
    era5_preprocess()


if __name__ == "__main__":
    main()
