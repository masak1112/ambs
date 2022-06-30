import json as js
import os
import argparse
import itertools as it
from pathlib import Path
from typing import Union, List, get_args
import zipfile as zf

from data_preprocess.extract_weatherbench import ExtractWeatherbench

# IDEA: type conversion (generic) => params_obj => bounds_checking (ds-specific)/ semantic checking


def source_dir(directory):
    dir = Path(directory)
    if not dir.exists():
        raise ValueError(f"Input directory {dir.absolute()} does not exist")
    return dir


def destination_dir(directory):
    dir = Path(directory)
    if not dir.exists():
        raise ValueError(f"Output directory: {dir.absolute()} does not exist.")
    return dir


def years(years):
    try:
        month_list = [int(x) for x in years]
    except ValueError as e:
        if not years == "all":
            raise ValueError(
                f"years must be either a list of years or 'all', not {months}."
            )
        month_list = -1

    return month_list


def months(months):
    try:
        month_list = [int(x) for x in months]
    except ValueError as e:
        if months == "DJF":
            month_list = [1, 2, 12]
        elif months == "MAM":
            month_list = [3, 4, 5]
        elif months == "JJA":
            month_list = [6, 7, 8]
        elif months == "SON":
            month_list = [9, 10, 11]
        elif months == "all":
            month_list = list(range(1, 13))
        else:
            raise ValueError(
                f"months-string '{months}' cannot be converted to list of months"
            )

    if not all(1 <= m <= 12 for m in months_list):
        errors = filter(lambda m: not 1 <= m <= 12, months_list)
        raise ValueError(
            f"all month integers must be within 1, ..., 12 not {list(errors)}"
        )

    return month_list


def variables(variables):
    vars = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "constants",
        "geopotential",
        "potential_vorticity",
        "relative_humidity",
        "specific_humidity",
        "temperature",
        "toa_incident_solar_radiation",
        "total_cloud_cover",
        "total_precipitation",
        "u_component_of_wind",
        "v_component_of_wind",
        "vorticity",
        "geopotential_500",
        "temperature_850",
    ]
    unknown_vars = [var for var in variables if var not in vars]
    if len(unknown_vars) > 0:
        raise ValueError(f"variables: {unknown_vars} are no known variables.")
    return variables


def get_data_files(variables: list, years, resolution, dirin: Path):
    """
    Get path to zip files and names of the yearly files within.
    :param variables: list of variables
    :param years: list of years
    :param months: list of months
    :param resolution:
    :param dirin: input directory
    :return lists paths to zips of variables
    """
    data_files = []
    zip_files = []
    res_str = f"{resolution}deg"
    for var in variables:
        var_dir = dirin / res_str / var
        if not var_dir.exists():
            raise ValueError(
                f"variable {var} is not available for resolution {res_str}"
            )

        zip_file = var_dir / f"{var}_{res_str}.zip"
        with zf.ZipFile(zip_file, "r") as myzip:
            names = myzip.namelist()
            if not all(any(str(year) in name for name in names) for year in years):
                raise ValueError(
                    f"variable {var} is not available for all years: {years}"
                )
            names = filter(lambda name: any(str(year) in name for year in years), names)

        data_files.append(list(names))
        zip_files.append(zip_file)

    return zip_files, data_files


def nyx(nyx):
    try:
        nyx = [int(n) for n in nyx]
    except ValueError as e:
        raise ValueError(f"number of grid points should be integers not {nyx}")
    if not all(n > 0 for n in nyx):
        raise ValueError(f"number of grid points should be > 0")

    return nyx


def coords(coords_sw):
    try:
        coords = [float(c) for c in coords_sw]
    except ValueError as e:
        raise ValueError(f"coordinates should be floats not {coords}")
    if not -90 <= coords[0] <= 90:
        raise ValueError(
            f"latitude of sw-corner is {coords[0]} but should be >= -90, <= 90"
        )
    if not 0 <= coords[1] <= 360:
        raise ValueError(
            f"latitude of sw-corner is {coords[0]} but should be >= 0, <= 360"
        )
    return coords


def main():
    # TODO consult Bing for defaults
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_dir",
        type=source_dir,
        help="Top-level directory where ERA5 grib-files are located under <year>/<month>.",
    )
    parser.add_argument(
        "destination_dir",
        type=destination_dir,
        help="Destination directory where the netCDF-files will be stored",
    )
    parser.add_argument(
        "years", nargs="+", type=int, help="Years of data to be processed."
    )
    parser.add_argument(
        "--resolution",
        "-r",
        choices=[1.40625, 2.8125, 5.625],
        default=5.625,
    )
    parser.add_argument(
        "--months",
        "-m",
        nargs="+",
        dest="months",
        default="all",
        type=months,
        help="Months of data. Can also be 'all' or season-strings, e.g. 'DJF'.",
    )
    parser.add_argument(
        "--variables",
        "-v",
        nargs="+",
        dest="variables",
        help="list of variables to extract",
        default=["geopotential_500", "temperature_850"],
        type=variables,
    )
    parser.add_argument(
        "--sw_corner",
        "-swc",
        dest="sw_corner",
        nargs="+",
        type=coords,
        default=(38.4, 0.0),
        help="Defines south-west corner of target domain (lat, lon)=(-90..90, 0..360)",
    )
    parser.add_argument(
        "--nyx",
        "-nyx",
        dest="nyx",
        nargs="+",
        type=nyx,
        default=(56, 92),
        help="Number of grid points in zonal and meridional direction.",
    )

    args = parser.parse_args()

    ne_corner = [
        coord + n * args.resolution for coord, n in zip(args.sw_corner, args.nyx)
    ]
    if not (-90 <= ne_corner[0] <= 90 and 0 <= ne_corner[1] <= 360):
        raise ValueError(
            f"number of grid points {args.nyx} will result in a invalid north-east corner: {ne_corner}"
        )

    weatherbench_extract = ExtractWeatherbench(
        args.source_dir,
        args.destination_dir,
        args.variables,
        args.years,
        args.months,
        (sw_corner[0], ne_corner[0]),
        (sw_corner[1], ne_corner[1]),
        args.resolution,
    )
    weatherbench_extract()


if __name__ == "__main__":
    main()
