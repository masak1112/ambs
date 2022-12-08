import os, glob
import logging

from zipfile import ZipFile
from typing import Union
from pathlib import Path
import multiprocessing as mp
import itertools as it
import sys

import pandas as pd
import xarray as xr

from utils.dataset_utils import get_filename_template

logging.basicConfig(level=logging.DEBUG)

class ExtractWeatherbench:
    max_years = list(range(1979, 2018))
    
    def __init__(
        self,
        dirin: Path,
        dirout: Path,
        variables: list[dict],
        years: Union[list[int], int],
        months: list[int],
        lat_range: tuple[float],
        lon_range: tuple[float],
        resolution: float,
    ):
        """
        This script performs several sanity checks and sets the class attributes accordingly.
        :param dirin: directory to the ERA5 reanalysis data
        :param dirout: directory where the output data will be saved
        :param variables: controlled dictionary for getting variables from ERA5-dataset, e.g. {"t": {"ml": "p850"}}
        :param years: list of year to to extract, -1 if all
        :param months: list of months to extract
        :param lat_range: domain of the latitude axis to extract
        :param lon_range: domain of the longitude axis to extract
        :param resolution: spacing on both lat, lon axis
        """
        
        self.dirin = dirin
        self.dirout = dirout

        if years[0] == -1:
            self.years = ExtractWeatherbench.max_years
        else:
            self.years = years
        self.months = months

        # TODO handle special variables for resolution 5.625 (temperature_850, geopotential_500)
        if resolution == 5.625:
            for var in variables:
                combined_name = f"{var['name']}_{var['lvl'][0]}"
                if combined_name in {"temperature_850", "geopotential_500"}:
                    var["name"] = combined_name
                    
        self.variables = variables

        self.lat_range = lat_range
        self.lon_range = lon_range

        self.resolution = resolution
              

    def __call__(self):
        """
        Run extraction.
        :return: -
        """
        logging.info("start extraction")

        zip_files, data_files = self.get_data_files()

        # extract archives => netcdf files (maybe use tempfiles ?)
        args = [
            (var_zip, file, self.dirout)
            for var_zip, files in zip(zip_files, data_files)
            for file in files
        ]
        with mp.Pool(20) as p:
            p.starmap(ExtractWeatherbench.extract_task, args)
        logging.info("finished extraction")
        
        # TODO: handle 3d data

        # load data
        files = [self.dirout / file for data_file in data_files for file in data_file]
        ds = xr.open_mfdataset(files, coords="minimal", compat="override")
        logging.info("opened dataset")
        ds.drop_vars("level")
        logging.info("data loaded")

        # select months
        ds = ds.isel(time=ds.time.dt.month.isin(self.months))

        # select region
        ds = ds.sel(lat=slice(*self.lat_range), lon=slice(*self.lon_range))
        logging.info("selected region")

        # split into monthly netcdf
        year_month_idx = pd.MultiIndex.from_arrays(
            [ds.time.dt.year.values, ds.time.dt.month.values]
        )
        ds.coords["year_month"] = ("time", year_month_idx)
        logging.info("constructed splitting-index")

        with mp.Pool(20) as p:
            p.map(
                ExtractWeatherbench.write_task,
                zip(ds.groupby("year_month"), it.repeat(self.dirout)),
                chunksize=5,
            )
        logging.info("wrote output")

    @staticmethod
    def extract_task(var_zip, file, dirout):
        with ZipFile(var_zip, "r") as myzip:
            myzip.extract(path=dirout, member=file)

    @staticmethod
    def write_task(args):
        (year_month, monthly_ds), dirout = args
        year, month = year_month
        logging.debug(f"{year}.{month:02d}: dropping index")
        monthly_ds = monthly_ds.drop_vars("year_month")
        try:
            logging.debug(f"{year}.{month:02d}: writing to netCDF")
            monthly_ds.to_netcdf(path=dirout / get_filename_template("weatherbench").format(year=year, month=month))
        except RuntimeError as e:
            logging.error(f"runtime error for writing {year}.{month}\n{str(e)}")
        logging.debug(f"{year}.{month:02d}: finished processing")

    def get_data_files(self):
        """
        Get path to zip files and names of the yearly files within.
        :return lists paths to zips of variables
        """
        data_files = []
        zip_files = []
        res_str = f"{self.resolution}deg"
        years = self.years
        for var in self.variables:
            var_dir = self.dirin / res_str / var["name"]
            if not var_dir.exists():
                raise ValueError(
                    f"variable {var} is not available for resolution {res_str}"
                )

            zip_file = var_dir / f"{var['name']}_{res_str}.zip"
            with ZipFile(zip_file, "r") as myzip:
                names = myzip.namelist()
                logging.debug(f"var:{var}\nyears:{years}\nnames:{names}")
                if not all(any(str(year) in name for name in names) for year in years):
                    missing_years = list(filter(lambda year: any(str(year) in name for name in names), years))
                    raise ValueError(
                        f"variable {var} is not available for years: {missing_years}"
                    )
                names = filter(
                    lambda name: any(str(year) in name for year in years), names
                )

            data_files.append(list(names))
            zip_files.append(zip_file)

        return zip_files, data_files
