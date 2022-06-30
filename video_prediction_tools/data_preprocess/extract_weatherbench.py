import os, glob
import logging

from zipfile import ZipFile
from typing import List, Union, Tuple
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
import xarray as xr


class ExtractWeatherbench:
    def __init__(
        self,
        dirin: Path,
        dirout: Path,
        variables: List[str],
        years: Union[List[int], int],
        months: List[int],
        lat_range: Tuple[float],
        lon_range: Tuple[float],
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

        self.years = years
        self.months = months

        self.variables = variables

        self.lat_range = lat_range
        self.lon_range = lon_range

        self.resolution = resolution

    def __call__(self):
        """
        Run extraction.
        :return: -
        """
        zip_files, data_files = self.get_data_files()

        # extract archives => netcdf files (maybe use tempfiles ?)
        args = [
            (var_zip, file, self.dirout)
            for var_zip, files in zip(zip_files, data_files)
            for file in files
        ]
        with Pool() as p:
            p.starmap(ExtractWeatherbench.extract_task, args)

        # load data
        files = [dirout / file for data_file in data_files for file in data_file]
        ds = xr.open_mfdataset(files, coords="minimal", compat="override")
        ds.drop_vars("level")

        # select months
        ds = ds.isel(time=ds.time.dt.month.isin(self.months))

        # select region
        ds = ds.sel(lat=slice(*self.lat_range), lon=slice(*self.lon_range))

        # split into monthly netcdf
        year_month_idx = pd.MultiIndex.from_arrays(
            [ds.time.dt.year.values, ds.time.dt.month.values]
        )
        ds.coords["year_month"] = ("time", year_month_idx)

        with Pool() as p:
            p.map(
                ExtractWeatherbench.write_task,
                zip(ds.groupby("year_month"), it.repeat(self.dirout)),
                chunksize=5,
            )

    @staticmethod
    def extract_task(var_zip, file, dirout):
        with ZipFile(var_zip, "r") as myzip:
            myzip.extract(path=dirout, member=file)

    @staticmethod
    def write_task(args):
        (year_month, monthly_ds), dirout = args
        year, month = year_month
        monthly_ds = monthly_ds.drop_vars("year_month")
        monthly_ds.to_netcdf(path=dirout / f"{year}_{month}.nc")

    def get_data_files(self):
        """
        Get path to zip files and names of the yearly files within.
        :return lists paths to zips of variables
        """
        data_files = []
        zip_files = []
        res_str = f"{self.resolution}deg"
        years = range() if self.years == -1 else self.years
        for var in self.variables:
            var_dir = self.dirin / res_str / var
            if not var_dir.exists():
                raise ValueError(
                    f"variable {var} is not available for resolution {res_str}"
                )

            zip_file = var_dir / f"{var}_{res_str}.zip"
            with ZipFile(zip_file, "r") as myzip:
                names = myzip.namelist()
                if not all(any(str(year) in name for name in names) for year in years):
                    missing_years = [
                        year
                        for name in names
                        if str(year) not in name
                        for year in years
                    ]
                    raise ValueError(
                        f"variable {var} is not available years: {missing_years}"
                    )
                names = filter(
                    lambda name: any(str(year) in name for year in years), names
                )

            data_files.append(list(names))
            zip_files.append(zip_file)

        return zip_files, data_files
