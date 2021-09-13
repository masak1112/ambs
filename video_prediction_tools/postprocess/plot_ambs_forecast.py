#!/usr/bin/env python
# coding: utf-8

"""
Script to plot video frame prediction forecasts of 2m temperature as well as the difference w.r.t. the ground truth.
Data is expected to be saved in netCDF-files produced by main_visualize_postprocess.py (= the postprocessing step of
the AMBS-workflow)
"""

import xarray as xr
import numpy as np
import pandas as pd
import sys 
from argparse import ArgumentParser
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy
from mpl_toolkits.basemap import Basemap


# The plot function

def create_plot(data: xr.DataArray, data_ref: xr.DataArray, varname: str, fcst_hour: int, plt_fname: str):
    """
    Creates filled contour plot of the forecast data as well as the difference between forecast and analysis (ground
    truth).
    ML: So far, only plotting of the 2m temperature is supported (with 12 predicted hours/frames)
    :param data: the forecasted data array to be plotted
    :param data_ref: the reference data ('ground truth')
    :param varname: the name of the variable
    :param plt_fname: the filename to the store the plot
    :return: -
    """
    method = create_plot.__name__

    try:
        coords = data.coords
        # handle coordinates and forecast times
        lat, lon = coords["lat"], coords["lon"]
        dates_fcst = pd.to_datetime(coords["time_forecast"].data)
    except Exception as err:
        print("%{0}: Could not retrieve expected coordinates lat, lon and time_forecast from data.".format(method))
        raise err

    lons, lats = np.meshgrid(lon, lat)

    date0 = dates_fcst[0] - (dates_fcst[1] - dates_fcst[0])
    date0_str = date0.strftime("%Y-%m-%d %H:%M UTC")

    fhhs = ((dates_fcst - date0) / pd.Timedelta('1 hour')).values

    # check data to be plotted since programme is not generic so far
    if np.shape(dates_fcst)[0] != 12:
        raise ValueError("%{0}: Currently, only 12 hour forecast can be handled properly.".format(method))

    if varname != "2t":
        raise ValueError("%{0}: Currently, only 2m temperature is plotted nicely properly.".format(method))

    # define levels
    clevs = np.arange(-10., 40., 1.)
    clevs_diff = np.arange(-10.5, 10.6, 1.)

    # create fig and subplot axes
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6,12))
    axes = axes.flatten()
    cbar_labs = ["°C", "K"]
    
    for t in np.arange(2):
        m = Basemap(projection='cyl', llcrnrlat=np.min(lat), urcrnrlat=np.max(lat),
                    llcrnrlon=np.min(lon), urcrnrlon=np.max(lon), resolution='l', ax=axes[t])
        m.drawcoastlines()
        x, y = m(lons, lats)
    
        lat_lab = [1, 0, 0, 0]
        axes[t].set_ylabel(u'Latitude', labelpad=32)
        lon_lab = list(np.zeros(4))
        if t == 1:
            lon_lab = [0, 0, 0, 1]
            axes[t].set_xlabel(u'Longitude', labelpad=17)
    
        m.drawmapboundary()
        m.drawparallels(np.arange(0, 90, 5),labels=lat_lab, xoffset=0.5)
        m.drawmeridians(np.arange(5, 355, 10),labels=lon_lab, yoffset=0.5)
        if t == 0:
            cs = m.contourf(x, y, data.isel(time_forecast=fcst_hour)-273.15, clevs, cmap=plt.get_cmap("jet"), ax=axes[t])
            cbar_ticks = None
        elif t == 1:
            cs = m.contourf(x, y, data.isel(time_forecast=fcst_hour)-data_ref.isel(time_forecast=fcst_hour),
                            clevs_diff, cmap=plt.get_cmap("PuOr"), ax=axes[t])
            cbar_ticks = list(np.arange(-10.5, -2., 2.)) + [-0.5, 0.5] + list(np.arange(2.5, 10.6, 2.))
            print(cbar_ticks)
        # add colorbar.
        pos = axes[t].get_position()

        cbar_ax = fig.add_axes([0.95, pos.y0+0.08*(t*2-1), 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(cs, cax=cbar_ax, orientation="vertical",  ticks=cbar_ticks)
        cbar.set_label(cbar_labs[t])
    # save to disk
    #plt.show()
    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=-0.5, wspace=0.05)
    plt.subplots_adjust(hspace=-0.5)
    #plt.tight_layout()

    plt.savefig(plt_fname, bbox_inches="tight")
    plt.close()


def main(args):

    parser = ArgumentParser()

    # add optional arguments which may be passed
    parser.add_argument("-fname", "--filename", dest="filename", type=str,
                        help="The path to the netCDF-file from which the plot data is retrieved.")
    parser.add_argument("-fhour", "--forecast_hour", dest="forecast_hour", type=int,
                        help="The forecast hour/lead time for which the plot should be created")

    args = parser.parse_args()

    filename = args.filename
    fhh = args.forecast_hour

    if os.path.isfile(filename):
        raise FileNotFoundError("Could not find the indictaed netCDF-file '{0}'".format(filename))

    with xr.open_dataset(filename) as dfile:
        t2m_fcst, t2m_ref = dfile["2t_fcst"], dfile["2t_ref"]

    create_plot(t2m_fcst, t2m_ref, "2t", fhh, "{0}_fhh{1:d}.png".format(filename[0:-3], fhh))


if __name__ == "__main__":
    main(sys.argv[1:])
