"""
Collection of functions to create plots
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2021-05-27"

import os
import numpy as np
import pandas as pd
import xarray as xr
# for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from general_utils import provide_default


def plot_cond_quantile(quantile_panel: xr.DataArray, data_marginal: xr.DataArray, plt_fname: str, opt: dict):
    """
    Creates conditional quantile plot
    :param quantile_panel: quantile panel created by calculate_cond_quantiles
    :param data_marginal: data array for which histogram will be plotted
    :param plt_fname: name of the plot-file to be created
    :param opt: options
    :return:
    """

    method = plot_cond_quantile.__name__

    if not isinstance(quantile_panel, xr.DataArray):
        raise ValueError("%{0}: quantile_panel must be a DataArray".format(method))

    if not isinstance(data_marginal, xr.DataArray):
        raise ValueError("%{0}: data_marginal must be a DataArray".format(method))

    if list(quantile_panel.coords) != ["bin_center", "quantile"]:
        raise ValueError("%{0}: The coordinates of quantile_panel must be ['bin_center', 'quantile']".format(method))

    bins_c = quantile_panel["bin_center"]
    bin_width = bins_c[1] - bins_c[0]
    bins = np.arange(bins_c[0]-bin_width/2., bins_c+1.5*bin_width/2, bin_width)
    quantiles = quantile_panel["quantile"]
    nquantiles = len(quantiles)
    if nquantiles%2 != 1:
        raise ValueError("%{0}: Number of quantiles must be odd.".format(method))

    ls_all = get_ls_mirrored(int(nquantiles/2))
    lw_all = list(np.full(nquantiles, 2.))
    lw_all[int(nquantiles/2)] = 1.5

    # start plotting
    figsize = provide_default(opt, "figsize", (12, 6))
    fig, ax = plt.subplots(figsize=figsize)

    # plot reference line
    ax.plot(bins_c, bins_c, color='k', label='reference 1:1', linewidth=1.)
    # plot conditional quantiles
    for iq in np.arange(nquantiles):
        ax.plot(bins_c, quantile_panel.isel(quantile=iq), ls=ls_all[iq], color="k", lw=lw_all[iq],
                label="{0:d}th quantile".format(int(quantiles[iq]*100.)))
    # plot histogram of marginal distribution
    ax2 = ax.twinx()
    xr.plot.hist(data_marginal, ax=ax2, bins=bins, color="k", alpha=0.3)
    ax2.set_yscale("log")

    ylabel = "{0} [{1}]".format(provide_default(quantile_panel.attr, "data_cond_longname", "conditiong variable"),
                                provide_default(quantile_panel.attr, "data_cond_unit", "unknown"))
    xlabel = "{0} [{1}]".format(provide_default(quantile_panel.attr, "data_cond_longname", "target variable"),
                                provide_default(quantile_panel.attr, "data_cond_unit", "unknown"))

    ax.set_ylabel(ylabel, fontsize=16)
    ax2.set_ylabel("counts", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)

    ax.tick_params(axis="both", labelsize=14)
    ax2.tick_params(axis="both", labelsize=14)

    fig.savefig(plt_fname)
    plt.close("all")


def plot_avg_eval_metrics(eval_ds, eval_metrics, fcst_prod_dict, varname, out_dir):
    """
    Plots error-metrics averaged over all predictions to file incl. 90%-confidence interval that is estimated by
    block bootstrapping.
    :param eval_ds: The dataset storing all evaluation metrics for each forecast (produced by init_metric_ds-method)
    :param eval_metrics: list of evaluation metrics
    :param fcst_prod_dict: dictionary of forecast products, e.g. {"persistence": "pfcst"}
    :param varname: the variable name for which the evaluation metrics are available
    :param out_dir: output directory to save the lots
    :return: a bunch of plots as png-files
    """
    method = plot_avg_eval_metrics.__name__

    # settings for block bootstrapping
    # sanity checks
    if not isinstance(eval_ds, xr.Dataset):
        raise ValueError("%{0}: Argument 'eval_ds' must be a xarray dataset.".format(method))

    if not isinstance(fcst_prod_dict, dict):
        raise ValueError("%{0}: Argument 'fcst_prod_dict' must be dictionary with short names of forecast product" +
                         "as key and long names as value.".format(method))

    try:
        nhours = np.shape(eval_ds.coords["fcst_hour"])[0]
    except Exception as err:
        print("%{0}: Input argument 'eval_ds' appears to be unproper.".format(method))
        raise err

    nmodels = len(fcst_prod_dict.values())
    colors = ["blue", "red", "black", "grey"]
    for metric in eval_metrics:
        # create a new figure object
        fig = plt.figure(figsize=(6, 4))
        ax = plt.axes([0.1, 0.15, 0.75, 0.75])
        hours = np.arange(1, nhours + 1)

        for ifcst, fcst_prod in enumerate(fcst_prod_dict.keys()):
            metric_name = "{0}_{1}_{2}".format(varname, fcst_prod, metric)
            try:
                metric2plt = eval_ds[metric_name + "_avg"]
                metric_boot = eval_ds[metric_name + "_bootstrapped"]
            except Exception as err:
                print("%{0}: Could not retrieve {1} and/or {2} from evaluation metric dataset."
                      .format(method, metric_name, metric_name + "_boot"))
                raise err
            # plot the data
            metric2plt_min = metric_boot.quantile(0.05, dim="iboot")
            metric2plt_max = metric_boot.quantile(0.95, dim="iboot")
            plt.plot(hours, metric2plt, label=fcst_prod, color=colors[ifcst], marker="o")
            plt.fill_between(hours, metric2plt_min, metric2plt_max, facecolor=colors[ifcst], alpha=0.3)
        # configure plot
        plt.xticks(hours)
        # automatic y-limits for PSNR wich can be negative and positive
        if metric != "psnr": ax.set_ylim(0., None)
        legend = ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        ax.set_xlabel("Lead time [hours]")
        ax.set_ylabel(metric.upper())
        plt_fname = os.path.join(out_dir, "evaluation_{0}".format(metric))
        print("Saving basic evaluation plot in terms of {1} to '{2}'".format(method, metric, plt_fname))
        plt.savefig(plt_fname)

    plt.close()

    return True


def create_geo_contour_plot(data, data_diff, varname, plt_fname):
    """
    Creates filled contour plot of forecast data and also draws contours for differences.
    ML: So far, only plotting of the 2m temperature is supported (with 12 predicted hours/frames)
    :param data: the forecasted data array to be plotted
    :param data_diff: the reference data ('ground truth')
    :param varname: the name of the variable
    :param plt_fname: the filename to the store the plot
    :return: -
    """
    method = create_geo_contour_plot.__name__

    try:
        coords = data.coords
        # handle coordinates and forecast times
        lat, lon = coords["lat"], coords["lon"]
        date0 = pd.to_datetime(coords["init_time"].data)
        fhhs = coords["fcst_hour"].data
    except Exception as err:
        print("%{0}: Could not retrieve expected coordinates lat, lon and time_forecast from data.".format(method))
        raise err

    lons, lats = np.meshgrid(lon, lat)

    date0_str = date0.strftime("%Y-%m-%d %H:%M UTC")

    # check data to be plotted since programme is not generic so far
    if np.shape(fhhs)[0] != 12:
        raise ValueError("%{0}: Currently, only 12 hour forecast can be handled properly.".format(method))

    if varname != "2t":
        raise ValueError("%{0}: Currently, only 2m temperature is plotted nicely properly.".format(method))

    # define levels
    clevs = np.arange(-10., 40., 1.)
    clevs_diff = np.arange(0.5, 10.5, 2.)
    clevs_diff2 = np.arange(-10.5, -0.5, 2.)

    # create fig and subplot axes
    fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(12, 6))
    axes = axes.flatten()

    # create all subplots
    for t, fhh in enumerate(fhhs):
        m = Basemap(projection='cyl', llcrnrlat=np.min(lat), urcrnrlat=np.max(lat),
                    llcrnrlon=np.min(lon), urcrnrlon=np.max(lon), resolution='l', ax=axes[t])
        m.drawcoastlines()
        x, y = m(lons, lats)
        if t % 6 == 0:
            lat_lab = [1, 0, 0, 0]
            axes[t].set_ylabel(u'Latitude', labelpad=30)
        else:
            lat_lab = list(np.zeros(4))
        if t / 6 >= 1:
            lon_lab = [0, 0, 0, 1]
            axes[t].set_xlabel(u'Longitude', labelpad=15)
        else:
            lon_lab = list(np.zeros(4))
        m.drawmapboundary()
        m.drawparallels(np.arange(0, 90, 5), labels=lat_lab, xoffset=1.)
        m.drawmeridians(np.arange(5, 355, 10), labels=lon_lab, yoffset=1.)
        cs = m.contourf(x, y, data.isel(fcst_hour=t) - 273.15, clevs, cmap=plt.get_cmap("jet"), ax=axes[t],
                        extend="both")
        cs_c_pos = m.contour(x, y, data_diff.isel(fcst_hour=t), clevs_diff, linewidths=0.5, ax=axes[t],
                             colors="black")
        cs_c_neg = m.contour(x, y, data_diff.isel(fcst_hour=t), clevs_diff2, linewidths=1, linestyles="dotted",
                             ax=axes[t], colors="black")
        axes[t].set_title("{0} +{1:02d}:00".format(date0_str, int(fhh)), fontsize=7.5, pad=4)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=-0.7,
                        wspace=0.05)
    # add colorbar.
    cbar_ax = fig.add_axes([0.3, 0.22, 0.4, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation="horizontal")
    cbar.set_label('°C')
    # save to disk
    plt.savefig(plt_fname, bbox_inches="tight")


# auxiliary functions
def get_ls_mirrored(n, ls_base=("--", ":")):

    nls_base = len(ls_base)
    lss = []
    for ilw in np.arange(n):
        if ilw < nls_base:
            lss.append(ls_base[ilw])
        else:
            lss.append("-")

    lss = lss + ["-"] + lss[::-1]

    return lss