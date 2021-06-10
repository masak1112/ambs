"""
Collection of auxiliary functions for statistical evaluation and class for Score-functions
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2021-05-xx"

import numpy as np
import xarray as xr
from typing import Union, List
from skimage.measure import compare_ssim as ssim

try:
    from tqdm import tqdm
    l_tqdm = True
except:
    l_tqdm = False
from general_utils import provide_default

# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]


def calculate_cond_quantiles(data_fcst: xr.DataArray, data_ref: xr.DataArray, factorization="calibration_refinement",
                             quantiles=(0.05, 0.5, 0.95)):
    """
    Calculate conditional quantiles of forecast and observation/reference data with selected factorization
    :param data_fcst: forecast data array
    :param data_ref: observational/reference data array
    :param factorization: factorization: "likelihood-base_rate" p(m|o) or "calibration_refinement" p(o|m)-> default
    :param quantiles: conditional quantiles
    :return quantile_panel: conditional quantiles of p(m|o) or p(o|m)
    """
    method = calculate_cond_quantiles.__name__

    # sanity checks
    if not isinstance(data_fcst, xr.DataArray):
        raise ValueError("%{0}: data_fcst must be a DataArray.".format(method))

    if not isinstance(data_ref, xr.DataArray):
        raise ValueError("%{0}: data_ref must be a DataArray.".format(method))

    if not (list(data_fcst.coords) == list(data_ref.coords) and list(data_fcst.dims) == list(data_ref.dims)):
        raise ValueError("%{0}: Coordinates and dimensions of data_fcst and data_ref must be the same".format(method))

    nquantiles = len(quantiles)
    if not nquantiles >= 3:
        raise ValueError("%{0}: quantiles must be a list/tuple of at least three float values ([0..1])".format(method))

    if factorization == "calibration_refinement":
        data_cond = data_fcst
        data_tar = data_ref
    elif factorization == "likelihood-base_rate":
        data_cond = data_ref
        data_tar = data_fcst
    else:
        raise ValueError("%{0}: Choose either 'calibration_refinement' or 'likelihood-base_rate' for factorization"
                         .format(method))

    # get and set some basic attributes
    data_cond_longname = provide_default(data_cond.attrs, "longname", "conditioning_variable")
    data_cond_unit = provide_default(data_cond.attrs, "unit", "unknown")

    data_tar_longname = provide_default(data_tar.attrs, "longname", "target_variable")
    data_tar_unit = provide_default(data_cond.attrs, "unit", "unknown")

    # get bins for conditioning
    data_cond_min, data_cond_max = np.floor(np.min(data_cond)), np.ceil(np.max(data_cond))
    bins = list(np.arange(int(data_cond_min), int(data_cond_max) + 1))
    bins_c = 0.5 * (np.asarray(bins[0:-1]) + np.asarray(bins[1:]))
    nbins = len(bins) - 1

    # get all possible bins from target and conditioning variable
    data_all_min, data_all_max = np.minimum(data_cond_min, np.floor(np.min(data_tar))),\
                                 np.maximum(data_cond_max, np.ceil(np.max(data_tar)))
    bins_all = list(np.arange(int(data_all_min), int(data_all_max) + 1))
    bins_c_all = 0.5 * (np.asarray(bins_all[0:-1]) + np.asarray(bins_all[1:]))
    # initialize quantile data array
    quantile_panel = xr.DataArray(np.full((len(bins_c_all), nquantiles), np.nan),
                                  coords={"bin_center": bins_c_all, "quantile": list(quantiles)},
                                  dims=["bin_center", "quantile"],
                                  attrs={"cond_var_name": data_cond_longname, "cond_var_unit": data_cond_unit,
                                         "tar_var_name": data_tar_longname, "tar_var_unit": data_tar_unit})
    
    print("%{0}: Start caclulating conditional quantiles for all {1:d} bins.".format(method, nbins))
    # fill the quantile data array
    for i in np.arange(nbins):
        # conditioning of ground truth based on forecast
        data_cropped = data_tar.where(np.logical_and(data_cond >= bins[i], data_cond < bins[i + 1]))
        # quantile-calculation
        quantile_panel.loc[dict(bin_center=bins_c[i])] = data_cropped.quantile(quantiles)

    return quantile_panel, data_cond


def avg_metrics(metric: da_or_ds, dim_name: str):
    """
    Averages metric over given dimension
    :param metric: DataArray or Dataset of metric that should be averaged
    :param dim_name: name of the dimension on which division into blocks is applied
    :return: DataArray or Dataset of metric averaged over given dimension. If a Dataset is passed, the averaged metrics
             carry the suffix "_avg" in their variable names.
    """
    method = perform_block_bootstrap_metric.__name__

    if not isinstance(metric, da_or_ds.__args__):
        raise ValueError("%{0}: Input metric must be a xarray DataArray or Dataset and not {1}".format(method,
                                                                                                       type(metric)))

    if isinstance(metric, xr.Dataset):
        list_vars = [varname for varname in metric.data_vars if dim_name in metric[varname].dims]
        if not list_vars:
            raise ValueError("%{0}: {1} is not a dimension in the input metric dataset".format(method, dim_name))

        metric2avg = metric[list_vars]
    else:
        if dim_name in metric.dims:
            pass
        else:
            raise ValueError("%{0}: {1} is not a dimension in the input metric data-array".format(method, dim_name))

        metric2avg = metric

    metric_avg = metric2avg.mean(dim=dim_name)

    if isinstance(metric, xr.Dataset):
        new_varnames = ["{0}_avg".format(var) for var in list_vars]
        metric_avg = metric_avg.rename(dict(zip(list_vars, new_varnames)))

    return metric_avg


def perform_block_bootstrap_metric(metric: da_or_ds, dim_name: str, block_length: int, nboots_block: int = 1000,
                                   seed: int = 42):
    """
    Performs block bootstrapping on metric along given dimension (e.g. along time dimension)
    :param metric:  DataArray or dataset of metric that should be bootstrapped
    :param dim_name: name of the dimension on which division into blocks is applied
    :param block_length: length of block (index-based)
    :param nboots_block: number of bootstrapping steps to be performed
    :param seed: seed for random block sampling (to be held constant for reproducability)
    :return: bootstrapped version of metric(-s)
    """

    method = perform_block_bootstrap_metric.__name__

    if not isinstance(metric, da_or_ds.__args__):
        raise ValueError("%{0}: Input metric must be a xarray DataArray or Dataset and not {1}".format(method,
                                                                                                       type(metric)))
    if dim_name not in metric.dims:
        raise ValueError("%{0}: Passed dimension cannot be found in passed metric.".format(method))

    metric = metric.sortby(dim_name)

    dim_length = np.shape(metric.coords[dim_name].values)[0]
    nblocks = int(np.floor(dim_length/block_length))

    if nblocks < 10:
        raise ValueError("%{0}: Less than 10 blocks are present with given block length {1:d}."
                         .format(method, block_length) + " Too less for bootstrapping.")

    # precompute metrics of block
    for iblock in np.arange(nblocks):
        ind_s, ind_e = iblock * block_length, (iblock + 1) * block_length
        metric_block_aux = metric.isel({dim_name: slice(ind_s, ind_e)}).mean(dim=dim_name)
        if iblock == 0:
            metric_val_block = metric_block_aux.expand_dims(dim={"iblock": 1}, axis=0).copy(deep=True)
        else:
            metric_val_block = xr.concat([metric_val_block, metric_block_aux.expand_dims(dim={"iblock": 1}, axis=0)],
                                         dim="iblock")

    metric_val_block["iblock"] = np.arange(nblocks)

    # get random blocks
    np.random.seed(seed)
    iblocks_boot = np.sort(np.random.randint(nblocks, size=(nboots_block, nblocks)))

    print("%{0}: Start block bootstrapping...".format(method))
    iterator_b = np.arange(nboots_block)
    if l_tqdm:
        iterator_b = tqdm(iterator_b)
    for iboot_b in iterator_b:
        metric_boot_aux = metric_val_block.isel(iblock=iblocks_boot[iboot_b, :]).mean(dim="iblock")
        if iboot_b == 0:
            metric_boot = metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0).copy(deep=True)
        else:
            metric_boot = xr.concat([metric_boot, metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0)], dim="iboot")

    # set iboot-coordinate
    metric_boot["iboot"] = np.arange(nboots_block)
    if isinstance(metric_boot, xr.Dataset):
        new_varnames = ["{0}_bootstrapped".format(var) for var in metric.data_vars]
        metric_boot = metric_boot.rename(dict(zip(metric.data_vars, new_varnames)))

    return metric_boot


class Scores:
    """
    Class to calculate scores and skill scores.
    """

    known_scores = ["mse", "psnr","ssim"]

    def __init__(self, score_name: str, dims: List[str]):
        """
        Initialize score instance.
        :param score_name:   name of score that is queried
        :param dims:         list of dimension over which the score shall operate
        :return:             Score instance
        """
        method = Scores.__init__.__name__

        self.metrics_dict = {"mse": self.calc_mse_batch , "psnr": self.calc_psnr_batch, "ssim":self.calc_ssim_batch}
        if set(self.metrics_dict.keys()) != set(Scores.known_scores):
            raise ValueError("%{0}: Known scores must coincide with keys of metrics_dict.".format(method))
        self.score_name = self.set_score_name(score_name)
        self.score_func = self.metrics_dict[score_name]
        # attributes set when run_calculation is called
        self.avg_dims = dims

    def run_calculation(self, model_data, ref_data, dims2avg=None, **kwargs):

        method = Scores.run_calculation.__name__

        model_data, ref_data = Scores.set_model_and_ref_data(model_data, ref_data, dims2avg=dims2avg)

        try:
            if self.avg_dims is None:
                result = self.score_func(model_data, ref_data, **kwargs)
            else:
                result = self.score_func(model_data, ref_data, **kwargs)
        except Exception as err:
            print("%{0}: Calculation of '{1}' was not successful. Inspect error message!".format(method,
                                                                                                 self.score_name))
            raise err

        return result

    def set_score_name(self, score_name):

        method = Scores.set_score_name.__name__

        if score_name in self.metrics_dict.keys():
            return score_name
        else:
            print("The following scores are currently implemented:".format(method))
            for score in self.metrics_dict.keys():
                print("* {0}".format(score))
            raise ValueError("%{0}: The selected score '{1}' cannot be selected.".format(method, score_name))

    def calc_mse_batch(self, data_fcst, data_ref, **kwargs):
        """
        Calculate mse of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged mse for each batch example
        """
        method = Scores.calc_mse_batch.__name__

        if kwargs:
            print("%{0}: Passed keyword arguments are without effect.".format(method))
        # sanity checks
        if self.avg_dims is None:
            print("%{0}: Squared difference is averaged over all dimensions.".format(method))
            dims = list(data_fcst.dims)
        else:
            dims = self.avg_dims

        mse = np.square(data_fcst - data_ref).mean(dim=dims)

        return mse

    def calc_psnr_batch(self, data_fcst, data_ref, **kwargs):
        """
        Calculate psnr of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged psnr for each batch example
        """
        method = Scores.calc_mse_batch.__name__

        if "pixel_max" in kwargs:
            pixel_max = kwargs.get("pixel_max")
        else:
            pixel_max = 1.

        mse = self.calc_mse_batch(data_fcst, data_ref)
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.
        else:
            psnr = 20.*np.log10(pixel_max / np.sqrt(mse))

        return psnr


    def calc_ssim_batch(self, data_fcast, data_ref, **kwargs):
        """
        Calculate ssim ealuation metric of forecast data w.r.t reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged ssim for each batch example
        """
        method = Scores.calc_ssim_batch.__name__
        print("shape of data_ref:",np.array(data_ref).shape)
        print("shape of data_fcast:",np.array(data_fcast).shape)
        print("max values of data forecast",data_fcast.max())
        print("min value of data forecadst",data_fcast.min())
        ssim_pred = ssim(data_ref[0,0,:,:], data_fcast[0,0,:,:])
        return ssim_pred
