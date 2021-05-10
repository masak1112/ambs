from typing import Union, Tuple, Dict, List
import numpy as np
import xarray as xr
import pandas as pd
try:
    from tqdm import tqdm
    l_tqdm = True
except:
    l_tqdm = False


def perform_block_bootstrap_metric(metric: xr.DataArray, dim_name: str, block_length: int, nboots_block: int = 1000,
                                   seed: int = 42):

    method = perform_block_bootstrap_metric.__name__

    if dim_name not in metric.dims:
        raise ValueError("%{0}: Passed dimension cannot be found in passed metric.".format(method))

    metric = metric.sortby(dim_name)

    dim_length = np.shape(metric.coords[dim_name].values)[0]
    nblocks = int(np.floor(dim_length/block_length))

    if nblocks < 10:
        raise ValueError("%{0}: Less than 10 blocks are present with given block length {1:d}.".format(method, block_length) +
                         " Too less for bootstrapping.")

    # get remaining coordinates and dimensions
    dims_old = list(metric.dims)
    dims_old.remove(dim_name)

    coords_new_block = {**{"iblock": np.arange(nblocks)}, **metric.drop("init_time").coords}
    coords_new_boot = {**{"iboot": np.arange(nboots_block)}, **metric.drop("init_time").coords}
    
    shape_block = tuple([a.shape[0] for a in coords_new_block.values()])
    shape_boot = tuple([a.shape[0] for a in coords_new_boot.values()])
    
    metric_val_block = xr.DataArray(np.full(shape_block, np.nan), coords=coords_new_block, dims=["iblock"] + dims_old)
    metric_boot = xr.DataArray(np.full(shape_boot, np.nan), coords=coords_new_boot,
                               dims=["iboot"] + dims_old)

    # precompute metrics of block
    for iblock in np.arange(nblocks):
        ind_s, ind_e = iblock * block_length, (iblock + 1) * block_length
        metric_val_block[iblock,...] = metric.isel({dim_name: slice(ind_s, ind_e)}).mean(dim=dim_name)

    # get random blocks
    np.random.seed(seed)
    iblocks_boot = np.sort(np.random.randint(nblocks, size=(nboots_block, nblocks)))

    print("%{0}: Start block bootstrapping...".format(method))
    iterator_b = np.arange(nboots_block)
    if l_tqdm:
        iterator_b = tqdm(iterator_b)
    for iboot_b in iterator_b:
        metric_boot[iboot_b,...] = metric_val_block.isel(iblock=iblocks_boot[iboot_b, :]).mean(dim="iblock")

    return metric_boot


class Scores:
    """
    Class to calculate scores and skill scores.
    """
    def __init__(self, score_name):

        self.metrics_dict = {"mse": Scores.calc_mse_batch , "psnr": Scores.calc_psnr_batch}
        self.score_name = Scores.set_score_name(score_name)
        self.score_func = self.metrics_dict[score_name]
        # attributes set when run_calculation is called
        self.avg_dims = None

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

    def set_model_and_ref_data(self, model_data: xr.DataArray, ref_data: xr.DataArray, dims2avg: List[str] = None):

        method = Scores.set_score_name.__name__

        coords = model_data.coords
        if not list(coords) == list(ref_data.coords):
            raise ValueError("%{0}: Input data arrays must have the same shape and coordinates.".format(method))

        if dims2avg is None:
            self.avg_dims = dims2avg
        else:
            for dim in dims2avg:
                if not dim in coords:
                    raise ValueError("%{0}: Dimension '{1}' does not exist in model and reference data"
                                     .format(method, dim))

        return model_data, ref_data

    def calc_mse_batch(self, data_fcst, data_ref, **kwargs):
        """
        Calculate mse of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged mse for each batch example
        """
        method = Scores.calc_mse_batch.__name__

        if kwargs is not None:
            print("%{0}: Passed keyword arguments are without effect.".format(method))
        # sanity checks
        if self.avg_dims is None:
            print("%{0}: Squeared difference is averaged over all dimensions.".format(method))
            dims = list(data_fcst.dims)
        else:
            dims = self.avg_dims

        mse = np.square(data_fcst - data_ref).mean(dim=dims)

        return mse

    def calc_psnr_batch(self, data_fcst, data_ref, **kwargs):
        """
        Calculate mse of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged mse for each batch example
        """
        method = Scores.calc_mse_batch.__name__

        if kwargs is not None:
            print("%{0}: Passed keyword arguments are without effect.".format(method))

        psnr = metrics.psnr_imgs(data_ref.values, data_fcst.values)

        return psnr


