# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

def known_datasets():
    """
    An auxilary function
    :return: dictionary of known datasets
    """
    dataset_mappings = {
        "era5": "ERA5Dataset",
        "moving_mnist": "MovingMnist",
        "gzprcp": "GzprcpDataset"
    }

    return dataset_mappings
