# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ == "2022-03-17"

from .base_dataset  import BaseDataset


class ERA5Dataset(BaseDataset):
    
    def __init__(self, *args, **kwargs):
        super(BaseDataset, *args, **kwargs)
        
        
