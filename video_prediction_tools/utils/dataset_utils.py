# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
functions providing info about available options
Provides:   * DATASET_META_LOCATION
            * DATASETS
            * get_dataset_info
"""

import json
from pathlib import Path
from typing import Dict, Any, List
#from dataclasses import dataclass #TODO use dataclass in python 3.7+

DATASET_META_LOCATION = Path(__file__).parent.parent / "data_split"
DATASETS = [path.name for path in DATASET_META_LOCATION.iterdir() if path.is_dir()]

DATE_TEMPLATE = "{year}-{month:02d}"

def get_filename_template(name: str) -> str:
    return f"{name}_{DATE_TEMPLATE}.nc"

def get_dataset_info(name: str) -> Dict[str,Any]:
    """Extract metainformation about dataset from corresponding JSON file."""
    file = DATASET_META_LOCATION / f"{name}/{name}.json"
    try:
        with open(file, "r") as f:
            return json.load(f) # TODO: input validation => specify schema
    except FileNotFoundError as e:
        raise ValueError("Information on dataset '{dataset}' doesnt exist.")
