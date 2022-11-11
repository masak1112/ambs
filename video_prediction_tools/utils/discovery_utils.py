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
from dataclasses import dataclass

DATASET_META_LOCATION = Path("../data_preprocess")
DATASETS = [file.stem for file in DATASET_META_LOCATION.iterdir() if file.suffix == ".json"]

@dataclass
class DatasetInfo:
    pass

@dataclass(frozen=True)
class Variable:
    name: str
    lvl: List[int]
    interpolation: str
    

def get_dataset_info(dataset: str) -> Dict[str,Any]:
    """Extract metainformation about dataset from corresponding JSON file."""
    file = DATASET_META_LOCATION / f"{dataset}.json"
    try:
        with open(file, "r") as f:
            return json.load(f) # TODO: input validation
    except FileNotFoundError as e:
        raise ValueError("Information on dataset '{dataset}' doesnt exist.")
