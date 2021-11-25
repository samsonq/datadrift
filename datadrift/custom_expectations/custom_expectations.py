"""
Implemented Custom Expectations for GE
"""
import os
import sys
import random
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy

import great_expectations as ge
from great_expectations.dataset import PandasDataset, MetaPandasDataset
import great_expectations.jupyter_ux
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
from great_expectations.exceptions import DataContextError

import warnings
warnings.filterwarnings("ignore")


class DataDriftDataset(PandasDataset):
    _data_asset_type = "DataDriftDataset"
    pass


class CustomExpectations:
    """
    """
    def __init__(self):
        pass
