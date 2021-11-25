"""
Shared expectations across all types of features in data.
"""
import os
import sys
import numpy as np
import pandas as pd

import great_expectations as ge
import great_expectations.jupyter_ux
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
from great_expectations.exceptions import DataContextError


class CommonExpectations():
    def __init__(self, validator, feature):
        self.feature = feature
        self.validator = validator
        self.expectations = []
        self.bins = None

    def get_expectations(self):
        return self.expectations

    def expect_column_kl_divergence_to_be_less_than(self, threshold=0.05):
        expectation_configuration = ExpectationConfiguration(**{
            "expectation_type": "expect_column_kl_divergence_to_be_less_than",
            "ge_cloud_id": None,
            "meta": {},
            "kwargs": {
                "column": self.feature,
                "partition_object": {
                    "values": self.bins["values"],
                    "weights": self.bins["weights"]
                },
                "threshold": threshold
            }
        })
        self.expectations.append(expectation_configuration)
        return expectation_configuration
