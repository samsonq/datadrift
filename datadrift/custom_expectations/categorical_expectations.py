import os
import sys
import numpy as np
import pandas as pd

import great_expectations as ge
import great_expectations.jupyter_ux
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
from great_expectations.exceptions import DataContextError

from .common_expectations import CommonExpectations
from .custom_expectations import *


class CategoricalExpectations(CommonExpectations):
    """
    """
    def __init__(self, validator, feature):
        super().__init__(validator, feature)
        #self.feature = feature
        #self.validator = validator
        #self.expectations = []
        self.bins = ge.dataset.util.build_categorical_partition_object(ge.dataset.PandasDataset(self.validator.active_batch.data.dataframe),
                                                                       self.feature)

        self.expect_column_min_to_be_between()
        self.expect_column_max_to_be_between()
        self.expect_column_kl_divergence_to_be_less_than()
        #self.expect_column_chisquare_test_p_value_to_be_greater_than()

    def get_expectations(self):
        return self.expectations

    def expect_column_chisquare_test_p_value_to_be_greater_than(self, threshold=0.05):
        expectation_configuration = ExpectationConfiguration(**{
              "expectation_type": "expect_column_chisquare_test_p_value_to_be_greater_than",
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

    def expect_column_min_to_be_between(self):
        expectation_configuration = ExpectationConfiguration(**{
              "expectation_type": "expect_column_min_to_be_between",
              "ge_cloud_id": None,
              "meta": {},
              "kwargs": {
                "column": self.feature,
                "max_value": 1.0,
                "min_value": 1.0
              }
            })
        self.expectations.append(expectation_configuration)
        return expectation_configuration

    def expect_column_max_to_be_between(self):
        expectation_configuration = ExpectationConfiguration(**{
              "expectation_type": "expect_column_max_to_be_between",
              "ge_cloud_id": None,
              "meta": {},
              "kwargs": {
                "column": self.feature,
                "max_value": 5.0,
                "min_value": 5.0
              }
            })
        self.expectations.append(expectation_configuration)
        return expectation_configuration
