import os
import sys
import numpy as np
import pandas as pd

import great_expectations as ge
import great_expectations.jupyter_ux
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
from great_expectations.exceptions import DataContextError


class AggregateExpectations:
    """
    """
    def __init__(self, validator):
        self.expectations = []
        self.validator = validator

        # TODO: list out all expect functions in class and call each of them, automate process
        self.expect_table_row_count_to_be_between()

    def get_expectations(self):
        return self.expectations

    def expect_table_row_count_to_be_between(self):
        """
        """
        expectation_configuration = ExpectationConfiguration(**{
            "expectation_type": "expect_table_row_count_to_be_between",
            "ge_cloud_id": None,
            "meta": {},
            "kwargs": {
                "max_value": 173610,
                "min_value": 100000
            }
        })
        self.expectations.append(expectation_configuration)
        return expectation_configuration
