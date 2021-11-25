"""
Parse JSON results of Great Expectations output to identify failed tests and the corresponding features involved. Reduces the
complexity of the JSON validations into a user-friendly and easily-interpretable format to quickly identify
potential features that have data drift.
"""
import os
import sys
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")


def parse_ge_results(path, save_path="."):
    """
    Parses Great Expectations JSON validation results to identify failed tests and the corresponding features. Also identifies
    any tests that threw exceptions during validation.
    :param path: path to JSON validation
    :param save_path: path to save CSV results of expectations
    #:return: (dict) {feature: [failed tests]}, (dict) {feature: [test exceptions]}
    :return: Pandas dataframe of features and failed tests
    """
    ge_results = {}
    exceptions = {}
    with open(path, "r") as f:
        ge_json = json.load(f)

    print("Parsing {} total expectations...".format(len(ge_json["results"])))
    for expectation in tqdm(ge_json["results"]):
        if "column" not in expectation["expectation_config"]["kwargs"].keys():
            continue
        feature_name = expectation["expectation_config"]["kwargs"]["column"]
        test = expectation["expectation_config"]["expectation_type"]
        if expectation["exception_info"]["raised_exception"]:
            if feature_name not in exceptions.keys():
                exceptions[feature_name] = [test]
            else:
                ge_results[feature_name].append(test)
        if not expectation["success"]:
            try:
                expected_value = [expectation["expectation_config"]["kwargs"]["min_value"],
                                  expectation["expectation_config"]["kwargs"]["max_value"]]
                observed_value = expectation["result"]["observed_value"]
            except:
                expected_value = "TBD"
                observed_value = "TBD"
            if feature_name not in ge_results.keys():
                ge_results[feature_name] = [[test, expected_value, observed_value]]
            else:
                ge_results[feature_name].append([test, expected_value, observed_value])

    results = pd.DataFrame({"Feature": [], "Test": [], "Type": [], "Expected": [], "Actual": []})
    print("Extracting expectation results from JSON...")
    for feature, tests in tqdm(ge_results.items(), total=len(ge_results)):
        for test in tests:
            results.loc[len(results)] = [feature, test[0], "Failed", test[1], test[2]]

    for feature, tests in tqdm(exceptions.items(), total=len(exceptions)):
        for test in tests:
            results.loc[len(results)] = [feature, test[0], "Exception", test[1], test[2]]

    results.to_csv(os.path.join(save_path, "ge_results.csv"))
    # return ge_results, exceptions
    return results
