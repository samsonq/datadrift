#################################################
# Encapsulates GE Test Suite Creation Functions #
#                                               #
#################################################
"""
Automate process to generate test suite and expectations for dataset, and then create checkpoints to evaluate datasets. These functions encapsulate the
code/process that is run on the notebooks, to make the process of expectation creation and evaluation much simpler, user-friendly, and efficient for
comparing many datasets.

Generates JSON expectations file in '/validations' folder that contains tests and results. Also creates static HTML dashboard that visualizes
expectations and results of new data compared with old data. Please see the 'main' function below as well as the parameters to specify when running.
"""
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import argparse
import great_expectations as ge
import great_expectations.jupyter_ux
from great_expectations.core.batch import BatchRequest
from great_expectations.profile.user_configurable_profiler import UserConfigurableProfiler
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.exceptions import DataContextError
from great_expectations.cli.datasource import sanitize_yaml_and_save_datasource, check_if_datasource_name_exists
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.data_context.types.resource_identifiers import ExpectationSuiteIdentifier
# from ruamel.yaml import YAML
# yaml = YAML()
import yaml
from pprint import pprint

# sys.path.insert("./custom_expectations", 0)
from custom_expectations import categorical_expectations, continuous_expectations, aggregate_expectations, \
    custom_expectations, utils  # expectations
from parse_results import parse_ge_results  # parsing GE JSON result outputs

import warnings

warnings.filterwarnings("ignore")

'''
def parse_args():
    """
    Get arguments to run Great Expectations tests.
    :return: program arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--datasource_name", type=str, default="data_drift",
                    help="Name of Data Source")
    ap.add_argument("-n", "--datasource_path", type=str, default="data_drift_detection",
                    help="Name of Expectation Suite")
    ap.add_argument("-n", "--overwrite", type=str, default="data_drift_detection",
                    help="Name of Expectation Suite")
    ap.add_argument("-n", "--expectation_suite_name", type=str, default="data_drift_detection",
                    help="Name of Expectation Suite")
    ap.add_argument("-n", "--expectation_data", type=str, default="data_drift_detection",
                    help="Name of Expectation Suite")
    ap.add_argument("-n", "--data_docs", type=str, default="data_drift_detection",
                    help="Name of Expectation Suite")
    ap.add_argument("-n", "--checkpoint_name", type=str, default="data_drift_detection",
                    help="Name of Expectation Suite")
    ap.add_argument("-n", "--checkpoint_data", type=str, default="data_drift_detection",
                    help="Name of Expectation Suite")
    return vars(ap.parse_args())
'''


def create_ge_datasource(datasource_name="data_drift", data_path="../data", overwrite=True):
    """
    Step 1 of the Great Expectations process to create a new datasource with the main data.
    :param datasource_name: name of new datasource
    :param data_path: path to data source
    :param overwrite: boolean to overwrite datasource if existing
    :return: GE datasources object
    """
    context = ge.get_context()
    yaml = f"""
    name: {datasource_name}
    class_name: Datasource
    execution_engine:
      class_name: PandasExecutionEngine
    data_connectors:
      default_inferred_data_connector_name:
        class_name: InferredAssetFilesystemDataConnector
        base_directory: {data_path}
        default_regex:
          group_names: 
            - data_asset_name
          pattern: (.*)
      default_runtime_data_connector_name:
        class_name: RuntimeDataConnector
        batch_identifiers:
          - default_identifier_name
    """
    try:
        yaml_result = context.test_yaml_config(yaml_config=yaml)
        assert len(yaml_result.get_available_data_asset_names()[
                       "default_inferred_data_connector_name"]) > 0, "No data sources available."
    except:
        print("Failed to create new GE datasource.")
        return

    if check_if_datasource_name_exists(context, datasource_name=datasource_name):
        if overwrite:
            sanitize_yaml_and_save_datasource(context, yaml, overwrite_existing=True)
        else:
            print("Data source {} already exists. Set overwrite=True to overwrite data source.".format(datasource_name))
            return
    else:
        sanitize_yaml_and_save_datasource(context, yaml, overwrite_existing=False)

    return context.list_datasources()


def create_ge_expectations_suite(expectation_suite_name="data_drift_detection",
                                 datasource_name="data_drift",
                                 dataset_name="example_data.csv",
                                 categorical_variables=["Prior_Claims"],
                                 continuous_variables=["Age", "Income"],
                                 data_docs=True):
    """
    Step 2 of the Great Expectations process to create a new expectation suite on a GE data source.
    :param expectation_suite_name: name of expectation suite to create
    :param datasource_name: name of datasource
    :param continuous_variables: list of categorical variables to create expectations for
    :param continuous_variables: list of continuous variables to create expectations for
    :param data_docs: boolean of whether to send checkpoint results to GE Data Docs
    :return: number of expectations created for datasource
    """
    context = ge.data_context.DataContext()
    batch_request = {'datasource_name': datasource_name,
                     'data_connector_name': 'default_inferred_data_connector_name',
                     'data_asset_name': dataset_name,
                     'limit': 1000}
    try:
        suite = context.get_expectation_suite(expectation_suite_name=expectation_suite_name)
        print(
            f'Loaded ExpectationSuite "{suite.expectation_suite_name}" containing {len(suite.expectations)} expectations.')
    except DataContextError:
        suite = context.create_expectation_suite(expectation_suite_name=expectation_suite_name)
        print(f'Created ExpectationSuite "{suite.expectation_suite_name}".')

    validator = context.get_validator(batch_request=BatchRequest(**batch_request),
                                      expectation_suite_name=expectation_suite_name)

    expectation_count = 0
    ### Create Data Expectations ###

    ## Table Level Aggregate Expectations ##
    # validator.expect_table_row_count_to_be_between(max_value=173610, min_value=173610)
    aggregate = aggregate_expectations.AggregateExpectations(validator)
    for expectation_config in aggregate.get_expectations():
        suite.add_expectation(expectation_configuration=expectation_config)
        expectation_count += 1

    ## Categorical Variables Expectations ##
    # Distributional Expectations #
    for categorical_var in categorical_variables:
        categorical = categorical_expectations.CategoricalExpectations(validator, categorical_var)
        for expectation_config in categorical.get_expectations():
            suite.add_expectation(expectation_configuration=expectation_config)
            expectation_count += 1

    ## Continuous Variables Expectations ##
    # Distributional Expectations #
    for continuous_var in continuous_variables:
        continuous = continuous_expectations.ContinuousExpectations(validator, continuous_var)
        for expectation_config in categorical.get_expectations():
            suite.add_expectation(expectation_configuration=expectation_config)
            expectation_count += 1

    context.save_expectation_suite(expectation_suite=suite, expectation_suite_name=expectation_suite_name)
    print(context.get_expectation_suite(expectation_suite_name=expectation_suite_name))
    print("{} expectations were created!".format(expectation_count))

    if data_docs:
        suite_identifier = ExpectationSuiteIdentifier(expectation_suite_name=expectation_suite_name)
        context.build_data_docs(resource_identifiers=[suite_identifier])
        context.open_data_docs(resource_identifier=suite_identifier)
    return expectation_count


def create_ge_checkpoint(checkpoint_name="checkpoint",
                         expectation_suite_name="data_drift_detection",
                         datasource_name="data_drift",
                         new_dataset_name="example_data_for_validation.csv",
                         data_docs=True):
    """
    Step 3 of the Great Expectations process to introduce a new dataset and run/validate the data through previously created expectations.
    :param checkpoint_name: name of checkpoint to create
    :param expectation_suite_name: name of expectation suite to create
    :param datasource_name: name of datasource
    :param new_dataset_name: name of new dataset to validate and create checkpoint with
    :param data_docs: boolean of whether to send checkpoint results to GE Data Docs
    :return: checkpoint configuration
    """
    context = ge.get_context()
    yaml_config = f"""
    name: {checkpoint_name}
    config_version: 1.0
    class_name: SimpleCheckpoint
    run_name_template: "%Y%m%d-%H%M%S-my-run-name-template"
    validations:
      - batch_request:
          datasource_name: {datasource_name}
          data_connector_name: default_inferred_data_connector_name
          data_asset_name: {new_dataset_name}
          data_connector_query:
            index: -1
        expectation_suite_name: {expectation_suite_name}
    """
    pprint(context.get_available_data_asset_names())  # print available datasources and expectation suites
    try:
        checkpoint = context.test_yaml_config(yaml_config=yaml_config)
    except:
        print("Failed to create GE checkpoint.")
    checkpoint_config = checkpoint.get_substituted_config().to_yaml_str()
    print(checkpoint_config)  # print checkpoint config
    context.add_checkpoint(**yaml.load(yaml_config))  # save checkpoint
    if data_docs:
        context.run_checkpoint(checkpoint_name=checkpoint_name)
        context.open_data_docs()
    print("Done creating checkpoint {}.".format(checkpoint_name))
    return checkpoint_config


def detect_drift():
    ge_config = yaml.load(open("ge_config.yaml", "r"))
    # args = parse_args()  # inputted arguments

    # context = ge.data_context.DataContext()
    # batch_request = {'datasource_name': args["datasource"], 'data_connector_name': 'default_inferred_data_connector_name', 'data_asset_name': 'april.csv', 'limit': 1000}
    # expectation_suite_name = args["name"]
    # validator = context.get_validator(batch_request=BatchRequest(**batch_request),
    #                                  expectation_suite_name=expectation_suite_name)

    # column_names = [f'"{column_name}"' for column_name in validator.columns()]
    # print(f"Data Columns: {', '.join(column_names)}.")

    # 1. Create Data Source
    create_ge_datasource(datasource_name=ge_config["datasource_name"],
                         data_path=ge_config["datasource_path"],
                         overwrite=ge_config["overwrite"])

    # 2. Create Expectation Suite
    create_ge_expectations_suite(expectation_suite_name=ge_config["expectation_suite_name"],
                                 datasource_name=ge_config["datasource_name"],
                                 dataset_name=ge_config["expectation_data"],
                                 categorical_variables=ge_config["categorical_variables"],
                                 continuous_variables=ge_config["continuous_variables"],
                                 data_docs=ge_config["data_docs"])

    # 3. Create Checkpoint
    create_ge_checkpoint(checkpoint_name=ge_config["checkpoint_name"],
                         expectation_suite_name=ge_config["expectation_suite_name"],
                         datasource_name=ge_config["datasource_name"],
                         new_dataset_name=ge_config["checkpoint_data"],
                         data_docs=ge_config["data_docs"])

    # Parse GE Results
    parse_ge_results()


if __name__ == "__main__":
    detect_drift()
