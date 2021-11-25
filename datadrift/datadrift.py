import os
import sys
from tqdm import tqdm
import random
import math
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from matplotlib import pyplot as plt
import seaborn as sns


class DataDriftDetection(object):
    """
    """
    def __init__(self, past_data, new_data, label_col):
        """

        :param past_data:
        :param new_data:
        :param label_col:
        """
        self.past_data = pd.read_csv(past_data)
        self.new_data = pd.read_csv(new_data)
        self.labels = np.unique(self.past_data[label_col])
        
    def _compute_statistics_numerical(self, feature):
        """

        :param feature:
        :return:
        """
        min_past, min_new = np.min(self.past_data[feature]), np.min(self.new_data[feature])
        max_past, max_new = np.max(self.past_data[feature]), np.max(self.new_data[feature])
        median_past, median_new = np.median(self.past_data[feature]), np.median(self.new_data[feature])
        mean_past, mean_new = np.mean(self.past_data[feature]), np.mean(self.new_data[feature])
        std_past, std_new = np.std(self.past_data[feature]), np.std(self.new_data[feature])
        variance_past, variance_new = std_past**2, std_new**2
        kurtosis_past, kurtosis_new = scipy.stats.kurtosis(self.past_data[feature]), scipy.stats.kurtosis(self.new_data[feature])
        skewness_past, skewness_new = scipy.stats.skew(self.past_data[feature]), scipy.stats.skew(self.new_data[feature])
        self.numerical_statistics_past = {"min": min_past, "max": max_past, "median": median_past, "mean": mean_past, 
                                          "std": std_past, "variance": variance_past, "kurtosis": kurtosis_past, "skew": skewness_past}
        self.numerical_statistics_new = {"min": min_new, "max": max_new, "median": median_new, "mean": mean_new, 
                                         "std": std_new, "variance": variance_new, "kurtosis": kurtosis_new, "skew":skewness_new}
        return self.numerical_statistics_past, self.numerical_statistics_new
    
    def _compute_statistics_categorical(self, feature):
        """

        :param feature:
        :return:
        """
        
        return

    def plot_numerical(self, feature, bivariate=False):
        """

        :param feature:
        :return:
        """
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.subplot(2, 2, 1)
        if bivariate:
            sns.histplot(data=self.past_data, x=feature, hue=self.labels, kde=True)
        else:
            sns.histplot(data=self.past_data, x=feature, kde=True)
        plt.title("Baseline Data Distribution for {}".format(feature))
        plt.subplot(2, 2, 2)
        if bivariate:
            sns.histplot(data=self.new_data, x=feature, hue=self.labels, kde=True)
        else:
            sns.histplot(data=self.new_data, x=feature, kde=True)
        plt.title("New Data Distribution for {}".format(feature))
        plt.show()

    def plot_categorical(self, feature):
        """

        :param feature:
        :return:
        """
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        plt.subplot(2, 2, 1)
        sns.barplot(x=x_column, y=y_column, data=self.past_data)
        plt.title("Baseline Data Distribution for {}".format(feature))
        plt.subplot(2, 2, 2)
        sns.barplot(x=x_column, y=y_column, data=self.new_data)
        plt.title("New Data Distribution for {}".format(feature))
        plt.show()
    
    def bivariate_correlation(self, feature1, feature2):
        """

        :param feature1:
        :param feature2:
        :return:
        """
        corr_past, p_val_past = scipy.stats.pearsonr(self.past_data[feature1], self.past_data[feature2])
        corr_new, p_val_new = scipy.stats.pearsonr(self.new_data[feature1], self.new_data[feature2])
        print("Correlation of past data: {}, p-value: {}".format(corr_past, p_val_past))
        print("Correlation of new data: {}, p-value: {}".format(corr_new, p_val_new))
        return (corr_past, p_val_past), (corr_new, p_val_new)
    
    def ks_test(self, feature):
        """

        :param feature:
        :return:
        """
        ks_stat, ks_p_val = scipy.stats.ks_2samp(self.past_data[feature], self.new_data[feature])
        print("KS Statistic: {}, p-value: {}".format(ks_stat, ks_p_val))
        return ks_stat, ks_p_val
