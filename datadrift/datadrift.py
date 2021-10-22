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
        self.past_data = ge.read_csv(past_data)
        self.new_data = ge.read_csv(new_data)
        self.labels = np.unique(self.past_data[label_col])
        
    def _compute_statistics_numerical(self, feature):
        min_past, min_new = np.min(self.past_data[feature]), np.min(self.new_data[feature])
        max_past, max_new = np.max(self.past_data[feature]), np.max(self.new_data[feature])
        median_past, median_new = np.median(self.past_data[feature]), np.median(self.new_data[feature])
        mean_past, mean_new = np.mean(self.past_data[feature]), np.mean(self.new_data[feature])
        std_past, std_new = np.std(self.past_data[feature]), np.std(self.new_data[feature])
        variance_past, variance_new = std_past**2, std_new**2
        kurtosis_past, kurtosis_new = scipy.stats.kurtosis(self.past_data[feature]), scipy.stats.kurtosis(self.new_data[feature])
        skewnewss_past, skewnewss_new = scipy.stats.skew(self.past_data[feature]), scipy.stats.skew(self.new_data[feature])
        self.numerical_statistics_past = {"min": min_past, "max": max_past, "median": median_past, "mean": mean_past, 
                                          "std": std_past, "variance": variance_past, "kurtosis": kurtosis_past, "skew": skewness_past}
        self.numerical_statistics_new = {"min": min_new, "max": max_new, "median": median_new, "mean": mean_new, 
                                         "std": std_new, "variance": variance_new, "kurtosis": kurtosis_new, "skew":skewness_new}
        return self.numerical_statistics_past, self.numerical_statistics_new
    
    def _compute_statistics_categorical(self, feature):
        
        return
    
    def bivariate_correlation(self, feature1, feature2):
        corr_past, p_val_past = scipy.stats.pearsonr(self.past_data[feature1], self.past_data[feature2])
        corr_new, p_val_new = scipy.stats.pearsonr(self.new_data[feature1], self.new_data[feature2])
        print("Correlation of past data: {}, p-value: {}".format(corr_past, p_val_past))
        print("Correlation of new data: {}, p-value: {}".format(corr_new, p_val_new))
        return (corr_past, p_val_past), (corr_new, p_val_new)
    
    def ks_test(self, feature):
        ks_stat, ks_p_val = scipy.stats.ks_2samp(self.past_data[feature], self.new_data[feature])
        