import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sandbox_functions import create_data, create_balanced_sample, assign_categorical_levels, verify_sample_belongs_to_population
from sandbox_functions import standardize_continuous_variables, remove_outliers, principal_components_analysis, get_representative_vars_by_pca
from sandbox_functions import plot_cummulated_variance_from_pca, split_data_into_test_and_train, implement_decision_tree
from sandbox_functions import evaluate_performance_of_decision_tree, get_subsets

class Report_result:
  def __init__(self, data: dict) -> None:
    self.result_data = data
    self.subset:pd.DataFrame = data['subset']

  def plot_histograms(self, columns = ['duration', 'amount', 'age']):

    for col in columns:
        plt.figure(figsize=(8, 6))
        plt.hist(self.subset[col], bins=30, color='blue', alpha=0.7)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

  def check_balance(self, target = 'credit_risk'):
      counts = np.bincount(self.subset[target])

      ones_count = counts[1]
      zeros_count = counts[0]

      return ones_count == zeros_count

  def plot_boxplots(self, columns = ['duration', 'amount', 'age']):

    for col in columns:
        plt.figure(figsize=(8, 6))
        self.subset.boxplot(column=col)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

  def count_outliers(self, columns = ['duration', 'amount', 'age']):
    outlier_counts = {}
    inlier_counts = {}

    for col in columns:
        Q1 = self.subset[col].quantile(0.25)
        Q3 = self.subset[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.subset[(self.subset[col] < lower_bound) | (self.subset[col] > upper_bound)]

        outlier_counts[col] = len(outliers)
        inlier_counts[col] = len(self.subset[col]) - outlier_counts[col]

    return outlier_counts, inlier_counts
