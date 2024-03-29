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
    self.generated_subsets:dict = {}
    self.pca_results:dict = {}

  def plot_histograms(self, columns = ['duration', 'amount', 'age'], use_named_set = False, set_name = ''):
    subset = self.generated_subsets[set_name] if(use_named_set) else self.subset

    for col in columns:
      plt.figure(figsize=(8, 6))
      plt.hist(subset[col], bins=30, color='blue', alpha=0.7)
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

  def plot_boxplots(self, columns = ['duration', 'amount', 'age'], use_named_set = False, set_name = ''):
    subset = self.generated_subsets[set_name] if(use_named_set) else self.subset

    for col in columns:
      plt.figure(figsize=(8, 6))
      subset.boxplot(column=col)
      plt.title(f'Histogram of {col}')
      plt.xlabel(col)
      plt.ylabel('Frequency')
      plt.grid(True)
      plt.show()

  def count_outliers(self, columns = ['duration', 'amount', 'age'], use_named_set = False, set_name = ''):
    subset = self.generated_subsets[set_name] if(use_named_set) else self.subset
    outlier_counts = {}
    inlier_counts = {}

    for col in columns:
      Q1 = subset[col].quantile(0.25)
      Q3 = subset[col].quantile(0.75)
      IQR = Q3 - Q1

      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR

      outliers = self.subset[(subset[col] < lower_bound) | (subset[col] > upper_bound)]

      outlier_counts[col] = len(outliers)
      inlier_counts[col] = len(subset[col]) - outlier_counts[col]

    return outlier_counts, inlier_counts

  def remove_outliers(self, columns = ['duration', 'amount', 'age'], outliers_id = 'outliers', inliers_id = 'inliers', use_named_set = False, set_name = ''):
    subset = self.generated_subsets[set_name] if(use_named_set) else self.subset

    no_outliers, outliers = remove_outliers(subset, columns)

    self.generated_subsets[outliers_id] = outliers
    self.generated_subsets[inliers_id] = no_outliers

  def principal_components_analysis(self, use_named_set = False, set_name = ''):
    subset = self.generated_subsets[set_name] if(use_named_set) else self.subset
    pca_id = f'{set_name}-pca' if(use_named_set) else 'full-subset-pca'

    pca_result, pca_instance = principal_components_analysis(subset)

    representative_columns = get_representative_vars_by_pca(subset, pca_instance)

    self.pca_results[pca_id] = {
      'results': pca_result,
      'instance': pca_instance,
      'representative_columns': representative_columns,
    }

    print(f'Variables that most accurately account for the variation: {representative_columns}')
    print(len(representative_columns))

  def plot_variance_from_pca(self, use_named_set = False, set_name = ''):
    pca_id = f'{set_name}-pca' if(use_named_set) else 'full-subset-pca'

    plot_cummulated_variance_from_pca(self.pca_results[pca_id]['instance'].explained_variance_ratio_)
