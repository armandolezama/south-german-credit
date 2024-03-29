import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from Report_result import Report_result

from sandbox_functions import create_data, create_balanced_sample, assign_categorical_levels, verify_sample_belongs_to_population
from sandbox_functions import standardize_continuous_variables, remove_outliers, principal_components_analysis, get_representative_vars_by_pca
from sandbox_functions import plot_cummulated_variance_from_pca, split_data_into_test_and_train, implement_decision_tree
from sandbox_functions import evaluate_performance_of_decision_tree, get_subsets

class Report:
  def __init__(self, subsets:list) -> None:
    self.current_subsets = subsets
    self.reports:list = [Report_result({'subset': dataset }) for dataset in subsets]

  def add_subset(self, new_subset: pd.DataFrame):
    self.current_subsets = [*self.current_subsets, new_subset]
