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

class Report_result:
  def __init__(self, data: dict) -> None:
    self.result_data = data
    self.subset = data['subset']

  def plot_histograms(self):

    columns = ['duration', 'amount', 'age']

    for col in columns:
        plt.figure(figsize=(8, 6))
        plt.hist(self.subset[col], bins=30, color='blue', alpha=0.7)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

