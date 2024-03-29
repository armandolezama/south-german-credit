import pandas as pd
from Report_result import Report_result
from sandbox_functions import get_subsets
class Report:
  def __init__(self, subsets:list) -> None:
    self.current_subsets = subsets
    self.reports:list = [Report_result({'subset': dataset }) for dataset in subsets]

  def add_subset(self, new_subset: pd.DataFrame):
    self.current_subsets = [*self.current_subsets, new_subset]
    self.reports = [*self.reports, Report_result({'subset': new_subset })]

  def generate_subsets(self, data_set, columns):
    return get_subsets(data_set, columns=columns)

  def add_multiple_subsets(self, new_subsets:list):
    self.current_subsets = [*self.current_subsets, *new_subsets]
    self.reports = [*self.reports, *[Report_result({'subset': dataset }) for dataset in new_subsets]]
