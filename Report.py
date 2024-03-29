import pandas as pd
from Report_result import Report_result

class Report:
  def __init__(self, subsets:list) -> None:
    self.current_subsets = subsets
    self.reports:list = [Report_result({'subset': dataset }) for dataset in subsets]

  def add_subset(self, new_subset: pd.DataFrame):
    self.current_subsets = [*self.current_subsets, new_subset]
