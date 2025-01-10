import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from abc import ABC, abstractmethod


class CV_Core(ABC):
  def __init__(self, df, conditions: dict):
    """
    dataset (pd.DataFrame) and the condition dictionary need to be passed to construct the class.
    condition dictionary contains the information (key):
        1. target: string
        2. features: string
        3. select_params: dict
        4. train_params: dict
        5. dev_segment: dict (containing key-value pairs of column_name: string and values: list)
        6. select_segment: dict (same as dev_segment)
        7. eval_criteria: dict
            a. eval_target: list of string
            b. eval_metric: list of functions
            c. eval_segment: list of dict (same as dev_segment)
    only 'target' is necessary key, others are optional
    """
    if 'target' not in conditions:
        raise KeyError('target name is not specified in condition')

    self._target = conditions['target']
    self._features = condtions.get('features', [x for x in df.columns if x!=condtions['target']])
    self._model_features = self._features
    self._select_params = conditions.get('select_params', {})
    self._train_params = conditions.get('train_params', {})
    self._dev_segment = conditions.get('dev_segment', '__all__')
    self._select_segment = conditions.get('select_segment', self._dev_segment)
