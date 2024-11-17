import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBClassifier
from collections import defaultdict
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from skleran.metrics import roc_auc_score


# Customized backward feature elimination with cross-validation

def drop_unimportant_variables(
    data: pd.DataFrame, 
    gb1: LGBClassifier, 
    target: str, # target column in df
    columns: list=None, # if columns is None then all numeric columns will be used
    oot1: pd.DataFrame=None,
    oot2: pd.DataFrame=None,
    exhaustive_search_threshold: int=30, # when to switch to exhaustive search (1 drop at a time)
    n_splits: int=4, # number of splits in RepeatedStratifiedKFold
    n_repeats: int=2, # number of repeats in RepeatedStratifiedKFold
    random_state: int=1,
    drop_fraction: float=0.05, # what fraction of feature importance to drop at each step (group dropping)
    pdf: bool=None,
    plot_title: str='',
    weighted: bool=False
    ):
    warnings.simplefilter("ignore", Warning)
    pd.options.mode.chained_assignment = None

    """
    Repeatedly drops the least importance features (recursive backwards feature elimination)
    from the model while tracking performance with cross-fold validation.
    """

    # If columns is None then all numeric columns will be used
    if columns is None:
        columns = data.columns
    columns = list(set(columns).intersection(set(data.select_dtypes(include='number').columns)))
    columns = [i for i in columns if i not in [target]]
    seg = 'VAL'

    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    df = data.loc[(~data[target].isna())].reset_index(drop=True)
    dfs, variables, metrics = {}, {}, {}

    if oot1 is not None:
        dfs['OOT1'] = oot1
    if oot2 is not None:
        dfs['OOT2'] = oot2
    dfs['DEV+VAL'] = df

    # Train the first gbm on all features available
    print('Training first model with', len(columns), 'available variables', end='')
    df = dfs['DEV+VAL']
    if weighted:
        print('Sample Weights Used')

    def score_folds(columns: str):
        """
        Train n_splits * n_repeats models using the features in columns and calculates the
        AUC and KS for each model and fold held out by RepeatedStratifiedKFold validation.
        Returns the total feature importance across all folds for each column in columns.
        """

        folds = {}
        importances = pd.Series(0, index=columns)

        for train_index, test_index in skf.split(df, df[target]):

            dfs['DEV'], dfs['VAL'] = df.iloc[train_index, :], df.iloc[test_index, :]
            train_weights = dfs['DEV']['SAMPLE_WEIGHT'] if weighted else None
            valid_weights = dfs['VAL']['SAMPLE_WEIGHT'] if weighted else None

            gb1.fit(
                dfs['DEV'][columns],
                dfs['DEV'][target],
                sample_weight=train_weights,
                eval_set=[(dfs['VAL'][columns], dfs['VAL'][target])],
                eval_sample_weight=[valid_weights]
            )

            fold = {}
            for i in dfs.keys():
                dfs[i]['y'] = gb1.predict_proba(dfs[i][columns])[:, 1]
                fold['AUC ' + i] = roc_auc_score(dfs[i][target], dfs[i]['y'])
                fold['KS ' + i] = ks_2samp(dfs[i].loc[dfs[i][target]==0, 'y'], dfs[i].loc[dfs[i][target]==1, 'y'])[0]
            folds[len(folds)] = fold

            # Track cumulative importance across all folds
            importances = importances + pd.Series(gb1.feature_importances_, columns)
            print('.', end='')

        variables[len(importances)] = columns
        metrics[len(importances)] = folds

        return importances