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
    pdf=None,
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

    condition = (len(columns) > exhaustive_search_threshold)
    while condition:
        importances = score_folds(columns)
        cumulative = importances.sort_values(ascending=True).cumsum().sort_values(ascending=False)
        keeps = list(cumulative[cumulative > importances.sum()*drop_fraction].index)
        drops = [i for i in columns if i not in keeps]
        if len(drops) > 100:
            drops = str(len(drops)) + ' variables'
        columns = keeps
        auc = pd.DataFrame(metrics[len(importances)]).T['AUC' + seg].mean()
        auc_train = pd.DataFrame(metrics[len(importances)]).T['AUC' + ' DEV'].mean()
        print('\nDropping:', drops, ',', len(columns), 'vars remaining, AUC:', round(auc, 4), 'training AUC:', round(auc_train, 4), end='')
        condition = (len(columns) > exhaustive_search_threshold) and (len(drops) > 0)

    print('\nIteratively drop feature that improves AUC the least. This will ensure highly correlated features are dropped first.', end='')
    while len(columns) > 1:
        auc_dict = {}
        for var in columns:
            auc_dict[var] = 0

        for train_index, test_index in skf.split(df, df[target]):
            train, test = df.iloc[train_index, :], df.iloc[test_index, :]
            train_weights = train['DEV']['SAMPLE_WEIGHT'] if weighted else None
            valid_weights = test['VAL']['SAMPLE_WEIGHT'] if weighted else None

            # For each variable, train a new model with every variable except that one
            for var in columns:
                except_var = [i for i in columns if i != var]
                gb1.fit(
                    train[except_var],
                    train[target],
                    sample_weight=train_weights,
                    eval_metric='auc',
                    eval_set=[(test[except_var], test[target])],
                    eval_sample_weight=[valid_weights],
                )
                y = gb1.predict_proba(test[except_var])[:, 1]
                auc_dict[var] = auc_dict[var] + roc_auc_score(test[target], y)

        # Drop variable that decreased AUC the least, (get max of auc_dict)
        drop_var = max(auc_dict, key=auc_dict.get)
        columns = [i for i in columns if i != drop_var]
        importances = score_folds(columns)
        auc = pd.DataFrame(metrics[len(importances)]).T['AUC' + seg].mean()
        auc_train = pd.DataFrame(metrics[len(importances)]).T['AUC' + ' DEV'].mean()
        print('\nDropping:', drop_var, ',', len(columns), 'vars remaining AUC:', round(auc, 4), 'training AUC:', round(auc_train, 4), end='')

    print(' Final Var: ', columns[0])

    # Plot fold AUCs
    keys = list(metrics.keys())

    # ax2 = ax1.twinx()
    colors = {'VAL': 'teal', 'OOT1': 'grey', 'OOT2': 'red'}
    for metric in ['AUC', 'KS']:
        fig, ax1 = plt.subplots(nrow=1, ncols=1, figsize=(12, 6), facecolor='white')
        plot_dfs = {}
        for seg in [j for j in list(dfs.keys()) if j not in ['DEV+VAL', 'DEV']]:
            plot_dfs[seg] = pd.DataFrame()
            for i in keys:
                plot_dfs[seg] = pd.concat([plot_dfs[seg], pd.DataFrame.from_dict(metrics[i], orient='index')[metric + ' ' + seg]], axis=1)
            plot_dfs[seg].columns = keys
            means = plot_dfs[seg].mean()
            stds = plot_dfs[seg].std()

            # Create violin plots
            plot_data = [np.random.normal(means.iloc[i], stds.iloc[i], size=1000) for i in range(len(means))]
            parts = ax1.violinplot(plot_data, width=.9, showmeans=False, showextrema=False, showmedians=False, points=1000)
            color = colors[seg]
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(.3)

        ax1.set_ylabel(metric)
        ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.01))
        ax1.set_xticks([i for i in range(plot_dfs[seg].shape[1] + 1)])
        ax1.set_xticklabels([''] + keys)
        ax1.set_xlabel('Number of Features')
        ax1.grid()

        import matplotlib.patches as mpatches
        teal = mpatches.Patch(color='teal', label='Validation')
        grey = mpatches.Patch(color='grey', label='OOT1')
        red = mpatches.Patch(color='red', label='OOT2')
        plt.legend(handles=[teal, grey, red])
        plt.title(plot_title + '\n' + metric + ' Confidence Interval')
        if pdf != None:
            pdf.savefig(fig)
        plt.show()

    var_df = pd.DataFrame([variables.keys(), variables.values()]).T
    var_df.columns = ['num_features', 'features']
    for seg in [j for j in list(dfs.keys()) if j not in ['DEV+VAL']]:
        for metric in ['AUC', 'KS']:
            var_df[seg + ' ' + metric + ' mean'] = [round(pd.DataFrame(metric[i]).T[metric + ' ' + seg].mean(), 4) for i in metrics.keys()]
            var_df[seg + ' ' + metric + ' std'] = [round(pd.DataFrame(metric[i]).T[metric + ' ' + seg].std(), 4) for i in metrics.keys()]

    warnings.resetwarnings()
    pd.options.mode.chained_assignment = 'warn'

    return var_df

# Plot ROC & KS plot

def plot_roc_rank(targets, estimate, df, title='', subtitle='', pdf=None, savefig=None):
    import sklearn.metrics as metrics
    from sklearn.metrics import roc_auc_score
    from scipy.stats import ks_2samp

    # Plot validation ROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title(subtitle + ' ROC Curve')
    for target in targets:
        val = df.dropna(subset=[target, estimate])
        preds = val[estimate]
        y_test = val[target]
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
        auc = metrics.auc(fpr, tpr)
        ks = round(ks_2samp(val.loc[val[target]==0, estimate], val.loc[val[target]==1, estimate])[0], 4)
        label = target + +' AUC = %0.3f' % auc + '\n' + target + ' KS = %0.3f' % ks
        if target == targets[-1]:
            label = label + '\nLoans: ' + str(len(val))
        ax1.plot(fpr, tpr, label=label)
    ax1.legend(loc='lower right')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=.3)
    ax1.set_xlim([-.02, 1.02])
    ax1.set_ylim([-.02, 1.02])
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positive Rate')
    ax1.grid(True)

    # Plot rank order
    edges = [val[estimate].quantiles((1+1)/10) for i in range(10)]
    val['bin'] = 0
    for i in range(len(edges)):
        val.loc[val[estimate] > edges[i], 'bin'] = i + 1
    val['bin'] = val['bin'] + 1
    labels = list(val.bin.unique())
    labels.sort()
    bins = np.arange(1, len(labels) + 0.5, 1)
    for target in targets:
        bads = val[['bin', target]].groupby(val['bin']).mean()[target]
        ax2.plot(bins, bads, label=target + ' Rate')
    estimates = val[['bin', estimate]].groupby(val['bin']).mean()[estimate]

    ax2.plot(bins, estimates, label='Model Average Estimate', color='grey')
    ax2.set_ylabel('Bad Rate')
    ax2.set_xlabel('Model Score Decile')
    ax2.set_xticks(bins)
    ax2.set_xticklabels(labels=labels)
    ax2.set_title(subtitle + ' Rank Order')
    ax2.set_ylim([-.02, 1.02])
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if pdf is not None:
        pdf.savefig(fig)
    if savefig:
        fig.savefig(f'{rawgraphpath}\{savefile}')
    ptl.show()

    return auc, ks
    
# PDP Plot function

def plot_pdps(train_df, target, bins=20, pdf=None, gbm=None, vars=[], plot_pdp=True, plot_estimate=True, dictionary=None, savefile=None, rawgraphpath=None):
    # df should have var, target, and model as columns
    df = train_df.copy()
    df[vars] = df[vars].fillna(-999999)
    if savefile:
        filenames = []

    for var in vars:

        if isinstance(gbm, lgb.basic.Booster):
            df['Model_Estimate'] = gbm.predict(df[vars].replace(-999999, np.nan))
        else:
            df['Model_Estimate'] = gbm.predict_proba(df[vars].replace(-999999, np.nan))[:, 1]
        temp = df[vars].copy()
        uniques = list(df[var].sample(200).unique())
        uniques.sort()
        uniques_dict = {}
        for i in uniques:
            temp[var] = i
            if isinstance(gbm, lgb.basic.Booster):
                uniques_dict[i] = gbm.predict(temp.replace(-999999, np.nan)).mean()  
            else: 
                uniques_dict[i] = gbm.predict_proba(temp.replace(-999999, np.nan))[:,:1].mean()
        
        pdp = pd.DataFrame.from_dict(uniques_dict, orient='index')
        pdp.columns = ['values']
        pdp['average'] = pdp.index
        df[var] = df[var].astype(float)

        if len(df[var].unique()) <= 3:
            df['edges'] = df[var].astype(float)
            target_rate = df[[var, target]].groupby(df[var], observed=False).mean()[target]
            model_estimate = df[[var, 'Model_Estimate']].groupby(df[var], observed=False).mean()['Model_Estimate']
            pdp[var] = pdp['average'].astype(float)
            pdp = pdp[[var, 'values']].groupby(pdp[var], observed=False).mean()['values'].dropna()
        else:
            array = np.arange()