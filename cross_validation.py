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
        self._dev_segment = self.parse_segmentation(self._dev_segment) if isinstance(self._dev_segment, dict) else self._dev_segment
        self._select_segment = self.parse_segmentation(self._select_segment) if isinstance(self._select_segment, dict) else self._select_segment
        self._eval_target = conditions['eval_criteria'].get('eval_target', [self._target,]) if 'eval_criteria' in conditions else [self._target,]
        self._eval_metric = conditions['eval_criteria'].get('eval_metric', [roc_auc_score,]) if 'eval_criteria' in conditions else [roc_auc_score,]
        self._eval_segment = conditions['eval_criteria'].get('eval_segment', ['__all__',]) if 'eval_criteria' in conditions else ['__all__',]
        self._eval_segment = [self.parse_segmentation(x) if isinstance(x, dict) else x for x in self._eval_segment]
        self._df = df
        self._df['cv_estimates'] = np.nan
        self._eval_result = []

    @property
    def target(self):
        return self._target

    @property
    def model_features(self):
        return self._model_features

    @property
    def df(self):
        return self._df

    @property
    def eval_result(self):
        return self._eval_result

    # Optional method to parse the segmentation criteria and convert into list of values
    def oarse_segmentation(self, segmentation: dict):
        """
        Optional method to parse the segmentation criteria and convert into list of values.
        Input segmentation dictionary containing pairs of (column_name, list of values to segment)
        Output segmentation dictionary that strictly in format of {string: list}
        Override if the input segmentation criteria not in the form of list
        """
        for key, values in segmentation.items():
            assert isinstance(values, list), f"the segment values of {key} is not a list, please override 'parse_segmentation' method to convert into list"
        return segmentation

    # Segment the dataset based on segment dictionary
    def get_segment(self, df, segm_info):
        """
        create the subset of the given data based on the provided segmentation information.
        """
        if segm_info == '__all__':
            return df
        assert isinstance(segm_info, dict), f"the segment values of {key} is not a list, please override 'parse_segmentation' method to convert into list"
        for key, values in segm_info.items():
            df = df[df[key].isin(values)]
        return df

    # @abstractmethod
    # def get_data(self):
    #     """
    #     Abstract method to get data based on conditions.
    #     Input: original dataset and condition dictionary.
    #     Output: the dataset used to build model.
    #     Must be implemented in a subclass.
    #     """
    #     pass

    # Optional features selection methods
    def get_selection(self, df, target, features, params, *arg, **kwargs):
        """
        Optional features selection methods to perform additional feature selection prior to cross validation modeling.
        Input: df, target, features, params, and additional parameters.
        Output: list of selected features.
        Implement if needed in a subclass.
        """
        # raise NotImplementedError("the 'get_selection' method should be implemented")
        print("No feature selection method has been implemented, all provided features will be used")
        print("Please implement your own feature selection method if needed")
        return features

    # Wrap up the feature selection function
    def feature_selection(self, df=None, target=None, features=None, params=None, select_segment=None, *args, **kwargs):
        """
        Wrap up the feature selection function,
        if feature selection not implemented, all starting features will be used
        """
        select_segment = self._select_segment if select_segment is None else select_segment
        df = self.get_segment(self._df, select_segment) if df is None else df
        target = self._target if target is None else target
        features = self._features if features is None else features
        params = self._select_params if params is None else params
        self._model_features = self.get_selection(df[df[target].notna()], target, features, params, *args, **kwargs)

    # Generate cross-validation fold index
    def get_cv_index(self, n_folds, random_state, stratify=None):
        """
        Generate cross-validation fold index.
        Input: number of folds, random state, stratify column.
        Output: add 'CV_INDEX' column to the model df.
        Override if needed in a subclass.
        """
        np.random.seed(random_state)
        if stratify is not None:
            self._df['CV_INDEX'] = self._df_groupby(stratify)[stratify].transform(lambda x: np.random.randint(n_folds, size=len(x)))
        else:
            self._df['CV_INDEX'] = np.random.randint(n_folds, size=self._df.shape[0])

    # Split the dataset into test, valid, and train at each iteration
    def get_model_data(self, i, dev_segment=None):
        """
        At each cross-validation iteration,
        Split the dataset into test and dev,
        Apply the segmentation criteria to test to get validation dataset, and to dev to get train dataset.
        """
        dev_segment = self._dev_segment if dev_segment is None else dev_segment
        test = self._df[self._df['CV_INDEX']==i]
        valid = self.get_segment(test, dev_segment)
        train = self.get_segment(self._df[self._df['CV_INDEX']!=i], dev_segment)
        return train, valid, test

    @abstractmethod
    def get_model(self):
        """
        Train the model based on train and valid data at current iteration.
        Input: 'train', 'valid', 'params', 'target', 'features'.
        Output: trained model. (the model object must be implemented with 'predict' method if not modify the 'get_scores' method to match)
        Must be implemented in a subclass
        """
        pass

    # Score the given dataset with provided model
    def get_scores(self, model, df, features):
        """
        Score the given dataset with provided model.
        Input: model object, scoring dataset, features.
        Output: model scores
        Current implementation needs 'predict' method for model object, override if not
        """
        return model.predict(df[features])

    # Fit the cross-validation model
    def fit(self, n_folds, random_state, stratify=None, params=None, dev_segment=None, target=None, features=None):
        """
        Perform the cross-validation modeling and score the entire dataset as 'cv_estimate'
        Input: hyperparameters used for cv model training, number of cv folds, random state, stratify column
        Output: update the 'cv_estimate' column for the entire dataset
        Override if needed in a subclass
        """
        params = self._train_params if params is None else params
        dev_segment = self._dev_segment is dev_segment is None else dev_segment
        target = self._target is target is None else target
        features = self._model_features is features is None else features

        self.get_cv_index(n_folds=n_folds, random_state=random_state, stratify=stratify)
        for i in range(n_folds):
            train, valid, test = self.get_model_data(i, dev_segment=dev_segment)
            model = self.get_model(train=train[train[target].notna()], valid=valid[valid[target].notna()], params=params, target=target, features=features)
            self._df.loc[test.index, 'cv_estimates'] = self.get_scores(model=model, df=self._df.loc[test.index], features=features)

    # Evaluate the test scores
    def eval(self, eval_segment=None, eval_metric=None, eval_target=None):
        """
        Evaluate the test score on different segments, different targets, and metrics
        """
        eval_segment = self._eval_segment if eval_segment is None else eval_segment
        eval_metric = self._eval_metric if eval_metric is None else eval_metric
        eval_target = self._eval_target if eval_target is None else eval_target
        self._eval_result = []

        for segment in eval_segment:
            df = self.get_segment(self._df, segment)
            for target in eval_target:
                for metric in eval_metric:
                    _df = df[df[target].notna() & df['cv_estimates'].notna()]
                    value = metric(_df[target], _df['cv_estimates'])
                    result = segment.copy() if isinstance(segment, dict) else {}
                    result['eval_target'] = target
                    result['eval_metric'] = metric.__name__
                    result['value'] = value
                    self._eval_result.append(result)
        return self._eval_result


# pre-defined cross-validation evaluation with lightgbm model and selection algorithm
# must implement the 'get_model' method
# optional 'get_selection' method

class CV_LightGBM(CV_Core):
    import lightgbm as lgb
    from lightgbm_functions import BackwardSelector, get_selected_features

    def get_model(self, train, valid, params, target, features):
        # train, valid = train[train[target].notna()], valid[valid[target].notna()]
        lgb_train = lgb.Dataset(train[features], train[target], free_raw_data = False)
        lgb_val = lgb.Dataset(valid[features], valid[target], reference = lgb_train, free_raw_data = False)

        eval_results = {}
        record_eval_callback = lgb.record_evaluation(eval_results)
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round = 500,
            valid_sets = lgb_val,
            valid_names = ['train', 'valid'],
            callbacks = [record_eval_callback]
        )
        return model

    def get_selection(self, df, target, features, params, n_splits=4, n_repeats=2, exhaustive_search_threshold=40, drop_fraction=0.05, *args, **kwargs):
        selector = BackwardSelector(target, features, params, df, n_splits=n_splits, n_repeats=n_repeats)
        selector.fit(exhaustive_search_threshold, drop_fraction)
        self._selector = selector
        return get_selected_features(selector, max_features=25)
