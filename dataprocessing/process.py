from datetime import timedelta
from itertools import product
import xgboost as xgb

import numpy as np
import pandas as pd


class TrainTestSplit:

    def __init__(self, test_size=0.2, days_between=365):
        self.test_size = test_size
        self.days_between = days_between

        self.train_end_date = None
        self.val_start_date = None
        self.ratio = None

    def __call__(self, *args):
        return self.make(*args)

    def make(self, train_features, train_outcomes):
        train_ind = train_features.index
        train_full_dates = train_ind.levels[0]
        train_before_year_date = train_full_dates[-1] - timedelta(days=self.days_between)
        train_before_year = train_features[:train_before_year_date]
        train_before_year_unique_dates = train_before_year.index.get_level_values(0).unique()
        train_max_date_num = int(len(train_before_year_unique_dates) * (1 - self.test_size))
        self.train_end_date = train_before_year_unique_dates[train_max_date_num]

        X_train, y_train = train_features.loc[:self.train_end_date], train_outcomes.loc[:self.train_end_date]

        self.val_start_date = self.train_end_date + timedelta(days=self.days_between)
        X_val, y_val = train_features.loc[self.val_start_date:], train_outcomes.loc[self.val_start_date:]

        self.ratio = (round(len(X_train) / (len(X_train) + len(X_val)) * 100),
                      round(len(X_val) / (len(X_train) + len(X_val)) * 100))

        return X_train, X_val, y_train, y_val


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = X.mean()
        self.std_ = X.std()

    def transform(self, X):
        if self.mean_ is not None and self.std_ is not None:
            X = X - self.mean_
            X = X / self.std_
            return X
        else:
            raise RuntimeError("You need to do the fit method before")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SmartLogtransformer:

    def __init__(self):
        self.smartshift = None

    def fit(self, X):
        self.smartshift = (np.clip(X.min(), -np.inf, 0)).abs()

    def transform(self, X):
        if self.smartshift is not None:
            X = np.log(np.clip(X + self.smartshift, 0, np.inf) + 1e-12)
            return X
        else:
            raise RuntimeError("You need to do the fit method before")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Pipeline:

    def __init__(self, steps: list):
        self.steps = steps

    def fit_transform(self, X):
        for name, trans in self.steps:
            X = trans.fit_transform(X)
        return X

    def fit(self, X):
        self.fit_transform(X)

    def transform(self, X):
        for name, trans in self.steps:
            X = trans.transform(X)
        return X

    def __str__(self):
        return "[" + ', \n'.join([f"({', '.join([name, str(obj)[:-30] + '>'])})"
                                  for name, obj in self.steps]) + "]"

    def __repr__(self):
        return self.__str__()


class GridSearchXgb:
    """
    Search of the max
    """
    def __init__(self, xgb_train_params, param_grid, scoring=None, verbose=False, score_fn=None):
        """

        :param xgb_train_params:
        :param param_grid:
        :param scoring:
        :param verbose:
        :param score_fn: the custom function from param_grid.keys() that return evaluating score. It is tuple where first
        point is score and second point is num_boost_round.
        """
        self.xgb_train_params = xgb_train_params
        self.param_grid = param_grid
        self.scoring = scoring
        self.verbose = verbose
        self.score_fn = score_fn

        self.iterator = None
        self.scores = None
        self.best_score_ = None
        self.best_params_ = None
        self.best_index_ = None
        self.cv_results_ = None
        self.cv_results_df_ = None
        self.best_num_boost_round = None
        self.best_xgb_train_params = None

    def get_score(self, cv_params):
        if self.score_fn is not None:
            return self.score_fn(cv_params)
        else:
            params_ = self.xgb_train_params.copy()
            for key in cv_params:
                params_["params"][key] = cv_params[key]
            bst = xgb.train(**params_)
            score = bst.best_score
            num_boost_round = bst.best_ntree_limit
            return score, num_boost_round

    def fit(self, verbose=None):
        # initialization
        if verbose is not None:
            self.verbose = verbose
        self.cv_results_ = []
        self.best_score_ = float("inf")
        self.best_params_ = None
        self.best_index_ = None
        self.best_num_boost_round = None
        self.best_xgb_train_params = None

        self.iterator = list(product(*self.param_grid.values()))
        for num, item in enumerate(self.iterator):
            cv_params = dict(zip(self.param_grid.keys(), item))
            score, num_boost_round = self.get_score(cv_params)
            if score < self.best_score_:
                self.best_score_ = score
                self.best_params_ = cv_params
                self.best_index_ = num
                self.best_num_boost_round = num_boost_round

            self.cv_results_.append(score)

            if self.verbose:
                if len(self.param_grid) == 1:
                    print(f"cv_params: {cv_params}, score: {score}")
                else:
                    if num == 0:
                        print("total:", len(self.iterator))
                    if (num + 1) % 10 == 0:
                        print(num + 1, end="")
                    else:
                        print(".", end="")
                    if num == len(self.iterator) - 1:
                        print()

        self.cv_results_ = np.array(self.cv_results_)

        self.cv_results_df_ = pd.DataFrame({"score": self.cv_results_}, index=self.iterator)
        self.cv_results_df_.index.name = ", ".join(self.param_grid.keys())

        self.best_xgb_train_params = self.xgb_train_params.copy()
        self.best_xgb_train_params["params"] = self.best_params_
