from datetime import timedelta
import numpy as np


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
