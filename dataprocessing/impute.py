class SimpleImputer:

    def __init__(self, strategy="median"):
        if strategy in ["median", "mean"]:
            self.strategy = strategy
        else:
            raise ValueError("strategy is not valid")
        self.values_for_fill = None

    def fit(self, X):
        if self.strategy == "median":
            self.values_for_fill = X.median()
        elif self.strategy == "mean":
            self.values_for_fill = X.mean()
        else:
            raise ValueError("strategy is not valid")

    def transform(self, X):
        if self.values_for_fill is not None:
            return X.fillna(self.values_for_fill)
        else:
            raise RuntimeError("You need to do the fit method before")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
