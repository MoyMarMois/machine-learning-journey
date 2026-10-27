import numpy as np

class StandardScaler:
    def __init__(self, with_mean: bool = True, with_var: bool = True):
        self.with_mean = with_mean
        self.with_var = with_var

        self.mean_ = None
        self.var_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        if self.with_var:
            self.var_ = np.var(X, axis=0)
            self.var_[self.var_ == 0] = 1
            self.scale_ = np.sqrt(self.var_)
        else:
            self.scale_ = np.ones(X.shape[1])

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.with_mean and self.mean_ is not None:
            X -= self.mean_
        if self.with_var and self.scale_ is not None:
            X  /= self.scale_

        return X

    def fit_transform(self, X):
        X = self.fit(X).transform(X)

        return X