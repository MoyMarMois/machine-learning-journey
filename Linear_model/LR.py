import numpy as np

class LinearRegressionButch:
    def __init__(self, n_iter: int = 1000, lr: float = 0.01):
        self.n_iter = n_iter
        self.lr = lr

        self.intercept_ = 0.0
        self.coefs_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        self.coefs_ = np.zeros(n_features)

        for _ in range(self.n_iter):
            y_pred = self.intercept_ + X @ self.coefs_

            errors = y - y_pred

            dw = (2 / n_samples) * (X.T @ errors)
            db = (2 / n_samples) * np.sum(errors)

            self.coefs_ -= self.lr * dw
            self.intercept_ -= self.lr * db

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.intercept + X @ self.coefs


class LinearRegressionSGD:
    def __init__(self, n_iter: int = 1000, lr: float = 0.01, random_state: int = 42):
        self.n_iter = n_iter
        self.lr = lr
        self.random_state = random_state

        self.coefs_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.coefs_ = np.zeros(n_features)

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_iter):

            indices = rng.permutation(n_samples)

            for idx in indices:

                X_i = X[idx]
                y_i = y[idx]

                y_pred = self.intercept_ + np.dot(X_i, self.coefs_)

                error = y_pred - y_i

                dw = 2 * X_i * error
                db = 2 * error

                self.intercept_ -= self.lr * db
                self.coefs_ -= self.lr * dw

    def predict(self, X):
        X = np.asarray(X)
        return self.intercept_ + X @ self.coefs_
