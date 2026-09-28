import numpy as np

class LinearRegressionBatch:
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
            y_pred = self.intercept_ + np.dot(X, self.coefs_)

            errors = y_pred - y

            dw = (2 / n_samples) * np.dot(X.T, errors)
            db = (2 / n_samples) * np.sum(errors)

            self.intercept_ -= self.lr * db
            self.coefs_ -= self.lr * dw

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + np.dot(X, self.coefs_)

class LinearRegressionSGD:
    def __init__(self, epochs: int = 1000, lr: float = 0.01, random_state: int = 42):
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

        self.intercept_ = 0.0
        self.coefs_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.coefs_ = np.zeros(n_features)

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.epochs):
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
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + np.dot(X, self.coefs_)

class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 0.01, random_state: int = 42):
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

        self.intercept_ = 0.0
        self.coefs_ = None

    def fit(self, X, y, method: str = "Batch"):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.coefs_ = np.zeros(n_features)

        if method == "Batch":
            self._fit_batch(X, y, n_samples)
        elif method == "SGD":
            self._fit_sgd(X, y, n_samples)
        else:
            raise ValueError(f"Неизвестный {method}, введите значение из списка ['Batch', 'SGD']")

    def _fit_batch(self, X, y, n_samples: int):
        for _ in range(self.epochs):
            y_pred = self.intercept_ + X @ self.coefs_
            errors = y_pred - y

            dw = (2 / n_samples) * (X.T @ errors)
            db = (2 / n_samples) * np.sum(errors)

            self.intercept_ -= self.lr * db
            self.coefs_ -= self.lr * dw

    def _fit_sgd(self, X, y, n_samples: int):
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.epochs):
            indices = rng.permutation(n_samples)

            for idx in indices:
                X_i = X[idx]
                y_i = y[idx]

                y_pred = self.intercept_ + X_i @ self.coefs_
                error = y_pred - y_i

                dw = 2 * X_i * error
                db = 2 * error

                self.intercept_ -= self.lr * db
                self.coefs_ -= self.lr * dw

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + X @ self.coefs_