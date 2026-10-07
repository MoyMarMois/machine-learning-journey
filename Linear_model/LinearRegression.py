import numpy as np

class LinearRegression:
    def __init__(self, max_iter: int = 1000, lr: float = 0.01, random_state: int = 42):
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state

        self.intercept_ = None
        self.coefs_ = None

    def fit(self, X, y, method: str = "gd", batch_size: int = 40):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.intercept_ = 0.0
        self.coefs_ = rng.normal(loc = 0.0, scale=0.01, size = n_features)

        method = method.lower()

        if method == "gd":
            self._fit_gd(X, y, n_samples)
        elif method == "sgd":
            self._fit_sgd(X, y, n_samples, rng)
        elif method == "mini-batch":
            self.batch = batch_size
            self._fit_mini_batch(X, y, n_samples, rng)
        else:
            raise ValueError(f"Неизвестный {method}, введите значение из списка ['gd', 'sgd', 'mini-batch']")

        return self

    def _fit_gd(self, X, y, n_samples):
        for _ in range(self.max_iter):
            y_pred = self.intercept_ + X @ self.coefs_
            errors = y_pred - y

            db = (2 / n_samples) * np.sum(errors)
            dw = (2 / n_samples) * (X.T @ errors)

            self.intercept_ -= self.lr * db
            self.coefs_ -= self.lr * dw

    def _fit_sgd(self, X, y, n_samples, rng):
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)

            for idx in indices:
                X_i = X[idx]
                y_i = y[idx]

                y_pred = self.intercept_ + X_i @ self.coefs_
                error = y_pred - y_i

                db = 2 * error
                dw = 2 * X_i * error

                self.intercept_ -= self.lr * db
                self.coefs_ -= self.lr * dw

    def _fit_mini_batch(self, X, y, n_samples, rng):
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, self.batch):
                end_idx = start_idx + self.batch

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                n_batch = X_batch.shape[0]

                y_pred = self.intercept_ + X_batch @ self.coefs_
                errors = y_pred - y_batch

                db = (2 / n_batch) * np.sum(errors)
                dw = (2 / n_batch) * X_batch.T @ errors

                self.intercept_ -= self.lr * db
                self.coefs_ -= self.lr * dw

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + X @ self.coefs_