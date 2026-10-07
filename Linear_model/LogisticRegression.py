import numpy as np

class LogisticRegression:
    def __init__(self, max_iter: int = 1000, lr: float = 0.01, random_state: int = 42):
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state

        self.intercept_ = None
        self.coefs_ = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, method: str = "full_batch", batch_size: int = 42):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.batch_size = batch_size

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.coefs_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.intercept_ = 0.0

        method = method.lower()
        if method == "full_batch":
            self._fit_full_batch(X, y, n_samples)
        elif method == "sgd":
            self._fit_sgd(X, y, n_samples, rng)
        elif method == "mini_batch":
            self._fit_mini_batch(X, y, n_samples, rng)
        else: 
            raise ValueError(f"Неизвестный {method}, введите значение из списка ['full_batch', 'sgd', 'mini_batch']")

        return self

    def _fit_full_batch(self, X, y, n_samples):
        for _ in range(self.max_iter):
            errors = self._sigmoid(self.intercept_ + X @ self.coefs_) - y
            dw = (1 / n_samples) * (X.T @ errors)
            db = (1 / n_samples) * np.sum(errors)

            self.intercept_ -= self.lr * db
            self.coefs_ -= self.lr * dw

    def _fit_sgd(self, X, y, n_samples, rng):
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            for idx in indices:
                X_i = X[idx]
                y_i = y[idx]

                error = self._sigmoid(self.intercept_ + X_i @ self.coefs_) - y_i
                dw = X_i.T * error
                db = error

                self.intercept_ -= self.lr * db
                self.coefs_ -= self.lr * dw

    def _fit_mini_batch(self, X, y, n_samples, rng):
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size

                X_batch = X_shuffled[start_idx: end_idx]
                y_batch = y_shuffled[start_idx: end_idx]
                n_batch = X_batch.shape[0]

                y_pred = self._sigmoid(self.intercept_ + X_batch @ self.coefs_)
                errors = y_pred - y_batch

                dw = (1 / n_batch) * (X_batch.T @ errors)
                db = (1 / n_batch) * np.sum(errors)

                self.intercept_ -= self.lr * db
                self.coefs_ -= self.lr * dw

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        y_proba = self._sigmoid(self.intercept_ + X @ self.coefs_)
        return np.column_stack((1 - y_proba, y_proba))

    def predict(self, X, threshold: float = 0.5):
        X = np.asarray(X, dtype=np.float64)
        proba = self.predict_proba(X)[:,1]
        return (proba >= threshold).astype(int)