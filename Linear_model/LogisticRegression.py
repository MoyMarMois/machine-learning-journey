import numpy as np

class LogisticRegression:
    def __init__(self, max_iter: int = 1000, lr: float = 0.01, random_state: int = 42):
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state

        self.intercept_ = 0.0
        self.coefs_ = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 /(1 + np.exp(-z))

    def fit(self, X, y, method: str = "batch"):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.coefs_ = np.zeros(n_features)

        if method == "batch":
            self._fit_batch(X, y, n_samples)
        elif method == "sgd":
            self._fit_sgd(X, y, n_samples)
        else:
            raise ValueError(f"Неизвестный {method}, введите значение из списка ['batch', 'sgd']")

        return self

    def _fit_batch(self, X, y, n_samples):
        for _ in range(self.max_iter):
            y_proba = self._sigmoid(self.intercept_ + X @ self.coefs_)

            errors = y_proba - y

            dw = (1 / n_samples) * (X.T @ errors)
            db = (1 / n_samples) * np.sum(errors)

            self.intercept_ -= self.lr * db
            self.coefs_ -= self.lr * dw

    def _fit_sgd(self, X, y, n_samples):
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)

            for idx in indices:
                X_i = X[idx]
                y_i = y[idx]

                y_proba = self._sigmoid(self.intercept_ + X_i @ self.coefs_)

                error = y_proba - y_i

                dw = X_i * error
                db = error

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

