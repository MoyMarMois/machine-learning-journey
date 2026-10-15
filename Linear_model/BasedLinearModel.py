import numpy as np

class BasedLinearModel:
    def __init__(self,
                penalty: str = None,
                alpha: float = 0.1,
                l1_ratio: float = 0.5,
                max_iter: int = 1000,
                lr: float = 0.01,
                random_state: int = 42):

        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state

        self.coefs_ = None
        self.intercept_ = None

    def _calculate_penalty(self, n_samples):
        if self.penalty is None or self.alpha == 0.0:
            return 0.0

        if self.penalty == "l2":
            return (2 * self.alpha / n_samples) * self.coefs_
        elif self.penalty == "l1":
            return (2 * self.alpha / n_samples) * np.sign(self.coefs_)
        elif self.penalty == "elasticnet":
            l2_grad = (2 * self.alpha * (1 - self.l1_ratio)/ n_samples) * self.coefs_
            l1_grad = (2 * self.alpha * self.l1_ratio / n_samples) * np.sign(self.coefs_)
            return l2_grad + l1_grad
        else:
            raise ValueError(f"Неизвестный penalty: {self.penalty}")

    def fit(self, X, y, method: str = "gd", batch_size: int = 32):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self.coefs_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.intercept_ = 0.0

        method = method.lower().replace("_", "-")

        if method == "gd":
            self._fit_gd(X, y, n_samples)
        elif method == "sgd":
            self._fit_sgd(X, y, n_samples, rng)
        elif method == "mini-batch":
            self.batch_size = batch_size
            self._fit_mini_batch(X, y, n_samples, rng)
        else:
            raise ValueError(f"Неизвестный method: {method}")

        return self

    def _fit_gd(self, X, y, n_samples):
        for _ in range(self.max_iter):
            dw, db = self._get_gradient(X, y, n_samples)
            self.coefs_ -= self.lr * dw
            self.intercept_ -= self.lr * db

    def _fit_sgd(self, X, y, n_samples, rng):
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            for idx in indices:
                X_i = X[idx: idx + 1]
                y_i = y[idx: idx + 1]
                dw, db = self._get_gradient(X_i, y_i, n_samples)

                self.coefs_ -= self.lr * dw
                self.intercept_ -= self.lr * db

    def _fit_mini_batch(self, X, y, n_samples, rng):
        for _ in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                dw, db = self._get_gradient(X_batch, y_batch, n_samples)

                self.coefs_ -= self.lr * dw
                self.intercept_ -= self.lr * db

class LinearRegression(BasedLinearModel):
    def _get_gradient(self, X_batch, y_batch, n_samples):
        n_batch = X_batch.shape[0]

        errors = self.intercept_ + X_batch @ self.coefs_ - y_batch
        dw = (2 / n_batch) * X_batch.T @ errors + self._calculate_penalty(n_samples)
        db = (2 / n_batch) * np.sum(errors)

        return dw, db

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.intercept_ + X @ self.coefs_

class LogisticRegression(BasedLinearModel):
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _calculate_penalty(self, n_samples):
            if self.penalty is None or self.alpha == 0.0:
                return 0.0
    
            if self.penalty == "l2":
                return (self.alpha / n_samples) * self.coefs_
            elif self.penalty == "l1":
                return (self.alpha / n_samples) * np.sign(self.coefs_)
            elif self.penalty == "elasticnet":
                l2_grad = (self.alpha * (1 - self.l1_ratio)/ n_samples) * self.coefs_
                l1_grad = (self.alpha * self.l1_ratio / n_samples) * np.sign(self.coefs_)
                return l2_grad + l1_grad
            else:
                raise ValueError(f"Неизвестный penalty: {self.penalty}")

    def _get_gradient(self, X_batch, y_batch, n_samples):
        n_batch = X_batch.shape[0]

        errors = self._sigmoid(self.intercept_ + X_batch @ self.coefs_) - y_batch
        dw = (1 / n_batch) * X_batch.T @ errors
        db = (1 / n_batch) * np.sum(errors)
        return dw, db

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        y_proba = self._sigmoid(self.intercept_ + X @ self.coefs_)
        return np.column_stack((1 - y_proba, y_proba))

    def predict(self, X, threshold: float = 0.5):
        X = np.asarray(X, dtype=np.float64)
        proba = self.predict_proba(X)[:,1]
        return (proba >= threshold).astype(int)