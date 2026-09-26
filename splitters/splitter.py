import numpy as np

class splitter:
    def __init__(self, test_size: float = 0.2, validate_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.validate_size = validate_size
        self.random_state = random_state

    def split(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)

        indices = rng.permutation(n_samples)

        test_count = int(n_samples * self.test_size)
        valid_count = int(n_samples * self.validate_size)

        test_idx = indices[:test_count]
        valid_idx = indices[test_count:test_count + valid_count]
        train_idx = indices[test_count + valid_count:]

        return X[train_idx], y[train_idx], X[valid_idx], y[valid_idx], X[test_idx], y[test_idx]