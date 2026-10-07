import numpy as np

class DataSplitter:
    def __init__(self, valid_size: float = 0.2, test_size: float = 0.1, random_state: int = 42):
        self.valid_size = valid_size
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(n_samples)

        valid_count = int(n_samples * self.valid_size)
        test_count = int(n_samples * self.test_size)

        valid_idx = indices[:valid_count]
        test_idx = indices[valid_count: test_count + valid_count]
        train_idx = indices[valid_count + test_count:]

        return X[train_idx], y[train_idx], X[valid_idx], y[valid_idx], X[test_idx], y[test_idx]
        