import numpy as np

def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype = np.float64).ravel()
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype = np.float64).ravel()
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype = np.float64).ravel()
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype = np.float64).ravel()

    SSres = np.mean(y_true - y_pred)
    SStot = np.mean(y_true - np.mean(y_true))

    if SSres == 0:
        return 0.0

    return 1 - (SSres / SStot)