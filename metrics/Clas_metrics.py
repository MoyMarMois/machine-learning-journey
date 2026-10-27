import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 0) & (y_pred == 1))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    return np.array([[tn, fp], [fn, tp]])

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    return np.mean(y_true == y_pred)

def recall(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if (tp + fn) == 0:
        return 0.0

    return tp / (tp + fn)

def precision(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel() 

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    if (tp + fp) == 0:
        return 0.0

    return tp / (tp + fp)

def f1_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    pr = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    return (2 * pr * rec) / (pr + rec)