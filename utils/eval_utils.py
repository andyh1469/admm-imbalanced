import numpy as np
from sklearn.metrics import accuracy_score

def accuracy(w, X, y, X_test, y_test):
    X_major = X[np.where(y == 1)[0]]
    y_major = y[np.where(y == 1)[0]]
    X_minor = X[np.where(y == -1)[0]]
    y_minor = y[np.where(y == -1)[0]]

    y_pred = np.rint(1 / (1 + np.exp(-X @ w)))
    y_pred = np.where(y_pred == 0, -1, y_pred)
    print('Train accuracy: ', accuracy_score(y, y_pred))

    y_pred = np.rint(1 / (1 + np.exp(-X_major @ w)))
    y_pred = np.where(y_pred == 0, -1, y_pred)
    print('Train accuracy (class 1): ', accuracy_score(y_major, y_pred))

    y_pred = np.rint(1 / (1 + np.exp(-X_minor @ w)))
    y_pred = np.where(y_pred == 0, -1, y_pred)
    print('Train accuracy (class 2): ', accuracy_score(y_minor, y_pred))

    y_pred = np.rint(1 / (1 + np.exp(-X_test @ w)))
    y_pred = np.where(y_pred == 0, -1, y_pred)
    print('Test accuracy: ', accuracy_score(y_test, y_pred))

    X_test_major = X_test[np.where(y_test == 1)[0]]
    y_test_major = y_test[np.where(y_test == 1)[0]]
    X_test_minor = X_test[np.where(y_test == -1)[0]]
    y_test_minor = y_test[np.where(y_test == -1)[0]]

    y_pred = np.rint(1 / (1 + np.exp(-X_test_major @ w)))
    y_pred = np.where(y_pred == 0, -1, y_pred)
    print('Test accuracy (class 1): ', accuracy_score(y_test_major, y_pred))

    y_pred = np.rint(1 / (1 + np.exp(-X_test_minor @ w)))
    y_pred = np.where(y_pred == 0, -1, y_pred)
    print('Test accuracy (class 2): ', accuracy_score(y_test_minor, y_pred),'\n')