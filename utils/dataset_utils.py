import logging

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dataset_logger")


def load_cifar10(class1, class2):
    if not (isinstance(class1, int) and isinstance(class2, int)) or class1 < 0 or class1 > 9 or class2 < 0 or class2 > 9:
        raise Exception("Please provide valid class label arguments")

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    train_mask = (y_train == class1) | (y_train == class2)
    test_mask = (y_test == class1) | (y_test == class2)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

    y_train = np.where(y_train == class2, -1, y_train)
    y_train = np.where(y_train == class1, 1, y_train)
    y_test = np.where(y_test == class2, -1, y_test)
    y_test = np.where(y_test == class1, 1, y_test)

    labels = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    logger.info("Successfully loaded CIFAR-10!")
    logger.info(f"Class 1: {labels[class1]}")
    logger.info(f"Class 2: {labels[class2]}")
    logger.info("-------------------------------------------------")
    return (X_train, y_train), (X_test, y_test)


def preprocess(X, y, size=None, isTrain=False):
    if not size:
        size = len(y)
    else:
        size = min(len(y), size)

    X_new = X[:size] / 255.0
    y_new = y[:size]

    if isTrain:
        shuffle(X, y)
        logger.info(f"Preprocessed {size} training samples")
        logger.info(f"X_train shape: {X_new.shape}")
        logger.info(f"Class 1 count: {np.count_nonzero(y_new == 1)}")
        logger.info(f"Class 2 count: {np.count_nonzero(y_new == -1)}")
    else:
        logger.info(f"Preprocessed {size} test samples")
        logger.info(f"X_test shape: {X_new.shape}")
        logger.info(f"Class 1 count: {np.count_nonzero(y_new == 1)}")
        logger.info(f"Class 2 count: {np.count_nonzero(y_new == -1)}")

    logger.info("-------------------------------------------------")
    return X_new, y_new


def create_imbalanced_train(X_train, y_train, percent):
    if percent < 50 or percent > 100 or not isinstance(percent, int):
        raise Exception("Please provide a valid percentage (integer from 50 to 100)")
    if len(y_train) < 100:
        raise Exception("Dataset too small")

    total = np.floor(len(y_train) / 100) * 100
    count = total * percent / 100
    while count > np.count_nonzero(y_train == 1) or (total - count) > np.count_nonzero(y_train == -1):
        total -= 100
        if total < 100:
            raise Exception("This split is not possible. Please provide a different percentage")
        count = total * percent / 100

    del_label1 = np.count_nonzero(y_train == 1) - count
    del_label2 = np.count_nonzero(y_train == -1) - (total - count)

    label1 = np.where(y_train == 1)[0]
    X_train = np.delete(X_train, label1[0 : int(del_label1)], axis=0)
    y_train = np.delete(y_train, label1[0 : int(del_label1)])
    label2 = np.where(y_train == -1)[0]
    X_train = np.delete(X_train, label2[0 : int(del_label2)], axis=0)
    y_train = np.delete(y_train, label2[0 : int(del_label2)])

    logger.info(f"Train set adjusted to {percent}% class imbalance")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"Class 1 count: {np.count_nonzero(y_train == 1)}")
    logger.info(f"Class 2 count: {np.count_nonzero(y_train == -1)}")
    logger.info("-------------------------------------------------")

    return X_train, y_train
