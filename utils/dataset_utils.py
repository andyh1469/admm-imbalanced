import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import numpy as np

def load_cifar10(class1, class2):
    if not (isinstance(class1,int) and isinstance(class2,int)) \
            or class1 < 0 or class1 > 9 or class2 < 0 or class2 > 9:
        raise Exception('Please provide valid class label arguments')

    (X_train,y_train), (X_test,y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.reshape((X_train.shape[0],-1))
    X_test = X_test.reshape((X_test.shape[0],-1))
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    train_mask = (y_train == class1) | (y_train == class2)
    test_mask = (y_test == class1) | (y_test == class2)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    y_train = np.where(y_train == class2, -1, y_train)
    y_train = np.where(y_train == class1, 1, y_train)
    y_test = np.where(y_test == class2, -1, y_test)
    y_test = np.where(y_test == class1, 1, y_test)

    labels = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    print('Successfully loaded CIFAR-10!')
    print('Class 1: {}'.format(labels[class1]))
    print('Class 2: {}\n'.format(labels[class2]))
    return (X_train,y_train), (X_test,y_test)

def preprocess(X, y, size=None, isTrain=False):
    if not size:
        size = len(y)
    else:
        size = min(len(y),size)

    X_new = normalize(X)[:size]
    y_new = y[:size]

    if isTrain:
        shuffle(X, y)
        print('Preprocessed {} training samples'.format(size))
        print('X_train shape: {}'.format(X_new.shape))
        print('Class 1 count: {}'.format(np.count_nonzero(y_new == 1)))
        print('Class 2 count: {}\n'.format(np.count_nonzero(y_new == -1)))
    else:
        print('Preprocessed {} test samples'.format(size))
        print('X_test shape: {}'.format(X_new.shape))
        print('Class 1 count: {}'.format(np.count_nonzero(y_new == 1)))
        print('Class 2 count: {}\n'.format(np.count_nonzero(y_new == -1)))
    return X_new, y_new


def create_imbalanced_train(X_train, y_train, percent):
    if percent < 50 or percent > 100 or not isinstance(percent,int):
        raise Exception('Please provide a valid percentage (integer from 50 to 100)')
    if len(y_train) < 100:
        raise Exception('Dataset too small')

    total = np.floor(len(y_train) / 100) * 100
    count = total * percent / 100
    while count > np.count_nonzero(y_train == 1) \
            or (total - count) > np.count_nonzero(y_train == -1):
        total -= 100
        if total < 100:
            raise Exception('This split is not possible. Please provide a different percentage')
        count = total * percent / 100

    del_label1 = np.count_nonzero(y_train == 1) - count
    del_label2 = np.count_nonzero(y_train == -1) - (total - count)

    label1 = np.where(y_train == 1)[0]
    X_train = np.delete(X_train, label1[0:int(del_label1)], axis=0)
    y_train = np.delete(y_train, label1[0:int(del_label1)])
    label2 = np.where(y_train == -1)[0]
    X_train = np.delete(X_train, label2[0:int(del_label2)], axis=0)
    y_train = np.delete(y_train, label2[0:int(del_label2)])

    print('Train set adjusted to {}% class imbalance'.format(percent))
    print('X_train shape: {}'.format(X_train.shape))
    print('Class 1 count: {}'.format(np.count_nonzero(y_train == 1)))
    print('Class 2 count: {}\n'.format(np.count_nonzero(y_train == -1)))

    return X_train, y_train