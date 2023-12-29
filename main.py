import time

from sklearn.linear_model import LogisticRegression

from solvers.admm import ADMM
from solvers.cvx import CVX
from utils import dataset_utils, eval_utils, log_utils

if __name__ == "__main__":
    timestamp = time.asctime()
    logger = log_utils.get_logger(timestamp)

    # load dataset
    (X_train, y_train), (X_test, y_test) = dataset_utils.load_cifar10(0, 2)

    # preprocess and create imbalanced train set
    X_train, y_train = dataset_utils.preprocess(X_train, y_train, size=1000, isTrain=True)
    X_train, y_train = dataset_utils.create_imbalanced_train(X_train, y_train, 90)

    # preprocess test set
    X_test, y_test = dataset_utils.preprocess(X_test, y_test)

    # ADMM
    logger.info("Running ADMM...")
    admm = ADMM(X_train, y_train, timestamp)
    start = time.time()
    admm.run(diff=1e-2)
    end = time.time()
    eval_utils.accuracy(admm.w, X_train, y_train, X_test, y_test)
    logger.info(f"ADMM completed in {round(end - start, 2)} seconds")
    logger.info("-------------------------------------------------")

    # CVX
    logger.info("Running CVX...")
    cvx = CVX(X_train, y_train)
    start = time.time()
    cvx.run()
    end = time.time()
    eval_utils.accuracy(cvx.w, X_train, y_train, X_test, y_test)
    logger.info(f"CVX completed in {round(end - start, 2)} seconds")
    logger.info("-------------------------------------------------")

    # unconstrained logistic regression
    logger.info("Running vanilla logistic regression...")
    clf = LogisticRegression(penalty=None).fit(X_train, y_train)
    eval_utils.accuracy(clf.coef_[0], X_train, y_train, X_test, y_test)
    logger.info("Vanilla logistic regression completed")
    logger.info("-------------------------------------------------")
