from utils import dataset_utils
from utils import eval_utils
from solvers.admm import ADMM
from solvers.cvx import CVX
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # load dataset
    (X_train,y_train), (X_test,y_test) = dataset_utils.load_cifar10(0,2)

    # preprocess and create imbalanced train set
    X_train, y_train = dataset_utils.preprocess(X_train, y_train, size=1000, isTrain=True)
    X_train, y_train = dataset_utils.create_imbalanced_train(X_train, y_train, 90)

    # preprocess test set
    X_test, y_test = dataset_utils.preprocess(X_test, y_test)

    # ADMM
    admm = ADMM(X_train, y_train)
    admm.run(diff=1e-2)
    print('ADMM results:')
    eval_utils.accuracy(admm.w, X_train, y_train, X_test, y_test)

    # CVX
    cvx = CVX(X_train, y_train)
    cvx.run()
    print('CVX results:')
    eval_utils.accuracy(cvx.w, X_train, y_train, X_test, y_test)

    # unconstrained logistic regression
    print('Running vanilla logistic regression...')
    clf = LogisticRegression(penalty=None).fit(X_train, y_train)
    print('Done!\n')
    print('Vanilla logistic regression results:')
    eval_utils.accuracy(clf.coef_[0], X_train, y_train, X_test, y_test)