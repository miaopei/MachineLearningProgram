import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model


def plot_line(x, y, theta=None, regressor=None):
    # Plot outputs
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))

    if regressor:
        Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = pred_val(theta, np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())


    plt.show()



# prediction values
def pred_val(theta, X, hard=True):
    # normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_val_func(theta, X)
    pred_value = np.where(pred_prob > 0.5, 1, 0)
    if hard:
        return pred_value
    else:
        return pred_prob



def logistic_grad_func(theta, x, y):
    # compute gradient
    grad = np.dot((logistic_val_func(theta, x) - y).T, np.c_[np.ones(x.shape[0]), x])
    grad = grad / x.shape[0]

    return grad


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def logistic_val_func(theta, x):
    # forwarding
    return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], theta.T))


def logistic_cost_func(theta, x, y):
    # compute cost (loss)
    y_hat = logistic_val_func(theta, x)
    cost = np.sum(y * np.log(y_hat)) + np.sum((1 - y) * np.log(1-y_hat))
    cost *= 1.0 / x.shape[0]
    return -cost


def logistic_grad_desc(theta, X_train, Y_train, lr=0.03, max_iter=500, converge_change=.00001):

    cost_iter = []
    cost = logistic_cost_func(theta, X_train, Y_train)
    cost_iter.append([0, cost])
    cost_change = 1
    i = 1
    while cost_change > converge_change and i < max_iter:
        pre_cost = cost
        # compute gradient
        grad = logistic_grad_func(theta, X_train, Y_train)
        theta -= lr * grad
        cost = logistic_cost_func(theta, X_train, Y_train)
        cost_iter.append([i, cost])
        cost_change = abs(cost - pre_cost)
        i += 1

    return theta, cost_iter


def logistic_regression():
    # load dataset
    dataset = datasets.load_iris()
    # Select only 2 dims
    X = dataset.data[0:100, 0:2]
    Y = dataset.target[:100, None]

    # split dataset into training and testing
    idx_trn = range(30)
    idx_trn.extend(range(50, 80))
    idx_tst = range(30,50)
    idx_tst.extend(range(80, 100))

    X_train = X[idx_trn]
    X_test = X[idx_tst]

    Y_train = Y[idx_trn]
    Y_test = Y[idx_tst]


    # Linear regression
    theta = np.random.rand(1, X_train.shape[1]+1)
    fitted_theta, cost_iter = logistic_grad_desc(theta, X_train, Y_train, lr=0.05, max_iter=50000)

    print('Coefficients: {}'.format(fitted_theta[0, :-1]))
    print('Intercept: {}'.format(fitted_theta[0, -2]))
    print('Accuracy: {}'.format(np.sum((pred_val(fitted_theta, X_test) == Y_test)) * 1.0 / X_test.shape[0]))

    # Plot Training
    # plot_line(X_train, Y_train, fitted_theta)
    # Plot Testing
    plot_line(X_test, Y_test, theta=fitted_theta)


def sklearn_logistic_regression():
    # load dataset
    dataset = datasets.load_iris()
    # Select only 2 dims
    X = dataset.data[0:100, 0:2]
    Y = dataset.target[:100, None]

    # split dataset into training and testing
    idx_trn = range(30)
    idx_trn.extend(range(50, 80))
    idx_tst = range(30,50)
    idx_tst.extend(range(80, 100))

    X_train = X[idx_trn]
    X_test = X[idx_tst]

    Y_train = Y[idx_trn]
    Y_test = Y[idx_tst]

    # Logistic regression
    regressor = linear_model.LogisticRegression()
    regressor.fit(X_train, np.ravel(Y_train))
    print('Coefficients: {}'.format(regressor.coef_))
    print('Intercept: {}'.format(regressor.intercept_))
    print('Accuracy:{}'.format(np.sum((regressor.predict(X_test) == np.ravel(Y_test)))*1.0/X_test.shape[0]))

    plot_line(X_test, Y_test, regressor=regressor)


def main():
    print('Class 1: Linear Regression Example')
    logistic_regression()

    print ('')

    print('sklearn Linear Regression Example')
    sklearn_logistic_regression()


if __name__ == "__main__":
    main()