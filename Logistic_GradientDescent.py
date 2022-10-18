import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def cost(X, theta, y):
    '''
    计算损失函数
    :param X:
    :param theta:
    :param y:
    :return:
    '''
    m, n = X.shape
    h = sigmoid((np.dot(X,theta)))
    cost = (-1.0/m) * np.sum(y.T * np.log(h)
                             + (1- y).T * np.log( 1 - h))
    return cost

def LR_gradient(X, y, alpha= 0.001, tol = 1e-6, max_iter = 100000):
    process = max_iter / 100 # 显示进度
    # matrix 更为方便
    X = np.mat(X)
    y = np.mat(y)
    m, n = X.shape
    theta = np.ones((n,1))
    iter = 0
    while ( iter < max_iter ):
        loss = cost(X, theta, y)
        # 梯度的下降法
        h = sigmoid( np.dot(X, theta)) # 预测值
        error = h - y # 误差项
        grad= (1.0/m) * np.dot(X.T, error)
        theta = theta - alpha * grad
        loss_new = cost(X, theta, y)
        if (iter % process == 0):
            # print(iter / max_iter)
            print("|",end="")
        if abs( loss - loss_new) < tol:
            break
        iter += 1
    print("\n迭代:",iter,"次","  最大迭代次数: ",max_iter, "损失: ",loss_new)
    return theta
def LR_gradient_norm(X, y, alpha= 0.001, Lambda = 0.5, tol = 1e-6, max_iter = 100000):
    process = max_iter / 100 # 显示进度
    # matrix 更为方便
    X = np.mat(X)
    y = np.mat(y)
    m, n = X.shape
    theta = np.ones((n,1))
    iter = 0
    while ( iter < max_iter ):
        loss = cost(X, theta, y)
        # 梯度的下降法
        h = sigmoid( np.dot(X, theta)) # 预测值
        error = h - y # 误差项
        grad= (1.0/m) * np.dot(X.T, error)
        theta = theta - alpha * grad - Lambda * (1.0/m) * theta
        loss_new = cost(X, theta, y)
        if (iter % process == 0):
            # print(iter / max_iter)
            print("|",end="")
        if abs( loss - loss_new) < tol:
            break
        iter += 1
    print("\n迭代:",iter,"次","  最大迭代次数: ",max_iter, "损失: ",loss_new)
    return theta