import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def newtons_method_gradient(X, theta ,y):
    m, n= X.shape
    return (1.0/m) * np.dot(X.T, sigmoid( np.dot(X, theta)) - y)

def newtons_method_hessian(X, theta ,y):
    # 负最大对数似然函数的二阶Hessian矩阵
    p = sigmoid( np.dot(X, theta))
    q = p.getA1()
    return X.T.dot(np.diag(q * (1-q)).dot(X))
    #return X.T.dot(np.diag(p * (1-p)).dot(X))

def cost(X, theta, y):
    '''
    计算损失函数
    '''
    m, n = X.shape
    h = sigmoid((np.dot(X,theta)))
    cost = (-1.0/m) * np.sum(y.T * np.log(h)
                             + (1- y).T * np.log( 1 - h))
    return cost

def LR_newton(X, y,tol = 1e-6, max_iter = 10000):
    process= max_iter / 100
    X = np.mat(X)
    y = np.mat(y)
    m, n = X.shape
    theta = np.zeros((n,1))

    iter = 0
    while( iter < max_iter):
        loss = cost(X,theta, y)

        # 牛顿迭代法
        grad = newtons_method_gradient(X, theta, y)
        hessian = newtons_method_hessian(X, theta, y)

        #theta = theta  - alpha * (1.0/m) * np.linalg.inv(hessian) * grad
        theta = theta - np.linalg.inv(hessian) * grad

        loss_new = cost(X, theta , y)

        if iter % process == 0:
            print("|",end="")
        if abs(loss - loss_new) < tol:
            break
        iter += 1

    print("\n迭代:",iter,"次",",最大迭代次数: ", max_iter, "损失: ",loss_new)
    return theta


