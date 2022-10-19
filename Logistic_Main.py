import time

import numpy as np
import data_in
import Logistic_Newton as LRN
import Logistic_GradientDescent as LRG
import matplotlib.pyplot as plt
def normalization(data):
    row, columns = data.shape
    ones = np.ones((row, 1))
    x = data
    data = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    data = np.append(ones, data, axis=1)
    return data

def predict(X, theta, y):
    m, n = X.shape
    pre = np.zeros((m,1))

    for i in range(m):
        map = LRG.sigmoid(np.dot(X[i,:],theta))
        # print(map)
        if map > 0.5:
            pre[i] = 1

    count = 0
    for i in range(m):
        if (pre[i] == y[i]):
            count +=1
    return count / m

def paint_result(X, y, w):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(X.shape[0]):
        if y[i] == 1:
            ax.scatter(X[i][1], X[i][2],color="red")
        else :
            ax.scatter(X[i][1], X[i][2], color="green")
    x = np.arange(min(X[:,1]),max(X[:,1]),0.001)
    y = (-w[0] - w[1] * x.T)/w[2]
    ax.plot(x, y.getA1())
    plt.show()

def minitype_data(X_tr,y_tr,X_test,y_test,theta_1, theta_2):
    # 数据对比:
    print("     theta_1           theta_2")
    for i in range(columns):
        print(theta_1[i].getA1(), "---", theta_2[i].getA1())
    paint_result(X_tr, y_tr, theta_1)
    paint_result(X_tr, y_tr, theta_2)
    paint_result(X_test, y_test, theta_1)
    paint_result(X_test, y_test, theta_2)

if __name__ == '__main__':
    # 选择数据集
    #telco = data_in.data_np_telco()
    telco = data_in.data_np_lr() # 小规模训练集

    rows, columns = telco.shape
    beta = 0.8 # 训练集占比
    # 训练集
    X_tr = normalization(telco[:int(rows*beta), :columns - 1])
    y_tr = telco[:int(rows*beta), -1:]

    # 测试集
    X_test = normalization(telco[int(rows*beta):, :columns - 1])
    y_test = telco[int(rows*beta):,-1:]

    print("Logistic Regression 梯度下降算法")
    time_start = time.time()
    theta_1 = LRG.LR_gradient(X_tr, y_tr,alpha=0.1, tol=1e-4,max_iter=5e5) # 梯度下降解法 零数据直接跑的准确率为0.715
    time_end = time.time()
    print("测试集预测准确率: ",predict(X_test,theta_1,y_test),end="//")
    #np.savetxt("./theta_1.txt", theta_1, fmt='%f')
    print('time cost', time_end - time_start, 's')

    print("Logistic Regression 梯度下降算法-正则化项")
    time_start = time.time()
    theta_2 = LRG.LR_gradient_norm(X_tr, y_tr, alpha=0.1,Lambda=0,tol=1e-4, max_iter=5e5)  # 梯度下降解法 零数据直接跑的准确率为0.715
    time_end = time.time()
    #np.savetxt("./theta_2.txt", theta_2, fmt='%f')
    print("测试集预测准确率: ", predict(X_test, theta_2, y_test),end="//")
    print('time cost', time_end - time_start, 's')

    print("Logistic Regression 牛顿迭代法")
    time_start = time.time()
    theta_3 = LRN.LR_newton(X_tr, y_tr, tol=1e-6, max_iter=1e4)
    time_end = time.time()
    #np.savetxt("./theta_3.txt", theta_3, fmt='%f')
    print("测试集预测准确率: ",predict(X_test,theta_3,y_test),end="//")
    print('time cost', time_end - time_start, 's')

    minitype_data(X_tr,y_tr,X_test,y_test,theta_1, theta_2)
    '''
    print("问题二#################")
    mul = np.ones(X_tr.shape)
    mul[:,2] *= 1000
    X_tr *= mul
    mul = np.ones(X_test.shape)
    mul[:, 2] *= 1000
    X_test *= mul
    print("Logistic Regression 梯度下降算法")
    time_start = time.time()
    theta_4 = LRG.LR_gradient(X_tr, y_tr,alpha=0.01, tol=1e-5,max_iter=1e4) # 梯度下降解法 零数据直接跑的准确率为0.715
    time_end = time.time()
    #np.savetxt("./theta_4.txt", theta_4, fmt='%f')
    print("测试集预测准确率: ",predict(X_test,theta_1,y_test),end="//")
    print('time cost', time_end - time_start, 's')

    print("Logistic Regression 牛顿迭代法")
    time_start = time.time()
    theta_5 = LRG.LR_gradient(X_tr, y_tr, alpha=0.01, tol=1e-5, max_iter=1e4)  # 梯度下降解法 零数据直接跑的准确率为0.715
    time_end = time.time()
    #np.savetxt("./theta_5.txt", theta_5, fmt='%f')
    print("测试集预测准确率: ", predict(X_test, theta_5, y_test), end="//")
    print('time cost', time_end - time_start, 's')
    '''