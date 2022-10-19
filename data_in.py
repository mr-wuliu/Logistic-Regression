import pandas as pd
import numpy as np

def data_np_X():
    pd_reader = pd.read_csv("./data/randn_data_regression_X.csv")
    result = np.array(pd_reader)
    return result

def data_np_Y():
    pd_reader = pd.read_csv("./data/randn_data_regression_y.csv")
    result = np.array(pd_reader)
    return result

def data_np_telco():
    pd_reader =pd.read_csv("./data/telco.csv")
    # 用平均值对空值进行填充
    pd_reader = pd_reader.fillna(pd_reader.mean())
    result = np.array(pd_reader, dtype=np.float64)
    return result
def data_np_lr():
    return np.array(pd.read_table('./data/LR_data.txt'))
if __name__ == '__main__':
    print(data_np_telco()[:,35])
