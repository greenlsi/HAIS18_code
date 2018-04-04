import numpy as np
import utils.store_dataset as sd

def add_zeros_col(train, test, total_cols):
    #Adding cols of zeros
    test_rows = test.shape[0]
    train_rows = train.shape[0]
    columns = total_cols
    z_train=np.zeros(train_rows)
    z_test=np.zeros(test_rows)
    x_train=np.zeros(shape=(train_rows, columns), dtype=np.float32)
    x_test=np.zeros(shape=(test_rows, columns), dtype=np.float32)
    for i in range(columns):
        if i<test.shape[1]:
            x_train[:,i]=train[:,i]
            x_test[:,i]=test[:,i]
        else:
            x_train[:, i] = z_train
            x_test[:, i] = z_test
    return x_train, x_test

def read_data(path, delim=","):
    data = sd.importCSVasPandas(path, delim)
    npdata=data.as_matrix()
    return npdata