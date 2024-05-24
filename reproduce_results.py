import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

def get_np_data(data, time_steps):
    np_data = []
    hp = data.iloc[:,:4]
    for i in range(len(hp)):
        kk = []
        for j in range(time_steps):
            dd = []
            dd += list(hp.iloc[i])
            dd.append(data.iloc[i]["train_loss_epoch"+str(j+1)])
            dd.append(data.iloc[i]["eval_loss_epoch"+str(j+1)])
            dd.append(data.iloc[i]["eval_acc_epoch"+str(j+1)])
            kk.append(dd)
        np_data.append(kk)
    return np.array(np_data)

def get_np_y_data(data):
    return np.array(list(data["eval_acc_epoch150"])).reshape(-1,1)

if __name__ == '__main__':
    e = int(input("Enter the E value (Can be 5, 10, 20, 30 or 60): "))

    # LOAD TEST DATA
    test_data = pd.read_csv('./data/test.csv')
    test_np_data_x = get_np_data(test_data, e)

    save_path = "./reproduced_results/e"+str(e)+"/"

    model = None
    if e == 5:
        model = load_model('./ckpt/e5/model_epoch_50.h5')
    elif e == 10:
        model = load_model('./ckpt/e10/model_epoch_35.h5')
    elif e == 20:
        model = load_model('./ckpt/e20/model_epoch_35.h5')
    elif e == 30:
        model = load_model('./ckpt/e30/model_epoch_35.h5')
    elif e == 60:
        model = load_model('./ckpt/e60/model_epoch_50.h5')
    else:
        print("Invalid E value")
        exit()

    result = model.predict(test_np_data_x)
    print(result)
    # write the result array to a file
    np.savetxt(save_path+"result.csv", result, delimiter=",")
