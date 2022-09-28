import h5py
import numpy as np

SG_FILEPATH = "/home/manuel/REDS/data/SG24_dataset.h5"
DG_FILEPATH = "/home/manuel/REDS/data/DG10_dataset.h5"

def read_data_SG():
    data = h5py.File(SG_FILEPATH,'r')
    predictors = np.asarray(data['Predictors'])
    target = np.asarray(data['Target'])
    user = np.asarray(data['User'])
    return data, predictors, target, user


def read_data_DG():
    data = h5py.File(DG_FILEPATH,'r')
    predictors = np.asarray(data['Predictors'])
    target = np.asarray(data['Target'])
    user = np.asarray(data['User'])
    data.close()
    return data, predictors, target, user

read_data_SG()