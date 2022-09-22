import h5py
import numpy as np

SG_FILEPATH = "/home/manuel/REDS/src/data/SG24_dataset.h5"
DG_FILEPATH = "/home/manuel/REDS/src/data/DG10_dataset.h5"

def read_data_SG():
    sg_data = h5py.File(SG_FILEPATH,'r')
    print(sg_data.keys())
    predictors = sg_data['Predictors']
    target = sg_data['Target']
    user = sg_data['User']
    sg_data.close()
    return predictors, target, user


def read_data_DG():
    data = h5py.File(DG_FILEPATH,'r')
    print(data.keys())
    predictors = data['Predictors']
    target = data['Target']
    user = data['User']
    print(predictors)
    data.close()
    return predictors, target, user

read_data_SG()