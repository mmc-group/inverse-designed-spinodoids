# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from parameters import *
from normalization import Normalization
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader

def getDataset():
    
    ######################################################
    dir = './'
    ######################################################
    
    data = pd.read_csv(dir+'data.csv')
    
    ######################################################
    data.volFrac = data.volFrac.astype(float)
    data.thetaX = data.thetaX.astype(float)
    data.thetaY = data.thetaY.astype(float)
    data.thetaZ = data.thetaZ.astype(float)
    ######################################################
    
    print('Data: ',data.shape)          

    ##############---INIT TENSORS---##############
    featureTensor = torch.tensor(data[featureNames].values)
    labelTensor = torch.tensor(data[labelNames].values)

    ##############---INIT NORMALIZATION---##############
    featureNormalization = Normalization(featureTensor)
    featureTensor = featureNormalization.normalize(featureTensor)
    
    ##############---INIT Dataset and loader---##############
    dataset =  TensorDataset(featureTensor.float(), labelTensor.float())
    l1 = round(len(dataset)*trainSplit)
    l2 = len(dataset) - l1
    print('train/test: ',[l1,l2])
    train_set, test_set = torch.utils.data.random_split(dataset, [l1,l2])
    return train_set, test_set, featureNormalization


#################################################     
def exportTensor(name,data,cols, header=True):
    df=pd.DataFrame.from_records(data.detach().numpy())
    if(header):
        df.columns = cols
    print(name)
    df.to_csv(name+".csv",header=header)

def exportList(name,data):
    arr=np.array(data)
    np.savetxt(name+".csv", [arr], delimiter=',')
    
    