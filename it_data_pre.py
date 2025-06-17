import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import xlrd
from PyEMD import EMD, Visualisation
from sklearn.decomposition import KernelPCA
import pylab as plt
os.chdir('D:/itransformer/test')
a=[21,38]

def read_data(path):
    data=pd.read_excel(path,header=None)
    data=data[100:]
    data=data.values
    x_data=data[:,:21]
    y_data=data[:,21:91]
    return x_data,y_data

    
def combin(*gk):
    return np.concatenate(*gk,axis=0,dtype='float32')

def sliding_window(x_data,sw_width=256,in_start=0):
    X=[]
    for _ in range(len(x_data)):
        in_end=in_start+sw_width
        if in_end<len(x_data):
            X.append(x_data[in_start:in_end])
        in_start+=16
    return np.array(X,dtype='float32')
def get_max_min(n):
    x=[]
    for i in range(0,n):
        x_data,_=read_data(f'case{i}.odb.xlsx')
        x.append(x_data)
    x=tuple(x)
    x_data=combin((x))
    x_data=x_data.reshape(-1,21)
    x_data=torch.tensor(x_data)
    max_value=torch.max(x_data,dim=0)
    min_value=torch.min(x_data,dim=0)
    return max_value,min_value

def max_minnormalize(dataset,max_value,min_value):
    dataset=dataset.reshape(-1,21)
    dataset=torch.tensor(dataset)

    dataset=(dataset-min_value.values)/(max_value.values-min_value.values)

    return dataset.reshape(-1,256,21)
print('here')
max_value,min_value=get_max_min(211)
print(max_value)

x=[]
y=[]
for i in range(0,211):
    x_data,y_data=read_data(f"case{i}.odb.xlsx")
    x_data=max_minnormalize(sliding_window(x_data,256,0),max_value,min_value)
    y_data=sliding_window(y_data,256,0)
    y_data=torch.tensor(np.average(y_data,axis=1))
    x.append(x_data)
    y.append(y_data)
    print(f'{i}x:',x_data.shape)
    print(f'{i}y:',y_data.shape)
    print(f'gk{i}')
x=tuple(x)
x_data=combin((x))
y=tuple(y)
y_data=combin((y))
y_data=torch.tensor(y_data,dtype=torch.float32)
#y_data=y_data.ge(0.05)
x=[]
y=[]
x_data=torch.tensor(x_data,dtype=torch.float32)
split=[51146,12787]
dataset=torch.utils.data.TensorDataset(*(x_data,y_data))
train_dataset,test_dataset=torch.utils.data.random_split(dataset,split)
torch.save(train_dataset[:][0],'train_features.pth')
torch.save(train_dataset[:][1],'train_labels.pth')
torch.save(test_dataset[:][0],'test_features.pth')
torch.save(test_dataset[:][1],'test_labels.pth')