import scipy.io
import time
import pdb
import numpy as np
from SSDL_GU import *
from sklearn.decomposition import SparseCoder
from numpy.linalg import norm
import sys
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from mnist import MNIST
from PIL import Image
import math
import os
import cv2

def Eigenface_f(Train_SET,Eigen_NUM):
    NN,Train_NUM=Train_SET.shape
    if NN<=Train_NUM:
        Mean_Image=mean(Train_SET,axis=1)
        Train_SET=Train_SET-np.dot(Mean_Image,np.ones(1,Train_NUM))
        R=np.dot(Train_SET,Train_SET.T)/(Train_NUM-1)
        V,S=Find_K_Max_Eigen(R,Eigen_NUM)
        disc_value=S
        disc_set=V
    else:
        Mean_Image=mean(Train_SET,axis=1)
        Train_SET=Train_SET-np.dot(Mean_Image,np.ones(1,Train_NUM))
        R=np.dot(Train_SET.T,Train_SET)/(Train_NUM-1)
        V,S=Find_K_Max_Eigen(R,Eigen_NUM)
        disc_value=S
        disc_set=np.zeros((NN,Eigen_NUM))
        Train_SET=Train_SET/math.sqrt(Train_NUM-1)
        for k in range(Eigen_NUM)
            disc_set[:,k]=(1./math.sqrt(disc_value[k]))*Train_SET*V[:,k]

def Dict_Ini(data,nCol,wayInit):
    m=data.shape[0]
    if wayInit=="pca":
        (D,disc_value,Mean_Image)=Eigenface_f(data,nCol-1)
        Mean_Image=preprocessing.normalize(Mean_Image.T, norm='l2').T
        D=np.hstack((D,Mean_Image))
    elif wayInit=="random":
        phi=np.random.randn(m,nCol)
        D=preprocessing.normalize(phi.T, norm='l2').T
    else:
        print("wayInit_error")
        exit()

# data = scipy.io.loadmat('clothes5.mat') # 读取mat文件
data = scipy.io.loadmat('T4.mat') # 读取mat文件
# print(data.keys())  # 查看mat文件中的所有变量
train_data=data['train_data']
train_data_reg=preprocessing.normalize(train_data.T, norm='l2').T
aa=np.stack((train_data,train_data),axis=1)
train_Annotation=data['train_Annotation']
test_data=data['test_data']
test_data_reg=preprocessing.normalize(test_data.T, norm='l2').T
test_Annotation=data['test_Annotation']
testNum=test_data.shape[1]
labelNum=test_Annotation.shape[0]
featureDim=test_data.shape[0]
atom_n=30
atomNum=[atom_n,atom_n,atom_n,atom_n,atom_n]
D0=np.empty((labelNum,train_data.shape[0],atom_n))
D0_reg=np.empty((labelNum,train_data.shape[0],atom_n))
# Dic_reg_para=np.empty((labelNum,atom_n))
xmu=np.array([0.05])
RankingLoss=np.zeros((xmu.shape[0]))
Average_Precision=np.zeros((xmu.shape[0]))
Coverage=np.zeros((xmu.shape[0]))
OneError=np.zeros((xmu.shape[0]))
for m in range(xmu.shape[0]):
    for i in range(labelNum):
        cdat=train_data_reg[:,train_Annotation[i,:]==1]
        nRow,nCol=cdat.shape
        if atomNum[i]>min(featureDim,cdat.shape[1]):
            wayInit1="pca"
            wayInit2="random"
            atomNum1=min(featureDim,cdat.shape[1])
            atomNum2=atomNum[i]-atomNum1
            dict1=Dict_Ini(cdat,atomNum1,wayInit1)
            dict2=Dict_Ini(cdat,atomNum2,wayInit2)
            the_dict=np.hstack((dict1,dict2))
        else:
            wayInit="pca"
            the_dict=Dict_Ini(cdat,atomNum(i),wayInit)
        D0[i]=the_dict
        D0_reg[i]=preprocessing.normalize(the_dict.T, norm='l2').T
pdb.set_trace()