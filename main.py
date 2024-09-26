import numpy as np
import pandas as pd 
import math
# 读取Excel文件  
mat1 = pd.read_excel('data1.xls', header=None).to_numpy()  
mat2 = pd.read_excel('data2.xls', header=None).to_numpy()  
  

def normalInformationEntropy(MAT):#计算x的信息熵
    X = np.zeros(MAT.shape[0])
    for k in range(MAT.shape[0]):
        X[k] = sum(MAT[k,:])
        X[k] = -X[k]*math.log2(X[k])
    return sum(X)
pass

def conditionalInformationEntropy(MAT):#计算条件熵
    #计算x信息熵
    X = np.zeros(MAT.shape[0])
    
    for k in range(MAT.shape[0]):
        X[k]= -sum((MAT[k,k:k+1])*math.log2(MAT[k,k:k+1]))
        X[k]= -(sum(MAT[k,:]))*math.log2(X[k])
        return sum(X)
pass

Hx_1=normalInformationEntropy(mat1)
Hy_1=normalInformationEntropy(mat1.T)
Hy_x1=conditionalInformationEntropy(mat1)
Hx_y1=conditionalInformationEntropy(mat1.T)
HXY_1= Hx_1 +Hy_x1

Hx_2=normalInformationEntropy(mat2)
Hy_2=normalInformationEntropy(mat2.T)
Hy_x2=conditionalInformationEntropy(mat2)
Hx_y2=conditionalInformationEntropy(mat2.T)
HXY_2= Hx_2 +Hy_x2

print(f"H(X1): {round(Hx_1,3)}")  
print(f"H(Y1): {round(Hy_1,3)}") 
print(f"H(Y|X1): {round((Hy_x1),3)}") 
print(f"H(X|Y1): {round((Hx_y1),3)}") 
print(f"HXY1:{round((HXY_1),3)}")

print(f"H(X2): {round(Hx_2,3)}")  
print(f"H(Y2): {round(Hy_2,3)}") 
print(f"H(Y|X2): {round((Hy_x2),3)}") 
print(f"H(X|Y2): {round((Hx_y2),3)}") 
print(f"HXY2:{round(HXY_2,3)}")

