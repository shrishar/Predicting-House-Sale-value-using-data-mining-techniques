from sklearn.neural_network import MLPRegressor
import pandas as pd
import os
import numpy as np
pwd=os.getcwd()
from sklearn.preprocessing import StandardScaler

train= pd.read_csv("D:\\college\\Sem 1\\Data Mining\\Data Mining Project-1 ( House sales)\\train.csv")
test= pd.read_csv("D:\\college\\Sem 1\\Data Mining\\Data Mining Project-1 ( House sales)\\test.csv")



colNameWithNAList=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
colNameWithNumList=['LotFrontage','LotArea','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MasVnrArea']



def dataFilter(data, targetColumn, columnName=[]):
    if len(columnName) == 0:  # if column names are not given then create X with all the attributes
        columnName = [col for col in data.columns if col not in targetColumn]
    else:
        columnName = [col for col in data.columns if col not in columnName]

    dataFrame = data[columnName]  # predictor variable
    labelFrame = data[[targetColumn]]  # target variable
    return dataFrame, np.asarray(labelFrame).flatten()

def replaceNaNValue(x):
    if x=="None":
        return
    else:
        return x

def calMode(data):
    myDict={}
    for i in data:
        if i!="None":
            if myDict.__contains__(i):
                myDict[i]=myDict[i]+1
            else:
                myDict.update({i:1})
    max=0
    val=""
    for key in myDict:
        if myDict[key]>max:
            max=myDict[key]
            val=key

    return val

def calSSE(target,predicted):#calculate sum of squared error
    m=len(target)
    SSE=((np.asarray(target)-np.asarray(predicted))**2)/float(2*m)
    return sum(SSE)

def predictMissingValues(data):
    colNameWithCatValue = list(set(list(data.columns)) - set(colNameWithNAList) - set(colNameWithNumList))
    for i in colNameWithNumList:
        if (data[i].isnull().any().any()):
            mean = data[i].mean()
            data[i]=data[i].fillna(mean)

    for j in colNameWithCatValue:
        if (data[j].isnull().any().any()):
            mode = calMode(data[j])
            data[j] = data[j].fillna(mode)

    return data

def normalizeDatas(train,test,isZScoreNormalize=False):

  if isZScoreNormalize:
      for i in train.columns:
          mean = train[i].mean()
          std = train[i].std()
          if std!=0:
            train[i] = train[i].apply(lambda d: float(d - mean) / float(std))  # perform z-score normalization
      for i in train.columns:
          mean = test[i].mean()
          std = test[i].std()
          if std != 0:
            test[i] = test[i].apply(lambda d: float(d - mean) / float(std))  # perform z-score normalization
      return train,test
  else:
    scaler = StandardScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    return train,test

dataTrainFrame,labSalePrice=dataFilter(train,'SalePrice',['SalePrice','Id'])
dataTestFrame,labDummyPrice=dataFilter(test,'Id')
fullData = pd.concat([dataTrainFrame,dataTestFrame],axis=0) #Combined both Train and Test Data set
#train['Type']='Train'
cleanedData=predictMissingValues(fullData)
hotEncode=pd.get_dummies(cleanedData,drop_first=True)
trainHotEncoded=hotEncode[:len(train)]
testHotEncoding=hotEncode[len(train):]

trainHotEncoded,testHotEncoding=normalizeDatas(trainHotEncoded,testHotEncoding,False)
mlp = MLPRegressor(hidden_layer_sizes=(50,),activation='logistic', max_iter=500, alpha=1e-4,solver='sgd', verbose=True, tol=0.0001, random_state=1,learning_rate_init=.001)
#hidden_layer_sizes=(50,) : there is one hidden layer having 50 hidden nodes
#hidden_layer_sizes=(10,20,) : there will be two hidden layer. first hidden layer will have 10 nodes and second hidden layer will have 20 nodes
#activation='logistic' : we are using logistic activation function. Other option is tanh
#max_iter=500 : The forward propogation and error back propogation will take place in a loop at the max 500 times or until convergence has reached
#alpha=1e-4 : regularization  parameter i.e. a penalty of 0.0001 for making model complex
#algorithm='sgd' : which algorithm to use to find optimal weight. 'sgd' refers to stochastic gradient descent
#verbose=True : Whether to print progress messages to stdout
#tol=1e-4 : represents tolerance i.e. threshold to check for convergence
#learning_rate_init=.001 : The initial learning rate used. It controls the step-size in updating the weights. Only used when algorithm='sgd' or 'adam'.
#random_state=1 : State or seed for random number generator

mlp.fit(trainHotEncoded, labSalePrice)

print ("Total Accuracy",mlp.score(trainHotEncoded, labSalePrice))
predictedPrice=mlp.predict(testHotEncoding)
print(predictedPrice)
