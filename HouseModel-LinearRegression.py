import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
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
ec=0.57721566490153286060651209008240243
def plotGraph(values,plotLine=False):
    fig, ax = plt.subplots(1, 1)
    plt.title('Predicted Value for Random Forest')
    mean=np.mean( train['SalePrice'])
    var=np.var(train['SalePrice'])
    beta=np.sqrt((6*var)/np.pi)
    mu=mean-beta*ec
    n_gum, binsGum, patchesGum=ax.hist(values, bins=25,normed=True, histtype='stepfilled', facecolor='red', alpha=0.4)
    if plotLine:
        gumLine = gumbel_r.pdf(binsGum, mu, beta)
        plt.plot(binsGum, gumLine, 'g--', label=" " + str(mu) + " variance " + str(beta))
train= pd.read_csv("D:\\college\\Sem 1\\Data Mining\\Data Mining Project-1 ( House sales)\\train.csv")
test= pd.read_csv("D:\\college\\Sem 1\\Data Mining\\Data Mining Project-1 ( House sales)\\test.csv")
dataTrainFrame,labSalePrice=dataFilter(train,'SalePrice',['SalePrice','Id'])
dataTestFrame,labDummyPrice=dataFilter(test,'Id')
fullData = pd.concat([dataTrainFrame,dataTestFrame],axis=0) #Combined both Train and Test Data set
datafullData=pd.DataFrame(fullData)
datafullData.to_csv('D:\\college\\Sem 1\\Data Mining\\Data Mining Project-1 ( House sales)\\fullData.csv')

#train['Type']='Train'
cleanedData=predictMissingValues(fullData)
df1=cleanedData[['SaleType','YrSold','GarageYrBlt','MasVnrType','Foundation','OverallQual','YearRemodAdd',
                       'GarageQual','Neighborhood','KitchenQual','ExterQual','Fireplaces','FullBath','TotalBsmtSF'
                        ,'BsmtCond','HeatingQC','1stFlrSF','GarageCars','GarageFinish','SaleCondition','OverallCond','LotConfig','BsmtQual'
                 ,'LotFrontage','LotArea','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath',
                 'BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                 'KitchenAbvGr','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF',
                 'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MasVnrArea'
]]
hotencoded1=pd.get_dummies(cleanedData,drop_first=True)

hotEncoded=pd.get_dummies(df1,drop_first=True)
trainData=hotEncoded[:len(train)]
testData=hotEncoded[len(train):]


pearson = hotencoded1.corr(method='pearson')
corr_with_target = pearson.ix[-1][:-1]
print(corr_with_target.sort(ascending=False))
corr_with_target_dict = corr_with_target.to_dict()

# List the attributes sorted from the most predictive by their correlation with Sale Price
print("FEATURE \tCORRELATION")
for attr in sorted(corr_with_target_dict.items(), key = lambda x: -abs(x[1])):
    print(attr)
att=pd.DataFrame(corr_with_target[abs(corr_with_target).argsort()[::1]])
att.to_csv('D:\\college\\Sem 1\\Data Mining\\Data Mining Project-1 ( House sales)\\att.csv')


# #feature selection
# lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(trainHotEncoded, labSalePrice)
# model1 = SelectFromModel(lsvc, prefit=True)
# X_new = model1.transform(trainHotEncoded)
# print(X_new)
# print(np.asarray(trainHotEncoded.columns[X_new.support_]).flatten())
# model1.fit(X_new,labSalePrice)
# predictedValue1=model1.predict(testHotEncoding)
# print("linear Svc Prediction",predictedValue1)



model =  RandomForestRegressor(n_estimators = 100 , oob_score = True, random_state = 42)
model.fit(trainData,labSalePrice)

# coef = pd.Series(model.feature_importances_, index = trainHotEncoded.columns).sort_values(ascending=False)
# plt.figure(figsize=(10, 5))
# coef.head(25).plot(kind='bar')
# plt.title('Feature Significance')
# plt.tight_layout()

#accuracy
scores=cross_val_score(model,trainData,labSalePrice,cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #train set accuracy


predictedValue=model.predict(testData)
plotGraph(predictedValue,False)
plt.show()
print("Random Forest Prediction",predictedValue)
dataPredictedvalue=pd.DataFrame(predictedValue)
dataPredictedvalue.to_csv('D:\\college\\Sem 1\\Data Mining\\Data Mining Project-1 ( House sales)\\RandomForestPredictValues.csv')
print("File written to drive successfully")