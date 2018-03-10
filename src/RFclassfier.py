from __future__ import print_function
import urllib
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import time


featureName =np.array(['roll_degree', 'pitch_degree', 'yaw_degree', 
                        'left_pressure','right_pressure', 
                        'left_hip_joint', 'right_hip_joint', 'left_knee_joint','right_knee_joint'])

target_name=np.array(['sitting', 'standing', 'walking', 'resting'])

def loadData():
    raw_data = pd.read_csv("../../data/0307/test1/exo_sample_data_with_targets.csv")
    print(raw_data.describe())
    print("all size is", np.array(raw_data).shape)
    labels = raw_data.pop("targets")
    xList = raw_data
    ## unbalance data: Stratifid sampling by labels
    xTrain, xTest, yTrain, yTest = train_test_split(xList, labels,
        test_size=0.2, random_state=226)
    return xTrain, xTest, yTrain,yTest

def ClassModel(xTrain, xTest, yTrain,yTest):
    missClassError = []
    nTreeList = range(300, 600, 50)
    for iTrees in nTreeList:
        depth = None
        maxFeat = 4 # try tweaking
        RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees,
            max_depth=depth, max_features=maxFeat,
            oob_score=False, random_state=531)
        
        start_time = time.time()
        RFModel.fit(xTrain,yTrain)
        end_time = time.time()-start_time
        print("using time is: ", end_time)
        ## Accumulate 
        prediction = RFModel.predict(xTest)
        correct = accuracy_score(yTest, prediction)
        missClassError.append(1.0 - correct)
    print("MissClassifcation Error" )
    print(min(missClassError))

    ## generate confusion matrix
    pList = prediction.tolist()
    confusionMat = confusion_matrix(yTest, pList)
    print('')
    print("Confusion Matrix")
    print(confusionMat)
    
    ## plot number of trees and error 
    plt.plot(nTreeList, missClassError)
    plt.xlabel('Number of Trees in Ensemble')
    plt.ylabel('Missclassifcation Error Rate')
    plt.show()

    ## plot feature importance 
    featureImportance = RFModel.feature_importances_
    featureImportance = featureImportance / featureImportance.max()
    idxSorted = np.argsort(featureImportance)
    barPos = np.arange(idxSorted.shape[0]) + .5
    plt.barh(barPos, featureImportance[idxSorted], align='center')
    plt.yticks(barPos, featureName[idxSorted])
    plt.xlabel('Variable Importance')
    plt.show()

    ## save model and load model
    # joblib.dump(RFModel, '../model/rfclf.pkl')

xTrain, xTest, yTrain,yTest = loadData()

ClassModel(xTrain, xTest, yTrain,yTest)
'''
# # test the model 
finalClasss = joblib.load('../model/rfclf.pkl')
test = np.array([0.948203,-107.960335,20.957085,115,24,-1.238987,-1.221456,1.692592,1.58796]).reshape(1,-1)
finalPrediction = finalClasss.predict(test)
if finalPrediction[0]==0:
    print("walking")
elif finalPrediction[0]==1:
    print("sitting")
elif finalPrediction[0]==2:
    print("resting")
elif finalPrediction[0]==3:
    print("standing")
else:
    print("no data")
'''

