import numpy as np
import pandas as pd
from matplotlib import pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.linear_model import enet_path
import math


# load data
featureName =np.array(['roll_degree', 'pitch_degree', 'yaw_degree', 
                        'left_pressure','right_pressure', 
                        'left_hip_joint', 'right_hip_joint', 'left_knee_joint','right_knee_joint'])

target_name=np.array(['sitting', 'standing', 'walking', 'resting'])

xList = []
raw_data = open("../../data/0307/test1/exo_sample_data_with_targets.csv")
for line in raw_data:
    row = line.strip().split(",")
    xList.append(row)

xNum = []
labels = []
for row in xList[1:-1]:
    labels.append(row.pop())
    l = len(row)
    attrRow = [float(row[i]) for i in range(1,l)]
    xNum.append(attrRow)

# number of rows and columns in x matrix
nrow = len(xNum)
ncol = len(xNum[1])

yOneVAll = []
labelSet = set(labels)
labelList = list(labelSet)
nlabels = len(labelList)
for i in range(nrow):
    yRow = [0.0] * nlabels
    index = labelList.index(labels[i])
    yRow[index] = 1.0
    yOneVAll.append(yRow)

# calculate means and variances
xMeans = []
xSD = []
for i in range(ncol):
    col = [xNum[j][i] for j in range(nrow)]
    mean = sum(col) / nrow
    xMeans.append(mean)
    colDiff = [(xNum[j][i] - mean) for j in range(nrow)]
    sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrow)])
    stdDev = math.sqrt(sumSq / nrow)
    xSD.append(stdDev)

xNormalized = []
for i in range(nrow):
    rowNormalized = [(xNum[i][j]) - xMeans[j]/xSD[j] for j in range(ncol)]
    xNormalized.append(rowNormalized)

# normalize y's to center
yMeans = []
ySD = []
for i in range(nlabels):
    col = [yOneVAll[j][i] for j in range(nrow)]
    mean = sum(col) / nrow
    yMeans.append(mean)
    colDiff = [(yOneVAll[j][i] - mean) for j in range(nrow)]
    sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrow)])
    stdDev = math.sqrt(sumSq/nrow)
    ySD.append(stdDev)

yNormalized = []
for i in range(nrow):
    rowNormalized = [(yOneVAll[i][j] - yMeans[j])/ySD[j] for j in range(nlabels)]
    yNormalized.append(rowNormalized)

# number of cross-validation folds
nxval = 10
nAlphas=500
misClass = [0.0] * nAlphas
for ixval in range(nxval):
    # Defne test and training index sets
    idxTest = [a for a in range(nrow) if a%nxval == ixval%nxval]
    idxTrain = [a for a in range(nrow) if a%nxval != ixval%nxval]
    # Defne test and training attribute and label sets
    xTrain = np.array([xNormalized[r] for r in idxTrain])
    xTest = np.array([xNormalized[r] for r in idxTest])
    yTrain = [yNormalized[r] for r in idxTrain]
    yTest = [yNormalized[r] for r in idxTest]
    labelsTest = [labels[r] for r in idxTest]
    # build model for each column in yTrain
    models = []
    lenTrain = len(yTrain)
    lenTest = nrow - lenTrain
    for iModel in range(nlabels):
        yTemp = np.array([yTrain[j][iModel] for j in range(lenTrain)])
        models.append(enet_path(xTrain, yTemp,l1_ratio=1,
            ft_intercept=False, eps=0.1e-3, n_alphas=nAlphas,
            return_models=False))
    for iStep in range(1,nAlphas):
        # Assemble the predictions for all the models, fnd largest
        # prediction and calc error
        allPredictions = []
        for iModel in range(nlabels):
            _, coefs, _ = models[iModel]
            predTemp = list(np.dot(xTest, coefs[:,iStep]))
            # un-normalize the prediction for comparison
            predUnNorm = [(predTemp[j]*ySD[iModel] + yMeans[iModel]) for j in range(len(predTemp))]
            allPredictions.append(predUnNorm)
        predictions = []
        for i in range(lenTest):
            listOfPredictions = [allPredictions[j][i] for j in range(nlabels) ]
            idxMax = listOfPredictions.index(max(listOfPredictions))
            if labelList[idxMax] != labelsTest[i]:
                misClass[iStep] += 1.0

misClassPlot = [misClass[i]/nrow for i in range(1, nAlphas)]
print("Misclassifcation Error Rate: ",min(misClassPlot))

# plot figure
plot.plot(misClassPlot)
plot.xlabel("Penalty Parameter Steps")
plot.ylabel(("Misclassifcation Error Rate"))
plot.show()
