import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time

def loadData():
    raw_data = pd.read_csv("../../data/0307/test1/exo_sample_data_with_targets.csv")
    print("all size is", np.array(raw_data).shape)
    dataset = raw_data.values
    X = dataset[:, 0:9].astype(float)
    Y = dataset[:, -1]
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y,test_size=0.2, random_state=226)
    return xTrain, xTest, yTrain, yTest

X_train, X_test, y_train, y_test = loadData()

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
svm_est = Pipeline([('scaler',StandardScaler()),('svc',OneVsRestClassifier(SVC()))])
# Cs = [0.001, 0.01, 0.1, 1, 10]
# gammas = [0.001, 0.01, 0.1, 1, 10]
Cs = [10]
gammas= [0.01]
param_grid = dict(svc__estimator__gamma=gammas, svc__estimator__C=Cs)


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=226)
rand_grid = GridSearchCV(svm_est, param_grid=param_grid, cv=cv, n_jobs=1)

# fitting process
start_time = time.time()
rand_grid.fit(X_train, y_train)  
end_time = time.time() - start_time
print("using time is: ", end_time)
print("best parameter: ",rand_grid.best_params_)

# predict process
prediction = rand_grid.predict(X_test)
correct = accuracy_score(y_test, prediction)
print("MissClassifcation Error:", 1-correct )


# generate confusion matrix
pList = prediction.tolist()
confusionMat = confusion_matrix(y_test, pList)
print('')
print("Confusion Matrix")
print(confusionMat)