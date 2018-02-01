import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.externals import joblib

sumv=0
ratios = []
rng = np.random.RandomState(42)

# Generate train data
# df1 = pd.read_csv("../../data/0309/test1/exo_raw_data.csv").fillna(0)
# df_train1 = df1.iloc[29000:30000,:].copy()
# df2 = pd.read_csv("../../data/0309/test2/exo_raw_data.csv").fillna(0)
# df_train2 =df_train1.append(df2.iloc[33000:34000,:].copy())
# df3 = pd.read_csv("../../data/0309/test3/exo_raw_data.csv").fillna(0)
# df_train3 = df_train2.append(df3.iloc[29000:30000,:].copy())
# df4 = pd.read_csv("../../data/0309/test4/exo_raw_data.csv").fillna(0)
# df_train = df_train3.append(df4.iloc[31000:32000,:].copy())
# print("df_train shape",df_train.shape)

# left_foot_train = df_train["left_pressure "]
# right_foot_train = df_train["right_pressure "]
# X_train = np.array(pd.concat([left_foot_train,right_foot_train],axis=1))


df = pd.read_csv("../../data/0309/test1/exo_raw_data.csv").fillna(0)
df_train = df.iloc[11000:14000,:].copy()
left_foot_train = df_train["left_pressure "]
right_foot_train = df_train["right_pressure "]
X_train = np.array(pd.concat([left_foot_train,right_foot_train],axis=1))
# Generate test data
df = pd.read_csv("../../data/0309/test1/exo_raw_data.csv").fillna(0)
df_test = df.iloc[0:1000,:].copy()
left_foot_test = df_test["left_pressure "]
right_foot_test = df_test["right_pressure "]
X_test = np.array(pd.concat([left_foot_test,right_foot_test],axis=1))

# Generate some abnormal data
'''
df_train = df.iloc[200:500,:].copy()
left_foot_train = df_train["left_pressure "]
right_foot_train = df_train["right_pressure "]
X_outliers = np.array(pd.concat([left_foot_train,right_foot_train],axis=1))
'''
X_outliers_left = rng.uniform(low=0, high=60, size=(1000, 1))
X_outliers_right = rng.uniform(low=0, high=60, size=(1000, 1))
X_outliers = np.concatenate((X_outliers_left,X_outliers_right),axis=1)
# plot training data
# fig = plt.figure("fig")
# plt.title("sitting")
# a = plt.scatter(left_foot_train,right_foot_train,c='blue',alpha=0.2,marker='o',s=20)
# plt.xlabel("left_pressure")
# plt.ylabel("right_pressure")
# plt.legend([a],
#             ["pressure"],
#             loc="upper right")
# plt.show(fig)


# fit the model
# clf = IsolationForest(max_samples=100, random_state=rng)
# clf.fit(X_train)

# save model
# joblib.dump(clf, '../model/sitting.pkl')
# load model
for clf in [joblib.load('../model/standing.pkl'), joblib.load('../model/walking.pkl'),
            joblib.load('../model/resting.pkl'), joblib.load('../model/sitting.pkl')]:

# clf = joblib.load('../model/walking.pkl')

# predict
# y_pred_train = clf.predict(X_train)

# y_pred_test = clf.predict(X_test)

    y_pred_outliers = clf.predict(X_outliers)
    for value in y_pred_outliers:
        if value>0:
            sumv= sumv+1
    ratio = sumv/len(y_pred_outliers)
    ratios.append(ratio)
    sumv=0

print("normal ratio is:", ratios)
print("sum ratio is:", np.sum(ratios))
if np.sum(ratios)>0.5:
    print("is normal!")
else:
    print("is abnormal!")
# print("y_pred_outliers: ", y_pred_outliers)



# plot all data
a = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                 s=20, edgecolor='k')

plt.title("anomaly detection")
plt.xlabel("left_pressure")
plt.ylabel("right_pressure")
plt.legend([a, b, c],
           ["training data",
            "new regular data",
            "new abnormal data"],
            loc="upper right")
plt.show()

