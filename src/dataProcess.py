import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
class ExoData(object):
    def __init__(self, ahrs_file, leftfoot_file, rightfoot_file):
        self.ahrs_file = ahrs_file
        self.leftfoot_file = leftfoot_file
        self.rightfoot_file = rightfoot_file

    def generatorData(self,):
        ahrs = pd.read_table(self.ahrs_file)
        self.roll_degree = ahrs["roll_degree "]
        self.pitch_degree = ahrs["pitch_degree "]
        self.yaw_degree = ahrs["yaw_degree "]
        left_foot = pd.read_table(self.leftfoot_file)
        right_foot = pd.read_table(self.rightfoot_file)
        self.left_pressure = left_foot["joint_radian "]
        self.right_pressure = right_foot["joint_radian "]


    def visualizationData(self,):
        time_len = np.arange(0,54539,)
        # fig1 = plt.figure('fig1')
        # plt.title("0309-test4-ahrs data")
        # plt.plot(time_len, self.roll_degree, 'g', label="roll_degree")
        # plt.plot(time_len, self.pitch_degree, 'b', label="pitch_degree")
        # plt.plot(time_len, self.yaw_degree, 'r', label="yaw_degree")
        # plt.xlabel("time series")
        # plt.ylabel("value")
        # plt.legend(loc="upper right")


        fig2 = plt.figure('fig2')
        plt.title("0307-test1-knee data")
        plt.plot(time_len, self.left_pressure, 'g', label="left_knee")
        plt.plot(time_len, self.right_pressure, 'b', label="right_knee")
        plt.xlabel("time series")
        plt.ylabel("value")
        plt.legend(loc="upper right")

        # plt.show(fig1)
        plt.show(fig2)

    def exoState(self,):
        state = pd.concat([self.roll_degree, self.pitch_degree, self.yaw_degree, self.left_pressure, self.right_pressure], axis=1)
        return state


# load raw sensor data
def loadData():
    exo = ExoData("../../data/0307/test1/ahrs", "../../data/0307/test1/left_knee","../../data/0307/test1/right_knee")
    exo.generatorData()
    exo.visualizationData()
    exo_raw_state = exo.exoState()
    exo_raw_state.to_csv("../../data/0307/test1/exo_raw_data3.csv")

loadData()
'''
'''
## plot all data
df = pd.read_csv("../../data/0307/test1/exo_sample_data_with_targets.csv")
left_hip_joint = df["left_hip_joint"]
right_hip_joint = df["left_hip_joint"]
# left_knee_joint = df["left_knee_joint"]
# right_knee_joint = df["right_knee_joint"]
time_len = np.arange(0,5453)

fig1 = plt.figure('fig1')
plt.title("0307-test1-all data")
plt.plot(time_len, left_hip_joint, 'b', label="left_hip_joint")
plt.plot(time_len, right_hip_joint, 'r', label="right_hip_joint")
# plt.plot(time_len, left_knee_joint, 'y', label="left_knee_joint")
# plt.plot(time_len, right_knee_joint, 'g', label="right_knee_joint")

plt.xlabel("time series")
plt.ylabel("value")
plt.legend(loc="upper right")
plt.show(fig1)
'''

def countData():
    raw_data = pd.read_csv("../../data/0307/test1/exo_sample_data_with_targets.csv")
    print(raw_data.describe())
    from collections import Counter
    labels = []
    for featVec in raw_data.values:
        labels.append(featVec[-1])
    labelCnt = Counter(labels)
    return labelCnt

print(countData())


