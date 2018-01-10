import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self.left_pressure = left_foot["pressure "]
        self.right_pressure = right_foot["pressure "]


    def visualizationData(self,):
        time_len = np.arange(0,31659)
        fig1 = plt.figure('fig1')
        plt.title("0309-test3-ahrs data")
        plt.plot(time_len, self.roll_degree, 'g', label="roll_degree")
        plt.plot(time_len, self.pitch_degree, 'b', label="pitch_degree")
        plt.plot(time_len, self.yaw_degree, 'r', label="yaw_degree")
        plt.xlabel("time series")
        plt.ylabel("value")
        plt.legend(loc="upper right")


        fig2 = plt.figure('fig2')
        plt.title("0309-test3-pressure data")
        plt.plot(time_len, self.left_pressure, 'g', label="left_pressure")
        plt.plot(time_len, self.right_pressure, 'b', label="right_pressure")
        plt.xlabel("time series")
        plt.ylabel("value")
        plt.legend(loc="upper right")

        plt.show(fig1)
        plt.show(fig2)

    def exoState(self,):
        state = pd.concat([self.roll_degree, self.pitch_degree, self.yaw_degree, self.left_pressure, self.right_pressure], axis=1)
        return state


# load raw sensor data
def loadData():
    exo = ExoData("../../data/0309/test3/ahrs", "../../data/0309/test3/left_foot","../../data/0309/test3/right_foot")
    exo.generatorData()
    exo.visualizationData()
    # exo_raw_state = exo.exoState()
    # exo_raw_state.to_csv("../../data/0309/test1/exo_raw_data.csv")

loadData()