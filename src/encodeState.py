import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import Dense, Input
from keras.models import Model, load_model 
import numpy as np
import pandas as pd
import csv
from keras import regularizers
import matplotlib.pyplot as plt

# loadData()
df = pd.read_csv("../../data/0307/test1/exo_raw_data.csv").fillna(0)
print(df.describe())
df_norm = (df - df.mean()) / df.std()
df_norm = df_norm - df_norm.min()
print(df_norm.describe())

df_norm = np.array(df_norm)
x_train = df_norm
time_series = np.arange(len(x_train))
# x_test = df_norm[44540:54540, :]
print(x_train.shape)
# print(x_test.shape)

def trainModel():
    # consturct autoencoder
    encoding_dim = 2
    input_data = Input(shape=(5,))
    # encoder layers
    encoded = Dense(5, activation='relu')(input_data)
    encoded = Dense(3, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(3, activation='relu')(encoder_output)
    decoded = Dense(5, activation='relu')(decoded)

    autoencoder = Model(inputs=input_data, outputs=decoded)
    # autoencoder.save("../model/autoencoderModel.h5")
    # construct the encoder model 
    encoder = Model(inputs=input_data, outputs=encoder_output)
    encoder.save("../model/encoderModel.h5")
    # compile and fit
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=1000, batch_size=64, shuffle=True)
    


def testModel():
    # load model and test
    # autoencoder = load_model("../model/autoencoderModel.h5")
    encoder = load_model("../model/encoderModel.h5")
    decoded_data = encoder.predict(x_train.reshape(len(x_train),5))

    with open("./test3.csv", "w",newline="") as f:
        writer = csv.writer(f)
        writer.writerows(decoded_data)
    
    # 1 dimentions
    # fig1 = plt.figure('fig1')
    # plt.title("encoded data")
    # plt.plot(time_series, decoded_data, 'b--', label="encoded data")
    # plt.xlabel("time series")
    # plt.ylabel("value")
    # plt.legend(loc="upper right")
    # plt.show()

    # 2 dimentions
    fig2 = plt.figure('fig2')
    plt.title("encoded data")
    T=decoded_data[:,0] + decoded_data[:,1]
    lo = plt.scatter(decoded_data[:,0], decoded_data[:,1], c=T, s=25, alpha=0.2, marker='o',label="2d")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    print("decoded_data: ", decoded_data)
    # print("ground trueth: ", x_train[110:120,:].reshape(10,5))

# trainModel()
testModel()

