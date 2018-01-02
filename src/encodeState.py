import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import Dense, Input
from keras.models import Model, load_model 
import numpy as np
import pandas as pd
from keras import regularizers


# loadData()
df = pd.read_csv("../data/0307/test1/exo_raw_data.csv").fillna(0)
print(df.describe())
df_norm = (df - df.mean()) / df.std()
df_norm = df_norm - df_norm.min()
print(df_norm.describe())

df_norm = np.array(df_norm)
x_train = df_norm
# x_test = df_norm[44540:54540, :]
print(x_train.shape)
# print(x_test.shape)

def trainModel():
    # consturct autoencoder
    encoding_dim = 1
    input_data = Input(shape=(5,))
    # encoder layers
    encoded = Dense(5, activation='relu')(input_data)
    encoded = Dense(3, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(3, activation='relu')(encoder_output)
    decoded = Dense(5, activation='relu')(decoded)

    autoencoder = Model(inputs=input_data, outputs=decoded)

    # construct the encoder model 
    encoder = Model(inputs=input_data, outputs=encoder_output)

    # compile and fit
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(x_train, x_train, epochs=500, batch_size=128, shuffle=True)
    autoencoder.save("../model/autoencoderModel.h5")


def testModel():
    # load model and test
    autoencoder = load_model("../model/autoencoderModel.h5")
    decoded_data = autoencoder.predict(x_train[100:110,:].reshape(10,5))
    print("decoded_data: ", decoded_data)
    print("ground trueth: ", x_train[100:110,:].reshape(10,5))

# trainModel()
testModel()

