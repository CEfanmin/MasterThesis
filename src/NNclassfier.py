
import os, time
os.environ['KERAS_BACKEND'] = 'tensorflow'
import win_unicode_console
win_unicode_console.enable()
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import regularizers
from matplotlib import pyplot as plt


# fix random seed for reproducibility
seed = 226
np.random.seed(seed)

## load dataset
def loadData():
    raw_data = pd.read_csv("../../data/0307/test1/exo_sample_data_with_targets.csv")
    print("all size is", np.array(raw_data).shape)
    dataset = raw_data.values
    X = dataset[:, 0:9].astype(float)
    Y = dataset[:, -1]

    ## encode labels
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    xTrain, xTest, yTrain, yTest = train_test_split(X, dummy_y,test_size=0.2, random_state=226)
    return xTrain, xTest, yTrain, yTest

## define the model
def BaselineModel():
    model = Sequential()
    model.add(Dense(32, input_dim = 9, activation='relu'))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

original_model = BaselineModel()

xTrain, xTest, yTrain, yTest = loadData()
start_time = time.time()
original_hist = original_model.fit(xTrain, yTrain, epochs=300, verbose=1,batch_size=32,
                                validation_data=(xTest, yTest))
end_time = time.time() - start_time
print("using time is: ", end_time)

original_model.save_weights("../model/DenseModel.h5")

# plot training loss
epochs = range(0, 300)
original_val_loss = original_hist.history['loss']
plt.plot(epochs, original_val_loss, 'b',label='Original model')
plt.xlabel('Epochs')
plt.ylabel('training loss')
plt.legend()
plt.show()
print("Saved model to disk")






# ## test model
# test = np.array([0.948203,-107.960335,20.957085,115,24,-1.238987,-1.221456,1.692592,1.58796]).reshape(1,9)
# original_model.load_weights('../model/DenseModel.h5')
# result = original_model.predict(test)
# print(result)

