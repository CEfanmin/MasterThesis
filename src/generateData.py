import pandas as pd
import numpy as np

df = pd.read_csv("../encodedata/test1.csv")
# print(df.head())
X_train = df.iloc[:50000,:].copy()
X_test = df.iloc[10000:11000,:].copy()  # test data for diffrient state:stand/walk/rest/sit

# z-score transform 
X_mean = X_train.mean()
X_std = X_train.std()
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_train = X_train
y_test = X_test

input_seq_len = 100  # using past 100 point
output_seq_len =10  # predict future 50 point

# in shape: (batch_size, time_steps, feature_dim)
def generate_train_samples(x=np.array(X_train).tolist(), y=np.array(y_train).tolist(), 
                           batch_size=10, input_seq_len=input_seq_len,
                           output_seq_len=output_seq_len):
    total_start_points=len(x)-input_seq_len-output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)  # random choice training data
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs,axis=0)
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    
    output_seq = np.take(y, output_batch_idxs,axis=0)
    return input_seq, output_seq

def generate_test_samples(x = np.array(X_test).tolist(), y = np.array(y_test).tolist(), 
                          input_seq_len = input_seq_len, 
                          output_seq_len = output_seq_len):
    total_samples = np.array(x).shape[0]
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)    
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)

    return input_seq, output_seq


