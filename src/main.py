from seq2seqModel import *
from matplotlib import pyplot as plt


# load data
x, y = generate_train_samples()
print("x.shape, y.shape is:")
print(x.shape, y.shape)
test_x, test_y = generate_test_samples()
print("test_x.shape, test_y.shape is:")
print(test_x.shape, test_y.shape)

def trainModel():
    total_iteractions = 200
    batch_size = 16
    KEEP_RATE = 0.5
    train_losses = []
    val_losses = []

    x = np.linspace(0, 40, 130)
    train_data_x = x[:110]

    rnn_model = build_graph(feed_previous=False)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)
        # writer = tf.summary.FileWriter("../logs/", sess.graph)
        print("Training losses: ")
        for i in range(total_iteractions):
            batch_input, batch_output = generate_train_samples(batch_size=batch_size)
            
            feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
            feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
            _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
            print(loss_t)
            
        temp_saver = rnn_model['saver']()
        save_path = temp_saver.save(sess, os.path.join('../seq2seqmodel/', 'indRNNmodel'))
            
    print("Checkpoint saved at: ", save_path)


def testModel():
    rnn_model = build_graph(feed_previous=True)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = rnn_model['saver']().restore(sess,  os.path.join('../seq2seqmodel/', 'seq2seqmodel'))
        feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} # batch prediction
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)

        final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
        final_preds = np.concatenate(final_preds, axis = 1)
        print("Test mse is: ", np.mean((final_preds - test_y)**2))
        test_y_expand = np.concatenate([test_y[i].reshape(-1) for i in range(0, test_y.shape[0], 10)], axis = 0)
        final_preds_expand = np.concatenate([final_preds[i].reshape(-1) for i in range(0, final_preds.shape[0], 10)], axis = 0)

        fig1 = plt.figure('fig1')
        plt.plot(final_preds_expand, color = 'orange', label = 'predicted')
        plt.plot(test_y_expand, color = 'blue', label = 'actual')
        plt.title("test model")
        plt.legend(loc="upper left")
        plt.show()

if __name__ =="__main__":
    trainModel()
    # testModel()

