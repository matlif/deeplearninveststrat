from datetime import datetime
from tensorflow.python.ops import rnn

import tensorflow as tf
import numpy as np
import os


class MyLSTM:

    def __init__(self, X_train, X_test, Y_train, Y_test, parameters):
        self.default_Parameters = {
            'hm_epochs': 200,
            'n_batches': 4,
            'rnn_size': 64,
            'num_layers': 1,
            'attention': False
            }
        self.parameters = parameters
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.n_classes = 1
        self.seq_len = 12

        self.hm_epochs = self.default_Parameters['hm_epochs']
        self.n_batches = self.default_Parameters['n_batches']
        self.rnn_size = self.default_Parameters['rnn_size']
        self.num_layers = self.default_Parameters['num_layers']
        self.attention = self.default_Parameters['attention']

        if parameters:
            self.hm_epochs = self.parameters['hm_epochs']
            self.n_batches = self.parameters['n_batches']
            self.rnn_size = self.parameters['rnn_size']
            self.num_layers = self.parameters['num_layers']
            self.attention = self.parameters['attention']

        self.batch_size = self.X_train.shape[0] / self.n_batches
        self.chunk_size = self.X_train.shape[2]
        tf.reset_default_graph()


    def createBaseFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def recurrent_neural_network(self, x):
        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        #     layer = {'weights':tf.Variable(np.random.normal(size=(rnn_size,n_classes)).astype('float32')),
        #              'biases':tf.Variable(tf.random_normal([n_classes]))}

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.chunk_size])
        x = tf.split(x, self.seq_len, 0)
        lstm_cells = []

        for _ in range(self.num_layers):
            cell = tf.contrib.rnn.LSTMCell(self.rnn_size, state_is_tuple=True,
                                               initializer=tf.contrib.layers.xavier_initializer())
            if self.attention:
                cell = tf.contrib.rnn.AttentionCellWrapper(cell, self.seq_len)

            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=1)
            lstm_cells.append(tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0))

        multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

        outputs, states = rnn.static_rnn(multi_cell, x, dtype=tf.float32)
        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']  # softmax layer
        return output


    def train_neural_network(self):
        x = tf.placeholder('float', [None, self.seq_len, self.chunk_size])
        y = tf.placeholder('float')
        prediction = self.recurrent_neural_network(x)
        cost = tf.losses.mean_squared_error(predictions=prediction, labels=y)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(self.hm_epochs):
                epoch_loss = 0
                for i in range(self.n_batches):
                    epoch_x = self.X_train[i * int(self.batch_size):(i + 1) * int(self.batch_size)]
                    epoch_y = self.Y_train[i * int(self.batch_size):(i + 1) * int(self.batch_size)]

                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c

                print('Epoch', epoch, 'completed out of', self.hm_epochs, 'loss:', np.mean(epoch_loss))

            pred_train = sess.run(prediction, feed_dict={x: self.X_train.astype('float32')})[:, 0]
            pred_test = sess.run(prediction, feed_dict={x: self.X_test.astype('float32')})[:, 0]

            print('Testing MAE:', np.mean(np.abs(pred_test - self.Y_test)))
            print('Testing MSE:', np.mean((pred_test - self.Y_test) ** 2))

            # nameList = [self.seq_len, self.hm_epochs, self.n_classes, self.n_batches, self.rnn_size, self.num_layers, self.attention]
            # nameList = [str(i) for i in nameList]
            # nameList = ''.join(nameList)
            # export_dir = './Models/{0}'.format('LSTM-' + datetime.strftime(datetime.now(), '%Y%m%d%H%M') +'-Paras-' + nameList + '/')
            # self.createBaseFolder(export_dir)
            #
            # tf.saved_model.simple_save(sess, export_dir, inputs={"x": self.X_train, "y": self.Y_train}, outputs={"z": pred_test})
            return pred_test