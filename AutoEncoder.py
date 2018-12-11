from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf


class myAutoEncoder:
    def __init__(self,n_input, n_hidden_1, n_hidden_2):
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2

    def encoder(self,x):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]))
            }

        biases = {'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                  'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2]))
                  }

        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        return layer_2

    def decoder(self,x):
        weights = {'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
                   'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input]))
                   }

        biases = {'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                  'decoder_b2': tf.Variable(tf.random_normal([self.n_input]))
                  }

        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h2']), biases['decoder_b2']))
        return layer_5

    def NormalizedStockData(self, quarterlyData):
        min_max_scaler = preprocessing.MinMaxScaler()
        trainData = quarterlyData[:72]
        trainData = pd.DataFrame(min_max_scaler.fit_transform(trainData), columns=trainData.columns.values)

        trainDataWithNoise = trainData + np.random.normal(0, 0.5, (trainData.shape[0], trainData.shape[1]))
        testData = pd.DataFrame(min_max_scaler.fit_transform(quarterlyData), columns=quarterlyData.columns.values)
        return trainData, trainDataWithNoise, testData
