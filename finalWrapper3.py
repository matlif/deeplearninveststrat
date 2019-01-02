from datetime import datetime
from tensorflow.python.ops import rnn

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

class algo2:
    # intial_prices, initial_indicators, initial_volume are N-d arrays

    def __init__(self, initial_prices, initial_indicators, initial_volume, stockNumber, tickerList):

        self.price_arr = initial_prices
        self.indicator_arr = initial_indicators
        self.volume_arr = initial_volume
        self.N = stockNumber
        self.n = 10

        self.tickers = tickerList

        self.months = 0
        self.avail_capital = 100000000
        self.port_value = {self.months: float(self.avail_capital)}

        # Initialize portfolio
        # argsort is ascending so negate indicator_arr
        self.positon_indices = np.argsort(-self.indicator_arr)[:self.n]

        # -1 indicates there is no position so no entry month
        self.entry_months = np.array([-1] * self.N)
        self.entry_months[self.positon_indices] = 0

        self.shares = np.array([0] * self.N)
        self.shares[self.positon_indices] = (self.avail_capital / self.n / self.price_arr[self.positon_indices]).astype(
            int)

        self.StockPosition = {'month' + str(self.months): {}}
        for idex in self.positon_indices:
            self.StockPosition['month' + str(self.months)][str(self.tickers[idex])] = self.shares[idex]

        self.avail_capital -= np.dot(self.price_arr, self.shares)

        self.tradingHistory = {'Buy': {self.months: self.tickers[self.positon_indices]}, 'Sell': {}}

        self.holdingStock = {self.months: self.tickers[self.positon_indices]}

    def execute_sell(self, sell_indices):
        # print('selling', self.tickers[sell_indices])
        for i in sell_indices:
            sell_gain = self.price_arr[-1, i] * self.shares[i]
            transaction_cost = 0.01 * self.shares[i]
            slip_cost = 0
            if self.shares[i] >= 0.1 * self.volume_arr[-1, i]:
                slip_cost = self.price_arr[-1, i] * 0.01 * self.shares[i]

            # Update capital and Reset shares
            self.avail_capital += (sell_gain - transaction_cost - slip_cost)
            self.shares[i] = 0
            # Reset entry_position  to -1 to denote no position
            self.entry_months[i] = -1
            print(
                f'Month{self.months} stock {self.tickers[i]} avaliable capital is {self.avail_capital}, the volums is {self.volume_arr[-1,i]}')

    def execute_buy(self, buy_indices):
        # print('buying', self.tickers[buy_indices])
        buy_money = self.avail_capital / len(buy_indices)

        for i in buy_indices:
            # Allocate avail_capital equally
            buy_price = self.price_arr[-1, i] + 0.01

            n_shares = buy_money // buy_price
            if n_shares >= 0.1 * self.volume_arr[-1, i]:
                buy_price = self.price_arr[-1, i] + 0.01 + 0.1 * self.price_arr[-1, i]
                n_shares = buy_money // buy_price

            self.shares[i] = n_shares
            self.entry_months[i] = self.months
            self.avail_capital = self.avail_capital - (n_shares * buy_price)

    def process_new_data(self, new_prices, new_indicators, new_volumes):
        self.months += 1
        self.price_arr = np.vstack((self.price_arr, new_prices))
        self.indicator_arr = np.vstack((self.indicator_arr, new_indicators))
        self.volume_arr = np.vstack((self.volume_arr, new_volumes))
        # Update the portfolios
        self.update_portfolio()
        self.port_value[self.months] = np.dot(self.price_arr[-1, :], self.shares) + self.avail_capital

    def update_portfolio(self):
        self.new_top_indices = np.argsort(-self.indicator_arr[-1])[:self.n]
        sell_indices = list()
        buy_indices = list()
        postitionList = list(self.positon_indices)

        # Try to add new indices not currently in portfolio to buy_indices
        for x in self.positon_indices:
            # if held for more than 12 months and not in new top indices
            if self.months - self.entry_months[x] >= 12 and x not in self.new_top_indices:
                sell_indices.append(x)
                postitionList.remove(x)

        # Add k best new indices to portfolio, k = number of stocks we will sell
        if sell_indices:
            for y in self.new_top_indices:
                if y not in self.positon_indices:
                    buy_indices.append(y)
                    postitionList.append(y)
                    # Stop once we have same number of stock we buy as sell
                    if len(buy_indices) == len(sell_indices):
                        break

            self.execute_sell(sell_indices)
            self.execute_buy(buy_indices)

        self.tradingHistory['Buy'][self.months] = self.tickers[buy_indices]
        self.tradingHistory['Sell'][self.months] = self.tickers[sell_indices]

        self.positon_indices = np.array(postitionList)

        self.holdingStock[self.months] = self.tickers[self.positon_indices]

        self.StockPosition['month' + str(self.months)] = {}
        for idex in self.positon_indices:
            self.StockPosition['month' + str(self.months)][str(self.tickers[idex])] = (
            f'share: {self.shares[idex]}', f'price:{self.price_arr[-1, idex]}')


class MyRNN:

    def __init__(self, X_train, X_test, Y_train, Y_test, parameters):
        self.default_Parameters = {
            'hm_epochs': 200,
            'n_batches': 4,
            'rnn_size': 64,
            'num_layers': 1,
            'attention': False,
            'learning_rate': 0.001
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
        self.learning_rate = self.default_Parameters['learning_rate']

        if parameters:
            self.hm_epochs = self.parameters['hm_epochs']
            self.n_batches = self.parameters['n_batches']
            self.rnn_size = self.parameters['rnn_size']
            self.num_layers = self.parameters['num_layers']
            self.attention = self.parameters['attention']
            self.learning_rate = self.parameters['learning_rate']

        self.batch_size = self.X_train.shape[0] / self.n_batches
        self.chunk_size = self.X_train.shape[2]
        tf.reset_default_graph()

    def createBaseFolder(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print('Error: Creating directory. ' + directory)
            pass


    def creatDirectory(self):
        nameList = [self.hm_epochs, self.n_batches, self.rnn_size, self.num_layers, self.learning_rate, self.attention]
        nameList = [str(i) for i in nameList]
        nameList = ''.join(nameList)
        if '.' in nameList:
            nameList = nameList.replace('.', '-')
        export_dir = './Results2/{0}'.format('BasicRNN-' + datetime.strftime(datetime.now(), '%Y%m%d%H%M') +'-Paras-' + nameList + '/')
        return export_dir

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
            cell = tf.contrib.rnn.BasicRNNCell(self.rnn_size)
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
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

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

            MAE =  np.mean(np.abs(pred_test - self.Y_test))
            MSE = np.mean((pred_test - self.Y_test) ** 2)

            print(f'Testing MAE: {MAE}')
            print(f'Testing MSE: {MSE}')
            return pred_test, MAE, MSE


class MyWrapper3:

    def __init__(self, data, parameters):
        self.parameters = parameters
        self.df = data
        self.df1 = self.df.copy()
        self.timeline = self.df[self.df['gvkey'] == 1209]['Date'].values.tolist()
        self.df.drop(['Ticker', 'Date', 'Adjusted Close', 'Volume'], axis=1, inplace=True)

        self.myTimeLine = self.timeline[71+12+1:]
        self.used_keys = list()

        self.df1 = self.df1[['gvkey', 'Date', 'Adjusted Close', 'Volume']]
        self.df1['Date'] = pd.to_datetime(self.df1['Date'])
        self.df1.set_index('Date', inplace=True)
        self.df1 = self.df1[self.df1.index >= datetime.strptime(self.myTimeLine[0], '%Y-%m-%d')]

        self.X_train_total = np.array([])
        self.X_test_total = np.array([])
        self.Y_train_total = np.array([])
        self.Y_test_total = np.array([])


    def DataProcess(self):
        X_train_total = []
        Y_train_total = []

        X_test_total = []
        Y_test_total = []

        seq_len = 12
        keys = list(self.df['gvkey'].unique())

        if 8214 in keys:
            keys.remove(8214)
        if 15350 in keys:
            keys.remove(15350)

        norm_dict = {}

        for key in keys:
            df_temp = self.df[self.df['gvkey'] == key].drop(['gvkey'], axis=1)
            if len(df_temp) != 115:
                #print(f'we drop the {key}')
                continue

            # Some columns have all Nan
            if df_temp.isnull().all().any():
                #print(f'we drop the {key}')
                continue

            self.used_keys.append(key)

            df_temp.ffill(inplace=True)
            norm_vector = df_temp.abs().max(axis=0).values
            # When a columns is all 0s, its max will be 0. We cannot divide by 0 so replace with 1.
            norm_vector[norm_vector == 0] = 1
            norm_dict[key] = norm_vector
            # Some columns have all 0s so dividing will introduce Nan
            df_temp = df_temp / norm_vector

            if df_temp.isnull().any().any():
                print(key, 'null')

            X_list = list()
            Y_list = list()
            # len(df_temp) is 115
            for i in range(len(df_temp) - seq_len - 1):
                X_list += [df_temp.iloc[i:i + seq_len].values]
                Y_list += [df_temp.iloc[i + seq_len + 1, -1]]

            X = np.array(X_list)
            Y = np.array(Y_list)
            # First 71 quarters are for training
            # The 72nd quarter starts to test
            # Some gvkeys have no insufficient data (dont have 12 quarters of data).
            # >=5 to ensure there is testing data.
            if X.shape[0] >= 5:
                X_train = X[:int(len(X) * 0.7)]
                X_test = X[int(len(X) * 0.7):]

                Y_train = Y[:int(len(X) * 0.7)]
                Y_test = Y[int(len(X) * 0.7):]

                X_train_total += [X_train]
                X_test_total += [X_test]

                Y_train_total += [Y_train]
                Y_test_total += [Y_test]

        self.X_train_total = np.vstack(X_train_total)
        self.X_test_total = np.vstack(X_test_total)

        self.Y_train_total = np.concatenate(Y_train_total)
        self.Y_test_total = np.concatenate(Y_test_total)


    def run(self):
        self.DataProcess()
        print('........Data Process Finished...........')
        print('..............Running Model.............')
        a = MyRNN(self.X_train_total, self.X_test_total, self.Y_train_total, self.Y_test_total, self.parameters)
        result, MAE, MSE = a.train_neural_network()

        export_dir = a.creatDirectory()
        a.createBaseFolder(export_dir)

        pred_result_split = np.split(result, len(result) / 31)
        pred_result_dict = dict(zip(self.used_keys, pred_result_split))
        pred_result = pd.DataFrame.from_dict(pred_result_dict, orient='index').T
        pred_result['Date'] = self.myTimeLine
        pred_result['Date'] = pd.to_datetime(pred_result['Date'])
        pred_result.set_index('Date', inplace=True)

        print('...........Prediction Finished..........')

        df1gb = self.df1.groupby('gvkey')
        final = {}
        for key, data in df1gb:
            if key in self.used_keys:
                data['PredictedNextIndicator'] = pred_result.loc[:, key]
                final[key] = data
                final[key]['PredictedNextIndicator'] = final[key]['PredictedNextIndicator'].shift(-1)
                final[key] = final[key].dropna()

        ff = final[1209]
        for k in self.used_keys[1:]:
            ff = pd.concat([ff, final[k]])

        df2 = ff.pivot(columns='gvkey', values=['Adjusted Close', 'Volume', 'PredictedNextIndicator'])
        periods = len(df2)
        prices = df2['Adjusted Close'].values
        indicators = df2['PredictedNextIndicator'].values
        volume = df2['Volume'].values * 1000000
        tickers = df2['Volume'].columns.values

        print('..............Back Test..........')
        mm = algo2(prices[0, :], indicators[0, :], volume[0, :], len(tickers), tickers)
        for i in range(1, periods):
            print(mm.avail_capital)
            mm.process_new_data(prices[i, :], indicators[i, :], volume[i, :])

        print('-------output result--------')
        spx = pd.read_excel('./monthendpricehistory.xls')
        spx['Date'] = pd.to_datetime(spx['Date'])
        spx.set_index('Date', inplace=True)
        spx = spx[['SPX']]
        spx = spx[spx.index >= datetime.strptime(self.myTimeLine[0], '%Y-%m-%d')]
        spxInitialShare = 100000000 / spx['SPX'].iloc[0]
        spx['SPXBench'] = spx['SPX'] * spxInitialShare

        pnl = [i - 100000000 for i in mm.port_value.values()]
        mysharpe = np.mean(pnl) / np.std(pnl, ddof=1)

        resultLogPath = export_dir+'/resultLog.txt'
        f = open(resultLogPath, "w+")
        f.write('Model Part:\n')
        f.write(f'MSE is {MSE}, MAE is {MAE}\n')
        f.write('Back Test Result:\n')
        f.write(str(mm.StockPosition))
        f.write(f'\n the PNL is {pnl}, \nSarpe Ratio is {mysharpe}')
        f.close()

        port_val = [i for i in mm.port_value.values()]
        tt = df2['Adjusted Close'].index.tolist()
        portfolio = pd.DataFrame(port_val, tt, columns=['PortfolioValue'])
        portfolio.index.name = 'Date'
        portfolio = pd.concat([portfolio, spx], axis=1)

        portfolio = portfolio.dropna()
        portfolio.to_csv(export_dir+'/MyPortfolio.csv')
        print(f'Finised, results path is {export_dir}')

        return self.parameters, MAE, MSE, pnl, mysharpe


if __name__ == '__main__':
    data = pd.read_csv('/Users/bowen/Desktop/H/deeplearninveststrat/Data/100_clean.csv', index_col=0)
    parameters = {
                    'hm_epochs': 20,
                    'n_batches': 4,
                    'rnn_size': 10,
                    'num_layers': 1,
                    'attention': True,
                    'learning_rate': 0.1
                }

    MyWrapper3 = MyWrapper3(data, parameters)
    parameters, MAE, MSE, pnl, mysharpe = MyWrapper3.run()
