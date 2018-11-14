import numpy as np
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
        self.port_value = {self.months:float(self.avail_capital)}

        # Initialize portfolio
        # argsort is ascending so negate indicator_arr
        self.positon_indices = np.argsort(-self.indicator_arr)[:self.n]

        # -1 indicates there is no position so no entry month
        self.entry_months = np.array([-1] * self.N)
        self.entry_months[self.positon_indices] = 0

        self.shares = np.array([0] * self.N)
        self.shares[self.positon_indices] = (self.avail_capital / self.n / self.price_arr[self.positon_indices]).astype(int)
        
        self.StockPosition = {'month' + str(self.months):{}}
        for idex in self.positon_indices:
        	self.StockPosition['month' + str(self.months)][str(self.tickers[idex])] = self.shares[idex]


        self.avail_capital -= np.dot(self.price_arr, self.shares)

        self.tradingHistory = {'Buy':{self.months:self.tickers[self.positon_indices]}, 'Sell':{}}

        self.holdingStock = {self.months:self.tickers[self.positon_indices]}

    def execute_sell(self, sell_indices):
        #print('selling', self.tickers[sell_indices])
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
            print(f'Month{self.months} stock {self.tickers[i]} avaliable capital is {self.avail_capital}, the volums is {self.volume_arr[-1,i]}')


    def execute_buy(self, buy_indices):
        #print('buying', self.tickers[buy_indices])
        buy_money = self.avail_capital / len(buy_indices)

        for i in buy_indices:
            # Allocate avail_capital equally
            buy_price = self.price_arr[-1, i] + 0.01

            n_shares = buy_money //buy_price
            if n_shares >= 0.1 * self.volume_arr[-1, i]:
                buy_price = self.price_arr[-1, i] + 0.01 + 0.1 * self.price_arr[-1, i]
                n_shares = buy_money //buy_price

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

        self.StockPosition['month'+str(self.months)] = {}
        for idex in self.positon_indices:
        	self.StockPosition['month'+str(self.months)][str(self.tickers[idex])] = (f'share: {self.shares[idex]}', f'price:{self.price_arr[-1, idex]}')



