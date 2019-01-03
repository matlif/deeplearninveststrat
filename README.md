# Deep Learning Investment Strategy
A factor-based quantitative investing strategy that employs deep neural networks to forecast company fundamentals, based on John Alberg and Zachary Lipton's paper "Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals".

# Data
please make sure you put these dataset in a proper directory so that you could read them

## 100_clean.csv
This dataset is our training and testing dataset. This dataset contains quarterly consecutive fundamental data of 100 stocks from 1990 to 2018. Because the paper utilized these fundamental data as features input to predict the EBIT/EV ratio, so we replicate this method. 

## monthendpricehistory.xls
This dataset is from CBOE official website, and it contains SPX index quarterly price. We used the SPX as the benchmark, and we compared our portfolio with the benchmark in our final report.

# Models
We tried to use the OLS, Basic RNN, LSTM and GRU to quarterly predict the EBIT/EV ratio. 

## linear_model.ipynb
In this jupyter notebook, we implemented the OLS to predict EBIT/EV ratio. Also, we summarized the ols results.

## finalWrapper.py 
There are three finalWrapper python files (finalWrapper.py ,finalWrapper2.py, finalWrapper3.py ), they are corresponding to the LSTM, GRU and Basic RNN model. The inputs of finalWrapper are dataset and model's parameters. The finalWrapper can achieve the following functions: 

* Automatically normalize data, adjusted data shape according to models' need
* Train models, make the prediction and generate the prediction results
* Run back testing 
* Automatically save each result in 'Results' folder  (If there's no 'Results' folder in your directory, don't worry, it will be created automatically)

# BackTesting.py
We built an efficient backtest module. It could record the quarterly portfolio value, detailed long/short trade history and the trading price of each trade.

# Pick the Best Model

## ParameterTuning.ipynb
We tried to pick the best model by inputting different **epoch number, batch size, network depth, neural numbers, learning rate and different structure**. We tried more than 400 pairs of parameters in total. This jupyter notebook imports the finalWrappers, and could help us tune these hyperparameters

# Miscellaneous

## AutoEncoder
### AutoEncoder.py and AutoencoderDenosing.ipynb

We implemented the AutoEncoder to denoise dataset, please see this jupyter notebook to find more detail. The **DenosingData.csv** is the final result.

## Data+Model+BackTest.ipynb
This jupyter notebook is the blue print of the finalWrapper. The reason why we keep this is that you could see how each percedure works clearly.

## attention_real_with_all_data_combined.ipynb
We tried to use the attention to enhance model performence. In this juputer notebook, it shows the results of attetion LSTM
