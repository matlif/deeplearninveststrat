# Deep Learning Investment Strategy
A factor-based quantitative investing strategy that employs deep neural networks to forecast company fundamentals, based on John Alberg and Zachary Lipton's paper "Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals".

# Data
Note: Datasets must be moved to the appropriate directory if you would like to run the code

## 100_clean.csv
This dataset is our training and testing dataset. It contains quarterly consecutive fundamental data of 100 stocks from 1990 to 2018.

## monthendpricehistory.xls
This dataset is from the CBOE official website, and it contains SPX index quarterly price. We used the SPX as the benchmark, and we compared our portfolio with the benchmark in our final report.

# Models
We tried to use the OLS, Basic RNN, LSTM and GRU to predict the EBIT/EV ratio quarterly. 

## linear_model.ipynb
In this Jupyter Notebook, we implemented OLS to predict the EBIT/EV ratio. Also, we summarize the OLS results.

## finalWrapper.py 
There are three finalWrapper python files (finalWrapper.py ,finalWrapper2.py, finalWrapper3.py ), they correspond to the LSTM, GRU and Basic RNN model. The inputs of finalWrapper are dataset and model parameters. The finalWrapper achieves the following: 

* Automatically normalize data, adjusted data shape according to models' needs
* Train models, make the prediction and generate the prediction results
* Run back testing 
* Automatically save each result in 'Results' folder  (If there's no 'Results' folder in your directory, don't worry, it will be created automatically)

# BackTesting.py
We built an efficient backtest module. It can record the quarterly portfolio value, detailed long/short trade history and the trading price of each trade.

# Pick the Best Model

## ParameterTuning.ipynb
We tried to pick the best model by inputting different **epoch number, batch size, network depth, neural numbers, learning rate and different structure**. We tried more than 400 pairs of parameters in total. This Jupyter Notebook imports the finalWrappers, and is used to help us tune the hyperparameters

# Miscellaneous

## AutoEncoder
### AutoEncoder.py and AutoencoderDenosing.ipynb

We implemented an AutoEncoder to denoise the dataset, please see this Jupyter Notebook for more detail. The **Data/denosing_data.csv** is the final result.

## Data+Model+BackTest.ipynb
This Jupyter Notebook is the blueprint of the finalWrapper. We kept this so that you can clearly see how each procedure works.

## attention_real_with_all_data_combined.ipynb
We tried to use the attention to enhance model performance. This Juputer Notebook, it shows the results of attention LSTM.
