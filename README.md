# Deep Learning Investment Strategy - In Progress

A factor-based quantitative investing strategy that employs deep neural networks to forecast company fundamentals, based on John Alberg and Zachary Lipton's paper "Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals".


# Data 
## 100_clean.csv
This data set is our training and testing dataset. This dataset contains quarterly consecutive fundamental data of 100 stocks from 1990 to 2018. Because the paper utilized these fundamental data as features input to predict the EBIT/EV ratio, so we replicate this method. 

## monthendpricehistory.xls
This dataset is from CBOE official website, and it contains SPX index quarterly price. We used the SPX as the benchmark, and we compared our portfolio with the benchmark in our final report.


# Tuning the Hyperparameters

Please download the 'TuningParameters.ipynb', it's a useful tool to tune parameters of our model. 

* The TuningParameters.ipynb can automatically normalize data, train models, make prediction and generate the prediction results and back testing results. In the end, these results would be automatically saved in the folder 'Results' (If there's no 'Results' folder in your directory, don't worry, it will be created automatically)

* Then, you could find the results in the 'Results' folder.

Happy Holidays!!


# Back Test
Thank you Jerry!! we got our backtest run smoothly and we could track the portfolio value and record position history, pretty cool. Good jobs guys!

