# Deep Learning Investment Strategy - In Progress

A factor-based quantitative investing strategy that employs deep neural networks to forecast company fundamentals, based on John Alberg and Zachary Lipton's paper "Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals".


# SignelAssetAndIndicator

In this notebook, we replicated the GRU structure, which was mentioned in the paper of "Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555."

We used the signal stock's relative fundamental and pricing data as input features to predict the indicator 'EBITV/EV'.The results showed we overfitted a little bit, therefore, the next step is to optimized the parameters and utilize more data

# Back Test
Thank you Jerry!! we got our backtest run smoothly and we could track the portfolio value and record position history, pretty cool. Good jobs guys!


# About the Wrapper

Please download the 'finalWrapper.py', it's a useful tool to tune parameters of our model. 

* After your download it, please see the lastest content of the code, where the '__main__' locates
* you could either change the parameters in the main function or you could import the 'finalWrapper.py' as an package and run it in the jupyter notebook

* The finalWrapper will automatically normalize data, train models, make prediction and generate the prediction results and back testing results. In the end, these results would be automatically saved in the folder 'Models' (If there's no 'Models' folder in your directory, don't worry, it will be created automatically)

*Then, you could find the results in the 'Models' folder.

Happy Holidays!!
