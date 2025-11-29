# Machine Learning for Trading - Stefan Jansen Repository Overview

Repository: https://github.com/stefan-jansen/machine-learning-for-trading
Stars: 16,000+

## Key Chapters and Topics

# ML for Trading - 2<sup>nd</sup> Edition

This [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d) aims to show how ML can add value to algorithmic trading strategies in a practical yet comprehensive way. It covers a broad range of ML techniques from linear regression to deep reinforcement learning and demonstrates how to build, backtest, and evaluate a trading strategy driven by model predictions.  

In four parts with **23 chapters plus an appendix**, it covers on **over 800 pages**:
- important aspects of data sourcing, **financial feature engineering**, and portfolio management, 
- the design and evaluation of long-short **strategies based on supervised and unsupervised ML algorithms**,
- how to extract tradeable signals from **financial text data** like SEC filings, earnings call transcripts or financial news,
- using **deep learning** models like CNN and RNN with market and alternative data, how to generate synthetic data with generative adversarial networks, and training a trading agent using deep reinforcement learning

<p align="center">
<a href="https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d">
<img src="https://ml4t.s3.amazonaws.com/assets/cover_toc_gh.png" width="75%">
</a>
</p>

This repo contains **over 150 notebooks** that put the concepts, algorithms, and use cases discussed in the book into action. They provide numerous examples that show:
- how to work with and extract signals from market, fundamental and alternative text and image data, 
- how to train and tune models that predict returns for different asset classes and investment horizons, including how to replicate recently published research, and 
- how to design, backtest, and evaluate trading strategies.

> We **highly recommend** reviewing the notebooks while reading the book; they are usually in an executed state and often contain additional information not included due to space constraints.  

In addition to the information in this repo, the book's [website](ml4trading.io) contains chapter summary and additional information.

## Join the ML4T Community!

To make it easy for readers to ask questions about the book's content and code examples, as well as the development and implementation of their own strategies and industry developments, we are hosting an online [platform](https://exchange.ml4trading.io/).

Please [join](https://exchange.ml4trading.io/) our community and connect with fellow traders interested in leveraging ML for trading strategies, share your experience, and learn from each other! 

## What's new in the 2<sup>nd</sup> Edition?

First and foremost, this [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=VMKJPZC4N36TTZZCWATP&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=8f331266-0d21-4c76-a3eb-d2e61d23bb31&pd_rd_w=kVGNF&pd_rd_wg=LYLKH&ref_=pd_gw_ci_mcx_mr_hp_d) demonstrates how you can extract signals from a diverse set of data sources and design trading strategies for different asset classes using a broad range of supervised, unsupervised, and reinforcement learning algorithms. It also provides relevant mathematical and statistical knowledge to facilitate the tuning of an algorithm or the interpretation of the results. Furthermore, it covers the financial background that will help you work with market and fundamental data, extract informative features, and manage the performance of a trading strategy.

From a practical standpoint, the 2nd edition aims to equip you with the conceptual understanding and tools to develop your own ML-based trading strategies. To this end, it frames ML as a critical element in a process rather than a standalone exercise, introducing the end-to-end ML for trading workflow from data sourcing, feature engineering, and model optimization to strategy design and backtesting.

More specifically, the ML4T workflow starts with generating ideas for a well-defined investment universe, collecting relevant data, and extracting informative features. It also involves designing, tuning, and evaluating ML models suited to the predictive task. Finally, it requires developing trading strategies to act on the models' predictive signals, as well as simulating and evaluating their performance on historical data using a backtesting engine. Once you decide to execute an algorithmic strategy in a real market, you will find yourself iterating over this workflow repeatedly to incorporate new information and a changing environment.

<p align="center">
<img src="https://i.imgur.com/kcgItgp.png" width="75%">
</p>

The [second edition](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d)'s emphasis on the ML4t workflow translates into a new chapter on [strategy backtesting](08_ml4t_workflow), a new [appendix](24_alpha_factor_library) describing over 100 different alpha factors, and many new practical applications. We have also rewritten most of the existing content for clarity and readability. 

The trading applications now use a broader range of data sources beyond daily US equity prices, including international stocks and ETFs. It also demonstrates how to use ML for an intraday strategy with minute-frequency equity data. Furthermore, it extends the coverage of alternative data sources to include SEC filings for sentiment analysis and return forecasts, as well as satellite images to classify land use. 

Another innovation of the second edition is to replicate several trading applications recently published in top journals: 
- [Chapter 18](18_convolutional_neural_nets) demonstrates how to apply convolutional neural networks to time series converted to image format for return predictions based on [Sezer and Ozbahoglu](https://www.researchgate.net/publication/324802031_Algorithmic_Financial_Trading_with_Deep_Convolutional_Neural_Networks_Time_Series_to_Image_Conversion_Approach) (2018). 
- [Chapter 20](20_autoencoders_for_conditional_risk_factors) shows how to extract risk factors conditioned on stock characteristics for asset pricing using autoencoders based on [Autoencoder Asset Pricing Models](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) by Shihao Gu, Bryan T. Kelly, and Dacheng Xiu (2019), and 
- [Chapter 21](21_gans_for_synthetic_time_series) shows how to create synthetic training data using generative adversarial networks based on [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks) by Jinsung Yoon, Daniel Jarrett, and Mihaela van der Schaar (2019).

All applications now use the latest available (at the time of writing) software versions such as pandas 1.0 and TensorFlow 2.2. There is also a customized version of Zipline that makes it easy to include machine learning model predictions when designing a trading strategy.

## Installation, data sources and bug reports

The code examples rely on a wide range of Python libraries from the data science and finance domains.

It is not necessary to try and install all libraries at once because this increases the likeliihood of encountering version conflicts. Instead, we recommend that you install the libraries required for a specific chapter as you go along.

> Update March 2022: `zipline-reloaded`, `pyfolio-reloaded`, `alphalens-reloaded`, and `empyrical-reloaded` are now available on the `conda-forge` channel. The channel `ml4t` only contains outdated versions and will soon be removed.

> Update April 2021: with the update of [Zipline](https://zipline.ml4trading.io), it is no longer necessary to use Docker. The installation instructions now refer to OS-specific environment files that should simplify your running of the notebooks.

> Update Februar 2021: code sample release 2.0 updates the conda environments provided by the Docker image to Python 3.8, Pandas 1.2, and TensorFlow 1.2, among others; the Zipline backtesting environment with now uses Python 3.6.

- The [installation](installation/README.md) directory contains detailed instructions on setting up and using a Docker image to run the notebooks. It also contains configuration files for setting up various `conda` environments and install the packages used in the notebooks directly on your machine if you prefer (and, depending on your system, are prepared to go the extra mile).
- To download and preprocess many of the data sources used in this book, see the instructions in the [README](data/README.md) file alongside various notebooks in the [data](data) directory.

> If you have any difficulties installing the environments, downloading the data or running the code, please raise a **GitHub issue** in the repo ([here](https://github.com/stefan-jansen/machine-learning-for-trading/issues)). Working with GitHub issues has been described [here](https://guides.github.com/features/issues/).

> **Update**: You can download the **[algoseek](https://www.algoseek.com)** data used in the book [here](https://www.algoseek.com/ml4t-book-data.html). See instructions for preprocessing in [Chapter 2](02_market_and_fundamental_data/02_algoseek_intraday/README.md) and an intraday example with a gradient boosting model in [Chapter 12](12_gradient_boosting_machines/10_intraday_features.ipynb).  

> **Update**: The [figures](figures) directory contains color versions of the charts used in the book. 

# Outline & Chapter Summary

The [book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=GZH2XZ35GB3BET09PCCA&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=91a679c7-f069-4a6e-bdbb-a2b3f548f0c8&pd_rd_w=2B0Q0&pd_rd_wg=GMY5S&ref_=pd_gw_ci_mcx_mr_hp_d) has four parts that address different challenges that arise when sourcing and working with market, fundamental and alternative data sourcing, developing ML solutions to various predictive tasks in the trading context, and designing and evaluating a trading strategy that relies on predictive signals generated by an ML model.

> The directory for each chapter contains a README with additional information on content, code examples and additional resources.  

[Part 1: From Data to Strategy Development](#part-1-from-data-to-strategy-development)
* [01 Machine Learning for Trading: From Idea to Execution](#01-machine-learning-for-trading-from-idea-to-execution)
* [02 Market & Fundamental Data: Sources and Techniques](#02-market--fundamental-data-sources-and-techniques)
* [03 Alternative Data for Finance: Categories and Use Cases](#03-alternative-data-for-finance-categories-and-use-cases)
* [04 Financial Feature Engineering: How to research Alpha Factors](#04-financial-feature-engineering-how-to-research-alpha-factors)
* [05 Portfolio Optimization and Performance Evaluation](#05-portfolio-optimization-and-performance-evaluation)

[Part 2: Machine Learning for Trading: Fundamentals](#part-2-machine-learning-for-trading-fundamentals)
* [06 The Machine Learning Process](#06-the-machine-learning-process)
* [07 Linear Models: From Risk Factors to Return Forecasts](#07-linear-models-from-risk-factors-to-return-forecasts)
* [08 The ML4T Workflow: From Model to Strategy Backtesting](#08-the-ml4t-workflow-from-model-to-strategy-backtesting)
* [09 Time Series Models for Volatility Forecasts and Statistical Arbitrage](#09-time-series-models-for-volatility-forecasts-and-statistical-arbitrage)
* [10 Bayesian ML: Dynamic Sharpe Ratios and Pairs Trading](#10-bayesian-ml-dynamic-sharpe-ratios-and-pairs-trading)
* [11 Random Forests: A Long-Short Strategy for Japanese Stocks](#11-random-forests-a-long-short-strategy-for-japanese-stocks)
* [12 Boosting your Trading Strategy](#12-boosting-your-trading-strategy)
* [13 Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning](#13-data-driven-risk-factors-and-asset-allocation-with-unsupervised-learning)

[Part 3: Natural Language Processing for Trading](#part-3-natural-language-processing-for-trading)
* [14 Text Data for Trading: Sentiment Analysis](#14-text-data-for-trading-sentiment-analysis)
* [15 Topic Modeling: Summarizing Financial News](#15-topic-modeling-summarizing-financial-news)
* [16 Word embeddings for Earnings Calls and SEC Filings](#16-word-embeddings-for-earnings-calls-and-sec-filings)

[Part 4: Deep & Reinforcement Learning](#part-4-deep--reinforcement-learning)
* [17 Deep Learning for Trading](#17-deep-learning-for-trading)
* [18 CNN for Financial Time Series and Satellite Images](#18-cnn-for-financial-time-series-and-satellite-images)
* [19 RNN for Multivariate Time Series and Sentiment Analysis](#19-rnn-for-multivariate-time-series-and-sentiment-analysis)
* [20 Autoencoders for Conditional Risk Factors and Asset Pricing](#20-autoencoders-for-conditional-risk-factors-and-asset-pricing)
* [21 Generative Adversarial Nets for Synthetic Time Series Data](#21-generative-adversarial-nets-for-synthetic-time-series-data)
* [22 Deep Reinforcement Learning: Building a Trading Agent](#22-deep-reinforcement-learning-building-a-trading-agent)
* [23 Conclusions and Next Steps](#23-conclusions-and-next-steps)
* [24 Appendix - Alpha Factor Library](#24-appendix---alpha-factor-library)

## Part 1: From Data to Strategy Development

The first part provides a framework for developing trading strategies driven by machine learning (ML). It focuses on the data that power the ML algorithms and strategies discussed in this book, outlines how to engineer and evaluates features suitable for ML models, and how to manage and measure a portfolio's performance while executing a trading strategy.

### 01 Machine Learning for Trading: From Idea to Execution

This [chapter](01_machine_learning_for_trading) explores industry trends that have led to the emergence of ML as a source of competitive advantage in the investment industry. We will also look at where ML fits into the investment process to enable algorithmic trading strategies. 

More specifically, it covers the following topics:
- Key trends behind the rise of ML in the investment industry
- The design and execution of a trading strategy that leverages ML
- Popular use cases for ML in trading

### 02 Market & Fundamental Data: Sources and Techniques

This [chapter](02_market_and_fundamental_data) shows how to work with market and fundamental data and describes critical aspects of the environment that they reflect. For example, familiarity with various order types and the trading infrastructure matter not only for the interpretation of the data but also to correctly design backtest simulations. We also illustrate how to use Python to access and manipulate trading and financial statement data.  

Practical examples demonstrate how to work with trading data from NASDAQ tick data and Algoseek minute bar data with a rich set of attributes capturing the demand-supply dynamic that we will later use for an ML-based intraday strategy. We also cover various data provider APIs and how to source financial statement information from the SEC.

<p align="center">
<img src="https://i.imgur.com/enaSo0C.png" title="Order Book" width="50%"/>
</p>
In particular, this chapter covers:

- How market data reflects the structure of the trading environment
- Working with intraday trade and quotes data at minute frequency
- Reconstructing the **limit order book** from tick data using NASDAQ ITCH 
- Summarizing tick data using various types of bars
- Working with eXtensible Business Reporting Language (XBRL)-encoded **electronic filings**
- Parsing and combining market and fundamental data to create a P/E series
- How to access various market and fundamental data sources using Python

### 03 Alternative Data for Finance: Categories and Use Cases

This [chapter](03_alternative_data) outlines categories and use cases of alternative data, describes criteria to assess the exploding number of sources and providers, and summarizes the current market landscape. 

It also demonstrates how to create alternative data sets by scraping websites, such as collecting earnings call transcripts for use with natural language processing (NLP) and sentiment analysis algorithms in the third part of the book.
 
More specifically, this chapter covers:

- Which new sources of signals have emerged during the alternative data revolution
- How individuals, business, and sensors generate a diverse set of alternative data
- Important categories and providers of alternative data
- Evaluating how the burgeoning supply of alternative data can be used for trading
- Working with alternative data in Python, such as by scraping the internet

### 04 Financial Feature Engineering: How to research Alpha Factors

If you are already familiar with ML, you know that feature engineering is a crucial ingredient for successful predictions. It matters at least as much in the trading domain, where academic and industry researchers have investigated for decades what drives asset markets and prices, and which features help to explain or predict price movements.

<p align="center">
<img src="https://i.imgur.com/UCu4Huo.png" width="70%">
</p>

This [chapter](04_alpha_factor_research) outlines the key takeaways of this research as a starting point for your own quest for alpha factors. It also presents essential tools to compute and test alpha factors, highlighting how the NumPy, pandas, and TA-Lib libraries facilitate the manipulation of data and present popular smoothing techniques like the wavelets and the Kalman filter that help reduce noise in data. After reading it, you will know about:
- Which categories of factors exist, why they work, and how to measure them,
- Creating alpha factors using NumPy, pandas, and TA-Lib,
- How to de-noise data using wavelets and the Kalman filter,
- Using Zipline to test individual and multiple alpha factors,
- How to use [Alphalens](https://github.com/quantopian/alphalens) to evaluate predictive performance.
 
### 05 Portfolio Optimization and Performance Evaluation

Alpha factors generate signals that an algorithmic strategy translates into trades, which, in turn, produce long and short positions. The returns and risk of the resulting portfolio determine whether the strategy meets the investment objectives.
<p align="center">
<img src="https://i.imgur.com/E2h63ZB.png" width="65%">
</p>

There are several approaches to optimize portfolios. These include the application of machine learning (ML) to learn hierarchical relationships among assets and treat them as complements or substitutes when designing the portfolio's risk profile. This [chapter](05_strategy_evaluation) covers:
- How to measure portfolio risk and return
- Managing portfolio weights using mean-variance optimization and alternatives
- Using machine learning to optimize asset allocation in a portfolio context
- Simulating trades and create a portfolio based on alpha factors using Zipline
- How to evaluate portfolio performance using [pyfolio](https://quantopian.github.io/pyfolio/)

## Part 2: Machine Learning for Trading: Fundamentals

The second part covers the fundamental supervised and unsupervised learning algorithms and illustrates their application to trading strategies. It also introduces the Quantopian platform that allows you to leverage and combine the data and ML techniques developed in this book to implement algorithmic strategies that execute trades in live markets.

### 06 The Machine Learning Process

This [chapter](06_machine_learning_process) kicks off Part 2 that illustrates how you can use a range of supervised and unsupervised ML models for trading. We will explain each model's assumptions and use cases before we demonstrate relevant applications using various Python libraries. 

There are several aspects that many of these models and their applications have in common. This chapter covers these common aspects so that we can focus on model-specific usage in the following chapters. It sets the stage by outlining how to formulate, train, tune, and evaluate the predictive performance of ML models as a systematic workflow. The content includes:

<p align="center">
<img src="https://i.imgur.com/5qisClE.png" width="65%">
</p>

- How supervised and unsupervised learning from data works
- Training and evaluating supervised learning models for regression and classification tasks
- How the bias-variance trade-off impacts predictive performance
- How to diagnose and address prediction errors due to overfitting
- Using cross-validation to optimize hyperparameters with a focus on time-series data
- Why financial data requires additional attention when testing out-of-sample

### 07 Linear Models: From Risk Factors to Return Forecasts

Linear models are standard tools for inference and prediction in regression and classification contexts. Numerous widely used asset pricing models rely on linear regression. Regularized models like Ridge and Lasso regression often yield better predictions by limiting the risk of overfitting. Typical regression applications identify risk factors that drive asset returns to manage risks or predict returns. Classification problems, on the other hand, include directional price forecasts.

<p align="center">
<img src="https://i.imgur.com/3Ph6jma.png" width="65%">
</p>

[Chapter 07](07_linear_models) covers the following topics:

- How linear regression works and which assumptions it makes
- Training and diagnosing linear regression models
- Using linear regression to predict stock returns
- Use regularization to improve the predictive performance
- How logistic regression works
- Converting a regression into a classification problem

### 08 The ML4T Workflow: From Model to Strategy Backtesting

This [chapter](08_ml4t_workflow) presents an end-to-end perspective on designing, simulating, and evaluating a trading strategy driven by an ML algorithm. 
We will demonstrate in detail how to backtest an ML-driven strategy in a historical market context using the Python libraries [backtrader](https://www.backtrader.com/) and [Zipline](https://zipline.ml4trading.io/index.html). 
The ML4T workflow ultimately aims to gather evidence from historical data that helps decide whether to deploy a candidate strategy in a live market and put financial resources at risk. A realistic simulation of your strategy needs to faithfully represent how security markets operate and how trades execute. Also, several methodological aspects require attention to avoid biased results and false discoveries that will lead to poor investment decisions.

<p align="center">
<img src="https://i.imgur.com/R9O0fn3.png" width="65%">
</p>

More specifically, after working through this chapter you will be able to:

- Plan and implement end-to-end strategy backtesting
- Understand and avoid critical pitfalls when implementing backtests
- Discuss the advantages and disadvantages of vectorized vs event-driven backtesting engines
- Identify and evaluate the key components of an event-driven backtester
- Design and execute the ML4T workflow using data sources at minute and daily frequencies, with ML models trained separately or as part of the backtest
- Use Zipline and backtrader to design and evaluate your own strategies 

### 09 Time Series Models for Volatility Forecasts and Statistical Arbitrage

This [chapter](09_time_series_models) focuses on models that extract signals from a time series' history to predict future values for the same time series. 
Time series models are in widespread use due to the time dimension inherent to trading. It presents tools to diagnose time series characteristics such as stationarity and extract features that capture potentially useful patterns. It also introduces univariate and multivariate time series models to forecast macro data and volatility patterns. 
Finally, it explains how cointegration identifies common trends across time series and shows how to develop a pairs trading strategy based on this crucial concept. 

<p align="center">
<img src="https://i.imgur.com/cglLgJ0.png" width="90%">
</p>

In particular, it covers:
- How to use time-series analysis to prepare and inform the modeling process
- Estimating and diagnosing univariate autoregressive and moving-average models
- Building autoregressive conditional heteroskedasticity (ARCH) models to predict volatility
- How to build multivariate vector autoregressive models
- Using cointegration to develop a pairs trading strategy

### 10 Bayesian ML: Dynamic Sharpe Ratios and Pairs Trading

Bayesian statistics allows us to quantify uncertainty about future events and refine estimates in a principled way as new information arrives. This dynamic approach adapts well to the evolving nature of financial markets. 
Bayesian approaches to ML enable new insights into the uncertainty around statistical metrics, parameter estimates, and predictions. The applications range from more granular risk management to dynamic updates of predictive models that incorporate changes in the market environment. 

<p align="center">
<img src="https://i.imgur.com/qOUPIDV.png" width="80%">
</p>

More specifically, this [chapter](10_bayesian_machine_learning) covers: 
- How Bayesian statistics applies to machine learning
- Probabilistic programming with PyMC3
- Defining and training machine learning models using PyMC3
- How to run state-of-the-art sampling methods to conduct approximate inference
- Bayesian ML applications to compute dynamic Sharpe ratios, dynamic pairs trading hedge ratios, and estimate stochastic volatility


### 11 Random Forests: A Long-Short Strategy for Japanese Stocks

This [chapter](11_decision_trees_random_forests) applies decision trees and random forests to trading. Decision trees learn rules from data that encode nonlinear input-output relationships. We show how to train a decision tree to make predictions for regression and classification problems, visualize and interpret the rules learned by the model, and tune the model's hyperparameters to optimize the bias-variance tradeoff and prevent overfitting.

The second part of the chapter introduces ensemble models that combine multiple decision trees in a randomized fashion to produce a single prediction with a lower error. It concludes with a long-short strategy for Japanese equities based on trading signals generated by a random forest model.

<p align="center">
<img src="https://i.imgur.com/S4s0rou.png" width="80%">
</p>

In short, this chapter covers:
- Use decision trees for regression and classification
- Gain insights from decision trees and visualize the rules learned from the data
- Understand why ensemble models tend to deliver superior results
- Use bootstrap aggregation to address the overfitting challenges of decision trees
- Train, tune, and interpret random forests
- Employ a random forest to design and evaluate a profitable trading strategy


### 12 Boosting your Trading Strategy

Gradient boosting is an alternative tree-based ensemble algorithm that often produces better results than random forests. The critical difference is that boosting modifies the data used to train each tree based on the cumulative errors made by the model. While random forests train many trees independently using random subsets of the data, boosting proceeds sequentially and reweights the data.
This [chapter](12_gradient_boosting_machines) shows how state-of-the-art libraries achieve impressive performance and apply boosting to both daily and high-frequency data to backtest an intraday trading strategy. 

<p align="center">
<img src="https://i.imgur.com/Re0uI0H.png" width="70%">
</p>

More specifically, we will cover the following topics:
- How does boosting differ from bagging, and how did gradient boosting evolve from adaptive boosting,
- Design and tune adaptive and gradient boosting models with scikit-learn,
- Build, optimize, and evaluate gradient boosting models on large datasets with the state-of-the-art implementations XGBoost, LightGBM, and CatBoost,
- Interpreting and gaining insights from gradient boosting models using [SHAP](https://github.com/slundberg/shap) values, and
- Using boosting with high-frequency data to design an intraday strategy.

### 13 Data-Driven Risk Factors and Asset Allocation with Unsupervised Learning

Dimensionality reduction and clustering are the main tasks for unsupervised learning: 
- Dimensionality reduction transforms the existing features into a new, smaller set while minimizing the loss of information. A broad range of algorithms exists that differ by how they measure the loss of information, whether they apply linear or non-linear transformations or the constraints they impose on the new feature set. 
- Clustering algorithms identify and group similar observations or features instead of identifying new features. Algorithms differ in how they define the similarity of observations and their assumptions about the resulting groups.

<p align="center">
<img src="https://i.imgur.com/Rfk7uCM.png" width="70%">
</p>

More specifically, this [chapter](13_unsupervised_learning) covers:
- How principal and independent component analysis (PCA and ICA) perform linear dimensionality reduction
- Identifying data-driven risk factors and eigenportfolios from asset returns using PCA
- Effectively visualizing nonlinear, high-dimensional data using manifold learning
- Using T-SNE and UMAP to explore high-dimensional image data
- How k-means, hierarchical, and density-based clustering algorithms work
- Using agglomerative clustering to build robust portfolios with hierarchical risk parity


## Part 3: Natural Language Processing for Trading

Text data are rich in content, yet unstructured in format and hence require more preprocessing so that a machine learning algorithm can extract the potential signal. The critical challenge consists of converting text into a numerical format for use by an algorithm, while simultaneously expressing the semantics or meaning of the content. 

The next three chapters cover several techniques that capture language nuances readily understandable to humans so that machine learning algorithms can also interpret them.

### 14 Text Data for Trading: Sentiment Analysis

Text data is very rich in content but highly unstructured so that it requires more preprocessing to enable an ML algorithm to extract relevant information. A key challenge consists of converting text into a numerical format without losing its meaning.
This [chapter](14_working_with_text_data) shows how to represent documents as vectors of token counts by creating a document-term matrix that, in turn, serves as input for text classification and sentiment analysis. It also introduces the Naive Bayes algorithm and compares its performance to linear and tree-based models.

In particular, in this chapter covers:
- What the fundamental NLP workflow looks like
- How to build a multilingual feature extraction pipeline using spaCy and TextBlob
- Performing NLP tasks like part-of-speech tagging or named entity recognition
- Converting tokens to numbers using the document-term matrix
- Classifying news using the naive Bayes model
- How to perform sentiment analysis using different ML algorithms

### 15 Topic Modeling: Summarizing Financial News

This [chapter](15_topic_modeling) uses unsupervised learning to model latent topics and extract hidden themes from documents. These themes can generate detailed insights into a large corpus of financial reports.
Topic models automate the creation of sophisticated, interpretable text features that, in turn, can help extract trading signals from extensive collections of texts. They speed up document review, enable the clustering of similar documents, and produce annotations useful for predictive modeling.
Applications include identifying critical themes in company disclosures, earnings call transcripts or contracts, and annotation based on sentiment analysis or using returns of related assets. 

<p align="center">
<img src="https://i.imgur.com/VVSnTCa.png" width="60%">
</p>

More specifically, it covers:
- How topic modeling has evolved, what it achieves, and why it matters
- Reducing the dimensionality of the DTM using latent semantic indexing
- Extracting topics with probabilistic latent semantic analysis (pLSA)
- How latent Dirichlet allocation (LDA) improves pLSA to become the most popular topic model
- Visualizing and evaluating topic modeling results -
- Running LDA using scikit-learn and gensim
- How to apply topic modeling to collections of earnings calls and financial news articles

### 16 Word embeddings for Earnings Calls and SEC Filings

This [chapter](16_word_embeddings) uses neural networks to learn a vector representation of individual semantic units like a word or a paragraph. These vectors are dense with a few hundred real-valued entries, compared to the higher-dimensional sparse vectors of the bag-of-words model. As a result, these vectors embed or locate each semantic unit in a continuous vector space.

Embeddings result from training a model to relate tokens to their context with the benefit that similar usage implies a similar vector. As a result, they encode semantic aspects like relationships among words through their relative location. They are powerful features that we will use with deep learning models in the following chapters.

<p align="center">
<img src="https://i.imgur.com/v8w9XLL.png" width="80%">
</p>

 More specifically, in this chapter, we will cover:
- What word embeddings are and how they capture semantic information
- How to obtain and use pre-trained word vectors
- Which network architectures are most effective at training word2vec models
- How to train a word2vec model using TensorFlow and gensim
- Visualizing and evaluating the quality of word vectors
- How to train a word2vec model on SEC filings to predict stock price moves
- How doc2vec extends word2vec and helps with sentiment analysis
- Why the transformerâ€™s attention mechanism had such an impact on NLP
- How to fine-tune pre-trained BERT models on financial data

## Part 4: Deep & Reinforcement Learning

Part four explains and demonstrates how to leverage deep learning for algorithmic trading. 
The powerful capabilities of deep learning algorithms to identify patterns in unstructured data make it particularly suitable for alternative data like images and text. 

The sample applications show, for exapmle, how to combine text and price data to predict earnings surprises from SEC filings, generate synthetic time series to expand the amount of training data, and train a trading agent using deep reinforcement learning.
Several of these applications replicate research recently published in top journals.

### 17 Deep Learning for Trading

This [chapter](17_deep_learning) presents feedforward neural networks (NN) and demonstrates how to efficiently train large models using backpropagation while managing the risks of overfitting. It also shows how to use TensorFlow 2.0 and PyTorch and how to optimize a NN architecture to generate trading signals.
In the following chapters, we will build on this foundation to apply various architectures to different investment applications with a focus on alternative data. These include recurrent NN tailored to sequential data like time series or natural language and convolutional NN, particularly well suited to image data. We will also cover deep unsupervised learning, such as how to create synthetic data using Generative Adversarial Networks (GAN). Moreover, we will discuss reinforcement learning to train agents that interactively learn from their environment.

<p align="center">
<img src="https://i.imgur.com/5cet0Fi.png" width="70%">
</p>

In particular, this chapter will cover
- How DL solves AI challenges in complex domains
- Key innovations that have propelled DL to its current popularity
- How feedforward networks learn representations from data
- Designing and training deep neural networks (NNs) in Python
- Implementing deep NNs using Keras, TensorFlow, and PyTorch
- Building and tuning a deep NN to predict asset returns
- Designing and backtesting a trading strategy based on deep NN signals

### 18 CNN for Financial Time Series and Satellite Images

CNN architectures continue to evolve. This chapter describes building blocks common to successful applications, demonstrates how transfer learning can speed up learning, and how to use CNNs for object detection.
CNNs can generate trading signals from images or time-series data. Satellite data can anticipate commodity trends via aerial images of agricultural areas, mines, or transport networks. Camera footage can help predict consumer activity; we show how to build a CNN that classifies economic activity in satellite images.
CNNs can also deliver high-quality time-series classification results by exploiting their structural similarity with images, and we design a strategy based on time-series data formatted like images. 

<p align="center">
<img src="https://i.imgur.com/PlLQV0M.png" width="60%">
</p>

More specifically, this [chapter](18_convolutional_neural_nets) covers:

- How CNNs employ several building blocks to efficiently model grid-like data
- Training, tuning and regularizing CNNs for images and time series data using TensorFlow
- Using transfer learning to streamline CNNs, even with fewer data
- Designing a trading strategy using return predictions by a CNN trained on time-series data formatted like images
- How to classify economic activity based on satellite images

### 19 RNN for Multivariate Time Series and Sentiment Analysis

Recurrent neural networks (RNNs) compute each output as a function of the previous output and new data, effectively creating a model with memory that shares parameters across a deeper computational graph. Prominent architectures include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) that address the challenges of learning long-range dependencies.
RNNs are designed to map one or more input sequences to one or more output sequences and are particularly well suited to natural language. They can also be applied to univariate and multivariate time series to predict market or fundamental data. This chapter covers how RNN can model alternative text data using the word embeddings that we covered in Chapter 16 to classify the sentiment expressed in documents.

<p align="center">
<img src="https://i.imgur.com/E9fOApg.png" width="60%">
</p>

More specifically, this chapter addresses:
- How recurrent connections allow RNNs to memorize patterns and model a hidden state
- Unrolling and analyzing the computational graph of RNNs
- How gated units learn to regulate RNN memory from data to enable long-range dependencies
- Designing and training RNNs for univariate and multivariate time series in Python
- How to learn word embeddings or use pretrained word vectors for sentiment analysis with RNNs
- Building a bidirectional RNN to predict stock returns using custom word embeddings

### 20 Autoencoders for Conditional Risk Factors and Asset Pricing

This [chapter](20_autoencoders_for_conditional_risk_factors) shows how to leverage unsupervised deep learning for trading. We also discuss autoencoders, namely, a neural network trained to reproduce the input while learning a new representation encoded by the parameters of a hidden layer. Autoencoders have long been used for nonlinear dimensionality reduction, leveraging the NN architectures we covered in the last three chapters.
We replicate a recent AQR paper that shows how autoencoders can underpin a trading strategy. We will use a deep neural network that relies on an autoencoder to extract risk factors and predict equity returns, conditioned on a range of equity attributes.

<p align="center">
<img src="https://i.imgur.com/aCmE0UD.png" width="60%">
</p>

More specifically, in this chapter you will learn about:
- Which types of autoencoders are of practical use and how they work
- Building and training autoencoders using Python
- Using autoencoders to extract data-driven risk factors that take into account asset characteristics to predict returns

### 21 Generative Adversarial Nets for Synthetic Time Series Data

This chapter introduces generative adversarial networks (GAN). GANs train a generator and a discriminator network in a competitive setting so that the generator learns to produce samples that the discriminator cannot distinguish from a given class of training data. The goal is to yield a generative model capable of producing synthetic samples representative of this class.
While most popular with image data, GANs have also been used to generate synthetic time-series data in the medical domain. Subsequent experiments with financial data explored whether GANs can produce alternative price trajectories useful for ML training or strategy backtests. We replicate the 2019 NeurIPS Time-Series GAN paper to illustrate the approach and demonstrate the results.

<p align="center">
<img src="https://i.imgur.com/W1Rp89K.png" width="60%">
</p>

More specifically, in this chapter you will learn about:
- How GANs work, why they are useful, and how they could be applied to trading
- Designing and training GANs using TensorFlow 2
- Generating synthetic financial data to expand the inputs available for training ML models and backtesting

### 22 Deep Reinforcement Learning: Building a Trading Agent

Reinforcement Learning (RL) models goal-directed learning by an agent that interacts with a stochastic environment. RL optimizes the agent's decisions concerning a long-term objective by learning the value of states and actions from a reward signal. The ultimate goal is to derive a policy that encodes behavioral rules and maps states to actions.
This [chapter](22_deep_reinforcement_learning) shows how to formulate and solve an RL problem. It covers model-based and model-free methods, introduces the OpenAI Gym environment, and combines deep learning with RL to train an agent that navigates a complex environment. Finally, we'll show you how to adapt RL to algorithmic trading by modeling an agent that interacts with the financial market while trying to optimize an objective function.

<p align="center">
<img src="https://i.imgur.com/lg0ofbZ.png" width="60%">
</p>

More specifically,this chapter will cover:

- Define a Markov decision problem (MDP)
- Use value and policy iteration to solve an MDP
- Apply Q-learning in an environment with discrete states and actions
- Build and train a deep Q-learning agent in a continuous environment
- Use the OpenAI Gym to design a custom market environment and train an RL agent to trade stocks

### 23 Conclusions and Next Steps

In this concluding chapter, we will briefly summarize the essential tools, applications, and lessons learned throughout the book to avoid losing sight of the big picture after so much detail.
We will then identify areas that we did not cover but would be worth focusing on as you expand on the many machine learning techniques we introduced and become productive in their daily use.

In sum, in this chapter, we will
- Review key takeaways and lessons learned
- Point out the next steps to build on the techniques in this book
- Suggest ways to incorporate ML into your investment process

### 24 Appendix - Alpha Factor Library

Throughout this book, we emphasized how the smart design of features, including appropriate preprocessing and denoising, typically leads to an effective strategy. This appendix synthesizes some of the lessons learned on feature engineering and provides additional information on this vital topic.

To this end, we focus on the broad range of indicators implemented by TA-Lib (see [Chapter 4](04_alpha_factor_research)) and WorldQuant's [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) paper (Kakushadze 2016), which presents real-life quantitative trading factors used in production with an average holding period of 0.6-6.4 days.

This chapter covers: 
- How to compute several dozen technical indicators using TA-Lib and NumPy/pandas,
- Creating the formulaic alphas describe in the above paper, and
- Evaluating the predictive quality of the results using various metrics from rank correlation and mutual information to feature importance, SHAP values and Alphalens. 


---

## Chapter Details

### 04 - Alpha Factor Research
# Financial Feature Engineering: How to research Alpha Factors

Algorithmic trading strategies are driven by signals that indicate when to buy or sell assets to generate superior returns relative to a benchmark such as an index. The portion of an asset's return that is not explained by exposure to this benchmark is called alpha, and hence the signals that aim to produce such uncorrelated returns are also called alpha factors.

If you are already familiar with ML, you may know that feature engineering is a key ingredient for successful predictions. This is no different in trading. Investment, however, is particularly rich in decades of research into how markets work and which features may work better than others to explain or predict price movements as a result. This chapter provides an overview as a starting point for your own search for alpha factors.

This chapter also presents key tools that facilitate the computing and testing alpha factors. We will highlight how the NumPy, pandas and TA-Lib libraries facilitate the manipulation of data and present popular smoothing techniques like the wavelets and the Kalman filter that help reduce noise in data.

We also preview how you can use the trading simulator Zipline to evaluate the predictive performance of (traditional) alpha factors. We discuss key alpha factor metrics like the information coefficient and factor turnover. An in-depth introduction to backtesting trading strategies that use machine learning follows in [Chapter 6](../08_ml4t_workflow), which covers the **ML4T workflow** that we will use throughout the book to evaluate trading strategies. 

Please see the [Appendix - Alpha Factor Library](../24_alpha_factor_library) for additional material on this topic, including numerous code examples that compute a broad range of alpha factors.

## Content

1. [Alpha Factors in practice: from data to signals](#alpha-factors-in-practice-from-data-to-signals)
2. [Building on Decades of Factor Research](#building-on-decades-of-factor-research)
    * [References](#references)
3. [Engineering alpha factors that predict returns](#engineering-alpha-factors-that-predict-returns)
    * [Code Example: How to engineer factors using pandas and NumPy](#code-example-how-to-engineer-factors-using-pandas-and-numpy)
    * [Code Example: How to use TA-Lib to create technical alpha factors](#code-example-how-to-use-ta-lib-to-create-technical-alpha-factors)
    * [Code Example: How to denoise your Alpha Factors with the Kalman Filter](#code-example-how-to-denoise-your-alpha-factors-with-the-kalman-filter)
    * [Code Example: How to preprocess your noisy signals using Wavelets](#code-example-how-to-preprocess-your-noisy-signals-using-wavelets)
    * [Resources](#resources)
4. [From signals to trades: backtesting with `Zipline`](#from-signals-to-trades-backtesting-with-zipline)
    * [Code Example: How to use Zipline to backtest a single-factor strategy](#code-example-how-to-use-zipline-to-backtest-a-single-factor-strategy)
    * [Code Example: Combining factors from diverse data sources on the Quantopian platform](#code-example-combining-factors-from-diverse-data-sources-on-the-quantopian-platform)
    * [Code Example: Separating signal and noise â€“ how to use alphalens](#code-example-separating-signal-and-noise--how-to-use-alphalens)
5. [Alternative Algorithmic Trading Libraries and Platforms](#alternative-algorithmic-trading-libraries-and-platforms)

## Alpha Factors in practice: from data to signals

Alpha factors are transformations of market, fundamental, and alternative data that contain predictive signals. They are designed to capture risks that drive asset returns. One set of factors describes fundamental, economy-wide variables such as growth, inflation, volatility, productivity, and demographic risk. Another set consists of tradeable investment styles such as the market portfolio, value-growth investing, and momentum investing.

There are also factors that explain price movements based on the economics or institutional setting of financial markets, or investor behavior, including known biases of this behavior. The economic theory behind factors can be rational, where the factors have high returns over the long run to compensate for their low returns during bad times, or behavioral, where factor risk premiums result from the possibly biased, or not entirely rational behavior of agents that is not arbitraged away.

## Building on Decades of Factor Research

In an idealized world, categories of risk factors should be independent of each other (orthogonal), yield positive risk premia, and form a complete set that spans all dimensions of risk and explains the systematic risks for assets in a given class. In practice, these requirements will hold only approximately.

### References

- [Dissecting Anomalies](http://schwert.ssb.rochester.edu/f532/ff_JF08.pdf) by Eugene Fama and Ken French (2008)
- [Explaining Stock Returns: A Literature Review](https://www.ifa.com/pdfs/explainingstockreturns.pdf) by James L. Davis (2001)
- [Market Efficiency, Long-Term Returns, and Behavioral Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=15108) by Eugene Fama (1997)
- [The Efficient Market Hypothesis and It's Critics](https://pubs.aeaweb.org/doi/pdf/10.1257/089533003321164958) by Burton Malkiel (2003)
- [The New Palgrave Dictionary of Economics](https://www.palgrave.com/us/book/9780333786765) (2008) by Steven Durlauf and Lawrence Blume, 2nd ed.
- [Anomalies and Market Efficiency](https://www.nber.org/papers/w9277.pdf) by G. William Schwert25 (Ch. 15 in Handbook of the- "Economics of Finance", by Constantinides, Harris, and Stulz, 2003)
- [Investor Psychology and Asset Pricing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=265132), by David Hirshleifer (2001)
- [Practical advice for analysis of large, complex data sets](https://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html), Patrick Riley, Unofficial Google Data Science Blog

## Engineering alpha factors that predict returns

Based on a conceptual understanding of key factor categories, their rationale and popular metrics, a key task is to identify new factors that may better capture the risks embodied by the return drivers laid out previously, or to find new ones. In either case, it will be important to compare the performance of innovative factors to that of known factors to identify incremental signal gains.

### Code Example: How to engineer factors using pandas and NumPy

The notebook [feature_engineering.ipynb](00_data/feature_engineering.ipynb) in the [data](00_data) directory illustrates how to engineer basic factors.

### Code Example: How to use TA-Lib to create technical alpha factors

The notebook [how_to_use_talib](02_how_to_use_talib.ipynb) illustrates the usage of TA-Lib, which includes a broad range of common technical indicators. These indicators have in common that they only use market data, i.e., price and volume information.

The notebook [common_alpha_factors](../24_alpha_factor_library/02_common_alpha_factors.ipynb) in th **appendix** contains dozens of additional examples.  

### Code Example: How to denoise your Alpha Factors with the Kalman Filter

The notebook [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) demonstrates the use of the Kalman filter using the `PyKalman` package for smoothing; we will also use it in [Chapter 9](../09_time_series_models) when we develop a pairs trading strategy.

### Code Example: How to preprocess your noisy signals using Wavelets

The notebook [kalman_filter_and_wavelets](03_kalman_filter_and_wavelets.ipynb) also demonstrates how to work with wavelets using the `PyWavelets` package.

### Resources

- [Fama French](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) Data Library
- [numpy](https://numpy.org/) website
    - [Quickstart Tutorial](https://numpy.org/devdocs/user/quickstart.html)
- [pandas](https://pandas.pydata.org/) website
    - [User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
    - [10 minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
    - [Python Pandas Tutorial: A Complete Introduction for Beginners](https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/)
- [alphatools](https://github.com/marketneutral/alphatools) - Quantitative finance research tools in Python
- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) - Package based on the work of Dr Marcos Lopez de Prado regarding his research with respect to Advances in Financial Machine Learning
- [PyKalman](https://pykalman.github.io/) documentation
- [Tutorial: The Kalman Filter](http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf)
- [Understanding and Applying Kalman Filtering](http://biorobotics.ri.cmu.edu/papers/sbp_papers/integrated3/kleeman_kalman_basics.pdf)
- [How a Kalman filter works, in pictures](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
- [PyWavelets](https://pywavelets.readthedocs.io/en/latest/) - Wavelet Transforms in Python
- [An Introduction to Wavelets](https://www.eecis.udel.edu/~amer/CISC651/IEEEwavelet.pdf) 
- [The Wavelet Tutorial](http://web.iitd.ac.in/~sumeet/WaveletTutorial.pdf)
- [Wavelets for Kids](http://www.gtwavelet.bme.gatech.edu/wp/kidsA.pdf)
- [The Barra Equity Risk Model Handbook](https://www.alacra.com/alacra/help/barra_handbook_GEM.pdf)
- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [Modern Investment Management: An Equilibrium Approach](https://www.amazon.com/Modern-Investment-Management-Equilibrium-Approach/dp/0471124109) by Bob Litterman, 2003
- [Quantitative Equity Portfolio Management: Modern Techniques and Applications](https://www.crcpress.com/Quantitative-Equity-Portfolio-Management-Modern-Techniques-and-Applications/Qian-Hua-Sorensen/p/book/9781584885580) by Edward Qian, Ronald Hua, and Eric Sorensen
- [Spearman Rank Correlation](https://statistics.laerd.com/statistical-guides/spearmans-rank-order-correlation-statistical-guide.php)

## From signals to trades: backtesting with `Zipline`

The open source [zipline](https://zipline.ml4trading.io/index.html) library is an event-driven backtesting system maintained and used in production by the crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) to facilitate algorithm-development and live-trading. It automates the algorithm's reaction to trade events and provides it with current and historical point-in-time data that avoids look-ahead bias.

- [Chapter 8](../08_ml4t_workflow) contains a more comprehensive introduction to Zipline.
- Please follow the [instructions](../installation) in the `installation` folder, including to address **know issues**.

### Code Example: How to use Zipline to backtest a single-factor strategy

The notebook [single_factor_zipline](04_single_factor_zipline.ipynb) develops and test a simple mean-reversion factor that measures how much recent performance has deviated from the historical average. Short-term reversal is a common strategy that takes advantage of the weakly predictive pattern that stock price increases are likely to mean-revert back down over horizons from less than a minute to one month.

### Code Example: Combining factors from diverse data sources on the Quantopian platform

The Quantopian research environment is tailored to the rapid testing of predictive alpha factors. The process is very similar because it builds on `zipline`, but offers much richer access to data sources. 

The notebook [multiple_factors_quantopian_research](05_multiple_factors_quantopian_research.ipynb) illustrates how to compute alpha factors not only from market data as previously but also from fundamental and alternative data.
    
### Code Example: Separating signal and noise â€“ how to use alphalens

The notebook [performance_eval_alphalens](06_performance_eval_alphalens.ipynb) introduces the [alphalens](http://quantopian.github.io/alphalens/) library for the performance analysis of predictive (alpha) factors, open-sourced by Quantopian. It demonstrates how it integrates with the backtesting library `zipline` and the portfolio performance and risk analysis library `pyfolio` that we will explore in the next chapter.

`alphalens` facilitates the analysis of the predictive power of alpha factors concerning the:
- Correlation of the signals with subsequent returns
- Profitability of an equal or factor-weighted portfolio based on a (subset of) the signals
- Turnover of factors to indicate the potential trading costs
- Factor-performance during specific events
- Breakdowns of the preceding by sector

The analysis can be conducted using `tearsheets` or individual computations and plots. The tearsheets are illustrated in the online repo to save some space.

- See [here](https://github.com/quantopian/alphalens/blob/master/alphalens/examples/alphalens_tutorial_on_quantopian.ipynb) for a detailed `alphalens` tutorial by Quantopian

## Alternative Algorithmic Trading Libraries and Platforms

- [QuantConnect](https://www.quantconnect.com/)
- [Alpha Trading Labs](https://www.alphalabshft.com/)
    - Alpha Trading Labs is no longer active
- [WorldQuant](https://www.worldquantvrc.com/en/cms/wqc/home/)
- Python Algorithmic Trading Library [PyAlgoTrade](http://gbeced.github.io/pyalgotrade/)
- [pybacktest](https://github.com/ematvey/pybacktest)
- [Trading with Python](http://www.tradingwithpython.com/)
- [Interactive Brokers](https://www.interactivebrokers.com/en/index.php?f=5041)


### 05 - Strategy Evaluation  
# Portfolio Optimization and Performance Evaluation

To test a strategy prior to implementation under market conditions, we need to simulate the trades that the algorithm would make and verify their performance. Strategy evaluation includes backtesting against historical data to optimize the strategy's parameters and forward-testing to validate the in-sample performance against new, out-of-sample data. The goal is to avoid false discoveries from tailoring a strategy to specific past circumstances.

In a portfolio context, positive asset returns can offset negative price movements. Positive price changes for one asset are more likely to offset losses on another the lower the correlation between the two positions. Based on how portfolio risk depends on the positionsâ€™ covariance, Harry Markowitz developed the theory behind modern portfolio management based on diversification in 1952. The result is mean-variance optimization that selects weights for a given set of assets to minimize risk, measured as the standard deviation of returns for a given expected return.

The capital asset pricing model (CAPM) introduces a risk premium, measured as the expected return in excess of a risk-free investment, as an equilibrium reward for holding an asset. This reward compensates for the exposure to a single risk factorâ€”the marketâ€”that is systematic as opposed to idiosyncratic to the asset and thus cannot be diversified away. 

Risk management has evolved to become more sophisticated as additional risk factors and more granular choices for exposure have emerged. The Kelly criterion is a popular approach to dynamic portfolio optimization, which is the choice of a sequence of positions over time; it has been famously adapted from its original application in gambling to the stock market by Edward Thorp in 1968.

As a result, there are several approaches to optimize portfolios that include the application of machine learning (ML) to learn hierarchical relationships among assets and treat their holdings as complements or substitutes with respect to the portfolio risk profile. This chapter will cover the following topics:

## Content

1. [How to measure portfolio performance](#how-to-measure-portfolio-performance)
    * [The (adjusted) Sharpe Ratio](#the-adjusted-sharpe-ratio)
    * [The fundamental law of active management](#the-fundamental-law-of-active-management)
2. [How to manage Portfolio Risk & Return](#how-to-manage-portfolio-risk--return)
    * [The evolution of modern portfolio management](#the-evolution-of-modern-portfolio-management)
    * [Mean-variance optimization](#mean-variance-optimization)
        - [Code Examples: Finding the efficient frontier in Python](#code-examples-finding-the-efficient-frontier-in-python)
    * [Alternatives to mean-variance optimization](#alternatives-to-mean-variance-optimization)
        - [The 1/N portfolio](#the-1n-portfolio)
        - [The minimum-variance portfolio](#the-minimum-variance-portfolio)
        - [The Black-Litterman approach](#the-black-litterman-approach)
        - [How to size your bets â€“ the Kelly rule](#how-to-size-your-bets--the-kelly-rule)
        - [Alternatives to MV Optimization with Python](#alternatives-to-mv-optimization-with-python)
    * [Hierarchical Risk Parity](#hierarchical-risk-parity)
3. [Trading and managing a portfolio with `Zipline`](#trading-and-managing-a-portfolio-with-zipline)
    * [Code Examples: Backtests with trades and portfolio optimization ](#code-examples-backtests-with-trades-and-portfolio-optimization-)
4. [Measure backtest performance with `pyfolio`](#measure-backtest-performance-with-pyfolio)
    * [Code Example: `pyfolio` evaluation from a `Zipline` backtest](#code-example-pyfolio-evaluation-from-a-zipline-backtest)

## How to measure portfolio performance

To evaluate and compare different strategies or to improve an existing strategy, we need metrics that reflect their performance with respect to our objectives. In investment and trading, the most common objectives are the **return and the risk of the investment portfolio**.

The return and risk objectives imply a trade-off: taking more risk may yield higher returns in some circumstances, but also implies greater downside. To compare how different strategies navigate this trade-off, ratios that compute a measure of return per unit of risk are very popular. Weâ€™ll discuss the **Sharpe ratio** and the **information ratio** (IR) in turn.

### The (adjusted) Sharpe Ratio

The ex-ante Sharpe Ratio (SR) compares the portfolio's expected excess portfolio to the volatility of this excess return, measured by its standard deviation. It measures the compensation as the average excess return per unit of risk taken. It can be estimated from data.

Financial returns often violate the iid assumptions. Andrew Lo has derived the necessary adjustments to the distribution and the time aggregation for returns that are stationary but autocorrelated. This is important because the time-series properties of investment strategies (for example, mean reversion, momentum, and other forms of serial correlation) can have a non-trivial impact on the SR estimator itself, especially when annualizing the SR from higher-frequency data.

- [The Statistics of Sharpe Ratios](https://www.jstor.org/stable/4480405?seq=1#page_scan_tab_contents), Andrew Lo, Financial Analysts Journal, 2002

### The fundamental law of active management

Itâ€™s a curious fact that Renaissance Technologies (RenTec), the top-performing quant fund founded by Jim Simons that we mentioned in [Chapter 1](../01_machine_learning_for_trading), has produced similar returns as Warren Buffet despite extremely different approaches. Warren Buffetâ€™s investment firm Berkshire Hathaway holds some 100-150 stocks for fairly long periods, whereas RenTec may execute 100,000 trades per day. How can we compare these distinct strategies?

ML is about optimizing objective functions. In algorithmic trading, the objectives are the return and the risk of the overall investment portfolio, typically relative to a benchmark (which may be cash, the risk-free interest rate, or an asset price index like the S&P 500).

A high Information Ratio (IR) implies attractive out-performance relative to the additional risk taken. The Fundamental Law of Active Management breaks the IR down into the information coefficient (IC) as a measure of forecasting skill, and the ability to apply this skill through independent bets. It summarizes the importance to play both often (high breadth) and to play well (high IC).

The IC measures the correlation between an alpha factor and the forward returns resulting from its signals and captures the accuracy of a manager's forecasting skills. The breadth of the strategy is measured by the independent number of bets an investor makes in a given time period, and the product of both values is proportional to the IR, also known as appraisal risk (Treynor and Black).

The fundamental law is important because it highlights the key drivers of outperformance: both accurate predictions and the ability to make independent forecasts and act on these forecasts matter. In practice, estimating the breadth of a strategy is difficult given the cross-sectional and time-series correlation among forecasts. 

- [Active Portfolio Management: A Quantitative Approach for Producing Superior Returns and Controlling Risk](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) by Richard Grinold and Ronald Kahn, 1999
- [How to Use Security Analysis to Improve Portfolio Selection](https://econpapers.repec.org/article/ucpjnlbus/v_3a46_3ay_3a1973_3ai_3a1_3ap_3a66-86.htm), Jack L Treynor and Fischer Black, Journal of Business, 1973
- [Portfolio Constraints and the Fundamental Law of Active Management](https://faculty.fuqua.duke.edu/~charvey/Teaching/BA491_2005/Transfer_coefficient.pdf), Clarke et al 2002

## How to manage Portfolio Risk & Return

Portfolio management aims to pick and size positions in financial instruments that achieve the desired risk-return trade-off regarding a benchmark. As a portfolio manager, in each period, you select positions that optimize diversification to reduce risks while achieving a target return. Across periods, these positions may require rebalancing to account for changes in weights resulting from price movements to achieve or maintain a target risk profile.

### The evolution of modern portfolio management

Diversification permits us to reduce risks for a given expected return by exploiting how imperfect correlation allows for one asset's gains to make up for another asset's losses. Harry Markowitz invented modern portfolio theory (MPT) in 1952 and provided the mathematical tools to optimize diversification by choosing appropriate portfolio weights.
 
### Mean-variance optimization

Modern portfolio theory solves for the optimal portfolio weights to minimize volatility for a given expected return, or maximize returns for a given level of volatility. The key requisite inputs are expected asset returns, standard deviations, and the covariance matrix. 
- [Portfolio Selection](https://www.math.ust.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf), Harry Markowitz, The Journal of Finance, 1952
- [The Capital Asset Pricing Model: Theory and Evidence](http://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S6B-FamaFrench-CAPM-JEP04.pdf), Eugene F. Fama and Kenneth R. French, Journal of Economic Perspectives, 2004

#### Code Examples: Finding the efficient frontier in Python

We can calculate an efficient frontier using scipy.optimize.minimize and the historical estimates for asset returns, standard deviations, and the covariance matrix. 
- The notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb) to compute the efficient frontier in python.

### Alternatives to mean-variance optimization

The challenges with accurate inputs for the mean-variance optimization problem have led to the adoption of several practical alternatives that constrain the mean, the variance, or both, or omit return estimates that are more challenging, such as the risk parity approach that we discuss later in this section.

#### The 1/N portfolio

Simple portfolios provide useful benchmarks to gauge the added value of complex models that generate the risk of overfitting. The simplest strategyâ€”an equally-weighted portfolioâ€”has been shown to be one of the best performers.

#### The minimum-variance portfolio

Another alternative is the global minimum-variance (GMV) portfolio, which prioritizes the minimization of risk. It is shown in the efficient frontier figure and can be calculated as follows by minimizing the portfolio standard deviation using the mean-variance framework.

#### The Black-Litterman approach

The Global Portfolio Optimization approach of Black and Litterman (1992) combines economic models with statistical learning and is popular because it generates estimates of expected returns that are plausible in many situations.
The technique assumes that the market is a mean-variance portfolio as implied by the CAPM equilibrium model. It builds on the fact that the observed market capitalization can be considered as optimal weights assigned to each security by the market. Market weights reflect market prices that, in turn, embody the marketâ€™s expectations of future returns.

- [Global Portfolio Optimization](http://www.sef.hku.hk/tpg/econ6017/2011/black-litterman-1992.pdf), Black, Fischer; Litterman, Robert
Financial Analysts Journal, 1992

#### How to size your bets â€“ the Kelly rule

The Kelly rule has a long history in gambling because it provides guidance on how much to stake on each of an (infinite) sequence of bets with varying (but favorable) odds to maximize terminal wealth. It was published as A New Interpretation of the Information Rate in 1956 by John Kelly who was a colleague of Claude Shannon's at Bell Labs. He was intrigued by bets placed on candidates at the new quiz show The $64,000 Question, where a viewer on the west coast used the three-hour delay to obtain insider information about the winners. 

Kelly drew a connection to Shannon's information theory to solve for the bet that is optimal for long-term capital growth when the odds are favorable, but uncertainty remains. His rule maximizes logarithmic wealth as a function of the odds of success of each game, and includes implicit bankruptcy protection since log(0) is negative infinity so that a Kelly gambler would naturally avoid losing everything.

- [A New Interpretation of Information Rate](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf), John Kelly, 1956
- [Beat the Dealer: A Winning Strategy for the Game of Twenty-One](https://www.amazon.com/Beat-Dealer-Winning-Strategy-Twenty-One/dp/0394703103), Edward O. Thorp,1966
- [Beat the Market: A Scientific Stock Market System](https://www.researchgate.net/publication/275756748_Beat_the_Market_A_Scientific_Stock_Market_System) , Edward O. Thorp,1967
- [Quantitative Trading: How to Build Your Own Algorithmic Trading Business](https://www.amazon.com/Quantitative-Trading-Build-Algorithmic-Business/dp/0470284889/ref=sr_1_2?s=books&ie=UTF8&qid=1545525861&sr=1-2), Ernie Chan, 2008

#### Alternatives to MV Optimization with Python

- The notebook [kelly_rule](05_kelly_rule.ipynb) demonstrates the application for the single and multiple asset case. 
- The latter result is also included in the notebook [mean_variance_optimization](04_mean_variance_optimization.ipynb), along with several other alternative approaches.

### Hierarchical Risk Parity

This novel approach developed by [Marcos Lopez de Prado](http://www.quantresearch.org/) aims to address three major concerns of quadratic optimizers, in general, and Markowitzâ€™s critical line algorithm (CLA), in particular: 
- instability, 
- concentration, and 
- underperformance. 

Hierarchical Risk Parity (HRP) applies graph theory and machine-learning to build a diversified portfolio based on the information contained in the covariance matrix. However, unlike quadratic optimizers, HRP does not require the invertibility of the covariance matrix. In fact, HRP can compute a portfolio on an ill-degenerated or even a singular covariance matrixâ€”an impossible feat for quadratic optimizers. Monte Carlo experiments show that HRP delivers lower out-of-sample variance than CLA, even though minimum variance is CLAâ€™s optimization objective. HRP also produces less risky portfolios out of sample compared to traditional risk parity methods. We will discuss HRP in more detail in [Chapter 13](../13_unsupervised_learning) when we discuss applications of unsupervised learning, including hierarchical clustering, to trading.

- [Building diversified portfolios that outperform out of sample](https://jpm.pm-research.com/content/42/4/59.short), Marcos LÃ³pez de Prado, The Journal of Portfolio Management 42, no. 4 (2016): 59-69.
- [Hierarchical Clustering Based Asset Allocation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2840729), Thomas Raffinot, 2016

We demonstrate how to implement HRP and compare it to alternatives in Chapter 13 on [Unsupervised Learning](../13_unsupervised_learning) where we also introduce hierarchical clustering.

## Trading and managing a portfolio with `Zipline`

The open source [zipline](https://zipline.ml4trading.io/index.html) library is an event-driven backtesting system maintained and used in production by the crowd-sourced quantitative investment fund [Quantopian](https://www.quantopian.com/) to facilitate algorithm-development and live-trading. It automates the algorithm's reaction to trade events and provides it with current and historical point-in-time data that avoids look-ahead bias. [Chapter 8 - The ML4T Workflow](../08_strategy_workflow) has a more detailed, dedicated introduction to backtesting using both `zipline` and `backtrader`. 

In [Chapter 4](../04_alpha_factor_research), we introduced `zipline` to simulate the computation of alpha factors from trailing cross-sectional market, fundamental, and alternative data. Now we will exploit the alpha factors to derive and act on buy and sell signals. 

### Code Examples: Backtests with trades and portfolio optimization 

The code for this section lives in the following two notebooks: 
- The notebooks in this section use the `conda` environment `backtest`. Please see the installation [instructions](../installation/README.md) for downloading the latest Docker image or alternative ways to set up your environment.
- The notebook [backtest_with_trades](01_backtest_with_trades.ipynb) simulates the trading decisions that build a portfolio based on the simple MeanReversion alpha factor from the last chapter using Zipline. We not explicitly optimize the portfolio weights and just assign positions of equal value to each holding.
- The notebook [backtest_with_pf_optimization](02_backtest_with_pf_optimization.ipynb) demonstrates how to use PF optimization as part of a simple strategy backtest. 

## Measure backtest performance with `pyfolio`

Pyfolio facilitates the analysis of portfolio performance and risk in-sample and out-of-sample using many standard metrics. It produces tear sheets covering the analysis of returns, positions, and transactions, as well as event risk during periods of market stress using several built-in scenarios, and also includes Bayesian out-of-sample performance analysis.

### Code Example: `pyfolio` evaluation from a `Zipline` backtest

The notebook [pyfolio_demo](03_pyfolio_demo.ipynb) illustrates how to extract the `pyfolio` input from the backtest conducted in the previous folder. It then proceeds to calculate several performance metrics and tear sheets using `pyfolio`

- This notebook requires the `conda` environment `backtest`. Please see the [installation instructions](../installation/README.md) for running the latest Docker image or alternative ways to set up your environment.

### 09 - Time Series Models
# From Volatility Forecasts to Statistical Arbitrage: Linear Time Series Models

In this chapter, we will build dynamic linear models to explicitly represent time and include variables observed at specific intervals or lags. A key characteristic of time-series data is their sequential order: rather than random samples of individual observations as in the case of cross-sectional data, our data are a single realization of a stochastic process that we cannot repeat.

Our goal is to identify **systematic patterns in time series** that help us predict how the time series will behave in the future. More specifically, we focus on models that extract signals from a historical sequence of the output and, optionally, other contemporaneous or lagged input variables to predict future values of the output. For example, we might try to predict future returns for a stock using past returns, combined with historical returns of a benchmark or macroeconomic variables. We focus on linear time-series models before turning to nonlinear models like recurrent or convolutional neural networks in Part 4. 

Time-series models are very popular given the time dimension inherent to trading. Key applications include the **prediction of asset returns and volatility**, as well as the identification of co-movements of asset price series. Time-series data are likely to become more prevalent as an ever-broader array of connected devices collects regular measurements with potential signal content.

We first introduce tools to diagnose time-series characteristics and to extract features that capture potential patterns. Then we introduce univariate and multivariate time-series models and apply them to forecast macro data and volatility patterns. We conclude with the concept of **cointegration** and how to apply it to develop a **pairs trading strategy**.

## Content

1. [Tools for diagnostics and feature extraction](#tools-for-diagnostics-and-feature-extraction)
    * [How to decompose time series patterns](#how-to-decompose-time-series-patterns)
    * [Rolling window statistics and moving averages](#rolling-window-statistics-and-moving-averages)
    * [How to measure autocorrelation](#how-to-measure-autocorrelation)
    * [How to diagnose and achieve stationarity](#how-to-diagnose-and-achieve-stationarity)
    * [How to apply time series transformations](#how-to-apply-time-series-transformations)
    * [How to diagnose and address unit roots](#how-to-diagnose-and-address-unit-roots)
    * [Code example: working with time series data](#code-example-working-with-time-series-data)
    * [Resources](#resources)
2. [Univariate Time Series Models](#univariate-time-series-models)
    * [How to build autoregressive models](#how-to-build-autoregressive-models)
    * [How to build moving average models](#how-to-build-moving-average-models)
    * [How to build ARIMA models and extensions](#how-to-build-arima-models-and-extensions)
    * [Code example: forecasting macro fundamentals with ARIMA and SARIMAX models](#code-example-forecasting-macro-fundamentals-with-arima-and-sarimax-models)
    * [How to use time series models to forecast volatility](#how-to-use-time-series-models-to-forecast-volatility)
    * [How to build a volatility-forecasting model](#how-to-build-a-volatility-forecasting-model)
    * [Code examples: volatility forecasts](#code-examples-volatility-forecasts)
    * [Resources](#resources-2)
3. [Multivariate Time Series Models](#multivariate-time-series-models)
    * [The vector autoregressive (VAR) model](#the-vector-autoregressive-var-model)
    * [Code example: How to use the VAR model for macro fundamentals forecasts](#code-example-how-to-use-the-var-model-for-macro-fundamentals-forecasts)
    * [Resources](#resources-3)
4. [Cointegration â€“ time series with a common trend](#cointegration--time-series-with-a-common-trend)
    * [Pairs trading: Statistical arbitrage with cointegration](#pairs-trading-statistical-arbitrage-with-cointegration)
    * [Alternative approaches to selecting and trading comoving assets](#alternative-approaches-to-selecting-and-trading-comoving-assets)
    * [Code example: Pairs trading in practice](#code-example-pairs-trading-in-practice)
        - [Computing distance-based heuristics to identify cointegrated pairs](#computing-distance-based-heuristics-to-identify-cointegrated-pairs)
        - [Precomputing the cointegration tests](#precomputing-the-cointegration-tests)
    * [Resources](#resources-4)

## Tools for diagnostics and feature extraction

Most of the examples in this section use data provided by the Federal Reserve that you can access using the pandas datareader that we introduced in [Chapter 2, Market and Fundamental Data](../02_market_and_fundamental_data). 

### How to decompose time series patterns

Time series data typically contains a mix of various patterns that can be decomposed into several components, each representing an underlying pattern category. In particular, time series often consist of the systematic components trend, seasonality and cycles, and unsystematic noise. These components can be combined in an additive, linear model, in particular when fluctuations do not depend on the level of the series, or in a non-linear, multiplicative model. 

- `pandas` Time Series and Date functionality [docs](https://pandas.pydata.org/pandas-docs/stable/timeseries.html)
- [Forecasting - Principles & Practice, Hyndman, R. and Athanasopoulos, G., ch.6 'Time Series Decomposition'](https://otexts.org/fpp2/decomposition.html)

### Rolling window statistics and moving averages

The pandas library includes very flexible functionality to define various window types, including rolling, exponentially weighted and expanding windows.

- `pandas` window function [docs](https://pandas.pydata.org/pandas-docs/stable/computation.html#window-functions)

### How to measure autocorrelation

Autocorrelation (also called serial correlation) adapts the concept of correlation to the time series context: just as the correlation coefficient measures the strength of a linear relationship between two variables, the autocorrelation coefficient measures the extent of a linear relationship between time series values separated by a given lag.

We present the following tools to measure autocorrelation:
- autocorrelation function (ACF)
- partial autocorrelation function (PACF)
- correlogram as a plot of ACF or PACF against the number of lags.

### How to diagnose and achieve stationarity

The statistical properties, such as the mean, variance, or autocorrelation, of a stationary time series are independent of the period, that is, they don't change over time. Hence, stationarity implies that a time series does not have a trend or seasonal effects and that descriptive statistics, such as the mean or the standard deviation, when computed for different rolling windows, are constant or do not change much over time.

### How to apply time series transformations

To satisfy the stationarity assumption of linear time series models, we need to transform the original time series, often in several steps. Common transformations include the application of the (natural) logarithm to convert an exponential growth pattern into a linear trend and stabilize the variance, or differencing.

### How to diagnose and address unit roots

Unit roots pose a particular problem for determining the transformation that will render a time series stationary. In practice, time series of interest rates or asset prices are often not stationary, for example, because there does not exist a price level to which the series reverts. The most prominent example of a non-stationary series is the random walk.

The defining characteristic of a unit-root non-stationary series is long memory: since current values are the sum of past disturbances, large innovations persist for much longer than for a mean-reverting, stationary series. Identifying the correct transformation, and in particular, the appropriate number and lags for differencing is not always clear-cut. We present a few heuristics to guide the process.

Statistical unit root tests are a common way to determine objectively whether (additional) differencing is necessary. These are statistical hypothesis tests of stationarity that are designed to determine whether differencing is required.

### Code example: working with time series data

- The notebook [tsa_and_stationarity](01_tsa_and_stationarity.ipynb) illustrates the concepts discussed in this section.

### Resources

- [Analysis of Financial Time Series, 3rd Edition, Ruey S. Tsay](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354)
- [Quantitative Equity Investing: Techniques and Strategies, Frank J. Fabozzi, Sergio M. Focardi, Petter N. Kolm](https://www.wiley.com/en-us/Quantitative+Equity+Investing%3A+Techniques+and+Strategies-p-9780470262474)
- `statsmodels` Time Series Analysis [docs](https://www.statsmodels.org/dev/tsa.html)

## Univariate Time Series Models

Univariate time series models relate the value of the time series at the point in time of interest to a linear combination of lagged values of the series and possibly past disturbance terms.

While exponential smoothing models are based on a description of the trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data. ARIMA(p, d, q) models require stationarity and leverage two building blocks:
- Autoregressive (AR) terms consisting of p-lagged values of the time series
- Moving average (MA) terms that contain q-lagged disturbances

### How to build autoregressive models

An AR model of order p aims to capture the linear dependence between time series values at different lags. It closely resembles a multiple linear regression on lagged values of the outcome.

### How to build moving average models

An MA model of order q uses q past disturbances rather than lagged values of the time series in a regression-like model. Since we do not observe the white-noise disturbance values, MA(q) is not a regression model like the ones we have seen so far. Rather than using least squares, MA(q) models are estimated using maximum likelihood (MLE).

### How to build ARIMA models and extensions

Autoregressive integrated moving-average ARIMA(p, d, q) models combine AR(p) and MA(q) processes to leverage the complementarity of these building blocks and simplify model development by using a more compact form and reducing the number of parameters, in turn reducing the risk of overfitting.

- statsmodels State-Space Models [docs](https://www.statsmodels.org/dev/statespace.html)

### Code example: forecasting macro fundamentals with ARIMA and SARIMAX models

We will build a SARIMAX model for monthly data on an industrial production time series for the 1988-2017 period. See notebook [arima_models](02_arima_models.ipynb) for implementation details.

### How to use time series models to forecast volatility

A particularly important area of application for univariate time series models is the prediction of volatility. The volatility of financial time series is usually not constant over time but changes, with bouts of volatility clustering together. Changes in variance create challenges for time series forecasting using the classical ARIMA models.

### How to build a volatility-forecasting model

The development of a volatility model for an asset-return series consists of four steps:
1. Build an ARMA time series model for the financial time series based on the serial dependence revealed by the ACF and PACF.
2. Test the residuals of the model for ARCH/GARCH effects, again relying on the ACF and PACF for the series of the squared residual.
3. Specify a volatility model if serial correlation effects are significant, and jointly estimate the mean and volatility equations.
4. Check the fitted model carefully and refine it if necessary.

### Code examples: volatility forecasts

The notebook [arch_garch_models](03_arch_garch_models.ipynb) demonstrates the usage of the ARCH library to estimate time series models for volatility forecasting with NASDAQ data.

### Resources

- NYU Stern [VLAB](https://vlab.stern.nyu.edu/)
- ARCH Library
    - [docs](https://arch.readthedocs.io/en/latest/index.html) 
    - [examples](http://nbviewer.jupyter.org/github/bashtage/arch/blob/master/examples/univariate_volatility_modeling.ipynb)

## Multivariate Time Series Models

Multivariate time series models are designed to capture the dynamic of multiple time series simultaneously and leverage dependencies across these series for more reliable predictions.

Univariate time-series models like the ARMA approach are limited to statistical relationships between a target variable and its lagged values or lagged disturbances and exogenous series in the ARMAX case. In contrast, multivariate time-series models also allow for lagged values of other time series to affect the target. This effect applies to all series, resulting in complex interactions.

In addition to potentially better forecasting, multivariate time series are also used to gain insights into cross-series dependencies. For example, in economics, multivariate time series are used to understand how policy changes to one variable, such as an interest rate, may affect other variables over different horizons. 

- [New Introduction to Multiple Time Series Analysis, LÃ¼tkepohl, Helmut, Springer, 2005](https://www.springer.com/us/book/9783540401728)

### The vector autoregressive (VAR) model

The vector autoregressive VAR(p) model extends the AR(p) model to k series by creating a system of k equations where each contains p lagged values of all k series.

VAR(p) models also require stationarity, so that the initial steps from univariate time-series modeling carry over. First, explore the series and determine the necessary transformations, and then apply the augmented Dickey-Fuller test to verify that the stationarity criterion is met for each series and apply further transformations otherwise. It can be estimated with OLS conditional on initial information or with MLE, which is equivalent for normally distributed errors but not otherwise.

If some or all of the k series are unit-root non-stationary, they may be cointegrated (see next section). This extension of the unit root concept to multiple time series means that a linear combination of two or more series is stationary and, hence, mean-reverting. 

### Code example: How to use the VAR model for macro fundamentals forecasts

The notebook [vector_autoregressive_model](04_vector_autoregressive_model.ipynb) demonstrates how to use `statsmodels` to estimate a VAR model for macro fundamentals time series.

### Resources

- `statsmodels` Vector Autoregression [docs](https://www.statsmodels.org/dev/vector_ar.html)
- [Time Series Analysis in Python with statsmodels](https://conference.scipy.org/proceedings/scipy2011/pdfs/statsmodels.pdf), Wes McKinney, Josef Perktold, Skipper Seabold, SciPY Conference 2011

## Cointegration â€“ time series with a common trend

The concept of an integrated multivariate series is complicated by the fact that all the component series of the process may be individually integrated but the process is not jointly integrated in the sense that one or more linear combinations of the series exist that produce a new stationary series.

In other words, a combination of two co-integrated series has a stable mean to which this linear combination reverts. A multivariate series with this characteristic is said to be co-integrated. This also applies when the individual series are integrated of a higher order and the linear combination reduces the overall order of integration. 

We demonstrate two major approaches to testing for cointegration:
- The Engleâ€“Granger two-step method
- The Johansen procedure

### Pairs trading: Statistical arbitrage with cointegration

Statistical arbitrage refers to strategies that employ some statistical model or method to take advantage of what appears to be relative mispricing of assets while maintaining a level of market neutrality.

Pairs trading is a conceptually straightforward strategy that has been employed by algorithmic traders since at least the mid-eighties (Gatev, Goetzmann, and Rouwenhorst 2006). The goal is to find two assets whose prices have historically moved together, track the spread (the difference between their prices), and, once the spread widens, buy the loser that has dropped below the common trend and short the winner. If the relationship persists, the long and/or the short leg will deliver profits as prices converge and the positions are closed. 

This approach extends to a multivariate context by forming baskets from multiple securities and trade one asset against a basket of two baskets against each other.

In practice, the strategy requires two steps: 
1. Formation phase: Identify securities that have a long-term mean-reverting relationship. Ideally, the spread should have a high variance to allow for frequent profitable trades while reliably reverting to the common trend.
2. Trading phase: Trigger entry and exit trading rules as price movements cause the spread to diverge and converge.

Several approaches to the formation and trading phases have emerged from increasingly active research in this area across multiple asset classes over the last several years. The next subsection outlines the key differences before we dive into an example application.

### Alternative approaches to selecting and trading comoving assets

A recent comprehensive survey of pairs trading strategies [Statistical Arbitrage Pairs Trading Strategies: Review
and Outlook](https://www.iwf.rw.fau.de/files/2016/03/09-2015.pdf), Krauss (2017) identifies four different methodologies plus a number of other more recent approaches, including ML-based forecasts:

- **Distance** approach: The oldest and most-studied method identifies candidate pairs with distance metrics like correlation and uses non-parametric thresholds like Bollinger Bands to trigger entry and exit trades. The computational simplicity allows for large-scale applications with demonstrated profitability across markets and asset classes for extended periods of time since Gatev, et al. (2006). However, performance has decayed more recently.
- **Cointegration** approach: As outlined previously, this approach relies on an econometric model of a long-term relationship among two or more variables and allows for statistical tests that promise more reliability than simple distance metrics. Examples in this category use the Engle-Granger and Johansen procedures to identify pairs and baskets of securities as well as simpler heuristics that aim to capture the concept (Vidyamurthy 2004). Trading rules often resemble the simple thresholds used with distance metrics.
- **Time-series** approach: With a focus on the trading phase, strategies in this category aim to model the spread as a mean-reverting stochastic process and optimize entry and exit rules accordingly (Elliott, Hoek, and Malcolm 2005). It assumes promising pairs have already been identified.
- **Stochastic control** approach: Similar to the time-series approach, the goal is to optimize trading rules using stochastic control theory to find value and policy functions to arrive at an optimal portfolio (Liu and Timmermann 2013). We will address this type of approach in Chapter 21, Reinforcement Learning.
- **Other approaches**: Besides pair identification based on unsupervised learning like principal component analysis (see Chapter 13, Unsupervised Learning) and statistical models like copulas (Patton 2012), machine learning has become popular more recently to identify pairs based on their relative price or return forecasts (Huck 2019). We will cover several ML algorithms that can be used for this purpose and illustrate corresponding multivariate pairs trading strategies in the coming chapters.

### Code example: Pairs trading in practice

The **distance approach** identifies pairs using the correlation of (normalized) asset prices or their returns and is simple and orders of magnitude less computationally intensive than cointegration tests. 
- The notebook [cointegration_tests](05_cointegration_tests.ipynb) illustrates this for a sample of ~150 stocks with four years of daily data: it takes ~30ms to compute the correlation with the returns of an ETF, compared to 18 seconds for a suite of cointegration tests (using statsmodels) - 600x slower.

The speed advantage is particularly valuable because the number of potential pairs is the product of the number of candidates to be considered on either side so that evaluating combinations of 100 stocks and 100 ETFs requires comparing 10,000 tests (weâ€™ll discuss the challenge of multiple testing bias below).

On the other hand, distance metrics do not necessarily select the most profitable pairs: correlation is maximized for perfect co-movement that in turn eliminates actual trading opportunities. Empirical studies confirm that the volatility of the price spread of cointegrated pairs is almost twice as high as the volatility of the price spread of distance pairs (Huck and Afawubo 2015).

To balance the tradeoff between computational cost and the quality of the resulting pairs, Krauss (2017) recommends a procedure that combines both approaches based on his literature review:
1. Select pairs with a stable spread that shows little drift to reduce the number of candidates
2. Test the remaining pairs with the highest spread variance for cointegration

This process aims to select cointegrated pairs with lower divergence risk while ensuring more volatile spreads that in turn generate higher profit opportunities.

A large number of tests introduce data snooping bias as discussed in Chapter 6, The Machine Learning Workflow: multiple testing is likely to increase the number of false positives that mistakenly reject the null hypothesis of no cointegration. While statistical significance may not be necessary for profitable trading (Chan 2008), a study of commodity pairs (Cummins and Bucca 2012) shows that controlling the familywise error rate to improve the testsâ€™ power according to Romano and Wolf (2010) can lead to better performance.

#### Computing distance-based heuristics to identify cointegrated pairs

- The notebook [cointegration_tests](05_cointegration_tests.ipynb) takes a closer look at how predictive various heuristics for the degree of comovement of asset prices are for the result of cointegration tests. The example code uses a sample of 172 stocks and 138 ETFs traded on the NYSE and NASDAQ with daily data from 2010 - 2019 provided by Stooq. 

The securities represent the largest average dollar volume over the sample period in their respective class; highly correlated and stationary assets have been removed. See the notebook [create_datasets](../data/create_datasets.ipynb) in the data folder of the GitHub repository for downloading for instructions on how to obtain the data and the notebook cointegration_tests for the relevant code and additional preprocessing and exploratory details.

#### Precomputing the cointegration tests

The notebook [statistical_arbitrage_with_cointegrated_pairs](06_statistical_arbitrage_with_cointegrated_pairs.ipynb) implements a statistical arbitrage strategy based on cointegration for the sample of stocks and ETFs and the 2017-2019 period.

It first generates and stores the cointegration tests for all candidate pairs and the resulting trading signals before we backtest a strategy based on these signals given the computational intensity of the process.

### Resources

- Quantopian offers various resources on pairs trading:
    - [Introduction to Pairs Trading](https://www.quantopian.com/lectures/introduction-to-pairs-trading)
    - [Quantopian Johansen](https://www.quantopian.com/posts/trading-baskets-co-integrated-with-spy)
    - [Quantopian PT](https://www.quantopian.com/posts/how-to-build-a-pairs-trading-strategy-on-quantopian)
    - [Pairs Trading Basics: Correlation, Cointegration And Strategy](https://blog.quantinsti.com/pairs-trading-basics/)
- Additional blog posts include:
    - [Pairs Trading using Data-Driven Techniques: Simple Trading Strategies Part 3](https://medium.com/auquan/pairs-trading-data-science-7dbedafcfe5a)
    - [Pairs Trading Johansen & Kalman](https://letianzj.github.io/kalman-filter-pairs-trading.html)
    - [Copulas](https://twiecki.io/blog/2018/05/03/copulas/) by Thomas Wiecki


### 17 - Deep Learning
# Deep Learning for Trading

This chapter kicks off part four, which covers how several deep learning (DL) modeling techniques can be useful for investment and trading. DL has achieved numerous breakthroughs in many domains ranging from image and speech recognition to robotics and intelligent agents that have drawn widespread attention and revived large-scale research into Artificial Intelligence (AI). The expectations are high that the rapid development will continue and many more solutions to difficult practical problems will emerge.

In this chapter, we will present feedforward neural networks to introduce key elements of working with neural networks relevant to the various DL architectures covered in the following chapters. More specifically, we will demonstrate how to train large models efficiently using the backpropagation algorithm and manage the risks of overfitting. We will also show how to use the popular Keras, TensorFlow 2.0, and PyTorch frameworks, which we will leverage throughout part four.

In the following chapters, we will build on this foundation to design various architectures suitable for different investment applications with a particular focus on alternative text and image data. These include recurrent neural networks (RNNs) tailored to sequential data such as time series or natural language, and Convolutional Neural Networks (CNNs), which are particularly well suited to image data but can also be used with time-series data. We will also cover deep unsupervised learning, including autoencoders and Generative Adversarial Networks (GANs) as well as reinforcement learning to train agents that interactively learn from their environment.

## Content

1. [Deep learning: How it differs and why it matters](#deep-learning-how-it-differs-and-why-it-matters)
    * [How hierarchical features help tame high-dimensional data](#how-hierarchical-features-help-tame-high-dimensional-data)
    * [Automating feature extraction: DL as representation learning](#automating-feature-extraction-dl-as-representation-learning)
    * [How DL relates to machine learning and artificial intelligence](#how-dl-relates-to-machine-learning-and-artificial-intelligence)
2. [Code example: Designing a neural network](#code-example-designing-a-neural-network)
    * [Key design choices](#key-design-choices)
    * [How to regularize deep neural networks](#how-to-regularize-deep-neural-networks)
    * [Training faster: Optimizations for deep learning](#training-faster-optimizations-for-deep-learning)
3. [Popular Deep Learning libraries](#popular-deep-learning-libraries)
    * [How to Leverage GPU Optimization](#how-to-leverage-gpu-optimization)
    * [How to use Tensorboard](#how-to-use-tensorboard)
    * [Code example: how to use PyTorch](#code-example-how-to-use-pytorch)
    * [Code example: How to use TensorFlow](#code-example-how-to-use-tensorflow)
4. [Code example: Optimizing a neural network for a long-short trading strategy](#code-example-optimizing-a-neural-network-for-a-long-short-trading-strategy)
    * [Optimizing the NN architecture](#optimizing-the-nn-architecture)
    * [Backtesting a long-short strategy based on ensembled signals](#backtesting-a-long-short-strategy-based-on-ensembled-signals)


## Deep learning: How it differs and why it matters

The machine learning (ML) algorithms covered in Part 2 work well on a wide variety of important problems, including on text data as demonstrated in Part 3. They have been less successful, however, in solving central AI problems such as recognizing speech or classifying objects in images. These limitations have motivated the development of DL, and the recent DL breakthroughs have greatly contributed to a resurgence of interest in AI. F

or a comprehensive introduction that includes and expands on many of the points in this section, see Goodfellow, Bengio, and Courville (2016), or for a much shorter version, see LeCun, Bengio, and Hinton (2015).

- [Deep Learning](https://www.deeplearningbook.org/), Ian Goodfellow, Yoshua Bengio and Aaron Courville, MIT Press, 2016
- [Deep learning](https://www.nature.com/articles/nature14539), Yann LeCun, Yoshua Bengio, and Geoffrey Hinton, Nature 2015
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), Michael A. Nielsen, Determination Press, 2015
- [The Quest for Artificial Intelligence - A History of Ideas and Achievements](https://ai.stanford.edu/~nilsson/QAI/qai.pdf), Nils J. Nilsson, Cambridge University Press, 2010
- [One Hundred Year Study on Artificial Intelligence (AI100)](https://ai100.stanford.edu/)
- [TensorFlow Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.71056&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false), Interactive, browser-based Deep Learning platform

### How hierarchical features help tame high-dimensional data

As discussed throughout Part 2, the key challenge of supervised learning is to generalize from training data to new samples. Generalization becomes exponentially more difficult as the dimensionality of the data increases. We encountered the root causes of these difficulties as the curse of dimensionality in Chapter 13, [Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity](../13_unsupervised_learning).

### Automating feature extraction: DL as representation learning

Many AI tasks like image or speech recognition require knowledge about the world. One of the key challenges is to encode this knowledge so a computer can utilize it. For decades, the development of ML systems required considerable domain expertise to transform the raw data (such as image pixels) into an internal representation that a learning algorithm could use to detect or classify patterns.

### How DL relates to machine learning and artificial intelligence

AI has a long history, going back at least to the 1950s as an academic field and much longer as a subject of human inquiry, but has experienced several waves of ebbing and flowing enthusiasm since (see [The Quest for Artificial Intelligence](https://ai.stanford.edu/~nilsson/QAI/qai.pdf), Nilsson, 2009 for an in-depth survey). 
- ML is an important subfield with a long history in related disciplines such as statistics and became prominent in the 1980s. 
- DL is a form of representation learning and itself a subfield of ML.

## Code example: Designing a neural network

To gain a better understanding of how NN work, the notebook [01_build_and_train_feedforward_nn](build_and_train_feedforward_nn.ipynb) formulates as simple feedforward architecture and forward propagation computations using matrix algebra and implements it using Numpy, the Python counterpart of linear algebra.

<p align="center">
<img src="https://i.imgur.com/UKCr9zi.png" width="85%">
</p>

### Key design choices

Some NN design choices resemble those for other supervised learning models. For example, the output is dictated by the type of the ML problem such as regression, classification, or ranking. Given the output, we need to select a cost function to measure prediction success and failure, and an algorithm that optimizes the network parameters to minimize the cost. 

NN-specific choices include the numbers of layers and nodes per layer, the connections between nodes of different layers, and the type of activation functions.

### How to regularize deep neural networks

The downside of the capacity of NN to approximate arbitrary functions is the greatly increased risk of overfitting. The best protection against overfitting is to train the model on a larger dataset. Data augmentation, e.g. by creating slightly modified versions of images, is a powerful alternative approach. The generation of synthetic financial training data for this purpose is an active research area that we will address in [Chapter 21](../21_gans_for_synthetic_time_series)

### Training faster: Optimizations for deep learning

Backprop refers to the computation of the gradient of the cost function with respect to the internal parameter we wish to update and the use of this information to update the parameter values. The gradient is useful because it indicates the direction of parameter change that causes the maximal increase in the cost function. Hence, adjusting the parameters according to the negative gradient produces an optimal cost reduction, at least for a region very close to the observed samples. See Ruder (2016) for an excellent overview of key gradient descent optimization algorithms.

- [Gradient Checking & Advanced Optimization](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization), Unsupervised Feature Learning and Deep Learning, Stanford University
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#momentum), Sebastian Ruder, 2016

## Popular Deep Learning libraries

Currently, the most popular DL libraries are [TensorFlow](https://www.tensorflow.org/) (supported by Google) and [PyTorch](https://pytorch.org/) (supported by Facebook). 

Development is very active with PyTorch at version 1.4 and TensorFlow at 2.2 as of March 2020. TensorFlow 2.0 adopted [Keras](https://keras.io/) as its main interface, effectively combining both libraries into one.
Additional options include:

- [Microsoft Cognitive Toolkit (CNTK)](https://github.com/Microsoft/CNTK)
- [Caffe](http://caffe.berkeleyvision.org/)
- [Thenao](http://www.deeplearning.net/software/theano/), developed at University of Montreal since 2007
- [Apache MXNet](https://mxnet.apache.org/), used by Amazon
- [Chainer](https://chainer.org/), developed by the Japanese company Preferred Networks
- [Torch](http://torch.ch/), uses Lua, basis for PyTorch
- [Deeplearning4J](https://deeplearning4j.org/), uses Java

### How to Leverage GPU Optimization

All popular Deep Learning libraries support the use of GPU, and some also allow for parallel training on multiple GPU. The most common types of GPU are produced by NVIDIA, and configuration requires installation and setup of the CUDA environment. The process continues to evolve and can be somewhat challenging depending on your computational environment. 

A more straightforward way to leverage GPU is via the the Docker virtualization platform. There are numerous images available that you can run in local container managed by Docker that circumvents many of the driver and version conflicts that you may otherwise encounter. Tensorflow provides docker images on its website that can also be used with Keras. 

- [Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](http://timdettmers.com/2018/11/05/which-gpu-for-deep-learning/), Tim Dettmers
- [A Full Hardware Guide to Deep Learning](http://timdettmers.com/2018/12/16/deep-learning-hardware-guide/), Tim Dettmers

### How to use Tensorboard

Tensorboard is a great visualization tool that comes with TensorFlow. It includes a suite of visualization tools to simplify the understanding, debugging, and optimization of neural networks.

You can use it to visualize the computational graph, plot various execution and performance metrics, and even visualize image data processed by the network. It also permits comparisons of different training runs.
When you run the how_to_use_keras notebook, and with TensorFlow installed, you can launch Tensorboard from the command line:

```python
tensorboard --logdir=/full_path_to_your_logs ## e.g. ./tensorboard
```
- [TensorBoard: Visualizing Learning](https://www.tensorflow.org/guide/summaries_and_tensorboard)

### Code example: how to use PyTorch

Pytorch has been developed at the Facebook AI Research group led by Yann LeCunn and the first alpha version released in September 2016. It provides deep integration with Python libraries like Numpy that can be used to extend its functionality, strong GPU acceleration, and automatic differentiation using its autograd system. It provides more granular control than Keras through a lower-level API and is mainly used as a deep learning research platform but can also replace NumPy while enabling GPU computation.

It employs eager execution, in contrast to the static computation graphs used by, e.g., Theano or TensorFlow. Rather than initially defining and compiling a network for fast but static execution, it relies on its autograd package for automatic differentiation of Tensor operations, i.e., it computes gradients â€˜on the flyâ€™ so that network structures can be partially modified more easily. This is called define-by-run, meaning that backpropagation is defined by how your code runs, which in turn implies that every single iteration can be different. The PyTorch documentation provides a detailed tutorial on this.

The notebook [how_to_use_pytorch](03_how_to_use_pytorch.ipynb) illustrates how to use the 1.4 release.

- [PyTorch Documentation](https://pytorch.org/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials)
- [PyTorch Ecosystem](https://pytorch.org/ecosystem)
    - [AllenNLP](https://allennlp.org/), state-of-the-art NLP platform developed by the Allen Institute for Artificial Intelligence
    - [Flair](https://github.com/zalandoresearch/flair),  simple framework for state-of-the-art NLP developed at Zalando
    - [fst.ai](http://www.fast.ai/), simplifies training NN using modern best practices; offers online training

### Code example: How to use TensorFlow

TensorFlow has become the leading deep learning library shortly after its release in September 2015, one year before PyTorch. TensorFlow 2.0 aims to simplify the API that has grown increasingly complex over time by making the Keras API, integrated into TensorFlow as part of the contrib package since 2017 its principal interface, and adopting eager execution. It will continue to focus on a robust implementation across numerous platforms but will make it easier to experiment and do research.

The notebook [how_to_use_tensorflow](04_how_to_use_tensorflow.ipynb) illustrates how to use the 2.0 release.

- [TensorFlow.org](https://www.tensorflow.org/)
- [Standardizing on Keras: Guidance on High-level APIs in TensorFlow 2.0](https://medium.com/tensorflow/standardizing-on-keras-guidance-on-high-level-apis-in-tensorflow-2-0-bad2b04c819a)
- [TensorFlow.js](https://js.tensorflow.org/), A JavaScript library for training and deploying ML models in the browser and on Node.js

## Code example: Optimizing a neural network for a long-short trading strategy

In practice, we need to explore variations for the design options for the NN architecture and how we train it from those we outlined previously because we can never be sure from the outset which configuration best suits the data. 

This code example explores various architectures for a simple feedforward neural network to predict daily stock returns using the dataset developed in [Chapter 12](../12_gradient_boosting_machines) (see the notebook [preparing_the_model_data](../12_gradient_boosting_machines/04_preparing_the_model_data.ipynb)).

To this end, we will define a function that returns a TensorFlow model based on several architectural input parameters and cross-validate alternative designs using the MultipleTimeSeriesCV we introduced in Chapter 7. To assess the signal quality of the model predictions, we build a simple ranking-based long-short strategy based on an ensemble of the models that perform best during the in-sample cross-validation period. To limit the risk of false discoveries, we then evaluate the performance of this strategy for an out-of-sample test period.

### Optimizing the NN architecture

The notebook [how_to_optimize_a_NN_architecure](04_how_to_use_tensorflow.ipynb) explores various options to build a simple feedforward Neural Network to predict asset returns. To develop our trading strategy, we use the daily stock returns for 995 US stocks for the eight-year period from 2010 to 2017. 

### Backtesting a long-short strategy based on ensembled signals

To translate our NN model into a trading strategy, we generate predictions, evaluate their signal quality, create rules that define how to trade on these predictions, and backtest the performance of a strategy that implements these rules. 

The notebook [backtesting_with_zipline](05_backtesting_with_zipline.ipynb) contains the code examples for this section.


### 19 - Recurrent Neural Networks
# RNN for Trading: Multivariate Time Series and Text Data

The major innovation of RNN is that each output is a function of both previous output and new data. As a result, RNN gain the ability to incorporate information on previous observations into the computation it performs on a new feature vector, effectively creating a model with memory. This recurrent formulation enables parameter sharing across a much deeper computational graph that includes cycles. Prominent architectures include Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) that aim to overcome the challenge of vanishing gradients associated with learning long-range dependencies, where errors need to be propagated over many connections. 

RNNs have been successfully applied to various tasks that require mapping one or more input sequences to one or more output sequences and are particularly well suited to natural language. RNN can also be applied to univariate and multivariate time series to predict market or fundamental data. This chapter covers how RNN can model alternative text data using the word embeddings that we covered in [Chapter 16](16_word_embeddings) to classify the sentiment expressed in documents.

## Content

1. [How recurrent neural nets work](#how-recurrent-neural-nets-work)
    * [Backpropagation through Time](#backpropagation-through-time)
    * [Alternative RNN Architectures](#alternative-rnn-architectures)
        - [Long-Short Term Memory](#long-short-term-memory)
        - [Gated Recurrent Units](#gated-recurrent-units)
2. [RNN for financial time series with TensorFlow 2](#rnn-for-financial-time-series-with-tensorflow-2)
    * [Code example: Univariate time-series regression: predicting the S&P 500](#code-example-univariate-time-series-regression-predicting-the-sp-500)
    * [Code example: Stacked LSTM for predicting weekly stock price moves and returns](#code-example-stacked-lstm-for-predicting-weekly-stock-price-moves-and-returns)
    * [Code example: Predicting returns instead of directional price moves](#code-example-predicting-returns-instead-of-directional-price-moves)
    * [Code example: Multivariate time-series regression for macro data](#code-example-multivariate-time-series-regression-for-macro-data)
3. [RNN for text data: sentiment analysis and return prediction](#rnn-for-text-data-sentiment-analysis-and-return-prediction)
    * [Code example: LSTM with custom word embeddings for sentiment classification](#code-example-lstm-with-custom-word-embeddings-for-sentiment-classification)
    * [Code example: Sentiment analysis with pretrained word vectors](#code-example-sentiment-analysis-with-pretrained-word-vectors)
    * [Code example: SEC filings for a bidirectional RNN GRU to predict weekly returns](#code-example-sec-filings-for-a-bidirectional-rnn-gru-to-predict-weekly-returns)

## How recurrent neural nets work

RNNs assume that the input data has been generated as a sequence such that previous data points impact the current observation and are relevant for predicting subsequent values. Thus, they allow for more complex input-output relationships than FFNNs and CNNs, which are designed to map one input vector to one output vector using a given number of computational steps. 
RNNs, in contrast, can model data for tasks where the input, the output, or both, are best represented as a sequence of vectors. 

For a thorough overview, see [chapter 10](https://www.deeplearningbook.org/contents/rnn.html in [Deep Learning](https://www.deeplearningbook.org/) by Goodfellow, Bengio, and Courville (2016).

### Backpropagation through Time

 RNNs are called recurrent because they apply the same transformations to every element of a sequence in a way that the output depends on the outcome of prior iterations. As a result, RNNs maintain an internal state that captures information about previous elements in the sequence akin to a memory.

The backpropagation algorithm that updates the weight parameters based on the gradient of the loss function with respect to the parameters involves a forward pass from left to right along the unrolled computational graph, followed by backward pass in the opposite direction.

- [Sequence Modeling: Recurrent and Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html), Deep Learning Book, Chapter 10, Ian Goodfellow, Yoshua Bengio and Aaron Courville, MIT Press, 2016
- [Supervised Sequence Labelling with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/preprint.pdf), Alex Graves, 2013
- [Tutorial on LSTM Recurrent Networks](http://people.idsia.ch/~juergen/lstm/sld001.htm), Juergen Schmidhuber, 2003
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Alternative RNN Architectures

RNNs can be designed in a variety of ways to best capture the functional relationship and dynamic between input and output data. In addition to the recurrent connections between the hidden states, there are several alternative approaches, including recurrent output relationships, bidirectional RNN, and encoder-decoder architectures.

#### Long-Short Term Memory

RNNs with an LSTM architecture have more complex units that maintain an internal state and contain gates to keep track of dependencies between elements of the input sequence and regulate the cellâ€™s state accordingly. These gates recurrently connect to each other instead of the usual hidden units we encountered above. They aim to address the problem of vanishing and exploding gradients by letting gradients pass through unchanged.

A typical LSTM unit combines four parameterized layers that interact with each other and the cell state by transforming and passing along vectors. These layers usually involve an input gate, an output gate, and a forget gate, but there are variations that may have additional gates or lack some of these mechanisms

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), Christopher Olah, 2015
- [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf), Rafal Jozefowicz, Ilya Sutskever, et al, 2015

#### Gated Recurrent Units

Gated recurrent units (GRU) simplify LSTM units by omitting the output gate. They have been shown to achieve similar performance on certain language modeling tasks but do better on smaller datasets.

- [Learning Phrase Representations using RNN Encoderâ€“Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf), Kyunghyun Cho, Yoshua Bengio, et al 2014
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555), Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio, 2014

## RNN for financial time series with TensorFlow 2

We illustrate how to build RNN using the Keras library for various scenarios. The first set of models includes regression and classification of univariate and multivariate time series. The second set of tasks focuses on text data for sentiment analysis using text data converted to word embeddings (see [Chapter 15](../15_word_embeddings)). 

- [Recurrent Neural Networks (RNN) with Keras](https://www.tensorflow.org/guide/keras/rnn)
- [Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras documentation](https://keras.io/getting-started/sequential-model-guide/)
- [LSTM documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Working with RNNs](https://keras.io/guides/working_with_rnns/) by Scott Zhu and Francois Chollet

### Code example: Univariate time-series regression: predicting the S&P 500

The notebook [univariate_time_series_regression](01_univariate_time_series_regression.ipynb) demonstrates how to get data into the requisite shape and how to forecast the S&P 500 index values using a Recurrent Neural Network. 

### Code example: Stacked LSTM for predicting weekly stock price moves and returns

We'll now build a slightly deeper model by stacking two LSTM layers using the Quandl stock price data. Furthermore, we will include features that are not sequential in nature, namely indicator variables that identify the ticker and time periods like month and year.
- See the [stacked_lstm_with_feature_embeddings](02_stacked_lstm_with_feature_embeddings.ipynb) notebook for implementation details.

### Code example: Predicting returns instead of directional price moves

The notebook [stacked_lstm_with_feature_embeddings_regression](03_stacked_lstm_with_feature_embeddings_regression.ipynb) illustrates how to adapt the model to the regression task of predicting returns rather than binary price changes.

### Code example: Multivariate time-series regression for macro data

So far, we have limited our modeling efforts to single time series. RNNs are naturally well suited to multivariate time series and represent a non-linear alternative to the Vector Autoregressive (VAR) models we covered in [Chapter 9, Time Series Models](../09_time_series_models).

The notebook [multivariate_timeseries](04_multivariate_timeseries.ipynb) demonstrates the application of RNNs to modeling and forecasting several time series using the same dataset we used for the [VAR example](../09_time_series_models/04_vector_autoregressive_model.ipynb), namely monthly data on consumer sentiment, and industrial production from the Federal Reserve's FRED service.

## RNN for text data: sentiment analysis and return prediction

### Code example: LSTM with custom word embeddings for sentiment classification

RNNs are commonly applied to various natural language processing tasks. We've already encountered sentiment analysis using text data in part three of [this book](https://www.amazon.com/Machine-Learning-Algorithmic-Trading-alternative/dp/1839217715?pf_rd_r=VMKJPZC4N36TTZZCWATP&pf_rd_p=c5b6893a-24f2-4a59-9d4b-aff5065c90ec&pd_rd_r=8f331266-0d21-4c76-a3eb-d2e61d23bb31&pd_rd_w=kVGNF&pd_rd_wg=LYLKH&ref_=pd_gw_ci_mcx_mr_hp_d).

This example shows how to learn custom embedding vectors while training an RNN on the classification task. This differs from the word2vec model that learns vectors while optimizing predictions of neighboring tokens, resulting in their ability to capture certain semantic relationships among words (see Chapter 16). Learning word vectors with the goal of predicting sentiment implies that embeddings will reflect how a token relates to the outcomes it is associated with.

The notebook [sentiment_analysis_imdb](05_sentiment_analysis_imdb.ipynb) illustrates how to apply an RNN model to text data to detect positive or negative sentiment (which can easily be extended to a finer-grained sentiment scale). We are going to use word embeddings to represent the tokens in the documents. We covered word embeddings in [Chapter 15, Word Embeddings](../15_word_embeddings). They are an excellent technique to convert text into a continuous vector representation such that the relative location of words in the latent space encodes useful semantic aspects based on the words' usage in context.

### Code example: Sentiment analysis with pretrained word vectors

In [Chapter 15, Word Embeddings](../15_word_embeddings), we showed how to learn domain-specific word embeddings. Word2vec, and related learning algorithms, produce high-quality word vectors, but require large datasets. Hence, it is common that research groups share word vectors trained on large datasets, similar to the weights for pretrained deep learning models that we encountered in the section on transfer learning in the [previous chapter](../17_convolutional_neural_nets).

The notebook [sentiment_analysis_pretrained_embeddings](06_sentiment_analysis_pretrained_embeddings.ipynb) illustrates how to use pretrained Global Vectors for Word Representation (GloVe) provided by the Stanford NLP group with the IMDB review dataset.

- [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/), Stanford AI Group
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/), Stanford NLP

### Code example: SEC filings for a bidirectional RNN GRU to predict weekly returns

In Chapter 16, we discussed important differences between product reviews and financial text data. While the former was useful to illustrate important workflows, in this section, we will tackle more challenging but also more relevant financial documents. 

More specifically, we will use the SEC filings data introduced in [Chapter 16](../16_word_embeddings) to learn word embeddings tailored to predicting the return of the ticker associated with the disclosures from before publication to one week after.

The notebook [sec_filings_return_prediction](07_sec_filings_return_prediction.ipynb) contains the code examples for this application. 

See the notebook [sec_preprocessing](../16_word_embeddings/06_sec_preprocessing.ipynb) in Chapter 16 and instructions in the data folder on GitHub on how to obtain the data.


### 22 - Deep Reinforcement Learning
# Deep Reinforcement Learning: Building a Trading Agent

Reinforcement Learning (RL) is a computational approach to goal-directed learning performed by an agent that interacts with a typically stochastic environment which the agent has incomplete information about. RL aims to automate how the agent makes decisions to achieve a long-term objective by learning the value of states and actions from a reward signal. The ultimate goal is to derive a policy that encodes behavioral rules and maps states to actions.

This chapter shows how to formulate an RL problem and how to apply various solution methods. It covers model-based and model-free methods, introduces the [OpenAI Gym](https://gym.openai.com/) environment, and combines deep learning with RL to train an agent that navigates a complex environment. Finally, we'll show you how to adapt RL to algorithmic trading by modeling an agent that interacts with the financial market while trying to optimize an objective function. 

#### Table of contents

1. [Key elements of a reinforcement learning system](#key-elements-of-a-reinforcement-learning-system)
    * [The policy: translating states into actions](#the-policy-translating-states-into-actions)
    * [Rewards: learning from actions](#rewards-learning-from-actions)
    * [The value function: optimal decisions for the long run](#the-value-function-optimal-decisions-for-the-long-run)
    * [The environment](#the-environment)
    * [Components of an interactive RL system](#components-of-an-interactive-rl-system)
2. [How to solve RL problems](#how-to-solve-rl-problems)
    * [Code example: dynamic programming â€“ value and policy iteration](#code-example-dynamic-programming--value-and-policy-iteration)
    * [Code example: Q-Learning](#code-example-q-learning)
3. [Deep Reinforcement Learning](#deep-reinforcement-learning)
    * [Value function approximation with neural networks](#value-function-approximation-with-neural-networks)
    * [The Deep Q-learning algorithm and extensions](#the-deep-q-learning-algorithm-and-extensions)
    * [The Open AI Gym â€“ the Lunar Lander environment](#the-open-ai-gym--the-lunar-lander-environment)
    * [Code example: Double Deep Q-Learning using Tensorflow](#code-example-double-deep-q-learning-using-tensorflow)
4. [Code example: deep RL for trading with TensorFlow 2 and OpenAI Gym](#code-example-deep-rl-for-trading-with-tensorflow-2-and-openai-gym)
    * [How to Design an OpenAI trading environment](#how-to-design-an-openai-trading-environment)
    * [How to build a Deep Q-learning agent for the stock market](#how-to-build-a-deep-q-learning-agent-for-the-stock-market)
5. [Resources](#resources)
    * [RL Algorithms](#rl-algorithms)
    * [Investment Applications](#investment-applications)

## Key elements of a reinforcement learning system

RL problems feature several elements that set them apart from the ML settings we have covered so far. The following two sections outline the key features required for defining and solving an RL problem by learning a policy that automates decisions. 
Weâ€™ll use the notation and generally follow [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (Sutton and Barto 2018) and David Silverâ€™s [UCL Courses on RL](https://www.davidsilver.uk/teaching/) that are recommended for further study beyond the brief summary that the scope of this chapter permits.

RL problems aim to optimize an agent's decisions based on an objective function vis-a-vis an environment.

### The policy: translating states into actions
At any point in time, the policy defines the agentâ€™s behavior. It maps any state the agent may encounter to one or several actions. In an environment with a limited number of states and actions, the policy can be a simple lookup table filled in during training. 

### Rewards: learning from actions

The reward signal is a single value that the environment sends to the agent at each time step. The agentâ€™s objective is typically to maximize the total reward received over time. Rewards can also be a stochastic function of the state and the actions. They are typically discounted to facilitate convergence and reflect the time decay of value.
 
### The value function: optimal decisions for the long run
The reward provides immediate feedback on actions. However, solving an RL problem requires decisions that create value in the long run. This is where the value function comes in: it summarizes the utility of states or of actions in a given state in terms of their long-term reward. 
 
### The environment
The environment presents information about its state to the agent, assigns rewards for actions, and transitions the agent to new states subject to probability distributions the agent may or may not know about. 
It may be fully or partially observable, and may also contain other agents. The design of the environment typically requires significant up-front design effort to facilitate goal-oriented learning by the agent during training.

RL problems differ by the complexity of their state and action spaces that can be either discrete or continuous. The latter requires ML to approximate a functional relationship between states, actions, and their value. They also require us to generalize from the subset of states and actions they are experienced by the agent during training.

### Components of an interactive RL system

The components of an RL system typically include:

- Observations by the agent of the state of the environment
- A set of actions that are available to the agent
- A policy that governs the agent's decisions

In addition, the environment emits a reward signal that reflects the new state resulting from the agent's action. At the core, the agent usually learns a value function that shapes its judgment over actions. The agent has an objective function to process the reward signal and translate the value judgments into an optimal policy.

## How to solve RL problems

RL methods aim to learn from experience on how to take actions that achieve a long-term goal. To this end, the agent and the environment interact over a sequence of discrete time steps via the interface of actions, state observations, and rewards that we described in the previous section.

There are numerous approaches to solving RL problems which implies finding rules for the agent's optimal behavior:

- **Dynamic programming** (DP) methods make the often unrealistic assumption of complete knowledge of the environment, but are the conceptual foundation for most other approaches.
- **Monte Carlo** (MC) methods learn about the environment and the costs and benefits of different decisions by sampling entire state-action-reward sequences.
- **Temporal difference** (TD) learning significantly improves sample efficiency by learning from shorter sequences. To this end, it relies on bootstrapping, which is defined as refining its estimates based on its own prior estimates.

Approaches for continuous state and/or action spaces often leverage ML to approximate a value or policy function. Hence, they integrate supervised learning, and in particular, the deep learning methods we discussed in the last several chapters. However, these methods face distinct challenges in the RL context:

- The reward signal does not directly reflect the target concept, such as a labeled sample
- The distribution of the observations depends on the agent's actions and the policy which is itself the subject of the learning process

### Code example: dynamic programming â€“ value and policy iteration

Finite MDPs are a simple yet fundamental framework. This section introduces the trajectories of rewards that the agent aims to optimize, and define the policy and value functions they are used to formulate the optimization problem and the Bellman equations that form the basis for the solution methods.

The notebook [gridworld_dynamic_programming](01_gridworld_dynamic_programming.ipynb) applies Value and Policy Iteration to a toy environment that consists of a 3 x 4 grid.

### Code example: Q-Learning

Q-learning was an early RL breakthrough when it was developed by Chris Watkins for his [PhD thesis]((http://www.cs.rhul.ac.uk/~chrisw/thesis.html)) in 1989 . It introduces incremental dynamic programming to control an MDP without knowing or modeling the transition and reward matrices that we used for value and policy iteration in the previous section. A convergence proof followed three years later by [Watkins and Dayan](http://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html).

Q-learning directly optimizes the action-value function, q, to approximate q*. The learning proceeds off-policy, that is, the algorithm does not need to select actions based on the policy that's implied by the value function alone. However, convergence requires that all state-action pairs continue to be updated throughout the training process. A straightforward way to ensure this is by using an Îµ-greedy policy.

The Q-learning algorithm keeps improving a state-action value function after random initialization for a given number of episodes. At each time step, it chooses an action based on an Îµ-greedy policy, and uses a learning rate, Î±, to update the value function based on the reward  and its current estimate of the value function for the next state.

The notebook [gridworld_q_learning](02_gridworld_q_learning.ipynb) demonstrates how to build a Q-learning agent using the 3 x 4 grid of states from the previous section.

## Deep Reinforcement Learning

This section adapts Q-Learning to continuous states and actions where we cannot use the tabular solution that simply fills an array with state-action values. Instead, we will see how to approximate the optimal state-value function using a neural network to build a deep Q network with various refinements to accelerate convergence. We will then see how we can use the [OpenAI Gym](http://gym.openai.com/docs/) to apply the algorithm to the [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) environment.

### Value function approximation with neural networks

As in other fields, deep neural networks have become popular for approximating value functions. However, ML faces distinct challenges in the RL context where the data is generated by the interaction of the model with the environment using a (possibly randomized) policy:

- With continuous states, the agent will fail to visit most states and, thus, needs to generalize.
- Supervised learning aims to generalize from a sample of independently and identically distributed samples that are representative and correctly labeled. In the RL context, there is only one sample per time step so that learning needs to occur online.
- Samples can be highly correlated when sequential states are similar and the behavior distribution over states and actions is not stationary, but changes as a result of the agent's learning.

### The Deep Q-learning algorithm and extensions

Deep Q learning estimates the value of the available actions for a given state using a deep neural network. It was introduced by Deep Mind's [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (2013), where RL agents learned to play games solely from pixel input.

The deep Q-learning algorithm approximates the action-value function, q, by learning a set of weights, Î¸, of a multi-layered Deep Q Network (DQN) that maps states to actions.

Several innovations have improved the accuracy and convergence speed of deep Q-Learning, namely:
- **Experience replay** stores a history of state, action, reward, and next state transitions and randomly samples mini-batches from this experience to update the network weights at each time step before the agent selects an Îµ-greedy action. It increases sample efficiency, reduces the autocorrelation of samples, and limits the feedback due to the current weights producing training samples that can lead to local minima or divergence.
- **Slowly-changing target network** weakens the feedback loop from the current network parameters on the neural network weight updates. Also invented by by Deep Mind in [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) (2015), it use a slowly-changing target network that has the same architecture as the Q-network, but its weights are only updated periodically. The target network generates the predictions of the next state value used to update the Q-Networks estimate of the current state's value.
- **Double deep Q-learning** addresses the bias of deep Q-Learning to overestimate action values because it purposely samples the highest action value. This bias can negatively affect the learning process and the resulting policy if it does not apply uniformly , as shown by Hado van Hasselt in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (2015). To decouple the estimation of action values from the selection of actions, Double Deep Q-Learning (DDQN) uses the weights, of one network to select the best action given the next state, and the weights of another network to provide the corresponding action value estimate.

### The Open AI Gym â€“ the Lunar Lander environment

The [OpenAI Gym](https://gym.openai.com/) is a RL platform that provides standardized environments to test and benchmark RL algorithms using Python. It is also possible to extend the platform and register custom environments.

The [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) (LL) environment requires the agent to control its motion in two dimensions, based on a discrete action space and low-dimensional state observations that include position, orientation, and velocity. At each time step, the environment provides an observation of the new state and a positive or negative reward. Each episode consists of up to 1,000 time steps.

### Code example: Double Deep Q-Learning using Tensorflow

The [lunar_lander_deep_q_learning](03_lunar_lander_deep_q_learning.ipynb) notebook implements a DDQN agent that uses TensorFlow and Open AI Gym's Lunar Lander environment.

## Code example: deep RL for trading with TensorFlow 2 and OpenAI Gym

To train a trading agent, we need to create a market environment that provides price and other information, offers trading-related actions, and keeps track of the portfolio to reward the agent accordingly.

### How to Design an OpenAI trading environment

The OpenAI Gym allows for the design, registration, and utilization of environments that adhere to its architecture, as described in its [documentation](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym). 
- The [trading_env.py](trading_env.py) file implements an example that illustrates how to create a class that implements the requisite `step()` and `reset()` methods.

The trading environment consists of three classes that interact to facilitate the agent's activities:
 1. The `DataSource` class loads a time series, generates a few features, and provides the latest observation to the agent at each time step. 
 2. `TradingSimulator` tracks the positions, trades and cost, and the performance. It also implements and records the results of a buy-and-hold benchmark strategy. 
 3. `TradingEnvironment` itself orchestrates the process. 
 
### How to build a Deep Q-learning agent for the stock market
 
The notebook [q_learning_for_trading](04_q_learning_for_trading.ipynb) demonstrates how to set up a simple game with a limited set of options, a relatively low-dimensional state, and other parameters that can be easily modified and extended to train the Deep Q-Learning agent used in [lunar_lander_deep_q_learning](03_lunar_lander_deep_q_learning.ipynb).
 
<p align="center">
<img src="https://i.imgur.com/lg0ofbZ.png" width="60%">
</p>


## Resources

- [Reinforcement Learning: An Introduction, 2nd eition](http://incompleteideas.net/book/RLbook2018.pdf), Richard S. Sutton and Andrew G. Barto, 2018
- [University College of London Course on Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html), David Silver, 2015
- [Implementation of Reinforcement Learning Algorithms](https://github.com/dennybritz/reinforcement-learning), Denny Britz
    - This repository provides code, exercises and solutions for popular Reinforcement Learning algorithms. These are meant to serve as a learning tool to complement the theoretical materials from Sutton/Baron and Silver (see above).

### RL Algorithms

- Q Learning
    - [Learning from Delayed Rewards](http://www.cs.rhul.ac.uk/~chrisw/thesis.html), PhD Thesis, Chris Watkins, 1989
    - [Q-Learning](http://www.gatsby.ucl.ac.uk/~dayan/papers/wd92.html), Machine Learning, 1992
- Deep Q Networks
    - [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), Mnih et al, 2013
    - We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. The model is a convolutional neural network, trained with a variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards. We apply our method to seven Atari 2600 games from the Arcade Learning Environment, with no adjustment of the architecture or learning algorithm. We find that it outperforms all previous approaches on six of the games and surpasses a human expert on three of them.
- Asynchronous Advantage Actor-Critic (A2C/A3C)
    - [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), Mnih, V. et al. 2016
    - We propose a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms and show that parallel actor-learners have a stabilizing effect on training allowing all four methods to successfully train neural network controllers. The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems as well as on a new task of navigating random 3D mazes using a visual input.
- Proximal Policy Optimization (PPO)
    - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al, 2017
    - We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.

- Trust Region Policy Optimization (TRPO)
    - [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al, 2015
    - We describe an iterative procedure for optimizing policies, with guaranteed monotonic improvement. By making several approximations to the theoretically-justified procedure, we develop a practical algorithm, called Trust Region Policy Optimization (TRPO). This algorithm is similar to natural policy gradient methods and is effective for optimizing large nonlinear policies such as neural networks. Our experiments demonstrate its robust performance on a wide variety of tasks: learning simulated robotic swimming, hopping, and walking gaits; and playing Atari games using images of the screen as input. Despite its approximations that deviate from the theory, TRPO tends to give monotonic improvement, with little tuning of hyperparameters.
    
- Deep Deterministic Policy Gradient (DDPG)
    - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), Lillicrap et al, 2015
    - We adapt the ideas underlying the success of Deep Q-Learning to the continuous action domain. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. Using the same learning algorithm, network architecture and hyper-parameters, our algorithm robustly solves more than 20 simulated physics tasks, including classic problems such as cartpole swing-up, dexterous manipulation, legged locomotion and car driving. Our algorithm is able to find policies whose performance is competitive with those found by a planning algorithm with full access to the dynamics of the domain and its derivatives. We further demonstrate that for many of the tasks the algorithm can learn policies end-to-end: directly from raw pixel inputs.
- Twin Delayed DDPG (TD3)
    - [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), Fujimoto et al, 2018
    - In value-based reinforcement learning methods such as deep Q-learning, function approximation errors are known to lead to overestimated value estimates and suboptimal policies. We show that this problem persists in an actor-critic setting and propose novel mechanisms to minimize its effects on both the actor and the critic. Our algorithm builds on Double Q-learning, by taking the minimum value between a pair of critics to limit overestimation. We draw the connection between target networks and overestimation bias, and suggest delaying policy updates to reduce per-update error and further improve performance. We evaluate our method on the suite of OpenAI gym tasks, outperforming the state of the art in every environment tested.
- Soft Actor-Critic (SAC)
    - [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018
    - Model-free deep reinforcement learning (RL) algorithms have been demonstrated on a range of challenging decision making and control tasks. However, these methods typically suffer from two major challenges: very high sample complexity and brittle convergence properties, which necessitate meticulous hyperparameter tuning. Both of these challenges severely limit the applicability of such methods to complex, real-world domains. In this paper, we propose soft actor-critic, an off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework. In this framework, the actor aims to maximize expected reward while also maximizing entropy. That is, to succeed at the task while acting as randomly as possible. Prior deep RL methods based on this framework have been formulated as Q-learning methods. By combining off-policy updates with a stable stochastic actor-critic formulation, our method achieves state-of-the-art performance on a range of continuous control benchmark tasks, outperforming prior on-policy and off-policy methods. Furthermore, we demonstrate that, in contrast to other off-policy algorithms, our approach is very stable, achieving very similar performance across different random seeds.
- Categorical 51-Atom DQN (C51)
    - [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Bellemare, et al 2017
    - In this paper we argue for the fundamental importance of the value distribution: the distribution of the random return received by a reinforcement learning agent. This is in contrast to the common approach to reinforcement learning which models the expectation of this return, or value. Although there is an established body of literature studying the value distribution, thus far it has always been used for a specific purpose such as implementing risk-aware behaviour. We begin with theoretical results in both the policy evaluation and control settings, exposing a significant distributional instability in the latter. We then use the distributional perspective to design a new algorithm which applies Bellman's equation to the learning of approximate value distributions. We evaluate our algorithm using the suite of games from the Arcade Learning Environment. We obtain both state-of-the-art results and anecdotal evidence demonstrating the importance of the value distribution in approximate reinforcement learning. Finally, we combine theoretical and empirical evidence to highlight the ways in which the value distribution impacts learning in the approximate setting.
    
### Investment Applications
- [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/abs/1706.10059), Zhengyao Jiang, Dixing Xu, Jinjun Liang 2017
    - Financial portfolio management is the process of constant redistribution of a fund into different financial products. This paper presents a financial-model-free Reinforcement Learning framework to provide a deep machine learning solution to the portfolio management problem. The framework consists of the Ensemble of Identical Independent Evaluators (EIIE) topology, a Portfolio-Vector Memory (PVM), an Online Stochastic Batch Learning (OSBL) scheme, and a fully exploiting and explicit reward function. This framework is realized in three instants in this work with a Convolutional Neural Network (CNN), a basic Recurrent Neural Network (RNN), and a Long Short-Term Memory (LSTM). They are, along with a number of recently reviewed or published portfolio-selection strategies, examined in three back-test experiments with a trading period of 30 minutes in a cryptocurrency market. Cryptocurrencies are electronic and decentralized alternatives to government-issued money, with Bitcoin as the best-known example of a cryptocurrency. All three instances of the framework monopolize the top three positions in all experiments, outdistancing other compared trading algorithms. Although with a high commission rate of 0.25% in the backtests, the framework is able to achieve at least 4-fold returns in 50 days.
    - [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio); corresponding GitHub repo
- [Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/pdf/1807.02787.pdf), Huang, Chien-Yi, 2018
- [Order placement with Reinforcement Learning](https://github.com/mjuchli/ctc-executioner)
    - CTC-Executioner is a tool that provides an on-demand execution/placement strategy for limit orders on crypto currency markets using Reinforcement Learning techniques. The underlying framework provides functionalities which allow to analyse order book data and derive features thereof. Those findings can then be used in order to dynamically update the decision making process of the execution strategy.
    - The methods being used are based on a research project (master thesis) currently proceeding at TU Delft.
    
- [Q-Trader](https://github.com/edwardhdlu/q-trader)
    - An implementation of Q-learning applied to (short-term) stock trading. The model uses n-day windows of closing prices to determine if the best action to take at a given time is to buy, sell or sit. As a result of the short-term state representation, the model is not very good at making decisions over long-term trends, but is quite good at predicting peaks and troughs.
    
