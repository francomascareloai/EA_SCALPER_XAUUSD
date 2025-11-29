# FinRL - Deep Reinforcement Learning for Trading

Repository: https://github.com/AI4Finance-Foundation/FinRL
Framework for financial reinforcement learning

<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>

# FinRLÂ®: Financial Reinforcement Learning [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]

[1.1]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_twitter_22x22.png
[1.2]: http://www.tensorlet.org/wp-content/uploads/2021/01/facebook-button_22x22.png
[1.3]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_google_22.xx_.png
[1.4]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_linkedin_22x22.png

[1]: https://twitter.com/intent/tweet?text=FinRL-Financial-Deep-Reinforcement-Learning%20&url=https://github.com/AI4Finance-Foundation/FinRL&hashtags=DRL&hashtags=AI
[2]: https://www.facebook.com/sharer.php?u=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL
[3]: https://plus.google.com/share?url=https://github.com/AI4Finance-Foundation/FinRL
[4]: https://www.linkedin.com/sharing/share-offsite/?url=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="55%"/>
</div>

[![Downloads](https://static.pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://static.pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Join Discord](https://img.shields.io/badge/Discord-Join-blue)](https://discord.gg/trsr8SXpW5)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)
![](https://img.shields.io/github/issues-raw/AI4Finance-Foundation/finrl?label=Issues)
![](https://img.shields.io/github/issues-closed-raw/AI4Finance-Foundation/finrl?label=Closed+Issues)
![](https://img.shields.io/github/issues-pr-raw/AI4Finance-Foundation/finrl?label=Open+PRs)
![](https://img.shields.io/github/issues-pr-closed-raw/AI4Finance-Foundation/finrl?label=Closed+PRs)

[FinGPT](https://github.com/AI4Finance-Foundation/ChatGPT-for-FinTech): Open-source for open-finance! Revolutionize FinTech.


[![](https://dcbadge.vercel.app/api/server/trsr8SXpW5)](https://discord.gg/trsr8SXpW5)

![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL&countColor=%23B17A)
[![](https://dcbadge.limes.pink/api/server/trsr8SXpW5)](https://discord.gg/trsr8SXpW5)



**Financial reinforcement learning (FinRLÂ®)** ([Document website](https://finrl.readthedocs.io/en/latest/index.html)) is **the first open-source framework** for financial reinforcement learning. FinRL has evolved into an **ecosystem**
* [FinRL-DeepSeek](https://github.com/AI4Finance-Foundation/FinRL_DeepSeek): LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents

| Dev Roadmap  | Stage | Users | Project | Description |
|----|----|----|----|----|
| 0.0 (Preparation) | entrance | practitioners | [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)| gym-style market environments |
| 1.0 (Proof-of-Concept)| full-stack | developers | [this repo](https://github.com/AI4Finance-Foundation/FinRL) | automatic pipeline |
| 2.0 (Professional) | profession | experts | [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | algorithms |
| 3.0 (Production) | service | hedge funds | [Podracer](https://github.com/AI4Finance-Foundation/FinRL_Podracer) | cloud-native deployment |


## Outline

  - [Overview](#overview)
  - [File Structure](#file-structure)
  - [Supported Data Sources](#supported-data-sources)
  - [Installation](#installation)
  - [Status Update](#status-update)
  - [Tutorials](#tutorials)
  - [Publications](#publications)
  - [News](#news)
  - [Citing FinRL](#citing-finrl)
  - [Join and Contribute](#join-and-contribute)
    - [Contributors](#contributors)
    - [Sponsorship](#sponsorship)
  - [LICENSE](#license)

## Overview

FinRL has three layers: market environments, agents, and applications.  For a trading task (on the top), an agent (in the middle) interacts with a market environment (at the bottom), making sequential decisions.

<div align="center">
<img align="center" src=figs/finrl_framework.png>
</div>

A quick start: Stock_NeurIPS2018.ipynb. Videos [FinRL](http://www.youtube.com/watch?v=ZSGJjtM-5jA) at [AI4Finance Youtube Channel](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g).


## File Structure

The main folder **finrl** has three subfolders **applications, agents, meta**. We employ a **train-test-trade** pipeline with three files: train.py, test.py, and trade.py.

```
FinRL
â”œâ”€â”€ finrl (main folder)
â”‚   â”œâ”€â”€ applications
â”‚   	â”œâ”€â”€ Stock_NeurIPS2018
â”‚   	â”œâ”€â”€ imitation_learning
â”‚   	â”œâ”€â”€ cryptocurrency_trading
â”‚   	â”œâ”€â”€ high_frequency_trading
â”‚   	â”œâ”€â”€ portfolio_allocation
â”‚   	â””â”€â”€ stock_trading
â”‚   â”œâ”€â”€ agents
â”‚   	â”œâ”€â”€ elegantrl
â”‚   	â”œâ”€â”€ rllib
â”‚   	â””â”€â”€ stablebaseline3
â”‚   â”œâ”€â”€ meta
â”‚   	â”œâ”€â”€ data_processors
â”‚   	â”œâ”€â”€ env_cryptocurrency_trading
â”‚   	â”œâ”€â”€ env_portfolio_allocation
â”‚   	â”œâ”€â”€ env_stock_trading
â”‚   	â”œâ”€â”€ preprocessor
â”‚   	â”œâ”€â”€ data_processor.py
â”‚       â”œâ”€â”€ meta_config_tickers.py
â”‚   	â””â”€â”€ meta_config.py
â”‚   â”œâ”€â”€ config.py
â”‚   â”œâ”€â”€ config_tickers.py
â”‚   â”œâ”€â”€ main.py
â”‚   â”œâ”€â”€ plot.py
â”‚   â”œâ”€â”€ train.py
â”‚   â”œâ”€â”€ test.py
â”‚   â””â”€â”€ trade.py
â”‚
â”œâ”€â”€ examples
â”œâ”€â”€ unit_tests (unit tests to verify codes on env & data)
â”‚   â”œâ”€â”€ environments
â”‚   	â””â”€â”€ test_env_cashpenalty.py
â”‚   â””â”€â”€ downloaders
â”‚   	â”œâ”€â”€ test_yahoodownload.py
â”‚   	â””â”€â”€ test_alpaca_downloader.py
â”œâ”€â”€ setup.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

## Supported Data Sources

|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Akshare](https://alpaca.markets/docs/introduction/)| CN Securities| 2015-now, 1day| Account-specific| OHLCV| Prices&Indicators|
|[Alpaca](https://docs.alpaca.markets/docs/getting-started)| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|[Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)| CN Securities| 1990-12-19-now, 5min| Account-specific| OHLCV| Prices&Indicators|
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggegrated trades, OHLCV| Prices&Indicators|
|[CCXT](https://docs.ccxt.com/en/latest/manual.html)| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|[EODhistoricaldata](https://eodhistoricaldata.com/financial-apis/)| US Securities| Frequency-specific, 1min| API-specific | OHLCV | Prices&Indicators|
|[IEXCloud](https://iexcloud.io/docs/api/)| NMS US securities|1970-now, 1 day|100 per second per IP|OHLCV| Prices&Indicators|
|[JoinQuant](https://www.joinquant.com/)| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|[QuantConnect](https://www.quantconnect.com/docs/v2)| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|
|[RiceQuant](https://www.ricequant.com/doc/rqdata/python/)| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
[Sinopac](https://sinotrade.github.io/zh_TW/tutor/prepare/terms/) | Taiwan securities | 2023-04-13~now, 1min | Account-specific | OHLCV | Prices&Indicators|
|[Tushare](https://tushare.pro/document/1?doc_id=131)| CN Securities, A share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[YahooFinance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|


<!-- |Data Source |Type |Max Frequency |Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |
|    AkShare |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Alpaca |  US Stocks, ETFs |  1 min |  OHLCV |  Prices, indicators |
|    Alpha Vantage | Stock, ETF, forex, crypto, technical indicators | 1 min |  OHLCV  & Prices, indicators |
|    Baostock |  CN Securities |  5 min |  OHLCV |  Prices, indicators |
|    Binance |  Cryptocurrency |  1 s |  OHLCV |  Prices, indicators |
|    CCXT |  Cryptocurrency |  1 min  |  OHLCV |  Prices, indicators |
|    currencyapi |  Exchange rate | 1 day |  Exchange rate | Exchange rate, indicators |
|    currencylayer |  Exchange rate | 1 day  |  Exchange rate | Exchange rate, indicators |
|    EOD Historical Data | US stocks, and ETFs |  1 day  |  OHLCV  | Prices, indicators |
|    Exchangerates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    findatapy |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Financial Modeling prep | US stocks, currencies, crypto |  1 min |  OHLCV  | Prices, indicators |
|    finnhub | US Stocks, currencies, crypto |   1 day |  OHLCV  | Prices, indicators |
|    Fixer |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    IEXCloud |  NMS US securities | 1 day  | OHLCV |  Prices, indicators |
|    JoinQuant |  CN Securities |  1 min  |  OHLCV |  Prices, indicators |
|    Marketstack | 50+ countries |  1 day  |  OHLCV | Prices, indicators |
|    Open Exchange Rates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    pandas\_datareader |  US Securities |  1 day |  OHLCV | Prices, indicators |
|    pandas-finance |  US Securities |  1 day  |  OHLCV  & Prices, indicators |
|    Polygon |  US Securities |  1 day  |  OHLCV  | Prices, indicators |
|    Quandl | 250+ sources |  1 day  |  OHLCV  | Prices, indicators |
|    QuantConnect |  US Securities |  1 s |  OHLCV |  Prices, indicators |
|    RiceQuant |  CN Securities |  1 ms  |  OHLCV |  Prices, indicators |
|    Sinopac   | Taiwan securities | 1min | OHLCV |  Prices, indicators |
|    Tiingo | Stocks, crypto |  1 day  |  OHLCV  | Prices, indicators |
|    Tushare |  CN Securities | 1 min  |  OHLCV |  Prices, indicators |
|    WRDS |  US Securities |  1 ms  |  Intraday Trades | Prices, indicators |
|    XE |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    Xignite |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    YahooFinance |  US Securities | 1 min  |  OHLCV  |  Prices, indicators |
|    ystockquote |  US Securities |  1 day  |  OHLCV | Prices, indicators | -->



OHLCV: open, high, low, and close prices; volume. adjusted_close: adjusted close price

Technical indicators: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'. Users also can add new features.


## Installation
+ [Install description for all operating systems (MAC OS, Ubuntu, Windows 10)](./docs/source/start/installation.rst)
+ [FinRL for Quantitative Finance: Install and Setup Tutorial for Beginners](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159)

## Status Update
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 2022-06-25
	0.3.5: Formal release of FinRL, neo_finrl is chenged to FinRL-Meta with related files in directory: *meta*.
* 2021-08-25
	0.3.1: pytorch version with a three-layer architecture, apps (financial tasks), drl_agents (drl algorithms), neo_finrl (gym env)
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Remove tensorflow 1.0 at this moment, under development to support tensorflow 2.0
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
</div>
</details>


## Tutorials

+ [Towardsdatascience] [Deep Reinforcement Learning for Automated Stock Trading](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)


## Publications

|Title |Conference/Journal |Link|Citations|Year|
|  ----  |  ----  |  ----  |  ----  |  ----  |
|Dynamic Datasets and Market Environments for Financial Reinforcement Learning| Machine Learning - Springer Nature| [paper](https://arxiv.org/abs/2304.13174) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 7 | 2024 |
|**FinRL-Meta**: FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning| NeurIPS 2022| [paper](https://arxiv.org/abs/2211.03107) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 37 | 2022 |
|**FinRL**: Deep reinforcement learning framework to automate trading in quantitative finance| ACM International Conference on AI in Finance (ICAIF) | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3955949) | 49 | 2021 |
|**FinRL**: A deep reinforcement learning library for automated stock trading in quantitative finance| NeurIPS 2020 Deep RL Workshop  | [paper](https://arxiv.org/abs/2011.09607) | 87 | 2020 |
|Deep reinforcement learning for automated stock trading: An ensemble strategy| ACM International Conference on AI in Finance (ICAIF) | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/2-Advance/FinRL_Ensemble_StockTrading_ICAIF_2020/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) | 154 | 2020 |
|Practical deep reinforcement learning approach for stock trading | NeurIPS 2018 Workshop on Challenges and Opportunities for AI in Financial Services| [paper](https://arxiv.org/abs/1811.07522) [code](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/FinRL/tree/master/examples))| 164 | 2018 |


## News
+ [å¤®å¹¿ç½‘] [2021 IDEAå¤§ä¼šäºŽç¦ç”°åœ†æ»¡è½å¹•ï¼šç¾¤è‹±èŸèƒè®ºé“AI å¤šé¡¹ç›®å‘å¸ƒäº®ç‚¹çº·å‘ˆ](http://tech.cnr.cn/techph/20211123/t20211123_525669092.shtml)
+ [å¤®å¹¿ç½‘] [2021 IDEAå¤§ä¼šå¼€å¯AIæ€æƒ³ç››å®´ æ²ˆå‘æ´‹ç†äº‹é•¿å‘å¸ƒå…­å¤§å‰æ²¿äº§å“](https://baijiahao.baidu.com/s?id=1717101783873523790&wfr=spider&for=pc)
+ [IDEAæ–°é—»] [2021 IDEAå¤§ä¼šå‘å¸ƒäº§å“FinRL-Metaâ€”â€”åŸºäºŽæ•°æ®é©±åŠ¨çš„å¼ºåŒ–å­¦ä¹ é‡‘èžé£Žé™©æ¨¡æ‹Ÿç³»ç»Ÿ](https://idea.edu.cn/news/20211213143128.html)
+ [çŸ¥ä¹Ž] [FinRL-MetaåŸºäºŽæ•°æ®é©±åŠ¨çš„å¼ºåŒ–å­¦ä¹ é‡‘èžå…ƒå®‡å®™](https://zhuanlan.zhihu.com/p/437804814)
+ [é‡åŒ–æŠ•èµ„ä¸Žæœºå™¨å­¦ä¹ ] [åŸºäºŽæ·±åº¦å¼ºåŒ–å­¦ä¹ çš„è‚¡ç¥¨äº¤æ˜“ç­–ç•¥æ¡†æž¶ï¼ˆä»£ç +æ–‡æ¡£)](https://www.mdeditor.tw/pl/p5Gg)
+ [è¿ç­¹ORå¸·å¹„] [é¢†è¯»è®¡åˆ’NO.10 | åŸºäºŽæ·±åº¦å¢žå¼ºå­¦ä¹ çš„é‡åŒ–äº¤æ˜“æœºå™¨äººï¼šä»ŽAlphaGoåˆ°FinRLçš„æ¼”å˜è¿‡ç¨‹](https://zhuanlan.zhihu.com/p/353557417)
+ [æ·±åº¦å¼ºåŒ–å®žéªŒå®¤] [ã€é‡ç£…æŽ¨èã€‘å“¥å¤§å¼€æºâ€œFinRLâ€: ä¸€ä¸ªç”¨äºŽé‡åŒ–é‡‘èžè‡ªåŠ¨äº¤æ˜“çš„æ·±åº¦å¼ºåŒ–å­¦ä¹ åº“](https://blog.csdn.net/deeprl/article/details/114828024)
+ [å•†ä¸šæ–°çŸ¥] [é‡‘èžç§‘æŠ€è®²åº§å›žé¡¾|AI4Finance: ä»ŽAlphaGoåˆ°FinRL](https://www.shangyexinzhi.com/article/4170766.html)
+ [Kaggle] [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199313)
+ [çŸ©æ± äº‘Matpool] [åœ¨çŸ©æ± äº‘ä¸Šå¦‚ä½•è¿è¡ŒFinRLè‚¡ç¥¨äº¤æ˜“ç­–ç•¥æ¡†æž¶](http://www.python88.com/topic/111918)
+ [è´¢æ™ºæ— ç•Œ] [é‡‘èžå­¦ä¼šå¸¸åŠ¡ç†äº‹é™ˆå­¦å½¬: æ·±åº¦å¼ºåŒ–å­¦ä¹ åœ¨é‡‘èžèµ„äº§ç®¡ç†ä¸­çš„åº”ç”¨](https://www.sohu.com/a/486837028_120929319)
+ [Neurohive] [FinRL: Ð³Ð»ÑƒÐ±Ð¾ÐºÐ¾Ðµ Ð¾Ð±ÑƒÑ‡ÐµÐ½Ð¸Ðµ Ñ Ð¿Ð¾Ð´ÐºÑ€ÐµÐ¿Ð»ÐµÐ½Ð¸ÐµÐ¼ Ð´Ð»Ñ Ñ‚Ñ€ÐµÐ¹Ð´Ð¸Ð½Ð³Ð°](https://neurohive.io/ru/gotovye-prilozhenija/finrl-glubokoe-obuchenie-s-podkrepleniem-dlya-trejdinga/)
+ [ICHI.PRO] [ì–‘ì  ê¸ˆìœµì„ìœ„í•œ FinRL: ë‹¨ì¼ ì£¼ì‹ ê±°ëž˜ë¥¼ìœ„í•œ íŠœí† ë¦¬ì–¼](https://ichi.pro/ko/yangjeog-geum-yung-eul-wihan-finrl-dan-il-jusig-geolaeleul-wihan-tyutolieol-61395882412716)
+ [çŸ¥ä¹Ž] [åŸºäºŽæ·±åº¦å¼ºåŒ–å­¦ä¹ çš„é‡‘èžäº¤æ˜“ç­–ç•¥ï¼ˆFinRL+Stable baselines3ï¼Œä»¥é“ç¼æ–¯30è‚¡ç¥¨ä¸ºä¾‹ï¼‰](https://zhuanlan.zhihu.com/p/563238735)
+ [çŸ¥ä¹Ž] [åŠ¨æ€æ•°æ®é©±åŠ¨çš„é‡‘èžå¼ºåŒ–å­¦ä¹ ](https://zhuanlan.zhihu.com/p/616799055)
+ [çŸ¥ä¹Ž] [FinRLçš„W&BåŒ–+è¶…å‚æ•°æœç´¢å’Œæ¨¡åž‹ä¼˜åŒ–(åŸºäºŽStable Baselines 3ï¼‰](https://zhuanlan.zhihu.com/p/498115373)
+ [çŸ¥ä¹Ž] [FinRL-Meta: æœªæ¥é‡‘èžå¼ºåŒ–å­¦ä¹ çš„å…ƒå®‡å®™](https://zhuanlan.zhihu.com/p/544621882)
+
## Citing FinRL

```
@article{dynamic_datasets,
    author = {Liu, Xiao-Yang and Xia, Ziyi and Yang, Hongyang and Gao, Jiechao and Zha, Daochen and Zhu, Ming and Wang, Christina Dan and Wang, Zhaoran and Guo, Jian},
    title = {Dynamic Datasets and Market Environments for Financial Reinforcement Learning},
    journal = {Machine Learning - Springer Nature},
    year = {2024}
}
```


```
@article{liu2022finrl_meta,
  title={FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning},
  author={Liu, Xiao-Yang and Xia, Ziyi and Rui, Jingyang and Gao, Jiechao and Yang, Hongyang and Zhu, Ming and Wang, Christina Dan and Wang, Zhaoran and Guo, Jian},
  journal={NeurIPS},
  year={2022}
}
```

```
@article{liu2021finrl,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Gao, Jiechao and Wang, Christina Dan},
    title   = {{FinRL}: Deep reinforcement learning framework to automate trading in quantitative finance},
    journal = {ACM International Conference on AI in Finance (ICAIF)},
    year    = {2021}
}

```

```
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    title   = {{FinRL}: A deep reinforcement learning library for automated stock trading in quantitative finance},
    journal = {Deep RL Workshop, NeurIPS 2020},
    year    = {2020}
}
```

```
@article{liu2018practical,
  title={Practical deep reinforcement learning approach for stock trading},
  author={Liu, Xiao-Yang and Xiong, Zhuoran and Zhong, Shan and Yang, Hongyang and Walid, Anwar},
  journal={NeurIPS Workshop on Deep Reinforcement Learning},
  year={2018}
}
```

We published [FinRL papers](http://tensorlet.org/projects/ai-in-finance/) that are listed at [Google Scholar](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=XsdPXocAAAAJ). Previous papers are given in the [list](https://github.com/AI4Finance-Foundation/FinRL/blob/master/tutorials/FinRL_papers.md).


## Join and Contribute

Welcome to **AI4Finance** community!

Please check [Contributing Guidances](https://github.com/AI4Finance-Foundation/FinRL-Tutorials/blob/master/Contributing.md).

### Contributors

Thank you!

<a href="https://github.com/AI4Finance-LLC/FinRL-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-LLC/FinRL-Library" />
</a>


## LICENSE

MIT License
```
Trademark Disclaimer

FinRLÂ® is a registered trademark.
This license does not grant permission to use the FinRL name, logo, or related trademarks
without prior written consent, except as permitted by applicable trademark law.
For trademark inquiries or permissions, please contact: contact@finrl.ai

```

**Disclaimer: We are sharing codes for academic purpose under the MIT education license. Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**


---

## Key Components

This folder has three subfolders:
+ applications: trading tasks,
+ agents: DRL algorithms, from ElegantRL, RLlib, or Stable Baselines 3 (SB3). Users can plug in any DRL lib and play.
+ meta: market environments, we merge the stable ones from the active [FinRL-Meta repo](https://github.com/AI4Finance-Foundation/FinRL-Meta).

Then, we employ a train-test-trade pipeline by three files: train.py, test.py, and trade.py.

```
FinRL
â”œâ”€â”€ finrl (this folder)
â”‚   â”œâ”€â”€ applications
â”‚   	â”œâ”€â”€ cryptocurrency_trading
â”‚   	â”œâ”€â”€ high_frequency_trading
â”‚   	â”œâ”€â”€ portfolio_allocation
â”‚   	â””â”€â”€ stock_trading
â”‚   â”œâ”€â”€ agents
â”‚   	â”œâ”€â”€ elegantrl
â”‚   	â”œâ”€â”€ rllib
â”‚   	â””â”€â”€ stablebaseline3
â”‚   â”œâ”€â”€ meta
â”‚   	â”œâ”€â”€ data_processors
â”‚   	â”œâ”€â”€ env_cryptocurrency_trading
â”‚   	â”œâ”€â”€ env_portfolio_allocation
â”‚   	â”œâ”€â”€ env_stock_trading
â”‚   	â”œâ”€â”€ preprocessor
â”‚   	â”œâ”€â”€ data_processor.py
â”‚   	â””â”€â”€ finrl_meta_config.py
â”‚   â”œâ”€â”€ config.py
â”‚   â”œâ”€â”€ config_tickers.py
â”‚   â”œâ”€â”€ main.py
â”‚   â”œâ”€â”€ train.py
â”‚   â”œâ”€â”€ test.py
â”‚   â”œâ”€â”€ trade.py
â”‚   â””â”€â”€ plot.py
```

