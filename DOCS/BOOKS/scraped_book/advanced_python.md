---
title: "Native python support"
url: "https://www.mql5.com/en/book/advanced/python"
hierarchy: []
scraped_at: "2025-11-28 09:48:56"
---

# Native python support

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Advanced language tools](/en/book/advanced "Advanced language tools")Native python support

* [Installing Python and the MetaTrader5 package](/en/book/advanced/python/python_install "Installing Python and the MetaTrader5 package")
* [Overview of functions of the MetaTrader5 package for Python](/en/book/advanced/python/python_funcs_overview "Overview of functions of the MetaTrader5 package for Python")
* [Connecting a Python script to the terminal and account](/en/book/advanced/python/python_init "Connecting a Python script to the terminal and account")
* [Error checking: last\_error](/en/book/advanced/python/python_last_error "Error checking: last_error")
* [Getting information about a trading account](/en/book/advanced/python/python_account_info "Getting information about a trading account")
* [Getting information about the terminal](/en/book/advanced/python/python_terminal_info "Getting information about the terminal")
* [Getting information about financial instruments](/en/book/advanced/python/python_symbols "Getting information about financial instruments")
* [Subscribing to order book changes](/en/book/advanced/python/python_marketbook "Subscribing to order book changes")
* [Reading quotes](/en/book/advanced/python/python_copyrates "Reading quotes")
* [Reading tick history](/en/book/advanced/python/python_copyticks "Reading tick history")
* [Calculating margin requirements and evaluating profits](/en/book/advanced/python/python_margin_profit "Calculating margin requirements and evaluating profits")
* [Checking and sending a trade order](/en/book/advanced/python/python_ordercheck_ordersend "Checking and sending a trade order")
* [Getting the number and list of active orders](/en/book/advanced/python/python_orders "Getting the number and list of active orders")
* [Getting the number and list of open positions](/en/book/advanced/python/python_positions "Getting the number and list of open positions")
* [Reading the history of orders and deals](/en/book/advanced/python/python_history_deals "Reading the history of orders and deals")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Native python support

The potential success of automated trading largely depends on the breadth of technology that is available in the implementation of the idea. As we have already seen in the previous sections, MQL5 allows you to go beyond strictly applied trading tasks and provides opportunities for integration with external services (for example, based on network functions and custom symbols), processing and storing data using relational databases, as well as connecting arbitrary libraries.

The last point allows you to ensure interaction with any software that provides API in the DLL format. Some developers use this method to connect to industrial distributed DBMSs (instead of the built-in SQLite), math packages like R or MATLAB, and other programming languages.

Python has become one of the most popular programming languages. Its feature is a compact core, which is complemented by packages which are ready-made collections of scripts for building application solutions. Traders benefit from the wide selection and functionality of the packages for fundamental market analysis (statistical calculations, data visualization) and testing of trading hypotheses, including machine learning.

Following this trend, MQ introduced Python support in MQL5 in 2019. This tighter "out-of-the-box" integration allows the complete transfer of technical analysis and trading algorithms to the Python environment.

From a technical point of view, integration is achieved by installing the "MetaTrader5" package in Python, which organizes interprocess interaction with the terminal (at the time of writing this, through the ipykernel/RPC mechanism).

Among the functions of the package, there are full analogs of the built-in MQL5 functions for obtaining information about the terminal, trading account, symbols in Market Watch, quotes, ticks, Depth of Market, orders, positions, and deals. In addition, the package allows you to switch trading accounts, send trade orders, check margin requirements, and evaluate potential profits/losses in real-time.

However, integration with Python has some limitations. In particular, it is not possible in Python to implement event handling such as OnTick, OnBookEvent, and others. Because of this, it is necessary to use an infinite loop to check new prices, much like we were forced to do in MQL5 scripts. The analysis of the execution of trade orders is just as difficult: in the absence of OnTradeTransaction, more code would be needed to know if a position was fully or partially closed. To bypass these restrictions, you can organize the interaction of the Python script and MQL5, for example, through sockets. The mql5.com site features articles with examples of the implementation of such a bridge.

Thus, it seems that it is only natural to use Python in conjunction with MetaTrader 5 for machine learning tasks that deal with quotes, ticks, or trading account history. Unfortunately, you can't get indicator readings in Python.

[Signal service client program in MQL5](/en/book/advanced/project/project_trade_signal_client_mql5 "Signal service client program in MQL5")

[Installing Python and the MetaTrader5 package](/en/book/advanced/python/python_install "Installing Python and the MetaTrader5 package")