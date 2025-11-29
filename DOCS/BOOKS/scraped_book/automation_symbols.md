---
title: "Financial instruments and Market Watch"
url: "https://www.mql5.com/en/book/automation/symbols"
hierarchy: []
scraped_at: "2025-11-28 09:48:10"
---

# Financial instruments and Market Watch

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Trading automation](/en/book/automation "Trading automation")Financial instruments and Market Watch

* [Getting available symbols and Market Watch lists](/en/book/automation/symbols/symbols_list "Getting available symbols and Market Watch lists")
* [Editing the Market Watch list](/en/book/automation/symbols/symbols_select "Editing the Market Watch list")
* [Checking if a symbol exists](/en/book/automation/symbols/symbols_exist "Checking if a symbol exists")
* [Checking the symbol data relevance](/en/book/automation/symbols/symbols_sync "Checking the symbol data relevance")
* [Getting the last tick of a symbol](/en/book/automation/symbols/symbols_tick "Getting the last tick of a symbol")
* [Schedules of trading and quoting sessions](/en/book/automation/symbols/symbols_sessions "Schedules of trading and quoting sessions")
* [Symbol margin rates](/en/book/automation/symbols/symbols_margin_rates "Symbol margin rates")
* [Overview of functions for getting symbol properties](/en/book/automation/symbols/symbols_info "Overview of functions for getting symbol properties")
* [Checking symbol status](/en/book/automation/symbols/symbols_state "Checking symbol status")
* [Price type for building symbol charts](/en/book/automation/symbols/symbols_chart_mode "Price type for building symbol charts")
* [Base, quote, and margin currencies of the instrument](/en/book/automation/symbols/symbols_currencies "Base, quote, and margin currencies of the instrument")
* [Price representation accuracy and change steps](/en/book/automation/symbols/symbols_point_tick "Price representation accuracy and change steps")
* [Permitted volumes of trading operations](/en/book/automation/symbols/symbols_volume "Permitted volumes of trading operations")
* [Trading permission](/en/book/automation/symbols/symbols_trade_mode "Trading permission")
* [Symbol trading conditions and order execution modes](/en/book/automation/symbols/symbols_execution_filling "Symbol trading conditions and order execution modes")
* [Margin requirements](/en/book/automation/symbols/symbols_margin "Margin requirements")
* [Pending order expiration rules](/en/book/automation/symbols/symbols_expiration "Pending order expiration rules")
* [Spreads and order distance from the current price](/en/book/automation/symbols/symbols_spreads_levels "Spreads and order distance from the current price")
* [Getting swap sizes](/en/book/automation/symbols/symbols_swaps "Getting swap sizes")
* [Current market information (tick)](/en/book/automation/symbols/symbols_tick_parts "Current market information (tick)")
* [Descriptive symbol properties](/en/book/automation/symbols/symbols_description "Descriptive symbol properties")
* [Depth of Market](/en/book/automation/symbols/symbols_market_depth "Depth of Market")
* [Custom symbol properties](/en/book/automation/symbols/symbols_custom "Custom symbol properties")
* [Specific properties (stock exchange, derivatives, bonds)](/en/book/automation/symbols/symbols_special "Specific properties (stock exchange, derivatives, bonds)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Financial instruments and Market Watch

MetaTrader 5 allows users to analyze and trade financial instruments (a.k.a. symbols or tickers), which form the basis of almost all terminal subsystems. Charts, indicators, and the price history of quotes exist in relation to trading symbols. The main functionality of the terminal is built on financial instruments such as trading orders, deals, control of margin requirements, and trading account history.

Via the terminal, brokers deliver to traders a specified list of symbols, from which each user chooses the preferred ones, forming the Market Watch. The Market Watch window determines the symbols for which the terminal requests online quotes and allows you to open charts and view the history.

The MQL5 API provides similar software tools that allow you to view and analyze the characteristics of all symbols, add them to the Market Watch, or exclude them from there.

In addition to standard symbols with information provided by brokers, MetaTrader 5 makes it possible to create custom symbols: their properties and price history can be loaded from arbitrary data sources and calculated using formulas or MQL programs. Custom symbols also participate in the Market Watch and can be used for [testing strategies](/en/book/automation/tester) and technical analysis, however, they also have a natural limitation — they cannot be traded online using regular MQL5 API tools, since these symbols are not available on the server. [Custom symbols](/en/book/advanced/custom_symbols) will be reviewed in a separate chapter, in the last, seventh part of the book.

A little while ago, in the relevant chapters, we have already touched on [time series](/en/book/applications/timeseries) with price data of individual symbols, including history paging using an example with [indicators](/en/book/applications/indicators_make/indicators_multisymbol). All this functionality actually assumes that the corresponding symbols are already enabled in the Market Watch. This is especially true for multicurrency indicators and Expert Advisors that refer not only to the working symbol of the chart but also to other symbols. In this chapter, we will learn how the Market Watch list is managed from MQL programs.

The chapter on charts has already described some of the symbol properties made available through [basic property-getter functions](/en/book/applications/charts/charts_main_properties) of a current chart (Point, Digits) since the chart cannot work without the symbol associated with it. Now we will study most of the properties of symbols, including their specification. Their full set can be found in the [MQL5 documentation on the website](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants "Information about instrument ").

[Trading automation](/en/book/automation "Trading automation")

[Getting available symbols and Market Watch lists](/en/book/automation/symbols/symbols_list "Getting available symbols and Market Watch lists")