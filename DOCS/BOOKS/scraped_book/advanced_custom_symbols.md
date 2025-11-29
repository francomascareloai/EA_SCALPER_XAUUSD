---
title: "Custom symbols"
url: "https://www.mql5.com/en/book/advanced/custom_symbols"
hierarchy: []
scraped_at: "2025-11-28 09:48:52"
---

# Custom symbols

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Advanced language tools](/en/book/advanced "Advanced language tools")Custom symbols

* [Creating and deleting custom symbols](/en/book/advanced/custom_symbols/custom_symbols_create_delete "Creating and deleting custom symbols")
* [Custom symbol properties](/en/book/advanced/custom_symbols/custom_symbols_properties "Custom symbol properties")
* [Setting margin rates](/en/book/advanced/custom_symbols/custom_symbols_margin "Setting margin rates")
* [Configuring quoting and trading sessions](/en/book/advanced/custom_symbols/custom_symbols_sessions "Configuring quoting and trading sessions")
* [Adding, replacing, and deleting quotes](/en/book/advanced/custom_symbols/custom_symbols_rates "Adding, replacing, and deleting quotes")
* [Adding, replacing, and removing ticks](/en/book/advanced/custom_symbols/custom_symbols_ticks "Adding, replacing, and removing ticks")
* [Translation of order book changes](/en/book/advanced/custom_symbols/custom_symbols_market_book "Translation of order book changes")
* [Custom symbol trading specifics](/en/book/advanced/custom_symbols/custom_symbols_trade_specifics "Custom symbol trading specifics")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Custom symbols

One of the interesting technical features of MetaTrader 5 is the support for custom financial instruments. These are the symbols that are defined not by the broker on the server side but by the trader directly in the terminal.

Custom symbols can be added to the Market Watch list along with standard symbols. The charts of such symbols with them can be used in a usual way.

The easiest way to create a custom symbol is to specify its calculation formula in the corresponding property. To do this, from the terminal interface, call the context menu in the Market Watch window, execute the Symbols command, go to the symbol hierarchy and its Custom branch, and press the Create symbol button. As a result, a dialog for setting the properties of the new symbol will open. At the same place, you can import external tick history (tab Ticks) or quotes (tab Bars) into similar tools, from files. This is discussed in detail in the MetaTrader 5 [documentation](https://www.metatrader5.com/en/terminal/help/trading_advanced/custom_instruments "MetaTrader 5 documentation").

However, the MQL5 API provides the most complete control over custom symbols.

For custom symbols, the API provides a group of functions working with [Financial instruments and Market Watch](/en/book/automation/symbols). In particular, such symbols can be listed from the program using standard functions such as SymbolsTotal, SymbolName, and SymbolInfo. We have already briefly touched on this possibility and provided an example in the section on [Custom symbol properties](/en/book/automation/symbols/symbols_custom). A distinctive feature of a custom symbol is the enabled flag (property) SYMBOL\_CUSTOM.

Using the built-in functions, you can splice Futures, generate random time series with specified characteristics, emulate renko, equal-range bars, equivolume, and other non-standard types of charts (for example, second timeframes). Also, unlike importing static files, software-controlled custom symbols can be generated in realt-time based on the data from web services such as cryptocurrency exchanges. The conversation on integrating MQL programs with the [web](/en/book/advanced/network) is still ahead, but this possibility cannot be ignored.

A custom symbol can be easily used to test strategies in the tester or as an additional method of technical analysis. However, this technology has its limitations.

Due to the fact that custom symbols are defined in the terminal and not on the server, they cannot be traded online. In particular, if you create a renko chart, trading strategies based on it will need to be adapted in one way or another so that trading signals and trades are actually separated by different symbols: artificial user and real brokerage. We will look at a couple of [solutions to the problem](/en/book/advanced/custom_symbols/custom_symbols_trade_specifics).

In addition, since the duration of all bars of one timeframe is the same in the platform, any emulation of bars with different periods (Renko, equivolume, etc.) is usually based on the smaller of the available M1 timeframes and does not provide a full time synchronization with reality. In other words, ticks belonging to such a bar are forced to have an artificial time within 60 seconds, even if a renko "brick" or a bar of a given volume actually required much more time to form. Otherwise, if we put ticks in real time, they would form the next M1 bars, violating the rules of renko or equivolume. Moreover, there are situations when a renko "brick" or other artificial bar should be created with a time interval smaller than 1 minute from the previous bar (for example, when there is increased fast volatility). In such cases, it will be necessary to change the time of historical bars in quotes of the custom instrument (shift them to the left "retroactively") or put future times on new bars (which is highly undesirable). This problem cannot be solved in a general way within the framework of user-defined symbols technology.

[Application of graphic resources in trading](/en/book/advanced/resources/resources_applied_usecase "Application of graphic resources in trading")

[Creating and deleting custom symbols](/en/book/advanced/custom_symbols/custom_symbols_create_delete "Creating and deleting custom symbols")