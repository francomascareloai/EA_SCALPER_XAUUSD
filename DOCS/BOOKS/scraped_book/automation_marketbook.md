---
title: "Depth of market"
url: "https://www.mql5.com/en/book/automation/marketbook"
hierarchy: []
scraped_at: "2025-11-28 09:48:05"
---

# Depth of market

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Trading automation](/en/book/automation "Trading automation")Depth of Market

* [Managing subscriptions to Depth of Market events](/en/book/automation/marketbook/marketbook_add_release "Managing subscriptions to Depth of Market events")
* [Receiving events about changes in the Depth of Market](/en/book/automation/marketbook/marketbook_events "Receiving events about changes in the Depth of Market")
* [Reading the current Depth of Market data](/en/book/automation/marketbook/marketbook_get "Reading the current Depth of Market data")
* [Using Depth of Market data in applied algorithms](/en/book/automation/marketbook/marketbook_application "Using Depth of Market data in applied algorithms")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Depth of market

In addition to several types of up-to-date market price data (Ask/Bid/Last) and the last traded volumes are received in the terminal in the form of [ticks](/en/book/automation/symbols/symbols_tick), MetaTrader 5 supports the Depth of Market (order book), which is an array of records about the volumes of placed buy and sell orders around the current market price. Volumes are aggregated at several levels above and below the current price, with the smallest increment of price movement according to the symbol specification. As we have seen, the maximum order book size (number of price levels) is set in the [SYMBOL\_TICKS\_BOOKDEPTH](/en/book/automation/symbols/symbols_market_depth) symbol property.

Terminal users know the Depth of Market feature in the interface and its operating principles. If you need further details, please see the [documentation](https://www.metatrader5.com/en/terminal/help/trading/depth_of_market "MetaTrader 5 documentation").

The order book contains extended market information which is commonly referred to as "market depth". Knowing it allows you to create more sophisticated trading systems.

Indeed, information about a tick is only a small slice of the order book. In a somewhat simplified sense, a tick is a 2-level order book with one nearest Ask price (available offer) and one nearest Bid price (available demand). Furthermore, ticks do not provide order volumes at these prices.

Depth of Market changes can occur much more frequently than ticks, since they affect not only the reaction to concluded deals but also changes in the volume of pending limit orders in the Depth of Market.

Usually, data providers for the order book and quotes (ticks, deals) are different instances, and tick events ([OnTick](/en/book/automation/experts/experts_ontick) in Expert Advisors or [OnCalculate](/en/book/applications/indicators_make/indicators_oncalculate) in indicators) do not match the Depth of Market events. Both threads arrive asynchronously and in parallel but eventually end up in the [event queue](/en/book/applications/runtime/runtime_events_overview) of an MQL program.

It is important to note that an order book is available, as a rule, for exchange instruments, but there are exceptions both in one direction and in the other:

* Depth of Market may be missing for one reason or another for an exchange instrument
* Depth of Market can be provided by a broker for an OTC instrument based on the information they have collected about their clients' orders

In MQL5, Depth of Market data is available for Expert Advisors and indicators. By using special functions ([MarketBookAdd](/en/book/automation/marketbook/marketbook_add_release), [MarketBookRelease](/en/book/automation/marketbook/marketbook_add_release)), programs can enable or disable their subscription to receive notifications about Depth of Market changes in the platform. To receive the notifications, the program must define the [OnBookEvent](/en/book/automation/marketbook/marketbook_events) event handler function in its code. After receiving a notification, the order book data can be read using the [MarketBookGet](/en/book/automation/marketbook/marketbook_get) function.

The terminal maintains the history of quotes and ticks, but not of the Depth of Market data. In particular, the user or an MQL program can download the history at the required retrospective (if the broker has it) and test Expert Advisors and indicators on it. 
  
In contrast, the Depth of Market is only broadcast online and is not available in the tester. A broker does not have an archive of Depth of Market data on the server. To emulate the behavior of the order book in the tester, you should collect the Depth of Market history online and then read it from the MQL program running in the tester. You can find ready-made products in the MQL5 Market.

[Specific properties (stock exchange, derivatives, bonds)](/en/book/automation/symbols/symbols_special "Specific properties (stock exchange, derivatives, bonds)")

[Managing subscriptions to Depth of Market events](/en/book/automation/marketbook/marketbook_add_release "Managing subscriptions to Depth of Market events")