---
title: "Creating Expert Advisors"
url: "https://www.mql5.com/en/book/automation/experts"
hierarchy: []
scraped_at: "2025-11-28 09:48:11"
---

# Creating Expert Advisors

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Trading automation](/en/book/automation "Trading automation")Creating Expert Advisors

* [Expert Advisors main event: OnTick](/en/book/automation/experts/experts_ontick "Expert Advisors main event: OnTick")
* [Basic principles and concepts: order, deal, and position](/en/book/automation/experts/experts_order_deal_position "Basic principles and concepts: order, deal, and position")
* [Types of trading operations](/en/book/automation/experts/experts_request_types "Types of trading operations")
* [Order types](/en/book/automation/experts/experts_order_type "Order types")
* [Order execution modes by price and volume](/en/book/automation/experts/experts_execution_filling "Order execution modes by price and volume")
* [Pending order expiration dates](/en/book/automation/experts/experts_pending_expiration "Pending order expiration dates")
* [Margin calculation for a future order: OrderCalcMargin](/en/book/automation/experts/experts_ordercalcmargin "Margin calculation for a future order: OrderCalcMargin")
* [Estimating the profit of a trading operation: OrderCalcProfit](/en/book/automation/experts/experts_ordercalcprofit "Estimating the profit of a trading operation: OrderCalcProfit")
* [MqlTradeRequest structure](/en/book/automation/experts/experts_mqltraderequest "MqlTradeRequest structure")
* [MqlTradeCheckResult structure](/en/book/automation/experts/experts_mqltradecheckresult "MqlTradeCheckResult structure")
* [Request validation: OrderCheck](/en/book/automation/experts/experts_ordercheck "Request validation: OrderCheck")
* [Request sending result: MqlTradeResult structure](/en/book/automation/experts/experts_mqltraderesult "Request sending result: MqlTradeResult structure")
* [Sending a trade request: OrderSend and OrderSendAsync](/en/book/automation/experts/experts_ordersend_ordersendasync "Sending a trade request: OrderSend and OrderSendAsync")
* [Buying and selling operations](/en/book/automation/experts/experts_market_buy_sell "Buying and selling operations")
* [Modying Stop Loss and/or Take Profit levels of a position](/en/book/automation/experts/experts_modify_position "Modying Stop Loss and/or Take Profit levels of a position")
* [Trailing stop](/en/book/automation/experts/experts_trailing_stop "Trailing stop")
* [Closing a position: full and partial](/en/book/automation/experts/experts_close "Closing a position: full and partial ")
* [Closing opposite positions: fill and partial](/en/book/automation/experts/experts_closeby "Closing opposite positions: fill and partial")
* [Placing a pending order](/en/book/automation/experts/experts_pending "Placing a pending order")
* [Modifying a pending order](/en/book/automation/experts/experts_modify_order "Modifying a pending order")
* [Deleting a pending order](/en/book/automation/experts/experts_remove_order "Deleting a pending order")
* [Getting a list of active orders](/en/book/automation/experts/experts_order_list "Getting a list of active orders")
* [Order properties (active and historical)](/en/book/automation/experts/experts_order_properties "Order properties (active and historical)")
* [Functions for reading properties of active orders](/en/book/automation/experts/experts_orderget_funcs "Functions for reading properties of active orders")
* [Selecting orders by properties](/en/book/automation/experts/experts_order_filter "Selecting orders by properties")
* [Getting the list of positions](/en/book/automation/experts/experts_position_list "Getting the list of positions")
* [Position properties](/en/book/automation/experts/experts_position_properties "Position properties")
* [Functions for reading position properties](/en/book/automation/experts/experts_positionget_funcs "Functions for reading position properties")
* [Deal properties](/en/book/automation/experts/experts_deal_properties "Deal properties")
* [Selecting orders and deals from history](/en/book/automation/experts/experts_history_select "Selecting orders and deals from history")
* [Functions for reading order properties from history](/en/book/automation/experts/experts_historyorderget_funcs "Functions for reading order properties from history")
* [Functions for reading deal properties from history](/en/book/automation/experts/experts_historydealget_funcs "Functions for reading deal properties from history")
* [Types of trading transactions](/en/book/automation/experts/experts_transaction_type "Types of trading transactions")
* [OnTradeTransaction event](/en/book/automation/experts/experts_ontradetransaction "OnTradeTransaction event")
* [Synchronous and asynchronous requests](/en/book/automation/experts/experts_sync_vs_async "Synchronous and asynchronous requests")
* [OnTrade event](/en/book/automation/experts/experts_ontrade "OnTrade event")
* [Monitoring trading environment changes](/en/book/automation/experts/experts_trade_state "Monitoring trading environment changes")
* [Creating multi-symbol Expert Advisors](/en/book/automation/experts/experts_multisymbol "Creating multi-symbol Expert Advisors")
* [Limitations and benefits of Expert Advisors](/en/book/automation/experts/experts_limitations "Limitations and benefits of Expert Advisors")
* [Creating Expert Advisors in the MQL Wizard](/en/book/automation/experts/experts_wizard "Creating Expert Advisors in the MQL Wizard")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Creating Expert Advisors

In this chapter, we begin to study the MQL5 trading API used to implement Expert Advisors. This type of program is perhaps the most complex and demanding in terms of error-free coding and the number and variety of technologies involved. In particular, we will need to utilize many of the skills acquired from the previous chapters, ranging from OOP to the applied aspects of working with graphical objects, indicators, symbols, and software environment settings.

Depending on the chosen trading strategy, the Expert Advisor developer may need to pay special attention to the following:

* Decision-making and order-sending speed (for HFT, High-Frequency Trading)
* Selecting the optimal portfolio of instruments based on their correlations and volatility (for cluster trading)
* Dynamically calculating lots and distance between orders (for martingale and grid strategies)
* Analysis of news or external data sources (this will be discussed in the 7th part of the book)

All such features should be optimally applied by the developer to the described trading mechanisms provided by the MQL5 API.

Next, we will consider in detail built-in functions for managing trading activity, the Expert Advisor event model, and specific data structures, and recall the basic principles of interaction between the terminal and the server, as well as the basic concepts for algorithmic trading in MetaTrader 5: order, deal, and position.

At the same time, due to the versatility of the material, many important nuances of Expert Advisor development, such as testing and optimization, are highlighted in the next chapter.

We have previously considered the [Design of MQL programs of various types](/en/book/applications/runtime/runtime_features_by_progtype), including Expert Advisors, as well as started [Features of starting and stopping programs](/en/book/applications/runtime/runtime_lifecycle). Despite the fact that an Expert Advisor is launched on a specific chart, for which a working symbol is defined, there are no obstacles to centrally manage trading of an arbitrary set of financial instruments. Such Expert Advisors are traditionally referred to as multicurrency, although in fact, their portfolio may include CFDs, stocks, commodities, and tickers of other markets.

In Expert Advisors, as well as in indicators, there are [Key events](/en/book/applications/runtime/runtime_oninit_ondeinit) [OnInit](/en/book/applications/runtime/runtime_oninit_ondeinit) [and](/en/book/applications/runtime/runtime_oninit_ondeinit) [OnDeinit](/en/book/applications/runtime/runtime_oninit_ondeinit). They are not mandatory, but, as a rule, they are present in the code for the preparation and regular completion of the program: we used them and will continue using them in the examples. In a separate section, we provided an [Overview of all event handling functions](/en/book/applications/runtime/runtime_events_overview): we have already studied some of them in detail by now (for example, [OnCalculate](/en/book/applications/indicators_make/indicators_oncalculate) indicator events and the [OnTimer](/en/book/applications/timer/timer_ontimer) timer).  Expert Advisor-specific events ([OnTick](/en/book/automation/experts/experts_ontick), [ontrade](/en/book/automation/experts/experts_ontrade), [OnTradeTransaction](/en/book/automation/experts/experts_ontradetransaction)) will be described in this chapter.

Expert Advisors can use the widest range of source data as trading signals: [quotes](/en/book/applications/timeseries/timeseries_copy_funcs_overview), [ticks](/en/book/applications/timeseries/timeseries_ticks_mqltick), [depth of market](/en/book/automation/marketbook), [trading account history](/en/book/automation/experts/experts_history_select), or indicator readings. In the latter case, the principles of creating indicator instances and reading values from their buffers are no different from those discussed in the chapter [Using ready-made indicators from MQL programs](/en/book/applications/indicators_use). In the Expert Advisor examples in the following sections, we will demonstrate most of these tricks.

It should be noted that trading functions can be used not only in Expert Advisors but also in scripts. We will see examples for both options.

[Current financial performance of the account](/en/book/automation/account/account_state "Current financial performance of the account")

[Expert Advisors main event: OnTick](/en/book/automation/experts/experts_ontick "Expert Advisors main event: OnTick")