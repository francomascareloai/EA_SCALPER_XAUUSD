---
title: "Expert Advisors main event: OnTick"
url: "https://www.mql5.com/en/book/automation/experts/experts_ontick"
hierarchy: []
scraped_at: "2025-11-28 09:48:07"
---

# Expert Advisors main event: OnTick

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Trading automation](/en/book/automation "Trading automation")[Creating Expert Advisors](/en/book/automation/experts "Creating Expert Advisors")Expert Advisors main event: OnTick

* Expert Advisors main event: OnTick
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

# Expert Advisors main event: OnTick

The OnTick event is generated by the terminal for Expert Advisors when a new tick appears containing the price of the current chart's working symbol on which the Expert Advisor is running. To handle this event, the OnTick function must be defined in the Expert Advisor code. It has the following prototype.

void OnTick(void)

As you can see, the function has no parameters. If necessary, the very value of the new price and other tick characteristics should be requested by calling [SymbolInfoTick](/en/book/automation/symbols/symbols_tick).

From the point of view of the reaction to the new tick event, this handler is similar to OnCalculate in indicators. However, OnCalculate can only be defined in indicators, and OnTick only in Expert Advisors (to be more precise, the OnTick function in the code of an indicator, script, or service will be simply ignored).

At the same time, the Expert Advisor does not have to contain the OnTick handler. In addition to this event, Expert Advisors can process the [OnTimer](/en/book/applications/timer/timer_ontimer), [OnBookEvent](/en/book/automation/marketbook/marketbook_events), and [OnChartEvent](/en/book/applications/events/events_onchartevent) events and perform all necessary trading operations from them.

All events in Expert Advisors are processed one after the other in the order they arrive, since Expert Advisors, like all other MQL programs, are single-threaded. If there is already an OnTick event in the queue or such an event is being processed, then new OnTick events are not queued.

An OnTick event is generated regardless of whether automatic trading is disabled or enabled (Algo trading button in the terminal interface). Disabled automatic trading means only restriction on sending trade requests from the Expert Advisors but does not prevent the Expert Advisor from running.

It should be remembered that tick events are generated only for one symbol, which is the symbol of the current chart. If the Expert Advisor is multicurrency, then getting ticks from other symbols should be organized in some alternative way, for example, using a spy indicator [EventTickSpy.mq5](/en/book/applications/events/events_custom) or subscription to market book events, as in [MarketBookQuasiTicks.mq5](/en/book/automation/marketbook/marketbook_application).

As a simple example, consider the Expert Advisor ExpertEvents.mq5. It defines handlers for all events that are usually used to launch trading algorithms. We will study some other events ([OnTrade](/en/book/automation/experts/experts_ontrade), [OnTradeTransaction](/en/book/automation/experts/experts_ontradetransaction), as well as tester events) later.

All handlers call the display helper function which outputs the current time (millisecond system counter label) and handler name in a multi-line comment.

| |
| --- |
| #define N\_LINES 25 #include <MQL5Book/Comments.mqh>     void Display(const string message) {    ChronoComment((string)GetTickCount() + ": " + message); } |

The OnTick event will be called automatically upon the arrival of new ticks. For timer and order book events, you need to activate the corresponding handlers using EventSetTimer and MarketBookAdd calls from OnInit.

| |
| --- |
| void OnInit() {    Print(\_\_FUNCTION\_\_);    EventSetTimer(2);    if(!MarketBookAdd(\_Symbol))    {       Print("MarketBookAdd failed:", \_LastError);    } }     void OnTick() {    Display(\_\_FUNCTION\_\_); }     void OnTimer() {    Display(\_\_FUNCTION\_\_); }     void OnBookEvent(const string &symbol) {    if(symbol == \_Symbol) // react only to order book of "our" symbol    {       Display(\_\_FUNCTION\_\_);    } } |

The chart change event is also available: it can be used to trade on markup based on graphical objects, by pressing buttons or hotkeys, as well as upon the arrival of custom events from other programs, for example, indicators like EventTickSpy.mq5.

| |
| --- |
| void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam) {    Display(\_\_FUNCTION\_\_); }     void OnDeinit(const int) {    Print(\_\_FUNCTION\_\_);    MarketBookRelease(\_Symbol);    Comment(""); } |

The following screenshot shows the result of the Expert Advisor operation on the chart.

![Comments with events of various types in the expert](/en/book/img/usdcnhm1_events.png "Comments with events of various types in the expert") 
Comments with events of various types in the Expert Advisor

Please note that the OnBookEvent event (if it is broadcast for a symbol) arrives more often than OnTick.

[Creating Expert Advisors](/en/book/automation/experts "Creating Expert Advisors")

[Basic principles and concepts: order, deal, and position](/en/book/automation/experts/experts_order_deal_position "Basic principles and concepts: order, deal, and position")