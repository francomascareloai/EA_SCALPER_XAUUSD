---
title: "Testing indicators"
url: "https://www.mql5.com/en/book/applications/indicators_make/indicators_test"
hierarchy: []
scraped_at: "2025-11-28 09:48:02"
---

# Testing indicators

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")[Creating custom indicators](/en/book/applications/indicators_make "Creating custom indicators")Testing indicators

* [Main characteristics of indicators](/en/book/applications/indicators_make/indicators_features "Main characteristics of indicators")
* [Main indicator event: OnCalculate](/en/book/applications/indicators_make/indicators_oncalculate "Main indicator event: OnCalculate")
* [Two types of indicators: for main window and subwindow](/en/book/applications/indicators_make/indicators_window_chart_separate "Two types of indicators: for main window and subwindow")
* [Setting the number of buffers and graphic plots](/en/book/applications/indicators_make/indicators_buffers_plots "Setting the number of buffers and graphic plots")
* [Assigning an array as a buffer: SetIndexBuffer](/en/book/applications/indicators_make/indicators_setindexbuffer "Assigning an array as a buffer: SetIndexBuffer")
* [Plot settings: PlotIndexSetInteger](/en/book/applications/indicators_make/indicators_plotindexsetinteger "Plot settings: PlotIndexSetInteger")
* [Buffer and chart mapping rules](/en/book/applications/indicators_make/indicators_buffer_to_plot_mapping "Buffer and chart mapping rules")
* [Applying directives to customize plots](/en/book/applications/indicators_make/indicators_properties "Applying directives to customize plots")
* [Setting plot names](/en/book/applications/indicators_make/indicators_labels "Setting plot names")
* [Visualizing data gaps (empty elements)](/en/book/applications/indicators_make/indicators_empty_value "Visualizing data gaps (empty elements)")
* [Indicators in separate subwindows: sizes and levels](/en/book/applications/indicators_make/indicators_separate_window "Indicators in separate subwindows: sizes and levels")
* [General properties of indicators: title and value accuracy](/en/book/applications/indicators_make/indicators_caption_digits "General properties of indicators: title and value accuracy")
* [Item-wise chart coloring](/en/book/applications/indicators_make/indicators_color "Item-wise chart coloring")
* [Skip drawing on initial bars](/en/book/applications/indicators_make/indicators_begin "Skip drawing on initial bars")
* [Waiting for data and managing visibility (DRAW\_NONE)](/en/book/applications/indicators_make/indicators_wait_none "Waiting for data and managing visibility (DRAW_NONE)")
* [Multicurrency and multitimeframe indicators](/en/book/applications/indicators_make/indicators_multisymbol "Multicurrency and multitimeframe indicators")
* [Tracking bar formation](/en/book/applications/indicators_make/indicators_newbars "Tracking bar formation")
* Testing indicators
* [Limitations and advantages of indicators](/en/book/applications/indicators_make/indicators_limitations "Limitations and advantages of indicators")
* [Creating an indicator draft in the MQL Wizard](/en/book/applications/indicators_make/indicators_wizard "Creating an indicator draft in the MQL Wizard")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Testing indicators

The built-in MetaTrader 5 tester supports two types of MQL programs: Expert Advisors and Indicators. Indicators are always tested in the visual window. But this applies only to testing an isolated indicator. If the indicator is created and called from an Expert Advisor programmatically, then this Expert Advisor together with the indicator(s) can be tested without visualization, at the user's discretion. We will study the technology of using indicators from the MQL code in the next chapter. The same technology will be used for integration with Expert Advisors.

At the same time, the indicator developer should pay attention to the fact that without visualization, the tester uses an accelerated calculation method for indicators called from Expert Advisors. The data is not calculated at every tick, but only when the relevant data is requested from indicator buffers (see the [CopyBuffer](/en/book/applications/indicators_use/indicators_copybuffer) function).

If the indicator has not yet been calculated on the current tick, it is calculated once at the first access to its data. If other requests are generated during the same tick, the calculated data is returned in the ready form. If the indicator buffers are not read on the current tick, it is not calculated. The on-demand calculation of indicators gives a significant acceleration in testing and optimization.

If a certain indicator requires precise calculations and cannot skip ticks, MQL5 can instruct the tester to enable indicator recalculation on every tick. This is done with the following directive:

| |
| --- |
| #property tester\_everytick\_calculate |

The word everytick in the directive refers specifically to the calculation of the indicator and does not affect the tick generation mode. In other words, ticks mean price changes generated by the tester, whether for every tick, for OHLC M1 prices, or for bar openings, and this tester setting remains in effect.

For the indicators that we have considered in this chapter, this property is not critical. It should also be noted that it only applies to operations in the strategy tester. In the terminal, indicators always receive OnCalculate events on each incoming tick (providing for the possibility to skip ticks if your calculations in OnCalculate take too much time and fail to complete before a new tick arrives).

As for the tester, the indicators are calculated on each tick under any of the following conditions:

* In visual mode
* If there is the tester\_everytick\_calculate directive
* If they have the [EventChartCustom](/en/book/applications/events/events_custom) call or the [OnChartEvent](/en/book/applications/events/events_onchartevent) or [OnTimer](/en/book/applications/timer/timer_ontimer) functions

Let's try to test the IndMultiSymbolMonitor.mq5 indicator from the previous section.

We select the main symbol and timeframe of the EURUSD, H1 chart. The tick generation method is "based on real ticks".

After starting the test, we should see the following entries in the log of the visual mode window:

| |
| --- |
| 2021.10.20 00:00:00   New bar(s) on: EURUSD USDCHF USDJPY , in-sync:false 2021.10.20 00:00:00   New bar(s) on: AUDUSD , in-sync:false 2021.10.20 00:00:00   New bar(s) on: GBPUSD , in-sync:false 2021.10.20 00:00:02   New bar(s) on: USDCAD , in-sync:false 2021.10.20 00:00:11   New bar(s) on: NZDUSD , in-sync:true 2021.10.20 01:00:04   New bar(s) on: EURUSD GBPUSD USDCHF USDJPY AUDUSD USDCAD NZDUSD , in-sync:true 2021.10.20 02:00:00   New bar(s) on: EURUSD USDJPY NZDUSD , in-sync:false 2021.10.20 02:00:00   New bar(s) on: USDCHF , in-sync:false 2021.10.20 02:00:01   New bar(s) on: AUDUSD , in-sync:false 2021.10.20 02:00:15   New bar(s) on: GBPUSD USDCAD , in-sync:true 2021.10.20 03:00:00   New bar(s) on: EURUSD AUDUSD NZDUSD , in-sync:false 2021.10.20 03:00:00   New bar(s) on: GBPUSD USDJPY USDCAD , in-sync:false 2021.10.20 03:00:12   New bar(s) on: USDCHF , in-sync:true |

As you can see, new bars appear on different symbols gradually. Usually, several events occur before the "in-sync" flag set to true appears.

You can run testing for other indicators of this chapter as well. Please note that if an MQL program queries the history of ticks, select the generation method "based on real ticks" in the tester.

Testing "by open prices" can only be used for indicators and Expert Advisors that are developed with support for this mode, for example, they calculate only by Open prices or analyze completed bars starting from the 1st one.

Attention! When testing indicators in the tester, the OnDeinit event does not work. Moreover, other finalization is not performed, for example, destructors of global objects are not called.

[Tracking bar formation](/en/book/applications/indicators_make/indicators_newbars "Tracking bar formation")

[Limitations and advantages of indicators](/en/book/applications/indicators_make/indicators_limitations "Limitations and advantages of indicators")