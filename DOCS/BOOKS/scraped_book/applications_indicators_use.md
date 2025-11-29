---
title: "Using ready-made indicators from MQL programs"
url: "https://www.mql5.com/en/book/applications/indicators_use"
hierarchy: []
scraped_at: "2025-11-28 09:48:23"
---

# Using ready-made indicators from MQL programs

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")Using ready-made indicators from MQL programs

* [Handles and counters of indicator owners](/en/book/applications/indicators_use/indicators_descriptors "Handles and counters of indicator owners")
* [A simple way to create indicator instances: iCustom](/en/book/applications/indicators_use/indicators_icustom "A simple way to create indicator instances: iCustom")
* [Checking the number of calculated bars: BarsCalculated](/en/book/applications/indicators_use/indicators_barscalculated "Checking the number of calculated bars: BarsCalculated")
* [Getting timeseries data from an indicator: CopyBuffer](/en/book/applications/indicators_use/indicators_copybuffer "Getting timeseries data from an indicator: CopyBuffer")
* [Support for multiple symbols and timeframes](/en/book/applications/indicators_use/indicators_multitimeframe "Support for multiple symbols and timeframes")
* [Overview of built-in indicators](/en/book/applications/indicators_use/indicators_standard "Overview of built-in indicators")
* [Using built-in indicators](/en/book/applications/indicators_use/indicators_standard_use "Using built-in indicators")
* [Advanced way to create indicators: IndicatorCreate](/en/book/applications/indicators_use/indicators_indicatorcreate "Advanced way to create indicators: IndicatorCreate")
* [Flexible creation of indicators with IndicatorCreate](/en/book/applications/indicators_use/indicators_flexible_create "Flexible creation of indicators with IndicatorCreate")
* [Overview of functions managing indicators on the chart](/en/book/applications/indicators_use/indicators_chart_review "Overview of functions managing indicators on the chart")
* [Combining output to main and auxiliary windows](/en/book/applications/indicators_use/indicators_chart_plus_subwindow "Combining output to main and auxiliary windows ")
* [Reading data from charts that have a shift](/en/book/applications/indicators_use/indicators_shifted "Reading data from charts that have a shift")
* [Deleting indicator instances: IndicatorRelease](/en/book/applications/indicators_use/indicators_indicatorrelease "Deleting indicator instances: IndicatorRelease")
* [Getting indicator settings by its handle](/en/book/applications/indicators_use/indicators_parameters "Getting indicator settings by its handle")
* [Defining data source for an indicator](/en/book/applications/indicators_use/indicators_apply_to "Defining data source for an indicator")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Using ready-made indicators from MQL programs

In the previous chapter, we learned how to develop custom indicators. Users can place them on charts and perform manual technical analysis with them. But this is not the only way to use indicators. MQL5 allows you to create instances of indicators and request their calculated data programmatically. This can be done both from other indicators, combining several simple ones into more complex ones, and from Expert Advisors that implement automatic or semi-automatic trading on indicator signals.

It is enough to know the indicator parameters, as well as the location and meaning of the calculated data in its public buffers, in order to organize the construction of these newly applied timeseries and gain access to them.

In this chapter, we will study the functions for creating and deleting indicators, as well as reading their buffers. This applies not only to custom indicators written in MQL5 but also to a large set of built-in indicators.

The general principles of programmatic interaction with indicators include several steps:

* Creating the indicator descriptor which is a unique identification number issued by the system in response to a certain function call ([iCustom](/en/book/applications/indicators_use/indicators_icustom) or [IndicatorCreate](/en/book/applications/indicators_use/indicators_indicatorcreate)) and through which the MQL code reports the name and parameters of the required indicator
* Reading data from the indicator buffers specified by the descriptor using the [CopyBuffer](/en/book/applications/indicators_use/indicators_copybuffer) function
* Freeing the handle ([IndicatorRelease](/en/book/applications/indicators_use/indicators_indicatorrelease)) if the indicator is no longer needed

Creating and freeing the descriptor are usually performed during program initialization and deinitialization, respectively, and buffers are read and analyzed repeatedly, as needed, for example, when ticks arrive.

In all cases, except for exotic ones, when it is required to dynamically change indicator settings during program execution, it is recommended to obtain indicator descriptors once in OnInit or in the constructor of the global object class.

All indicator creation functions have at least 2 parameters: symbol and timeframe. Instead of a symbol, you can pass NULL, which means the current instrument. Also, the value 0 corresponds to the current timeframe. Optionally, you can use built-in variables \_Symbol and \_Period. If necessary, you can set an arbitrary symbol and timeframe that are not related to the chart. Thus, in particular, it is possible to implement multi-asset and multi-timeframe indicators.

You can't access the indicator data immediately after creating its instance because the calculation of buffers takes some time. Before reading the data, you should check their readiness using the [BarsCalculated](/en/book/applications/indicators_use/indicators_barscalculated) function (it also takes a descriptor argument and returns the number of calculated bars). Otherwise, an error will be received instead of data. Although it is not critical as it does not cause the program to stop and unload, the absence of data will make the program useless.

Further in this chapter, for brevity, we will refer to the creation of instances of indicators and obtaining their descriptors simply as "creating indicators". It should be distinguished from the similar term "creating custom indicators", by which we meant writing the source code of indicators in the previous chapter.

[Creating an indicator draft in the MQL Wizard](/en/book/applications/indicators_make/indicators_wizard "Creating an indicator draft in the MQL Wizard")

[Handles and counters of indicator owners](/en/book/applications/indicators_use/indicators_descriptors "Handles and counters of indicator owners")