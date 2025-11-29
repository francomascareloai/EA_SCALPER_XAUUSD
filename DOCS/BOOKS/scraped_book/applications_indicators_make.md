---
title: "Creating custom indicators"
url: "https://www.mql5.com/en/book/applications/indicators_make"
hierarchy: []
scraped_at: "2025-11-28 09:48:25"
---

# Creating custom indicators

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")Creating custom indicators

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
* [Testing indicators](/en/book/applications/indicators_make/indicators_test "Testing indicators")
* [Limitations and advantages of indicators](/en/book/applications/indicators_make/indicators_limitations "Limitations and advantages of indicators")
* [Creating an indicator draft in the MQL Wizard](/en/book/applications/indicators_make/indicators_wizard "Creating an indicator draft in the MQL Wizard")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Creating custom indicators

Indicators are one of the most popular types of MQL programs. They are a simple yet powerful tool for technical analysis. The main mechanism of their use is the processing of the initial price data using formulas for creating derivative timeseries. This enables the evaluation and visualization of specific characteristics of market processes. Any timeseries, including those obtained as a result of indicator calculations, can be fed into another indicator, and so on. Formulas of many well-known indicators (for example, [MACD](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "MetaTrader 5 Help")) actually consist of calls to several interrelated indicators.

Terminal users are undoubtedly familiar with many built-in indicators, and they also know that the list of available indicators can be expanded using the MQL5 language. From the user's point of view, built-in and custom indicators implemented in MQL5 work in exactly the same way.

As a rule, indicators display their operation results in the form of lines, histograms, and other graphical constructions in the price chart window. Each such chart is visualized on the basis of calculated timeseries, which are stored inside the indicators in special arrays called indicator buffers: they are available for viewing in the terminal Data Window along with the OHLC prices. However, indicators can provide extra functionality in addition to buffers or may have no buffers at all. For example, indicators are often used to solve problems where you need to create [graphic objects](/en/book/applications/objects), manage the chart and its [properties](/en/book/applications/charts/charts_properties_overview), and interact with the user (see [OnChartEvent](/en/book/applications/events/events_onchartevent)).

In this chapter we will study the basic principles of creating indicators in MQL5. Such indicators are usually called "custom" because the user can write them from scratch or compile them from ready-made source codes. In the next chapter, we will turn to the issues of programmatic management of custom and built-in indicators, which will allow us to construct more complex indicators and pave the way for indicator-based trading signals and filters for Expert Advisors.

A little later, we will master the technology of introducing indicators into executable MQL programs in the form of [resources](/en/book/advanced/resources).

[Working with real tick arrays in MqlTick structures](/en/book/applications/timeseries/timeseries_ticks_mqltick "Working with real tick arrays in MqlTick structures")

[Main characteristics of indicators](/en/book/applications/indicators_make/indicators_features "Main characteristics of indicators")