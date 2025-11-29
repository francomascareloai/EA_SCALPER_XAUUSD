---
title: "Working with charts"
url: "https://www.mql5.com/en/book/applications/charts"
hierarchy: []
scraped_at: "2025-11-28 09:48:17"
---

# Working with charts

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")Working with charts

* [Functions for getting the basic properties of the current chart](/en/book/applications/charts/charts_main_properties "Functions for getting the basic properties of the current chart")
* [Chart identification](/en/book/applications/charts/charts_id "Chart identification")
* [Getting the list of charts](/en/book/applications/charts/charts_list "Getting the list of charts")
* [Getting the symbol and timeframe of an arbitrary chart](/en/book/applications/charts/charts_symbol_period "Getting the symbol and timeframe of an arbitrary chart")
* [Overview of functions for working with the complete set of properties](/en/book/applications/charts/charts_properties_overview "Overview of functions for working with the complete set of properties")
* [Descriptive chart properties](/en/book/applications/charts/charts_string_properties "Descriptive chart properties")
* [Checking the status of the main window](/en/book/applications/charts/charts_window_state "Checking the status of the main window")
* [Getting the number and visibility of windows/subwindows](/en/book/applications/charts/charts_count_visibility "Getting the number and visibility of windows/subwindows")
* [Chart display modes](/en/book/applications/charts/charts_mode "Chart display modes")
* [Managing the visibility of chart elements](/en/book/applications/charts/charts_show_elements "Managing the visibility of chart elements")
* [Horizontal shifts](/en/book/applications/charts/charts_shift "Horizontal shifts")
* [Horizontal scale (by time)](/en/book/applications/charts/charts_scale_time "Horizontal scale (by time)")
* [Vertical scale (by price and indicator readings)](/en/book/applications/charts/charts_scale_price "Vertical scale (by price and indicator readings)")
* [Colors](/en/book/applications/charts/charts_color "Colors")
* [Mouse and keyboard control](/en/book/applications/charts/charts_keyboard_mouse "Mouse and keyboard control")
* [Undocking chart window](/en/book/applications/charts/charts_floating "Undocking chart window")
* [Getting MQL program drop coordinates on a chart](/en/book/applications/charts/charts_on_drop "Getting MQL program drop coordinates on a chart")
* [Translation of screen coordinates to time/price and vice versa](/en/book/applications/charts/charts_coordinates "Translation of screen coordinates to time/price and vice versa")
* [Scrolling charts along the time axis](/en/book/applications/charts/charts_navigate "Scrolling charts along the time axis")
* [Chart redraw request](/en/book/applications/charts/charts_redraw "Chart redraw request")
* [Switching symbol and timeframe](/en/book/applications/charts/charts_set_symbol_period "Switching symbol and timeframe")
* [Managing indicators on the chart](/en/book/applications/charts/charts_indicators "Managing indicators on the chart")
* [Opening and closing charts](/en/book/applications/charts/charts_open_close "Opening and closing charts")
* [Working with tpl chart templates](/en/book/applications/charts/charts_tpl "Working with tpl chart templates")
* [Saving a chart image](/en/book/applications/charts/charts_screenshot "Saving a chart image")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Working with charts

Most MQL programs, such as scripts, indicators, and Expert Advisors, are executed on charts. Only services run in the background, without being tied to a schedule. A rich set of functions is provided for obtaining and changing the properties of graphs, analyzing their list, and searching for other running programs.

Since charts are the natural environment for indicators, we have already had a chance to get acquainted with some of these features in the previous indicator chapters. In this chapter, we will study all these functions in a targeted manner.

When working with charts, we will use the concept of a window. A window is a dedicated area that displays price charts and/or indicator charts. The top and, as a rule, the largest window contains price charts, has the number 0, and always exists. All additional windows added to the lower part when placing indicators are numbered from 1 and higher (numbering from top to bottom). Each subwindow exists only as long as it has at least one indicator.

Since the user can delete all indicators in an arbitrary subwindow, including the one that is not the last (the lowest), the indexes of the remaining subwindows can decrease.

The event model of charts related to receiving and processing notifications about events on charts and generating custom events will be discussed in a [separate chapter](/en/book/applications/events).

In addition to the "charts in windows" discussed here, MetaTrader 5 also allows you to create "charts in objects". We will deal with [graphical objects](/en/book/applications/objects) in the next chapter.

[High-precision timer: EventSetMillisecondTimer](/en/book/applications/timer/timer_event_set_millisecond "High-precision timer: EventSetMillisecondTimer")

[Functions for getting the basic properties of the current chart](/en/book/applications/charts/charts_main_properties "Functions for getting the basic properties of the current chart")