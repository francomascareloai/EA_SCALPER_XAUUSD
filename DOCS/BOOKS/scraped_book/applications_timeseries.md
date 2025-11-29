---
title: "Timeseries"
url: "https://www.mql5.com/en/book/applications/timeseries"
hierarchy: []
scraped_at: "2025-11-28 09:48:16"
---

# Timeseries

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")Timeseries

* [Symbols and timeframes](/en/book/applications/timeseries/timeseries_symbol_period "Symbols and timeframes")
* [Technical aspects of timeseries organization and storage](/en/book/applications/timeseries/timeseries_storage_tech "Technical aspects of timeseries organization and storage")
* [Getting characteristics of price arrays](/en/book/applications/timeseries/timeseries_properties "Getting characteristics of price arrays")
* [Number of available bars (Bars/iBars)](/en/book/applications/timeseries/timeseries_bars "Number of available bars (Bars/iBars)")
* [Search bar index by time (iBarShift)](/en/book/applications/timeseries/timeseries_ibarshift "Search bar index by time (iBarShift)")
* [Overview of Copy functions for obtaining arrays of quotes](/en/book/applications/timeseries/timeseries_copy_funcs_overview "Overview of Copy functions for obtaining arrays of quotes")
* [Getting quotes as an array of MqlRates structures](/en/book/applications/timeseries/timeseries_mqlrates "Getting quotes as an array of MqlRates structures")
* [Separate request for arrays of prices, volumes, spreads, time](/en/book/applications/timeseries/timeseries_ohlcvs "Separate request for arrays of prices, volumes, spreads, time")
* [Reading price, volume, spread, and time by bar index](/en/book/applications/timeseries/timeseries_single_value "Reading price, volume, spread, and time by bar index")
* [Finding the maximum and minimum values in a timeseries](/en/book/applications/timeseries/timeseries_highest_lowest "Finding the maximum and minimum values in a timeseries")
* [Working with real tick arrays in MqlTick structures](/en/book/applications/timeseries/timeseries_ticks_mqltick "Working with real tick arrays in MqlTick structures")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Timeseries

Time series are arrays of data in which the indexes of the elements correspond to ordered time samples. Due to the application specifics of the terminal, almost all the information a trader needs is provided in the form of time series. These include, in particular, arrays of quotes, ticks, readings of technical indicators, and others. The vast majority of MQL programs also work with this data, and therefore a group of functions in the MQL5 API has been allocated for them, which we will consider in this section.

The way of accessing arrays in MQL5 enables developers to set one of two indexing directions:

* Normal (forward) — the numbering of elements goes from the beginning of the array to the end (from old counts to new ones)
* Reverse (timeseries) — the numbering goes from the end of the array to the beginning (from new counts to old ones)

We have already covered this issue in the section [Array indexing direction as in timeseries](/en/book/common/arrays/arrays_as_series).

Changing the indexing mode is performed using the ArraySetAsSeries function and does not affect the physical layout of the array in memory. Only the way of accessing elements by number changes: in the normal indexing we get the i-th element as array[i], while in the timeseries mode the equivalent formula is array[N - i - 1], where N is the size of the array (it is called "equivalent" because the application developer does not need to do such a recalculation everywhere as it is automatically done by the terminal if the timeseries indexing mode is set for the array). This is illustrated by the following table (for a character array of 10 elements).

| | | | | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Array elements | A | B | C | D | E | F | G | H | I | J |
| Regular index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| Index as in timeseries | 9 | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |

Recall that array indexing always starts from zero.

When it comes to arrays of quotes and other constantly updated data, new elements are physically appended to the end of the array. However, from a trading point of view, the most recent data should be taken into account and taken as a starting point when analyzing history. That is why it is convenient to always have the current (last) bar under index 0, and count the previous ones from it into the past. Thus, we get the timeseries indexing.

By default, arrays are indexed from left to right. If we imagine that such an array is displayed on a standard MetaTrader 5 chart, then purely visually, the element with index 0 will be at the extreme left position and the last one at the extreme right. In timeseries with reverse indexing, the 0th element corresponds to the rightmost position, and the last element corresponds to the leftmost position. Since timeseries store the history of price data for financial instruments in relation to time, the most recent data in them is always to the right of the old ones.

The element with the zero index in the timeseries array contains information about the latest symbol quote. The zero bar is usually incomplete as it continues to form.

Another characteristic of a quote timeseries is its period, that is, the time interval between adjacent readings. This period is also called "timeframe" and can be reformulated more precisely. The timeframe is a period of time during which one bar of quotes is formed, and its beginning and end are aligned in absolute time with the same step. For example, in the "1 hour" (H1) timeframe, the bars start strictly at 0 minutes of every hour of the day. The beginning of each such period is included in the current bar, and the end belongs to the next bar.

The [Symbols and timeframes](/en/book/applications/timeseries/timeseries_symbol_period) chapter provides a complete list of standard timeframes.

Within the framework of the timeseries concept, as a rule, buffers of technical [indicators](/en/book/applications/indicators_make) also work, but we will study their features later.

If necessary, in any MQL program, you can request the values of timeseries for any symbol and timeframe, as well as the values of indicators calculated for any symbol and timeframe. This data is obtained by using [Copy](/en/book/applications/timeseries/timeseries_copy_funcs_overview) [functions](/en/book/applications/timeseries/timeseries_copy_funcs_overview), among which there are several reading arrays of prices of different types separately (for example, Open, High, Low, Close) or [MqlRates](/en/book/applications/timeseries/timeseries_mqlrates) structure arrays containing all characteristics of each bar.

Bars and ticks 
  
In addition to bars with quotes, MetaTrader 5 provides users and MQL programs with the ability to analyze ticks, which are elementary price changes, on the basis of which bars are built. Each tick contains time accurate to the millisecond, several types of prices (Bid, Ask, Last), and flags describing the essence of the changes, as well as the trading volume of the transaction. We will study the corresponding structure MqlTick a little later, in the chapter [Working with arrays of real ticks](/en/book/applications/timeseries/timeseries_ticks_mqltick). 
  
Depending on the type of trading instrument, bars can be built based on Bid or Last prices. In particular, Last prices are available for exchange-traded instruments, which also broadcast the [Depth of Market prices](/en/book/automation/marketbook). For non-exchange instruments such as Forex or CFDs, the Bid price is used. 
  
The periods during which there were no price changes do not generate bars. This is how the price is presented in MetaTrader 5. For example, if the timeframe is equal to 1 day (D1), then a couple of bars for the weekend, as a rule, are absent, and Monday immediately follows Friday. 
  
A quote bar appears if at least one tick has occurred in the corresponding time interval. At the same time, the bar opening time is always aligned strictly with the period border, even if the first tick arrived later (as it usually happens). For example, the first M1 bar of the day can be formed at 00:05 if there were no ticks for 4 minutes after midnight, and then the price change happened at 00:05:15 (that is, at the 15th second of the fifth minute). Thus, a tick is included in a particular bar based on the following ratio of timestamps: Topen <=Ttick < Topen + P, where Topen is the bar opening time, Ttick is the tick time, Topen + P is the opening time of the next potential bar after the period P ("potential" bar is called because its presence depends on other ticks).

[Restrictions for scripts and services](/en/book/applications/script_service/script_service_limitations "Restrictions for scripts and services")

[Symbols and timeframes](/en/book/applications/timeseries/timeseries_symbol_period "Symbols and timeframes")