---
title: "Functions for working with time"
url: "https://www.mql5.com/en/book/common/timing"
hierarchy: []
scraped_at: "2025-11-28 09:49:14"
---

# Functions for working with time

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Functions for working with time

* [Local and server time](/en/book/common/timing/timing_local_server "Local and server time")
* [Daylight saving time (local)](/en/book/common/timing/timing_daylight_saving "Daylight saving time (local)")
* [Universal Time](/en/book/common/timing/timing_gmt "Universal Time")
* [Pausing a program](/en/book/common/timing/timing_sleep "Pausing a program")
* [Time interval counters](/en/book/common/timing/timing_count "Time interval counters")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Functions for working with time

Time is a fundamental factor in most processes, and plays an important applied role for trading.

As we know, the main coordinate system in trading is based on two dimensions: price and time. They are displayed on the chart along the vertical and horizontal axes, respectively. Later, we will touch on another important axis, which can be represented as being perpendicular to the first two and going deep into the chart, on which trading volumes are marked. But for now, let's focus on time.

This measurement is common to all charts, uses the same units of measurement, and, strange as it may sound, is characterized by constancy (the course of time is predictable).

The terminal provides a plethora of built-in tools related to the calculation and analysis of time. So, we will get acquainted with them gradually, as we move through the chapters of the book, from simple to complex.

In this chapter, we will study the functions that allow you to control the time and pause the program activity for a specified interval.

In the [Date and time](/en/book/common/conversions/conversions_datetime) chapter, in the section on data transformation, we already saw a couple of functions related to time: TimeToStruct and StructToTime. They split a value of the datetime type into components or vice versa, construct datetime from individual fields: recall that they are summarized in the MqlDateTime structure.

| |
| --- |
| struct MqlDateTime {     int year;        // year (1970 – 3000)    int mon;         // month (1 – 12)     int day;         // day (1 – 31)     int hour;        // hours (0 – 23)     int min;         // minutes (0 – 59)     int sec;         // seconds (0 – 59)     int day\_of\_week; // day of the week, numbered from 0 (Sunday) to 6 (Saturday)                     // according to enum ENUM\_DAY\_OF\_WEEK    int day\_of\_year; // ordinal number of the day in the year, starting from 0 (January 1) }; |

But where can an MQL program get the datetime value from?

For example, historical prices and times are reflected in quotes, while current live data arrives as ticks. Both have timestamps, which we will learn how to get in the relevant sections: about [timeseries](/en/book/applications/timeseries) and [terminal events](/en/book/applications/events). However, an MQL program can query the current time by itself (without prices or other trading information) using several functions.

Several functions were required because the system is distributed: it consists of a client terminal and a broker server located in arbitrary parts of the world, which, quite likely, belong to different time zones.

Any time zone is characterized by a temporary offset relative to the global reference point of time, Greenwich Mean Time (GMT). As a rule, a time zone offset is an integer number of hours N (although there are also exotic zones with a half-hour step) and therefore it is indicated as GMT + N or GMT-N, depending on whether the zone is east or west of the meridian. For example, Continental Europe, located east of London, uses Central European Time (CET) equal to GMT + 1, or Eastern European Time (Eastern European Time, EET) equal to GMT + 2, while in America there are "negative" zones, such like Eastern Standard Time (EST) or GMT-5.

It should be noted that GMT corresponds to astronomical (solar) time, which is slightly non-linear as the Earth's rotation is gradually slowing down. In this regard, in recent decades, there has actually been a transition to a more accurate timekeeping system (based on atomic clocks), in which global time is called Coordinated Universal Time (UTC). In many application areas, including trading, the difference between GMT and UTC is not significant, so the time zone designations in the new UTC±N format and the old GMT±N should be considered analogs. For example, many brokers already specify session times in UTC in their specifications, while the MQL5 API has historically used GMT notation.

The MQL5 API allows you to find out the current time of the terminal (in fact, the local time of the computer) and the server time: they are returned by the functions TimeLocal and TimeCurrent, respectively. In addition, an MQL program can get the current GMT time (function TimeGMT) based on the Windows timezone settings. Thus, a trader and a programmer get a binding of local time to the global one, and by the difference between local and server time, one can determine the "timezone" of the server and quotes. But there are a couple of interesting points here.

First, in many countries, there is a practice of switching to the Daylight Saving Time (DST). Usually, this means adding 1 hour to standard (winter) time from about March/April to October/November (in the northern hemisphere, in the southern it is vice versa). At the same time, GMT/UTC time always remains constant, i.e., it is not subject to DST correction, and therefore various options for convergence/discrepancy between client and server time are potentially possible:

* transition dates may vary from country to country
* some countries do not implement daylight saving time

Because of this, some MQL programs need to keep track of such time zone changes if the algorithms are based on reference to intraday time (for example, news releases) and not to price movements or volume concentrations.

And if the time translation on the user's computer is quite easy to determine, thanks to the TimeDaylightSavings function, then there is no ready-made analog for server time.

Second, the regular MetaTrader 5 tester, in which we can debug or evaluate MQL programs of such types as Expert Advisors and indicators, unfortunately, does not emulate the time of the trade server. Instead, all three of the above functions TimeLocal, TimeGMT, and TimeCurrent, will return the same time, i.e. the timezone is always virtually GMT.

Absolute and relative time 
  
Time accounting in algorithms, as in life, can be carried out in absolute or relative coordinates. Every moment in the past, present, and future is described by an absolute value to which we can refer in order to indicate the beginning of an accounting period or the time an economic news is released. It is this time that we store in MQL5 using the datetime type. At the same time, it is often required to look into the future or retreat into the past for a given number of time units from the current moment. In this case, we are not interested in the absolute value, but in the time interval. 
  
In particular, algorithms have the concept of a timeout, which is a period of time during which a certain action must be performed, and if it is not performed for any reason, we cancel it and stop waiting for the result (because, apparently, something went wrong). You can measure the interval in different units: hours, seconds, milliseconds, or even microseconds (after all, computers are now fast). 
  
In MQL5, some time-related functions work with absolute values (for example, [TimeLocal](/en/book/common/timing/timing_local_server), [TimeCurrent](/en/book/common/timing/timing_local_server)), and the part with intervals (for example, [GetTickCount](/en/book/common/timing/timing_count), [GetMicrosecondCount](/en/book/common/timing/timing_count)). 
  
However, the measurement of intervals or the activation of the program at specified intervals can be implemented not only via the functions from this section but also using built-in timers that work according to the well-known principle of an alarm clock. When enabled, they use special events to notify MQL programs and the functions we implement to handle these events — [OnTimer](/en/book/applications/timer/timer_ontimer) (they are similar to OnStart). We will cover this aspect of time management in a separate section, after studying the general concept of events in MQL5 (see [Overview of event handling functions](/en/book/applications/runtime/runtime_events_overview)).

[Flushing global variables to disk](/en/book/common/globals/globals_flush "Flushing global variables to disk")

[Local and server time](/en/book/common/timing/timing_local_server "Local and server time")