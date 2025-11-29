---
title: "Date and Time"
url: "https://www.mql5.com/en/docs/dateandtime"
hierarchy: []
scraped_at: "2025-11-28 09:30:16"
---

# Date and Time

[MQL5 Reference](/en/docs "MQL5 Reference")Date and Time

* [TimeCurrent](/en/docs/dateandtime/timecurrent "TimeCurrent")
* [TimeTradeServer](/en/docs/dateandtime/timetradeserver "TimeTradeServer")
* [TimeLocal](/en/docs/dateandtime/timelocal "TimeLocal")
* [TimeGMT](/en/docs/dateandtime/timegmt "TimeGMT")
* [TimeDaylightSavings](/en/docs/dateandtime/timedaylightsavings "TimeDaylightSavings")
* [TimeGMTOffset](/en/docs/dateandtime/timegmtoffset "TimeGMTOffset")
* [TimeToStruct](/en/docs/dateandtime/timetostruct "TimeToStruct")
* [StructToTime](/en/docs/dateandtime/structtotime "StructToTime")

# Date and Time

This is the group of functions for working with data of [datetime](/en/docs/basis/types/integer/datetime) type (an integer that represents the number of seconds elapsed from 0 hours of January 1, 1970).

To arrange high-resolution counters and timers, use the [GetTickCount()](/en/docs/common/gettickcount) function, which produces values in milliseconds.

| Function | Action |
| --- | --- |
| [TimeCurrent](/en/docs/dateandtime/timecurrent) | Returns the last known server time (time of the last quote receipt) in the datetime format |
| [TimeTradeServer](/en/docs/dateandtime/timetradeserver) | Returns the current calculation time of the trade server |
| [TimeLocal](/en/docs/dateandtime/timelocal) | Returns the local computer time in datetime format |
| [TimeGMT](/en/docs/dateandtime/timegmt) | Returns GMT in datetime format with the Daylight Saving Time by local time of the computer, where the client terminal is running |
| [TimeDaylightSavings](/en/docs/dateandtime/timedaylightsavings) | Returns the sign of Daylight Saving Time switch |
| [TimeGMTOffset](/en/docs/dateandtime/timegmtoffset) | Returns the current difference between GMT time and the local computer time in seconds, taking into account DST switch |
| [TimeToStruct](/en/docs/dateandtime/timetostruct) | Converts a datetime value into a variable of MqlDateTime structure type |
| [StructToTime](/en/docs/dateandtime/structtotime) | Converts a variable of MqlDateTime structure type into a datetime value |

[StringTrimRight](/en/docs/strings/stringtrimright "StringTrimRight")

[TimeCurrent](/en/docs/dateandtime/timecurrent "TimeCurrent")