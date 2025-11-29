---
title: "Working with timer"
url: "https://www.mql5.com/en/book/applications/timer"
hierarchy: []
scraped_at: "2025-11-28 09:48:13"
---

# Working with timer

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")Working with timer

* [Turning timer on and off](/en/book/applications/timer/timer_event_set "Turning timer on and off")
* [Timer event: OnTimer](/en/book/applications/timer/timer_ontimer "Timer event: OnTimer")
* [High-precision timer: EventSetMillisecondTimer](/en/book/applications/timer/timer_event_set_millisecond "High-precision timer: EventSetMillisecondTimer")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Working with timer

For many applied tasks, it is important to be able to perform actions on a schedule, with some specified interval. In MQL5, this functionality is provided by the timer, a system time counter that can be configured to send regular notifications to an MQL program.

There are several functions for setting or canceling timer notifications in the MQL5 API: EventSetTimer, EventSetMillisecondTimer, EventKillTimer. The notifications themselves enter the program as events of a special type: the OnTimer handler is reserved for them in the source code. This group of functions will be discussed in this chapter.

Recall that in MQL5 events can only be received by interactive programs running on charts, that is, indicators and Expert Advisors. [Scripts](/en/book/applications/script_service/scripts) and [Services](/en/book/applications/script_service/services) do not support any events, including those from the timer.

However, in the chapter [Functions for working with time](/en/book/common/timing), we have already touched on related topics:

* Getting the timestamps of the current local or server clock ([TimeLocal / TimeCurrent](/en/book/common/timing/timing_local_server))
* Pausing the execution of the program for a specified period using [Sleep](/en/book/common/timing/timing_sleep)
* Getting the state of the computer's system time counter, counted from the start of the operating system ([GetTickCount](/en/book/common/timing/timing_count)) or since the launch of MQL-program ([GetMicrosecondCount](/en/book/common/timing/timing_count))

These options are open to absolutely all types of MQL programs.

In the previous chapters, we have already used the timer functions many times, although their formal description will be given only now. Due to the fact that timer events are available only in indicators or Expert Advisors, it would be difficult to study it before the programs themselves. After we have mastered the creation of indicators, the topic of timers will become a logical continuation.

Basically, we used timers to wait for the timeseries to be built. Such examples can be found in the sections [Waiting for data](/en/book/applications/indicators_make/indicators_wait_none), [Multicurrency and multitimeframe indicators](/en/book/applications/indicators_make/indicators_multisymbol), [Support for multiple symbols and timeframes](/en/book/applications/indicators_use/indicators_multitimeframe), [Using built-in indicators](/en/book/applications/indicators_use/indicators_standard_use).

In addition, we timed (every 5 seconds) the type of the subordinate indicator in the indicator "animation" demo in the section [Deleting indicator instances](/en/book/applications/indicators_use/indicators_indicatorrelease).

[Defining data source for an indicator](/en/book/applications/indicators_use/indicators_apply_to "Defining data source for an indicator")

[Turning timer on and off](/en/book/applications/timer/timer_event_set "Turning timer on and off")