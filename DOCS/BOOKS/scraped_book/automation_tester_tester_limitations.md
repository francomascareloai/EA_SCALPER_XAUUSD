---
title: "Limitations of functions in the tester"
url: "https://www.mql5.com/en/book/automation/tester/tester_limitations"
hierarchy: []
scraped_at: "2025-11-28 09:48:44"
---

# Limitations of functions in the tester

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Trading automation](/en/book/automation "Trading automation")[Testing and optimization of Expert Advisors](/en/book/automation/tester "Testing and optimization of Expert Advisors")Limitations of functions in the tester

* [Generating ticks in tester](/en/book/automation/tester/tester_ticks "Generating ticks in tester")
* [Time management in the tester: timer, Sleep, GMT](/en/book/automation/tester/tester_time "Time management in the tester: timer, Sleep, GMT")
* [Testing visualization: chart, objects, indicators](/en/book/automation/tester/tester_chart_limits "Testing visualization: chart, objects, indicators")
* [Multicurrency testing](/en/book/automation/tester/tester_multicurrency_sync "Multicurrency testing")
* [Optimization criteria](/en/book/automation/tester/tester_criterion "Optimization criteria")
* [Getting testing financial statistics: TesterStatistics](/en/book/automation/tester/tester_testerstatistics "Getting testing financial statistics: TesterStatistics")
* [OnTester event](/en/book/automation/tester/tester_ontester "OnTester event")
* [Auto-tuning: ParameterGetRange and ParameterSetRange](/en/book/automation/tester/tester_parameterrange "Auto-tuning: ParameterGetRange and ParameterSetRange")
* [Group of OnTester events for optimization control](/en/book/automation/tester/tester_ontester_init_pass_deinit "Group of OnTester events for optimization control")
* [Sending data frames from agents to the terminal](/en/book/automation/tester/tester_frameadd "Sending data frames from agents to the terminal")
* [Getting data frames in terminal](/en/book/automation/tester/tester_framenext "Getting data frames in terminal")
* [Preprocessor directives for the tester](/en/book/automation/tester/tester_directives "Preprocessor directives for the tester")
* [Managing indicator visibility: TesterHideIndicators](/en/book/automation/tester/tester_testerhideindicators "Managing indicator visibility: TesterHideIndicators")
* [Emulation of deposits and withdrawals](/en/book/automation/tester/tester_withdraw_deposit "Emulation of deposits and withdrawals")
* [Forced test stop: TesterStop](/en/book/automation/tester/tester_testerstop "Forced test stop: TesterStop")
* [Big Expert Advisor example](/en/book/automation/tester/tester_example_ea "Big Expert Advisor example")
* [Mathematical calculations](/en/book/automation/tester/tester_math_calc "Mathematical calculations")
* [Debugging and profiling](/en/book/automation/tester/tester_debug_profile "Debugging and profiling")
* Limitations of functions in the tester

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Limitations of functions in the tester

When using the tester, you should take into account some restrictions imposed on built-in functions. Some of the MQL5 API functions are never executed in the strategy tester and some work only in single passes but not during optimization.

So, to increase performance when optimizing Expert Advisors, the [Comment](/en/book/common/output/output_comment), [Print](/en/book/common/output/output_print), and [PrintFormat](/en/book/common/output/output_print) functions are not executed.

The exception is the use of these functions inside the OnInit handler which is done to make it easier to find possible causes of initialization errors.

Functions that provide interaction with the "world" are not executed in the strategy tester. These include [MessageBox](/en/book/common/output/output_messagebox), [PlaySound](/en/book/common/output/output_sound), [SendFTP](/en/book/advanced/network/network_ftp), [SendMail](/en/book/advanced/network/network_email), [SendNotification](/en/book/advanced/network/network_push), [WebRequest](/en/book/advanced/network/network_http), and functions for working with [sockets](/en/book/advanced/network/network_socket_create_connect).

In addition, many functions for working with charts and objects have no effect. In particular, you will not be able to change the symbol or period of the current chart by calling [ChartSetSymbolPeriod](/en/book/applications/charts/charts_set_symbol_period), list all indicators (including subordinate ones) with [ChartIndicatorGet](/en/book/applications/charts/charts_indicators), work with templates [ChartSaveTemplate](/en/book/applications/charts/charts_tpl), and so on.

In the tester, even in the visual mode, interactive chart, object, keyboard and mouse events are not generated for the [OnChartEvent](/en/book/applications/events/events_onchartevent) handler.

[Debugging and profiling](/en/book/automation/tester/tester_debug_profile "Debugging and profiling")

[Advanced language tools](/en/book/advanced "Advanced language tools")