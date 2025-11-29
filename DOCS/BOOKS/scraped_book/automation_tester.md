---
title: "Testing and optimization of Expert Advisors"
url: "https://www.mql5.com/en/book/automation/tester"
hierarchy: []
scraped_at: "2025-11-28 09:48:04"
---

# Testing and optimization of Expert Advisors

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Trading automation](/en/book/automation "Trading automation")Testing and optimization of Expert Advisors

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
* [Limitations of functions in the tester](/en/book/automation/tester/tester_limitations "Limitations of functions in the tester")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Testing and optimization of Expert Advisors

Development of Expert Advisors implies not only and not so much the implementation of a trading strategy in MQL5 but to a greater extent testing its financial performance, finding optimal settings, and debugging (searching for and correcting errors) in various situations. All this can be done in the integrated MetaTrader 5 tester.

The tester works for various currencies and supports several tick generation modes: based on opening prices of the selected timeframe, on OHLC prices of the M1 timeframe, on artificially generated ticks, and on the real tick history. This way you can choose the optimal ratio of speed and accuracy of trading simulation.

The tester settings allow you to set the testing time interval in the past, the size of the deposit, and the leverage; they are used to emulate requotes and specific account features (including the size of commissions, margins, session schedules, limiting the number of lots). All the details of working with the tester from the user's point of view can be found in [terminal documentation](https://www.metatrader5.com/en/terminal/help/algotrading/testing "terminal documentation").

Earlier, we already briefly discussed working with the tester, in particular, in the section [Testing indicators](/en/book/applications/indicators_make/indicators_test). Let's recall that the tester control functions and their optimization are not available for indicators, unlike for Expert Advisors. However, personally, I would like to see an option of adaptive self-tuning of indicators: all that is needed is to support the OnTester handler in them, which we will present in a [separate section](/en/book/automation/tester/tester_ontester).

As you know, various modes are available for optimization, such as direct enumeration of combinations of Expert Advisor input parameters, accelerated genetic algorithm, mathematical calculations, or sequential runs through symbols in Market Watch. As an optimization criterion, you can use both well-known metrics such as profitability, Sharpe ratio, recovery factor, and expected payoff, as well as "custom" variables embedded in the source code by the developer of the Expert Advisor. In the context of this book, it is assumed that the reader is already familiar with the principles of setting up, running, and interpreting optimization results because in this chapter we will begin to study the tester control API. Those interested can refresh their knowledge with the help of the relevant section of [documentation](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization "strategy optimization").

A particularly important function of the tester is multi-threaded optimization, which can be performed using local and distributed (network) agent programs, including those in the MQL5 Cloud Network. A single testing run (with specific input parameters) launched manually by the user, or one of the many runs called during optimization (when we implement enumeration of parameter values in given ranges) is performed in a separate program — the agent. Technically, this is a metatester64.exe file, and the copies of its processes can be seen in the Windows Task Manager during testing and optimization. It is due to this that the tester is multi-threaded.

The terminal is a dispatcher that distributes tasks to local and remote agents. It launches local agents if necessary. When optimizing, by default, several agents are launched; their quantity corresponds to the number of processor cores. After executing the next task for testing an Expert Advisor with the specified parameters, the agent returns the results to the terminal.

Each agent creates its own trading and software environment. All agents are isolated from each other and from the client terminal.

In particular, the agent has its own global variables and its own [file sandbox](/en/book/common/files), including the folder where detailed agent logs are written: Tester/Agent-IPaddress-Port/Logs. Here Tester is the tester installation directory (during a standard installation together with MetaTrader 5, this is the subfolder where the terminal is installed). The name of the directory Agent-IPaddress-Port, instead of IPaddress and Port, will contain the specific network address and port values that are used to communicate with the terminal. For local agents, this is the address 127.0.0.1 and the range of ports, by default, starting from 3000 (for example, on a computer with 4 cores, we will see agents on ports 3000, 3001, 3002, 3003).

When testing an Expert Advisor, all file operations are performed in the Tester/Agent-IPaddress-Port/MQL5/Files folder. However, it is possible to implement interaction between local agents and the client terminal (as well as between different copies of the terminal on the same computer) via a [shared folder](/en/book/common/files). For this, when opening a file with the [FileOpen](/en/book/common/files/files_open_close) function, the FILE\_COMMON flag must be specified. Another way to transfer data from agents to the terminal is provided by the [frames](/en/book/automation/tester/tester_frameadd) mechanism.

The agent's local sandbox is automatically cleared before each test due to security reasons (to prevent different Expert Advisors from reading each other's data).

A folder with the quotes history is created next to the file sandbox for each agent: Tester/Agent-IPaddress-Port/bases/ServerName/Symbol/. In the next section, we briefly remind you how it is formed.

The results of individual test runs and optimizations are stored by the terminal in a special cache which can be found in the installation directory, in the subfolder Tester/cache/. Test results are stored in files with the extension tst, and the optimization results are stored in opt files. Both formats are open-sourced by MetaQuotes developers, so you can implement your own batch analytical data processing, or use ready-made source codes from the codebase on the mql5.com website.

In this chapter, first, we will consider the basic principles of how MQL programs work in the tester, and then we will learn how to interact with it in practice.

[Creating Expert Advisors in the MQL Wizard](/en/book/automation/experts/experts_wizard "Creating Expert Advisors in the MQL Wizard")

[Generating ticks in tester](/en/book/automation/tester/tester_ticks "Generating ticks in tester")