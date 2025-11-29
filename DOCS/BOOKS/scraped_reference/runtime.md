---
title: "MQL5 Programs"
url: "https://www.mql5.com/en/docs/runtime"
hierarchy: []
scraped_at: "2025-11-28 09:30:45"
---

# MQL5 Programs

[MQL5 Reference](/en/docs "MQL5 Reference")MQL5 programs

* [Program Running](/en/docs/runtime/running "Program Running")
* [Trade Permission](/en/docs/runtime/tradepermission "Trade Permission")
* [Client Terminal Events](/en/docs/runtime/event_fire "Client Terminal Events")
* [Resources](/en/docs/runtime/resources "Resources")
* [Call of Imported Functions](/en/docs/runtime/imports "Call of Imported Functions")
* [Runtime Errors](/en/docs/runtime/errors "Runtime Errors")
* [Testing Trading Strategies](/en/docs/runtime/testing "Testing Trading Strategies")

# MQL5 Programs

For the mql5-program to operate, it must be compiled (Compile button or F7 key). Compilation should pass without errors (some warnings are possible; they should be analyzed). At this process, an executable file with the same name and with EX5 extension must be created in the corresponding directory, terminal\_dir\MQL5\Experts, terminal\_dir\MQL5\indicators or terminal\_dir\MQL5\scripts. This file can be run.

Operating features of MQL5 programs are described in the following sections:

* [Program running](/en/docs/runtime/running) – order of calling predefined event-handlers.
* [Testing trading strategies](/en/docs/runtime/testing) – operating features of MQL5 programs in the Strategy Tester.
* [Client terminal events](/en/docs/runtime/event_fire) – description of events, which can be processed in programs.
* [Call of imported functions](/en/docs/runtime/imports) – description order, allowed parameters, search details and call agreement for imported functions.
* [Runtime errors](/en/docs/runtime/errors) – getting information about runtime and critical errors.

Expert Advisors, custom indicators and scripts are attached to one of opened charts by Drag'n'Drop method from the Navigator window.

For an expert Advisor to stop operating, it should be removed from a chart. To do it select "Expert list" in chart context menu, then select an Expert Advisor from list and click "Remove" button. Operation of Expert Advisors is also affected by the state of the "AutoTrading" button.

In order to stop a custom indicator, it should be removed from a chart.

Custom indicators and Expert Advisors work until they are explicitly removed from a chart; information about attached Expert Advisors and Indicators is saved between client terminal sessions.

Scripts are executed once and are deleted automatically upon operation completion or change of the current chart state, or upon client terminal shutdown. After the restart of the client terminal scripts are not started, because the information about them is not saved.

Maximum one Expert Advisor, one script and unlimited number of indicators can operate in one chart.

Services do not require to be bound to a chart to work and are designed to perform auxiliary functions. For example, in a service, you can create a [custom symbol](/en/docs/customsymbols), open its chart, receive data for it in an endless loop using the [network functions](/en/docs/network) and constantly update it.

[MessageBox](/en/docs/constants/io_constants/messbconstants "MessageBox")

[Program Running](/en/docs/runtime/running "Program Running")