---
title: "General principles for executing MQL programs"
url: "https://www.mql5.com/en/book/applications/runtime"
hierarchy: []
scraped_at: "2025-11-28 09:48:19"
---

# General principles for executing MQL programs

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")General principles for executing MQL programs

* [Designing MQL programs of various types](/en/book/applications/runtime/runtime_features_by_progtype "Designing MQL programs of various types")
* [Threads](/en/book/applications/runtime/runtime_threads "Threads")
* [Overview of event handling functions](/en/book/applications/runtime/runtime_events_overview "Overview of event handling functions")
* [Features of starting and stopping programs of various types](/en/book/applications/runtime/runtime_lifecycle "Features of starting and stopping programs of various types")
* [Reference events of indicators and Expert Advisors: OnInit and OnDeinit](/en/book/applications/runtime/runtime_oninit_ondeinit "Reference events of indicators and Expert Advisors: OnInit and OnDeinit")
* [The main function of scripts and services: OnStart](/en/book/applications/runtime/runtime_onstart "The main function of scripts and services: OnStart")
* [Programmatic removal of Expert Advisors and scripts: ExpertRemove](/en/book/applications/runtime/runtime_remove "Programmatic removal of Expert Advisors and scripts: ExpertRemove")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# General principles for executing MQL programs

All MQL programs can be broadly divided into several groups depending on their capabilities and features.

Most programs, such as Expert Advisors, indicators, and scripts, work in the context of a chart. In other words, they start executing only after they are attached to one of the open charts by using the Attach to Chart context menu command in the Navigator tree or by dragging and dropping from Navigator to the chart.

In contrast, services cannot be placed on the chart, as they are designed to perform long, cyclic actions in the background. For example, in a service, you can create a [custom symbol](/en/book/advanced/custom_symbols) and then receive its data and keep updating it in an endless loop using network functions. Another logical application of a service is monitoring the trading account and the network connection, as a part of a solution that notifies the user about communication problems.

It is important to note that indicators and Expert Advisors are saved on the chart between terminal working sessions. In other words, if, for example, a user runs an indicator on the chart and then, without explicitly deleting it, closes MetaTrader 5, then the next time the terminal starts, the indicator will be restored along with the chart, including all its settings.

By the way, linking indicators and Expert Advisors to the chart is the basis for templates (see the [Documentation](https://www.metatrader5.com/en/terminal/help/charts_advanced/templates_profiles "documentation on mql5.com")). The user can create a set of programs to be used on a chart, configure them and save the set in a special file with the tpl extension. This is done using the context menu command Templates -> Save. After that, you can apply the template to any new chart (command Templates -> Upload) and run all linked programs. Templates are stored in the directory MQL5/Profiles/Templates/ by default.

Another consequence of attaching to a chart is that closing a chart results in unloading all MQL programs that were placed on it. However, MetaTrader 5 saves all closed charts in a specific way (at least for a while) and therefore, if the chart was closed by accident, it can be restored along with all programs (and [graphic objects](/en/book/applications/objects)) using the command File -> Open Remote.

If for some reason the terminal fails to load chart files, the entire state of MQL programs (settings and location) will be lost. Basically, the same applies to [graphic objects](/en/book/applications/objects) — programs can add them for their own needs and expect that these objects are located on the chart. Make backup copies of charts. Each chart is a file with the extension chr. Such files are stored by default in the directory MQL5/Profiles/Charts/Default/. This is the standard profile created when the platform is installed. You can create other profiles with the menu command File -> Profiles and then switch between them (see the [Documentation](https://www.metatrader5.com/en/terminal/help/charts_advanced/templates_profiles "documentation on mql5.com")).

If necessary, you can stop an Expert Advisor and remove it from the chart using the context menu command Expert list (called by pressing the right mouse button in the chart window). It opens the Experts dialog with a list of all Expert Advisors running in the terminal. In this list, select an Expert Advisor that you no longer need and press Remove.

Indicators can also be removed explicitly, using a similar context menu command Indicator List. It opens a dialog with a list of indicators running on the current chart, in which you can select a specific indicator and click the button Remove. In addition, most indicators display various graphical constructions, such as lines and histograms, on the chart, which can also be deleted using the relevant context menu commands.

In contrast to indicators and Expert Advisors, scripts are not permanently attached to a chart. In standard mode, the script is removed from the chart automatically after the task assigned to it is completed, if this is a one-time action. If a script has a loop for periodic, repetitive actions, it will, of course, continue its work until the loop is interrupted in one way or another, but no longer than until the end of the session. Closing the terminal causes the script to become detached from the chart. After restarting MetaTrader 5, scripts are not restored on charts.

Please note that if you switch the chart to another symbol or timeframe, the script running on it will be unloaded. But indicators and Expert Advisors will continue to work, however, they will be re-initialized. Initialization rules for them are different. These details will be discussed in the section [Features of starting and stopping programs of various types](/en/book/applications/runtime/runtime_lifecycle).

Only one Expert Advisor, only one script, and any number of indicators can be placed on the chart. The Expert Advisor, the script, and all indicators will work in parallel (simultaneously).

As for services, their created and running instances are automatically restored after loading the terminal. The service instance can be stopped or deleted using the context menu in the Services section of the Navigator window.

The following table summarizes the properties described above in a summary form.

| Program type | Link to chart | Quantity on the chart | Recovery of the  session |
| --- | --- | --- | --- |
| Indicator | Required | Multiple | With chart or template |
| Expert Advisor | Required | Maximum 1 | With chart or template |
| Script | Required | Maximum 1 | Not supported |
| Service | Not supported | 0 | With terminal |

All MQL programs are executed in the client terminal and therefore work only while the terminal is open. For constant program control over the account, use a VPS.

[Creating application programs](/en/book/applications "Creating application programs")

[Designing MQL programs of various types](/en/book/applications/runtime/runtime_features_by_progtype "Designing MQL programs of various types")