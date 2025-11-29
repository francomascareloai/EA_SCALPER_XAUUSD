---
title: "User interaction"
url: "https://www.mql5.com/en/book/common/output"
hierarchy: []
scraped_at: "2025-11-28 09:49:01"
---

# User interaction

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")User interaction

* [Logging messages](/en/book/common/output/output_print "Logging messages")
* [Alerts](/en/book/common/output/output_alert "Alerts")
* [Displaying messages in the chart window](/en/book/common/output/output_comment "Displaying messages in the chart window")
* [Message dialog box](/en/book/common/output/output_messagebox "Message dialog box")
* [Sound alerts](/en/book/common/output/output_sound "Sound alerts")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# User interaction

The connection of the program with the "outside world" is always bidirectional, and the means for organizing it can be conditionally divided into categories for input and output of data. In the classic version, the user provides the program with some settings and receives a result from it. If the program integrates with some external application or service, input and output, as a rule, are carried out using special exchange protocols (via files, network, shared memory, etc.), bypassing the user interface.

The MQL program execution environment allows you to organize interaction with the MetaTrader 5 user in many ways.

In this chapter, we will look at the simplest of them, which allow you to display messages in a log or graph, show a simple dialog box, and issue sound alerts.

Recall that the standard for entering data into an MQL program is [input variables](/en/book/basis/variables/input_variables). However, they can only be set at program initialization. Changing the program properties through the settings dialog means "restarting" it with new values (later we will talk about some of the special cases connected with a type of MQL program due to which the restart is in quotation marks).

More flexible interactive relation implies the ability to control the behavior of the program without stopping it. In elementary cases, the MessageBox dialog box (for example), which we will discuss below, would be suitable for this, but for most practical applications this is not enough.

Therefore, in the following parts of the book, we will significantly expand the list of tools for implementing the user interface and learn how to create interactive programs based on interface [objects](/en/book/applications/objects), display graphical information in [indicators](/en/book/applications/indicators_make) or [resources](/en/book/advanced/resources), send push notifications to user's mobile devices, and much more.

[Time interval counters](/en/book/common/timing/timing_count "Time interval counters")

[Logging messages](/en/book/common/output/output_print "Logging messages")