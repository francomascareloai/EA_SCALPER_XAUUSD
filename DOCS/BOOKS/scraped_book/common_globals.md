---
title: "Client terminal global variables"
url: "https://www.mql5.com/en/book/common/globals"
hierarchy: []
scraped_at: "2025-11-28 09:49:03"
---

# Client terminal global variables

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Client terminal global variables

* [Writing and reading global variables](/en/book/common/globals/globals_set_get "Writing and reading global variables")
* [Checking the existence and last activity time](/en/book/common/globals/globals_exist_time "Checking the existence and last activity time")
* [Getting a list of global variables](/en/book/common/globals/globals_list "Getting a list of global variables")
* [Deleting global variables](/en/book/common/globals/globals_delete "Deleting global variables")
* [Temporary global variables](/en/book/common/globals/globals_temp "Temporary global variables")
* [Synchronizing programs using global variables](/en/book/common/globals/globals_condition "Synchronizing programs using global variables")
* [Flushing global variables to disk](/en/book/common/globals/globals_flush "Flushing global variables to disk")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Client terminal global variables

In the previous chapter, we studied MQL5 functions that work with files. They provide wide, flexible options for writing and reading arbitrary data. However, sometimes an MQL program needs an easier way to save and restore the state of an attribute between runs.

For example, we want to calculate certain statistics: how many times the program was launched, how many instances of it are executed in parallel on different charts, etc. It is impossible to accumulate this information within the program itself. There must be some kind of external long-term storage. But it would be expensive to create a file for this, though it is also feasible.

Many programs are designed to interact with each other, i.e., they must somehow exchange information. If we are talking about integration with a program external to the terminal, or about transferring a large amount of data, then it is really difficult to do it without using files. However, when there is not enough data to be sent, and all programs are written in MQL5 and run inside MetaTrader 5, the use of files seems redundant. The terminal provides a simpler technology for this case: global variables.

A global variable is a named location in the terminal's shared memory. It can be created, modified, or deleted by any MQL program, but will not belong to it exclusively, and is available to all other MQL programs. The name of a global variable is any unique (among all variables) string of no more than 63 characters. This string does not have to meet the requirements for variable identifiers in MQL5, since global variables of the terminal are not variables in the usual sense. The programmer does not define them in the source code according to the syntax we learned in [Variables](/en/book/basis/variables), they are not an integral part of the MQL program, and any action with them is performed only by calling one of the special functions that we will describe in this chapter.

The global variables allow you to store only values of type double. If necessary, you can pack/convert values of other types to double or use part of the variable name (following a certain prefix, for example) to store strings.

While the terminal is running, global variables are stored in RAM and are available almost instantly: the only overhead is associated with function calls. This definitely gives a headstart to global variables against using files, since when dealing with the latter, obtaining a handle is a relatively slow process, and the handle itself consumes some additional resources.

At the end of the terminal session, global variables are unloaded into a special file (gvariables.dat) and then restored from it the next time you run the terminal.

A particular global variable is automatically destroyed by the terminal if it has not been claimed within 4 weeks. This behavior relies on keeping track of and storing the time of the last use of a variable, where use refers to setting a new value or reading an old one (but not checking for existence or getting the time of last use).

Please note that global variables are not tied to an account, profile, or any other characteristics of the trading environment. Therefore, if they are supposed to store something related to the environment (for example, some general limits for a particular account), variable names should be constructed taking into account all factors that affect the algorithm and decision-making. To distinguish between global variables of multiple instances of the same Experts Advisor (EA), you may need to add a working symbol, timeframe, or "magic number" from the EA settings to the name.

In addition to MQL programs, global variables can also be manually created by the user. The list of existing global variables, as well as the means of their interactive management, can be found in the dialog opened in the terminal by the command Tools -> Global Variables (F3).

By using the corresponding buttons here you can Add and Delete global variables, and double-clicking in columns Variable or Meaning allows you to edit the name or value of a particular variable. The following hotkeys work from the keyboard: F2 for name editing, F3 for value editing, Ins for adding a new variable, Del for deleting the selected variable.

A little later, we will study two main types of MQL programs — Expert Advisors and Indicators. Their special feature is the ability to run in the tester, where functions for global variables also work. However, global variables are created, stored, and managed by the tester agent in the tester. In other words, the lists of terminal global variables are not available in the tester, and those variables that are created by the program under test belong to a specific agent, and their lifetime is limited to one test pass. That is, the agent's global variables are not visible from other agents and will be removed at the end of the test run. In particular, if the EA is [optimized](/en/book/automation/tester) on several agents, it can manipulate global variables to "communicate" with the indicators it uses in the context of the same agent since they are executed there together, but on parallel agents, other copies of the EA will form their own lists of variables.

Data exchange between MQL programs using global variables is not the only available, and not always the most appropriate way. In particular, EAs and indicators are interactive types of MQL programs that can generate and accept [events on charts](/en/book/applications/events). You can pass various types of information in event parameters. In addition, arrays of calculated data can be prepared and provided to other MQL programs in the form of [indicator buffers](/en/book/applications/indicators_make/indicators_setindexbuffer). MQL programs located on charts can use UI [graphic objects](/en/book/applications/objects) to transfer and store information.

From the technical point of view, the maximum number of global variables is limited only by the resources of the operating system. However, for a large number of elements, it is recommended to use more suitable means: [files](/en/book/common/files) or [databases](/en/book/advanced/sqlite).

[File or folder selection dialog](/en/book/common/files/files_select "File or folder selection dialog")

[Writing and reading global variables](/en/book/common/globals/globals_set_get "Writing and reading global variables")