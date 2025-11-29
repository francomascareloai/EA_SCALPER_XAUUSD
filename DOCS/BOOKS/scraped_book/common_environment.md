---
title: "MQL program execution environment"
url: "https://www.mql5.com/en/book/common/environment"
hierarchy: []
scraped_at: "2025-11-28 09:49:05"
---

# MQL program execution environment

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")MQL program execution environment

* [Getting a general list of terminal and program properties](/en/book/common/environment/env_listing "Getting a general list of terminal and program properties")
* [Terminal build number](/en/book/common/environment/env_build "Terminal build number")
* [Program type and license](/en/book/common/environment/env_type_license "Program type and license")
* [Terminal and program operating modes](/en/book/common/environment/env_mode "Terminal and program operating modes")
* [Permissions](/en/book/common/environment/env_permissions "Permissions")
* [Checking network connections](/en/book/common/environment/env_connectivity "Checking network connections")
* [Computing resources: memory, disk, and CPU](/en/book/common/environment/env_resources "Computing resources: memory, disk, and CPU")
* [Screen specifications](/en/book/common/environment/env_screen "Screen specifications")
* [Terminal and program string properties](/en/book/common/environment/env_descriptive "Terminal and program string properties")
* [Custom properties: Bar limit and interface language](/en/book/common/environment/env_bar_lang "Custom properties: Bar limit and interface language")
* [Binding a program to runtime properties](/en/book/common/environment/env_signature "Binding a program to runtime properties")
* [Checking keyboard status](/en/book/common/environment/env_keyboard "Checking keyboard status")
* [Checking the MQL program status and reason for termination](/en/book/common/environment/env_stop "Checking the MQL program status and reason for termination")
* [Programmatically closing the terminal and setting a return code](/en/book/common/environment/env_terminal_close "Programmatically closing the terminal and setting a return code")
* [Handling runtime errors](/en/book/common/environment/env_last_error "Handling runtime errors")
* [User-defined errors](/en/book/common/environment/env_user_error "User-defined errors")
* [Debug management](/en/book/common/environment/env_debug_break "Debug management")
* [Predefined variables](/en/book/common/environment/env_variables "Predefined variables")
* [Predefined constants of the MQL5 language](/en/book/common/environment/env_constants "Predefined constants of the MQL5 language")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# MQL program execution environment

As we know, the source texts of an MQL program after compilation into a binary executable code in the format ex5 are ready to work in the terminal or on test agents. Thus, a terminal or a tester provides a common environment within which MQL programs "live".

Recall that the built-in tester supports only 2 types of MQL programs: Expert Advisors and indicators. We will talk in detail about the types of MQL programs and their features in the fifth part of the book. Meanwhile, in this chapter, we will focus on those MQL5 API functions that are common to all types, and allow you to analyze the execution environment and, to some extent, control it.

Most environment properties are read-only through functions TerminalInfoInteger, TerminalInfoDouble, TerminalInfoString, MQLInfoInteger, and MQLInfoString. From the names you can understand that each function returns values of a certain type. Such an architecture leads to the fact that the applied meaning of the properties combined in one function can be very different. Another grouping can be provided by the implementation of your own object layer in MQL5 (an example will be given a little later, in the section on using [properties for binding to the program environment](/en/book/common/environment/env_signature)).

The specified set of functions has an explicit logical division into general terminal properties (with the "Terminal" prefix) and properties of a separate MQL program (with the "MQL" prefix). However, in many cases, it is required to jointly analyze the similar characteristics of both the terminal and the program. For example, permissions to use a DLL, or perform trading operations are issued both to the terminal as a whole and to a specific program. That is why it makes sense to consider the functions from this in a complex, as a whole.

Only some of the environment properties associated with error codes are writable, in particular, resetting a previous error (ResetLastError) and setting a user error (SetUserError).

Also in this chapter, we will look at the functions for closing the terminal within a program (TerminalClose, SetReturnError) and pausing the program in the debugger (Debug Break).

[Sound alerts](/en/book/common/output/output_sound "Sound alerts")

[Getting a general list of terminal and program properties](/en/book/common/environment/env_listing "Getting a general list of terminal and program properties")