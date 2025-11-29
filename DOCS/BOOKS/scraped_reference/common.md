---
title: "Common Functions"
url: "https://www.mql5.com/en/docs/common"
hierarchy: []
scraped_at: "2025-11-28 09:30:24"
---

# Common Functions

[MQL5 Reference](/en/docs "MQL5 Reference")Common Functions

* [Alert](/en/docs/common/alert "Alert")
* [CheckPointer](/en/docs/common/checkpointer "CheckPointer")
* [Comment](/en/docs/common/comment "Comment")
* [CryptEncode](/en/docs/common/cryptencode "CryptEncode")
* [CryptDecode](/en/docs/common/cryptdecode "CryptDecode")
* [DebugBreak](/en/docs/common/debugbreak "DebugBreak")
* [ExpertRemove](/en/docs/common/expertremove "ExpertRemove")
* [GetPointer](/en/docs/common/getpointer "GetPointer")
* [GetTickCount](/en/docs/common/gettickcount "GetTickCount")
* [GetTickCount64](/en/docs/common/gettickcount64 "GetTickCount64")
* [GetMicrosecondCount](/en/docs/common/getmicrosecondcount "GetMicrosecondCount")
* [MessageBox](/en/docs/common/messagebox "MessageBox")
* [PeriodSeconds](/en/docs/common/periodseconds "PeriodSeconds")
* [PlaySound](/en/docs/common/playsound "PlaySound")
* [Print](/en/docs/common/print "Print")
* [PrintFormat](/en/docs/common/printformat "PrintFormat")
* [ResetLastError](/en/docs/common/resetlasterror "ResetLastError")
* [ResourceCreate](/en/docs/common/resourcecreate "ResourceCreate")
* [ResourceFree](/en/docs/common/resourcefree "ResourceFree")
* [ResourceReadImage](/en/docs/common/resourcereadimage "ResourceReadImage")
* [ResourceSave](/en/docs/common/resourcesave "ResourceSave")
* [SetReturnError](/en/docs/common/setreturnerror "SetReturnError")
* [SetUserError](/en/docs/common/setusererror "SetUserError")
* [Sleep](/en/docs/common/sleep "Sleep")
* [TerminalClose](/en/docs/common/terminalclose "TerminalClose")
* [TesterHideIndicators](/en/docs/common/testerhideindicators "TesterHideIndicators")
* [TesterStatistics](/en/docs/common/testerstatistics "TesterStatistics")
* [TesterStop](/en/docs/common/testerstop "TesterStop")
* [TesterDeposit](/en/docs/common/testerdeposit "TesterDeposit")
* [TesterWithdrawal](/en/docs/common/testerwithdrawal "TesterWithdrawal")
* [TranslateKey](/en/docs/common/translatekey "TranslateKey")
* [ZeroMemory](/en/docs/common/zeromemory "ZeroMemory")

# Common Functions

General-purpose functions not included into any specialized group are listed here.

| Function | Action |
| --- | --- |
| [Alert](/en/docs/common/alert) | Displays a message in a separate window |
| [CheckPointer](/en/docs/common/checkpointer) | Returns the type of the object pointer |
| [Comment](/en/docs/common/comment) | Outputs a comment in the left top corner of the chart |
| [CryptEncode](/en/docs/common/cryptencode) | Transforms the data from array with the specified method |
| [CryptDecode](/en/docs/common/cryptdecode) | Performs the inverse transformation of the data from array |
| [DebugBreak](/en/docs/common/debugbreak) | Program breakpoint in debugging |
| [ExpertRemove](/en/docs/common/expertremove) | Stops Expert Advisor and unloads it from the chart |
| [GetPointer](/en/docs/common/getpointer) | Returns the object [pointer](/en/docs/basis/types/object_pointers) |
| [GetTickCount](/en/docs/common/gettickcount) | Returns the number of milliseconds that have elapsed since the system was started |
| [GetTickCount64](/en/docs/common/gettickcount64) | Returns the number of milliseconds that have elapsed since the system was started |
| [GetMicrosecondCount](/en/docs/common/getmicrosecondcount) | Returns the number of microseconds that have elapsed since the start of MQL5 program |
| [MessageBox](/en/docs/common/messagebox) | Creates, displays a message box and manages it |
| [PeriodSeconds](/en/docs/common/periodseconds) | Returns the number of seconds in the period |
| [PlaySound](/en/docs/common/playsound) | Plays a sound file |
| [Print](/en/docs/common/print) | Displays a message in the log |
| [PrintFormat](/en/docs/common/printformat) | Formats and prints the sets of symbols and values in a log file in accordance with a preset format |
| [ResetLastError](/en/docs/common/resetlasterror) | Sets the value of a predetermined variable [\_LastError](/en/docs/predefined/_lasterror) to zero |
| [ResourceCreate](/en/docs/common/resourcecreate) | Creates an image resource based on a data set |
| [ResourceFree](/en/docs/common/resourcefree) | Deletes [dynamically created resource](/en/docs/common/resourcecreate#dynamic_resourcecreate) (freeing the memory allocated for it) |
| [ResourceReadImage](/en/docs/common/resourcereadimage) | Reads data from the graphical resource [created by ResourceCreate() function](/en/docs/common/resourcecreate#dynamic_resourcecreate) or [saved in EX5 file during compilation](/en/docs/runtime/resources#resource_include) |
| [ResourceSave](/en/docs/common/resourcesave) | Saves a resource into the specified file |
| [SetUserError](/en/docs/common/setusererror) | Sets the predefined variable \_LastError into the value equal to ERR\_USER\_ERROR\_FIRST + user\_error |
| [SetReturnError](/en/docs/common/setreturnerror) | Sets the code that returns the terminal process when completing the operation. |
| [Sleep](/en/docs/common/sleep) | Suspends execution of the current Expert Advisor or script within a specified interval |
| [TerminalClose](/en/docs/common/terminalclose) | Commands the terminal to complete operation |
| [TesterHideIndicators](/en/docs/common/testerhideindicators) | Sets the mode of displaying/hiding indicators used in an EA |
| [TesterStatistics](/en/docs/common/testerstatistics) | It returns the value of a specified statistic calculated based on testing results |
| [TesterStop](/en/docs/common/testerstop) | Gives program operation completion command when [testing](https://www.metatrader5.com/en/terminal/help/algotrading/testing) |
| [TesterDeposit](/en/docs/common/testerdeposit) | Emulates depositing funds during a test. It can be used in some money management systems |
| [TesterWithdrawal](/en/docs/common/testerwithdrawal) | Emulates the operation of money withdrawal in the process of testing |
| [TranslateKey](/en/docs/common/translatekey) | Returns a Unicode character by a virtual key code |
| [ZeroMemory](/en/docs/common/zeromemory) | Resets a variable passed to it by reference. The variable can be of any type, except for classes and structures that have constructors. |

[\_IsX64](/en/docs/predefined/_isx64 "_IsX64")

[Alert](/en/docs/common/alert "Alert")