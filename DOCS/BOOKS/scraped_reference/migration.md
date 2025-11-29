---
title: "Moving from MQL4 to MQL5"
url: "https://www.mql5.com/en/docs/migration"
hierarchy: []
scraped_at: "2025-11-28 09:30:12"
---

# Moving from MQL4 to MQL5

[MQL5 Reference](/en/docs "MQL5 Reference")Moving from MQL4

* [Language Basics](/en/docs/basis "Language Basics")
* [Constants, Enumerations and Structures](/en/docs/constants "Constants, Enumerations and Structures")
* [MQL5 programs](/en/docs/runtime "MQL5 programs")
* [Predefined Variables](/en/docs/predefined "Predefined Variables")
* [Common Functions](/en/docs/common "Common Functions")
* [Array Functions](/en/docs/array "Array Functions")
* [Matrix and Vector Methods](/en/docs/matrix "Matrix and Vector Methods")
* [Conversion Functions](/en/docs/convert "Conversion Functions")
* [Math Functions](/en/docs/math "Math Functions")
* [String Functions](/en/docs/strings "String Functions")
* [Date and Time](/en/docs/dateandtime "Date and Time")
* [Account Information](/en/docs/account "Account Information")
* [Checkup](/en/docs/check "Checkup")
* [Event Handling](/en/docs/event_handlers "Event Handling")
* [Market Info](/en/docs/marketinformation "Market Info")
* [Economic Calendar](/en/docs/calendar "Economic Calendar")
* [Timeseries and Indicators Access](/en/docs/series "Timeseries and Indicators Access")
* [Custom Symbols](/en/docs/customsymbols "Custom Symbols")
* [Chart Operations](/en/docs/chart_operations "Chart Operations")
* [Trade Functions](/en/docs/trading "Trade Functions")
* [Trade Signals](/en/docs/signals "Trade Signals")
* [Network Functions](/en/docs/network "Network Functions")
* [Global Variables of the Terminal](/en/docs/globals "Global Variables of the Terminal")
* [File Functions](/en/docs/files "File Functions")
* [Custom Indicators](/en/docs/customind "Custom Indicators")
* [Object Functions](/en/docs/objects "Object Functions")
* [Technical Indicators](/en/docs/indicators "Technical Indicators")
* [Working with Optimization Results](/en/docs/optimization_frames "Working with Optimization Results")
* [Working with Events](/en/docs/eventfunctions "Working with Events")
* [Working with OpenCL](/en/docs/opencl "Working with OpenCL")
* [Working with databases](/en/docs/database "Working with databases")
* [Working with DirectX](/en/docs/directx "Working with DirectX")
* [Python Integration](/en/docs/python_metatrader5 "Python Integration")
* [ONNX models](/en/docs/onnx "ONNX models")
* [Standard Library](/en/docs/standardlibrary "Standard Library")
* Moving from MQL4
* [List of MQL5 Functions](/en/docs/function_indices "List of MQL5 Functions")
* [List of MQL5 Constants](/en/docs/constant_indices "List of MQL5 Constants")

# Moving from MQL4 to MQL5

MQL5 is the evolution of its predecessor - the MQL4 programming language, in which numerous indicators, scripts, and Expert Advisors were written. Despite the fact that the new programming language is maximally compatible with the previous-generation language, there are still some differences between these languages. And when transferring programs these differences should be noted.

This section contains information intended to facilitate the adaptation of codes to the new MQL5 language for programmers who know MQL4.

First it should be noted:

* The new language does not contain functions start(), init() and deinit().
* No limit for the number of indicator buffers.
* DLLs are loaded immediately after loading an Expert Advisor (or any other mql5 program).
* Check of logical conditions is shortened.
* When limits of an array are exceeded, the current performance is terminated (critically - with the output of an errors).
* Precedence of operators like in C + +.
* The language offers the implicit type cast (even from string to a number).
* Local variables are not initialized automatically (except for strings).
* Common local arrays are automatically deleted.

## Special Functions init, start and deinit

The MQL4 language contained only three predefined functions that could be used in the indicator, script or Expert Advisor (not taking into account the include files \*.mqh and library files). In MQL5 there are no such functions, but there are their analogues. The table shows the approximate correspondence of functions.

| MQL4 | MQL5 |
| --- | --- |
| init | OnInit |
| start | OnStart |
| deinit | OnDeinit |

Functions [OnInit](/en/docs/event_handlers/oninit) and [OnDeinit](/en/docs/event_handlers/ondeinit) perform the same role as init and deinit in MQL4 - they are designed to locate the code, which must be performed during initialization and deinitialization of mql5 programs. You can either just rename these functions accordingly, or leave them as they are, but add calls of these functions in corresponding places.

Example:

| |
| --- |
| void OnInit()   { //--- Call function upon initialization    init();   } void OnDeinit(const int reason)   { //--- Call function upon deinitialization    deinit(); //---   } |

The start function is replaced by [OnStart](/en/docs/event_handlers/onstart) only in scripts. In Expert Advisors and indicators it should be renamed to [OnTick](/en/docs/event_handlers/ontick) and [OnCalculate](/en/docs/event_handlers/oncalculate), respectively. The code that is to be executed during a mql5 program operation should be located in these three functions:

| mql5-program | main function |
| --- | --- |
| [script](/en/docs#script) | OnStart |
| [indicator](/en/docs#indicator) | OnCalculate |
| [Expert Advisor](/en/docs#expert) | OnTick |

If the indicator or script code does not contain the main function, or the function name differs from the required one, the call of this function is not performed. It means, if the source code of a script doesn't contain OnStart, such a code will be compiled as an Expert Advisor.

If an indicator code doesn't contain the OnCalculate function, the compilation of such an indicator is impossible.

## Predefined Variables

In MQL5 there are no such predefined variables as Ask, Bid, Bars. Variables Point and Digits have a slightly different spelling:

| MQL4 | MQL5 |
| --- | --- |
| Digits | \_Digits |
| Point | \_Point |
| | \_LastError |
| | \_Period |
| | \_Symbol |
| | \_StopFlag |
| | \_UninitReason |

## Access to Timeseries

In MQL5 there are no such predefined timeseries as Open[], High[], Low[], Close[], Volume[] and Time[]. The necessary depth of a timeseries can now be set using corresponding [functions to access timeseries](/en/docs/series).

## Expert Advisors

Expert Advisors in MQL5 do not require the obligatory presence of functions that handle the [events](/en/docs/basis/function/events) of a new tick receipt - OnTick, as it was in MQL4 (the start function in MQL4 is executed when a new tick is received). In MQL5 Expert Advisors can contain pre-defined handler functions of several types of events:

* [OnTick](/en/docs/event_handlers/ontick) – receipt of a new tick;
* [OnTimer](/en/docs/event_handlers/ontimer) – timer event;

* [OnTrade](/en/docs/event_handlers/ontrade) - trade event;

* [OnChartEvent](/en/docs/event_handlers/onchartevent) – events of input from the keyboard and mouse, events of a graphic object moving, event of a text editing completion in the entry field of the LabelEdit object;
* [OnBookEvent](/en/docs/event_handlers/onbookevent) – event of Depth of Market status change.

## Custom Indicators

In MQL4, the number of indicator buffers is limited and can't exceed 8. In MQL5 there are no such limitations, but it should be remembered that each indicator buffer requires allocation of a certain part of memory for its location in the terminal, so the new possibility should not be abused.

MQL4 offered only 6 types of custom indicator plotting; while MQL5 now offers 18 [drawing styles](/en/docs/constants/indicatorconstants/drawstyles#enum_draw_type). The names of drawing types haven't changed, but the ideology of the graphical representation of indicators has changed significantly.

The direction of indexing in indicator buffers also differs. By default, in MQL5 all the indicator buffers have the behavior of common arrays, i.e. 0 indexed element is the oldest one in the history, and as the index increases, we move from the oldest data to the most recent ones.

The only function for working with [custom indicators](/en/docs/customind) that was preserved from MQL4 is [SetIndexBuffer](/en/docs/customind/setindexbuffer). But its call has changed; now you should specify [type of data to be stored in an array](/en/docs/constants/indicatorconstants/customindicatorproperties#enum_indexbuffer_type_enum), linked to the indicator buffer.

Properties of custom indicators also have changed and expanded. New functions for [accessing timeseries](/en/docs/series) have been added, so the total calculation algorithm must be reconsidered.

## Graphical Objects

The number of graphical objects in MQL5 has been significantly increased. Besides, graphical objects can now be positioned in time with the accuracy of a second in a chart of any timeframe - now object anchor points are not rounded off to the bar opening time in the current price chart.

For the Arrow, Text and Label objects now you can specify [binding methods](/en/docs/constants/objectconstants/enum_anchorpoint), and for the Label, Button, Chart, Bitmap Label and Edit objects you can set [chart corner to which an object is attached](/en/docs/constants/objectconstants/enum_basecorner).

[SubwinOff](/en/docs/standardlibrary/controls/cappdialog/cappdialogsubwinoff "SubwinOff")

[List of MQL5 Functions](/en/docs/function_indices "List of MQL5 Functions")