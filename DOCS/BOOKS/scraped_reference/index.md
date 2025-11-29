---
title: "MQL5 Reference"
url: "https://www.mql5.com/en/docs"
hierarchy: []
scraped_at: "2025-11-28 09:30:06"
---

# MQL5 Reference

MQL5 Reference

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
* [Moving from MQL4](/en/docs/migration "Moving from MQL4")
* [List of MQL5 Functions](/en/docs/function_indices "List of MQL5 Functions")
* [List of MQL5 Constants](/en/docs/constant_indices "List of MQL5 Constants")

MQL5 Help as One File:

[mql5.chm](https://www.mql5.com/files/docs/mql5.chm "mql5.chm")
[mql5.pdf](https://www.mql5.com/files/docs/mql5.pdf "mql5.pdf")
English

[mql5\_russian.chm](https://www.mql5.com/files/docs/mql5_russian.chm "mql5_russian.chm")
[mql5\_russian.pdf](https://www.mql5.com/files/docs/mql5_russian.pdf "mql5_russian.pdf")
Russian

[mql5\_german.chm](https://www.mql5.com/files/docs/mql5_german.chm "mql5_german.chm")
[mql5\_german.pdf](https://www.mql5.com/files/docs/mql5_german.pdf "mql5_german.pdf")
German

[mql5\_spanish.chm](https://www.mql5.com/files/docs/mql5_spanish.chm "mql5_spanish.chm")
[mql5\_spanish.pdf](https://www.mql5.com/files/docs/mql5_spanish.pdf "mql5_spanish.pdf")
Spanish

[mql5\_french.chm](https://www.mql5.com/files/docs/mql5_french.chm "mql5_french.chm")
[mql5\_french.pdf](https://www.mql5.com/files/docs/mql5_french.pdf "mql5_french.pdf")
French

[mql5\_chinese.chm](https://www.mql5.com/files/docs/mql5_chinese.chm "mql5_chinese.chm")
[mql5\_chinese.pdf](https://www.mql5.com/files/docs/mql5_chinese.pdf "mql5_chinese.pdf")
Chinese

[mql5\_italian.chm](https://www.mql5.com/files/docs/mql5_italian.chm "mql5_italian.chm")
[mql5\_italian.pdf](https://www.mql5.com/files/docs/mql5_italian.pdf "mql5_italian.pdf")
Italian

[mql5\_portuguese.chm](https://www.mql5.com/files/docs/mql5_portuguese.chm "mql5_portuguese.chm")
[mql5\_portuguese.pdf](https://www.mql5.com/files/docs/mql5_portuguese.pdf "mql5_portuguese.pdf")
Portuguese

[mql5\_turkish.chm](https://www.mql5.com/files/docs/mql5_turkish.chm "mql5_turkish.chm")
[mql5\_turkish.pdf](https://www.mql5.com/files/docs/mql5_turkish.pdf "mql5_turkish.pdf")
Turkish

[mql5\_japanese.chm](https://www.mql5.com/files/docs/mql5_japanese.chm "mql5_japanese.chm")
[mql5\_japanese.pdf](https://www.mql5.com/files/docs/mql5_japanese.pdf "mql5_japanese.pdf")
Japanese

[mql5\_korean.chm](https://www.mql5.com/files/docs/mql5_korean.chm "mql5_korean.chm")
[mql5\_korean.pdf](https://www.mql5.com/files/docs/mql5_korean.pdf "mql5_korean.pdf")
Korean

![Automated Trading Language Documentation](https://c.mql5.com/i/docs/background_docs.png "Automated Trading Language Documentation")

# MQL5 Reference

MetaQuotes Language 5 (MQL5) is a high-level language designed for developing technical indicators, trading robots and utility applications, which automate financial trading. MQL5 has been developed by [MetaQuotes](https://www.metaquotes.net) for their trading platform. The language syntax is very close to C++ enabling programmers to develop applications in the object-oriented programming (OOP) style.

In addition to the MQL5 language, the trading platform package also includes the [MetaEditor IDE](https://www.metatrader5.com/en/metaeditor/help) with highly advanced code writing tools, such as templates, snippets, debugging, profiling and auto completion tools, as well as built-in [MQL5 Storage](https://www.metatrader5.com/en/metaeditor/help/mql5storage) enabling file versioning.

The language support is available on the MQL5 Algotrading community website, which contains a huge [free CodeBase](https://www.mql5.com/en/code) and a plethora of [articles](https://www.mql5.com/en/articles). These articles cover all the aspects of the modern trading, including neural networks, statistics and analysis, high-frequency trading, arbitrage, testing and optimization of trading strategies, use of trading automation robots, and more.

Traders and MQL5 program developers can communicate on the forum, order and develop applications using the [Freelance](https://www.mql5.com/en/job) service, as well as buy and sell protected programs in the [Market](https://www.mql5.com/en/market) of automated trading applications.

The MQL5 language provides specialized [trading functions](/en/docs/trading) and predefined [event handlers](/en/docs/basis/function/events) to help programmers develop Expert Advisors (EAs), which automatically control trading processes following specific trading rules. In addition to EAs, MQL5 allows developing custom [technical indicators](/en/docs/customind), scripts and libraries.

This MQL5 language reference contains functions, operations, reserved words and other language constructions divided into categories. The reference also provides descriptions of [Standard Library](/en/docs/standardlibrary) classes used for developing trading strategies, control panels, custom graphics and enabling file access.

Additionally, the CodeBase contains the [ALGLIB](https://www.mql5.com/en/code/1146) numerical analysis library, which can be used for solving various mathematical problems.

 

### Algo Trading Books

Starting to learn something new is always challenging. To assist beginners, we have released two comprehensive books on MQL5 programming, designed for anyone who wish to master the creation of trading robots and applications for algorithmic trading.

These books offer a systematic and structured presentation of the material to make the learning process significantly easier. Detailed code examples, which explain the step-by-step creation of trading robots and applications, allow for a deeper understanding of algorithmic trading nuances. The books include numerous practical exercises to help reinforce the acquired knowledge and develop programming skills in real trading environments.

["MQL5 Programming for Traders](https://www.mql5.com/en/book "MQL5 Tutorial")" is the most complete and detailed tutorial on MQL5, suitable for programmers of all levels. Beginners will learn the basics: the book introduces development tools and basic programming concepts. Based on this material, you will create, compile and run your first application in the MetaTrader 5 trading platform. Users with experience in other programming languages can immediately proceed to the application part: creating trading robots and analytical applications in MQL5.

["Neural Networks for Algorithmic Trading with MQL5](https://www.mql5.com/en/neurobook "Book on Neural Networks with MQL5, OpenCL and Python")" is a guide to using machine learning methods in trading robots for the MetaTrader 5 platform. You will be progressively introduced to the fundamentals of neural networks and their application in algorithmic trading. As you advance, you will build and train your own AI solution, gradually adding new features. In addition to learning MQL5, you will gain Python and OpenCL programming skills and explore integrated matrix and vector methods, which enable the solution of complex mathematical problems with concise and efficient code.

 

### Articles on the development of trading applications

MQL5 Articles are an excellent resource for exploring the full potential of the language, covering a wide range of practical algorithmic trading tasks. For easy navigation, all articles are categorized into sections such as [Example](https://www.mql5.com/en/articles/examples), [Expert Advisors](https://www.mql5.com/en/articles/expert_advisors), [Machine Learning](https://www.mql5.com/en/articles/machine_learning) and more. Every month, dozens of new articles are published on the [MQL5 Algotrading community](https://www.mql5.com/) website, written by traders for other traders. Read and discuss these articles to master modern algorithmic trading. For beginners, we have compiled a list of [16 recommended articles](https://www.metatrader5.com/en/metaeditor/help/articles) for a quick immersion into MQL5. 
 

## Types of MQL5 Applications

MQL5 programs are divided into five specialized types based on the trading automation tasks that they implement:

* Expert Advisor is an automated trading system linked to a chart. An Expert Advisor contains [event](/en/docs/basis/function/events) handlers to manage predefined events which activate execution of appropriate trading strategy elements. For example, an event of program initialization and deinitializtion, new ticks, timer events, changes in the Depth of Market, chart and custom events. 
 In addition to calculating trading signals based on the implemented rules, Expert Advisors can also automatically execute trades and send them directly to a trading server. Expert Advisors are stored in <Terminal\_Directory>\MQL5\Experts.
* Custom Indicators is a technical indicator developed by a user in addition to standard indicators integrated into the trading platform. Custom indicators, as well as standard ones, cannot trade automatically, but only implement analytical functions. Custom indicators can utilize values of other indicators for calculations, and can be called from Expert Advisors. 
 Custom indicators are stored in <Terminal\_Directory>\MQL5\Indicators.
* Script is a program for a single execution of an action. Unlike Expert Advisors, scripts do not handle any event except for trigger. A script code must contain the OnStart handler function. 
 Scripts are stored in <Terminal\_DIrectory>\MQL5\Scripts.

* Service is a program that, unlike indicators, Expert Advisors and scripts, does not require to be bound to a chart to work. Like scripts, services do not handle any event except for trigger. To launch a service, its code should contain the OnStart handler function. Services do not accept any other events except Start, but they are able to send custom events to charts using [EventChartCustom](/en/docs/eventfunctions/eventchartcustom). Services are stored in <terminal\_directory>\MQL5\Services.

* Library is a set of custom functions. Libraries are intended to store and distribute commonly used algorithms of custom programs. 
 Libraries are stored in <Terminal\_Directory>\MQL5\Libraries.
* Include File is a source text of the most frequently used blocks of custom programs. Such files can be included into the source texts of Expert Advisors, scripts, custom indicators, and libraries at the compiling stage. The use of included files is more preferable than the use of libraries because of additional burden occurring at calling library functions. 
 Include files can be stored in the same directory where the original file is located. In this case the [#include](/en/docs/basis/preprosessor/include) directive with double quotes is used. Another option is to store include files in <Terminal\_Directory>\MQL5\Include. In this case #include with angle brackets should be used.

 

© 2000-2025, [MetaQuotes Ltd.](https://www.metaquotes.net)

[Language Basics](/en/docs/basis "Language Basics")