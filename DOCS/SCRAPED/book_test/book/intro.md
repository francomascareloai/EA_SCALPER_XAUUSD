---
title: "Introduction to MQL5 and development environment"
url: "https://www.mql5.com/en/book/intro"
hierarchy: []
scraped_at: "2025-11-28 10:14:41"
---

# Introduction to MQL5 and development environment

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Introduction to MQL5 and development environment

* Introduction to MQL5 and development environment
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* [First program](/en/book/intro/first_program "First program")
* [Data types and values](/en/book/intro/types_and_values "Data types and values")
* [Variables and identifiers](/en/book/intro/variables_and_identifiers "Variables and identifiers")
* [Assignment and initialization, expressions and arrays](/en/book/intro/init_assign_express "Assignment and initialization, expressions and arrays")
* [Data input](/en/book/intro/data_input "Data input")
* [Error fixing and debugging](/en/book/intro/errors_debug "Error fixing and debugging")
* [Data output](/en/book/intro/a_data_output "Data output")
* [Formatting, indentation, and spaces](/en/book/intro/b_formatting "Formatting, indentation, and spaces")
* [Mini summary](/en/book/intro/c_summing_up "Mini summary")
* [Programming fundamentals](/en/book/basis "Programming fundamentals")
* [Object Oriented Programming](/en/book/oop "Object Oriented Programming")
* [Common APIs](/en/book/common "Common APIs")
* [Creating application programs](/en/book/applications "Creating application programs")
* [Trading automation](/en/book/automation "Trading automation")
* [Advanced language tools](/en/book/advanced "Advanced language tools")
* [Conclusion](/en/book/conclusion "Conclusion")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Introduction to MQL5 and development environment

One of the most important changes in MQL5 in its reincarnation in MetaTrader 5 is that it supports the object-oriented programming (OOP) concept. At the time of its appearance, the preceding MQL4 (the language of MetaTrader 4) was conventionally compared to the C programming language, while it is more reasonable to liken MQL5 to C++. In all fairness, it should be noted that today all OOP tools that initially had only been available in MQL5 were transferred into MQL4. However, users who scarcely know programming still perceive OOP as something too complicated.

In a sense, this book is aiming at making complex things simple. It is not to replace, but to be added to the MQL5 Language Reference that is supplied with the terminal and also available on the mql5.com website.

In this book, we are going to consistently tell you about all the components and techniques of programming in MQL5, taking baby steps so that each iteration is clear and the OOP technology gradually unlocks its potential that is especially notable, as with any powerful tool, when it is used properly and reasonably. As a result, the developers of MQL programs will be able to choose a preferred programming style suitable for a specific task, i.e., not only the object-oriented but also the 'old' procedural one, as well as use various combinations of them.

Users of the trading terminal can be conveniently classified into "programmers" (those who have already some experience in programming in at least one language) and "non-programmers" ("pure" traders interested in the customization capacity of the terminal using MQL5). The former ones can optionally skip the first and the second parts of this book describing the basic concepts of language and immediately start learning about the specific APIs (Application Programming Interfaces) embedded in MetaTrader 5. For the latter ones, progressive reading is recommended.

Among the category of "programmers," those knowing C++ have the best advantages, since MQL5 and C++ are similar. However, this "medal" has its reverse side. The matter is that MQL5 does not completely match with C++ (especially when compared to the recent standards). Therefore, attempts to write one structure or another through habit "as on pluses" will frequently be interrupted by unexpected errors of the compiler. Considering specific elements of the language, we will do our best to point out these differences.

Technical analysis, executing trading orders, or integration with external data sources — all these functions are available to the terminal users both from the user interface and via software tools embedded in MQL5.

Since MQL5 programs must perform different functions, there are some specialized program types supported in MetaTrader 5. This is a standard technique in many software systems. For example, in Windows, along with usual windowing programs, there are command-line-driven programs and services.

The following program types are available in MQL5:

* Indicators — programs aimed at graphically displaying data arrays computed by a given formula, normally based on the series of quotes;
* Expert Advisors — programs to automate trading completely or partly;
* Scripts — programs intended for performing one action at a time; and
* Services — programs for performing permanent background actions.

We will discuss the purposes and special features of each type in detail later. It is important to note now that they all are created in MQL5 and have much in common. Therefore, we will start learning with common features and gradually get to know about the specificity of each type.

The essential technical feature of MetaTrader consists in exerting the entire control in the client terminal, while commands initiated in it are sent to the server. In other words, MQL-based applications can only work within the client terminal, most of them requiring a 'live' connection to the server to function properly. No applications are installed on the server. The server just processes the orders received from the client terminal and returns the changes in the trading environment. These changes also become available to MQL5 programs.

Most types of MQL5 programs are executed in the chart context, i.e., to launch a program, you should 'throw' it onto the desired chart. The exception is only a special type, i.e., services: They are intended for background operation, without being attached to the chart.

We recall that all MQL5 programs are inside the working MetaTrader 5 folder, in the nested folder named /MQL5/<type>, where <type> is, respectively:

* Indicators
* Experts
* Scripts
* Services

Based on the MetaTrader 5 installation technique, the path to the working folder can be different (particularly, with the limited user rights in Windows, in a normal mode or portable). For example, it can be:

| |
| --- |
| C:/Program Files/MetaTrader 5/ |

or

| |
| --- |
| C:/Users/<username>/AppData/Roaming/MetaQuotes/Terminal/<instance\_id>/ |

The user can get to know where this folder is located exactly by executing the File -> Open data catalog command (it is available in both terminal and editor). Moreover, when creating a new program, you don't need to think of looking up the correct folder due to using the MQL Wizard embedded in the editor. It is called for by the File -> New command and allows selecting the required type of the MQL5 program. The relevant text file containing a source code template will be created automatically where necessary upon completing the Master and then opened for editing.

In the MQL5 folder, there are other nested folders, along with the above ones, and they are also directly related to MQL5 programming, but we will refer to them later.

 

| | |
| --- | --- |
| MQL5 Programming for Traders — Source Codes from the Book. Part 1 | [MQL5 Programming for Traders — Source Codes from the Book. Part 1](https://www.mql5.com/en/code/45590/) |
| Примеры из книги также доступны в публичном проекте \MQL5\Shared Projects\MQL5Book. | Examples from the book are also available in the [public project](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#public) \MQL5\Shared Projects\MQL5Book |

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")

[Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")