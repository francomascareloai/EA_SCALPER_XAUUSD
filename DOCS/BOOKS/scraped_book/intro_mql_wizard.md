---
title: "MQL Wizard and program draft"
url: "https://www.mql5.com/en/book/intro/mql_wizard"
hierarchy: []
scraped_at: "2025-11-28 09:47:37"
---

# MQL Wizard and program draft

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")MQL Wizard and program draft

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* MQL Wizard and program draft
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

# MQL Wizard and program draft

Here we will consider the simplest MQL program that does not really do anything. It is aimed at introducing the process of writing a source code in the editor, compiling it, and launching it in the terminal. Following the steps below independently, you will make sure that programming is available to casual users and start adapting to the integrated development environment of MQL5 programs. It will always be needed for consolidating the material covered.

The simplest MQL5 programs are scripts. Therefore, it is a script that we are going to try and create. For this purpose, let's start MQL5 Wizard (File -> New). In the first step, we will select Script in the list of types and press Next:

![Creating a Script Using MQL Wizard. Step 1](/en/book/img/wizard_start_en.png "Creating a Script Using MQL Wizard. Step 1")

Creating a script using MQL Wizard. Step 1

In the second step, we will introduce the script name in the Name field, having added it after the default folder mentioned above and a backslash: "Scripts\". For instance, let's name the script "Hello" (that is, the Name field will contain the line: "Scripts\Hello") and, without changing anything else, press Finish.

![Creating a Script Using MQL Wizard. Step 2](/en/book/img/wizard_script_name_params_en.png "Creating a Script Using MQL Wizard. Step 2")

Creating a script using MQL Wizard. Step 2

As a result, the Wizard will create a file named Hello.mq5 and open it for editing. The file is located in folder MQL5/Scripts (standard location for scripts) because we have used the default folder; however, we could add any sub-folder or even a sub-folder hierarchy. For instance, if we write "Scripts\Exercise\Hello" in the Name field at Wizard Step 1, then the Exercise sub-folder will be created in the Scripts folder automatically, and the file Hello.mq5 will be located in that sub-folder.

All examples from this book will be located in the MQL5Book folders inside catalogs allocated for the MQL programs of relevant types. This is necessary to facilitate installing the examples into your working copy of the terminal and rule out any name conflicts with other MQL programs you have already installed. 
  
For example, file Hello.mq5 delivered as part of this book is located in MQL5\Scripts\MQL5Book\p1\, where p1 means Part 1 this example relates to.

The resulting template of script Hello.mq5 contains the following text:

| |
| --- |
| //+------------------------------------------------------------------+ //|                                                        Hello.mq5 | //|                                  Copyright 2021, MetaQuotes Ltd. | //|                                             https://www.mql5.com | //+------------------------------------------------------------------+   #property copyright "Copyright 2021, MetaQuotes Ltd." #property link      "https://www.mql5.com" #property version   "1.00"   //+------------------------------------------------------------------+ //| Script program start function                                    | //+------------------------------------------------------------------+ void OnStart() { }   //+------------------------------------------------------------------+ |

It is this script that is shown in the preceding screenshots of MetaEditor and MetaTrader 5.

All strings starting with "//" are the comments and do not affect the program intent. They are neither processed by the compiler nor executed by the terminal. They are only used to exchange explanatory information among developers or to visually emphasize the code parts to enhance the text readability. For instance, in this template, the file starts with a block containing a comment where the script name and the author's copyright are expected to be specified. The second block of comments is the heading for the main function of the script — it is referred to in more detail below. Finally, the last comment string visually emphasizes the file end.

Three strings starting with a special directive, #property, provide the compiler with some attributes it builds into the program in a special manner. In our case, they are not important so far and can even be deleted. The specific directories are available to each MQL program type — we will know about them as soon as we proceed to learning the particular program types.

The main part of the script, where we are going to describe the essence of the program actions, is represented by the OnStart function. Here we have to learn the concepts of 'code block' and 'function'.

[Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")

[Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")