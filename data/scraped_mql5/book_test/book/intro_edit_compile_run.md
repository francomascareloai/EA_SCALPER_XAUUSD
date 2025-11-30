---
title: "Editing, compiling, and running programs"
url: "https://www.mql5.com/en/book/intro/edit_compile_run"
hierarchy: []
scraped_at: "2025-11-28 10:14:12"
---

# Editing, compiling, and running programs

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Editing, compiling, and running programs

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* Editing, compiling, and running programs
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

# Editing, compiling, and running programs

All MetaTrader 5 programs are compilable. That is, a source code written in MQL5 must be compiled to obtain the binary representation that will be exactly the one executed in the terminal.

Programs are edited and compiled using MetaEditor.

![Editing an MQL-Program in MetaEditor](/en/book/img/me_edit_en.png "Editing an MQL-Program in MetaEditor")

Editing an MQL program in MetaEditor

Source code is a text written according to the MQL5 rules and saved as a file having the extension of mq5. The file containing a compiled program will have the same name, while its extension will be ex5.

In the simplest case, one executable file corresponds with one file containing the source code; however, as we will see later, coding complex programs frequently requires splitting the source code into multiple files: The main one and some supporting ones that are enabled from the main file in a special manner. In this case, the main file must still have the extension of mq5, while those enabled from it must have the extension of mqh. Then statements from all source files will get into the executable file being generated. Thus, multiple files containing the source code may be the starting point for creating one executable file/program. All this mentioned here to complete the picture is going to be presented in the second part of the book.

We will use the term MQL5 syntax to denote the set of all rules that allow constructing programs in MQL5. Only the strict adherence to the syntax allows coding programs compatible with the compiler. In fact, teaching to code consists of sequentially introducing all the rules of a particular language that is MQL5, in our case. And this is the main purpose of this book.

To compile a source code, we can use the command MetaEditor File -> Compile or just press F7. However, there are some other, special methods to compile – we will discuss them later. Compiling is accompanied by displaying the changing status in the editor log (where an MQL5 program consists of multiple files containing the source code, and enabling each file is marked in a single log line).

![Compiling an MQL Program in MetaEditor](/en/book/img/me_compile_en.png "Compiling an MQL Program in MetaEditor")

Compiling an MQL5 program in MetaEditor

An indication of a successful compilation is zero errors ("0 errors"). Warnings do not affect the compilation results, they just inform on potential issues. Therefore, it is recommended to fix them on the same basis as errors (we will tell you later how to do that). Ideally, there should not be any warnings ("0 warnings").

Upon the successful compilation of an mq5 file, we get a same-name file with the extension of ex5. MetaTrader 5 Navigator displays as a tree all executable ex5 files located in folder MQL5 and its subfolders, including the one just compiled.

![MetaTrader 5 Navigator with a Compiled MQL Program](/en/book/img/mt_navex_en.png "MetaTrader 5 Navigator with a Compiled MQL Program")

MetaTrader 5 Navigator with a compiled MQL5 program

Ready programs are launched in the terminal using any methods familiar to the user. For instance, any program, other than a service, can be dragged with the mouse from Navigator to the chart. We will talk about the features of services separately.

Besides, developers often need the program to be executed in the debugging mode to find what causes the errors. There are multiple special commands for this purpose, and we will refer to them in [Bug fixing and debugging](/en/book/intro/errors_debug).

[Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")

[MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")