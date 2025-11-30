---
title: "First program"
url: "https://www.mql5.com/en/book/intro/first_program"
hierarchy: []
scraped_at: "2025-11-28 10:14:54"
---

# First program

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")First program

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* First program
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

# First program

Let's try to add to the script something simple but illustrative to demonstrate its operation. Let's rename the modified script as HelloChart.mq5.

In many programming textbooks, the initial example prints the sacramental "Hello, world". In MQL5, a similar greeting could appear as follows:

| |
| --- |
| void OnStart() {   Print("Hello, world"); } |

But we will make it more informative:

| |
| --- |
| void OnStart() {   Print("Hello, ", Symbol()); } |

Thus, we have added only one string with some language structures.

Here, Print is the name of the function embedded in the terminal and intended to display messages in the Expert Advisors log (tab Expert Advisors in the Tools window; despite its name Expert Advisors, the tab collects messages from MQL programs of all types). Unlike the function OnStart that we are defining independently, the Print function is defined for us in advance and forever. Print is one of many embedded functions constructing the MQL5 API (application programming interface).

The new line in our code denotes the statement to call the Print function sending into it the list of arguments (in parentheses) that will be printed in the log. Arguments in the list are separated by commas. In this case, there are two arguments: Line "Hello " and call for another embedded function, Symbol, that returns the name of the active instrument on the current chart (the value obtained from it will immediately get into the list of arguments of function Print, into the location from which the Symbol function has been called).

The Symbol function does not have any parameters and, therefore, nothing is sent into it inside parentheses.

For instance, if the script is located on the "EURUSD" chart, then calling the function Symbol() will return "EURUSD" and, in terms of the program being executed, the statement regarding calling the function Print will have a new look: Print("Hello, ", "EURUSD"). From a user's point of view, of course, all these calls for functions and the dynamic substitution of intermediary results are smooth and immediate. However, for a programmer, it is important to fully realize how the program is executed step by step to avoid logical errors and achieve strict compliance with the plan conceived.

The "Hello " line in double quotation marks is referred to as the literal, i.e., a fixed sequence of characters perceived by the computer as a text, as it is (as it is introduced in the source code of the program).

Thus, the printing statement above must print the two arguments one by one in the log, which should result in actually joining the two lines and obtaining "Hello, EURUSD".

Importantly, the comma inside the quotation marks will be printed in the log as a part of the line and is not processed in any special manner. Unlike that, the comma that is placed after the closing quotation mark and before calling Symbol() is the separating character in the argument list, i.e., affects the program behavior. If the first comma is omitted, the program will not lose its correctness, although it will print the word "Hello" without a comma after it. However, if the second comma is omitted, the program will stop being compiled, since the syntax of the function argument list will be broken: All values in it (in our case, these are two lines) must be separated by commas.

The compiler error will appear as follows:

| |
| --- |
| 'Symbol' - some operator expected        HelloChart.mq5        16        19 |

The compiler 'complains' of the lack of something before mentioning Symbol. This will break the compilation, and the executable file of the program is not created. Therefore, we will put the comma back in place.

This example shows us how important it is to strictly follow the syntax of the language. The same characters can work differently, being in different parts of the program. Thus, even a small omission may be critical. For instance, note the semicolon at the end of the line calling Print. The semicolon means the end of the statement here. If we forget to put it, strange compiler errors may occur.

To see this, we will try to remove this semicolon and re-compile the script. This results in obtaining new errors with the description of the problem and its place in the source code.

![Compilation errors in the MetaEditor log](/en/book/img/me_error_en.png "Compilation errors in the MetaEditor log")

Compilation errors in the MetaEditor log

 

| |
| --- |
| '}' - semicolon expected        HelloChart.mq5        17        1 '}' - unexpected end of program        HelloChart.mq5        17        1 |

The first error explicitly specifies the absence of the semicolon expected by the compiler. The second error is propagated: The closing brace signaling the end of the program had been detected before the current statement ended. In the compiler's opinion, it continues, because it has not encountered the semicolon yet. It is obvious how to fix the errors: The semicolon must be placed back in the right position in the statement.

Let's compile and launch the fixed script. Although it is executed very quickly and removed from the chart practically immediately and a record confirming the script operation appears in the Experts log.

| |
| --- |
| HelloChart (EURUSD,H1)        Hello, EURUSD |

[Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")

[Data types and values](/en/book/intro/types_and_values "Data types and values")