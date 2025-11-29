---
title: "Data output"
url: "https://www.mql5.com/en/book/intro/a_data_output"
hierarchy: []
scraped_at: "2025-11-28 09:47:34"
---

# Data output

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Data output

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* [First program](/en/book/intro/first_program "First program")
* [Data types and values](/en/book/intro/types_and_values "Data types and values")
* [Variables and identifiers](/en/book/intro/variables_and_identifiers "Variables and identifiers")
* [Assignment and initialization, expressions and arrays](/en/book/intro/init_assign_express "Assignment and initialization, expressions and arrays")
* [Data input](/en/book/intro/data_input "Data input")
* [Error fixing and debugging](/en/book/intro/errors_debug "Error fixing and debugging")
* Data output
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

# Data output

In the case of our script, data are output by simply recording the greeting into the log using the Print function. Where necessary, MQL5 allows saving the results in files and databases, sending over the Internet, and displaying as graphical series (in indicators) or objects on charts.

The simplest way to communicate some simple momentary information to the user without making him or her looking into the log (which is a service tool for monitoring the operation of programs and may be hidden from the screen) is provided by the MQL5 API function Comment. It can be used exactly as that of Print. However, its execution results in displaying the text not in the log, but on the current chart, in its upper left corner.

For instance, having replaced Print with Comment in the text script, we will obtain such a function Greeting:

| |
| --- |
| void OnStart() {   Comment(Greeting(GreetingHour), ", ", Symbol()); } |

Having launched the changed script in the terminal, we will see the following:

![Displaying text information on the chart using function Comment](/en/book/img/comment.png "Displaying text information on the chart using function Comment")

Displaying text information on the chart using the Comment function

If we need both display the text for the user and draw their attention to a change in the environment, related to the new information, it is better to use function Alert. It sends a notification into a separate terminal window that pops up over the main window, accompanying it with a sound alert. It is useful, for example, in case of a trade signal or non-routine events requiring the user's intervention.

The syntax of Alert is identical to that of Print and Comment.

The image below shows the result of the Alert function operation.

![Displaying a notification using function Alert](/en/book/img/alert-ru.png "Displaying a notification using function Alert")

Displaying a notification using the Alert function

Script versions with functions Comment and Alert are not attached to this book for the reader to independently try and edit GoodTime2.mq5 and reproduce the screenshots provided herein.

[Error fixing and debugging](/en/book/intro/errors_debug "Error fixing and debugging")

[Formatting, indentation, and spaces](/en/book/intro/b_formatting "Formatting, indentation, and spaces")