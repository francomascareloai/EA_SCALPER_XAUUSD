---
title: "Data types and values"
url: "https://www.mql5.com/en/book/intro/types_and_values"
hierarchy: []
scraped_at: "2025-11-28 09:47:48"
---

# Data types and values

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Data types and values

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* [First program](/en/book/intro/first_program "First program")
* Data types and values
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

# Data types and values

Along with calling the embedded function Symbol, we could also use our own function that we have defined in the source code. Suppose we would like to print in the log not just "Hello", but different greetings depending on the time of day. We will determine the time of day accurate to hours: 0-8 is morning, 8-16 is afternoon, and 16-24 is evening.

It is logical to suggest that the definition structure of the new function must be similar to that of the function OnStart already familiar to us. However, its name must be unique, i.e., it should not duplicate the names of other functions or reserved words. We will study the list of these words further in this textbook, while now luckily suggesting that the word Greeting can be used as a name.

Like the Symbol function, this function must return a string; this time, however, the string must be one of the following phrases, depending on the time of day: "Good morning", "Good afternoon", or "Good evening".

Guided by common sense, we are using the common concept of string here. Apparently, it is familiar to the compiler, because we saw how it had generated a program printing the predefined text. Thus, we have smoothly approached to the concept of types in the programming language, one of the types being a string, i.e., a sequence of characters.

In MQL5, this type is described by the keyword string. This is the second type we know, the first one was void. We have already seen a value of this type, without knowing it was that: It is the literal "Hello, ". When we just insert a constant (particularly, something like a quoted text) into the source code, its type description is not required: defines the correct type automatically.

Using the OnStart function description as a sample, we can suggest how the function Greeting should appear for a first approximation.

| |
| --- |
| string Greeting() { } |

This text indicates our intention to create the Greeting function, which can return an arbitrary value of the string type. However, for the function to really return something, it is necessary to use a special statement with the return operator. It is one of many MQL5 operators: We will explore them all later. If the function has a return value type other than void, it must contain the operator return.

Particularly, to return the former greeting string "Hello, " from the function, we should write:

| |
| --- |
| string Greeting() {   return "Hello, "; } |

Operator return stops the function execution and sends out what is to the right of it, as a result. "Out" hides the source code fragment, from which the function was called.

We have not explored all the options for writing expressions that could form an arbitrary string. However, the simplest instance with the quoted text is transferred here without any changes. It is important that the return value type coincides with the function type, as in our case. At the end of the statement, we put a semicolon.

However, we wanted to generate different greetings depending on the time of day. Therefore, the function must have an hour-defining parameter that can take values ranging from 0 through 23. Obviously, the hour number is an integer, i.e., a number that has no fractional part. It is clear that the time does not stop within an hour, and minutes are counted in it, the number of minutes being an integer, too. Then again, it is pointless to determine the time of day accurately to a minute. Therefore, we will limit ourselves to choosing the greeting by the hour number only.

For integer values, there is a special type int in MQL5. This value should be sent to the function Greeting from another place in the program, from which this function will be called. Here we have first faced the necessity of describing a named memory cell, that is, a variable.

[First program](/en/book/intro/first_program "First program")

[Variables and identifiers](/en/book/intro/variables_and_identifiers "Variables and identifiers")