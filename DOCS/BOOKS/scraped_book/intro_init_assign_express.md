---
title: "Assignment and initialization, expressions and arrays"
url: "https://www.mql5.com/en/book/intro/init_assign_express"
hierarchy: []
scraped_at: "2025-11-28 09:47:46"
---

# Assignment and initialization, expressions and arrays

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Assignment and initialization, expressions and arrays

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* [First program](/en/book/intro/first_program "First program")
* [Data types and values](/en/book/intro/types_and_values "Data types and values")
* [Variables and identifiers](/en/book/intro/variables_and_identifiers "Variables and identifiers")
* Assignment and initialization, expressions and arrays
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

# Assignment and initialization, expressions and arrays

An array is a named set of same-type cells that are located in memory contiguously, each being accessible by its index. In a sense, it is a composite variable characterized by a common identifier, type of values stored, and quantity of numbered elements.

For instance, an array of 5 integers can be described as follows:

| |
| --- |
| int array[5]; |

Array size is specified in square brackets after the name. Elements are numbered from 0 through N-1, where N is the array size. They are accessed, i.e., the values are read, using a similar syntax. For example, to print the first element of the above array into the log, we could write the following statement:

| |
| --- |
| Print(array[0]); |

Please note that index 0 corresponds to the very first element. To print the last element, the statement would be replaced with the following:

| |
| --- |
| Print(array[4]); |

It is supposed, of course, that before printing an element of the array, a useful value has once been written into it. This record is made using a special statement, i.e., assignment operator. A special feature of this operator is the use of the symbol '=', to the left of which the array element (or variable) is specified, in which the record is made, while to the right of it the value to be recorded or its 'equivalent' is specified. Here, 'equivalent' hides the language ability to compute expressions of arithmetic, logic, and other types (we will learn them in Part 2). Syntax of the expressions is mostly similar to the rules of writing the equations learned in school-time arithmetic and algebra. For example, operations of addition ('+'), subtraction ('-'), multiplication ('\*'), and division ('/') can be used in an expression.

Below are examples of operators to fill out some elements of the array above.

| |
| --- |
| array[0] = 10;                       // 10 array[1] = array[0] + 1;             // 11 array[2] = array[0] \* array[1] + 1;  // 111 |

These statements demonstrate various methods of assignment and constructing expressions: In the first string, literal 10 is written into element array[0], while in the second and third lines, the expressions are used, computing which leads to obtaining the results specified for visual clarity in comments.

Where array elements (or variables, in a general case) are involved in an expression, the computer reads their values from memory during program execution and performs the above operations with them.

It is necessary to distinguish the use of variables and array elements to the left of and to the right of the '=' character in the assignment statement: On the left, there is a 'receiver' of the processed data (it is always single), while on the right, there are the 'sources' of initial data for computing (there can be many 'sources' in an expression, like in the last string of this example, where the values of elements array[0] and array[1] are multiplied together).

In our examples, the '=' character was used to assign the values to the elements of a predefined array. However, it is sometimes convenient to assign initial values to variables and arrays immediately upon defining them. This is called initialization. The '=' character is used for it, too. Let's consider this syntax in the context of our applied task.

Let's describe the array of strings with the greeting options inside the function Greeting:

| |
| --- |
| string Greeting(int hour) {   string messages[3] = {"Good morning", "Good afternoon", "Good evening"};   return "Hello, "; } |

In the statement added, not only the messages array with 3 elements is defined, but also its initialization, i.e., filling with the desired initial values. Initialization highlights the '=' character upon variable/array name and type description. For a variable, it is necessary to specify only one value after '=' (without braces), while for an array, as we can see, we can write several values separated by commas and enclosed in braces.

Do not confuse initialization with assignment. The former is specified in defining a variable/array (and is made once), while the latter occurs in specific statements (the same variable or array element can be assigned with different values over and over again). Array elements can only be assigned separately: MQL5 does not support assigning all elements at a time, as is the case with initialization.

The messages array, being defined inside the function, is available only inside it, like the parameter hour. Then we will see how we can describe variables available throughout the program code.

How shall we transform the incoming value of hour with the hour number into one of the three elements?

Recall that, according to our idea, hour can have values from 0 through 23. If we divide it by 8 exactly, we will obtain the values from 0 through 2. For instance, dividing 1 by 8 will give us 0, and 7 by 8 will give 0 (in exact division, the fractional part is neglected). However, dividing 8 by 8 is 1, so all numbers through 15 will give us 1 when divided by 8. Numbers 16-23 will correspond with the division result of 2. Integers 0, 1, and 2 obtained shall be used as indexes to read the messages array element.

In MQL5, operation '/' allows computing the exact division for integers.

Expression to obtain the division results is similar to those we have recently considered for the array, just the parameter hour and operation '/' must be used. We will use the following statement as a demonstration of a possible implementation of the hour transformation into the element index:

| |
| --- |
| int index = hour / 8; |

Here, a new integer variable, index, is defined and initialized by the value of the above expression.

However, we can omit saving the intermediate value in the index variable and immediately transfer this expression (to the right of '=') inside square brackets, where the array element number is specified.

Then in the statement with operator return, we can extract the relevant greeting as follows:

| |
| --- |
| string Greeting(int hour) {   string messages[3] = {"Good morning", "Good afternoon", "Good evening"};   return messages[hour / 8]; } |

The function is more or less ready. After a couple of sections, we will make some corrections, though. So far, let's save the project in a file under another name, GoodTime0.mq5, and try to call our function. For this reason, in OnStart, we will use the call for Greeting inside the Print call.

| |
| --- |
| void OnStart() {   Print(Greeting(0), ", ", Symbol()); } |

We have saved the separating comma (put inside lateral "Hello, ") between the greeting and the instrument name. Now there are three arguments in the Print function call: The first and the last ones will be computed on the fly using calls, respectively, of functions Greeting and Symbol, while the comma will be sent for printing as it is.

So far, we are sending the constant '0' into the function Greeting. It is its value that will get into the hour parameter. Having compiled and launched the program, we can make sure that it prints the desired text in the log.

| |
| --- |
| GoodTime0 (EURUSD,H1)        Good morning, EURUSD |

However, in practice, greetings must be selected dynamically, depending on the time specified by the user.

Thus, we have approached the need for arranging data input.

[Variables and identifiers](/en/book/intro/variables_and_identifiers "Variables and identifiers")

[Data input](/en/book/intro/data_input "Data input")