---
title: "Variables and identifiers"
url: "https://www.mql5.com/en/book/intro/variables_and_identifiers"
hierarchy: []
scraped_at: "2025-11-28 10:15:14"
---

# Variables and identifiers

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Variables and identifiers

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* [First program](/en/book/intro/first_program "First program")
* [Data types and values](/en/book/intro/types_and_values "Data types and values")
* Variables and identifiers
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

# Variables and identifiers

A variable is a memory cell having a unique name (to be referred to without any errors), which can store the values of a certain type. This ability is ensured by the fact that the compiler allocates for the variable just enough memory that is required for it in the special internal format: Each type is sized and has a relevant memory storing format. More details on this are given in Part 2.

Basically, there is a stricter term, identifier, in the program, which term is used for the names of variables, functions, and many other entities to be learned later herein. Identifier follows some rules. In particular, it may only contain Latin characters, numbers, and underscores; and it may not start with a number. This is why the word 'Greeting' chosen for the function earlier meets these requirements.

Values of a variable can be different, and they can be changed using special statements during the program execution.

Along with its type and name, a variable is characterized by the context, i.e., an area in the program, where it is defined and can be used without any errors of compiler. Our example will probably facilitate understanding this concept without any detailed technical reasoning in the beginning.

The matter is that a particular instance of a variable is the function parameter. The parameter is intended for sending a certain value into the function. Hereof it is obvious that the code fragment, where there is such a variable, must be limited to the body of the function. In other words, the parameter can be used in all statements inside the function block, but not outside. If the programming language allowed such liberties, this would become a source of many errors due to the potential possibility to 'spoil' the function inside from a random program fragment that is not related to the function.

In any case, it is a slightly simplified definition of a variable, which is sufficient for this introductory section. We will consider some finer nuances later.

Hence, let's generalize our knowledge of variables and parameters: They must have type, name, and context. We write the first two characteristics in the code explicitly, while the last one results from the definition location.

Let's see how we can define the parameter of the hour number in the Greeting function. We already know the desired type, it's int, and we can logically choose the name: hour.

| |
| --- |
| string Greeting(int hour) {   return "Hello, "; } |

This function will still return "Hello," whatever the hour. Now we should add some statements that would select different strings to return, based on the value of parameter hour. Please remember that there are three possible function response options: "Good morning", "Good afternoon", and "Good evening". We could suppose that we need 3 variables to describe these strings. However, it is much more convenient to use an array in such cases, which ensures a unified method of coding algorithms with access to elements.

[Data types and values](/en/book/intro/types_and_values "Data types and values")

[Assignment and initialization, expressions and arrays](/en/book/intro/init_assign_express "Assignment and initialization, expressions and arrays")