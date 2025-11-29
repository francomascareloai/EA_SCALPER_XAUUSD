---
title: "Functions"
url: "https://www.mql5.com/en/book/basis/functions"
hierarchy: []
scraped_at: "2025-11-28 10:15:55"
---

# Functions

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")Functions

* [Function definition](/en/book/basis/functions/functions_definition "Function definition")
* [Function call](/en/book/basis/functions/functions_call "Function call")
* [Parameters and arguments](/en/book/basis/functions/functions_parameters "Parameters and arguments")
* [Value parameters and reference parameters](/en/book/basis/functions/functions_ref_value "Value parameters and reference parameters")
* [Optional parameters](/en/book/basis/functions/functions_parameters_default "Optional parameters")
* [Return values](/en/book/basis/functions/functions_return "Return values")
* [Function declaration](/en/book/basis/functions/functions_declaration "Function declaration")
* [Recursion](/en/book/basis/functions/functions_recursive "Recursion")
* [Function overloading](/en/book/basis/functions/functions_overloading "Function overloading")
* [Function pointers (typedef)](/en/book/basis/functions/functions_typedef "Function pointers (typedef)")
* [Inlining](/en/book/basis/functions/functions_inline "Inlining")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Functions

A function is a named block with statements. Almost the entire application algorithm of the program is contained in functions. Outside of functions, only auxiliary operations are performed, such as creating and deleting global variables.

The execution of statements within a function occurs when we call that function. Some functions, the main ones, are called automatically by the terminal when various events occur. They are also referred to as the MQL program entry points or event handlers. In particular, we already know that when we run a script on a chart, the terminal calls its main function OnStart. In other types of programs, there are other functions called by the terminal, which we will discuss in detail in the [fifth](/en/book/applications) and [sixth](/en/book/automation) chapters covering the trading architecture of the MQL5 API.

In this chapter, we will learn how to define and declare a function, how to describe and pass parameters to it, and how to return the result of its work from the function.

We will also talk about function overloading, i.e., the ability to provide multiple functions with the same name, and how this can be useful.

Finally, we will get acquainted with a new type: a pointer to a function.

[Empty statement](/en/book/basis/statements/statements_null "Empty statement")

[Function definition](/en/book/basis/functions/functions_definition "Function definition")