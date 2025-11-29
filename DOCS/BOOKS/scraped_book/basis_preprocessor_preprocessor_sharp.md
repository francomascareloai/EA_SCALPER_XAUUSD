---
title: "Special operators '#' and '##' inside #define definitions"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_sharp"
hierarchy: []
scraped_at: "2025-11-28 09:49:40"
---

# Special operators '#' and '##' inside #define definitions

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Special operators '#' and '##' inside #define definitions

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* [Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")
* [Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")
* [Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")
* Special operators '#' and '##' inside #define definitions
* [Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")
* [Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")
* [Conditional compilation (#ifdef/#ifndef/#else/#endif)](/en/book/basis/preprocessor/preprocessor_ifdefs "Conditional compilation (#ifdef/#ifndef/#else/#endif)")
* [General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Special operators '#' and '##' inside #define definitions

Inside macro definitions, two special operators can be used:

* a single hash symbol '#' before the name of a macro parameter turns the contents of that parameter into a string; it is allowed only in function macros;
* a double hash symbol '##' between two words (tokens) combines them, and if the token is a macro parameter, then its value is substituted, but if the token is a macro name, it is substituted as is, without expanding the macro; if as a result of "gluing" another macro name is obtained, it is expanded;

In the examples in this book, we often used the following macro:

| |
| --- |
| #define PRT(A) Print(#A, "=", (A)) |

It calls the Print function, in which the passed expression is displayed as a string thanks to #A, and after the sign "equal", the actual value of A is printed.

To demonstrate '##', let's consider another macro:

| |
| --- |
| #define COMBINE(A,B,X) A##B(X) |

With it, we can actually generate a call to the SQN macro defined above:

| |
| --- |
| Print(COMBINE(SQ,N,2)); // 4 |

The literals SQ and N are concatenated, after which the macro SQN expands to ((2)\*(2)) and produces the result 4.

The following macro allows you to create a variable definition in code by generating its name given the parameters of the macro:

| |
| --- |
| #define VAR(TYPE,N) TYPE var##N = N |

Then the line of code:

| |
| --- |
| VAR(int, 3); |

is equivalent to the following:

| |
| --- |
| int var3 = 3; |

Concatenation of tokens allows the implementation of a loop shorthand over the array elements using a macro.

| |
| --- |
| #define for\_each(I, A) for(int I = 0, max\_##I = ArraySize(A); I < max\_##I; ++I)    // describe and somehow fill in the array x double x[]; // ... // implement loop through the array for\_each(i, x) {    x[i] = i \* i; } |

[Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")

[Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")