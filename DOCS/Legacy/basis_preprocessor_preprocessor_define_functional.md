---
title: "#define form as a pseudo-function"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_define_functional"
hierarchy: []
scraped_at: "2025-11-28 09:49:29"
---

# #define form as a pseudo-function

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Form of #define as a pseudo-function

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* [Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")
* [Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")
* Form of #define as a pseudo-function
* [Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")
* [Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")
* [Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")
* [Conditional compilation (#ifdef/#ifndef/#else/#endif)](/en/book/basis/preprocessor/preprocessor_ifdefs "Conditional compilation (#ifdef/#ifndef/#else/#endif)")
* [General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# #define form as a pseudo-function

The syntax of the parametric form #define is similar to a function.

| |
| --- |
| #define macro\_identifier(parameter,...) text\_with\_parameters |

Such a macro has one or more parameters in parentheses. Parameters are separated by commas. Each parameter is a simple identifier (often a single letter). Moreover, all parameters of one macro must have different identifiers.

It is important that there is no space between the identifier and the opening parenthesis, otherwise the macro will be treated as a simple form in which the replacement text starts with an opening parenthesis.

After this directive is registered, the preprocessor will search the source codes for lines of the form:

| |
| --- |
| macro\_identifier(expression,...) |

Arbitrary expressions can be specified instead of parameters. The number of arguments must match the number of macro parameters. All found occurrences will be replaced with text\_with\_parameters, in which, in turn, the parameters will be replaced with the passed expressions. Each parameter can occur several times, in any order.

For example, the following macro finds the maximum of two values:

| |
| --- |
| #define MAX(A,B) ((A) > (B) ? (A) : (B)) |

If the code contains the statement:

| |
| --- |
| int z = MAX(x, y); |

it will be "expanded" by the preprocessor into:

| |
| --- |
| int z = ((x) > (y) ? (x) : (y)); |

Macro substitution will work for any data type (for which the operations applied inside the macro are valid).

However, substitution can also have side effects. For example, if the actual parameter is a function call or statement that modifies the variable (say, ++x), then the corresponding action can be performed multiple times (instead of the intended one time). In the case of MAX, this will happen twice: during the comparison and when getting values in one of the branches of the '?:' operator. In this regard, it makes sense to convert such macros into functions whenever possible (especially considering that in MQL5 functions are automatically inlined).

There are parentheses around the parameters and around the entire macro definition. They are used to ensure that the substitution of expressions as parameters or the macro itself inside other expressions does not distort the computing order due to different priorities. Let's say the macro defines the multiplication of two parameters (not yet enclosed in parentheses):

| |
| --- |
| #define MUL(A,B) A \* B |

Then the use of the macro with the following expressions will produce unexpected results:

| |
| --- |
| int x = MUL(1 + 2, 3 + 4); // 1 + 2 \* 3 + 4 |

Instead of multiplication (1 + 2) \* (3 + 4) which gives 21, we have 1 + 2 \* 3 + 4, i.e., 11. The appropriate macro definition should be like this:

| |
| --- |
| #define MUL(A,B) ((A) \* (B)) |

You can specify another macro as a macro parameter. In addition, you can also insert other macros in a macro definition. All such macros will be replaced sequentially. For example:

| |
| --- |
| #define SQ3(X) (X \* X \* X) #define ABS(X) MathAbs(SQ3(X)) #define INC(Y) (++(Y)) |

Then the following code will print 504 (MathAbs is a built-in function that returns the modulus of a number, i.e. without a sign):

| |
| --- |
| int x = -10; Print(ABS(INC(x))); // -> ABS(++(Y)) // -> MathAbs(SQ3(++(Y))) // -> MathAbs((++(Y))\*(++(Y))\*(++(Y))) // -> MathAbs(-9\*-8\*-7) // -> 504 |

In the variable x, the value -7 will remain (due to the triple increment).

A macro definition can contain unmatched parentheses. This technique is used, as a rule, in a pair of macros, one of which should open a certain piece of code, and the other should close it. In this case, unmatched parentheses in each of them will become matched. In particular, in standard library files available in the MetaTrader 5 distribution package, in Controls/Defines.mqh, the EVENT\_MAP\_BEGIN and EVENT\_MAP\_END macros are defined. They are used to form the event processing function in graphical objects.

The preprocessor reads the entire source text of the program line by line, starting from the main mq5 file and inserting the texts from the header files encountered in place. By the time any line of code is read, a certain set of macros that are already defined is formed. It does not matter in which order the macros were defined: it is quite possible that one macro refers in its definition to another, which was described both above and below in the text. It is only important that in the line of source code where the macro name is used, the definitions of all referenced macros are known.

Consider an example.

| |
| --- |
| #define NEG(x) (-SQN(x))\*TEN #define SQN(x) ((x)\*(x)) #define TEN 10 ... Print(NEG(2)); // -40 |

Here, the NEG macro uses the SQN and TEN macros, which are described below it. And this does not prevent us from successfully using it in the code after all three #define-s.

However, if we change the relative position of the rows to the following:

| |
| --- |
| #define NEG(x) (-SQN(x))\*TEN #define SQN(x) ((x)\*(x)) ... Print(NEG(2)); // error: 'TEN' - undeclared identifier ... #define TEN 10 |

we get an "undeclared identifier" compilation error.

[Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")

[Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")