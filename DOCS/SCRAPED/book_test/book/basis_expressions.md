---
title: "Expressions"
url: "https://www.mql5.com/en/book/basis/expressions"
hierarchy: []
scraped_at: "2025-11-28 10:15:52"
---

# Expressions

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")Expressions

* [Basic concepts](/en/book/basis/expressions/expressions_overview "Basic concepts")
* [Assignment operation](/en/book/basis/expressions/operator_assignment "Assignment operation")
* [Arithmetic operations](/en/book/basis/expressions/operators_arithmetic "Arithmetic operations")
* [Increment and decrement](/en/book/basis/expressions/increment_decrement "Increment and decrement")
* [Comparison operations](/en/book/basis/expressions/operators_relational "Comparison operations")
* [Logical operations](/en/book/basis/expressions/operators_logical "Logical operations")
* [Bitwise operations](/en/book/basis/expressions/operators_bitwise "Bitwise operations")
* [Modification operations](/en/book/basis/expressions/operators_compound "Modification operations")
* [Conditional ternary operator](/en/book/basis/expressions/operator_conditional "Conditional ternary operator")
* [Comma](/en/book/basis/expressions/operator_comma "Comma")
* [Special operators sizeof and typename](/en/book/basis/expressions/operators_sizeof_typename "Special operators sizeof and typename")
* [Grouping with parentheses](/en/book/basis/expressions/operators_parentheses "Grouping with parentheses")
* [Priorities of operations](/en/book/basis/expressions/operators_precedence "Priorities of operations")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Expressions

Expressions are essential elements of any programming language. Whatever applied idea underlies an algorithm, it is eventually reduced to data processing, that is, to computations. The expression describes computing some result from one or more predefined values. The values are called operands, while the actions performed with them are denoted by operations or operators.

As operators that allow manipulating with operands, independent characters or their sequences are used in expressions, such as '+' for addition or '\*' for multiplication. They all form several groups, such as arithmetic, bitwise, comparison, logic, and some specialized ones.

We have already used expressions in the previous sections of this book, such as to initialize variables. In the simplest case, the expression is a constant (literal) that is the only operand, while the computation result is equal to the operand value. However, operands can also be variables, array elements, function call results (for which the function is called directly from the expression), nested expressions, and other entities.

All operators substitute (return) their result into the parent expression, directly into the place where there were operands, which allows combining them making quite complex hierarchic structures. For example, in the following expression, the result of multiplying variables b by c is added to the value of variable a, and then the value obtained will be stored in variable v:

| |
| --- |
| v = a + b \* c; |

In this section, we consider the general principles of constructing and computing various expressions, as well as the standard set of operators supported in MQL5 for the built-in types. Later on, in the part dealing with OOP, we will know how operators can be reloaded (redefined) for custom types, i.e., structures and classes, which will allow us to use objects in expressions and perform nonstandard actions with them.

[Using arrays](/en/book/basis/arrays/arrays_usage "Using arrays")

[Basic concepts](/en/book/basis/expressions/expressions_overview "Basic concepts")