---
title: "Statements"
url: "https://www.mql5.com/en/book/basis/statements"
hierarchy: []
scraped_at: "2025-11-28 10:15:58"
---

# Statements

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")Statements

* [Compound statements (blocks of code)](/en/book/basis/statements/statements_compound "Compound statements (blocks of code)")
* [Declaration/definition statements](/en/book/basis/statements/statements_declaration "Declaration/definition statements")
* [Simple statements (expressions)](/en/book/basis/statements/statements_expression "Simple statements (expressions)")
* [Overview of control statements](/en/book/basis/statements/statements_control "Overview of control statements")
* [For loop](/en/book/basis/statements/statements_for "For loop")
* [While loop](/en/book/basis/statements/statements_while "While loop")
* [Do loop](/en/book/basis/statements/statements_do "Do loop")
* [If selection](/en/book/basis/statements/statements_if "If selection")
* [Switch selection](/en/book/basis/statements/statements_switch "Switch selection")
* [Break jump](/en/book/basis/statements/statements_break "Break jump")
* [Continue jump](/en/book/basis/statements/statements_continue "Continue jump")
* [Return jump](/en/book/basis/statements/statements_return "Return jump")
* [Empty statement](/en/book/basis/statements/statements_null "Empty statement")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Statements

So far, we've learned about data types, variable declarations, and their use in expressions for calculations. However, these are only small bricks in the building with which the program can be compared. Even the simplest program consists of larger blocks that allow you to group related data processing operations and control the sequence of their execution. These blocks are called statements, and we have actually already used some of them.

In particular, the declaration of a variable (or several variables) is a statement. Assigning the expression evaluation result to a variable is also a statement. Strictly speaking, the assignment operation itself is part of the expression, so it is more correct to call such a statement a statement of expression. By the way, an expression may not contain an assignment operator (for example, if it simply calls some function that does not return a value, such as Print("Hello");).

Program execution is the progressive execution of statements: from top to bottom and from left to right (if there are several statements on one line). In the simplest case, their sequence is performed linearly, one after the other. For most programs, this is not enough, so there are various control statements. They allow you to organize loops (repeating calculations) in programs and the selection of algorithm operation options depending on the conditions.

Statements are special syntactic constructions that represent the source text written according to the rules. Statements of a particular type have their own rules, but there is something in common. Statements of all types end with a ';' except for the [compound statement](/en/book/basis/statements/statements_compound). It can do without a semicolon because its beginning and end are set by a pair of curly brackets. It is important to note that thanks to the compound statement, we can include sets of statements inside other statements, building arbitrary hierarchical structures of algorithms.

In this chapter, we will get acquainted with all types of MQL5 control statements, as well as consolidate the features of declaration and expression statements.

[Explicit type conversion](/en/book/basis/conversion/conversion_explicit "Explicit type conversion")

[Compound statements (blocks of code)](/en/book/basis/statements/statements_compound "Compound statements (blocks of code)")