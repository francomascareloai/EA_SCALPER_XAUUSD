---
title: "Type conversion"
url: "https://www.mql5.com/en/book/basis/conversion"
hierarchy: []
scraped_at: "2025-11-28 09:48:38"
---

# Type conversion

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")Type conversion

* [Implicit type conversion](/en/book/basis/conversion/conversion_implicit "Implicit type conversion")
* [Arithmetic type conversions](/en/book/basis/conversion/conversion_arithmetic "Arithmetic type conversions")
* [Explicit type conversion](/en/book/basis/conversion/conversion_explicit "Explicit type conversion")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Type conversion

In this section, we will consider the concept of type conversion, limiting ourselves to built-in data types for now. Later, after studying OOP, we will supplement it with the nuances inherent in object types.

Type conversion in MQL5 is the process of changing the data type of a variable or expression. MQL5 supports three main types of type conversion: implicit, arithmetic, and explicit.

[Implicit type conversion](/en/book/basis/conversion/conversion_implicit):

* Occurs automatically when a variable of one type is used in a context that expects another type. For example, integer values can be implicitly converted to real values.

[Arithmetic type conversion](/en/book/basis/conversion/conversion_arithmetic):

* Arises during arithmetic operations with operands of different types. The compiler attempts to maintain maximum accuracy but warns about potential data loss. For instance, in integer division, the result is converted to a real type.

[Explicit type conversion](/en/book/basis/conversion/conversion_explicit):

* Gives the programmer control over type conversion. It is done in two forms: C-style ((target)) and "functional" style (target()). It is used when you need to explicitly instruct the compiler to perform a conversion between types, for example, when rounding real numbers or when successive type conversions are required.

Understanding the differences between implicit, arithmetic, and explicit type conversion is crucial for ensuring the correct execution of operations and avoiding data loss. This knowledge helps programmers effectively utilize this mechanism in MQL5 development.

[Priorities of operations](/en/book/basis/expressions/operators_precedence "Priorities of operations")

[Implicit type conversion](/en/book/basis/conversion/conversion_implicit "Implicit type conversion")