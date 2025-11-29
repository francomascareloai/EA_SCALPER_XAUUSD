---
title: "Mathematical functions"
url: "https://www.mql5.com/en/book/common/maths"
hierarchy: []
scraped_at: "2025-11-28 09:49:11"
---

# Mathematical functions

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Mathematical functions

* [The absolute value of a number](/en/book/common/maths/maths_abs "The absolute value of a number")
* [Maximum and minimum of two numbers](/en/book/common/maths/maths_max_min "Maximum and minimum of two numbers")
* [Rounding functions](/en/book/common/maths/maths_rounding "Rounding functions")
* [Remainder after division (Modulo operation)](/en/book/common/maths/maths_mod "Remainder after division (Modulo operation)")
* [Powers and roots](/en/book/common/maths/maths_pow_sqrt "Powers and roots")
* [Exponential and logarithmic functions](/en/book/common/maths/maths_exp_log "Exponential and logarithmic functions")
* [Trigonometric functions](/en/book/common/maths/maths_trig "Trigonometric functions")
* [Hyperbolic functions](/en/book/common/maths/maths_hyper "Hyperbolic functions")
* [Normality test for real numbers](/en/book/common/maths/maths_nan "Normality test for real numbers")
* [Random number generation](/en/book/common/maths/maths_rand "Random number generation")
* [Endianness control in integers](/en/book/common/maths/maths_byte_swap "Endianness control in integers")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Mathematical functions

The most popular mathematical functions are usually available in all modern programming languages, and MQL5 is no exception. In this chapter, we'll take a look at several groups of out-of-the-box functions. These include rounding, trigonometric, hyperbolic, exponential, logarithmic, and power functions, as well as a few special ones, such as generating random numbers and checking real numbers for normality.  

Most of the functions have two names: full (with the prefix "Math" and capitalization) and abbreviated (without a prefix, in lowercase letters). We will provide both options: they work the same way. The choice can be made based on the formatting style of the source codes.

Since mathematical functions perform some calculations and return a result as a real number, potential errors can lead to a situation where the result is undefined. For example, you cannot take the square root of a negative number or take the logarithm of zero. In such cases, the functions return special values that are not numbers (NaN, Not A Number). We have already faced them in the sections [Real numbers](/en/book/basis/builtin_types/float_numbers), [Arithmetic operations](/en/book/basis/expressions/operators_arithmetic), and [Numbers to strings and back](/en/book/common/conversions/conversions_numbers). The number correctness and the absence of errors can be analyzed using the MathIsValidNumber and MathClassify functions (see section [Checking real numbers for normality](/en/book/common/maths/maths_nan)).

The presence of at least one operand with a value of NaN will cause any subsequent computations implying this operand, including function calls, to also result in NaN.

For self-study and visual material, you can use the MathPlot.mq5 script as an attachment, which allows you to display mathematical function graphs with one argument from those described. The script uses the standard drawing library Graphic.mqh provided in MetaTrader 5 (outside the scope of this book). Below is a sample of what a hyperbolic sine curve might look like in the MetaTrader 5 window.

![Hyperbolic sine chart](/en/book/img/sinhplot.png "Hyperbolic sine chart")

Hyperbolic sine chart in the MetaTrader 5 window

[Zeroing objects and arrays](/en/book/common/arrays/zero_memory "Zeroing objects and arrays")

[The absolute value of a number](/en/book/common/maths/maths_abs "The absolute value of a number")