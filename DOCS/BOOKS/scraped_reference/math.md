---
title: "Mathematical Functions"
url: "https://www.mql5.com/en/docs/math"
hierarchy: []
scraped_at: "2025-11-28 09:30:22"
---

# Mathematical Functions

[MQL5 Reference](/en/docs "MQL5 Reference")Math Functions

* [MathAbs](/en/docs/math/mathabs "MathAbs")
* [MathArccos](/en/docs/math/matharccos "MathArccos")
* [MathArcsin](/en/docs/math/matharcsin "MathArcsin")
* [MathArctan](/en/docs/math/matharctan "MathArctan")
* [MathArctan2](/en/docs/math/matharctan2 "MathArctan2")
* [MathClassify](/en/docs/math/mathclassify "MathClassify")
* [MathCeil](/en/docs/math/mathceil "MathCeil")
* [MathCos](/en/docs/math/mathcos "MathCos")
* [MathExp](/en/docs/math/mathexp "MathExp")
* [MathFloor](/en/docs/math/mathfloor "MathFloor")
* [MathLog](/en/docs/math/mathlog "MathLog")
* [MathLog10](/en/docs/math/mathlog10 "MathLog10")
* [MathMax](/en/docs/math/mathmax "MathMax")
* [MathMin](/en/docs/math/mathmin "MathMin")
* [MathMod](/en/docs/math/mathmod "MathMod")
* [MathPow](/en/docs/math/mathpow "MathPow")
* [MathRand](/en/docs/math/mathrand "MathRand")
* [MathRound](/en/docs/math/mathround "MathRound")
* [MathSin](/en/docs/math/mathsin "MathSin")
* [MathSqrt](/en/docs/math/mathsqrt "MathSqrt")
* [MathSrand](/en/docs/math/mathsrand "MathSrand")
* [MathTan](/en/docs/math/mathtan "MathTan")
* [MathIsValidNumber](/en/docs/math/mathisvalidnumber "MathIsValidNumber")
* [MathExpm1](/en/docs/math/mathexpm1 "MathExpm1")
* [MathLog1p](/en/docs/math/mathlog1p "MathLog1p")
* [MathArccosh](/en/docs/math/matharccosh "MathArccosh")
* [MathArcsinh](/en/docs/math/matharcsinh "MathArcsinh")
* [MathArctanh](/en/docs/math/matharctanh "MathArctanh")
* [MathCosh](/en/docs/math/mathcosh "MathCosh")
* [MathSinh](/en/docs/math/mathsinh "MathSinh")
* [MathTanh](/en/docs/math/mathtanh "MathTanh")
* [MathSwap](/en/docs/math/mathswap "MathSwap")

# Mathematical Functions

A set of mathematical and trigonometric functions.

Math functions were originally designed to perform relevant operations on scalar values. From this build on, most of the functions can be applied to [matrices and vectors](/en/docs/basis/types/matrix_vector). These include MathAbs, MathArccos, MathArcsin, MathArctan, MathCeil, MathCos, MathExp, MathFloor, MathLog, MathLog10, MathMod, MathPow, MathRound, MathSin, MathSqrt, MathTan, MathExpm1, MathLog1p, MathArccosh, MathArcsinh, MathArctanh, MathCosh, MathSinh, and MathTanh. Such operations imply element-wise handling of matrices or vectors. Example:

| |
| --- |
| //---   matrix a= {{1, 4}, {9, 16}};   Print("matrix a=\n",a);   a=MathSqrt(a);   Print("MatrSqrt(a)=\n",a);   /\*    matrix a=    [[1,4]     [9,16]]    MatrSqrt(a)=    [[1,2]     [3,4]]   \*/ |

For [MathMod](/en/docs/math/mathmod) and [MathPow](/en/docs/math/mathpow), the second element can be either a scalar or a matrix/vector of the appropriate size.

| Function | Action |
| --- | --- |
| [MathAbs](/en/docs/math/mathabs) | Returns absolute value (modulus) of the specified numeric value |
| [MathArccos](/en/docs/math/matharccos) | Returns the arc cosine of x in radians |
| [MathArcsin](/en/docs/math/matharcsin) | Returns the arc sine of x in radians |
| [MathArctan](/en/docs/math/matharctan) | Returns the arc tangent of x in radians |
| [MathArctan2](/en/docs/math/matharctan2) | Return the angle (in radians) whose tangent is the quotient of two specified numbers |
| [MathClassify](/en/docs/math/mathclassify) | Returns the type of a real number |
| [MathCeil](/en/docs/math/mathceil) | Returns integer numeric value closest from above |
| [MathCos](/en/docs/math/mathcos) | Returns the cosine of a number |
| [MathExp](/en/docs/math/mathexp) | Returns exponent of a number |
| [MathFloor](/en/docs/math/mathfloor) | Returns integer numeric value closest from below |
| [MathLog](/en/docs/math/mathlog) | Returns natural logarithm |
| [MathLog10](/en/docs/math/mathlog10) | Returns the logarithm of a number by base 10 |
| [MathMax](/en/docs/math/mathmax) | Returns the maximal value of the two numeric values |
| [MathMin](/en/docs/math/mathmin) | Returns the minimal value of the two numeric values |
| [MathMod](/en/docs/math/mathmod) | Returns the real remainder after the division of two numbers |
| [MathPow](/en/docs/math/mathpow) | Raises the base to the specified power |
| [MathRand](/en/docs/math/mathrand) | Returns a pseudorandom value within the range of 0 to 32767 |
| [MathRound](/en/docs/math/mathround) | Rounds of a value to the nearest integer |
| [MathSin](/en/docs/math/mathsin) | Returns the sine of a number |
| [MathSqrt](/en/docs/math/mathsqrt) | Returns a square root |
| [MathSrand](/en/docs/math/mathsrand) | Sets the starting point for generating a series of pseudorandom integers |
| [MathTan](/en/docs/math/mathtan) | Returns the tangent of a number |
| [MathIsValidNumber](/en/docs/math/mathisvalidnumber) | Checks the correctness of a real number |
| [MathExpm1](/en/docs/math/mathexpm1) | Returns the value of the expression MathExp(x)-1 |
| [MathLog1p](/en/docs/math/mathlog1p) | Returns the value of the expression MathLog(1+x) |
| [MathArccosh](/en/docs/math/matharccosh) | Returns the hyperbolic arccosine |
| [MathArcsinh](/en/docs/math/matharcsinh) | Returns the hyperbolic arcsine |
| [MathArctanh](/en/docs/math/matharctanh) | Returns the hyperbolic arctangent |
| [MathCosh](/en/docs/math/mathcosh) | Returns the hyperbolic cosine |
| [MathSinh](/en/docs/math/mathsinh) | Returns the hyperbolic sine |
| [MathTanh](/en/docs/math/mathtanh) | Returns the hyperbolic tangent |
| [MathSwap](/en/docs/math/mathswap) | Change the order of bytes in the [ushort](/en/docs/basis/types/integer/integertypes)/uint/ushort types value |

[StringFormat](/en/docs/convert/stringformat "StringFormat")

[MathAbs](/en/docs/math/mathabs "MathAbs")