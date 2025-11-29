---
title: "Cancelling macro substitution (#undef)"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_undef"
hierarchy: []
scraped_at: "2025-11-28 09:49:39"
---

# Cancelling macro substitution (#undef)

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Cancelling macro substitution (#undef)

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* [Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")
* [Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")
* [Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")
* [Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")
* Cancelling macro substitution (#undef)
* [Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")
* [Conditional compilation (#ifdef/#ifndef/#else/#endif)](/en/book/basis/preprocessor/preprocessor_ifdefs "Conditional compilation (#ifdef/#ifndef/#else/#endif)")
* [General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Cancelling macro substitution (#undef)

Substitutions registered with #define can be undone if they are no longer needed after a particular piece of code. For these purposes, the #undef directive is used.

| |
| --- |
| #undef macro\_identifier |

In particular, it is useful if you need to define the same macro in different ways in different parts of the code. If the identifier specified in #define has already been registered somewhere in earlier lines of code (by another #define directive), then the old definition is replaced with the new one, and the preprocessor generates the "macro redefinition" warning. The use of #undef avoids the warning while explicitly indicating the programmer's intention not to use a particular macro further down the code.

#undef cannot undefine [predefined macros](/en/book/basis/preprocessor/preprocessor_predefined).

[Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")

[Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")