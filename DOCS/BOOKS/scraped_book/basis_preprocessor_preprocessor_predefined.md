---
title: "Predefined preprocessor constants"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_predefined"
hierarchy: []
scraped_at: "2025-11-28 09:49:35"
---

# Predefined preprocessor constants

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Predefined preprocessor constants

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* [Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")
* [Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")
* [Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")
* [Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")
* [Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")
* Predefined preprocessor constants
* [Conditional compilation (#ifdef/#ifndef/#else/#endif)](/en/book/basis/preprocessor/preprocessor_ifdefs "Conditional compilation (#ifdef/#ifndef/#else/#endif)")
* [General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Predefined preprocessor constants

MQL5 has several predefined constants that are equivalent to simple macros, but they are defined by the compiler itself. The following table lists some of their names and meanings.

| Name | Description |
| --- | --- |
| \_\_COUNTER\_\_ | Counter (each mention in the text during macro expansion results in an increase of 1) |
| \_\_DATE\_\_ | Compilation date (day) |
| \_\_DATETIME\_\_ | Compilation date and time |
| \_\_FILE\_\_ | The name of the compiled file |
| \_\_FUNCSIG\_\_ | Current function signature |
| \_\_FUNCTION\_\_ | Current function name |
| \_\_LINE\_\_ | Line number in the compiled file |
| \_\_MQLBUILD\_\_, \_\_MQL5BUILD\_\_ | Compiler version |
| \_\_RANDOM\_\_ | Random number of type ulong |
| \_\_PATH\_\_ | Path to compiled file |
| \_DEBUG | Defined when compiling in debug mode |
| \_RELEASE | Defined when compiling in normal mode |

[Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")

[Conditional compilation (#ifdef/#ifndef/#else/#endif)](/en/book/basis/preprocessor/preprocessor_ifdefs "Conditional compilation (#ifdef/#ifndef/#else/#endif)")