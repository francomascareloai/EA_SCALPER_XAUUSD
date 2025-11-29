---
title: "Conditional compilation (#ifdef/#ifndef/#else/#endif)"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_ifdefs"
hierarchy: []
scraped_at: "2025-11-28 09:49:37"
---

# Conditional compilation (#ifdef/#ifndef/#else/#endif)

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Conditional compilation (#ifdef/#ifndef/#else/#endif)

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* [Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")
* [Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")
* [Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")
* [Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")
* [Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")
* [Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")
* Conditional compilation (#ifdef/#ifndef/#else/#endif)
* [General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Conditional compilation (#ifdef/#ifndef/#else/#endif)

Conditional compilation directives allow you to include and exclude code fragments from the compilation process. The #ifdef and #ifndef directives mark the beginning of the code fragment they control. The fragment ends with the #endif directive. In the simplest case, the #ifdef syntax is as follows:

| |
| --- |
| #ifdef macro\_identifier   statements #endif |

If a macro with the specified identifier is defined above in the code using #define, then this code fragment will participate in compilation. Otherwise, it is excluded. In addition to the macros defined in the application code, the environment provides a set of predefined constants, in particular, the \_RELEASE and \_DEBUG flags (see section [Predefined constants](/en/book/basis/preprocessor/preprocessor_predefined)): their names can also be checked in conditional compilation directives.

The extended form #ifdef allows the specification of two pieces of code: the first will be included if the macro identifier is defined, and the second if it is not. To do this, a fragment separator #else is inserted between #ifdef and #endif.

| |
| --- |
| #ifdef macro\_identifier   statesments\_true #else   statements\_false #endif |

The #ifndef directive works similarly, but fragments are included and excluded according to the reverse logic: if the macro specified in the header is not defined, the first fragment is compiled, and if it is defined, the second fragment is compiled.

For example, depending on the presence of the DEMO macro substitution, we may or may not call the function for calculating Fibonacci numbers.

| |
| --- |
| #ifdef DEMO    Print("Fibo is disabled in the demo"); #else    FillFibo(); #endif |

In this case, if the DEMO mode is enabled, instead of calling the function, a message would be displayed in the log, but since in the Preprocessor.mq5 script and all the included files there is no #define DEMO definition, compilation proceeds according to branch #else, that is, the call to the FillFibo function gets into the executable ex5 file.

Directives can be nested.

| |
| --- |
| #ifdef \_DEBUG    Print("Debugging"); #else    #ifdef \_RELEASE       Print("Normal run");    #else       Print("Undefined mode!");    #endif #endif |

[Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")

[General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")