---
title: "Preprocessor"
url: "https://www.mql5.com/en/book/basis/preprocessor"
hierarchy: []
scraped_at: "2025-11-28 09:48:40"
---

# Preprocessor

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")Preprocessor

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* [Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")
* [Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")
* [Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")
* [Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")
* [Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")
* [Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")
* [Conditional compilation (#ifdef/#ifndef/#else/#endif)](/en/book/basis/preprocessor/preprocessor_ifdefs "Conditional compilation (#ifdef/#ifndef/#else/#endif)")
* [General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Preprocessor

Up to this moment, we have been studying MQL5 programming, assuming that source codes are processed by the compiler, which converts their textual representation into binary (executable by the terminal). However, the first tool that reads and, if necessary, converts source codes is the preprocessor. This utility built into MetaEditor is controlled by special directives inserted directly into the source code. It can solve a number of problems that programmers face when preparing source codes.

Similarly to the C++ preprocessor, MQL5 supports the definition of macro substitutions (#define), conditional compilation (#ifdef) and inclusion of other source files (#include ). In this chapter, we will explore these possibilities. Some of them have limitations compared to C++.

In addition to the standard directives, the MQL5 preprocessor has its own specific ones, in particular, a set of MQL program properties ([#property](/en/book/basis/preprocessor/preprocessor_properties)), and functions import from separate EX5 and DLLs (#import). We will address them in the fifth, sixth and seventh parts when studying various types of MQL programs.

All preprocessor directives begin with a hash sign '#' followed by a keyword and additional parameters, the syntax of which depends on the type of directive.

It is recommended to start a preprocessor directive from the very beginning of the line, or at least after a whitespace indent (if the directives are nested). Inserting a directive inside source code statements is considered a bad programming style (unlike MQL5, the C++ preprocessor does not allow this at all).

Preprocessor directives are not language statements and should not be terminated with a ';'. Directives usually continue to the end of the current line. In some cases, they can be extended in a special way for the following lines, which will be discussed separately.

The directives are executed sequentially, in the same order in which they occur in the text and taking into account the processing of previous directives. For example, if another file is connected to a file using the [#include](/en/book/basis/preprocessor/preprocessor_include) directive and a substitution rule is defined in the included file using [#define](/en/book/basis/preprocessor/preprocessor_define_simple), then this rule starts working for all subsequent lines of code, including the header files included later.

The preprocessor does not process comments.

[Inlining](/en/book/basis/functions/functions_inline "Inlining")

[Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")