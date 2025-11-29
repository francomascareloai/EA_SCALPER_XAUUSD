---
title: "Overview of macro substitution directives"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_define_overview"
hierarchy: []
scraped_at: "2025-11-28 09:49:30"
---

# Overview of macro substitution directives

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Overview of macro substitution directives

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* Overview of macro substitution directives
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

# Overview of macro substitution directives

Macro substitution directives include two forms of the #define directive:

* simple, usually to define a constant
* defining a macro as a pseudo-function with parameters

In addition, there is a #undef directive to undo any of the previous #define definitions. If #undef is not used, each defined macro is valid until the end of source compilation.

Macros are registered and then used in code by name, following the rules of identifiers. By convention, macro names are written in capital letters. Macro names can overlap the names of variables, functions, and other elements of the source code. Purposeful use of this fact allows the flexibility to change and generate source code on the fly. However, an unintentional coincidence of a macro name with a program element will result in errors.

The principle of operation of both forms of macro substitutions is the same. Using the #define directive, an identifier is introduced, which is associated with a certain piece of text — a definition. If the preprocessor finds a given identifier later in the source code, it replaces it with the text associated with it. We emphasize that the macro name can be used in compiled code only after registration (this is similar to the variable declaration principles, but only at the compilation stage).

Replacing a macro name with its definition is called expansion. The analysis of the source code occurs progressively and by one line in a pass, but the expansion in each line can be performed an arbitrary number of times, as in a loop, as long as macro names are found in the result. You cannot include the same name in a macro definition: when substituting, such a macro will result in an "unknown identifier" error.

In Part 3 of the book, we'll learn about [templates](/en/book/oop/templates), which also allow you to generate (or, in fact, replicate) source code, but with different rules. If there are both, macro substitution directives and templates in the source code, the macros are expanded first, and then the code is generated from the templates.

Macro names are highlighted in red in MetaEditor.

[Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")

[Simple form of #define](/en/book/basis/preprocessor/preprocessor_define_simple "Simple form of #define")