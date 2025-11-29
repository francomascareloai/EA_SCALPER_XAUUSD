---
title: "Including source files (#include)"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_include"
hierarchy: []
scraped_at: "2025-11-28 09:49:42"
---

# Including source files (#include)

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Including source files (#include)

* Including source files (#include)
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

# Including source files (#include)

The #include directive is used to include the contents of another file into the source code. The directive produces the same action as if the programmer copies the text from the include file to the clipboard and pastes it into the current file at the place where the directive is used.

Splitting source code into multiple files is a common practice when writing complex programs. Such programs are built on a modular basis so that each module/file contains logically related code that solves one or more related tasks.

Include files are also used to distribute libraries (sets of ready-made algorithms). The same library can be included in different programs. In this case, the library update (the update of its header file) will be automatically applied in all programs during their next compilation.

If the main files of MQL programs must have the mq5 extension, then the include files commonly have the extension mqh ('h' at the end of the word means "header"). At the same time, it is permissible to use the #include directive for other types of text files, for example, \*.txt (see below). In any case, when a file is included, the final program combined from the main mq5 file and all headers must still be syntactically correct. For example, including a file with binary information (like a png image) will break the compilation.

There are two types of #include statements:

| |
| --- |
| #include <file\_name> #include "file\_name" |

In the first one, the file name is enclosed in angle brackets. The compiler searches for such files in the terminal data directory in the MQL5/Include/ subfolder.

For the second one, with the name in quotes, the search is performed in the same directory which contains the current file that uses the #include statement.

In both cases, the file can be located in subfolders within the search directory. In this case, you should specify the entire relative hierarchy of folders before the file name in the directive. For example, along with MetaTrader 5, there are many commonly used boot files, among which is DateTime.mqh with a set of methods for working with date and time (they are designed as structures, the language constructs that we will discuss in Part 3 devoted to OOP). The DateTime.mqh file is located in the Tools folder. To include it in your source code, you should use the following directive:

| |
| --- |
| #include <Tools/DateTime.mqh> |

To demonstrate how to include a header file from the same folder as the source file with the directive, let's consider the file Preprocessor.mq5. It contains the following directive:

| |
| --- |
| #include "Preprocessor.mqh" |

It refers to the Preprocessor.mqh file, which is really located next to Preprocessor.mq5.

An include file can, in turn, include other files. In particular, inside Preprocessor.mqh there is the following code:

| |
| --- |
| double array[] = {    #include "Preprocessor.txt" }; |

It means that the contents of the array are initialized from the given text file. If we look inside Preprocessor.txt, we will see the text that complies with the array initialization syntax rules:

| |
| --- |
| 1, 2, 3, 4, 5 |

Thus, it is possible to collect source code from custom components, including generating it using other programs.

Note that if the file specified in the directive is not found, the compilation will fail.

The order in which multiple files are included determines the order in which the preprocessor directives in them are processed.

[Preprocessor](/en/book/basis/preprocessor "Preprocessor")

[Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")