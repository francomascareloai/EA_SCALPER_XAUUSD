---
title: "Simple form of #define"
url: "https://www.mql5.com/en/book/basis/preprocessor/preprocessor_define_simple"
hierarchy: []
scraped_at: "2025-11-28 09:49:33"
---

# Simple form of #define

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Preprocessor](/en/book/basis/preprocessor "Preprocessor")Simple form of #define

* [Including source files (#include)](/en/book/basis/preprocessor/preprocessor_include "Including source files (#include)")
* [Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")
* Simple form of #define
* [Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")
* [Special operators '#' and '##' inside #define definitions](/en/book/basis/preprocessor/preprocessor_sharp "Special operators '#' and '##' inside #define definitions")
* [Cancelling macro substitution (#undef)](/en/book/basis/preprocessor/preprocessor_undef "Cancelling macro substitution (#undef)")
* [Predefined preprocessor constants](/en/book/basis/preprocessor/preprocessor_predefined "Predefined preprocessor constants")
* [Conditional compilation (#ifdef/#ifndef/#else/#endif)](/en/book/basis/preprocessor/preprocessor_ifdefs "Conditional compilation (#ifdef/#ifndef/#else/#endif)")
* [General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Simple form of #define

The simple form of the #define directive registers an identifier and the character sequence by which the identifier should be replaced everywhere in the source codes after the directive, up to the end of the program, or before the #undef directive with the same identifier.

Its syntax is:

| |
| --- |
| #define macro\_identifier [text] |

The text starts after the identifier and continues to the end of the current line. The identifier and text must be separated by an arbitrary number of spaces or tabs. If the required sequence of characters is too long, then for readability you can split it into several lines by putting a backslash character '\' at the end of the line.

| |
| --- |
| #define macro\_identifier text\_beginning \                            text\_continued \                            text\_ending |

The text can consist of any language constructs: constants, operators, identifiers, and punctuation marks. If you substitute macro\_identifier instead of the found constructs in the source code, all of them will be included in the compilation.

The simple form is traditionally used for several purposes:

1. Flag declarations, which are then used for [conditional compilation](/en/book/basis/preprocessor/preprocessor_ifdefs) checks;
2. Named constant declarations;
3. Abbreviated notation of common statements.

The first point is characterized by the fact that nothing needs to be specified after the identifier - the presence of a directive with a name is already enough for the corresponding identifier to be registered and can be used in conditional directives [#ifdef/ifndef](/en/book/basis/preprocessor/preprocessor_ifdefs). For them, it is only important whether the identifier exists or not, i.e. it works in the flag mode: declared / not declared. For example, the following directive defines the DEMO flag:

| |
| --- |
| #define DEMO |

It can then be used, say, to build a demo version of the program from which certain functions are excluded (see the example in the conditional compilation section).

The second way to use a simple directive allows you to replace the "magic numbers" in the source code with friendly names. "Magic numbers" are constants inserted into the source text, the meaning of which is not always clear (because a number is just a number: it is desirable to at least explain it in a comment). In addition, the same value can be scattered throughout different parts of the code, and if the programmer decides to change it to another, then he will have to do this in all places (and hope that he did not miss anything).

With a named macro, these two problems are easily solved. For example, a script can prepare an array with Fibonacci numbers to a certain maximum depth. Then it makes sense to define a macro with a predefined array size and use it in the description of the array itself (Preprocessor.mq5).

| |
| --- |
| #define MAX\_FIBO 10   int fibo[MAX\_FIBO]; // 10   void FillFibo() {    int prev = 0;    int result = 1;      for(int i = 0; i < MAX\_FIBO; ++i) // i < 10    {       int temp = result;       result = result + prev;       fibo[i] = result;       prev = temp;    } } |

If the programmer subsequently decides that the size of the array needs to be increased, it is enough for him to do this in one place - in the #define directive. Thus, the directive actually defines a certain parameter of the algorithm, which is "hardwired" into the source code and is not available for user configuration. The need for this arises quite often.

The question may arise how defining through #define differs from a constant variable in the global context. Indeed, we could declare a variable with the same name and purpose, and even preserve the uppercase letters:

| |
| --- |
| const int MAX\_FIBO = 10; |

However, in this case, MQL5 will not allow defining an array with the specified size, since only constants are allowed in square brackets, i.e. literals (and a constant variable, despite its similar name, is not a constant). To solve this problem, we could define an array as dynamic (without specifying a size first) and then allocate memory for it using the [ArrayResize](/en/book/common/arrays/arrays_dynamic) function - passing a variable as a size is not difficult here.

An alternative way to define a named constant is provided by enums, but is limited to integer values ​​only. For example:

| |
| --- |
| enum {    MAX\_FIBO = 10 }; |

But macro can contain a value of any type.

| |
| --- |
| #define TIME\_LIMIT     D'2023.01.01'  #define MIN\_GRID\_STEP  0.005 |

The search for macro names in source texts for replacement is performed taking into account the syntax of the language, that is, indivisible elements, such as variable identifiers or string literals, will remain unchanged, even if they include a substring that matches one of the macros. For example, given the macro XYZ below, the variable XYZAXES will be kept as it is, and the name XYZ (because it is exactly the same as the macro) will be changed to ABC.

| |
| --- |
| #define XYZ ABC int XYZAXES = 3; // int XYZAXES = 3 int XYZ = 0;     // int ABC = 0 |

Macro substitutions allow you to embed your code in the source code of other programs. This technique is usually used by libraries that are distributed as mqh header files and connected to programs using the [#include](/en/book/basis/preprocessor/preprocessor_include) directives.

In particular, for scripts, we can define our own library implementation of the OnStartfunction, which must perform some additional actions without affecting the original functionality of the program.

| |
| --- |
| void OnStart() {    Print("OnStart wrapper started");    // ... additional actions    \_OnStart();    // ... additional actions    Print("OnStart wrapper stopped"); }   #define OnStart \_OnStart |

Suppose this part is in the included header file (Preprocessor.mqh).

Then the original function OnStart (in Preprocessor.mq5) will be renamed by the preprocessor in the source code to \_OnStart (it is understood that this identifier is not used anywhere else for some other purpose). And the new version of OnStart from the header calls \_OnStart, "wrapping" it into additional statements.

The third common way to use the simple #define is to shorten the notation of language constructs. For example, the title of an infinite loop can be denoted with one word LOOP:

| |
| --- |
| #define LOOP for( ; !IsStopped() ; ) |

And then applied in code:

| |
| --- |
| LOOP {    // ...    Sleep(1000); } |

This method is also the main technique for using the #define directive with parameters (see below).

[Overview of macro substitution directives](/en/book/basis/preprocessor/preprocessor_define_overview "Overview of macro substitution directives")

[Form of #define as a pseudo-function](/en/book/basis/preprocessor/preprocessor_define_functional "Form of #define as a pseudo-function")