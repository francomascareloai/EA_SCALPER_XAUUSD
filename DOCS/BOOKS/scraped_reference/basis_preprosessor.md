---
title: "Preprocessor"
url: "https://www.mql5.com/en/docs/basis/preprosessor"
hierarchy: []
scraped_at: "2025-11-28 09:31:39"
---

# Preprocessor

[MQL5 Reference](/en/docs "MQL5 Reference")[Language Basics](/en/docs/basis "Language Basics")Preprocessor

* [Macro substitution (#define)](/en/docs/basis/preprosessor/constant "Macro substitution (#define)")
* [Program Properties (#property)](/en/docs/basis/preprosessor/compilation "Program Properties (#property)")
* [Including Files (#include)](/en/docs/basis/preprosessor/include "Including Files (#include)")
* [Importing Functions (#import)](/en/docs/basis/preprosessor/import "Importing Functions (#import)")
* [Conditional Compilation (#ifdef, #ifndef, #else, #endif)](/en/docs/basis/preprosessor/conditional_compilation "Conditional Compilation (#ifdef, #ifndef, #else, #endif)")

# Preprocessor

Preprocessor is a special subsystem of the MQL5 compiler that is intended for preparation of the program source code immediately before the program is compiled.

Preprocessor allows enhancement of the source code readability. The code can be structured by including of specific files containing source codes of mql5-programs. The possibility to assign mnemonic names to specific constants contributes to enhancement of the code readability.

Preprocessor also allows determining specific parameters of mql5-programs:

* [Declare constants](/en/docs/basis/preprosessor/constant)
* [Set program properties](/en/docs/basis/preprosessor/compilation)
* [Include files in program text](/en/docs/basis/preprosessor/include)
* [Import functions](/en/docs/basis/preprosessor/import)
* [Conditional Compilation](/en/docs/basis/preprosessor/conditional_compilation)

The preprocessor directives are used by the compiler to preprocess the source code before compiling it. The directive always begins with #, therefore the compiler prohibits using the symbol in names of variables, functions etc.

Each directive is described by a separate entry and is valid until the line break. You cannot use several directives in one entry. If the directive entry is too big, it can be broken into several lines using the '\' symbol. In this case, the next line is considered a continuation of the directive entry.

| |
| --- |
| //+------------------------------------------------------------------+ //|  foreach pseudo-operator                                         | //+------------------------------------------------------------------+ #define ForEach(index, array) for (int index = 0,                    \    max\_##index=ArraySize((array));                                   \    index<max\_##index; index++)     //+------------------------------------------------------------------+ //| Script program start function                                    | //+------------------------------------------------------------------+ void OnStart()   {    string array[]={"12","23","34","45"}; //--- bypass the array using ForEach    ForEach(i,array)      {       PrintFormat("%d: array[%d]=%s",i,i,array[i]);      }   } //+------------------------------------------------------------------+ /\* Output result      0: array[0]=12    1: array[1]=23    2: array[2]=34    3: array[3]=45 \*/ |

For the compiler, all these three #define directive lines look like a single long line. The example above also applies ## character which is a merge operator used in the #define macros to merge the two macro tokens into one. The tokens merge operator cannot be the first or last one in a macro definition.

[Creating and Deleting Objects](/en/docs/basis/variables/object_live "Creating and Deleting Objects")

[Macro substitution (#define)](/en/docs/basis/preprosessor/constant "Macro substitution (#define)")