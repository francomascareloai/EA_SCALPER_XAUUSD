---
title: "Macro substitution (#define)"
url: "https://www.mql5.com/en/docs/basis/preprosessor/constant"
hierarchy: []
scraped_at: "2025-11-28 09:31:41"
---

# Macro substitution (#define)

[MQL5 Reference](/en/docs "MQL5 Reference")[Language Basics](/en/docs/basis "Language Basics")[Preprocessor](/en/docs/basis/preprosessor "Preprocessor")Macro substitution (#define)

* Macro substitution (#define)
* [Program Properties (#property)](/en/docs/basis/preprosessor/compilation "Program Properties (#property)")
* [Including Files (#include)](/en/docs/basis/preprosessor/include "Including Files (#include)")
* [Importing Functions (#import)](/en/docs/basis/preprosessor/import "Importing Functions (#import)")
* [Conditional Compilation (#ifdef, #ifndef, #else, #endif)](/en/docs/basis/preprosessor/conditional_compilation "Conditional Compilation (#ifdef, #ifndef, #else, #endif)")

# Macro substitution (#define)

The preprocessor directives are used by the compiler to preprocess the source code before compiling it. The directive always begins with #, therefore the compiler prohibits using the symbol in names of variables, functions etc.

Each directive is described by a separate entry and is valid until the line break. You cannot use several directives in one entry. If the directive entry is too big, it can be broken into several lines using the '\' symbol. In this case, the next line is considered a continuation of the directive entry.

The #define directive can be used to assign mnemonic names to constants. There are two forms:

| |
| --- |
| #define identifier expression                   // parameter-free form #define identifier(par1,... par8) expression    // parametric form |

The #define directive substitutes expression for all further found entries of identifier in the source text. The identifier is replaced only if it is a separate token. The identifier is not replaced if it is part of a comment, part of a string, or part of another longer identifier.

The constant identifier is governed by the same rules as variable names. The value can be of any type:

| |
| --- |
| #define ABC               100 #define PI                3.14 #define COMPANY\_NAME      "MetaQuotes Software Corp." ... void ShowCopyright()   {    Print("Copyright  2001-2009, ",COMPANY\_NAME);    Print("https://www.metaquotes.net");   } |

expression can consist of several tokens, such as keywords, constants, constant and non-constant expressions. expression ends with the end of the line and can't be transferred to the next line.

Example:

| |
| --- |
| #define TWO        2 #define THREE      3 #define INCOMPLETE TWO+THREE #define COMPLETE  (TWO+THREE) void OnStart()   {    Print("2 + 3\*2 = ",INCOMPLETE\*2);    Print("(2 + 3)\*2 = ",COMPLETE\*2);   } // Result // 2 + 3\*2 = 8 // (2 + 3)\*2 = 10 |

 

### Parametric Form #define

With the parametric form, all the subsequent found entries of identifier will be replaced by expression taking into account the actual parameters. For example:

| |
| --- |
| // example with two parameters a and b #define A 2+3 #define B 5-1 #define MUL(a, b) ((a)\*(b))   double c=MUL(A,B); Print("c=",c); /\* expression double c=MUL(A,B); is equivalent to double c=((2+3)\*(5-1)); \*/ // Result // c=20 |

Be sure to enclose parameters in parentheses when using the parameters in expression, as this will help avoid non-obvious errors that are hard to find. If we rewrite the code without using the brackets, the result will be different:

| |
| --- |
| // example with two parameters a and b #define A 2+3 #define B 5-1 #define MUL(a, b) a\*b   double c=MUL(A,B); Print("c=",c); /\* expression double c=MUL(A,B); is equivalent to double c=2+3\*5-1; \*/ // Result // c=16 |

When using the parametric form, maximum 8 parameters are allowed.

| |
| --- |
| // correct parametric form #define LOG(text)  Print(\_\_FILE\_\_,"(",\_\_LINE\_\_,") :",text)   // one parameter - 'text'    // incorrect parametric form          #define WRONG\_DEF(p1, p2, p3, p4, p5, p6, p7, p8, p9)   p1+p2+p3+p4 // more than 8 parameters from p1 to p9 |

### 

### The #undef directive

The #undef directive cancels declaration of the macro substitution, defined before.

Example:

| |
| --- |
| #define MACRO   void func1()   { #ifdef MACRO    Print("MACRO is defined in ",\_\_FUNCTION\_\_);    #else    Print("MACRO is not defined in ",\_\_FUNCTION\_\_); #endif   }   #undef MACRO   void func2()   { #ifdef MACRO    Print("MACRO is defined in ",\_\_FUNCTION\_\_); #else    Print("MACRO is not defined in ",\_\_FUNCTION\_\_); #endif   }   void OnStart()   {    func1();    func2();   }   /\* Result:  MACRO is defined in func1  MACRO is not defined in func2 \*/ |

See also

[Identifiers](/en/docs/basis/syntax/identifiers), [Character Constants](/en/docs/basis/types/integer/symbolconstants)

[Preprocessor](/en/docs/basis/preprosessor "Preprocessor")

[Program Properties (#property)](/en/docs/basis/preprosessor/compilation "Program Properties (#property)")