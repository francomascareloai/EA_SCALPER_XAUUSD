---
title: "Including Files (#include)"
url: "https://www.mql5.com/en/docs/basis/preprosessor/include"
hierarchy: []
scraped_at: "2025-11-28 09:30:08"
---

# Including Files (#include)

[MQL5 Reference](/en/docs "MQL5 Reference")[Language Basics](/en/docs/basis "Language Basics")[Preprocessor](/en/docs/basis/preprosessor "Preprocessor")Including Files (#include)

* [Macro substitution (#define)](/en/docs/basis/preprosessor/constant "Macro substitution (#define)")
* [Program Properties (#property)](/en/docs/basis/preprosessor/compilation "Program Properties (#property)")
* Including Files (#include)
* [Importing Functions (#import)](/en/docs/basis/preprosessor/import "Importing Functions (#import)")
* [Conditional Compilation (#ifdef, #ifndef, #else, #endif)](/en/docs/basis/preprosessor/conditional_compilation "Conditional Compilation (#ifdef, #ifndef, #else, #endif)")

# Including Files (#include)

The #include command line can be placed anywhere in the program, but usually all inclusions are placed at the beginning of the source code. Call format:

| |
| --- |
| #include <file\_name> #include "file\_name" |

Examples:

| |
| --- |
| #include <WinUser32.mqh> #include "mylib.mqh" |

The preprocessor replaces the line #include <file\_name> with the content of the file WinUser32.mqh. Angle brackets indicate that the WinUser32.mqh file will be taken from the standard directory (usually it is terminal\_installation\_directory\MQL5\Include). The current directory is not included in the search.

If the file name is enclosed in quotation marks, the search is made in the current directory (which contains the main source file). The standard directory is not included in the search.

See also

[Standard Library](/en/docs/standardlibrary), [Importing Functions](/en/docs/basis/preprosessor/import)

[Program Properties (#property)](/en/docs/basis/preprosessor/compilation "Program Properties (#property)")

[Importing Functions (#import)](/en/docs/basis/preprosessor/import "Importing Functions (#import)")