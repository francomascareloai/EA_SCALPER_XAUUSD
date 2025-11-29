---
title: "Development and connection of binary format libraries"
url: "https://www.mql5.com/en/book/advanced/libraries"
hierarchy: []
scraped_at: "2025-11-28 09:48:43"
---

# Development and connection of binary format libraries

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Advanced language tools](/en/book/advanced "Advanced language tools")Development and connection of binary format libraries

* [Creation of ex5 libraries; export of functions](/en/book/advanced/libraries/libraries_export "Creation of ex5 libraries; export of functions ")
* [Including libraries; #import of functions](/en/book/advanced/libraries/libraries_import "Including libraries; #import of functions")
* [Library file search order](/en/book/advanced/libraries/libraries_path_lookup "Library file search order")
* [DLL connection specifics](/en/book/advanced/libraries/libraries_dll "DLL connection specifics")
* [Classes and templates in MQL5 libraries](/en/book/advanced/libraries/libraries_class_template "Classes and templates in MQL5 libraries")
* [Importing functions from .NET libraries](/en/book/advanced/libraries/libraries_dotnet "Importing functions from .NET libraries")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Development and connection of binary format libraries

In addition to the specialized types of MQL programs − [Expert Advisors](/en/book/automation/experts), [indicators](/en/book/applications/indicators_make), [scripts, and services](/en/book/applications/script_service) — the MetaTrader 5 platform allows you to create and connect independent binary modules with arbitrary functionality, compiled as ex5 files or commonly used DLLs (Dynamic Link Library), standard for Windows. These can be analytical algorithms, graphical visualization, network interaction with web services, control of external programs, or the operating system itself. In any case, such libraries work in the terminal not as independent MQL programs but in conjunction with a program of any of the above 4 types.

The idea of integrating the library and the main (parent) program is that the library exports certain functions, i.e., declares them available for use from the outside, and the program imports their prototypes. It is the description of prototypes — sets of names, lists of parameters, and return values — that allows you to call these functions in the code without having their implementation.

Then, during the launch of the MQL program, the early dynamic linking is performed. This implies loading the library after the main program and establishing correspondence between the imported prototypes and the exported functions available in the library. Establishing one-to-one correspondences by names, parameter lists, and return types is a prerequisite for successful loading. If no corresponding exported implementation can be found for the import description of at least one function, the execution of the MQL program will be canceled (it will end with an error at the startup stage).

![Communication-component diagram of an MQL program with libraries](/en/book/img/lib.png "Communication-component diagram of an MQL program with libraries")

Communication-component diagram of an MQL program with libraries

You cannot select an included library when starting an MQL program. This linking is set by the developer when compiling the main program along with library imports. However, the user can manually replace one ex5/dll file with another between program starts (provided that the prototypes of the implemented exported functions match in the libraries). This can be used, for example, to switch the user interface language if the libraries contain labeled string resources. However, libraries are most often used as a commercial product with some know-how, which the author is not ready to distribute in the form of open header files.

For programmers who have come to MQL5 from other environments and are already familiar with the DLL technology, we would like to add a note about late dynamic linking, which is one of the advantages of DLLs. Full dynamic connection of one MQL program (or DLL module) to another MQL program during execution is impossible. The only similar action that MQL5 allows you to do "on the go" is linking an Expert Advisor and an indicator via [iCustom](/en/book/applications/indicators_use/indicators_icustom) or [IndicatorCreate](/en/book/applications/indicators_use/indicators_indicatorcreate), where the indicator acts as a dynamically linked library (however, programmatic interaction with has to be done through the indicators API, which means increased overhead for CopyBuffer, compared to direct function calls via export/#import).

Note that in normal cases, when an MQL program is compiled from sources without importing external functions, static linking is used, that is, the generated binary code directly refers to the called functions since they are known at the time of compilation.

Strictly speaking, a library can also rely on other libraries, i.e., it can import some of the functions. In theory, the chain of such dependencies can be even longer: for example, an MQL program includes library A, library A uses library B, and library B, in turn, uses library C. However, such chains are undesirable because they complicate the distribution and installation of the product, as well as make identifying the causes of potential startup problems more difficult. Therefore, libraries are usually connected directly to the parent MQL program.

In this chapter, we will describe the process of creating libraries in MQL5, exporting and importing functions (including restrictions on the data types used in them), as well as connecting external (ready-made) DLLs. DLL development is beyond the scope of this book.

[Example of searching for a trading strategy using SQLite](/en/book/advanced/sqlite/sqlite_example_ts "Example of searching for a trading strategy using SQLite")

[Creation of ex5 libraries; export of functions](/en/book/advanced/libraries/libraries_export "Creation of ex5 libraries; export of functions ")