---
title: "Common MQL5 APIs"
url: "https://www.mql5.com/en/book/common"
hierarchy: []
scraped_at: "2025-11-28 10:15:05"
---

# Common MQL5 APIs

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Common APIs

* [Built-in type conversions](/en/book/common/conversions "Built-in type conversions")
* [Working with strings and symbols](/en/book/common/strings "Working with strings and symbols")
* [Working with arrays](/en/book/common/arrays "Working with arrays")
* [Mathematical functions](/en/book/common/maths "Mathematical functions")
* [Working with files](/en/book/common/files "Working with files")
* [Client terminal global variables](/en/book/common/globals "Client terminal global variables")
* [Functions for working with time](/en/book/common/timing "Functions for working with time")
* [User interaction](/en/book/common/output "User interaction")
* [MQL program execution environment](/en/book/common/environment "MQL program execution environment")
* [Matrices and vectors](/en/book/common/matrices "Matrices and vectors")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Common MQL5 APIs

In the previous parts of the book, we studied the basic concepts, syntax, and rules for using MQL5 language constructs. However, this is only a foundation for writing real programs that meet trader requirements, such as analytical data processing and automatic trading. Solving such tasks would not be possible without a wide range of built-in functions and means of interaction with the MetaTrader 5 terminal, which make up the MQL5 API.

In this chapter, we will start mastering the MQL5 API and will continue to do so until the end of the book, gradually getting familiar with all the specialized subsystems.

The list of technologies and capabilities provided to any MQL program by the kernel (the runtime environment of MQL programs inside the terminal) is very large. This is why it makes sense to start with the simplest things that can be useful in most programs. In particular, here we will look at functions specialized for work with arrays, strings, files, data transformation, user interaction, mathematical functions, and environmental control.

Previously, we learned to describe our own [functions](/en/book/basis/functions) in MQL5 and call them. The built-in functions of the MQL5 API are available from the source code, as they say, "out of the box", i.e. without any preliminary description.

It is important to note that, unlike in C++, no additional preprocessor directives are required to include a specific set of built-in functions in a program. The names of all MQL5 API functions are present in the global context (namespace), always and unconditionally.

On the one hand, this is convenient, but on the other hand, it requires you to be aware of a possible name conflict. If you accidentally try to use one of the names of the built-in functions, it will override the standard implementation, which can lead to unexpected consequences: at best, you get a compiler error about ambiguous overload, and at worst, all the usual calls will be redirected to the new implementation, without any warnings.

In theory, similar names can be used in other contexts, for example, as a class method name or in a dedicated (user) namespace. In such cases, calling a global function can be done using the context resolution operator: we discussed this situation in the section [Nested types, namespaces, and the '::' context operator](/en/book/oop/classes_and_interfaces/classes_namespace_context).

 

| | |
| --- | --- |
| MQL5 Programming for Traders — Source Codes from the Book. Part 4 | [MQL5 Programming for Traders — Source Codes from the Book. Part 4](https://www.mql5.com/en/code/45593) |
| Примеры из книги также доступны в публичном проекте \MQL5\Shared Projects\MQL5Book. | Examples from the book are also available in the [public project](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#public) \MQL5\Shared Projects\MQL5Book |

[Absent template specialization](/en/book/oop/templates/templates_specialization "Absent template specialization")

[Built-in type conversions](/en/book/common/conversions "Built-in type conversions")