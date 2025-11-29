---
title: "Structures and unions"
url: "https://www.mql5.com/en/book/oop/structs_and_unions"
hierarchy: []
scraped_at: "2025-11-28 09:47:55"
---

# Structures and unions

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")Structures and unions

* [Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")
* [Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")
* [Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")
* [Constructors and destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor "Constructors and destructors")
* [Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")
* [Structure layout and inheritance](/en/book/oop/structs_and_unions/structs_composition "Structure layout and inheritance")
* [Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")
* [Unions](/en/book/oop/structs_and_unions/unions "Unions")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Structures and unions

A structure is the object type that is easiest to understand, so we'll start our introduction to OOP with it. Structures have a lot in common with classes, which are the main building blocks in OOP, so knowledge of structures will help in the future when moving to classes. At the same time, structures have certain differences, some of which can be considered limitations, and some are considered advantages. In particular, structures cannot have [virtual functions](/en/book/oop/classes_and_interfaces/classes_virtual_override), but they can be used for integration with third-party DLLs.

The choice between structures and classes in the implementation of the algorithm is traditionally based on the requirements for access to the elements of the object and the presence of internal business logic. If a simple container with structured data is needed and its state does not need to be checked for correctness (in programming this is called an "invariant"), then a structure will do just fine. If you want to restrict access and support writing and reading according to some rules (which are formalized in the form of functions assigned to the object, which we will discuss later), then it is better to use classes.

MQL5 has built-in types of structures that describe entities that are in demand for trading, in particular, rates (MqlRates), ticks (MqlTick), date and time (MqlDateTime), trade requests (MqlTradeRequest), requests' results (MqlTradeResult) and many others. We will talk about them in Part 6 of this book.

[Object Oriented Programming](/en/book/oop "Object Oriented Programming")

[Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")