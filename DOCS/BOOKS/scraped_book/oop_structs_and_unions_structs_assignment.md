---
title: "Copying structures"
url: "https://www.mql5.com/en/book/oop/structs_and_unions/structs_assignment"
hierarchy: []
scraped_at: "2025-11-28 09:49:15"
---

# Copying structures

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")Copying structures

* [Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")
* [Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")
* Copying structures
* [Constructors and destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor "Constructors and destructors")
* [Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")
* [Structure layout and inheritance](/en/book/oop/structs_and_unions/structs_composition "Structure layout and inheritance")
* [Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")
* [Unions](/en/book/oop/structs_and_unions/unions "Unions")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Copying structures

Structures of the same type can be copied entirely into each other using the '=' assignment operator. Let's demonstrate this rule using an example of the structure Result. We get the first instance of r from the calculate function.

| |
| --- |
| void OnStart() {    ...    Result r = calculate(s);    r.print();    // will output to the log:    // 0.5 1 ok    // 1.00000 2.00000 3.00000    ...    Result r2;    r2 = r;    r2.print();    // will output to the log the same values:    // 0.5 1 ok    // 1.00000 2.00000 3.00000 } |

Then, the variable Result r2 was additionally created, and the contents of the r variable, all fields concurrently, were duplicated into it. The accuracy of the operation can be verified by outputting to the log using the method print (the lines are given in the comments).

It should be noted that defining two types of structures with the same set of fields does not make the two types the same. It is not possible to assign a structure to another one completely, only memberwise assignment is permitted in such cases.

A little later, we'll talk about structure inheritance, which will give you more options for copying. The fact is that copying works not only between structures of the same type but also between related types. However, there are important nuances, which we will cover in the [Layout and inheritance of structures](/en/book/oop/structs_and_unions/structs_composition) section.

[Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")

[Constructors and destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor "Constructors and destructors")