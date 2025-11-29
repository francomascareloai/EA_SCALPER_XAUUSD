---
title: "Functions (methods) in structures"
url: "https://www.mql5.com/en/book/oop/structs_and_unions/structs_methods"
hierarchy: []
scraped_at: "2025-11-28 09:49:27"
---

# Functions (methods) in structures

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")Functions (methods) in structures

* [Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")
* Functions (methods) in structures
* [Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")
* [Constructors and destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor "Constructors and destructors")
* [Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")
* [Structure layout and inheritance](/en/book/oop/structs_and_unions/structs_composition "Structure layout and inheritance")
* [Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")
* [Unions](/en/book/oop/structs_and_unions/unions "Unions")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Functions (methods) in structures

After receiving a result from the calculate function, it would be desirable to print it to the log, but the Print function does not work with user-defined types: they themselves must provide a way to output information.

| |
| --- |
| void OnStart() {    Settings s = {D'2021.01.01', 1000, PRICE\_CLOSE, 8};    Result r = calculate(s);    // Print(r);  // error: 'r' - objects are passed by reference only    // Print(&r); // error: 'r' - class type expected } |

The comments show the attempts to call the Print function for the structure, and what follows thereafter. The first error is caused by the fact that structure instances are objects, and objects must be passed to functions by reference. At the same time, Print is expecting a value (one or several). The use of an ampersand before the variable name in the second Print call means in MQL5 that the pointer is received, and it is not a reference as one might think. Pointers in MQL5 are only supported for class objects (not structures), hence the second "class type expected" error. We will learn more about pointers in the next chapter (see [Classes and interfaces](/en/book/oop/classes_and_interfaces)).

We could specify in the Print call all the members of the structure separately (using dereference), but this is rather troublesome.

For those cases when it is necessary to process the contents of the structure in a special way, it is possible to define functions inside the structure. The syntax of the definition is no different from the familiar global context functions, but the definition itself is located inside the structure block.

Such functions are called methods. Since they are located in the context of the corresponding block, the fields of the structure can be accessed from them without the dereference operator. As an example, let's write the implementation of the function print in the Resultstructure.

| |
| --- |
| struct Result {    ...    void print()    {       Print(probability, " ", direction, " ", status);       ArrayPrint(coef);    } }; |

Calling a method of the structure instance is as simple as reading its field: the same '.' operator is used.

| |
| --- |
| void OnStart() {    Settings s = {D'2021.01.01', 1000, PRICE\_CLOSE, 8};    Result r = calculate(s);    r.print(); } |

[Chapter on Classes](/en/book/oop/classes_and_interfaces) will cover methods in more detail.

[Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")

[Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")