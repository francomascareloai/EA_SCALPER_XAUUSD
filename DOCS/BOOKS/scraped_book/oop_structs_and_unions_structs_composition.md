---
title: "Structure layout and inheritance"
url: "https://www.mql5.com/en/book/oop/structs_and_unions/structs_composition"
hierarchy: []
scraped_at: "2025-11-28 09:49:24"
---

# Structure layout and inheritance

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")Structure layout and inheritance

* [Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")
* [Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")
* [Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")
* [Constructors and destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor "Constructors and destructors")
* [Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")
* Structure layout and inheritance
* [Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")
* [Unions](/en/book/oop/structs_and_unions/unions "Unions")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Structure layout and inheritance

Structures can have other structures as their fields. For example, let's define the Inclosure structure and use this type for the field data in the Main structure (StructsComposition.mq5):

| |
| --- |
| struct Inclosure {    double X, Y; };   struct Main {    Inclosure data;    int code; };   void OnStart() {    Main m = {{0.1, 0.2}, -1}; // aggregate initialization    m.data.X = 1.0;            // assignment element by element    m.data.Y = -1.0; } |

In the initialization list, the field data is represented by an additional level of curly brackets with field values ​​Inclosure. To access fields of such a structure, you need to use two dereference operations.

If the nested structure is not used anywhere else, it can be declared directly inside the outer one.

| |
| --- |
| struct Main2 {    struct Inclosure2    {      double X, Y;    }    data;    int code; }; |

Another way of laying out structures is inheritance. This mechanism is typically used for building class hierarchies (and will be discussed in detail in the corresponding [section](/en/book/oop/classes_and_interfaces/classes_inheritance)), but it is also available for structs.

When defining a new type of structure, the programmer can indicate the type of the parent structure in its header, after the colon sign (it must be defined earlier in the source code). As a result, all fields of the parent structure will be added to the daughter structure (at its beginning), and the own fields of the new structure will be located in memory behind the parent ones.

| |
| --- |
| struct Main3 : Inclosure {    int code; }; |

The parent structure here is not nested, but an integral part of the daughter structure. Because of it, filling fields does not require additional curly brackets when initializing, or a chain of multiple dereference operators.

| |
| --- |
| Main3 m3 = {0.1, 0.2, -1};    m3.X = 1.0;    m3.Y = -1.0; |

All three considered structures Main, Main2, and Main3 have the same memory representation and size of 20 bytes. But they are different types.

| |
| --- |
| Print(sizeof(Main));   // 20    Print(sizeof(Main2));  // 20    Print(sizeof(Main3));  // 20 |

As we said before (see [Copying Structures](/en/book/oop/structs_and_unions/structs_assignment)), the assignment operator '=', can be used to copy related types of structures, more specifically those that are linked by an inheritance chain. In other words, a structure of a parent type can be written into a structure of a daughter type (in this case, the fields added in the derived structure will remain untouched), or vice versa, a daughter type structure can be written into a parent type structure (in this case, "extra" fields will be cut off).

For example:

| |
| --- |
| Inclosure in = {10, 100};    m3 = in; |

Here, variable m3 has a type Main3 inherited from Inclosure. As a result of the assignment m3 = in, the fields X and Y (the common part for both types) will be copied from the variable in of the base type into the fields X and Y in the variable m3 of the derived type. The field code of the variable m3 will remain unchanged.

It does not matter whether the child structure is a direct descendant of the ancestor or a distant one, i.e. the chain of inheritance can be long. Such copying of common fields works between "children", "grandchildren" and other combinations of types from different branches of the "family tree".

If the parent structure only has constructors with parameters, it must be called from the initialization list when the derived structure constructor is inherited. For example,

| |
| --- |
| struct Base {    const int mode;    string s;    Base(const int m) : mode(m) { } };   struct Derived : Base {    double data[10];    // if we remove the constructor, we get an error:    Derived() : Base(1) { } // 'Base' - wrong parameters count }; |

In the Base constructor, we fill in the field mode. Since it has the modifier const, the constructor is the only way to set a value for it, and this must be done in the form of a special initialization syntax after the colon (you can no longer assign a constant in the body of the constructor). Having an explicit constructor causes the compiler to not generate an implicit (parameterless) constructor. However, we do not have an explicit parameterless constructor in the structure Base, and in its absence, any derived class does not know how to correctly call the Base constructor with a parameter. Therefore, in the structure Derived, it is required to explicitly initialize the base constructor: this is also done using the initialization syntax in the constructor header, after the sign ':' - in this case, we call Base(1).

If we remove the constructor Derived, we get an "invalid number of parameters" error in the base constructor, because the compiler tries to call the constructor for Base by default (which should have 0 parameters).

We'll cover the syntax and inheritance mechanism in more detail in the [Class Chapter](/en/book/oop/classes_and_interfaces).

[Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")

[Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")