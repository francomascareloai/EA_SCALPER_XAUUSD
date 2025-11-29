---
title: "Unions"
url: "https://www.mql5.com/en/book/oop/structs_and_unions/unions"
hierarchy: []
scraped_at: "2025-11-28 09:49:23"
---

# Unions

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")Unions

* [Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")
* [Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")
* [Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")
* [Constructors and destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor "Constructors and destructors")
* [Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")
* [Structure layout and inheritance](/en/book/oop/structs_and_unions/structs_composition "Structure layout and inheritance")
* [Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")
* Unions

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Unions

A union is a user-defined type composed of fields located in the same memory area, due to which they overlap each other. This makes it possible to write a value of one type to a union, and then read its internal representation (at the bit level) in the interpretation for another type. Thus it is possible to provide non-standard conversion from one type to another.

Union fields can be of any built-in type, except for strings, dynamic arrays, and pointers. Also, in unions, you can use structures with the same simple field types and without constructors/destructors.

The compiler allocates for the union a memory cell with a size equal to the maximum size among the types of all elements. So, for the union with fields like long (8 bytes) and int (4 bytes), 8 bytes will be allocated.

All fields of the union are located at the same memory address, that is, they are aligned at the beginning of the union (they have an offset of 0, which can be checked using offsetof, see section [Packing Structures](/en/book/oop/structs_and_unions/structs_pack_dll)).

The syntax for describing a union is similar to the structure but uses the union keyword. It is followed by an identifier and then a block of code with a list of fields.

For example, an algorithm might use an array of type double to store various settings, simply because the type double is one of those with a maximum size in bytes equal to 8. Let's say among the settings there are numbers like ulong. Since the type double is not guaranteed to accurately reproduce large ulong values, you need to use a union to "pack" the ulong into a double and "unpack" it back.

| |
| --- |
| #define MAX\_LONG\_IN\_DOUBLE       9007199254740992 // FYI: ULONG\_MAX            18446744073709551615   union ulong2double {    ulong U;   // 8 bytes    double D;  // 8 bytes }; ulong2double converter;   void OnStart() {    Print(sizeof(ulong2double)); // 8        const ulong value = MAX\_LONG\_IN\_DOUBLE + 1;        double d = value; // possible loss of data due to type conversion    ulong result = d; // possible loss of data due to type conversion        Print(d, " / ", value, " -> ", result);    // 9007199254740992.0 / 9007199254740993 -> 9007199254740992        converter.U = value;    double r = converter.D;    Print(r);               // 4.450147717014403e-308    Print(offsetof(ulong2double, U), " ", offsetof(ulong2double, D)); // 0 0 } |

The size of the structure ulong2double is equal to 8 since both its fields have this size. Thus, the fields U and D overlap completely.

In the realm of integers, 9007199254740992 is the largest value that is guaranteed with robust storage in double. In this example, we are trying to store one more number in double.

The standard conversion from ulong to double results in loss of precision: after writing 9007199254740993 into a variable d of type double we read from its already "rounded" value 9007199254740992 (for additional information about the subtleties of storing numbers in the type double, see. section [Real numbers](/en/book/basis/builtin_types/float_numbers)).

When using the converter, the number 9007199254740993 is written to the union "as is", without conversions, since we are assigning it to a U field of type ulong. Its representation in terms of double is available, again without conversions, from field D. We can copy it to other variables and arrays like double without worrying.

Although the resulting value double looks strange, it exactly matches the original integer if it needs to be extracted by reverse conversion: write to a D field of type double, then read from a U field of type ulong.

A union can have constructors and destructors, as well as methods. By default, union members have public access rights, but this can be adjusted using access modifiers, as in the structure.

[Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")

[Classes and interfaces](/en/book/oop/classes_and_interfaces "Classes and interfaces")