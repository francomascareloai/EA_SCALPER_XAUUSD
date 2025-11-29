---
title: "Access rights"
url: "https://www.mql5.com/en/book/oop/structs_and_unions/structs_access"
hierarchy: []
scraped_at: "2025-11-28 09:49:26"
---

# Access rights

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")Access rights

* [Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")
* [Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")
* [Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")
* [Constructors and destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor "Constructors and destructors")
* [Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")
* [Structure layout and inheritance](/en/book/oop/structs_and_unions/structs_composition "Structure layout and inheritance")
* Access rights
* [Unions](/en/book/oop/structs_and_unions/unions "Unions")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Access rights

If necessary, in the description of the structure, you can use special keywords, which represent access modifiers that limit the visibility of fields from outside the structure. There are three modifiers: public, protected, and private. By default, all structure members are public, which is equivalent to the following entry (using the Result structure as an example):

| |
| --- |
| struct Result { public:    double probability;    double coef[3];    int direction;    string status;    ... }; |

All members below the modifier receive the appropriate access rights until another modifier is encountered or the structure block ends. There can be many sections with different access rights, however, they can be modified arbitrarily.

Members marked as protected are available only from the code of this structure and descendant structures, i.e., it is assumed that they must have public methods, otherwise, no one will be able to access such fields.

Members marked as private are accessible only from within the structure's code. For example, if you add private before the status field, you will most likely need a method to read the status by external code (getStatus).

| |
| --- |
| struct Result { public:    double probability;    double coef[3];    int direction;     private:    string status;     public:    string getStatus()    {       return status;    }    ... }; |

It will be possible to set the status only through the parameter of the second constructor. Accessing the field directly will result in the error "no access to private member 'status' of structure 'Result'":

| |
| --- |
| // error: // cannot access to private member 'status' declared in structure 'Result' r.status = "message"; |

In classes, the default access is private. This follows the principle of encapsulation, which we will cover in the [Chapter on Classes](/en/book/oop/classes_and_interfaces).

[Structure layout and inheritance](/en/book/oop/structs_and_unions/structs_composition "Structure layout and inheritance")

[Unions](/en/book/oop/structs_and_unions/unions "Unions")