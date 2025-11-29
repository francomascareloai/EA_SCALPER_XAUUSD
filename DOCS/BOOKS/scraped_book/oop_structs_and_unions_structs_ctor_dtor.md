---
title: "Constructors and destructors"
url: "https://www.mql5.com/en/book/oop/structs_and_unions/structs_ctor_dtor"
hierarchy: []
scraped_at: "2025-11-28 09:49:19"
---

# Constructors and destructors

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")Constructors and destructors

* [Definition of structures](/en/book/oop/structs_and_unions/structs_definition "Definition of structures")
* [Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")
* [Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")
* Constructors and destructors
* [Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")
* [Structure layout and inheritance](/en/book/oop/structs_and_unions/structs_composition "Structure layout and inheritance")
* [Access rights](/en/book/oop/structs_and_unions/structs_access "Access rights")
* [Unions](/en/book/oop/structs_and_unions/unions "Unions")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Constructors and destructors

Among the methods that can be defined for a structure, there are special ones: constructors and destructors.

A constructor has the same name as the structure name and does not return a value (type void). The constructor, if defined, will be called at the time of initialization for each new instance of the structure. Due to this, in the constructor, the initial state of the structure can be calculated in a special way.

A structure can have multiple constructors with different sets of parameters, and the compiler will choose the appropriate one based on the number and type of arguments when defining the variable.

For example, we can describe a pair of constructors in the structure Result: one without parameters, and the second one with one string type parameter to set the status.

| |
| --- |
| struct Result {    ...    void Result()    {       status = "ok";    }    void Result(string s)    {       status = s;    } }; |

By the way, a constructor without parameters is called a default constructor. If there are no explicit constructors, the compiler implicitly creates a default constructor for any structure that contains strings and dynamic arrays to pad these fields with zeros.

It is important that fields of other types (for example, all numeric) are not reset to zero, regardless of whether the structure has a default constructor, and therefore the initial values of the elements after memory allocation will be random. You should either create constructors or make sure that the correct values are assigned in your code immediately after the object is created.

The presence of explicit constructors makes it impossible to use the aggregate initialization syntax. Because of it, the line Result r = {}; in the calculate method will not be compiled. Now we have the right to use only one of the constructors that we provided ourselves. For example, the following statements call the parameterless constructor:

| |
| --- |
| Result r1;    Result r2(); |

And creating a structure with a filled status can be done like this:

| |
| --- |
| Result r3("success"); |

The default constructor (explicit or implicit) is also called when an array of structures is created. For example, the following statement allocates memory for 10 structures with results and initializes them with a default constructor:

| |
| --- |
| Result array[10]; |

A destructor is a function that will be called when the structure object is being destroyed. The destructor has the same name as the structure name, but is prefixed with a tilde character '~'. The destructor, like the constructor, does not return a value, but it does not take parameters either.

There can only be one destructor.

You cannot explicitly call the destructor. The program itself does this when exiting a block of code where a local structure variable was defined, or when freeing an array of structures.

The purpose of the destructor is to release any dynamic resources if the structure allocated them in the constructor. For example, a structure can have the persistence property, that is, save its state to a file when it is unloaded from memory and restore it when the program creates it again. In this case, a descriptor that needs to be opened and closed is used in the built-in [file functions](/en/book/common/files).

Let's define a destructor in the Result structure and add constructors along the way so that all these methods keep track of the number of object instances (as they are created and destroyed).

| |
| --- |
| struct Result {    ...    void Result()    {       static int count = 0;       Print(\_\_FUNCSIG\_\_, " ", ++count);       status = "ok";    }      void Result(string s)    {       static int count = 0;       Print(\_\_FUNCSIG\_\_, " ", ++count);       status = s;    }      void ~Result()    {       static int count = 0;       Print(\_\_FUNCSIG\_\_, " ", ++count);    } }; |

Three static variables named count exist independently of each other: each of them counts in the context of its own function.

As a result of running the script, we will receive the following log:

| |
| --- |
| Result::Result() 1 Result::Result() 2 Result::Result() 3 Result::~Result() 1 Result::~Result() 2 0.5 1 ok 1.00000 2.00000 3.00000 Result::Result(string) 1 0.5 1 ok 1.00000 2.00000 3.00000 Result::~Result() 3 Result::~Result() 4 |

Let's figure out, what it means.

The first instance of the structure is created in the function OnStart, in the same line where calculate is called. When entering the constructor, the counter value count is initialized once with zero and then incremented each time the constructor is executed, so for the first time, the value 1 is output.

Inside the calculate function, a local variable of type Result is defined; it is registered under number 2.

The third structure instance is not so obvious. The point is that to pass the result from the function, the compiler implicitly creates a temporary variable, where it copies the data of the local variable. It is likely that this behavior will change in the future, and then the local instance will "move" out of the function without duplication.

The last constructor call is in a method with a string parameter, so the call count is 1.

It is important that the total number of calls to both constructors is the same as the number of calls to the destructor: 4.

We'll talk more about [constructors](/en/book/oop/classes_and_interfaces/classes_ctors) and [destructors](/en/book/oop/classes_and_interfaces/classes_dtors) in the Chapter on Classes.

[Copying structures](/en/book/oop/structs_and_unions/structs_assignment "Copying structures")

[Packing structures in memory and interacting with DLLs](/en/book/oop/structs_and_unions/structs_pack_dll "Packing structures in memory and interacting with DLLs")