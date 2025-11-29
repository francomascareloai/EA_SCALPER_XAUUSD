---
title: "Definition of structures"
url: "https://www.mql5.com/en/book/oop/structs_and_unions/structs_definition"
hierarchy: []
scraped_at: "2025-11-28 09:49:17"
---

# Definition of structures

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")Definition of structures

* Definition of structures
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

# Definition of structures

A structure consists of variables, which can be built-in or other user-defined types. The purpose of the structure is to combine logically related data in a single container. Suppose we have a function that performs a certain calculation and accepts a set of parameters: number of bars that show a history of quotes for analysis, date when the analysis started, price type, and number of signals allocated (for example, harmonics).

| |
| --- |
| double calculate(datetime start, int barNumber,                  ENUM\_APPLIED\_PRICE price, int components); |

In reality, there may be more parameters and it won't be easy to pass them to the function as a list. Moreover, based on the results of several calculations, it makes sense to save some of the best settings in some kind of array. Therefore, it is convenient to represent a set of parameters as a single object.

The description of the structure with the same variables looks as follows:

| |
| --- |
| struct Settings {    datetime start;    int barNumber;    ENUM\_APPLIED\_PRICE price;    int components; }; |

The description starts with the keyword struct followed by the identifier of our choice. This is followed by a block of code in curly brackets, and inside it are descriptions of variables included in the structure. Additionally, these are called fields or members of a structure. There is a semicolon after the curly brackets since the whole notation is a statement defining a new type, and ';' is required after statements.

Once the type is defined, we can apply it in the same way as built-in types. In particular, the new type allows you to describe variable Settings in the program in the usual way.

| |
| --- |
| Settings s; |

It is important to note that a single structure description allows you to create an arbitrary number of structure variables and even arrays of this type. Each structure instance will have its own set of elements, and they will contain independent values.

To access members of a structure, a special dereference operator is provided – the dot character '.'. To the left of it should be a variable of structure type, and to the right – an identifier of one of the fields available in it. Here's how you can assign a value to a structure element:

| |
| --- |
| void OnStart() {    Settings s;    s.start = D'2021.01.01';    s.barNumber = 1000;    s.price = PRICE\_CLOSE;    s.components = 8; } |

There is a more convenient way to fill in the structure which is the aggregate initialization. In this case, the sign '=' is written to the right of the structure variable, followed by a comma-separated list of initial values ​​of all fields in curly brackets.

| |
| --- |
| Settings s = {D'2021.01.01', 1000, PRICE\_CLOSE, 8}; |

The types of the value must match the corresponding element types. It is allowed to specify fewer values than the number of fields: then the remaining fields will receive zero values.

Note that this method only works when the variable is initialized, at the time of its definition. It is impossible to assign the contents of an already existing structure in this way, we will get a compilation error.

| |
| --- |
| Settings s;    // error: '{' - parameter conversion not allowed    s = {D'2021.01.01', 1000, PRICE\_CLOSE, 8}; |

Using the dereference operator, you can also read the value of a structure element. For example, we use the number of bars to calculate the number of components.

| |
| --- |
| s.components = (int)(MathSqrt(s.barNumber) + 1); |

Here MathSqrt is the built-in [square root](/en/book/common/maths/maths_pow_sqrt) function.

We have introduced a new type, Settings, to make it easier to pass a set of parameters to a function. Now it can be used as the only parameter of the updated function calculate:

| |
| --- |
| double calculate(Settings &settings); |

Notice the ampersand '&' in front of the parameter name, which means [passing by reference](/en/book/basis/functions/functions_ref_value). Structures can only be passed as parameters by reference.

Structures are also useful if you need to return a set of values from a function rather than a single value. Let's imagine that the calculate function should return not a value of the type double, but several coefficients and some trading recommendations (trade direction and probability of success). Then we can define the type of the structure Result and use it in the function prototype (Structs.mq5).

| |
| --- |
| struct Result {    double probability;    double coef[3];    int direction;    string status; };   Result calculate(Settings &settings) {    if(settings.barNumber > 1000) // edit fields    {       settings.components = (int)(MathSqrt(settings.barNumber) + 1);    }    // ...    // emulate getting the result    Result r = {};    r.direction = +1;    for(int i = 0; i < 3; i++) r.coef[i] = i + 1;    return r; } |

The empty curly brackets in the line Result r = {} represent the minimal aggregate initializer: it fills all fields of the structure with zeros.

The definition and declaration of the structure type can, if necessary, be done separately (as a rule, the declaration goes in the header mqh file, and the definition is in the mq5 file). This extended syntax will be covered in the [Chapter on Classes](/en/book/oop/classes_and_interfaces/classes_declaration_definition).

[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")

[Functions (methods) in structures](/en/book/oop/structs_and_unions/structs_methods "Functions (methods) in structures")