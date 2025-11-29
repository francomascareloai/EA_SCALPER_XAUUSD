---
title: "Features of built-in and object types in templates"
url: "https://www.mql5.com/en/book/oop/templates/templates_for_standard_and_object_types"
hierarchy: []
scraped_at: "2025-11-28 09:49:47"
---

# Features of built-in and object types in templates

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Templates](/en/book/oop/templates "Templates")Features of built-in and object types in templates

* [Template header](/en/book/oop/templates/templates_header "Template header")
* [General template operation principles](/en/book/oop/templates/templates_principles "General template operation principles")
* [Templates vs preprocessor macros](/en/book/oop/templates/templates_vs_macro "Templates vs preprocessor macros")
* Features of built-in and object types in templates
* [Function templates](/en/book/oop/templates/templates_functions "Function templates")
* [Object type templates](/en/book/oop/templates/templates_objects "Object type templates")
* [Method templates](/en/book/oop/templates/templates_methods "Method templates")
* [Nested templates](/en/book/oop/templates/templates_nested "Nested templates")
* [Absent template specialization](/en/book/oop/templates/templates_specialization "Absent template specialization")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Features of built-in and object types in templates

It should be kept in mind that 3 important aspects impose restrictions on the applicability of types in a template:

* Whether the type is built-in or user-defined (user-defined types require parameters to be passed by reference, and built-in ones will not allow a literal to be passed by reference);
* Whether the object type is a class (only classes support pointers);
* A set of operations performed on data of the appropriate types in the template algorithm.

Let's say we have a Dummy structure (see script TemplatesMax.mq5):

| |
| --- |
| struct Dummy {    int x; }; |

If we try to call the Max function for two instances of the structure, we will get a bunch of error messages, with mains as the following: "objects can only be passed by reference" and "you cannot apply a template."

| |
| --- |
| // ERRORS:    // 'object1' - objects are passed by reference only    // 'Max' - cannot apply template    Dummy object1, object2;    Max(object1, object2); |

The pinnacle of the problem is passing template function parameters by value, and this method is incompatible with any object type. To solve it, you can change the type of parameters to links:

| |
| --- |
| template<typename T> T Max(T &value1, T &value2) {    return value1 > value2 ? value1 : value2; } |

The old error will go away, but then we will get a new error: "'>' - illegal operation use" ("'>' - illegal operation use"). The point is that the Max template has an expression with the '>' comparison operator. Therefore, if a custom type is substituted into the template, the '>' operator must be overloaded in the template (and the structure Dummy does not have it: we'll get to that shortly). For more complex functions, you will likely need to overload a much larger number of operators. Fortunately, the compiler tells you exactly what is missing.

However, changing the method of passing function parameters by reference additionally led to the previous call not working as such:

| |
| --- |
| Print(Max<ulong>(1000, 10000000)); |

Now it generates errors: "parameter passed as reference, variable expected". Thus, our function template stopped working with literals and other temporary values ​​(in particular, it is impossible to directly pass an expression or the result of calling another function into it).

One might think that the universal way out of the situation would be template function overloading, i.e., the definition of both options, that differs only in the ampersand in the parameters:

| |
| --- |
| template<typename T> T Max(T &value1, T &value2) {    return value1 > value2 ? value1 : value2; }    template<typename T> T Max(T value1, T value2) {    return value1 > value2 ? value1 : value2; } |

But it won't work. Now the compiler throws the error "ambiguous function overload with the same parameters":

| |
| --- |
| 'Max' - ambiguous call to overloaded function with the same parameters could be one of 2 function(s)    T Max(T&,T&)    T Max(T,T) |

The final, working overload would require the modifier const to be added to the links. Along the way, we added the operator Print to the template Max so that we can see in the log which overload is being called and which parameter type T corresponds to.

| |
| --- |
| template<typename T> T Max(const T &value1, const T &value2) {    Print(\_\_FUNCSIG\_\_, " T=", typename(T));    return value1 > value2 ? value1 : value2; }     template<typename T> T Max(T value1, T value2) {    Print(\_\_FUNCSIG\_\_, " T=", typename(T));    return value1 > value2 ? value1 : value2; }     struct Dummy {    int x;    bool operator>(const Dummy &other) const    {       return x > other.x;    } }; |

We have also implemented an overload of the operator '>' in the Dummy structure. Therefore, all Max function calls in the test script are completed successfully: both for built-in and user-defined types, as well as for literals and variables. The outputs that go into the log:

| |
| --- |
| double Max<double>(double,double) T=double 1.0 datetime Max<datetime>(datetime,datetime) T=datetime 2021.10.10 00:00:00 ulong OnStart::Max<ulong>(ulong,ulong) T=ulong 10000000 Dummy Max<Dummy>(const Dummy&,const Dummy&) T=Dummy |

An attentive reader will notice that we now have two identical functions that differ only in the way parameters are passed (by value and by reference), and this is exactly the situation against which the use of templates is directed. Such duplication can be costly if the function body is not as simple as ours. This can be solved by the usual methods: separate the implementation into a separate function and call it from both "overloads", or call one "overload" from the other (an optional parameter was required to avoid the first version of Max calling itself and, resulting in stack overflows):

| |
| --- |
| template<typename T> T Max(T value1, T value2) {    // calling a function with parameters by reference    return Max(value1, value2, true); }     template<typename T> T Max(const T &value1, const T &value2, const bool ref = false) {    return (T)(value1 > value2 ? value1 : value2); } |

We still have to consider one more point associated with user-defined types, namely the use of pointers in templates (recall, that they apply only to class objects). Let's create a simple class Data and try to call the template function Max for pointers to its objects.

| |
| --- |
| class Data { public:    int x;    bool operator>(const Data &other) const    {       return x > other.x;    } };     void OnStart() {    ...     Data \*pointer1 = new Data();    Data \*pointer2 = new Data();    Max(pointer1, pointer2);    delete pointer1;    delete pointer2; } |

We will see in the log that 'T=Data\*', i.e. the pointer attribute, hits the inline type. This suggests that, if necessary, you can write another overload of the template function, which will be responsible only for pointers.

| |
| --- |
| template<typename T> T \*Max(T \*value1, T \*value2) {    Print(\_\_FUNCSIG\_\_, " T=", typename(T));    return value1 > value2 ? value1 : value2; } |

In this case, the attribute of the pointer '\*' is already present in the template parameters, and so type inference results in 'T=Data'. This approach allows you to provide a separate template implementation for pointers.

If there are multiple templates that are suitable for generating an instance with specific types, the most specialized version of the template is chosen. In particular, when calling the function Max with pointer arguments, two templates with parameters T (T=Data\*) and T\* (T=Data), but since the former can take both values ​​and pointers, it is more general than the latter, which only works with pointers. Therefore, the second one will be chosen for pointers. In other words, the fewer modifiers in the actual type that is substituted for T, the more preferable the template variant. In addition to the attribute of the pointer '\*', this also includes the modifier const. The parameters const T\* or const T are more specialized than just T\* or T, respectively.

[Templates vs preprocessor macros](/en/book/oop/templates/templates_vs_macro "Templates vs preprocessor macros")

[Function templates](/en/book/oop/templates/templates_functions "Function templates")