---
title: "Nested templates"
url: "https://www.mql5.com/en/book/oop/templates/templates_nested"
hierarchy: []
scraped_at: "2025-11-28 09:49:50"
---

# Nested templates

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Templates](/en/book/oop/templates "Templates")Nested templates

* [Template header](/en/book/oop/templates/templates_header "Template header")
* [General template operation principles](/en/book/oop/templates/templates_principles "General template operation principles")
* [Templates vs preprocessor macros](/en/book/oop/templates/templates_vs_macro "Templates vs preprocessor macros")
* [Features of built-in and object types in templates](/en/book/oop/templates/templates_for_standard_and_object_types "Features of built-in and object types in templates")
* [Function templates](/en/book/oop/templates/templates_functions "Function templates")
* [Object type templates](/en/book/oop/templates/templates_objects "Object type templates")
* [Method templates](/en/book/oop/templates/templates_methods "Method templates")
* Nested templates
* [Absent template specialization](/en/book/oop/templates/templates_specialization "Absent template specialization")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Nested templates

Templates can be nested within classes/structures or within other class/structure templates. The same is true for unions.

In the section [Unions](/en/book/oop/structs_and_unions/unions), we saw the ability to "convert" long values ​​to double and back again without loss of precision.

Now we can use templates to write a universal "converter"(TemplatesConverter.mq5). The template class Converter has two parameters T1 and T2, indicating the types between which the conversion will be performed. To write a value according to the rules of one type and read according to the rules of another, we again need a union. We will also make it a template (DataOverlay) with parameters U1 and U2, and define it inside the class.

The class provides a convenient transformation by overloading the operators [], in the implementation of which the union fields are written and read.

| |
| --- |
| template<typename T1,typename T2> class Converter { private:    template<typename U1,typename U2>    union DataOverlay    {       U1 L;       U2 D;    };        DataOverlay<T1,T2> data;     public:    T2 operator[](const T1 L)    {       data.L = L;       return data.D;    }        T1 operator[](const T2 D)    {       data.D = D;       return data.L;    } }; |

The union is used to describe the field DataOverlay<T1,T2>data within the class. We could use T1 and T2 directly in DataOverlay and not make this union a template. But to demonstrate the technique itself, the parameters of the outer template are passed to the inner template when the data field is generated. Inside the DataOverlay, the same pair of types will be known as U1 and U2 (in addition to T1 and T2).

Let's see the template in action.

| |
| --- |
| #define MAX\_LONG\_IN\_DOUBLE       9007199254740992     void OnStart() {    Converter<double,ulong> c;        const ulong value = MAX\_LONG\_IN\_DOUBLE + 1;        double d = value; // possible loss of data due to type conversion    ulong result = d; // possible loss of data due to type conversion        Print(value == result); // false        double z = c[value];    ulong restored = c[z];        Print(value == restored); // true } |

[Method templates](/en/book/oop/templates/templates_methods "Method templates")

[Absent template specialization](/en/book/oop/templates/templates_specialization "Absent template specialization")