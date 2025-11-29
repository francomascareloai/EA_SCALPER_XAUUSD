---
title: "General template operation principles"
url: "https://www.mql5.com/en/book/oop/templates/templates_principles"
hierarchy: []
scraped_at: "2025-11-28 09:49:45"
---

# General template operation principles

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Templates](/en/book/oop/templates "Templates")General template operation principles

* [Template header](/en/book/oop/templates/templates_header "Template header")
* General template operation principles
* [Templates vs preprocessor macros](/en/book/oop/templates/templates_vs_macro "Templates vs preprocessor macros")
* [Features of built-in and object types in templates](/en/book/oop/templates/templates_for_standard_and_object_types "Features of built-in and object types in templates")
* [Function templates](/en/book/oop/templates/templates_functions "Function templates")
* [Object type templates](/en/book/oop/templates/templates_objects "Object type templates")
* [Method templates](/en/book/oop/templates/templates_methods "Method templates")
* [Nested templates](/en/book/oop/templates/templates_nested "Nested templates")
* [Absent template specialization](/en/book/oop/templates/templates_specialization "Absent template specialization")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# General template operation principles

Let's recall [functions overload](/en/book/basis/functions/functions_overloading). It consists in defining several versions of a function with different parameters, including situations when the number of parameters is the same, but their types are different. Often an algorithm of such functions is the same for parameters of different types. For example, MQL5 has a built-in function [MathMax](/en/book/common/maths/maths_max_min) that returns the largest of the two values passed to it:

| |
| --- |
| double MathMax(double value1, double value2); |

Although a prototype is only provided for the type double, the function is actually capable of working with argument pairs of other numeric types, such as int or datetime. In other words, the function is an overloaded kernel for built-in numerical types. If we wanted to achieve the same effect in our source code, we would have to overload the function by duplicating it with different parameters, like so:

| |
| --- |
| double Max(double value1, double value2) {    return value1 > value2 ? value1 : value2; }   int Max(int value1, int value2) {    return value1 > value2 ? value1 : value2; }   datetime Max(datetime value1, datetime value2) {    return value1 > value2 ? value1 : value2; } |

All implementations (function bodies) are the same. Only the parameter types change.

This is when templates are useful. By using them, we can describe one sample of the algorithm with the required implementation, and the compiler itself will generate several instances of it for the specific types involved in the program. Generation occurs on the fly during compilation and is imperceptible to the programmer (unless there is an error in the template) The source code obtained automatically is not inserted into the program text, but is directly converted into binary code (ex5 file).

In the template, one or more parameters are formal designations of types, for which, at the compilation stage, according to special type inference rules, real types will be selected from among built-in or user-defined ones. For example, the Max function can be described using the following template with the T type parameter:

| |
| --- |
| template<typename T> T Max(T value1, T value2) {    return value1 > value2 ? value1 : value2; } |

And then - apply it for variables of various types (see TemplatesMax.mq5):

| |
| --- |
| void OnStart() {    double d1 = 0, d2 = 1;    datetime t1 = D'2020.01.01', t2 = D'2021.10.10';    Print(Max(d1, d2));    Print(Max(t1, t2));    ... } |

In this case, the compiler will automatically generate variants of the function Max for the types double and datetime.

The template itself does not generate source code. To do this, you need to create an instance of the template in one way or another: call a template function or mention the name of a template class with specific types to create an object or a derived class.

Until this is done, the entire pattern is ignored by the compiler. For example, we can write the following supposedly template function, which actually contains syntactically incorrect code. However, the compilation of a module with this function will succeed as long as it is not called anywhere.

| |
| --- |
| template<typename T> void function() {   it's not a comment, but it's not source code either    !%^&\* } |

For each use of the template, the compiler determines the real types that match the formal parameters of the template. Based on this information, template source code is automatically generated for each unique combination of parameters. This is the instance.

So, in the given example of the Max function, we called the template function twice: for the pair of variables of type double, and for the pair of variables of type datetime. This resulted in two instances of the Max function with source code for the matches T=double and T=datetime. Of course, if the same template is called in other parts of the code for the same types, no new instances will be generated. A new instance of the template is required only if the template is applied to another type (or set of types, if there is more than 1 parameter).

Please note that the template Max has one parameter, and it sets the type for two input parameters of the function and its return value at once. In other words, the template declaration is capable of imposing certain restrictions on the types of valid arguments.

If we were to call Max on variables of different types, the compiler would not be able to determine the type to instantiate the template and would throw the error "ambiguous template parameters, must be 'double' or 'datetime'":

| |
| --- |
| Print(Max(d1, t1)); // template parameter ambiguous,                     // could be 'double' or 'datetime' |

This process of discovering the actual types for template parameters based on the context in which the template is used is called type deduction. In MQL5, type inference is available only for function and method templates.

For classes, structures, and unions, a different way of binding types to template parameters is used: the required types are explicitly specified in angle brackets when creating a template instance (if there are several parameters, then the corresponding number of types is indicated as a comma-separated list). For more on this, see the section [Object type templates](/en/book/oop/templates/templates_objects).

The same explicit method can be applied to functions as an alternative to automatic type inference.

For example, we can generate and call an instance of Max for type ulong:

| |
| --- |
| Print(Max<ulong>(1000, 10000000)); |

In this case, if not for the explicit indication, the template function would be associated with the type int (based on the values ​​of integer constants).

[Template header](/en/book/oop/templates/templates_header "Template header")

[Templates vs preprocessor macros](/en/book/oop/templates/templates_vs_macro "Templates vs preprocessor macros")