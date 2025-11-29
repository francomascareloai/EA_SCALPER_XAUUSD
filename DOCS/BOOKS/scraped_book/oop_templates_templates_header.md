---
title: "Template header"
url: "https://www.mql5.com/en/book/oop/templates/templates_header"
hierarchy: []
scraped_at: "2025-11-28 09:49:48"
---

# Template header

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Templates](/en/book/oop/templates "Templates")Template header

* Template header
* [General template operation principles](/en/book/oop/templates/templates_principles "General template operation principles")
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

# Template header

In MQL5, you can make functions, object types (classes, structures, unions) or separate methods within them templated. In any case, the template description has a title:

| |
| --- |
| template <typename T [, typename Ti ... ]> |

The header starts with the template keyword, followed by a comma-separated list of formal parameters in angle brackets: each parameter is denoted by the typename keyword and an identifier. Identifiers must be unique within a particular definition.

The keyword typename in the template header tells the compiler that the following identifier should be treated as a type. In the future, the MQL5 compiler is likely to support other kinds of non-type parameters, as the C++ compiler does.

This use of typename should not be confused with the built-in [operator](/en/book/basis/expressions/operators_sizeof_typename) [typename](/en/book/basis/expressions/operators_sizeof_typename), which returns a string with the type name of the passed argument.

A template header is followed by a usual definition of a function (method) or class (structure, union), in which the formal parameters of the template (identifiers T, Ti) are used in instructions and expressions in those places where the syntax requires a type name. For example, for template functions, template parameters describe the types of the function parameters or return value, and in a template class, a template parameter can designate a field type.

A template is an entire definition. A template ends with a definition of an entity (function, method, class, structure, union) preceded by the template heading.

For template parameter names, it is customary to take one- or two-character identifiers in uppercase.

The minimum number of parameters is 1, the maximum is 64.

The main use cases for parameters (using the T parameter as an example) include:

* type when describing fields, local variables in functions/methods, their parameters and return values ​​(T variable\_name; T function(T parameter\_name));
* one of the components of a fully qualified type name, in particular: T::SubType, T.StaticMember;
* construction of new types with modifiers: const T, pointer T \*, reference T &, array T[], typedef functions T(\*func)(T);
* construction of new template types: T<Type>, Type<T>, including when inheriting from templates (see section [Template specialization, which is not present](/en/book/oop/templates/templates_specialization));
* typecasting (T) with the ability to add modifiers and creating objects via new T();
* sizeof(T) as a primitive replacement for value parameters that are absent in MQL templates (at the time of writing the book).

[Templates](/en/book/oop/templates "Templates")

[General template operation principles](/en/book/oop/templates/templates_principles "General template operation principles")