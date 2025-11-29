---
title: "Templates"
url: "https://www.mql5.com/en/book/oop/templates"
hierarchy: []
scraped_at: "2025-11-28 09:47:58"
---

# Templates

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")Templates

* [Template header](/en/book/oop/templates/templates_header "Template header")
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

# Templates

In modern programming languages, there are many built-in features that allow you to avoid code duplication and, thereby, minimize the number of errors and increase programmer productivity. In MQL5, such tools include the already known [functions](/en/book/basis/functions), object types with inheritance support ([classes](/en/book/oop/classes_and_interfaces) and [structures](/en/book/oop/structs_and_unions)), [preprocessor macros](/en/book/basis/preprocessor/preprocessor_define_functional), and the ability to [include files](/en/book/basis/preprocessor/preprocessor_include). But this list would be incomplete without templates.

A template is a specially crafted generic definition of a function or object type from which the compiler can automatically generate working instances of that function or object type. The resulting instances contain the same algorithm but operate on variables of different types, corresponding to the specific conditions for using the template in the source code.

For C++ connoisseurs, we note that MQL5 templates do not support many features of C++ templates, in particular:

* parameters that are not types;
* parameters with default values;
* variable number of parameters;
* specialization of classes, structures, and associations (full and partial);
* templates for templates.

On the one hand, this reduces the potential of templates in MQL5, but, on the other hand, it simplifies the learning of the material for those who are unfamiliar with these technologies.

[Inheritance management: final and delete](/en/book/oop/classes_and_interfaces/classes_final_delete "Inheritance management: final and delete")

[Template header](/en/book/oop/templates/templates_header "Template header")