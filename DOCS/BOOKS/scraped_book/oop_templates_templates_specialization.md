---
title: "Absent template specialization"
url: "https://www.mql5.com/en/book/oop/templates/templates_specialization"
hierarchy: []
scraped_at: "2025-11-28 09:49:12"
---

# Absent template specialization

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Templates](/en/book/oop/templates "Templates")Absent template specialization

* [Template header](/en/book/oop/templates/templates_header "Template header")
* [General template operation principles](/en/book/oop/templates/templates_principles "General template operation principles")
* [Templates vs preprocessor macros](/en/book/oop/templates/templates_vs_macro "Templates vs preprocessor macros")
* [Features of built-in and object types in templates](/en/book/oop/templates/templates_for_standard_and_object_types "Features of built-in and object types in templates")
* [Function templates](/en/book/oop/templates/templates_functions "Function templates")
* [Object type templates](/en/book/oop/templates/templates_objects "Object type templates")
* [Method templates](/en/book/oop/templates/templates_methods "Method templates")
* [Nested templates](/en/book/oop/templates/templates_nested "Nested templates")
* Absent template specialization

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Absent template specialization

In some cases, it may be necessary to provide a template implementation for a particular type (or set of types) in a way that differs from the generic one. For example, it usually makes sense to prepare a special version of the swap function for pointers or arrays. In such cases, C++ allows you to do what is called template specialization, that is, to define a version of the template in which the generic type parameter T is replaced by the required concrete type.

When specializing function and method templates, specific types must be specified for all parameters. This is called complete specialization.

In the case of C++ object type templates, specialization can be not only complete but also partial: it specifies the type of only some of the parameters (and the rest will be inferred or specified when the template is instantiated). There can be several partial specializations: the only condition for this is that each specialization must describe a unique combination of types.

Unfortunately, there is no specialization in MQL5 in the full sense of the word.

Template function specialization is no different from overloading. For example, given the following template func:

| |
| --- |
| template<typename T> void func(T t) { ... } |

it is allowed to provide its custom implementation for a given type (such as string) in one of the forms:

| |
| --- |
| // explicit specialization  template<> void func(string t) { ... } |

or:

| |
| --- |
| // normal overload  void func(string t) { ... } |

Only one of the forms must be selected. Otherwise, we get a compilation error "'func' - function already defined and has body".

As for the specialization of classes, inheritance from templates with an indication of specific types for some of the template parameters can be considered as an equivalent of their partial specialization. Template methods can be overridden in a derived class.

The following example (TemplatesExtended.mq5) shows several options for using template parameters as parent types, including cases where one of them is specified as specific.

| |
| --- |
| #define RTTI Print(typename(this))     class Base { public:    Base() { RTTI; } };     template<typename T>  class Derived : public T { public:    Derived() { RTTI; } };      template<typename T>  class Base1 {    Derived<T> object; public:    Base1() { RTTI; } };      template<typename T>                // complete "specialization" class Derived1 : public Base1<Base> // 1 of 1 parameter is set  { public:    Derived1() { RTTI; } };      template<typename T,typename E>  class Base2 : public T { public:    Base2() { RTTI; } };      template<typename T>                    // partial "specialization" class Derived2 : public Base2<T,string> // 1 of 2 parameters is set  { public:    Derived2() { RTTI; } }; |

We will provide an instantiation of an object according to a template using a variable:

| |
| --- |
| Derived2<Derived1<Base>> derived2; |

Debug type logging using the RTTI macro produces the following result:

| |
| --- |
| Base Derived<Base> Base1<Base> Derived1<Base> Base2<Derived1<Base>,string> Derived2<Derived1<Base>> |

When developing [libraries](/en/book/advanced/libraries) that come as closed binary, you must ensure that templates are explicitly instantiated for all types that future users of the library are expected to work with. You can do this by explicitly calling function templates and creating objects with type parameters in some auxiliary function, for example, bound to the initialization of a global variable.

[Nested templates](/en/book/oop/templates/templates_nested "Nested templates")

[Common APIs](/en/book/common "Common APIs")