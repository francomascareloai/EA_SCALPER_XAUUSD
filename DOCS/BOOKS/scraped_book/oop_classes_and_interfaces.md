---
title: "Classes and interfaces"
url: "https://www.mql5.com/en/book/oop/classes_and_interfaces"
hierarchy: []
scraped_at: "2025-11-28 09:48:00"
---

# Classes and interfaces

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")Classes and interfaces

* [OOP fundamentals: Abstraction](/en/book/oop/classes_and_interfaces/classes_abstraction "OOP fundamentals: Abstraction")
* [OOP fundamentals: Encapsulation](/en/book/oop/classes_and_interfaces/classes_encapsulation "OOP fundamentals: Encapsulation")
* [OOP fundamentals: Inheritance](/en/book/oop/classes_and_interfaces/classes_oop_inheritance "OOP fundamentals: Inheritance")
* [OOP fundamentals: Polymorphism](/en/book/oop/classes_and_interfaces/classes_polymorphism "OOP fundamentals: Polymorphism")
* [OOP fundamentals: Composition (design)](/en/book/oop/classes_and_interfaces/classes_composition "OOP fundamentals: Composition (design)")
* [Class definition](/en/book/oop/classes_and_interfaces/classes_definition "Class definition")
* [Access rights](/en/book/oop/classes_and_interfaces/classes_access_rights "Access rights")
* [Constructors: default, parametric, and copying](/en/book/oop/classes_and_interfaces/classes_ctors "Constructors: default, parametric, and copying")
* [Destructors](/en/book/oop/classes_and_interfaces/classes_dtors "Destructors")
* [Self-reference: this](/en/book/oop/classes_and_interfaces/classes_this "Self-reference: this")
* [Inheritance](/en/book/oop/classes_and_interfaces/classes_inheritance "Inheritance")
* [Dynamic creation of objects: new and delete](/en/book/oop/classes_and_interfaces/classes_new_delete_pointers "Dynamic creation of objects: new and delete")
* [Pointers](/en/book/oop/classes_and_interfaces/classes_pointers "Pointers")
* [Virtual methods (virtual and override)](/en/book/oop/classes_and_interfaces/classes_virtual_override "Virtual methods (virtual and override)")
* [Static members](/en/book/oop/classes_and_interfaces/classes_static "Static members")
* [Nested types, namespaces, and the context operator '::'](/en/book/oop/classes_and_interfaces/classes_namespace_context "Nested types, namespaces, and the context operator '::'")
* [Splitting class declaration and definition](/en/book/oop/classes_and_interfaces/classes_declaration_definition "Splitting class declaration and definition")
* [Abstract classes and interfaces](/en/book/oop/classes_and_interfaces/classes_abstract_interfaces "Abstract classes and interfaces ")
* [Operator overloading](/en/book/oop/classes_and_interfaces/classes_operator_overloading "Operator overloading")
* [Object type сasting: dynamic\_cast and pointer void \*](/en/book/oop/classes_and_interfaces/classes_dynamic_cast_void "Object type сasting: dynamic_cast and pointer void *")
* [Pointers, references, and const](/en/book/oop/classes_and_interfaces/classes_ref_pointers_const "Pointers, references, and const")
* [Inheritance management: final and delete](/en/book/oop/classes_and_interfaces/classes_final_delete "Inheritance management: final and delete")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Classes and interfaces

Classes are the main building block in the program development based on the OOP approach. In a global sense, the term class refers to a collection of something (things, people, formulas, etc.) that have some common characteristics. In the context of OOP, this logic is preserved: one class generates objects that have the same set of properties and behavior.

In the previous chapters of this book, we familiarized ourselves with the built-in MQL5 types such as double, int or string. The compiler knows how to store values of these types and what operations can be performed on them. However, these types may not be very convenient to use when describing any application area. For example, a trader has to work with such entities as a trading strategy, a signal filter, a currency basket, and a portfolio of open positions. Each of them consists of a whole set of related properties, subject to specific processing and consistency rules.

A program to automate actions with these objects could consist only of built-in types and simple functions, but then you would have to come up with tricky ways to store and link properties. This is where the OOP technology comes to the rescue, providing ready-made, unified, and intuitive mechanisms for this.

OOP proposes to write all the instructions for storing properties, filling them correctly, and performing permitted operations on objects of a particular user-defined type in a single container with source code. It combines variables and functions in a certain way. Containers are divided into classes, structures, and associations if you list them in descending order of capabilities and relevance.

We have already had an encounter with structures and associations in the [previous chapter](/en/book/oop/structs_and_unions). This knowledge will be useful for classes as well, but classes provide more tools from the OOP arsenal.

By analogy with a structure, a class is a description of a user-defined type with an arbitrary internal storage method and rules for working with it. Based on it, the program can create instances of this class, the objects that should be considered composite variables.

All user-defined types share some of the basic concepts that you might call OOP theory, but they are especially relevant for classes. These include:

* abstraction
* encapsulation
* inheritance
* polymorphism
* composition (design)

Despite the tricky names, they indicate quite simple and familiar norms of the real world, transferred to the world of programming. We'll start our dive into OOP by looking at these concepts. As for the syntax for describing classes and how to create objects — we will discuss it later.

[Unions](/en/book/oop/structs_and_unions/unions "Unions")

[OOP fundamentals: Abstraction](/en/book/oop/classes_and_interfaces/classes_abstraction "OOP fundamentals: Abstraction")