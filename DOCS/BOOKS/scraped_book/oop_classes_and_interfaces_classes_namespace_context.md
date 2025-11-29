---
title: "Nested types, namespaces, and the context operator '::'"
url: "https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_namespace_context"
hierarchy: []
scraped_at: "2025-11-28 09:49:07"
---

# Nested types, namespaces, and the context operator '::'

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Classes and interfaces](/en/book/oop/classes_and_interfaces "Classes and interfaces")Nested types, namespaces, and the context operator '::'

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
* Nested types, namespaces, and the context operator '::'
* [Splitting class declaration and definition](/en/book/oop/classes_and_interfaces/classes_declaration_definition "Splitting class declaration and definition")
* [Abstract classes and interfaces](/en/book/oop/classes_and_interfaces/classes_abstract_interfaces "Abstract classes and interfaces ")
* [Operator overloading](/en/book/oop/classes_and_interfaces/classes_operator_overloading "Operator overloading")
* [Object type сasting: dynamic\_cast and pointer void \*](/en/book/oop/classes_and_interfaces/classes_dynamic_cast_void "Object type сasting: dynamic_cast and pointer void *")
* [Pointers, references, and const](/en/book/oop/classes_and_interfaces/classes_ref_pointers_const "Pointers, references, and const")
* [Inheritance management: final and delete](/en/book/oop/classes_and_interfaces/classes_final_delete "Inheritance management: final and delete")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Nested types, namespaces, and the context operator '::'

Classes, structures, and unions can be described not only in the global context but also within another class or structure. And even more: the definition can be done inside the function. This allows you to describe all the entities necessary for the operation of any class or structure within the appropriate context and thereby avoid potential name conflicts.

In particular, in the drawing program, the structure for storing coordinates Pair has been defined globally so far. As the program grows, it is quite possible that another entity called Pair will be needed (especially given the rather generic name). Therefore, it is desirable to move the description of the structure inside the class Shape (Shapes6.mq5).

| |
| --- |
| class Shape { public:    struct Pair    {       int x, y;       Pair(int a, int b): x(a), y(b) { }    };    ... }; |

The nested descriptions have access permissions in accordance with the specified section modifiers. In this case, we have made the name Pair publicly available. Inside the class Shape, the handling of the Pair structure type does not change in any way due to the transfer. However, in external code, you must specify a fully qualified name that includes the name of the external class (context), the context selection operator '::' and the internal entity identifier itself. For example, to describe a variable with a pair of coordinates, you would write:

| |
| --- |
| Shape::Pair coordinates(0, 0); |

The level of nesting when describing entities is not limited, so a fully qualified name can contain identifiers of multiple levels (contexts) separated by '::'. For example, we could wrap all drawing classes inside the outer class Drawing, in the public section.

| |
| --- |
| class Drawing { public:    class Shape    {    public:       struct Pair       {          ...       };    };    class Rectangle : public Shape    {       ...    };    ... }; |

Then fully qualified type names (e.g. for use in OnStart or other external functions) would be lengthened:

| |
| --- |
| Drawing::Shape::Rect coordinates(0, 0); Drawing::Rectangle rect(200, 100, 70, 50, clrBlue); |

On the one hand, this is inconvenient, but on the other hand, it is sometimes a necessity in large projects with a large number of classes. In our small project, this approach is used only to demonstrate the technical feasibility.

To combine logically related classes and structures into named groups, MQL5 provides an easier way than including them in an "empty" wrapper class.

A namespace is declared using the keyword namespace followed by the name and a block of curly braces that includes all the necessary definitions. Here's what the same paint program looks like using namespace:

| |
| --- |
| namespace Drawing {    class Shape    {    public:       struct Pair       {          ...       };    };    class Rectangle : public Shape    {       ...    };    ... } |

There are two main differences: the internal contents of the space are always available publicly (access modifiers are not applicable in it) and there is no semicolon after the closing curly brace.

Let's add the method move to the class Shape, which takes the structure Pair as a parameter:

| |
| --- |
| class Shape { public:    ...    Shape \*move(const Pair &pair)    {       coordinates.x += pair.x;       coordinates.y += pair.y;       return &this;    } }; |

Then, in the function OnStart, you can organize the shift of all shapes by a given value by calling this function:

| |
| --- |
| void OnStart() {    //draw a random set of shapes    for(int i = 0; i < 10; ++i)    {       Drawing::Shape \*shape = addRandomShape();       // move all shapes       shape.move(Drawing::Shape::Pair(100, 100));       shape.draw();       delete shape;    } } |

Note that the types Shape and Pair have to be described with full names: Drawing::Shape and Drawing::Shape::Pair respectively.

There may be several blocks with the same space name: all their contents will fall into one logically unified context with the specified name.

Identifiers defined in the global context, in particular all built-in functions of the MQL5 API, are also available through the context selection operator not preceded by any notation. For example, here's what a call to the function Print might look like:

| |
| --- |
| ::Print("Done!"); |

When the call is made from any function defined in the global context, there is no need for such an entry.

Necessity can manifest itself inside any class or structure if an element of the same name (function, variable or constant) is defined in them. For example, let's add the method Print to the class Shape:

| |
| --- |
| static void Print(string x)    {       // empty       // (likely will output it to a separate log file later)    } |

Since the test implementations of the draw method in derived classes call Print, they are now redirected to this Print method: from several identical identifiers, the compiler chooses the one that is defined in a closer context. In this case, the definition in the base class is closer to the shapes than the global context. As a result, logging output from shape classes will be suppressed.

However, calling Print from the function OnStart still works (because it is outside the context of the class Shape).

| |
| --- |
| void OnStart() {    ...    Print("Done!"); } |

To "fix" debug printing in classes, you need to precede all Print calls with a global context selection operator:

| |
| --- |
| class Rectangle : public Shape {    ...    void draw() override    {       ::Print("Drawing rectangle"); // reprint via global Print(...)    } }; |

[Static members](/en/book/oop/classes_and_interfaces/classes_static "Static members")

[Splitting class declaration and definition](/en/book/oop/classes_and_interfaces/classes_declaration_definition "Splitting class declaration and definition")