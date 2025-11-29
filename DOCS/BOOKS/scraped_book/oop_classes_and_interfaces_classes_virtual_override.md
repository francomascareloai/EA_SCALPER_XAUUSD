---
title: "Virtual methods (virtual and override)"
url: "https://www.mql5.com/en/book/oop/classes_and_interfaces/classes_virtual_override"
hierarchy: []
scraped_at: "2025-11-28 09:49:22"
---

# Virtual methods (virtual and override)

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Object Oriented Programming](/en/book/oop "Object Oriented Programming")[Classes and interfaces](/en/book/oop/classes_and_interfaces "Classes and interfaces")Virtual methods (virtual and override)

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
* Virtual methods (virtual and override)
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

# Virtual methods (virtual and override)

Classes are intended to describe external programming interfaces and provide their internal implementation. Since the functionality of our test program is to draw various shapes, we have described several variables in the class Shape and its descendants for future implementation, and also reserved the method draw for the interface.

In the base class Shape, it shouldn't and can't do anything because Shape is not a concrete shape: we'll convert Shape to an abstract class later (we will talk more about [abstract classes and interfaces](/en/book/oop/classes_and_interfaces/classes_abstract_interfaces) later).

Let's redefine the draw method in the Rectangle, Ellipse and other derived classes (Shapes3.mq5). This involves copying the method and modifying its content accordingly. Although many refer to this process as "overriding", we will distinguish between the two terms, reserving "overriding" exclusively for virtual methods, which will be discussed later.

Strictly speaking, redefining a method only requires the method name to match. However, to ensure consistent usage throughout the code, it is essential to maintain the same parameter list and return type.

| |
| --- |
| class Rectangle : public Shape {    ...    void draw()    {       Print("Drawing rectangle");    } }; |

Since we don't know how to draw on the screen yet, we'll just output the message to the log.

It is important to note that by providing a new implementation of the method in the derived class, we thereby get 2 versions of the method: one refers to the built-in base object (inner Shape), and the other to the derived one (outer Rectangle).

The first will be called for a variable of type Shape, and the second one for a variable of type Rectangle.

In a longer inheritance chain, a method can be redefined and propagated even more times.

You can change an access type of a new method, for example, make it public if it was protected, or vice versa. But in this case, we left the draw method in the public section.

If necessary, the programmer can call the implementation of the method of any of the progenitor classes: for this, a special [context resolution operator](/en/book/oop/classes_and_interfaces/classes_namespace_context) is used – two colons '::'. In particular, we could call the draw implementation from the class Rectangle from the method draw of the class Square: for this, we specify the name of the desired class, '::' and the method name, for example, Rectangle::draw(). Calling draw without specifying the context implies a method of the current class, and therefore if you do it from the method draw itself, you will get an infinite [recursion](/en/book/basis/functions/functions_recursive), and ultimately, a stack overflow and program crash.

| |
| --- |
| class Square : public Rectangle { public:    ...    void draw()    {       Rectangle::draw();       Print("Drawing square");    } }; |

Then calling draw on the object Square would log two lines:

| |
| --- |
| Square s(100, 200, 50, clrGreen);    s.draw(); // Drawing rectangle              // Drawing square |

Binding a method to a class in which it is declared provides the static dispatch (or static binding): the compiler decides which method to call at the compilation stage and "hardwires" the found match into binary code.

During the decision process, the compiler looks for the method to be called in the object of the class for which the dereference ('.') is performed. If the method is present, it is called, and if not, the compiler checks the parent class for the presence of the method, and so on, through the inheritance chain until the method is found. If the method is not found in any of the classes in the chain, an "undeclared identifier" compilation error will occur.

In particular, the following code calls the setColor method on the object Rectangle:

| |
| --- |
| Rectangle r(100, 200, 75, 50, clrBlue);    r.setColor(clrWhite); |

However, this method is defined only in the base class Shape and is built in once in all descendant classes, and therefore it will be executed here.

Let's try to start drawing arbitrary shapes from an array in the function OnStart (recall that we have duplicated and modified the method draw in all descendant classes).

| |
| --- |
| for(int i = 0; i < 10; ++i)    {       shapes[i].draw();    } |

Oddly enough, nothing is output to the log. This happens because the program calls the method draw of the class Shape.

There is a major drawback of static dispatch here: when we use a pointer to a base class to store an object of a derived class, the compiler chooses a method based on the type of the pointer, not the object. The fact is that at the compilation stage, it is not yet known what class object it will point to during program execution.

Thus, there is a need for a more flexible approach: a dynamic dispatch (or binding), which would defer the choice of a method (from among all the overridden versions of the method in the descendant chain) to runtime. The choice must be made based on analysis of the actual class of the object at the pointer. It is dynamic dispatching that provides the principle of [polymorphism](/en/book/oop/classes_and_interfaces/classes_polymorphism).

This approach is implemented in MQL5 using virtual methods. In the description of such a method, the keyword virtual must be added at the beginning of the header.

Let's declare the method draw in the class Shape (Shapes4.mq5) as virtual. This will automatically make all versions of it in derived classes virtual as well.

| |
| --- |
| class Shape {    ...    virtual void draw()    {    } }; |

Once a method is virtualized, modifying it in derived classes is called overriding rather than redefinition. Overriding requires the name, parameter types, and return value of the method to match (taking into account the presence/absence of const modifiers).

Note that overriding virtual functions is different from [function overloading](/en/book/basis/functions/functions_overloading). Overloading uses the same function name, but with different parameters (in particular, we saw the possibility of overloading a constructor in the example of structures, see [Constructors and Destructors](/en/book/oop/structs_and_unions/structs_ctor_dtor)), and overriding requires full matching of function signatures. 
  
Overridden functions must be defined in different classes that are related by inheritance relationships. Overloaded functions must be in the same class – otherwise, it will not be an overload but, most likely, a redefinition (and it will work differently, see further analysis of the example OverrideVsOverload.mq5).

If you run a new script, the expected lines will appear in the log, signaling calls to specific versions of the draw method in each of the classes.

| |
| --- |
| Drawing square Drawing circle Drawing triangle Drawing ellipse Drawing triangle Drawing rectangle Drawing square Drawing triangle Drawing square Drawing triangle |

In derived classes where a virtual method is overridden, it is recommended to add the keyword override to its header (although this is not required).

| |
| --- |
| class Rectangle : public Shape {    ...    void draw() override    {       Print("Drawing rectangle");    } }; |

This allows the compiler to know that we are overriding the method on purpose. If in the future the API of the base class suddenly changes and the overridden method is no longer virtual (or simply removed), the compiler will generate an error message: "method is declared with 'override' specifier, but does not override any base class method". Keep in mind that even adding or removing the modifier const from a method changes its signature, and overriding may be broken due to this.

The keyword virtual before an overridden method is also allowed but not required.

For dynamic dispatching to work, the compiler generates a table of virtual functions for each class. An implicit field is added to each object with a link to the given table of its class. The table is populated by the compiler based on information about all virtual methods and their overridden versions along the inheritance chain of a particular class.

A call to a virtual method is encoded in the binary image of the program in a special way: first, the table is looked up in search of a version for a class of a particular object (located at the pointer), and then a transition is made to the appropriate function.

As a result, dynamic dispatch is slower than static dispatch.

In MQL5, classes always contain a table of virtual functions, regardless of the presence of virtual methods.

If a virtual method returns a pointer to a class, then when it is overridden, it is possible to change (make it more specific, highly specialized) the object type of the return value. In other words, the type of the pointer can be not only the same as in the initial declaration of the virtual method but also any of its successors. Such types are called "covariant" or interchangeable.

For example, if we made the method setColor virtual in the class Shape:

| |
| --- |
| class Shape {    ...    virtual Shape \*setColor(const color c)    {       backgroundColor = c;       return &this;    }    ... }; |

we could override it in the class Rectangle like this (only as a demonstration of the technology):

| |
| --- |
| class Rectangle : public Shape {    ...    virtual Rectangle \*setColor(const color c) override    {       // call original method       // (by pre-lightening the color,       // no matter what for)       Rectangle::setColor(c | 0x808080);       return &this;    } }; |

Note that the return type is a pointer to Rectangle instead of Shape.

It makes sense to use a similar trick if the overridden version of the method changes something in that part of the object that does not belong to the base class, so that the object, in fact, no longer corresponds to the allowed state (invariant) of the base class.

Our example with drawing shapes is almost ready. It remains to fill the virtual methods draw with real content. We will do this in the chapter [Graphics](/en/book/applications/objects/objects_color_style) (see example ObjectShapesDraw.mq5), but we will improve it after studying [graphic resources](/en/book/advanced/resources/resources_resourcesave).

Taking into account the inheritance concept, the procedure by which the compiler chooses the appropriate method looks a bit confusing. Based on the method name and the specific list of arguments (their types) in the call instruction, a list of all available candidate methods is compiled. 
  
For non-virtual methods, at the beginning only methods of the current class are analyzed. If none of them matches, the compiler will continue searching the base class (and then more distant ancestors until it finds a match). If among the methods of the current class, there is a suitable one (even if the implicit conversion of argument types is necessary), it will be picked. If the base class had a method with more appropriate argument types (no conversion or fewer conversions), the compiler still won't get to it. In other words, non-virtual methods are analyzed starting from the class of the current object towards the ancestors to the first "working" match. 
  
For virtual methods, the compiler first finds the required method by name in the pointer class and then selects the implementation in the table of virtual functions for the most instantiated class (furthest descendant) in which this method is overridden in the chain between the pointer type and the object type. In this case, implicit argument conversion can also be used if there is no exact match between the types of arguments.

Let's consider the following example (OverrideVsOverload.mq5). There are 4 classes that are chained: Base, Derived, Concrete and Special. All of them contain methods with type arguments int and float. In the function OnStart, the integer i and the real f variables are used as arguments for all method calls.

| |
| --- |
| class Base { public:    void nonvirtual(float v)    {       Print(\_\_FUNCSIG\_\_, " ", v);    }    virtual void process(float v)    {       Print(\_\_FUNCSIG\_\_, " ", v);    } };   class Derived : public Base { public:    void nonvirtual(int v)    {       Print(\_\_FUNCSIG\_\_, " ", v);    }    virtual void process(int v) // override    // error: 'Derived::process' method is declared with 'override' specifier,    // but does not override any base class method    {       Print(\_\_FUNCSIG\_\_, " ", v);    } };   class Concrete : public Derived { };   class Special : public Concrete { public:    virtual void process(int v) override    {       Print(\_\_FUNCSIG\_\_, " ", v);    }    virtual void process(float v) override    {       Print(\_\_FUNCSIG\_\_, " ", v);    } }; |

First, we create an object of class Concrete and a pointer to it Base \*ptr. Then we call non-virtual and virtual methods for them. In the second part, the methods of the object Special are called through the class pointers Base and Derived.

| |
| --- |
| void OnStart() {    float f = 2.0;    int i = 1;      Concrete c;    Base \*ptr = &c;        // Static link tests      ptr.nonvirtual(i); // Base::nonvirtual(float), conversion int -> float    c.nonvirtual(i);   // Derived::nonvirtual(int)      // warning: deprecated behavior, hidden method calling    c.nonvirtual(f);   // Base::nonvirtual(float), because                       // method selection ended in Base,                       // Derived::nonvirtual(int) does not suit to f      // Dynamic link tests      // attention: there is no method Base::process(int), also    // there are no process(float) overrides in classes up to and including Concrete    ptr.process(i);    // Base::process(float), conversion int -> float    c.process(i);      // Derived::process(int), because                       // there is no override in Concrete,                       // and the override in Special does not count      Special s;    ptr = &s;    // attention: there is no method Base::process(int) in ptr    ptr.process(i);    // Special::process(float), conversion int -> float    ptr.process(f);    // Special::process(float)      Derived \*d = &s;    d.process(i);      // Special::process(int)      // warning: deprecated behavior, hidden method calling    d.process(f);      // Special::process(float) } |

The log output is shown below.

| |
| --- |
| void Base::nonvirtual(float) 1.0 void Derived::nonvirtual(int) 1 void Base::nonvirtual(float) 2.0 void Base::process(float) 1.0 void Derived::process(int) 1 void Special::process(float) 1.0 void Special::process(float) 2.0 void Special::process(int) 1 void Special::process(float) 2.0 |

The ptr.nonvirtual(i) call is made using static binding, and the integer i is preliminarily cast to the parameter type, float.

The call c.nonvirtual(i) is also static, and since there is no void nonvirtual(int) method in the class Concrete, the compiler finds such a method in the parent class Derived.

Calling the function of the same name on the same object with a value of type float leads the compiler to the method Base::nonvirtual(float) because Derived::nonvirtual(int) is not suitable (the conversion would lead to a loss of precision). Along the way, the compiler issues a "deprecated behavior, hidden method calling" warning.

Overloaded methods may look like overridden (have the same name but different parameters) but they are different because they are located in different classes. When a method in a derived class overrides a method in a parent class, it replaces the behavior of the parent class method which can sometimes cause unexpected effects. The programmer might expect the compiler to choose another suitable method (as in overloading), but instead the subclass is invoked.

To avoid potential warnings, if the implementation of the parent class is necessary, it should be written as exactly the same function in the derived class, and the base class should be called from it.

| |
| --- |
| class Derived : public Base { public:    ...    // this override will suppress the warning    // "deprecated behavior, hidden method calling"    void nonvirtual(float v)    {       Base::nonvirtual(v);       Print(\_\_FUNCSIG\_\_, " ", v);    } ... |

Let's go back to tests in OnStart.

Calling ptr.process(i) demonstrates the confusion between overloading and overriding described above. The Base class has a process(float) virtual method, and the class Derived adds a new virtual method process(int), which is not overriding in this case because parameter types are different. The compiler selects a method by name in the base class and checks the virtual function table for overrides in the inheritance chain up to the Concrete class (inclusive, this is the object class by pointer). Since no overrides were found, the compiler took Base::process(float) and applied the type conversion of the argument to the parameter (int to float).

If we followed the rule of always writing the word override where redefining is implied and added it to Derived, we would get an error:

| |
| --- |
| class Derived : public Base {    ...    virtual void process(int v) override // error!    {       Print(\_\_FUNCSIG\_\_, " ", v);    } }; |

The compiler would report "'Derived::process' method is declared with 'override' specifier, but does not override any base class method". This would serve as a hint to fixing the problem.

Calling process(i) on the Concrete object is done with Derived::process(int). Although we have an even further redefinition in the class Special, it is irrelevant because it's done in the inheritance chain after the Concrete class.

When the pointer ptr is later assigned to the Special object, calls to process(i) and process(f) are resolved by the compiler as Special::process(float) because Special overrides Base::process(float). The choice of the float parameter occurs for the same reason as described earlier: the method Base::process(float) is overridden by Special.

If we apply the pointer d of type Derived, then we finally get the expected call Special::process(int) for the string d.process(i). The point is that process(int) is defined in Derived, and falls into the scope of the compiler's search.

Note that the Special class not only overrides the inherited virtual methods but also overloads two methods in the class itself.

Do not call a virtual function from a constructor or destructor! While technically possible, the virtual behavior in the constructor and destructor is completely lost and you might get unexpected results. Not only explicit but also indirect calls should be avoided (for example, when a simple method is called from a constructor, which in turn calls a virtual one). 
  
Let's analyze the situation in more detail using the example of a constructor. The fact is that at the time of the constructor's work, the object is not yet fully assembled along the entire inheritance chain, but only up to the current class. All derived part have yet to be "finished" around the existing core. Therefore, all later virtual method overrides (if any) are not yet available at this point. As a result, the current version of the method will be called from the constructor.

[Pointers](/en/book/oop/classes_and_interfaces/classes_pointers "Pointers")

[Static members](/en/book/oop/classes_and_interfaces/classes_static "Static members")