---
title: "Object Oriented Programming in MQL5"
url: "https://www.mql5.com/en/book/oop"
hierarchy: []
scraped_at: "2025-11-28 09:47:25"
---

# Object Oriented Programming in MQL5

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Object Oriented Programming

* [Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")
* [Classes and interfaces](/en/book/oop/classes_and_interfaces "Classes and interfaces")
* [Templates](/en/book/oop/templates "Templates")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Object Oriented Programming in MQL5

At some point during the process of software development, the problem of the built-in types and set of functions not being sufficient for the effective implementation of requirements becomes apparent. The complexity of managing many small entities that make up the program grows like a snowball and requires using some kind of technology capable of improving the convenience, productivity, and quality of the programmer's work.

One of these technologies, implemented at the level of many programming languages, is called Object-Oriented, and the programming style based on it is called Object-Oriented Programming (OOP), respectively. The MQL5 programming language also supports it and therefore belongs to the family of object-oriented languages, like C++.

From the name of the technology, it can be concluded that it is organized around objects. Essentially, an object is a variable of a user-defined type, i.e., a type defined by a programmer using MQL5 tools. The opportunity to create types that model the subject area makes programs more understandable and simplifies their writing and maintenance.

In MQL5, there are several methods to define a new type, and each method is characterized by some features that we will describe in the relevant sections. Depending on the method of description, user-defined types are divided into classes, structures, and associations. Each of them can combine data and algorithms, i.e., describe the state and behavior of applied objects.

In Part 1 of the book, we brought up the quote from one of the fathers of programming, Nicklaus Wirth, that programs are a symbiosis of algorithms and data structures. So, the objects are essentially mini-programs — each is responsible for solving its own, albeit small, but logically complete task. By composing objects into a single system, you can build a service or product of arbitrary complexity. Thus, with the OOP we get a new interpretation of the principle of "divide and conquer".

OOP should be thought of as a more powerful and flexible alternative to the procedural programming style we explored in Part Two. At the same time, both approaches should not be contrasted: if necessary, they can be combined, and in the simplest tasks, OOP can be left aside.

So, in this third Part of the book, we will study the basics of OOP and the possibilities of their practical implementation in MQL5. In addition, we will talk about templates, interfaces, and namespaces.

 

| | |
| --- | --- |
| MQL5 Programming for Traders — Source Codes from the Book. Part 3 | [MQL5 Programming for Traders — Source Codes from the Book. Part 3](https://www.mql5.com/en/code/45592) |
| Примеры из книги также доступны в публичном проекте \MQL5\Shared Projects\MQL5Book. | Examples from the book are also available in the [public project](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#public) \MQL5\Shared Projects\MQL5Book |

[General program properties (#property)](/en/book/basis/preprocessor/preprocessor_properties "General program properties (#property)")

[Structures and unions](/en/book/oop/structs_and_unions "Structures and unions")