---
title: "Variables"
url: "https://www.mql5.com/en/book/basis/variables"
hierarchy: []
scraped_at: "2025-11-28 10:15:37"
---

# Variables

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")Variables

* [Declaration and definition of variables](/en/book/basis/variables/define_vs_declare "Declaration and definition of variables")
* [Context, scope, and lifetime of variables](/en/book/basis/variables/scope_and_lifetime "Context, scope, and lifetime of variables")
* [Initialization](/en/book/basis/variables/initialization "Initialization")
* [Static variables](/en/book/basis/variables/static_variables "Static variables")
* [Constant variables](/en/book/basis/variables/const_variables "Constant variables")
* [Input variables](/en/book/basis/variables/input_variables "Input variables")
* [External variables](/en/book/basis/variables/variables_extern "External variables")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Variables

In this chapter, we will learn the basic principles of working with variables in MQL5, namely those relating to embedded data types. In particular, we will consider the declaration and definition of variables, special features of initialization as the context requires, lifetime, and basic modifiers changing the properties of variables. Later on, relying on this knowledge, we will extend the abilities of variables with new custom types (unions, custom enumerations, and aliases), classes, pointers, and references.

Variables in MQL5 provide a mechanism for storing data of various types, playing an important role in organizing program logic and operations with market information. This section includes the following subsections:

[Declaration and definition of variables](/en/book/basis/variables/define_vs_declare):

* Variable declaration is the step of creating them in a program. In this section, we look at how to declare and define variables, as well as how to specify their types.

[Context, scope, and lifetime of variables](/en/book/basis/variables/scope_and_lifetime):

* Variables can exist in different contexts and scopes, which affects their availability and lifetime. This subsection covers these aspects, helping you understand how variables interact with your code.

[Initialization](/en/book/basis/variables/initialization):

* Initialization of variables involves assigning them initial values. We study methods of initialization, helping to avoid undefined program behavior.

[Static variables](/en/book/basis/variables/static_variables):

* Static variables retain their values between function calls. This section explains how to use static variables to store information between different code executions.

[Constant variables](/en/book/basis/variables/const_variables):

* Constant variables represent values that do not change during program execution. This section describes their usage and characteristics.

[Input variables](/en/book/basis/variables/input_variables):

* Input variables are used in trading robots to configure strategy parameters. We will see how to use them to create flexible and customizable trading systems.

[External variables](/en/book/basis/variables/variables_extern):

* External variables allow users to interact with the program as their values can be changed without the need to modify the code. This section explains how external variables work.

[Void type](/en/book/basis/builtin_types/void "Void type")

[Declaration and definition of variables](/en/book/basis/variables/define_vs_declare "Declaration and definition of variables")