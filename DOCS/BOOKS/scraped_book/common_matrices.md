---
title: "Matrices and vectors"
url: "https://www.mql5.com/en/book/common/matrices"
hierarchy: []
scraped_at: "2025-11-28 09:48:59"
---

# Matrices and vectors

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Matrices and vectors

* [Types of matrices and vectors](/en/book/common/matrices/matrices_types "Types of matrices and vectors")
* [Creating and initializing matrices and vectors](/en/book/common/matrices/matrices_init "Creating and initializing matrices and vectors")
* [Copying matrices, vectors, and arrays](/en/book/common/matrices/matrices_copy "Copying matrices, vectors, and arrays")
* [Copying timeseries to matrices and vectors](/en/book/common/matrices/matrices_copyrates "Copying timeseries to matrices and vectors")
* [Copying tick history to matrices and vectors](/en/book/common/matrices/matrices_copyticks "Copying tick history to matrices and vectors")
* [Evaluation of expressions with matrices and vectors](/en/book/common/matrices/matrices_expressions "Evaluation of expressions with matrices and vectors")
* [Manipulating matrices and vectors](/en/book/common/matrices/matrices_manipulations "Manipulating matrices and vectors")
* [Products of matrices and vectors](/en/book/common/matrices/matrices_mul "Products of matrices and vectors")
* [Transformations (decomposition) of matrices](/en/book/common/matrices/matrices_decomposition "Transformations (decomposition) of matrices")
* [Obtaining statistics](/en/book/common/matrices/matrices_stats "Obtaining statistics")
* [Characteristics of matrices and vectors](/en/book/common/matrices/matrices_characteristics "Characteristics of matrices and vectors")
* [Solving equations](/en/book/common/matrices/matrices_sle "Solving equations")
* [Machine learning methods](/en/book/common/matrices/matrices_ml "Machine learning methods")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Matrices and vectors

The MQL5 language provides special object data types: matrices and vectors. They can be used to solve a large class of mathematical problems. These types provide methods for writing concise and understandable code close to the mathematical notation of linear or differential equations.

All programming languages support the concept of an array, which is a collection of multiple elements. Most algorithms, especially in algorithmic trading, are constructed on the bases of numeric type arrays (int, double) or structures. Array elements can be accessed by index, which enables the implementation of operations inside loops. As we know, arrays can have one, two, or more dimensions.

Relatively simple data storing and processing tasks can usually be implemented by using arrays. But when it comes to complex mathematical problems, the large number of nested loops makes working with arrays difficult in terms of both programming and reading code. Even the simplest linear algebra operations require a lot of code and a good understanding of mathematics. This task can be simplified by the [functional paradigm](https://en.wikipedia.org/wiki/Functional_programming "Wikipedia: functional programming") of programming, embodied in the form of matrix and vector method functions. These actions perform a lot of routine actions "behind the scenes".

Modern technologies such as machine learning, neural networks, and 3D graphics make extensive use of linear algebra problem solving, which uses operations with vectors and matrices. The new data types have been added to MQL5 for quick and convenient work with such objects.

At the time of writing the book, the set of functions for working with matrices and vectors was actively developed, so many interesting new items may not be mentioned here. Follow the release notes and articles section on the mql5.com site.

In this chapter, we will consider a brief description. For further details about matrices and vectors, please see the corresponding help section [Matrix and vector methods](https://www.mql5.com/en/docs/matrix "MQL5 Documentation").

It is also assumed that the reader is familiar with the Linear Algebra theory. If necessary, you can always turn to reference literature and manuals on the web.

[Predefined constants of the MQL5 language](/en/book/common/environment/env_constants "Predefined constants of the MQL5 language")

[Types of matrices and vectors](/en/book/common/matrices/matrices_types "Types of matrices and vectors")