---
title: "Working with arrays"
url: "https://www.mql5.com/en/book/common/arrays"
hierarchy: []
scraped_at: "2025-11-28 09:49:09"
---

# Working with arrays

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Working with arrays

* [Logging arrays](/en/book/common/arrays/arrays_print "Logging arrays")
* [Dynamic arrays](/en/book/common/arrays/arrays_dynamic "Dynamic arrays")
* [Array measurement](/en/book/common/arrays/arrays_metrics "Array measurement")
* [Initializing and populating arrays](/en/book/common/arrays/arrays_init_fill "Initializing and populating arrays")
* [Copying and editing arrays](/en/book/common/arrays/arrays_edit "Copying and editing arrays")
* [Moving (swapping) arrays](/en/book/common/arrays/arrays_move_swap "Moving (swapping) arrays")
* [Comparing, sorting, and searching in arrays](/en/book/common/arrays/arrays_compare_sort_search "Comparing, sorting, and searching in arrays")
* [Timeseries indexing direction in arrays](/en/book/common/arrays/arrays_as_series "Timeseries indexing direction in arrays")
* [Zeroing objects and arrays](/en/book/common/arrays/zero_memory "Zeroing objects and arrays")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Working with arrays

It is difficult to imagine any program and especially one related to trading, without arrays. We have already studied the general principles of describing and using arrays in the [Arrays chapter](/en/book/basis/arrays). They are organically complemented by a set of built-in functions for working with arrays.

Some of them provide ready-made implementations of the most commonly used array operations, such as finding the maximum and minimum, sorting, inserting, and deleting elements.

However, there are a number of functions without which it is impossible to use arrays of specific types. In particular, a dynamic array must first allocate memory before working with it, and arrays with data for indicator buffers (we will study this MQL program type in Part 5 of the book) use a special order of element indexing, set by a special function.

And we will begin looking at functions for working with arrays with the output operation to the log. We already saw it in previous chapters of the book and will be useful in many subsequent ones.

Since MQL5 arrays can be multidimensional (from 1 to 4 dimensions), we will need to refer to the dimension numbers further in the text. We will call them numbers, starting with the first, which is more familiar geometrically and which emphasizes the fact that an array must have at least one dimension (even if it is empty). However, array elements for each dimension are numbered, as is customary in MQL5 (and in many other programming languages), from zero. Thus, for an array described as array[5][10], the first dimension is 5 and the second is 10.

[Universal formatted data output to a string](/en/book/common/strings/strings_format "Universal formatted data output to a string")

[Logging arrays](/en/book/common/arrays/arrays_print "Logging arrays")