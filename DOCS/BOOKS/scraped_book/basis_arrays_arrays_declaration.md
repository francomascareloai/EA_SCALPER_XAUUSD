---
title: "Description of arrays"
url: "https://www.mql5.com/en/book/basis/arrays/arrays_declaration"
hierarchy: []
scraped_at: "2025-11-28 09:49:36"
---

# Description of arrays

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Programming fundamentals](/en/book/basis "Programming fundamentals")[Arrays](/en/book/basis/arrays "Arrays")Description of arrays

* [Array characteristics](/en/book/basis/arrays/arrays_overview "Array characteristics")
* Description of arrays
* [Using arrays](/en/book/basis/arrays/arrays_usage "Using arrays")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Description of arrays

Array description inherits some features of variable descriptions. To start with, we should note that arrays may be global and local, based on the place of their declaration. Similarly to variables, modifiers const and static can also be used in describing an array. For a one-dimension fixed-size array, the declaration syntax appears as follows:

| |
| --- |
| type static1D[size]; |

Here, type and static1D denote the type name of elements and the array identifier, respectively, while size in square brackets is a size-defining integer constant.

For multidimensional arrays, several sizes must be specified, according to the quantity of dimensions:

| |
| --- |
| type static2D[size1][size2]; type static3D[size1][size2][size3]; type static4D[size1][size2][size3][size4]; |

Dynamic arrays are described in a similar manner, except that a skip is made in the first square brackets (before using such an array, the required memory volume must be allocated for it using the ArrayResize function, see the section dealing with [dynamic arrays](/en/book/common/arrays/arrays_dynamic)).

| |
| --- |
| type dynamic1D[]; type dynamic2D[][size2]; type dynamic3D[][size2][size3]; type dynamic4D[][size2][size3][size4]; |

For fixed-size arrays, initialization is permitted: Initial values are specified for the elements after the equal sign, as a comma-separated list, the entire list being enclosed in braces. For example:

| |
| --- |
| int array1D[3] = {10, 20, 30}; |

Here, a 3-sized integer array takes the values of 10, 20, and 30.

With an initialization list, there is no need to specify the array size in square brackets (for the first dimension). The compiler will assess the size automatically by the list length. For example:

| |
| --- |
| int array1D[] = {10, 20, 30}; |

Initial values can be both constants and the constant expressions, i.e., formulas the compiler can compute during compilation. For example, the following array is filled with the number of seconds in a minute, hour, day, and week (representation as formulas is more illustrative than 86400 or 604800):

| |
| --- |
| int seconds[] = {60, 60 \* 60, 60 \* 60 \* 24, 60 \* 60 \* 24 \* 7}; |

Such values are usually designed as a preprocessor macro in the code beginning, and then the name of this macro is inserted everywhere where it is necessary in the text. This option is described in the section related to the [Preprocessor](/en/book/basis/preprocessor/preprocessor_define_overview).

The number of initializing elements may not exceed the array size. Otherwise, the compiler will give the error message, "too many initializers". If the quantity of values is smaller than the array size, the resting elements are initialized by zero. Therefore, there is a brief notation to initialize the entire array by zeros:

| |
| --- |
| int array2D[2][3] = {0}; |

Or just empty braces:

| |
| --- |
| int array2D[2][3] = {}; |

It works regardless of the number of dimensions.

To initialize multidimensional arrays, the lists must be nested. For example:

| |
| --- |
| int array2D[3][2] = {{1, 2}, {3, 4}, {5, 6}}; |

Here, the first-dimension size of the array is 3; therefore, two commas frame 3 elements inside the external braces. However, since the array is two-dimensional, each of its elements is an array, in turn, the size of each being 2. This is why each element represents a list in braces, each list containing 2 values.

Supposing, we need a transposed array (the first size is 2, and the second one is 3), then its initialization will change:

| |
| --- |
| int array2D[2][3] = {{1, 3, 5}, {2, 4, 6}}; |

We can skip one or more values in the initialization list, if necessary, having marked their places with commas. All skipped elements will also be initialized by zero.

| |
| --- |
| int array1D[3] = {, , 30}; |

Here, the first elements will be equal to 0.

The language syntax permits placing a comma after the last element:

| |
| --- |
| string messages[] = {   "undefined",   "success",   "error", }; |

This simplifies adding new elements, especially for multi-string entries. Particularly, if we forget to enter a comma before the newly added element in a string array, the old and the new strings will turn out to be fused within one element (with the same index), while no new element will appear. Moreover, some arrays may be generated automatically (by another program or by macros). Therefore, the unified appearance of all elements is natural.

"Heap" and "Stack" 
  
 With arrays that can potentially be large, it is important to make the distinction between global and local location in memory. 
  
Memory for global variables and arrays is distributed within the 'heap', i.e., free memory available to the program. This memory is not practically limited by anything, apart from the physical characteristics of your computer and operating system. The name of 'heap' is explained by the fact that differently sized memory areas are always either allocated or deallocated by the program, which results in the free areas being randomly scattered within the entire bulk. 
  
Local variables and arrays are located in the stack, i.e., a limited memory area preliminarily allocated for the program, especially for local elements. The name of 'stack' derives from the fact that, during the algorithm execution, the nested calls of functions take place, which accumulate their internal data according to the "piled-up" principle: For instance, OnStart is called by the terminal, a function from your applied code is called from OnStart, then your other function is called from the previous one, etc. At the same time, when entering each function, its local variables are created that continue being there when the nested function is called. It creates local variables, too, which get onto the stack somewhat over the preceding ones. As a result, a stack usually contains some layers of the local data from all functions that had been activated on the path to the current code string. Not until the function being on the top of the stack is completed, its local data will be removed from there. Generally, the stack is a storage that works according to the FILO/LIFO (First In Last Out, Last In First Out) principle. 
  
Since the stack size is limited, it is recommended to create only local variables in it. However, arrays can be quite large to exhaust the entire stack very soon. At the same time, the program execution is completed with an error. Therefore, we should describe arrays at a global level as static (static) or allocate memory for them dynamically (this is also done from the heap).

[Array characteristics](/en/book/basis/arrays/arrays_overview "Array characteristics")

[Using arrays](/en/book/basis/arrays/arrays_usage "Using arrays")