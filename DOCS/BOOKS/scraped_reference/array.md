---
title: "Group of Functions for Working with Arrays"
url: "https://www.mql5.com/en/docs/array"
hierarchy: []
scraped_at: "2025-11-28 09:31:16"
---

# Group of Functions for Working with Arrays

[MQL5 Reference](/en/docs "MQL5 Reference")Array Functions

* [ArrayBsearch](/en/docs/array/arraybsearch "ArrayBsearch")
* [ArrayCopy](/en/docs/array/arraycopy "ArrayCopy")
* [ArrayCompare](/en/docs/array/arraycompare "ArrayCompare")
* [ArrayFree](/en/docs/array/arrayfree "ArrayFree")
* [ArrayGetAsSeries](/en/docs/array/arraygetasseries "ArrayGetAsSeries")
* [ArrayInitialize](/en/docs/array/arrayinitialize "ArrayInitialize")
* [ArrayFill](/en/docs/array/arrayfill "ArrayFill")
* [ArrayIsDynamic](/en/docs/array/arrayisdynamic "ArrayIsDynamic")
* [ArrayIsSeries](/en/docs/array/arrayisseries "ArrayIsSeries")
* [ArrayMaximum](/en/docs/array/arraymaximum "ArrayMaximum")
* [ArrayMinimum](/en/docs/array/arrayminimum "ArrayMinimum")
* [ArrayPrint](/en/docs/array/arrayprint "ArrayPrint")
* [ArrayRange](/en/docs/array/arrayrange "ArrayRange")
* [ArrayResize](/en/docs/array/arrayresize "ArrayResize")
* [ArrayInsert](/en/docs/array/arrayinsert "ArrayInsert")
* [ArrayRemove](/en/docs/array/arrayremove "ArrayRemove")
* [ArrayReverse](/en/docs/array/array_reverse "ArrayReverse")
* [ArraySetAsSeries](/en/docs/array/arraysetasseries "ArraySetAsSeries")
* [ArraySize](/en/docs/array/arraysize "ArraySize")
* [ArraySort](/en/docs/array/arraysort "ArraySort")
* [ArraySwap](/en/docs/array/arrayswap "ArraySwap")
* [ArrayToFP16](/en/docs/array/arraytofp16 "ArrayToFP16")
* [ArrayToFP8](/en/docs/array/arraytofp8 "ArrayToFP8")
* [ArrayFromFP16](/en/docs/array/arrayfromfp16 "ArrayFromFP16")
* [ArrayFromFP8](/en/docs/array/arrayfromfp8 "ArrayFromFP8")

# Group of Functions for Working with Arrays

[Arrays](/en/docs/basis/variables#array_define) are allowed to be maximum four-dimensional. Each dimension is indexed from 0 to dimension\_size-1. In a particular case of a one-dimensional array of 50 elements, calling of the first element will appear as array[0], of the last one - as array[49].

| Function | Action |
| --- | --- |
| [ArrayBsearch](/en/docs/array/arraybsearch) | Returns index of the first found element in the first array dimension |
| [ArrayCopy](/en/docs/array/arraycopy) | Copies one array into another |
| [ArrayCompare](/en/docs/array/arraycompare) | Returns the result of comparing two arrays of [simple types](/en/docs/basis/types#base_types) or custom structures without [complex objects](/en/docs/basis/types#complex_types) |
| [ArrayFree](/en/docs/array/arrayfree) | Frees up buffer of any dynamic array and sets the size of the zero dimension in 0. |
| [ArrayGetAsSeries](/en/docs/array/arraygetasseries) | Checks direction of array indexing |
| [ArrayInitialize](/en/docs/array/arrayinitialize) | Sets all elements of a numeric array into a single value |
| [ArrayFill](/en/docs/array/arrayfill) | Fills an array with the specified value |
| [ArrayIsSeries](/en/docs/array/arrayisseries) | Checks whether an array is a timeseries |
| [ArrayIsDynamic](/en/docs/array/arrayisdynamic) | Checks whether an array is dynamic |
| [ArrayMaximum](/en/docs/array/arraymaximum) | Search for an element with the maximal value |
| [ArrayMinimum](/en/docs/array/arrayminimum) | Search for an element with the minimal value |
| [ArrayPrint](/en/docs/array/arrayprint) | Prints an array of a simple type or a simple structure into journal |
| [ArrayRange](/en/docs/array/arrayrange) | Returns the number of elements in the specified dimension of the array |
| [ArrayResize](/en/docs/array/arrayresize) | Sets the new size in the first dimension of the array |
| [ArrayInsert](/en/docs/array/arrayinsert) | Inserts the specified number of elements from a source array to a receiving one starting from a specified index |
| [ArrayRemove](/en/docs/array/arrayremove) | Removes the specified number of elements from the array starting with a specified index |
| [ArrayReverse](/en/docs/array/array_reverse) | Reverses the specified number of elements in the array starting with a specified index |
| [ArraySetAsSeries](/en/docs/array/arraysetasseries) | Sets the direction of array indexing |
| [ArraySize](/en/docs/array/arraysize) | Returns the number of elements in the array |
| [ArraySort](/en/docs/array/arraysort) | Sorting of numeric arrays by the first dimension |
| [ArraySwap](/en/docs/array/arrayswap) | Swaps the contents of two dynamic arrays of the same type |
| [ArrayToFP16](/en/docs/array/arraytofp16) | Copies an array of type float or double into an array of type [ushort](/en/docs/basis/types/integer/integertypes#ushort) with the given format |
| [ArrayToFP8](/en/docs/array/arraytofp8) | Copies an array of type float or double into an array of type [uchar](/en/docs/basis/types/integer/integertypes#uchar) with the given format |
| [ArrayFromFP16](/en/docs/array/arrayfromfp16) | Copies an array of type [ushort](/en/docs/basis/types/integer/integertypes#ushort) into an array of float or double type with the given format |
| [ArrayFromFP8](/en/docs/array/arrayfromfp8) | Copies an array of type [uchar](/en/docs/basis/types/integer/integertypes#uchar) into an array of float or double type with the given format |

[ZeroMemory](/en/docs/common/zeromemory "ZeroMemory")

[ArrayBsearch](/en/docs/array/arraybsearch "ArrayBsearch")