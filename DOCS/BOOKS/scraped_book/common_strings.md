---
title: "Working with strings and symbols"
url: "https://www.mql5.com/en/book/common/strings"
hierarchy: []
scraped_at: "2025-11-28 09:49:08"
---

# Working with strings and symbols

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Working with strings and symbols

* [Initialization and measurement of strings](/en/book/common/strings/strings_init "Initialization and measurement of strings")
* [String concatenation](/en/book/common/strings/strings_concatenation "String concatenation")
* [String comparison](/en/book/common/strings/strings_comparison "String comparison")
* [Changing the character case and trimming spaces](/en/book/common/strings/strings_case_trim "Changing the character case and trimming spaces")
* [Finding, replacing, and extracting string fragments](/en/book/common/strings/strings_find_replace_split "Finding, replacing, and extracting string fragments")
* [Working with symbols and code pages](/en/book/common/strings/strings_codepages "Working with symbols and code pages")
* [Universal formatted data output to a string](/en/book/common/strings/strings_format "Universal formatted data output to a string")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Working with strings and symbols

Although computers take their name from the verb "compute", they are equally successful in processing not only numbers but also any unstructured information, the most famous example of which is text. In MQL programs, text is also used everywhere, from the names of the programs themselves to comments in trade orders. To work with the text in MQL5, there is a built-in [string type](/en/book/basis/builtin_types/strings), which allows you to operate on character sequences of arbitrary length.

To perform typical actions with strings, the MQL5 API provides a wide range of functions that can be conditionally divided into groups according to their purpose, such as string initialization, their addition, searching and replacing fragments within strings, converting strings to character arrays, accessing individual characters, as well as formatting.

Most of the functions in this chapter return an indication of the execution status: success or error. For functions with result type bool, true is usually a success, and false is an error. For functions with result type int a value of 0 or -1 can be considered an error: this is stated in the description of each function. In all these cases, the developer can find out the essence of the problem. To do this, call the [GetLastError](/en/book/common/environment/env_last_error) function and get the specific error code: a list of all codes with explanations is available in the documentation. It's important to call GetLastError immediately after receiving the error flag because calling each following instruction in the algorithm can lead to another error.

[Type complex](/en/book/common/conversions/conversions_complex "Type complex")

[Initialization and measurement of strings](/en/book/common/strings/strings_init "Initialization and measurement of strings")