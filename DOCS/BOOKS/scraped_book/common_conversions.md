---
title: "Built-in type conversions"
url: "https://www.mql5.com/en/book/common/conversions"
hierarchy: []
scraped_at: "2025-11-28 09:48:58"
---

# Built-in type conversions

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Built-in type conversions

* [Numbers to strings and vice versa](/en/book/common/conversions/conversions_numbers "Numbers to strings and vice versa")
* [Normalization of doubles](/en/book/common/conversions/conversions_normalize "Normalization of doubles")
* [Date and time](/en/book/common/conversions/conversions_datetime "Date and time")
* [Color](/en/book/common/conversions/conversions_color "Color")
* [Structures](/en/book/common/conversions/conversions_structs "Structures")
* [Enumerations](/en/book/common/conversions/conversions_enums "Enumerations")
* [Type complex](/en/book/common/conversions/conversions_complex "Type complex")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Built-in type conversions

Programs often operate with different data types. We have already encountered mechanisms of explicit and implicit casting of built-in types in the [Types Casting](/en/book/basis/conversion) section. They provide universal conversion methods that are not always suitable, for one reason or another. The MQL5 API provides a set of conversion functions using which a programmer can manage data conversions from one type to another and configure conversion results.

Among the most frequently used functions are those which convert various types to strings or vice versa. Specifically, this includes conversions for numbers, dates and times, colors, structures, and enums. Some types have additional specific operations.

This section considers various data conversion methods, providing programmers with the necessary tools to work with a variety of data types in trading robots. It includes the following subsections:

[Numbers to strings and vice versa](/en/book/common/conversions/conversions_numbers):

* This subsection explores methods for converting numerical values to strings and vice versa. It covers important aspects such as number formatting and handling various number systems.

[Normalization of doubles](/en/book/common/conversions/conversions_normalize):

* Normalizing double numbers is an important aspect when working with financial data. This section discusses normalization methods, ways to avoid precision loss, and processing floating-point values.

[Date and time](/en/book/common/conversions/conversions_datetime):

* Conversion of date and time plays a key role in trading strategies. This subsection discusses methods for working with dates, time intervals, and special data types like datetime.

[Color](/en/book/common/conversions/conversions_color):

* In MQL5, colors are represented by a special data type. The subsection examines the conversion of color values, their representation and use in graphical elements of trading robots.

[Structures](/en/book/common/conversions/conversions_structs):

* Data conversion within structures is an important topic when dealing with complex structured data. We will see methods of interacting with structures and their elements.

[Enumerations](/en/book/common/conversions/conversions_enums):

* Enumerations provide named constants and enhance code readability. This subsection discusses how to convert enumeration values and effectively use them in a program.

[Type complex](/en/book/common/conversions/conversions_complex):

* The complex type is designed to work with complex numbers. This section considers methods for converting and using complex numbers.

 

We will study all such functions in this chapter.

[Common APIs](/en/book/common "Common APIs")

[Numbers to strings and vice versa](/en/book/common/conversions/conversions_numbers "Numbers to strings and vice versa")