---
title: "Formatting, indentation, and spaces"
url: "https://www.mql5.com/en/book/intro/b_formatting"
hierarchy: []
scraped_at: "2025-11-28 09:47:54"
---

# Formatting, indentation, and spaces

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Formatting, indentation, and spaces

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* [First program](/en/book/intro/first_program "First program")
* [Data types and values](/en/book/intro/types_and_values "Data types and values")
* [Variables and identifiers](/en/book/intro/variables_and_identifiers "Variables and identifiers")
* [Assignment and initialization, expressions and arrays](/en/book/intro/init_assign_express "Assignment and initialization, expressions and arrays")
* [Data input](/en/book/intro/data_input "Data input")
* [Error fixing and debugging](/en/book/intro/errors_debug "Error fixing and debugging")
* [Data output](/en/book/intro/a_data_output "Data output")
* Formatting, indentation, and spaces
* [Mini summary](/en/book/intro/c_summing_up "Mini summary")
* [Programming fundamentals](/en/book/basis "Programming fundamentals")
* [Object Oriented Programming](/en/book/oop "Object Oriented Programming")
* [Common APIs](/en/book/common "Common APIs")
* [Creating application programs](/en/book/applications "Creating application programs")
* [Trading automation](/en/book/automation "Trading automation")
* [Advanced language tools](/en/book/advanced "Advanced language tools")
* [Conclusion](/en/book/conclusion "Conclusion")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Formatting, indentation, and spaces

MQL5 is among the so-called free-form languages, such as C-like and many other languages. This means that placing service symbols, such as brackets or operators, and keywords may be random, provided that syntactic rules are followed. The syntax only limits the mutual sequence of those symbols and words, while the indentation size at each string start or the number of spaces between the elements of the statement have no meaning for the compiler. In any place in the text, where a space needs to be inserted to separate language elements from each other, such as a variable type keyword and a variable identifier, a larger number of spaces can be used. Moreover, instead of spaces, it is allowed to use other symbols that denote empty space, such as tabulation and line breaks.

If there is a separating symbol (we will learn more about them in Part 2) between some elements of the statement, such as a comma ',' between function parameters, then there is no need for using any spaces at all.

Changes in formatting the source code do not modify the executable code.

Basically, there are many non-free-form languages. In some of them, forming a code block, which is performed using brace matching in MQL5, is based on equal indents from the left edge.

Due to free formatting, MQL5 allows programmers to use multiple different techniques to form the source code in order to improve its readability, visibility, and easier internal navigation.

Let's consider some examples of how the source text of the Greeting function can be recorded from our script, without changing its intent.

Here is the most 'packed' version without any excessive spaces or line breaks (a line break denoted here with the symbol '\' is only added to comply with the restrictions on publishing source codes in this book).

| |
| --- |
| string Greeting(int hour){string messages[3]={"Good morning",\ "Good afternoon","Good evening"};return messages[hour%24/8];} |

Here is the version, in which excessive spaces and line breaks are inserted.

| |
| --- |
| string Greeting ( int hour )   {     string messages [ 3 ]             = {                 "Good morning" ,                 "Good afternoon" ,                 "Good evening"               } ;            return messages [ hour % 24 / 8 ] ;   } |

MetaEditor has a built-in code styler that allows automatically formatting the source code of the current file in compliance with one of the styles supported. A specific style can be selected in dialog Tools -> Settings -> Styler. A style is applied using Tools -> Styler command.

You should keep in mind that your spacing freedom is limited. In particular, you may not insert spaces into identifiers, keywords, or numbers. Otherwise, the compiler won't be able to recognize them. For example, if we insert just one space between digits 2 and 4 in the number 24, the compiler will return a bunch of errors trying to compile the script.

Here is a knowingly incorrectly modified string:

| |
| --- |
| return messages[hour % 2 4 / 8]; |

Here is the error log:

| |
| --- |
| 'GoodTime2.mq5'        GoodTime2.mq5        1        1 '4' - some operator expected        GoodTime2.mq5        19        28 '[' - unbalanced left parenthesis        GoodTime2.mq5        19        18 '8' - some operator expected        GoodTime2.mq5        19        32 ']' - semicolon expected        GoodTime2.mq5        19        33 ']' - unexpected token        GoodTime2.mq5        19        33 5 errors, 0 warnings                6        1 |

Compiler messages may not always appear clear. It should be considered that, even upon the very first (in succession) error, there is a high probability that the internal representation of the program (as the compiler perceived it in 'mid-sentence') differs considerably from what the programmer has suggested. In particular, in this case, only the first and the second errors contain the key to understanding the problem, while all other ones are propagated.

According to the first error, the compiler expected to find the symbol of an operation between 2 and 4 (as it perceives 2 and 4 as two different numbers and not as 24 separated by a space). Alternative logic consists in the fact that a closing square bracket is omitted here, and the compiler displayed the second error: "'[' - unbalanced left parenthesis." After that running through the expression gets completely shattered, due to which the subsequent number 8 and closing bracket ']' appear inappropriate to the compiler. But in fact, if we just delete the excessive space between 2 and 4, the situation will become normal.

It is, of course, much easier to perform such an error analysis where we have intentionally added the issue. We do not always understand in practice how to remedy one situation or another. Even in the case above, supposing that you have received this broken code from another programmer and the array elements do not contain such trivial information, another correction option is easy to suspect: Either 2 or 4 must be left, because the author has probably desired to replace one number with another and not cleaned the 'footprints'.

[Data output](/en/book/intro/a_data_output "Data output")

[Mini summary](/en/book/intro/c_summing_up "Mini summary")