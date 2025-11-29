---
title: "Error fixing and debugging"
url: "https://www.mql5.com/en/book/intro/errors_debug"
hierarchy: []
scraped_at: "2025-11-28 10:15:30"
---

# Error fixing and debugging

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Error fixing and debugging

* [Introduction to MQL5 and development environment](/en/book/intro "Introduction to MQL5 and development environment")
* [Editing, compiling, and running programs](/en/book/intro/edit_compile_run "Editing, compiling, and running programs")
* [MQL Wizard and program draft](/en/book/intro/mql_wizard "MQL Wizard and program draft")
* [Statements, code blocks, and functions](/en/book/intro/statement_blocks "Statements, code blocks, and functions")
* [First program](/en/book/intro/first_program "First program")
* [Data types and values](/en/book/intro/types_and_values "Data types and values")
* [Variables and identifiers](/en/book/intro/variables_and_identifiers "Variables and identifiers")
* [Assignment and initialization, expressions and arrays](/en/book/intro/init_assign_express "Assignment and initialization, expressions and arrays")
* [Data input](/en/book/intro/data_input "Data input")
* Error fixing and debugging
* [Data output](/en/book/intro/a_data_output "Data output")
* [Formatting, indentation, and spaces](/en/book/intro/b_formatting "Formatting, indentation, and spaces")
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

# Error fixing and debugging

Programming art relies on the ability to instruct the program what and how it must do and also to protect it against potentially doing something wrong. The latter one is unfortunately much more difficult to execute due to multiple not very obvious factors affecting the program behavior. Incorrect data, insufficient resources, somebody else's and one's own coding errors are just to name some of the problems.

Nobody is insured against errors in coding programs. Errors may occur at different stages and are conveniently divided into:

* Compilation errors returned by the compiler when identifying a source code that does not meet the required syntax (we have already learned about such errors above); it is easiest to fix them because the compiler searches for them;
* Program runtime errors returned by the terminal, if an incorrect condition occurs in the program, such as division by zero, computing the square root of a negative, or an attempt to refer to a non-existing element of the array, as in our case; they are more difficult to detect since they usually occur not at any values of input parameters, but only in specific conditions;
* Program designing errors that lead to its complete shutdown without any tips from the terminal, such as sticking at an infinite loop; such errors may turn out to be the most complex in terms of locating and reproducing them, while the reproducibility of a problem in the program is a necessary condition for fixing it afterward; and
* Hidden errors, where the program seems to work smoothly, but the result provided is not correct; it is easy to detect if 2\*2 is not 4, while it is much more difficult to notice the discrepancies.

But let's get back to the specific situation with our script. According to the error message provided to us by the MQL program runtime environment, the following statement is wrong:

| |
| --- |
| return messages[hour / 8] |

In computing the index of an element from the array, depending on the value of the hour variable, a value may be obtained that goes beyond the array size of three.

The debugger embedded in MetaEditor allows making sure that it really happens. All its commands are collected in the Debug menu. They provide many useful functions. Here we are going to only settle on two: Debut -> Start on Real Data (F5) and Debug -> Start on History Data (Ctrl+F5). You can read about the other ones in the MetaEditor Help.

Both commands compile the program in a special manner — with the debugging information. Such a version of the program is not optimized as in standard compilation (more details on optimization, please see Documentation), while at the same time, it allows using the debugging information to 'look inside' the program during execution: See the states of variables and function call stacks.

The difference between debugging on real data and on history data consists in starting the program on an online chart with the former one and on the tester chart in a visual mode with the latter one. To instruct the editor on what exactly chart and with which settings to use, i.e., symbol, timeframe, date range, etc., you should preliminarily open the dialog Settings -> Debug and fill out the required fields in it. Option Use specified settings must be enabled. If it is disabled, the first symbol from the Market Watch and timeframe H1 will be used in online debugging, while tester settings are used when debugging on history data.

Please note that only indicators and Expert Advisors can be debugged in the tester. Only online debugging is available to scripts.

Let's run our script using F5 and enter 100 in parameter GreetingHour to reproduce the above problem situation. The script will start executing, and the terminal will practically immediately display an error message and request for opening the debugger.

| |
| --- |
| Critical error while running script 'GoodTime1 (EURUSD,H1)'. Array out of range. Continue in debugger? |

Having responded in the affirmative, we will get into MetaEditor where the current string is highlighted in the source code, in which the error has occurred (please give a notice of the green arrow in the left field).

![MetaEditor in the debugging mode in case of an error](/en/book/img/me_debug_en.png "MetaEditor in the debugging mode in case of an error")

MetaEditor in the debugging mode in case of an error

The current call stack is displayed in the lower left window part: All functions are listed in it (in bottom-up order), which had been called before the code execution stopped at the current string. In particular, in our script, the OnStart function was called (by the terminal itself), and the Greeting function was called from it (we called it from our code). An overview panel is in the lower right part of the window. Names of variables can be entered into it, or the entire expressions into the Expression column, and watch their values in the Values columns in the same string.

For instance, we can use the Add command of the context menu or double-click with the mouse on the first free string to enter the expression "hour / 8" and make sure that it is equal to 12.

Since debugging stopped resulting from an error, there is no sense to continue the program; therefore we can execute the Debug -> Stop command (Shift+F5).

In more complex cases of a not so obvious problem source, the debugger allows the string-by-string monitoring of the sequence of executing the statements and the contents of variables.

To solve the problem, it is necessary to ensure that, in the code, the element index always falls within the range of 0-2, i.e., complies with the array size. Strictly speaking, we should have written some additional statements checking the data entered for correctness (in our case, GreetingHour can only take a value within the range of 0-23), and then either display a tip or fix it automatically in case of violation of the conditions.

Within this introductory project, we will not go beyond a simple correction: We will improve the expression that computes the element index so that its result always falls within the required range. For this purpose, let's learn about one more operator — the modulus operator that only works for integers. To denote this operation, we use symbol '%'. The result of the modulus operation is the remainder of the integer division of dividend by the divisor. For example:

| |
| --- |
| 11 % 5 = 1 |

Here, with the integer division of 11 by 5, we would obtain 2, which corresponds with the largest factor of 5 within 11, which is 10. The remainder between 11 and 10 is exactly 1.

To fix the error in function Greeting, suffice to preliminarily perform the modulus division of hour by 24, which will ensure that the hour number will range within 0-23. Function Greeting will look as follows:

| |
| --- |
| string Greeting(int hour) {   string messages[3] = {"Good morning", "Good afternoon", "Good evening"};   return messages[hour % 24 / 8]; } |

Although this correction will surely work well (we are going to check it in a minute), it does not concern another problem that is left beyond our focus. The matter is that the GreetingHour parameter is of the int type, i.e., it can take both positive and negative values. If we tried to enter -‌8, for instance, or a 'more negative' number, then we would get the same runtime error, i.e., going beyond the array; just, in this case, the index does not exceed the highest value (array size) but becomes smaller than the lowest one (particularly, -8 leads to referring to the -1st element, interestingly, the values from -7 to -1 being displayed onto the 0th element and do not cause any error).

To fix this problem, we will replace the type of parameter GreetingHour with the unsigned integer: We will use uint instead of int (we will tell about all available types in part two, and here it is uint that we need). Guided by the limit for the non-negativity of values, built in at the compiler level for uint, MQL5 will independently ensure that neither the user (in the properties dialog) nor the program (in its computation) "goes negative."

Let's save the new version of the script as GoodTime2, compile, and launch it. We enter the value of 100 for the GreetingHour parameter and make sure that, this time, the script is executed without any errors, while the greeting "Good morning" is printed in the terminal log. It is the expected (correct) behavior since we can use a calculator and check that the remainder of the modulus division of 100 by 24 gives 4, while the integer division of 4 by 8 is 0, which means morning, in our case. From the user's point of view, of course, this behavior can be considered as unexpected. However, entering 100 as the hour number was also an unexpected user action. The user probably thought that our program would go down. But this did not happen, and this is a good point. Of course, with real programs, the values entered must be validated and the user must be notified about bugs.

As an additional measure of preventing from entering a wrong number, we will also use a special MQL5 feature to give a more detailed and friendly name to the input parameter. For this purpose, we will use a comment after the input parameter description in the same string. For example, like this:

| |
| --- |
| input uint GreetingHour = 0; // Greeting Hour (0-23) |

Please note that we have written the words from the variable name separately in the comment (it is not an identifier in the code anymore, but a tip for the user in it). Moreover, we added the range of valid values in parentheses. When launching the script, the previous GreetingHour will appear in the dialog to enter the parameters as follows:

| |
| --- |
| Greeting Hour (0-23) |

Now we can be sure that, if 100 is entered as the hour, it is not our fault.

A careful reader may wonder why we have defined the Greeting function with the hour parameter and send GreetingHour into it if we could use the input parameter in it directly. Function, as a discrete logical fragment of a code, is formed for both dividing the program into visible and easy-to-understand parts and reusing them subsequently. Functions are usually called from several parts of the program or are part of a library that is connected to multiple different programs. Therefore, a properly written function must be independent of the external context and can be moved among programs.

For instance, if we need to transfer our function Greeting into another script, it will stop being compiled, since there won't be the GreetingHour parameter in it. It is not quite correct to require adding it, because the other script can compute the time in another manner. In other words, when writing a function, we should do our best to avoid unnecessary external dependencies. Instead, we should declare the function parameters that can be filled out with the calling code.

[Data input](/en/book/intro/data_input "Data input")

[Data output](/en/book/intro/a_data_output "Data output")