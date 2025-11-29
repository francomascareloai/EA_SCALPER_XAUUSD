---
title: "Working with databases"
url: "https://www.mql5.com/en/docs/database"
hierarchy: []
scraped_at: "2025-11-28 09:31:20"
---

# Working with databases

[MQL5 Reference](/en/docs "MQL5 Reference")Working with databases

* [DatabaseOpen](/en/docs/database/databaseopen "DatabaseOpen")
* [DatabaseClose](/en/docs/database/databaseclose "DatabaseClose")
* [DatabaseImport](/en/docs/database/databaseimport "DatabaseImport")
* [DatabaseExport](/en/docs/database/databaseexport "DatabaseExport")
* [DatabasePrint](/en/docs/database/databaseprint "DatabasePrint")
* [DatabaseTableExists](/en/docs/database/databasetableexists "DatabaseTableExists")
* [DatabaseExecute](/en/docs/database/databaseexecute "DatabaseExecute")
* [DatabasePrepare](/en/docs/database/databaseprepare "DatabasePrepare")
* [DatabaseReset](/en/docs/database/databasereset "DatabaseReset")
* [DatabaseBind](/en/docs/database/databasebind "DatabaseBind")
* [DatabaseBindArray](/en/docs/database/databasebindarray "DatabaseBindArray")
* [DatabaseRead](/en/docs/database/databaseread "DatabaseRead")
* [DatabaseReadBind](/en/docs/database/databasereadbind "DatabaseReadBind")
* [DatabaseFinalize](/en/docs/database/databasefinalize "DatabaseFinalize")
* [DatabaseTransactionBegin](/en/docs/database/databasetransactionbegin "DatabaseTransactionBegin")
* [DatabaseTransactionCommit](/en/docs/database/databasetransactioncommit "DatabaseTransactionCommit")
* [DatabaseTransactionRollback](/en/docs/database/databasetransactionrollback "DatabaseTransactionRollback")
* [DatabaseColumnsCount](/en/docs/database/databasecolumnscount "DatabaseColumnsCount")
* [DatabaseColumnName](/en/docs/database/databasecolumnname "DatabaseColumnName")
* [DatabaseColumnType](/en/docs/database/databasecolumntype "DatabaseColumnType")
* [DatabaseColumnSize](/en/docs/database/databasecolumnsize "DatabaseColumnSize")
* [DatabaseColumnText](/en/docs/database/databasecolumntext "DatabaseColumnText")
* [DatabaseColumnInteger](/en/docs/database/databasecolumninteger "DatabaseColumnInteger")
* [DatabaseColumnLong](/en/docs/database/databasecolumnlong "DatabaseColumnLong")
* [DatabaseColumnDouble](/en/docs/database/databasecolumndouble "DatabaseColumnDouble")
* [DatabaseColumnBlob](/en/docs/database/databasecolumnblob "DatabaseColumnBlob")

# Working with databases

The functions for working with databases apply the popular and easy-to-use [SQLite](https://www.sqlite.org/index.html) engine. The convenient feature of this engine is that the entire database is located in a single file on a user PC's hard disk.

The functions allow for convenient creation of tables, adding data to them, performing modifications and sampling using simple SQL requests:

* receiving trading history and quotes from any formats,
* saving optimization and test results,
* preparing and exchanging data with other analysis packages,
* storing MQL5 application settings and status.

Queries allow using [statistical](/en/docs/database#math) and [mathematical](/en/docs/database#stats) functions.

The functions for working with databases allow you to replace the most repetitive large data array handling operations with SQL requests, so that it is often possible to use the [DatabaseExecute](/en/docs/database/databaseexecute)/[DatabasePrepare](/en/docs/database/databaseprepare) calls instead of programming complex loops and comparisons. Use the [DatabaseReadBind](/en/docs/database/databasereadbind) function to conveniently obtain query results in a ready-made structure. The function allows reading all record fields at once within a single call.

To accelerate reading, writing and modification, a database can be opened/created in RAM with the DATABASE\_OPEN\_MEMORY flag, although such a database is available only to a specific application and is not shared. When working with databases located on the hard disk, bulk data inserts/changes should be wrapped in transactions using [DatabaseTransactionBegin](/en/docs/database/databasetransactionbegin)/DatabaseTransactionCommit/DatabaseTransactionRollback. This accelerates the process hundreds of times.

To start working with the functions, read the article [SQLite: Native handling of SQL databases in MQL5](https://www.mql5.com/en/articles/7463).

| Function | Action |
| --- | --- |
| [DatabaseOpen](/en/docs/database/databaseopen) | Opens or creates a database in a specified file |
| [DatabaseClose](/en/docs/database/databaseclose) | Closes a database |
| [DatabaseImport](/en/docs/database/databaseimport) | Imports data from a file into a table |
| [DatabaseExport](/en/docs/database/databaseexport) | Exports a table or an SQL request execution result to a CSV file |
| [DatabasePrint](/en/docs/database/databaseprint) | Prints a table or an SQL request execution result in the Experts journal |
| [DatabaseTableExists](/en/docs/database/databasetableexists) | Checks the presence of the table in a database |
| [DatabaseExecute](/en/docs/database/databaseexecute) | Executes a request to a specified database |
| [DatabasePrepare](/en/docs/database/databaseprepare) | Creates a handle of a request, which can then be executed using DatabaseRead() |
| [DatabaseReset](/en/docs/database/databasereset) | Resets a request, like after calling [DatabasePrepare()](/en/docs/database/databaseprepare) |
| [DatabaseBind](/en/docs/database/databasebind) | Sets a parameter value in a request |
| [DatabaseBindArray](/en/docs/database/databasebindarray) | Sets an array as a parameter value |
| [DatabaseRead](/en/docs/database/databaseread) | Moves to the next entry as a result of a request |
| [DatabaseReadBind](/en/docs/database/databasereadbind) | Moves to the next record and reads data into the structure from it |
| [DatabaseFinalize](/en/docs/database/databasefinalize) | Removes a request created in DatabasePrepare() |
| [DatabaseTransactionBegin](/en/docs/database/databasetransactionbegin) | Starts transaction execution |
| [DatabaseTransactionCommit](/en/docs/database/databasetransactioncommit) | Completes transaction execution |
| [DatabaseTransactionRollback](/en/docs/database/databasetransactionrollback) | Rolls back transactions |
| [DatabaseColumnsCount](/en/docs/database/databasecolumnscount) | Gets the number of fields in a request |
| [DatabaseColumnName](/en/docs/database/databasecolumnname) | Gets a field name by index |
| [DatabaseColumnType](/en/docs/database/databasecolumntype) | Gets a field type by index |
| [DatabaseColumnSize](/en/docs/database/databasecolumnsize) | Gets a field size in bytes |
| [DatabaseColumnText](/en/docs/database/databasecolumntext) | Gets a field value as a string from the current record |
| [DatabaseColumnInteger](/en/docs/database/databasecolumninteger) | Gets the int type value from the current record |
| [DatabaseColumnLong](/en/docs/database/databasecolumnlong) | Gets the long type value from the current record |
| [DatabaseColumnDouble](/en/docs/database/databasecolumndouble) | Gets the double type value from the current record |
| [DatabaseColumnBlob](/en/docs/database/databasecolumnblob) | Gets a field value as an array from the current record |

Statistical functions:

* mode – [mode](https://en.wikipedia.org/wiki/Mode_(statistics))
* median – [median](https://en.wikipedia.org/wiki/Median) (50th percentile)
* percentile\_25 – 25th [percentile](https://en.wikipedia.org/wiki/Quantile)
* percentile\_75
* percentile\_90
* percentile\_95
* percentile\_99
* stddev or stddev\_samp — sample standard deviation
* stddev\_pop — population standard deviation
* variance or var\_samp — sample variance
* var\_pop — population variance

Mathematical functions

* [acos(X)](https://sqlite.org/lang_mathfunc.html#acos) – arccosine in radians
* [acosh(X)](https://sqlite.org/lang_mathfunc.html#acosh) – hyperbolic arccosine
* [asin(X)](https://sqlite.org/lang_mathfunc.html#asin) – arcsine in radians
* [asinh(X)](https://sqlite.org/lang_mathfunc.html#asinh) – hyperbolic arcsine
* [atan(X)](https://sqlite.org/lang_mathfunc.html#atan) – arctangent in radians
* [atan2(X,Y)](https://sqlite.org/lang_mathfunc.html#atan2) – arctangent in radians of the X/Y ratio
* [atanh(X)](https://sqlite.org/lang_mathfunc.html#atanh) – hyperbolic arctangent
* [ceil(X)](https://sqlite.org/lang_mathfunc.html#ceil) – rounding up to an integer
* [ceiling(X)](https://sqlite.org/lang_mathfunc.html#ceil) – rounding up to an integer
* [cos(X)](https://sqlite.org/lang_mathfunc.html#cos) – angle cosine in radians
* [cosh(X)](https://sqlite.org/lang_mathfunc.html#cosh) – hyperbolic cosine
* [degrees(X)](https://sqlite.org/lang_mathfunc.html#degrees) – convert radians into the angle
* [exp(X)](https://sqlite.org/lang_mathfunc.html#exp) – exponent
* [floor(X)](https://sqlite.org/lang_mathfunc.html#floor) – rounding down to an integer
* [ln(X)](https://sqlite.org/lang_mathfunc.html#ln) – natural logarithm
* [log(B,X)](https://sqlite.org/lang_mathfunc.html#log) – logarithm to the indicated base
* [log(X)](https://sqlite.org/lang_mathfunc.html#log) – decimal logarithm
* [log10(X)](https://sqlite.org/lang_mathfunc.html#log) – decimal logarithm
* [log2(X)](https://sqlite.org/lang_mathfunc.html#log2) – logarithm to base 2
* [mod(X,Y)](https://sqlite.org/lang_mathfunc.html#mod) – remainder of division
* [pi()](https://sqlite.org/lang_mathfunc.html#pi) – approximate Pi
* [pow(X,Y)](https://sqlite.org/lang_mathfunc.html#pow) – power by the indicated base
* [power(X,Y)](https://sqlite.org/lang_mathfunc.html#pow) – power by the indicated base
* [radians(X)](https://sqlite.org/lang_mathfunc.html#radians) – convert the angle into radians
* [sin(X)](https://sqlite.org/lang_mathfunc.html#sin) – angle sine in radians
* [sinh(X)](https://sqlite.org/lang_mathfunc.html#sinh) – hyperbolic sine
* [sqrt(X)](https://sqlite.org/lang_mathfunc.html#sqrt) – square root
* [tan(X)](https://sqlite.org/lang_mathfunc.html#tan) – angle tangent in radians
* [tanh(X)](https://sqlite.org/lang_mathfunc.html#tanh) – hyperbolic tangent
* [trunc(X)](https://sqlite.org/lang_mathfunc.html#trunc) – truncate to an integer closest to 0

Example:

| |
| --- |
| select   count(\*) as book\_count,   cast(avg(parent) as integer) as mean,   cast(median(parent) as integer) as median,   mode(parent) as mode,   percentile\_90(parent) as p90,   percentile\_95(parent) as p95,   percentile\_99(parent) as p99 from moz\_bookmarks; |

[CLExecutionStatus](/en/docs/opencl/clexecutionstatus "CLExecutionStatus")

[DatabaseOpen](/en/docs/database/databaseopen "DatabaseOpen")