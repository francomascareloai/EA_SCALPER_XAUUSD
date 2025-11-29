---
title: "SQLite database"
url: "https://www.mql5.com/en/book/advanced/sqlite"
hierarchy: []
scraped_at: "2025-11-28 09:48:49"
---

# SQLite database

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Advanced language tools](/en/book/advanced "Advanced language tools")SQLite database

* [Principles of database operations in MQL5](/en/book/advanced/sqlite/sqlite_intro "Principles of database operations in MQL5")
* [SQL Basics](/en/book/advanced/sqlite/sqlite_overview "SQL Basics")
* [Structure of tables: data types and restrictions](/en/book/advanced/sqlite/sqlite_scheme_types "Structure of tables: data types and restrictions")
* [OOP (MQL5) and SQL integration: ORM concept](/en/book/advanced/sqlite/sqlite_orm "OOP (MQL5) and SQL integration: ORM concept")
* [Creating, opening, and closing databases](/en/book/advanced/sqlite/sqlite_db_create_open_close "Creating, opening, and closing databases")
* [Executing queries without MQL5 data binding](/en/book/advanced/sqlite/sqlite_simple_queries "Executing queries without MQL5 data binding")
* [Checking if a table exists in the database](/en/book/advanced/sqlite/sqlite_table_exists "Checking if a table exists in the database")
* [Preparing bound queries: DatabasePrepare](/en/book/advanced/sqlite/sqlite_prepare "Preparing bound queries: DatabasePrepare")
* [Deleting and resetting prepared queries](/en/book/advanced/sqlite/sqlite_reset "Deleting and resetting prepared queries")
* [Binding data to query parameters: DatabaseBind/Array](/en/book/advanced/sqlite/sqlite_bind "Binding data to query parameters: DatabaseBind/Array")
* [Executing prepared queries: DatabaseRead/Bind](/en/book/advanced/sqlite/sqlite_read "Executing prepared queries: DatabaseRead/Bind")
* [Reading fields separately: DatabaseColumn Functions](/en/book/advanced/sqlite/sqlite_columns "Reading fields separately: DatabaseColumn Functions")
* [Examples of CRUD operations in SQLite via ORM objects](/en/book/advanced/sqlite/sqlite_crud_examples "Examples of CRUD operations in SQLite via ORM objects")
* [Transactions](/en/book/advanced/sqlite/sqlite_transactions "Transactions")
* [Import and export of database tables](/en/book/advanced/sqlite/sqlite_export_import "Import and export of database tables")
* [Printing tables and SQL queries to logs](/en/book/advanced/sqlite/sqlite_print "Printing tables and SQL queries to logs")
* [Example of searching for a trading strategy using SQLite](/en/book/advanced/sqlite/sqlite_example_ts "Example of searching for a trading strategy using SQLite")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# SQLite database

MetaTrader 5 provides native support for the SQLite database. It is a light yet fully functional database management system (DBMS). Traditionally, such systems are focused on processing data tables, where records of the same type are stored with a common set of attributes, and different correspondences (links or relations) can be established between records of different types (i.e. tables), and therefore such databases are also called relational. We have already considered examples of such connections between structures of the [economic calendar](/en/book/advanced/calendar/calendar_overview), but the calendar database is stored inside the terminal, and the functions of this section will allow you to create arbitrary databases from MQL programs.

The specialization of the DBMS on these data structures allows you to optimize — speed up and simplify — many popular operations such as sorting, searching, filtering, summing up, or calculating other aggregate functions for large amounts of data.

However, there is another side to this: DBMS programming requires its own SQL (Structured Query Language), and knowledge of pure MQL5 will not be enough. Unlike MQL5, which refers to imperative languages (those using operators indicating what, how, in what sequence to do), SQL is declarative, that is, it describes the initial data and the desired result, without specifying how and in what sequence to perform calculations. The meaning of the algorithm in SQL is described in the form of SQL queries. A query is an analog of a separate MQL5 operator, formed as a string using a special syntax.

Instead of programming complex loops and comparisons, we can simply call SQLite functions (for example, [DatabaseExecute](/en/book/advanced/sqlite/sqlite_simple_queries) or [Database Prepare](/en/book/advanced/sqlite/sqlite_prepare)) by passing SQL queries to them. To get query results into a ready-made MQL5 structure, you can use the [DatabaseReadBind](/en/book/advanced/sqlite/sqlite_read) function. This will allow you to read all the fields of the record (structure) at once in one call.

With the help of database functions, it is easy to create tables, add records to them, make modifications, and make selections according to complex conditions, for example, for tasks such as:

* Obtaining trading history and quotes
* Saving optimization and testing results
* Preparing and exchanging data with other analysis packages
* Analyzing economic calendar data
* Storing settings and states of MQL5 programs

In addition, a wide range of common, statistical, and mathematical functions can be used in SQL queries. Moreover, expressions with their participation can be calculated even without creating a table.

SQLite does not require a separate application, configuration, and administration, is not resource-demanding, and supports most commands of the popular SQL92 standard. An added convenience is that the entire database resides in a single file on the hard drive on the user's computer and can be easily transferred or backed up. However, to speed up read, write, and modification operations, the database can also be opened/created in RAM with the flag [DATABASE\_OPEN\_MEMORY](/en/book/advanced/sqlite/sqlite_db_create_open_close), however, in this case, such a database will be available only to this particular program and cannot be used for joint work of several programs.

It is important to note that the relative simplicity of SQLite, compared to full-featured DBMSs, comes with some limitations. In particular, SQLite does not have a dedicated process (system service or application) that would provide centralized access to the database and table management API, which is why parallel, shared access to the same database (file) from different processes is not guaranteed. So, if you need to simultaneously read and write to the database from optimization agents that execute instances of the same Expert Advisor, you will need to write code in it to synchronize access (otherwise, the data being written and read will be in an inconsistent state: after all, the order of writing, modifying, deleting, and reading from concurrent unsynchronized processes are random). Moreover, attempts to change the database at the same time may result in the MQL program receiving "database busy" errors (and the requested operation is not performed). The only scenario that does not require synchronization of parallel operations with SQLite is when only read operations are involved.

We will present only the basics of SQL to the extent necessary to start applying it. A complete description of the syntax and how SQL works is beyond the scope of this book. Check out the documentation on the SQLite site. However, please note that MQL5 and MetaEditor support a limited subset of commands and [SQL syntax constructions](/en/book/advanced/sqlite/sqlite_overview).

MQL Wizard in MetaEditor has an embedded option to create a database, which immediately offers to create the first table by defining a list of its fields. Also, the Navigator provides a separate tab for working with databases.

Using the Wizard or the context menu of the Navigator, you can create an empty database (a file on disk, placed by default, in the directory MQL5/Files) of supported formats (\*.db, \*.sql, \*.sqlite and others). In addition, in the context menu, you can import the entire database from an sql file or individual tables from csv files.

An existing or created database can be easily opened through the same menu. After that, its tables will appear in the Navigator, and the right, main area of the window will display a panel with tools for debugging SQL queries and a table with the results. For example, double-clicking on a table name performs a quick query of all record fields, which corresponds to the "SELECT \* FROM 'table'" statement that appears in the input field at the top.

![Viewing SQLite Database in MetaEditor](/en/book/img/sqlview.png "Viewing SQLite Database in MetaEditor")

Viewing SQLite Database in MetaEditor

You can edit the request and click the Execute button to activate it. Potential SQL syntax errors are output in the log.

For further details about the Wizard, the import/export of databases, and the interactive work with them, please see [MetaEditor documentation](https://www.metatrader5.com/en/metaeditor/help/database "MetaEditor documentation").

[Reading and writing data over a secure socket connection](/en/book/advanced/network/network_socket_tls_send_read "Reading and writing data over a secure socket connection")

[Principles of database operations in MQL5](/en/book/advanced/sqlite/sqlite_intro "Principles of database operations in MQL5")