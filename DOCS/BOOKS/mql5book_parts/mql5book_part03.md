# MQL5 Book - Part 3 (Pages 401-600)

## Page 401

Part 4. Common APIs
401 
4.4 Mathematical functions
4.4.1 0 Random number generation
Many algorithms in trading require the generation of random numbers. MQL5 provides two functions
that initialize and then poll the pseudo-random integer generator.
To get a better "randomness", you can use the Alglib library available in MetaTrader 5 (see
MQL5/Include/Math/Alglib/alglib.mqh).
void MathSrand(int seed) ≡ void srand(int seed)
The function sets some initial state of the pseudo-random integer generator. It should be called once
before starting the algorithm. The random values themselves should be obtained using the sequential
call of the MathRand function.
By initializing a generator with the same seed value, you can get reproducible sequences of numbers.
The seed value is not the first random number obtained from MathRand. The generator maintains some
internal state, which at each moment of time (between calls to it for a new random number) is
characterized by an integer value which is available from the program as the built-in uint variable
_ RandomSeed. It is this initial state value that establishes the MathSrand call.
Generator operation on every call to MathRand is described by two formulas:
Xn = Tf(Xp)
R = Gf(Xn)
The Tf function is called transition. It calculates the new internal state of the Xn generator based on
the previous Xp state.
The Gf function generates another "random" value that the function MathRand will return, using a new
internal state for this.
In MQL5, these formulas are implemented as follows (pseudocode):
Tf: _RandomSeed = _RandomSeed * 214013 + 2531011
Gf: MathRand = (_RandomSeed >> 16) & 0x7FFF
It is recommended to pass the GetTickCount or TimeLocal function as the seed value.
int MathRand() ≡ int rand()
The function returns a pseudo-random integer in the range from 0 to 32767. The sequence of
generated numbers varies depending on the opening initialization done by calling MathSrand.
An example of working with the generator is given in the MathRand.mq5 file. It calculates statistics on
the distribution of generated numbers over a given number of subranges (baskets). Ideally, we should
get a uniform distribution.

---

## Page 402

Part 4. Common APIs
402
4.4 Mathematical functions
#define LIMIT 1000 // number of attempts (generated numbers)
#define STATS 10   // number of baskets
   
int stats[STATS] = {}; // calculation of statistics of hits in baskets
   
void OnStart()
{
   const int bucket = 32767 / STATS;
   // generator reset
   MathSrand((int)TimeLocal());
   // repeat the experiment in a loop
   for(int i = 0; i < LIMIT; ++i)
   {
      // getting a new random number and updating statistics
      stats[MathRand() / bucket]++;
   }
   ArrayPrint(stats);
}
An example of results for three runs (each time we will get a new sequence):
 96  93 117  76  98  88 104 124 113  91
110  81 106  88 103  90 105 102 106 109
 89  98  98 107 114  90 101 106  93 104
4.4.1 1  Endianness control in integers
Various information systems, at the hardware level, use different byte orders when representing
numbers in memory. Therefore, when integrating MQL programs with the "outside world", in particular,
when implementing network communication protocols or reading/writing files of common formats, it
may be necessary to change the byte order.
Windows computers apply little-endian (starting with the least significant byte), i.e., the lowest byte
comes first in the memory cell allocated for the variable, then follows the byte with higher bits, and so
on. The alternative big-endian (starting with the highest digit, the most significant byte) is widely used
on the Internet. In this case, the first byte in the memory cell is the byte with the high bits, and the
last byte is the low bit. It is this order that is similar to how we write numbers "from left to right" in
ordinary life. For example, the value 1 234 starts with 1  and stands for thousands, followed by a 2 for
hundreds, a 3 for tens, and the last 4 is just four (low order).
Let's see the default byte order in MQL5. To do this, we will use the script MathSwap.mq5.
It describes a concatenation pattern that allows you to convert an integer to an array of bytes:

---

## Page 403

Part 4. Common APIs
403
4.4 Mathematical functions
template<typename T>
union ByteOverlay
{
   T value;
   uchar bytes[sizeof(T)];
   ByteOverlay(const T v) : value(v) { }
   void operator=(const T v) { value = v; }
};
This code allows you to visually divide the number into bytes and enumerate them with indices from the
array.
In OnStart, we describe the uint variable with the value 0x1 2345678 (note that the digits are
hexadecimal; in such a notation they exactly correspond to byte boundaries: every 2 digits is a
separate byte). Let's convert the number to an array and output it to the log.
void OnStart()
{
   const uint ui = 0x12345678;
   ByteOverlay<uint> bo(ui);
   ArrayPrint(bo.bytes); // 120  86  52  18 <==> 0x78 0x56 0x34 0x12
   ...
The ArrayPrint function can't print numbers in hexadecimal, so we see their decimal representation, but
it's easy to convert them to base 1 6 and make sure they match the original bytes. Visually, they go in
reverse order: i.e., under the 0th index in the array is 0x78, and then 0x56, 0x34 and 0x1 2. Obviously,
this order starts with the least-significant byte (indeed, we are in the Windows environment).
Now let's get familiar with the function MathSwap, which MQL5 provides to change the byte order.
integer MathSwap(integer value)
The function returns an integer in which the byte order of the passed argument is reversed. The
function takes parameters of type ushort/uint/ulong (i.e. 2, 4, 8 bytes in size).
Let's try the function in action:
   const uint ui = 0x12345678;
   PrintFormat("%I32X -> %I32X", ui, MathSwap(ui));
   const ulong ul = 0x0123456789ABCDEF;
   PrintFormat("%I64X -> %I64X", ul, MathSwap(ul));
Here is the result:
   12345678 -> 78563412
   123456789ABCDEF -> EFCDAB8967452301
Let's try to log an array of bytes after converting the value 0x1 2345678 with MathSwap:
   bo = MathSwap(ui);    // put the result of MathSwap into ByteOverlay
   ArrayPrint(bo.bytes); //  18  52  86 120 <==> 0x12 0x34 0x56 0x78
In a byte with index 0, where it used to be 0x78, there is now 0x1 2, and in elements with other
numbers, the values are also exchanged.

---

## Page 404

Part 4. Common APIs
404
4.5 Working with files
4.5 Working with files
It is difficult to find a program that does not use data input-output. We already know that MQL
programs can receive settings via input variables and output information to the log as we used the
latter in almost all test scripts. But in most cases, this is not enough.
For example, quite a significant part of program customization includes amounts of data that cannot be
accommodated in the input parameters. A program may need to be integrated with some external
analytical tools, i.e., uploading market information in a standard or specialized format, processing and
then loading it into the terminal in a new form, in particular, as trading signals, a set of neural network
weights or decision tree coefficients. Furthermore, it can be convenient to maintain a separate log for
an MQL program.
The file subsystem provides the most universal opportunities for such tasks. The MQL5 API provides a
wide range of functions for working with files, including functions to create, delete, search, write, and
read the files. We will study all this in this chapter.
All file operations in MQL5 are limited to a special area on the disk, which is called a sandbox. This is
done for security reasons so that no MQL program can be used for malicious purposes and harm your
computer or operating system.
Advanced users can avoid this limitation using special measures, which we will discuss later. But this
should only be done in exceptional cases while observing precautions and accepting all
responsibility.
For each instance of the terminal installed on the computer, the sandbox root directory is located at
<terminal_ data_ folder>/MQL5/Files/. From the MetaEditor, you can open the data folder using the
command File -> Open Data Folder. If you have sufficient access rights on the computer, this directory
is usually the same place where the terminal is installed. If you do not have the required permissions,
the path will look like this:
X:/Users/<user_name>/AppData/Roaming/MetaQuotes/Terminal/<instance_id>/MQL5/Files/
Here X is a drive letter where the system is installed, <user_ name> is the Windows user login,
<instance_ id> is a unique identifier of the terminal instance. The Users folder also has an alias
"Documents and Settings".
Please note that in the case of a remote connection to a computer via RDP (Remote Desktop
Protocol), the terminal will always use the Roaming directory and its subdirectories even if you have
administrator rights.
Let's recall that the MQL5 folder in the data directory is the place where all MQL programs are stored:
both their source codes and compiled ex5 files. Each type of MQL program, including indicators, Expert
Advisors, scripts, and others, has a dedicated subfolder in the MQL5 folder. So the Files folder for
working files is next to them.
In addition to this individual sandbox of each copy of the terminal on the computer, there is a common,
shared sandbox for all terminals: they can communicate through it. The path to it runs through the
home folder of the Windows user and may differ depending on the version of the operating system. For
example, in standard installations of Windows 7, 8, and 1 0, it looks like this:
X:/Users/<user_name>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/
Again, the folder can be easily accessed through MetaTrader: run the command File -> Open Shared
Data Folder, and you will be inside the Common folder.

---

## Page 405

Part 4. Common APIs
405
4.5 Working with files
Some types of MQL programs (Expert Advisors and indicators) can be executed not only in the terminal
but also in the tester. When running in it, the shared sandbox remains accessible, and instead of a
single instance sandbox, a folder inside the test agent is used. As a rule, it looks like:
X:/<terminal_path>/Tester/Agent-IP-port/MQL5/Files/
This may not be visible in the MQL program itself, i.e., all file functions work in exactly the same way.
However, from the user's point of view, it may seem that there is some kind of problem. For example, if
the program saves the results of its work to a file, it will be deleted in the tester's agent folder after the
run is completed (as if the file had never been created). This routine approach is designed to prevent
potentially valuable data of one program from leaking into another program that can be tested on the
same agent some time later (especially since agents can be shared). Other technologies are provided
for transferring files to agents and returning results from agents to the terminal, which we will discuss
in the fifth Part of the book.
To get around the sandbox limitation, you can use Windows' ability to assign symbolic links to file
system objects. In our case, the connections (junction) are best suited for redirecting access to folders
on the local computer. They are created using the following command (meaning the Windows command
line):
mklink /J new_name existing_target
The parameter new_ name is the name of the new virtual folder that will point to the real folder
existing_ target.
To create connections to external folders outside the sandbox, it is recommended to create a
dedicated folder inside MQL5/Files, for example, Links. Then, having entered it, you can execute the
above command by selecting new_ name and substituting the real path outside the sandbox as
existing_ target. For example, the following command will create in the folder Links a new link named
Settings, which will provide access to the MQL5/Presets folder:
mklink /J Settings "..\..\Presets\"
The relative path "..\..\" assumes that the command is executed in the specified MQL5/Files/Links
folder. A combination of two dots ".." indicates the transition from the current folder to the parent.
Specified twice, this combination instructs to go up the path hierarchy twice. As a result, the target
folder (existing_ target) will be generated as MQL5/Presets. But in the existing_ target parameter, you
can also specify an absolute path.
You can delete symbolic links like regular files (but, of course, you should first make sure that it is the
folder with the arrow icon in its lower left corner that is being deleted, i.e. the link, and not the original
folder). It is recommended to do this immediately, as soon as you no longer need to go beyond the
sandbox. The fact is that the created virtual folders become available to all MQL programs, not just
yours, and it is not known how other people's programs can use the additional freedom.
Many sections of the chapter deal with file names. They act as file system element identifiers and have
similar rules, including some restrictions.
Please note that the file name cannot contain some characters that play special roles in the file system
('<', '>', '/', '\\', '"', ':', '| ', '* ', '?'), as well as any characters with codes from 0 to 31  inclusive.
The following file names are also reserved for special use in the operating system and cannot be used:
CON, PRN, AUX, NUL, COM1 , COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9, LPT1 , LPT2,
LPT3, LPT4, LPT5, LPT6, LPT7, LPT8, LPT9.

---

## Page 406

Part 4. Common APIs
406
4.5 Working with files
It should be noted that the Windows file system does not see the fundamental difference between
letters in different cases, so names like "Name", "NAME", and "name" refer to the same element.
Windows allows both backslashes '\\' and forward slashes '/' to be used as a separator character
between path components (subfolders and files). However, the backslash needs to be screened (that is,
actually written twice) in MQL5 strings, because the '\' character itself is special: it is used to
construct control character sequences, such as '\r', '\n', '\t' and others (see section Character types).
For example, the following paths are equivalent: "MQL5Book/file.txt" and "MQL5Book\\file.txt".
The dot character '.' serves as a separator between the name and the extension. If a file system
element has multiple dots in its identifier, then the extension is the fragment to the right of the
rightmost dot, and everything to the left of it is the name. The title (before the dot) or extension (after
the dot) can be empty. For example, the file name without an extension is "text", and the file without a
name (only with the extension) is ".txt".
The total length of the path and file name in Windows has limitations. At the same time, to manage
files in MQL5, it should be taken into account that the path to the sandbox will be added to their
path and name, i.e., even less space will be allocated for the names of file objects in MQL function
calls. By default, the overall length limit is the system constant MAX_PATH, which is equal 260.
Starting from Windows 1 0 (build 1 607), you can increase this limit to 32767. To do this, you need
to save the following text in a .reg file and run it by adding it to the Windows Registry.
[HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem]
"LongPathsEnabled"=dword:00000001
For other versions of Windows, you can use workarounds from the command line. In particular, you
can shorten the path using the connections discussed above (by creating a virtual folder with a
short path). You can also use the shell command -subst, For example, subst z: c:\ very\ long\ path
(see Windows Help for details).
4.5.1  Information storage methods: text and binary
We have already seen in many previous sections that the same information can be represented in
textual and binary forms. For example, numbers of int, long, and double formats, date and time
(datetime) and colors (color) are stored in memory as a sequence of bytes of a certain length. This
method is compact and is better for computer interpretation, but it is more convenient for a human to
analyze information in a text form, although it takes longer. Therefore, we paid much attention to
converting numbers to strings and vice versa, and to functions for working with strings.
At the file level, the division into the binary and textual representation of data is also preserved. A
binary file is designed to store data in the same internal representation that is used in memory. The
text file contains a string representation.
Text files are commonly used for standard formats such as CSV (Comma Separated Values), JSON
(JavaScript Object Notation), XML (Extensible Markup Language), HTML (HyperText Markup
Language).
Binary files, of course, also have standard formats for many applications, in particular for images (PNG,
GIF, JPG, BMP), sounds (WAV, MP3), or compressed archives (ZIP). However, the binary format
initially assumes greater protection and low-level work with data, and therefore is more often used to
solve internal problems, when only storage efficiency and data availability for a specific program are
important. In other words, objects of any applied structures and classes can easily save and restore

---

## Page 407

Part 4. Common APIs
407
4.5 Working with files
their state in a binary file, actually making a memory impression and not worrying about compatibility
with any standard.
In theory, we could manually convert the data to strings when writing to a binary file and then convert
it back from strings to numbers (or structures, or arrays) when reading the file. This would be similar to
what the text file mode automatically provides but would require additional effort. The text file mode
saves us from such a routine. Moreover, the MQL5 file subsystem implicitly performs several optional
but important operations that are necessary when working with text.
First, the concept of text is based on some general rules of using delimiter characters. In particular, it
is assumed that all texts consist of strings. This way it is more convenient to read and analyze them
algorithmically. Therefore, there are special characters that separate one string from another.
Here we are faced with the first difficulties associated with the fact that different operating systems
accept different combinations of these characters. In Windows, the line separator is the sequence of
two characters '\r\n' (either as hexadecimal codes: 0xD 0xA, or as the name CRLF, which stands for
Carriage Return and Line Feed). In Unix and Linux, the single character '\n' is the standard, but some
versions and programs under MacOS may use the single character '\r'.
Although MetaTrader 5 runs under Windows, we have no guarantee that any resulting text file will not
be saved with unusual separators. If we were to read it in binary mode and check for delimiters
ourselves to form strings, these discrepancies would require specific handling. Here the text mode of
file operation in MQL5 comes to the rescue: it automatically normalizes line breaks when reading and
writing.
MQL5 might not fix line breaks for all cases. In particular, a single character '\r' will not be
interpreted as '\r\n' when reading a text file, while a single '\n' is correctly interpreted as '\r\n'.
Secondly, strings can be stored in memory in multiple representations. By default, string (type string)
in MQL5 consists of two-byte characters. This provides support for the universal Unicode encoding,
which is nice because it includes all national scripts. However, in many cases, such universality is not
required (for example, when storing numbers or messages in English), in which case it is more efficient
to use strings of single-byte characters in the ANSI encoding. The MQL5 API functions allow you to
choose the preferred way of writing strings in text mode into files. But if we control writing in our MQL
program, we can guarantee the validity and reliability of switching from Unicode to single-byte
characters. In this case, when integrating with some external software or web service, the ANSI code
page in its files can be any. In this regard, the following point arises.
Thirdly, due to the presence of many different languages, you need to be prepared for texts in various
ANSI encodings. Without the correct interpretation of the encoding, the text can be written or read
with distortions, or even become unreadable. We saw it in the section Working with symbols and code
pages. This is why file functions already include means for correct character processing: it is enough to
specify the desired or expected encoding in the parameters. The choice of encoding is described in
more detail in a separate section.
And finally, the text mode has built-in support for the well-known CSV format. Since trading often
requires tabular data, CSV is well suited for this. In a text file in CSV mode, the MQL5 API functions
process not only delimiters for wrapping lines of text but also an additional delimiter for the border of
columns (fields in each row of the table). This is usually a tab character '\t', a comma ',' or a
semicolon ';'. For example, here is what a CSV file with Forex news looks like ( a comma-separated
fragment is shown):

---

## Page 408

Part 4. Common APIs
408
4.5 Working with files
Title,Country,Date,Time,Impact,Forecast,Previous
Bank Holiday,JPY,08-09-2021,12:00am,Holiday,,
CPI y/y,CNY,08-09-2021,1:30am,Low,0.8%,1.1%
PPI y/y,CNY,08-09-2021,1:30am,Low,8.6%,8.8%
Unemployment Rate,CHF,08-09-2021,5:45am,Low,3.0%,3.1%
German Trade Balance,EUR,08-09-2021,6:00am,Low,13.9B,12.6B
Sentix Investor Confidence,EUR,08-09-2021,8:30am,Low,29.2,29.8
JOLTS Job Openings,USD,08-09-2021,2:00pm,Medium,9.27M,9.21M
FOMC Member Bostic Speaks,USD,08-09-2021,2:00pm,Medium,,
FOMC Member Barkin Speaks,USD,08-09-2021,4:00pm,Medium,,
BRC Retail Sales Monitor y/y,GBP,08-09-2021,11:01pm,Low,4.9%,6.7%
Current Account,JPY,08-09-2021,11:50pm,Low,1.71T,1.87T
And here it is, for clarity, in the form of a table:
Title
Country
Date
Time
Impact
Forecast
Previous
Bank Holiday
JPY
08-09-2021 
1 2:00am
Holiday
 
 
CPI y/y
CNY
08-09-2021 
1 :30am
Low
0.8%
1 .1 %
PPI y/y
CNY
08-09-2021 
1 :30am
Low
8.6%
8.8%
Unemployment Rate
CHF
08-09-2021 
5:45am
Low
3.0%
3.1 %
German Trade Balance
EUR
08-09-2021 
6:00am
Low
1 3.9B
1 2.6B
Sentix Investor Confidence
EUR
08-09-2021 
8:30am
Low
29.2
29.8
JOLTS Job Openings
USD
08-09-2021 
2:00pm
Medium
9.27M
9.21 M
FOMC Member Bostic Speaks
USD
08-09-2021 
2:00pm
Medium
 
 
FOMC Member Barkin Speaks
USD
08-09-2021 
4:00pm
Medium
 
 
BRC Retail Sales Monitor y/y
GBP
08-09-2021 
1 1 :01 pm
Low
4.9%
6.7%
Current Account
JPY
08-09-2021 
1 1 :50pm
Low
1 .71 T
1 .87T
4.5.2 Writing and reading files in simplified mode
Among the MQL5 file functions that are intended for writing and reading data, there is a division into 2
unequal groups. The first of these includes two functions: FileSave and FileLoad, which allow you to
write or read data in binary mode in a single function call. On the one hand, this approach has an
undeniable advantage, the simplicity, but on the other hand, it has some limitations (more on those
below). In the second large group, all file functions are used differently: it is required to call several of
them sequentially in order to perform a logically complete read or write operation. This seems more
complex, but it provides flexibility and control over the process. The functions of the second group
operate with special integers – file descriptors, which should be obtained using the FileOpen function
(see the next section).
Let's view the formal description of these two functions, and then consider their example
(FileSaveLoad.mq5).

---

## Page 409

Part 4. Common APIs
409
4.5 Working with files
bool FileSave(const string filename, const void &data[], const int flag = 0)
The function writes all elements of the passed data array to a binary file named filename. The filename
parameter may contain not only the file name but also the names of folders of several levels of nesting:
the function will create the specified folders if they do not already exist. If the file exists, it will be
overwritten (unless occupied by another program).
As the data parameter, an array of any built-in types can be passed, except for strings. It can also be
an array of simple structures containing fields of built-in types with the exception of strings, dynamic
arrays, and pointers. Classes are also not supported.
The flag parameter may, if necessary, contain the predefined constant FILE_COMMON, which means
creating and writing a file to the common data directory of all terminals (Common/Files/). If the flag is
not specified (which corresponds to the default value of 0), then the file is written to the regular data
directory (if the MQL program is running in the terminal) or to the testing agent directory (if it happens
in the tester). In the last two cases, the MQL5/Files/ sandbox is used inside the directory, as described
at the beginning of the chapter.
The function returns an indication of operation success (true) or error (false).
long FileLoad(const string filename, void &data[], const int flag = 0)
The function reads the entire contents of a binary file filename to the specified data array. The file
name may include a folder hierarchy within the MQL5/Files or Common/Files sandbox.
The data array must be of any built-in type except string, or a simple structure type (see above).
The flag parameter controls the selection of the directory where the file is searched and opened: by
default (with a value of 0) it is the standard sandbox, but if the value FILE_COMMON is set, then it is
the sandbox shared by all terminals.
The function returns the number of items read, or -1  on error.
Note that the data from the file is read in blocks of one array element. If the file size is not a multiple
of the element size, then the remaining data is skipped (not read). For example, if the file size is 1 0
bytes, reading it into an array of double type (sizeof(double)=8) will result in only 8 bytes actually being
loaded, i.e. 1  element (and the function will return 1 ). The remaining 2 bytes at the end of the file will
be ignored.
In the FileSaveLoad.mq5 script we define two structures for tests.

---

## Page 410

Part 4. Common APIs
41 0
4.5 Working with files
struct Pair
{
   short x, y;
};
  
struct Simple
{
   double d;
   int i;
   datetime t;
   color c;
   uchar a[10]; // fixed size array allowed
   bool b;
   Pair p;      // compound fields (nested simple structures) are also allowed
   
   // strings and dynamic arrays will cause a compilation error when used
   // FileSave/FileLoad: structures or classes containing objects are not allowed
   // string s;
   // uchar a[];
   
   // pointers are also not supported
   // void *ptr;
};
The Simple structure contains fields of most allowed types, as well as a composite field with the Pair
structure type. In the OnStart function, we fill in a small array of the Simple type.
void OnStart()
{
   Simple write[] =
   {
      {+1.0, -1, D'2021.01.01', clrBlue, {'a'}, true, {1000, 16000}},
      {-1.0, -2, D'2021.01.01', clrRed,  {'b'}, true, {1000, 16000}},
   };
   ...
We will select the file for writing data together with the MQL5Book subfolder so that our experiments do
not mix with your working files:
   const string filename = "MQL5Book/rawdata";
Let's write an array to a file, read it into another array, and compare them.
   PRT(FileSave(filename, write/*, FILE_COMMON*/)); // true
   
   Simple read[];
   PRT(FileLoad(filename, read/*, FILE_COMMON*/)); // 2
   
   PRT(ArrayCompare(write, read)); // 0
FileLoad returned 2, i.e., 2 elements (2 structures) were read. If the comparison result is 0, that
means that the data matched. You can open the folder in your favorite file manager
MQL5/Files/MQL5Book and make sure that there is the 'rawdata' file (it is not recommended to view its
contents using a text editor, we suggest using a viewer that supports binary mode).

---

## Page 411

Part 4. Common APIs
41 1 
4.5 Working with files
Further in the script, we convert the read array of structures into bytes and output them to the log in
the form of hexadecimal codes. This is a kind of memory dump, and it allows you to understand what
binary files are.
   uchar bytes[];
   for(int i = 0; i < ArraySize(read); ++i)
   {
      uchar temp[];
      PRT(StructToCharArray(read[i], temp));
      ArrayCopy(bytes, temp, ArraySize(bytes));
   }
   ByteArrayPrint(bytes);
Result:
 [00] 00 | 00 | 00 | 00 | 00 | 00 | F0 | 3F | FF | FF | FF | FF | 00 | 66 | EE | 5F | 
 [16] 00 | 00 | 00 | 00 | 00 | 00 | FF | 00 | 61 | 00 | 00 | 00 | 00 | 00 | 00 | 00 | 
 [32] 00 | 00 | 01 | E8 | 03 | 80 | 3E | 00 | 00 | 00 | 00 | 00 | 00 | F0 | BF | FE | 
 [48] FF | FF | FF | 00 | 66 | EE | 5F | 00 | 00 | 00 | 00 | FF | 00 | 00 | 00 | 62 | 
 [64] 00 | 00 | 00 | 00 | 00 | 00 | 00 | 00 | 00 | 01 | E8 | 03 | 80 | 3E | 
Because the built-in ArrayPrint function can't print in hexadecimal format, we had to develop our own
function ByteArrayPrint (here we will not give its source code, see the attached file).
Next, let's remember that FileLoad is able to load data into an array of any type, so we will read the
same file using it directly into an array of bytes.
   uchar bytes2[];
   PRT(FileLoad(filename, bytes2/*, FILE_COMMON*/)); // 78,  39 * 2
   PRT(ArrayCompare(bytes, bytes2)); // 0, equality
A successful comparison of two byte arrays shows that FileLoad can operate with raw data from the file
in an arbitrary way, in which it is instructed (there is no information in the file that it stores an array of
Simple structures).
It is important to note here that since the byte type has a minimum size (1 ), it is a multiple of any file
size. Therefore, any file is always read into a byte array without a remainder. Here the FileLoad
function has returned the number 78 (the number of elements is equal to the number of bytes). This is
the size of the file (two structures of 39 bytes each).
Basically, the ability of FileLoad to interpret data for any type requires care and checks on the part of
the programmer. In particular, further in the script, we read the same file into an array of structures
MqlDateTime. This, of course, is wrong, but it works without errors.
   MqlDateTime mdt[];
   PRT(sizeof(MqlDateTime)); // 32
   PRT(FileLoad(filename, mdt)); // 2
 // attention: 14 bytes left unread
   ArrayPrint(mdt);
The result contains a meaningless set of numbers:

---

## Page 412

Part 4. Common APIs
41 2
4.5 Working with files
        [year]      [mon] [day]     [hour]    [min]    [sec] [day_of_week] [day_of_year]
[0]          0 1072693248    -1 1609459200        0 16711680            97             0
[1] -402587648    4096003     0  -20975616 16777215  6286950     -16777216    1644167168
Because the size of MqlDateTime is 32, then only two such structures fit in a 78-byte file, and 1 4 more
bytes remain superfluous. The presence of a residue indicates a problem. But even if there is no
residue, this does not guarantee the meaningfulness of the operation performed, because two different
sizes can, purely by chance, fit an integer (but different) number of times in the length of the file.
Moreover, two structures that are different in meaning can have the same size, but this does not mean
that they should be written and read from one to the other.
Not surprisingly, the log of the array of structures MqlDateTime shows strange values, since it was, in
fact, a completely different data type.
To make reading somewhat more careful, the script implements an analog of the FileLoad function –
MyFileLoad. We will analyze this function in detail, as well as its pair MyFileSave, in the following
sections, when learning new file functions and using them to model the internal structure
FileSave/FileLoad. In the meantime, just note that in our version, we can check for the presence of an
unread remainder in the file and display a warning.
To conclude, let's look at a couple more potential errors demonstrated in the script.
   /*
  // compilation error, string type not supported here
   string texts[];
   FileSave("any", texts); // parameter conversion not allowed
   */
   
   double data[];
   PRT(FileLoad("any", data)); // -1
   PRT(_LastError); // 5004, ERR_CANNOT_OPEN_FILE
The first one happens at compile time (which is why the code block is commented out) because string
arrays are not allowed.
The second is to read a non-existent file, which is why FileLoad returns -1 . An explanatory error code
can be easily obtained using GetLastError (or _ LastError).
4.5.3 Opening and closing files
To write and read data from a file, most MQL5 functions require that the file be opened first. For this
purpose, there is the FileOpen function. After performing the required operations, the open file should
be closed using the FileClose function. The fact is that an open file may, depending on the applied
options, be blocked for access from other programs. In addition, file operations are buffered in memory
(cache) for performance reasons, and without closing the file, new data may not be physically uploaded
to it for some time. This is especially critical if the data being written is waiting for an external program
(for example, when integrating an MQL program with other systems). We learn about an alternative way
to flush the buffer to disk from the description of the FileFlush function.
A special integer referred to as the descriptor is associated with an open file in an MQL program. It is
returned by the FileOpen function. All operations related to accessing or modifying the internal contents
of a file require this identifier to be specified in the corresponding API functions. Those functions that

---

## Page 413

Part 4. Common APIs
41 3
4.5 Working with files
operate on the entire file (copy, delete, move, check for existence) do not require a descriptor. You do
not need to open the file to perform these steps.
i n t F i l eOp en ( co n s t s tr i n g  fi l en a m e, i n t fl a g s , co n s t s h o r t d el i m i ter  =  '\ t', u i n t co d ep a g e =  C P_AC P)
i n t F i l eOp en ( co n s t s tr i n g  fi l en a m e, i n t fl a g s , co n s t s tr i n g  d el i m i ter , u i n t co d ep a g e =  C P_AC P)
The function opens a file with the specified name, in the mode specified by the flags parameter. The
filename parameter may contain subfolders before the actual file name. In this case, if the file is
opened for writing and the required folder hierarchy does not yet exist, it will be created.
The flags parameter must contain a combination of constants describing the required mode of working
with the file. The combination is performed using the operations of bitwise OR. Below is a table of
available constants.
Identifier
Value
Description
FILE_READ
1
The file is opened for reading
FILE_WRITE
2
The file is opened for writing
FILE_BIN
4
Binary read-write mode, no data conversion from string to
string
FILE_CSV
8
File of CSV type; the data being written is converted to text of
the appropriate type (Unicode or ANSI, see below), and when
reading, the reverse conversion is performed from the text to
the required type (specified in the reading function); one CSV
record is a single line of text, delimited by newline characters
(usually CRLF); inside the CSV record, the elements are
separated by a delimiter character (parameter delimiter);
FILE_TXT
1 6
Plain text file, similar to CSV mode, but a delimiter character
is not used (the value of the parameter delimiter is ignored)
FILE_ANSI
32
ANSI type strings (single-byte characters)
FILE_UNICODE
64
Unicode type strings (double-byte characters)
FILE_SHARE_READ
1 28
Shared read access from several programs
FILE_SHARE_WRITE
256
Shared writing access by multiple programs
FILE_REWRITE
51 2
Permission to overwrite a file (if it already exists) in functions
FileCopy and FileMove
FILE_COMMON
4096
File location in the shared folder of all client
terminals /Terminal/Common/Files (the flag is used when
opening files (FileOpen), copying files (FileCopy, FileMove)
and checking the existence of files (FileIsExist))
When opening a file, one of the FILE_WRITE, FILE_READ flags or their combination must be specified.
The FILE_SHARE_READ and FILE_SHARE_WRITE flags do not replace or cancel the need to specify the
FILE_READ and FILE_WRITE flags.

---

## Page 414

Part 4. Common APIs
41 4
4.5 Working with files
The MQL program execution environment always buffers files for reading, which is equivalent to
implicitly adding the FILE_READ flag. Because of this, FILE_SHARE_READ should always be used to
work properly with shared files (even if another process is known to have a write-only file open).
If none of the FILE_CSV, FILE_BIN, FILE_TXT flags is specified, FILE_CSV is assumed as the highest
priority. If more than one of these three flags is specified, the highest priority passed is applied (they
are listed above in descending order of priority).
For text files, the default mode is FILE_UNICODE.
The delimiter parameter affecting only CSV, could be of type ushort or string. In the second case, if the
length of the string is greater than 1 , only its first character will be used.
The codepage parameter only affects files opened in text mode (FILE_TXT or FILE_CSV), and only if
FILE_ANSI mode is selected for strings. If the strings are stored in Unicode (FILE_UNICODE), the code
page is not important.
If successful, the function returns a file descriptor, a positive integer. It is unique only within a
particular MQL program; it makes no sense to share it with other programs. For further work with the
file, the descriptor is passed to calls to other functions.
On error, the result is INVALID_HANDLE (-1 ). The essence of the error should be clarified from the
code returned by the GetLastError function.
All operating mode settings made at the time the file is opened remain unchanged for as long as the file
is open. If it becomes necessary to change the mode, the file should be closed and reopened with the
new parameters.
For each open file, the MQL program execution environment maintains an internal pointer, i.e. the
current position within the file. Immediately after opening the file, the pointer is set to the beginning
(position 0). In the process of writing or reading, the position is shifted appropriately, according to the
amount of data transmitted or received from various file functions. It is also possible to directly
influence the position (move back or forward). All these opportunities will be discussed in the following
sections.
FILE_READ and FILE_WRITE in various combinations allow you to achieve several scenarios:
• FILE_READ – open a file only if it exists; otherwise, the function returns an error and no new file is
created.
• FILE_WRITE – creating a new file if it does not already exist, or opening an existing file, and its
contents are cleared and the size is reset to zero.
• FILE_READ| FILE_WRITE – open an existing file with all its contents or create a new file if it does
not already exist.
As you can see, some scenarios are inaccessible only due to flags. In particular, you cannot open a file
for writing only if it already exists. This can be achieved using additional functions, for example,
FileIsExist. Also, it will not be possible to "automatically" reset a file opened for a combination of
reading and writing: in this case, MQL5 always leaves the contents.
To append data to a file, one must not only open the file in FILE_READ| FILE_WRITE mode, but also
move the current position within the file to its end by calling FileSeek.
The correct description of the shared access to the file is a prerequisite for successful execution of File
Open. This aspect is managed as follows.

---

## Page 415

Part 4. Common APIs
41 5
4.5 Working with files
• If neither of the FILE_SHARE_READ and FILE_SHARE_WRITE flags is specified, then the current
program gets exclusive access to the file if it opens it first. If the same file has already been
opened by someone before (by another program,or by the same program), the function call will fail.
• When the FILE_SHARE_READ flag is set, the program allows subsequent requests to open the same
file for reading. If at the time of the function call the file is already open for reading by another or
the same program, and this flag is not set, the function will fail.
• When the FILE_SHARE_WRITE flag is set, the program allows subsequent requests to open the
same file for writing. If at the time of the function call the file is already open for writing by another
or the same program, and this flag is not set, the function will fail.
Access sharing is checked not only in relation to other MQL programs or processes external to
MetaTrader 5, but also in relation to the same MQL program if it reopens the file.
Thus, the least conflicting mode implies that both flags are specified, but it still does not guarantee that
the file will be opened if someone has already been issued a descriptor to it with no sharing. However,
more stringent rules should be followed depending on the planned reads or writes.
For example, when opening a file for reading, it makes sense to leave the opportunity for others to read
it. Additionally, you can probably allow others to write to it, if it is a file that is being replenished (for
example, a journal). However, when opening a file for writing, it is hardly worth leaving write access to
others: this would lead to unpredictable data overlay.  
void FileClose(int handle)
The function closes a previously opened file by its handle.
After the file is closed, its handle in the program becomes invalid: an attempt to call any file function on
it will result in an error. However, you can use the same variable to store a different handle if you
reopen the same or a different file.
When the program terminates, open files are forcibly closed, and the write buffer, if it is not empty, is
written to disk. However, it is recommended to close files explicitly.
Closing a file when you're finished working with it is an important rule to follow. This is due not only to
the caching of the information being written, which may remain in RAM for some time and not saved to
disk (as already mentioned above), if the file is not closed. In addition, an open file consumes some
internal resource of the operating system, and we are not talking about disk space. The number of
simultaneously open files is limited (maybe several hundred or thousands depending on Windows
settings). If many programs keep a large number of files open, this limit may be reached and attempts
to open new files will fail.
In this regard, it is desirable to protect yourself from the possible loss of descriptors using a wrapper
class that would open a file and receive a descriptor when creating an object, and the descriptor would
be released and the file closed automatically in the destructor.
We will create a wrapper class after testing the pure FileOpen and FileClose functions.
But before diving into file specifics, let's prepare a new version of the macro to illustrate an output of
our functions to the call log. The new version was required because, until now, macros like PRT and
PRTS (used in previous sections) "absorbed" function return values during printing. For example, we
wrote:

---

## Page 416

Part 4. Common APIs
41 6
4.5 Working with files
PRT(FileLoad(filename, read));
Here the result of the FileLoad call is sent to the log, but it is not possible to get it in the calling string
of code. To tell the truth, we did not need it. But now the FileOpen function will return a file descriptor,
and should be stored in a variable for further manipulation of the file.
There are two problems with the old macros. First, they are based on the function Print, which
consumes the passed data (sending it to the log) but does not itself return anything. Second, any value
for a variable with a result can only be obtained from an expression, and a Print call cannot be made a
part of an expression due to the fact that it has the type void.
To solve these problems, we need a print helper function that returns a printable value. And we will
pack its call into a new PRTF macro:
#include <MQL5Book/MqlError.mqh>
  
#define PRTF(A) ResultPrint(#A, (A))
  
template<typename T>
T ResultPrint(const string s, const T retval = 0)
{
   const string err = E2S(_LastError) + "(" + (string)_LastError + ")";
   Print(s, "=", retval, " / ", (_LastError == 0 ? "ok" : err));
 ResetLastError();// clear the error flag for the next call
   return retval;
}
Using the '#' magic string conversion operator, we get a detailed descriptor of the code fragment
(expression A) that is passed as the first argument to ResultPrint. The expression itself (the macro
argument) is evaluated (if there is a function, it is called), and its result is passed as the second
argument to ResultPrint. Next, the usual Print function comes into play, and finally, the same result is
returned to the calling code.
In order not to look into the Help for decoding error codes, an E2S macro was prepared that uses the
MQL_ERROR enumeration with all MQL5 errors. It can be found in the header file
MQL5/Include/MQL5Book/MqlError.mqh. The new macro and the ResultPrint function are defined in the
PRTF.mqh file, next to the test scripts.
In the FileOpenClose.mq5 script, let's try to open different files, and, in particular, the same file will
open several times in parallel. This is usually avoided in real programs. A single handle to a particular
file in a program instance is sufficient for most tasks.
One of the files, MQL5Book/rawdata, must already exist since it was created by a script from the
section Writing and reading files in simplified mode. Another file will be created during the test.
We will choose the file type FILE_BIN. working with FILE_TXT or FILE_CSV would be similar at this
stage.
Let's reserve an array for file descriptors so that at the end of the script we close all files at once.
First, let's open MQL5Book/rawdata in reading mode without access sharing. Assuming that the file is
not in use by any third party application, we can expect the handle to be successfully received.

---

## Page 417

Part 4. Common APIs
41 7
4.5 Working with files
void OnStart()
{
   int ha[4] = {}; // array for test file handles 
   
   // this file must exist after running FileSaveLoad.mq5
   const string rawdata = "MQL5Book/rawdata";
   ha[0] = PRTF(FileOpen(rawdata, FILE_BIN | FILE_READ)); // 1 / ok
If we try to open the same file again, we will encounter an error because neither the first nor the
second call allows sharing.
 ha[1] = PRTF(FileOpen(rawdata, FILE_BIN | FILE_READ)); // -1 / CANNOT_OPEN_FILE(5004)
Let's close the first handle, open the file again, but with shared read permissions, and make sure that
reopening now works (although it also needs to allow shared reading):
   FileClose(ha[0]);
   ha[0] = PRTF(FileOpen(rawdata, FILE_BIN | FILE_READ | FILE_SHARE_READ)); // 1 / ok
   ha[1] = PRTF(FileOpen(rawdata, FILE_BIN | FILE_READ | FILE_SHARE_READ)); // 2 / ok
Opening a file for writing (FILE_WRITE) will not work, because the two previous calls of FileOpen only
allow FILE_SHARE_READ.
   ha[2] = PRTF(FileOpen(rawdata, FILE_BIN | FILE_READ | FILE_WRITE | FILE_SHARE_READ));
   // -1 / CANNOT_OPEN_FILE(5004)
Now let's try to create a new file MQL5Book/newdata. If you open it as read-only, the file will not be
created.
   const string newdata = "MQL5Book/newdata";
   ha[3] = PRTF(FileOpen(newdata, FILE_BIN | FILE_READ));
   // -1 / CANNOT_OPEN_FILE(5004)
To create a file, you must specify the FILE_WRITE mode (the presence of FILE_READ is not critical
here, but it makes the call more universal: as we remember, in this combination, the instruction
guarantees that either the old file will be opened, if it exists, or a new one will be created).
   ha[3] = PRTF(FileOpen(newdata, FILE_BIN | FILE_READ | FILE_WRITE)); // 3 / ok
Let's try to write something to a new file using the function FileSave known to us. It acts as an
"external player", since it works with the file bypassing our descriptor, in much the same way as it
could be done by another MQL program or a third-party application.
   long x[1] = {0x123456789ABCDEF0};
   PRTF(FileSave(newdata, x)); // false
This call fails because the handle was opened without sharing permissions. Close and reopen the file
with maximum "permissions".
   FileClose(ha[3]);
   ha[3] = PRTF(FileOpen(newdata, 
      FILE_BIN | FILE_READ | FILE_WRITE | FILE_SHARE_READ | FILE_SHARE_WRITE)); // 3 / ok
This time FileSave works as expected.

---

## Page 418

Part 4. Common APIs
41 8
4.5 Working with files
   PRTF(FileSave(newdata, x)); // true
You can look in the folder MQL5/Files/MQL5Book/ and find there the newdata file, 8 bytes long.
Note that after we close the file, its descriptor is returned to the free descriptor pool, and the next time
a file (maybe another file) is opened, the same number comes into play again.
For a neat shutdown, we will explicitly close all open files.
   for(int i = 0; i < ArraySize(ha); ++i)
   {
      if(ha[i] != INVALID_HANDLE)
      {
        FileClose(ha[i]);
      }
   }
}
4.5.4 Managing file descriptors
Since we need to constantly remember about open files and to release local descriptors on any exit
from functions, it would be efficient to entrust the entire routine to special objects.
This approach is well-known in programming and is called Resource Acquisition Is Initialization (RAII).
Using RAII makes it easier to control resources and ensure they are in the correct state. In particular,
this is especially effective if the function that opens the file (and creates an owner object for it) exits
from several different places.
The scope of RAII is not limited to files. In the section Object type templates, we created the AutoPtr
class, which manages a pointer to an object. It was another example of this concept, since a pointer is
also a resource (memory), and it is very easy to lose it as well as it is resource-consuming to release it
in several different branches of the algorithm. 
A file wrapper class can be useful in another way as well. The file API does not provide a function that
would allow you to get the name of a file by a descriptor (despite the fact that such a relationship
certainly exists internally). At the same time, inside the object, we can store this name and implement
our own binding to the descriptor.
In the simplest case, we need some class that stores a file descriptor and automatically closes it in the
destructor. An example implementation is shown in the FileHandle.mqh file.

---

## Page 419

Part 4. Common APIs
41 9
4.5 Working with files
class FileHandle
{
   int handle;
public:
   FileHandle(const int h = INVALID_HANDLE) : handle(h)
   {
   }
   
   FileHandle(int &holder, const int h) : handle(h)
   {
      holder = h;
   }
   
   int operator=(const int h)
   {
      handle = h;
      return h;
   }
   ...
Two constructors, as well as an overloaded assignment operator, ensure that an object is bound to a
file (descriptor). The second constructor allows you to pass a reference to a local variable (from the
calling code), which will additionally get a new descriptor. This will be a kind of external alias for the
same descriptor, which can be used in the usual way in other function calls.
But you can do without an alias too. For these cases, the class defines the operator '~', which returns
the value of the internal handle variable.
   int operator~() const
   {
      return handle;
   }
Finally, the most important thing for which the class was implemented is the smart destructor:

---

## Page 420

Part 4. Common APIs
420
4.5 Working with files
   ~FileHandle()
   {
      if(handle != INVALID_HANDLE)
      {
         ResetLastError();
         // will set internal error code if handle is invalid
         FileGetInteger(handle, FILE_SIZE);
         if(_LastError == 0)
         {
            #ifdef FILE_DEBUG_PRINT
               Print(__FUNCTION__, ": Automatic close for handle: ", handle);
            #endif
            FileClose(handle);
         }
         else
         {
            PrintFormat("%s: handle %d is incorrect, %s(%d)", 
               __FUNCTION__, handle, E2S(_LastError), _LastError);
         }
      }
   }
In it, after several checks, FileClose is called for the controlled handle variable. The point is that the file
can be explicitly closed elsewhere in the program, although this is no longer required with this class. As
a result, the descriptor may become invalid by the time the destructor is called when the execution of
the algorithm leaves the block in which the FileHandle object is defined. To find this out, a dummy call
to the FileGetInteger function is used. It is a dummy because it doesn't do anything useful. If the
internal error code remains 0 after the call, the descriptor is valid.
We can omit all these checks and simply write the following:
   ~FileHandle()
   {
      if(handle != INVALID_HANDLE)
      {
         FileClose(handle);
      }
   }
If the descriptor is corrupted, FileClose won't return any warning. But we have added checks to be able
to output diagnostic information.
Let's try the FileHandle class in action. The test script for it is called FileHandle.mq5.

---

## Page 421

Part 4. Common APIs
421 
4.5 Working with files
const string dummy = "MQL5Book/dummy";
   
void OnStart()
{
   // creating a new file or open an existing one and reset it
   FileHandle fh1(PRTF(FileOpen(dummy, 
      FILE_TXT | FILE_WRITE | FILE_SHARE_WRITE | FILE_SHARE_READ))); // 1
   // another way to connect the descriptor via '='
   int h = PRTF(FileOpen(dummy, 
      FILE_TXT | FILE_WRITE | FILE_SHARE_WRITE | FILE_SHARE_READ)); // 2
   FileHandle fh2 = h;
   // and another supported syntax:
   // int f;
   // FileHandle ff(f, FileOpen(dummy,
   //    FILE_TXT | FILE_WRITE | FILE_SHARE_WRITE | FILE_SHARE_READ));
   
   // data is supposed to be written here
   // ...
   
   // close the file manually (this is not necessary; only done to demonstrate 
   // that the FileHandle will detect this and won't try to close it again)
   FileClose(~fh1); // operator '~' applied to an object returns a handle
   
   // descriptor handle in variable 'h' bound to object 'fh2' is not manually closed
   // and will be automatically closed in the destructor
}
According to the output in the log, everything works as planned:
   FileHandle::~FileHandle: Automatic close for handle: 2
   FileHandle::~FileHandle: handle 1 is incorrect, INVALID_FILEHANDLE(5007)
However, if there are lots of files, creating a tracking object copy for each of them can become an
inconvenience. For such situations, it makes sense to design a single object that collects all descriptors
in a given context (for example, inside a function).
Such a class is implemented in the FileHolder.mqh file and is shown in the FileHolder.mq5 script. One
copy of FileHolder itself creates upon request auxiliary observing objects of the FileOpener class, which
shares common features with FileHandle, especially the destructor, as well as the handle field.
To open a file via FileHolder, you should use its FileOpen method (its signature repeats the signature of
the standard FileOpen function). 

---

## Page 422

Part 4. Common APIs
422
4.5 Working with files
class FileHolder
{
   static FileOpener *files[];
   int expand()
   {
      return ArrayResize(files, ArraySize(files) + 1) - 1;
   }
public:
   int FileOpen(const string filename, const int flags, 
                const ushort delimiter = '\t', const uint codepage = CP_ACP)
   {
      const int n = expand();
      if(n > -1)
      {
         files[n] = new FileOpener(filename, flags, delimiter, codepage);
         return files[n].handle;
      }
      return INVALID_HANDLE;
   }
All FileOpener objects add up in the files array for tracking their lifetime. In the same place, zero
elements mark the moments of registration of local contexts (blocks of code) in which FileHolder
objects are created. The FileHolder constructor is responsible for this.
   FileHolder()
   {
      const int n = expand();
      if(n > -1)
      {
         files[n] = NULL;
      }
   }
As we know, during the execution of a program, it enters nested code blocks (it calls functions). If they
require the management of local file descriptors, the FileHolder objects (one per block or less) should
be described there. According to the rules of the stack (first in, last out), all such descriptions add up
at files and then are released in reverse order as the program leaves the contexts. The destructor is
called at each such moment.

---

## Page 423

Part 4. Common APIs
423
4.5 Working with files
   ~FileHolder()
   {
      for(int i = ArraySize(files) - 1; i >= 0; --i)
      {
         if(files[i] == NULL)
         {
            // decrement array and exit
            ArrayResize(files, i);
            return;
         }
         
         delete files[i];
      }
   }
Its task is to remove the last FileOpener objects in the array up to the first encountered zero element,
which indicates the boundary of the context (further in the array are descriptors from another, external
context).
You can study the whole class on your own.
Let's look at its use in the test script FileHolder.mq5. In addition to the OnStart function, it has
SubFunc. Operations with files are performed in both contexts.

---

## Page 424

Part 4. Common APIs
424
4.5 Working with files
const string dummy = "MQL5Book/dummy";
   
void SubFunc()
{
   Print(__FUNCTION__, " enter");
   FileHolder holder;
   int h = PRTF(holder.FileOpen(dummy, 
      FILE_BIN | FILE_WRITE | FILE_SHARE_WRITE | FILE_SHARE_READ));
   int f = PRTF(holder.FileOpen(dummy, 
      FILE_BIN | FILE_WRITE | FILE_SHARE_WRITE | FILE_SHARE_READ));
   // use h and f
   // ...
   // no need to manually close files and track early function exits
   Print(__FUNCTION__, " exit");
}
void OnStart()
{
   Print(__FUNCTION__, " enter");
   
   FileHolder holder;
   int h = PRTF(holder.FileOpen(dummy, 
      FILE_BIN | FILE_WRITE | FILE_SHARE_WRITE | FILE_SHARE_READ));
   // writing data and other actions on the file by descriptor
   // ...
   /*
   int a[] = {1, 2, 3};
   FileWriteArray(h, a);
   */
   
   SubFunc();
   SubFunc();
   
 if(rand() >32000) // simulate branching by conditions
   {
      // thanks to the holder we don't need an explicit call
      // FileClose(h);
      Print(__FUNCTION__, " return");
      return; // there can be many exits from the function
   }
   
   /*
     ... more code
   */
   
   // thanks to the holder we don't need an explicit call
   // FileClose(h);
   Print(__FUNCTION__, " exit");
}
We have not closed any handles manually, instances of FileHolder will do it automatically in the
destructors.

---

## Page 425

Part 4. Common APIs
425
4.5 Working with files
Here is an example of logging output:
OnStart enter
holder.FileOpen(dummy,FILE_BIN|FILE_WRITE|FILE_SHARE_WRITE|FILE_SHARE_READ)=1 / ok
SubFunc enter
holder.FileOpen(dummy,FILE_BIN|FILE_WRITE|FILE_SHARE_WRITE|FILE_SHARE_READ)=2 / ok
holder.FileOpen(dummy,FILE_BIN|FILE_WRITE|FILE_SHARE_WRITE|FILE_SHARE_READ)=3 / ok
SubFunc exit
FileOpener::~FileOpener: Automatic close for handle: 3
FileOpener::~FileOpener: Automatic close for handle: 2
SubFunc enter
holder.FileOpen(dummy,FILE_BIN|FILE_WRITE|FILE_SHARE_WRITE|FILE_SHARE_READ)=2 / ok
holder.FileOpen(dummy,FILE_BIN|FILE_WRITE|FILE_SHARE_WRITE|FILE_SHARE_READ)=3 / ok
SubFunc exit
FileOpener::~FileOpener: Automatic close for handle: 3
FileOpener::~FileOpener: Automatic close for handle: 2
OnStart exit
FileOpener::~FileOpener: Automatic close for handle: 1
4.5.5 Selecting an encoding for text mode
For written text files, the encoding should be chosen based on the characteristics of the text or
adjusted to the requirements of external programs for which the generated files are intended. If there
are no external requirements, you can follow the rule to always use ANSI for plain texts with numbers,
English letters and punctuation (a table of 1 28 such international characters is given in the section
String comparison). When working with various languages or special characters, use UTF-8 or Unicode,
i.e. respectively:
int u8 = FileOpen("utf8.txt", FILE_WRITE | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
int u0 = FileOpen("unicode.txt", FILE_WRITE | FILE_TXT | FILE_UNICODE);
For example, these settings are useful for saving the names of financial instruments to a file, since they
sometimes use special characters that denote currencies or trading modes.
Reading your own files should not be a problem, because it is enough to specify the same encoding
settings when reading as you did when writing. However, text files can come from different sources.
Their encoding may be unknown, or subject to change without prior notice. Therefore, here comes the
question of what to do if some of the files can be supplied as single-byte strings (ANSI), some as two-
byte strings (Unicode), and some as UTF-8 encoding.
Encoding can be selected via the input parameters of the program. However, this is effective only for
one file, and if you have to open many different files, their encodings may not match. Therefore, it is
desirable to instruct the system to make the correct model choice on the fly (from file to file).
MQL5 does not allow 1 00% automatic detection and application of correct encodings, however, there
is one most universal mode for reading a variety of text files. To do this, you need to set the following
input parameters of the FileOpen function:
int h = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
There are several factors at work.
First, the UTF-8 encoding transparently skips the mentioned 1 28 characters in any ANSI encoding (i.e.
they are transmitted "one to one").

---

## Page 426

Part 4. Common APIs
426
4.5 Working with files
Second, it is the most popular for Internet protocols.
Third, MQL5 has an additional built-in analysis for text formatting in two-byte Unicode, which allows you
to automatically switch the file operation mode to FILE_UNICODE, if necessary, regardless of the
specified parameters. The fact is that files in Unicode format are usually preceded by a special pair of
identifiers: 0xFFFE, or vice versa, 0xFEFF. This sequence is called the Byte Order Mark (BOM). It is
needed because, as we know, bytes can be stored inside numbers in a different order on different
platforms (this was discussed in the section Endianness control in integers).
The FILE_UNICODE format uses a 2-byte integer (code) per character, so byte order becomes
important, unlike other encodings. The Windows byte order BOM is 0xFFFE. If the MQL5 core finds this
label at the beginning of a text file, its reading will automatically switch to Unicode mode.
Let's see how the different mode settings work with text files of different encodings. For this, we will
use the FileText.mq5 script and several text files with the same content, but in different encodings (the
size in bytes is indicated in brackets):
·ansi1 252.txt (50): European encoding 1 252 (it will be displayed in full without distortion in
Windows with the European language)
·unicode1 .txt (1 02): two-byte Unicode, at the beginning is the inherent Windows BOM 0xFFFE
·unicode2.txt (1 00): two-byte Unicode without BOM (in general, BOM is optional)
·unicode3.txt (1 02): two-byte Unicode, at the beginning there is BOM inherent to Unix, 0xFEFF
·utf8.txt (54): UTF-8 encoding
In the OnStart function, we will read these files in loops with different settings of FileOpen. Please note
that by using FileHandle (reviewed in the previous section) we don't have to worry about closing files:
everything happens automatically within each iteration.

---

## Page 427

Part 4. Common APIs
427
4.5 Working with files
void OnStart()
{
   Print("=====> UTF-8");
   for(int i = 0; i < ArraySize(texts); ++i)
   {
      FileHandle fh(FileOpen(texts[i], FILE_READ | FILE_TXT | FILE_ANSI, 0, CP_UTF8));
      Print(texts[i], " -> ", FileReadString(~fh));
   }
   
   Print("=====> Unicode");
   for(int i = 0; i < ArraySize(texts); ++i)
   {
      FileHandle fh(FileOpen(texts[i], FILE_READ | FILE_TXT | FILE_UNICODE));
      Print(texts[i], " -> ", FileReadString(~fh));
   }
   
   Print("=====> ANSI/1252");
   for(int i = 0; i < ArraySize(texts); ++i)
   {
      FileHandle fh(FileOpen(texts[i], FILE_READ | FILE_TXT | FILE_ANSI, 0, 1252));
      Print(texts[i], " -> ", FileReadString(~fh));
   }
}
The FileReadString function reads a string from a file. We'll cover it in the section on writing and
reading variables.
Here is an example log with the script execution results:
=====> UTF-8
MQL5Book/ansi1252.txt -> This is a text with special characters: ?? / ? / ?
MQL5Book/unicode1.txt -> This is a text with special characters: ±Σ / £ / ¥
MQL5Book/unicode2.txt -> T
MQL5Book/unicode3.txt -> ??
MQL5Book/utf8.txt -> This is a text with special characters: ±Σ / £ / ¥
=====> Unicode
MQL5Book/ansi1252.txt ->                    ›     
MQL5Book/unicode1.txt -> This is a text with special characters: ±Σ / £ / ¥
MQL5Book/unicode2.txt -> This is a text with special characters: ±Σ / £ / ¥
MQL5Book/unicode3.txt ->   èÊ   èÊ   ?   Úç     èÊ      Úç  èÊ ?     ? úÒ ?   Úç úÒ    
MQL5Book/utf8.txt -> ÑÐ ?    ïÚ  Øù û©       ÙÀ  ïÚ üò ? j   ¼ú ?   
=====> ANSI/1252
MQL5Book/ansi1252.txt -> This is a text with special characters: ±? / £ / ¥
MQL5Book/unicode1.txt -> This is a text with special characters: ±Σ / £ / ¥
MQL5Book/unicode2.txt -> T
MQL5Book/unicode3.txt -> þÿ
MQL5Book/utf8.txt -> This is a text with special characters: Â±Î£ / Â£ / Â¥
The unicode1 .txt file is always read correctly because it has BOM 0xFFFE, and the system ignores the
settings in the source code. However, if the label is missing or is big-endian, this auto-detection does
not work. Also, when setting FILE_UNICODE, we lose the ability to read single-byte texts and UTF-8.

---

## Page 428

Part 4. Common APIs
428
4.5 Working with files
As a result, the aforementioned combination of FILE_ANSI and CP_UTF8 should be considered more
resistant to variations in formatting. Selecting a specific national code page is only recommended when
required explicitly.
Despite the significant help provided for the programmer from the API when working with files in text
mode, we can, if necessary, avoid the FILE_TXT or FILE_CSV mode, and open a text file in binary mode
FILE_BINARY. This will shift all the complexity of parsing text and determining the encoding onto the
shoulders of the programmer, but it will allow them to support other non-standard formats. But the
main point here is that text can be read from and written to a file opened in binary mode. However, the
opposite, in the general case, is impossible. A binary file with arbitrary data (which means, it does not
contain strings exclusively) opened in text mode will most likely be interpreted as text "gibberish". If
you need to write binary data to a text file, first use the CryptEncode function and CRYPT_BASE64
encoding.
4.5.6 Writing and reading arrays
Two MQL5 functions are intended for writing and reading arrays: FileWriteArray and FileReadArray. With
binary files, they allow you to handle arrays of any built-in type other than strings, as well as arrays of
simple structures that do not contain string fields, objects, pointers, and dynamic arrays. These
limitations are related to the optimization of the writing and reading processes, which is possible due to
the exclusion of types with variable lengths. Strings, objects, and dynamic arrays are just like that.
At the same time, when working with text files, these functions are able to operate on arrays of type
string (other types of arrays in files with FILE_TXT/FILE_CSV mode are not allowed by these
functions). Such arrays are stored in a file in the following format: one element per line.
If you need to store structures or classes without type restrictions in a file, use type-specific functions
that process one value per call. They are described in two sections on writing and reading variables of
built-in types: for binary and text files.
In addition, support for structures with strings can be organized through internal optimization of
information storage. For example, instead of string fields, you can use integer fields, which will contain
the indices of the corresponding strings in a separate array with strings. Given the possibility of
redefining many operations (in particular, the assignment) using OOP tools and obtaining a structural
element of an array by number, the appearance of the algorithm will practically not change. But when
writing, you can first open a file in binary mode and call FileWriteArray for an array with a simplified
structure type and then reopen the file in text mode and add an array of all strings to it using the
second FileWriteArray call. To read such a file, you should provide a header at the beginning of it
containing the number of elements in the arrays in order to pass it as the count parameter into
FileReadArray (see further along).  
If you need to save or read not an array of structures, but a single structure, use the FileWriteStruct
and FileReadStruct functions which are described in the next section.
Let's study function signatures and then consider a general example (FileArray.mq5).
uint FileWriteArray(int handle, const void &array[], int start = 0, int count = WHOLE_ARRAY)
The function writes the array array to a file with the handle descriptor. The array can be
multidimensional. The start and count parameters allow to set the range of elements; by default, it is
equal to the entire array. In the case of multidimensional arrays, the start index and the number of
elements count refer to continuous numbering across all dimensions, not the first dimension of the

---

## Page 429

Part 4. Common APIs
429
4.5 Working with files
array. For example, if the array has the configuration [][5], then the start value equal to 7 will point to
the element with indexes [1 ][2], and count = 2 will add the element [1 ][3] to it.
The function returns the number of written elements. In case of an error, it will be 0.
If handle is received in binary mode, arrays can be of any built-in type except strings, or simple
structure types. If handle is opened in any of the text modes, the array must be of type string.
uint FileReadArray(int handle, const void &array[], int start = 0, int count = WHOLE_ARRAY)
The function reads data from a file with the handle descriptor into an array. The array can be
multidimensional and dynamic. For multidimensional arrays, the start and count parameters work on
the basis of the continuous numbering of elements in all dimensions, described above. A dynamic array,
if necessary, automatically increases in size to fit the data being read. If start is greater than the
original length of the array, these intermediate elements will contain random data after memory
allocation (see the example).
Pay attention that the function cannot control whether the configuration of the array used when
writing the file matches the configuration of the receiving array when reading. Basically, there is no
guarantee that the file being read was written with FileWriteArray.
To check the validity of the data structure, some predefined formats of initial headers or other
descriptors inside files are usually used. The functions themselves will read any contents of the file
within its size and place it in the specified array.
If handle is received in binary mode, arrays can be any of the built-in non-string types or simple
structure types. If handle is opened in text mode, the array must be of type string.
Let's check the work both in binary and in text mode using the FileArray.mq5 script. To do this, we will
reserve two file names.
const string raw = "MQL5Book/array.raw";
const string txt = "MQL5Book/array.txt";
Three arrays of type long and two arrays of type string are described in the OnStart function. Only the
first array of each type is filled with data, and all the rest will be checked for reading after the files are
written.
void OnStart()
{
   long numbers1[][2] = {{1, 4}, {2, 5}, {3, 6}};
   long numbers2[][2];
   long numbers3[][2];
   
   string text1[][2] = {{"1.0", "abc"}, {"2.0", "def"}, {"3.0", "ghi"}};
   string text2[][2];
   ...
In addition, to test operations with structures, the following 3 types are defined:

---

## Page 430

Part 4. Common APIs
430
4.5 Working with files
struct TT
{
   string s1;
   string s2;
};
  
struct B
{
private:
   int b;
public:
   void setB(const int v) { b = v; }
};
  
struct XYZ : public B
{
   color x, y, z;
};
We will not be able to use a structure of the TT type in the described functions because it contains
string fields. It is needed to demonstrate a potential compilation error in a commented statement (see
further along). Inheritance between structures B and XYZ, as well as the presence of a closed field, are
not an obstacle for the functions FileWriteArray and FileReadArray.
The structures are used to declare a pair of arrays:
 TTtt[]; // empty, because data is not important
   XYZ xyz[1];
   xyz[0].setB(-1);
   xyz[0].x = xyz[0].y = xyz[0].z = clrRed;
Let's start with binary mode. Let's create a new file or open an existing file, dumping its contents.
Then, in three FileWriteArray calls, we will try to write three arrays: numbers1 , text1  and xyz.
   int writer = PRTF(FileOpen(raw, FILE_BIN | FILE_WRITE)); // 1 / ok
   PRTF(FileWriteArray(writer, numbers1)); // 6 / ok
   PRTF(FileWriteArray(writer, text1)); // 0 / FILE_NOTTXT(5012)
   PRTF(FileWriteArray(writer, xyz)); // 1 / ok
   FileClose(writer);
   ArrayPrint(numbers1);
Arrays numbers1  and xyz are written successfully, as indicated by the number of items written. The
text1  array fails with a FILE_NOTTXT(501 2) error because string arrays require the file to be opened
in text mode. Therefore the content xyz will be located in the file immediately after all elements of
numbers1 .
Note that each write (or read) function starts writing (or reading) data to the current position
within the file, and shifts it by the size of the written or read data. If this pointer is at the end of the
file before the write operation, the file size is increased. If the end of the file is reached while
reading, the pointer no longer moves and the system raises a special internal error code 5027
(FILE_ENDOFFILE). In a new file of the zero size, the beginning and end are the same.
From an array text1 , 0 items were written, so nothing in the file reminds you that between two
successful calls FileWriteArray there was one failure.

---

## Page 431

Part 4. Common APIs
431 
4.5 Working with files
In the test script, we simply output the result of the function and the status (error code) to the log, but
in a real program, we should analyze problems on the go and take some actions: fix something in the
parameters, in the file settings, or interrupt the process with a message to the user.
Let's read a file into the numbers2 array.
   int reader = PRTF(FileOpen(raw, FILE_BIN | FILE_READ)); // 1 / ok
   PRTF(FileReadArray(reader, numbers2)); // 8 / ok
   ArrayPrint(numbers2);
Since two different arrays were written to the file (not only numbers1 , but also xyz), 8 elements were
read into the receiving array (i.e., the entire file to the end, because otherwise was not specified using
parameters).
Indeed, the size of the structure XYZ is 1 6 bytes (4 fields of 4 bytes: one int and three color), which
corresponds to one row in the array numbers2 (2 elements of type long). In this case, it's a
coincidence. As noted above, the functions have no idea about the configuration and size of the raw
data and can read anything into any array: the programmer must monitor the validity of the operation.
Let's compare the initial and received states. Source array numbers1 :
       [,0][,1]
   [0,]   1   4
   [1,]   2   5
   [2,]   3   6
Resulting array numbers2:
                 [,0]          [,1]
   [0,]             1             4
   [1,]             2             5
   [2,]             3             6
   [3,] 1099511627775 1095216660735
The beginning of the numbers2 array completely matches the original numbers1  array, i.e., writing and
reading through the file work properly.
The last row is entirely occupied by a single structure XYZ (with correct values, but incorrect
representation as two numbers of type long).
Now we get to the file beginning (using the FileSeek function, which we will discuss later in the section
Position control within a file) and call FileReadArray indicating the number and quantity of elements,
i.e., we perform a partial reading.
   PRTF(FileSeek(reader, 0, SEEK_SET)); // true
   PRTF(FileReadArray(reader, numbers3, 10, 3));
   FileClose(reader);
   ArrayPrint(numbers3);
Three elements are read from the file and placed, starting at index 1 0, into the receiving array
numbers3. Since the file is read from the beginning, these elements are the values 1 , 4, 2. And since a
two-dimensional array has the configuration [][2], the through index 1 0 points to the element [5,0].
Here's what it looks like in memory:

---

## Page 432

Part 4. Common APIs
432
4.5 Working with files
       [,0][,1]
   [0,]   1   4
   [1,]   1   4
   [2,]   2   6
   [3,]   0   0
   [4,]   0   0
   [5,]   1   4
   [6,]   2   0
Items marked in yellow are random (may change for different script runs). It is possible that they will
all be zero, but this is not guaranteed. The numbers3 array initially was empty and the FileReadArray
call initiated an allocation of memory required to receive 3 elements at offset 1 0 (total 1 3). The
selected block is not filled with anything, and only 3 numbers are read from the file. Therefore,
elements with through indices from 0 to 9 (i.e. the first 5 rows), as well as the last one, with index 1 3,
contain garbage.
Multidimensional arrays are scaled along the first dimension, and therefore an increase of 1  number
means adding the entire configuration along higher dimensions. In this case, the distribution concerns a
series of two numbers ([][2]). In other words, the requested size 1 3 is rounded up to a multiple of two,
that is, 1 4.
Finally, let's test how the functions work with string arrays. Let's create a new file or open an existing
file, dumping its contents. Then, in two FileWriteArray calls, we will write the text1  and numbers1 
arrays.
   writer = PRTF(FileOpen(txt, FILE_TXT | FILE_ANSI | FILE_WRITE)); // 1 / ok
   PRTF(FileWriteArray(writer, text1)); // 6 / ok
   PRTF(FileWriteArray(writer, numbers1)); // 0 / FILE_NOTBIN(5011)
   FileClose(writer);
The string array is saved successfully. The numeric array is ignored with a FILE_NOTBIN(501 1 ) error
because it must open the file in binary mode.
When trying to write an array of structures tt, we get a compilation error with a lengthy message
"structures or classes with objects are not allowed". What the compiler actually means is that it
doesn't like fields like string (it is assumed that strings and dynamic arrays have an internal
representation of some service objects). Thus, despite the fact that the file is opened in text mode and
there are only text fields in the structure, this combination is not supported in MQL5.
   // COMPILATION ERROR: structures or classes containing objects are not allowed
   FileWriteArray(writer, tt);
The presence of string fields makes the structure "complicated" and unsuitable for working with
functions FileWriteArray/FileReadArray in any mode.
After running the script, you can change to the directory MQL5/Files/MQL5Book and examine the
contents of the generated files.
Earlier, in the section Writing and reading files in simplified mode, we discussed the FileSave and
FileLoad functions. In the test script (FileSaveLoad.mq5), we have implemented the equivalent versions
of these functions using FileWriteArray and FileReadArray. But we have not seen them in detail. Since
we are now familiar with these new functions, we can examine the source code:

---

## Page 433

Part 4. Common APIs
433
4.5 Working with files
template<typename T>
bool MyFileSave(const string name, const T &array[], const int flags = 0)
{
   const int h = FileOpen(name, FILE_BIN | FILE_WRITE | flags);
   if(h == INVALID_HANDLE) return false;
   FileWriteArray(h, array);
   FileClose(h);
   return true;
}
   
template<typename T>
long MyFileLoad(const string name, T &array[], const int flags = 0)
{
   const int h = FileOpen(name, FILE_BIN | FILE_READ | flags);
   if(h == INVALID_HANDLE) return -1;
   const uint n = FileReadArray(h, array, 0, (int)(FileSize(h) / sizeof(T)));
   // this version has the following check added compared to the standard FileLoad:
   // if the file size is not a multiple of the structure size, print a warning
   const ulong leftover = FileSize(h) - FileTell(h);
   if(leftover != 0)
   {
      PrintFormat("Warning from %s: Some data left unread: %d bytes", 
         __FUNCTION__, leftover);
      SetUserError((ushort)leftover);
   }
   FileClose(h);
   return n;
}
MyFileSave is built on a single call of FileWriteArray, and MyFileLoad on FileReadArray call, between a
pair of FileOpen/FileClose calls. In both cases, all available data is written and read. Thanks to
templates, our functions are also able to accept arrays of arbitrary types. But if any unsupported type
(for example, a class) is deduced as a meta parameter T, then a compilation error will occur, as is the
case with incorrect access to built-in functions.
4.5.7 Writing and reading structures (binary files)
In the previous section, we learned how to perform I/O operations on arrays of structures. When
reading or writing is related to a separate structure, it is more convenient to use the pair of functions
FileWriteStruct and FileReadStruct.
uint FileWriteStruct(int handle, const void &data, int size = -1 )
The function writes the contents of a simple data structure to a binary file with the handle descriptor.
As we know, such structures can only contain fields of built-in non-string types and nested simple
structures.
The main feature of the function is the size parameter. It helps to set the number of bytes to be
written, which allows us to discard some part of the structure (its end). By default, the parameter is -1 ,
which means that the entire structure is saved. If size is greater than the size of the structure, the
excess is ignored, i.e., only the structure is written, sizeof(data) bytes.

---

## Page 434

Part 4. Common APIs
434
4.5 Working with files
On success, the function returns the number of bytes written, on error it returns 0.
uint FileReadStruct(int handle, void &data, int size = -1 )
The function reads content from a binary file with the handle descriptor to the data structure. The size
parameter specifies the number of bytes to be read. If it is not specified or exceeds the size of the
structure, then the exact size of the specified structure is used.
On success, the function returns the number of bytes read, on error it returns 0.
The option to cut off the end of the structure is present only in the FileWriteStruct and FileReadStruct
functions. Therefore, their use in a loop becomes the most suitable alternative for saving and reading
an array of trimmed structures: the FileWriteArray and FileReadArray functions do not have this
capability, and writing and reading by individual fields can be more resource-intensive (we will look at
the corresponding functions in the following sections).
It should be noted that in order to use this feature, you should design your structures in such a way
that all temporary and intermediary calculation fields that should not be saved are located at the end of
the structure.
Let's look at examples of using these two functions in the script FileStruct.mq5.
Suppose we want to archive the latest quotes from time to time, in order to be able to check their
invariance in the future or to compare with similar periods from other providers. Basically, this can be
done manually through the Symbols dialog (in the Bars tab) in MetaTrader 5. But this would require
extra effort and adherence to a schedule. It is much easier to do this automatically, from the program.
In addition, manual export of quotes is done in CSV text format, and we may need to send files to an
external server. Therefore, it is desirable to save them in a compact binary form. In addition to this,
let's assume that we are not interested in information about ticks, spread and real volumes (which are
always empty for Forex symbols).
In the section Comparing, sorting, and searching in arrays, we considered the MqlRates structure and
the CopyRates function. They will be described in detail later, while now we will use them once more as
a testing ground for file operations.
Using the size parameter in FileWriteStruct, we can save only part of the MqlRates structure, without
the last fields.
At the beginning of the script, we define the macros and the name of the test file.
#define BARLIMIT 10 // number of bars to write
#define HEADSIZE 10 // size of the header of our format 
const string filename = "MQL5Book/struct.raw";
Of particular interest is the HEADSIZE constant. As mentioned earlier, file functions as such are not
responsible for the consistency of the data in the file, and the types of structures into which this data is
read. The programmer must provide such control in their code. Therefore, a certain header is usually
written at the beginning of the file, with the help of which you can, firstly, make sure that this is a file of
the required format, and secondly, save the meta-information in it that is necessary for proper reading.
In particular, the title may indicate the number of entries. Strictly speaking, the latter is not always
necessary, because we can read the file gradually until it ends. However, it is more efficient to allocate
memory for all expected records at once, based on the counter in the header.
For our purposes, we have developed a simple structure FileHeader.

---

## Page 435

Part 4. Common APIs
435
4.5 Working with files
struct FileHeader
{
   uchar signature[HEADSIZE];
   int n;
   FileHeader(const int size = 0) : n(size)
   {
      static uchar s[HEADSIZE] = {'C','A','N','D','L','E','S','1','.','0'};
      ArrayCopy(signature, s);
   }
};
It starts with the text signature "CANDLES" (in the signature field), the version number "1 .0" (same
location), and the number of entries (the n field). Since we cannot use a string field for the signature
(then the structure would no longer be simple and meet the requirements of file functions), the text is
actually packed into the uchar array of the fixed size HEADSIZE. Its initialization in the instance is done
by the constructor based on the local static copy.
In the OnStart function, we request the BARLIMIT of the last bars, open the file in FILE_WRITE mode,
and write the header followed by the resulting quotes in a truncated form to the file.
void OnStart()
{
   MqlRates rates[], candles[];
   int n = PRTF(CopyRates(_Symbol, _Period, 0, BARLIMIT, rates)); // 10 / ok
   if(n < 1) return;
  
   // create a new file or overwrite the old one from scratch
   int handle = PRTF(FileOpen(filename, FILE_BIN | FILE_WRITE)); // 1 / ok
  
 FileHeaderfh(n);// header with the actual number of entries
  
   // first write the header
   PRTF(FileWriteStruct(handle, fh)); // 14 / ok
  
   // then write the data
   for(int i = 0; i < n; ++i)
   {
      FileWriteStruct(handle, rates[i], offsetof(MqlRates, tick_volume));
   }
   FileClose(handle);
   ArrayPrint(rates);
   ...
As the size parameter value in the FileWriteStruct function, we use an expression with a familiar
operator offsetof: offsetof(MqlRates, tick_ volume), i.e., all fields starting with tick_ volume are discarded
when writing to the file.
To test the data reading, let's open the same file in FILE_READ mode and read the FileHeader
structure.

---

## Page 436

Part 4. Common APIs
436
4.5 Working with files
   handle = PRTF(FileOpen(filename, FILE_BIN | FILE_READ)); // 1 / ok
   FileHeader reference, reader;
   PRTF(FileReadStruct(handle, reader)); // 14 / ok
   // if the headers don't match, it's not our data
   if(ArrayCompare(reader.signature, reference.signature))
   {
      Print("Wrong file format; 'CANDLES' header is missing");
      return;
   }
The reference structure contains the unchanged default header (signature). The reader structure got
1 4 bytes from the file. If the two signatures match, we can continue to work, since the file format
turned out to be correct, and the reader.n field contains the number of entries read from the file. We
allocate and zero out the required size memory for the receiving array candles, and then read all
entries into it.
   PrintFormat("Reading %d candles...", reader.n);
 ArrayResize(candles, reader.n);// allocate memory for the expected data in advance
   ZeroMemory(candles);
   
   for(int i = 0; i < reader.n; ++i)
   {
      FileReadStruct(handle, candles[i], offsetof(MqlRates, tick_volume));
   }
   FileClose(handle);
   ArrayPrint(candles);
}
Zeroing was required because the MqlRates structures are read partially, and the remaining fields would
contain garbage without zeroing.
Here is the log showing the initial data (as a whole) for XAUUSD,H1 .
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.08.16 03:00:00 1778.86 1780.58 1778.12 1780.56          3049        5             0
[1] 2021.08.16 04:00:00 1780.61 1782.58 1777.10 1777.13          4633        5             0
[2] 2021.08.16 05:00:00 1777.13 1780.25 1776.99 1779.21          3592        5             0
[3] 2021.08.16 06:00:00 1779.26 1779.26 1776.67 1776.79          2535        5             0
[4] 2021.08.16 07:00:00 1776.79 1777.59 1775.50 1777.05          2052        6             0
[5] 2021.08.16 08:00:00 1777.03 1777.19 1772.93 1774.35          3213        5             0
[6] 2021.08.16 09:00:00 1774.38 1775.41 1771.84 1773.33          4527        5             0
[7] 2021.08.16 10:00:00 1773.26 1777.42 1772.84 1774.57          4514        5             0
[8] 2021.08.16 11:00:00 1774.61 1776.67 1773.69 1775.95          3500        5             0
[9] 2021.08.16 12:00:00 1775.96 1776.12 1773.68 1774.44          2425        5             0
Now let's see what was read.

---

## Page 437

Part 4. Common APIs
437
4.5 Working with files
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.08.16 03:00:00 1778.86 1780.58 1778.12 1780.56             0        0             0
[1] 2021.08.16 04:00:00 1780.61 1782.58 1777.10 1777.13             0        0             0
[2] 2021.08.16 05:00:00 1777.13 1780.25 1776.99 1779.21             0        0             0
[3] 2021.08.16 06:00:00 1779.26 1779.26 1776.67 1776.79             0        0             0
[4] 2021.08.16 07:00:00 1776.79 1777.59 1775.50 1777.05             0        0             0
[5] 2021.08.16 08:00:00 1777.03 1777.19 1772.93 1774.35             0        0             0
[6] 2021.08.16 09:00:00 1774.38 1775.41 1771.84 1773.33             0        0             0
[7] 2021.08.16 10:00:00 1773.26 1777.42 1772.84 1774.57             0        0             0
[8] 2021.08.16 11:00:00 1774.61 1776.67 1773.69 1775.95             0        0             0
[9] 2021.08.16 12:00:00 1775.96 1776.12 1773.68 1774.44             0        0             0
The quotes match, but the last three fields in each structure are empty.
You can open the MQL5/Files/MQL5Book folder and examine the internal representation of the
struct.raw file (use a viewer that supports binary mode; an example is shown below).
Options for presenting a binary file with quotes archive in an external viewer
Here is a typical way to display binary files: the left column shows addresses (offsets from the
beginning of the file), byte codes are in the middle column, and the symbolic representations of the
corresponding bytes are shown in the right column. The first and second columns use the hexadecimal
notation for numbers. The characters in the right column may differ depending on the selected ANSI
code page. It makes sense to pay attention to them only in those fragments where the presence of
text is known. In our case, the signature "CANDLES1 .0" is clearly "manifested" at the very beginning.
Numbers should be analyzed by the middle column. In this column for example, after the signature, you
can see the 4-byte value 0x0A000000, i.e., 0x0000000A in an inverted form (remember the section
Endianness control in integers): this is 1 0, the number of structures written.

---

## Page 438

Part 4. Common APIs
438
4.5 Working with files
4.5.8 Writing and reading variables (binaries)
If a structure contains fields of types that are prohibited for simple structures (strings, dynamic arrays,
pointers), then it will not be possible to write it to a file or read from a file using the functions
considered earlier. The same goes for class objects. However, such entities usually contain most of the
data in programs and also require saving and restoring their state.
Using the example of the header structure in the previous section, it was clearly shown that strings
(and other types of variable length) can be avoided, but in this case, one has to invent alternative, more
cumbersome implementations of algorithms (for example, replacing a string with an array of
characters).
To write and read data of arbitrary complexity, MQL5 provides sets of lower-level functions which
operate on a single value of a particular type: double, float, int/uint, long/ulong, or string. All other
built-in MQL5 types are equivalent to integers of different sizes: char/uchar is 1  byte, short/ushort is 2
bytes, color is 4 bytes, enumerations are 4 bytes, and datetime is 8 bytes. Such functions can be
called atomic (i.e., indivisible), because the functions for reading and writing to files at the bit level no
longer exist.
Of course, element-by-element writing or reading also removes the restriction on file operations with
dynamic arrays.
As for pointers to objects, in the spirit of the OOP paradigm, we can allow them to save and restore
objects: it is enough to implement in each class an interface (a set of methods) that is responsible for
transferring important content to files and back, and using low-level functions. Then, if we come across
a pointer field to another object as part of the object, we simply delegate saving or reading to it, and in
turn, it will deal with its fields, among which there may be other pointers, and the delegation will
continue deeper until will cover all elements.
Please note that in this section we will look at atomic functions for binary files. Their counterparts
for text files will be presented in the next section. All functions in this section return the number of
bytes written, or 0 in case of an error.
uint FileWriteDouble(int handle, double value)
uint FileWriteFloat(int handle, float value)
uint FileWriteLong(int handle, long value)
The functions write the value of the corresponding type passed in the parameter value (double, float,
long) to a binary file with the handle descriptor.
uint FileWriteInteger(int handle, int value, int size = INT_VALUE)
The function writes the value integer to a binary file with the handle descriptor. The size of the value in
bytes is set by the size parameter and can be one of the predefined constants: CHAR_VALUE (1 ),
SHORT_VALUE (2), INT_VALUE (4, default), which corresponds to types char, short and int (signed and
unsigned).
The function supports an undocumented writing mode of a 3-byte integer. Its use is not recommended.
The file pointer moves by the number of bytes written (not by the int size).
uint FileWriteString(int handle, const string value, int length = -1 )
The function writes a string from the value parameter to a binary file with the handle descriptor. You
can specify the number of characters to write the length parameter. If it is less than the length of the

---

## Page 439

Part 4. Common APIs
439
4.5 Working with files
string, only the specified part of the string will be included in the file. If length is -1  or is not specified,
the entire string is transferred to the file without the terminal null. If length is greater than the length
of the string, extra characters are filled with zeros.
Note that when writing to a file opened with the FILE_UNICODE flag (or without the FILE_ANSI flag),
the string is saved in the Unicode format (each character takes up 2 bytes). When writing to a file
opened with the FILE_ANSI flag, each character occupies 1  byte (foreign language characters may be
distorted).
The FileWriteString function can also work with text files. This aspect of its application is described
in the next section.
double FileReadDouble(int handle)
float FileReadFloat(int handle)
long FileReadLong(int handle)
The functions read a number of the appropriate type, double, float or long, from a binary file with the
specified descriptor. If necessary, convert the result to ulong (if an unsigned long is expected in the file
at that position).
int FileReadInteger(int handle, int size = INT_VALUE)
The function reads an integer value from a binary file with the handle descriptor. The value size in bytes
is specified in the size parameter.
Since the result of the function is of type int, it must be explicitly converted to the required target type
if it is different from int (i.e. to uint, or short/ushort, or char/uchar). Otherwise, you will at least get a
compiler warning and at most a loss of sign.
The fact is that when reading CHAR_VALUE or SHORT_VALUE, the default result is always positive (i.e.
corresponds to uchar and ushort, which are wholly "fit" in int). In these cases, if the numbers are
actually of types uchar and ushort, the compiler warnings are purely nominal, since we are already sure
that inside the value of type int only 1  or 2 low bytes are filled, and they are unsigned. This happens
without distortion.
However, when storing signed values (types char and short) in the file, conversion becomes necessary
because, without it, negative values will turn into inverse positive ones with the same bit representation
(see the 'Signed and unsigned integers' part in the Arithmetic type conversions section).
In any case, it is better to avoid warnings by explicit type conversion.
The function supports 3-byte integer reading mode. Its use is not recommended.
The file pointer moves by the number of bytes read (not by the size int).
string FileReadString(int handle, int size = -1 )
The function reads a string of the specified size in characters from a file with the handle descriptor. The
size parameter must be set when working with a binary file (the default value is only suitable for text
files that use separator characters). Otherwise, the string is not read (the function returns an empty
string), and the internal error code _ LastError is 501 6 (FILE_BINSTRINGSIZE).
Thus, even at the stage of writing a string to a binary file, you need to think about how the string will be
read. There are three main options:

---

## Page 440

Part 4. Common APIs
440
4.5 Working with files
·Write strings with a null terminal character at the end. In this case, they will have to be analyzed
character by character in a loop and combine characters into a string until 0 is encountered.
·Always write a string of the fixed (predefined) length. The length should be chosen with a margin for
most scenarios, or according to the specification (terms of reference, protocol, etc.), but this is
uneconomical and does not give a 1 00% guarantee that some rare string will not be shortened
when writing to a file.
·Write the length as an integer before the string.
The FileReadString function can also work with text files. This aspect of its application is described
in the next section.
Also note that if the size parameter is 0 (which can happen during some calculations), then the function
does not read: the file pointer remains in the same place and the function returns an empty string.
As an example for this section, we will improve the FileStruct.mq5 script from the previous section. The
new program name is FileAtomic.mq5.
The task remains the same: save a given number of truncated MqlRates structures with quotes to a
binary file. But now the FileHeader structure will become a class (and the format signature will be
stored in a string, not in an array of characters). A header of this type and an array of quotes will be
part of another control class Candles, and both classes will be inherited from the Persistent interface
for writing arbitrary objects to a file and reading from a file.
Here is the interface:
interface Persistent
{
   bool write(int handle);
   bool read(int handle);
};
In the FileHeader class, we will implement the saving and checking of the format signature (let's
change it to "CANDLES/1 .1 ") and of the names of the current symbol and chart timeframe (more
about _ Symbol and _ Period).
Writing is done in the implementation of the write method inherited from the interface.
class FileHeader : public Persistent
{
   const string signature;
public:
   FileHeader() : signature("CANDLES/1.1") { }
   bool write(int handle) override
   {
      PRTF(FileWriteString(handle, signature, StringLen(signature)));
      PRTF(FileWriteInteger(handle, StringLen(_Symbol), CHAR_VALUE));
      PRTF(FileWriteString(handle, _Symbol));
      PRTF(FileWriteString(handle, PeriodToString(), 3));
      return true;
   }
The signature is written exactly according to its length since the sample is stored in the object and the
same length will be set when reading.

---

## Page 441

Part 4. Common APIs
441 
4.5 Working with files
For the instrument of the current chart, we first save the length of its name in the file (1  byte is enough
for lengths up to 255), and only then we save the string itself.
The name of the timeframe never exceeds 3 symbols, if the constant prefix "PERIOD_" is excluded from
it, therefore a fixed length is chosen for this string. The timeframe name without a prefix is obtained in
the auxiliary function PeriodToString: it is in a separate header file Periods.mqh (it will be discussed in
more detail in the section Symbols and timeframes).
Reading is performed in read method in the reverse order (of course, it is assumed that the reading will
be performed in a different, new object).
   bool read(int handle) override
   {
      const string sig = PRTF(FileReadString(handle, StringLen(signature)));
      if(sig != signature)
      {
         PrintFormat("Wrong file format, header is missing: want=%s vs got %s", 
            signature, sig);
         return false;
      }
      const int len = PRTF(FileReadInteger(handle, CHAR_VALUE));
      const string sym = PRTF(FileReadString(handle, len));
      if(_Symbol != sym)
      {
         PrintFormat("Wrong symbol: file=%s vs chart=%s", sym, _Symbol);
         return false;
      }
      const string stf = PRTF(FileReadString(handle, 3));
      if(_Period != StringToPeriod(stf))
      {
         PrintFormat("Wrong timeframe: file=%s(%s) vs chart=%s", 
            stf, EnumToString(StringToPeriod(stf)), EnumToString(_Period));
         return false;
      }
      return true;
   }
If any of the properties (signature, symbol, timeframe) does not match in the file and on the current
chart, the function returns false to indicate an error.
The reverse transformation of the timeframe name into the ENUM_TIMEFRAMES enumeration is done
by the function StringToPeriod, also from the file Periods.mqh.
The main Candles class for requesting, saving and reading the archive of quotes is as follows.

---

## Page 442

Part 4. Common APIs
442
4.5 Working with files
class Candles : public Persistent
{
   FileHeader header;
   int limit;
   MqlRates rates[];
public:
   Candles(const int size = 0) : limit(size)
   {
      if(size == 0) return;
      int n = PRTF(CopyRates(_Symbol, _Period, 0, limit, rates));
      if(n < 1)
      {
 limit =0; // initialization failed
      }
 limit =n; // may be less than requested
   }
The fields are the header of the FileHeader type, the requested number of bars limit, and an array
receiving MqlRates structures from MetaTrader 5. The array is filled in the constructor. In case of an
error, the limit field is reset to zero.
Being derived from the Persistent interface, the Candles class requires the implementation of methods
write and read. In the write method, we first instruct the header object to save itself, and then append
the number of quotes, the date range (for reference), and the array itself to the file.
   bool write(int handle) override
   {
      if(!limit) return false; // no data
      if(!header.write(handle)) return false;
      PRTF(FileWriteInteger(handle, limit));
      PRTF(FileWriteLong(handle, rates[0].time));
      PRTF(FileWriteLong(handle, rates[limit - 1].time));
      for(int i = 0; i < limit; ++i)
      {
         FileWriteStruct(handle, rates[i], offsetof(MqlRates, tick_volume));
      }
      return true;
   }
Reading is done in reverse order:

---

## Page 443

Part 4. Common APIs
443
4.5 Working with files
   bool read(int handle) override
   {
      if(!header.read(handle))
      {
         return false;
      }
      limit = PRTF(FileReadInteger(handle));
      ArrayResize(rates, limit);
      ZeroMemory(rates);
      // dates need to be read: they are not used, but this shifts the position in the file;
      // it was possible to explicitly change the position, but this function has not yet been studied
      datetime dt0 = (datetime)PRTF(FileReadLong(handle));
      datetime dt1 = (datetime)PRTF(FileReadLong(handle));
      for(int i = 0; i < limit; ++i)
      {
         FileReadStruct(handle, rates[i], offsetof(MqlRates, tick_volume));
      }
      return true;
   }
In a real program for archiving quotes, the presence of a range of dates would allow building their
correct sequence over a long history by the file headers and, to some extent, would protect against
arbitrary renaming of files.
There is a simple print method to control the process:
   void print() const
   {
      ArrayPrint(rates);
   }
In the main function of the script, we create two Candles objects, and using one of them, we first save
the quotes archive and then restore it with the help of the other. Files are managed by the wrapper
FileHandle that we already know (see section File descriptor management).

---

## Page 444

Part 4. Common APIs
444
4.5 Working with files
const string filename = "MQL5Book/atomic.raw";
  
void OnStart()
{
   // create a new file and reset the old one
   FileHandle handle(PRTF(FileOpen(filename, 
      FILE_BIN | FILE_WRITE | FILE_ANSI | FILE_SHARE_READ)));
   // form data
   Candles output(BARLIMIT);
   // write them to a file
   if(!output.write(~handle))
   {
      Print("Can't write file");
      return;
   }
   output.print();
  
   // open the newly created file for checking
   handle = PRTF(FileOpen(filename, 
      FILE_BIN | FILE_READ | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE));
   // create an empty object to receive quotes
   Candles inputs;
   // read data from the file into it
   if(!inputs.read(~handle))
   {
      Print("Can't read file");
   }
   else
   {
      inputs.print();
   }
Here is an example of logs of initial data for XAUUSD,H1 :

---

## Page 445

Part 4. Common APIs
445
4.5 Working with files
FileOpen(filename,FILE_BIN|FILE_WRITE|FILE_ANSI|FILE_SHARE_READ)=1 / ok
CopyRates(_Symbol,_Period,0,limit,rates)=10 / ok
FileWriteString(handle,signature,StringLen(signature))=11 / ok
FileWriteInteger(handle,StringLen(_Symbol),CHAR_VALUE)=1 / ok
FileWriteString(handle,_Symbol)=6 / ok
FileWriteString(handle,PeriodToString(),3)=3 / ok
FileWriteInteger(handle,limit)=4 / ok
FileWriteLong(handle,rates[0].time)=8 / ok
FileWriteLong(handle,rates[limit-1].time)=8 / ok
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.08.17 15:00:00 1791.40 1794.57 1788.04 1789.46          8157        5             0
[1] 2021.08.17 16:00:00 1789.46 1792.99 1786.69 1789.69          9285        5             0
[2] 2021.08.17 17:00:00 1789.76 1790.45 1780.95 1783.30          8165        5             0
[3] 2021.08.17 18:00:00 1783.30 1783.98 1780.53 1782.73          5114        5             0
[4] 2021.08.17 19:00:00 1782.69 1784.16 1782.09 1782.49          3586        6             0
[5] 2021.08.17 20:00:00 1782.49 1786.23 1782.17 1784.23          3515        5             0
[6] 2021.08.17 21:00:00 1784.20 1784.85 1782.73 1783.12          2627        6             0
[7] 2021.08.17 22:00:00 1783.10 1785.52 1782.37 1785.16          2114        5             0
[8] 2021.08.17 23:00:00 1785.11 1785.84 1784.71 1785.80           922        5             0
[9] 2021.08.18 01:00:00 1786.30 1786.34 1786.18 1786.20            13        5             0
And here is an example of the recovered data (recall that the structures are saved in a truncated form
according to our hypothetical technical task):
FileOpen(filename,FILE_BIN|FILE_READ|FILE_ANSI|FILE_SHARE_READ|FILE_SHARE_WRITE)=2 / ok
FileReadString(handle,StringLen(signature))=CANDLES/1.1 / ok
FileReadInteger(handle,CHAR_VALUE)=6 / ok
FileReadString(handle,len)=XAUUSD / ok
FileReadString(handle,3)=H1 / ok
FileReadInteger(handle)=10 / ok
FileReadLong(handle)=1629212400 / ok
FileReadLong(handle)=1629248400 / ok
                 [time]  [open]  [high]   [low] [close] [tick_volume] [spread] [real_volume]
[0] 2021.08.17 15:00:00 1791.40 1794.57 1788.04 1789.46             0        0             0
[1] 2021.08.17 16:00:00 1789.46 1792.99 1786.69 1789.69             0        0             0
[2] 2021.08.17 17:00:00 1789.76 1790.45 1780.95 1783.30             0        0             0
[3] 2021.08.17 18:00:00 1783.30 1783.98 1780.53 1782.73             0        0             0
[4] 2021.08.17 19:00:00 1782.69 1784.16 1782.09 1782.49             0        0             0
[5] 2021.08.17 20:00:00 1782.49 1786.23 1782.17 1784.23             0        0             0
[6] 2021.08.17 21:00:00 1784.20 1784.85 1782.73 1783.12             0        0             0
[7] 2021.08.17 22:00:00 1783.10 1785.52 1782.37 1785.16             0        0             0
[8] 2021.08.17 23:00:00 1785.11 1785.84 1784.71 1785.80             0        0             0
[9] 2021.08.18 01:00:00 1786.30 1786.34 1786.18 1786.20             0        0             0
It is easy to make sure that the data is stored and read correctly. And now let's see how they look
inside the file:

---

## Page 446

Part 4. Common APIs
446
4.5 Working with files
Viewing the internal structure of a binary file with an archive of quotes in an external program
Here, various fields of our header are highlighted with color: signature, symbol name length, symbol
name, timeframe name, etc.
4.5.9 Writing and reading variables (text files)
Text files have their own set of functions for atomic (element-by-element) saving and for reading data.
It is slightly different from the binary files set in the previous section. It should also be noted that there
are no analog functions for writing/reading a structure or an array of structures to a text file. If you try
to use any of these functions with a text file, they will have no effect but will raise an internal error
code of 501 1  (FILE_NOTBIN).
As we already know, text files in MQL5 have two forms: plain text and text in CSV format. The
corresponding mode, FILE_TXT or FILE_CSV, is set when the file is opened and cannot be changed
without closing and reacquiring the handle. The difference between them appears only when reading
files. Both modes are recorded in the same way.
In the TXT mode, each call to the read function (any of the functions we'll look at in this section) finds
the next newline in the file (a '\n' character or a pair of '\r\n') and processes everything up to it. The
point of processing is to convert the text from the file into a value of a specific type corresponding to
the called function. In the simplest case, if the FileReadString function is called, no processing is
performed (the string is returned "as is").
In the CSV mode, each time the read function is called, the text in the file is logically split not only by
newlines but also by an additional delimiter specified when opening the file. The rest of the processing
of the fragment from the current position of the file to the nearest delimiter is similar.

---

## Page 447

Part 4. Common APIs
447
4.5 Working with files
In other words, reading the text and transferring the internal position within the file is done in
fragments from delimiter to delimiter, where delimiter means not only the delimiter character in the
FileOpen parameter list but also a newline ('\n', '\r\n'), as well as the beginning and end of the file.
The additional delimiter has the same effect on writing text to FILE_TXT and FILE_CSV files, but only
when using the FileWrite function: it automatically inserts this character between the recorded
elements. The FileWriteString function separator is ignored.
Let's view the formal descriptions of the functions, and then consider an example in FileTxtCsv.mq5.
uint FileWrite(int handle, ...)
The function belongs to the category of functions that take a variable number of parameters. Such
parameters are indicated in the function prototype with an ellipsis. Only built-in data types are
supported. To write structures or class objects, you must dereference their elements and pass them
individually.
The function writes all arguments passed after the first one to a text file with the handle descriptor.
Arguments are separated by commas, as in a normal argument list. The number of arguments output
to the file cannot exceed 63.
When output, numeric data is converted to text format according to the rules of the standard
conversion to (string). Values or type double output to 1 6 significant digits, either in traditional format
or scientific exponent format (the more compact option is chosen). Data of the float type is displayed
with an accuracy of 7 significant digits. To display real numbers with a different precision or in an
explicitly specified format, use the DoubleToString function (see Numbers to strings and vice versa).
Values of the datetime type are output in the format "YYYY.MM.DD hh:mm:ss" (see Date and time).
A standard color (from the list of web colors) is displayed as a name, a non-standard color is displayed
as a triple of RGB component values (see Color), separated by commas (note: comma is the most
common separator character in CSV).
For enumerations, an integer denoting the element is displayed instead of its identifier (name). For
example, when writing FRIDAY (from ENUM_DAY_OF_WEEK, see Enumerations) we get number 5 in the
file.
Values of the bool type are output as the strings "true" or "false".
If a delimiter character other than 0 was specified when opening the file, it will be inserted between two
adjacent lines resulting from the conversion of the corresponding arguments.
Once all arguments are written to the file, a line terminator '\r\n' is added.
The function returns the number of bytes written, or 0 in case of an error.
uint FileWriteString(int handle, const string text, int length = -1 )
The function writes the text string parameter to a text file with the handle descriptor. The length
parameter is only applicable for binary files and is ignored in this context (the line is written in full).
The FileWriteString function can also work with binary files. This application of the function is
described in the previous section.
Any separators (between elements in a line) and newlines must be inserted/added by the programmer.

---

## Page 448

Part 4. Common APIs
448
4.5 Working with files
The function returns the number of bytes written (in FILE_UNICODE mode this will be 2 times the
length of the string in characters) or 0 in case of an error.
string FileReadString(int handle, int length = -1 )
The function reads a string up to the next delimiter from a file with the handle descriptor (delimiter
character in a CSV file, linefeed character in any file, or until the end of the file). The length parameter
only applies to binary files and is ignored in this context.
The resulting string can be converted to a value of the required type using standard reduction rules or
using conversion functions. Alternatively, specialized read functions can be used: FileReadBool,
FileReadDatetime, FileReadNumber are described below.
In case of an error, an empty string will be returned. The error code can be found through the variable
_ LastError or function GetLastError. In particular, when the end of the file is reached, the error code
will be 5027 (FILE_ENDOFFILE).
bool FileReadBool(int handle)
The function reads a fragment of a CSV file up to the next delimiter, or until the end of the line and
converts it to a value of type bool. If the fragment contains the text "true" (in any case, including
mixed case, for example, "True"), or a non-zero number, we get true. In other cases, we get false.
The word "true" must occupy the entire read element. Even if the string starts with "true", but has a
continuation (for example, "True Volume"), we get false.
datetime FileReadDatetime(int handle)
The function reads from a CSV file a string of one of the following formats: "YYYY.MM.DD hh:mm:ss",
"YYYY.MM.DD" or "hh:mm:ss", and converts it to a value of the datetime type. If the fragment does not
contain a valid textual representation of the date and/or time, the function will return zero or "weird"
time, depending on what characters it can interpret as date and time fragments. For empty or non-
numeric strings, we get the current date with zero time.
More flexible date and time reading (with more formats supported) can be achieved by combining two
functions: StringToTime(FileReadString(handle)). For further details about StringToTime see Date and
time.
double FileReadNumber(int handle)
The function reads a fragment from the CSV file up to the next delimiter or until the end of the line, and
converts it to a value of type double according to standard type casting rules.
Please note that the double may lose the precision of very large values, which can affect the reading of
large numbers of types long/ulong (the value after which integers inside double are distorted is
90071 99254740992: an example of such a phenomenon is given in the section Unions).
Functions discussed in the previous section, including FileReadDouble, FileReadFloat,
FileReadInteger, FileReadLong, and FileReadStruct, cannot be applied to text files.
The FileTxtCsv.mq5 script demonstrates how to work with text files. Last time we uploaded quotes to a
binary file. Now let's do it in TXT and CSV formats.
Basically, MetaTrader 5 allows you to export and import quotes in CSV format from the "Symbols"
dialog. But for educational purposes, we will reproduce this process. In addition, the software
implementation allows you to deviate from the exact format that is generated by default. A fragment of
the XAUUSD H1  history exported in the standard way is shown below.

---

## Page 449

Part 4. Common APIs
449
4.5 Working with files
<DATE> » <TIME> » <OPEN> » <HIGH> » <LOW> » <CLOSE> » <TICKVOL> » <VOL> » <SPREAD>
2021.01.04 » 01:00:00 » 1909.07 » 1914.93 » 1907.72 » 1913.10 » 4230 » 0 » 5
2021.01.04 » 02:00:00 » 1913.04 » 1913.64 » 1909.90 » 1913.41 » 2694 » 0 » 5
2021.01.04 » 03:00:00 » 1913.41 » 1918.71 » 1912.16 » 1916.61 » 6520 » 0 » 5
2021.01.04 » 04:00:00 » 1916.60 » 1921.89 » 1915.49 » 1921.79 » 3944 » 0 » 5
2021.01.04 » 05:00:00 » 1921.79 » 1925.26 » 1920.82 » 1923.19 » 3293 » 0 » 5
2021.01.04 » 06:00:00 » 1923.20 » 1923.71 » 1920.24 » 1922.67 » 2146 » 0 » 5
2021.01.04 » 07:00:00 » 1922.66 » 1922.99 » 1918.93 » 1921.66 » 3141 » 0 » 5
2021.01.04 » 08:00:00 » 1921.66 » 1925.60 » 1921.47 » 1922.99 » 3752 » 0 » 5
2021.01.04 » 09:00:00 » 1922.99 » 1925.54 » 1922.47 » 1924.80 » 2895 » 0 » 5
2021.01.04 » 10:00:00 » 1924.85 » 1935.16 » 1924.59 » 1932.07 » 6132 » 0 » 5
Here, in particular, we may not be satisfied with the default separator character (tab, denoted as '"'),
the order of the columns, or the fact that the date and time are divided into two fields.
In our script, we will choose comma as a separator, and we will generate the columns in the order of
the fields of the MqlRates structure. Unloading and subsequent test reading will be performed in the
FILE_TXT and FILE_CSV modes.
const string txtfile = "MQL5Book/atomic.txt";
const string csvfile = "MQL5Book/atomic.csv";
const short delimiter = ',';
Quotes will be requested at the beginning of the function OnStart in the standard way:
void OnStart()
{
   MqlRates rates[];   
   int n = PRTF(CopyRates(_Symbol, _Period, 0, 10, rates)); // 10
We will specify the names of the columns in the array separately, and also combine them using the
helper function StringCombine. Separate titles are required because we combine them into a common
title using a selectable delimiter character (an alternative solution could be based on StringReplace).
We encourage you to work with the source code StringCombine independently: it does the opposite
operation with respect to the built-in StringSplit.
   const string columns[] = {"DateTime", "Open", "High", "Low", "Close", 
                             "Ticks", "Spread", "True"};
   const string caption = StringCombine(columns, delimiter) + "\r\n";
The last column should have been called "Volume", but we will use its example to check the
performance of the function FileReadBool. You may assume that the current name implies "True
Volume" (but such a string would not be interpreted as true).
Next, let's open two files in the FILE_TXT and FILE_CSV modes, and write the prepared header into
them.
   int fh1 = PRTF(FileOpen(txtfile, FILE_TXT | FILE_ANSI | FILE_WRITE, delimiter));//1
   int fh2 = PRTF(FileOpen(csvfile, FILE_CSV | FILE_ANSI | FILE_WRITE, delimiter));//2
  
   PRTF(FileWriteString(fh1, caption)); // 48
   PRTF(FileWriteString(fh2, caption)); // 48
Since the FileWriteString function does not automatically add a newline, we have added "\r\n" to the
caption variable.

---

## Page 450

Part 4. Common APIs
450
4.5 Working with files
   for(int i = 0; i < n; ++i)
   {
      FileWrite(fh1, rates[i].time, 
         rates[i].open, rates[i].high, rates[i].low, rates[i].close, 
         rates[i].tick_volume, rates[i].spread, rates[i].real_volume);
      FileWrite(fh2, rates[i].time, 
         rates[i].open, rates[i].high, rates[i].low, rates[i].close, 
         rates[i].tick_volume, rates[i].spread, rates[i].real_volume);
   }
   
   FileClose(fh1);
   FileClose(fh2);
Writing structure fields from the rates array is done in the same way, by calling FileWrite in a loop for
each of the two files. Recall that the FileWrite function automatically inserts a delimiter character
between arguments and adds "\r\n" at the string ends. Of course, it was possible to independently
convert all output values to strings and send them to a file using FileWriteString, but then we would
have to take care of separators and newlines ourselves. In some cases, they are not needed, for
example, if you are writing in JSON format in a compact form (essentially in one giant line).
Thus, at the recording stage, both files were managed in the same way and turned out to be the same.
Here is an example of their content for XAUUSD,H1  (your results may vary):
DateTime,Open,High,Low,Close,Ticks,Spread,True
2021.08.19 12:00:00,1785.3,1789.76,1784.75,1789.06,4831,5,0
2021.08.19 13:00:00,1789.06,1790.02,1787.61,1789.06,3393,5,0
2021.08.19 14:00:00,1789.08,1789.95,1786.78,1786.89,3536,5,0
2021.08.19 15:00:00,1786.78,1789.86,1783.73,1788.82,6840,5,0
2021.08.19 16:00:00,1788.82,1792.44,1782.04,1784.02,9514,5,0
2021.08.19 17:00:00,1784.04,1784.27,1777.14,1780.57,8526,5,0
2021.08.19 18:00:00,1780.55,1784.02,1780.05,1783.07,5271,6,0
2021.08.19 19:00:00,1783.06,1783.15,1780.73,1782.59,3571,7,0
2021.08.19 20:00:00,1782.61,1782.96,1780.16,1780.78,3236,10,0
2021.08.19 21:00:00,1780.79,1780.9,1778.54,1778.65,1017,13,0
Differences in working with these files will begin to appear at the reading stage.
Let's open a text file for reading and "scan" it using the FileReadString function in a loop, until it
returns an empty string (i.e., until the end of the file).
   string read;
   fh1 = PRTF(FileOpen(txtfile, FILE_TXT | FILE_ANSI | FILE_READ, delimiter)); // 1
   Print("===== Reading TXT");
   do
   {
      read = PRTF(FileReadString(fh1));
   }
   while(StringLen(read) > 0);
The log will show something like this:

---

## Page 451

Part 4. Common APIs
451 
4.5 Working with files
===== Reading TXT
FileReadString(fh1)=DateTime,Open,High,Low,Close,Ticks,Spread,True / ok
FileReadString(fh1)=2021.08.19 12:00:00,1785.3,1789.76,1784.75,1789.06,4831,5,0 / ok
FileReadString(fh1)=2021.08.19 13:00:00,1789.06,1790.02,1787.61,1789.06,3393,5,0 / ok
FileReadString(fh1)=2021.08.19 14:00:00,1789.08,1789.95,1786.78,1786.89,3536,5,0 / ok
FileReadString(fh1)=2021.08.19 15:00:00,1786.78,1789.86,1783.73,1788.82,6840,5,0 / ok
FileReadString(fh1)=2021.08.19 16:00:00,1788.82,1792.44,1782.04,1784.02,9514,5,0 / ok
FileReadString(fh1)=2021.08.19 17:00:00,1784.04,1784.27,1777.14,1780.57,8526,5,0 / ok
FileReadString(fh1)=2021.08.19 18:00:00,1780.55,1784.02,1780.05,1783.07,5271,6,0 / ok
FileReadString(fh1)=2021.08.19 19:00:00,1783.06,1783.15,1780.73,1782.59,3571,7,0 / ok
FileReadString(fh1)=2021.08.19 20:00:00,1782.61,1782.96,1780.16,1780.78,3236,10,0 / ok
FileReadString(fh1)=2021.08.19 21:00:00,1780.79,1780.9,1778.54,1778.65,1017,13,0 / ok
FileReadString(fh1)= / FILE_ENDOFFILE(5027)
Every call of FileReadString reads the entire line (up to '\r\n') in the FILE_TXT mode. To separate it
into elements, we should implement additional processing. Optionally, we can use the FILE_CSV mode.
Let's do the same for the CSV file.
   fh2 = PRTF(FileOpen(csvfile, FILE_CSV | FILE_ANSI | FILE_READ, delimiter)); // 2
   Print("===== Reading CSV");
   do
   {
      read = PRTF(FileReadString(fh2));
   }
   while(StringLen(read) > 0);
This time there will be many more entries in the log:

---

## Page 452

Part 4. Common APIs
452
4.5 Working with files
===== Reading CSV
FileReadString(fh2)=DateTime / ok
FileReadString(fh2)=Open / ok
FileReadString(fh2)=High / ok
FileReadString(fh2)=Low / ok
FileReadString(fh2)=Close / ok
FileReadString(fh2)=Ticks / ok
FileReadString(fh2)=Spread / ok
FileReadString(fh2)=True / ok
FileReadString(fh2)=2021.08.19 12:00:00 / ok
FileReadString(fh2)=1785.3 / ok
FileReadString(fh2)=1789.76 / ok
FileReadString(fh2)=1784.75 / ok
FileReadString(fh2)=1789.06 / ok
FileReadString(fh2)=4831 / ok
FileReadString(fh2)=5 / ok
FileReadString(fh2)=0 / ok
...
FileReadString(fh2)=2021.08.19 21:00:00 / ok
FileReadString(fh2)=1780.79 / ok
FileReadString(fh2)=1780.9 / ok
FileReadString(fh2)=1778.54 / ok
FileReadString(fh2)=1778.65 / ok
FileReadString(fh2)=1017 / ok
FileReadString(fh2)=13 / ok
FileReadString(fh2)=0 / ok
FileReadString(fh2)= / FILE_ENDOFFILE(5027)
The point is that the FileReadString function in the FILE_CSV mode takes into account the delimiter
character and splits the strings into elements. Every FileReadString call returns a single value (cell)
from a CSV table. Obviously, the resulting strings need to be subsequently converted to the appropriate
types.
This problem can be solved in a generalized form using specialized functions FileReadDatetime,
FileReadNumber, FileReadBool. However, in any case, the developer must keep track of the number of
the current readable column and determine its practical meaning. An example of such an algorithm is
given in the third step of the test. It uses the same CSV file (for simplicity, we close it at the end of
each step and open it at the beginning of the next one).
To simplify the assignment of the next field in the MqlRates structure by the column number, we have
created a child structure MqlRates that contains one template method set:

---

## Page 453

Part 4. Common APIs
453
4.5 Working with files
struct MqlRatesM : public MqlRates
{
   template<typename T>
   void set(int field, T v)
   {
      switch(field)
      {
         case 0: this.time = (datetime)v; break;
         case 1: this.open = (double)v; break;
         case 2: this.high = (double)v; break;
         case 3: this.low = (double)v; break;
         case 4: this.close = (double)v; break;
         case 5: this.tick_volume = (long)v; break;
         case 6: this.spread = (int)v; break;
         case 7: this.real_volume = (long)v; break;
      }
   }
};
In the OnStart function, we have described an array of one such structure, where we will add the
incoming values. The array was required to simplify logging with ArrayPrint (there is no ready-made
function in MQL5 for printing a structure by itself).
   Print("===== Reading CSV (alternative)");
   MqlRatesM r[1];
   int count = 0;
   int column = 0;
   const int maxColumn = ArraySize(columns);
The count variable that counts the records was required not only for statistics but also as a means to
skip the first line, which contains headers and not data. The current column number is tracked in the
column variable. Its maximum value should not exceed the number of columns maxColumn.
Now we only have to open the file and read elements from it in a loop using various functions until an
error occurs, in particular, an expected error such as 5027 (FILE_ENDOFFILE), that is, the end of the
file is reached.
When the column number is 0, we apply the FileReadDatetime function. For other columns use
FileReadNumber. The exception is the case of the first line with headers: for this we call the
FileReadBool function to demonstrate how it would react to the "True" header that was deliberately
added to the last column.

---

## Page 454

Part 4. Common APIs
454
4.5 Working with files
   fh2 = PRTF(FileOpen(csvfile, FILE_CSV | FILE_ANSI | FILE_READ, delimiter)); // 1
   do
   {
      if(column)
      {
         if(count == 1) // demo for FileReadBool on the 1st record with headers
         {
            r[0].set(column, PRTF(FileReadBool(fh2)));
         }
         else
         {
            r[0].set(column, FileReadNumber(fh2));
         }
      }
      else // 0th column is the date and time
      {
         ++count;
         if(count >1) // the structure from the previous line is ready
         {
            ArrayPrint(r, _Digits, NULL, 0, 1, 0);
         }
         r[0].time = FileReadDatetime(fh2);
      }
      column = (column + 1) % maxColumn;
   }
   while(_LastError == 0); // exit when end of file 5027 is reached (FILE_ENDOFFILE)
   
   // printing the last structure
   if(column == maxColumn - 1)
   {
      ArrayPrint(r, _Digits, NULL, 0, 1, 0);
   }
This is what is logged:

---

## Page 455

Part 4. Common APIs
455
4.5 Working with files
===== Reading CSV (alternative)
FileOpen(csvfile,FILE_CSV|FILE_ANSI|FILE_READ,delimiter)=1 / ok
FileReadBool(fh2)=false / ok
FileReadBool(fh2)=false / ok
FileReadBool(fh2)=false / ok
FileReadBool(fh2)=false / ok
FileReadBool(fh2)=false / ok
FileReadBool(fh2)=false / ok
FileReadBool(fh2)=true / ok
2021.08.19 00:00:00   0.00   0.00  0.00    0.00          0     0       1
2021.08.19 12:00:00 1785.30 1789.76 1784.75 1789.06       4831     5       0
2021.08.19 13:00:00 1789.06 1790.02 1787.61 1789.06       3393     5       0
2021.08.19 14:00:00 1789.08 1789.95 1786.78 1786.89       3536     5       0
2021.08.19 15:00:00 1786.78 1789.86 1783.73 1788.82       6840     5       0
2021.08.19 16:00:00 1788.82 1792.44 1782.04 1784.02       9514     5       0
2021.08.19 17:00:00 1784.04 1784.27 1777.14 1780.57       8526     5       0
2021.08.19 18:00:00 1780.55 1784.02 1780.05 1783.07       5271     6       0
2021.08.19 19:00:00 1783.06 1783.15 1780.73 1782.59       3571     7       0
2021.08.19 20:00:00 1782.61 1782.96 1780.16 1780.78       3236    10       0
2021.08.19 21:00:00 1780.79 1780.90 1778.54 1778.65       1017    13       0
As you see, of all the headers, only the last one is converted to the true value, and all the previous ones
are false.
The content of the read structures is the same as the original data.
4.5.1 0 Managing position in a file
As we already know, the system associates a certain pointer with each open file: it determines the
place in the file (offset from its beginning) where data will be written or read from the next time any I/O
function is called. After the function is executed, the pointer is shifted by the size of the written or read
data.
In some cases, you want to change the position of the pointer without I/O operations. In particular,
when we need to append data to the end of a file, we open it in "mixed" mode FILE_READ | 
FILE_WRITE, and then we must somehow end up at the end of the file (otherwise we will start
overwriting the data from the beginning). We could call the read functions while there is something to
read (thus shifting the pointer), but this is not efficient. It is better to use the special function FileSeek.
And the FileTell function allows getting the actual value of the pointer (position in the file).
In this section, we'll explore these and a couple of other functions related to the current position in a
file. Some of them work the same way for files in text and binary mode, while others are different.
bool FileSeek(int handle, long offset, ENUM_FILE_POSITION origin)
The function moves the file pointer by the offset number of bytes using origin as a reference which is
one of the predefined positions described in the ENUM_FILE_POSITION enumeration. The offset can be
either positive (moving to the end of the file and beyond) or negative (moving to the beginning).
ENUM_FILE_POSITION has the following members:
• SEEK_SET for the file beginning
• SEEK_CUR for the current position
• SEEK_END for the file end

---

## Page 456

Part 4. Common APIs
456
4.5 Working with files
If the calculation of the new position relative to the anchor point gave a negative value (i.e., an offset
to the left of the beginning of the file is requested), then the file pointer will be set to the beginning of
the file.
If you set the position beyond the end of the file (the value is greater than the file size), then the
subsequent writing to the file will be made not from the end of the file, but from the set position. In this
case, undefined values will be written between the previous end of the file and the given position (see
below).
The function returns true on success and false in case of an error.
ulong FileTell(int handle)
For a file opened with the handle descriptor, the function returns the current position of the internal
pointer (an offset relative to the beginning of the file). In case of an error, ULONG_MAX ((ulong)-1 ) will
be returned. The error code is available in the _ LastError variable, or through the GetLastError function.
bool FileIsEnding(int handle)
The function returns an indication of whether the pointer is at the end of the handle file. If so, the
result is true.
bool FileIsLineEnding(int handle)
For a text file with the handle descriptor, the function returns a sign of whether the file pointer is at the
end of the line (immediately after the newline characters '\n' or '\r\n'). In other words, the return
value true means that the current position is at the beginning of the next line (or at the end of the file).
For binary files, the result is always false.
The test script for the aforementioned functions is called FileCursor.mq5. It works with three files: two
binary and one text.
const string fileraw = "MQL5Book/cursor.raw";
const string filetxt = "MQL5Book/cursor.csv";
const string file100 = "MQL5Book/k100.raw";
To simplify logging of the current position, along with the end-of-file (End-Of-File, EOF) and end-of-line
(End-Of-Line, EOL) signs, we have created a helper function FileState.
string FileState(int handle)
{
   return StringFormat("P:%I64d, F:%s, L:%s", 
      FileTell(handle),
      (string)FileIsEnding(handle),
      (string)FileIsLineEnding(handle));
}
The scenario for testing the functions on a binary file includes the following steps.
Create a new or open an existing fileraw file ("MQL5Book/cursor.raw") in read/write mode.
Immediately after opening, and then after each operation, we output the current state of the file by
calling FileState.

---

## Page 457

Part 4. Common APIs
457
4.5 Working with files
void OnStart()
{
   int handle;
   Print("\n * Phase I. Binary file");
   handle = PRTF(FileOpen(fileraw, FILE_BIN | FILE_WRITE | FILE_READ));
   Print(FileState(handle));
   ...
Move the pointer to the end of the file, which will allow us to append data to this file every time the
script is executed (and not overwrite it from the beginning). The most obvious way to refer to the file
end: null offset relative to origin=SEEK_ END.
   PRTF(FileSeek(handle, 0, SEEK_END));
   Print(FileState(handle));
If the file is no longer empty (not new), we can read existing data at its arbitrary position (relative or
absolute). In particular, if the origin parameter of the FileSeek function is equal to SEEK_CUR, that
means that with a negative offset the current position will move the corresponding number of bytes
back (to the left), and with positive it will move forward (to the right).
In this example, we are trying to step back by the size of one value of type int. A little later we will see
that in this place there should be a field day_ of_ year (last field) of the structure MqlDateTime, because
we write it to a file in subsequent instructions, and this data is available from the file on the next run.
The read value is logged for comparison with what was previously saved.
   if(PRTF(FileSeek(handle, -1 * sizeof(int), SEEK_CUR)))
   {
      Print(FileState(handle));
      PRTF(FileReadInteger(handle));
   }
In a new empty file, the FileSeek call will end with error 4003 (INVALID_PARAMETER), and the if
statement block will not be executed.
Next, the file is filled with data. First, the current local time of the computer (8 bytes of datetime) is
written with FileWriteLong.
   datetime now = TimeLocal();
   PRTF(FileWriteLong(handle, now));
   Print(FileState(handle));
Then we try to step back from the current location by 4 bytes (-4) and read long.
   PRTF(FileSeek(handle, -4, SEEK_CUR));
   long x = PRTF(FileReadLong(handle));
   Print(FileState(handle));
This attempt will end with error 501 5 (FILE_READERROR), because we were at the end of the file and
after shifting 4 bytes to the left, we cannot read 8 bytes from the right (size long). However, as we will
see from the log, as a result of this unsuccessful attempt, the pointer will still move back to the end of
the file.
If you step back by 8 bytes (-8), the subsequent reading of the long value will be successful, and both
time values, including the original and one received from the file, must match.

---

## Page 458

Part 4. Common APIs
458
4.5 Working with files
   PRTF(FileSeek(handle, -8, SEEK_CUR));
   Print(FileState(handle));
   x = PRTF(FileReadLong(handle));
   PRTF((now == x));
Finally, write the MqlDateTime structure filled with the same time to the file. The position in the file will
increase by 32 (the size of the structure in bytes).
   MqlDateTime mdt;
   TimeToStruct(now, mdt);
   StructPrint(mdt); // display the date/time in the log visually
   PRTF(FileWriteStruct(handle, mdt)); // 32 = sizeof(MqlDateTime)
   Print(FileState(handle));
   FileClose(handle);
After the first run of the script for the scenario with the file fileraw (MQL5Book/cursor.raw) we get
something like the following (the time will be different):
first run 
 * Phase I. Binary file
FileOpen(fileraw,FILE_BIN|FILE_WRITE|FILE_READ)=1 / ok
P:0, F:true, L:false
FileSeek(handle,0,SEEK_END)=true / ok
P:0, F:true, L:false
FileSeek(handle,-1*sizeof(int),SEEK_CUR)=false / INVALID_PARAMETER(4003)
FileWriteLong(handle,now)=8 / ok
P:8, F:true, L:false
FileSeek(handle,-4,SEEK_CUR)=true / ok
FileReadLong(handle)=0 / FILE_READERROR(5015)
P:8, F:true, L:false
FileSeek(handle,-8,SEEK_CUR)=true / ok
P:0, F:false, L:false
FileReadLong(handle)=1629683392 / ok
(now==x)=true / ok
  2021     8    23      1    49    52             1           234
FileWriteStruct(handle,mdt)=32 / ok
P:40, F:true, L:false
According to the status, the file size is initially zero because the position is "P:0" after the shift to the
end of the file ("F:true"). After each recording (using FileWriteLong and FileWriteStruct) the position P
is increased by the size of the written data.
After the second run of the script, you can notice some changes in the log:

---

## Page 459

Part 4. Common APIs
459
4.5 Working with files
second run
 * Phase I. Binary file
FileOpen(fileraw,FILE_BIN|FILE_WRITE|FILE_READ)=1 / ok
P:0, F:false, L:false
FileSeek(handle,0,SEEK_END)=true / ok
P:40, F:true, L:false
FileSeek(handle,-1*sizeof(int),SEEK_CUR)=true / ok
P:36, F:false, L:false
FileReadInteger(handle)=234 / ok
FileWriteLong(handle,now)=8 / ok
P:48, F:true, L:false
FileSeek(handle,-4,SEEK_CUR)=true / ok
FileReadLong(handle)=0 / FILE_READERROR(5015)
P:48, F:true, L:false
FileSeek(handle,-8,SEEK_CUR)=true / ok
P:40, F:false, L:false
FileReadLong(handle)=1629683397 / ok
(now==x)=true / ok
  2021     8    23      1    49    57             1           234
FileWriteStruct(handle,mdt)=32 / ok
P:80, F:true, L:false
First, the size of the file after opening is 40 (according to the position "P:40" after the shift to the end
of the file). Each time the script is run, the file will grow by 40 bytes.
Second, since the file is not empty, it is possible to navigate through it and read the "old" data. In
particular, after retreating to -1 *sizeof(int) from the current position (which is also the end of the file),
we successfully read the value 234 which is the last field of the structure MqlDateTime (it is the
number of the day in a year and it will most likely be different for you).
The second test scenario works with the text csv file filetxt (MQL5Book/cursor.csv). We will also open it
in the combined read and write mode, but will not move the pointer to the end of the file. Because of
this, every run of the script will overwrite the data, starting from the beginning of the file. To make it
easy to spot the differences, the numbers in the first column of the CSV are randomly generated. In
the second column, the same strings are always substituted from the template in the StringFormat
function.
   Print(" * Phase II. Text file");
   srand(GetTickCount());
   // create a new file or open an existing file for writing/overwriting
   // from the very beginning and subsequent reading; inside CSV data (Unicode)
   handle = PRTF(FileOpen(filetxt, FILE_CSV | FILE_WRITE | FILE_READ, ','));
   // three rows of data (number,string pair in each), separated by '\n'
   // note that the last element does not end with a newline '\n'
   // this is optional, but allowed
   string content = StringFormat(
      "%02d,abc\n%02d,def\n%02d,ghi", 
      rand() % 100, rand() % 100, rand() % 100);
   // '\n' will be replaced with '\r\n' automatically, thanks to FileWriteString
   PRTF(FileWriteString(handle, content));
Here is an example of generated data:

---

## Page 460

Part 4. Common APIs
460
4.5 Working with files
34,abc
20,def
02,ghi
Then we return to the beginning of the file and read it in a loop with FileReadString, constantly logging
the status.
   PRTF(FileSeek(handle, 0, SEEK_SET));
   Print(FileState(handle));
   // count the lines in the file using the FileIsLineEnding feature
   int lineCount = 0;
   while(!FileIsEnding(handle))
   {
      PRTF(FileReadString(handle));
      Print(FileState(handle));
      // FileIsLineEnding also equals true when FileIsEnding equals true,
      // even if there is no trailing '\n' character
      if(FileIsLineEnding(handle)) lineCount++;
   }
   FileClose(handle);
   PRTF(lineCount);
Below are the logs for the file filetxt after the first and second run of the script. First one first:
first run
 * Phase II. Text file
FileOpen(filetxt,FILE_CSV|FILE_WRITE|FILE_READ,',')=1 / ok
FileWriteString(handle,content)=44 / ok
FileSeek(handle,0,SEEK_SET)=true / ok
P:0, F:false, L:false
FileReadString(handle)=08 / ok
P:8, F:false, L:false
FileReadString(handle)=abc / ok
P:18, F:false, L:true
FileReadString(handle)=37 / ok
P:24, F:false, L:false
FileReadString(handle)=def / ok
P:34, F:false, L:true
FileReadString(handle)=96 / ok
P:40, F:false, L:false
FileReadString(handle)=ghi / ok
P:46, F:true, L:true
lineCount=3 / ok
And here is the second one:

---

## Page 461

Part 4. Common APIs
461 
4.5 Working with files
second run
 * Phase II. Text file
FileOpen(filetxt,FILE_CSV|FILE_WRITE|FILE_READ,',')=1 / ok
FileWriteString(handle,content)=44 / ok
FileSeek(handle,0,SEEK_SET)=true / ok
P:0, F:false, L:false
FileReadString(handle)=34 / ok
P:8, F:false, L:false
FileReadString(handle)=abc / ok
P:18, F:false, L:true
FileReadString(handle)=20 / ok
P:24, F:false, L:false
FileReadString(handle)=def / ok
P:34, F:false, L:true
FileReadString(handle)=02 / ok
P:40, F:false, L:false
FileReadString(handle)=ghi / ok
P:46, F:true, L:true
lineCount=3 / ok
As you can see, the file does not change in size, but different numbers are written at the same offsets.
Because this CSV file has two columns, after every second value we read, we see an EOL flag ("L:true")
cocked.
The number of detected lines is 3, despite the fact that there are only 2 newline characters in the file:
the last (third) line ends with the file.
Finally, the last test scenario uses the file file1 00 (MQL5Book/k1 00.raw) to move the pointer past the
end of the file (to the mark of 1 000000 bytes), and thereby increase its size (reserves disk space for
potential future write operations).
   Print(" * Phase III. Allocate large file");
   handle = PRTF(FileOpen(file100, FILE_BIN | FILE_WRITE));
   PRTF(FileSeek(handle, 1000000, SEEK_END));
   // to change the size, you need to write at least something
   PRTF(FileWriteInteger(handle, 0xFF, 1));
   PRTF(FileTell(handle));
   FileClose(handle);
The log output for this script does not change from run to run, however, the random data that ends up
in the space allocated for the file may differ (its contents are not shown here: use an external binary
viewer).
 * Phase III. Allocate large file
FileOpen(file100,FILE_BIN|FILE_WRITE)=1 / ok
FileSeek(handle,1000000,SEEK_END)=true / ok
FileWriteInteger(handle,0xFF,1)=1 / ok
FileTell(handle)=1000001 / ok
4.5.1 1  Getting file properties
In the process of working with files, in addition to directly writing and reading data, it often becomes
necessary to analyze their properties. One of the main properties, the file size, can be obtained using

---

## Page 462

Part 4. Common APIs
462
4.5 Working with files
the FileSize function. But there are a few more characteristics which can be requested using
FileGetInteger.
Please note that the FileSize function requires an open file handle. FileGetInteger has some properties,
including the size, that can be recognized by the file name, and you do not need to open it first.
ulong FileSize(int handle)
The function returns the size of an open file by its descriptor. In case of an error, the result is equal to
0, which is a valid size for the normal execution of the function, so you should always analyze potential
errors using _ LastError (or GetLastError).
The file size can also be obtained by moving the pointer to the end of the file FileSeek(handle, 0,
SEEK_ END) and calling FileTell(handle). These two functions are described in the previous section.
l o n g  F i l eGetIn teg er ( i n t h a n d l e, E N U M _F IL E _PR OPE R TY_IN TE GE R  p r o p er ty)
l o n g  F i l eGetIn teg er ( co n s t s tr i n g  fi l en a m e, E N U M _F IL E _PR OPE R TY_IN TE GE R  p r o p er ty, b o o l  co m m o n  =  fa l s e)
The function has two options: to work through an open file descriptor, and by the file name (including a
closed one).
The function returns one of the file properties specified in the property parameter. The list of valid
properties is different for each of the options (see below). Even though the value type is long, depending
on the requested property, it can contain not only an integer number but also datetime or bool:
perform the required typecast explicitly.
When requesting a property by the file name, you can additionally use the common parameter to
specify in which folder the file should be searched: the current terminal folder MQL5/Files (false,
default) or the common folder Users/<user_ name>...MetaQuotes/Terminal/Common/Files (true). If the
MQL program is running in the tester, the working directory is located inside the test agent folder
(Tester/<agent>/MQL5/Files), see the introduction of the chapter Working with files.
The following table lists all the members of ENUM_FILE_PROPERTY_INTEGER.

---

## Page 463

Part 4. Common APIs
463
4.5 Working with files
Property
Description
FILE_EXISTS *
Check for existence (similar to FileIsExist)
FILE_CREATE_DATE *
Creation date
FILE_MODIFY_DATE *
Last modified date
FILE_ACCESS_DATE *
Last access date
FILE_SIZE *
File size in bytes (similar to FileSize)
FILE_POSITION
Pointer position in the file (similar to FileTell)
FILE_END
Position at the end of the file (similar to FileIsEnding)
FILE_LINE_END
Position at the end of a string (similar to FileIsLineEnding)
FILE_IS_COMMON
File opened in terminals shared folder (FILE_COMMON)
FILE_IS_TEXT
File opened as text (FILE_TXT)
FILE_IS_BINARY
File opened as binary (FILE_BIN)
FILE_IS_CSV
File opened as CSV (FILE_CSV)
FILE_IS_ANSI
File opened as ANSI (FILE_ANSI)
FILE_IS_READABLE
File opened for reading (FILE_READ)
FILE_IS_WRITABLE
File opened for writing (FILE_WRITE)
Properties allowed for use by filename are marked with an asterisk. If you try to get other properties,
the second version of the function will return an error 4003 (INVALID_PARAMETER).
Some properties can change while working with an open file: FILE_MODIFY_DATE, FILE_ACCESS_DATE,
FILE_SIZE, FILE_POSITION, FILE_END, FILE_LINE_END (for text files only).
In case of an error, the result of the call is -1 .
The second version of the function allows you to check if the specified name is the name of a file or
directory. If a directory is specified when getting properties by name, the function will set a special
internal error code 501 8 (ERR_MQL_FILE_IS_DIRECTORY), while the returned value will be correct.
We will test the functions of this section using the script FileProperties.mq5. It will work on a file with a
predefined name.
const string fileprop = "MQL5Book/fileprop";
At the beginning of OnStart, let's try to request the size by a wrong descriptor (it was not received
through the File Open call). After FileSize, the _ LastError variable check is required, and FileGetInteger
immediately returns a special value, an error indicator (-1 ).

---

## Page 464

Part 4. Common APIs
464
4.5 Working with files
void OnStart()
{
   int handle = 0;
   ulong size = FileSize(handle);
   if(_LastError)
   {
      Print("FileSize error=", E2S(_LastError) + "(" + (string)_LastError + ")");
      // We will get: FileSize 0, error=WRONG_FILEHANDLE(5008)
   }
   
   PRTF(FileGetInteger(handle, FILE_SIZE)); // -1 / WRONG_FILEHANDLE(5008)
Next, we create a new file or open an existing file and reset it, and then write the test text.
   handle = PRTF(FileOpen(fileprop, FILE_TXT | FILE_WRITE | FILE_ANSI)); // 1
   PRTF(FileWriteString(handle, "Test Text\n")); // 11
We selectively request some of the properties.
   PRTF(FileGetInteger(fileprop, FILE_SIZE)); // 0, not written to the disk yet
   PRTF(FileGetInteger(handle, FILE_SIZE)); // 11
   PRTF(FileSize(handle)); // 11
   PRTF(FileGetInteger(handle, FILE_MODIFY_DATE)); //1629730884, number of seconds since 1970
   PRTF(FileGetInteger(handle, FILE_IS_TEXT)); // 1, bool true
   PRTF(FileGetInteger(handle, FILE_IS_BINARY)); // 0, bool false
Information about the length of the file by its descriptor takes into account the current caching buffer,
and by the file name, the actual length will become available only after the file is closed, or if you call
the FileFlush function (see section Force write cache to disk).
The function returns dates and times as the number of seconds of the standard epoch since January 1 ,
1 970, which corresponds to the datetime type and can be brought to it.
The request for file open flags (its mode) is successful for the function version with a descriptor, in
particular, we received a response that the file is text and not binary. However, the next similar request
for a filename will fail because the property is only supported when a valid handle is passed. This
happens even though the name points to the same file that we have opened.
   PRTF(FileGetInteger(fileprop, FILE_IS_TEXT)); // -1 / INVALID_PARAMETER(4003)
Let's wait for one second, close the file, and check the modification date again (this time by name,
since the descriptor is no longer valid).
   Sleep(1000);
   FileClose(handle);
   PRTF(FileGetInteger(fileprop, FILE_MODIFY_DATE)); // 1629730885 / ok
Here you can clearly see that the time has increased by 1 .
Finally, make sure that properties are available for directories (folders).
   PRTF((datetime)FileGetInteger("MQL5Book", FILE_CREATE_DATE));
   // We will get: 2021.08.09 22:38:00 / FILE_IS_DIRECTORY(5018)
Since all examples of the book are located in the "MQL5Book" folder, it must already exist. However,
your actual creation time will be different. The FILE_IS_DIRECTORY error code in this case is displayed

---

## Page 465

Part 4. Common APIs
465
4.5 Working with files
for us by the PRTF macro. In the working program, the function call should be made without a macro,
and then the code should be read in _ LastError.
4.5.1 2 Force write cache to disk
File writing and reading in MQL5 are cached. This means that a certain buffer in memory is maintained
for the data, due to which the efficiency of work is increased. So, the data transferred using function
calls during writing gets into the output buffer, and only after it is full, the physical writing to the disk
takes place. When reading, on the contrary, more data is read from the disk into the buffer than the
program requested using functions (if it is not the end of the file), and subsequent read operations
(which are very likely) are faster.
Caching is a standard technology used in most applications and at the level of the operating system
itself. However, besides its pros, caching has its cons as well.
In particular, if files are used as a means of data exchange between programs, delayed writing can
significantly slow down communication and make it less predictable, since the buffer size can be quite
large, and the frequency of its "dumping" to disk can be adjusted according to some algorithms.
For example, in MetaTrader 5 there is a whole category of MQL programs for copying trading signals
from one instance of the terminal to another. They tend to use files to transfer information, and it's
very important to them that caching doesn't slow things down. For this case, MQL5 provides the
FileFlush function.
void FileFlush(int handle)
The function performs a forced flush to a disk of all data remaining in the I/O file buffer for the file with
the handle descriptor.
If you do not use this function, then part of the data "sent" from the program may, in the worst case,
get to the disk only when the file is closed.
This feature provides greater guarantees for the safety of valuable data in case of unforeseen events
(such as an operating system or program hang). However, on the other hand, frequent FileFlush calls
during mass recording are not recommended, as they can adversely affect performance.
If the file is opened in the mixed mode, simultaneously for writing and reading, the FileFlush function
must be called between reads and writes to the file.
As an example, consider the script FileFlush.mq5, in which we implement two modes that simulate the
operation of the deal copier. We will need to run two instances of the script on different charts, with
one of them becoming the data sender and the other one becoming the recipient.
The script has two input parameters: EnableFlashing allows you to compare the actions of programs
using the FileFlush function and without it, and UseCommonFolder indicates the need to create a file
that acts as a means of data transfer, to choose from: in the folder of the current instance of the
terminal or in a shared folder (in the latter case, you can test data transfer between different
terminals).

---

## Page 466

Part 4. Common APIs
466
4.5 Working with files
#property script_show_inputs
input bool EnableFlashing = false;
input bool UseCommonFolder = false;
Recall that in order for a dialog with input variables to appear when the script is launched, you must
additionally set the script_ show_ inputs property.
The name of the transit file is specified in the dataport variable. Option UseCommonFolder controls the
FILE_COMMON flag added to the set of mode switches for opened files in the File Open function.
const string dataport = "MQL5Book/dataport";
const int flag = UseCommonFolder ? FILE_COMMON : 0;
The main OnStart function actually consists of two parts: settings for the opened file and a loop that
periodically sends or receives data.
We will need to run two instances of the script, and each will have its own file descriptor pointing to the
same file on disk but opened in different modes.
void OnStart()
{
   bool modeWriter = true; // by default the script should write data
   int count = 0;          // number of writes/reads made
   // create a new or reset the old file in read mode, as a "sender"
   int handle = PRTF(FileOpen(dataport, 
      FILE_BIN | FILE_WRITE | FILE_SHARE_READ | flag));
   // if writing is not possible, most likely another instance of the script is already writing to the file,
   // so we try to open it for reading
   if(handle == INVALID_HANDLE)
   {
      // if it is possible to open the file for reading, we will continue to work as a "receiver"
      handle = PRTF(FileOpen(dataport, 
         FILE_BIN | FILE_READ | FILE_SHARE_WRITE | FILE_SHARE_READ | flag));
      if(handle == INVALID_HANDLE)
      {
         Print("Can't open file"); // something is wrong
         return;
      }
      modeWriter = false; // switch model/role
   }
In the beginning, we are trying to open the file in FILE_WRITE mode, without sharing write permission
(FILE_SHARE_WRITE), so the first instance of the running script will capture the file and prevent the
second one from working in write mode. The second instance will get an error and INVALID_HANDLE
after the first call to FileOpen and will try to open the file in the read mode (FILE_READ) with the
second FileOpen call using the FILE_SHARE_WRITE parallel write flag. Ideally, this should work. Then,
the modeWriter variable will be set to false to indicate the actual role of the script.
The main operating loop has the following structure:

---

## Page 467

Part 4. Common APIs
467
4.5 Working with files
   while(!IsStopped())
   {
      if(modeWriter)
      {
         // ...write test data
      }
      else
      {
         // ...read test data
      }
      Sleep(5000);
   }
The loop is executed until the user deletes the script from the chart manually: this will be signaled by
the IsStopped function. Inside the loop, the action is triggered every 5 seconds by calling the Sleep
function, which "freezes" the program for the specified number of milliseconds (5000 in this case). This
is done to make it easier to analyze ongoing changes and to avoid too frequent state logs. In a real
program without detailed logs, you can send data every 1 00 milliseconds or even more often.
The transmitted data will include the current time (one datetime value, 8 bytes). In the first branch of
the instruction if(modeWriter), where the file is written, we call FileWriteLong with the last count
(obtained from the function TimeLocal), increase the operation counter by 1  (count++) and output the
current state to the log.
         long temp = TimeLocal(); // get the current local time datetime
         FileWriteLong(handle, temp); // append it to the file (every 5 seconds)
         count++;
         if(EnableFlashing)
         {
            FileFlush(handle);
         }
         Print(StringFormat("Written[%d]: %I64d", count, temp));
It is important to note that calling the FileFlush function after each entry is done only if the input
parameter EnableFlashing is set to true.
In the second branch of the if operator, in which we read the data, we first reset the internal error flag
by calling ResetLastError. This is necessary because we are going to read the data from the file as long
as there is any data. Once there is no more data to read, the program will get a specific error code
501 5 (ERR_FILE_READERROR).
Since the built-in MQL5 timers, including the Sleep function, have limited accuracy (approximately 1 0
ms), we cannot exclude the situation where two consecutive writes occurred between two consecutive
attempts to read a file. For example, one reading occurred at 1 0:00:00'200, and the second at
1 0:00:05'21 0 (in the notation "hours:minutes:seconds' milliseconds"). In this case, two recordings
occurred in parallel: one at 1 0:00:00'205, and the second at 1 0:00:05'205, and both fell into the
above period. Such a situation is unlikely but possible. Even with absolutely precise time intervals, the
MQL5 runtime system may be forced to choose between two running scripts (which one to invoke
earlier than the other) if the total number of programs is large and there are not enough processor
cores for all of them.
MQL5 provides high-precision timers (up to microseconds), but this is not critical for the current task.

---

## Page 468

Part 4. Common APIs
468
4.5 Working with files
The nested loop is needed for one more reason. Immediately after the script is launched as a
"receiver" of data, it must process all the records from the file that have accumulated since the launch
of the "sender" (it is unlikely that both scripts can be launched simultaneously). Probably someone
would prefer a different algorithm: skip all the "old" records and keep track of only the new ones. This
can be done, but the "lossless" option is implemented here.
         ResetLastError();
         while(true)// loop as long as there is data and no problems
         {
            bool reportedEndBeforeRead = FileIsEnding(handle);
            ulong reportedTellBeforeRead = FileTell(handle);
  
            temp = FileReadLong(handle);
            // if there is no more data, we will get an error 5015 (ERR_FILE_READERROR)
            if(_LastError)break; // exit the loop on any error
  
            // here the data is received without errors
            count++;
            Print(StringFormat("Read[%d]: %I64d\t"
               "(size=%I64d, before=%I64d(%s), after=%I64d)", 
               count, temp, 
               FileSize(handle), reportedTellBeforeRead, 
               (string)reportedEndBeforeRead, FileTell(handle)));
         }
Please note the following point. The metadata about the file opened for reading, such as its size,
returned by the FileSize function (see Getting file properties) does not change after the file is opened.
If another program later adds something to the file we opened for reading, its "detectable" length will
not be updated even if we call FileFlash for the read descriptor. It would be possible to close and reopen
the file (before each read, but this is not efficient): then the new length would appear for the new
descriptor. But we will do without it, with the help of another trick.
The trick is to keep reading data using read functions (in our case FileReadLong) for as long as they
return data without errors. It is important not to use other functions that operate on metadata. In
particular, due to the fact that the read-only end-of-file remains constant, checking with the
FileIsEnding function (see Position control within a file) will give true at the old position, despite the
possible replenishment of the file from another process. Moreover, an attempt to move the internal file
pointer to the end (FileSeek(handle, 0, SEEK_ END); for the FileSeek function see the same section) will
not jump to the actual end of the data, but to the outdated position where the end was located at the
time of opening.
The function tells us the real position inside the file FileTell (see the same section). As information is
added to the file from another instance of the script and read in this loop, the pointer will move further
and further to the right, exceeding, however strange it is, FileSize. For a visual demonstration of how
the pointer moves beyond the file size, let's save its values before and after calling FileReadLong, and
then output the values along with the size to the log.
Once reading with FileReadLong generates any error, the inner loop will break. Regular loop exit implies
error 501 5 (ERR_FILE_READERROR). In particular, it occurs when there is no data available for reading
at the current position in the file.
The last successfully read data is output to the log, and it is easy to compare it with what the sender
script output there.

---

## Page 469

Part 4. Common APIs
469
4.5 Working with files
Let's run a new script twice. To distinguish between its copies, we'll do it on the charts of different
instruments.
When running both scripts, it is important to observe the same value of the UseCommonFolder
parameter. Let's leave it in our tests equal to false since we will be doing everything in one
terminal. Data transfer between different terminals with UseCommonFolder set to true is suggested
for independent testing.
First, let's run the first instance on the EURUSD,H1  chart, leaving all the default settings, including
EnableFlashing=false. Then, we will run the second instance on the XAUUSD,H1  chart (also with default
settings). The log will be as follows (your time will be different):
(EURUSD,H1) *
(EURUSD,H1) FileOpen(dataport,FILE_BIN|FILE_WRITE|FILE_SHARE_READ|flag)=1 / ok
(EURUSD,H1) Written[1]: 1629652995
(XAUUSD,H1) *
(XAUUSD,H1) FileOpen(dataport,FILE_BIN|FILE_WRITE|FILE_SHARE_READ|flag)=-1 / CANNOT_OPEN_FILE(5004)
(XAUUSD,H1) FileOpen(dataport,FILE_BIN|FILE_READ|FILE_SHARE_WRITE|FILE_SHARE_READ|flag)=1 / ok
(EURUSD,H1) Written[2]: 1629653000
(EURUSD,H1) Written[3]: 1629653005
(EURUSD,H1) Written[4]: 1629653010
(EURUSD,H1) Written[5]: 1629653015
The sender successfully opened the file for writing and started sending data every 5 seconds, according
to the lines with the word "Written" and to the increasing values. Less than 5 seconds after the sender
was started, the receiver was also started. It gave an error message because it could not open the file
for writing. But then it successfully opened the file for reading. However, there are no records indicating
that it was able to find the transmitted data in the file. The data remained "hanging" in the sender's
cache.
Let's stop both scripts and run them again in the same sequence: first, we run the sender on EURUSD,
and then the receiver on XAUUSD. But this time we will specify EnableFlashing=true for the sender.
Here's what happens in the log:
(EURUSD,H1) *
(EURUSD,H1) FileOpen(dataport,FILE_BIN|FILE_WRITE|FILE_SHARE_READ|flag)=1 / ok
(EURUSD,H1) Written[1]: 1629653638
(XAUUSD,H1) *
(XAUUSD,H1) FileOpen(dataport,FILE_BIN|FILE_WRITE|FILE_SHARE_READ|flag)=-1 / CANNOT_OPEN_FILE(5004)
(XAUUSD,H1) FileOpen(dataport,FILE_BIN|FILE_READ|FILE_SHARE_WRITE|FILE_SHARE_READ|flag)=1 / ok
(XAUUSD,H1) Read[1]: 1629653638 (size=8, before=0(false), after=8)
(EURUSD,H1) Written[2]: 1629653643
(XAUUSD,H1) Read[2]: 1629653643 (size=8, before=8(true), after=16)
(EURUSD,H1) Written[3]: 1629653648
(XAUUSD,H1) Read[3]: 1629653648 (size=8, before=16(true), after=24)
(EURUSD,H1) Written[4]: 1629653653
(XAUUSD,H1) Read[4]: 1629653653 (size=8, before=24(true), after=32)
(EURUSD,H1) Written[5]: 1629653658
The same file is again successfully opened in different modes in both scripts, but this time the written
values are regularly read by the receiver.
It is interesting to note that before each next data reading, except for the first one, the FileIsEnding
function returns true (displayed in the same string as the received data, in parentheses after the
"before" string). Thus, there is a sign that we are at the end of the file, but then FileReadLong

---

## Page 470

Part 4. Common APIs
470
4.5 Working with files
successfully reads a value supposedly outside of the file limit and shifts the position to the right. For
example, the entry "size=8, before=8(true), after=1 6" means that the file size is reported to the MQL
program as 8, the current pointer before the call to FileReadLong is also equal to 8 and the end-of-file
sign is enabled. After a successful call to FileReadLong, the pointer is moved to 1 6. However, on the
next and all other iterations, we see "size=8" again, and the pointer gradually moves further and
further out of the file.
Since the write in the sender and the read in the receiver occur once every 5 seconds, depending on
their loop offset phases, we can observe the effect of a different delay between the two operations, up
to almost 5 seconds in the worst case. However, this does not mean that cache flushing is so slow. In
fact, it is almost an instant process. To ensure a more rapid change detection, you can reduce the
sleep period in loops (please note that this test, if the delay is too short, will quickly fill the log – unlike
a real program, new data is always generated here as this is the sender's current time to the nearest
second).
Incidentally, you can run multiple recipients, as opposed to the sender which must be only one. The log
below shows the operation of a sender on EURUSD and of two recipients on the XAUUSD and USDRUB
charts.
(EURUSD,H1) *
(EURUSD,H1) FileOpen(dataport,FILE_BIN|FILE_WRITE|FILE_SHARE_READ|flag)=1 / ok
(EURUSD,H1) Written[1]: 1629671658
(XAUUSD,H1) *
(XAUUSD,H1) FileOpen(dataport,FILE_BIN|FILE_WRITE|FILE_SHARE_READ|flag)=-1 / CANNOT_OPEN_FILE(5004)
(XAUUSD,H1) FileOpen(dataport,FILE_BIN|FILE_READ|FILE_SHARE_WRITE|FILE_SHARE_READ|flag)=1 / ok
(XAUUSD,H1) Read[1]: 1629671658 (size=8, before=0(false), after=8)
(EURUSD,H1) Written[2]: 1629671663
(USDRUB,H1) *
(USDRUB,H1) FileOpen(dataport,FILE_BIN|FILE_WRITE|FILE_SHARE_READ|flag)=-1 / CANNOT_OPEN_FILE(5004)
(USDRUB,H1) FileOpen(dataport,FILE_BIN|FILE_READ|FILE_SHARE_WRITE|FILE_SHARE_READ|flag)=1 / ok
(USDRUB,H1) Read[1]: 1629671658 (size=16, before=0(false), after=8)
(USDRUB,H1) Read[2]: 1629671663 (size=16, before=8(false), after=16)
(XAUUSD,H1) Read[2]: 1629671663 (size=8, before=8(true), after=16)
(EURUSD,H1) Written[3]: 1629671668
(USDRUB,H1) Read[3]: 1629671668 (size=16, before=16(true), after=24)
(XAUUSD,H1) Read[3]: 1629671668 (size=8, before=16(true), after=24)
(EURUSD,H1) Written[4]: 1629671673
(USDRUB,H1) Read[4]: 1629671673 (size=16, before=24(true), after=32)
(XAUUSD,H1) Read[4]: 1629671673 (size=8, before=24(true), after=32)
(EURUSD,H1) Written[5]: 1629671678
By the time the third script on USDRUB was launched, there were already 2 records of 8 bytes in the
file, so the inner loop immediately performed 2 iterations from FileReadLong, and the file size "seems"
to be equal to 1 6.
4.5.1 3 Deleting a file and checking if it exists
Checking if a file exists and deleting it are critical actions related to the file system, i.e., to the external
environment in which files "live". So far, we've looked at functions that manipulate the internal contents
of files. Starting with this section, the focus will shift towards functions that manage files as indivisible
units.

---

## Page 471

Part 4. Common APIs
471 
4.5 Working with files
bool FileIsExist(const string filename, int flag = 0)
The function checks if a file with the name filename exists and returns true if it does. The search
directory is selected using the flag parameter: if it is 0 (the default value), the file is searched in the
directory of the current terminal instance (MQL5/Files); if flag equals FILE_COMMON, the common
directory of all terminals Users/<user>...MetaQuotes/Terminal/Common/Files is checked. If the MQL
program is running in the tester, the working directory is located inside the tester agent folder
(Tester/<agent>/MQL5/Files), see an introductory part of the chapter Working with files.
The specified name may belong not to a file but to a directory. In this case, the FileIsExist function will
return false and a pseudo-error 501 8 (FILE_IS_DIRECTORY) will be logged into the _ LastError variable.
bool FileDelete(const string filename, int flag = 0)
The function deletes the file with the specified name filename. The flag parameter specifies the location
of the file. With the default value, deletion is performed in the working directory of the current terminal
instance (MQL5/Files) or tester agent (Tester/<agent>/MQL5/Files) if the program is running in the
tester. If flag equals FILE_COMMON, the file must be located in the common folder of all terminals
(/Terminal/Common/Files).
The function returns a sign of success (true) or error (false).
This function does not allow deleting directories. For this purpose, use the FolderDelete function (see
Working with folders).
To see how the described functions work, we will use the script FileExist.mq5. We will do several
manipulations with a temporary file.
const string filetemp = "MQL5Book/temp";
void OnStart()
{
   PRTF(FileIsExist(filetemp)); // false / FILE_NOT_EXIST(5019)
   PRTF(FileDelete(filetemp));  // false / FILE_NOT_EXIST(5019)
   
   int handle = PRTF(FileOpen(filetemp, FILE_TXT | FILE_WRITE | FILE_ANSI)); // 1
   
   PRTF(FileIsExist(filetemp)); // true
   PRTF(FileDelete(filetemp));  // false / CANNOT_DELETE_FILE(5006)
   
   FileClose(handle);
   
   PRTF(FileIsExist(filetemp)); // true
   PRTF(FileDelete(filetemp));  // true
   PRTF(FileIsExist(filetemp)); // false / FILE_NOT_EXIST(5019)
   
   PRTF(FileIsExist("MQL5Book")); // false / FILE_IS_DIRECTORY(5018)
   PRTF(FileDelete("MQL5Book"));  // false / FILE_IS_DIRECTORY(5018)
}
The file does not initially exist, so both functions FileIsExist and FileDelet return false, and the error
code is 501 9 (FILE_NOT_EXIST).
We then create a file, and the FileIsExist function reports its presence. However, it cannot be deleted
because it is open and busy with our process (error code 5006, CANNOT_DELETE_FILE).

---

## Page 472

Part 4. Common APIs
472
4.5 Working with files
Once the file is closed, it can be deleted.
At the end of the script, the "MQL5Book" directory is checked and an attempt is made to delete it.
FileIsExist returns false because it's not a file, however the error code 501 8 (FILE_IS_DIRECTORY)
specifies that it's a directory.
4.5.1 4 Copying and moving files
The main operations on files at the file system level are copying and moving. For these purposes, MQL5
implements two functions with identical prototypes.
bool FileCopy(const string source, int flag, const string destination, int mode)
The function copies the source file to the destination file. Both mentioned parameters can contain only
file names, or names together with prefixing paths (folder hierarchies) in MQL5 sandboxes. The flag and
mode parameters determine, in which working folder the source file is searched and which working
folder is the target: 0 means it is a folder of the local current instance of the terminal (or the tester
agent, if the program is running in the tester), and the value FILE_COMMON means the common folder
for all terminals.
In addition, in the mode parameter, you can optionally specify the FILE_REWRITE constant (if you need
to combine FILE_REWRITE and FILE_COMMON, this is done using the bitwise operator OR (| )). In the
absence of FILE_REWRITE, copying over an existing file is prohibited. In other words, if the file with the
path and name specified in the destination parameter already exists, you must confirm your intention
to overwrite it by setting FILE_REWRITE. If this is not done, the function call will fail.
The function returns true upon successful completion or false in case of an error.
Copying may fail if the source or destination file is occupied (opened) by another process.
When copying files, their metadata (creation time, access rights, alternative data streams) is
usually saved. If you need to perform "pure" copying of only the data of the file itself, you can use
successive calls FileLoad and FileSave, see Writing and reading files in simplified mode.
bool FileMove(const string source, int flag, const string destination, int mode)
The function moves or renames a file. The source path and name are specified in the source parameter
and the target path and name are specified in destination.
The list of parameters and their operating principles are the same as for the FileCopy function. Roughly
speaking, FileMove does the same work as FileCopy, but it additionally deletes the original file after a
successful copy.
Let's learn how to work with functions in practice using the script FileCopy.mq5. It has two variables
with the file names. Both files do not exist when the script is run.
const string source = "MQL5Book/source";
const string destination = "MQL5Book/destination";
In OnStart, we perform a sequence of actions according to a simple scenario. First, we try to copy the
source file from the local working directory to the destination file of the general directory. As expected,
we get false, and the error code in _ LastError will be 501 9 (FILE_NOT_EXIST).

---

## Page 473

Part 4. Common APIs
473
4.5 Working with files
void OnStart()
{
   PRTF(FileCopy(source, 0, destination, FILE_COMMON)); // false / FILE_NOT_EXIST(5019)
   ...
Therefore, we will create a source file in the usual way, write some data and flush it onto the disk.
   int handle = PRTF(FileOpen(source, FILE_TXT | FILE_WRITE)); // 1
   PRTF(FileWriteString(handle, "Test Text\n")); // 22
   FileFlush(handle);
Since the file was left open and the FILE_SHARE_READ permission was not specified when opening,
access to it in other ways (bypassing the handle) is still blocked. Hence, the next copy attempt will fail
again.
   PRTF(FileCopy(source, 0, destination, FILE_COMMON)); // false / CANNOT_OPEN_FILE(5004)
Let's close the file and try again. But first, let's output the properties of the resulting file to the log:
when it was created and modified. Both properties will contain the current timestamp of your
computer.
   FileClose(handle);
   PRTF(FileGetInteger(source, FILE_CREATE_DATE)); // 1629757115, example
   PRTF(FileGetInteger(source, FILE_MODIFY_DATE)); // 1629757115, example
Let's wait for 3 seconds before calling FileCopy. This will allow you to see the difference in the
properties of the original file and its copy. This pause has nothing to do with the previous lock on the
file: we could copy immediately after we closed the file, or even while writing it if the
FILE_SHARE_READ option was enabled.
   Sleep(3000);
Let's copy the file. This time the operation succeeds. Let's see the copy properties.
   PRTF(FileCopy(source, 0, destination, FILE_COMMON)); // true
   PRTF(FileGetInteger(destination, FILE_CREATE_DATE, true)); // 1629757118, +3 seconds
   PRTF(FileGetInteger(destination, FILE_MODIFY_DATE, true)); // 1629757115, example
Each file has its own creation time (for a copy it is 3 seconds later than for the original), but the
modification time is the same (the copy has inherited the properties of the original).
Now let's try to move the copy back to the local folder. It cannot be done without the FILE_REWRITE
option because there is no permission to overwrite the original file.
   PRTF(FileMove(destination, FILE_COMMON, source, 0)); // false / FILE_CANNOT_REWRITE(5020)
By changing the value of the parameter, we will achieve a successful file transfer.
   PRTF(FileMove(destination, FILE_COMMON, source, FILE_REWRITE)); // true
Finally, the original file is also deleted to leave a clean environment for new experiments with this script.
   ...
   FileDelete(source);
}

---

## Page 474

Part 4. Common APIs
474
4.5 Working with files
4.5.1 5 Searching for files and folders
MQL5 allows you to search for files and folders within terminal sandboxes, tester agents, and the
common sandbox for all terminals (for more details about sandboxes, see the chapter introduction
Working with files). If you know exactly the required file/directory name and location, use the
FileIsExist function.
long FileFindFirst(const string filter, string &found, int flag = 0)
The function starts searching for files and folders according to the passed filter. The filter can contain a
path consisting of subfolders within the sandbox and must contain the exact name or name pattern of
the file system elements that are searched for. The filter parameter cannot be empty.
A template is a string that contains one or more wildcard characters. There are two types of such
characters: the asterisk ('*') replaces any number of any characters (including zero), and the question
mark ('?') replaces no more than one of any character. For example, the filter "*" will find all files and
folders, and "???.*" will find only those having the name no longer than 3 characters, and the extension
may or may not be present. Files with the "csv" extension can be found by the "*.csv" filter (but note
that the folder can also have an extension). Filter "*." finds elements without an extension, and ".*"
finds elements without a name. However, the following should be remembered here.
In many versions of Windows, two kinds of names are generated for file system elements: long (by
default, up to 260 characters) and short (in the 8.3 format inherited from MS-DOS). The second
kind is automatically generated from the long name if it exceeds 8 characters or the extension is
longer than 3. This generation of short names can be disabled on the system if no software uses
them, but they are usually enabled.
Files are searched in both types of names, which is why the returned list may contain elements that
are unexpected at first glance. In particular, a short name, if generated by the system from a long
name, always contains an initial part before the dot, up to 8 characters long. It may accidentally
find a match with the desired pattern.
If you need to find files with several extensions, or with different fragments in the name that cannot be
generalized by one pattern, you will have to repeat the search process several times with different
settings.
The search is performed only in a specific folder (either in the root folder of the sandbox if there is no
path in the filter, or in the specified subfolder if the filter contains a path) and does not go into
subdirectories.
The search is not case-sensitive. For example, a request for "*.txt" files will also return files with the
extension "TXT", "Txt", etc.
If a file or folder with a matching name is found, that name is placed in the output parameter found
(requires a variable because the result is passed by reference) and the function returns a search
handle: this will need to be passed to the FileFindNext function to continue iterating over matching
items if there are many.
In the found parameter, only the name and extension are returned, without the path (folder hierarchy)
that might have been specified in the filter.
If the item found is a folder, a '\' (backslash) character is appended to the right of its name.
The flag parameter allows the selection of the search area between the local working folder of the
current copy of the terminal (by value 0) or the common folder of all terminals (by value

---

## Page 475

Part 4. Common APIs
475
4.5 Working with files
FILE_COMMON). When an MQL program is executed in a tester, its local sandbox (0) is located in the
tester agent directory.
After the search procedure is completed, the received handle should be freed by calling FileFindClose
(see further along).
bool FileFindNext(long handle, string &found)
The function continues searching for suitable elements of the file system, started by the FileFindFirst
function. The first parameter is the descriptor received from FileFindFirst, due to which all the previous
search conditions are applied.
If the next element is found, its name is passed to the calling code via the argument found, and the
function returns true.
If there are no more elements, the function returns false.
void FileFindClose(long handle)
The function closes the search descriptor received as a result of the call FileFindFirst.
The function must be called after the search procedure is completed in order to free system resources.
As an example, let's consider the script FileFind.mq5. In the previous sections, we tested many other
scripts that created files in the directory MQL5/Files/MQL5Book. Request a list of all such files.
void OnStart()
{
   string found; // receiving variable
   // start searching and get descriptor 
   long handle = PRTF(FileFindFirst("MQL5Book/*", found));
   if(handle != INVALID_HANDLE)
   {
      do
      {
         Print(found);
      }
      while(FileFindNext(handle, found));
      FileFindClose(handle);
   }
}
Even if you have cleared this directory, you can copy the sample files supplied with the book in various
encodings into it. So the script FileFind.mq5 should output at least the following list (the order of
enumeration may change):
ansi1252.txt
unicode1.txt
unicode2.txt
unicode3.txt
utf8.txt
To simplify the search process, the script has an auxiliary function DirList. It contains all the necessary
calls to built-in functions and a loop for building a string array with a list of elements that match the
filter.

---

## Page 476

Part 4. Common APIs
476
4.5 Working with files
bool DirList(const string filter, string &result[], bool common = false)
{
   string found[1];
   long handle = FileFindFirst(filter, found[0]);
   if(handle == INVALID_HANDLE) return false;
   do
   {
      if(ArrayCopy(result, found, ArraySize(result)) != 1) break;
   }
   while(FileFindNext(handle, found[0]));
   FileFindClose(handle);
   
   return true;
}
With it, we will request a list of directories in the local sandbox. To do this, we use the assumption that
directories usually do not have an extension (in theory, this is not always the case, and therefore a
more strict request for a list of subfolders should be implemented differently by those who wish). The
filter for elements with no extension is "*." (you can check it with the command dir in Windows shell
"dir *."). However, this template causes error 5002 (WRONG_FILENAME) in MQL5 functions.
Therefore, we will specify a more "vague" template "*.?": it means elements without an extension or
with an extension of 1  character.
void OnStart()
{
   ...
   string list[];
   // try to request elements without extension
   // (works on the Windows command line)
   PRTF(DirList("*.", list)); // false / WRONG_FILENAME(5002)
   
   // expand the condition: the extension must be no more than 1 character
   if(DirList("*.?", list))
   {
      ArrayPrint(list);
      // example: "MQL5Book\" "Tester\"
   }
}
In my MetaTrader 5 instance, the script finds two folders "MQL5Book\" and "Tester\". You should have
the first one too if you ran the previous test scripts.
4.5.1 6 Working with folders
It is difficult to imagine a file system without the ability to structure stored information through an
arbitrary hierarchy of directories – containers for sets of logically related files. At the MQL5 level, this
feature is also supported. If necessary, we can create, clean up and delete folders using the built-in
functions FolderCreate, FolderClean, and FolderDelete.
Earlier, we have already seen one way to create a folder, and, perhaps, not even one, but the entire
required hierarchy of subfolders at once. For this, when creating (opening) a file using FileOpen, or

---

## Page 477

Part 4. Common APIs
477
4.5 Working with files
when copying it (FileCopy, FileMove), you should specify not just a name, but precede it with the
required path. For example,
   FileCopy("MQL5Book/unicode1.txt", 0, "ABC/DEF/code.txt", 0);
This statement will create the "ABC" folder in the sandbox, the "DEF" folder in it, and copy the file
there under a new name (the source file must exist).
If you do not want to create a source file in advance, you can create a dummy file on the go:
   uchar dummy[];
   FileSave("ABC/DEF/empty", dummy);
Here we will get the same folder hierarchy as in the previous example but with a zero-size "empty" file.
With such approaches, the creation of folders becomes some sort of a by-product of working with files.
However, sometimes it is required to operate with folders as independent entities and without side
effects, in particular, just create an empty folder. This is offered by the FolderCreate function.
bool FolderCreate(const string folder, int flag = 0)
The function creates a folder named folder, which can include a path (several top-level folder names).
In either case, a single folder or folder hierarchy is created in the sandbox defined by the flag
parameter. By default when flag is 0, the local working folder MQL5/Files of terminal or tester agent (if
the program is running in the tester) is used. If flag equals FILE_COMMON, the shared folder of all
terminals is used.
The function returns true on success, or if the folder already exists. In case of an error, the result is
false.
bool FolderClean(const string folder, int flag = 0)
The function deletes all files and folders of any nesting level (together with all content) in the specified
folder directory. The flag parameter specifies the sandbox (local or global) in which the action takes
place.
Use this feature with caution, as all files and subfolders (with their files) are permanently deleted.
bool FolderDelete(const string folder, int flag = 0)
The function deletes the specified folder (folder). Before calling the function, the folder must be empty,
otherwise it cannot be deleted.
Techniques for working with these three functions are demonstrated in the script FileFolder.mq5. You
can execute this script in the debug mode step by step (statement by statement) and watch in the file
manager how folders and files appear and disappear. However, please note that before executing the
next instruction, you should use the file manager to exit the created folders up to the "MQL5Book"
level, because otherwise the folders may be occupied by the file manager, and this will disrupt the
script.
We first create several subfolders as a by-product of writing an empty dummy file into them.

---

## Page 478

Part 4. Common APIs
478
4.5 Working with files
void OnStart()
{
   const string filename = "MQL5Book/ABC/DEF/dummy";
   uchar dummy[];
   PRTF(FileSave(filename, dummy)); // true
Next, we create another folder at the bottom nesting level with FolderCreate: This time the folder
appears on its own, without the helper file.
   PRTF(FolderCreate("MQL5Book/ABC/GHI")); // true
If you try to delete the "DEF" folder, it will fail because it is not empty (there is a file there).
   PRTF(FolderDelete("MQL5Book/ABC/DEF")); // false / CANNOT_DELETE_DIRECTORY(5024)
In order to remove it, you must first clear it, and the easiest way to do this is with FolderClean. But we
will try to simulate a common situation when some files in the folders being cleared can be locked by
other MQL programs, external applications, or the terminal itself. Let's open the file for reading and call
FolderClean.
   int handle = PRTF(FileOpen(filename, FILE_READ)); // 1
   PRTF(FolderClean("MQL5Book/ABC")); // false / CANNOT_CLEAN_DIRECTORY(5025)
The function returns false and exposes error code 5025 (CANNOT_CLEAN_DIRECTORY). After we close
the file, cleaning and deleting the entire folder hierarchy succeeds.
   FileClose(handle);
   PRTF(FolderClean("MQL5Book/ABC")); // true
   PRTF(FolderDelete("MQL5Book/ABC")); // true
}
Potential locks are more likely when using a shared terminal directory, where the same file or folder
can be "claimed" by different program instances. But even in a local sandbox, you should not forget
about possible conflicts (for example, if a csv file is opened in Excel). Implement detailed diagnostics
and error output for the code parts that work with folders, so that the user can notice and fix the
problem.
4.5.1 7 File or folder selection dialog
In the group of functions for working with files and folders, there is one that allows to interactively
request the name of a file or folder, as well as a group of files from the user in order to pass this
information to an MQL program. Calling the FileSelectDialog function causes a standard Windows
window for selecting files and folders to appear in the terminal.
Since the dialog interrupts the execution of the MQL program until it is closed, the function call is
allowed only in two types of MQL programs that are executed in separate threads: EAs and scripts
(see Types of MQL programs). Using this function is prohibited in indicators and services: the
former are executed in the terminal's interface thread (and stopping them would freeze updating
the charts of the corresponding instruments), while the latter are executed in the background and
cannot access the user interface.
All elements of the file system that the function works with are located inside the sandbox, i.e., in the
directory of the current copy of the terminal or testing agent (if the program is running in the tester),
in the subfolder MQL5/Files.

---

## Page 479

Part 4. Common APIs
479
4.5 Working with files
If the FSD_COMMON_FOLDER flag is present in the flags parameter (see further), a common sandbox of
all terminals Users/<user>...MetaQuotes/Terminal/Common/Files is used.
The appearance of the dialog depends on the Windows operating system. One of the possible interface
options is shown below.
Windows file and folder selection dialog
int FileSelectDialog(const string caption, const string initDir, const string filter,
     uint flags, string &filenames[], const string defaultName)
The function displays a standard Windows dialog for opening or creating a file or selecting a folder. The
title is specified in the caption parameter. If the value is NULL, the standard title is used: "Open" for
reading or "Save as" for writing a file, or "Select folder", depending on the flags in the flags parameter.
The initDir parameter allows you to set the initial folder for which the dialog will open. If set to NULL,
the contents of the MQL5/Files folder will be shown. The same folder is used if a non-existent path is
specified in initDir.
Using the filter parameter, you can limit the set of file extensions that will be shown in the dialog box.
Files of other formats will be hidden. NULL means no restrictions.
The format of the filter string is as follows:
"<description 1>|<extension 1>|<description 2>|<extension 2>..."
Any string is allowed as description. You can write any filter with the substituted characters '*' and '?'
that we discussed in the section Finding files and folders as extensions. Symbol '| ' is a delimiter.
Since the adjacent description and extension form a logically related pair, the total number of elements
in the line must be even, and the number of delimiters must be odd.
Each combination of description and extension generates a separate selection in the dialog's drop-down
list. The description is shown to the user and the extension is used for filtering.

---

## Page 480

Part 4. Common APIs
480
4.5 Working with files
For example, "Text documents (*.txt)| *.txt| All files (*.*)| *.*", while the first extension "Text documents
(*.txt)| *.txt" will be selected as the default file type.
In the flags parameter, you can indicate a bit mask specifying the operating modes using the '| '
operator. The following constants are defined for it:
• FSD_WRITE_FILE – file writing mode ("Save as"). In the absence of this flag, the read mode
("Open") is used by default. If this flag is present, the input of an arbitrary new name is always
allowed, regardless of the FSD_FILE_MUST_EXIST flag.
• FSD_SELECT_FOLDER – folder selection mode (only one and only existing). With this flag, all other
flags except FSD_COMMON_FOLDER are ignored or cause an error. You cannot explicitly request
the creation of a folder, but it is possible to create a folder interactively in the dialog and
immediately select it.
• FSD_ALLOW_MULTISELECT – permission to select multiple files in read mode. This flag is ignored if
FSD_WRITE_FILE or FSD_SELECT_FOLDER is specified.
• FSD_FILE_MUST_EXIST – the selected files must exist. If the user tries to specify an arbitrary
name, the dialog will display a warning and remain open. This flag is ignored if FSD_WRITE_FILE
mode is specified.
• FSD_COMMON_FOLDER – the dialog is opened for a common sandbox of all client terminals.
The function will fill an array of strings filenames with the names of the selected files or folder. If the
array is dynamic, its size changes to fit the actual amount of data, in particular, expands or truncates
down to 0 if nothing was selected. If the array is fixed, it must be large enough to accept the expected
data. Otherwise, an error 4007 (ARRAY_RESIZE_ERROR) will occur.
The defaultName parameter specifies the default file/folder name, which will be substituted into the
corresponding input field immediately after opening the dialog. If the parameter is NULL, the field will
be initially empty.
If the defaultName parameter is set, then during non-visual testing of the MQL program,
FileSelectDialog call will return 1  and the defaultName value itself will be copied to the filenames
array.
The function returns the number of items selected (0 if the user didn't select anything), or -1  if there
was an error.
Consider examples of how the function works in the script FileSelect.mq5. In the OnStart function, we
will sequentially call FileSelectDialog with different settings. As long as the user selects something
(doesn't click the "Cancel" button in the dialog), the test continues all the way to the last step (even if
the function executes with an error code).
void OnStart()
{
 string filenames[]; // a dynamic array suitable for any call
 string fixed[1]; // too small array if there are more than 1 files
 const stringfilter = // filter example
      "Text documents (*.txt)|*.txt"
      "|Files with short names|????.*"
      "|All files (*.*)|*.*";
First, we will ask the user for one file from the "MQL5Book" folder. You can select an existing file or
enter a new name (because there is no FSD_FILE_MUST_EXIST flag).

---

## Page 481

Part 4. Common APIs
481 
4.5 Working with files
   Print("Open a file");
   if(PRTF(FileSelectDialog(NULL, "MQL5book", filter, 
      0, filenames, NULL)) == 0) return;             // 1
   ArrayPrint(filenames);                            // "MQL5Book\utf8.txt"
Assuming that the folder contains at least 5 files from the book delivery, one of them is selected here.
Then we will make a similar request in "for writing" mode (with the FSD_WRITE_FILE flag).
   Print("Save as a file");
   if(PRTF(FileSelectDialog(NULL, "MQL5book", NULL, 
      FSD_WRITE_FILE, filenames, NULL)) == 0) return;// 1 
   ArrayPrint(filenames);                            // "MQL5Book\newfile"
Here the user will also be able to select both an existing file and enter a new name. A check of whether
the user is going to overwrite an existing file must be done by the programmer (the dialog does not
generate warnings).
Now let's check the selection of multiple files (FSD_ALLOW_MULTISELECT) in a dynamic array.
   if(PRTF(FileSelectDialog(NULL, "MQL5book", NULL, 
     FSD_FILE_MUST_EXIST | FSD_ALLOW_MULTISELECT, filenames, NULL)) == 0) return; // 5
   ArrayPrint(filenames);
   // "MQL5Book\ansi1252.txt" "MQL5Book\unicode1.txt" "MQL5Book\unicode2.txt"
   // "MQL5Book\unicode3.txt" "MQL5Book\utf8.txt"
The presence of the FSD_FILE_MUST_EXIST flag means that the dialog will display a warning and
remain open if you try to enter a new name.
If we try to select more than one file in a fixed-size array in a similar way, we will get an error.
   Print("Open multiple files (fixed, choose more than 1 file for error)");
   if(PRTF(FileSelectDialog(NULL, "MQL5book", NULL, 
      FSD_FILE_MUST_EXIST | FSD_ALLOW_MULTISELECT, fixed, NULL)) == 0) return;
   // -1 / ARRAY_RESIZE_ERROR(4007)
   ArrayPrint(fixed); // null
Finally, let's check folder operations (FSD_SELECT_FOLDER).
   Print("Select a folder");
   if(PRTF(FileSelectDialog(NULL, "MQL5book/nonexistent", NULL, 
      FSD_SELECT_FOLDER, filenames, NULL)) == 0) return; // 1
   ArrayPrint(filenames); // "MQL5Book"
In this case, the non-existent subfolder "nonexistent" is specified as the start path, so the dialog will
open in the root of the sandbox MQL5/Files. There we chose "MQL5book".
If we combine an invalid combination of flags, we get another error.

---

## Page 482

Part 4. Common APIs
482
4.5 Working with files
   if(PRTF(FileSelectDialog(NULL, "MQL5book", NULL, 
      FSD_SELECT_FOLDER | FSD_WRITE_FILE, filenames, NULL)) == 0) return;
   // -1 / INTERNAL_ERROR(4001)
   ArrayPrint(filenames); // "MQL5Book"
}
Due to an error, the function did not modify the passed array, and the old "MQL5Book" element
remained in it.
In this test, we deliberately checked the results only for 0 in order to demonstrate all options,
regardless of the presence of errors. In a real program, check the result of the function taking into
account errors, i.e. with conditions for three outcomes: choice made (>0), choice not made (==0), and
error (<0).
4.6 Client terminal global variables
In the previous chapter, we studied MQL5 functions that work with files. They provide wide, flexible
options for writing and reading arbitrary data. However, sometimes an MQL program needs an easier
way to save and restore the state of an attribute between runs.
For example, we want to calculate certain statistics: how many times the program was launched, how
many instances of it are executed in parallel on different charts, etc. It is impossible to accumulate this
information within the program itself. There must be some kind of external long-term storage. But it
would be expensive to create a file for this, though it is also feasible.
Many programs are designed to interact with each other, i.e., they must somehow exchange
information. If we are talking about integration with a program external to the terminal, or about
transferring a large amount of data, then it is really difficult to do it without using files. However, when
there is not enough data to be sent, and all programs are written in MQL5 and run inside MetaTrader 5,
the use of files seems redundant. The terminal provides a simpler technology for this case: global
variables.
A global variable is a named location in the terminal's shared memory. It can be created, modified, or
deleted by any MQL program, but will not belong to it exclusively, and is available to all other MQL
programs. The name of a global variable is any unique (among all variables) string of no more than 63
characters. This string does not have to meet the requirements for variable identifiers in MQL5, since
global variables of the terminal are not variables in the usual sense. The programmer does not define
them in the source code according to the syntax we learned in Variables, they are not an integral part
of the MQL program, and any action with them is performed only by calling one of the special functions
that we will describe in this chapter.
The global variables allow you to store only values of type double. If necessary, you can pack/convert
values of other types to double or use part of the variable name (following a certain prefix, for example)
to store strings.
While the terminal is running, global variables are stored in RAM and are available almost instantly: the
only overhead is associated with function calls. This definitely gives a headstart to global variables
against using files, since when dealing with the latter, obtaining a handle is a relatively slow process,
and the handle itself consumes some additional resources.
At the end of the terminal session, global variables are unloaded into a special file (gvariables.dat) and
then restored from it the next time you run the terminal.

---

## Page 483

Part 4. Common APIs
483
4.6 Client terminal global variables
A particular global variable is automatically destroyed by the terminal if it has not been claimed within
4 weeks. This behavior relies on keeping track of and storing the time of the last use of a variable,
where use refers to setting a new value or reading an old one (but not checking for existence or getting
the time of last use).
Please note that global variables are not tied to an account, profile, or any other characteristics of
the trading environment. Therefore, if they are supposed to store something related to the
environment (for example, some general limits for a particular account), variable names should be
constructed taking into account all factors that affect the algorithm and decision-making. To
distinguish between global variables of multiple instances of the same Experts Advisor (EA), you may
need to add a working symbol, timeframe, or "magic number" from the EA settings to the name.
In addition to MQL programs, global variables can also be manually created by the user. The list of
existing global variables, as well as the means of their interactive management, can be found in the
dialog opened in the terminal by the command Tools -> Global Variables (F3).
By using the corresponding buttons here you can Add and Delete global variables, and double-clicking in
columns Variable or Meaning allows you to edit the name or value of a particular variable. The following
hotkeys work from the keyboard: F2 for name editing, F3 for value editing, Ins for adding a new
variable, Del for deleting the selected variable.
A little later, we will study two main types of MQL programs – Expert Advisors and Indicators. Their
special feature is the ability to run in the tester, where functions for global variables also work.
However, global variables are created, stored, and managed by the tester agent in the tester. In other
words, the lists of terminal global variables are not available in the tester, and those variables that are
created by the program under test belong to a specific agent, and their lifetime is limited to one test
pass. That is, the agent's global variables are not visible from other agents and will be removed at the
end of the test run. In particular, if the EA is optimized on several agents, it can manipulate global
variables to "communicate" with the indicators it uses in the context of the same agent since they are
executed there together, but on parallel agents, other copies of the EA will form their own lists of
variables.
Data exchange between MQL programs using global variables is not the only available, and not
always the most appropriate way. In particular, EAs and indicators are interactive types of MQL
programs that can generate and accept events on charts. You can pass various types of information
in event parameters. In addition, arrays of calculated data can be prepared and provided to other
MQL programs in the form of indicator buffers. MQL programs located on charts can use UI graphic
objects to transfer and store information.
From the technical point of view, the maximum number of global variables is limited only by the
resources of the operating system. However, for a large number of elements, it is recommended to use
more suitable means: files or databases.
4.6.1  Writing and reading global variables
The MQL5 API provides 2 functions to write and read global variables: GlobalVariableSet and
GlobalVariableGet (in two versions).
datetime GlobalVariableSet(const string name, double value)
The function sets a new value to the 'name' global variable. If the variable did not exist before the
function was called, it will be created. If the variable already exists, the previous value will be replaced
by value.

---

## Page 484

Part 4. Common APIs
484
4.6 Client terminal global variables
If successful, the function returns the variable modification time (the current local time of the
computer). In case of an error, we get 0.
double GlobalVariableGet(const string name)
bool GlobalVariableGet(const string name, double &value)
The function returns the value of the 'name' global variable. The result of calling the first version of the
function contains just the value of the variable (in case of success) or 0 (in case of error). Since the
variable can contain the value of 0 (which is the same as an error indication), this option requires
parsing the internal error code _ LastError if zero is received, to distinguish the standard version from
the non-standard one. In particular, if an attempt is made to read a variable that does not exist, an
internal error 4501  (GLOBALVARIABLE_NOT_FOUND) is generated.
This function version is convenient to use in algorithms where getting zero is a suitable analog of the
default initialization for a previously nonexistent variable (see example below). If the absence of a
variable needs to be handled in a special way (in particular, to calculate some other starting value), you
should first check the existence of the variable using the GlobalVariableCheck function and, depending
on its result, execute different code branches. Optionally, you can use the second version.
The second version of the function returns true or false depending on the success of the execution. If
successful, the value of the global variable of the terminal is placed in the receiving value variable,
passed by reference as the second parameter. If there is no variable, we get false.
In the test script GlobalsRunCount.mq5, we use a global variable to count the number of times it ran.
The name of the variable is the name of the source file.
const string gv = __FILE__;
Recall that the built-in macro __FILE__ (see Predefined constants) is expanded by the compiler into the
name of the compiled file, i.e., in this case, "GlobalsRunCount.mq5".
In the OnStart function, we will try to read the given global variable and save the result in the local
count variable. If there was no global variable yet, we get 0, which is okay for us (we start counting
from zero).
Before saving the value in count, it is necessary to typecast it to (int), since the GlobalVariableGet
function returns double, and without the cast, the compiler generates a warning about potential data
loss (it doesn't know that we plan to store only integers).
void OnStart()
{
   int count = (int)PRTF(GlobalVariableGet(gv));
   count++;
   PRTF(GlobalVariableSet(gv, count));
   Print("This script run count: ", count);
}
Then we increment the counter by 1  and write it back to the global variable with GlobalVariableSet. If
we run the script several times, we will get something like the following log entries (your timestamps
will be different):

---

## Page 485

Part 4. Common APIs
485
4.6 Client terminal global variables
GlobalVariableGet(gv)=0.0 / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalVariableSet(gv,count)=2021.08.29 16:04:40 / ok
This script run count: 1
GlobalVariableGet(gv)=1.0 / ok
GlobalVariableSet(gv,count)=2021.08.29 16:05:00 / ok
This script run count: 2
GlobalVariableGet(gv)=2.0 / ok
GlobalVariableSet(gv,count)=2021.08.29 16:05:21 / ok
This script run count: 3
It is important to note that on the first run, we received a value of 0, and the internal error flag 4501 
was generated. All subsequent calls are executed without problems since the variable exists (it can be
seen in the "Global Variables" window of the terminal). Those who wish may close the terminal, restart
it and execute the script again: the counter will continue to increase from the previous value.
4.6.2 Checking the existence and last activity time
As we saw in the previous section, you can check the existence of a global variable by trying to read its
value: if this does not result in an error code in _ LastError, then the global variable exists, and we have
already obtained its value and can use it in the algorithm. However, if under some conditions you only
need to check for the existence, but not read the global variable, it is more convenient to use another
function specifically designed for this: GlobalVariableCheck.
There is another way to check, namely, using the GlobalVariableTime function. As its name implies, it
allows you to find out the last time a variable was used. But if the variable does not exist, then the time
of its use is absent, i.e., it is equal to 0.
bool GlobalVariableCheck(const string name)
The function checks for the existence of a global variable with the specified name and returns the
result: true (the variable exists) or false (no variable).
datetime GlobalVariableTime(const string name)
The function returns the time the global variable with the specified name was last used. The fact of use
can be represented by the modification or reading of the variable value.
Checking for the variable existence with GlobalVariableCheck or getting its time through
GlobalVariableTime do not change the time of use.
In the script GlobalsRunCheck.mq5, we will slightly supplement the code from GlobalsRunCount.mq5 so
that at the very beginning of the function OnStart check for the presence of a variable and the time of
its use.
void OnStart()
{
   PRTF(GlobalVariableCheck(gv));
   PRTF(GlobalVariableTime(gv));
   ...
The code below is unchanged. Meanwhile, note that the gv variable defined via __FILE__ will this time
contain the new script name "GlobalsRunCheck.mq5" as the name of the global variable (i.e., each
script has its own global counter).

---

## Page 486

Part 4. Common APIs
486
4.6 Client terminal global variables
All runs except the very first one will show true from the GlobalVariableCheck function (the variable
exists) and the time of the variable from the previous run. Here is an example log:
GlobalVariableCheck(gv)=false / ok
GlobalVariableTime(gv)=1970.01.01 00:00:00 / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalVariableGet(gv)=0.0 / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalVariableSet(gv,count)=2021.08.29 16:59:35 / ok
This script run count: 1
GlobalVariableCheck(gv)=true / ok
GlobalVariableTime(gv)=2021.08.29 16:59:35 / ok
GlobalVariableGet(gv)=1.0 / ok
GlobalVariableSet(gv,count)=2021.08.29 16:59:45 / ok
This script run count: 2
GlobalVariableCheck(gv)=true / ok
GlobalVariableTime(gv)=2021.08.29 16:59:45 / ok
GlobalVariableGet(gv)=2.0 / ok
GlobalVariableSet(gv,count)=2021.08.29 16:59:56 / ok
This script run count: 3
4.6.3 Getting a list of global variables
Quite often, an MQL program is required to look through the existing global variables and select those
meeting some criteria. For example, if a program uses part of a variable name to store textual
information, then only the prefix is known in advance. The purpose of this prefix is to identify "its own"
variable, and the "payload" attached to the prefix does not allow searching for a variable by the exact
name.
The MQL5 API has two functions that allow you to enumerate global variables.
int GlobalVariablesTotal()
The function returns the total number of global variables.
string GlobalVariableName(int index)
The function returns the name of the global variable by its index number in the list of global variables.
The index parameter with the number of the requested variable must be in the range from 0 to
GlobalVariablesTotal() - 1 .
In case of an error, the function will return NULL, and the error code can be obtained from the service
variable _ LastError or the GetLastError function.
Let's test this pair of functions using the script GlobalsList.mq5.

---

## Page 487

Part 4. Common APIs
487
4.6 Client terminal global variables
void OnStart()
{
   PRTF(GlobalVariableName(1000000));
   int n = PRTF(GlobalVariablesTotal());
   for(int i = 0; i < n; ++i)
   {
      const string name = GlobalVariableName(i);
      PrintFormat("%d %s=%f", i, name, GlobalVariableGet(name));
   }
}
The first string deliberately asks for the name of a variable with a large number, which, most likely,
does not exist, and that fact should cause an error. Next, a request is made for the real number of
variables and a loop through all of them, with the output of the name and value. The log below includes
variables created by previous test scripts and one third-party variable.
GlobalVariableName(1000000)= / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalVariablesTotal()=3 / ok
0 GlobalsRunCheck.mq5=3.000000
1 GlobalsRunCount.mq5=4.000000
2 abracadabra=0.000000
The order in which the terminal returns variables by an index is not defined.
4.6.4 Deleting global variables
If necessary, an MQL program can delete a global variable or a group of them that has become
redundant. The list of global variables consumes some computer resources, and the good programming
style suggests that resources should be freed whenever possible.
bool GlobalVariableDel(const string name)
The function removes the 'name' global variable. On success, the function returns true, otherwise
returns false.
int GlobalVariablesDeleteAll(const string prefix = NULL, datetime limit = 0)
The function deletes global variables with the specified prefix in the name and with a usage time older
than the limit parameter value.
If the NULL prefix (default) or an empty string is specified, then all global variables that also match the
deletion criterion by date (if it's set) fall under the deletion criterion.
If the limit parameter is zero (default), then global variables with any date taking into account the
prefix are deleted.
If both parameters are specified, then global variables that match both, the prefix and the time
criterion, are deleted.
Be careful: calling GlobalVariablesDeleteAll without parameters will remove all variables.
The function returns the number of deleted variables.
Consider the script GlobalsDelete.mq5, exploiting two new features.

---

## Page 488

Part 4. Common APIs
488
4.6 Client terminal global variables
void OnStart()
{
   PRTF(GlobalVariableDel("#123%"));
   PRTF(GlobalVariablesDeleteAll("#123%"));
   ...
In the beginning, an attempt is made to delete non-existent global variables by their exact name and
prefix. Both have no effect on existing variables.
Calling GlobalVariablesDeleteAll with a filter by time in the past (more than 4 weeks ago) also has a
zero result, because the terminal deletes such old variables automatically (such variables cannot exist).
   PRTF(GlobalVariablesDeleteAll(NULL, D'2021.01.01'));
Then, we create a variable with the name "abracadabra" (if it did not exist) and immediately delete it.
These calls should succeed.
   PRTF(GlobalVariableSet(abracadabra, 0));
   PRTF(GlobalVariableDel(abracadabra));
Finally, let's delete the variables starting with the "GlobalsRun" prefix: they should have been created
by the test scripts from the two previous sections on file names (respectively, "GlobalsRunCount.mq5"
and "GlobalsRunCheck.mq5").
   PRTF(GlobalVariablesDeleteAll("GlobalsRun"));
   PRTF(GlobalVariablesTotal());
}
The script should output something like the following set of strings to the log (some indicators depend
on external conditions and startup time).
GlobalVariableDel(#123%)=false / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalVariablesDeleteAll(#123%)=0 / ok
GlobalVariablesDeleteAll(NULL,D'2021.01.01')=0 / ok
GlobalVariableSet(abracadabra,0)=2021.08.30 14:02:32 / ok
GlobalVariableDel(abracadabra)=true / ok
GlobalVariablesDeleteAll(GlobalsRun)=2 / ok
GlobalVariablesTotal()=0 / ok
In the end, we printed out the total number of remaining global variables (in this case, we got 0, i.e.,
there are no variables). It may differ for you if the global variables were created by other MQL
programs or by the user.
4.6.5 Temporary global variables
In the global variables subsystem of the terminal, it is possible to make some variables temporary: they
are stored only in memory and are not written to disk when the terminal is closed.
Due to their specific nature, temporary global variables are used exclusively for data exchange between
MQL programs and are not suitable for saving states between MetaTrader 5 launches. One of the most
obvious uses for temporary variables is various metrics of operational activity (for example, counters of
running program copies) that should be dynamically recalculated on every startup, rather than being
restored from disk.

---

## Page 489

Part 4. Common APIs
489
4.6 Client terminal global variables
A global variable should be declared temporary in advance, before assigning any value to it, using the
function GlobalVariableTemp.
Unfortunately, it is impossible to find out by the name of a global variable whether it is temporary:
MQL5 does not provide means for this.
Temporary variables can only be created using MQL programs. Temporary variables are displayed in
the "Global Variables" window along with ordinary (persistent) global variables, but the user does not
have the ability to add their own temporary variable from the GUI.
bool GlobalVariableTemp(const string name)
The function creates a new global variable with the specified name, which will exist only until the end of
the current terminal session.
If a variable with the same name already exists, it will not be converted to a temporary variable.
However, if a variable does not exist yet, it will get the value 0. After that, you can work with it as
usual, in particular, assign other values using the GlobalVariableSet function.
We will show an example of this function along with the functions of the next section.
4.6.6 Synchronizing programs using global variables
Since global variables exist outside of MQL programs, they are useful for organizing external flags that
control multiple copies of the same program or pass signals between different programs. The simplest
example is to limit the number of copies of a program that can be run. This may be necessary to
prevent accidental duplication of the Expert Advisor on different charts (due to which trade orders may
double), or to implement a demo version.
At first glance, such a check could be done in the source code as follows.
void OnStart()
{
   const string gv = "AlreadyRunning";
   // if the variable exists, then one instance is already running
   if(GlobalVariableCheck(gv)) return;
   // create a variable as a flag signaling the presence of a working copy
   GlobalVariableSet(gv, 0);
   
   while(!IsStopped())
   {
       // work cycle
   }
   // delete variable before exit
   GlobalVariableDel(gv);
}
The simplest version is shown here using a script as an example. For other types of MQL programs, the
general concept of checking will be the same, although the location of instructions may differ: instead
of an endless work cycle, Expert Advisors and indicators use their characteristic event handlers
repeatedly called by the terminal. We will study these problems later.

---

## Page 490

Part 4. Common APIs
490
4.6 Client terminal global variables
The problem with the presented code is that it does not take into account the parallel execution of MQL
programs.
An MQL program usually runs in its own thread. For three out of four types of MQL programs, namely
for Expert Advisors, scripts, and services, the system definitely allocates separate threads. As for
indicators, one common thread is allocated to all their instances, working on the same combination of
working symbol and timeframe. But indicators on different combinations still belong to different
threads.
Almost always, a lot of threads are running in the terminal – much more than the number of processor
cores. Because of this, each thread from time to time is suspended by the system to allow other
threads to work. Since all such switching between threads happens very quickly, we, as users, do not
notice this "inner organization". However, each suspension can affect the sequence in which different
threads access the shared resources. Global variables are such resources.
From the program's point of view, a pause can occur between any adjacent instructions. If knowing
this, we look again at our example, it is not difficult to see a place where the logic of working with a
global variable can be broken.
Indeed, the first copy (thread) can perform a check and find no variable but be immediately suspended.
As a result, before it has time to create the variable with its next instruction, the execution context
switches to the second copy. That one also won't find the variable and will decide to continue working,
like the first one. For clarity, the identical source code of the two copies is shown below as two columns
of instructions in the order of their interleaved execution.
Copy 1
Copy 2
void OnStart()              
{                                      
   const string gv = "AlreadyRunning"; 
                                       
   if(GlobalVariableCheck(gv)) return; 
   // no variable
                                       
   GlobalVariableSet(gv, 0);           
   // "I am the first and only"
   while(!IsStopped())                 
                                       
   {                                   
      ;                                
                                       
   }                                   
   GlobalVariableDel(gv);              
                                       
}                                      
void OnStart()              
{                                      
                                       
   const string gv = "AlreadyRunning"; 
                                       
   if(GlobalVariableCheck(gv)) return; 
   // still no variable
                                       
   GlobalVariableSet(gv, 0);           
   // "No, I'm the first and only one"
   while(!IsStopped())                 
   {                                   
                                       
      ;                                
   }                                   
                                       
   GlobalVariableDel(gv);              
}                                      
Of course, such a scheme for switching between threads has a fair amount of conventionality. But in
this case, the very possibility of violating the logic of the program is important, even in one single
string. When there are many programs (threads), the probability of unforeseen actions with common
resources increases. This may be enough to take the EA to a loss at the most unexpected moment or
to get distorted technical analysis estimates.

---

## Page 491

Part 4. Common APIs
491 
4.6 Client terminal global variables
The most frustrating thing about errors of this kind is that they are very difficult to detect. The
compiler is not able to detect them, and they manifest themselves sporadically at runtime. But if the
error does not reveal itself for a long time, this does not mean that there is no error.
To solve such problems, it is necessary to somehow synchronize the access of all copies of programs to
shared resources (in this case, to global variables).
In computer science, there is a special concept – a mutex (mutual exclusion) – which is an object for
providing exclusive access to a shared resource from parallel programs. A mutex prevents data from
being lost or corrupted due to asynchronous changes. Usually, accessing a mutex synchronizes
different programs due to the fact that only one of them can edit protected data by capturing the
mutex at a particular moment, and the rest are forced to wait until the mutex is released.
There are no ready-made mutexes in MQL5 in their pure form. But for global variables, a similar effect
can be obtained by the following function, which we will consider.
bool GlobalVariableSetOnCondition(const string name, double value, double precondition)
The function sets a new value of the existing global variable name provided that its current value is
equal to precondition.
On success, the function returns true. Otherwise, it returns false, and the error code will be available in
_ LastError. In particular, if the variable does not exist, the function will generate an
ERR_GLOBALVARIABLE_NOT_FOUND (4501 ) error.
The function provides atomic access to a global variable, that is, it performs two actions in an
inseparable way: it checks its current value, and if it matches the condition, it assigns to the variable a
new value.
The equivalent function code can be represented approximately as follows (why it is "approximately" we
will explain later):
bool GlobalVariableZetOnCondition(const string name, double value, double precondition)
{
   bool result = false;
   { /* enable interrupt protection */ }
   if(GlobalVariableCheck(name) && (GlobalVariableGet(name) == precondition))
   {
      GlobalVariableSet(name, value);
      result = true;
   }
   { /* disable interrupt protection */ }
   return result;
}
Implementing code like this, which works as intended, is impossible for two reasons. First, there is
nothing to implement blocks that enable and disable interrupt protection in pure MQL5 (inside the built-
in GlobalVariableSetOnCondition function this is provided by the kernel itself). Second, the
GlobalVariableGet function call changes the last time the variable was used, while the
GlobalVariableSetOnCondition function does not change it if the precondition was not met.
To demonstrate how to use GlobalVariableSetOnCondition, we will turn to a new MQL program type:
services. We will study them in detail in a separate section. For now, it should be noted that their
structure is very similar to scripts: for both, there is only one main function (entry point), OnStart. The

---

## Page 492

Part 4. Common APIs
492
4.6 Client terminal global variables
only significant difference is that the script runs on the chart, while the service runs by itself (in the
background).
The need to replace scripts with services is explained by the fact that the applied meaning of the task
in which we use GlobalVariableSetOnCondition, consists in counting the number of running instances of
the program, with the possibility of setting a limit. In this case, collisions with simultaneous modification
of the shared counter can occur only at the moment of launching multiple programs. However, with
scripts, it is quite difficult to run several copies of them on different charts in a relatively short period of
time. For services, on the contrary, the terminal interface has a convenient mechanism for batch
(group) launch. In addition, all activated services will automatically start at the next boot of the
terminal.
The proposed mechanism for counting the number of copies will also be in demand for MQL programs of
other types. Since Expert Advisors and indicators remain attached to the charts even when the terminal
is turned off, the next time it is turned on, all programs read their settings and shared resources almost
simultaneously. Therefore, if a limit on the number of copies is built into some Expert Advisors and
indicators, it is critical to synchronize the counting based on global variables.
First, let's consider a service that implements copy control in a naive mode, without using
GlobalVariableSetOnCondition, and make sure that the problem of counter failures is real. The services
are located in a dedicated subdirectory in the general source code directory, so here is the expanded
path − MQL5/Services/MQL5Book/p4/GlobalsNoCondition.mq5.
At the beginning of the service file there should be a directive:
#property service
In the service, we will provide 2 input variables to set a limit on the number of allowed copies running in
parallel and a delay to emulate execution interruption due to a massive load on the disk and CPU of the
computer, which often happens when the terminal is launched. This will make it easier to reproduce the
problem without having to restart the terminal many times hoping to get out of sync. So, we are going
to catch a bug that can only occur sporadically, but at the same time, if it happens, it is fraught with
serious consequences.
input int limit = 1;       // Limit
input int startPause = 100;// Delay(ms)
Delay emulation is based on the Sleep function.
void Delay()
{
   if(startPause > 0)
   {
      Sleep(startPause);
   }
}
First of all, a temporary global variable is declared inside the OnStart function. Since it is designed to
count running copies of the program, it makes no sense to make it constant: every time you start the
terminal, you need to count again.

---

## Page 493

Part 4. Common APIs
493
4.6 Client terminal global variables
void OnStart()
{
   PRTF(GlobalVariableTemp(__FILE__));
   ...
To avoid the case when a user creates a variable of the same name in advance and assigns a negative
value to it, we introduce protection.
   int count = (int)GlobalVariableGet(__FILE__);
   if(count < 0)
   {
      Print("Negative count detected. Not allowed.");
      return;
   }
Next, the fragment with the main functionality begins. If the counter is already greater than or equal to
the maximum allowable quantity, we interrupt the program launch.
   if(count >= limit)
   {
      PrintFormat("Can't start more than %d copy(s)", limit);
      return;
   }
Otherwise, we increase the counter by 1  and write it to the global variable. In advance, we emulate the
delay in order to provoke a situation when another program could intervene between reading a variable
and writing it in our program.
   Delay();
   PRTF(GlobalVariableSet(__FILE__, count + 1));
If this really happens, our copy of the program will increment and assign an already obsolete, incorrect
value. It will result in a situation where in another copy of the program running in parallel with ours, the
same count value has already been processed or will be processed again.
The useful work of the service is represented by the following loop.
   int loop = 0;
   while(!IsStopped())
   {
      PrintFormat("Copy %d is working [%d]...", count, loop++);
      // ...
      Sleep(3000);
   }
After the user stops the service (for this, the interface has a context menu; more on that will follow),
the cycle will end, and we need to decrement the counter.

---

## Page 494

Part 4. Common APIs
494
4.6 Client terminal global variables
   int last = (int)GlobalVariableGet(__FILE__);
   if(last > 0)
   {
      PrintFormat("Copy %d (out of %d) is stopping", count, last);
      Delay();
      PRTF(GlobalVariableSet(__FILE__, last - 1));
   }
   else
   {
      Print("Count underflow");
   }
}
Compiled services fall into the corresponding branch of the "Navigator".
Services in the "N avigator" and their context menu
By right-clicking, we will open the context menu and create two instances of the service
GlobalsNoCondition.mq5 by calling the Add service command twice. In this case, each time a dialog will
open with the service settings, where you should leave the default values for the parameters.
It is important to note that the Add service command starts the created service immediately. But
we don't need this. Therefore, immediately after launching each copy, we have to call the context
menu again and execute the Stop command (if a specific instance is selected), or Stop everything
(if the program, i.e., the entire group of generated instances, is selected).
The first instance of the service will by default have a name that completely matches the service file
("GlobalsNoCondition"), and in all subsequent instances, an incrementing number will be automatically
added. In particular, the second instance is listed as "GlobalsNoCondition 1 ". The terminal allows you
to rename instances to arbitrary text using the Rename command, but we won't do that.
Now everything is ready for the experiment. Let's try to run two instances at the same time. To do
this, let's run the Run All command for the corresponding GlobalsNoCondition branch.

---

## Page 495

Part 4. Common APIs
495
4.6 Client terminal global variables
Let's remind that a limit of 1  instance was set in the parameters. However, according to the logs, it
didn't work.
GlobalsNoCondition    GlobalVariableTemp(GlobalsNoCondition.mq5)=true / ok
GlobalsNoCondition 1  GlobalVariableTemp(GlobalsNoCondition.mq5)=false / GLOBALVARIABLE_EXISTS(4502)
GlobalsNoCondition    GlobalVariableSet(GlobalsNoCondition.mq5,count+1)=2021.08.31 17:47:17 / ok
GlobalsNoCondition    Copy 0 is working [0]...
GlobalsNoCondition 1  GlobalVariableSet(GlobalsNoCondition.mq5,count+1)=2021.08.31 17:47:17 / ok
GlobalsNoCondition 1  Copy 0 is working [0]...
GlobalsNoCondition    Copy 0 is working [1]...
GlobalsNoCondition 1  Copy 0 is working [1]...
GlobalsNoCondition    Copy 0 is working [2]...
GlobalsNoCondition 1  Copy 0 is working [2]...
GlobalsNoCondition    Copy 0 is working [3]...
GlobalsNoCondition 1  Copy 0 is working [3]...
GlobalsNoCondition    Copy 0 (out of 1) is stopping
GlobalsNoCondition    GlobalVariableSet(GlobalsNoCondition.mq5,last-1)=2021.08.31 17:47:26 / ok
GlobalsNoCondition 1  Count underflow
Both copies "think" that they are number 0 (output "Copy 0" out of the work loop) and their total
number is erroneously equal to 1  because that is the value that both copies have stored in the counter
variable.
It is because of this that when services are stopped (the Stop everything command), we received a
message about an incorrect state ("Count underflow"): after all, each of the copies is trying to
decrease the counter by 1 , and as a result, the one that was executed second received a negative
value.
To solve the problem, you need to use the GlobalVariableSetOnCondition function. Based on the source
code of the previous service, an improved version GlobalsWithCondition.mq5 was prepared. In general, it
reproduces the logic of its predecessor, but there are significant differences.
Instead of just calling GlobalVariableSet to increase the counter, a more complex structure had to be
written.

---

## Page 496

Part 4. Common APIs
496
4.6 Client terminal global variables
   const int maxRetries = 5;
   int retry = 0;
   
   while(count < limit && retry < maxRetries)
   {
      Delay();
      if(PRTF(GlobalVariableSetOnCondition(__FILE__, count + 1, count))) break;
      // condition is not met (count is obsolete), assignment failed,
      // let's try again with a new condition if the loop does not exceed the limit
      count = (int)GlobalVariableGet(__FILE__);
      PrintFormat("Counter is already altered by other instance: %d", count);
      retry++;
   }
   
   if(count == limit || retry == maxRetries)
   {
      PrintFormat("Start failed: count: %d, retries: %d", count, retry);
      return;
   }
   ...
Since the GlobalVariableSetOnCondition function may not write a new counter value, if the old one is
already obsolete, we read the global variable again in the loop and repeat attempts to increment it until
the maximum allowable counter value is exceeded. The loop condition also limits the number of
attempts. If the loop ends with a violation of one of the conditions, then the counter update failed, and
the program should not continue to run.
Synchronization strategies
 In theory, there are several standard strategies for implementing shared resource capture.
The first is to soft-check if the resource is free and then lock it only if it is free at that moment. If it
is busy, the algorithm plans the next attempt after a certain period, and at this time it is engaged in
other tasks (which is why this approach is preferable for programs that have several areas of
activity/responsibility). An analog of this scheme of behavior in the transcription for the
GlobalVariableSetOnCondition function is a single call, without a loop, exiting the current block on
failure. Variable change is postponed "until better times".
The second strategy is more persistent, and it is applied in our script. This is a loop that repeats a
request for a resource for a given number of times, or a predefined time (the allowable timeout
period for the resource). If the loop expires and a positive result is not reached (calling the function
GlobalVariableSetOnCondition never returned true), the program also exits the current block and
probably plans to try again later.
Finally, the third strategy, the toughest one, involves requesting a resource "to the bitter end". It
can be thought of as an infinite loop with a function call. This approach makes sense to use in
programs that are focused on one specific task and cannot continue to work without a seized
resource. In MQL5, use the loop while(!IsStopped()) for this and don't forget to call Sleep inside.
It's important to note here the potential problem with "hard" grabbing multiple resources. Imagine
that an MQL program modifies several global variables (which is, in theory, a common situation). If
one copy of it captures one variable, and the second copy captures another, and both will wait for

---

## Page 497

Part 4. Common APIs
497
4.6 Client terminal global variables
the release, their mutual blocking (deadlock) will come.
Based on the foregoing, sharing of global variables and other resources (for example, files) should
be carefully designed and analyzed for locks and the so-called "race conditions", when the parallel
execution of programs leads to an undefined result (depending on the order of their work).
After the completion of the work cycle in the new version of the service, the counter decrement
algorithm has been changed in a similar way.
   retry = 0;
   int last = (int)GlobalVariableGet(__FILE__);
   while(last > 0 && retry < maxRetries)
   {
      PrintFormat("Copy %d (out of %d) is stopping", count, last);
      Delay();
      if(PRTF(GlobalVariableSetOnCondition(__FILE__, last - 1, last))) break;
      last = (int)GlobalVariableGet(__FILE__);
      retry++;
   }
   
   if(last <= 0)
   {
      PrintFormat("Unexpected exit: %d", last);
   }
   else
   {
      PrintFormat("Stopped copy %d: count: %d, retries: %d", count, last, retry);
   }
As an experiment, let's create three instances for the new service. In the settings of each of them, in
the Limit parameter, we specify 2 instances (to conduct a test under changed conditions). Recall that
creating each instance immediately launches it, which we do not need, and therefore each newly
created instance should be stopped.
The instances will get the default names "GlobalsWithCondition", "GlobalsWithCondition 1 ", and
"GlobalsWithCondition 2".
When everything is ready, we run all instances at once and get something like this in the log.

---

## Page 498

Part 4. Common APIs
498
4.6 Client terminal global variables
GlobalsWithCondition 2  GlobalVariableTemp(GlobalsWithCondition.mq5)= »
                        » false / GLOBALVARIABLE_EXISTS(4502)
GlobalsWithCondition 1  GlobalVariableTemp(GlobalsWithCondition.mq5)= »
                        » false / GLOBALVARIABLE_EXISTS(4502)
GlobalsWithCondition    GlobalVariableTemp(GlobalsWithCondition.mq5)=true / ok
GlobalsWithCondition    GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,count+1,count)= »
                        » true / ok
GlobalsWithCondition 1  GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,count+1,count)= »
                        » false / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalsWithCondition 2  GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,count+1,count)= »
                        » false / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalsWithCondition 1  Counter is already altered by other instance: 1
GlobalsWithCondition    Copy 0 is working [0]...
GlobalsWithCondition 2  Counter is already altered by other instance: 1
GlobalsWithCondition 1  GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,count+1,count)=true / ok
GlobalsWithCondition 1  Copy 1 is working [0]...
GlobalsWithCondition 2  GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,count+1,count)= »
                        » false / GLOBALVARIABLE_NOT_FOUND(4501)
GlobalsWithCondition 2  Counter is already altered by other instance: 2
GlobalsWithCondition 2  Start failed: count: 2, retries: 2
GlobalsWithCondition    Copy 0 is working [1]...
GlobalsWithCondition 1  Copy 1 is working [1]...
GlobalsWithCondition    Copy 0 is working [2]...
GlobalsWithCondition 1  Copy 1 is working [2]...
GlobalsWithCondition    Copy 0 is working [3]...
GlobalsWithCondition 1  Copy 1 is working [3]...
GlobalsWithCondition    Copy 0 (out of 2) is stopping
GlobalsWithCondition    GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,last-1,last)=true / ok
GlobalsWithCondition    Stopped copy 0: count: 2, retries: 0
GlobalsWithCondition 1  Copy 1 (out of 1) is stopping
GlobalsWithCondition 1  GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,last-1,last)=true / ok
GlobalsWithCondition 1  Stopped copy 1: count: 1, retries: 0
First of all, pay attention to the random, but at the same time visual demonstration of the described
effect of context switching for parallel running programs. The first instance that created a temporary
variable was "GlobalsWithCondition" without a number: this can be seen from the result of the function
GlobalVariableTemp which is true. However, in the log, this line occupies only the third position, and the
two previous ones contain the results of calling the same function in copies under the names with
numbers 1  and 2; in those the function GlobalVariableTemp returned false. This means that these
copies checked the variable later, although their threads then overtook the unnumbered
"GlobalsWithCondition" thread and ended up in the log earlier.
But let's get back to our main program counting algorithm. The instance "GlobalsWithCondition" was
the first to pass the check, and started working under the internal identifier "Copy 0" (we cannot find
out from the service code how the user named the instance: there is no such function in the MQL5 API,
at least not at the moment).
Thanks to the function GlobalVariableSetOnCondition, in instances 1  and 2 ("GlobalsWithCondition 1 ",
"GlobalsWithCondition 2"), the fact of modifying the counter was detected: it was 0 at the start, but
GlobalsWithCondition increased it by 1 . Both late instances output the message "Counter is already
altered by other instance: 1 ". One of these instances ("GlobalsWithCondition 1 ") ahead of number 2,
managed to get a new value of 1  from the variable and increase it to 2. This is indicated by a
successful call GlobalVariableSetOnCondition (it returned true). And that, there was a message about it
starting to work, "Copy 1  is working".

---

## Page 499

Part 4. Common APIs
499
4.6 Client terminal global variables
The fact that the value of the internal counter is the same as the external instance number, is purely
coincidental. It could well be that "GlobalsWithCondition 2" had started before "GlobalsWithCondition
1 " (or in some other sequence, given that there are three copies). Then the outer and inner numbering
would be different. You can repeat the experiment starting and stopping all services many times, and
the sequence in which the instances increment the counter variable will most likely be different. But in
any case, the limit on the total number will cut off one extra instance.
When the last instance of "GlobalsWithCondition 2" is granted access to a global variable, value 2 is
already stored there. Since this is the limit we set, the program does not start.
GlobalVariableSetOnCondition(GlobalsWithCondition.mq5,count+1,count)= »
» false / GLOBALVARIABLE_NOT_FOUND(4501)
Counter is already altered by other instance: 2
Start failed: count: 2, retries: 2
Further along, copies of "GlobalsWithCondition" and "GlobalsWithCondition 1 " "spin" in the work cycle
until the services are stopped.
You can try to stop only one instance. Then it will be possible to launch another one that previously
received a ban on execution due to exceeding the quota.
Of course, the proposed version of protection against parallel modification is effective only for
coordinating the behavior of your own programs, but not for limiting a single copy of the demo version,
since the user can simply delete the global variable. For this purpose, global variables can be used in a
different way - in relation to the chart ID: an MQL program works only for as long as its created global
variable contains its ID graphic arts. Other ways to control shared data (counters and other
information) is provided by resources and database.
4.6.7 Flushing global variables to disk
To optimize performance, global variables reside in memory while the terminal is running. However, as
we know, variables are stored between sessions in a special file. This applies to all global variables
except temporary variables. Normally writing variables to a file happens when the terminal closes.
However, if your computer suddenly crashes, your data may be lost. Therefore, it can be useful to
forcibly initiate writing in order to guarantee the safety of data in any unforeseen situations. For this
purpose, the MQL5 API provides the GlobalVariablesFlush function.
void GlobalVariablesFlush()
The function forces the contents of global variables to be written to disk. The function has no
parameters and returns nothing.
The simplest example is given in the script GlobalsFlush.mq5.
void OnStart()
{
   GlobalVariablesFlush();
}
With it, you can flush variables to disk at any time, if necessary. You can use your preferred file
manager and make sure that the date and time of the gvariables.dat file change immediately after the
script is run. However, note that the file will only be updated if the global variables have been edited in
any way or just read (this changes the access time) since the previous save.

---

## Page 500

Part 4. Common APIs
500
4.6 Client terminal global variables
This script is useful for those who keep the terminal turned on for a long time, and programs that
modify global variables are executed in it.
4.7 Functions for working with time
Time is a fundamental factor in most processes, and plays an important applied role for trading.
As we know, the main coordinate system in trading is based on two dimensions: price and time. They
are displayed on the chart along the vertical and horizontal axes, respectively. Later, we will touch on
another important axis, which can be represented as being perpendicular to the first two and going
deep into the chart, on which trading volumes are marked. But for now, let's focus on time.
This measurement is common to all charts, uses the same units of measurement, and, strange as it
may sound, is characterized by constancy (the course of time is predictable).
The terminal provides a plethora of built-in tools related to the calculation and analysis of time. So, we
will get acquainted with them gradually, as we move through the chapters of the book, from simple to
complex.
In this chapter, we will study the functions that allow you to control the time and pause the program
activity for a specified interval.
In the Date and time chapter, in the section on data transformation, we already saw a couple of
functions related to time: TimeToStruct and StructToTime. They split a value of the datetime type into
components or vice versa, construct datetime from individual fields: recall that they are summarized in
the MqlDateTime structure.
struct MqlDateTime
{ 
   int year;        // year (1970 — 3000)
   int mon;         // month (1 — 12) 
   int day;         // day (1 — 31) 
   int hour;        // hours (0 — 23) 
   int min;         // minutes (0 — 59) 
   int sec;         // seconds (0 — 59) 
   int day_of_week; // day of the week, numbered from 0 (Sunday) to 6 (Saturday)
                    // according to enum ENUM_DAY_OF_WEEK
   int day_of_year; // ordinal number of the day in the year, starting from 0 (January 1)
};
But where can an MQL program get the datetime value from?
For example, historical prices and times are reflected in quotes, while current live data arrives as ticks.
Both have timestamps, which we will learn how to get in the relevant sections: about timeseries and
terminal events. However, an MQL program can query the current time by itself (without prices or
other trading information) using several functions.
Several functions were required because the system is distributed: it consists of a client terminal and a
broker server located in arbitrary parts of the world, which, quite likely, belong to different time zones.
Any time zone is characterized by a temporary offset relative to the global reference point of time,
Greenwich Mean Time (GMT). As a rule, a time zone offset is an integer number of hours N (although
there are also exotic zones with a half-hour step) and therefore it is indicated as GMT + N or GMT-N,

---

## Page 501

Part 4. Common APIs
501 
4.7 Functions for working with time
depending on whether the zone is east or west of the meridian. For example, Continental Europe,
located east of London, uses Central European Time (CET) equal to GMT + 1 , or Eastern European
Time (Eastern European Time, EET) equal to GMT + 2, while in America there are "negative" zones,
such like Eastern Standard Time (EST) or GMT-5.
It should be noted that GMT corresponds to astronomical (solar) time, which is slightly non-linear
as the Earth's rotation is gradually slowing down. In this regard, in recent decades, there has
actually been a transition to a more accurate timekeeping system (based on atomic clocks), in
which global time is called Coordinated Universal Time (UTC). In many application areas, including
trading, the difference between GMT and UTC is not significant, so the time zone designations in
the new UTC±N format and the old GMT±N should be considered analogs. For example, many
brokers already specify session times in UTC in their specifications, while the MQL5 API has
historically used GMT notation.
The MQL5 API allows you to find out the current time of the terminal (in fact, the local time of the
computer) and the server time: they are returned by the functions TimeLocal and TimeCurrent,
respectively. In addition, an MQL program can get the current GMT time (function TimeGMT) based on
the Windows timezone settings. Thus, a trader and a programmer get a binding of local time to the
global one, and by the difference between local and server time, one can determine the "timezone" of
the server and quotes. But there are a couple of interesting points here.
First, in many countries, there is a practice of switching to the Daylight Saving Time (DST). Usually,
this means adding 1  hour to standard (winter) time from about March/April to October/November (in
the northern hemisphere, in the southern it is vice versa). At the same time, GMT/UTC time always
remains constant, i.e., it is not subject to DST correction, and therefore various options for
convergence/discrepancy between client and server time are potentially possible:
• transition dates may vary from country to country
• some countries do not implement daylight saving time
Because of this, some MQL programs need to keep track of such time zone changes if the algorithms
are based on reference to intraday time (for example, news releases) and not to price movements or
volume concentrations.
And if the time translation on the user's computer is quite easy to determine, thanks to the
TimeDaylightSavings function, then there is no ready-made analog for server time.
Second, the regular MetaTrader 5 tester, in which we can debug or evaluate MQL programs of such
types as Expert Advisors and indicators, unfortunately, does not emulate the time of the trade server.
Instead, all three of the above functions TimeLocal, TimeGMT, and TimeCurrent, will return the same
time, i.e. the timezone is always virtually GMT.
Absolute and relative time
Time accounting in algorithms, as in life, can be carried out in absolute or relative coordinates.
Every moment in the past, present, and future is described by an absolute value to which we can
refer in order to indicate the beginning of an accounting period or the time an economic news is
released. It is this time that we store in MQL5 using the datetime type. At the same time, it is often
required to look into the future or retreat into the past for a given number of time units from the
current moment. In this case, we are not interested in the absolute value, but in the time interval.
In particular, algorithms have the concept of a timeout, which is a period of time during which a
certain action must be performed, and if it is not performed for any reason, we cancel it and stop
waiting for the result (because, apparently, something went wrong). You can measure the interval in

---

## Page 502

Part 4. Common APIs
502
4.7 Functions for working with time
different units: hours, seconds, milliseconds, or even microseconds (after all, computers are now
fast).
In MQL5, some time-related functions work with absolute values (for example, TimeLocal,
TimeCurrent), and the part with intervals (for example, GetTickCount, GetMicrosecondCount).
However, the measurement of intervals or the activation of the program at specified intervals can
be implemented not only via the functions from this section but also using built-in timers that work
according to the well-known principle of an alarm clock. When enabled, they use special events to
notify MQL programs and the functions we implement to handle these events – OnTimer (they are
similar to OnStart). We will cover this aspect of time management in a separate section, after
studying the general concept of events in MQL5 (see Overview of event handling functions).
4.7.1  Local and server time
There are always two types of time on the MetaTrader 5 platform: local (client) and server (broker).
Local time corresponds to the time of the computer on which the terminal is running, and increases
continuously, at the same rate as in the real world.
Server time flows differently. The basis for it is set by the time on the broker’s computer, however, the
client receives information about it only together with the next price changes, which are packed into
special structures called ticks (see the section about MqlTick) and are passed to MQL programs using
events.
Thus, the updated server time becomes known in the terminal only as a result of a change in the price
of at least one financial instrument on the market, that is, from among those selected in the Market
Watch window. The last known time of the server is displayed in the title bar of this window. If there are
no ticks, the server time in the terminal stands still. This is especially noticeable on weekends and
holidays when all exchanges and Forex platforms are closed.
In particular, on a Sunday, the server time will most likely be displayed as Friday evening. The only
exceptions are those instances of MetaTrader 5 that offer continuously traded instruments such as
cryptocurrencies. However, even in this case, during periods of low volatility, server time can noticeably
lag behind local time.
All functions in this section operate on time with an accuracy of up to a second (the accuracy of time
representation in the datetime type).
To get local and server time, the MQL5 API provides three functions: TimeLocal, TimeCurrent, and
TimeTradeServer. All three functions have two versions of the prototype: the first one returns the time
as a value of the datetime type, and the second one additionally accepts by reference and fills the
MqlDateTime structure with time components.
datetime TimeLocal()
datetime TimeLocal(MqlDateTime &dt)
The function returns the local computer time in the datetime format.
It is important to note that time includes Daylight Savings Time if enabled. I.e., TimeLocal equals the
standard time of the computer's time zone, minus the correction TimeDaylightSavings. Conditionally,
the formula can be represented as follows:

---

## Page 503

Part 4. Common APIs
503
4.7 Functions for working with time
TimeLocal summer() = TimeLocal winter() - TimeDaylightSavings()
Here TimeDaylightSavings usually equals -3600, that is, moving the clock forward 1  hour (1  hour is
lost). So the summer value of TimeLocal is greater than the winter value (with equal astronomical time
of day) relative to UTC. For example, if in winter TimeLocal equals UTC+2, then in summer it is UTC+3.
UTC can be obtained using the TimeGMT function.
datetime TimeCurrent()
datetime TimeCurrent(MqlDateTime &dt)
The function returns the last known server time in the datetime format. This is the time of arrival of the
last quote from the list of all financial instruments in the Market Watch. The only exception is the
OnTick event handler in Expert Advisors, where this function will return the time of the processed tick
(even if ticks with a more recent time have already appeared in the Market Watch).
Also, note that the time on the horizontal axis of all charts in MetaTrader 5 corresponds to the server
time (in history). The last (current, rightmost) bar contains TimeCurrent. See details in the Charts
section.
datetime TimeTradeServer()
datetime TimeTradeServer(MqlDateTime &dt)
The function returns the estimated current time of the trade server. Unlike TimeCurrent, the results of
which may not change if there are no new quotes, TimeTradeServer allows you to get an estimate of
continuously increasing server time. The calculation is based on the last known difference between the
time zones of the client and the server, which is added to the current local time.
In the tester, the TimeTradeServer value is always equal to TimeCurrent.
An example of how the functions work is given in the script TimeCheck.mq5.
The main function has an infinite loop that logs all types of time every second until the user stops the
script.
void OnStart()
{
   while(!IsStopped())
   {
      PRTF(TimeLocal());
      PRTF(TimeCurrent());
      PRTF(TimeTradeServer());
      PRTF(TimeTradeServerExact());
      Sleep(1000);
   }
}
In addition to the standard functions, a custom function TimeTradeServerExact is applied here.

---

## Page 504

Part 4. Common APIs
504
4.7 Functions for working with time
datetime TimeTradeServerExact()
{
   enum LOCATION
   {
      LOCAL, 
      SERVER, 
   };
   static datetime now[2] = {}, then[2] = {};
   static int shiftInHours = 0;
   static long shiftInSeconds = 0;
   
   // constantly detect the last 2 timestamps here and there
   then[LOCAL] = now[LOCAL];
   then[SERVER] = now[SERVER];
   now[LOCAL] = TimeLocal();
   now[SERVER] = TimeCurrent();
   
   // at the first call we don't have 2 labels yet,
   // needed to calculate the stable difference
   if(then[LOCAL] == 0 && then[SERVER] == 0) return 0;
   // when the time course is the same on the client and on the server,
   // and the server is not "frozen" due to weekends/holidays,
   // updating difference
   if(now[LOCAL] - now[SERVER] == then[LOCAL] - then[SERVER]
   && now[SERVER] != then[SERVER])
   {
      shiftInSeconds = now[LOCAL] - now[SERVER];
      shiftInHours = (int)MathRound(shiftInSeconds / 3600.0);
      // debug print
      PrintFormat("Shift update: hours: %d; seconds: %lld", shiftInHours, shiftInSeconds);
   }
   
   // NB: The built-in function TimeTradeServer calculates like this:
   //                TimeLocal() - shiftInHours * 3600
   return (datetime)(TimeLocal() - shiftInSeconds);
}
It was required because the algorithm of the built-in TimeTradeServer function may not suit everyone.
The built-in function finds the difference between local and server time in hours (that is, the time zone
difference), and then gets the server time as a local time correction for this difference. As a result, if
the minutes and seconds go on the client and server not synchronously (which is very likely), the
standard approximation of server time will show the minutes and seconds of the client, not the server.
Ideally, the local clocks of all computers should be synchronized with global time, but in practice,
deviations occur. So, if there is even a small shift on one of the sides, TimeTradeServer can no longer
repeat the time on the server with the highest precision.
In our implementation of the same function in MQL5, we do not round the difference between the client
and server time to hourly timezones. Instead, the exact difference in seconds is used in the calculation.
That's why TimeTradeServerExact returns the time at which minutes and seconds go just like on the
server.

---

## Page 505

Part 4. Common APIs
505
4.7 Functions for working with time
Here is an example of a log generated by the script.
TimeLocal()=2021.09.02 16:03:34 / ok
TimeCurrent()=2021.09.02 15:59:39 / ok
TimeTradeServer()=2021.09.02 16:03:34 / ok
TimeTradeServerExact()=1970.01.01 00:00:00 / ok
It can be seen that the time zones of the client and server are the same, but there is a
desynchronization of several minutes (for clarity). On the first call, TimeTradeServerExact returned 0.
Further, the data for calculating the difference will already arrive, and we will see all four time types,
uniformly "walking" with an interval of a few seconds.
TimeLocal()=2021.09.02 16:03:35 / ok
TimeCurrent()=2021.09.02 15:59:40 / ok
TimeTradeServer()=2021.09.02 16:03:35 / ok
Shift update: hours: 0; seconds: 235
TimeTradeServerExact()=2021.09.02 15:59:40 / ok
TimeLocal()=2021.09.02 16:03:36 / ok
TimeCurrent()=2021.09.02 15:59:41 / ok
TimeTradeServer()=2021.09.02 16:03:36 / ok
Shift update: hours: 0; seconds: 235
TimeTradeServerExact()=2021.09.02 15:59:41 / ok
TimeLocal()=2021.09.02 16:03:37 / ok
TimeCurrent()=2021.09.02 15:59:41 / ok
TimeTradeServer()=2021.09.02 16:03:37 / ok
TimeTradeServerExact()=2021.09.02 15:59:42 / ok
TimeLocal()=2021.09.02 16:03:38 / ok
TimeCurrent()=2021.09.02 15:59:43 / ok
TimeTradeServer()=2021.09.02 16:03:38 / ok
TimeTradeServerExact()=2021.09.02 15:59:43 / ok
4.7.2 Daylight saving time (local)
To determine whether local clocks are switched to daylight saving time, MQL5 provides the
TimeDaylightSavings function. It takes settings from your operating system.
Determining the daylight saving time on a server is not as easy. To do this, you need to implement
MQL5 analysis of quotes, economic calendar events, or a rollover/swap time in the account trading
history. In the example below, we will show one of the options.
int TimeDaylightSavings()
The function returns the correction in seconds if daylight savings time has been applied. Winter time is
standard for each time zone, so the correction for this period is zero. In conditional form, the formula
for obtaining the correction can be written as follows:
TimeDaylightSavings() = TimeLocal winter() - TimeLocal summer()
For example, if the standard timezone (winter) is equal to UTC+3 (that is, the zone time is 3 hours
ahead of UTC), then during the transition to daylight saving time (summer) we add 1  hour and get
UTC+4. Wherein TimeDaylightSavings will return -3600.
An example of using the function is given in the script TimeSummer.mq5, which also suggests one of the
possible empirical ways to identify the appropriate mode on the server.

---

## Page 506

Part 4. Common APIs
506
4.7 Functions for working with time
void OnStart()
{
   PRTF(TimeLocal());          // local time of the terminal
   PRTF(TimeCurrent());        // last known server time
   PRTF(TimeTradeServer());    // estimated server time
   PRTF(TimeGMT());            // GMT time (calculation from local via time zone shift)
   PRTF(TimeGMTOffset());      // time zone shift compare to GMT, in seconds
   PRTF(TimeDaylightSavings());// correction for summer time in seconds
   ...
First, let's display all types of time and its correction provided by MQL5 (functions TimeGMT and
TimeGMTOffset will be presented in the next section on Universal Time, but their meaning should
already be generally clear from the previous description).
The script is supposed to run on trading days. The entries in the log will correspond to the settings of
your computer and the broker's server.
TimeLocal()=2021.09.09 22:06:17 / ok
TimeCurrent()=2021.09.09 22:06:10 / ok
TimeTradeServer()=2021.09.09 22:06:17 / ok
TimeGMT()=2021.09.09 19:06:17 / ok
TimeGMTOffset()=-10800 / ok
TimeDaylightSavings()=0 / ok
In this case, the client's time zone is 3 hours off from GMT (UTC+3), there is no adjustment for
daylight saving time.
Now let's take a look at the server. Based on the value of the TimeCurrent function, we can determine
the current time of the server, but not its standard time zone, since this time may involve the transition
to daylight saving time (MQL5 does not provide information about whether it is used at all and whether
it is currently enabled).
To determine the real time zone of the server and the daylight saving time, we will use the fact that the
server time translation affects quotes. Like most empirical methods for solving problems, this one may
not give completely correct results in certain circumstances. If a comparison with other sources shows
discrepancies, a different method should be chosen.
The Forex market opens on Sunday at 22:00 UT (this corresponds to the beginning of morning trading
in the Asia-Pacific region) and closes on Friday at 22:00 (the close of trading in America). This means
that on servers in the UTC+2 zone (Eastern Europe), the first bars will appear at exactly 0 hours 0
minutes on Monday. According to Central European time, which corresponds to UTC+1 , the trading
week starts at 23:00 on Sunday.
Having calculated the statistics of the intraday shift of the first bar H1  after each weekend break, we
will get an estimate of the server's time zone. Of course, for this, it is better to use the most liquid
Forex instrument, which is EURUSD.
If two maximum intraday shifts are found in the statistics for an annual period, and they are located
next to each other, this will mean that the broker is switching to daylight saving time and vice versa.
Note that the summer and winter time periods are not equal. So, when switching to summer time in
early March and returning to winter time in early November, we get about 8 months of summer time.
This will affect the ratio of maximums in the statistics.

---

## Page 507

Part 4. Common APIs
507
4.7 Functions for working with time
Having two time zones, we can easily determine which of them is active at the moment and, thereby,
find out the current presence or absence of a correction for daylight saving time.
When switching clocks to daylight saving time, the broker's timezone will change from UTC+2 to
UTC+3, which will shift the beginning of the week from 22:00 to 21 :00. This will affect the structure of
H1  bars: visually on the chart, we will see three bars on Sunday evening instead of two.
Changing hours from winter (UTC+2) to summer (UTC+3) time on the EURUSD H1 chart
To implement this, we have a separate function, ServerTimeZone. The call of the built-in CopyTime
function is responsible for getting quotes, or bar timestamps, to be more precise (we will study this
function in the section on access to timeseries).
ServerTime ServerTimeZone(const string symbol = NULL)
{
  const int year = 365 * 24 * 60 * 60;
  datetime array[];
  if(PRTF(CopyTime(symbol, PERIOD_H1, TimeCurrent() - year, TimeCurrent(), array)) > 0)
  {
     // here we get about 6000 bars in the array
     const int n = ArraySize(array);
     PrintFormat("Got %d H1 bars, ~%d days", n, n / 24);
     // (-V-) loop through H1 bars
     ...
  }
}
The CopyTime function receives the working instrument, H1  timeframe, and the range of dates for the
last year, as parameters. The NULL value instead of the instrument means the symbol of the current
chart where the script will be placed, so it is recommended to select the window with EURUSD. The

---

## Page 508

Part 4. Common APIs
508
4.7 Functions for working with time
PERIOD_H1  constant corresponds to H1 , as you might guess. We are already familiar with the
TimeCurrent function: it will return the current, latest known time of the server. And if we subtract from
it the number of seconds in a year, which is placed into the year variable, we will get the date and time
exactly one year ago. The results will go into the array.
To calculate statistics on how many times a week was opened by a bar at a specific hour, we reserve
the hours[24] array. The calculation will be performed in a loop through the resulting array, that is, by
bars from the past to the present. At each iteration, the opening hour of the week being viewed will be
stored in the current variable. When the loop ends, the server's current time zone will remain in current,
since the current week will be processed last.
     // (-v-) cycle through H1 bars
     int hours[24] = {};
     int current = 0;
     for(int i = 0; i < n; ++i)
     {
     // (-V-) processing of the i-th bar H1
        ...
     }
     
     Print("Week opening hours stats:");
     ArrayPrint(hours);
Inside the days loop, we will use the datetime class from the header file MQL5Book/DateTime.mqh (see
Date and time).
        // (-v-) processing the i-th bar H1
        // find the day of the week of the bar
        const ENUM_DAY_OF_WEEK weekday = TimeDayOfWeek(array[i]);
        // skip all days except Sunday and Monday
        if(weekday > MONDAY) continue;
        // analyze the first bar H1 of the next trading week
        // find the hour of the first bar after the weekend
        current = _TimeHour();
        // calculate open hours statistics
        hours[current]++;
        
        // skip next 2 days
        // (because the statistics for the beginning of this week have already been updated)
        i += 48;
The proposed algorithm is not optimal, but it does not require understanding the technical details of
timeseries organization, which are not yet known to us.
Some weeks are unformatted (begin after the holidays). If this situation happens in the last week, the
current variable will contain an unusual offset. This can be verified by statistics: for the resulting hour,
there will be a very small number of recorded "openings" of the week. In the test script, in this case, a
message is simply displayed in the log. In practice, you should clarify the standard opening for the
previous one to two weeks.

---

## Page 509

Part 4. Common APIs
509
4.7 Functions for working with time
     // (-V-) cycle through H1 bars
     ...
     if(hours[current] <= 52 / 4)
     {
        // TODO: check for previous weeks
        Print("Extraordinary week detected");
     }
If the broker does not switch to daylight saving time, the statistics will have one maximum, which will
include all or almost all weeks. If the broker practices a time zone change, there will be two highs in
the statistics.
     // find the most frequent time shift
     int max = ArrayMaximum(hours);
     // then check if there is another regular shift
     hours[max] = 0;
     int sub = ArrayMaximum(hours);
We need to determine how significant the second extreme is (i.e. different from random holidays that
could shift the start of the week). To do this, we evaluate the statistics for a quarter of the year (52
weeks / 4). If this limit is exceeded, the broker supports daylight saving time.
     int DST = 0;
     if(hours[sub] > 52 / 4)
     {
        // basically, DST is supported 
        if(current == max || current == sub)
        {
           if(current == MathMin(max, sub))
              DST =fabs(max -sub); // DST is enabled now
        }
     }
If the offset of the opening of the current week (in the current variable) coincides with one of the two
main extremes, then the current week opened normally, and it can be used to draw a conclusion about
the time zone (this protective condition is necessary because we do not have a correction for the non-
standard weeks and only a warning is issued instead).
Now everything is ready to form the response of our function: the server time zone and the sign of the
enabled daylight saving time.
 current +=2 +DST;// +2 to get offset from UTC
     current %= 24;
 // timezones are always in the range [UTC-12,UTC+12]
     if(current > 12) current = current - 24;
Since we have two characteristics to return from a function (current and DST), and besides that, we
can tell the called code whether the broker uses daylight saving time to begin with (even if it is winter
now), it makes sense to declare a special structure ServerTime with all required fields.

---

## Page 510

Part 4. Common APIs
51 0
4.7 Functions for working with time
struct ServerTime
{
 intoffsetGMT;      // timezone in seconds relative to UTC/GMT
 intoffsetDST;      // DST correction in seconds (included in offsetGMT)
 boolsupportDST;    // DST correction detected in quotes in principle
 stringdescription; // result description
};
Then, in the ServerTimeZone function, we can fill in and return such a structure as a result of the work.
     ServerTime st = {};
     st.description = StringFormat("Server time offset: UTC%+d, including DST%+d", current, DST);
     st.offsetGMT = -current * 3600;
     st.offsetDST = -DST * 3600;
     return st;
If for some reason the function cannot get quotes, we will return an empty structure.
ServerTime ServerTimeZone(const string symbol = NULL)
{
  const int year = 365 * 24 * 60 * 60;
  datetime array[];
  if(PRTF(CopyTime(symbol, PERIOD_H1, TimeCurrent() - year, TimeCurrent(), array)) > 0)
  {
     ...
     return st;
  }
  ServerTime empty = {-INT_MAX, -INT_MAX, false};
  return empty;
}
Let's check the new function in action, for which in OnStart we add the following instructions:
   ...
   ServerTime st = ServerTimeZone();
   Print(st.description);
   Print("ServerGMTOffset: ", st.offsetGMT);
   Print("ServerTimeDaylightSavings: ", st.offsetDST);
}
Let's look at the possible results.
CopyTime(symbol,PERIOD_H1,TimeCurrent()-year,TimeCurrent(),array)=6207 / ok
Got 6207 H1 bars, ~258 days
Week opening hours stats:
52  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
Server time offset: UTC+2, including DST+0
ServerGMTOffset: -7200
ServerTimeDaylightSavings: 0
According to the collected statistics of H1  bars, the week for this broker opens strictly at 00:00 on
Monday. Thus, the real time zone is equal to UTC+2, and there is no correction for summer time, i.e.,
the server time must match EET (UTC+2). However, in practice, as we saw in the first part of the log,
the time on the server differs from GMT by 3 hours.

---

## Page 511

Part 4. Common APIs
51 1 
4.7 Functions for working with time
Here we can assume that we met a server that works all year round in summer time. In that case, the
function ServerTimeZone will not be able to distinguish the correction from the additional hour in the
"time zone": as a result, the DST mode will be equal to zero, and the GMT time calculated from the
server quotes will shift to the right by an hour from the real one. Or our initial assumption that quotes
start arriving at 22:00 on Sunday does not correspond to the mode of operation of this server. Such
points should be clarified with the broker's support service.
4.7.3 Universal Time
In MQL5, you can find out the global GMT (UTC) based on the computer's local time and its time zone.
datetime TimeGMT()
datetime TimeGMT(MqlDateTime &dt)
The function returns GMT in the datetime format, counting it from the local time of the computer,
taking into account the transition to winter or summer time.
Generalized calculation formula:
TimeGMT() = TimeLocal() + TimeGMTOffset()
Thus, the accuracy of the representation of universal time depends on the correct setting of the clock
on the local computer. Ideally, the value retrieved should match the value known to the server.
For trading strategies based on external economic news, it is easiest to use calendars in the GMT time
zone: then upcoming events can be tracked by TimeGMT. To bind an event to the server time on the
chart, you should correct the event for the difference between the server time zone and GMT
(TimeTradeServer() - TimeGMT()). But remember that MQL5 has its own built-in calendar.
int TimeGMTOffset()
The function returns the current difference between GMT and the computer's local time in seconds,
based on the time zone setting in Windows, taking into account the current daylight savings time. In
most cases, the time zone is given as an integer number of hours relative to GMT, so TimeGMTOffset is
equal to the time zone multiplied by -3600 (converted to seconds). For example, in winter the time
zone can be equal to UTC + 2, which gives an offset of -7200, and in summer it can be UTC + 3, which
gives -1 0800. The minus is needed, because positive time zones when converting their time to GMT
require subtraction of the above number of seconds, and negative ones require additions.
A script using TimeGMT and TimeGMTOffset was shown in the previous section.
4.7.4 Pausing a Program
As we saw earlier in the examples, programs sometimes need to repeat certain actions periodically,
either on a simple schedule or after previous attempts have failed. When this is done in a loop, it is
recommended to pause the program regularly to prevent too frequent requests and unnecessary load
on the CPU, as well as to allow time for external "players" to do their work (for example, if we are
waiting for data from another program, loading the history of quotes, etc.).
For this purpose, MQL5 provides the Sleep function. This section gives its formal description, and an
example will be given in the next section, along with the functions for time interval measurements.

---

## Page 512

Part 4. Common APIs
51 2
4.7 Functions for working with time
void Sleep(int milliseconds)
The function pauses the execution of the MQL program for the specified number of milliseconds. After
their expiration, the instructions following the Sleep call will continue to be executed.
It makes sense to use the function in the first place in scripts and services because these types of
programs have no other way to wait.
In Expert Advisors and indicators, it is recommended to use timers and the OnTimer event. In this
scheme, the MQL program returns control to the terminal and will be called after a specified interval.
Moreover, the Sleep function cannot be called from indicators, since they are executed in terminal
interface threads, the suspension of which will affect the rendering of charts.
If the user interrupts the MQL program from the terminal interface while it is waiting for the call to
complete Sleep, the exit from the function occurs immediately (within 1 00ms), i.e., the pause ends
ahead of schedule. This will set the stop flag _ StopFlag (also available via the function IsStopped), and
the program should stop execution as quickly and correctly as possible.
4.7.5 Time interval counters
To detect a time interval up to a second, it is enough to take the difference between two datetime
values obtained using TimeLocal. However, sometimes we need even higher accuracy. For this purpose,
MQL5 allows you to get system millisecond (GetTickCount, GetTickCount64) or microsecond
(GetMicrosecondCount) counters.
uint GetTickCount()
ulong GetTickCount64()
The functions return the number of milliseconds that have passed since the operating system was
loaded. The timing accuracy is limited by the standard system timer (~1 0-1 5 milliseconds). For a more
accurate measurement of intervals, use the GetMicrosecondCount function.
In case of the GetTickCount function, the return type uint predetermines the period of time after which
the counter will overflow: approximately 49.7 days. In other words, the countdown will start again from
0 if the computer has not been turned off for such a long time.
In contrast, the GetTickCount64 function returns ulong values, and this counter will not overflow in the
foreseeable future (584'942'41 7 years).
ulong GetMicrosecondCount()
The function returns the number of microseconds that have passed since the start of the MQL program.
Examples of using the counter functions and Sleep are summarized in the script TimeCount.mq5.

---

## Page 513

Part 4. Common APIs
51 3
4.7 Functions for working with time
void OnStart()
{
   const uint startMs = GetTickCount();
   const ulong startMcs =  GetMicrosecondCount();
   
   // loop for 5 seconds
   while(PRTF(GetTickCount()) < startMs + 5000)
   {
      PRTF(GetMicrosecondCount());
      Sleep(1000);
   }
   
   PRTF(GetTickCount() - startMs);
   PRTF(GetMicrosecondCount() - startMcs);
}
Here's what the log output of the script might look like.
GetTickCount()=12912811 / ok
GetMicrosecondCount()=278 / ok
GetTickCount()=12913903 / ok
GetMicrosecondCount()=1089845 / ok
GetTickCount()=12914995 / ok
GetMicrosecondCount()=2182216 / ok
GetTickCount()=12916087 / ok
GetMicrosecondCount()=3273823 / ok
GetTickCount()=12917179 / ok
GetMicrosecondCount()=4365889 / ok
GetTickCount()=12918271 / ok
GetTickCount()-startMs=5460 / ok
GetMicrosecondCount()-startMcs=5458271 / ok
4.8 User interaction
The connection of the program with the "outside world" is always bidirectional, and the means for
organizing it can be conditionally divided into categories for input and output of data. In the classic
version, the user provides the program with some settings and receives a result from it. If the program
integrates with some external application or service, input and output, as a rule, are carried out using
special exchange protocols (via files, network, shared memory, etc.), bypassing the user interface.
The MQL program execution environment allows you to organize interaction with the MetaTrader 5 user
in many ways.
In this chapter, we will look at the simplest of them, which allow you to display messages in a log or
graph, show a simple dialog box, and issue sound alerts.
Recall that the standard for entering data into an MQL program is input variables. However, they can
only be set at program initialization. Changing the program properties through the settings dialog
means "restarting" it with new values (later we will talk about some of the special cases connected with
a type of MQL program due to which the restart is in quotation marks).

---

## Page 514

Part 4. Common APIs
51 4
4.8 User interaction
More flexible interactive relation implies the ability to control the behavior of the program without
stopping it. In elementary cases, the MessageBox dialog box (for example), which we will discuss below,
would be suitable for this, but for most practical applications this is not enough.
Therefore, in the following parts of the book, we will significantly expand the list of tools for
implementing the user interface and learn how to create interactive programs based on interface
objects, display graphical information in indicators or resources, send push notifications to user's
mobile devices, and much more.
4.8.1  Logging messages
Logging is the most common way to inform the user of current information about the program's
operation. This may be the status of a regular completion, an indication of progress during a long
calculation, or debugging data for finding and reproducing errors.
Unfortunately, no programmer is immune to errors in their code. Therefore, developers usually try to
leave the so-called "breadcrumb trail": logging the main stages of program execution (at least, the
sequence of function calls).
We are already familiar with two logging functions − Print and PrintFormat. We used them in the
examples in previous sections. We had to "put them into use" ahead of time in a simplified mode since
it is almost impossible to do without them.
One function call generates, as a rule, one record. However, if a newline character ('\n') is encountered
in the output string, it will split the information into two parts.
Note that all Print and PrintFormat calls are transformed into log entries on the Experts tab of the
Toolbox window. Although the tab is called Experts, it collects the results of all print instructions,
regardless of the MQL program type.
Logs are stored in files organized according to the principle "one day = one file": they have the names
YYYYMMDD.log (Y for year, M for month, and D for day). Files are located in <data
directory>/MQL5/Logs (do not confuse them with the terminal system logs in the folder <data
directory>/Logs).
Note that during bulk logging (if Print function calls generate a large amount of information in a
short time), the terminal displays only some entries in the window. This is done to optimize
performance. In addition, the user is in any case not able to see all the messages on the go. In
order to see the full version of the log, you need to run the View command of the context menu. As
a result, a window with a log will open.
It should also be kept in mind that information from the log is cached when written to disk, that is,
it is written to files in large blocks in a lazy mode, which is why at any given time the log file, as a
rule, does not contain the most recent entries (although they are visible in a window). To initiate a
cache flush to the disk, you can run the command View or Open in the log context menu.
Each log entry is preceded by a time to the nearest millisecond, as well as the name of the program
(and its graphics) that generated or caused this message.

---

## Page 515

Part 4. Common APIs
51 5
4.8 User interaction
void Print(argument, ...)
The function prints one or more values to the expert log, in one line (if the output data does not contain
the character '\n').
Arguments can be of any built-in type. They are separated by commas. The number of parameters
cannot exceed 64. Their variable number is indicated by an ellipsis in the prototype, but MQL5 does not
allow you to describe your own functions with a similar characteristic: only some built-in API functions
have a variable number of parameters (in particular, StringFormat, Print, PrintFormat, and Comment).
For structures and classes, you should implement a built-in print method, or display their fields
separately.
Also, the function is not capable of handling arrays. You can display them element by element, or use
the function ArrayPrint.
Values of type double are output by the function with an accuracy of up to 1 6 significant digits
(together in the mantissa and the fractional part). A number can be displayed in either traditional or
scientific format (with an exponent), whichever is more compact. Values of type float are displayed with
an accuracy of 7 decimal places. To display real numbers with a different precision, or to explicitly
specify the format, you must use the PrintFormat function.
Values of type bool output as the strings "true" or "false".
Dates are displayed with the day and time specified with maximum accuracy (up to a second), in the
format "YYYY.MM.DD hh:mm:ss". To display the date in a different format, use the TimeToString
function (see section Date and time).
Enumeration values are displayed as integers. To display element names use the EnumToString function
(see section Enumerations).
Single-byte and double-byte characters are also output as integers. To display symbols as characters
or letters, use the functions CharToString or ShortToString see section Working with symbols and code
pages).
Values of the color type are displayed either as a string with a triple of numbers indicating the intensity
of each color component ("R, G, B") or as a color name if this color is present in the color set.
For more information about converting values of different types to strings, see the chapter Data
Conversion of Built-in Types (particularly in sections Numbers to strings and vice versa, Date and time,
Color).
When working in the strategy tester in single pass mode (testing Expert Advisor or indicator), results of
the function Print are output to the test agent log.
When working in the strategy tester in the mode optimization, logging is suppressed for performance
reasons, so the Print function has no visible effect. However, all expressions given as arguments are
evaluated.
All arguments, after being converted to a string representation, are concatenated into one common
string without any delimiter characters. If required, such characters must be explicitly written in the
argument list. For example,

---

## Page 516

Part 4. Common APIs
51 6
4.8 User interaction
int x;
bool y;
datetime z;
...
Print(x, ", ", y, ", ", z);
Here, 3 variables are logged, separated by commas. If it were not for the intermediate literals ", ", the
values of the variables would be stuck together in the log entry.
Lots of cases of applying Print can be found starting from the very first sections of the book (for
example, First program, Assignment and initialization, expressions and arrays, and in others).  
As a new way of working with Print we will implement a simple class that will allow you to display a
sequence of arbitrary values without specifying a separator character between each neighboring value.
We use the '<<' operator overload approach, similar to what is used in the C++ I/O streams
(std::cout).
The class definition will be placed in a separate header file OutputStream.mqh. A class is shown below in
a simplified form.

---

## Page 517

Part 4. Common APIs
51 7
4.8 User interaction
class OutputStream
{
protected:
   ushort delimiter;
   string line;
   
   // add the next argument, separated by a separator (if any)
   void appendWithDelimiter(const string v)
   {
      line += v;
      if(delimiter != 0)
      {
         line += ShortToString(delimiter);
      }
   }
   
public:
   OutputStream(ushort d = 0): delimiter(d) { }
   
   template<typename T>
   OutputStream *operator<<(const T v)
   {
      appendWithDelimiter((string)v);
      return &this;
   }
   
   OutputStream *operator<<(OutputStream &self)
   {
      if(&this == &self)
      {
         print(line);// output of the composed string
         line = NULL;
      }
      return &this;
   }
};
Its point is to accumulate in a string variable line string representations of any arguments passed using
the '<<' operator. If a separator character is specified in the class constructor, it will automatically be
inserted between the arguments. Since the overloaded operator returns a pointer to an object, we can
chainpass a sequence of arguments:
OutputStream out(',');
out << x << y << z << out;
As an attribute of the end of data collection, and for the actual output of the content line into the log,
an overload of the same operator for the object itself is used.
The real class is somewhat more complicated. In particular, it allows you to set not only the separator
character but also the accuracy of displaying real numbers, as well as flags for selecting fields in date
and time values. In addition, the class supports character printing, ushort, in the form of characters
(instead of integer codes), the simplified output of arrays (into a separate string), colors in hexadecimal

---

## Page 518

Part 4. Common APIs
51 8
4.8 User interaction
format as a single value (and not a triple of numbers separated by commas, since the comma is often
used as a separator character, and then the color components in the log look like 3 different variables).
A demonstration of using the class is given in the script OutputStream.mq5.
void OnStart()
{
   OutputStream os(5, ',');
   
   bool b = true;
   datetime dt = TimeCurrent();
   color clr = C'127, 128, 129';
   int array[] = {100, 0, -100};
   os << M_PI << "text" << clrBlue << b << array << dt << clr << '@' << os;
   
   /*
      output example
      
      3.14159,text,clrBlue,true
      [100,0,-100]
      2021.09.07 17:38,clr7F8081,@
   */
}
void PrintFormat(const string format, ...) ≡ void printf(const string format, ...)
The function logs a set of arguments based on the specified format string. The format parameter not
only provides a free text output string template that is displayed "as is", but can also contain escape
sequences that describe how specific arguments are to be formatted.
The total number of parameters, including the format string, cannot exceed 64. Restrictions on
parameter types are similar to functions print.
PrintFormat working and formatting principles are identical to those described for the StringFormat
function (see section Universal formatted data output to a string). The only difference is that
StringFormat returns the formed string to the calling code, and print format sends to the journal. We
can say that PrintFormat has the following conditional equivalent:
Print(StringFormat(<list of arguments as is, including format>))
In addition to the full name PrintFormat you can use a shorter alias printf.
Like the Print function, PrintFormat has some specific features when working in the tester in the
optimization mode: its output to the log is suppressed to improve performance.
We have already met in many sections scripts that use PrintFormat, for example, Return transition,
Color, Dynamic arrays, File descriptor management, Getting a list of global variables.
4.8.2 Alerts
In this section, the signal will mean the Alert function to issue warnings to the terminal user.

---

## Page 519

Part 4. Common APIs
51 9
4.8 User interaction
The term "alert" has multiple meanings in MetaTrader 5. There are 2 contexts in which it is used:
• User-configurable (manually) alerts in the Alerts tab in the Toolbox panel. Using them, you can
track the triggering of simple conditions for exceeding the set values by price, volume or time, and
issue notifications in various ways.
• Program "alerts" generated from the MQL code by the Alert function. They have nothing to do with
the previous ones.
void Alert(argument, ...)
The function displays a message in a non-modal dialog box, accompanied by a standard sound signal
(according to the selection in the Options dialog, on the tab Events, in the terminal). If the window is
hidden, it will be shown on top of the main terminal window (it can then be closed, minimized, or moved
away while continuing to work with the main window). The message is also added to the Expert log,
marked as "Alert".
There is no command in the MetaTrader 5 interface to manually open the alert window if it was
previously closed. To see the list of warnings again (in its pure form, without the need to filter the
log), you will need to generate a new signal somehow.
Passing arguments, displaying information and the general principles of the function are exactly the
same as what was stated for the Print function.
Demonstration of the Alert function with a screenshot was shown in the introductory greetings example
in the first chapter, in the section Data output.
Use Alert instead of Print in cases where it is necessary to draw the user's attention to the displayed
information. However, it should not be abused, since the frequent appearance of the window can hinder
the user's work, force them to ignore messages or stop the MQL program. Provide an algorithm in your
program to limit the frequency of possible message generation.
4.8.3 Displaying messages in the chart window
As we have seen in the previous sections, MQL5 allows you to output messages to the log or alert
window. The first method is primarily for technical information and cannot guarantee that the user will
notice the message (because the log window may be hidden). At the same time, the second method
can seem too intrusive if used to display frequently changing program status. An intermediate option
offers the function Comment.
void Comment(argument, ...)
The function displays a message composed of all the passed arguments in the upper left corner of the
chart. The message remains there until this or some other program removes it or replaces it with
another one.
The window can contain only one comment: on each call of Comment the old content (if any) is
replaced with the new one.
To clear a comment, just call the function with an empty string: Comment("").
The number of parameters must not exceed 64. Only built-in type arguments are supported. The
concepts of forming the resulting string from the passed values are similar to those described for the
function Print.

---

## Page 520

Part 4. Common APIs
520
4.8 User interaction
The total length of the displayed message is limited to 2045 characters. If the limit is exceeded, the
end of the line will be cut off.
The current content of a comment is one of the string properties of the chart, which can be found
by calling the function ChartGetString(NULL, CHART_ COMMENT). We will talk about this and other
properties of charts (not only string ones) in a separate chapter.
Same as in the Print, PrintFormat, and Alert functions, the string arguments may contain a newline
character ('\n' or '\r\n'), which will cause the message to be split into the appropriate number of
strings. For Comment this is the only way to show a multi-line message. If you can call them several
times to get the same effect using the print and signal functions, then with Comment this cannot be
done, since each call will replace the old string with the new one.
An example of work of the function Comment is shown in the image of the window with the welcome
script from the first chapter, in the section Data output.
Additionally, we will develop a class and simplified functions for displaying multi-line comments based on
a ring buffer of a given size. The test script (OutputComment.mq5) and the header file with the class
code (Comments.mqh) are included in the book.
class Comments
{
 const int capacity; // maximum number of strings
 const bool reverse; // display order (new ones on top if true)
 string lines[];     // text buffer
 int cursor;         // where to put the next string
 int size;           // actual number of strings saved
   
public:
   Comments(const int limit = N_LINES, const bool r = false):
      capacity(limit), reverse(r), cursor(0), size(0)
   {
      ArrayResize(lines, capacity);
   }
   
   void add(const string line);
   void clear();
};
The main work is done by the method add.

---

## Page 521

Part 4. Common APIs
521 
4.8 User interaction
void Comments::add(const string line)
{
   ...
   // if the passed text contains multiple strings,
   // split it into elements by newline character
   string inputs[];
   const int n = StringSplit(line, '\n', inputs);
   
   // add all new elements to the ring buffer
   // overwriting the oldest entries at the cursor
   // cursor increases by capacity module (reset to 0 on overflow)
   for(int i = 0; i < n; ++i)
   {
      lines[cursor] = inputs[reverse ? n - i - 1 : i];
      cursor = (cursor + 1) % capacity;
      if(size < capacity) size++;
   }
   // concatenate all text entries in forward or reverse order
   // gluing with newline characters
   string result = "";
   for(int i = 0, k = size == capacity ? cursor % capacity : 0;
      i < size; ++i, k = ++k % capacity)
   {
      if(reverse)
      {
         result = lines[k] + "\n" + result;
      }
      else
      {
         result += lines[k] + "\n";
      }
   }
   
   // output the result
   Comment(result);
}
If necessary, the comment, and text buffer can be cleared by the method clear, or by calling
add(NULL).
void Comments::clear()
{
   Comment("");
   cursor = 0;
   size = 0;
}
Given such a class, you can define an object with the required buffer capacity and output direction, and
then use its methods.

---

## Page 522

Part 4. Common APIs
522
4.8 User interaction
Comments c(30/*capacity*/, true/*order*/);
   
void function()
{
   ...
   c.add("123");
}
But to simplify the generation of comments in the usual functional style, by analogy with the function
Comment, a couple of helper functions are implemented.
void MultiComment(const string line = NULL)
{
   static Comments com(N_LINES, true);
   com.add(line);
}
void ChronoComment(const string line = NULL)
{
   static Comments com(N_LINES, false);
   com.add(line);
}
They differ only in the direction of the buffer output. MultiComment displays rows in reverse
chronological order, i.e. most recent at the top, like on a bulletin board. This function is recommended
for an indefinitely long episodic display of information with the preservation of history. ChronoComment
displays rows in forward order, i.e. new ones are added at the bottom. This function is recommended
for batch output of multi-line messages.
The number of buffer lines is N_LINES (1 0) by default. If you define this macro with a different value
before including the header file, it will resize.
The test script contains a loop in which messages are periodically generated.
void OnStart()
{
   for(int i = 0; i < 50 && !IsStopped(); ++i)
   {
      if((i + 1) % 10 == 0) MultiComment();
      MultiComment("Line " + (string)i + ((i % 3 == 0) ? "\n  (details)" : ""));
      Sleep(1000);
   }
   MultiComment();
}
At every tenth iteration, the comment is cleared. At every third iteration, a message is created from
two lines (for the rest - from one). A delay of 1  second allows you to see the dynamics in action.
Here is an example of the window while the script is running (in "new messages on top" mode).

---

## Page 523

Part 4. Common APIs
523
4.8 User interaction
Multi-line comments on the chart
Displaying multi-line information in a comment has rather limited capabilities. If you need to organize
data output by columns, highlighting with color or different fonts, reaction to mouse clicks, or arbitrary
locations on the chart, you should use graphical objects.
4.8.4 Message dialog box
The MQL5 API provides the MessageBox function to interactively prompt the user to confirm actions or
select an option for handling a particular situation.
int MessageBox(const string message, const string caption = NULL, int flags = 0)
The function opens a modeless dialog box with the given message (message), header (caption), and
settings (flags). The window remains visible on top of the main terminal window until the user closes it
by clicking on one of the available buttons (see further along).
The message is also displayed in the expert log with the "Message" mark.
If the caption parameter is NULL, the name of the MQL program is used as the title.
The flags parameter must contain a combination of bit flags combined with an OR ('| ') operation. The
general set of supported flags is divided into 3 groups that define:
·a set of buttons in the dialog
·icon image in the dialog
·selection of the active button by default
The following table lists the constants and flag values for defining dialog buttons.

---

## Page 524

Part 4. Common APIs
524
4.8 User interaction
Constant
Value
Description
MB_OK
0x0000
1  OK button (default)
MB_OKCANCEL
0x0001 
2 buttons: OK and Cancel
MB_ABORTRETRYIGNORE
0x0002
3 buttons: Abort, Retry, Ignore
MB_YESNOCANCEL
0x0003
3 buttons: Yes, No, Cancel
MB_YESNO
0x0004
2 buttons: Yes and No
MB_RETRYCANCEL
0x0005
2 buttons: Retry and Cancel
MB_CANCELTRYCONTINUE
0x0006
3 buttons: Cancel, Try Again, Continue
The following table lists the available images (displayed to the left of the message).
Constant
Value
Description
MB_ICONSTOP
MB_ICONERROR
MB_ICONHAND
0x001 0
STOP sign
MB_ICONQUESTION
0x0020
Question mark
MB_ICONEXCLAMATION
MB_ICONWARNING
0x0030
Exclamation point
MB_ICONINFORMATION
MB_ICONASTERISK
0x0040
Information sign
All icons depend on the operating system version. The examples shown may differ on your computer.
The following values are reserved for selecting the active button.
Constant
Value
Description
MB_DEFBUTTON1 
0x0000
The first button (default) if none of the other constants are
selected
MB_DEFBUTTON2
0x01 00
The second button
MB_DEFBUTTON3
0x0200
The third button
MB_DEFBUTTON4
0x0300
The fourth button
The question may arise about what this fourth button is if the above constants allow you to set no
more than three. The fact is that among the flags there is also MB_HELP (0x00004000). It
instructs to show the Help button in the dialog. Then it can become the fourth in a row if there are
three main buttons. However, clicking on the Help button does not close the dialog, unlike other
buttons. According to the Windows standard, a help file can be associated with the program, which

---

## Page 525

Part 4. Common APIs
525
4.8 User interaction
should open with the necessary help when the Help button is pressed. However, MQL programs do
not currently support this technology.
The function returns one of the predefined values depending on how the dialog was closed (which
button was pressed).
Constant
Value
Description
IDOK
1
OK button
IDCANCEL
2
Cancel button
IDABORT
3
Abort button
IDRETRY
4
Retry button
IDIGNORE
5
Ignore button
IDYES
6
Yes button
IDNO
7
No button
IDTRYAGAIN
1 0
Try Again button
IDCONTINUE
1 1 
Continue button
If the message box has a Cancel button, then the function returns IDCANCEL when the ESC key is
pressed (in addition to the Cancel button). If the message box does not have a Cancel button, pressing
ESC has no effect.
Calling MessageBox suspends the execution of the current MQL program until the user closes the
dialog. For this reason, using MessageBox is prohibited in indicators, since the indicators are executed
in the interface thread of the terminal, and waiting for the user's response would slow down the update
of the charts.
Also, the function cannot be used in services, because they have no connection with the user interface,
while other types of MQL programs are executed in the context of the chart.
When working in the strategy tester, the MessageBox function has no effect and returns 0.
After getting the result from the function call, you can process it in the way you want, for example:
   int result = MessageBox("Continue?", NULL, MB_YESNOCANCEL);
 // use 'switch' or 'if' as needed
   switch(result)
   {
   case IDYES:
     // ...
     break;
   case IDNO:
     // ...
     break;
   case IDCANCEL:
     // ...
     break;

---

## Page 526

Part 4. Common APIs
526
4.8 User interaction
   }
The MessageBox function can be tested using the OutputMessage.mq5 script, in which the user can
select the parameters of the dialog using input variables and see it in action.
Groups of settings for buttons, icons, and the default selected button, as well as return codes, are
described in special enumerations: ENUM_MB_BUTTONS, ENUM_MB_ICONS, ENUM_MB_DEFAULT,
ENUM_MB_RESULT. This provides visual input through drop-down lists and simplifies their conversion to
strings using EnumToString.
For example, here is how the first two enumerations are defined.
enum ENUM_MB_BUTTONS
{
   _OK = MB_OK,                                      // Ok
   _OK_CANCEL = MB_OKCANCEL,                         // Ok | Cancel
   _ABORT_RETRY_IGNORE = MB_ABORTRETRYIGNORE,        // Abort | Retry | Ignore
   _YES_NO_CANCEL = MB_YESNOCANCEL,                  // Yes | No | Cancel
   _YES_NO = MB_YESNO,                               // Yes | No
   _RETRY_CANCEL = MB_RETRYCANCEL,                   // Retry | Cancel
   _CANCEL_TRYAGAIN_CONTINUE = MB_CANCELTRYCONTINUE, // Cancel | Try Again | Continue
};
   
enum ENUM_MB_ICONS
{
   _ICON_NONE = 0,                                  // None
   _ICON_QUESTION = MB_ICONQUESTION,                // Question
   _ICON_INFORMATION_ASTERISK = MB_ICONINFORMATION, // Information (Asterisk)
   _ICON_WARNING_EXCLAMATION = MB_ICONWARNING,      // Warning (Exclamation)
   _ICON_ERROR_STOP_HAND = MB_ICONERROR,            // Error (Stop, Hand)
};
The rest can be found in the source code.
They are then used as input variable types (with element comments providing a more user-friendly
presentation in the user interface).

---

## Page 527

Part 4. Common APIs
527
4.8 User interaction
input string Message = "Message";
input string Caption = "";
input ENUM_MB_BUTTONS Buttons = _OK;
input ENUM_MB_ICONS Icon = _ICON_NONE;
input ENUM_MB_DEFAULT Default = _DEF_BUTTON1;
   
void OnStart()
{
   const string text = Message + "\n"
      + EnumToString(Buttons) + ", "
      + EnumToString(Icon) + ","
      + EnumToString(Default);
   ENUM_MB_RESULT result = (ENUM_MB_RESULT)
      MessageBox(text, StringLen(Caption) ? Caption : NULL, Buttons | Icon | Default);
   Print(EnumToString(result));
}
The script displays the specified message in the window, along with the specified dialog settings. The
result of the dialogue is displayed in the log.
A screenshot of the selection of options and the resulting dialog are shown in the following images.
Window properties dialog
Received message dialog box

---

## Page 528

Part 4. Common APIs
528
4.8 User interaction
4.8.5 Sound alerts
To work with sound, the MQL5 API provides one function: PlaySound.
bool PlaySound(const string soundfile)
The function plays the specified sound file in the format wav.
If the file name is specified without a path (for example, "Ring.wav"), it must be located in the Sounds
folder inside the terminal installation directory. If needed, you can organize subfolders inside the
Sounds folder. In such cases, the file name in the soundfile parameter should be preceded by a relative
path. For example, "Example/Ring.wav" refers to the folders and file Sounds/Example/Ring.wav inside
the terminal installation directory.
In addition, you can use sound files located in any other MQL5 subfolder in the terminal's data
directory. Such a path must be preceded by a leading slash (forward single '/' or double backslash '\
\'), which is the delimiter character you use between adjacent folder levels in the file system. For
example, if the sound file Demo.wav is in the terminal_ data_ directory/MQL5/Files, then in the PlaySound
call, we will write the path "/Files/Demo.wav".
Calling the function with a NULL parameter stops the sound from playing. Calling a function with a new
file while the old one is still playing will cause the old one to be interrupted and the new one to start
playing.
In addition to files located in the file system, a path to the resources – data blocks embedded in the
MQL program – can be passed to the function. In particular, a developer can create a sound resource
from a file that is available locally at compile time within a sandbox. All resources are located inside the
ex5 file, which ensures that the user has them and simplifies the distribution of the program as a single
module.
A detailed article about all ways of using resources, including not only sound but also images, arbitrary
binary and text data, and dependent programs (indicators), is presented in the corresponding section in
the seventh part of the book.
The PlaySound function returns true if the file is found, or false otherwise. Note that even if the file is
not an audio file and cannot be played, the function will return true.
Sound playback is performed asynchronously, in parallel with the execution of subsequent program
instructions. In other words, the function returns control to the calling code immediately after the call,
without waiting for the audio effect to complete.
In the strategy tester, the PlaySound function is not executed.
The OutputSound.mq5 script allows you to test the operation of the function.

---

## Page 529

Part 4. Common APIs
529
4.8 User interaction
void OnStart()
{
   PRTF(PlaySound("new.txt"));
   PRTF(PlaySound("abracadabra.wav"));
   const uint start = GetTickCount();
   PRTF(PlaySound("request.wav"));
   PRTF(GetTickCount() - start);
}
The program is trying to play multiple files. The file "new.txt" exists (created specifically for testing),
the file "abracadabra.wav" does not exist, and the "request.wav" file is included in the standard
distribution of MetaTrader 5. The time of the last function call is measured using a pair of calls to
GetTickCount.
As a result of running the script, we get the following log entries:
PlaySound(new.txt)=true / ok
PlaySound(abracadabra.wav)=false / FILE_NOT_EXIST(5019)
PlaySound(request.wav)=true / ok
GetTickCount()-start=0 / ok
The file "new.txt" was found and therefore the function returned true, although it did not produce a
sound. A call for a second, non-existent file returned false, and the error code in _ LastError is 501 9
(FILE_NOT_EXIST). Finally, playing the last file (assuming it exists) should succeed in every sense: the
function will return true, and the terminal will play the audio. The call processing time is virtually zero
(the duration of the sound does not matter).
4.9 MQL program execution environment
As we know, the source texts of an MQL program after compilation into a binary executable code in the
format ex5 are ready to work in the terminal or on test agents. Thus, a terminal or a tester provides a
common environment within which MQL programs "live".
Recall that the built-in tester supports only 2 types of MQL programs: Expert Advisors and indicators.
We will talk in detail about the types of MQL programs and their features in the fifth part of the book.
Meanwhile, in this chapter, we will focus on those MQL5 API functions that are common to all types,
and allow you to analyze the execution environment and, to some extent, control it.
Most environment properties are read-only through functions TerminalInfoInteger, TerminalInfoDouble,
TerminalInfoString, MQLInfoInteger, and MQLInfoString. From the names you can understand that each
function returns values of a certain type. Such an architecture leads to the fact that the applied
meaning of the properties combined in one function can be very different. Another grouping can be
provided by the implementation of your own object layer in MQL5 (an example will be given a little later,
in the section on using properties for binding to the program environment).
The specified set of functions has an explicit logical division into general terminal properties (with the
"Terminal" prefix) and properties of a separate MQL program (with the "MQL" prefix). However, in
many cases, it is required to jointly analyze the similar characteristics of both the terminal and the
program. For example, permissions to use a DLL, or perform trading operations are issued both to the
terminal as a whole and to a specific program. That is why it makes sense to consider the functions
from this in a complex, as a whole.

---

## Page 530

Part 4. Common APIs
530
4.9 MQL program execution environment
Only some of the environment properties associated with error codes are writable, in particular,
resetting a previous error (ResetLastError) and setting a user error (SetUserError).
Also in this chapter, we will look at the functions for closing the terminal within a program
(TerminalClose, SetReturnError) and pausing the program in the debugger (Debug Break).
4.9.1  Getting a general list of terminal and program properties
The available built-in functions for obtaining environment properties use a generic approach: the
properties of each specific type are combined into a separate function with a single argument that
specifies the requested property. There are enumerations defined to identify properties: each element
describes one property.
As we will see below, this approach is often used in the MQL5 API and in other areas, including
application areas. In particular, similar sets of functions are used to get the properties of trading
accounts and financial instruments.
Properties of three simple types, int, double, and string, are sufficient to describe the environment.
However, not only integer properties are presented using values of type int, but also logical flags (in
particular, permissions/prohibitions, presence of a network connection, etc.), as well as other built-in
enumerations (for example, types of MQL programs and types of licenses).
Given the conditional division into terminal properties and properties of a particular MQL program, there
are the following functions that describe the environment.
int MQLInfoInteger(ENUM_MQL_INFO_INTEGER p)
int TerminalInfoInteger(ENUM_TERMINAL_INFO_INTEGER p)
double TerminalInfoDouble(ENUM_TERMINAL_INFO_DOUBLE p)
string MQLInfoString(ENUM_MQL_INFO_STRING p)
string TerminalInfoString(ENUM_TERMINAL_INFO_STRING p)
These prototypes map value types to enum types. For example, terminal properties of type int are
summarized in ENUM_TERMINAL_INFO_INTEGER, and its properties of type double are listed in
ENUM_TERMINAL_INFO_DOUBLE, etc. The list of available enums and their elements can be found in
the documentation, in the sections on Terminal properties and MQL programs.
In the following sections, we'll take a look at all the properties, grouped based on their purpose. But
here we turn to the problem of obtaining a general list of all existing properties and their values. This is
often necessary to identify "bottlenecks" or features of the operation of MQL programs on specific
instances of the terminal. A rather common situation is when an MQL program works on one computer,
but does not work at all, or works exhibits some problems on another.
The list of properties is constantly updated as the platform develops, so it is advisable to make their
request not on the basis of a list hardwired into the source code, but automatically.
In the Enumerations section, we have created a template function EnumToArray to get a complete list
of enumeration elements (file EnumToArray.mqh). Also in that section, we introduced the script
ConversionEnum.mq5, which uses the specified header file. In the script, a helper function process was
implemented, which received an array with enumeration element codes and output them to the log. We
will take these developments as a starting point for further improvement.

---

## Page 531

Part 4. Common APIs
531 
4.9 MQL program execution environment
We need to modify the process function in such a way, that we not only get a list of the elements of a
particular enumeration but also query the corresponding properties using one of the built-in property
functions.
Let's give the new version of the script a name, Environment.mq5.
Since the properties of the environment are scattered across several different functions (in this case,
five), you need to learn how to pass to the new version of the function process a pointer to the required
built-in function (see section Function pointers (typedef)). However, MQL5 does not allow assigning the
address of a built-in function to a function pointer. This can only be done with an application function
implemented in MQL5. Therefore, we will create wrapper functions. For example:
int _MQLInfoInteger(const ENUM_MQL_INFO_INTEGER p)
{
   return MQLInfoInteger(p);
}
// example of pointer type description  
typedef int (*IntFuncPtr)(const ENUM_MQL_INFO_INTEGER property);
// initialization of pointer variables
IntFuncPtr ptr1 = _MQLInfoInteger;  // ok
IntFuncPtr ptr2 = MQLInfoInteger;   // compilation error
A "double" for MQLInfoInteger is shown above (obviously, it should have a different, but preferably
similar, name). Other functions are "packed" in a similar way. There will be five in total.
If in the old version of process there was only one template parameter specifying an enumeration, in
the new one we also need to pass the type of the return value (since MQL5 does not "understand" the
words in the name of enumerations): even though the ending "INTEGER" is present in the name
ENUM_MQL_INFO_INTEGER, the compiler is not able to associate it with the type int).
However, in addition to linking the types of the return value and the enumeration, we need to somehow
pass to the function process a pointer to the appropriate wrapper function (one of the five we defined
earlier). After all, the compiler itself cannot determine by an argument, for example, of
ENUM_MQL_INFO_INTEGER type, that MQLInfoInteger needs to be called.
To solve this problem, a special template structure was created that combines all three factors
together.
template<typename E, typename R>
struct Binding
{
public:
   typedef R (*FuncPtr)(const E property);
   const FuncPtr f;
   Binding(FuncPtr p): f(p) { }
};
The two template parameters allow you to specify the type of the function pointer (FuncPtr) with the
desired combination of result and input parameters. The structure instance has the f field for a pointer
to that newly defined type.
Now a new version of the process function can be described as follows.

---

## Page 532

Part 4. Common APIs
532
4.9 MQL program execution environment
template<typename E, typename R>
void process(Binding<E, R> &b)
{
   E e = (E)0; // turn off the warning about the lack of initialization
   int array[];
   // get a list of enum elements into an array
   int n = EnumToArray(e, array, 0, USHORT_MAX);
   Print(typename(E), " Count=", n);
   ResetLastError();
   // display the name and value for each element,
   // obtained by calling a pointer in the Binding structure
   for(int i = 0; i < n; ++i)
   {
      e = (E)array[i];
      R r = b.f(e); // call the function, then parse _LastError
      const int snapshot = _LastError;
      PrintFormat("% 3d %s=%s", i, EnumToString(e), (string)r +
         (snapshot != 0 ? E2S(snapshot) + " (" + (string)snapshot + ")" : ""));
      ResetLastError();
   }
}
The input argument is the Binding structure. It contains a pointer to a specific function for obtaining
properties (this field will be filled in by the calling code).
This version of the algorithm logs the sequence number, the property identifier, and its value. Again,
note that the first number in each entry will contain the element's ordinal in the enumeration, not the
value (values can be assigned to elements with gaps). Optionally you can add an output of a variable e
"in its pure form" inside the instructions print format.
In addition, you can modify the process so that it collects into an array (or other container, such as a
map) the resulting property values and returns them "outside".
It would be a potential error to refer to the function pointer directly in the instruction print format along
with the _ LastError error code analysis. The point is that the sequence of evaluation of function
arguments (see section Parameters and Arguments) and operands in an expression (see section Basic
concepts) is not defined in this case. Therefore, when a pointer is called on the same line where
_ LastError is read, the compiler may decide to execute the second before the first. As a result, we will
see an irrelevant error code (for example, from a previous function call).
But that's not all. Built-in variable _ LastError can change its value almost anywhere in the evaluation of
an expression if any operation fails. In particular, the function EnumToString can potentially raise an
error code if a value is passed as an argument that is not in the enumeration. In this snippet, we are
immune to this problem because our function EnumToArray returns an array with only checked (valid)
enumeration elements. However, in general cases, in any "compound" instruction, there may be many
places where _ LastError will be changed. In this regard, it is desirable to fix the error code immediately
after the action which we are interested in (here it is a function call by a pointer), saving it to an
intermediate variable snapshot.
But let's go back to the main issue. We can finally organize a call of the new function process to obtain
various properties of the software environment.

---

## Page 533

Part 4. Common APIs
533
4.9 MQL program execution environment
void OnStart()
{
   process(Binding<ENUM_MQL_INFO_INTEGER, int>(_MQLInfoInteger));
   process(Binding<ENUM_TERMINAL_INFO_INTEGER, int>(_TerminalInfoInteger));
   process(Binding<ENUM_TERMINAL_INFO_DOUBLE, double>(_TerminalInfoDouble));
   process(Binding<ENUM_MQL_INFO_STRING, string>(_MQLInfoString));
   process(Binding<ENUM_TERMINAL_INFO_STRING, string>(_TerminalInfoString));
}
Below is a snippet of the generated log entries.
ENUM_MQL_INFO_INTEGER Count=15
  0 MQL_PROGRAM_TYPE=1
  1 MQL_DLLS_ALLOWED=0
  2 MQL_TRADE_ALLOWED=0
  3 MQL_DEBUG=1
...
  7 MQL_LICENSE_TYPE=0
...
ENUM_TERMINAL_INFO_INTEGER Count=50
  0 TERMINAL_BUILD=2988
  1 TERMINAL_CONNECTED=1
  2 TERMINAL_DLLS_ALLOWED=0
  3 TERMINAL_TRADE_ALLOWED=0
...
  6 TERMINAL_MAXBARS=100000
  7 TERMINAL_CODEPAGE=1251
  8 TERMINAL_MEMORY_PHYSICAL=4095
  9 TERMINAL_MEMORY_TOTAL=8190
 10 TERMINAL_MEMORY_AVAILABLE=7813
 11 TERMINAL_MEMORY_USED=377
 12 TERMINAL_X64=1
...
ENUM_TERMINAL_INFO_DOUBLE Count=2
  0 TERMINAL_COMMUNITY_BALANCE=0.0 (MQL5_WRONG_PROPERTY,4512)
  1 TERMINAL_RETRANSMISSION=0.0
ENUM_MQL_INFO_STRING Count=2
  0 MQL_PROGRAM_NAME=Environment
  1 MQL_PROGRAM_PATH=C:\Program Files\MT5East\MQL5\Scripts\MQL5Book\p4\Environment.ex5
ENUM_TERMINAL_INFO_STRING Count=6
  0 TERMINAL_COMPANY=MetaQuotes Software Corp.
  1 TERMINAL_NAME=MetaTrader 5
  2 TERMINAL_PATH=C:\Program Files\MT5East
  3 TERMINAL_DATA_PATH=C:\Program Files\MT5East
  4 TERMINAL_COMMONDATA_PATH=C:\Users\User\AppData\Roaming\MetaQuotes\Terminal\Common
  5 TERMINAL_LANGUAGE=Russian
These and other properties will be described in the following sections.
It is worth noting that some properties are inherited from previous stages of platform development and
are left only for compatibility. In particular, the TERMINAL_X64 property in TerminalInfoInteger

---

## Page 534

Part 4. Common APIs
534
4.9 MQL program execution environment
returns an indication of whether the terminal is 64-bit. Today, the development of 32-bit versions has
been discontinued, and therefore this property is always equal to 1  (true).
4.9.2 Terminal build number
Since the terminal is constantly being improved and new features appear in its new versions, an MQL
program may need to analyze the current version in order to apply different algorithm options. In
addition, no program is immune to errors, including the terminal itself. Therefore, if problems occur,
you should provide a diagnostic output that includes the current version of the terminal. This can help
in reproducing and fixing bugs.
You can get the build number of the terminal using the TERMINAL_BUILD property in
ENUM_TERMINAL_INFO_INTEGER.
if(TerminalInfoInteger(TERMINAL_BUILD) >= 3000)
{
   ...
}
Recall that the build number of the compiler with which the program is built is available in the source
code through the macro definitions __MQLBUILD__ or __MQL5BUILD__ (see Predefined Constants).
4.9.3 Program type and license
The same source code can somehow be included in MQL programs of different types. In addition to the
option of including source codes (preprocessor directive #include) into a common product at the
compilation stage, it is also possible to assemble the libraries – binary program modules connected to
the main program at the execution stage.
However, some functions are only allowed to be used in certain types of programs. For example, the
OrderCalcMargin function cannot be used in indicators. Although this limitation does not seem to be
fundamentally justified, the developer of a universal algorithm for calculating collateral funds, which can
be built into not only Expert Advisors but also indicators, should take this nuance into account and
provide an alternative calculation method for indicators.
A complete list of restrictions on program types will be given in a suitable section of each chapter. In all
such cases, it is important to know the type of the "parent" program.
To determine the program type, there is the MQL_PROGRAM_TYPE property in
ENUM_MQL_INFO_INTEGER. Possible property values are described in the ENUM_PROGRAM_TYPE
enumeration.
Identifier
Value
Description
PROGRAM_SCRIPT
1
Script
PROGRAM_EXPERT
2
Expert Advisor
PROGRAM_INDICATOR
4
Indicator
PROGRAM_SERVICE
5
Service

---

## Page 535

Part 4. Common APIs
535
4.9 MQL program execution environment
In the log snippet in the previous section, we saw that the PROGRAM_SCRIPT property is set to 1 
because our test is a script. To get a string description, you can use the function EnumToString.
ENUM_PROGRAM_TYPE type = (ENUM_PROGRAM_TYPE)MQLInfoInteger(MQL_PROGRAM_TYPE);
Print(EnumToString(type));
Another property of an MQL program that is convenient to analyze for enabling/disabling certain
features is the type of license. As you know, MQL programs can be distributed freely or within the
MQL5 Market. Moreover, the program in the store can be purchased or downloaded as a demo version.
These factors are easy to check and, if desired, adapt the algorithms for them. For these purposes,
there is the MQL_LICENSE_TYPE property in ENUM_MQL_INFO_INTEGER, which uses the
ENUM_LICENSE_TYPE enumeration as a type.
Identifier
Value
Description
LICENSE_FREE
0
Free unlimited version
LICENSE_DEMO
1
Demo version of a paid product from the Market
that works only in the strategy tester
LICENSE_FULL
2
Purchased licensed version, allows at least 5
activations (can be increased by the seller)
LICENSE_TIME
3
Time-limited version (not implemented yet)
It is important to note here that the license refers to the binary ex5 module from which the request is
made using MQLInfoInteger(MQL_ LICENSE_ TYPE). Within a library, this function will return the library's
own license, not the main program that the library is linked to.
As an example to test both functions of this section, a simple service EnvType.mq5 is included with the
book. It does not contain a work cycle and therefore will terminate immediately after executing the two
instructions in OnStart.
#property service
   
void OnStart()
{
   Print(EnumToString((ENUM_PROGRAM_TYPE)MQLInfoInteger(MQL_PROGRAM_TYPE)));
   Print(EnumToString((ENUM_LICENSE_TYPE)MQLInfoInteger(MQL_LICENSE_TYPE)));
}
To simplify its launch, i.e., to eliminate the need to create an instance of the service and run it through
the context menu of the Navigator in the terminal, it is proposed to use the debugger: just open the
source code in MetaEditor and execute the command Debugging -> Start on real data (F5, or button in
the toolbar).
We should get the following log entries:
EnvType (debug)PROGRAM_SERVICE
EnvType (debug)LICENSE_FREE
Here you can clearly see that the type of program is a service, and there is actually no license (free
use).

---

## Page 536

Part 4. Common APIs
536
4.9 MQL program execution environment
4.9.4 Terminal and program operating modes
The MetaTrader 5 environment provides a solution to various tasks at the intersection of trading and
programming, which necessitates several modes of operation of both the terminal itself and a specific
program.
Using the MQL5 API, you can distinguish between regular online activity and backtesting, between
source code debugging (in order to identify potential errors) and performance analysis (search for
bottlenecks in the code), as well as between a local copy of the terminal and the cloud one
(MetaTrader VPS).
The modes are described by flags, each of which contains a value of a boolean type: true or false.
Identifier
Description
MQL_DEBUG
The program is running in debug mode
MQL_PROFILER
The program works in code profiling mode
MQL_TESTER
The program works in the tester
MQL_FORWARD
The program is executed in the process of forward testing
MQL_OPTIMIZATION
The program is running in the optimization process
MQL_VISUAL_MODE
The program is running in visual testing mode
MQL_FRAME_MODE
The Expert Advisor is executed on the chart in the mode of collecting frames
of optimization results
TERMINAL_VPS
The terminal works on a virtual server MetaTrader Virtual Hosting (MetaTrader
VPS)
The MQL_FORWARD, MQL_OPTIMIZATION, and MQL_VISUAL_MODE flags imply the presence of the
MQL_TESTER flag set.
Some pairwise combinations of flags are mutually exclusive, i.e., such flags cannot be enabled at the
same time.
In particular, the presence of MQL_FRAME_MODE excludes MQL_TESTER, and vice versa.
MQL_OPTIMIZATION excludes MQL_VISUAL_MODE, and MQL_PROFILER excludes MQL_DEBUG.
We will study all the flags related to testing (MQL_TESTER, MQL_VISUAL_MODE) in the sections
devoted to Expert Advisors and, in part, to indicators. Everything related to Expert Advisor optimization
(MQL_OPTIMIZATION, MQL_FORWARD, MQL_FRAME_MODE) will be covered in a separate section.
Now let's get acquainted with the principles of reading flags using the example of debugging
(MQL_DEBUG) and profiling (MQL_PROFILER) modes. At the same time, let's recall how these modes
are activated from the MetaEditor (for details, see the documentation, in sections Debugging and
Profiling).
We will use the EnvMode.mq5 script.

---

## Page 537

Part 4. Common APIs
537
4.9 MQL program execution environment
void OnStart()
{
   PRTF(MQLInfoInteger(MQL_TESTER));
   PRTF(MQLInfoInteger(MQL_DEBUG));
   PRTF(MQLInfoInteger(MQL_PROFILER));
   PRTF(MQLInfoInteger(MQL_VISUAL_MODE));
   PRTF(MQLInfoInteger(MQL_OPTIMIZATION));
   PRTF(MQLInfoInteger(MQL_FORWARD));
   PRTF(MQLInfoInteger(MQL_FRAME_MODE));
}
Before running the program, you should check the debugging/profiling settings. To do this, in
MetaEditor, run the command Tools -> Options and check the field values in the Debugging/Profiling tab.
If the option Use specified settings is enabled, then it is the values of the underlying fields that will
affect the financial instrument chart and the timeframe on which the program will be launched. If the
option is disabled, the first financial instrument in Market Watch and the H1  timeframe will be used.
At this stage, the choice of option is not critical.
After preparations, run the script using the command Debug -> Start on Real Data (F5). Since the script
only prints the requested properties to the log (and we don't need breakpoints in it), its execution will
be instantaneous. If step-by-step debugging is needed, we could put a breakpoint (F9) on any
statement in the source code, and the script execution would freeze there for any period we need,
making it possible to study the contents of all variables in MetaEditor, and also move line by line (F1 0)
along the algorithm.
In the MetaTrader 5 log (Experts tab), we will see the following:
MQLInfoInteger(MQL_TESTER)=0 / ok
MQLInfoInteger(MQL_DEBUG)=1 / ok
MQLInfoInteger(MQL_PROFILER)=0 / ok
MQLInfoInteger(MQL_VISUAL_MODE)=0 / ok
MQLInfoInteger(MQL_OPTIMIZATION)=0 / ok
MQLInfoInteger(MQL_FORWARD)=0 / ok
MQLInfoInteger(MQL_FRAME_MODE)=0 / ok
Flags of all modes are reset, except for MQL_DEBUG.
Now let's run the same script from the Navigator in MetaTrader 5 (just drag it with the mouse to any
chart). We will get an almost identical set of flags, but this time MQL_DEBUG will be equal to 0
(because the program was executed in a regular way, and not under a debugger).
Please note that the launch of the program with debugging is preceded by its recompilation in a
special mode when service information permitting debugging is added to the executable file. Such
binary file is larger and slower than usual. Therefore, after debugging is completed, before being
used in real trading, transferred to the customer, or uploaded to the Market, the program should be
recompiled with the File -> Compile (F7) command.
The compilation method does not directly affect the MQL_DEBUG property. The debug version of
the program, as we can see, can be launched in the terminal without a debugger, and MQL_DEBUG
will be reset in this case. Two built-in macros allow you to determine the compilation method:
_DEBUG and _RELEASE (see section Predefined Constants). They are constants, not functions,
because this property is "hardwired" into the program at compile time, and cannot then be changed
(unlike the runtime environment).

---

## Page 538

Part 4. Common APIs
538
4.9 MQL program execution environment
Now let's execute in MetaEditor the command Debug -> Start Profiling on Real Data. Of course, there is
no particular point in profiling such a simple script, but our task now is to make sure that the
appropriate flag is turned on in the environment properties. Indeed, opposite the MQL_PROFILER there
is 1  now.
MQLInfoInteger(MQL_TESTER)=0 / ok
MQLInfoInteger(MQL_DEBUG)=0 / ok
MQLInfoInteger(MQL_PROFILER)=1 / ok
...
The launch of the program with profiling is also preceded by its recompilation in another special
mode, which adds other service information to the binary file that is necessary to measure the
speed of instruction execution. After analyzing the profiler report and fixing bottlenecks, you should
recompile the program in the usual way.
In principle, debugging and profiling can be performed both online and in the tester (MQL_TESTER) on
historical data, but the tester only supports Expert Advisors and indicators. Therefore, it is impossible
to see the set MQL_TESTER or MQL_VISUAL_MODE flag in the script example.
As you know, MetaTrader 5 allows you to test trading programs in quick mode (without a chart) and in
visual mode (on a separate chart). It is in the second case that the MQL_VISUAL_MODE properties will
be enabled. It makes sense to check it, in particular, to disable manipulations with graphic objects in
the absence of visualization.
To debug in visual mode using history, you must first enable the option Use visual mode for debugging on
history in the MetaEditor settings dialog. Analytical programs (indicators) are always tested in visual
mode.
Keep in mind that online debugging is not safe for trading Expert Advisors.
4.9.5 Permissions
MetaTrader 5 provides features for restricting the execution of certain actions by MQL programs for
security reasons. Some of these restrictions are two-level, i.e., they are set separately for the terminal
as a whole and for a specific program. Terminal settings have a priority or act as default values for the
settings of any MQL program. For example, a trader can disable all automated trading by checking the
corresponding box in the MetaTrader 5 settings dialog. In this case, private trading permissions set
earlier to specific robots in their dialogs become invalid.
In the MQL5 API, such restrictions (or vice versa, permissions) are available for reading via the
functions TerminalInfoInteger and MQLInfoInteger. Since they have the same effect on an MQL
program, the program must check general and specific prohibitions equally carefully (to avoid
generating an error when trying to perform an illegal action). Therefore, this section provides a list of
all options of different levels.
All permissions are boolean flags, i.e., they store the values of true or false.
Identifier
Description
TERMINAL_DLLS_ALLOWED
Permission to use the DLL
TERMINAL_TRADE_ALLOWED
Permission to trade automatically online

---

## Page 539

Part 4. Common APIs
539
4.9 MQL program execution environment
Identifier
Description
TERMINAL_EMAIL_ENABLED
Permission to send emails (SMTP server and login must be
specified in the terminal settings)
TERMINAL_FTP_ENABLED
Permission to send files via FTP to the specified server
(including reports for the trading account specified in the
terminal settings)
TE R M IN AL _N O TIF ICATIO N S _E N AB L E D 
Permission to send push notifications to a smartphone
MQL_DLLS_ALLOWED
Permission to use the DLL for this program
MQL_TRADE_ALLOWED
Permission for a program to trade automatically
MQL_SIGNALS_ALLOWED
Permission for a program to work with signals
Permission to use a DLL at the terminal level means that when running an MQL program that contains a
link to some dynamic library, the 'Enable DLL Import' flag on the Dependencies tab will be enabled by
default in its properties dialog. If the flag is cleared in the terminal settings, then the option in the
properties of the MQL program will be disabled by default. In any case, the user must allow imports for
the individual program (there is one exception for scripts, which is discussed below). Otherwise, the
program will not run.
In other words, the TERMINAL_DLLS_ALLOWED and MQL_DLLS_ALLOWED flags can be checked either
by a program without binding to a DLL, or by a program with binding, but for this program,
MQL_DLLS_ALLOWED must be unambiguously equal to true (due to the fact that it has already
started). Thus, as part of software systems that require a DLL, it probably makes sense to provide an
independent utility that would monitor the state of the flag and display diagnostics for the user if it is
suddenly turned off. For example, an Expert Advisor may require an indicator that uses a DLL. Then,
before trying to load the indicator and get its handle, the EA can check the TERMINAL_DLLS_ALLOWED
flag and generate a warning if the flag is reset.
For scripts, the behavior is slightly different because the script settings dialog only opens if the
#property script_ show_ inputs directive is present in the source code. If it is not present, then the dialog
appears when the TERMINAL_DLLS_ALLOWED flag is reset in the terminal settings (and the user must
enable the flag in order for the script to work). When the general flag TERMINAL_DLLS_ALLOWED is
enabled, the script is run without user confirmation, i.e., the MQL_DLLS_ALLOWED value is assumed to
be true (according to TERMINAL_DLLS_ALLOWED).
When working in the tester, the TERMINAL_TRADE_ALLOWED and MQL_TRADE_ALLOWED flags are
always equal to true. However, in indicators, access to all trading functions is prohibited regardless
of these flags. The tester does not allow the testing of MQL programs with DLL dependencies.
The TERMINAL_EMAIL_ENABLED, TERMINAL_FTP_ENABLED, and
TERMINAL_NOTIFICATIONS_ENABLED flags are critical for the send mail, SendFTP, and send
notification functions, which are described in the Network functions section. The
MQL_SIGNALS_ALLOWED flag affects the availability of a group of functions that manage the mql5.com
trading signal subscription  (not discussed in this book). Its state corresponds to the option 'Allow
changing signal settings' in the Common tab of MQL program properties.
Since checking some properties requires additional effort, it makes sense to wrap the flags in a class
that hides multiple calls to various system functions in its methods. This is all the more necessary
because some permissions are not limited to the above options. For example, permission to trade can

---

## Page 540

Part 4. Common APIs
540
4.9 MQL program execution environment
be set (or removed) not only at the terminal or MQL program level but also for an individual financial
instrument – according to its specification from your broker and the exchange sessions. Therefore, at
this step, we will present a draft of the Permissions class which will only contain familiar elements, and
then we will improve for particular application APIs.
Thanks to the class which acts as a program layer, the programmer does not have to remember which
permissions are defined for TerminalInfo functions and which of them are defined for MqlInfo functions.
The source code is in the EnvPermissions.mq5 file.
class Permissions
{
public:
   static bool isTradeEnabled(const string symbol = NULL, const datetime session = 0)
   {
      // TODO: will be supplemented by applied checks of the symbol and sessions
      return PRTF(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
          && PRTF(MQLInfoInteger(MQL_TRADE_ALLOWED));
   }
   static bool isDllsEnabledByDefault()
   {
      return (bool)PRTF(TerminalInfoInteger(TERMINAL_DLLS_ALLOWED));
   }
   static bool isDllsEnabled()
   {
      return (bool)PRTF(MQLInfoInteger(MQL_DLLS_ALLOWED));
   }
   
   static bool isEmailEnabled()
   {
      return (bool)PRTF(TerminalInfoInteger(TERMINAL_EMAIL_ENABLED));
   }
   
   static bool isFtpEnabled()
   {
      return (bool)PRTF(TerminalInfoInteger(TERMINAL_FTP_ENABLED));
   }
   
   static bool isPushEnabled()
   {
      return (bool)PRTF(TerminalInfoInteger(TERMINAL_NOTIFICATIONS_ENABLED));
   }
   
   static bool isSignalsEnabled()
   {
      return (bool)PRTF(MQLInfoInteger(MQL_SIGNALS_ALLOWED));
   }
};
All class methods are static and are called in OnStart.

---

## Page 541

Part 4. Common APIs
541 
4.9 MQL program execution environment
void OnStart()
{
   Permissions::isTradeEnabled();
   Permissions::isDllsEnabledByDefault();
   Permissions::isDllsEnabled();
   Permissions::isEmailEnabled();
   Permissions::isPushEnabled();
   Permissions::isSignalsEnabled();
}
An example of generated logs is shown below.
TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)=1 / ok
MQLInfoInteger(MQL_TRADE_ALLOWED)=1 / ok
TerminalInfoInteger(TERMINAL_DLLS_ALLOWED)=0 / ok
MQLInfoInteger(MQL_DLLS_ALLOWED)=0 / ok
TerminalInfoInteger(TERMINAL_EMAIL_ENABLED)=0 / ok
TerminalInfoInteger(TERMINAL_NOTIFICATIONS_ENABLED)=0 / ok
MQLInfoInteger(MQL_SIGNALS_ALLOWED)=0 / ok
For self-study, the script has a built-in (but commented out) ability to connect system DLLs to read the
contents of the Windows clipboard. We will consider the creation and use of libraries, in particular the
#import directive, in the seventh part of the book, in the section Libraries.
Let's assume that the global DLL import option is disabled in the terminal disabled (this is the
recommended setting for security reasons). Then, if DLLs are connected to the script, it will be possible
to run the script only by allowing import in its individual settings dialog, as a result of which
MQLInfoInteger(MQL_ DLLS_ ALLOWED) will be returning 1  (true). If the global permission for the DLL is
given, then we get TerminalInfoInteger(TERMINAL_ DLLS_ ALLOWED)=1 , and MQL_DLLS_ALLOWED will
inherit this value.
4.9.6 Checking network connections
As you know, the MetaTrader 5 platform is a distributed system that includes several links. In addition
to the client terminal and broker server, it includes the MQL5 community, the Market, cloud services,
and much more. In fact, the client part is also distributed, consisting of a terminal and testing agents
which can be deployed on multiple computers on a local network. In this case, the connection between
any links can potentially be broken for one reason or another. Although the MetaTrader 5 infrastructure
tries to automatically restore its functionality, it is not always possible to do this quickly.
Therefore, in MQL programs, one should take into account the possibility of a connection loss. The
MQL5 API allows you to control the most important connections: with the trade server and the MQL5
community. The following properties are available in TerminalInfoInteger.
Identifier
Description
TERMINAL_CONNECTED
Connection to the trading server
TERMINAL_PING_LAST
The last known ping to the trade server in microseconds
TE R M IN AL _CO M M U N ITY_ACCO U N T
Availability of MQL5.community authorization data in the
terminal

---

## Page 542

Part 4. Common APIs
542
4.9 MQL program execution environment
Identifier
Description
TE R M IN AL _CO M M U N ITY_CO N N E CTIO N 
Connection to MQL5.community
TERMINAL_MQID
Availability of MetaQuotes ID for sending push notifications
All properties except TERMINAL_PING_LAST are boolean flags. TERMINAL_PING_LAST contains a
value of type int.
In addition to the connection, an MQL program often needs to make sure that the data it has is up to
date. In particular, the checked TERMINAL_CONNECTED flag does not yet mean that the quotes you
are interested in are synchronized with the server. To do this, you need to additionally check
SymbolIsSynchronized or SeriesInfoInteger(..., SERIES_ SYNCHRONIZED). These features will be
discussed in the chapter on timeseries.
The TerminalInfoDouble function supports another interesting property: TERMINAL_RETRANSMISSION.
It denotes the percentage of network packets resent in TCP/IP protocol for all running applications and
services on this computer. Even on the fastest and most properly configured network, packet loss
sometimes occurs and, as a result, there will be no confirmation of packet delivery between the
recipient and the sender. In such cases, the lost packet is resent. The terminal itself does not count
the TERMINAL_RETRANSMISSION indicator but requests it once a minute in the operating system.
A high value of this metric may indicate external problems (Internet connection, your provider, local
network, or computer issues), which can worsen the quality of the terminal connection.
If there is a confirmed connection to the community (TERMINAL_COMMUNITY_CONNECTION), an MQL
program can query the user's current balance by calling
TerminalInfoDouble(TERMINAL_ COMMUNITY_ BALANCE). This allows you to use an automated
subscription to paid trading signals (API documentation is available on the mql5.com website).
Let's check the listed properties using the script EnvConnection.mq5.
void OnStart()
{
   PRTF(TerminalInfoInteger(TERMINAL_CONNECTED));
   PRTF(TerminalInfoInteger(TERMINAL_PING_LAST));
   PRTF(TerminalInfoInteger(TERMINAL_COMMUNITY_ACCOUNT));
   PRTF(TerminalInfoInteger(TERMINAL_COMMUNITY_CONNECTION));
   PRTF(TerminalInfoInteger(TERMINAL_MQID));
   PRTF(TerminalInfoDouble(TERMINAL_RETRANSMISSION));
   PRTF(TerminalInfoDouble(TERMINAL_COMMUNITY_BALANCE));
}
Here is a log example (the values will match your settings).
TerminalInfoInteger(TERMINAL_CONNECTED)=1 / ok
TerminalInfoInteger(TERMINAL_PING_LAST)=49082 / ok
TerminalInfoInteger(TERMINAL_COMMUNITY_ACCOUNT)=0 / ok
TerminalInfoInteger(TERMINAL_COMMUNITY_CONNECTION)=0 / ok
TerminalInfoInteger(TERMINAL_MQID)=0 / ok
TerminalInfoDouble(TERMINAL_RETRANSMISSION)=0.0 / ok
TerminalInfoDouble(TERMINAL_COMMUNITY_BALANCE)=0.0 / ok

---

## Page 543

Part 4. Common APIs
543
4.9 MQL program execution environment
4.9.7 Computing resources: memory, disk, and CPU
Like all programs, MQL applications consume computer resources, including memory, disk space, and
CPU. Taking into account that the terminal itself is resources-intensive (in particular, due to the
potential download of quotes and ticks for multiple financial instruments with a long history), sometimes
it is necessary to analyze and control the situation in terms of the proximity of available limits.
The MQL5 API provides several properties that allow you to estimate the maximum achievable and
expended resources. The properties are summarized in the ENUM_MQL_INFO_INTEGER and
ENUM_TERMINAL_INFO_INTEGER enumerations.
Identifier
Description
MQL_MEMORY_LIMIT
Maximum possible amount of dynamic memory for an MQL
program in Kb
MQL_MEMORY_USED
Memory used by an MQL program in Mb
MQL_HANDLES_USED
Number of class objects
TERMINAL_MEMORY_PHYSICAL
Physical RAM in the system in Mb
TERMINAL_MEMORY_TOTAL
Memory (physical+swap file, i.e. virtual) available to the
terminal (agent) process in Mb
TERMINAL_MEMORY_AVAILABLE
Free memory of the terminal (agent) process in Mb, part of
TOTAL
TERMINAL_MEMORY_USED
Memory used by the terminal (agent) in Mb, part of TOTAL
TERMINAL_DISK_SPACE
Free disk space, taking into account possible quotas for the
MQL5/Files folder of the terminal (agent), in Mb
TERMINAL_CPU_CORES
Number of processor cores in the system
TERMINAL_OPENCL_SUPPORT
Supported OpenCL  version as 0x0001 0002 = 1 .2; "0" means
that OpenCL is not supported
The maximum amount of memory available to an MQL program is described by the
MQL_MEMORY_LIMIT property. This is the only property listed that uses kilobytes (Kb). All others are
returned in megabytes (Mb). As a rule, MQL_MEMORY_LIMIT is equal to TERMINAL_MEMORY_TOTAL,
i.e., all memory available on the computer can be allocated to one MQL program by default. However,
the terminal, in particular its cloud implementation for MetaTrader VPS, and cloud testing agents may
limit the memory for a single MQL program. Then MQL_MEMORY_LIMIT will be significantly less than
TERMINAL_MEMORY_TOTAL.
Since Windows typically creates a swap file that is equal in size to physical memory (RAM), the
TERMINAL_MEMORY_TOTAL property can be up to 2 times the size of TERMINAL_MEMORY_PHYSICAL.
All available virtual memory TERMINAL_MEMORY_TOTAL is divided between used
(TERMINAL_MEMORY_USED) and still free (TERMINAL_MEMORY_AVAILABLE) memory.
The book comes with the script EnvProvision.mq5, which logs all specified properties.

---

## Page 544

Part 4. Common APIs
544
4.9 MQL program execution environment
void OnStart()
{
   PRTF(MQLInfoInteger(MQL_MEMORY_LIMIT)); // Kb!
   PRTF(MQLInfoInteger(MQL_MEMORY_USED));
   PRTF(TerminalInfoInteger(TERMINAL_MEMORY_PHYSICAL));
   PRTF(TerminalInfoInteger(TERMINAL_MEMORY_TOTAL));
   PRTF(TerminalInfoInteger(TERMINAL_MEMORY_AVAILABLE));
   PRTF(TerminalInfoInteger(TERMINAL_MEMORY_USED));
   PRTF(TerminalInfoInteger(TERMINAL_DISK_SPACE));
   PRTF(TerminalInfoInteger(TERMINAL_CPU_CORES));
   PRTF(TerminalInfoInteger(TERMINAL_OPENCL_SUPPORT));
   
   uchar array[];
   PRTF(ArrayResize(array, 1024 * 1024 * 10)); // allocate 10 Mb
   PRTF(MQLInfoInteger(MQL_MEMORY_USED));
   PRTF(TerminalInfoInteger(TERMINAL_MEMORY_AVAILABLE));
   PRTF(TerminalInfoInteger(TERMINAL_MEMORY_USED));
}
After the initial output of the properties, we allocate 1 0 Mb for the array and then check the memory
again. A result example is shown below (you will have your own values).
MQLInfoInteger(MQL_MEMORY_LIMIT)=8388608 / ok
MQLInfoInteger(MQL_MEMORY_USED)=1 / ok
TerminalInfoInteger(TERMINAL_MEMORY_PHYSICAL)=4095 / ok
TerminalInfoInteger(TERMINAL_MEMORY_TOTAL)=8190 / ok
TerminalInfoInteger(TERMINAL_MEMORY_AVAILABLE)=7842 / ok
TerminalInfoInteger(TERMINAL_MEMORY_USED)=348 / ok
TerminalInfoInteger(TERMINAL_DISK_SPACE)=4528 / ok
TerminalInfoInteger(TERMINAL_CPU_CORES)=4 / ok
TerminalInfoInteger(TERMINAL_OPENCL_SUPPORT)=0 / ok
ArrayResize(array,1024*1024*10)=10485760 / ok
MQLInfoInteger(MQL_MEMORY_USED)=11 / ok
TerminalInfoInteger(TERMINAL_MEMORY_AVAILABLE)=7837 / ok
TerminalInfoInteger(TERMINAL_MEMORY_USED)=353 / ok
Note that the total virtual memory (81 90) is twice the physical memory (4095). The amount of
memory available for the script is 8388608 Kb, which is almost equal to the entire memory of 81 90
Mb. Free (7842) and used (348) system memory also add up to 81 90.
If before allocating memory for an array, the MQL program occupied 1  Mb, then after allocating it, it is
already 1 1  Mb. Meanwhile, the amount of memory occupied by the terminal increased by only 5 Mb
(from 348 to 353), since some resources were reserved in advance.
4.9.8 Screen specifications
Several properties provided by the function TerminalInfoInteger, refer to the video subsystem of the
computer.

---

## Page 545

Part 4. Common APIs
545
4.9 MQL program execution environment
Identifier
Description
TERMINAL_SCREEN_DPI
Resolution of information output to the screen is measured in
the number of dots per linear inch (DPI, Dots Per Inch)
TERMINAL_SCREEN_LEFT
Left coordinate of the virtual screen
TERMINAL_SCREEN_TOP
Top coordinate of the virtual screen
TERMINAL_SCREEN_WIDTH
Virtual screen width
TERMINAL_SCREEN_HEIGHT
Virtual screen height
TERMINAL_LEFT
Left coordinate of the terminal relative to the virtual screen
TERMINAL_TOP
Top coordinate of the terminal relative to the virtual screen
TERMINAL_RIGHT
Right coordinate of the terminal relative to the virtual screen
TERMINAL_BOTTOM
Bottom coordinate of the terminal relative to the virtual
screen
Knowing the TERMINAL_SCREEN_DPI parameter, you can set the dimensions of graphic objects so that
they look the same on monitors with different resolutions. For example, if you want to create a button
with a visible size of X centimeters, then you can specify it as the number of screen dots (pixels) using
the following function:
int cm2pixels(const double x)
{
   static const double inch2cm = 2.54; // 1 inch equals 2.54 cm
   return (int)(x / inch2cm * TerminalInfoInteger(TERMINAL_SCREEN_DPI));
}
The virtual screen is a bounding box of all monitors. If there is more than one monitor in the system
and the order of their arrangement differs from strictly left to right, then the left coordinate of the
virtual screen may turn out to be negative, and the center (reference point) will be on the border of two
monitors (in the upper left corner of the main monitor).
Virtual screen from multiple monitors

---

## Page 546

Part 4. Common APIs
546
4.9 MQL program execution environment
If the system has one monitor, then the size of the virtual screen fully corresponds to it.
The terminal coordinates do not take into account its possible current maximization (that is, if the main
window is maximized, the properties return the unmaximized size, although the terminal is expanded to
the entire monitor).
In the EnvScreen.mq5 script, check reading screen properties.
void OnStart()
{
   PRTF(TerminalInfoInteger(TERMINAL_SCREEN_DPI));
   PRTF(TerminalInfoInteger(TERMINAL_SCREEN_LEFT));
   PRTF(TerminalInfoInteger(TERMINAL_SCREEN_TOP));
   PRTF(TerminalInfoInteger(TERMINAL_SCREEN_WIDTH));
   PRTF(TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT));
   PRTF(TerminalInfoInteger(TERMINAL_LEFT));
   PRTF(TerminalInfoInteger(TERMINAL_TOP));
   PRTF(TerminalInfoInteger(TERMINAL_RIGHT));
   PRTF(TerminalInfoInteger(TERMINAL_BOTTOM));
}
Here is an example of the resulting log entries.
TerminalInfoInteger(TERMINAL_SCREEN_DPI)=96 / ok
TerminalInfoInteger(TERMINAL_SCREEN_LEFT)=0 / ok
TerminalInfoInteger(TERMINAL_SCREEN_TOP)=0 / ok
TerminalInfoInteger(TERMINAL_SCREEN_WIDTH)=1440 / ok
TerminalInfoInteger(TERMINAL_SCREEN_HEIGHT)=900 / ok
TerminalInfoInteger(TERMINAL_LEFT)=126 / ok
TerminalInfoInteger(TERMINAL_TOP)=41 / ok
TerminalInfoInteger(TERMINAL_RIGHT)=1334 / ok
TerminalInfoInteger(TERMINAL_BOTTOM)=836 / ok
In addition to the general sizes of the screen and the terminal window, MQL programs quite often need
to analyze the current size of the chart (daughter window inside the terminal). For these purposes,
there is a special set of functions (in particular, ChartGetInteger), which we will discuss in the Charts
section.
4.9.9 Terminal and program string properties
The MQLInfoString and TerminalInfoString functions can be used to find out several string properties of
the terminal and MQL program.
Identifier
Description
MQL_PROGRAM_NAME
The name of the running MQL program
MQL_PROGRAM_PATH
Path for this running MQL program
TERMINAL_LANGUAGE
Terminal language
TERMINAL_COMPANY
Name of the company (broker)

---

## Page 547

Part 4. Common APIs
547
4.9 MQL program execution environment
Identifier
Description
TERMINAL_NAME
Terminal name
TERMINAL_PATH
The folder from which the terminal is launched
TERMINAL_DATA_PATH
The folder where terminal data is stored
TERMINAL_COMMONDATA_PATH
The shared folder of all client terminals installed on the
computer
The name of the running program (MQL_PROGRAM_NAME) usually coincides with the name of the main
module (mq5 file) but may differ. In particular, if your source code compiles to a library which is
imported into another MQL program (Expert Advisor, indicator, script, or service), then the
MQL_PROGRAM_NAME property will return the name of the main program, not the library (the library is
not an independent program that can be run).
We discussed the arrangement of working terminal folders in Working with files. Using the listed
properties, you can find out where the terminal is installed (TERMINAL_PATH), as well as find the
working data of the current terminal instance (TERMINAL_DATA_PATH) and of all instances
(TERMINAL_COMMONDATA_PATH).
A simple script EnvDescription.mq5 logs all these properties.
void OnStart()
{
   PRTF(MQLInfoString(MQL_PROGRAM_NAME));
   PRTF(MQLInfoString(MQL_PROGRAM_PATH));
   PRTF(TerminalInfoString(TERMINAL_LANGUAGE));
   PRTF(TerminalInfoString(TERMINAL_COMPANY));
   PRTF(TerminalInfoString(TERMINAL_NAME));
   PRTF(TerminalInfoString(TERMINAL_PATH));
   PRTF(TerminalInfoString(TERMINAL_DATA_PATH));
   PRTF(TerminalInfoString(TERMINAL_COMMONDATA_PATH));
}
Below is an example result.
MQLInfoString(MQL_PROGRAM_NAME)=EnvDescription / ok
MQLInfoString(MQL_PROGRAM_PATH)= »
» C:\Program Files\MT5East\MQL5\Scripts\MQL5Book\p4\EnvDescription.ex5 / ok
TerminalInfoString(TERMINAL_LANGUAGE)=Russian / ok
TerminalInfoString(TERMINAL_COMPANY)=MetaQuotes Software Corp. / ok
TerminalInfoString(TERMINAL_NAME)=MetaTrader 5 / ok
TerminalInfoString(TERMINAL_PATH)=C:\Program Files\MT5East / ok
TerminalInfoString(TERMINAL_DATA_PATH)=C:\Program Files\MT5East / ok
TerminalInfoString(TERMINAL_COMMONDATA_PATH)= »
» C:\Users\User\AppData\Roaming\MetaQuotes\Terminal\Common / ok
The interface language of the terminal can be found not only as a string in the TERMINAL_LANGUAGE
property but also as a code page number (see the TERMINAL_CODEPAGE property in the next section).

---

## Page 548

Part 4. Common APIs
548
4.9 MQL program execution environment
4.9.1 0 Custom properties: Bar limit and interface language
Among the properties of the terminal, there are two special properties which the user can change
interactively. These include the default maximum number of bars displayed on each chart (it
corresponds to the value of the Max. bars in the window field in the Options dialog, as well as the
interface language (selected using the View -> Languages command).
Identifier
Description
TERMINAL_MAXBARS
Maximum number of bars on the chart
TERMINAL_CODEPAGE
Code page number of the language selected in the client
terminal
Please note that the TERMINAL_MAXBARS value sets the upper limit for displaying bars, but in fact,
their number may be less if the depth of the available quotes history is not sufficient on any timeframe.
On the other hand, the length of the history may also exceed the specified limit TERMINAL_MAXBARS.
Then you can find the number of potentially available bars using the function from the timeseries
property group: SeriesInfoInteger with the SERIES_BARS_COUNT property. Please note that the
TERMINAL_MAXBARS value directly affects the consumption of RAM.
4.9.1 1  Binding a program to runtime properties
As an example of working with the properties described in the previous sections, let's consider the
popular task of binding an MQL program to a hardware environment to protect it from copying. When
the program is distributed through the MQL5 Market, the binding is provided by the service itself.
However, if the program is developed on a custom basis, it can be linked either to the account number,
or to the name of the client, or to the available properties of the terminal (computer). The first is not
always convenient, because many traders have several live accounts (probably with different brokers),
not to mention demo accounts with a limited validity period. The second may be fictional or too
commonplace. Therefore, we will implement a prototype algorithm for binding a program to a selected
set of environment properties. More serious security schemes could probably use a DLL and directly
read device hardware labels from Windows, but not every client will agree to run potentially unsafe
libraries.
Our protection option is presented in the script EnvSignature.mq5. The script calculates hashes from
the given properties of the environment and creates a unique signature (imprint) based on them.
Hashing is a special processing of arbitrary information, as a result of which a new block of data is
created that has the following characteristics (they are guaranteed by the algorithm used):
·Matching hash values for two original data sets means, with almost 1 00% probability, that the data
are identical (the probability of a random match is negligible).
·If the original data changes, their hash value will also change.
·It is impossible to mathematically restore the original data from the hash value (they remain
secret) unless a complete enumeration of possible initial values is performed (if their initial size
increases and there is no information about their structure, the problem is unsolvable in the
foreseeable future).
·The hash size is fixed (does not depend on the amount of initial data).

---

## Page 549

Part 4. Common APIs
549
4.9 MQL program execution environment
Suppose one of the environment properties is described by the string:
"TERMINAL_LANGUAGE=German". It can be obtained with a simple statement like the following
(simplified):
string language = EnumToString(TERMINAL_LANGUAGE) +
            "=" + TerminalInfoString(TERMINAL_LANGUAGE);
The actual language will match the settings. Having a hypothetical Hash function, we can compute the
signature.
string signature = Hash(language);
When there are more properties, we simply repeat the procedure for all of them, or request a hash
from the combined strings (so far this is pseudo-code, not part of the real program).
string properties[];
// fill in the property lines as you wish
// ...
string signature;
for(int i = 0; i < ArraySize(properties); ++i)
{
   signature += properties[i];
}
return Hash(signature);
The received signature can be reported by the user to the program developer, who will "sign" it in a
special way, upon receiving a validation string suitable only for this signature. The signature is also
based on hashing and requires knowledge of some secret (password phrase), known only to the
developer and hard-coded into the program (for the verification phase).
The developer will pass the validation string to the user who then will be able to run the program by
specifying this string in the parameters.
When launched without a validation string, the program should generate a new signature for the current
environment, print it to the log, and exit (this information should be passed to the developer). With an
invalid validation string, the program should display an error message and exit.
Several launch modes can be provided for the developer himself: with a signature, but without a
validation string (to generate the last one), or with a signature and a validation string (here the
program will re-sign the signature and compare it with the specified validation string just for checking).
Let's estimate how selective such protection will be. After all, the binding here is not performed to a
unique identifier of anything.
The following table provides statistics on two characteristics: screen size and RAM. Obviously, the
values will change over time, but the approximate distribution will remain the same: a few characteristic
values will be the most popular, while some "new" advanced and "old" ones that are going out of
circulation will make up decreasing "tails".

---

## Page 550

Part 4. Common APIs
550
4.9 MQL program execution environment
Screen
1 920x1 080
1 536x864
1 440x900
1 366x768
800x600
RAM
21 %
7%
5%
1 0%
4%
4Gb    20%
4.20
1 .40
1 .00
2.0
0.8
8Gb    20%
4.20
1 .40
1 .00
2.0
0.8
1 6Gb  1 5%
3.1 5
1 .05
0.75
1 .5
0.6
32Gb  1 0%
2.1 0
0.70
0.50
1 .0
0.4
64Gb    5%
1 .05
0.35
0.25
0.5
0.2
Pay attention to the cells with the largest values, because they mean the same signatures (unless we
introduce an element of randomness into them, which will be discussed below). In this case, two
combinations of characteristics in the upper left corner are most likely, with each at 4.2%. But these
are only two features. If you add the interface language, time zone, number of cores, and working data
path (preferably shared, since it contains the Windows username) to the evaluated environment, then
the number of potential matches will noticeably decrease.
For hashing, we use the built-in CryptEncode function (it will be described in the Cryptography section)
that supports the SHA256 hashing method. As its name suggests, it produces a hash that is 256 bits
long, i.e., 32 bytes. If we needed to show it to the user, then we would translate it into text in
hexadecimal representation and get a 64-character long string.
To make the signature shorter, we will convert it using Base64 encoding (it is also supported by the
CryptEncode function and its counterpart CryptDecode), which will give a 44-character long string.
Unlike a one-way hash operation, Base64 encoding is reversible, i.e. the original data can be recovered
from it.  
The main operations are implemented by the EnvSignature class. It defines the data string which should
accumulate certain fragments describing the environment. The public interface consists of several
overloaded versions of the append function to add strings with environment properties. Essentially, they
join the name of the requested property and its value using some abstract element returned by the
virtual 'pepper' method as a link. The derived class will define it as a specific string (but it can be
empty).

---

## Page 551

Part 4. Common APIs
551 
4.9 MQL program execution environment
class EnvSignature
{
private:
   string data;
protected:
   virtual string pepper() = 0;
public:
   bool append(const ENUM_TERMINAL_INFO_STRING e)
   {
      return append(EnumToString(e) + pepper() + TerminalInfoString(e));
   }
   bool append(const ENUM_MQL_INFO_STRING e)
   {
      return append(EnumToString(e) + pepper() + MQLInfoString(e));
   }
   bool append(const ENUM_TERMINAL_INFO_INTEGER e)
   {
      return append(EnumToString(e) + pepper()
        + StringFormat("%d", TerminalInfoInteger(e)));
   }
   bool append(const ENUM_MQL_INFO_INTEGER e)
   {
      return append(EnumToString(e) + pepper()
        + StringFormat("%d", MQLInfoInteger(e)));
   }
To add an arbitrary string to an object, there is a generic method append, which is called in the above
methods.
   bool append(const string s)
   {
      data += s;
      return true;
   }
Optionally, the developer can add a so-called "salt" to the hashed data. This is an array with randomly
generated data which further complicates hash reversal. Each generation of the signature will be
different from the previous one, even though the environment remains constant. The implementation of
this feature as well as of other more specific protection aspects (such as the use of symmetric
encryption and dynamic calculation of the secret) are left for independent study.
Since the environment consists of well-known properties (their list is limited by MQL5 API constants),
and not all of them are sufficiently unique, our defense, as we calculated, can generate the same
signatures for different users if we do not use the salt. The signature match will not allow identifying the
source of the license leak if it happened.
Therefore, you can increase the effectiveness of protection by changing the method of presenting
properties before hashing for each customer. Of course, the method itself should not be disclosed. In
the considered example, this implies changing the contents of the pepper method and recompiling the
product. This can be expensive, but it allows you to avoid using random salt.
With the property string filled in, we can generate a signature. This is done using the emit method.

---

## Page 552

Part 4. Common APIs
552
4.9 MQL program execution environment
   string emit() const
   {
      uchar pack[];
      if(StringToCharArray(data + secret(), pack, 0, 
         StringLen(data) + StringLen(secret()), CP_UTF8) <= 0) return NULL;
   
      uchar key[], result[];
      if(CryptEncode(CRYPT_HASH_SHA256, pack, key, result) <= 0) return NULL;
      Print("Hash bytes:");
      ArrayPrint(result);
   
      uchar text[];
      CryptEncode(CRYPT_BASE64, result, key, text);
      return CharArrayToString(text);
   }
The method adds a certain secret (a sequence of bytes known only to the developer and located inside
the program) to the data and calculates the hash for the shared string. The secret is obtained from the
virtual secret method, which will also define the derived class.
The resulting byte array with the hash is encoded into a string using Base64.
Now comes the most important class function: check. It is this function that implements the signature
from the developer and checks it from the user.

---

## Page 553

Part 4. Common APIs
553
4.9 MQL program execution environment
   bool check(const string sig, string &validation)
   {
      uchar bytes[];
      const int n = StringToCharArray(sig + secret(), bytes, 0, 
         StringLen(sig) + StringLen(secret()), CP_UTF8);
      if(n <= 0) return false;
      
      uchar key[], result1[], result2[];
      if(CryptEncode(CRYPT_HASH_SHA256, bytes, key, result1) <= 0) return false;
      
      /*
        WARNING
        The following code should only be present in the developer utility.
        The program supplied to the user must compile without this if.
      */
      #ifdef I_AM_DEVELOPER
      if(StringLen(validation) == 0)
      {
         if(CryptEncode(CRYPT_BASE64, result1, key, result2) <= 0) return false;
         validation = CharArrayToString(result2);
         return true;
      }
      #endif
      uchar values[];
      // the exact length is needed to not append terminating '0'
      if(StringToCharArray(validation, values, 0, 
         StringLen(validation)) <= 0) return false;
      if(CryptDecode(CRYPT_BASE64, values, key, result2) <= 0) return false;
      
      return ArrayCompare(result1, result2) == 0;
   }
During normal operation (for the user), the method calculates the hash from the received signature,
supplemented by the secret, and compares it with the value from the validation string (it must first be
decoded from Base64 into the raw binary representation of the hash). If the two hashes match, the
validation is successful: the validation string matches the property set. Obviously, an empty validation
string (or a string entered at random) will not pass the test.
On the developer's machine, the I_AM_DEVELOPER macro must be defined in the source code for the
signature utility, which results in an empty validation string being handled differently. In this case, the
resulting hash is Base64 encoded, and this string is passed out through the validation parameter. Thus,
the utility will be able to display a ready-made validation string for the given signature to the developer.
To create an object, you need a certain derived class that defines strings with the secret and pepper.

---

## Page 554

Part 4. Common APIs
554
4.9 MQL program execution environment
// WARNING: change the macro to your own set of random bytes
#define PROGRAM_SPECIFIC_SECRET "<PROGRAM-SPECIFIC-SECRET>"
// WARNING: choose your characters to link in pairs name'='value 
#define INSTANCE_SPECIFIC_PEPPER "=" // obvious single sign is selected for demo
// WARNING: the following macro needs to be disabled in the real product,
//          it should only be in the signature utility
#define I_AM_DEVELOPER
#ifdef I_AM_DEVELOPER
#define INPUT input
#else
#define INPUT const
#endif
INPUT string Signature = "";
INPUT string Secret = PROGRAM_SPECIFIC_SECRET;
INPUT string Pepper = INSTANCE_SPECIFIC_PEPPER;
class MyEnvSignature : public EnvSignature
{
protected:
   virtual string secret() override
   {
      return Secret;
   }
   virtual string pepper() override
   {
      return Pepper;
   }
};
Let's quickly pick a few properties to fill in the signature.
void FillEnvironment(EnvSignature &env)
{
   // the order is not important, you can mix
   env.append(TERMINAL_LANGUAGE);
   env.append(TERMINAL_COMMONDATA_PATH);
   env.append(TERMINAL_CPU_CORES);
   env.append(TERMINAL_MEMORY_PHYSICAL);
   env.append(TERMINAL_SCREEN_DPI);
   env.append(TERMINAL_SCREEN_WIDTH);
   env.append(TERMINAL_SCREEN_HEIGHT);
   env.append(TERMINAL_VPS);
   env.append(MQL_PROGRAM_TYPE);
}
Now everything is ready to test our protection scheme in the OnStart function. But first, let's look at
the input variables. Since the same program will be compiled in two versions, for the end user and for
the developer, there are two sets of input variables: for entering registration data by the user and for
generating this data based on the developer's signature. The input variables intended for the developer
have been described above using the INPUT macro. Only the validation string is available to the user.

---

## Page 555

Part 4. Common APIs
555
4.9 MQL program execution environment
input string Validation = "";
When the string is empty, the program will collect the environment data, generate a new signature, and
print it to the log. This completes the work of the script since access to the useful code has not yet
been confirmed.
void OnStart()
{
   MyEnvSignature env;
    string signature;
   if(StringLen(Signature) > 0)
   {
     // ... here will be the code to be signed by the author
   }
   else
   {
      FillEnvironment(env);
      signature = env.emit();
   }
   
   if(StringLen(Validation) == 0)
   {
      Print("Validation string from developer is required to run this script");
      Print("Environment Signature is generated for current state...");
      Print("Signature:", signature);
      return;
   }
   else
   {
     // ... check the validation string here
   }
   Print("The script is validated and running normally");
   // ... actual working code is here
}
If the variable Validation is filled, we check its compliance with the signature and terminate the work in
case of failure.

---

## Page 556

Part 4. Common APIs
556
4.9 MQL program execution environment
   if(StringLen(Validation) == 0)
   {
      ...
   }
   else
   {
      validation = Validation; // need a non-const argument
      const bool accessGranted = env.check(Signature, validation);
      if(!accessGranted)
      {
         Print("Wrong validation string, terminating");
         return;
      }
      // success
   }
   Print("The script is validated and running normally");
   // ... actual working code is here
}
If there are no discrepancies, the algorithm proceeds to the working code of the program.
On the developer's side (in the version of the program that was built with the I_AM_DEVELOPER
macro), a signature can be introduced. We restore the state of the MyEnvSignature object using the
signature and calculate the validation string.
void OnStart()
{
   ...
   if(StringLen(Signature) > 0)
   {
      #ifdef I_AM_DEVELOPER
      if(StringLen(Validation) == 0)
      {
         string validation;
         if(env.check(Signature, validation))
           Print("Validation:", validation);
         return;
      }
      signature = Signature; 
      #endif
   }
   ...
The developer can not only specify the signature but also validate it: in this case, the code execution
will continue in the user mode (for debugging purposes).
If you wish, you can simulate a change in the environment, for example, as follows:

---

## Page 557

Part 4. Common APIs
557
4.9 MQL program execution environment
      FillEnvironment(env);
      // artificially make a change in the environment (add a time zone)
      // env.append("Dummy" + (string)(TimeGMTOffset() - TimeDaylightSavings()));
      const string update = env.emit();
      if(update != signature)
      {
         Print("Signature and environment mismatch");
         return;
      }
Let's look at a few test logs.
When you first run the EnvSignature.mq5 script, the "user" will see something like the following log
(values will vary due to environment differences):
Hash bytes:
  4 249 194 161 242  28  43  60 180 195  54 254  97 223 144 247 216 103 238 245 244 224   7  68 101 253 248 134  27 102 202 153
Validation string from developer is required to run this script
Environment Signature is generated for current state...
Signature:BPnCofIcKzy0wzb+Yd+Q99hn7vX04AdEZf34hhtmypk=
It sends the generated signature to the "developer" (there are no actual users during the test, so all
the roles of "user" and "developer" are quoted), who enters it into the signing utility (compiled with the
I_AM_DEVELOPER macro), in the Signature parameter. As a result, the program will generate a
validation string:
Validation:YBpYpQ0tLIpUhBslIw+AsPhtPG48b0qut9igJ+Tk1fQ=
The "developer" sends it back to the "user", and the "user", by entering it into the Validation
parameter, will get the activated script:
Hash bytes:
  4 249 194 161 242  28  43  60 180 195  54 254  97 223 144 247 216 103 238 245 244 224   7  68 101 253 248 134  27 102 202 153
The script is validated and running normally
To demonstrate the effectiveness of protection, let's duplicate the script as a service: to do this, let's
copy the file to the folder MQL5/Services/MQL5Book/p4/ and replace the following line in the source
code:
#property script_show_inputs
with the following line:
#property service
Let's compile the service, create and run its instance, and specify the previously received validation
string in the input parameters. As a result, the service will abort (before reaching the statements with
the required code) with the following message:
Hash bytes:
147 131  69  39  29 254  83 141  90 102 216 180 229 111   2 246 245  19  35 205 223 145 194 245  67 129  32 108 178 187 232 113
Wrong validation string, terminating
The point is that among the properties of the environment we have used the string
MQL_PROGRAM_TYPE. Therefore, an issued license for one type of program will not work for another
type of program, even if it is running on the same user's computer.

---

## Page 558

Part 4. Common APIs
558
4.9 MQL program execution environment
4.9.1 2 Checking keyboard status
The TerminalInfoInteger function can be used to find out the state of the control keys, which are also
called virtual. These include, in particular, Ctrl, Alt, Shift, Enter, Ins, Del, Esc, arrows, and so on. They
are called virtual because keyboards, as a rule, provide several ways to generate the same control
action. For example, Ctrl, Shift, and Alt are duplicated to the left and right of the spacebar, while the
cursor can be moved both by dedicated keys and by the main ones when Fn is pressed. Thus, this
function cannot distinguish between control methods at the physical level (for example, the left and
right Shift).
The API defines constants for the following keys:
Identifier
Description
TERMINAL_KEYSTATE_LEFT
Left Arrow
TERMINAL_KEYSTATE_UP
Up Arrow
TERMINAL_KEYSTATE_RIGHT
Right Arrow
TERMINAL_KEYSTATE_DOWN
Down Arrow
TERMINAL_KEYSTATE_SHIFT
Shift
TERMINAL_KEYSTATE_CONTROL
Ctrl
TERMINAL_KEYSTATE_MENU
Windows
TERMINAL_KEYSTATE_CAPSLOCK
CapsLock
TERMINAL_KEYSTATE_NUMLOCK
NumLock
TERMINAL_KEYSTATE_SCRLOCK
ScrollLock
TERMINAL_KEYSTATE_ENTER
Enter
TERMINAL_KEYSTATE_INSERT
Insert
TERMINAL_KEYSTATE_DELETE
Delete
TERMINAL_KEYSTATE_HOME
Home
TERMINAL_KEYSTATE_END
End
TERMINAL_KEYSTATE_TAB
Tab
TERMINAL_KEYSTATE_PAGEUP
PageUp
TERMINAL_KEYSTATE_PAGEDOWN
PageDown
TERMINAL_KEYSTATE_ESCAPE
Escape
The function returns a two-byte integer value that reports the current state of the requested key using
a pair of bits.
The least significant bit keeps track of keystrokes since the last function call. For example, if
TerminalInfoInteger(TERMINAL_ KEYSTATE_ ESCAPE) returned 0 at some point, and then the user

---

## Page 559

Part 4. Common APIs
559
4.9 MQL program execution environment
pressed Escape, then on the next call, TerminalInfoInteger(TERMINAL_ KEYSTATE_ ESCAPE) will return
1 . If the key is pressed again, the value will return to 0.
For keys responsible for switching input modes, such as CapsLock, NumLock, and ScrollLock, the
position of the bit indicates whether the corresponding mode is enabled or disabled.
The most significant bit of the second byte (0x8000) is set if the key is pressed (and not released) at
the current moment.
This feature cannot be used to track pressing of alphanumeric and functional keys. For this purpose, it
is necessary to implement the OnChartEvent handler and intercept messages with the
CHARTEVENT_KEYDOWN code in the program. Please note that events are generated on the chart and
are only available for Expert Advisors and indicators. Programs of other types (scripts and services) do
not support the event programming model.
The EnvKeys.mq5 script includes a loop through all TERMINAL_KEYSTATE constants.
void OnStart()
{
   for(ENUM_TERMINAL_INFO_INTEGER i = TERMINAL_KEYSTATE_TAB;
      i <= TERMINAL_KEYSTATE_SCRLOCK; ++i)
   {
      const string e = EnumToString(i);
      // skip values that are not enum elements
      if(StringFind(e, "ENUM_TERMINAL_INFO_INTEGER") == 0) continue;
      PrintFormat("%s=%4X", e, (ushort)TerminalInfoInteger(i));
   }
}
You can experiment with keystrokes and enable/disable keyboard modes to see how the values change
in the log.
For example, if capitalization is disabled by default, we will see the following log:
TERMINAL_KEYSTATE_SCRLOCK= 0
If we press the ScrollLock key and, without releasing it, run the script again, we get the following log:
TERMINAL_KEYSTATE_CAPSLOCK=8001
That is, the mode is already on and the key is pressed. Let's release the key, and the next time the
script will return:
TERMINAL_KEYSTATE_SCRLOCK= 1
The mode remained on, but the key was released.
TerminalInfoInteger is not suitable for checking the status of keys (TERMINAL_KEYSTATE_XYZ) in
dependent indicators created by the iCustom or IndicatorCreate call. In them, the function always
returns 0, even if the indicator was added to the chart using ChartIndicatorAdd.
Also, the function does not work when the MQL program chart is not active (the user has switched to
another one). MQL5 does not provide means for permanent control of the keyboard.

---

## Page 560

Part 4. Common APIs
560
4.9 MQL program execution environment
4.9.1 3 Checking the MQL program status and reason for termination
We have already encountered the IsStopped function in different examples across the book. It must be
called from time to time in cases where the MQL program performs lengthy calculations. This allows
you to check if the user initiated the closing of the program (i.e. if they tried to remove it from the
chart).
bool IsStopped() ≡ bool _StopFlag
The function returns true if the program was interrupted by the user (for example, by pressing the
Delete button in the dialog opened by the Expert List command in the context menu).
The program is given 3 seconds to properly pause calculations, save intermediate results if necessary,
and complete its work. If this does not happen, the program will be removed from the chart forcibly.
Instead of the IsStopped function, you can check the value of the built-in _ StopFlag variable.
The test script EnvStop.mq5 emulates lengthy calculations in a loop: search for prime numbers.
Conditions for exiting the while loop are written using the IsStopped function. Therefore, when the user
deletes the script, the loop is interrupted in the usual way and the log displays the statistics of found
prime numbers log (the script could also save the numbers to a file).

---

## Page 561

Part 4. Common APIs
561 
4.9 MQL program execution environment
bool isPrime(int n)
{
   if(n < 1) return false;
   if(n <= 3) return true;
   if(n % 2 == 0) return false;
   const int p = (int)sqrt(n);
   int i = 3;
   for( ; i <= p; i += 2)
   {
      if(n % i == 0) return false;
   }
   
   return true;
}
   
void OnStart()
{
   int count = 0;
   int candidate = 1;
   
   while(!IsStopped()) // try to replace it with while(true)
   {
      // emulate long calculations
      if(isPrime(candidate))
      {
         Comment("Count:", ++count, ", Prime:", candidate);
      }
      ++candidate;
      Sleep(10);
   }
   Comment("");
   Print("Total found:", count);
}
If we replace the loop condition with true (infinite loop), the script will stop responding to the user's
request to stop and will be unloaded from the chart forcibly. As a result, we will see the "Abnormal
termination" error in the log, and the comment in the upper left corner of the window remains
uncleaned. Thus, all instructions that in this example symbolize saving data and clearing busy resources
(and this could be, for example, deleting your own graphic objects from the window) are ignored.
After a stop request has been sent to the program (and the value _ StopFlag equals true), the reason
for the termination can be found using the UninitializeReason function.
Unfortunately, this feature is only available for Expert Advisors and indicators.
int UninitializeReason() ≡ int _UninitReason
The function returns one of the predefined codes describing the reasons for deinitialization.

---

## Page 562

Part 4. Common APIs
562
4.9 MQL program execution environment
Constant
Value
Description
REASON_PROGRAM
0
ExpertRemove function only available in Expert
Advisors and scripts was called
REASON_REMOVE
1
Program removed from the chart
REASON_RECOMPILE
2
Program recompiled
REASON_CHARTCHANGE
3
Chart symbol or period changed
REASON_CHARTCLOSE
4
Chart closed
REASON_PARAMETERS
5
Program input parameters changed
REASON_ACCOUNT
6
Another account is connected or a reconnection
to the trading server occurred
REASON_TEMPLATE
7
Another chart template applied
REASON_INITFAILED
8
OnInit event handler returned an error flag
REASON_CLOSE
9
Terminal closed
Instead of a function, you can access the built-in global variable _ UninitReason.
The deinitialization reason code is also passed as a parameter to the OnDeinit event handler function.
Later, when studying Program start and stop features, we will see an indicator
(Indicators/MQL5Book/p5/LifeCycle.mq5) and an Expert Advisor (Experts/MQL5Book/p5/LifeCycle.mq5)
that log the reasons for deinitialization and allow you to explore the behavior of programs depending on
user actions.
4.9.1 4 Programmatically closing the terminal and setting a return code
The MQL5 API contains several functions not only for reading but also for modifying the program
environment. One of the most radical of them is TerminalClose. Using this function, an MQL program
can close the terminal (without user confirmation!).
bool TerminalClose(int retcode)
The function has one parameter retcode which is the code returned by the terminal64.exe process to
the Windows operating system. Such codes can be analyzed in batch files (*.bat and *.cmd), as well as
in shell scripts (Windows Script Host (WSH), which supports VBScript and JScript, or Windows
PowerShell (WPS), with .ps* files) and other automation tools (for example, the built-in Windows
scheduler, the Linux support subsystem under Windows with *.sh files, etc.).
The function does not immediately stop the terminal, but sends a termination command to the
terminal.
If the result of the call is true, it means that the command has been successfully "accepted for
consideration", and the terminal will try to close as quickly as possible, but correctly (generating a
notification and stopping other running MQL programs). In the calling code, of course, all preparations
must also be made for the immediate termination of work (in particular, all previously opened files
should be closed), and after the function call, control should be returned to the terminal.

---

## Page 563

Part 4. Common APIs
563
4.9 MQL program execution environment
Another function associated with the process return code is SetReturnError. It allows you to pre-assign
this code without sending an immediate close command.
void SetReturnError(int retcode)
The function sets the code that the terminal process will return to the Windows system after closing.
Please note that the terminal does not need to be forcibly closed by the TerminalClose function.
Regular closing of the terminal by the user will also occur with the specified code. Also, this code will
enter the system if the terminal closes due to an unexpected critical error.
If the SetReturnError function was called repeatedly and/or from different MQL programs, the terminal
will return the last set code.
Let's test these functions using the EnvClose.mq5 script.
#property script_show_inputs
   
input int ReturnCode = 0;
input bool CloseTerminalNow = false;
   
void OnStart()
{
   if(CloseTerminalNow)
   {
      TerminalClose(ReturnCode);
   }
   else
   {
      SetReturnError(ReturnCode);
   }
}
To test it in action, we also need the file envrun.bat (located in the folder MQL5/Files/MQL5Book/).
terminal64.exe
@echo Exit code: %ERRORLEVEL%
In fact, it only launches the terminal, and after its completion displays the resulting code to the
console. The file should be placed in the terminal folder (or the current instance of MetaTrader 5 from
among several installed in the system should be registered in the PATH system variable).
For example, if we start the terminal using the bat file, and execute the script EnvClose.mq5, for
example, with parameters ReturnCode=1 00, CloseTerminalNow=true, we will see something like this in
the console:
Microsoft Windows [Version 10.0.19570.1000]
(c) 2020 Microsoft Corporation. All rights reserved.
C:\Program Files\MT5East>envrun
C:\Program Files\MT5East>terminal64.exe
Exit code: 100
C:\Program Files\MT5East>
As a reminder, MetaTrader 5 supports various options when launched from the command line (see
details in the documentation section Running the trading platform). Thus, it is possible to organize, for

---

## Page 564

Part 4. Common APIs
564
4.9 MQL program execution environment
example, batch testing of various Expert Advisors or settings, as well as sequential switching between
thousands of monitored accounts, which would be unrealistic to achieve with the constant parallel
operation of so many instances on one computer.
4.9.1 5 Handling runtime errors
Any program written correctly enough to compile without errors is still not immune to runtime errors.
They can occur both due to an oversight of the developer and due to unforeseen circumstances that
have arisen in the software environment (such as Internet connection loss, running out of memory,
etc.). But no less likely is the situation when the error occurs due to incorrect application of the
program. In all these cases, the program must be able to analyze the essence of the problem and
process it adequately.
Each MQL5 statement is a potential source of runtime errors. If such an error occurs, the terminal
saves a descriptive code to the special _ LastError variable. Make sure to analyze the code immediately
after each statement, since potential errors in subsequent statements can overwrite this value.
Please note that there are a number of critical errors that will immediately abort program execution
when they occur:
• Zero divide
• Index out of range
• Incorrect object pointer
For a complete list of error codes and what they mean, see the documentation.
In the Opening and closing files section, we've already addressed the problem of diagnosing errors as
part of writing a useful PRTF macro. There, in particular, we have seen an auxiliary header file
MQL5/Include/MQL5Book/MqlError.mqh, in which the MQL_ERROR enumeration allow easy conversion of
the numeric error code into a name using EnumToString.
enum MQL_ERROR
{
   SUCCESS = 0, 
   INTERNAL_ERROR = 4001, 
   WRONG_INTERNAL_PARAMETER = 4002, 
   INVALID_PARAMETER = 4003, 
   NOT_ENOUGH_MEMORY = 4004, 
   ...
   // start of area for errors defined by the programmer (see next section)
   USER_ERROR_FIRST = 65536, 
};
#define E2S(X) EnumToString((MQL_ERROR)(X))
Here, as the X parameter of the E2S macro, we should have the _ LastError variable or its equivalent
GetLastError function.
int GetLastError() ≡ int _LastError
The function returns the code of the last error that occurred in the MQL program statements. Initially,
while there are no errors, the value is 0. The difference between reading _ LastError and calling the
GetLastError function is purely syntactic (choose the appropriate option in accordance with the
preferred style).

---

## Page 565

Part 4. Common APIs
565
4.9 MQL program execution environment
It should be borne in mind that regular error-free execution of statements does not reset the error
code. Calling GetLastError also does not do it.
Thus, if there is a sequence of actions, in which only one will set an error flag, this flag will be returned
by the function for subsequent (successful) actions. For example,
// _LastError = 0 by default
action1; // ok, _LastError doe not change
action2; // error, _LastError = X
action3; // ok, _LastError does not change, i.e. is still equal to X
action4; // another error, _LastError = Y
action5; // ok, _LastError does not change, that is, it is still equal to Y
action6; // ok, _LastError does not change, that is, it is still equal to Y
This behavior would make it difficult to localize the problem area. To avoid this, there is a separate
ResetLastError function that resets the _ LastError variable to 0.
void ResetLastError()
The function sets the value of the built-in _ LastError variable to zero.
It is recommended to call the function before any action that can lead to an error and after which you
are going to analyze errors using GetLastError.
A good example of using both functions is the already mentioned PRTF macro (PRTF.mqh file). Its code
is shown below:
#include <MQL5Book/MqlError.mqh>
   
#define PRTF(A) ResultPrint(#A, (A))
   
template<typename T>
T ResultPrint(const string s, const T retval = NULL)
{
   const int snapshot = _LastError; // recording _LastError at input
   const string err = E2S(snapshot) + "(" + (string)snapshot + ")";
   Print(s, "=", retval, " / ", (snapshot == 0 ? "ok" : err));
   ResetLastError(); // clear the error flag for the next calls
   return retval;
}
The purpose of the macro and of the ResultPrint function wrapped into it is to log the passed value,
which is the current error code, and to immediately clear the error code. Thus, successive application
of PRTF on a number of statements always ensures that the error (or success indication) printed to the
log corresponds to the last statement with which the value of the retval parameter was obtained.
We need to save _ LastError in the intermediate local variable snapshot because _ LastError can change
its value almost anywhere in the evaluation of an expression if any operation fails. In this particular
example, the E2S macro uses the EnumToString function which may raise its own error code if a value
that is not in the enumeration is passed as an argument. Then, in the subsequent parts of the same
expression, when forming a string, we will see not the initial error but the raised one.
There may be several places in any statement where _ LastError suddenly changes. In this regard, it is
desirable to record the error code immediately after the desired action.

---

## Page 566

Part 4. Common APIs
566
4.9 MQL program execution environment
4.9.1 6 User-defined errors
The developer can use the built-in _ LastError variable for their own applied purposes. This is facilitated
by the SetUserError function.
void SetUserError(ushort user_error)
The function sets the built-in _ LastError variable to the ERR_ USER_ ERROR_ FIRST + user_ error value,
where ERR_USER_ERROR_FIRST is 65536. All codes below this value are reserved for system errors.
Using this mechanism, you can partially bypass the MQL5 limitation associated with the fact that
exceptions are not supported in the language.
Quite often, functions use the return value as a sign of an error. However, there are algorithms where
the function must return a value of the application type. Let's talk about double. If the function has a
definition range from minus to plus infinity, any value we choose to indicate an error (for example, 0)
will be indistinguishable from the actual result of the calculation. In the case of double, of course, there
is an option to return a specially constructed NaN value (Not a Number, see section Checking real
numbers for normality). But what if the function returns a structure or a class object? One of the
possible solutions is to return the result via a parameter by reference or pointer, but such a form
makes it impossible to use functions as operands of expressions.  
In the context of classes, let's consider the special functions called 'constructors'. They return a new
instance of the object. However, sometimes circumstances prevent you from constructing the whole
object, and then the calling code seems to get the object but should not use it. It's good if the class
can provide an additional method that would allow you to check the usefulness of the object. But as a
uniform alternative approach (for example, covering all classes), we can use SetUserError.
In the Operator overloading section, we encountered the Matrix class. We will supplement it with
methods for calculating the determinant and inverse matrix, and then use it to demonstrate user errors
(see file Matrix.mqh). Overloaded operators were defined for matrices, allowing them to be combined
into chains of operators in a single expression, and therefore it would be inconvenient to implement a
check for potential errors in it.
Our Matrix class is a custom alternative implementation for the recently added MQL5 built-in object
type matrix.
We start by validating input parameters in the Matrix main class constructors. If someone tries to
create a zero-size matrix, let's set a custom error ERR_USER_MATRIX_EMPTY (one of several
provided).

---

## Page 567

Part 4. Common APIs
567
4.9 MQL program execution environment
enum ENUM_ERR_USER_MATRIX
{
   ERR_USER_MATRIX_OK = 0, 
   ERR_USER_MATRIX_EMPTY =  1, 
   ERR_USER_MATRIX_SINGULAR = 2, 
   ERR_USER_MATRIX_NOT_SQUARE = 3
};
   
class Matrix
{
   ...
public:
   Matrix(const int r, const int c) : rows(r), columns(c)
   {
      if(rows <= 0 || columns <= 0)
      {
         SetUserError(ERR_USER_MATRIX_EMPTY);
      }
      else
      {
         ArrayResize(m, rows * columns);
         ArrayInitialize(m, 0);
      }
   }
These new operations are only defined for square matrices, so let's create a derived class with an
appropriate size constraint.
class MatrixSquare : public Matrix
{
public:
   MatrixSquare(const int n, const int _ = -1) : Matrix(n, n)
   {
      if(_ != -1 && _ != n)
      {
         SetUserError(ERR_USER_MATRIX_NOT_SQUARE);
      }
   }
   ...
The second parameter in the constructor should be absent (it is assumed to be equal to the first one),
but we need it because the Matrix class has a template transposition method, in which all types of T
must support a constructor with two integer parameters.

---

## Page 568

Part 4. Common APIs
568
4.9 MQL program execution environment
class Matrix
{
   ...
   template<typename T>
   T transpose() const
   {
      T result(columns, rows);
      for(int i = 0; i < rows; ++i)
      {
         for(int j = 0; j < columns; ++j)
         {
            result[j][i] = this[i][(uint)j];
         }
      }
      return result;
   }
Due to the fact that there are two parameters in the MatrixSquare constructor, we also have to check
them for mandatory equality. If they are not equal, we set the ERR_USER_MATRIX_NOT_SQUARE
error.
Finally, during the calculation of the inverse matrix, we can find that the matrix is degenerate (the
determinant is 0). The error ERR_USER_MATRIX_SINGULAR is reserved for this case.
class MatrixSquare : public Matrix
{
public:
   ...
   MatrixSquare inverse() const
   {
      MatrixSquare result(rows);
      const double d = determinant();
      if(fabs(d) > DBL_EPSILON)
      {
         result = complement().transpose<MatrixSquare>() * (1 / d);
      }
      else
      {
         SetUserError(ERR_USER_MATRIX_SINGULAR);
      }
      return result;
   }
   
   MatrixSquare operator!() const
   {
      return inverse();
   }
   ...
For visual error output, a static method has been added to the log, returning the
ENUM_ERR_USER_MATRIX enumeration, which is easy to pass to EnumToString:

---

## Page 569

Part 4. Common APIs
569
4.9 MQL program execution environment
   static ENUM_ERR_USER_MATRIX lastError()
   {
      if(_LastError >= ERR_USER_ERROR_FIRST)
      {
         return (ENUM_ERR_USER_MATRIX)(_LastError - ERR_USER_ERROR_FIRST);
      }
      return (ENUM_ERR_USER_MATRIX)_LastError;
   }
The full code of all methods can be found in the attached file.
We will check application error codes in the test script EnvError.mq5.
First, let's make sure that the class works: invert the matrix and check that the product of the original
matrix and the inverted one is equal to the identity matrix.
void OnStart()
{
   Print("Test matrix inversion (should pass)");
   double a[9] =
   {
      1,  2,  3, 
      4,  5,  6, 
      7,  8,  0, 
   };
      
   ResetLastError();
   Matrix SquaremA(a);   // assign data to the original matrix
   Print("Input");
   mA.print();
   MatrixSquare mAinv(3);
   mainv = !mA;          // invert and store in another matrix
   Print("Result");
   mAinv.print();
   
   Print("Check inverted by multiplication");
   Matrix Squaretest(3); // multiply the first by the second
   test = mA * mAinv;
   test.print();         // get identity matrix
   Print(EnumToString(Matrix::lastError())); // ok
   ...
This code snippet generates the following log entries.

---

## Page 570

Part 4. Common APIs
570
4.9 MQL program execution environment
Test matrix inversion (should pass)
Input
1.00000 2.00000 3.00000
4.00000 5.00000 6.00000
7.00000 8.00000 0.00000
Result
-1.77778  0.88889 -0.11111
 1.55556 -0.77778  0.22222
-0.11111  0.22222 -0.11111
Check inverted by multiplication
 1.00000 +0.00000  0.00000
 -0.00000   1.00000  +0.00000
0.00000 0.00000 1.00000
ERR_USER_MATRIX_OK
Note that in the identity matrix, due to floating point errors, some zero elements are actually very
small values close to zero, and therefore they have signs.
Then, let's see how the algorithm handles the degenerate matrix.
   Print("Test matrix inversion (should fail)");
   double b[9] =
   {
     -22, -7, 17, 
     -21, 15,  9, 
     -34,-31, 33
   };
   
   MatrixSquare mB(b);
   Print("Input");
   mB.print();
   ResetLastError();
   Print("Result");
   (!mB).print();
   Print(EnumToString(Matrix::lastError())); // singular
   ...
The results are presented below.
Test matrix inversion (should fail)
Input
-22.00000  -7.00000  17.00000
-21.00000  15.00000   9.00000
-34.00000 -31.00000  33.00000
Result
0.0 0.0 0.0
0.0 0.0 0.0
0.0 0.0 0.0
ERR_USER_MATRIX_SINGULAR
In this case, we simply display an error description. But in a real program, it should be possible to
choose a continuation option, depending on the nature of the problem.
Finally, we will simulate situations for the two remaining applied errors.

---

## Page 571

Part 4. Common APIs
571 
4.9 MQL program execution environment
   Print("Empty matrix creation");
   MatrixSquare m0(0);
   Print(EnumToString(Matrix::lastError()));
   
   Print("'Rectangular' square matrix creation");
   MatrixSquare r12(1, 2);
   Print(EnumToString(Matrix::lastError()));
}
Here we describe an empty matrix and a supposedly square matrix but with different sizes.
Empty matrix creation
ERR_USER_MATRIX_EMPTY
'Rectangular' square matrix creation
ERR_USER_MATRIX_NOT_SQUARE
In these cases, we cannot avoid creating an object because the compiler does this automatically.
Of course, this test clearly violates contracts (the specifications of data and actions, that classes and
methods "consider" as valid). However, in practice, arguments are often obtained from other parts of
the code, in the course of processing large, "third-party" data, and detecting deviations from
expectations is not that easy.
The ability of a program to "digest" incorrect data without fatal consequences is the most important
indicator of its quality, along with producing correct results for correct input data.
4.9.1 7 Debug management
The built-in debugger in MetaEditor allows setting breakpoints in the source code, which are the lines
on which the program execution should be suspended. Sometimes this system fails, i.e., the pause does
not work, and then you can use the DebugBreak function explicitly enforces the stop.
void DebugBreak()
Calling the function pauses the program and activates the editor window in the debug mode, with all the
tools for viewing variables and the call stack and for continuing further execution step by step.
Program execution is interrupted only if the program is launched from the editor in the debug mode (by
commands Debug -> Start on Real Data or Start in History Data). In all other modes, including regular
launch (in the terminal) and profiling, the function has no effect.
4.9.1 8 Predefined variables
Each MQL program has a certain general set of global variables provided by the terminal: we have
already covered most of them in the previous sections, and below is a summary table. Almost all
variables are read-only. The exception is the variable _ LastError, which can be reset by the
ResetLastError function.

---

## Page 572

Part 4. Common APIs
572
4.9 MQL program execution environment
Variable
Value
_LastError
Last error value, an analog of the GetLastError function
_StopFlag
Program stop flag, an analog of the IsStopped function
_UninitReason
Program deinitialization reason code, an analog of the UninitializeReason function
_RandomSeed
Current internal state of the pseudo-random integer generator
_IsX64
Flag of a 64-bit terminal, analog of TerminalInfoInteger for the TERMINAL_X64
property
In addition, for MQL programs running in the chart context of a chart, such as Expert Advisors, scripts,
and indicators, the language provides predefined variables with chart properties (they also cannot be
changed from the program).
Variable
Value
_Symbol
Name of the current chart symbol, an analog of the Symbol function
_Period
Current chart timeframe, an analog of the Period function
_Digits
The number of decimal places in the price of the current chart symbol, an analog of
the Digits function
_Point
Point size in the prices of the current symbol (in the quote currency), an analog of
the Point function
_AppliedTo
Type of data on which the indicator is calculated (only for indicators)
4.9.1 9 Predefined constants of the MQL5 language
This section describes all the constants defined by the runtime environment for any program. We have
already seen some of them in previous sections. Some constants relate to applied MQL5 programming
aspects, which will be presented in later chapters.

---

## Page 573

Part 4. Common APIs
573
4.9 MQL program execution environment
Constant
Description
Value
CHARTS_MAX
The maximum possible number of simultaneously
open charts
1 00
clrNONE
No color
-1 
(0xFFFFFFFF)
EMPTY_VALUE
Empty value in the indicator buffer
DBL_MAX
INVALID_HANDLE
Invalid handle
-1 
NULL
Any type null
0
WHOLE_ARRAY
The number of elements until the end of the array,
i.e., the entire array will be processed
-1 
WRONG_VALUE
A constant can be implicitly cast to any
enumeration type
-1 
As shown in the Files chapter, the INVALID_HANDLE constant can be used to validate file descriptors.
The WHOLE_ARRAY constant is intended for functions working with arrays that require specifying the
number of elements in the processed arrays: If it is necessary to process all the array values from the
specified position to the end, specify the WHOLE_ARRAY value.
The EMPTY_VALUE constant is usually assigned to those elements in indicator buffers, which should not
be drawn on the chart. In other words, this constant means a default empty value. Later, we will
describe how it can be replaced for a specific indicator buffer with another value, for example, 0.
The WRONG_VALUE constant is intended for those cases when it is required to designate an incorrect
enumeration value.
In addition, two constants have different values depending on the compilation method.
Constant
Description
IS_DEBUG_MODE
An attribute of running an mq5 program in the debug mode: It is non-
zero in the debug mode and 0 otherwise
IS_PROFILE_MODE
An attribute of running an mq5 program in the profiling mode: It is non-
zero in the profiling mode and 0 otherwise
The IS_PROFILE_MODE constant allows you to change the operation of the program for the correct
collection of information in the profiling mode. Profiling allows you to measure the execution time of
individual program fragments (functions and individual lines).
The compiler sets the IS_PROFILE_MODE constant value during compilation. Normally, it is set to 0.
When the program is launched in a profiling mode, a special compilation is performed, and in this case,
a non-zero value is used instead of IS_PROFILE_MODE.
The IS_DEBUG_MODE constant works in a similar way: it is equal to 0 as a result of native compilation
and is greater than 0 after debug compilation. It is useful in cases where it is necessary to slightly

---

## Page 574

Part 4. Common APIs
574
4.9 MQL program execution environment
change the operation of the MQL program for verification purposes: for example, to output additional
information to the log or to create auxiliary graphical objects on the chart.
The preprocessor defines _DEBUG and _RELEASE constants that are similar in meaning (see Predefined
preprocessor constants).
More detailed information about the program operation mode can be found at runtime using the
MQLInfoInteger function (see Terminal and program operating modes). In particular, the debug build of
a program can be run without a debugger.
4.1 0 Matrices and vectors
The MQL5 language provides special object data types: matrices and vectors. They can be used to
solve a large class of mathematical problems. These types provide methods for writing concise and
understandable code close to the mathematical notation of linear or differential equations.
All programming languages support the concept of an array, which is a collection of multiple elements.
Most algorithms, especially in algorithmic trading, are constructed on the bases of numeric type arrays
(int, double) or structures. Array elements can be accessed by index, which enables the
implementation of operations inside loops. As we know, arrays can have one, two, or more dimensions.
Relatively simple data storing and processing tasks can usually be implemented by using arrays. But
when it comes to complex mathematical problems, the large number of nested loops makes working
with arrays difficult in terms of both programming and reading code. Even the simplest linear algebra
operations require a lot of code and a good understanding of mathematics. This task can be simplified
by the functional paradigm of programming, embodied in the form of matrix and vector method
functions. These actions perform a lot of routine actions "behind the scenes".
Modern technologies such as machine learning, neural networks, and 3D graphics make extensive use
of linear algebra problem solving, which uses operations with vectors and matrices. The new data types
have been added to MQL5 for quick and convenient work with such objects.
At the time of writing the book, the set of functions for working with matrices and vectors was
actively developed, so many interesting new items may not be mentioned here. Follow the release
notes and articles section on the mql5.com site.
In this chapter, we will consider a brief description. For further details about matrices and vectors,
please see the corresponding help section Matrix and vector methods.
It is also assumed that the reader is familiar with the Linear Algebra theory. If necessary, you can
always turn to reference literature and manuals on the web.
4.1 0.1  Types of matrices and vectors
A vector is a one-dimensional array of the real or complex type, while a matrix is a two-dimensional
array of the real or complex type. Thus, the list of valid numeric types for the elements of these objects
includes double (considered the default type), float, and complex.
From the point of view of linear algebra (but not the compiler!) a prime number is also a minimal vector,
and a vector, in turn, can be considered as a special case of a matrix.
The vector, depending on the type of elements, is described using one of the vector (with or without
suffix) keywords:

---

## Page 575

Part 4. Common APIs
575
4.1 0 Matrices and vectors
• vector is a vector with elements of type double
• vectorf is a vector with elements of type float
• vectorc is a vector with elements of type complex
Although vectors can be vertical and horizontal, MQL5 does not make such a division. The required
orientation of the vector is determined (implied) by the position of the vector in the expression.
The following operations are defined on vectors: addition and multiplication, as well as the Norm (with
the relevant norm method) which gets the vector length or module.
You can think of a matrix as an array, where the first index is the row number and the second index is
the column number. However, the numbering of rows and columns, unlike linear algebra, starts from
zero, as in arrays.
Two dimensions of matrices are also called axes and are numbered as follows: 0 for the horizontal axis
(along rows) and 1  for the vertical axis (along columns). Axis numbers are used in many matrix
functions. In particular, when we talk about splitting a matrix into parts, horizontal splitting means
cutting between rows, and vertical splitting means cutting between columns.
Depending on the type of elements, the matrix is described using one of the matrix (with or without
suffix) keywords:
• matrix is a matrix with elements of type double
• matrixf is a matrix with elements of type float
• matrixc is a matrix with elements of type complex
For application in template functions, you can use the notation matrix<double> , matrix<float> ,
matrix<complex> , vector<double> , vector<float> , vector<complex> instead of the corresponding
types.
vectorf v_f1 = {0, 1, 2, 3,};
vector<float> v_f2 = v_f1;
matrix m = {{0, 1}, {2, 3}};
void OnStart()
{
   Print(v_f2);
   Print(m);
}
When logged, matrices and vectors are printed as sequences of numbers separated by commas and
enclosed in square brackets.

---

## Page 576

Part 4. Common APIs
576
4.1 0 Matrices and vectors
[0,1,2,3]
[[0,1]
 [2,3]]
The following algebraic operations are defined for matrices:
• Addition of same-size matrices
• Multiplication of matrices of a suitable size, when the number of columns in the first matrix must be
equal to the number of rows in the second matrix
• Multiplication of a matrix by a column vector and multiplication of a row vector by a matrix
according to the matrix multiplication rules (a vector is, in this sense, a special case of a matrix)
• Multiplying a matrix by a number
In addition, matrix and vector types have built-in methods that correspond to analogs of the NumPy
library (a popular package for machine learning in Python), so you can get more hints in the
documentation and library examples. A complete list of methods can be found in the corresponding
section of MQL5 help.
Unfortunately, MQL5 does not provide for casting matrices and vectors of one type to another (for
example, from double to float). Also, a vector is not automatically treated by the compiler as a matrix
(with one column or row) in expressions where a matrix is expected. This means that the concept of
inheritance (characteristic of OOP) between matrices and vectors does not exist, despite the apparent
relationship between these structures.
4.1 0.2 Creating and initializing matrices and vectors
There are several ways to declare and initialize matrices and vectors. They can be divided into several
categories according to their purpose.
·Declaration without specifying the size
·Declaration with the size specified
·Declaration with initialization
·Static creation methods
·Non-static (re-)configuration and initialization methods
The simplest creation method is a declaration without specifying a size, i.e., without allocating memory
for the data. To do this, just specify the type and name of the variable:
matrix         matrix_a;   // matrix of type double
matrix<double> matrix_a1;  // double type matrix inside function or class templates
matrix<float>  matrix_a2;  // float matrix
vector         vector_v;   // vector of type double
vector<double> vector_v1;  // another notation of a double-type vector creation
vector<float>  vector_v2;  // vector of type float
Then you can change the size of the created objects and fill them with the desired values. They can
also be used in built-in matrix and vector methods to get the results of calculations. All of these
methods will be discussed by groups in sections within this chapter.
You can declare a matrix or vector with a size specified. This will allocate memory but without any
initialization. To do this, after the variable name in parentheses, specify the size(s) (for a matrix, the
first one is the number of rows and the second one is the number of columns):

---

## Page 577

Part 4. Common APIs
577
4.1 0 Matrices and vectors
matrix         matrix_a(128, 128);      // you can specify as parameters
matrix<double> matrix_a1(nRows, nCols); // both constants and variables
matrix<float>  matrix_a2(nRows, 1);     // analog of column vector
vector         vector_v(256);
vector<double> vector_v1(nSize);
vector<float>  vector_v2(nSize +16);    // expression as a parameter
The third way to create objects is by declaration with initialization. The sizes of matrices and vectors in
this case are determined by the initialization sequence indicated in curly brackets:
matrix         matrix_a = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
matrix<double> matrix_a1 =matrix_a;     // must be matrices of the same type
matrix<float>  matrix_a2 = {{1, 2}, {3, 4}};
vector         vector_v = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
vector<double> vector_v1 = {1, 5, 2.4, 3.3};
vector<float>  vector_v2 =vector_v1;    // must be vectors of the same type
There are also static methods for creating matrices and vectors of a specified size with initialization in
a certain way (specifically for one or another canonical form). All of them are listed below and have
similar prototypes (vectors differ from matrices only in the absence of a second dimension).
static matrix<T> matrix<T>::Eye∫Tri(const ulong rows, const ulong cols, const int diagonal = 0);
static matrix<T> matrix<T>::Identity∫Ones∫Zeros(const ulong rows, const ulong cols);
static matrix<T> matrix<T>::Full(const ulong rows, const ulong cols, const double value);
·Eye constructs a matrix with ones on the specified diagonal and zeros elsewhere
·Tri constructs a matrix with ones on and below the specified diagonal and zeros elsewhere
·Identity constructs an identity matrix of the specified size
·Ones constructs a matrix (or vector) filled with ones
·Zeros constructs a matrix (or vector) filled with zeros
·Full constructs a matrix (or vector) filled with the given value in all elements
If necessary, you can turn any existing matrix into an identity matrix, for which you should apply a non-
static method Identity (no parameters).
Let's demonstrate the methods in action:
matrix         matrix_a = matrix::Eye(4, 5, 1);
matrix<double> matrix_a1 = matrix::Full(3, 4, M_PI);
matrixf        matrix_a2 = matrixf::Identity(5, 5);
matrixf<float> matrix_a3 = matrixf::Ones(5, 5);
matrix         matrix_a4 = matrix::Tri(4, 5, -1);
vector         vector_v = vector::Ones(256);
vectorf        vector_v1 = vector<float>::Zeros(16);
vector<float>  vector_v2 = vectorf::Full(128, float_value);
Additionally, there are non-static methods to initialize a matrix/vector with given values: Init and Fill.
void matrix<T>::Init(const ulong rows, const ulong cols, func_reference rule = NULL, ...)
void matrix<T>::Fill(const T value)
An important advantage of the Init method (which is present for constructors as well) is the ability to
specify in the parameters an initializing function for filling the elements of a matrix/vector according to
a given law (see example below).

---

## Page 578

Part 4. Common APIs
578
4.1 0 Matrices and vectors
A reference to such a function can be passed after the sizes by specifying its identifier without quotes
in the rules parameter (this is not a pointer in the sense of typedef (*pointer)(...) and not a string with a
name).
The initializing function must have a reference to the object being filled as the first parameter and may
also have additional parameters: in this case, the values for them are passed to Init or a constructor
after the function reference. If the rule link is not specified, it will simply create a matrix of specified
dimensions.
The Init method also allows changing the matrix configuration.
Let's view everything stated above using small examples.
matrix m(2, 2);
m.Fill(10);
Print("matrix m \n", m);
/*
  matrix m
  [[10,10]
  [10,10]]
*/
m.Init(4, 6);
Print("matrix m \n", m);
/*
  matrix m
  [[10,10,10,10,0.0078125,32.00000762939453]
  [0,0,0,0,0,0]
  [0,0,0,0,0,0]
  [0,0,0,0,0,0]]
*/
Here the Init method was used to resize an already initialized matrix, which resulted in the new
elements being filled with random values.
The following function fills the matrix with numbers that increase exponentially:
template<typename T>
void MatrixSetValues(matrix<T> &m, const T initial = 1)
{
   T value = initial;
   for(ulong r = 0; r < m.Rows(); r++)
   {
      for(ulong c = 0; c < m.Cols(); c++)
      {
         m[r][c] = value;
         value *= 2;
      }
   }
}
Then it can be used to create a matrix.

---

## Page 579

Part 4. Common APIs
579
4.1 0 Matrices and vectors
void OnStart()
{
   matrix M(3, 6, MatrixSetValues);
   Print("M = \n", M);
}
The execution result is:
M = 
[[1,2,4,8,16,32]
 [64,128,256,512,1024,2048]
 [4096,8192,16384,32768,65536,131072]]
In this case, the values for the parameter of the initializing function were not specified following its
identifier in the constructor call, and therefore the default value (1 ) was used. But we can, for example,
pass a start value of -1  for the same MatrixSetValues, which will fill the matrix with a negative row.
   matrix M(3, 6, MatrixSetValues, -1);
4.1 0.3 Copying matrices, vectors, and arrays
The simplest and most common way to copy matrices and vectors is through the assignment operator
'='.
matrix a = {{2, 2}, {3, 3}, {4, 4}};
matrix b = a + 2;
matrix c;
Print("matrix a \n", a);
Print("matrix b \n", b);
c.Assign(b);
Print("matrix c \n", c);
This snippet generates the following log entries:
matrix a
[[2,2]
 [3,3]
 [4,4]]
matrix b
[[4,4]
 [5,5]
 [6,6]]
matrix c
[[4,4]
 [5,5]
 [6,6]]
The Copy and Assign methods can also be used to copy matrices and vectors. The difference between
Assign and Copy is that Assign allows you to copy not only matrices but also arrays.

---

## Page 580

Part 4. Common APIs
580
4.1 0 Matrices and vectors
bool matrix<T>::Copy(const matrix<T> &source)
bool matrix<T>::Assign(const matrix<T> &source)
bool matrix<T>::Assign(const T &array[])
Similar methods and prototypes are also available for vectors.
Through Assign, it is possible to write a vector to a matrix: the result will be a one-row matrix.
bool matrix<T>::Assign(const vector<T> &v)
You can also assign a matrix to a vector: it will be unwrapped, i.e., all rows of the matrix will be lined up
in one row (equivalent to calling the Flat method).
bool vector<T>::Assign(const matrix<T> &m)
At the time of writing this chapter, there was no method in MQL5 for exporting a matrix or vector to an
array, although there is a mechanism for "transferring" data (see the Swap method further).
The example below shows how an integer array int_ arr is copied into a matrix of type double. In this
case, the resulting matrix automatically adjusts to the size of the copied array.
matrix double_matrix = matrix::Full(2, 10, 3.14);
Print("double_matrix before Assign() \n", double_matrix);
int int_arr[5][5] = {{1, 2}, {3, 4}, {5, 6}};
Print("int_arr: ");
ArrayPrint(int_arr);
double_matrix.Assign(int_arr);
Print("double_matrix after Assign(int_arr) \n", double_matrix);
We have the following output in the log.
double_matrix before Assign() 
[[3.14,3.14,3.14,3.14,3.14,3.14,3.14,3.14,3.14,3.14]
 [3.14,3.14,3.14,3.14,3.14,3.14,3.14,3.14,3.14,3.14]]
 
int_arr: 
    [,0][,1][,2][,3][,4]
[0,]   1   2   0   0   0
[1,]   3   4   0   0   0
[2,]   5   6   0   0   0
[3,]   0   0   0   0   0
[4,]   0   0   0   0   0
double_matrix after Assign(int_arr) 
[[1,2,0,0,0]
 [3,4,0,0,0]
 [5,6,0,0,0]
 [0,0,0,0,0]
 [0,0,0,0,0]]
So, the method Assign can be used to switch from arrays to matrices with automatic size and type
conversion.
A more efficient (fast and not involving copying) way to transfer data between matrices, vectors, and
arrays is to use Swap methods.

---

## Page 581

Part 4. Common APIs
581 
4.1 0 Matrices and vectors
bool matrix<T>::Swap(vector<T> &vec)
bool matrix<T>::Swap(matrix<T> &vec)
bool matrix<T>::Swap(T &arr[]) 
bool vector<T>::Swap(vector<T> &vec)
bool vector<T>::Swap(matrix<T> &vec)
bool vector<T>::Swap(T &arr[])
They work similarly to ArraySwap: Internal pointers to buffers with data inside two objects are
swapped. As a result, elements of a matrix or vector disappear in the source object and appear in the
receiving array, or, vice versa, they move from the array to the matrix or vector.
The Swap method allows working with dynamic arrays, including multidimensional ones. A certain
condition applies to the constant sizes of the highest dimensions of a multidimensional array (array[]
[N1 ][N2]...): The product of these dimensions must be a multiple of the size of the matrix or vector. So,
an array of [][2][3] is redistributed in blocks of 6 elements. Therefore, it is interchangeable with
matrices and vectors of size 6, 1 2, 1 8, etc.
4.1 0.4 Copying timeseries to matrices and vectors
The matrix<T>::CopyRates method copies timeseries with the quoting history directly into a matrix or
vector. This method works similarly to the functions which we will cover in detail in Part 5, in the
chapter on timeseries, namely: CopyRates and separate Copy functions for each field of the MqlRates
structure.
bool matrix<T>::CopyRates(const string symbol, ENUM_TIMEFRAMES tf, ulong rates_mask,
   ulong start, ulong count)
bool matrix<T>::CopyRates(const string symbol, ENUM_TIMEFRAMES tf, ulong rates_mask,
   datetime from, ulong count)
bool matrix<T>::CopyRates(const string symbol, ENUM_TIMEFRAMES tf, ulong rates_mask,
   datetime from, datetime to)
In the parameters, you need to specify the symbol, timeframe, and the range of requested bars: either
by number and quantity, or by date range. The data is copied so that the oldest element is placed at
the beginning of the matrix/vector.
The rates_ mask parameter specifies a combination of flags from the ENUM_COPY_RATES enumeration
with a set of available fields. The combination of flags allows you to get several timeseries from history
in one request. In this case, the order of rows in the matrix will correspond to the order of values in the
ENUM_COPY_RATES enumeration, in particular, the row with High data in the matrix will always be
above the row with Low data.
When copying to a vector, only one value from the ENUM_COPY_RATES enumeration can be specified.
Otherwise, an error will occur.

---

## Page 582

Part 4. Common APIs
582
4.1 0 Matrices and vectors
Identifier
Value
Description
COPY_RATES_OPEN
1
Open prices
COPY_RATES_HIGH
2
High prices
COPY_RATES_LOW
4
Low prices
COPY_RATES_CLOSE
8
Close prices
COPY_RATES_TIME
1 6
Bar opening times
COPY_RATES_VOLUME_TICK
32
Tick volumes
COPY_RATES_VOLUME_REAL
64
Real volumes
COPY_RATES_SPREAD
1 28
Spreads
Combinations
COPY_RATES_OHLC
1 5
Open, High, Low, Close
COPY_RATES_OHLCT
31 
Open, High, Low, Close, Time
We will view an example of using this function in the Solving equations section.
4.1 0.5 Copying tick history to matrices or vectors
As in the case with bars, you can copy ticks to a vector or matrix. This is done by CopyTicks and
CopyTicksRange method overloads. They work on a basis similar to the CopyTicks and CopyTicksRange
functions, but they receive data into the caller. These functions will be described in detail in Part 5, in
the section about arrays of real ticks in MqlTick structures. Here we will only show the prototypes and
mention the main points.
bool matrix<T>::CopyTicks(const string symbol, uint flags, ulong from_msc, uint count)
bool vector<T>::CopyTicks(const string symbol, uint flags, ulong from_msc, uint count)
bool matrix<T>::CopyTicksRange(const string symbol, uint flags, ulong from_msc, ulong to_msc)
bool matrix<T>::CopyTicksRange(const string symbol, uint flags, ulong from_msc, ulong to_msc)
The symbol parameter sets the name of the financial instrument for which the ticks are requested. The
tick range can be specified in different ways:
·In CopyTicks, it can be specified as a number of ticks (the count parameter), starting from some
moment (from_ msc), in milliseconds
·In CopyTicksRange, it can be a range of two points in time (from from_ msc to to_ msc).
The composition of the copied data about each tick is specified in the flags parameter as a bitmask of
values from the ENUM_COPY_TICKS enumeration.

---

## Page 583

Part 4. Common APIs
583
4.1 0 Matrices and vectors
Identifier
Value
Description
COPY_TICKS_INFO
1
Ticks generated by Bid and/or Ask
changes
COPY_TICKS_TRADE
2
Ticks generated by Last and Volume
changes
COPY_TICKS_ALL
3
All ticks
COPY_TICKS_TIME_MS
1  << 8
Time in milliseconds
COPY_TICKS_BID
1  << 9
Bid price
COPY_TICKS_ASK
1  << 1 0
Ask price
COPY_TICKS_LAST
1  << 1 1 
Last price
COPY_TICKS_VOLUME
1  << 1 2
Volume
COPY_TICKS_FLAGS
1  << 1 3
Tick flags
The first three bits (low byte) determine the set of requested ticks, and the remaining bits (high byte)
determine the properties of these ticks.
High-byte flags can only be combined for matrices since only one row with the values of a particular
field from all ticks is placed in the vector. Thus, only one bit of the most significant byte should be
selected to fill the vector.
When selecting several properties of ticks in the process of filling the matrix, the order of rows in it will
correspond to the order of elements in the enumeration. For example, the Bid price will always appear
in the row higher (with a lower index) than the row with Ask prices.
An example of working with both, ticks and vectors, will be presented in the section on machine
learning.
4.1 0.6 Evaluation of expressions with matrices and vectors
You can perform mathematical operations element by element (use operators) over matrices and
vectors, such as addition, subtraction, multiplication, and division. For these operations, both objects
must be of the same type and have the same dimensions. Each member of the matrix/vector interacts
with the corresponding element of the second matrix/vector.
As the second term (multiplier, subtrahend, or divisor), you can also use a scalar of the corresponding
type (double, float, or complex). In this case, each element of the matrix or vector will be processed
taking into account that scalar.

---

## Page 584

Part 4. Common APIs
584
4.1 0 Matrices and vectors
matrix matrix_a = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
matrix matrix_b = {{1, 2, 3}, {4, 5, 6}};
matrix matrix_c1 = matrix_a + matrix_b;
matrix matrix_c2 = matrix_b - matrix_a;
matrix matrix_c3 = matrix_a * matrix_b;   // Hadamard product (element-by-element)
matrix matrix_c4 = matrix_b / matrix_a;
matrix_c1 = matrix_a + 1;
matrix_c2 = matrix_b - double_value;
matrix_c3 = matrix_a * M_PI;
matrix_c4 = matrix_b / 0.1;
matrix_a += matrix_b;                     // operations "in place" are possible 
matrix_a /= 2;
In-place operations modify the original matrix (or vector) by placing the result into it, unlike regular
binary operations in which the operands are left unchanged, and a new object is created for the result.
Besides, matrices and vectors can be passed as a parameter to most mathematical functions. In this
case, the matrix or vector is processed element by element. For example:
matrix a = {{1, 4}, {9, 16}};
Print("matrix a=\n", a);
a = MathSqrt(a);
Print("MatrSqrt(a)=\n", a);
/*
   matrix a=
   [[1,4]
    [9,16]]
   MatrSqrt(a)=
   [[1,2]
    [3,4]]
*/
In the case of MathMod and MathPow, the second parameter can be either a scalar, or a matrix, or a
vector of the appropriate size.
4.1 0.7 Manipulating matrices and vectors
When working with matrices and vectors, basic manipulations are available without any calculations.
Exclusively matrix methods are provided at the beginning of the list, while the last four methods are also
applicable to vectors.
·Transpose: matrix transposition
·Col, Row, Diag: extract and set rows, columns, and diagonals by number
·TriL, TriU: get the lower and upper triangular matrix by the number of the diagonal
·SwapCols, SwapRows: rearrange rows and columns indicated by numbers
·Flat: set and get a matrix element by a through index
·Reshape: reshape a matrix "in place"
·Split, Hsplit, Vsplit: split a matrix into several submatrices
·resize: resize a matrix or vector "in place";

---

## Page 585

Part 4. Common APIs
585
4.1 0 Matrices and vectors
·Compare, CompareByDigits: compare two matrices or two vectors with a given precision of real
numbers
·Sort: sort "in place" (permutation of elements) and by getting a vector or matrix of indexes
·clip: limit the range of values of elements "in place"
Note that vector splitting is not provided.
Below are the prototype methods for matrices.
matrix<T> matrix<T>::Transpose()
vector matrix<T>::Col∫Row(const ulong n)
void matrix<T>::Col∫Row(const vector v, const ulong n)
vector matrix<T>::Diag(const int n = 0)
void matrix<T>::Diag(const vector v, const int n = 0)
matrix<T> matrix<T>::TriL∫TriU(const int n = 0)
bool matrix<T>::SwapCols∫SwapRows(const ulong n1 , const ulong n2)
T matrix<T>::Flat(const ulong i)
bool matrix<T>::Flat(const ulong i, const T value)
bool matrix<T>::Resize(const ulong rows, const ulong cols, const ulong reserve = 0)
void matrix<T>::Reshape(const ulong rows, const ulong cols)
ulong matrix<T>::Compare(const matrix<T> &m, const T epsilon)
ulong matrix<T>::CompareByDigits(const matrix &m, const int digits)
bool matrix<T>::Split(const ulong nparts, const int axis, matrix<T> &splitted[])
void matrix<T>::Split(const ulong &parts[], const int axis, matrix<T> &splitted[])
bool matrix<T>::Hsplit∫Vsplit(const ulong nparts, matrix<T> &splitted[])
void matrix<T>::Hsplit∫Vsplit(const ulong &parts[], matrix<T> &splitted[])
void matrix<T>::Sort(func_reference compare = NULL, T context)
void matrix<T>::Sort(const int  axis, func_reference compare = NULL, T context)
matrix<T> matrix<T>::Sort(func_reference compare = NULL, T context)
matrix<T> matrix<T>::Sort(const int axis, func_reference compare = NULL, T context)
bool matrix<T>::Clip(const T min, const T max)
For vectors, there is a smaller set of methods.
bool vector<T>::Resize(const ulong size, const ulong reserve = 0)
ulong vector<T>::Compare(const vector<T> &v, const T epsilon)
ulong vector<T>::CompareByDigits(const vector<T> &v, const int digits)
void vector<T>::Sort(func_reference compare = NULL, T context)
vector vector<T>::Sort(func_reference compare = NULL, T context)
bool vector<T>::Clip(const T min, const T max)
Matrix transposition example:

---

## Page 586

Part 4. Common APIs
586
4.1 0 Matrices and vectors
matrix a = {{0, 1, 2}, {3, 4, 5}};
Print("matrix a \n", a);
Print("a.Transpose() \n", a.Transpose());
/*
   matrix a
   [[0,1,2]
    [3,4,5]]
   a.Transpose()
   [[0,3]
    [1,4]
    [2,5]]
*/
Several examples of setting different diagonals using the Diag method:
vector v1 = {1, 2, 3};
matrix m1;
m1.Diag(v1);
Print("m1\n", m1);
/* 
   m1
   [[1,0,0]
    [0,2,0]
    [0,0,3]]
*/
  
matrix m2;
m2.Diag(v1, -1);
Print("m2\n", m2);
/*
   m2
   [[0,0,0]
    [1,0,0]
    [0,2,0]
    [0,0,3]]
*/
  
matrix m3;
m3.Diag(v1, 1);
Print("m3\n", m3);
/*
   m3
   [[0,1,0,0]
    [0,0,2,0]
    [0,0,0,3]]
*/
Changing the matrix configuration using Reshape:

---

## Page 587

Part 4. Common APIs
587
4.1 0 Matrices and vectors
matrix matrix_a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
Print("matrix_a\n", matrix_a);
/*
   matrix_a
   [[1,2,3]
    [4,5,6]
    [7,8,9]
    [10,11,12]]
*/
  
matrix_a.Reshape(2, 6);
Print("Reshape(2,6)\n", matrix_a);
/*
   Reshape(2,6)
   [[1,2,3,4,5,6]
    [7,8,9,10,11,12]]
*/
  
matrix_a.Reshape(3, 5);
Print("Reshape(3,5)\n", matrix_a);
/*
   Reshape(3,5)
   [[1,2,3,4,5]
    [6,7,8,9,10]
    [11,12,0,3,0]]
*/
  
matrix_a.Reshape(2, 4);
Print("Reshape(2,4)\n", matrix_a);
/*
   Reshape(2,4)
   [[1,2,3,4]
    [5,6,7,8]]
*/
We will apply the splitting of matrices into submatrices in an example when Solving equations.
The Col and Row methods allow not only getting columns or rows of a matrix by their number but also
inserting them "in place" into previously defined matrices. In this case, neither the dimensions of the
matrix nor the values of elements outside the column vector (for the case Col) or a row vector (for the
case Row) will change.
If either of these two methods is applied to a matrix the dimensions of which have not yet been set,
then a null matrix of size [N * M] will be created, where N and M are defined differently for Col and Row,
based on the length of the vector and the given column or row index:
·For Col, N is the length of the column vector and M is by 1  greater than the specified index of the
inserted column
·For Row, N is by 1  greater than the specified index of the inserted row and M is the length of the
row vector

---

## Page 588

Part 4. Common APIs
588
4.1 0 Matrices and vectors
At the time of writing this chapter, MQL5 did not provide methods for full-fledged insertion of rows and
columns with the expansion of subsequent elements, as well as for excluding specified rows and
columns.
4.1 0.8 Products of matrices and vectors
Matrix multiplication is one of the basic operations in various numerical methods. For example, it is
often used when implementing forward and backward propagation methods in neural network layers.
Various kinds of convolutions can also be attributed to the category of matrix products. The group of
such functions in MQL5 looks like this:
• MatMul: the matrix product of two matrices
• Power: raise a square matrix to the specified integer power
• Inner: the inner product of two matrices
• Outer: the outer product of two matrices or two vector
• Kron: the Kronecker product of two matrices, a matrix and a vector, a vector and a matrix, or two
vectors
• CorrCoef: calculate the Pearson correlation between rows or columns of a matrix, or between
vectors
• Cov: calculate the covariance matrix of rows or columns of a matrix, or between two vectors
• Correlate: calculate the mutual correlation (cross-correlation) of two vectors
• Convolve: calculate discrete linear convolution of two vectors
• Dot: the scalar product of two vectors
To give a general idea of how to manage these methods, we will give their prototypes (in the following
order: from matrix, through mixed matrix-vector, to vector).
matrix<T> matrix<T>::MatMul(const matrix<T> &m)
matrix<T> matrix<T>::Power(const int power)
matrix<T> matrix<T>::Inner(const matrix<T> &m)
matrix<T> matrix<T>::Outer(const matrix<T> &m)
matrix<T> matrix<T>::Kron(const matrix<T> &m)
matrix<T> matrix<T>::Kron(const vector<T> &v)
matrix<T> matrix<T>::CorrCoef(const bool rows = true)
matrix<T> matrix<T>::Cov(const bool rows = true)
matrix<T> vector<T>::Cov(const vector<T> &v)
                  T vector<T>::CorrCoef(const vector<T> &v)
vector<T> vector<T>::Correlate(const vector<T> &v, ENUM_VECTOR_CONVOLVE mode)
vector<T> vector<T>::Convolve(const vector<T> &v, ENUM_VECTOR_CONVOLVE mode)
matrix<T> vector<T>::Outer(const vector<T> &v)
matrix<T> vector<T>::Kron(const matrix<T> &m)
matrix<T> vector<T>::Kron(const vector<T> &v)
                  T vector<T>::Dot(const vector<T> &v)
Here is a simple example of the matrix product of two matrices using the MatMul method:

---

## Page 589

Part 4. Common APIs
589
4.1 0 Matrices and vectors
matrix a = {{1, 0, 0},
            {0, 1, 0}};
matrix b = {{4, 1},
            {2, 2},
            {1, 3}};
matrix c1 = a.MatMul(b);
matrix c2 = b.MatMul(a);
Print("c1 = \n", c1);
Print("c2 = \n", c2);
/*
   c1 = 
   [[4,1]
    [2,2]]
   c2 = 
   [[4,1,0]
    [2,2,0]
    [1,3,0]]
*/
Matrices of the form A[M,N] * B[N,K] = C[M,K] can be multiplied, i.e., the number of columns in the
first matrix must be equal to the number of rows in the second matrix. If the dimensions are not
consistent, the result is an empty matrix.
When multiplying a matrix and a vector, two options are allowed:
• The horizontal vector (row) is multiplied by the matrix on the right, the length of the vector is equal
to the number of matrix rows
• The matrix is multiplied by a vertical vector (column) on the right, the length of the vector is equal
to the number of columns of the matrix
Vectors can also be multiplied with each other. In MatMul, this is always equivalent to the dot product
(the Dot method) of a row vector by a column vector, and the option when a column vector is multiplied
by a row vector and a matrix is obtained is supported by another method: Outer.
Let's demonstrate the Outer product of vector v5 by vector v3, and in reverse order. In both cases, a
column vector is implied on the left, and a row vector is implied on the right.

---

## Page 590

Part 4. Common APIs
590
4.1 0 Matrices and vectors
vector v3 = {1, 2, 3};
vector v5 = {1, 2, 3, 4, 5};
Print("v5 = \n", v5);
Print("v3 = \n", v3);
Print("v5.Outer(v3) = m[5,3] \n", v5.Outer(v3));
Print("v3.Outer(v5) = m[3,5] \n", v3.Outer(v5));
/*
   v5 =
   [1,2,3,4,5]
   v3 =
   [1,2,3]
   v5.Outer(v3) = m[5,3]
   [[1,2,3]
    [2,4,6]
    [3,6,9]
    [4,8,12]
    [5,10,15]]
   v3.Outer(v5) = m[3,5]
   [[1,2,3,4,5]
    [2,4,6,8,10]
    [3,6,9,12,15]]
*/
4.1 0.9 Transformations (decomposition) of matrices
Matrix transformations are the most commonly used operations when working with data. However,
many complex transformations cannot be performed analytically and with absolute accuracy.
Matrix transformations (or in other words, decompositions) are methods that decompose a matrix into
its component parts, which makes it easier to calculate more complex matrix operations. Matrix
decomposition methods, also called matrix factorization methods, are the basis of linear algebra
algorithms, such as solving systems of linear equations and calculating the inverse of a matrix or
determinant.
In particular, Singular Values Decomposition (SVD) is widely used in machine learning, which allows you
to represent the original matrix as a product of three other matrices. SVD decomposition is used to
solve a variety of problems, from least squares approximation to compression and image recognition.
List of available methods:
·Cholesky: calculate the Cholesky decomposition
·Eig: calculate eigenvalues and right eigenvectors of a square matrix
·Eig Vals: calculate eigenvalues of the common matrix
·LU: implement LU factorization of a matrix as a product of a lower triangular matrix and an upper
triangular matrix
·LUP: implement LUP factorization with partial rotation, which is an LU factorization with row
permutations only: PA=LU
·QR: implement QR factorization of the matrix
·SVD: singular value decomposition
Below are the method prototypes.

---

## Page 591

Part 4. Common APIs
591 
4.1 0 Matrices and vectors
bool matrix<T>::Cholesky(matrix<T> &L)
bool matrix<T>::Eig(matrix<T> &eigen_vectors, vector<T> &eigen_values)
bool matrix<T>::EigVals(vector<T> &eigen_values)
bool matrix<T>::LU(matrix<T> &L, matrix<T> &U)
bool matrix<T>::LUP(matrix<T> &L, matrix<T> &U, matrix<T> &P)
bool matrix<T>::QR(matrix<T> &Q, matrix<T> &R)
bool matrix<T>::SVD(matrix<T> &U, matrix<T> &V, vector<T> &singular_values)
Let's show an example of a singular value decomposition using the SVD method (see. file
MatrixSVD.mq5). First, we initialize the original matrix.
matrix a = {{0, 1, 2, 3, 4, 5, 6, 7, 8}};
a = a - 4;
a.Reshape(3, 3);
Print("matrix a \n", a);
Now let's make an SVD decomposition:
matrix U, V;
vector singular_values;
a.SVD(U, V, singular_values);
Print("U \n", U);
Print("V \n", V);
Print("singular_values = ", singular_values);
Let's check the expansion: the following equality must hold: U * "singular diagonal" * V = A.
matrix matrix_s;
matrix_s.Diag(singular_values);
Print("matrix_s \n", matrix_s);
matrix matrix_vt = V.Transpose();
Print("matrix_vt \n", matrix_vt);
matrix matrix_usvt = (U.MatMul(matrix_s)).MatMul(matrix_vt);
Print("matrix_usvt \n", matrix_usvt);
Let's compare the resulting and original matrix for errors.
ulong errors = (int)a.Compare(matrix_usvt, 1e-9);
Print("errors=", errors);
The log should look like this:

---

## Page 592

Part 4. Common APIs
592
4.1 0 Matrices and vectors
matrix a
[[-4,-3,-2]
 [-1,0,1]
 [2,3,4]]
U
[[-0.7071067811865474,0.5773502691896254,0.408248290463863]
 [-6.827109697437648e-17,0.5773502691896253,-0.8164965809277256]
 [0.7071067811865472,0.5773502691896255,0.4082482904638627]]
V
[[0.5773502691896258,-0.7071067811865474,-0.408248290463863]
 [0.5773502691896258,1.779939029415334e-16,0.8164965809277258]
 [0.5773502691896256,0.7071067811865474,-0.408248290463863]]
singular_values = [7.348469228349533,2.449489742783175,3.277709923350408e-17]
  
matrix_s
[[7.348469228349533,0,0]
 [0,2.449489742783175,0]
 [0,0,3.277709923350408e-17]]
matrix_vt
[[0.5773502691896258,0.5773502691896258,0.5773502691896256]
 [-0.7071067811865474,1.779939029415334e-16,0.7071067811865474]
 [-0.408248290463863,0.8164965809277258,-0.408248290463863]]
matrix_usvt
[[-3.999999999999997,-2.999999999999999,-2]
 [-0.9999999999999981,-5.977974170712231e-17,0.9999999999999974]
 [2,2.999999999999999,3.999999999999996]]
errors=0
Another practical case of applying the Convolve method is included in the example in Machine learning
methods.
4.1 0.1 0 Obtaining statistics
The methods listed below are designed to obtain descriptive statistics for matrices and vectors. All of
them apply to a vector or a matrix as a whole, as well as to a given matrix axis (horizontally or
vertically). When applied entirely to an object, these functions return a scalar (singular). When applied
to a matrix along any of the axes, a vector is returned.
The general appearance of prototypes:
T vector<T>::Method(const vector<T> &v)
T matrix<T>::Method(const matrix<T> &m)
vector<T> matrix<T>::Method(const matrix<T> &m, const int axis)
The list of methods:
·ArgMax, ArgMin: find indexes of maximum and minimum values
·Max, Min: find the maximum and minimum values
·Ptp: find a range of values
·Sum, Prod: calculate the sum or product of elements
·CumSum, CumProd: calculate the cumulative sum or product of elements

---

## Page 593

Part 4. Common APIs
593
4.1 0 Matrices and vectors
·Median, Mean, Average: calculate the median, arithmetic mean, or weighted arithmetic mean
·Std, Var: calculate standard deviation and variance
·Percentile, Quantile: calculate percentiles and quantiles
·RegressionMetric: calculate one of the predefined regression metrics, such as errors of deviation
from the regression line on the matrix/vector data
An example of calculating the standard deviation and percentiles for the range of bars (in points) of the
current symbol and timeframe is given in the MatrixStdPercentile.mq5 file.
input int BarCount = 1000;
input int BarOffset = 0;
   
void OnStart()
{
   // getting current chart quotes
   matrix rates;
   rates.CopyRates(_Symbol, _Period, COPY_RATES_OPEN | COPY_RATES_CLOSE, 
      BarOffset, BarCount);
   // calculating price increments on bars
   vector delta = MathRound((rates.Row(1) - rates.Row(0)) / _Point);
   // debug print of initial bars
   rates.Resize(rates.Rows(), 10);
   Normalize(rates);
   Print(rates);
   // printing increment metrics
   PRTF((int)delta.Std());
   PRTF((int)delta.Percentile(90));
   PRTF((int)delta.Percentile(10));
}
Log:
(EURUSD,H1)[[1.00832,1.00808,1.00901,1.00887,1.00728,1.00577,1.00485,1.00652,1.00538,1.00409]
(EURUSD,H1) [1.00808,1.00901,1.00887,1.00728,1.00577,1.00485,1.00655,1.00537,1.00412,1.00372]]
(EURUSD,H1)(int)delta.Std()=163 / ok
(EURUSD,H1)(int)delta.Percentile(90)=170 / ok
(EURUSD,H1)(int)delta.Percentile(10)=-161 / ok
4.1 0.1 1  Characteristics of matrices and vectors
The following group of methods can be used to obtain the main characteristics of matrices:
·Rows, Cols: the number of rows and columns in the matrix
·Norm: one of the predefined matrix norms (ENUM_MATRIX_NORM)
·Cond: the condition number of the matrix
·Det: the determinant of a square nondegenerate matrix
·SLogDet: calculates the sign and logarithm of the matrix determinant
·Rank: the rank of the matrix
·Trace: the sum of elements along the diagonals of the matrix (trace)
·Spectrum: the spectrum of a matrix as a set of its eigenvalues

---

## Page 594

Part 4. Common APIs
594
4.1 0 Matrices and vectors
In addition, the following characteristics are defined for vectors:
·Size: the length of the vector
·Norm: one of the predefined vector norms (ENUM_VECTOR_NORM)
The sizes of objects (as well as the indexing of elements in them) use values of the ulong type.
ulong matrix<T>::Rows()
ulong matrix<T>::Cols()
ulong vector<T>::Size()
Most of the other characteristics are real numbers.
double vector<T>::Norm(const ENUM_VECTOR_NORM norm, const int norm_p = 2)
double matrix<T>::Norm(const ENUM_MATRIX_NORM norm)
double matrix<T>::Cond(const ENUM_MATRIX_NORM norm)
double matrix<T>::Det()
double matrix<T>::SLogDet(int &sign)
double matrix<T>::Trace()
The rank and spectrum are, respectively, an integer and a vector.
int matrix<T>::Rank()
vector matrix<T>::Spectrum()
Matrix rank calculation example:
matrix a = matrix::Eye(4, 4);
Print("matrix a (eye)\n", a);
Print("a.Rank()=", a.Rank());
   
a[3, 3] = 0;
Print("matrix a (defective eye)\n", a);
Print("a.Rank()=", a.Rank());
   
matrix b = matrix::Ones(1, 4);
Print("b \n", b);
Print("b.Rank()=", b.Rank());
   
matrix zeros = matrix::Zeros(4, 1);
Print("zeros \n", zeros);
Print("zeros.Rank()=", zeros.Rank());
And here is the result of the script execution:

---

## Page 595

Part 4. Common APIs
595
4.1 0 Matrices and vectors
matrix a (eye)
[[1,0,0,0]
 [0,1,0,0]
 [0,0,1,0]
 [0,0,0,1]]
a.Rank()=4
  
matrix a (defective eye)
[[1,0,0,0]
 [0,1,0,0]
 [0,0,1,0]
 [0,0,0,0]]
a.Rank()=3
  
b
[[1,1,1,1]]
b.Rank()=1
   
zeros
[[0]
 [0]
 [0]
 [0]]
zeros.Rank()=0
4.1 0.1 2 Solving equations
In machine learning methods and optimization problems, it is often required to find a solution to a
system of linear equations. MQL5 contains four methods that allow solving such equations depending on
the matrix type.
• Solve solves a linear matrix equation or a system of linear algebraic equations
• LstSq  solves a system of linear algebraic equations approximately (for non-square or degenerate
matrices)
• Inv calculates a multiplicative inverse matrix relative to a square non-singular matrix using the
Jordan-Gauss method
• PInv calculates the pseudo-inverse matrix by the Moore-Penrose method
Following are the method prototypes.
vector<T> matrix<T>::Solve(const vector<T> b)
vector<T> matrix<T>::LstSq(const vector<T> b)
matrix<T> matrix<T>::Inv()
matrix<T> matrix<T>::PInv()
The Solve and LstSq methods imply the solution of a system of equations of the form A*X=B, where A is
a matrix, B is a vector passed through a parameter with the values of the function (or "dependent
variable").
Let's try to apply the LstSq method to solve a system of equations, which is a model of ideal portfolio
trading (in our case, we will analyze a portfolio of the main Forex currencies). To do this, on a given

---

## Page 596

Part 4. Common APIs
596
4.1 0 Matrices and vectors
number of "historical" bars, we need to find such lot sizes for each currency, with which the balance
line tends to be a constantly growing straight line.
Let's denote the i-th currency pair as Si. Its quote at the bar with the k index is equal to Si[k]. The
numbering of bars will go from the past to the future, as in matrices and vectors populated by the
CopyRates method. Thus, the beginning of the collected quotes for training the model corresponds to
the bar marked with the number 0, but on the timeline, it will be the oldest historical bar (of those that
we process, according to the algorithm settings). The bars on the right (to the future) from it are
numbered 1 , 2, and so on, up to the total number of bars on which the user will order the calculation.
A change in the price of a symbol between the 0th bar and the Nth bar determines the profit (or loss)
by the time of the Nth bar.
Taking into account the set of currencies, we get, for example, the following profit equation for the 1 st
bar:
(S1[1] - S1[0]) * X1 + (S2[1] - S2[0]) * X2 + ... + (Sm[1] - Sm[0]) * Xm = B
Here m is the total number of characters, Xi is the lot size of each symbol, and B is the floating profit
(conditional balance, if you lock in the profit).
For simplicity, let's shorten the notation. Let's move from absolute values to price increments (Ai [k] =
Si [k]-Si [0]). Taking into account the movement through bars, we will obtain several expressions for the
virtual balance curve:
A1[1] * X1 + A2[1] * X2 + ... + Am[1] * Xm = B[1]
A1[2] * X1 + A2[2] * X2 + ... + Am[2] * Xm = B[2]
...
A1[K] * X1 + A2[K] * X2 + ... + Am[K] * Xm = B[K]
Successful trading is characterized by a constant profit on each bar, i.e., the model for the right-
handed vector B is a monotonically increasing function, ideally a straight line.
Let's implement this model and select the X coefficients for it based on quotes. Since we do not yet
know the application APIs, we will not code a full-fledged trading strategy. Let's just build a virtual
balance chart using the GraphPlot function from the standard header file Graphic.mqh (we have already
used it to demonstrate mathematical functions).
The full source code for the new example is in the script MatrixForexBasket.mq5.
In the input parameters, let the user choose the total number of bars for data sampling (BarCount), as
well as the bar number within this selection (BarOffset) on which the conditional past ends and the
conditional future begins.
A model will be built on the conditional past (the above system of linear equations will be solved), and a
forward test will be performed on the conditional future.
input int BarCount = 20;  // BarCount (known "history" and "future")
input int BarOffset = 10; // BarOffset (where "future" starts)
input ENUM_CURVE_TYPE CurveType = CURVE_LINES;
To fill the vector with an ideal balance, we write the ConstantGrow function: it will be used later during
initialization.

---

## Page 597

Part 4. Common APIs
597
4.1 0 Matrices and vectors
void ConstantGrow(vector &v)
{
   for(ulong i = 0; i < v.Size(); ++i)
   {
      v[i] = (double)(i + 1);
   }
}
The list of traded instruments (major Forex pairs) is hard-set at the beginning of the OnStart function
— edit it to suit your requirements and trading environment.
void OnStart()
{
   const string symbols[] =
   {
      "EURUSD", "GBPUSD", "USDJPY", "USDCAD", 
      "USDCHF", "AUDUSD", "NZDUSD"
   };
   const int size = ArraySize(symbols);
   ...
Let's create the rates matrix in which symbol quotes will be added, the model vector with desired
balance curve, and the auxiliary close vector for a symbol-by-symbol request for bar closing prices (the
data from it will be copied into the columns of the rates matrix).
   matrix rates(BarCount, size);
   vector model(BarCount - BarOffset, ConstantGrow);
   vector close;
In a symbol loop, we copy the closing prices into the close vector, calculate price increments, and write
them in the corresponding column of the rates matrix.
   for(int i = 0; i < size; i++)
   {
      if(close.CopyRates(symbols[i], _Period, COPY_RATES_CLOSE, 0, BarCount))
      {
         // calculate increments (profit on all and on each bar in one line)
         close -= close[0];
         // adjust the profit to the pip value
         close *= SymbolInfoDouble(symbols[i], SYMBOL_TRADE_TICK_VALUE) /
            SymbolInfoDouble(symbols[i], SYMBOL_TRADE_TICK_SIZE);
         // place the vector in the matrix column
         rates.Col(close, i);
      }
      else
      {
         Print("vector.CopyRates(%d, COPY_RATES_CLOSE) failed. Error ", 
            symbols[i], _LastError);
         return;
      }
   }
   ...
We will consider the calculation of one price point value (in the deposit currency) in Part 5.

---

## Page 598

Part 4. Common APIs
598
4.1 0 Matrices and vectors
It is also important to note, that bars with the same indexes may have different timestamps on
different financial instruments, for example, if there was a holiday in one of the countries and the
market was closed (outside of Forex, symbols may, in theory, have different trading session schedules).
To solve this problem, we would need a deeper analysis of quotes, taking into account bar times and
their synchronization before inserting them into the rates matrix. We do not do this here to maintain
simplicity, and also because the Forex market operates according to the same rules most of the time.
We split the matrix into two parts: the initial part will be used to find a solution (this emulates
optimization on history), and the subsequent part will be used for a forward test (calculation of
subsequent balance changes).
   matrix split[];
   if(BarOffset > 0)
   {
      // training on BarCount - BarOffset bars
      // check on BarOffset bars
      ulong parts[] = {BarCount - BarOffset, BarOffset};
      rates.Split(parts, 0, split);
   }
  
   // solve the system of linear equations for the model
   vector x = (BarOffset > 0) ? split[0].LstSq(model) : rates.LstSq(model);
   Print("Solution (lots per symbol): ");
   Print(x);
   ...
Now, when we have a solution, let's build the balance curve for all bars of the sample (the ideal
"historical" part will be at the beginning, and then the "future" part will begin, which was not used to
adjust the model).
   vector balance = vector::Zeros(BarCount);
   for(int i = 1; i < BarCount; ++i)
   {
      balance[i] = 0;
      for(int j = 0; j < size; ++j)
      {
         balance[i] += (float)(rates[i][j] * x[j]);
      }
   }
   ...
Let's estimate the quality of the solution by the R2 criterion.

---

## Page 599

Part 4. Common APIs
599
4.1 0 Matrices and vectors
   if(BarOffset > 0)
   {
      // make a copy of the balance
      vector backtest = balance;
      // select only "historical" bars for backtesting
      backtest.Resize(BarCount - BarOffset);
      // bars for the forward test have to be copied manually
      vector forward(BarOffset);
      for(int i = 0; i < BarOffset; ++i)
      {
         forward[i] = balance[BarCount - BarOffset + i];
      }
      // compute regression metrics independently for both parts
      Print("Backtest R2 = ", backtest.RegressionMetric(REGRESSION_R2));
      Print("Forward R2 = ", forward.RegressionMetric(REGRESSION_R2));
   }
   else
   {
      Print("R2 = ", balance.RegressionMetric(REGRESSION_R2));
   }
   ...
To display the balance curve on a chart, you need to transfer data from a vector to an array.
   double array[];
   balance.Swap(array);
   
   // print the values of the changing balance with an accuracy of 2 digits
   Print("Balance: ");
   ArrayPrint(array, 2);
  
   // draw the balance curve in the chart object ("backtest" and "forward")
   GraphPlot(array, CurveType);
}
Here is an example of a log obtained by running the script on EURUSD,H1 .
Solution (lots per symbol): 
[-0.0057809334,-0.0079846876,0.0088985749,-0.0041461736,-0.010710154,-0.0025694175,0.01493552]
Backtest R2 = 0.9896645616246145
Forward R2 = 0.8667852183780984
Balance: 
 0.00  1.68  3.38  3.90  5.04  5.92  7.09  7.86  9.17  9.88 
 9.55 10.77 12.06 13.67 15.35 15.89 16.28 15.91 16.85 16.58
And here is what the virtual balance curve looks like.

---

## Page 600

Part 4. Common APIs
600
4.1 0 Matrices and vectors
Virtual balance of trading a portfolio of currencies by lots according to the  decision
The left half has a more even shape and a higher R2, which is not surprising because the model (X
variables) was adjusted specifically for it.
Just out of interest, we will increase the depth of training and verification by 1 0 times, that is, we will
set in the parameters BarCount = 200 and BarOffset = 1 00. We will get a new picture.


---

