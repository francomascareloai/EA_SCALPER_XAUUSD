---
title: "File Functions"
url: "https://www.mql5.com/en/docs/files"
hierarchy: []
scraped_at: "2025-11-28 09:30:26"
---

# File Functions

[MQL5 Reference](/en/docs "MQL5 Reference")File Functions

* [FileSelectDialog](/en/docs/files/fileselectdialog "FileSelectDialog")
* [FileFindFirst](/en/docs/files/filefindfirst "FileFindFirst")
* [FileFindNext](/en/docs/files/filefindnext "FileFindNext")
* [FileFindClose](/en/docs/files/filefindclose "FileFindClose")
* [FileIsExist](/en/docs/files/fileisexist "FileIsExist")
* [FileOpen](/en/docs/files/fileopen "FileOpen")
* [FileClose](/en/docs/files/fileclose "FileClose")
* [FileCopy](/en/docs/files/filecopy "FileCopy")
* [FileDelete](/en/docs/files/filedelete "FileDelete")
* [FileMove](/en/docs/files/filemove "FileMove")
* [FileFlush](/en/docs/files/fileflush "FileFlush")
* [FileGetInteger](/en/docs/files/filegetinteger "FileGetInteger")
* [FileIsEnding](/en/docs/files/fileisending "FileIsEnding")
* [FileIsLineEnding](/en/docs/files/fileislineending "FileIsLineEnding")
* [FileReadArray](/en/docs/files/filereadarray "FileReadArray")
* [FileReadBool](/en/docs/files/filereadbool "FileReadBool")
* [FileReadDatetime](/en/docs/files/filereaddatetime "FileReadDatetime")
* [FileReadDouble](/en/docs/files/filereaddouble "FileReadDouble")
* [FileReadFloat](/en/docs/files/filereadfloat "FileReadFloat")
* [FileReadInteger](/en/docs/files/filereadinteger "FileReadInteger")
* [FileReadLong](/en/docs/files/filereadlong "FileReadLong")
* [FileReadNumber](/en/docs/files/filereadnumber "FileReadNumber")
* [FileReadString](/en/docs/files/filereadstring "FileReadString")
* [FileReadStruct](/en/docs/files/filereadstruct "FileReadStruct")
* [FileSeek](/en/docs/files/fileseek "FileSeek")
* [FileSize](/en/docs/files/filesize "FileSize")
* [FileTell](/en/docs/files/filetell "FileTell")
* [FileWrite](/en/docs/files/filewrite "FileWrite")
* [FileWriteArray](/en/docs/files/filewritearray "FileWriteArray")
* [FileWriteDouble](/en/docs/files/filewritedouble "FileWriteDouble")
* [FileWriteFloat](/en/docs/files/filewritefloat "FileWriteFloat")
* [FileWriteInteger](/en/docs/files/filewriteinteger "FileWriteInteger")
* [FileWriteLong](/en/docs/files/filewritelong "FileWriteLong")
* [FileWriteString](/en/docs/files/filewritestring "FileWriteString")
* [FileWriteStruct](/en/docs/files/filewritestruct "FileWriteStruct")
* [FileLoad](/en/docs/files/fileload "FileLoad")
* [FileSave](/en/docs/files/filesave "FileSave")
* [FolderCreate](/en/docs/files/foldercreate "FolderCreate")
* [FolderDelete](/en/docs/files/folderdelete "FolderDelete")
* [FolderClean](/en/docs/files/folderclean "FolderClean")

# File Functions

This is a group of functions for working with files.

For security reasons, work with files is strictly controlled in the MQL5 language. Files with which file operations are conducted using MQL5 means cannot be outside the file sandbox.

There are two directories (with subdirectories) in which working files can be located:

* terminal\_data\_folder\MQL5\FILES\ (in the terminal menu select to view "File" - "Open the data directory");
* the common folder for all the terminals installed on a computer - usually located in the directory C:\Documents and Settings\All Users\Application Data\MetaQuotes\Terminal\Common\Files.

There is a program method to obtain names of these catalogs using the [TerminalInfoString()](/en/docs/check/terminalinfostring) function, using the [ENUM\_TERMINAL\_INFO\_STRING](/en/docs/constants/environment_state/terminalstatus#enum_terminal_info_string) enumeration:

| |
| --- |
| //--- Folder that stores the terminal data    string terminal\_data\_path=TerminalInfoString(TERMINAL\_DATA\_PATH); //--- Common folder for all client terminals    string common\_data\_path=TerminalInfoString(TERMINAL\_COMMONDATA\_PATH); |

Work with files from other directories is prohibited.

If the file is opened for writing using [FileOpen()](/en/docs/files/fileopen), all subfolders specified in the path will be created if there are no such ones.

File functions allow working with so-called "named pipes". To do this, simply call [FileOpen()](/en/docs/files/fileopen) function with appropriate parameters.

| Function | Action |
| --- | --- |
| [FileSelectDialog](/en/docs/files/fileselectdialog) | Create a file or folder opening/creation dialog |
| [FileFindFirst](/en/docs/files/filefindfirst) | Starts the search of files in a directory in accordance with the specified filter |
| [FileFindNext](/en/docs/files/filefindnext) | Continues the search started by the FileFindFirst() function |
| [FileFindClose](/en/docs/files/filefindclose) | Closes search handle |
| [FileOpen](/en/docs/files/fileopen) | Opens a file with a specified name and flag |
| [FileDelete](/en/docs/files/filedelete) | Deletes a specified file |
| [FileFlush](/en/docs/files/fileflush) | Writes to a disk all data remaining in the input/output file buffer |
| [FileGetInteger](/en/docs/files/filegetinteger) | Gets an integer property of a file |
| [FileIsEnding](/en/docs/files/fileisending) | Defines the end of a file in the process of reading |
| [FileIsLineEnding](/en/docs/files/fileislineending) | Defines the end of a line in a text file in the process of reading |
| [FileClose](/en/docs/files/fileclose) | Closes a previously opened file |
| [FileIsExist](/en/docs/files/fileisexist) | Checks the existence of a file |
| [FileCopy](/en/docs/files/filecopy) | Copies the original file from a local or shared folder to another file |
| [FileMove](/en/docs/files/filemove) | Moves or renames a file |
| [FileReadArray](/en/docs/files/filereadarray) | Reads arrays of any type except for string from the file of the BIN type |
| [FileReadBool](/en/docs/files/filereadbool) | Reads from the file of the CSV type a string from the current position till a delimiter (or till the end of a text line) and converts the read string to a value of bool type |
| [FileReadDatetime](/en/docs/files/filereaddatetime) | Reads from the file of the CSV type a string of one of the formats: "YYYY.MM.DD HH:MM:SS", "YYYY.MM.DD" or "HH:MM:SS" - and converts it into a datetime value |
| [FileReadDouble](/en/docs/files/filereaddouble) | Reads a double value from the current position of the file pointer |
| [FileReadFloat](/en/docs/files/filereadfloat) | Reads a float value from the current position of the file pointer |
| [FileReadInteger](/en/docs/files/filereadinteger) | Reads int, short or char value from the current position of the file pointer |
| [FileReadLong](/en/docs/files/filereadlong) | Reads a long type value from the current position of the file pointer |
| [FileReadNumber](/en/docs/files/filereadnumber) | Reads from the file of the CSV type a string from the current position till a delimiter (or til the end of a text line) and converts the read string into double value |
| [FileReadString](/en/docs/files/filereadstring) | Reads a string from the current position of a file pointer from a file |
| [FileReadStruct](/en/docs/files/filereadstruct) | Reads the contents from a binary file  into a structure passed as a parameter, from the current position of the file pointer |
| [FileSeek](/en/docs/files/fileseek) | Moves the position of the file pointer by a specified number of bytes relative to the specified position |
| [FileSize](/en/docs/files/filesize) | Returns the size of a corresponding open file |
| [FileTell](/en/docs/files/filetell) | Returns the current position of the file pointer of a corresponding open file |
| [FileWrite](/en/docs/files/filewrite) | Writes data to a file of CSV or TXT type |
| [FileWriteArray](/en/docs/files/filewritearray) | Writes arrays of any type except for string into a file of BIN type |
| [FileWriteDouble](/en/docs/files/filewritedouble) | Writes value of the double type from the current position of a file pointer into a binary file |
| [FileWriteFloat](/en/docs/files/filewritefloat) | Writes value of the float type from the current position of a file pointer into a binary file |
| [FileWriteInteger](/en/docs/files/filewriteinteger) | Writes value of the int type from the current position of a file pointer into a binary file |
| [FileWriteLong](/en/docs/files/filewritelong) | Writes value of the long type from the current position of a file pointer into a binary file |
| [FileWriteString](/en/docs/files/filewritestring) | Writes the value of a string parameter into a BIN or TXT file starting from the current position of the file pointer |
| [FileWriteStruct](/en/docs/files/filewritestruct) | Writes the contents of a structure passed as a parameter into a binary file, starting from the current position of the file pointer |
| [FileLoad](/en/docs/files/fileload) | Reads all data of a specified binary file into a passed array of numeric types or simple structures |
| [FileSave](/en/docs/files/filesave) | Writes to a binary file all elements of an array passed as a parameter |
| [FolderCreate](/en/docs/files/foldercreate) | Creates a folder in the Files directory |
| [FolderDelete](/en/docs/files/folderdelete) | Removes a selected directory. If the folder is not empty, then it can't be removed |
| [FolderClean](/en/docs/files/folderclean) | Deletes all files in the specified folder |

[GlobalVariablesTotal](/en/docs/globals/globalvariablestotal "GlobalVariablesTotal")

[FileSelectDialog](/en/docs/files/fileselectdialog "FileSelectDialog")