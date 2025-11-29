---
title: "Working with files"
url: "https://www.mql5.com/en/book/common/files"
hierarchy: []
scraped_at: "2025-11-28 09:49:02"
---

# Working with files

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Common APIs](/en/book/common "Common APIs")Working with files

* [Information storage methods: text and binary](/en/book/common/files/files_modes_bin_txt "Information storage methods: text and binary")
* [Writing and reading files in simplified mode](/en/book/common/files/files_save_load "Writing and reading files in simplified mode")
* [Opening and closing files](/en/book/common/files/files_open_close "Opening and closing files")
* [Managing file descriptors](/en/book/common/files/files_handles "Managing file descriptors")
* [Selecting an encoding for text mode](/en/book/common/files/files_txt_codepage "Selecting an encoding for text mode")
* [Writing and reading arrays](/en/book/common/files/files_arrays "Writing and reading arrays")
* [Writing and reading structures (binary files)](/en/book/common/files/files_bin_structs "Writing and reading structures (binary files)")
* [Writing and reading variables (binary files)](/en/book/common/files/files_bin_atomic "Writing and reading variables (binary files)")
* [Writing and reading variables (text files)](/en/book/common/files/files_txt_atomic "Writing and reading variables (text files)")
* [Managing position in a file](/en/book/common/files/files_cursor "Managing position in a file")
* [Getting file properties](/en/book/common/files/files_properties "Getting file properties")
* [Force write cache to disk](/en/book/common/files/files_flush "Force write cache to disk")
* [Deleting a file and checking if it exists](/en/book/common/files/files_exist_delete "Deleting a file and checking if it exists")
* [Copying and moving files](/en/book/common/files/files_copy_move "Copying and moving files")
* [Searching for files and folders](/en/book/common/files/files_find "Searching for files and folders")
* [Working with folders](/en/book/common/files/files_folders "Working with folders")
* [File or folder selection dialog](/en/book/common/files/files_select "File or folder selection dialog")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Working with files

It is difficult to find a program that does not use data input-output. We already know that MQL programs can receive settings via [input variables](/en/book/basis/variables/input_variables) and output information to the log as we used the latter in almost all test scripts. But in most cases, this is not enough.

For example, quite a significant part of program customization includes amounts of data that cannot be accommodated in the input parameters. A program may need to be integrated with some external analytical tools, i.e., uploading market information in a standard or specialized format, processing and then loading it into the terminal in a new form, in particular, as trading signals, a set of neural network weights or decision tree coefficients. Furthermore, it can be convenient to maintain a separate log for an MQL program.

The file subsystem provides the most universal opportunities for such tasks. The MQL5 API provides a wide range of functions for working with files, including functions to create, delete, search, write, and read the files. We will study all this in this chapter.

All file operations in MQL5 are limited to a special area on the disk, which is called a sandbox. This is done for security reasons so that no MQL program can be used for malicious purposes and harm your computer or operating system.

Advanced users can avoid this limitation using special measures, which we will discuss later. But this should only be done in exceptional cases while observing precautions and accepting all responsibility.

For each instance of the terminal installed on the computer, the sandbox root directory is located at <terminal\_data\_folder>/MQL5/Files/. From the MetaEditor, you can open the data folder using the command File -> Open Data Folder. If you have sufficient access rights on the computer, this directory is usually the same place where the terminal is installed. If you do not have the required permissions, the path will look like this:

| |
| --- |
| X:/Users/<user\_name>/AppData/Roaming/MetaQuotes/Terminal/<instance\_id>/MQL5/Files/ |

Here X is a drive letter where the system is installed, <user\_name> is the Windows user login, <instance\_id> is a unique identifier of the terminal instance. The Users folder also has an alias "Documents and Settings".

Please note that in the case of a remote connection to a computer via RDP (Remote Desktop Protocol), the terminal will always use the Roaming directory and its subdirectories even if you have administrator rights.

Let's recall that the MQL5 folder in the data directory is the place where all MQL programs are stored: both their source codes and compiled ex5 files. Each type of MQL program, including indicators, Expert Advisors, scripts, and others, has a dedicated subfolder in the MQL5 folder. So the Files folder for working files is next to them.

In addition to this individual sandbox of each copy of the terminal on the computer, there is a common, shared sandbox for all terminals: they can communicate through it. The path to it runs through the home folder of the Windows user and may differ depending on the version of the operating system. For example, in standard installations of Windows 7, 8, and 10, it looks like this:

| |
| --- |
| X:/Users/<user\_name>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/ |

Again, the folder can be easily accessed through MetaTrader: run the command File -> Open Shared Data Folder, and you will be inside the Common folder.

Some types of MQL programs (Expert Advisors and indicators) can be executed not only in the terminal but also in the tester. When running in it, the shared sandbox remains accessible, and instead of a single instance sandbox, a folder inside the test agent is used. As a rule, it looks like:

| |
| --- |
| X:/<terminal\_path>/Tester/Agent-IP-port/MQL5/Files/ |

This may not be visible in the MQL program itself, i.e., all file functions work in exactly the same way. However, from the user's point of view, it may seem that there is some kind of problem. For example, if the program saves the results of its work to a file, it will be deleted in the tester's agent folder after the run is completed (as if the file had never been created). This routine approach is designed to prevent potentially valuable data of one program from leaking into another program that can be tested on the same agent some time later (especially since agents can be shared). Other technologies are provided for transferring files to agents and returning results from agents to the terminal, which we will discuss in the fifth Part of the book.

To get around the sandbox limitation, you can use Windows' ability to assign symbolic links to file system objects. In our case, the connections (junction) are best suited for redirecting access to folders on the local computer. They are created using the following command (meaning the Windows command line):

| |
| --- |
| mklink /J new\_name existing\_target |

The parameter new\_name is the name of the new virtual folder that will point to the real folder existing\_target.

To create connections to external folders outside the sandbox, it is recommended to create a dedicated folder inside MQL5/Files, for example, Links. Then, having entered it, you can execute the above command by selecting new\_name and substituting the real path outside the sandbox as existing\_target. For example, the following command will create in the folder Links a new link named Settings, which will provide access to the MQL5/Presets folder:

| |
| --- |
| mklink /J Settings "..\..\Presets\" |

The relative path "..\..\" assumes that the command is executed in the specified MQL5/Files/Links folder. A combination of two dots ".." indicates the transition from the current folder to the parent. Specified twice, this combination instructs to go up the path hierarchy twice. As a result, the target folder (existing\_target) will be generated as MQL5/Presets. But in the existing\_target parameter, you can also specify an absolute path.

You can delete symbolic links like regular files (but, of course, you should first make sure that it is the folder with the arrow icon in its lower left corner that is being deleted, i.e. the link, and not the original folder). It is recommended to do this immediately, as soon as you no longer need to go beyond the sandbox. The fact is that the created virtual folders become available to all MQL programs, not just yours, and it is not known how other people's programs can use the additional freedom.

Many sections of the chapter deal with file names. They act as file system element identifiers and have similar rules, including some restrictions.

Please note that the file name cannot contain some characters that play special roles in the file system ('<', '>', '/', '\\', '"', ':', '|', '\* ', '?'), as well as any characters with codes from 0 to 31 inclusive.

The following file names are also reserved for special use in the operating system and cannot be used: CON, PRN, AUX, NUL, COM1, COM2, COM3, COM4, COM5, COM6, COM7, COM8, COM9, LPT1, LPT2, LPT3, LPT4, LPT5, LPT6, LPT7, LPT8, LPT9.

It should be noted that the Windows file system does not see the fundamental difference between letters in different cases, so names like "Name", "NAME", and "name" refer to the same element.

Windows allows both backslashes '\\' and forward slashes '/' to be used as a separator character between path components (subfolders and files). However, the backslash needs to be screened (that is, actually written twice) in MQL5 strings, because the '\' character itself is special: it is used to construct control character sequences, such as '\r', '\n', '\t' and others (see section [Character types](/en/book/basis/builtin_types/characters)). For example, the following paths are equivalent: "MQL5Book/file.txt" and "MQL5Book\\file.txt".

The dot character '.' serves as a separator between the name and the extension. If a file system element has multiple dots in its identifier, then the extension is the fragment to the right of the rightmost dot, and everything to the left of it is the name. The title (before the dot) or extension (after the dot) can be empty. For example, the file name without an extension is "text", and the file without a name (only with the extension) is ".txt".

The total length of the path and file name in Windows has limitations. At the same time, to manage files in MQL5, it should be taken into account that the path to the sandbox will be added to their path and name, i.e., even less space will be allocated for the names of file objects in MQL function calls. By default, the overall length limit is the system constant MAX\_PATH, which is equal 260. Starting from Windows 10 (build 1607), you can increase this limit to 32767. To do this, you need to save the following text in a .reg file and run it by adding it to the Windows Registry. 
  
[HKEY\_LOCAL\_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem] 
"LongPathsEnabled"=dword:00000001 
  
For other versions of Windows, you can use workarounds from the command line. In particular, you can shorten the path using the connections discussed above (by creating a virtual folder with a short path). You can also use the shell command -subst, For example, subst z: c:\very\long\path (see Windows Help for details).

[Endianness control in integers](/en/book/common/maths/maths_byte_swap "Endianness control in integers")

[Information storage methods: text and binary](/en/book/common/files/files_modes_bin_txt "Information storage methods: text and binary")