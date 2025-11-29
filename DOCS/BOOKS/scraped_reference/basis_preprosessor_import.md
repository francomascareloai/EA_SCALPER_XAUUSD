---
title: "Importing Function (#import)"
url: "https://www.mql5.com/en/docs/basis/preprosessor/import"
hierarchy: []
scraped_at: "2025-11-28 09:31:37"
---

# Importing Function (#import)

[MQL5 Reference](/en/docs "MQL5 Reference")[Language Basics](/en/docs/basis "Language Basics")[Preprocessor](/en/docs/basis/preprosessor "Preprocessor")Importing Functions (#import)

* [Macro substitution (#define)](/en/docs/basis/preprosessor/constant "Macro substitution (#define)")
* [Program Properties (#property)](/en/docs/basis/preprosessor/compilation "Program Properties (#property)")
* [Including Files (#include)](/en/docs/basis/preprosessor/include "Including Files (#include)")
* Importing Functions (#import)
* [Conditional Compilation (#ifdef, #ifndef, #else, #endif)](/en/docs/basis/preprosessor/conditional_compilation "Conditional Compilation (#ifdef, #ifndef, #else, #endif)")

# Importing Function (#import)

Functions are imported from compiled MQL5 modules (\*.ex5 files) and from operating system modules (\*.dll files). The module name is specified in the #import directive. For compiler to be able to correctly form the imported function call and organize proper [transmission parameters](/en/docs/basis/function/parameterpass), the full description of [functions](/en/docs/basis/function) is needed. Function descriptions immediately follow the #import "module name" directive. New command #import (can be without parameters) completes the block of imported function descriptions.

| |
| --- |
| #import "file\_name"     func1 define;     func2 define;     ...     funcN define; #import |

Imported functions can have any names. Functions having the same names but from different modules can be imported at the same time. Imported functions can have names that coincide with the names of built-in functions. Operation of [scope resolution](/en/docs/basis/operations/other#context_allow) defines which of the functions should be called.

The order of searching for a file specified after the #import keyword is described in [Call of Imported Functions](/en/docs/runtime/imports).

Since the imported functions are outside the compiled module, the compiler can not verify the validity of passed parameters. Therefore, to avoid run-time errors, one must accurately describe the composition and order of parameters passed to imported functions. Parameters passed to imported functions (both from EX5, and from the DLL-module) can have default values.

The following can't be used for parameters in imported functions:

* [pointers](/en/docs/basis/types/object_pointers) (\*);
* links to objects that contain [dynamic arrays](/en/docs/basis/types/dynamic_array) and/or pointers.

Classes, string arrays or complex objects that contain strings and/or dynamic arrays of any types cannot be passed as a parameter to functions imported from DLL.

Examples:

| |
| --- |
| #import "stdlib.ex5" string ErrorDescription(int error\_code); int    RGB(int red\_value,int green\_value,int blue\_value); bool   CompareDoubles(double number1,double number2); string DoubleToStrMorePrecision(double number,int precision); string IntegerToHexString(int integer\_number); #import "ExpertSample.dll" int    GetIntValue(int); double GetDoubleValue(double); string GetStringValue(string); double GetArrayItemValue(double &arr[],int,int); bool   SetArrayItemValue(double &arr[],int,int,double); double GetRatesItemValue(double &rates[][6],int,int,int); #import |

To import functions during execution of a mql5 program, early binding is used. This means that the library is loaded during the loading of a program using its ex5 program.

It's not recommended to use a fully qualified name of the loadable module of type Drive:\Directory\FileName.Ext. MQL5 libraries are loaded from the terminal\_dir\MQL5\Libraries folder.

If the imported function has different call versions for 32- and 64-bit Windows versions, both of them should be imported, and the right function version should be called explicitly using the [\_IsX64](/en/docs/predefined/_isx64) variable.

Example:

| |
| --- |
| #import "user32.dll" //--- For the 32-bit system int    MessageBoxW(uint hWnd,string lpText,string lpCaption,uint uType); //--- For the 64-bit system int    MessageBoxW(ulong hWnd,string lpText,string lpCaption,uint uType); #import //+------------------------------------------------------------------+ //|  MessageBox\_32\_64\_bit uses the proper version of MessageBoxW()   | //+------------------------------------------------------------------+ int MessageBox\_32\_64\_bit()   {    int res=-1;    //--- If we are using the 64-bit Windows    if(\_IsX64)      {       ulong hwnd=0;       res=MessageBoxW(hwnd,"64-bit MessageBoxW call example","MessageBoxW 64 bit",MB\_OK|MB\_ICONINFORMATION);      }    else  // We are using the 32-bit Windows      {       uint hwnd=0;       res=MessageBoxW(hwnd,"32-bit MessageBoxW call example","MessageBoxW 32 bit",MB\_OK|MB\_ICONINFORMATION);      }    return (res);   } //+------------------------------------------------------------------+ //| Script program start function                                    | //+------------------------------------------------------------------+ void OnStart()   { //---    int ans=MessageBox\_32\_64\_bit();    PrintFormat("MessageBox\_32\_64\_bit returned %d",ans);   } |

## Importing functions from .NET libraries

To work with .NET library functions, simply import DLL itself without defining specific functions. MetaEditor automatically imports all functions it is possible to work with:

* Simple structures (POD, plain old data) — structures that contain only simple data types.
* Public static functions having parameters, in which only simple types and POD structures or their arrays are used

To call functions from the library, simply import it:

| |
| --- |
| #import "TestLib.dll" //+------------------------------------------------------------------+ //| Script program start function                                    | //+------------------------------------------------------------------+ void OnStart()   {    int x=41;    TestClass::Inc(x);    Print(x);   } |

The C# code of the Inc function of the TestClass looks as follows:

| |
| --- |
| public class TestClass   {    public static void Inc(ref int x)      {       x++;      }   } |

As a result of execution, the script returns the value of 42.

See also

[Including Files](/en/docs/basis/preprosessor/include)

[Including Files (#include)](/en/docs/basis/preprosessor/include "Including Files (#include)")

[Conditional Compilation (#ifdef, #ifndef, #else, #endif)](/en/docs/basis/preprosessor/conditional_compilation "Conditional Compilation (#ifdef, #ifndef, #else, #endif)")