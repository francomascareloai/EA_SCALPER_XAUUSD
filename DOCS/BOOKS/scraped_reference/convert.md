---
title: "Conversion Functions"
url: "https://www.mql5.com/en/docs/convert"
hierarchy: []
scraped_at: "2025-11-28 09:30:36"
---

# Conversion Functions

[MQL5 Reference](/en/docs "MQL5 Reference")Conversion Functions

* [CharToString](/en/docs/convert/chartostring "CharToString")
* [CharArrayToString](/en/docs/convert/chararraytostring "CharArrayToString")
* [CharArrayToStruct](/en/docs/convert/chararraytostruct "CharArrayToStruct")
* [StructToCharArray](/en/docs/convert/structtochararray "StructToCharArray")
* [ColorToARGB](/en/docs/convert/colortoargb "ColorToARGB")
* [ColorToString](/en/docs/convert/colortostring "ColorToString")
* [DoubleToString](/en/docs/convert/doubletostring "DoubleToString")
* [EnumToString](/en/docs/convert/enumtostring "EnumToString")
* [IntegerToString](/en/docs/convert/integertostring "IntegerToString")
* [ShortToString](/en/docs/convert/shorttostring "ShortToString")
* [ShortArrayToString](/en/docs/convert/shortarraytostring "ShortArrayToString")
* [TimeToString](/en/docs/convert/timetostring "TimeToString")
* [NormalizeDouble](/en/docs/convert/normalizedouble "NormalizeDouble")
* [StringToCharArray](/en/docs/convert/stringtochararray "StringToCharArray")
* [StringToColor](/en/docs/convert/stringtocolor "StringToColor")
* [StringToDouble](/en/docs/convert/stringtodouble "StringToDouble")
* [StringToInteger](/en/docs/convert/stringtointeger "StringToInteger")
* [StringToShortArray](/en/docs/convert/stringtoshortarray "StringToShortArray")
* [StringToTime](/en/docs/convert/stringtotime "StringToTime")
* [StringFormat](/en/docs/convert/stringformat "StringFormat")

# Conversion Functions

This is a group of functions that provide conversion of data from one format into another.

The [NormalizeDouble()](/en/docs/convert/normalizedouble) function must be specially noted as it provides the necessary accuracy of the price presentation. In trading operations, no unnormalized prices may be used if their accuracy even a digit exceeds that required by the trade server.

| Function | Action |
| --- | --- |
| [CharToString](/en/docs/convert/chartostring) | Converting a symbol code into a one-character string |
| [DoubleToString](/en/docs/convert/doubletostring) | Converting a numeric value to a text line with a specified accuracy |
| [EnumToString](/en/docs/convert/enumtostring) | Converting an enumeration value of any type to string |
| [NormalizeDouble](/en/docs/convert/normalizedouble) | Rounding of a floating point number to a specified accuracy |
| [StringToDouble](/en/docs/convert/stringtodouble) | Converting a string containing a symbol representation of number into number of double type |
| [StringToInteger](/en/docs/convert/stringtointeger) | Converting a string containing a symbol representation of number into number of long type |
| [StringToTime](/en/docs/convert/stringtotime) | Converting a string containing time or date in "yyyy.mm.dd [hh:mi]" format into datetime type |
| [TimeToString](/en/docs/convert/timetostring) | Converting a value containing time in seconds elapsed since 01.01.1970 into a string of "yyyy.mm.dd hh:mi" format |
| [IntegerToString](/en/docs/convert/integertostring) | Converting int into a string of preset length |
| [ShortToString](/en/docs/convert/shorttostring) | Converting symbol code (unicode) into one-symbol string |
| [ShortArrayToString](/en/docs/convert/shortarraytostring) | Copying array part into a string |
| [StringToShortArray](/en/docs/convert/stringtoshortarray) | Symbol-wise copying a string to a selected part of array of ushort type |
| [CharArrayToString](/en/docs/convert/chararraytostring) | Converting symbol code (ansi) into one-symbol array |
| [StringToCharArray](/en/docs/convert/stringtochararray) | Symbol-wise copying a string converted from Unicode to ANSI, to a selected place of array of uchar type |
| [CharArrayToStruct](/en/docs/convert/chararraytostruct) | Copy uchar type array to [POD structure](/en/docs/basis/types/classes#simple_structure) |
| [StructToCharArray](/en/docs/convert/structtochararray) | Copy [POD structure](/en/docs/basis/types/classes#simple_structure) to uchar type array |
| [ColorToARGB](/en/docs/convert/colortoargb) | Converting color type to uint type to receive ARGB representation of the color. |
| [ColorToString](/en/docs/convert/colortostring) | Converting color value into string as "R,G,B" |
| [StringToColor](/en/docs/convert/stringtocolor) | Converting "R,G,B" string or string with color name into color type value |
| [StringFormat](/en/docs/convert/stringformat) | Converting number into string according to preset format |

See also

[Use of a Codepage](/en/docs/constants/io_constants/codepageusage)

[BlasL3HeR2K](/en/docs/matrix/openblas/blas_level3/blasl3her2k "BlasL3HeR2K")

[CharToString](/en/docs/convert/chartostring "CharToString")