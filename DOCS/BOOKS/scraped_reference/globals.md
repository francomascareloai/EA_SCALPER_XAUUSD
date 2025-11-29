---
title: "Global Variables of the Client Terminal"
url: "https://www.mql5.com/en/docs/globals"
hierarchy: []
scraped_at: "2025-11-28 09:30:28"
---

# Global Variables of the Client Terminal

[MQL5 Reference](/en/docs "MQL5 Reference")Global Variables of the Terminal

* [GlobalVariableCheck](/en/docs/globals/globalvariablecheck "GlobalVariableCheck")
* [GlobalVariableTime](/en/docs/globals/globalvariabletime "GlobalVariableTime")
* [GlobalVariableDel](/en/docs/globals/globalvariabledel "GlobalVariableDel")
* [GlobalVariableGet](/en/docs/globals/globalvariableget "GlobalVariableGet")
* [GlobalVariableName](/en/docs/globals/globalvariablename "GlobalVariableName")
* [GlobalVariableSet](/en/docs/globals/globalvariableset "GlobalVariableSet")
* [GlobalVariablesFlush](/en/docs/globals/globalvariablesflush "GlobalVariablesFlush")
* [GlobalVariableTemp](/en/docs/globals/globalvariabletemp "GlobalVariableTemp")
* [GlobalVariableSetOnCondition](/en/docs/globals/globalvariablesetoncondition "GlobalVariableSetOnCondition")
* [GlobalVariablesDeleteAll](/en/docs/globals/globalvariablesdeleteall "GlobalVariablesDeleteAll")
* [GlobalVariablesTotal](/en/docs/globals/globalvariablestotal "GlobalVariablesTotal")

# Global Variables of the Client Terminal

There is a group set of functions for working with global variables.

Global variables of the client terminal should not be mixed up with variables declared in the [global scope](/en/docs/basis/variables/global) of the mql5 program.

Global variables are kept in the client terminal for 4 weeks since the last access, then they will be deleted automatically. An access to a global variable is not only setting of a new value, but reading of the global variable value, as well.

Global variables of the client terminal are accessible simultaneously from all mql5 programs launched in the client terminal.

| Function | Action |
| --- | --- |
| [GlobalVariableCheck](/en/docs/globals/globalvariablecheck) | Checks the existence of a global variable with the specified name |
| [GlobalVariableTime](/en/docs/globals/globalvariabletime) | Returns time of the last accessing the global variable |
| [GlobalVariableDel](/en/docs/globals/globalvariabledel) | Deletes a global variable |
| [GlobalVariableGet](/en/docs/globals/globalvariableget) | Returns the value of a global variable |
| [GlobalVariableName](/en/docs/globals/globalvariablename) | Returns the name of a global variable by its ordinal number in the list of global variables |
| [GlobalVariableSet](/en/docs/globals/globalvariableset) | Sets the new value to a global variable |
| [GlobalVariablesFlush](/en/docs/globals/globalvariablesflush) | Forcibly saves contents of all global variables to a disk |
| [GlobalVariableTemp](/en/docs/globals/globalvariabletemp) | Sets the new value to a global variable, that exists only in the current session of the terminal |
| [GlobalVariableSetOnCondition](/en/docs/globals/globalvariablesetoncondition) | Sets the new value of the existing global variable by condition |
| [GlobalVariablesDeleteAll](/en/docs/globals/globalvariablesdeleteall) | Deletes global variables with the specified prefix in their names |
| [GlobalVariablesTotal](/en/docs/globals/globalvariablestotal) | Returns the total number of global variables |

[SendNotification](/en/docs/network/sendnotification "SendNotification")

[GlobalVariableCheck](/en/docs/globals/globalvariablecheck "GlobalVariableCheck")