---
title: "Object Functions"
url: "https://www.mql5.com/en/docs/objects"
hierarchy: []
scraped_at: "2025-11-28 09:30:30"
---

# Object Functions

[MQL5 Reference](/en/docs "MQL5 Reference")Object Functions

* [ObjectCreate](/en/docs/objects/objectcreate "ObjectCreate")
* [ObjectName](/en/docs/objects/objectname "ObjectName")
* [ObjectDelete](/en/docs/objects/objectdelete "ObjectDelete")
* [ObjectsDeleteAll](/en/docs/objects/objectdeleteall "ObjectsDeleteAll")
* [ObjectFind](/en/docs/objects/objectfind "ObjectFind")
* [ObjectGetTimeByValue](/en/docs/objects/objectgettimebyvalue "ObjectGetTimeByValue")
* [ObjectGetValueByTime](/en/docs/objects/objectgetvaluebytime "ObjectGetValueByTime")
* [ObjectMove](/en/docs/objects/objectmove "ObjectMove")
* [ObjectsTotal](/en/docs/objects/objectstotal "ObjectsTotal")
* [ObjectSetDouble](/en/docs/objects/objectsetdouble "ObjectSetDouble")
* [ObjectSetInteger](/en/docs/objects/objectsetinteger "ObjectSetInteger")
* [ObjectSetString](/en/docs/objects/objectsetstring "ObjectSetString")
* [ObjectGetDouble](/en/docs/objects/objectgetdouble "ObjectGetDouble")
* [ObjectGetInteger](/en/docs/objects/objectgetinteger "ObjectGetInteger")
* [ObjectGetString](/en/docs/objects/objectgetstring "ObjectGetString")
* [TextSetFont](/en/docs/objects/textsetfont "TextSetFont")
* [TextOut](/en/docs/objects/textout "TextOut")
* [TextGetSize](/en/docs/objects/textgetsize "TextGetSize")

# Object Functions

This is the group of functions intended for working with graphic objects relating to any specified chart.

The functions defining the properties of graphical objects, as well as [ObjectCreate()](/en/docs/objects/objectcreate) and [ObjectMove()](/en/docs/objects/objectmove) operations for creating and moving objects along the chart are actually used for sending commands to the chart. If these functions are executed successfully, the command is included in the common queue of the chart events. Visual changes in the properties of graphical objects are implemented when handling the queue of the chart events.

Thus, do not expect an immediate visual update of graphical objects after calling these functions. Generally, the graphical objects on the chart are updated automatically by the terminal following the change events - a new quote arrival, resizing the chart window, etc. Use [ChartRedraw()](/en/docs/chart_operations/chartredraw) function to forcefully update the graphical objects.

| Function | Action |
| --- | --- |
| [ObjectCreate](/en/docs/objects/objectcreate) | Creates an object of the specified type in a specified chart |
| [ObjectName](/en/docs/objects/objectname) | Returns the name of an object of the corresponding type in the specified chart (specified chart subwindow) |
| [ObjectDelete](/en/docs/objects/objectdelete) | Removes the object with the specified name from the specified chart (from the specified chart subwindow) |
| [ObjectsDeleteAll](/en/docs/objects/objectdeleteall) | Removes all objects of the specified type from the specified chart (from the specified chart subwindow) |
| [ObjectFind](/en/docs/objects/objectfind) | Searches for an object with the specified ID by the name |
| [ObjectGetTimeByValue](/en/docs/objects/objectgettimebyvalue) | Returns the time value for the specified object price value |
| [ObjectGetValueByTime](/en/docs/objects/objectgetvaluebytime) | Returns the price value of an object for the specified time |
| [ObjectMove](/en/docs/objects/objectmove) | Changes the coordinates of the specified object anchor point |
| [ObjectsTotal](/en/docs/objects/objectstotal) | Returns the number of objects of the specified type in the specified chart (specified chart subwindow) |
| [ObjectGetDouble](/en/docs/objects/objectgetdouble) | Returns the double value of the corresponding object property |
| [ObjectGetInteger](/en/docs/objects/objectgetinteger) | Returns the integer value of the corresponding object property |
| [ObjectGetString](/en/docs/objects/objectgetstring) | Returns the string value of the corresponding object property |
| [ObjectSetDouble](/en/docs/objects/objectsetdouble) | Sets the value of the corresponding object property |
| [ObjectSetInteger](/en/docs/objects/objectsetinteger) | Sets the value of the corresponding object property |
| [ObjectSetString](/en/docs/objects/objectsetstring) | Sets the value of the corresponding object property |
| [TextSetFont](/en/docs/objects/textsetfont) | Sets the font for displaying the text using drawing methods (Arial 20 used by default) |
| [TextOut](/en/docs/objects/textout) | Transfers the text to the custom array (buffer) designed for creation of a graphical [resource](/en/docs/common/resourcecreate) |
| [TextGetSize](/en/docs/objects/textgetsize) | Returns the string's width and height at the current [font settings](/en/docs/objects/textsetfont) |

Every graphical object should have a name unique within one [chart](/en/docs/chart_operations), including its subwindows. Changing of a name of a graphic object generates two events: event of deletion of an object with the old name, and event of creation of an object with a new name.

After an object is created or an [object property](/en/docs/constants/objectconstants/enum_object_property) is modified it is recommended to call the [ChartRedraw()](/en/docs/chart_operations/chartredraw) function, which commands the client terminal to forcibly draw a chart (and all [visible](/en/docs/constants/objectconstants/visible) objects in it).

[PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger "PlotIndexGetInteger")

[ObjectCreate](/en/docs/objects/objectcreate "ObjectCreate")