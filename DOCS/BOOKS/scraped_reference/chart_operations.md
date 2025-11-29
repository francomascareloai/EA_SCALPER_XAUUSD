---
title: "Chart Operations"
url: "https://www.mql5.com/en/docs/chart_operations"
hierarchy: []
scraped_at: "2025-11-28 09:31:14"
---

# Chart Operations

[MQL5 Reference](/en/docs "MQL5 Reference")Chart Operations

* [ChartApplyTemplate](/en/docs/chart_operations/chartapplytemplate "ChartApplyTemplate")
* [ChartSaveTemplate](/en/docs/chart_operations/chartsavetemplate "ChartSaveTemplate")
* [ChartWindowFind](/en/docs/chart_operations/chartwindowfind "ChartWindowFind")
* [ChartTimePriceToXY](/en/docs/chart_operations/charttimepricetoxy "ChartTimePriceToXY")
* [ChartXYToTimePrice](/en/docs/chart_operations/chartxytotimeprice "ChartXYToTimePrice")
* [ChartOpen](/en/docs/chart_operations/chartopen "ChartOpen")
* [ChartFirst](/en/docs/chart_operations/chartfirst "ChartFirst")
* [ChartNext](/en/docs/chart_operations/chartnext "ChartNext")
* [ChartClose](/en/docs/chart_operations/chartclose "ChartClose")
* [ChartSymbol](/en/docs/chart_operations/chartsymbol "ChartSymbol")
* [ChartPeriod](/en/docs/chart_operations/chartperiod "ChartPeriod")
* [ChartRedraw](/en/docs/chart_operations/chartredraw "ChartRedraw")
* [ChartSetDouble](/en/docs/chart_operations/chartsetdouble "ChartSetDouble")
* [ChartSetInteger](/en/docs/chart_operations/chartsetinteger "ChartSetInteger")
* [ChartSetString](/en/docs/chart_operations/chartsetstring "ChartSetString")
* [ChartGetDouble](/en/docs/chart_operations/chartgetdouble "ChartGetDouble")
* [ChartGetInteger](/en/docs/chart_operations/chartgetinteger "ChartGetInteger")
* [ChartGetString](/en/docs/chart_operations/chartgetstring "ChartGetString")
* [ChartNavigate](/en/docs/chart_operations/chartnavigate "ChartNavigate")
* [ChartID](/en/docs/chart_operations/chartid "ChartID")
* [ChartIndicatorAdd](/en/docs/chart_operations/chartindicatoradd "ChartIndicatorAdd")
* [ChartIndicatorDelete](/en/docs/chart_operations/chartindicatordelete "ChartIndicatorDelete")
* [ChartIndicatorGet](/en/docs/chart_operations/chartindicatorget "ChartIndicatorGet")
* [ChartIndicatorName](/en/docs/chart_operations/chartindicatorname "ChartIndicatorName")
* [ChartIndicatorsTotal](/en/docs/chart_operations/chartindicatorstotal "ChartIndicatorsTotal")
* [ChartWindowOnDropped](/en/docs/chart_operations/chartwindowondropped "ChartWindowOnDropped")
* [ChartPriceOnDropped](/en/docs/chart_operations/chartpriceondropped "ChartPriceOnDropped")
* [ChartTimeOnDropped](/en/docs/chart_operations/charttimeondropped "ChartTimeOnDropped")
* [ChartXOnDropped](/en/docs/chart_operations/chartxondropped "ChartXOnDropped")
* [ChartYOnDropped](/en/docs/chart_operations/chartyondropped "ChartYOnDropped")
* [ChartSetSymbolPeriod](/en/docs/chart_operations/chartsetsymbolperiod "ChartSetSymbolPeriod")
* [ChartScreenShot](/en/docs/chart_operations/chartscreenshot "ChartScreenShot")

# Chart Operations

Functions for setting chart properties ([ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartSetString](/en/docs/chart_operations/chartsetstring)) are asynchronous and are used for sending update commands to a chart. If these functions are executed successfully, the command is included in the common queue of the chart events. Chart property changes are implemented along with handling of the events queue of this chart.

Thus, do not expect an immediate update of the chart after calling asynchronous functions. Use the [ChartRedraw()](/en/docs/chart_operations/chartredraw) function to forcedly update the chart appearance and properties.

| Function | Action |
| --- | --- |
| [ChartApplyTemplate](/en/docs/chart_operations/chartapplytemplate) | Applies a specific template from a specified file to the chart |
| [ChartSaveTemplate](/en/docs/chart_operations/chartsavetemplate) | Saves current chart settings in a template with a specified name |
| [ChartWindowFind](/en/docs/chart_operations/chartwindowfind) | Returns the number of a subwindow where an indicator is drawn |
| [ChartTimePriceToXY](/en/docs/chart_operations/charttimepricetoxy) | Converts the coordinates of a chart from the time/price representation to the X and Y coordinates |
| [ChartXYToTimePrice](/en/docs/chart_operations/chartxytotimeprice) | Converts the X and Y coordinates on a chart to the time and price values |
| [ChartOpen](/en/docs/chart_operations/chartopen) | Opens a new chart with the specified symbol and period |
| [ChartClose](/en/docs/chart_operations/chartclose) | Closes the specified chart |
| [ChartFirst](/en/docs/chart_operations/chartfirst) | Returns the ID of the first chart of the client terminal |
| [ChartNext](/en/docs/chart_operations/chartnext) | Returns the chart ID of the chart next to the specified one |
| [ChartSymbol](/en/docs/chart_operations/chartsymbol) | Returns the symbol name of the specified chart |
| [ChartPeriod](/en/docs/chart_operations/chartperiod) | Returns the period value of the specified chart |
| [ChartRedraw](/en/docs/chart_operations/chartredraw) | Calls a forced redrawing of a specified chart |
| [ChartSetDouble](/en/docs/chart_operations/chartsetdouble) | Sets the double value for a corresponding property of the specified chart |
| [ChartSetInteger](/en/docs/chart_operations/chartsetinteger) | Sets the integer value (datetime, int, color, bool or char) for a corresponding property of the specified chart |
| [ChartSetString](/en/docs/chart_operations/chartsetstring) | Sets the string value for a corresponding property of the specified chart |
| [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) | Returns the double value property of the specified chart |
| [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) | Returns the integer value property of the specified chart |
| [ChartGetString](/en/docs/chart_operations/chartgetstring) | Returns the string value property of the specified chart |
| [ChartNavigate](/en/docs/chart_operations/chartnavigate) | Performs shift of the specified chart by the specified number of bars relative to the specified position in the chart |
| [ChartID](/en/docs/chart_operations/chartid) | Returns the ID of the current chart |
| [ChartIndicatorAdd](/en/docs/chart_operations/chartindicatoradd) | Adds an indicator with the specified handle into a specified chart window |
| [ChartIndicatorDelete](/en/docs/chart_operations/chartindicatordelete) | Removes an indicator with a specified name from the specified chart window |
| [ChartIndicatorGet](/en/docs/chart_operations/chartindicatorget) | Returns the handle of the indicator with the specified short name in the specified chart window |
| [ChartIndicatorName](/en/docs/chart_operations/chartindicatorname) | Returns the short name of the indicator by the number in the indicators list on the specified chart window |
| [ChartIndicatorsTotal](/en/docs/chart_operations/chartindicatorstotal) | Returns the number of all indicators applied to the specified chart window. |
| [ChartWindowOnDropped](/en/docs/chart_operations/chartwindowondropped) | Returns the number (index) of the chart subwindow the Expert Advisor or script has been dropped to |
| [ChartPriceOnDropped](/en/docs/chart_operations/chartpriceondropped) | Returns the price coordinate of the chart point the Expert Advisor or script has been dropped to |
| [ChartTimeOnDropped](/en/docs/chart_operations/charttimeondropped) | Returns the time coordinate of the chart point the Expert Advisor or script has been dropped to |
| [ChartXOnDropped](/en/docs/chart_operations/chartxondropped) | Returns the X coordinate of the chart point the Expert Advisor or script has been dropped to |
| [ChartYOnDropped](/en/docs/chart_operations/chartyondropped) | Returns the Y coordinate of the chart point the Expert Advisor or script has been dropped to |
| [ChartSetSymbolPeriod](/en/docs/chart_operations/chartsetsymbolperiod) | Changes the symbol value and a period of the specified chart |
| [ChartScreenShot](/en/docs/chart_operations/chartscreenshot) | Provides a screenshot of the chart of its current state in a GIF, PNG or BMP format depending on specified extension |

[CustomBookAdd](/en/docs/customsymbols/custombookadd "CustomBookAdd")

[ChartApplyTemplate](/en/docs/chart_operations/chartapplytemplate "ChartApplyTemplate")