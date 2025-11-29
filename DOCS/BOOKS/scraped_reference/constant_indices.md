---
title: "List of MQL5 Constants"
url: "https://www.mql5.com/en/docs/constant_indices"
hierarchy: []
scraped_at: "2025-11-28 09:30:50"
---

# List of MQL5 Constants

[MQL5 Reference](/en/docs "MQL5 Reference")List of MQL5 Constants

* [Language Basics](/en/docs/basis "Language Basics")
* [Constants, Enumerations and Structures](/en/docs/constants "Constants, Enumerations and Structures")
* [MQL5 programs](/en/docs/runtime "MQL5 programs")
* [Predefined Variables](/en/docs/predefined "Predefined Variables")
* [Common Functions](/en/docs/common "Common Functions")
* [Array Functions](/en/docs/array "Array Functions")
* [Matrix and Vector Methods](/en/docs/matrix "Matrix and Vector Methods")
* [Conversion Functions](/en/docs/convert "Conversion Functions")
* [Math Functions](/en/docs/math "Math Functions")
* [String Functions](/en/docs/strings "String Functions")
* [Date and Time](/en/docs/dateandtime "Date and Time")
* [Account Information](/en/docs/account "Account Information")
* [Checkup](/en/docs/check "Checkup")
* [Event Handling](/en/docs/event_handlers "Event Handling")
* [Market Info](/en/docs/marketinformation "Market Info")
* [Economic Calendar](/en/docs/calendar "Economic Calendar")
* [Timeseries and Indicators Access](/en/docs/series "Timeseries and Indicators Access")
* [Custom Symbols](/en/docs/customsymbols "Custom Symbols")
* [Chart Operations](/en/docs/chart_operations "Chart Operations")
* [Trade Functions](/en/docs/trading "Trade Functions")
* [Trade Signals](/en/docs/signals "Trade Signals")
* [Network Functions](/en/docs/network "Network Functions")
* [Global Variables of the Terminal](/en/docs/globals "Global Variables of the Terminal")
* [File Functions](/en/docs/files "File Functions")
* [Custom Indicators](/en/docs/customind "Custom Indicators")
* [Object Functions](/en/docs/objects "Object Functions")
* [Technical Indicators](/en/docs/indicators "Technical Indicators")
* [Working with Optimization Results](/en/docs/optimization_frames "Working with Optimization Results")
* [Working with Events](/en/docs/eventfunctions "Working with Events")
* [Working with OpenCL](/en/docs/opencl "Working with OpenCL")
* [Working with databases](/en/docs/database "Working with databases")
* [Working with DirectX](/en/docs/directx "Working with DirectX")
* [Python Integration](/en/docs/python_metatrader5 "Python Integration")
* [ONNX models](/en/docs/onnx "ONNX models")
* [Standard Library](/en/docs/standardlibrary "Standard Library")
* [Moving from MQL4](/en/docs/migration "Moving from MQL4")
* [List of MQL5 Functions](/en/docs/function_indices "List of MQL5 Functions")
* List of MQL5 Constants

# List of MQL5 Constants

All MQL5 constants in alphabetical order.

| Constant | Description | Usage |
| --- | --- | --- |
| \_\_DATE\_\_ | File compilation date without time (hours, minutes and seconds are equal to 0) | [Print](/en/docs/common/print) |
| \_\_DATETIME\_\_ | File compilation date and time | [Print](/en/docs/common/print) |
| \_\_FILE\_\_ | Name of the currently compiled file | [Print](/en/docs/common/print) |
| \_\_FUNCSIG\_\_ | Signature of the function in whose body the macro is located. Logging of the full description of functions can be useful in the identification of [overloaded functions](/en/docs/basis/function/functionoverload) | [Print](/en/docs/common/print) |
| \_\_FUNCTION\_\_ | Name of the function, in whose body the macro is located | [Print](/en/docs/common/print) |
| \_\_LINE\_\_ | Line number in the source code, in which the macro is located | [Print](/en/docs/common/print) |
| \_\_MQLBUILD\_\_, \_\_MQL5BUILD\_\_ | Compiler build number | [Print](/en/docs/common/print) |
| \_\_PATH\_\_ | An absolute path to the file that is currently being compiled | [Print](/en/docs/common/print) |
| ACCOUNT\_ASSETS | The current assets of an account | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_BALANCE | Account balance in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_COMMISSION\_BLOCKED | The current blocked commission amount on an account | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_COMPANY | Name of a company that serves the account | [AccountInfoString](/en/docs/account/accountinfostring) |
| ACCOUNT\_CREDIT | Account credit in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_CURRENCY | Account currency | [AccountInfoString](/en/docs/account/accountinfostring) |
| ACCOUNT\_EQUITY | Account equity in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_LEVERAGE | Account leverage | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_LIABILITIES | The current liabilities on an account | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_LIMIT\_ORDERS | Maximum allowed number of active pending orders | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_LOGIN | Account number | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_MARGIN | Account margin used in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_MARGIN\_FREE | Free margin of an account in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_MARGIN\_INITIAL | Initial margin. The amount reserved on an account to cover the margin of all pending orders | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_MARGIN\_LEVEL | Account margin level in percents | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_MARGIN\_MAINTENANCE | Maintenance margin. The minimum equity reserved on an account to cover the minimum amount of all open positions | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_MARGIN\_SO\_CALL | Margin call level. Depending on the set ACCOUNT\_MARGIN\_SO\_MODE is expressed in percents or in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_MARGIN\_SO\_MODE | Mode for setting the minimal allowed margin | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_MARGIN\_SO\_SO | Margin stop out level. Depending on the set ACCOUNT\_MARGIN\_SO\_MODE is expressed in percents or in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_NAME | Client name | [AccountInfoString](/en/docs/account/accountinfostring) |
| ACCOUNT\_PROFIT | Current profit of an account in the deposit currency | [AccountInfoDouble](/en/docs/account/accountinfodouble) |
| ACCOUNT\_SERVER | Trade server name | [AccountInfoString](/en/docs/account/accountinfostring) |
| ACCOUNT\_STOPOUT\_MODE\_MONEY | Account stop out mode in money | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_STOPOUT\_MODE\_PERCENT | Account stop out mode in percents | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_TRADE\_ALLOWED | [Allowed trade](/en/docs/runtime/tradepermission) for the current account | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_TRADE\_EXPERT | [Allowed trade](/en/docs/runtime/tradepermission) for an Expert Advisor | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_TRADE\_MODE | Account trade mode | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_TRADE\_MODE\_CONTEST | Contest account | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_TRADE\_MODE\_DEMO | Demo account | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ACCOUNT\_TRADE\_MODE\_REAL | Real account | [AccountInfoInteger](/en/docs/account/accountinfointeger) |
| ALIGN\_CENTER | Centered (only for the Edit object) | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger), [ChartScreenShot](/en/docs/customind) |
| ALIGN\_LEFT | Left alignment | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger), [ChartScreenShot](/en/docs/customind) |
| ALIGN\_RIGHT | Right alignment | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger), [ChartScreenShot](/en/docs/customind) |
| ANCHOR\_CENTER | Anchor point strictly in the center of the object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_LEFT | Anchor point to the left in the center | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_LEFT\_LOWER | Anchor point at the lower left corner | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_LEFT\_UPPER | Anchor point at the upper left corner | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_LOWER | Anchor point below in the center | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_RIGHT | Anchor point to the right in the center | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_RIGHT\_LOWER | Anchor point at the lower right corner | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_RIGHT\_UPPER | Anchor point at the upper right corner | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ANCHOR\_UPPER | Anchor point above in the center | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| BASE\_LINE | Main line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| BOOK\_TYPE\_BUY | Buy order (Bid) | [MqlBookInfo](/en/docs/constants/structures/mqlbookinfo) |
| BOOK\_TYPE\_BUY\_MARKET | Buy order by Market | [MqlBookInfo](/en/docs/constants/structures/mqlbookinfo) |
| BOOK\_TYPE\_SELL | Sell order (Offer) | [MqlBookInfo](/en/docs/constants/structures/mqlbookinfo) |
| BOOK\_TYPE\_SELL\_MARKET | Sell order by Market | [MqlBookInfo](/en/docs/constants/structures/mqlbookinfo) |
| BORDER\_FLAT | Flat form | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| BORDER\_RAISED | Prominent form | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| BORDER\_SUNKEN | Concave form | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| CHAR\_MAX | Maximal value, which can be represented by char type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| CHAR\_MIN | Minimal value, which can be represented by char type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| [CHART\_AUTOSCROLL](/en/docs/constants/chartconstants/charts_samples#chart_autoscroll) | Mode of automatic moving to the right border of the chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| CHART\_BARS | Display as a sequence of bars | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger) |
| CHART\_BEGIN | Chart beginning (the oldest prices) | [ChartNavigate](/en/docs/chart_operations/chartnavigate) |
| [CHART\_BRING\_TO\_TOP](/en/docs/constants/chartconstants/charts_samples#chart_bring_to_top) | Show chart on top of other charts | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| CHART\_CANDLES | Display as Japanese candlesticks | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger) |
| [CHART\_COLOR\_ASK](/en/docs/constants/chartconstants/charts_samples#chart_color_ask) | Ask price level color | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_BACKGROUND](/en/docs/constants/chartconstants/charts_samples#chart_color_background) | Chart background color | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_BID](/en/docs/constants/chartconstants/charts_samples#chart_color_bid) | Bid price level color | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_CANDLE\_BEAR](/en/docs/constants/chartconstants/charts_samples#chart_color_candle_bear) | Body color of a bear candlestick | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_CANDLE\_BULL](/en/docs/constants/chartconstants/charts_samples#chart_color_candle_bull) | Body color of a bull candlestick | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_CHART\_DOWN](/en/docs/constants/chartconstants/charts_samples#chart_color_chart_down) | Color for the down bar, shadows and body borders of bear candlesticks | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_CHART\_LINE](/en/docs/constants/chartconstants/charts_samples#chart_color_chart_line) | Line chart color and color of "Doji" Japanese candlesticks | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_CHART\_UP](/en/docs/constants/chartconstants/charts_samples#chart_color_chart_up) | Color for the up bar, shadows and body borders of bull candlesticks | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_FOREGROUND](/en/docs/constants/chartconstants/charts_samples#chart_color_foreground) | Color of axes, scales and OHLC line | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_GRID](/en/docs/constants/chartconstants/charts_samples#chart_color_grid) | Grid color | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_LAST](/en/docs/constants/chartconstants/charts_samples#chart_color_last) | Line color of the last executed deal price (Last) | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_STOP\_LEVEL](/en/docs/constants/chartconstants/charts_samples#chart_color_stop_level) | Color of stop order levels (Stop Loss and Take Profit) | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COLOR\_VOLUME](/en/docs/constants/chartconstants/charts_samples#chart_color_volume) | Color of volumes and position opening levels | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_COMMENT](/en/docs/constants/chartconstants/charts_samples#chart_comment) | Text of a comment in a chart | [ChartSetString](/en/docs/chart_operations/chartsetstring), [ChartGetString](/en/docs/chart_operations/chartgetstring) |
| CHART\_CURRENT\_POS | Current position | [ChartNavigate](/en/docs/chart_operations/chartnavigate) |
| [CHART\_DRAG\_TRADE\_LEVELS](/en/docs/constants/chartconstants/charts_samples#chart_drag_trade_levels) | Permission to drag trading levels on a chart with a mouse. The drag mode is enabled by default (true value) | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_EVENT\_MOUSE\_MOVE](/en/docs/constants/chartconstants/charts_samples#chart_event_mouse_move) | Send notifications of mouse move and mouse click events ([CHARTEVENT\_MOUSE\_MOVE](/en/docs/constants/chartconstants/enum_chartevents)) to all mql5 programs on a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_EVENT\_OBJECT\_CREATE](/en/docs/constants/chartconstants/charts_samples#chart_event_object_create) | Send a notification of an event of new object creation ([CHARTEVENT\_OBJECT\_CREATE](/en/docs/constants/chartconstants/enum_chartevents)) to all mql5-programs on a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_EVENT\_OBJECT\_DELETE](/en/docs/constants/chartconstants/charts_samples#chart_event_object_delete) | Send a notification of an event of object deletion ([CHARTEVENT\_OBJECT\_DELETE](/en/docs/constants/chartconstants/enum_chartevents)) to all mql5-programs on a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_FIRST\_VISIBLE\_BAR](/en/docs/constants/chartconstants/charts_samples#chart_first_visible_bar) | Number of the first visible bar in the chart. Indexing of bars is the same as for [timeseries](/en/docs/series). | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_FIXED\_MAX](/en/docs/constants/chartconstants/charts_samples#chart_fixed_max) | Fixed  chart maximum | [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) |
| [CHART\_FIXED\_MIN](/en/docs/constants/chartconstants/charts_samples#chart_fixed_min) | Fixed  chart minimum | [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) |
| [CHART\_FIXED\_POSITION](/en/docs/constants/chartconstants/charts_samples#chart_fixed_position) | Chart fixed position from the left border in percent value. Chart fixed position is marked by a small gray triangle on the horizontal time axis. It is displayed only if the automatic chart scrolling to the right on tick incoming is disabled (see CHART\_AUTOSCROLL property). The bar on a fixed position remains in the same place when zooming in and out. | [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) |
| [CHART\_FOREGROUND](/en/docs/constants/chartconstants/charts_samples#chart_foreground) | Price chart in the foreground | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_HEIGHT\_IN\_PIXELS](/en/docs/constants/chartconstants/charts_samples#chart_height_in_pixels) | Chart height in pixels | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_IS\_OBJECT](/en/docs/constants/chartconstants/charts_samples#chart_is_object) | Identifying "Chart" ([OBJ\_CHART)](/en/docs/constants/objectconstants/enum_object) object – returns true for a graphical object. Returns false for a real chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| CHART\_LINE | Display as a line drawn by Close prices | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger) |
| [CHART\_MODE](/en/docs/constants/chartconstants/charts_samples#chart_mode) | Chart type (candlesticks, bars or line) | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_MOUSE\_SCROLL](/en/docs/constants/chartconstants/charts_samples#chart_mouse_scroll) | Scrolling the chart horizontally using the left mouse button. Vertical scrolling is also available if the value of any following properties is set to true: CHART\_SCALEFIX, CHART\_SCALEFIX\_11 or CHART\_SCALE\_PT\_PER\_BAR | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_POINTS\_PER\_BAR](/en/docs/constants/chartconstants/charts_samples#chart_points_per_bar) | Scale in points per bar | [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) |
| [CHART\_PRICE\_MAX](/en/docs/constants/chartconstants/charts_samples#chart_price_max) | Chart maximum | [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) |
| [CHART\_PRICE\_MIN](/en/docs/constants/chartconstants/charts_samples#chart_price_min) | Chart minimum | [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) |
| [CHART\_SCALE](/en/docs/constants/chartconstants/charts_samples#chart_scale) | Scale | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SCALE\_PT\_PER\_BAR](/en/docs/constants/chartconstants/charts_samples#chart_scale_pt_per_bar) | Scale to be specified in points per bar | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SCALEFIX](/en/docs/constants/chartconstants/charts_samples#chart_scalefix) | Fixed scale mode | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SCALEFIX\_11](/en/docs/constants/chartconstants/charts_samples#chart_scalefix_11) | Scale 1:1 mode | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHIFT](/en/docs/constants/chartconstants/charts_samples#chart_shift) | Mode of price chart indent from the right border | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHIFT\_SIZE](/en/docs/constants/chartconstants/charts_samples#chart_shift_size) | The size of the zero bar indent from the right border in percents | [ChartSetDouble](/en/docs/chart_operations/chartsetdouble), [ChartGetDouble](/en/docs/chart_operations/chartgetdouble) |
| [CHART\_SHOW\_ASK\_LINE](/en/docs/constants/chartconstants/charts_samples#chart_show_ask_line) | Display Ask values as a horizontal line in a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_BID\_LINE](/en/docs/constants/chartconstants/charts_samples#chart_show_bid_line) | Display Bid values as a horizontal line in a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_DATE\_SCALE](/en/docs/constants/chartconstants/charts_samples#chart_show_date_scale) | Showing the time scale on a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_GRID](/en/docs/constants/chartconstants/charts_samples#chart_show_grid) | Display grid in the chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_LAST\_LINE](/en/docs/constants/chartconstants/charts_samples#chart_show_last_line) | Display Last values as a horizontal line in a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_OBJECT\_DESCR](/en/docs/constants/chartconstants/charts_samples#chart_show_object_descr) | Pop-up descriptions of graphical objects | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_OHLC](/en/docs/constants/chartconstants/charts_samples#chart_show_ohlc) | Show OHLC values in the upper left corner | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_ONE\_CLICK](/en/docs/constants/chartconstants/charts_samples#chart_show_one_click) | Showing the ["One click trading"](https://www.metatrader5.com/en/terminal/help/startworking/settings "\"One click trading\"") panel on a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_PERIOD\_SEP](/en/docs/constants/chartconstants/charts_samples#chart_show_period_sep) | Display vertical separators between adjacent periods | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_PRICE\_SCALE](/en/docs/constants/chartconstants/charts_samples#chart_show_price_scale) | Showing the price scale on a chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_TRADE\_LEVELS](/en/docs/constants/chartconstants/charts_samples#chart_show_trade_levels) | Displaying trade levels in the chart (levels of open positions, Stop Loss, Take Profit and pending orders) | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_SHOW\_VOLUMES](/en/docs/constants/chartconstants/charts_samples#chart_show_volumes) | Display volume in the chart | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_VISIBLE\_BARS](/en/docs/constants/chartconstants/charts_samples#chart_visible_bars) | The number of bars on the chart that can be displayed | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| CHART\_VOLUME\_HIDE | Volumes are not shown | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger) |
| CHART\_VOLUME\_REAL | Trade volumes | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger) |
| CHART\_VOLUME\_TICK | Tick volumes | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger) |
| [CHART\_WIDTH\_IN\_BARS](/en/docs/constants/chartconstants/charts_samples#chart_width_in_bars) | Chart width in bars | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_WIDTH\_IN\_PIXELS](/en/docs/constants/chartconstants/charts_samples#chart_width_in_pixels) | Chart width in pixels | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_WINDOW\_HANDLE](/en/docs/constants/chartconstants/charts_samples#chart_window_handle) | Chart window handle (HWND) | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_WINDOW\_IS\_VISIBLE](/en/docs/constants/chartconstants/charts_samples#chart_window_is_visible) | Visibility of subwindows | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_WINDOW\_YDISTANCE](/en/docs/constants/chartconstants/charts_samples#chart_window_ydistance) | The distance between the upper frame of the indicator subwindow and the upper frame of the main chart window, along the vertical Y axis, in pixels. In case of a mouse event, the cursor coordinates are passed in terms of the coordinates of the main chart window, while the coordinates of graphical objects in an indicator subwindow are set relative to the upper left corner of the subwindow. The value is required for converting the absolute coordinates of the main chart to the local coordinates of a subwindow for correct work with the graphical objects, whose coordinates are set relative to  the upper left corner of the subwindow frame. | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| [CHART\_WINDOWS\_TOTAL](/en/docs/constants/chartconstants/charts_samples#chart_windows_total) | The total number of chart windows, including indicator subwindows | [ChartSetInteger](/en/docs/chart_operations/chartsetinteger), [ChartGetInteger](/en/docs/chart_operations/chartgetinteger) |
| CHARTEVENT\_CHART\_CHANGE | Change of the chart size or modification of chart properties through the Properties dialog | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_CLICK | Clicking on a chart | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_CUSTOM | Initial number of an event from a range of custom events | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_CUSTOM\_LAST | The final number of an event from a range of custom events | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_KEYDOWN | Keystrokes | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_MOUSE\_MOVE | Mouse move, mouse clicks (if [CHART\_EVENT\_MOUSE\_MOVE](/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer)=true is set for the chart) | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_OBJECT\_CHANGE | [Graphical object](/en/docs/constants/objectconstants/enum_object) property changed via the properties dialog | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_OBJECT\_CLICK | Clicking on a [graphical object](/en/docs/constants/objectconstants/enum_object) | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_OBJECT\_CREATE | [Graphical object](/en/docs/constants/objectconstants/enum_object) created (if [CHART\_EVENT\_OBJECT\_CREATE](/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer)=true is set for the chart) | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_OBJECT\_DELETE | [Graphical object](/en/docs/constants/objectconstants/enum_object) deleted (if [CHART\_EVENT\_OBJECT\_DELETE](/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer)=true is set for the chart) | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_OBJECT\_DRAG | Drag and drop of a [graphical object](/en/docs/constants/objectconstants/enum_object) | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTEVENT\_OBJECT\_ENDEDIT | End of text editing in the graphical object Edit | [OnChartEvent](/en/docs/event_handlers/onchartevent) |
| CHARTS\_MAX | The maximum possible number of simultaneously open charts in the terminal | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| CHIKOUSPAN\_LINE | Chikou Span line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| clrAliceBlue | Alice Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrAntiqueWhite | Antique White | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrAqua | Aqua | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrAquamarine | Aquamarine | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBeige | Beige | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBisque | Bisque | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBlack | Black | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBlanchedAlmond | Blanched Almond | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBlue | Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBlueViolet | Blue Violet | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBrown | Brown | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrBurlyWood | Burly Wood | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrCadetBlue | Cadet Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrChartreuse | Chartreuse | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrChocolate | Chocolate | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrCoral | Coral | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrCornflowerBlue | Cornflower Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrCornsilk | Cornsilk | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrCrimson | Crimson | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkBlue | Dark Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkGoldenrod | Dark Goldenrod | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkGray | Dark Gray | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkGreen | Dark Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkKhaki | Dark Khaki | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkOliveGreen | Dark Olive Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkOrange | Dark Orange | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkOrchid | Dark Orchid | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkSalmon | Dark Salmon | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkSeaGreen | Dark Sea Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkSlateBlue | Dark Slate Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkSlateGray | Dark Slate Gray | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkTurquoise | Dark Turquoise | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDarkViolet | Dark Violet | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDeepPink | Deep Pink | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDeepSkyBlue | Deep Sky Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDimGray | Dim Gray | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrDodgerBlue | Dodger Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrFireBrick | Fire Brick | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrForestGreen | Forest Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrGainsboro | Gainsboro | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrGold | Gold | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrGoldenrod | Goldenrod | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrGray | Gray | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrGreen | Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrGreenYellow | Green Yellow | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrHoneydew | Honeydew | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrHotPink | Hot Pink | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrIndianRed | Indian Red | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrIndigo | Indigo | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrIvory | Ivory | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrKhaki | Khaki | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLavender | Lavender | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLavenderBlush | Lavender Blush | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLawnGreen | Lawn Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLemonChiffon | Lemon Chiffon | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightBlue | Light Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightCoral | Light Coral | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightCyan | Light Cyan | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightGoldenrod | Light Goldenrod | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightGray | Light Gray | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightGreen | Light Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightPink | Light Pink | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightSalmon | Light Salmon | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightSeaGreen | Light Sea Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightSkyBlue | Light Sky Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightSlateGray | Light Slate Gray | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightSteelBlue | Light Steel Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLightYellow | Light Yellow | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLime | Lime | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLimeGreen | Lime Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrLinen | Linen | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMagenta | Magenta | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMaroon | Maroon | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumAquamarine | Medium Aquamarine | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumBlue | Medium Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumOrchid | Medium Orchid | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumPurple | Medium Purple | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumSeaGreen | Medium Sea Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumSlateBlue | Medium Slate Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumSpringGreen | Medium Spring Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumTurquoise | Medium Turquoise | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMediumVioletRed | Medium Violet Red | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMidnightBlue | Midnight Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMintCream | Mint Cream | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMistyRose | Misty Rose | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrMoccasin | Moccasin | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrNavajoWhite | Navajo White | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrNavy | Navy | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrNONE | Absence of color | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| clrOldLace | Old Lace | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrOlive | Olive | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrOliveDrab | Olive Drab | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrOrange | Orange | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrOrangeRed | Orange Red | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrOrchid | Orchid | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPaleGoldenrod | Pale Goldenrod | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPaleGreen | Pale Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPaleTurquoise | Pale Turquoise | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPaleVioletRed | Pale Violet Red | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPapayaWhip | Papaya Whip | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPeachPuff | Peach Puff | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPeru | Peru | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPink | Pink | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPlum | Plum | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPowderBlue | Powder Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrPurple | Purple | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrRed | Red | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrRosyBrown | Rosy Brown | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrRoyalBlue | Royal Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSaddleBrown | Saddle Brown | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSalmon | Salmon | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSandyBrown | Sandy Brown | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSeaGreen | Sea Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSeashell | Seashell | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSienna | Sienna | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSilver | Silver | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSkyBlue | Sky Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSlateBlue | Slate Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSlateGray | Slate Gray | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSnow | Snow | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSpringGreen | Spring Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrSteelBlue | Steel Blue | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrTan | Tan | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrTeal | Teal | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrThistle | Thistle | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrTomato | Tomato | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrTurquoise | Turquoise | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrViolet | Violet | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrWheat | Wheat | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrWhite | White | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrWhiteSmoke | White Smoke | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrYellow | Yellow | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| clrYellowGreen | Yellow Green | [Web Colors](/en/docs/constants/objectconstants/webcolors) |
| CORNER\_LEFT\_LOWER | Center of coordinates is in the lower left corner of the chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| CORNER\_LEFT\_UPPER | Center of coordinates is in the upper left corner of the chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| CORNER\_RIGHT\_LOWER | Center of coordinates is in the lower right corner of the chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| CORNER\_RIGHT\_UPPER | Center of coordinates is in the upper right corner of the chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| CP\_ACP | The current Windows ANSI code page. | [CharArrayToString](/en/docs/convert/chararraytostring), [StringToCharArray](/en/docs/convert/stringtochararray), [FileOpen](/en/docs/files/fileopen) |
| CP\_MACCP | The current system Macintosh code page. Note: This value is mostly used in earlier created program codes and is of no use now, since modern Macintosh computers use Unicode for encoding. | [CharArrayToString](/en/docs/convert/chararraytostring), [StringToCharArray](/en/docs/convert/stringtochararray), [FileOpen](/en/docs/files/fileopen) |
| CP\_OEMCP | The current system OEM code page. | [CharArrayToString](/en/docs/convert/chararraytostring), [StringToCharArray](/en/docs/convert/stringtochararray), [FileOpen](/en/docs/files/fileopen) |
| CP\_SYMBOL | Symbol code page | [CharArrayToString](/en/docs/convert/chararraytostring), [StringToCharArray](/en/docs/convert/stringtochararray), [FileOpen](/en/docs/files/fileopen) |
| CP\_THREAD\_ACP | The Windows ANSI code page for the current thread. | [CharArrayToString](/en/docs/convert/chararraytostring), [StringToCharArray](/en/docs/convert/stringtochararray), [FileOpen](/en/docs/files/fileopen) |
| CP\_UTF7 | UTF-7 code page. | [CharArrayToString](/en/docs/convert/chararraytostring), [StringToCharArray](/en/docs/convert/stringtochararray), [FileOpen](/en/docs/files/fileopen) |
| CP\_UTF8 | UTF-8 code page. | [CharArrayToString](/en/docs/convert/chararraytostring), [StringToCharArray](/en/docs/convert/stringtochararray), [FileOpen](/en/docs/files/fileopen) |
| CRYPT\_AES128 | AES encryption with 128 bit key (16 bytes) | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| CRYPT\_AES256 | AES encryption with 256 bit key (32 bytes) | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| CRYPT\_ARCH\_ZIP | ZIP archives | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| CRYPT\_BASE64 | BASE64 | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| CRYPT\_DES | DES encryption with 56 bit key (7 bytes) | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| CRYPT\_HASH\_MD5 | MD5 HASH calculation | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| CRYPT\_HASH\_SHA1 | SHA1 HASH calculation | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| CRYPT\_HASH\_SHA256 | SHA256 HASH calculation | [CryptEncode](/en/docs/common/cryptencode), [CryptDecode](/en/docs/common/cryptdecode) |
| DBL\_DIG | Number of significant decimal digits for double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_EPSILON | Minimal value, which satisfies the condition: 1.0+DBL\_EPSILON != 1.0 (for double type) | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_MANT\_DIG | Bits count in a mantissa for double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_MAX | Maximal value, which can be represented by double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_MAX\_10\_EXP | Maximal decimal value of exponent degree for double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_MAX\_EXP | Maximal binary value of exponent degree for double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_MIN | Minimal positive value, which can be represented by double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_MIN\_10\_EXP | Minimal decimal value of exponent degree for double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DBL\_MIN\_EXP | Minimal binary value of exponent degree for double type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| DEAL\_COMMENT | Deal comment | [HistoryDealGetString](/en/docs/trading/historydealgetstring) |
| DEAL\_COMMISSION | Deal commission | [HistoryDealGetDouble](/en/docs/trading/historydealgetdouble) |
| DEAL\_ENTRY | Deal entry - entry in, entry out, reverse | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_ENTRY\_IN | Entry in | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_ENTRY\_INOUT | Reverse | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_ENTRY\_OUT | Entry out | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_MAGIC | Deal magic number (see [ORDER\_MAGIC](/en/docs/constants/tradingconstants/orderproperties)) | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_ORDER | Deal [order number](/en/docs/trading/historyordergetticket) | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_POSITION\_ID | [Identifier of a position](/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer), in the opening, modification or change of which this deal took part. Each position has a unique identifier that is assigned to all deals executed for the symbol during the entire lifetime of the position. | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_PRICE | Deal price | [HistoryDealGetDouble](/en/docs/trading/historydealgetdouble) |
| DEAL\_PROFIT | Deal profit | [HistoryDealGetDouble](/en/docs/trading/historydealgetdouble) |
| DEAL\_SWAP | Cumulative swap on close | [HistoryDealGetDouble](/en/docs/trading/historydealgetdouble) |
| DEAL\_SYMBOL | Deal symbol | [HistoryDealGetString](/en/docs/trading/historydealgetstring) |
| DEAL\_TIME | Deal time | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TIME\_MSC | The time of a deal execution in milliseconds since 01.01.1970 | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE | Deal type | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_BALANCE | Balance | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_BONUS | Bonus | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_BUY | Buy | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_BUY\_CANCELED | Canceled buy deal. There can be a situation when a previously executed buy deal is canceled. In this case, the type of the previously executed deal (DEAL\_TYPE\_BUY) is changed to DEAL\_TYPE\_BUY\_CANCELED, and its profit/loss is zeroized. Previously obtained profit/loss is charged/withdrawn using a separated balance operation | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_CHARGE | Additional charge | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_COMMISSION | Additional commission | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_COMMISSION\_AGENT\_DAILY | Daily agent commission | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_COMMISSION\_AGENT\_MONTHLY | Monthly agent commission | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_COMMISSION\_DAILY | Daily commission | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_COMMISSION\_MONTHLY | Monthly commission | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_CORRECTION | Correction | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_CREDIT | Credit | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_INTEREST | Interest rate | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_SELL | Sell | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_TYPE\_SELL\_CANCELED | Canceled sell deal. There can be a situation when a previously executed sell deal is canceled. In this case, the type of the previously executed deal (DEAL\_TYPE\_SELL) is changed to DEAL\_TYPE\_SELL\_CANCELED, and its profit/loss is zeroized. Previously obtained profit/loss is charged/withdrawn using a separated balance operation | [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) |
| DEAL\_VOLUME | Deal volume | [HistoryDealGetDouble](/en/docs/trading/historydealgetdouble) |
| [DRAW\_ARROW](/en/docs/customind/indicators_examples/draw_arrow) | Drawing arrows | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_BARS](/en/docs/customind/indicators_examples/draw_bars) | Display as a sequence of bars | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_CANDLES](/en/docs/customind/indicators_examples/draw_candles) | Display as a sequence of candlesticks | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_ARROW](/en/docs/customind/indicators_examples/draw_color_arrow) | Drawing multicolored arrows | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_BARS](/en/docs/customind/indicators_examples/draw_color_bars) | Multicolored bars | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_CANDLES](/en/docs/customind/indicators_examples/draw_color_candles) | Multicolored candlesticks | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_HISTOGRAM](/en/docs/customind/indicators_examples/draw_color_histogram) | Multicolored histogram from the zero line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_HISTOGRAM2](/en/docs/customind/indicators_examples/draw_color_histogram2) | Multicolored histogram of the two indicator buffers | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_LINE](/en/docs/customind/indicators_examples/draw_color_line) | Multicolored line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_SECTION](/en/docs/customind/indicators_examples/draw_color_section) | Multicolored section | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_COLOR\_ZIGZAG](/en/docs/customind/indicators_examples/draw_color_zigzag) | Multicolored ZigZag | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_FILLING](/en/docs/customind/indicators_examples/draw_filling) | Color fill between the two levels | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_HISTOGRAM](/en/docs/customind/indicators_examples/draw_histogram) | Histogram from the zero line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_HISTOGRAM2](/en/docs/customind/indicators_examples/draw_histogram2) | Histogram of the two indicator buffers | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_LINE](/en/docs/customind/indicators_examples/draw_line) | Line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_NONE](/en/docs/customind/indicators_examples/draw_none) | Not drawn | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_SECTION](/en/docs/customind/indicators_examples/draw_section) | Section | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| [DRAW\_ZIGZAG](/en/docs/customind/indicators_examples/draw_zigzag) | Style Zigzag allows vertical section on the bar | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles) |
| ELLIOTT\_CYCLE | Cycle | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_GRAND\_SUPERCYCLE | Grand Supercycle | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_INTERMEDIATE | Intermediate | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_MINOR | Minor | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_MINUETTE | Minuette | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_MINUTE | Minute | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_PRIMARY | Primary | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_SUBMINUETTE | Subminuette | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ELLIOTT\_SUPERCYCLE | Supercycle | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| EMPTY\_VALUE | Empty value in an indicator buffer | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| ERR\_ACCOUNT\_WRONG\_PROPERTY | Wrong account property ID | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_ARRAY\_BAD\_SIZE | Requested array size exceeds 2 GB | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_ARRAY\_RESIZE\_ERROR | Not enough memory for the relocation of an array, or an attempt to change the size of a static array | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_BOOKS\_CANNOT\_ADD | Depth Of Market can not be added | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_BOOKS\_CANNOT\_DELETE | Depth Of Market can not be removed | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_BOOKS\_CANNOT\_GET | The data from Depth Of Market can not be obtained | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_BOOKS\_CANNOT\_SUBSCRIBE | Error in subscribing to receive new data from Depth Of Market | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_BUFFERS\_NO\_MEMORY | Not enough memory for the distribution of indicator buffers | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_BUFFERS\_WRONG\_INDEX | Wrong indicator buffer index | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CANNOT\_CLEAN\_DIRECTORY | Failed to clear the directory (probably one or more files are blocked and removal operation failed) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CANNOT\_DELETE\_DIRECTORY | The directory cannot be removed | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CANNOT\_DELETE\_FILE | File deleting error | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CANNOT\_OPEN\_FILE | File opening error | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHAR\_ARRAY\_ONLY | Must be an array of type char | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_CANNOT\_CHANGE | Failed to change chart symbol and period | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_CANNOT\_CREATE\_TIMER | Failed to create timer | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_CANNOT\_OPEN | Chart opening error | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_INDICATOR\_CANNOT\_ADD | Error adding an indicator to chart | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_INDICATOR\_CANNOT\_DEL | Error deleting an indicator from the chart | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_INDICATOR\_NOT\_FOUND | Indicator not found on the specified chart | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_NAVIGATE\_FAILED | Error navigating through chart | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_NO\_EXPERT | No Expert Advisor in the chart that could handle the event | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_NO\_REPLY | Chart does not respond | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_NOT\_FOUND | Chart not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_SCREENSHOT\_FAILED | Error creating screenshots | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_TEMPLATE\_FAILED | Error applying template | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_WINDOW\_NOT\_FOUND | Subwindow containing the indicator was not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_WRONG\_ID | Wrong chart ID | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_WRONG\_PARAMETER | Error value of the parameter for the [function of working with charts](/en/docs/chart_operations) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CHART\_WRONG\_PROPERTY | Wrong chart property ID | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_CUSTOM\_WRONG\_PROPERTY | Wrong ID of the custom indicator property | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_DIRECTORY\_NOT\_EXIST | Directory does not exist | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_DOUBLE\_ARRAY\_ONLY | Must be an array of type double | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_BINSTRINGSIZE | String size must be specified, because the file is opened as binary | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_CACHEBUFFER\_ERROR | Not enough memory for cache to read | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_CANNOT\_REWRITE | File can not be rewritten | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_IS\_DIRECTORY | This is not a file, this is a directory | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_ISNOT\_DIRECTORY | This is a file, not a directory | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_NOT\_EXIST | File does not exist | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_NOTBIN | The file must be opened as a binary one | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_NOTCSV | The file must be opened as CSV | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_NOTTOREAD | The file must be opened for reading | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_NOTTOWRITE | The file must be opened for writing | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_NOTTXT | The file must be opened as a text | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_NOTTXTORCSV | The file must be opened as a text or CSV | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_READERROR | File reading error | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FILE\_WRITEERROR | Failed to write a resource to a file | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FLOAT\_ARRAY\_ONLY | Must be an array of type float | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FTP\_SEND\_FAILED | File sending via ftp failed | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_FUNCTION\_NOT\_ALLOWED | Function is not allowed for call | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_GLOBALVARIABLE\_EXISTS | Global variable of the client terminal with the same name already exists | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_GLOBALVARIABLE\_NOT\_FOUND | Global variable of the client terminal is not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_HISTORY\_NOT\_FOUND | Requested history not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_HISTORY\_WRONG\_PROPERTY | Wrong ID of the history property | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INCOMPATIBLE\_ARRAYS | Copying incompatible arrays. String array can be copied only to a string array, and a numeric array - in numeric array only | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INCOMPATIBLE\_FILE | A text file must be for string arrays, for other arrays - binary | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_CANNOT\_ADD | Error applying an indicator to chart | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_CANNOT\_APPLY | The indicator cannot be applied to another indicator | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_CANNOT\_CREATE | Indicator cannot be created | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_CUSTOM\_NAME | The first parameter in the array must be the name of the custom indicator | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_DATA\_NOT\_FOUND | Requested data not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_NO\_MEMORY | Not enough memory to add the indicator | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_PARAMETER\_TYPE | Invalid parameter type in the array when creating an indicator | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_PARAMETERS\_MISSING | No parameters when creating an indicator | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_UNKNOWN\_SYMBOL | Unknown symbol | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_WRONG\_HANDLE | Wrong indicator handle | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_WRONG\_INDEX | Wrong index of the requested indicator buffer | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INDICATOR\_WRONG\_PARAMETERS | Wrong number of parameters when creating an indicator | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INT\_ARRAY\_ONLY | Must be an array of type int | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INTERNAL\_ERROR | Unexpected internal error | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INVALID\_ARRAY | Array of a wrong type, wrong size, or a damaged object of a dynamic array | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INVALID\_DATETIME | Invalid date and/or time | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INVALID\_FILEHANDLE | A file with this handle was closed, or was not opening at all | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INVALID\_PARAMETER | Wrong parameter when calling the system function | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INVALID\_POINTER | Wrong pointer | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_INVALID\_POINTER\_TYPE | Wrong type of pointer | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_LONG\_ARRAY\_ONLY | Must be an array of type long | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_MAIL\_SEND\_FAILED | Email sending failed | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_MARKET\_LASTTIME\_UNKNOWN | Time of the last tick is not known (no ticks) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_MARKET\_NOT\_SELECTED | Symbol is not selected in MarketWatch | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_MARKET\_SELECT\_ERROR | Error adding or deleting a symbol in MarketWatch | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_MARKET\_UNKNOWN\_SYMBOL | Unknown symbol | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_MARKET\_WRONG\_PROPERTY | Wrong identifier of a symbol property | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_MQL5\_WRONG\_PROPERTY | Wrong identifier of the program property | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NO\_STRING\_DATE | No date in the string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NOT\_ENOUGH\_MEMORY | Not enough memory to perform the system function | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NOTIFICATION\_SEND\_FAILED | Failed to send a [notification](/en/docs/network/sendnotification) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NOTIFICATION\_TOO\_FREQUENT | Too frequent sending of notifications | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NOTIFICATION\_WRONG\_PARAMETER | Invalid parameter for sending a notification – an empty string or [NULL](/en/docs/basis/types/void) has been passed to the [SendNotification()](/en/docs/network/sendnotification) function | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NOTIFICATION\_WRONG\_SETTINGS | Wrong settings of notifications in the terminal (ID is not specified or permission is not set) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NOTINITIALIZED\_STRING | Not initialized string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_NUMBER\_ARRAYS\_ONLY | Must be a numeric array | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OBJECT\_ERROR | Error working with a graphical object | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OBJECT\_GETDATE\_FAILED | Unable to get date corresponding to the value | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OBJECT\_GETVALUE\_FAILED | Unable to get value corresponding to the date | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OBJECT\_NOT\_FOUND | Graphical object was not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OBJECT\_WRONG\_PROPERTY | Wrong ID of a graphical object property | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_ONEDIM\_ARRAYS\_ONLY | Must be a one-dimensional array | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_BUFFER\_CREATE | Failed to create an [OpenCL buffer](/en/docs/opencl/clbuffercreate) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_CONTEXT\_CREATE | Error creating the [OpenCL context](/en/docs/opencl/clcontextcreate) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_EXECUTE | [OpenCL program](/en/docs/opencl/clexecute) runtime error | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_INTERNAL | Internal error occurred when [running OpenCL](/en/docs/opencl/clexecute) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_INVALID\_HANDLE | Invalid [OpenCL handle](/en/docs/opencl/clprogramcreate) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_KERNEL\_CREATE | Error creating an [OpenCL kernel](/en/docs/opencl/clkernelcreate) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_NOT\_SUPPORTED | [OpenCL functions](/en/docs/opencl) are not supported on this computer | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_PROGRAM\_CREATE | Error occurred when [compiling an OpenCL program](/en/docs/opencl/clprogramcreate) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_QUEUE\_CREATE | Failed to create a run queue in OpenCL | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_SET\_KERNEL\_PARAMETER | Error occurred when [setting parameters](/en/docs/opencl/clsetkernelarg) for the OpenCL kernel | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_TOO\_LONG\_KERNEL\_NAME | Too long kernel name [(OpenCL kernel)](/en/docs/opencl/clkernelcreate) | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_WRONG\_BUFFER\_OFFSET | Invalid offset in the OpenCL buffer | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_OPENCL\_WRONG\_BUFFER\_SIZE | Invalid size of the OpenCL buffer | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_PLAY\_SOUND\_FAILED | Sound playing failed | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_RESOURCE\_NAME\_DUPLICATED | The names of the [dynamic](/en/docs/common/resourcecreate) and the [static](/en/docs/runtime/resources) resource match | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_RESOURCE\_NAME\_IS\_TOO\_LONG | The resource name exceeds 63 characters | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_RESOURCE\_NOT\_FOUND | Resource with this name has not been found in EX5 | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_RESOURCE\_UNSUPPORTED\_TYPE | Unsupported resource type or its size exceeds 16 Mb | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_SERIES\_ARRAY | Timeseries cannot be used | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_SHORT\_ARRAY\_ONLY | Must be an array of type short | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_SMALL\_ARRAY | Too small array, the starting position is outside the array | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_SMALL\_ASSERIES\_ARRAY | The receiving array is declared as AS\_SERIES, and it is of insufficient size | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRING\_OUT\_OF\_MEMORY | Not enough memory for the string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRING\_RESIZE\_ERROR | Not enough memory for the relocation of string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRING\_SMALL\_LEN | The string length is less than expected | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRING\_TIME\_ERROR | Error converting string to date | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRING\_TOO\_BIGNUMBER | Too large number, more than ULONG\_MAX | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRING\_UNKNOWNTYPE | Unknown data type when converting to a string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRING\_ZEROADDED | 0 added to the string end, a useless operation | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRINGPOS\_OUTOFRANGE | Position outside the string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_STRUCT\_WITHOBJECTS\_ORCLASS | The structure contains objects of strings and/or dynamic arrays and/or structure of such objects and/or classes | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_SUCCESS | The operation completed successfully | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TERMINAL\_WRONG\_PROPERTY | Wrong identifier of the terminal property | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TOO\_LONG\_FILENAME | Too long file name | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TOO\_MANY\_FILES | More than 64 files cannot be opened at the same time | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TOO\_MANY\_FORMATTERS | Amount of format specifiers more than the parameters | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TOO\_MANY\_PARAMETERS | Amount of parameters more than the format specifiers | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TRADE\_DEAL\_NOT\_FOUND | Deal not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TRADE\_DISABLED | Trading by Expert Advisors prohibited | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TRADE\_ORDER\_NOT\_FOUND | Order not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TRADE\_POSITION\_NOT\_FOUND | Position not found | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TRADE\_SEND\_FAILED | Trade request sending failed | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_TRADE\_WRONG\_PROPERTY | Wrong trade property ID | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_USER\_ERROR\_FIRST | [User defined](/en/docs/common/setusererror) errors start with this code | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WEBREQUEST\_CONNECT\_FAILED | Failed to connect to specified URL | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WEBREQUEST\_INVALID\_ADDRESS | Invalid URL | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WEBREQUEST\_REQUEST\_FAILED | HTTP request failed | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WEBREQUEST\_TIMEOUT | Timeout exceeded | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_DIRECTORYNAME | Wrong directory name | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_FILEHANDLE | Wrong file handle | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_FILENAME | Invalid file name | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_FORMATSTRING | Invalid format string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_INTERNAL\_PARAMETER | Wrong parameter in the inner call of the client terminal function | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_STRING\_DATE | Wrong date in the string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_STRING\_OBJECT | Damaged string object | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_STRING\_PARAMETER | Damaged parameter of string type | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_WRONG\_STRING\_TIME | Wrong time in the string | [GetLastError](/en/docs/check/getlasterror) |
| ERR\_ZEROSIZE\_ARRAY | An array of zero length | [GetLastError](/en/docs/check/getlasterror) |
| FILE\_ACCESS\_DATE | Date of the last access to the file | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_ANSI | Strings of ANSI type (one byte symbols). Flag is used in [FileOpen()](/en/docs/files/fileopen). | [FileOpen](/en/docs/files/fileopen) |
| FILE\_BIN | Binary read/write mode (without string to string conversion). Flag is used in [FileOpen()](/en/docs/files/fileopen). | [FileOpen](/en/docs/files/fileopen) |
| FILE\_COMMON | The file path in the common folder of all client terminals \Terminal\Common\Files. Flag is used in [FileOpen()](/en/docs/files/fileopen), [FileCopy()](/en/docs/files/filecopy), [FileMove()](/en/docs/files/filemove) and in [FileIsExist()](/en/docs/files/fileisexist) functions. | [FileOpen](/en/docs/files/fileopen), [FileCopy](/en/docs/files/filecopy), [FileMove](/en/docs/files/filemove), [FileIsExist](/en/docs/files/fileisexist) |
| FILE\_CREATE\_DATE | Date of creation | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_CSV | CSV file (all its elements are converted to strings of the appropriate type, Unicode or ANSI, and separated by separator). Flag is used in [FileOpen()](/en/docs/files/fileopen). | [FileOpen](/en/docs/files/fileopen) |
| FILE\_END | Get the end of file sign | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_EXISTS | Check the existence | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_IS\_ANSI | The file is opened as ANSI (see [FILE\_ANSI](/en/docs/constants/io_constants/fileflags)) | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_IS\_BINARY | The file is opened as a binary file (see [FILE\_BIN](/en/docs/constants/io_constants/fileflags)) | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_IS\_COMMON | The file is opened in a shared folder of all terminals (see [FILE\_COMMON](/en/docs/constants/io_constants/fileflags)) | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_IS\_CSV | The file is opened as CSV (see [FILE\_CSV](/en/docs/constants/io_constants/fileflags)) | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_IS\_READABLE | The opened file is readable (see [FILE\_READ](/en/docs/constants/io_constants/fileflags)) | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_IS\_TEXT | The file is opened as a text file (see [FILE\_TXT](/en/docs/constants/io_constants/fileflags)) | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_IS\_WRITABLE | The opened file is writable (see [FILE\_WRITE](/en/docs/constants/io_constants/fileflags)) | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_LINE\_END | Get the end of line sign | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_MODIFY\_DATE | Date of the last modification | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_POSITION | Position of a pointer in the file | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_READ | File is opened for reading. Flag is used in [FileOpen()](/en/docs/files/fileopen). When opening a file specification of FILE\_WRITE and/or FILE\_READ is required. | [FileOpen](/en/docs/files/fileopen) |
| FILE\_REWRITE | Possibility for the file rewrite using functions [FileCopy()](/en/docs/files/filecopy) and [FileMove()](/en/docs/files/filemove). The file should exist or should be opened for writing, otherwise the file will not be opened. | [FileCopy](/en/docs/files/filecopy), [FileMove](/en/docs/files/filemove) |
| FILE\_SHARE\_READ | Shared access for reading from several programs. Flag is used in [FileOpen()](/en/docs/files/fileopen), but it does not replace the necessity to indicate FILE\_WRITE and/or the FILE\_READ flag when opening a file. | [FileOpen](/en/docs/files/fileopen) |
| FILE\_SHARE\_WRITE | Shared access for writing from several programs. Flag is used in [FileOpen()](/en/docs/files/fileopen), but it does not replace the necessity to indicate FILE\_WRITE and/or the FILE\_READ flag when opening a file. | [FileOpen](/en/docs/files/fileopen) |
| FILE\_SIZE | File size in bytes | [FileGetInteger](/en/docs/files/filegetinteger) |
| FILE\_TXT | Simple text file (the same as csv file, but without taking into account the separators). Flag is used in [FileOpen()](/en/docs/files/fileopen). | [FileOpen](/en/docs/files/fileopen) |
| FILE\_UNICODE | Strings of UNICODE type (two byte symbols). Flag is used in [FileOpen()](/en/docs/files/fileopen). | [FileOpen](/en/docs/files/fileopen) |
| FILE\_WRITE | File is opened for writing. Flag is used in [FileOpen()](/en/docs/files/fileopen). When opening a file specification of FILE\_WRITE and/or FILE\_READ is required. | [FileOpen](/en/docs/files/fileopen) |
| FLT\_DIG | Number of significant decimal digits for float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_EPSILON | Minimal value, which satisfies the condition: 1.0+DBL\_EPSILON != 1.0 (for float type) | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_MANT\_DIG | Bits count in a mantissa for float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_MAX | Maximal value, which can be represented by float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_MAX\_10\_EXP | Maximal decimal value of exponent degree for float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_MAX\_EXP | Maximal binary value of exponent degree for float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_MIN | Minimal positive value, which can be represented by float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_MIN\_10\_EXP | Minimal decimal value of exponent degree for float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FLT\_MIN\_EXP | Minimal binary value of exponent degree for float type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| FRIDAY | Friday | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger), [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote), [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) |
| GANN\_DOWN\_TREND | Line corresponding to the downward trend | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| GANN\_UP\_TREND | Line corresponding to the uptrend line | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| GATORJAW\_LINE | Jaw line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| GATORLIPS\_LINE | Lips line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| GATORTEETH\_LINE | Teeth line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| IDABORT | "Abort" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDCANCEL | "Cancel" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDCONTINUE | "Continue" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDIGNORE | "Ignore" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDNO | "No" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDOK | "OK" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDRETRY | "Retry" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDTRYAGAIN | "Try Again" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IDYES | "Yes" button has been pressed | [MessageBox](/en/docs/common/messagebox) |
| IND\_AC | Accelerator Oscillator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_AD | Accumulation/Distribution | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_ADX | Average Directional Index | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_ADXW | ADX by Welles Wilder | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_ALLIGATOR | Alligator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_AMA | Adaptive Moving Average | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_AO | Awesome Oscillator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_ATR | Average True Range | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_BANDS | Bollinger Bands® | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_BEARS | Bears Power | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_BULLS | Bulls Power | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_BWMFI | Market Facilitation Index | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_CCI | Commodity Channel Index | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_CHAIKIN | Chaikin Oscillator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_CUSTOM | Custom indicator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_DEMA | Double Exponential Moving Average | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_DEMARKER | DeMarker | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_ENVELOPES | Envelopes | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_FORCE | Force Index | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_FRACTALS | Fractals | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_FRAMA | Fractal Adaptive Moving Average | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_GATOR | Gator Oscillator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_ICHIMOKU | Ichimoku Kinko Hyo | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_MA | Moving Average | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_MACD | MACD | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_MFI | Money Flow Index | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_MOMENTUM | Momentum | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_OBV | On Balance Volume | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_OSMA | OsMA | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_RSI | Relative Strength Index | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_RVI | Relative Vigor Index | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_SAR | Parabolic SAR | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_STDDEV | Standard Deviation | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_STOCHASTIC | Stochastic Oscillator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_TEMA | Triple Exponential Moving Average | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_TRIX | Triple Exponential Moving Averages Oscillator | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_VIDYA | Variable Index Dynamic Average | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_VOLUMES | Volumes | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| IND\_WPR | Williams' Percent Range | [IndicatorCreate](/en/docs/series/indicatorcreate), [IndicatorParameters](/en/docs/series/indicatorparameters) |
| INDICATOR\_CALCULATIONS | Auxiliary buffers for intermediate calculations | [SetIndexBuffer](/en/docs/customind/setindexbuffer) |
| INDICATOR\_COLOR\_INDEX | Color | [SetIndexBuffer](/en/docs/customind/setindexbuffer) |
| INDICATOR\_DATA | Data to draw | [SetIndexBuffer](/en/docs/customind/setindexbuffer) |
| INDICATOR\_DIGITS | Accuracy of drawing of indicator values | [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger) |
| INDICATOR\_HEIGHT | Fixed height of the indicator's window (the preprocessor command [#property indicator\_height](/en/docs/basis/preprosessor/compilation)) | [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger) |
| INDICATOR\_LEVELCOLOR | Color of the level line | [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger) |
| INDICATOR\_LEVELS | Number of levels in the indicator window | [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger) |
| INDICATOR\_LEVELSTYLE | Style of the level line | [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger) |
| INDICATOR\_LEVELTEXT | Level description | [IndicatorSetString](/en/docs/customind/indicatorsetstring) |
| INDICATOR\_LEVELVALUE | Level value | [IndicatorSetDouble](/en/docs/customind/indicatorsetdouble) |
| INDICATOR\_LEVELWIDTH | Thickness of the level line | [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger) |
| INDICATOR\_MAXIMUM | Maximum of the indicator window | [IndicatorSetDouble](/en/docs/customind/indicatorsetdouble) |
| INDICATOR\_MINIMUM | Minimum of the indicator window | [IndicatorSetDouble](/en/docs/customind/indicatorsetdouble) |
| INDICATOR\_SHORTNAME | Short indicator name | [IndicatorSetString](/en/docs/customind/indicatorsetstring) |
| INT\_MAX | Maximal value, which can be represented by int type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| INT\_MIN | Minimal value, which can be represented by int type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| INVALID\_HANDLE | Incorrect handle | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| IS\_DEBUG\_MODE | Flag that a mq5-program operates in debug mode | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| IS\_PROFILE\_MODE | Flag that a mq5-program operates in profiling mode | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| KIJUNSEN\_LINE | Kijun-sen line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| LICENSE\_DEMO | A trial version of a paid product from the Market. It works only in the strategy tester | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| LICENSE\_FREE | A free unlimited version | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| LICENSE\_FULL | A purchased licensed version allows at least 5 activations. The number of activations is specified by seller. Seller may increase the allowed number of activations | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| LICENSE\_TIME | A version with a limited term license | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| LONG\_MAX | Maximal value, which can be represented by long type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| LONG\_MIN | Minimal value, which can be represented by long type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| LOWER\_BAND | Lower limit | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| LOWER\_HISTOGRAM | Bottom histogram | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| LOWER\_LINE | Bottom line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| M\_1\_PI | 1/pi | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_2\_PI | 2/pi | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_2\_SQRTPI | 2/sqrt(pi) | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_E | e | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_LN10 | ln(10) | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_LN2 | ln(2) | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_LOG10E | log10(e) | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_LOG2E | log2(e) | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_PI | pi | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_PI\_2 | pi/2 | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_PI\_4 | pi/4 | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_SQRT1\_2 | 1/sqrt(2) | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| M\_SQRT2 | sqrt(2) | [Mathematical Constants](/en/docs/constants/namedconstants/mathsconstants) |
| MAIN\_LINE | Main line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| MB\_ABORTRETRYIGNORE | Message window contains three buttons: Abort, Retry and Ignore | [MessageBox](/en/docs/common/messagebox) |
| MB\_CANCELTRYCONTINUE | Message window contains three buttons: Cancel, Try Again, Continue | [MessageBox](/en/docs/common/messagebox) |
| MB\_DEFBUTTON1 | The first button MB\_DEFBUTTON1 - is default, if the other buttons MB\_DEFBUTTON2, MB\_DEFBUTTON3, or MB\_DEFBUTTON4 are not specified | [MessageBox](/en/docs/common/messagebox) |
| MB\_DEFBUTTON2 | The second button is default | [MessageBox](/en/docs/common/messagebox) |
| MB\_DEFBUTTON3 | The third button is default | [MessageBox](/en/docs/common/messagebox) |
| MB\_DEFBUTTON4 | The fourth button is default | [MessageBox](/en/docs/common/messagebox) |
| MB\_ICONEXCLAMATION, MB\_ICONWARNING | The exclamation/warning sign icon | [MessageBox](/en/docs/common/messagebox) |
| MB\_ICONINFORMATION, MB\_ICONASTERISK | The encircled i sign | [MessageBox](/en/docs/common/messagebox) |
| MB\_ICONQUESTION | The question sign icon | [MessageBox](/en/docs/common/messagebox) |
| MB\_ICONSTOP, MB\_ICONERROR, MB\_ICONHAND | The STOP sign icon | [MessageBox](/en/docs/common/messagebox) |
| MB\_OK | Message window contains only one button: OK. Default | [MessageBox](/en/docs/common/messagebox) |
| MB\_OKCANCEL | Message window contains two buttons: OK and Cancel | [MessageBox](/en/docs/common/messagebox) |
| MB\_RETRYCANCEL | Message window contains two buttons: Retry and Cancel | [MessageBox](/en/docs/common/messagebox) |
| MB\_YESNO | Message window contains two buttons: Yes and No | [MessageBox](/en/docs/common/messagebox) |
| MB\_YESNOCANCEL | Message window contains three buttons: Yes, No and Cancel | [MessageBox](/en/docs/common/messagebox) |
| MINUSDI\_LINE | Line –DI | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| MODE\_EMA | Exponential averaging | [Smoothing Methods](/en/docs/constants/indicatorconstants/enum_ma_method) |
| MODE\_LWMA | Linear-weighted averaging | [Smoothing Methods](/en/docs/constants/indicatorconstants/enum_ma_method) |
| MODE\_SMA | Simple averaging | [Smoothing Methods](/en/docs/constants/indicatorconstants/enum_ma_method) |
| MODE\_SMMA | Smoothed averaging | [Smoothing Methods](/en/docs/constants/indicatorconstants/enum_ma_method) |
| MONDAY | Monday | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger), [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote), [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) |
| MQL\_DEBUG | The flag, that indicates the debug mode | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_DLLS\_ALLOWED | The permission to use DLL for the given executed program | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_FRAME\_MODE | The flag, that indicates the Expert Advisor operating in [gathering optimization result frames mode](/en/docs/event_handlers/ontesterpass) | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_LICENSE\_TYPE | Type of license of the EX5 module. The license refers to the EX5 module, from which a request is made using MQLInfoInteger(MQL\_LICENSE\_TYPE). | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_MEMORY\_LIMIT | Maximum possible amount of dynamic memory for MQL5 program in MB | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_MEMORY\_USED | The memory size used by MQL5 program in MB | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_OPTIMIZATION | The flag, that indicates the optimization process | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_PROFILER | The flag, that indicates the program operating in the code profiling mode | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_PROGRAM\_NAME | Name of the mql5-program executed | [MQLInfoString](/en/docs/check/mqlinfostring) |
| MQL\_PROGRAM\_PATH | Path for the given executed program | [MQLInfoString](/en/docs/check/mqlinfostring) |
| MQL\_PROGRAM\_TYPE | Type of the mql5 program | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_SIGNALS\_ALLOWED | The permission to modify the Signals for the given executed program | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_TESTER | The flag, that indicates the tester process | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_TRADE\_ALLOWED | The [permission to trade](/en/docs/runtime/tradepermission) for the given executed program | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| MQL\_VISUAL\_MODE | The flag, that indicates the visual tester process | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| NULL | Zero for any types | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| OBJ\_ALL\_PERIODS | The object is drawn in all timeframes | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| [OBJ\_ARROW](/en/docs/constants/objectconstants/enum_object/obj_arrow) | Arrow | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_BUY](/en/docs/constants/objectconstants/enum_object/obj_arrow_buy) | Buy Sign | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_CHECK](/en/docs/constants/objectconstants/enum_object/obj_arrow_check) | Check Sign | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_DOWN](/en/docs/constants/objectconstants/enum_object/obj_arrow_down) | Arrow Down | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_LEFT\_PRICE](/en/docs/constants/objectconstants/enum_object/obj_arrow_left_price) | Left Price Label | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_RIGHT\_PRICE](/en/docs/constants/objectconstants/enum_object/obj_arrow_right_price) | Right Price Label | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_SELL](/en/docs/constants/objectconstants/enum_object/obj_arrow_sell) | Sell Sign | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_STOP](/en/docs/constants/objectconstants/enum_object/obj_arrow_stop) | Stop Sign | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_THUMB\_DOWN](/en/docs/constants/objectconstants/enum_object/obj_arrow_thumb_down) | Thumbs Down | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_THUMB\_UP](/en/docs/constants/objectconstants/enum_object/obj_arrow_thumb_up) | Thumbs Up | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROW\_UP](/en/docs/constants/objectconstants/enum_object/obj_arrow_up) | Arrow Up | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ARROWED\_LINE](/en/docs/constants/objectconstants/enum_object/obj_arrowed_line) | Arrowed Line | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_BITMAP](/en/docs/constants/objectconstants/enum_object/obj_bitmap) | Bitmap | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_BITMAP\_LABEL](/en/docs/constants/objectconstants/enum_object/obj_bitmap_label) | Bitmap Label | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_BUTTON](/en/docs/constants/objectconstants/enum_object/obj_button) | Button | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_CHANNEL](/en/docs/constants/objectconstants/enum_object/obj_channel) | Equidistant Channel | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_CHART](/en/docs/constants/objectconstants/enum_object/obj_chart) | Chart | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_CYCLES](/en/docs/constants/objectconstants/enum_object/obj_cycles) | Cycle Lines | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_EDIT](/en/docs/constants/objectconstants/enum_object/obj_edit) | Edit | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ELLIOTWAVE3](/en/docs/constants/objectconstants/enum_object/obj_elliotwave3) | Elliott Correction Wave | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ELLIOTWAVE5](/en/docs/constants/objectconstants/enum_object/obj_elliotwave5) | Elliott Motive Wave | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_ELLIPSE](/en/docs/constants/objectconstants/enum_object/obj_ellipse) | Ellipse | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_EVENT](/en/docs/constants/objectconstants/enum_object/obj_event) | The "Event" object corresponding to an event in the economic calendar | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_EXPANSION](/en/docs/constants/objectconstants/enum_object/obj_expansion) | Fibonacci Expansion | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_FIBO](/en/docs/constants/objectconstants/enum_object/obj_fibo) | Fibonacci Retracement | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_FIBOARC](/en/docs/constants/objectconstants/enum_object/obj_fiboarc) | Fibonacci Arcs | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_FIBOCHANNEL](/en/docs/constants/objectconstants/enum_object/obj_fibochannel) | Fibonacci Channel | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_FIBOFAN](/en/docs/constants/objectconstants/enum_object/obj_fibofan) | Fibonacci Fan | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_FIBOTIMES](/en/docs/constants/objectconstants/enum_object/obj_fibotimes) | Fibonacci Time Zones | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_GANNFAN](/en/docs/constants/objectconstants/enum_object/obj_gannfan) | Gann Fan | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_GANNGRID](/en/docs/constants/objectconstants/enum_object/obj_ganngrid) | Gann Grid | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_GANNLINE](/en/docs/constants/objectconstants/enum_object/obj_gannline) | Gann Line | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_HLINE](/en/docs/constants/objectconstants/enum_object/obj_hline) | Horizontal Line | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_LABEL](/en/docs/constants/objectconstants/enum_object/obj_label) | Label | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| OBJ\_NO\_PERIODS | The object is not drawn in all timeframes | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_D1 | The object is drawn in day charts | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_H1 | The object is drawn in 1-hour chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_H12 | The object is drawn in 12-hour chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_H2 | The object is drawn in 2-hour chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_H3 | The object is drawn in 3-hour chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_H4 | The object is drawn in 4-hour chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_H6 | The object is drawn in 6-hour chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_H8 | The object is drawn in 8-hour chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M1 | The object is drawn in 1-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M10 | The object is drawn in 10-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M12 | The object is drawn in 12-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M15 | The object is drawn in 15-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M2 | The object is drawn in 2-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M20 | The object is drawn in 20-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M3 | The object is drawn in 3-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M30 | The object is drawn in 30-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M4 | The object is drawn in 4-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M5 | The object is drawn in 5-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_M6 | The object is drawn in 6-minute chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_MN1 | The object is drawn in month charts | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJ\_PERIOD\_W1 | The object is drawn in week charts | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| [OBJ\_PITCHFORK](/en/docs/constants/objectconstants/enum_object/obj_pitchfork) | Andrews’ Pitchfork | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_RECTANGLE](/en/docs/constants/objectconstants/enum_object/obj_rectangle) | Rectangle | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_RECTANGLE\_LABEL](/en/docs/constants/objectconstants/enum_object/obj_rectangle_label) | The "Rectangle label" object for creating and designing the custom graphical interface. | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_REGRESSION](/en/docs/constants/objectconstants/enum_object/obj_regression) | Linear Regression Channel | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_STDDEVCHANNEL](/en/docs/constants/objectconstants/enum_object/obj_stddevchannel) | Standard Deviation Channel | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_TEXT](/en/docs/constants/objectconstants/enum_object/obj_text) | Text | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_TREND](/en/docs/constants/objectconstants/enum_object/obj_trend) | Trend Line | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_TRENDBYANGLE](/en/docs/constants/objectconstants/enum_object/obj_trendbyangle) | Trend Line By Angle | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_TRIANGLE](/en/docs/constants/objectconstants/enum_object/obj_triangle) | Triangle | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| [OBJ\_VLINE](/en/docs/constants/objectconstants/enum_object/obj_vline) | Vertical Line | [Object Types](/en/docs/constants/objectconstants/enum_object) |
| OBJPROP\_ALIGN | Horizontal text alignment in the "Edit" object (OBJ\_EDIT) | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_ANCHOR | Location of the anchor point of a graphical object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_ANGLE | Angle.  For the objects with no angle specified, created from a program, the value is equal to [EMPTY\_VALUE](/en/docs/constants/namedconstants/otherconstants) | [ObjectSetDouble](/en/docs/objects/objectsetdouble), [ObjectGetDouble](/en/docs/objects/objectgetdouble) |
| OBJPROP\_ARROWCODE | Arrow code for the Arrow object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_BACK | Object in the background | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_BGCOLOR | The background color for  OBJ\_EDIT, OBJ\_BUTTON, OBJ\_RECTANGLE\_LABEL | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_BMPFILE | The name of BMP-file for Bitmap Label. See also [Resources](/en/docs/runtime/resources) | [ObjectSetString](/en/docs/objects/objectsetstring), [ObjectGetString](/en/docs/objects/objectgetstring) |
| OBJPROP\_BORDER\_COLOR | Border color for the OBJ\_EDIT and OBJ\_BUTTON objects | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_BORDER\_TYPE | Border type for the "Rectangle label" object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_CHART\_ID | ID of the "Chart" object ([OBJ\_CHART](/en/docs/constants/objectconstants/enum_object)). It allows working with the properties of this object like with a normal chart using the functions described in [Chart Operations](/en/docs/chart_operations), but there some [exceptions](/en/docs/constants/objectconstants/enum_object_property#objprop_chart_id_exception). | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_CHART\_SCALE | The scale for the Chart object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_COLOR | Color | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_CORNER | The corner of the chart to link a graphical object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_CREATETIME | Time of object creation | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_DATE\_SCALE | Displaying the time scale for the Chart object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_DEGREE | Level of the Elliott Wave Marking | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_DEVIATION | Deviation for the Standard Deviation Channel | [ObjectSetDouble](/en/docs/objects/objectsetdouble), [ObjectGetDouble](/en/docs/objects/objectgetdouble) |
| OBJPROP\_DIRECTION | Trend of the Gann object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_DRAWLINES | Displaying lines for marking the Elliott Wave | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_ELLIPSE | Showing the full ellipse of the Fibonacci Arc object ([OBJ\_FIBOARC](/en/docs/constants/objectconstants/enum_object)) | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_FILL | Fill an object with color (for OBJ\_RECTANGLE, OBJ\_TRIANGLE, OBJ\_ELLIPSE, OBJ\_CHANNEL, OBJ\_STDDEVCHANNEL, OBJ\_REGRESSION) | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_FONT | Font | [ObjectSetString](/en/docs/objects/objectsetstring), [ObjectGetString](/en/docs/objects/objectgetstring) |
| OBJPROP\_FONTSIZE | Font size | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_HIDDEN | Prohibit showing of the name of a graphical object in the list of objects from the terminal menu "Charts" - "Objects" - "List of objects". The true value allows to hide an object from the list. By default, true is set to the objects that display calendar events, trading history and to the objects [created from MQL5 programs](/en/docs/objects/objectcreate). To see such [graphical objects](/en/docs/constants/objectconstants/enum_object) and access their properties, click on the "All" button in the "List of objects" window. | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_LEVELCOLOR | Color of the line-level | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_LEVELS | Number of levels | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_LEVELSTYLE | Style of the line-level | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_LEVELTEXT | Level description | [ObjectSetString](/en/docs/objects/objectsetstring), [ObjectGetString](/en/docs/objects/objectgetstring) |
| OBJPROP\_LEVELVALUE | Level value | [ObjectSetDouble](/en/docs/objects/objectsetdouble), [ObjectGetDouble](/en/docs/objects/objectgetdouble) |
| OBJPROP\_LEVELWIDTH | Thickness of the line-level | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_NAME | Object name | [ObjectSetString](/en/docs/objects/objectsetstring), [ObjectGetString](/en/docs/objects/objectgetstring) |
| OBJPROP\_PERIOD | Timeframe for the Chart object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_PRICE | Price coordinate | [ObjectSetDouble](/en/docs/objects/objectsetdouble), [ObjectGetDouble](/en/docs/objects/objectgetdouble) |
| OBJPROP\_PRICE\_SCALE | Displaying the price scale for the Chart object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_RAY | A vertical line goes through all the windows of a chart | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_RAY\_LEFT | Ray goes to the left | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_RAY\_RIGHT | Ray goes to the right | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_READONLY | Ability to edit text in the Edit object | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_SCALE | Scale (properties of Gann objects and Fibonacci Arcs) | [ObjectSetDouble](/en/docs/objects/objectsetdouble), [ObjectGetDouble](/en/docs/objects/objectgetdouble) |
| OBJPROP\_SELECTABLE | Object availability | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_SELECTED | Object is selected | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_STATE | Button state (pressed / depressed) | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_STYLE | Style | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_SYMBOL | Symbol for the Chart object | [ObjectSetString](/en/docs/objects/objectsetstring), [ObjectGetString](/en/docs/objects/objectgetstring) |
| OBJPROP\_TEXT | Description of the object (the text contained in the object) | [ObjectSetString](/en/docs/objects/objectsetstring), [ObjectGetString](/en/docs/objects/objectgetstring) |
| OBJPROP\_TIME | Time coordinate | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_TIMEFRAMES | Visibility of an object at timeframes | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_TOOLTIP | The text of a tooltip. If the property is not set, then the tooltip generated automatically by the terminal is shown. A tooltip can be disabled by setting the "\n" (line feed) value to it | [ObjectSetString](/en/docs/objects/objectsetstring), [ObjectGetString](/en/docs/objects/objectgetstring) |
| OBJPROP\_TYPE | Object type | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_WIDTH | Line thickness | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_XDISTANCE | The distance in pixels along the X axis from the binding corner (see [note](/en/docs/constants/objectconstants/enum_object_property#distance_fixedsize)) | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_XOFFSET | The X coordinate of the upper left corner of the [rectangular visible area](/en/docs/constants/objectconstants/enum_object_property#visual_rectangle) in the graphical objects "Bitmap Label" and "Bitmap" (OBJ\_BITMAP\_LABEL and OBJ\_BITMAP). The value is set in pixels relative to the upper left corner of the original image. | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_XSIZE | The object's width along the X axis in pixels. Specified for  OBJ\_LABEL (read only), OBJ\_BUTTON, OBJ\_CHART, OBJ\_BITMAP, OBJ\_BITMAP\_LABEL, OBJ\_EDIT, OBJ\_RECTANGLE\_LABEL objects. | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_YDISTANCE | The distance in pixels along the Y axis from the binding corner (see [note](/en/docs/constants/objectconstants/enum_object_property#distance_fixedsize)) | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_YOFFSET | The Y coordinate of the upper left corner of the [rectangular visible area](/en/docs/constants/objectconstants/enum_object_property#visual_rectangle) in the graphical objects "Bitmap Label" and "Bitmap" (OBJ\_BITMAP\_LABEL and OBJ\_BITMAP). The value is set in pixels relative to the upper left corner of the original image. | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_YSIZE | The object's height along the Y axis in pixels. Specified for  OBJ\_LABEL (read only), OBJ\_BUTTON, OBJ\_CHART, OBJ\_BITMAP, OBJ\_BITMAP\_LABEL, OBJ\_EDIT, OBJ\_RECTANGLE\_LABEL objects. | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| OBJPROP\_ZORDER | Priority of a graphical object for receiving events of clicking on a chart ([CHARTEVENT\_CLICK](/en/docs/constants/chartconstants/enum_chartevents)). The default zero value is set when creating an object; the priority can be increased if necessary. When applying objects one over another, only one of them with the highest priority will receive the CHARTEVENT\_CLICK event. | [ObjectSetInteger](/en/docs/objects/objectsetinteger), [ObjectGetInteger](/en/docs/objects/objectgetinteger) |
| ORDER\_COMMENT | Order comment | [OrderGetString](/en/docs/trading/ordergetstring), [HistoryOrderGetString](/en/docs/trading/historyordergetstring) |
| ORDER\_FILLING\_FOK | This filling policy means that an order can be filled only in the specified amount. If the necessary amount of a financial instrument is currently unavailable in the market, the order will not be executed. The required volume can be filled using several offers available on the market at the moment. | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_FILLING\_IOC | This mode means that a trader agrees to execute a deal with the volume maximally available in the market within that indicated in the order. In case the the entire volume of an order cannot be filled, the available volume of it will be filled, and the remaining volume will be canceled. | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_FILLING\_RETURN | This policy is used only for market orders (ORDER\_TYPE\_BUY and ORDER\_TYPE\_SELL), limit and stop limit orders (ORDER\_TYPE\_BUY\_LIMIT, ORDER\_TYPE\_SELL\_LIMIT, ORDER\_TYPE\_BUY\_STOP\_LIMIT and ORDER\_TYPE\_SELL\_STOP\_LIMIT ) and only for the symbols with Market or Exchange [execution](/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_trade_execution). In case of partial filling a market or limit order with remaining volume is not canceled but processed further. For the activation of the ORDER\_TYPE\_BUY\_STOP\_LIMIT and ORDER\_TYPE\_SELL\_STOP\_LIMIT orders, a corresponding limit order ORDER\_TYPE\_BUY\_LIMIT/ORDER\_TYPE\_SELL\_LIMIT with the ORDER\_FILLING\_RETURN execution type is created. | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_MAGIC | ID of an Expert Advisor that has placed the order (designed to ensure that each Expert Advisor places its own unique number) | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_POSITION\_ID | [Position identifier](/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer) that is set to an order as soon as it is executed. Each executed order results in a [deal](/en/docs/constants/tradingconstants/dealproperties) that opens or modifies an already existing position. The identifier of exactly this position is set to the executed order at this moment. | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_PRICE\_CURRENT | The current price of the order symbol | [OrderGetDouble](/en/docs/trading/ordergetdouble), [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) |
| ORDER\_PRICE\_OPEN | Price specified in the order | [OrderGetDouble](/en/docs/trading/ordergetdouble), [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) |
| ORDER\_PRICE\_STOPLIMIT | The Limit order price for the StopLimit order | [OrderGetDouble](/en/docs/trading/ordergetdouble), [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) |
| ORDER\_SL | Stop Loss value | [OrderGetDouble](/en/docs/trading/ordergetdouble), [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) |
| ORDER\_STATE | Order state | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_CANCELED | Order canceled by client | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_EXPIRED | Order expired | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_FILLED | Order fully executed | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_PARTIAL | Order partially executed | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_PLACED | Order accepted | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_REJECTED | Order rejected | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_REQUEST\_ADD | Order is being registered (placing to the trading system) | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_REQUEST\_CANCEL | Order is being deleted (deleting from the trading system) | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_REQUEST\_MODIFY | Order is being modified (changing its parameters) | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_STATE\_STARTED | Order checked, but not yet accepted by broker | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_SYMBOL | Symbol of the order | [OrderGetString](/en/docs/trading/ordergetstring), [HistoryOrderGetString](/en/docs/trading/historyordergetstring) |
| ORDER\_TIME\_DAY | Good till current trade day order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_DONE | Order execution or cancellation time | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_DONE\_MSC | Order execution/cancellation time in milliseconds since 01.01.1970 | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_EXPIRATION | Order expiration time | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_GTC | Good till cancel order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_SETUP | Order setup time | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_SETUP\_MSC | The time of placing an order for execution in milliseconds since 01.01.1970 | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_SPECIFIED | Good till expired order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TIME\_SPECIFIED\_DAY | The order will be effective till 23:59:59 of the specified day. If this time is outside a trading session, the order expires in the nearest trading time. | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TP | Take Profit value | [OrderGetDouble](/en/docs/trading/ordergetdouble), [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) |
| ORDER\_TYPE | Order type | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_BUY | Market Buy order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_BUY\_LIMIT | Buy Limit pending order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_BUY\_STOP | Buy Stop pending order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_BUY\_STOP\_LIMIT | Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_FILLING | Order filling type | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_SELL | Market Sell order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_SELL\_LIMIT | Sell Limit pending order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_SELL\_STOP | Sell Stop pending order | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_SELL\_STOP\_LIMIT | Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_TYPE\_TIME | Order lifetime | [OrderGetInteger](/en/docs/trading/ordergetinteger), [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) |
| ORDER\_VOLUME\_CURRENT | Order current volume | [OrderGetDouble](/en/docs/trading/ordergetdouble), [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) |
| ORDER\_VOLUME\_INITIAL | Order initial volume | [OrderGetDouble](/en/docs/trading/ordergetdouble), [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) |
| PERIOD\_CURRENT | Current timeframe | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_D1 | 1 day | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_H1 | 1 hour | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_H12 | 12 hours | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_H2 | 2 hours | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_H3 | 3 hours | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_H4 | 4 hours | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_H6 | 6 hours | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_H8 | 8 hours | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M1 | 1 minute | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M10 | 10 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M12 | 12 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M15 | 15 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M2 | 2 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M20 | 20 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M3 | 3 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M30 | 30 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M4 | 4 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M5 | 5 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_M6 | 6 minutes | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_MN1 | 1 month | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PERIOD\_W1 | 1 week | [Chart Timeframes](/en/docs/constants/chartconstants/enum_timeframes) |
| PLOT\_ARROW | Arrow code for style DRAW\_ARROW | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_ARROW\_SHIFT | Vertical shift of arrows for style DRAW\_ARROW | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_COLOR\_INDEXES | The number of colors | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_DRAW\_BEGIN | Number of initial bars without drawing and values in the DataWindow | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_DRAW\_TYPE | Type of graphical construction | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_EMPTY\_VALUE | An empty value for plotting, for which there is no drawing | [PlotIndexSetDouble](/en/docs/customind/plotindexsetdouble) |
| PLOT\_LABEL | The name of the indicator graphical series to display in the DataWindow. When working with complex graphical styles requiring several indicator buffers for display, the names for each buffer can be specified using ";" as a separator. Sample code is shown in [DRAW\_CANDLES](/en/docs/customind/indicators_examples/draw_candles) | [PlotIndexSetString](/en/docs/customind/plotindexsetstring) |
| PLOT\_LINE\_COLOR | The index of a buffer containing the drawing color | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_LINE\_STYLE | Drawing line style | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_LINE\_WIDTH | The thickness of the drawing line | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_SHIFT | Shift of indicator plotting along the time axis in bars | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLOT\_SHOW\_DATA | Sign of display of construction values in the DataWindow | [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger), [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) |
| PLUSDI\_LINE | Line +DI | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| POINTER\_AUTOMATIC | Pointer of any objects created automatically (not using new()) | [CheckPointer](/en/docs/common/checkpointer) |
| POINTER\_DYNAMIC | Pointer of the object created by the [new()](/en/docs/basis/operators/newoperator) operator | [CheckPointer](/en/docs/common/checkpointer) |
| POINTER\_INVALID | Incorrect pointer | [CheckPointer](/en/docs/common/checkpointer) |
| POSITION\_COMMENT | Position comment | [PositionGetString](/en/docs/trading/positiongetstring) |
| POSITION\_COMMISSION | Commission | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| POSITION\_IDENTIFIER | Position identifier is a unique number that is assigned to every newly opened position and doesn't change during the entire lifetime of the position. Position turnover doesn't change its identifier. | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_MAGIC | Position magic number (see [ORDER\_MAGIC](/en/docs/constants/tradingconstants/orderproperties)) | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_PRICE\_CURRENT | Current price of the position symbol | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| POSITION\_PRICE\_OPEN | Position open price | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| POSITION\_PROFIT | Current profit | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| POSITION\_SL | Stop Loss level of opened position | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| POSITION\_SWAP | Cumulative swap | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| POSITION\_SYMBOL | Symbol of the position | [PositionGetString](/en/docs/trading/positiongetstring) |
| POSITION\_TIME | Position open time | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_TIME\_MSC | Position opening time in milliseconds since 01.01.1970 | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_TIME\_UPDATE | Position changing time in seconds since 01.01.1970 | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_TIME\_UPDATE\_MSC | Position changing time in milliseconds since 01.01.1970 | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_TP | Take Profit level of opened position | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| POSITION\_TYPE | Position type | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_TYPE\_BUY | Buy | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_TYPE\_SELL | Sell | [PositionGetInteger](/en/docs/trading/positiongetinteger) |
| POSITION\_VOLUME | Position volume | [PositionGetDouble](/en/docs/trading/positiongetdouble) |
| PRICE\_CLOSE | Close price | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| PRICE\_HIGH | The maximum price for the period | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| PRICE\_LOW | The minimum price for the period | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| PRICE\_MEDIAN | Median price, (high + low)/2 | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| PRICE\_OPEN | Open price | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| PRICE\_TYPICAL | Typical price, (high + low + close)/3 | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| PRICE\_WEIGHTED | Average price, (high + low + close + close)/4 | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| PROGRAM\_EXPERT | Expert | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| PROGRAM\_INDICATOR | Indicator | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| PROGRAM\_SCRIPT | Script | [MQLInfoInteger](/en/docs/check/mqlinfointeger) |
| REASON\_ACCOUNT | Another account has been activated or reconnection to the trade server has occurred due to changes in the account settings | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_CHARTCHANGE | Symbol or chart period has been changed | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_CHARTCLOSE | Chart has been closed | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_CLOSE | Terminal has been closed | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_INITFAILED | This value means that [OnInit()](/en/docs/event_handlers/oninit) handler has returned a nonzero value | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_PARAMETERS | Input parameters have been changed by a user | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_PROGRAM | Expert Advisor terminated its operation by calling the [ExpertRemove()](/en/docs/common/expertremove) function | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_RECOMPILE | Program has been recompiled | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_REMOVE | Program has been deleted from the chart | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| REASON\_TEMPLATE | A new template has been applied | [UninitializeReason](/en/docs/check/uninitializereason), [OnDeinit](/en/docs/event_handlers/ondeinit) |
| SATURDAY | Saturday | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger), [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote), [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) |
| SEEK\_CUR | Current position of a file pointer | [FileSeek](/en/docs/files/fileseek) |
| SEEK\_END | File end | [FileSeek](/en/docs/files/fileseek) |
| SEEK\_SET | File beginning | [FileSeek](/en/docs/files/fileseek) |
| SENKOUSPANA\_LINE | Senkou Span A line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| SENKOUSPANB\_LINE | Senkou Span B line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| SERIES\_BARS\_COUNT | Bars count for the symbol-period for the current moment | [SeriesInfoInteger](/en/docs/series/seriesinfointeger) |
| SERIES\_FIRSTDATE | The very first date for the symbol-period for the current moment | [SeriesInfoInteger](/en/docs/series/seriesinfointeger) |
| SERIES\_LASTBAR\_DATE | Open time of the last bar of the symbol-period | [SeriesInfoInteger](/en/docs/series/seriesinfointeger) |
| SERIES\_SERVER\_FIRSTDATE | The very first date in the history of the symbol on the server regardless of the timeframe | [SeriesInfoInteger](/en/docs/series/seriesinfointeger) |
| SERIES\_SYNCHRONIZED | Symbol/period data synchronization flag for the current moment | [SeriesInfoInteger](/en/docs/series/seriesinfointeger) |
| SERIES\_TERMINAL\_FIRSTDATE | The very first date in the history of the symbol in the client terminal, regardless of the timeframe | [SeriesInfoInteger](/en/docs/series/seriesinfointeger) |
| SHORT\_MAX | Maximal value, which can be represented by short type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| SHORT\_MIN | Minimal value, which can be represented by short type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| SIGNAL\_BASE\_AUTHOR\_LOGIN | Author login | [SignalBaseGetString](/en/docs/signals/signalbasegetstring) |
| SIGNAL\_BASE\_BALANCE | Account balance | [SignalBaseGetDouble](/en/docs/signals/signalbasegetdouble) |
| SIGNAL\_BASE\_BROKER | Broker name (company) | [SignalBaseGetString](/en/docs/signals/signalbasegetstring) |
| SIGNAL\_BASE\_BROKER\_SERVER | Broker server | [SignalBaseGetString](/en/docs/signals/signalbasegetstring) |
| SIGNAL\_BASE\_CURRENCY | Signal base currency | [SignalBaseGetString](/en/docs/signals/signalbasegetstring) |
| SIGNAL\_BASE\_DATE\_PUBLISHED | Publication date (date when it become available for subscription) | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_DATE\_STARTED | Monitoring starting date | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_EQUITY | Account equity | [SignalBaseGetDouble](/en/docs/signals/signalbasegetdouble) |
| SIGNAL\_BASE\_GAIN | Account gain | [SignalBaseGetDouble](/en/docs/signals/signalbasegetdouble) |
| SIGNAL\_BASE\_ID | Signal ID | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_LEVERAGE | Account leverage | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_MAX\_DRAWDOWN | Account maximum drawdown | [SignalBaseGetDouble](/en/docs/signals/signalbasegetdouble) |
| SIGNAL\_BASE\_NAME | Signal name | [SignalBaseGetString](/en/docs/signals/signalbasegetstring) |
| SIGNAL\_BASE\_PIPS | Profit in pips | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_PRICE | Signal subscription price | [SignalBaseGetDouble](/en/docs/signals/signalbasegetdouble) |
| SIGNAL\_BASE\_RATING | Position in rating | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_ROI | Return on Investment (%) | [SignalBaseGetDouble](/en/docs/signals/signalbasegetdouble) |
| SIGNAL\_BASE\_SUBSCRIBERS | Number of subscribers | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_TRADE\_MODE | Account type (0-real, 1-demo, 2-contest) | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_BASE\_TRADES | Number of trades | [SignalBaseGetInteger](/en/docs/signals/signalbasegetinteger) |
| SIGNAL\_INFO\_CONFIRMATIONS\_DISABLED | The flag enables synchronization without confirmation dialog | [SignalInfoGetInteger](/en/docs/signals/signalinfogetinteger), [SignalInfoSetInteger](/en/docs/signals/signalinfosetinteger) |
| SIGNAL\_INFO\_COPY\_SLTP | Copy Stop Loss and Take Profit flag | [SignalInfoGetInteger](/en/docs/signals/signalinfogetinteger), [SignalInfoSetInteger](/en/docs/signals/signalinfosetinteger) |
| SIGNAL\_INFO\_DEPOSIT\_PERCENT | Deposit percent (%) | [SignalInfoGetInteger](/en/docs/signals/signalinfogetinteger), [SignalInfoSetInteger](/en/docs/signals/signalinfosetinteger) |
| SIGNAL\_INFO\_EQUITY\_LIMIT | Equity limit | [SignalInfoGetDouble](/en/docs/signals/signalinfogetdouble), [SignalInfoSetDouble](/en/docs/signals/signalinfosetdouble) |
| SIGNAL\_INFO\_ID | Signal id, r/o | [SignalInfoGetInteger](/en/docs/signals/signalinfogetinteger), [SignalInfoSetInteger](/en/docs/signals/signalinfosetinteger) |
| SIGNAL\_INFO\_NAME | Signal name, r/o | [SignalInfoGetString](/en/docs/signals/signalinfogetstring) |
| SIGNAL\_INFO\_SLIPPAGE | Slippage (used when placing market orders in synchronization of positions and copying of trades) | [SignalInfoGetDouble](/en/docs/signals/signalinfogetdouble), [SignalInfoSetDouble](/en/docs/signals/signalinfosetdouble) |
| SIGNAL\_INFO\_SUBSCRIPTION\_ENABLED | "Copy trades by subscription" permission flag | [SignalInfoGetInteger](/en/docs/signals/signalinfogetinteger), [SignalInfoSetInteger](/en/docs/signals/signalinfosetinteger) |
| SIGNAL\_INFO\_TERMS\_AGREE | "Agree to terms of use of Signals service" flag, r/o | [SignalInfoGetInteger](/en/docs/signals/signalinfogetinteger), [SignalInfoSetInteger](/en/docs/signals/signalinfosetinteger) |
| SIGNAL\_INFO\_VOLUME\_PERCENT | Maximum percent of deposit used (%), r/o | [SignalInfoGetDouble](/en/docs/signals/signalinfogetdouble), [SignalInfoSetDouble](/en/docs/signals/signalinfosetdouble) |
| SIGNAL\_LINE | Signal line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| STAT\_BALANCE\_DD | Maximum balance drawdown in monetary terms. In the process of trading, a balance may have numerous drawdowns; here the largest value is taken | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_BALANCE\_DD\_RELATIVE | Balance drawdown in monetary terms that was recorded at the moment of the maximum balance drawdown as a percentage (STAT\_BALANCE\_DDREL\_PERCENT). | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_BALANCE\_DDREL\_PERCENT | Maximum balance drawdown as a percentage. In the process of trading, a balance may have numerous drawdowns, for each of which the relative drawdown value in percents is calculated. The greatest value is returned | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_BALANCEDD\_PERCENT | Balance drawdown as a percentage that was recorded at the moment of the maximum balance drawdown in monetary terms (STAT\_BALANCE\_DD). | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_BALANCEMIN | Minimum balance value | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_CONLOSSMAX | Maximum loss in a series of losing trades. The value is less than or equal to zero | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_CONLOSSMAX\_TRADES | The number of trades that have formed STAT\_CONLOSSMAX (maximum loss in a series of losing trades) | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_CONPROFITMAX | Maximum profit in a series of profitable trades. The value is greater than or equal to zero | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_CONPROFITMAX\_TRADES | The number of trades that have formed STAT\_CONPROFITMAX (maximum profit in a series of profitable trades) | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_CUSTOM\_ONTESTER | The value of the calculated custom optimization criterion returned by the [OnTester()](/en/docs/event_handlers/ontester) function | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_DEALS | The number of deals | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_EQUITY\_DD | Maximum equity drawdown in monetary terms. In the process of trading, numerous drawdowns may appear on the equity; here the largest value is taken | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_EQUITY\_DD\_RELATIVE | Equity drawdown in monetary terms that was recorded at the moment of the maximum equity drawdown in percent (STAT\_EQUITY\_DDREL\_PERCENT). | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_EQUITY\_DDREL\_PERCENT | Maximum equity drawdown as a percentage. In the process of trading, an equity may have numerous drawdowns, for each of which the relative drawdown value in percents is calculated. The greatest value is returned | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_EQUITYDD\_PERCENT | Drawdown in percent that was recorded at the moment of the maximum equity drawdown in monetary terms (STAT\_EQUITY\_DD). | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_EQUITYMIN | Minimum equity value | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_EXPECTED\_PAYOFF | Expected payoff | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_GROSS\_LOSS | Total loss, the sum of all negative trades. The value is less than or equal to zero | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_GROSS\_PROFIT | Total profit, the sum of all profitable (positive) trades. The value is greater than or equal to zero | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_INITIAL\_DEPOSIT | The value of the initial deposit | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_LONG\_TRADES | Long trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_LOSS\_TRADES | Losing trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_LOSSTRADES\_AVGCON | Average length of a losing series of trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_MAX\_CONLOSS\_TRADES | The number of trades in the longest series of losing trades STAT\_MAX\_CONLOSSES | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_MAX\_CONLOSSES | The total loss of the longest series of losing trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_MAX\_CONPROFIT\_TRADES | The number of trades in the longest series of profitable trades STAT\_MAX\_CONWINS | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_MAX\_CONWINS | The total profit of the longest series of profitable trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_MAX\_LOSSTRADE | Maximum loss – the lowest value of all losing trades. The value is less than or equal to zero | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_MAX\_PROFITTRADE | Maximum profit – the largest value of all profitable trades. The value is greater than or equal to zero | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_MIN\_MARGINLEVEL | Minimum value of the margin level | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_PROFIT | Net profit after testing, the sum of STAT\_GROSS\_PROFIT and STAT\_GROSS\_LOSS (STAT\_GROSS\_LOSS is always less than or equal to zero) | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_PROFIT\_FACTOR | Profit factor, equal to  the ratio of STAT\_GROSS\_PROFIT/STAT\_GROSS\_LOSS. If STAT\_GROSS\_LOSS=0, the profit factor is equal to [DBL\_MAX](/en/docs/constants/namedconstants/typeconstants) | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_PROFIT\_LONGTRADES | Profitable long trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_PROFIT\_SHORTTRADES | Profitable short trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_PROFIT\_TRADES | Profitable trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_PROFITTRADES\_AVGCON | Average length of a profitable series of trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_RECOVERY\_FACTOR | Recovery factor, equal to the ratio of STAT\_PROFIT/STAT\_BALANCE\_DD | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_SHARPE\_RATIO | Sharpe ratio | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_SHORT\_TRADES | Short trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_TRADES | The number of trades | [TesterStatistics](/en/docs/common/testerstatistics) |
| STAT\_WITHDRAWAL | Money withdrawn from an account | [TesterStatistics](/en/docs/common/testerstatistics) |
| STO\_CLOSECLOSE | Calculation is based on Close/Close prices | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| STO\_LOWHIGH | Calculation is based on Low/High prices | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| STYLE\_DASH | Broken line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) |
| STYLE\_DASHDOT | Dash-dot line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) |
| STYLE\_DASHDOTDOT | Dash - two points | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) |
| STYLE\_DOT | Dotted line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) |
| STYLE\_SOLID | Solid line | [Drawing Styles](/en/docs/constants/indicatorconstants/drawstyles#enum_line_style) |
| SUNDAY | Sunday | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger), [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote), [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) |
| SYMBOL\_ASK | Ask - best buy offer | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_ASKHIGH | Maximal Ask of the day | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_ASKLOW | Minimal Ask of the day | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_BANK | Feeder of the current quote | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_BASIS | The underlying asset of a derivative | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_BID | Bid - best sell offer | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_BIDHIGH | Maximal Bid of the day | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_BIDLOW | Minimal Bid of the day | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_CALC\_MODE\_CFD | CFD mode - calculation of margin and profit for CFD | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_CFDINDEX | CFD index mode - calculation of margin and profit for CFD by indexes | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_CFDLEVERAGE | CFD Leverage mode - calculation of margin and profit for CFD at leverage trading | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_EXCH\_FUTURES | Futures mode –  calculation of margin and profit for trading futures contracts on a stock exchange | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_EXCH\_FUTURES\_FORTS | FORTS Futures mode –  calculation of margin and profit for trading futures contracts on FORTS. | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_EXCH\_STOCKS | Exchange mode – calculation of margin and profit for trading securities on a stock exchange | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_FOREX | Forex mode - calculation of profit and margin for Forex | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_FUTURES | Futures mode - calculation of margin and profit for futures | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CALC\_MODE\_SERV\_COLLATERAL | Collateral mode - a symbol is used as a non-tradable asset on a trading account. | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_CURRENCY\_BASE | Basic currency of a symbol | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_CURRENCY\_MARGIN | Margin currency | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_CURRENCY\_PROFIT | Profit currency | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_DESCRIPTION | Symbol description | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_DIGITS | Digits after a decimal point | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_EXPIRATION\_DAY | The order is valid till the end of the day | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_EXPIRATION\_GTC | The order is valid during the unlimited time period, until it is explicitly canceled | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_EXPIRATION\_MODE | Flags of allowed order [expiration modes](/en/docs/constants/environment_state/marketinfoconstants#symbol_expiration_mode) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_EXPIRATION\_SPECIFIED | The expiration time is specified in the order | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_EXPIRATION\_SPECIFIED\_DAY | The expiration date is specified in the order | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_EXPIRATION\_TIME | Date of the symbol trade end (usually used for futures) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_FILLING\_FOK | This policy means that a deal can be executed only with the specified volume. If the necessary amount of a financial instrument is currently unavailable in the market, the order will not be executed. The required volume can be filled using several offers available on the market at the moment. | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_FILLING\_IOC | In this case a trader agrees to execute a deal with the volume maximally available in the market within that indicated in the order. In case the order cannot be filled completely, the available volume of the order will be filled, and the remaining volume will be canceled. The possibility of using IOC orders is determined at the trade server. | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_FILLING\_MODE | Flags of allowed order [filling modes](/en/docs/constants/environment_state/marketinfoconstants#symbol_filling_mode) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_ISIN | The name of a symbol in the ISIN system (International Securities Identification Number). The International Securities Identification Number is a 12-digit alphanumeric code that uniquely identifies a security. The presence of this symbol property is determined on the side of a trade server. | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_LAST | Price of the last deal | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_LASTHIGH | Maximal Last of the day | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_LASTLOW | Minimal Last of the day | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_MARGIN\_INITIAL | Initial margin means the amount in the margin currency required for opening a position with the volume of one lot. It is used for checking a client's assets when he or she enters the market. | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_MARGIN\_MAINTENANCE | The maintenance margin. If it is set, it sets the margin amount in the margin currency of the symbol, charged from one lot. It is used for checking a client's assets when his/her account state changes. If the maintenance margin is equal to 0, the initial margin is used. | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_OPTION\_MODE | Option type | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_OPTION\_MODE\_EUROPEAN | European option may only be exercised on a specified date (expiration, execution date, delivery date) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_OPTION\_MODE\_AMERICAN | American option may be exercised on any trading day on or before expiry. The period within which a buyer can exercise the option is specified for it | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_OPTION\_RIGHT | Option right (Call/Put) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_OPTION\_RIGHT\_CALL | A call option gives you the right to buy an asset at a specified price | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_OPTION\_RIGHT\_PUT | A put option gives you the right to sell an asset at a specified price | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_OPTION\_STRIKE | The strike price of an option. The price at which an option buyer can buy (in a Call option) or sell (in a Put option) the underlying asset, and the option seller is obliged to sell or buy the appropriate amount of the underlying asset. | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_ORDER\_LIMIT | Limit orders are allowed (Buy Limit and Sell Limit) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_ORDER\_MARKET | Market orders are allowed (Buy and Sell) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_ORDER\_MODE | Flags of allowed [order types](/en/docs/constants/environment_state/marketinfoconstants#symbol_order_mode) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_ORDER\_SL | Stop Loss is allowed | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_ORDER\_STOP | Stop orders are allowed (Buy Stop and Sell Stop) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_ORDER\_STOP\_LIMIT | Stop-limit orders are allowed (Buy Stop Limit and Sell Stop Limit) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_ORDER\_TP | Take Profit is allowed | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_PATH | Path in the symbol tree | [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) |
| SYMBOL\_POINT | Symbol point value | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SELECT | Symbol is selected in Market Watch | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SESSION\_AW | Average weighted price of the current session | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_BUY\_ORDERS | Number of Buy orders at the moment | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SESSION\_BUY\_ORDERS\_VOLUME | Current volume of Buy orders | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_CLOSE | Close price of the current session | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_DEALS | Number of deals in the current session | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SESSION\_INTEREST | Summary open interest | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_OPEN | Open price of the current session | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_PRICE\_LIMIT\_MAX | Maximal price of the current session | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_PRICE\_LIMIT\_MIN | Minimal price of the current session | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_PRICE\_SETTLEMENT | Settlement price of the current session | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_SELL\_ORDERS | Number of Sell orders at the moment | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SESSION\_SELL\_ORDERS\_VOLUME | Current volume of Sell orders | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_TURNOVER | Summary turnover of the current session | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SESSION\_VOLUME | Summary volume of current session deals | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SPREAD | Spread value in points | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SPREAD\_FLOAT | Indication of a floating spread | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_START\_TIME | Date of the symbol trade beginning (usually used for futures) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_LONG | Long swap value | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_SWAP\_MODE | Swap calculation model | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_CURRENCY\_DEPOSIT | Swaps are charged in money, in client deposit currency | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_CURRENCY\_MARGIN | Swaps are charged in money in margin currency of the symbol | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_CURRENCY\_SYMBOL | Swaps are charged in money in base currency of the symbol | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_DISABLED | Swaps disabled (no swaps) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_INTEREST\_CURRENT | Swaps are charged as the specified annual interest from the instrument price at calculation of swap (standard bank year is 360 days) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_INTEREST\_OPEN | Swaps are charged as the specified annual interest from the open price of position (standard bank year is 360 days) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_POINTS | Swaps are charged in points | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_REOPEN\_BID | Swaps are charged by reopening positions. At the end of a trading day the position is closed. Next day it is reopened by the current Bid price +/- specified number of points (parameters SYMBOL\_SWAP\_LONG and SYMBOL\_SWAP\_SHORT) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_MODE\_REOPEN\_CURRENT | Swaps are charged by reopening positions. At the end of a trading day the position is closed. Next day it is reopened by the close price +/- specified number of points (parameters SYMBOL\_SWAP\_LONG and SYMBOL\_SWAP\_SHORT) | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_ROLLOVER3DAYS | Day of week to charge 3 days swap rollover | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_SWAP\_SHORT | Short swap value | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_TICKS\_BOOKDEPTH | Maximal number of requests shown in [Depth of Market](/en/docs/marketinformation/marketbookget). For symbols that have no queue of requests, the value is equal to zero. | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TIME | Time of the last quote | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_CALC\_MODE | Contract price calculation mode | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_CONTRACT\_SIZE | Trade contract size | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_TRADE\_EXECUTION\_EXCHANGE | Exchange execution | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_EXECUTION\_INSTANT | Instant execution | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_EXECUTION\_MARKET | Market execution | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_EXECUTION\_REQUEST | Execution by request | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_EXEMODE | Deal execution mode | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_FREEZE\_LEVEL | Distance to freeze trade operations in points | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_MODE | Order execution type | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_MODE\_CLOSEONLY | Allowed only position close operations | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_MODE\_DISABLED | Trade is disabled for the symbol | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_MODE\_FULL | No trade restrictions | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_MODE\_LONGONLY | Allowed only long positions | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_MODE\_SHORTONLY | Allowed only short positions | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_STOPS\_LEVEL | Minimal indention in points from the current close price to place Stop orders | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_TRADE\_TICK\_SIZE | Minimal price change | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_TRADE\_TICK\_VALUE | Value of SYMBOL\_TRADE\_TICK\_VALUE\_PROFIT | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_TRADE\_TICK\_VALUE\_LOSS | Calculated tick price for a losing position | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_TRADE\_TICK\_VALUE\_PROFIT | Calculated tick price for a profitable position | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_VOLUME | Volume of the last deal | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_VOLUME\_LIMIT | Maximum allowed aggregate volume of an open position and pending orders in one direction (buy or sell) for the symbol. For example, with the limitation of 5 lots, you can have an open buy position with the volume of 5 lots and place a pending order Sell Limit with the volume of 5 lots. But in this case you cannot place a Buy Limit pending order (since the total volume in one direction will exceed the limitation) or place Sell Limit with the volume more than 5 lots. | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_VOLUME\_MAX | Maximal volume for a deal | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_VOLUME\_MIN | Minimal volume for a deal | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_VOLUME\_STEP | Minimal volume change step for deal execution | [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) |
| SYMBOL\_VOLUMEHIGH | Maximal day volume | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| SYMBOL\_VOLUMELOW | Minimal day volume | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) |
| TENKANSEN\_LINE | Tenkan-sen line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| TERMINAL\_BUILD | The client terminal build number | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_CODEPAGE | Number of [the code page of the language](/en/docs/constants/io_constants/codepageusage) installed in the client terminal | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_COMMONDATA\_PATH | Common path for all of the terminals installed on a computer | [TerminalInfoString](/en/docs/check/terminalinfostring) |
| TERMINAL\_COMMUNITY\_ACCOUNT | The flag indicates the presence of MQL5.community authorization data in the terminal | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_COMMUNITY\_BALANCE | Balance in MQL5.community | [TerminalInfoDouble](/en/docs/check/terminalinfodouble) |
| TERMINAL\_COMMUNITY\_CONNECTION | Connection to MQL5.community | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_COMPANY | Company name | [TerminalInfoString](/en/docs/check/terminalinfostring) |
| TERMINAL\_CONNECTED | Connection to a trade server | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_CPU\_CORES | The number of CPU cores in the system | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_DATA\_PATH | Folder in which terminal data are stored | [TerminalInfoString](/en/docs/check/terminalinfostring) |
| TERMINAL\_DISK\_SPACE | Free disk space for the MQL5\Files folder of the terminal (agent), MB | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_DLLS\_ALLOWED | Permission to use DLL | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_EMAIL\_ENABLED | Permission to send e-mails using SMTP-server and login, specified in the terminal settings | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_FTP\_ENABLED | Permission to send reports using FTP-server and login, specified in the terminal settings | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_LANGUAGE | Language of the terminal | [TerminalInfoString](/en/docs/check/terminalinfostring) |
| TERMINAL\_MAXBARS | The maximal bars count on the chart | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_MEMORY\_AVAILABLE | Free memory of the terminal (agent) process, MB | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_MEMORY\_PHYSICAL | Physical memory in the system, MB | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_MEMORY\_TOTAL | Memory available to the process of the terminal (agent), MB | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_MEMORY\_USED | Memory used by the terminal (agent), MB | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_MQID | The flag indicates the presence of MetaQuotes ID data for [Push notifications](/en/docs/network/sendnotification) | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_NAME | Terminal name | [TerminalInfoString](/en/docs/check/terminalinfostring) |
| TERMINAL\_NOTIFICATIONS\_ENABLED | Permission to send notifications to smartphone | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_OPENCL\_SUPPORT | The version of the supported OpenCL in the format of 0x00010002 = 1.2.  "0" means that OpenCL is not supported | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_PATH | Folder from which the terminal is started | [TerminalInfoString](/en/docs/check/terminalinfostring) |
| TERMINAL\_PING\_LAST | The last known value of a ping to a trade server in microseconds. One second comprises of one million microseconds | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_SCREEN\_DPI | The resolution of information display on the screen is measured as number of Dots in a line per Inch (DPI). | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_TRADE\_ALLOWED | [Permission to trade](/en/docs/runtime/tradepermission) | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| TERMINAL\_X64 | Indication of the "64-bit terminal" | [TerminalInfoInteger](/en/docs/check/terminalinfointeger) |
| THURSDAY | Thursday | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger), [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote), [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) |
| TRADE\_ACTION\_DEAL | Place a trade order for an immediate execution with the specified parameters (market order) | [MqlTradeRequest](/en/docs/constants/structures/mqltraderequest) |
| TRADE\_ACTION\_MODIFY | Modify the parameters of the order placed previously | [MqlTradeRequest](/en/docs/constants/structures/mqltraderequest) |
| TRADE\_ACTION\_PENDING | Place a trade order for the execution under specified conditions (pending order) | [MqlTradeRequest](/en/docs/constants/structures/mqltraderequest) |
| TRADE\_ACTION\_REMOVE | Delete the pending order placed previously | [MqlTradeRequest](/en/docs/constants/structures/mqltraderequest) |
| TRADE\_ACTION\_SLTP | Modify Stop Loss and Take Profit values of an opened position | [MqlTradeRequest](/en/docs/constants/structures/mqltraderequest) |
| TRADE\_RETCODE\_CANCEL | Request canceled by trader | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_CLIENT\_DISABLES\_AT | Autotrading disabled by client terminal | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_CONNECTION | No connection with the trade server | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_DONE | Request completed | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_DONE\_PARTIAL | Only part of the request was completed | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_ERROR | Request processing error | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_FROZEN | Order or position frozen | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_INVALID | Invalid request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_INVALID\_EXPIRATION | Invalid order expiration date in the request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_INVALID\_FILL | Invalid [order filling type](/en/docs/constants/tradingconstants/orderproperties#enum_order_type_filling) | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_INVALID\_ORDER | Incorrect or prohibited [order type](/en/docs/constants/tradingconstants/orderproperties#enum_order_type) | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_INVALID\_PRICE | Invalid price in the request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_INVALID\_STOPS | Invalid stops in the request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_INVALID\_VOLUME | Invalid volume in the request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_LIMIT\_ORDERS | The number of pending orders has reached the limit | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_LIMIT\_VOLUME | The volume of orders and positions for the symbol has reached the limit | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_LOCKED | Request locked for processing | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_MARKET\_CLOSED | Market is closed | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_NO\_CHANGES | No changes in request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_NO\_MONEY | There is not enough money to complete the request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_ONLY\_REAL | Operation is allowed only for live accounts | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_ORDER\_CHANGED | Order state changed | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_PLACED | Order placed | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_POSITION\_CLOSED | Position with the specified [POSITION\_IDENTIFIER](/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer) has already been closed | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_PRICE\_CHANGED | Prices changed | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_PRICE\_OFF | There are no quotes to process the request | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_REJECT | Request rejected | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_REQUOTE | Requote | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_SERVER\_DISABLES\_AT | Autotrading disabled by server | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_TIMEOUT | Request canceled by timeout | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_TOO\_MANY\_REQUESTS | Too frequent requests | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_RETCODE\_TRADE\_DISABLED | Trade is disabled | [MqlTradeResult](/en/docs/constants/structures/mqltraderesult) |
| TRADE\_TRANSACTION\_DEAL\_ADD | Adding a deal to the history. The action is performed as a result of an order execution or performing operations with an account balance. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_DEAL\_DELETE | Deleting a deal from the history. There may be cases when a previously executed deal is deleted from a server. For example, a deal has been deleted in an external trading system (exchange) where it was previously transferred by a broker. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_DEAL\_UPDATE | Updating a deal in the history. There may be cases when a previously executed deal is changed on a server. For example, a deal has been changed in an external trading system (exchange) where it was previously transferred by a broker. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_HISTORY\_ADD | Adding an order to the history as a result of execution or cancellation. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_HISTORY\_DELETE | Deleting an order from the orders history. This type is provided for enhancing functionality on a trade server side. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_HISTORY\_UPDATE | Changing an order located in the orders history. This type is provided for enhancing functionality on a trade server side. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_ORDER\_ADD | Adding a new open order. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_ORDER\_DELETE | Removing an order from the list of the open ones. An order can be deleted from the open ones as a result of setting an appropriate request or execution (filling) and moving to the history. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_ORDER\_UPDATE | Updating an open order. The updates include not only evident changes from the client terminal or a trade server sides but also changes of an order state when setting it (for example, transition from [ORDER\_STATE\_STARTED](/en/docs/constants/tradingconstants/orderproperties#enum_order_state) to [ORDER\_STATE\_PLACED](/en/docs/constants/tradingconstants/orderproperties#enum_order_state) or from [ORDER\_STATE\_PLACED](/en/docs/constants/tradingconstants/orderproperties#enum_order_state) to [ORDER\_STATE\_PARTIAL](/en/docs/constants/tradingconstants/orderproperties#enum_order_state), etc.). | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_POSITION | Changing a position not related to a deal execution. This type of transaction shows that a position has been changed on a trade server side. Position volume, open price, Stop Loss and Take Profit levels can be changed. Data on changes are submitted in [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) structure via OnTradeTransaction handler. Position change (adding, changing or closing), as a result of a deal execution, does not lead to the occurrence of TRADE\_TRANSACTION\_POSITION transaction. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TRADE\_TRANSACTION\_REQUEST | Notification of the fact that a trade request has been processed by a server and processing result has been received. Only type field (trade transaction type) must be analyzed for such transactions in [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) structure. The second and third parameters of [OnTradeTransaction](/en/docs/event_handlers/ontradetransaction) (request and result) must be analyzed for additional data. | [MqlTradeTransaction](/en/docs/constants/structures/mqltradetransaction) |
| TUESDAY | Tuesday | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger), [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote), [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) |
| TYPE\_BOOL | bool | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_CHAR | char | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_COLOR | color | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_DATETIME | datetime | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_DOUBLE | double | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_FLOAT | float | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_INT | int | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_LONG | long | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_SHORT | short | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_STRING | string | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_UCHAR | uchar | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_UINT | uint | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_ULONG | ulong | [MqlParam](/en/docs/constants/structures/mqlparam) |
| TYPE\_USHORT | ushort | [MqlParam](/en/docs/constants/structures/mqlparam) |
| UCHAR\_MAX | Maximal value, which can be represented by uchar type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| UINT\_MAX | Maximal value, which can be represented by uint type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| ULONG\_MAX | Maximal value, which can be represented by ulong type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| UPPER\_BAND | Upper limit | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| UPPER\_HISTOGRAM | Upper histogram | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| UPPER\_LINE | Upper line | [Indicators Lines](/en/docs/constants/indicatorconstants/lines) |
| USHORT\_MAX | Maximal value, which can be represented by ushort type | [Numerical Type Constants](/en/docs/constants/namedconstants/typeconstants) |
| VOLUME\_REAL | Trade volume | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| VOLUME\_TICK | Tick volume | [Price Constants](/en/docs/constants/indicatorconstants/prices) |
| WEDNESDAY | Wednesday | [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger), [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote), [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) |
| WHOLE\_ARRAY | Means the number of items remaining until the end of the array, i.e., the entire array will be processed | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |
| WRONG\_VALUE | The constant can be implicitly [cast](/en/docs/basis/types/casting) to any [enumeration](/en/docs/basis/types/integer/enumeration) type | [Other Constants](/en/docs/constants/namedconstants/otherconstants) |

[List of MQL5 Functions](/en/docs/function_indices "List of MQL5 Functions")