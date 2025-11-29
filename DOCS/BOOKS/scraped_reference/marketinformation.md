---
title: "Getting Market Information"
url: "https://www.mql5.com/en/docs/marketinformation"
hierarchy: []
scraped_at: "2025-11-28 09:30:38"
---

# Getting Market Information

[MQL5 Reference](/en/docs "MQL5 Reference")Market Info

* [SymbolsTotal](/en/docs/marketinformation/symbolstotal "SymbolsTotal")
* [SymbolExist](/en/docs/marketinformation/symbolexist "SymbolExist")
* [SymbolName](/en/docs/marketinformation/symbolname "SymbolName")
* [SymbolSelect](/en/docs/marketinformation/symbolselect "SymbolSelect")
* [SymbolIsSynchronized](/en/docs/marketinformation/symbolissynchronized "SymbolIsSynchronized")
* [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble "SymbolInfoDouble")
* [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger "SymbolInfoInteger")
* [SymbolInfoString](/en/docs/marketinformation/symbolinfostring "SymbolInfoString")
* [SymbolInfoMarginRate](/en/docs/marketinformation/symbolinfomarginrate "SymbolInfoMarginRate")
* [SymbolInfoTick](/en/docs/marketinformation/symbolinfotick "SymbolInfoTick")
* [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote "SymbolInfoSessionQuote")
* [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade "SymbolInfoSessionTrade")
* [MarketBookAdd](/en/docs/marketinformation/marketbookadd "MarketBookAdd")
* [MarketBookRelease](/en/docs/marketinformation/marketbookrelease "MarketBookRelease")
* [MarketBookGet](/en/docs/marketinformation/marketbookget "MarketBookGet")

# Getting Market Information

These are functions intended for receiving information about the market state.

| Function | Action |
| --- | --- |
| [SymbolsTotal](/en/docs/marketinformation/symbolstotal) | Returns the number of available (selected in Market Watch or all) symbols |
| [SymbolExist](/en/docs/marketinformation/symbolexist) | Checks if a symbol with a specified name exists |
| [SymbolName](/en/docs/marketinformation/symbolname) | Returns the name of a specified symbol |
| [SymbolSelect](/en/docs/marketinformation/symbolselect) | Selects a symbol in the Market Watch window or removes a symbol from the window |
| [SymbolIsSynchronized](/en/docs/marketinformation/symbolissynchronized) | Checks whether data of a selected symbol in the terminal are [synchronized](/en/docs/series/timeseries_access#synchronized) with data on the trade server |
| [SymbolInfoDouble](/en/docs/marketinformation/symbolinfodouble) | Returns the double value of the symbol for the corresponding property |
| [SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger) | Returns a value of an integer type (long, datetime, int or bool) of a specified symbol for the corresponding property |
| [SymbolInfoString](/en/docs/marketinformation/symbolinfostring) | Returns a value of the string type of a specified symbol for the corresponding property |
| [SymbolInfoMarginRate](/en/docs/marketinformation/symbolinfomarginrate) | Returns the margin rates depending on the order type and direction |
| [SymbolInfoTick](/en/docs/marketinformation/symbolinfotick) | Returns the current prices for the specified symbol in a variable of the [MqlTick](/en/docs/constants/structures/mqltick) type |
| [SymbolInfoSessionQuote](/en/docs/marketinformation/symbolinfosessionquote) | Allows receiving time of beginning and end of the specified quoting sessions for a specified symbol and day of week. |
| [SymbolInfoSessionTrade](/en/docs/marketinformation/symbolinfosessiontrade) | Allows receiving time of beginning and end of the specified trading sessions for a specified symbol and day of week. |
| [MarketBookAdd](/en/docs/marketinformation/marketbookadd) | Provides opening of Depth of Market for a selected symbol, and subscribes for receiving notifications of the DOM changes |
| [MarketBookRelease](/en/docs/marketinformation/marketbookrelease) | Provides closing of Depth of Market for a selected symbol, and cancels the subscription for receiving notifications of the DOM changes |
| [MarketBookGet](/en/docs/marketinformation/marketbookget) | Returns a structure array [MqlBookInfo](/en/docs/constants/structures/mqlbookinfo) containing records of the Depth of Market of a specified symbol |

[OnTesterPass](/en/docs/event_handlers/ontesterpass "OnTesterPass")

[SymbolsTotal](/en/docs/marketinformation/symbolstotal "SymbolsTotal")