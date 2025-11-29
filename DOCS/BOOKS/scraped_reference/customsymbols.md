---
title: "Custom symbols"
url: "https://www.mql5.com/en/docs/customsymbols"
hierarchy: []
scraped_at: "2025-11-28 09:30:10"
---

# Custom symbols

[MQL5 Reference](/en/docs "MQL5 Reference")Custom Symbols

* [CustomSymbolCreate](/en/docs/customsymbols/customsymbolcreate "CustomSymbolCreate")
* [CustomSymbolDelete](/en/docs/customsymbols/customsymboldelete "CustomSymbolDelete")
* [CustomSymbolSetInteger](/en/docs/customsymbols/customsymbolsetinteger "CustomSymbolSetInteger")
* [CustomSymbolSetDouble](/en/docs/customsymbols/customsymbolsetdouble "CustomSymbolSetDouble")
* [CustomSymbolSetString](/en/docs/customsymbols/customsymbolsetstring "CustomSymbolSetString")
* [CustomSymbolSetMarginRate](/en/docs/customsymbols/customsymbolsetmarginrate "CustomSymbolSetMarginRate")
* [CustomSymbolSetSessionQuote](/en/docs/customsymbols/customsymbolsetsessionquote "CustomSymbolSetSessionQuote")
* [CustomSymbolSetSessionTrade](/en/docs/customsymbols/customsymbolsetsessiontrade "CustomSymbolSetSessionTrade")
* [CustomRatesDelete](/en/docs/customsymbols/customratesdelete "CustomRatesDelete")
* [CustomRatesReplace](/en/docs/customsymbols/customratesreplace "CustomRatesReplace")
* [CustomRatesUpdate](/en/docs/customsymbols/customratesupdate "CustomRatesUpdate")
* [CustomTicksAdd](/en/docs/customsymbols/customticksadd "CustomTicksAdd")
* [CustomTicksDelete](/en/docs/customsymbols/customticksdelete "CustomTicksDelete")
* [CustomTicksReplace](/en/docs/customsymbols/customticksreplace "CustomTicksReplace")
* [CustomBookAdd](/en/docs/customsymbols/custombookadd "CustomBookAdd")

# Custom symbols

Functions for creating and editing the custom symbol properties.

When connecting the terminal to a certain trade server, a user is able to [work with time series](/en/docs/series) of the financial symbols provided by a broker. Available financial symbols are displayed as a list in the Market Watch window. A separate group of functions allows [receiving data on the symbol properties](/en/docs/marketinformation), trading sessions and market depth updates.

The group of functions described in this section allows creating custom symbols. To do this, users are able to apply the trade server's existing symbols, text files or external data sources.

| Function | Action |
| --- | --- |
| [CustomSymbolCreate](/en/docs/customsymbols/customsymbolcreate) | Create a custom symbol with the specified name in the specified group |
| [CustomSymbolDelete](/en/docs/customsymbols/customsymboldelete) | Delete a custom symbol with the specified name |
| [CustomSymbolSetInteger](/en/docs/customsymbols/customsymbolsetinteger) | Set the integer type property value for a custom symbol |
| [CustomSymbolSetDouble](/en/docs/customsymbols/customsymbolsetdouble) | Set the real type property value for a custom symbol |
| [CustomSymbolSetString](/en/docs/customsymbols/customsymbolsetstring) | Set the string type property value for a custom symbol |
| [CustomSymbolSetMarginRate](/en/docs/customsymbols/customsymbolsetmarginrate) | Set the margin rates depending on the order type and direction for a custom symbol |
| [CustomSymbolSetSessionQuote](/en/docs/customsymbols/customsymbolsetsessionquote) | Set the start and end time of the specified quotation session for the specified symbol and week day |
| [CustomSymbolSetSessionTrade](/en/docs/customsymbols/customsymbolsetsessiontrade) | Set the start and end time of the specified trading session for the specified symbol and week day |
| [CustomRatesDelete](/en/docs/customsymbols/customratesdelete) | Delete all bars from the price history of the custom symbol in the specified time interval |
| [CustomRatesReplace](/en/docs/customsymbols/customratesreplace) | Fully replace the price history of the custom symbol within the specified time interval with the data from the MqlRates type array |
| [CustomRatesUpdate](/en/docs/customsymbols/customratesupdate) | Add missing bars to the custom symbol history and replace existing data with the ones from the MqlRates type array |
| [CustomTicksAdd](/en/docs/customsymbols/customticksadd) | Adds data from an array of the MqlTick type to the price history of a custom symbol. The custom symbol must be selected in the Market Watch window |
| [CustomTicksDelete](/en/docs/customsymbols/customticksdelete) | Delete all ticks from the price history of the custom symbol in the specified time interval |
| [CustomTicksReplace](/en/docs/customsymbols/customticksreplace) | Fully replace the price history of the custom symbol within the specified time interval with the data from the MqlTick type array |
| [CustomBookAdd](/en/docs/customsymbols/custombookadd) | Passes the status of the Depth of Market for a custom symbol |

[iSpread](/en/docs/series/ispread "iSpread")

[CustomSymbolCreate](/en/docs/customsymbols/customsymbolcreate "CustomSymbolCreate")