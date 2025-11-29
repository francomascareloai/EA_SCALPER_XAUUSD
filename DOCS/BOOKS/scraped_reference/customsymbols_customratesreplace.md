---
title: "CustomRatesReplace"
url: "https://www.mql5.com/en/docs/customsymbols/customratesreplace"
hierarchy: []
scraped_at: "2025-11-28 09:31:45"
---

# CustomRatesReplace

[MQL5 Reference](/en/docs "MQL5 Reference")[Custom Symbols](/en/docs/customsymbols "Custom Symbols")CustomRatesReplace

* [CustomSymbolCreate](/en/docs/customsymbols/customsymbolcreate "CustomSymbolCreate")
* [CustomSymbolDelete](/en/docs/customsymbols/customsymboldelete "CustomSymbolDelete")
* [CustomSymbolSetInteger](/en/docs/customsymbols/customsymbolsetinteger "CustomSymbolSetInteger")
* [CustomSymbolSetDouble](/en/docs/customsymbols/customsymbolsetdouble "CustomSymbolSetDouble")
* [CustomSymbolSetString](/en/docs/customsymbols/customsymbolsetstring "CustomSymbolSetString")
* [CustomSymbolSetMarginRate](/en/docs/customsymbols/customsymbolsetmarginrate "CustomSymbolSetMarginRate")
* [CustomSymbolSetSessionQuote](/en/docs/customsymbols/customsymbolsetsessionquote "CustomSymbolSetSessionQuote")
* [CustomSymbolSetSessionTrade](/en/docs/customsymbols/customsymbolsetsessiontrade "CustomSymbolSetSessionTrade")
* [CustomRatesDelete](/en/docs/customsymbols/customratesdelete "CustomRatesDelete")
* CustomRatesReplace
* [CustomRatesUpdate](/en/docs/customsymbols/customratesupdate "CustomRatesUpdate")
* [CustomTicksAdd](/en/docs/customsymbols/customticksadd "CustomTicksAdd")
* [CustomTicksDelete](/en/docs/customsymbols/customticksdelete "CustomTicksDelete")
* [CustomTicksReplace](/en/docs/customsymbols/customticksreplace "CustomTicksReplace")
* [CustomBookAdd](/en/docs/customsymbols/custombookadd "CustomBookAdd")

# CustomRatesReplace

Fully replaces the price history of the custom symbol within the specified time interval with the data from the [MqlRates](/en/docs/constants/structures/mqlrates) type array.

| |
| --- |
| int  CustomRatesReplace(    const string     symbol,             // symbol name    datetime         from,               // start date    datetime         to,                 // end date    const MqlRates&  rates[],            // array for the data to be applied to a custom symbol    uint             count=WHOLE\_ARRAY   // number of the rates[] array elements to be used    ); |

Parameters

symbol

[in]  Custom symbol name.

from

[in]  Time of the first bar in the price history within the specified range to be updated.

to

[in]  Time of the last bar in the price history within the specified range to be updated.

rates[]

[in]   Array of the [MqlRates](/en/docs/constants/structures/mqlrates) type history data for M1.

count=WHOLE\_ARRAY

[in]  Number of the rates[] array elements to be used for replacement. [WHOLE\_ARRAY](/en/docs/constants/namedconstants/otherconstants) means that all rates[] array elements should be used for replacement.

Return Value

Number of updated bars or -1 in case of an [error](/en/docs/constants/errorswarnings/errorcodes).

Note

If the bar from the rates[] array goes beyond the specified range, it is ignored. If such a bar is already present in the price history and enters the given range, it is replaced. All other bars in the current price history outside the specified range remain unchanged. The rates[] array data should be correct regarding OHLC prices, while the bars opening time should correspond to the M1 [timeframe](/en/docs/constants/chartconstants/enum_timeframes).

 

Example:

| |
| --- |
| //+------------------------------------------------------------------+ //|                                           CustomRatesReplace.mq5 | //|                                  Copyright 2024, MetaQuotes Ltd. | //|                                             https://www.mql5.com | //+------------------------------------------------------------------+ #property copyright "Copyright 2024, MetaQuotes Ltd." #property link      "https://www.mql5.com" #property version   "1.00"   #define   CUSTOM\_SYMBOL\_NAME     Symbol()+".C"     // custom symbol name #define   CUSTOM\_SYMBOL\_PATH     "Forex"           // name of the group, in which a symbol is to be created #define   CUSTOM\_SYMBOL\_ORIGIN   Symbol()          // name of a symbol a custom one is to be based on   #define   DATARATES\_COUNT        4                 // number of bars sent to the journal   //+------------------------------------------------------------------+ //| Script program start function                                    | //+------------------------------------------------------------------+ void OnStart()   { //--- get the error code when creating a custom symbol    int create=CreateCustomSymbol(CUSTOM\_SYMBOL\_NAME, CUSTOM\_SYMBOL\_PATH, CUSTOM\_SYMBOL\_ORIGIN);     //--- if the error code is not 0 (successful symbol creation) and not 5304 (symbol has already been created) - leave    if(create!=0 && create!=5304)       return;   //--- get the number of standard symbol bars    int bars=Bars(CUSTOM\_SYMBOL\_ORIGIN, PERIOD\_M1);        //--- get the data of all bars of the standard symbol minute timeframe into the MqlRates array    MqlRates rates[]={};    ResetLastError();    if(CopyRates(CUSTOM\_SYMBOL\_ORIGIN, PERIOD\_M1, 0, bars, rates)!=bars)      {       PrintFormat("CopyRates(%s, PERIOD\_M1, 0, %d) failed. Error %d", CUSTOM\_SYMBOL\_ORIGIN, bars, GetLastError());       return;      }   //--- set the copied data to the minute history of the custom symbol    ResetLastError();    if(CustomRatesUpdate(CUSTOM\_SYMBOL\_NAME, rates)<0)      {       PrintFormat("CustomRatesUpdate(%s) failed. Error %d", CUSTOM\_SYMBOL\_NAME, GetLastError());       return;      }       //--- after updating the historical data, get the number of custom symbol bars    bars=Bars(CUSTOM\_SYMBOL\_NAME, PERIOD\_M1);     //--- get the data of all bars of the custom symbol minute timeframe into the MqlRates array    ResetLastError();    if(CopyRates(CUSTOM\_SYMBOL\_NAME, PERIOD\_M1, 0, bars, rates)!=bars)      {       PrintFormat("CopyRates(%s, PERIOD\_M1, 0, %d) failed. Error %d", CUSTOM\_SYMBOL\_NAME, bars, GetLastError());       return;      }   //--- print the last DATARATES\_COUNT bars of the custom symbol minute history in the journal    int digits=(int)SymbolInfoInteger(CUSTOM\_SYMBOL\_NAME, SYMBOL\_DIGITS);    PrintFormat("Last %d bars of the custom symbol's minute history:", DATARATES\_COUNT);    ArrayPrint(rates, digits, NULL, bars-DATARATES\_COUNT, DATARATES\_COUNT);     //--- change the two penultimate data bars in the custom symbol minute history    datetime time\_from= rates[bars-3].time;    datetime time\_to  = rates[bars-2].time;     //--- make all prices of the two penultimate bars equal to the open prices of these bars in the 'rates' array    rates[bars-3].high=rates[bars-3].open;    rates[bars-3].low=rates[bars-3].open;    rates[bars-3].close=rates[bars-3].open;        rates[bars-2].high=rates[bars-2].open;    rates[bars-2].low=rates[bars-2].open;    rates[bars-2].close=rates[bars-2].open;     //--- replace existing bars with data from the modified 'rates' array    ResetLastError();    int replaced=CustomRatesUpdate(CUSTOM\_SYMBOL\_NAME, rates);    if(replaced<0)      {       PrintFormat("CustomRatesUpdate(%s) failed. Error %d", CUSTOM\_SYMBOL\_NAME, GetLastError());       return;      }       //--- after changing two bars of historical data, get the number of custom symbol bars again    bars=Bars(CUSTOM\_SYMBOL\_NAME, PERIOD\_M1);     //--- get the data of all bars of the custom symbol minute timeframe again    ResetLastError();    if(CopyRates(CUSTOM\_SYMBOL\_NAME, PERIOD\_M1, 0, bars, rates)!=bars)      {       PrintFormat("CopyRates(%s, PERIOD\_M1, 0, %d) failed. Error %d", CUSTOM\_SYMBOL\_NAME, bars, GetLastError());       return;      }   //--- print the last DATARATES\_COUNT bars of the updated custom symbol minute history in the journal    PrintFormat("\nLast %d bars after applying CustomRatesUpdate() with %d replaced bars:", DATARATES\_COUNT, replaced);    ArrayPrint(rates, digits, NULL, bars-DATARATES\_COUNT, DATARATES\_COUNT);       //--- display a hint about the script termination keys on the chart comment    Comment(StringFormat("Press 'Esc' to exit or 'Del' to delete the '%s' symbol and exit", CUSTOM\_SYMBOL\_NAME)); //--- wait for pressing the Esc or Del keys to exit in an endless loop    while(!IsStopped() && TerminalInfoInteger(TERMINAL\_KEYSTATE\_ESCAPE)==0)      {       Sleep(16);       //--- when pressing Del, delete the created custom symbol and its data       if(TerminalInfoInteger(TERMINAL\_KEYSTATE\_DELETE)<0)         {          //--- delete bar data          int deleted=CustomRatesDelete(CUSTOM\_SYMBOL\_NAME, 0, LONG\_MAX);          if(deleted>0)             PrintFormat("%d history bars of the custom symbol '%s' were successfully deleted", deleted, CUSTOM\_SYMBOL\_NAME);                    //--- delete tick data          deleted=CustomTicksDelete(CUSTOM\_SYMBOL\_NAME, 0, LONG\_MAX);          if(deleted>0)             PrintFormat("%d history ticks of the custom symbol '%s' were successfully deleted", deleted, CUSTOM\_SYMBOL\_NAME);                    //--- delete symbol          if(DeleteCustomSymbol(CUSTOM\_SYMBOL\_NAME))             PrintFormat("Custom symbol '%s' deleted successfully", CUSTOM\_SYMBOL\_NAME);          break;         }      } //--- clear the chart before exiting    Comment("");    /\*    result:    Last 4 bars of the custom symbol's minute history:                     [time]  [open]  [high]   [low] [close] [tick\_volume] [spread] [real\_volume]    [0] 2024.07.29 13:37:00 1.08394 1.08396 1.08388 1.08390            16        1             0    [1] 2024.07.29 13:38:00 1.08389 1.08400 1.08389 1.08398            35        1             0    [2] 2024.07.29 13:39:00 1.08398 1.08410 1.08394 1.08410            29        1             0    [3] 2024.07.29 13:40:00 1.08409 1.08414 1.08408 1.08414            14        1             0        Last 4 bars after applying CustomRatesUpdate() with 250820 replaced bars:                     [time]  [open]  [high]   [low] [close] [tick\_volume] [spread] [real\_volume]    [0] 2024.07.29 13:37:00 1.08394 1.08396 1.08388 1.08390            16        1             0    [1] 2024.07.29 13:38:00 1.08389 1.08389 1.08389 1.08389            35        1             0    [2] 2024.07.29 13:39:00 1.08398 1.08398 1.08398 1.08398            29        1             0    [3] 2024.07.29 13:40:00 1.08409 1.08414 1.08408 1.08414            14        1             0    \*/   } //+------------------------------------------------------------------+ //| Create a custom symbol, return an error code                     | //+------------------------------------------------------------------+ int CreateCustomSymbol(const string symbol\_name, const string symbol\_path, const string symbol\_origin=NULL)   { //--- define the name of a symbol a custom one is to be based on    string origin=(symbol\_origin==NULL ? Symbol() : symbol\_origin);     //--- if failed to create a custom symbol and this is not error 5304, report this in the journal    ResetLastError();    int error=0;    if(!CustomSymbolCreate(symbol\_name, symbol\_path, origin))      {       error=GetLastError();       if(error!=5304)          PrintFormat("CustomSymbolCreate(%s, %s, %s) failed. Error %d", symbol\_name, symbol\_path, origin, error);      } //--- successful    return(error);   } //+------------------------------------------------------------------+ //| Remove a custom symbol                                           | //+------------------------------------------------------------------+ bool DeleteCustomSymbol(const string symbol\_name)   { //--- hide the symbol from the Market Watch window    ResetLastError();    if(!SymbolSelect(symbol\_name, false))      {       PrintFormat("SymbolSelect(%s, false) failed. Error %d", GetLastError());       return(false);      }        //--- if failed to delete a custom symbol, report this in the journal and return 'false'    ResetLastError();    if(!CustomSymbolDelete(symbol\_name))      {       PrintFormat("CustomSymbolDelete(%s) failed. Error %d", symbol\_name, GetLastError());       return(false);      } //--- successful    return(true);   } |

 

See also

[CustomRatesDelete](/en/docs/customsymbols/customratesdelete), [CustomRatesUpdate](/en/docs/customsymbols/customratesupdate), [CopyRates](/en/docs/series/copyrates)

[CustomRatesDelete](/en/docs/customsymbols/customratesdelete "CustomRatesDelete")

[CustomRatesUpdate](/en/docs/customsymbols/customratesupdate "CustomRatesUpdate")