---
title: "CustomSymbolSetInteger"
url: "https://www.mql5.com/en/docs/customsymbols/customsymbolsetinteger"
hierarchy: []
scraped_at: "2025-11-28 09:31:49"
---

# CustomSymbolSetInteger

[MQL5 Reference](/en/docs "MQL5 Reference")[Custom Symbols](/en/docs/customsymbols "Custom Symbols")CustomSymbolSetInteger

* [CustomSymbolCreate](/en/docs/customsymbols/customsymbolcreate "CustomSymbolCreate")
* [CustomSymbolDelete](/en/docs/customsymbols/customsymboldelete "CustomSymbolDelete")
* CustomSymbolSetInteger
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

# CustomSymbolSetInteger

Sets the integer type property value for a custom symbol.

| |
| --- |
| bool  CustomSymbolSetInteger(    const string              symbol\_name,      // symbol name    ENUM\_SYMBOL\_INFO\_INTEGER  property\_id,      // property ID    long                      property\_value    // property value    ); |

Parameters

symbol\_name

[in]  Custom symbol name.

property\_id

[in]  Symbol property ID. The value can be one of the values of the [ENUM\_SYMBOL\_INFO\_INTEGER](/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_integer) enumeration.

property\_value

[in]  A long type variable containing the property value.

Return Value

true – success, otherwise – false. To get information about the error, call the [GetLastError()](/en/docs/check/getlasterror) function.

Note

The minute and tick history of the custom symbol is completely removed if any of these properties is changed in the symbol specification:

* SYMBOL\_CHART\_MODE – price type for constructing bars (Bid or Last)
* SYMBOL\_DIGITS – number of digits after the decimal point to display the price

After deleting the custom symbol history, the terminal attempts to create a new history using the updated properties. The same happens when the custom symbol properties are changed manually.

 

Example:

| |
| --- |
| //+------------------------------------------------------------------+ //|                                       CustomSymbolSetInteger.mq5 | //|                                  Copyright 2024, MetaQuotes Ltd. | //|                                             https://www.mql5.com | //+------------------------------------------------------------------+ #property copyright "Copyright 2024, MetaQuotes Ltd." #property link      "https://www.mql5.com" #property version   "1.00"   #define   CUSTOM\_SYMBOL\_NAME     Symbol()+".C"  // custom symbol name #define   CUSTOM\_SYMBOL\_PATH     "Forex"        // name of the group, in which a symbol is to be created #define   CUSTOM\_SYMBOL\_ORIGIN   Symbol()       // name of a symbol a custom one is to be based on   //+------------------------------------------------------------------+ //| Script program start function                                    | //+------------------------------------------------------------------+ void OnStart()   { //--- get the error code when creating a custom symbol    int create=CreateCustomSymbol(CUSTOM\_SYMBOL\_NAME, CUSTOM\_SYMBOL\_PATH, CUSTOM\_SYMBOL\_ORIGIN);     //--- if the error code is not 0 (successful symbol creation) and not 5304 (symbol has already been created) - leave    if(create!=0 && create!=5304)       return;        //--- get and print in the journal the properties of the symbol the custom one is based on //--- (trading mode, Stop order installation level and trading operations freezing distance)    ENUM\_SYMBOL\_TRADE\_EXECUTION origin\_exe\_mode = (ENUM\_SYMBOL\_TRADE\_EXECUTION)SymbolInfoInteger(CUSTOM\_SYMBOL\_ORIGIN, SYMBOL\_TRADE\_EXEMODE);    int origin\_stops\_level = (int)SymbolInfoInteger(CUSTOM\_SYMBOL\_ORIGIN, SYMBOL\_TRADE\_STOPS\_LEVEL);    int origin\_freeze\_level= (int)SymbolInfoInteger(CUSTOM\_SYMBOL\_ORIGIN, SYMBOL\_TRADE\_FREEZE\_LEVEL);        PrintFormat("The '%s' symbol from which the custom '%s' was created:\n"+                "  Deal execution mode: %s\n  Stops Level: %d\n  Freeze Level: %d",                CUSTOM\_SYMBOL\_ORIGIN, CUSTOM\_SYMBOL\_NAME,                StringSubstr(EnumToString(origin\_exe\_mode), 23), origin\_stops\_level, origin\_freeze\_level);     //--- set other values for the custom symbol properties    ResetLastError();    bool res=true;    res &=CustomSymbolSetInteger(CUSTOM\_SYMBOL\_NAME, SYMBOL\_TRADE\_EXEMODE, SYMBOL\_TRADE\_EXECUTION\_MARKET);    res &=CustomSymbolSetInteger(CUSTOM\_SYMBOL\_NAME, SYMBOL\_TRADE\_STOPS\_LEVEL, 10);    res &=CustomSymbolSetInteger(CUSTOM\_SYMBOL\_NAME, SYMBOL\_TRADE\_FREEZE\_LEVEL, 3);   //--- if there was an error when setting any of the properties, display an appropriate message in the journal    if(!res)       Print("CustomSymbolSetInteger() failed. Error ", GetLastError());     //--- get and print in the journal the modified custom symbol properties //--- (trading mode, Stop order installation level and trading operations freezing distance)    ENUM\_SYMBOL\_TRADE\_EXECUTION custom\_exe\_mode = (ENUM\_SYMBOL\_TRADE\_EXECUTION)SymbolInfoInteger(CUSTOM\_SYMBOL\_NAME, SYMBOL\_TRADE\_EXEMODE);    int custom\_stops\_level = (int)SymbolInfoInteger(CUSTOM\_SYMBOL\_NAME, SYMBOL\_TRADE\_STOPS\_LEVEL);    int custom\_freeze\_level= (int)SymbolInfoInteger(CUSTOM\_SYMBOL\_NAME, SYMBOL\_TRADE\_FREEZE\_LEVEL);        PrintFormat("Custom symbol '%s' based on '%s':\n"+                "  Deal execution mode: %s\n  Stops Level: %d\n  Freeze Level: %d",                CUSTOM\_SYMBOL\_NAME, CUSTOM\_SYMBOL\_ORIGIN,                 StringSubstr(EnumToString(custom\_exe\_mode), 23), custom\_stops\_level, custom\_freeze\_level);     //--- display a hint about the script termination keys on the chart comment    Comment(StringFormat("Press 'Esc' to exit or 'Del' to delete the '%s' symbol and exit", CUSTOM\_SYMBOL\_NAME));   //--- wait for pressing the Esc or Del keys to exit in an endless loop    while(!IsStopped() && TerminalInfoInteger(TERMINAL\_KEYSTATE\_ESCAPE)==0)      {       Sleep(16);       //--- when pressing Del, delete the created custom symbol       if(TerminalInfoInteger(TERMINAL\_KEYSTATE\_DELETE)<0)         {          if(DeleteCustomSymbol(CUSTOM\_SYMBOL\_NAME))             PrintFormat("Custom symbol '%s' deleted successfully", CUSTOM\_SYMBOL\_NAME);          break;         }      } //--- clear the chart before exiting    Comment("");    /\*    result:    The 'EURUSD' symbol from which the custom 'EURUSD.C' was created:      Deal execution mode: INSTANT      Stops Level: 0      Freeze Level: 0    Custom symbol 'EURUSD.C' based on 'EURUSD':      Deal execution mode: MARKET      Stops Level: 10      Freeze Level: 3    \*/   } //+------------------------------------------------------------------+ //| Create a custom symbol, return an error code                     | //+------------------------------------------------------------------+ int CreateCustomSymbol(const string symbol\_name, const string symbol\_path, const string symbol\_origin=NULL)   { //--- define the name of a symbol a custom one is to be based on    string origin=(symbol\_origin==NULL ? Symbol() : symbol\_origin);     //--- if failed to create a custom symbol and this is not error 5304, report this in the journal    ResetLastError();    int error=0;    if(!CustomSymbolCreate(symbol\_name, symbol\_path, origin))      {       error=GetLastError();       if(error!=5304)          PrintFormat("CustomSymbolCreate(%s, %s, %s) failed. Error %d", symbol\_name, symbol\_path, origin, error);      } //--- successful    return(error);   } //+------------------------------------------------------------------+ //| Remove a custom symbol                                           | //+------------------------------------------------------------------+ bool DeleteCustomSymbol(const string symbol\_name)   { //--- hide the symbol from the Market Watch window    ResetLastError();    if(!SymbolSelect(symbol\_name, false))      {       PrintFormat("SymbolSelect(%s, false) failed. Error %d", GetLastError());       return(false);      }        //--- if failed to delete a custom symbol, report this in the journal and return 'false'    ResetLastError();    if(!CustomSymbolDelete(symbol\_name))      {       PrintFormat("CustomSymbolDelete(%s) failed. Error %d", symbol\_name, GetLastError());       return(false);      } //--- successful    return(true);   } |

 

See also

[SymbolInfoInteger](/en/docs/marketinformation/symbolinfointeger)

[CustomSymbolDelete](/en/docs/customsymbols/customsymboldelete "CustomSymbolDelete")

[CustomSymbolSetDouble](/en/docs/customsymbols/customsymbolsetdouble "CustomSymbolSetDouble")