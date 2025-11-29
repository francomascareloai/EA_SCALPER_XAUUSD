---
title: "Generation of custom events"
url: "https://www.mql5.com/en/book/applications/events/events_custom"
hierarchy: []
scraped_at: "2025-11-28 09:48:08"
---

# Generation of custom events

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")[Interactive events on charts](/en/book/applications/events "Interactive events on charts")Generation of custom events

* [Event handling function OnChartEvent](/en/book/applications/events/events_onchartevent "Event handling function OnChartEvent")
* [Event-related chart properties](/en/book/applications/events/events_properties "Event-related chart properties")
* [Chart change event](/en/book/applications/events/events_chart "Chart change event")
* [Keyboard events](/en/book/applications/events/events_keyboard "Keyboard events")
* [Mouse events](/en/book/applications/events/events_mouse "Mouse events")
* [Graphical object events](/en/book/applications/events/events_objects "Graphical object events")
* Generation of custom events

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Generation of custom events

In addition to standard events, the terminal supports the ability to programmatically generate custom events, the essence and content of which are determined by the MQL program. Such events are added to the general queue of chart events and can be processed in the function OnChartEvent by all interested programs.

A special range of 65536 integer identifiers is reserved for custom events: from CHARTEVENT\_CUSTOM to CHARTEVENT\_CUSTOM\_LAST inclusive. In other words, the custom event must have the ID CHARTEVENT\_CUSTOM + n, where n is between 0 and 65535. CHARTEVENT\_CUSTOM\_LAST is exactly equal to CHARTEVENT\_CUSTOM + 65535.

Custom events are sent to the chart using the EventChartCustom function.

bool EventChartCustom(long chartId, ushort customEventId, 
   long lparam, double dparam, string sparam)

chartId is the identifier of the event recipient chart, while 0 indicates the current chart; customEventId is the event ID (selected by the MQL program developer). This identifier is automatically added to the CHARTEVENT\_CUSTOM value and converted to an integer type. This value will be passed to the OnChartEvent handler as the first argument. Other parameters of EventChartCustom correspond to the standard event parameters in OnChartEvent with types long, double and string, and may contain arbitrary information.

The function returns true in case of successful queuing of the user event or false in case of an error (the error code will become available in \_LastError).

As we approach the most complex and important part of our book devoted directly to trading automation, we will begin to solve applied problems that will be useful in the development of trading robots. Now, in the context of demonstrating the capabilities of custom events, let's turn to the multicurrency (or, more generally, multisymbol) analysis of the trading environment.

A little earlier, in the chapter on indicators, we considered [multicurrency indicators](/en/book/applications/indicators_make/indicators_multisymbol) but did not pay attention to an important point: despite the fact that the indicators processed quotes of different symbols, the calculation itself was launched in the OnCalculate handler, which was triggered by the arrival of a new tick of only one symbol – the working symbol of the chart. It turns out that the ticks of other instruments are essentially skipped. For example, if the indicator works on symbol A, when its tick arrives, we simply take the last known ticks of other symbols (B, C, D), but it is likely that other ticks managed to slip through each of them.

If you place a multicurrency indicator on the most liquid instrument (where ticks are received most often), this is not so critical. However, different instruments can be faster than others at different times of the day, and if an analytical or trading algorithm requires the fastest possible response to new quotes of all instruments in the portfolio, we are faced with the fact that the current solution does not suit us.

Unfortunately, the standard event of a new tick arrival works in MQL5 only for one symbol, which is the working symbol of the current chart. In indicators, the [OnCalculate](/en/book/applications/indicators_make/indicators_oncalculate) handler is called at such moments, and the [OnTick](/en/book/automation/experts/experts_ontick) handler is called in Expert Advisors.

Therefore, it is necessary to invent some mechanism so that the MQL program can receive notifications about ticks on all instruments of interest. This is where custom events will help us. Of course, this is not necessary for programs that analyze only one instrument.  

We will now develop an example of the EventTickSpy.mq5 indicator, which, being launched on a specific symbol X, will be able to send tick notifications from its OnCalculate function using EventChartCustom. As a result, in the handler OnChartEvent, which is specially prepared to receive such notifications, it will be possible to collect notifications from different instances of the indicator from different symbols.

This example is provided for illustration purposes. Subsequently, when studying multicurrency automated trading, we will adapt this technique for more convenient use in Expert Advisors.

First of all, let's think of a custom event number for the indicator. Since we are going to send tick notifications for many different symbols from some given list, we can choose different tactics here. For example, you can select one event identifier, and pass the number of the symbol in the list and/or the name of the symbol in the lparam and sparam parameters, respectively. Or you can take some constant (greater than and equal to CHARTEVENT\_CUSTOM) and get event numbers by adding the symbol number to this constant (then we have all parameters free, in particular, lparam and dparam, and they can be used to transfer prices Ask, Bid or something else).

We will focus on the option when there is one event code. Let's declare it in the TICKSPY macro. This will be the default value, which the user can change to avoid collisions (albeit unlikely) with other programs if necessary.

| |
| --- |
| #define TICKSPY 0xFEED // 65261 |

This value is taken on purpose as being rather far removed from the first allowed CHARTEVENT\_CUSTOM.

During the initial (interactive) launch of the indicator, the user must specify the list of instruments whose ticks the indicator should track. For this purpose, we will describe the input string variable SymbolList with a comma-separated list of symbols.

The identifier of the user event is set in the message parameter.

Finally, we need the identifier of the receiving chart to pass the event. We will provide the Chart parameter for this purpose. The user should not edit it: in the first instance of the indicator launched manually, the chart is known implicitly by attaching it to the chart. In other copies of the indicator that our first instance will run programmatically, this parameter will fill the algorithm with a call of the function ChartID (see below).

| |
| --- |
| input string SymbolList = "EURUSD,GBPUSD,XAUUSD,USDJPY"; // List of symbols separated by commas (example) input ushort message = TICKSPY;                          // Custom message input longchart = 0;                                     // Receiving chart (do not edit) |

In the SymbolList parameter, for example, a list with four common tools is indicated. Edit it as needed to suit your Market Watch.

In the OnInit handler, we convert the list to the Symbols array of symbols, and then in a loop we run the same indicator for all symbols from the array, except for the current one (as a rule, there is such a match, because the current symbol is already being processed by this initial copy of the indicator).

| |
| --- |
| string Symbols[];     void OnInit() {    PrintFormat("Starting for chart %lld, msg=0x%X [%s]", Chart, Message, SymbolList);    if(Chart == 0)    {       if(StringLen(SymbolList) > 0)       {          const int n = StringSplit(SymbolList, ',', Symbols);          for(int i = 0; i < n; ++i)          {             if(Symbols[i] != \_Symbol)             {                ResetLastError();                // run the same indicator on another symbol with different settings,                // in particular, we pass our ChartID to receive notifications back                iCustom(Symbols[i], PERIOD\_CURRENT, MQLInfoString(MQL\_PROGRAM\_NAME),                   "", Message, ChartID());                if(\_LastError != 0)                {                   PrintFormat("The symbol '%s' seems incorrect", Symbols[i]);                }             }          }       }       else       {          Print("SymbolList is empty: tracking current symbol only!");          Print("To monitor other symbols, fill in SymbolList, i.e."             " 'EURUSD,GBPUSD,XAUUSD,USDJPY'");       }    } } |

At the beginning of OnInit, information about the launched instance of the indicator is displayed in the log so that it is clear what is happening.

If we chose the option with separate event codes for each character, we would have to call iCustom as follows (addi to message):

| |
| --- |
| iCustom(Symbols[i], PERIOD\_CURRENT, MQLInfoString(MQL\_PROGRAM\_NAME), "",       Message + i, ChartID()); |

Note that the non-zero value of the Chart parameter implies that this copy is launched programmatically and that it should monitor a single symbol, that is, the working symbol of the chart. Therefore, we don't need to pass a list of symbols when running the slave copies.

In the OnCalculate function, which is called when a new tick is received, we send the Message custom event to the Chart chart by calling EventChartCustom. In this case, the lparam parameter is not used (equal to 0). In the dparam parameter, we pass the current (last) price price[0] (this is Bid or Last, depending on what type of price the chart is based on: it is also the price of the last tick processed by the chart), and we pass the symbol name in the sparam parameter.

| |
| --- |
| int OnCalculate(const int rates\_total, const int prev\_calculated,    const int, const double &price[]) {    if(prev\_calculated)    {       ArraySetAsSeries(price, true);       if(Chart > 0)       {          // send a tick notification to the parent chart          EventChartCustom(Chart, Message, 0, price[0], \_Symbol);       }       else       {          OnSymbolTick(\_Symbol, price[0]);       }    }       return rates\_total; } |

In the original instance of the indicator, where the Chart parameter is 0, we directly call a special function, a kind of a multiasset tick handler OnSymbolTick. In this case, there is no need to call EventChartCustom: although such a message will still arrive on the chart and this copy of the indicator, the transmission takes several milliseconds and loads the queue in vain.

The only purpose of OnSymbolTick in this demo is to print the name of the symbol and the new price in the log.

| |
| --- |
| void OnSymbolTick(const string &symbol, const double price) {    Print(symbol, " ", DoubleToString(price,       (int)SymbolInfoInteger(symbol, SYMBOL\_DIGITS))); } |

Of course, the same function is called from the OnChartEvent handler in the receiving (source) copy of the indicator, provided that our message has been received. Recall that the terminal calls OnChartEvent only in the interactive copy of the indicator (applied to the chart) and does not appear in those copies that we created "invisible" using iCustom.

| |
| --- |
| void OnChartEvent(const int id,    const long &lparam, const double &dparam, const string &sparam) {    if(id >= CHARTEVENT\_CUSTOM + Message)    {       OnSymbolTick(sparam, dparam);       // OR (if using custom event range):       // OnSymbolTick(Symbols[id - CHARTEVENT\_CUSTOM - Message], dparam);    } } |

We could avoid sending either the price or the name of the symbol in our event since the general list of symbols is known in the initial indicator (which initiated the process), and therefore we could somehow tell it the number of the symbol from the list. This could be done in the lparam parameter or, as mentioned above, by adding a number to the base constant of the user event. Then the original indicator, while receiving events, could take a symbol by index from the array and get all the information about the last tick using [SymbolInfoTick](/en/book/automation/symbols/symbols_tick), including different types of prices.

Let's run the indicator on the EURUSD chart with default settings, including the "EURUSD,GBPUSD,XAUUSD,USDJPY" test list. Here is the log:

| |
| --- |
| 16:45:48.745 (EURUSD,H1) Starting for chart 0, msg=0xFEED [EURUSD,GBPUSD,XAUUSD,USDJPY] 16:45:48.761 (GBPUSD,H1) Starting for chart 132358585987782873, msg=0xFEED [] 16:45:48.761 (USDJPY,H1) Starting for chart 132358585987782873, msg=0xFEED [] 16:45:48.761 (XAUUSD,H1) Starting for chart 132358585987782873, msg=0xFEED [] 16:45:48.777 (EURUSD,H1) XAUUSD 1791.00 16:45:49.120 (EURUSD,H1) EURUSD 1.13068 \* 16:45:49.135 (EURUSD,H1) USDJPY 115.797 16:45:49.167 (EURUSD,H1) XAUUSD 1790.95 16:45:49.167 (EURUSD,H1) USDJPY 115.796 16:45:49.229 (EURUSD,H1) USDJPY 115.797 16:45:49.229 (EURUSD,H1) XAUUSD 1790.74 16:45:49.369 (EURUSD,H1) XAUUSD 1790.77 16:45:49.572 (EURUSD,H1) GBPUSD 1.35332 16:45:49.572 (EURUSD,H1) XAUUSD 1790.80 16:45:49.791 (EURUSD,H1) XAUUSD 1790.80 16:45:49.791 (EURUSD,H1) USDJPY 115.796 16:45:49.931 (EURUSD,H1) EURUSD 1.13069 \* 16:45:49.931 (EURUSD,H1) XAUUSD 1790.86 16:45:49.931 (EURUSD,H1) USDJPY 115.795 16:45:50.056 (EURUSD,H1) USDJPY 115.793 16:45:50.181 (EURUSD,H1) XAUUSD 1790.88 16:45:50.321 (EURUSD,H1) XAUUSD 1790.90 16:45:50.399 (EURUSD,H1) EURUSD 1.13066 \* 16:45:50.727 (EURUSD,H1) EURUSD 1.13067 \* 16:45:50.773 (EURUSD,H1) GBPUSD 1.35334 |

Please note that in the column with (symbol,timeframe) which is the source of the record, we first see the starting indicator instances on four requested symbols.

After launch, the first tick was XAUUSD, not EURUSD. Further symbol ticks come with approximately equal intensity, interspersed. EURUSD ticks are marked with asterisks, so you can get an idea of how many other ticks would have been missed without notifications.

Timestamps have been saved in the left column for reference.

Places where two prices of two consecutive events from the same symbol coincide usually indicate that the Ask price has changed (we simply do not display it here).

A little later, after studying the trading MQL5 API, we will apply the same principle to respond to multicurrency ticks in Expert Advisors.

[Graphical object events](/en/book/applications/events/events_objects "Graphical object events")

[Trading automation](/en/book/automation "Trading automation")