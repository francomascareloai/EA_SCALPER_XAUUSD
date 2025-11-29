# MQL5 Book - Part 11 (Pages 2001-2047)

## Page 2001

Part 7. Advanced language tools
2001 
7.9 Native python support
7.9.2 Overview of functions of the MetaTrader5 package for Python
The API functions available in Python can be conditionally divided into 2 groups: functions that have full
analogs in the MQL5 API and functions available only in Python. The presence of the second group is
partly due to the fact that the connection between Python and MetaTrader 5 must be technically
organized before application functions can be used. This explains the presence and purpose of a pair of
functions initialize and shutdown: the first establishes a connection to the terminal, and the second one
terminates it.
It is important that during the initialization process, the required copy of the terminal can be launched
(if it has not been executed yet) and a specific trading account can be selected. In addition, it is
possible to change the trading account in the context of an already opened connection to the terminal:
this is done by the login function.
After connecting to the terminal, a Python script can get a summary of the terminal version using the
version function. Full information about the terminal is available through terminal_ info which is a
complete analog of three TerminalInfo functions, as if they were united in one call.
The following table lists the Python application functions and their counterparts in the MQL5 API.
Python
MQL5
last_error
GetLastError (Attention! Python has its native error codes)
account_info
AccountInfoInteger, AccountInfoDouble, AccountInfoString
terminal_info
TerminalInfoInteger, TerminalInfoDouble, TerminalInfoDouble
symbols_total
SymbolsTotal (all symbols, including custom and disabled)
symbols_get
SymbolsTotal + SymbolInfo functions
symbol_info
SymbolInfoInteger, SymbolInfoDouble, SymbolInfoString
symbol_info_tick
SymbolInfoTick
symbol_select
SymbolSelect
market_book_add
MarketBookAdd
market_book_get
MarketBookGet
market_book_release
MarketBookRelease
copy_rates_from
CopyRates (by the number of bars, starting from date/time)
copy_rates_from_pos
CopyRates (by the number of bars, starting from the bar number)
copy_rates_range
CopyRates (in the date/time range)
copy_ticks_from
CopyTicks (by the number of ticks, starting from the specified time)
copy_ticks_range
CopyTicksRange (in the specified time range)
orders_total
OrdersTotal

---

## Page 2002

Part 7. Advanced language tools
2002
7.9 Native python support
Python
MQL5
orders_get
OrdersTotal + OrderGet functions
order_calc_margin
OrderCalcMargin
order_calc_profit
OrderCalcProfit
order_check
OrderCheck
order_send
OrderSend
positions_total
PositionsTotal
positions_get
PositionsTotal + PositionGet functions
history_orders_total
HistoryOrdersTotal
history_orders_get
HistoryOrdersTotal + HistoryOrderGet functions
history_deals_total
HistoryDealsTotal
history_deals_get
HistoryDealsTotal + HistoryDealGet functions
Functions from the Python API have several features.
As already noted, functions can have named parameters: when a function is called, such parameters
are specified together with a name and value, in each pair of name and value they are combined with
the equal sign '='. The order of specifying named parameters is not important (unlike positional
parameters, which are used in MQL5 and must follow the strict order specified by the function
prototype).
Python functions operate on data types native to Python. This includes not only the usual numbers and
strings but also several composite types, somewhat similar to MQL5 arrays and structures.
For example, many functions return special Python data structures: tuple and namedtuple.
A tuple is a sequence of elements of an arbitrary type. It can be thought of as an array, but unlike an
array, the elements of a tuple can be of different types. You can also think of a tuple as a set of
structure fields.
An even closer resemblance to structure can be found with named tuples, where each element is given
an ID. Only an index can be used to access an element in a common tuple (in square brackets, as in
MQL5, that is, [i]). However, we can apply the dereference operator (dot '.') to a named tuple to get
its "property " just like in the MQL5 structure (tuple.field).
Also, tuples and named tuples cannot be edited in code (that is, they are constants).
Another popular type is a dictionary: an associative array that stores key and value pairs, and the types
of both can vary. The dictionary value is accessed using the operator [], and the key (whatever type it
is, for example, a string) is indicated between the square brackets, which makes dictionaries similar to
arrays. A dictionary cannot have two pairs with the same key, that is, the keys are always unique. In
particular, a named tuple can easily be turned into a dictionary using the method namedtuple._ asdict().

---

## Page 2003

Part 7. Advanced language tools
2003
7.9 Native python support
7.9.3 Connecting a Python script to the terminal and account
The initialize function establishes a connection with the MetaTrader 5 terminal and has 2 forms: short
(without parameters) and full (with several optional parameters, the first of them is path and it is
positional, and all the rest are named).
bool initialize()
bool initialize(path, account = <ACCOUNT>, password = <"PASSWORD">,
   server = <"SERVER">, timeout = 60000, portable = False)
The path parameter sets the path to the terminal file (metatrader64.exe) (note that this is an unnamed
parameter, unlike all the others, so if it is specified, it must come first in the list).
If the path is not specified, the module will try to find the executable file on its own (the developers do
not disclose the exact algorithm). To eliminate ambiguities, use the second form of the function with
parameters.
In the account parameter, you can specify the number of the trading account. If it is not specified,
then the last trading account in the selected instance of the terminal will be used.
The password for the trading account is specified in the password parameter and can also be omitted:
in this case, the password stored in the terminal database for the specified trading account is
automatically substituted.
The server parameter is processed in a similar way with the trade server name (as it is specified in the
terminal): if it is not specified, then the server saved in the terminal database for the specified trading
account is automatically substituted.
The timeout parameter indicates the timeout in milliseconds that is given for the connection (if it is
exceeded, an error will occur). The default value is 60000 (60 seconds).  
The portable parameter contains a flag for launching the terminal in the portable mode (default is
False).
The function returns True in case of successful connection to the MetaTrader 5 terminal and False
otherwise.
If necessary, when making a call initialize, the MetaTrader 5 terminal can be launched.
For example, connection to a specific trading account is performed as follows.
import MetaTrader5 as mt5 
if not mt5.initialize(login = 562175752, server = "MetaQuotes-Demo", password = "abc"):
   print("initialize() failed, error code =", mt5.last_error()) 
   quit()
...
The login function also connects to the trading account with the specified parameters. But this implies
that the connection with the terminal has already been established, that is, the function is usually used
to change the account.
bool login(account, password = <"PASSWORD">, server = <"SERVER">, timeout = 60000)
The trading account number is provided in the account parameter. This is a required unnamed
parameter, meaning it must come first in the list.

---

## Page 2004

Part 7. Advanced language tools
2004
7.9 Native python support
The password, server, and timeout parameters are identical to the relevant parameters of the initialize
function.
The function returns True in case of successful connection to the trading account and False otherwise.
shutdown()
The shutdown function closes the previously established connection to the MetaTrader 5 terminal.
The example for the above functions will be provided in the next section.
When the connection is established, the script can find the version of the terminal.
tuple version()
The version function returns brief information about the version of the MetaTrader 5 terminal as a tuple
of three values: version number, build number, and build date.
Field type
Description
integer
MetaTrader 5 terminal version (current, 500)
integer
Build number (for example, 3456)
string
Build date (e.g. '25 Feb 2022')
In case of an error, the function returns None, and the error code can be obtained using last_ error.
More complete information about the terminal can be obtained using the terminal_ info function.
7.9.4 Error checking: last_error
The last_ error function returns information about the last Python error.
int last_error()
Integer error codes differ from the codes that are allocated for MQL5 errors and returned by the
standard GetLastError function. In the following table, the abbreviation IPC refers to the term "Inter-
Process Communication".

---

## Page 2005

Part 7. Advanced language tools
2005
7.9 Native python support
Constant
Meaning
Description
R E S _S _O K 
1
S u c c e ss
R E S _E _F AIL 
-1 
Com m on  e r r or 
R E S _E _IN VAL ID _P AR AM S 
-2 
In va l i d  a r g u m e n t s/p a r a m e t e r s
R E S _E _N O _M E M O R Y
-3 
M e m or y a l l oc a t i on  e r r or 
R E S _E _N O T_F O U N D 
-4 
R e q u e st e d  h i st or y n ot  fou n d 
R E S _E _IN VAL ID _VE R S IO N 
-5 
Ve r si on  n ot  su p p or t e d 
R E S _E _AU TH _F AIL E D 
-6 
Au t h or i z a t i on  e r r or 
R E S _E _U N S U P P O R TE D 
-7 
M e t h od  n ot  su p p or t e d 
R E S _E _AU TO _TR AD IN G _D IS AB L E D 
-8 
Al g o t r a d i n g  i s d i sa b l e d 
R E S _E _IN TE R N AL _F AIL 
-1 0 0 0 0 
G e n e r a l  i n t e r n a l  IP C e r r or 
R E S _E _IN TE R N AL _F AIL _S E N D 
-1 0 0 0 1 
In t e r n a l  e r r or  se n d i n g  IP C d a t a 
R E S _E _IN TE R N AL _F AIL _R E CE IVE 
-1 0 0 0 2 
In t e r n a l  e r r or  se n d i n g  IP C d a t a 
R E S _E _IN TE R N AL _F AIL _IN IT
-1 0 0 0 3 
IP C i n t e r n a l  i n i t i a l i z a t i on  e r r or 
R E S _E _IN TE R N AL _F AIL _CO N N E CT
-1 0 0 0 3 
N o IP C
R E S _E _IN TE R N AL _F AIL _TIM E O U T
-1 0 0 0 5 
IP C t i m e ou t 
In the following script (MQL5/Scripts/MQL5Book/Python/init.py), in the case of an error when
connecting to the terminal, we display the error code and exit.
import MetaTrader5 as mt5
# show MetaTrader5 package version
print("MetaTrader5 package version: ", mt5.__version__)  #  5.0.37
   
# let's try to establish a connection or launch the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error()) 
   quit()
... # the working part of the script will be here
# terminate the connection to the terminal
mt5.shutdown()
7.9.5 Getting information about a trading account
The account_ info function obtains full information about the current trading account.
namedtuple account_info()
The function returns information as a structure of named tuples (namedtuple). In case of an error, the
result is None.

---

## Page 2006

Part 7. Advanced language tools
2006
7.9 Native python support
Using this function, you can use one call to get all the information that is provided by
AccountInfoInteger, AccountInfoDouble, and AccountInfoString in MQL5, with all variants of supported
properties. The names of the fields in the tuple correspond to the names of the enumeration elements
without the "ACCOUNT_" prefix, reduced to lowercase.
The following script MQL5/Scripts/MQL5Book/Python/accountinfo.py is included with the book.
import MetaTrader5 as mt5
  
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize(): 
   print("initialize() failed, error code =", mt5.last_error())
   quit()
   
account_info = mt5.account_info()
if account_info != None:
   # display trading account data as is
   print(account_info) 
   # display data about the trading account in the form of a dictionary
   print("Show account_info()._asdict():")
   account_info_dict = mt5.account_info()._asdict()
   for prop in account_info_dict:
      print("  {}={}".format(prop, account_info_dict[prop]))
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
The result should be something like this.

---

## Page 2007

Part 7. Advanced language tools
2007
7.9 Native python support
AccountInfo(login=25115284, trade_mode=0, leverage=100, limit_orders=200, margin_so_mode=0, ... 
Show account_info()._asdict(): 
  login=25115284 
  trade_mode=0 
  leverage=100 
  limit_orders=200 
  margin_so_mode=0 
  trade_allowed=True 
  trade_expert=True 
  margin_mode=2 
  currency_digits=2 
  fifo_close=False 
  balance=99511.4 
  credit=0.0 
  profit=41.82 
  equity=99553.22 
  margin=98.18 
  margin_free=99455.04 
  margin_level=101398.67590140559 
  margin_so_call=50.0 
  margin_so_so=30.0 
  margin_initial=0.0 
  margin_maintenance=0.0 
  assets=0.0 
  liabilities=0.0 
  commission_blocked=0.0 
  name=MetaQuotes Dev Demo 
  server=MetaQuotes-Demo 
  currency=USD 
  company=MetaQuotes Software Corp. 
7.9.6 Getting information about the terminal
The terminal_ info function allows you to get the status and parameters of the connected MetaTrader 5
terminal.
namedtuple terminal_info()
On success, the function returns the information as a structure of named tuples (namedtuple), and in
case of an error, it returns None.
In one call of this function, you can get all the information that is provided by TerminalInfoInteger,
TerminalInfoDouble, and TerminalInfoDouble in MQL5, with all variants of supported properties. The
names of the fields in the tuple correspond to the names of the enumeration elements without the
"TERMINAL_" prefix, reduced to lowercase.
For example (see MQL5/Scripts/MQL5Book/Python/terminalinfo.py):

---

## Page 2008

Part 7. Advanced language tools
2008
7.9 Native python support
import MetaTrader5 as mt5
  
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   quit() 
   
# display brief information about the MetaTrader 5 version
print(mt5.version()) 
# display full information about the settings and the state of the terminal
terminal_info = mt5.terminal_info()
if terminal_info != None: 
   # display terminal data as is
   print(terminal_info) 
   # display the data as a dictionary
   print("Show terminal_info()._asdict():")
   terminal_info_dict = mt5.terminal_info()._asdict()
   for prop in terminal_info_dict: 
      print("  {}={}".format(prop, terminal_info_dict[prop]))
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown() 
We should be something like the following.
[500, 3428, '14 Sep 2022'] 
TerminalInfo(community_account=True, community_connection=True, connected=True,.... 
Show terminal_info()._asdict(): 
  community_account=True 
  community_connection=True 
  connected=True 
  dlls_allowed=False 
  trade_allowed=False 
  tradeapi_disabled=False 
  email_enabled=False 
  ftp_enabled=False 
  notifications_enabled=False 
  mqid=False 
  build=2366 
  maxbars=5000 
  codepage=1251 
  ping_last=77850 
  community_balance=707.10668201585 
  retransmission=0.0 
  company=MetaQuotes Software Corp. 
  name=MetaTrader 5 
  language=Russian 
  path=E:\ProgramFiles\MetaTrader 5 
  data_path=E:\ProgramFiles\MetaTrader 5 
  commondata_path=C:\Users\User\AppData\Roaming\MetaQuotes\Terminal\Common 

---

## Page 2009

Part 7. Advanced language tools
2009
7.9 Native python support
7.9.7 Getting information about financial instruments
The group of functions of the MetaTrader5 package provides information about financial instruments.
The symbol_ info function returns information about one financial instrument as a named tuple
structure.
namedtuple symbol_info(symbol)
The name of the desired financial instrument is specified in the symbol parameter.
One call provides all the information that can be obtained using three MQL5 functions
SymbolInfoInteger, SymbolInfoDouble, and SymbolInfoString with all properties. The names of the fields
in the named tuple are the same as the names of the enumeration elements used in the specified
functions but without the "SYMBOL_" prefix and in lowercase.
In case of an error, the function returns None.
Attention! To ensure successful function execution, the requested symbol must be selected in
Market Watch. This can be done from Python by calling symbol_ select (see further).
Example (MQL5/Scripts/MQL5Book/Python/eurj py.py):
import MetaTrader5 as mt5
   
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   quit()
   
# make sure EURJPY is present in the Market Watch, or abort the algorithm
selected = mt5.symbol_select("EURJPY", True)
if not selected:
   print("Failed to select EURJPY")
   mt5.shutdown()
   quit()
   
# display the properties of the EURJPY symbol
symbol_info = mt5.symbol_info("EURJPY")
if symbol_info != None:
   # display the data as is (as a tuple)
   print(symbol_info)
   # output a couple of specific properties
   print("EURJPY: spread =", symbol_info.spread, ", digits =", symbol_info.digits)
   # output symbol properties as a dictionary
   print("Show symbol_info(\"EURJPY\")._asdict():")
   symbol_info_dict = mt5.symbol_info("EURJPY")._asdict()
   for prop in symbol_info_dict:
      print("  {}={}".format(prop, symbol_info_dict[prop]))
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
Result:

---

## Page 2010

Part 7. Advanced language tools
201 0
7.9 Native python support
SymbolInfo(custom=False, chart_mode=0, select=True, visible=True, session_deals=0, session_buy_orders=0, session_sell_orders=0, ... 
EURJPY: spread = 17, digits = 3 
Show symbol_info()._asdict(): 
  custom=False 
  chart_mode=0 
  select=True 
  visible=True 
  ...
  time=1585069682 
  digits=3 
  spread=17 
  spread_float=True 
  ticks_bookdepth=10 
  trade_calc_mode=0 
  trade_mode=4 
  ...
  trade_exemode=1 
  swap_mode=1 
  swap_rollover3days=3 
  margin_hedged_use_leg=False 
  expiration_mode=7 
  filling_mode=1 
  order_mode=127 
  order_gtc_mode=0 
  ...
  bid=120.024 
  ask=120.041 
  last=0.0 
  ...
  point=0.001 
  trade_tick_value=0.8977708350166538 
  trade_tick_value_profit=0.8977708350166538 
  trade_tick_value_loss=0.8978272580355541 
  trade_tick_size=0.001 
  trade_contract_size=100000.0 
  ...
  volume_min=0.01 
  volume_max=500.0 
  volume_step=0.01 
  volume_limit=0.0 
  swap_long=-0.2 
  swap_short=-1.2 
  margin_initial=0.0 
  margin_maintenance=0.0 
  margin_hedged=100000.0 
  ...
  currency_base=EUR 
  currency_profit=JPY 
  currency_margin=EUR 
  ...
bool symbol_select(symbol, enable = None)

---

## Page 2011

Part 7. Advanced language tools
201 1 
7.9 Native python support
The symbol_ select function adds the specified symbol to Market Watch or removes it. The symbol is
specified in the first parameter. The second parameter is passed as True or False, which means
showing or hiding the symbol, respectively.
If the second optional unnamed parameter is omitted, then by Python's type casting rules, bool(none)
is equivalent to False.
The function is an analog of SymbolSelect.
int symbols_total()
The symbols_ total function returns the number of all instruments in the MetaTrader 5 terminal, taking
into account custom symbols and those not currently shown in the Market Watch window. This is the
analog of the function SymbolsTotal(false).
Next symbols_ get function returns an array of tuples with information about all instruments or favorite
instruments with names matching the specified filter in the optional named parameter group.
tuple[] symbols_get(group = "PATTERN")
Each element in the array tuple is a named tuple with a full set of symbol properties (we saw a similar
tuple above in the context of the description of the symbol_ info function).
Since there is only one parameter, its name can be omitted when calling the function.
In case of an error, the function will return a special value of None.
The group parameter allows you to select symbols by name, optionally using the substitution (wildcard)
character '*' at the beginning and/or end of the searched string. '*' means 0 or any number of
characters. Thus, you can organize a search for a substring that occurs in the name with an arbitrary
number of other characters before or after the specified fragment. For example, "EUR*" means
symbols that start with "EUR" and have any name extension (or just "EUR"). The "*EUR*" filter will
return symbols with the names containing the "EUR" substring anywhere.
Also, the group parameter may contain multiple conditions separated by commas. Each condition can
be specified as a mask using '*'. To exclude symbols, you can use the logical negation sign '!'. In this
case, all conditions are applied sequentially, i.e., first you need to specify the inclusion conditions, and
then the exclusion conditions. For example, group="*, !*EUR*" means that we need to select all
symbols first and then exclude those that contain "EUR" in the name (anywhere).
For example, to display information about cross-currency rates, except for the 4 major Forex
currencies, you can run the following query:
crosses = mt5.symbols_get(group = "*,!*USD*,!*EUR*,!*JPY*,!*GBP*")
print('len(*,!*USD*,!*EUR*,!*JPY*,!*GBP*):', len(crosses)) # the size of the resulting array - the number of crosses
for s in crosses: 
   print(s.name, ":", s) 
An example of the result:

---

## Page 2012

Part 7. Advanced language tools
201 2
7.9 Native python support
len(*,!*USD*,!*EUR*,!*JPY*,!*GBP*):  10 
AUDCAD : SymbolInfo(custom=False, chart_mode=0, select=True, visible=True, session_deals=0, session_buy_orders=0, session... 
AUDCHF : SymbolInfo(custom=False, chart_mode=0, select=True, visible=True, session_deals=0, session_buy_orders=0, session... 
AUDNZD : SymbolInfo(custom=False, chart_mode=0, select=True, visible=True, session_deals=0, session_buy_orders=0, session... 
CADCHF : SymbolInfo(custom=False, chart_mode=0, select=False, visible=False, session_deals=0, session_buy_orders=0, sessi... 
NZDCAD : SymbolInfo(custom=False, chart_mode=0, select=False, visible=False, session_deals=0, session_buy_orders=0, sessi... 
NZDCHF : SymbolInfo(custom=False, chart_mode=0, select=False, visible=False, session_deals=0, session_buy_orders=0, sessi... 
NZDSGD : SymbolInfo(custom=False, chart_mode=0, select=False, visible=False, session_deals=0, session_buy_orders=0, sessi... 
CADMXN : SymbolInfo(custom=False, chart_mode=0, select=False, visible=False, session_deals=0, session_buy_orders=0, sessi... 
CHFMXN : SymbolInfo(custom=False, chart_mode=0, select=False, visible=False, session_deals=0, session_buy_orders=0, sessi... 
NZDMXN : SymbolInfo(custom=False, chart_mode=0, select=False, visible=False, session_deals=0, session_buy_orders=0, sessi... 
The symbol_ info_ tick function can be used to get the last tick for the specified financial instrument.
tuple symbol_info_tick(symbol)
The only mandatory parameter specifies the name of the financial instrument.
The information is returned as a tuple with the same fields as in the MqlTick structure. The function is
an analog of SymbolInfoTick.
None is returned if an error occurs.
For the function to work properly, the symbol must be enabled in Market Watch. Let's demonstrate it in
the script MQL5/Scripts/MQL5Book/Python/gbpusdtick.py.
import MetaTrader5 as mt5
   
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   quit()
   
# try to include the GBPUSD symbol in the Market Watch
selected=mt5.symbol_select("GBPUSD", True)
if not selected:
   print("Failed to select GBPUSD")
   mt5.shutdown()
   quit()
   
# display the last tick of the GBPUSD symbol as a tuple
lasttick = mt5.symbol_info_tick("GBPUSD")
print(lasttick)
# display the values of the tick fields in the form of a dictionary
print("Show symbol_info_tick(\"GBPUSD\")._asdict():")
symbol_info_tick_dict = lasttick._asdict()
for prop in symbol_info_tick_dict:
   print("  {}={}".format(prop, symbol_info_tick_dict[prop]))
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
The result should be as follows:

---

## Page 2013

Part 7. Advanced language tools
201 3
7.9 Native python support
Tick(time=1585070338, bid=1.17264, ask=1.17279, last=0.0, volume=0, time_msc=1585070338728, flags=2, volume_real=0.0) 
Show symbol_info_tick._asdict(): 
  time=1585070338 
  bid=1.17264 
  ask=1.17279 
  last=0.0 
  volume=0 
  time_msc=1585070338728 
  flags=2 
  volume_real=0.0
7.9.8 Subscribing to order book changes
The Python API includes three functions for working with the order book.
bool market_book_add(symbol)
The market_ book_ add function subscribes to receive events about order book changes for the specified
symbol. The name of the required financial instrument is indicated in a single unnamed parameter.
The function returns a boolean success indication.
The function is an analog of MarketBookAdd. After completing work with the order book, the
subscription should be canceled by calling market_ book_ release (see further).
tuple[] market_book_get(symbol)
The market_ book_ get function requests the current contents of the order book for the specified
symbol. The result is returned as a tuple (array) of BookInfo records. Each entry is an analog of the
MqlBookInfo structure, and from the Python point of view, this is a named tuple with the fields "type",
"price", "volume", "volume_real". In case of an error, the None value is returned.
Note that for some reason in Python, the field is called volume_ dbl, although in MQL5 the
corresponding field is called volume_ real.
To work with this function, you must first subscribe to receive order book events using the
market_ book_ add function.
The function is an analog of MarketBookGet. Please note that a Python script cannot receive
OnBookEvent events directly and should poll the contents of the glass in a loop.
bool market_book_release(symbol)
The market_ book_ release function cancels the subscription for order book change events for the
specified symbol. On success, the function returns True. The function is an analog of
MarketBookRelease.
Let's take a simple example (see MQL5/Scripts/MQL5Book/Python/eurusdbook.py).

---

## Page 2014

Part 7. Advanced language tools
201 4
7.9 Native python support
import MetaTrader5 as mt5
import time               # connect a pack for the pause
   
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   mt5.shutdown()
   quit()
   
# subscribe to receive DOM updates for the EURUSD symbol
if mt5.market_book_add('EURUSD'):
   # run 10 times a loop to read data from the order book
   for i in range(10):
      # get the contents of the order book
      items = mt5.market_book_get('EURUSD')
      # display the entire order book in one line as is
      print(items)
      # now display each price level separately in the form of a dictionary, for clarity
      for it in items or []:
         print(it._asdict())
      # let's pause for 5 seconds before the next request for data from the order book
      time.sleep(5) 
   # unsubscribe to order book changes
   mt5.market_book_release('EURUSD')
else:
   print("mt5.market_book_add('EURUSD') failed, error code =", mt5.last_error())
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
An example of the result:

---

## Page 2015

Part 7. Advanced language tools
201 5
7.9 Native python support
(BookInfo(type=1, price=1.20036, volume=250, volume_dbl=250.0), BookInfo(type=1, price=1.20029, volume=100, volume...
{'type': 1, 'price': 1.20036, 'volume': 250, 'volume_dbl': 250.0}
{'type': 1, 'price': 1.20029, 'volume': 100, 'volume_dbl': 100.0}
{'type': 1, 'price': 1.20028, 'volume': 50, 'volume_dbl': 50.0}
{'type': 1, 'price': 1.20026, 'volume': 36, 'volume_dbl': 36.0}
{'type': 2, 'price': 1.20023, 'volume': 36, 'volume_dbl': 36.0}
{'type': 2, 'price': 1.20022, 'volume': 50, 'volume_dbl': 50.0}
{'type': 2, 'price': 1.20021, 'volume': 100, 'volume_dbl': 100.0}
{'type': 2, 'price': 1.20014, 'volume': 250, 'volume_dbl': 250.0}
(BookInfo(type=1, price=1.20035, volume=250, volume_dbl=250.0), BookInfo(type=1, price=1.20029, volume=100, volume...
{'type': 1, 'price': 1.20035, 'volume': 250, 'volume_dbl': 250.0}
{'type': 1, 'price': 1.20029, 'volume': 100, 'volume_dbl': 100.0}
{'type': 1, 'price': 1.20027, 'volume': 50, 'volume_dbl': 50.0}
{'type': 1, 'price': 1.20025, 'volume': 36, 'volume_dbl': 36.0}
{'type': 2, 'price': 1.20023, 'volume': 36, 'volume_dbl': 36.0}
{'type': 2, 'price': 1.20022, 'volume': 50, 'volume_dbl': 50.0}
{'type': 2, 'price': 1.20021, 'volume': 100, 'volume_dbl': 100.0}
{'type': 2, 'price': 1.20014, 'volume': 250, 'volume_dbl': 250.0}
(BookInfo(type=1, price=1.20037, volume=250, volume_dbl=250.0), BookInfo(type=1, price=1.20031, volume=100, volume...
{'type': 1, 'price': 1.20037, 'volume': 250, 'volume_dbl': 250.0}
{'type': 1, 'price': 1.20031, 'volume': 100, 'volume_dbl': 100.0}
{'type': 1, 'price': 1.2003, 'volume': 50, 'volume_dbl': 50.0}
{'type': 1, 'price': 1.20028, 'volume': 36, 'volume_dbl': 36.0}
{'type': 2, 'price': 1.20025, 'volume': 36, 'volume_dbl': 36.0}
{'type': 2, 'price': 1.20023, 'volume': 50, 'volume_dbl': 50.0}
{'type': 2, 'price': 1.20022, 'volume': 100, 'volume_dbl': 100.0}
{'type': 2, 'price': 1.20016, 'volume': 250, 'volume_dbl': 250.0}
...
7.9.9 Reading quotes
The Python API allows you to get arrays of prices (bars) using three functions that differ in the way you
specify the range of requested data: by bar numbers or by time. All functions are similar to different
forms of CopyRates.
For all functions, the first two parameters are used to specify the name of the symbol and timeframe.
The timeframes are listed in the TIMEFRAME enumeration, which is similar to the enumeration
ENUM_TIMEFRAMES in MQL5.
Please note: In Python, the elements of this enumeration are prefixed with TIMEFRAME_, while the
elements of a similar enumeration in MQL5 are prefixed with PERIOD_.
Identifier
Description
TIMEFRAME_M1 
1  minute
TIMEFRAME_M2
2 minutes
TIMEFRAME_M3
3 minutes
TIMEFRAME_M4
4 minutes
TIMEFRAME_M5
5 minutes

---

## Page 2016

Part 7. Advanced language tools
201 6
7.9 Native python support
Identifier
Description
TIMEFRAME_M6
6 minutes
TIMEFRAME_M1 0
1 0 minutes
TIMEFRAME_M1 2
1 2 minutes
TIMEFRAME_M1 2
1 5 minutes
TIMEFRAME_M20
20 minutes
TIMEFRAME_M30
30 minutes
TIMEFRAME_H1 
1  hour
TIMEFRAME_H2
2 hours
TIMEFRAME_H3
3 hours
TIMEFRAME_H4
4 hours
TIMEFRAME_H6
6 hours
TIMEFRAME_H8
8 hours
TIMEFRAME_H1 2
1 2 hours
TIMEFRAME_D1 
1  day
TIMEFRAME_W1 
1  week
TIMEFRAME_MN1 
1  month
All three functions return bars as a numpy batch array with named columns time, open, high, low, close,
tick_ volume, spread, and real_ volume. The numpy.ndarray array is a more efficient analog of named
tuples. To access columns, use square bracket notation, array[' column' ].
None is returned if an error occurs.
All function parameters are mandatory and unnamed.
numpy.ndarray copy_rates_from(symbol, timeframe, date_from, count)
The copy_ rates_ from function requests bars starting from the specified date (date_ from) in the number
of count bars. The date can be set by the datetime object, or as the number of seconds since
1 970.01 .01 .
When creating the datetime object, Python uses the local time zone, while the MetaTrader 5 terminal
stores tick and bar open times in UTC (GMT, no offset). Therefore, to execute functions that use time,
it is necessary to create datetime variables in UTC. To configure timezones, you can use the pytz
package. For example (see MQL5/Scripts/MQL5Book/Python/eurusdrates.py):

---

## Page 2017

Part 7. Advanced language tools
201 7
7.9 Native python support
from datetime import datetime
import MetaTrader5 as mt5   
import pytz                    # import the pytz module to work with the timezone
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   mt5.shutdown()
   quit()
   
# set the timezone to UTC
timezone = pytz.timezone("Etc/UTC")
   
# create a datetime object in the UTC timezone so that the local timezone offset is not applied
utc_from = datetime(2022, 1, 10, tzinfo = timezone)
   
# get 10 bars from EURUSD H1 starting from 10/01/2022 in the UTC timezone
rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_H1, utc_from, 10)
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
   
# display each element of the received data (tuple)
for rate in rates:
   print(rate)
A sample of received data:
(1641567600, 1.12975, 1.13226, 1.12922, 1.13017, 8325, 0, 0)
(1641571200, 1.13017, 1.13343, 1.1299, 1.13302, 7073, 0, 0)
(1641574800, 1.13302, 1.13491, 1.13293, 1.13468, 5920, 0, 0)
(1641578400, 1.13469, 1.13571, 1.13375, 1.13564, 3723, 0, 0)
(1641582000, 1.13564, 1.13582, 1.13494, 1.13564, 1990, 0, 0)
(1641585600, 1.1356, 1.13622, 1.13547, 1.13574, 1269, 0, 0)
(1641589200, 1.13572, 1.13647, 1.13568, 1.13627, 1031, 0, 0)
(1641592800, 1.13627, 1.13639, 1.13573, 1.13613, 982, 0, 0)
(1641596400, 1.1361, 1.13613, 1.1358, 1.1359, 692, 1, 0)
(1641772800, 1.1355, 1.13597, 1.13524, 1.1356, 1795, 10, 0)
numpy.ndarray copy_rates_from_pos(symbol, timeframe, start, count)
The copy_ rates_ from_ pos function requests bars starting from the specified start index, in the quantity
of count.
The MetaTrader 5 terminal renders bars only within the limits of the history available to the user on
the charts. The number of bars that are available to the user is set in the settings by the parameter
"Max. bars in the window".
The following example (MQL5/Scripts/MQL5Book/Python/ratescorr.py) shows a graphic representation of
the correlation matrix of several currencies based on quotes.

---

## Page 2018

Part 7. Advanced language tools
201 8
7.9 Native python support
import MetaTrader5 as mt5
import pandas as pd              # connect the pandas module to output data
import matplotlib.pyplot as plt  # connect the matplotlib module for drawing
   
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   mt5.shutdown()
   quit()
   
# create a path in the sandbox for the image with the result
image = mt5.terminal_info().data_path + r'\MQL5\Files\MQL5Book\ratescorr'
   
# the list of working currencies for calculating correlation
sym = ['EURUSD','GBPUSD','USDJPY','USDCHF','AUDUSD','USDCAD','NZDUSD','XAUUSD']
   
# copy the closing prices of bars into DataFrame structures
d = pd.DataFrame()
for i in sym:        # last 1000 M1 bars for each symbol
   rates = mt5.copy_rates_from_pos(i, mt5.TIMEFRAME_M1, 0, 1000)
   d[i] = [y['close'] for y in rates]
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
   
# calculate the price change as a percentage
rets = d.pct_change()
   
# compute correlations
corr = rets.corr()
   
# draw the correlation matrix
fig = plt.figure(figsize = (5, 5))
fig.add_axes([0.15, 0.1, 0.8, 0.8])
plt.imshow(corr, cmap = 'RdYlGn', interpolation = 'none', aspect = 'equal')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation = 'vertical')
plt.yticks(range(len(corr)), corr.columns)
plt.show()
plt.savefig(image)
The image file ratescorr.png is formed in the sandbox of the current working copy of MetaTrader 5.
Interactive display of an image in a separate window using a call to plt.show() may not work if your
Python installation does not include the Optional Features "tcl/tk and IDLE" or if you do not add the pip
install.tk package.

---

## Page 2019

Part 7. Advanced language tools
201 9
7.9 Native python support
Forex currency correlation matrix
numpy.ndarray copy_rates_range(symbol, timeframe, date_from, date_to)
The copy_ rates_ range function allows you to get bars in the specified date and time range, between
date_ from and date_ to: both values are given as the number of seconds since the beginning of 1 970, in
the UTC time zone (because Python uses datetime local timezone, you should convert using the module
pytz). The result includes bars with times of opening, time >= date_ from and time <= date_ to.
In the following script, we will request bars in a specific time range.

---

## Page 2020

Part 7. Advanced language tools
2020
7.9 Native python support
from datetime import datetime
import MetaTrader5 as mt5
import pytz             # connect the pytz module to work with the timezone
import pandas as pd     # connect the pandas module to display data in a tabular form
   
pd.set_option('display.max_columns', 500) # how many columns to show
pd.set_option('display.width', 1500)      # max. table width to display
   
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   quit()
   
# set the timezone to UTC
timezone = pytz.timezone("Etc/UTC")
# create datetime objects in UTC timezone so that local timezone offset is not applied
utc_from = datetime(2020, 1, 10, tzinfo=timezone)
utc_to = datetime(2020, 1, 10, minute = 30, tzinfo=timezone)
   
# get bars for USDJPY M5 for period 2020.01.10 00:00 - 2020.01.10 00:30 in UTC timezone
rates = mt5.copy_rates_range("USDJPY", mt5.TIMEFRAME_M5, utc_from, utc_to)
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
   
# create a DataFrame from the received data
rates_frame = pd.DataFrame(rates)
# convert time from number of seconds to datetime format
rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit = 's')
   
# output data
print(rates_frame)
An example of the result:
                 time     open     high      low    close  tick_volume  spread  real_volume 
0 2020-01-10 00:00:00  109.513  109.527  109.505  109.521           43       2            0 
1 2020-01-10 00:05:00  109.521  109.549  109.518  109.543          215       8            0 
2 2020-01-10 00:10:00  109.543  109.543  109.466  109.505           98      10            0 
3 2020-01-10 00:15:00  109.504  109.534  109.502  109.517          155       8            0 
4 2020-01-10 00:20:00  109.517  109.539  109.513  109.527           71       4            0 
5 2020-01-10 00:25:00  109.526  109.537  109.484  109.520          106       9            0 
6 2020-01-10 00:30:00  109.520  109.524  109.508  109.510          205       7            0 
7.9.1 0 Reading tick history
The Python API includes two functions for reading the real tick history: copy_ ticks_ from with an
indication of the number of ticks starting from the specified date, and copy_ ticks_ range for all ticks for
the specified period.

---

## Page 2021

Part 7. Advanced language tools
2021 
7.9 Native python support
Both functions have four required unnamed parameters, the first of which specifies the symbol. The
second parameter specifies the initial time of the requested ticks. The third parameter indicates either
the required number of ticks is passed (in the copy_ ticks_ from function) or the end time of ticks (in the
copy_ ticks_ range function).
The last parameter determines what kind of ticks will be returned. It can contain one of the following
flags (COPY_TICKS):
Identifier
Description
COPY_TICKS_ALL
All ticks
COPY_TICKS_INFO
Ticks containing Bid and/or Ask price changes
COPY_TICKS_TRADE
Ticks containing changes in the Last price and/or volume (Volume)
Both functions return ticks as an array numpy.ndarray (from the package numpy) with named columns
time, bid, ask, last, and flags. The value of the field flags is a combination of bit flags from the
TICK_FLAG enumeration: each bit means a change in the corresponding field with the tick property.
Identifier
Changed tick property
TICK_FLAG_BID
Bid price
TICK_FLAG_ASK
Ask price
TICK_FLAG_LAST
Last price
TICK_FLAG_VOLUME
Volume
TICK_FLAG_BUY
Last Buy price
TICK_FLAG_SELL
Last Sell price
numpy.ndarray copy_ticks_from(symbol, date_from, count, flags)
The copy_ ticks_ from function requests ticks starting from the specified time (date_ from) in the given
quantity (count).
The function is an analog of CopyTicks.
numpy.array copy_ticks_range(symbol, date_from, date_to, flags)
The copy_ ticks_ range function allows you to get ticks for the specified time range.
The function is an analog of CopyTicksRange.
In the following example (MQL5/Scripts/MQL5Book/Python/copyticks.py), we generate an interactive
web page with a tick chart (note: the plotly package is used here; to install it in Python, run the
command pip install plotly).

---

## Page 2022

Part 7. Advanced language tools
2022
7.9 Native python support
import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime
   
# connect to terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   quit()
   
# set the name of the file to save to the sandbox
path = mt5.terminal_info().data_path + r'\MQL5\Files\MQL5Book\copyticks.html'
   
# copy 1000 EURUSD ticks from a specific moment in history
utc = pytz.timezone("Etc/UTC") 
rates = mt5.copy_ticks_from("EURUSD", \
datetime(2022, 5, 25, 1, 15, tzinfo = utc), 1000, mt5.COPY_TICKS_ALL)
bid = [x['bid'] for x in rates]
ask = [x['ask'] for x in rates]
time = [x['time'] for x in rates]
time = pd.to_datetime(time, unit = 's')
   
# terminate the connection to the terminal
mt5.shutdown()
   
# connect the graphics package and draw 2 rows of ask and bid prices on the web page
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
data = [go.Scatter(x = time, y = bid), go.Scatter(x = time, y = ask)]
plot(data, filename = path)
Here's what the result might look like.

---

## Page 2023

Part 7. Advanced language tools
2023
7.9 Native python support
Chart with ticks received in a Python script
Webpage copyticks.html is generated in the subdirectory MQL5/Files/MQL5Book.
7.9.1 1  Calculating margin requirements and evaluating profits
A Python developer can directly calculate the margin and potential profit or loss of the proposed
trading operation in the script using the order_ calc_ margin and order_ calc_ profit functions. In the case
of successful execution, the result of any function is a real number; otherwise, it's None.
float order_calc_margin(action, symbol, volume, price)
The order_ calc_ margin function returns the margin amount (in the account currency) required to
complete the specified trade operation action which can be one of the two elements of the
ENUM_ORDER_TYPE enumeration: ORDER_TYPE_BUY or ORDER_TYPE_SELL. The following parameters
specify the name of the financial instrument, the volume of the trade operation, and the opening price.
The function is an analog of OrderCalcMargin.
float order_calc_profit(action, symbol, volume, price_open, price_close)
The order_ calc_ profit function returns the amount of profit or loss (in the account currency) for the
specified type of trade, symbol, and volume, as well as the difference between market entry and exit
prices.
The function is an analog of OrderCalcProfit.
It is recommended to check the margin and the expected result of the trading operation before sending
an order.

---

## Page 2024

Part 7. Advanced language tools
2024
7.9 Native python support
7.9.1 2 Checking and sending a trade order
If necessary, you can trade directly from a Python script. The pair of functions order_ check and
order_ send allows you to pre-check and then execute a trading operation.
For both functions, the only parameter is the request structure TradeRequest (it can be initialized as a
dictionary in Python, see an example). The structure fields are exactly the same as for
MqlTradeRequest.
OrderCheckResult order_check(request)
The order_ check function checks the correctness of trade request fields and the sufficiency of funds to
complete the required trading operation.
The result of the function is returned as the OrderCheckResult structure. It repeats the structure of
MqlTradeCheckResult but additionally contains the request field with a copy of the original request.
The order_ check function is an analog of OrderCheck.
Example (MQL5/Scripts/MQL5Book/python/ordercheck.py):

---

## Page 2025

Part 7. Advanced language tools
2025
7.9 Native python support
import MetaTrader5 as mt5
   
# let's establish a connection to the MetaTrader 5 terminal
...   
# get account currency for information
account_currency=mt5.account_info().currency
print("Account currency:", account_currency)
   
# get the necessary properties of the deal symbol
symbol = "USDJPY"
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
   print(symbol, "not found, can not call order_check()")
   mt5.shutdown()
   quit()
   
point = mt5.symbol_info(symbol).point
# if the symbol is not available in the Market Watch, add it
if not symbol_info.visible:
   print(symbol, "is not visible, trying to switch on")
   if not mt5.symbol_select(symbol, True):
      print("symbol_select({}) failed, exit", symbol)
      mt5.shutdown()
      quit()
   
# prepare the query structure as a dictionary
request = \
{
   "action": mt5.TRADE_ACTION_DEAL,
   "symbol": symbol,
   "volume": 1.0,
   "type": mt5.ORDER_TYPE_BUY,
   "price": mt5.symbol_info_tick(symbol).ask,
   "sl": mt5.symbol_info_tick(symbol).ask - 100 * point,
   "tp": mt5.symbol_info_tick(symbol).ask + 100 * point,
   "deviation": 10,
   "magic": 234000,
   "comment": "python script",
   "type_time": mt5.ORDER_TIME_GTC,
   "type_filling": mt5.ORDER_FILLING_RETURN,
}
   
# run the test and display the result as is
result = mt5.order_check(request)
print(result)                       # [?this is not in the help log?]
   
# convert the result to a dictionary and output element by element
result_dict = result._asdict()
for field in result_dict.keys():
   print("   {}={}".format(field, result_dict[field]))
   # if this is the structure of a trade request, then output it element by element as well

---

## Page 2026

Part 7. Advanced language tools
2026
7.9 Native python support
   if field == "request":
      traderequest_dict = result_dict[field]._asdict()
      for tradereq_filed in traderequest_dict:
         print("       traderequest: {}={}".format(tradereq_filed,
         traderequest_dict[tradereq_filed]))
   
# terminate the connection to the terminal
mt5.shutdown()
Result:
Account currency: USD
OrderCheckResult(retcode=0, balance=10000.17, equity=10000.17, profit=0.0, margin=1000.0,...
   retcode=0
   balance=10000.17
   equity=10000.17
   profit=0.0
   margin=1000.0
   margin_free=9000.17
   margin_level=1000.017
   comment=Done
   request=TradeRequest(action=1, magic=234000, order=0, symbol='USDJPY', volume=1.0, price=144.128,...
       traderequest: action=1
       traderequest: magic=234000
       traderequest: order=0
       traderequest: symbol=USDJPY
       traderequest: volume=1.0
       traderequest: price=144.128
       traderequest: stoplimit=0.0
       traderequest: sl=144.028
       traderequest: tp=144.228
       traderequest: deviation=10
       traderequest: type=0
       traderequest: type_filling=2
       traderequest: type_time=0
       traderequest: expiration=0
       traderequest: comment=python script
       traderequest: position=0
       traderequest: position_by=0
OrderSendResult order_send(request)
The order_ send function sends a request from the terminal to the trading server to make a trade
operation.
The result of the function is returned as the OrderSendResult structure. It repeats the structure of
MqlTradeResult but additionally contains the request field with a copy of the original request.
The function is an analog of OrderSend.
Example (MQL5/Scripts/MQL5Book/python/ordersend.py):

---

## Page 2027

Part 7. Advanced language tools
2027
7.9 Native python support
import time 
import MetaTrader5 as mt5 
   
# let's establish a connection to the MetaTrader 5 terminal
...   
# assign the properties of the working symbol
symbol = "USDJPY"
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
   print(symbol, "not found, can not trade")
   mt5.shutdown()
   quit()
   
# if the symbol is not available in the Market Watch, add it
if not symbol_info.visible:
   print(symbol, "is not visible, trying to switch on")
   if not mt5.symbol_select(symbol, True):
      print("symbol_select({}) failed, exit", symbol)
      mt5.shutdown()
      quit()
   
# let's prepare the request structure for the purchase
lot = 0.1
point = mt5.symbol_info(symbol).point
price = mt5.symbol_info_tick(symbol).ask
deviation = 20
request = \
{ 
   "action": mt5.TRADE_ACTION_DEAL, 
   "symbol": symbol, 
   "volume": lot, 
   "type": mt5.ORDER_TYPE_BUY, 
   "price": price, 
   "sl": price - 100 * point, 
   "tp": price + 100 * point, 
   "deviation": deviation, 
   "magic": 234000, 
   "comment": "python script open", 
   "type_time": mt5.ORDER_TIME_GTC, 
   "type_filling": mt5.ORDER_FILLING_RETURN, 
}
   
# send a trade request to open a position
result = mt5.order_send(request)
# check the execution result
print("1. order_send(): by {} {} lots at {}".format(symbol, lot, price));
if result.retcode != mt5.TRADE_RETCODE_DONE:
   print("2. order_send failed, retcode={}".format(result.retcode))
   # request the result as a dictionary and display it element by element
   result_dict = result._asdict()
   for field in result_dict.keys():

---

## Page 2028

Part 7. Advanced language tools
2028
7.9 Native python support
      print("   {}={}".format(field, result_dict[field]))
      # if this is the structure of a trade request, then output it element by element as well
      if field == "request":
         traderequest_dict = result_dict[field]._asdict()
         for tradereq_filed in traderequest_dict: 
            print("       traderequest: {}={}".format(tradereq_filed,
            traderequest_dict[tradereq_filed]))
   print("shutdown() and quit")
   mt5.shutdown()
   quit()
   
print("2. order_send done, ", result)
print("   opened position with POSITION_TICKET={}".format(result.order))
print("   sleep 2 seconds before closing position #{}".format(result.order))
time.sleep(2)
# create a request to close 
position_id = result.order
price = mt5.symbol_info_tick(symbol).bid
request = \
{
   "action": mt5.TRADE_ACTION_DEAL, 
   "symbol": symbol, 
   "volume": lot, 
   "type": mt5.ORDER_TYPE_SELL, 
   "position": position_id, 
   "price": price, 
   "deviation": deviation, 
   "magic": 234000, 
   "comment": "python script close", 
   "type_time": mt5.ORDER_TIME_GTC, 
   "type_filling": mt5.ORDER_FILLING_RETURN, 
} 
# send a trade request to close the position
result = mt5.order_send(request)
# check the execution result
print("3. close position #{}: sell {} {} lots at {}".format(position_id,
symbol, lot, price));
if result.retcode != mt5.TRADE_RETCODE_DONE:
   print("4. order_send failed, retcode={}".format(result.retcode))
   print("   result", result)
else: 
   print("4. position #{} closed, {}".format(position_id, result))
   # request the result as a dictionary and display it element by element
   result_dict = result._asdict()
   for field in result_dict.keys():
      print("   {}={}".format(field, result_dict[field])) 
      # if this is the structure of a trade request, then output it element by element as well
      if field == "request": 
         traderequest_dict = result_dict[field]._asdict()
         for tradereq_filed in traderequest_dict:
            print("       traderequest: {}={}".format(tradereq_filed,

---

## Page 2029

Part 7. Advanced language tools
2029
7.9 Native python support
            traderequest_dict[tradereq_filed]))
   
# terminate the connection to the terminal
mt5.shutdown()
Result:
1. order_send(): by USDJPY 0.1 lots at 144.132
2. order_send done,  OrderSendResult(retcode=10009, deal=1445796125, order=1468026008, volume=0.1, price=144.132,...
   opened position with POSITION_TICKET=1468026008
   sleep 2 seconds before closing position #1468026008
3. close position #1468026008: sell USDJPY 0.1 lots at 144.124
4. position #1468026008 closed, OrderSendResult(retcode=10009, deal=1445796155, order=1468026041, volume=0.1, price=144.124,...
   retcode=10009
   deal=1445796155
   order=1468026041
   volume=0.1
   price=144.124
   bid=144.124
   ask=144.132
   comment=Request executed
   request_id=2
   retcode_external=0
   request=TradeRequest(action=1, magic=234000, order=0, symbol='USDJPY', volume=0.1, price=144.124, stoplimit=0.0,...
       traderequest: action=1
       traderequest: magic=234000
       traderequest: order=0
       traderequest: symbol=USDJPY
       traderequest: volume=0.1
       traderequest: price=144.124
       traderequest: stoplimit=0.0
       traderequest: sl=0.0
       traderequest: tp=0.0
       traderequest: deviation=20
       traderequest: type=1
       traderequest: type_filling=2
       traderequest: type_time=0
       traderequest: expiration=0
       traderequest: comment=python script close
       traderequest: position=1468026008
       traderequest: position_by=0
7.9.1 3 Getting the number and list of active orders
The Python API provides the following functions for working with active orders.
int orders_total()
The orders_ total function returns the number of active orders.
The function is an analog of Orders Total.
Detailed information about each order can be obtained using the orders_ get function, which has several
options with the ability to filter by symbol or ticket. Either way, the function returns the array of named

---

## Page 2030

Part 7. Advanced language tools
2030
7.9 Native python support
tuples TradeOrder (field names match ENUM_ORDER_PROPERTY_enumerations without the "ORDER_"
prefix and reduced to lowercase). In case of an error, the result is None.
namedtuple[] orders_get()
namedtuple[] orders_get(symbol = <"SYMBOL">)
namedtuple[] orders_get(group = <"PATTERN">)
namedtuple[] orders_get(ticket = <TICKET>)
The orders_ get function without parameters returns orders for all symbols.
The optional named parameter symbol makes it possible to specify a specific symbol name for order
selection.
The optional named parameter group is intended for specifying a search pattern using the wildcard
character '*' (as a substitute for an arbitrary number of any characters, including zero characters in
the given place of the pattern) and the condition logical negation character '!'. The filter template
operation principle was described in the section Getting information about financial instruments.
If the ticket parameter is specified, a certain order is searched.
In one function call, you can get all active orders. It is an analog of the combined use of OrdersTotal,
OrderSelect, and OrderGet functions.
In the next example (MQL5/Scripts/MQL5Book/Python/ordersget.py), we request information about
orders using different ways.

---

## Page 2031

Part 7. Advanced language tools
2031 
7.9 Native python support
import MetaTrader5 as mt5
import pandas as pd
pd.set_option('display.max_columns', 500) # how many columns to show
pd.set_option('display.width', 1500)      # max. table width to display
   
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize(): 
   print("initialize() failed, error code =", mt5.last_error())
   quit()
   
# display information about active orders on the GBPUSD symbol
orders = mt5.orders_get(symbol = "GBPUSD")
if orders is None:
   print("No orders on GBPUSD, error code={}".format(mt5.last_error()))
else:
   print("Total orders on GBPUSD:", len(orders))
   # display all active orders
   for order in orders:
      print(order)
print()
   
# getting a list of orders on symbols whose names contain "*GBP*"
gbp_orders = mt5.orders_get(group="*GBP*")
if gbp_orders is None: 
   print("No orders with group=\"*GBP*\", error code={}".format(mt5.last_error()))
else: 
   print("orders_get(group=\"*GBP*\")={}".format(len(gbp_orders)))
   # display orders as a table using pandas.DataFrame
   df = pd.DataFrame(list(gbp_orders), columns = gbp_orders[0]._asdict().keys())
   df.drop(['time_done', 'time_done_msc', 'position_id', 'position_by_id',
   'reason', 'volume_initial', 'price_stoplimit'], axis = 1, inplace = True)
   df['time_setup'] = pd.to_datetime(df['time_setup'], unit = 's')
   print(df)
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
The sample result is below:
Total orders on GBPUSD: 2 
TradeOrder(ticket=554733548, time_setup=1585153667, time_setup_msc=1585153667718, time_done=0, time_done_msc=0, time_expiration=0, type=3, type_time=0, ... 
TradeOrder(ticket=554733621, time_setup=1585153671, time_setup_msc=1585153671419, time_done=0, time_done_msc=0, time_expiration=0, type=2, type_time=0, ... 
  
orders_get(group="*GBP*")=4 
      ticket          time_setup  time_setup_msc  type ... volume_current  price_open   sl   tp  price_current  symbol comment external_id 
0  554733548 2020-03-25 16:27:47   1585153667718     3 ...            0.2     1.25379  0.0  0.0        1.16803  GBPUSD                     
1  554733621 2020-03-25 16:27:51   1585153671419     2 ...            0.2     1.14370  0.0  0.0        1.16815  GBPUSD                     
2  554746664 2020-03-25 16:38:14   1585154294401     3 ...            0.2     0.93851  0.0  0.0        0.92428  EURGBP                     
3  554746710 2020-03-25 16:38:17   1585154297022     2 ...            0.2     0.90527  0.0  0.0        0.92449  EURGBP    

---

## Page 2032

Part 7. Advanced language tools
2032
7.9 Native python support
7.9.1 4 Getting the number and list of open positions
The positions_ total function returns the number of open positions.
int positions_total()
The function is an analog of PositionsTotal.
To get detailed information about each position, use the positions_ get function which has multiple
options. All variants return an array of named tuples TradePosition with keys corresponding to position
properties (see elements of ENUM_POSITION_PROPERTY_enumerations, without the "POSITION_"
prefix, in lowercase). In case of an error, the result is None.
namedtuple[] positions_get()
namedtuple[] positions_get(symbol = <"SYMBOL">)
namedtuple[] positions_get(group = <"PATTERN">)
namedtuple[] positions_get(ticket = <TICKET>)
The function without parameters returns all open positions.
The function with the symbol parameter allows the selection of positions for the specified symbol.
The function with the group parameter provides filtering by search mask with wildcards '*' (any
characters are replaced) and logical negation of the condition '!'. For details see the section Getting
information about financial instruments.
A version with the ticket parameters selects a position with a specific ticket (POSITION_TICKET
property).
The positions_ get function can be used to get all positions and their properties in one call, which makes
it similar to a bunch of PositionsTotal, PositionSelect, and PositionGet functions.
In the script MQL5/Scripts/MQL5Book/Python/positionsget.py, we request positions for a specific
symbol and search mask.

---

## Page 2033

Part 7. Advanced language tools
2033
7.9 Native python support
import MetaTrader5 as mt5
import pandas as pd
pd.set_option('display.max_columns', 500) # how many columns to show
pd.set_option('display.width', 1500)      # max. table width to display
   
# let's establish a connection to the MetaTrader 5 terminal
if not mt5.initialize():
   print("initialize() failed, error code =", mt5.last_error())
   quit()
   
# get open positions on USDCHF
positions = mt5.positions_get(symbol = "USDCHF")
if positions == None: 
   print("No positions on USDCHF, error code={}".format(mt5.last_error()))
elif len(positions) > 0:
   print("Total positions on USDCHF =", len(positions))
 # display all open positions
   for position in positions:
      print(position)
   
# get a list of positions on symbols whose names contain "*USD*"
usd_positions = mt5.positions_get(group = "*USD*") 
if usd_positions == None:
    print("No positions with group=\"*USD*\", error code={}".format(mt5.last_error())) 
elif len(usd_positions) > 0: 
   print("positions_get(group=\"*USD*\") = {}".format(len(usd_positions)))
   # display the positions as a table using pandas.DataFrame
   df=pd.DataFrame(list(usd_positions), columns = usd_positions[0]._asdict().keys())
   df['time'] = pd.to_datetime(df['time'], unit='s')
   df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'],
   axis=1, inplace=True)
   print(df)
   
# complete the connection to the MetaTrader 5 terminal
mt5.shutdown()
Here's what the result might be:
Total positions on USDCHF = 1
TradePosition(ticket=1468454363, time=1664217233, time_msc=1664217233239, time_update=1664217233,
   time_update_msc=1664217233239, type=1, magic=0, identifier=1468454363, reason=0, volume=0.01, price_open=0.99145,
   sl=0.0, tp=0.0, price_current=0.9853, swap=-0.01, profit=6.24, symbol='USDCHF', comment='', external_id='')
positions_get(group="*USD*") = 2
       ticket                time  type  ...  identifier  volume  price_open  ... _current  swap  profit  symbol comment
0  1468454363 2022-09-26 18:33:53     1  ...  1468454363    0.01     0.99145  ...  0.98530 -0.01    6.24  USDCHF        
1  1468475849 2022-09-26 18:44:00     0  ...  1468475849    0.01     1.06740  ...  1.08113  0.00   13.73  GBPUSD        

---

## Page 2034

Part 7. Advanced language tools
2034
7.9 Native python support
7.9.1 5 Reading the history of orders and deals
Working with orders and deals in the account history using Python scripts is also possible. For these
purposes, there are functions history_ orders_ total, history_ orders_ get, history_ deals_ total, and
history_ deals_ get.
int history_orders_total(date_from, date_to)
The history_ orders_ total function returns the number of orders in the trading history in the specified
time interval. Each of the parameters is set by the datetime object or as the number of seconds since
1 970.01 .01 .
The function is an analog of HistoryOrdersTotal.
The history_ orders_ get function is available in several versions and supports order filtering by substring
in symbol name, ticket, or position ID. All variants return an array of named tuples TradeOrder (field
names match ENUM_ORDER_PROPERTY_enumerations without the "ORDER_" prefix and in lowercase).
If there are no matching orders, the array will be empty. In case of an error, the function will return
None.
namedtuple[] history_orders_get(date_from, date_to, group = <"PATTERN">)
namedtuple[] history_orders_get(ticket = <ORDER_TICKET>)
namedtuple[] history_orders_get(position = <POSITION_ID>)
The first version selects orders within the specified time range (similar to history_ orders_ total). In the
optional named parameter group, you can specify a search pattern for a substring of the symbol name
(you can use the wildcard characters '*' and negation '!' in it, see the section Getting information
about financial instruments).
The second version is designed to search for a specific order by its ticket.
The last version selects orders by position ID (ORDER_POSITION_ID property).
Either option is equivalent to calling several MQL5 functions: HistoryOrdersTotal, HistoryOrderSelect,
and HistoryOrderGet-functions.
Let's see on an example of the script historyordersget.py how to get the number and list of historical
orders for different conditions.

---

## Page 2035

Part 7. Advanced language tools
2035
7.9 Native python support
from datetime import datetime 
import MetaTrader5 as mt5 
import pandas as pd 
pd.set_option('display.max_columns', 500) # how many columns to show 
pd.set_option('display.width', 1500)      # max. table width for display
...   
# get the number of orders in the history for the period (total and *GBP*)
from_date = datetime(2022, 9, 1)
to_date = datetime.now()
total = mt5.history_orders_total(from_date, to_date)
history_orders=mt5.history_orders_get(from_date, to_date, group="*GBP*")
# print(history_orders)
if history_orders == None: 
   print("No history orders with group=\"*GBP*\", error code={}".format(mt5.last_error())) 
else :
   print("history_orders_get({}, {}, group=\"*GBP*\")={} of total {}".format(from_date,
   to_date, len(history_orders), total))
   
# display all canceled historical orders for ticket position 0
position_id = 0
position_history_orders = mt5.history_orders_get(position = position_id)
if position_history_orders == None:
   print("No orders with position #{}".format(position_id))
   print("error code =", mt5.last_error())
elif len(position_history_orders) > 0:
   print("Total history orders on position #{}: {}".format(position_id,
   len(position_history_orders)))
   # display received orders as is
   for position_order in position_history_orders:
      print(position_order)
   # display these orders as a table using pandas.DataFrame
   df = pd.DataFrame(list(position_history_orders),
   columns = position_history_orders[0]._asdict().keys())
   df.drop(['time_expiration', 'type_time', 'state', 'position_by_id', 'reason', 'volume_current',
   'price_stoplimit','sl','tp', 'time_setup_msc', 'time_done_msc', 'type_filling', 'external_id'],
   axis = 1, inplace = True)
   df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
   df['time_done'] = pd.to_datetime(df['time_done'], unit='s')
   print(df)
...
The result of the script (given with abbreviations):

---

## Page 2036

Part 7. Advanced language tools
2036
7.9 Native python support
history_orders_get(2022-09-01 00:00:00, 2022-09-26 21:50:04, group="*GBP*")=15 of total 44
   
Total history orders on position #0: 14
TradeOrder(ticket=1437318706, time_setup=1661348065, time_setup_msc=1661348065049, time_done=1661348083,
   time_done_msc=1661348083632, time_expiration=0, type=2, type_time=0, type_filling=2, state=2, magic=0,
   position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=0.99301,
   sl=0.0, tp=0.0, price_current=0.99311, price_stoplimit=0.0, symbol='EURUSD', comment='', external_id='')
TradeOrder(ticket=1437331579, time_setup=1661348545, time_setup_msc=1661348545750, time_done=1661348551,
   time_done_msc=1661348551354, time_expiration=0, type=2, type_time=0, type_filling=2, state=2, magic=0,
   position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=0.99281,
   sl=0.0, tp=0.0, price_current=0.99284, price_stoplimit=0.0, symbol='EURUSD', comment='', external_id='')
TradeOrder(ticket=1437331739, time_setup=1661348553, time_setup_msc=1661348553935, time_done=1661348563,
   time_done_msc=1661348563412, time_expiration=0, type=2, type_time=0, type_filling=2, state=2, magic=0,
   position_id=0, position_by_id=0, reason=3, volume_initial=0.01, volume_current=0.01, price_open=0.99285,
   sl=0.0, tp=0.0, price_current=0.99286, price_stoplimit=0.0, symbol='EURUSD', comment='', external_id='')
...
   
        ticket          time_setup           time_done  type  ... _initial  price_open  price_current  symbol comment
0   1437318706 2022-08-24 13:34:25 2022-08-24 13:34:43     2          0.01     0.99301        0.99311  EURUSD        
1   1437331579 2022-08-24 13:42:25 2022-08-24 13:42:31     2          0.01     0.99281        0.99284  EURUSD        
2   1437331739 2022-08-24 13:42:33 2022-08-24 13:42:43     2          0.01     0.99285        0.99286  EURUSD
...
We can see that in September, there were only 44 orders, 1 5 of which included the GBP currency (an
odd number due to the open position). The history contains 1 4 canceled orders.
int history_deals_total(date_from, date_to)
The history_ deals_ total function returns the number of deals in history for the specified period.
The function is an analog of HistoryDealsTotal.
The history_ deals_ get function has several forms and is designed to select trades with the ability to
filter by order ticket or position ID. All forms of the function return an array of named tuples TradeDeal,
with fields reflecting properties from the ENUM_DEAL_PROPERTY_enumerations (the prefix "DEAL_"
has been removed from the field names and lowercase has been applied). In case of an error, we get
None.
namedtuple[] history_deals_get(date_from, date_to, group = <"PATTERN">)
namedtuple[] history_deals_get(ticket = <ORDER_TICKET>)
namedtuple[] history_deals_get(position = <POSITION_ID>)
The first form of the function is similar to requesting historical orders using history_ orders_ get.
The second form allows the selection of deals generated by a specific order by its ticket (the
DEAL_ORDER property).
Finally, the third form requests deals that have formed a position with a given ID (the
DEAL_POSITION_ID property).
The function allows you to get all transactions together with their properties in one call, which is
analogous to the bunch of HistoryDealsTotal, HistoryDealSelect, and HistoryDealGet-functions.
Here is the main part of the test script historydealsget.py.

---

## Page 2037

Part 7. Advanced language tools
2037
7.9 Native python support
# set the time range
from_date = datetime(2020, 1, 1)
to_date = datetime.now() 
   
# get trades for symbols whose names do not contain either "EUR" or "GBP"
deals = mt5.history_deals_get(from_date, to_date, group="*,!*EUR*,!*GBP*") 
if deals == None: 
   print("No deals, error code={}".format(mt5.last_error()))
elif len(deals) > 0: 
   print("history_deals_get(from_date, to_date, group=\"*,!*EUR*,!*GBP*\") =",
   len(deals)) 
   # display all received deals as they are
   for deal in deals: 
      print("  ",deal) 
   # display these trades as a table using pandas.DataFrame
   df = pd.DataFrame(list(deals), columns = deals[0]._asdict().keys()) 
   df['time'] = pd.to_datetime(df['time'], unit='s')
   df.drop(['time_msc','commission','fee'], axis = 1, inplace = True)
   print(df) 
An example of result:
history_deals_get(from_date, to_date, group="*,!*EUR*,!*GBP*") = 12
   TradeDeal(ticket=1109160642, order=0, time=1632188460, time_msc=1632188460852, type=2, entry=0, magic=0, position_id=0, reason=0, volume=0.0, price=0.0, commission=0.0, swap=0.0, profit=10000.0, fee=0.0, symbol='', comment='', external_id='')
   TradeDeal(ticket=1250629232, order=1268074569, time=1645709385, time_msc=1645709385815, type=0, entry=0, magic=0, position_id=1268074569, reason=0, volume=0.01, price=1970.98, commission=0.0, swap=0.0, profit=0.0, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1250639814, order=1268085019, time=1645709950, time_msc=1645709950618, type=1, entry=1, magic=0, position_id=1268074569, reason=0, volume=0.01, price=1970.09, commission=0.0, swap=0.0, profit=-0.89, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1250639928, order=1268085129, time=1645709955, time_msc=1645709955502, type=1, entry=0, magic=0, position_id=1268085129, reason=0, volume=0.01, price=1969.98, commission=0.0, swap=0.0, profit=0.0, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1250640111, order=1268085315, time=1645709965, time_msc=1645709965148, type=0, entry=1, magic=0, position_id=1268085129, reason=0, volume=0.01, price=1970.17, commission=0.0, swap=0.0, profit=-0.19, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1250640309, order=1268085512, time=1645709973, time_msc=1645709973623, type=1, entry=0, magic=0, position_id=1268085512, reason=0, volume=0.1, price=1970.09, commission=0.0, swap=0.0, profit=0.0, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1250640400, order=1268085611, time=1645709978, time_msc=1645709978701, type=0, entry=1, magic=0, position_id=1268085512, reason=0, volume=0.1, price=1970.22, commission=0.0, swap=0.0, profit=-1.3, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1250640616, order=1268085826, time=1645709988, time_msc=1645709988277, type=1, entry=0, magic=0, position_id=1268085826, reason=0, volume=1.1, price=1969.95, commission=0.0, swap=0.0, profit=0.0, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1250640810, order=1268086019, time=1645709996, time_msc=1645709996990, type=0, entry=1, magic=0, position_id=1268085826, reason=0, volume=1.1, price=1969.88, commission=0.0, swap=0.0, profit=7.7, fee=0.0, symbol='XAUUSD', comment='', external_id='')
   TradeDeal(ticket=1445796125, order=1468026008, time=1664199450, time_msc=1664199450488, type=0, entry=0, magic=234000, position_id=1468026008, reason=3, volume=0.1, price=144.132, commission=0.0, swap=0.0, profit=0.0, fee=0.0, symbol='USDJPY', comment='python script op', external_id='')
   TradeDeal(ticket=1445796155, order=1468026041, time=1664199452, time_msc=1664199452567, type=1, entry=1, magic=234000, position_id=1468026008, reason=3, volume=0.1, price=144.124, commission=0.0, swap=0.0, profit=-0.56, fee=0.0, symbol='USDJPY', comment='python script cl', external_id='')
   TradeDeal(ticket=1446217804, order=1468454363, time=1664217233, time_msc=1664217233239, type=1, entry=0, magic=0, position_id=1468454363, reason=0, volume=0.01, price=0.99145, commission=0.0, swap=0.0, profit=0.0, fee=0.0, symbol='USDCHF', comment='', external_id='')
   
        ticket       order                time t…  e…  … position_id  volume       price    profit  symbol  comment external_id
0   1109160642           0 2021-09-21 01:41:00  2   0              0    0.00     0.00000  10000.00                             
1   1250629232  1268074569 2022-02-24 13:29:45  0   0     1268074569    0.01  1970.98000      0.00  XAUUSD                     
2   1250639814  1268085019 2022-02-24 13:39:10  1   1     1268074569    0.01  1970.09000     -0.89  XAUUSD                     
3   1250639928  1268085129 2022-02-24 13:39:15  1   0     1268085129    0.01  1969.98000      0.00  XAUUSD                     
4   1250640111  1268085315 2022-02-24 13:39:25  0   1     1268085129    0.01  1970.17000     -0.19  XAUUSD                     
5   1250640309  1268085512 2022-02-24 13:39:33  1   0     1268085512    0.10  1970.09000      0.00  XAUUSD                     
6   1250640400  1268085611 2022-02-24 13:39:38  0   1     1268085512    0.10  1970.22000     -1.30  XAUUSD                     
7   1250640616  1268085826 2022-02-24 13:39:48  1   0     1268085826    1.10  1969.95000      0.00  XAUUSD                     
8   1250640810  1268086019 2022-02-24 13:39:56  0   1     1268085826    1.10  1969.88000      7.70  XAUUSD                     
9   1445796125  1468026008 2022-09-26 13:37:30  0   0     1468026008    0.10   144.13200      0.00  USDJPY  python script op   
10  1445796155  1468026041 2022-09-26 13:37:32  1   1     1468026008    0.10   144.12400     -0.56  USDJPY  python script cl   
11  1446217804  1468454363 2022-09-26 18:33:53  1   0     1468454363    0.01     0.99145      0.00  USDCHF                     

---

## Page 2038

Part 7. Advanced language tools
2038
7.1 0 Built-in support for parallel computing: OpenCL
7.1 0 Built-in support for parallel computing: OpenCL
OpenCL is an open parallel programming standard that allows you to create applications for
simultaneous execution on many cores of modern processors, different in architecture, in particular,
graphic (GPU) or central (CPU).
In other words, OpenCL allows you to use all the cores of the central processor or all the computing
power of the video card for computing one task, which ultimately reduces the program execution time.
Therefore, the use of OpenCL is very useful for computationally intensive tasks, but it is important to
note that the algorithms for solving these tasks must be divisible into parallel threads. These include,
for example, training neural networks, Fourier transform, or solving systems of equations of large
dimensions.
For example, in relation to the trading specifics, an increase in performance can be achieved with a
script, indicator, or Expert Advisor that performs a complex and lengthy analysis of historical data for
several symbols and timeframes, and the calculation for each of which does not depend on others.
At the same time, beginners often have a question whether it is possible to speed up the testing and
optimization of Expert Advisors using OpenCL. The answers to both questions are no. Testing
reproduces the real process of sequential trading, and therefore each next bar or tick depends on
the results of the previous ones, which makes it impossible to parallelize the calculations of one
pass. As for optimization, the tester's agents only support CPU cores. This is due to the complexity
of a full-fledged analysis of quotes or ticks, tracking positions and calculating balance and equity.
However, if complexity doesn't scare you, you can implement your own optimization engine on the
graphics card cores by transferring all the calculations that emulate the trading environment with
the required reliability to OpenCL.
OpenCL means Open Computing Language. It is similar to the C and C++ languages, and therefore, to
MQL5. However, in order to prepare ("compile") an OpenCL program, pass input data to it, run it in
parallel on several cores, and obtain calculation results, a special programming interface (a set of
functions) is used. This OpenCL API is also available for MQL programs that wish to implement parallel
execution.
To use OpenCL, it is not necessary to have a video card on your PC as the presence of a central
processor is enough. In any case, special drivers from the manufacturer are required (OpenCL version
1 .1  and higher is required). If your computer has games or other software (for example, scientific,
video editor, etc.) that work directly with video cards, then the necessary software layer is most likely
already available. This can be checked by trying to run an MQL program in the terminal with an OpenCL
call (at least a simple example from the terminal delivery, see further).
If there is no OpenCL support, you will see an error in the log.
OpenCL OpenCL not found, please install OpenCL drivers
If there is a suitable device on your computer and OpenCL support has been enabled for it, the terminal
will display a message with the name and type of this device (there may be several devices). For
example:
OpenCL Device #0: CPU GenuineIntel Intel(R) Core(TM) i7-2700K CPU @ 3.50GHz with OpenCL 1.1 (8 units, 3510 MHz, 16301 Mb, version 2.0, rating 25)
OpenCL Device #1: GPU Intel(R) Corporation Intel(R) UHD Graphics 630 with OpenCL 2.1 (24 units, 1200 MHz, 13014 Mb, version 26.20.100.7985, rating 73)
The procedure for installing drivers for various devices is described in the article on mql5.com. Support
extends to the most popular devices from Intel, AMD, ATI, and Nvidia.

---

## Page 2039

Part 7. Advanced language tools
2039
7.1 0 Built-in support for parallel computing: OpenCL
In terms of the number of cores and the speed of distributed computing, central processors are
significantly inferior to graphics cards, but a good multi-core central processor will be quite enough to
significantly increase performance.
Important: If your computer has a video card with OpenCL support, then you do not need to install
OpenCL software emulation on the CPU!
OpenCL device drivers automate the distribution of calculations across cores. For example, if you need
to perform a million of calculations of the same type with different vectors, and there are only a
thousand cores at your disposal, then the drivers will automatically start each next task as the previous
ones are ready and the cores are released.
Preparatory operations for setting up the OpenCL runtime environment in an MQL program are
performed only once using the functions of the above OpenCL API. 
1 .Creating a context for an OpenCL program (selecting a device, such as a video card, CPU, or any
available): CLContextCreate(CL_ USE_ ANY). The function will return a context descriptor (an integer,
let's denote it conditionally ContextHandle).
2. Creating an OpenCL program in the received context: it is compiled based on the source code in the
OpenCL language using the CLProgramCreate function call, to which the text of the code is passed
through the parameter Source:CLProgramCreate(ContextHandle, Source, BuildLog). The function will
return the program handle (integer ProgramHandle). It is important to note here that inside the
source code of this program, there must be functions (at least one) marked with a special keyword
_ _ kernel (or simply kernel): they contain the parts of the algorithm to be parallelized (see example
below). Of course, in order to simplify (decompose the source code), the programmer can divide the
logical subtasks of the kernel function into other auxiliary functions and call them from the kernel:
at the same time, there is no need to mark the auxiliary functions with the word kernel.
3. Registering a kernel to execute by the name of one of those functions that are marked in the code
of the OpenCL program as kernel-forming: CLKernelCreate(ProgramHandle, KernelName). Calling this
function will return a handle to the kernel (an integer, let's say, KernelHandle). You can prepare
many different functions in OpenCL code and register them as different kernels.
4. If necessary, creating buffers for data arrays passed by reference to the kernel and for returned
values/arrays: CLBufferCreate(ContextHandle, Size * sizeof(double), CL_ MEM_ READ_ WRITE), etc.
Buffers are also identified and managed with descriptors.
Next, once or several times, if necessary, (for example, in indicator or Expert Advisor event handlers),
calculations are performed directly according to the following scheme:
I. Passing input data and/or binding input/output buffers with CLSetKernelArg(KernelHandle,...) and/or
CLSetKernelArgMem(KernelHandle,..., BufferHandle). The first function provides the setting of a
scalar value, and the second is equivalent to passing or receiving a value (or an array of values) by
reference. At this stage, data is moved from MQL5 to the OpenCL execution core.
CLBufferWrite(BufferHandle,...) writes data to the buffer. Parameters and buffers will become
available to the OpenCL program during kernel execution.
II.Performing Parallel Computations by Calling a Specific Kernel CLExecute(KernelHandle,...). The
kernel function will be able to write the results of its work to the output buffer.
III.Getting results with CLBufferRead(BufferHandle). At this stage, data is moved back from OpenCL to
MQL5.

---

## Page 2040

Part 7. Advanced language tools
2040
7.1 0 Built-in support for parallel computing: OpenCL
After completion of calculations, all descriptors should be released:
CLBufferFree(BufferHandle),CLKernelFree(KernelHandle), CLProgramFree(ProgramHandle), and
CLContextFree(ContextHandle).
This sequence is conventionally indicated in the following diagram.
Scheme of interaction between an MQL program and an OpenCL attachment
It is recommended to write the OpenCL source code in separate text files, which can then be
connected to the MQL5 program using resource variables.
The standard header library supplied with the terminal contains a wrapper class for working with
OpenCL: MQL5/Include/OpenCL/OpenCL.mqh.
Examples of using OpenCL can be found in the folder MQL5/Scripts/Examples/OpenCL/. In particular,
there is the MQL5/Scripts/Examples/OpenCL/Double/Wavelet.mq5 script, which produces a wavelet
transform of the time series (you can take an artificial curve according to the stochastic Weierstrass
model or the increment in prices of the current financial instrument). In any case, the initial data for
the algorithm is an array which is a two-dimensional image of a series.
When running this script, same as when running any other MQL program with OpenCL code, the
terminal will select the fastest device (if there are several of them, and the specific device was not

---

## Page 2041

Part 7. Advanced language tools
2041 
7.1 0 Built-in support for parallel computing: OpenCL
selected in the program itself or was not already defined earlier). Information about this is displayed in
the Journal tab (terminal log, not experts).
Scripts script Wavelet (EURUSD,H1) loaded successfully
OpenCL  device #0: GPU NVIDIA Corporation NVIDIA GeForce GTX 1650 with OpenCL 3.0 (16 units, 1560 MHz, 4095 Mb, version 512.72)
OpenCL  device #1: GPU Intel(R) Corporation Intel(R) UHD Graphics 630 with OpenCL 3.0 (24 units, 1150 MHz, 6491 Mb, version 27.20.100.8935)
OpenCL  device performance test started
OpenCL  device performance test successfully finished
OpenCL  device #0: GPU NVIDIA Corporation NVIDIA GeForce GTX 1650 with OpenCL 3.0 (16 units, 1560 MHz, 4095 Mb, version 512.72, rating 129)
OpenCL  device #1: GPU Intel(R) Corporation Intel(R) UHD Graphics 630 with OpenCL 3.0 (24 units, 1150 MHz, 6491 Mb, version 27.20.100.8935, rating 136)
Scripts script Wavelet (EURUSD,H1) removed
As a result of execution, the script displays in the Experts tab records with calculation speed
measurements in the usual way (in series, on the CPU) and in parallel (on OpenCL cores).
OpenCL: GPU device 'Intel(R) UHD Graphics 630' selected
time CPU=5235 ms, time GPU=125 ms, CPU/GPU ratio: 41.880000
The ratio of speeds, depending on the specifics of the task, can reach tens.
The script displays on the chart the original image, its derivative in the form of increments, and the
result of the wavelet transform.
The original simulated series, its increments and wavelet transform
Please note that the graphic objects remain on the chart after the script finished working. They will
need to be removed manually.
Here is how the source OpenCL code of the wavelet transform looks like, implemented in a separate file
MQL5/Scripts/Examples/OpenCL/Double/Kernels/wavelet.cl.

---

## Page 2042

Part 7. Advanced language tools
2042
7.1 0 Built-in support for parallel computing: OpenCL
// increased calculation accuracy double is required
// (by default, without this directive we get float)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
   
// Helper function Morlet
double Morlet(const double t)
{
   return exp(-t * t * 0.5) * cos(M_2_PI * t);
}
   
// OpenCL kernel function
__kernel void Wavelet_GPU(__global double *data, int datacount,
   int x_size, int y_size, __global double *result)
{
   size_t i = get_global_id(0);
   size_t j = get_global_id(1);
   double a1 = (double)10e-10;
   double a2 = (double)15.0;
   double da = (a2 - a1) / (double)y_size;
   double db = ((double)datacount - (double)0.0) / x_size;
   double a = a1 + j * da;
   double b = 0 + i * db;
   double B = (double)1.0;
   double B_inv = (double)1.0 / B;
   double a_inv = (double)1.0 / a;
   double dt = (double)1.0;
   double coef = (double)0.0;
   
   for(int k = 0; k < datacount; k++)
   {
      double arg = (dt * k - b) * a_inv;
      arg = -B_inv * arg * arg;
      coef = coef + exp(arg);
   }
   
   double sum = (float)0.0;
   for(int k = 0; k < datacount; k++)
   {
      double arg = (dt * k - b) * a_inv;
      sum += data[k] * Morlet(arg);
   }
   sum = sum / coef;
   uint pos = (int)(j * x_size + i);
   result[pos] = sum;
}
Full information about the OpenCL syntax, built-in functions and principles of operation can be found on
the official website of Khronos Group.
In particular, it is interesting to note that OpenCL supports not only the usual scalar numeric data
types (starting from char and ending with double) but also vector (u)charN, (u)shortN, (u)intN, (u)longN,

---

## Page 2043

Part 7. Advanced language tools
2043
7.1 0 Built-in support for parallel computing: OpenCL
floatN, doubleN, where N = {2| 3| 4| 8| 1 6} and denotes the length of the vector. In this example, this is
not used.
In addition to the mentioned keyword kernel, an important role in the organization of parallel computing
is played by the get_ global_ id function: it allows you to find in the code the number of the
computational subtask that is currently running. Obviously, the calculations in different subtasks should
be different (otherwise it would not make sense to use many cores). In this example, since the task
involves the analysis of a two-dimensional image, it is more convenient to identify its fragments using
two orthogonal coordinates. In the above code, we get them using two calls, get_ global_ id(0) and
get_ global_ id(1 ).
Actually, we set the data dimension for the task ourselves when calling the MQL5 function CLExecute
(see further).
In the file Wavelet.mq5, the OpenCL source code is included using the directive:
#resource "Kernels/wavelet.cl" as string cl_program
The image size is set by macros:
#define SIZE_X 600
#define SIZE_Y 200
To manage OpenCL, the standard library with the class COpenCL is used. Its methods have similar
names and internally use the corresponding built-in OpenCL functions from the MQL5 API. It is
suggested that you familiarize yourself with it.
#include <OpenCL/OpenCL.mqh>
In a simplified form (without error checking and visualization), the MQL code that launches the
transformation is shown below. Wavelet transform-related actions are summarized in the CWavelet
class.
class CWavelet
{
protected:
   ...
   int        m_xsize;              // image dimensions along the axes
   int        m_ysize;
   double     m_wavelet_data_GPU[]; // result goes here
   COpenCL    m_OpenCL;             // wrapper object
   ...
};
The main parallel computing is organized by its method CalculateWavelet_ GPU.

---

## Page 2044

Part 7. Advanced language tools
2044
7.1 0 Built-in support for parallel computing: OpenCL
bool CWavelet::CalculateWavelet_GPU(double &data[], uint &time)
{
   int datacount = ArraySize(data); // image size (number of dots)
   
   // compile the cl-program according to its source code
   m_OpenCL.Initialize(cl_program, true);
   
   // register a single kernel function from the cl file
   m_OpenCL.SetKernelsCount(1);
   m_OpenCL.KernelCreate(0, "Wavelet_GPU");
   
   // register 2 buffers for input and output data, write the input array
   m_OpenCL.SetBuffersCount(2);
   m_OpenCL.BufferFromArray(0, data, 0, datacount, CL_MEM_READ_ONLY);
   m_OpenCL.BufferCreate(1, m_xsize * m_ysize * sizeof(double), CL_MEM_READ_WRITE);
   m_OpenCL.SetArgumentBuffer(0, 0, 0);
   m_OpenCL.SetArgumentBuffer(0, 4, 1);
   
   ArrayResize(m_wavelet_data_GPU, m_xsize * m_ysize);
   uint work[2];              // task of analyzing a two-dimensional image - hence the dimension 2
   uint offset[2] = {0, 0};   // start from the very beginning (or you can skip something)
   work[0] = m_xsize;
   work[1] = m_ysize;
   
   // set input data   
   m_OpenCL.SetArgument(0, 1, datacount);
   m_OpenCL.SetArgument(0, 2, m_xsize);
   m_OpenCL.SetArgument(0, 3, m_ysize);
   
   time = GetTickCount();     // cutoff time for speed measurement
   // start computing on the GPU, two-dimensional task
   m_OpenCL.Execute(0, 2, offset, work);
   
   // get results into output buffer
   m_OpenCL.BufferRead(1, m_wavelet_data_GPU, 0, 0, m_xsize * m_ysize);
   
   time = GetTickCount() - time;
   
   m_OpenCL.Shutdown(); // free all resources - call all necessary functions CL***Free
   return true;
}
In the source code of the example, there is a commented out line calling PreparePriceData to prepare
an input array based on real prices: you can activate it instead of the previous line with the
PrepareModelData call (which generates an artificial number).

---

## Page 2045

Part 7. Advanced language tools
2045
7.1 0 Built-in support for parallel computing: OpenCL
void OnStart()
{
   int momentum_period = 8;
   double price_data[];
   double momentum_data[];
   PrepareModelData(price_data, SIZE_X + momentum_period);
   
   // PreparePriceData("EURUSD", PERIOD_M1, price_data, SIZE_X + momentum_period);
   
   PrepareMomentumData(price_data, momentum_data, momentum_period);
   ... // visualization of the series and increments
   CWavelet wavelet;
   uint time_gpu = 0;
   wavelet.CalculateWavelet_GPU(momentum_data, time_gpu);
   ... // visualization of the result of the wavelet transform
}
A special set of error codes (with the ERR_OPENCL_ prefix, starting with code 51 00,
ERR_OPENCL_NOT_SUPPORTED) has been allocated for operations with OpenCL. The codes are
described in the help. If there are problems with the execution of OpenCL programs, the terminal
outputs detailed diagnostics to the log, indicating error codes.

---

## Page 2046

Conclusion
2046
 
Conclusion
This section concludes the book. Throughout severn parts and numerous chapters, we have explored
various aspects of MQL5 programming, starting from the language basics and advancing to related
sophisticated technologies that enable a gradual transition from creating individual trader-specific tools
to complex trading systems and products.
The knowledge you gain will assist you in bringing various ideas to life and achieving success in the
world of professional algorithmic trading.
• Develop applications and sell them through the Market, the largest store of programs for
MetaTrader with a ready infrastructure for authors. The Market provides access to a huge
audience, offering product protection and licensing along with an integrated system for accepting
payments.
• Develop custom applications via Freelance. Access the entire array of development orders and
benefit from a convenient working system and payment protection.
• Share your experience by publishing your code in the Code Base. Present your programs to
thousands of traders from the MQL5.community.
And, of course, keep learning. The www.mql5.com website features a wealth of information and ready-
made algorithms:
• Programming articles, in which professional authors address practical problems.
• Forum where you can exchange experiences and seek advice from other developers.
• Code Base with program source codes to aid in learning the capabilities of the MQL5 languages and
creating your own programs.
Finally, I would like to remind you that software development involves not only programming but also
many other equally important areas: writing technical specifications (even if only for yourself),
designing, prototyping, creating user interface design, providing documentation, and further support. All
these aspects significantly influence the efficiency of your work as a programmer and the quality of the
final result.
In particular, most practical tasks can be broken down into standard algorithms and principles that
different language programmers have been using for a long time. This includes design patterns,
collections of data structures optimized for specific tasks, and tools for automating development. All of
this should be applied in the MetaTrader 5 platform with the help of MQL5 and in addition to it. While
the book is just the first step on the path to professional growth.

---

## Page 2047

Conclusion
2047
 
Join www.mql5.com — the community of trading robot developers!


---

