# MQL5 Book - Part 7 (Pages 1201-1400)

## Page 1201

Part 6. Trading automation
1 201 
6.4 Creating Expert Advisors
enum TRADE_RETCODE
{
   OK_0           = 0,      // no standard constant
   REQUOTE        = 10004,  // TRADE_RETCODE_REQUOTE
   REJECT         = 10006,  // TRADE_RETCODE_REJECT
   CANCEL         = 10007,  // TRADE_RETCODE_CANCEL
   PLACED         = 10008,  // TRADE_RETCODE_PLACED
   DONE           = 10009,  // TRADE_RETCODE_DONE
   DONE_PARTIAL   = 10010,  // TRADE_RETCODE_DONE_PARTIAL
   ERROR          = 10011,  // TRADE_RETCODE_ERROR
   TIMEOUT        = 10012,  // TRADE_RETCODE_TIMEOUT
   INVALID        = 10013,  // TRADE_RETCODE_INVALID
   INVALID_VOLUME = 10014,  // TRADE_RETCODE_INVALID_VOLUME
   INVALID_PRICE  = 10015,  // TRADE_RETCODE_INVALID_PRICE
   INVALID_STOPS  = 10016,  // TRADE_RETCODE_INVALID_STOPS
   TRADE_DISABLED = 10017,  // TRADE_RETCODE_TRADE_DISABLED
   MARKET_CLOSED  = 10018,  // TRADE_RETCODE_MARKET_CLOSED
   ...
};
   
#define TRCSTR(X) EnumToString((TRADE_RETCODE)(X))
So, the use of TRCSTR(r.retcode), where r is a structure, will provide a minimal description of the
numeric code.
We will consider an example of applying a macro and analyzing a structure in the next section about the
OrderCheck function.
6.4.1 1  Request validation: OrderCheck
To perform any trading operation, the MQL program must first fill the MqlTradeRequest structure with
the necessary data. Before sending it to the server using trading functions, it makes sense to check it
for formal correctness and evaluate the consequences of the request, in particular, the amount of
margin that will be required and the remaining free funds. This check is performed by the OrderCheck
function.
bool OrderCheck(const MqlTradeRequest &request, MqlTradeCheckResult &result)
If there are not enough funds if the parameters are filled incorrectly, the function returns false. In
addition, the function also reacts with a refusal when the trading is disabled, both in the terminal as a
whole and for a specific program. For the error code check the retcode field of the result structure.
Successful check of structure request and the trading environment ends with the status true, however,
this does not guarantee that the requested operation will certainly succeed if it is repeated using the
functions OrderSend or OrderSendAsync. Trading conditions may change between calls or the broker on
the server may have settings applied for a specific external trading system that cannot be satisfied in
the formal verification algorithm that is performed by OrderCheck.
To obtain a description of the expected financial result, you should analyze the fields of the structure
result.
Unlike the OrderCalcMargin function which calculates the estimated margin required for only one
proposed position or order, OrderCheck takes into account, albeit in a simplified mode, the general

---

## Page 1202

Part 6. Trading automation
1 202
6.4 Creating Expert Advisors
state of the trading account. So it fills the margin field in the MqlTradeCheckResult structure and other
related fields (margin_ free, margin_ level) with cumulative variables that will be formed after the
execution of the order. For example, if a position is already open for any instrument at the time of the
OrderCheck call and the request being checked increases the position, the margin field will reflect the
amount of deposit, including previous margin liabilities. If the new order contains an operation in the
opposite direction, the margin will not increase (in reality, it should decrease, because a position should
be closed completely on a netting account and the hedging margin should be applied for opposite
positions on a hedging account; however, the function does not perform such accurate calculations).
First of all, OrderCheck is useful for programmers at the initial stage of getting acquainted with the
trading API in order to experiment with requests without sending them to the server.
Let's test the performance of the fOrderCheck unction using a simple non-trading Expert Advisor
CustomOrderCheck.mq5. We made it an Expert Advisor and not a script for ease of use: this way it will
remain on the chart after being launched with the current settings, which can be easily edited by
changing individual input parameters. With a script, we would have to start over by setting the fields
each time from the default values.
To run the check, let's set a timer in OnInit.
void OnInit()
{
   // initiate pending execution
   EventSetTimer(1);
}
As for the timer handler, the main algorithm will be implemented there. At the very beginning we cancel
the timer since we need the code to be executed once, and then wait for the user to change the
parameters.
void OnTimer()
{
   // execute the code once and wait for new user settings
   EventKillTimer();
   ...
}
The Expert Advisor's input parameters completely repeat the set of fields of the trade request
structure.

---

## Page 1203

Part 6. Trading automation
1 203
6.4 Creating Expert Advisors
input ENUM_TRADE_REQUEST_ACTIONS Action = TRADE_ACTION_DEAL;
input ulong Magic;
input ulong Order;
input string Symbol;    // Symbol (empty = current _Symbol)
input double Volume;    // Volume (0 = minimal lot)
input double Price;     // Price (0 = current Ask)
input double StopLimit;
input double SL;
input double TP;
input ulong Deviation;
input ENUM_ORDER_TYPE Type;
input ENUM_ORDER_TYPE_FILLING Filling;
input ENUM_ORDER_TYPE_TIME ExpirationType;
input datetime ExpirationTime;
input string Comment;
input ulong Position;
input ulong PositionBy;
Many of them do not affect the check and financial performance but are left so that you can be sure of
this.
By default, the state of the variables corresponds to the request to open a position with the minimum
lot of the current instrument. In particular, the Type parameter without explicit initialization will get the
value of 0, which is equal to the ORDER_TYPE_BUY member of the ENUM_ORDER_TYPE structure. In
the Action parameter, we specified an explicit initialization because 0 does not correspond to any
element of the ENUM_TRADE_REQUEST_ACTIONS enumeration (the first element of
TRADE_ACTION_DEAL is 1 ).
void OnTimer()
{
   ...
   // initialize structures with zeros
   MqlTradeRequest request = {};
   MqlTradeCheckResult result = {};
   
   // default values
   const bool kindOfBuy = (Type & 1) == 0;
   const string symbol = StringLen(Symbol) == 0 ? _Symbol : Symbol;
   const double volume = Volume == 0 ?
      SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN) : Volume;
   const double price = Price == 0 ?
      SymbolInfoDouble(symbol, kindOfBuy ? SYMBOL_ASK : SYMBOL_BID) : Price;
   ...
Let's fill in the structure. Real robots usually only need to assign a few fields, but since this test is
generic, we must ensure that any parameters that the user enters are passed.

---

## Page 1204

Part 6. Trading automation
1 204
6.4 Creating Expert Advisors
   request.action = Action;
   request.magic = Magic;
   request.order = Order;
   request.symbol = symbol;
   request.volume = volume;
   request.price = price;
   request.stoplimit = StopLimit;
   request.sl = SL;
   request.tp = TP;
   request.deviation = Deviation;
   request.type = Type;
   request.type_filling = Filling;
   request.type_time = ExpirationType;
   request.expiration = ExpirationTime;
   request.comment = Comment;
   request.position = Position;
   request.position_by = PositionBy;
   ...
Please note that here we do not normalize prices and lots yet, although it is required in the real
program. Thus, this test makes it possible to enter "uneven" values and make sure that they lead to an
error. In the following examples, normalization will be enabled.
Then we call OrderCheck and log the request and result structures. We are only interested in the
retcode field of the latter, so it is additionally printed with "decryption" as text, macro TRCSTR
(TradeRetcode.mqh). You can also analyze a string field comment, but its format may change so that it
is more suitable for display to the user.
   ResetLastError();
   PRTF(OrderCheck(request, result));
   StructPrint(request, ARRAYPRINT_HEADER);
   Print(TRCSTR(result.retcode));
   StructPrint(result, ARRAYPRINT_HEADER, 2);
   ...
The output of structures is provided by a helper function StructPrint which is based on ArrayPrint.
Because of this, we will still get a "raw" display of data. In particular, the elements of enumerations are
represented by numbers "as is". Later we will develop a function for a more transparent (user-friendly)
MqlTradeRequest structure output (see TradeUtils.mqh).
To facilitate the analysis of the results, at the beginning of the OnTimer function we will display the
current state of the account, and at the end, for comparison, we will calculate the margin for a given
trading operation using the function OrderCalcMargin.

---

## Page 1205

Part 6. Trading automation
1 205
6.4 Creating Expert Advisors
void OnTimer()
{
   PRTF(AccountInfoDouble(ACCOUNT_EQUITY));
   PRTF(AccountInfoDouble(ACCOUNT_PROFIT));
   PRTF(AccountInfoDouble(ACCOUNT_MARGIN));
   PRTF(AccountInfoDouble(ACCOUNT_MARGIN_FREE));
   PRTF(AccountInfoDouble(ACCOUNT_MARGIN_LEVEL));
   ...
   // filling in the structure MqlTradeRequest
   // calling OrderCheck and printing results
   ...
   double margin = 0;
   ResetLastError();
   PRTF(OrderCalcMargin(Type, symbol, volume, price, margin));
   PRTF(margin);
}
Below is an example of logs for XAUUSD with default settings.
AccountInfoDouble(ACCOUNT_EQUITY)=15565.22 / ok
AccountInfoDouble(ACCOUNT_PROFIT)=0.0 / ok
AccountInfoDouble(ACCOUNT_MARGIN)=0.0 / ok
AccountInfoDouble(ACCOUNT_MARGIN_FREE)=15565.22 / ok
AccountInfoDouble(ACCOUNT_MARGIN_LEVEL)=0.0 / ok
OrderCheck(request,result)=true / ok
[action] [magic] [order] [symbol] [volume] [price] [stoplimit] [sl] [tp] [deviation] [type] »
       1       0       0 "XAUUSD"     0.01 1899.97        0.00 0.00 0.00           0      0 »
 » [type_filling] [type_time]        [expiration] [comment] [position] [position_by] [reserved]
 »             0           0 1970.01.01 00:00:00 ""                 0             0          0
OK_0
[retcode] [balance] [equity] [profit] [margin] [margin_free] [margin_level] [comment] [reserved]
        0  15565.22 15565.22     0.00    19.00      15546.22       81922.21 "Done"             0
OrderCalcMargin(Type,symbol,volume,price,margin)=true / ok
margin=19.0 / ok
The next example shows an estimate of the expected increase in margin on the account, where there is
already an open position which we are going to double.

---

## Page 1206

Part 6. Trading automation
1 206
6.4 Creating Expert Advisors
AccountInfoDouble(ACCOUNT_EQUITY)=9999.540000000001 / ok
AccountInfoDouble(ACCOUNT_PROFIT)=-0.83 / ok
AccountInfoDouble(ACCOUNT_MARGIN)=79.22 / ok
AccountInfoDouble(ACCOUNT_MARGIN_FREE)=9920.32 / ok
AccountInfoDouble(ACCOUNT_MARGIN_LEVEL)=12622.49431961626 / ok
OrderCheck(request,result)=true / ok
[action] [magic] [order]  [symbol] [volume] [price] [stoplimit] [sl] [tp] [deviation] [type] »
       1       0       0 "PLZL.MM"      1.0 12642.0         0.0  0.0  0.0           0      0 »
 » [type_filling] [type_time]        [expiration] [comment] [position] [position_by] [reserved]
 »              0           0 1970.01.01 00:00:00 ""                 0             0          0
OK_0
[retcode] [balance] [equity] [profit] [margin] [margin_free] [margin_level] [comment] [reserved]
        0  10000.87  9999.54    -0.83   158.26       9841.28        6318.43 "Done"             0
OrderCalcMargin(Type,symbol,volume,price,margin)=true / ok
margin=79.04000000000001 / ok
Try changing any request parameters and see if the request is successful. Incorrect parameter
combinations will cause error codes from the standard list, but since there are many more invalid
options than reserved ones (the most common errors), the function can often return the generic code
TRADE_RETCODE_INVALID (1 001 3). In this regard, it is recommended to implement your own
structure checks with a greater degree of diagnostics.
When sending real requests to the server, the same TRADE_RETCODE_INVALID code is used under
various unforeseen circumstances, for example, when trying to re-edit an order whose modification
operation has already been started (but has not yet been completed) in the external trading system.
6.4.1 2 Request sending result: MqlTradeResult structure
In response to a trade request executed by the OrderSend or OrderSendAsync functions, which we'll
cover in the next section, the server returns the request processing results. For this purpose, a special
predefined structure is used MqlTradeResult.
struct MqlTradeResult 
{ 
   uint     retcode;          // Operation result code 
   ulong    deal;             // Transaction ticket, if it is completed 
   ulong    order;            // Order ticket, if it is placed 
   double   volume;           // Trade volume confirmed by the broker 
   double   price;            // Trade price confirmed by the broker 
   double   bid;              // Current market bid price 
   double   ask;              // Current market offer price 
   string   comment;          // Broker's comment on the operation 
   uint     request_id;       // Request identifier, set by the terminal when sending  
   uint     retcode_external; // Response code of the external trading system 
};
The following table describes its fields.

---

## Page 1207

Part 6. Trading automation
1 207
6.4 Creating Expert Advisors
Field
Description
retcode
Trade server return code
deal
Deal ticket if it is performed (during the TRADE_ACTION_DEAL
trading operation)
order
Order ticket if it is placed (during the TRADE_ACTION_PENDING
trading operation)
volume
Trade volume confirmed by the broker (depends on the order
execution modes)
price
The price in the deal confirmed by the broker (depends on the
deviation field in the trade request, execution mode, and the trading
operation)
bid
Current market bid price
ask
Current market ask price
comment
Broker's comment on the trade (by default, it is filled in with the
decryption of the trade server return code)
request_id
Request ID which is set by the terminal when sending it to the trade
server
retcode_external
Error code returned by the external trading system
As we will see below when conducting trading operations, a variable of type MqlTradeResult is passed as
the second parameter by reference in the OrderSend or OrderSendAsync function. It returns the result.
When sending a trade request to the server, the terminal sets the request_ id identifier to a unique
value. This is necessary for the analysis of subsequent trading events, which is required if an
asynchronous function OrderSendAsync is used. This identifier allows you to associate the sent request
with the result of its processing passed to the OnTradeTransaction event handler.
The presence and types of errors in the retcode_ external field depend on the broker and the external
trading system into which trading operations are forwarded.
Request results are analyzed in different ways, depending on the trading operations and the way they
are sent. We will deal with this in subsequent sections on specific actions: market buy and sell, placing
and deleting pending orders, and modifying and closing positions.
6.4.1 3 Sending a trade request: OrderSend and OrderSendAsync
To perform trading operations, the MQL5 API provides two functions: OrderSend and OrderSendAsync.
Just like OrderCheck, they perform a formal check of the request parameters passed in the form of the
MqlTradeRequest structure and then, if successful, send a request to the server.  
The difference between the two functions is as follows. OrderSend expects for the order to be queued
for processing on the server and receives meaningful data from it into the fields of the MqlTradeResult
structure which is passed as the second function parameter. OrderSendAsync immediately returns
control to the calling code regardless of how the server responds. At the same time, from all fields of

---

## Page 1208

Part 6. Trading automation
1 208
6.4 Creating Expert Advisors
the MqlTradeResult structure except retcode, important information is filled only into request_ id. Using
this request identifier, an MQL program can receive further information about the progress of
processing this request in the OnTradeTransaction event. An alternative approach is to periodically
analyze the lists of orders, deals, and positions. This can also be done in a loop, setting some timeout in
case of communication problems.
It's important to note that despite the "Async" suffix in the second function's name, the first function
without this suffix is also not fully synchronous. The fact is that the result of order processing by the
server, in particular, the execution of a deal (or, probably, several deals based on one order) and the
opening of a position, generally occurs asynchronously in an external trading system. So the OrderSend
function also requires delayed collection and analysis of the consequences of request execution, which
MQL programs must, if necessary, implement themselves. We'll look at an example of truly
synchronous sending of a request and receiving all of its results later (see MqlTradeSync.mqh).
bool OrderSend(const MqlTradeRequest &request, MqlTradeResult &result)
The function returns true in case of a successful basic check of the request structure in the terminal
and a few additional checks on the server. However, this only indicates the acceptance of the order by
the server and does not guarantee a successful execution of the trade operation.
The trade server can fill the field deal or order values in the returned result structure if this data is
known at the time the server formats an answer to the OrderSend call. However, in the general case,
the events of deal execution or placing limit orders corresponding to an order can occur after the
response is sent to the MQL program in the terminal. Therefore, for any type of trade request, when
receiving the execution OrderSend result, it is necessary to check the trade server return code retcode
and external trading system response code retcode_ external (if necessary) which are available in the
returned result structure. Based on them, you should decide whether to wait for pending actions on the
server or take your own actions.
Each accepted order is stored on the trade server pending processing until any of the following events
that affect its life cycle occurs:
• execution when a counter request appears
• triggered when the execution price arrives
• expiration date
• cancellation by the user or MQL program
• removal by the broker (for example, in case of clearing or shortage of funds, Stop Out)
The OrderSendAsync prototype completely repeats that of OrderSend.
bool OrderSendAsync(const MqlTradeRequest &request, MqlTradeResult &result)
The function is intended for high-frequency trading, when, according to the conditions of the algorithm,
it is unacceptable to waste time waiting for a response from the server. The use of OrderSendAsync
does not speed up request processing by the server or request sending to the external trading system.
Attention! In the tester, the OrderSendAsync function works like OrderSend. This makes it difficult
to debug the pending processing of asynchronous requests.
The function returns true upon a successful sending of the request to the MetaTrader 5 server.
However, this does not mean that the request reached the server and was accepted for processing. At
the same time, the response code in the receiving result structure contains the
TRADE_RETCODE_PLACED (1 0008) value, that is, "the order has been placed".

---

## Page 1209

Part 6. Trading automation
1 209
6.4 Creating Expert Advisors
When processing the received request, the server will send a response message to the terminal about a
change in the current state of positions, orders and deals, which leads to the generation of the OnTrade
event in an MQL program. There, the program can analyze the new trading environment and account
history. We will look at relevant examples below.
Also, the details of the trader request execution on the server can be tracked using the
OnTradeTransaction handler. At the same time, it should be considered that as a result of the execution
of one trade request, the OnTradeTransaction handler will be called multiple times. For example, when
sending a market buy request, it is accepted for processing by the server, a corresponding 'buy' order
is created for the account, the order is executed and the trader is performed, as a result of which it is
removed from the list of open orders and added to the history of orders. Then the trade is added to the
history and a new position is created. For each of these events, the OnTradeTransaction function will be
called.
Let's start with a simple Expert Advisor example CustomOrderSend.mq5. It allows you to set all fields of
the request in the input parameters, which is similar to CustomOrderCheck.mq5, but further differs in
that it sends a request to the server instead of a simple check in the terminal. Run the Expert Advisor
on your demo account. After completing the experiments, don't forget to remove the Expert Advisor
from the chart or close the chart so that you don't send a test request every next launch of the
terminal.
The new example has several other improvements. First of all the input parameter Async is added.
input bool Async = false;
This option allows selecting the function that will send the request to the server. By default, the
parameter equals to false and the OrderSend function is used. If you set it to true, OrderSendAsync will
be called.
In addition, with this example, we will begin to describe and complete a special set of functions in the
header file TradeUtils.mqh, which will come in handy to simplify the coding of robots. All functions are
placed in the namespace TU (from "Trade Utilities"), and first, we introduce functions for convenient
output to the structure log MqlTradeRequest and MqlTradeResult.

---

## Page 1210

Part 6. Trading automation
1 21 0
6.4 Creating Expert Advisors
namespace TU
{
   string StringOf(const MqlTradeRequest &r)
   {
      SymbolMetrics p(r.symbol);
      
      // main block: action, type, symbol      
      string text = EnumToString(r.action);
      if(r.symbol != NULL) text += ", " + r.symbol;
      text += ", " + EnumToString(r.type);
      // volume block
      if(r.volume != 0) text += ", V=" + p.StringOf(r.volume, p.lotDigits);
      text += ", " + EnumToString(r.type_filling);
      // block of all prices 
      if(r.price != 0) text += ", @ " + p.StringOf(r.price);
      if(r.stoplimit != 0) text += ", X=" + p.StringOf(r.stoplimit);
      if(r.sl != 0) text += ", SL=" + p.StringOf(r.sl);
      if(r.tp != 0) text += ", TP=" + p.StringOf(r.tp);
      if(r.deviation != 0) text += ", D=" + (string)r.deviation;
      // pending orders expiration block
      if(IsPendingType(r.type)) text += ", " + EnumToString(r.type_time);
      if(r.expiration != 0) text += ", " + TimeToString(r.expiration);
      // modification block
      if(r.order != 0) text += ", #=" + (string)r.order;
      if(r.position != 0) text += ", #P=" + (string)r.position;
      if(r.position_by != 0) text += ", #b=" + (string)r.position_by;
      // auxiliary data
      if(r.magic != 0) text += ", M=" + (string)r.magic;
      if(StringLen(r.comment)) text += ", " + r.comment;
      
      return text;
   }
   
   string StringOf(const MqlTradeResult &r)
   {
      string text = TRCSTR(r.retcode);
      if(r.deal != 0) text += ", D=" + (string)r.deal;
      if(r.order != 0) text += ", #=" + (string)r.order;
      if(r.volume != 0) text += ", V=" + (string)r.volume;
      if(r.price != 0) text += ", @ " + (string)r.price; 
      if(r.bid != 0) text += ", Bid=" + (string)r.bid; 
      if(r.ask != 0) text += ", Ask=" + (string)r.ask; 
      if(StringLen(r.comment)) text += ", " + r.comment;
      if(r.request_id != 0) text += ", Req=" + (string)r.request_id;
      if(r.retcode_external != 0) text += ", Ext=" + (string)r.retcode_external;
      
      return text;
   }
   ...
};

---

## Page 1211

Part 6. Trading automation
1 21 1 
6.4 Creating Expert Advisors
The purpose of the functions is to provide all significant (non-empty) fields in a concise but convenient
form: they are displayed in one line with a unique designation for each.
As you can see, the function uses the SymbolMetrics class for MqlTradeRequest. It facilitates the
normalization of multiple prices or volumes for the same instrument. Don't forget that the normalization
of prices and volumes is a prerequisite for preparing a correct trade request.
   class SymbolMetrics
   {
   public:
      const string symbol;
      const int digits;
      const int lotDigits;
      
      SymbolMetrics(const string s): symbol(s),
         digits((int)SymbolInfoInteger(s, SYMBOL_DIGITS)),
         lotDigits((int)MathLog10(1.0 / SymbolInfoDouble(s, SYMBOL_VOLUME_STEP)))
      { }
         
      double price(const double p)
      {
         return TU::NormalizePrice(p, symbol);
      }
      
      double volume(const double v)
      {
         return TU::NormalizeLot(v, symbol);
      }
   
      string StringOf(const double v, const int d = INT_MAX)
      {
         return DoubleToString(v, d == INT_MAX ? digits : d);
      }
   };
The direct normalization of values is entrusted to auxiliary functions NormalizePrice and NormalizeLot
(the scheme of the latter is identical to what we saw in the file LotMarginExposure.mqh).
   double NormalizePrice(const double price, const string symbol = NULL)
   {
      const double tick = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
      return MathRound(price / tick) * tick;
   }
If we connect TradeUtils.mqh, the example CustomOrderSend.mq5 has the following form (the omitted
code fragments '...' remained unchanged from CustomOrderCheck.mq5).

---

## Page 1212

Part 6. Trading automation
1 21 2
6.4 Creating Expert Advisors
void OnTimer()
{
   ...
   MqlTradeRequest request = {};
   MqlTradeCheckResult result = {};
   
   TU::SymbolMetrics sm(symbol);
   
   // fill in the request structure
   request.action = Action;
   request.magic = Magic;
   request.order = Order;
   request.symbol = symbol;
   request.volume = sm.volume(volume);
   request.price = sm.price(price);
   request.stoplimit = sm.price(StopLimit);
   request.sl = sm.price(SL);
   request.tp = sm.price(TP);
   request.deviation = Deviation;
   request.type = Type;
   request.type_filling = Filling;
   request.type_time = ExpirationType;
   request.expiration = ExpirationTime;
   request.comment = Comment;
   request.position = Position;
   request.position_by = PositionBy;
   
   // send the request and display the result
   ResetLastError();
   if(Async)
   {
      PRTF(OrderSendAsync(request, result));
   }
   else
   {
      PRTF(OrderSend(request, result));
   }
   Print(TU::StringOf(request));
   Print(TU::StringOf(result));
}
Due to the fact that prices and volume are now normalized, you can try to enter uneven values into the
corresponding input parameters. They are often obtained in programs during calculations, and our code
converts them according to the symbol specification.
With default settings, the Expert Advisor creates a request to buy the minimum lot of the current
instrument by market and makes it using the OrderSend function.

---

## Page 1213

Part 6. Trading automation
1 21 3
6.4 Creating Expert Advisors
OrderSend(request,result)=true / ok
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.12462
DONE, D=1250236209, #=1267684253, V=0.01, @ 1.12462, Bid=1.12456, Ask=1.12462, Request executed, Req=1
As a rule, with allowed trading, this operation should be completed successfully (status DONE, comment
"Request executed"). In the result structure, we immediately received the deal number D.
If we open Expert Advisor settings and replace the value of the parameter Async with true, we will send
a similar request but with the OrderSendAsync function.
OrderSendAsync(request,result)=true / ok
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.12449
PLACED, Order placed, Req=2
In this case, the status is PLACED, and the trade number at the time the function returns is not known.
We only know the unique request ID Req=2. To get the deal and position number, you need to intercept
the TRADE_TRANSACTION_REQUEST message with the same request ID in the OnTradeTransaction
handler, where the filled structure will be received as a MqlTradeResult parameter.
From the user's point of view, both requests should be equally fast.
It will be possible to compare the performance of these two functions directly in the code of an MQL
program using another example of an Expert Advisor (see the section on synchronous and asynchronous
requests), which we will consider after studying the model of trading events.
It should be noted that trading events are sent to the OnTradeTransaction handler (if present in the
code), regardless of which function is used to send requests, OrderSend or OrderSendAsync. The
situation is as follows: in case of applying OrderSend some or all information about the execution of
the order is immediately available in the receiving MqlTradeResult structure. However, in the
general case, the result is distributed over time and volume, for example, when one order is "filled"
into several deals. Then complete information can be obtained from trading events or by analyzing
the history of transactions and orders.
If you try to send a deliberately incorrect request, for example, change the order type to a pending
ORDER_TYPE_BUY_STOP, you will get an error message, because for such orders you should use the
TRADE_ACTION_PENDING action. Furthermore, they should be located at a distance from the current
price (we use the market price by default). Before this test, it is important not to forget to change the
query mode back to synchronous (Async=false) to immediately see the error in the MqlTradeResult
structure after ending the OrderSend call. Otherwise, OrderSendAsync would return true, but the order
would still not be set, and the program could receive information about this only in OnTradeTransaction
which we don't have yet.
OrderSend(request,result)=false / TRADE_SEND_FAILED(4756)
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, @ 1.12452, ORDER_TIME_GTC
REQUOTE, Bid=1.12449, Ask=1.12452, Requote, Req=5
In this case, the error reports an invalid Requote price.
Examples of using functions to perform specific trading actions will be presented in the following
sections.

---

## Page 1214

Part 6. Trading automation
1 21 4
6.4 Creating Expert Advisors
6.4.1 4 Buying and selling operations
In this section, we finally begin to study the application of MQL5 functions for specific trading tasks.
The purpose of these functions is to fill in the MqlTradeRequest structure in a special way and call the
OrderSend or OrderSendAsync function.
The first action we will learn is buying or selling a financial instrument at the current market price. The
procedure for performing this action includes:
·Creating a market order based on a submitted order
·Executing a deal (one or several) under an order
·The result should be an open position
As we saw in the section on types of trading operations, instant buy/sell corresponds to the
TRADE_ACTION_DEAL element in the ENUM_TRADE_REQUEST_ACTIONS enumeration. Therefore,
when filling the MqlTradeRequest structure, write TRADE_ACTION_DEAL in the action field.
The trade direction is set using the type field which should contain one of the order types:
ORDER_TYPE_BUY or ORDER_TYPE_SELL.
Of course, to buy or sell, you need to specify the name of the symbol in the symbol field and its desired
volume in the volume field.
The type_ filling field must be filled with one of the fill policies from the enumeration
ENUM_ORDER_TYPE_FILLING, which is chosen based on the character property
SYMBOL_FILLING_MODE with allowed policies.
Optionally, the program can fill in the fields with protective price levels (sl and tp), a comment
(comment), and an Expert Advisor ID (magic).
The contents of other fields are set differently depending on the price execution mode for the selected
symbol. In some modes, certain fields have no effect. For example, in the Request Execution and
Instant Execution modes, the field with the price must be filled in with a suitable price (the last known
Ask for buying and Bid for selling), and the deviation field may contain the maximum allowable deviation
of the price from the set price for the successful execution of a deal. In Exchange Execution and Market
Execution, these fields are ignored. In order to simplify the source code, you can fill in the price and
slippage uniformly in all modes, but in the last two options, the price will still be selected and
substituted by the trade server according to the rules of the modes.
Other fields of the MqlTradeRequest structure not mentioned here are not used for these trading
operations.
The following table summarizes the rules for filling the fields for different execution modes. The required
fields are marked with an asterisk, while optional fields are marked with a plus.

---

## Page 1215

Part 6. Trading automation
1 21 5
6.4 Creating Expert Advisors
Field
Request
Instant
Exchange
Market
action
*
*
*
*
symbol
*
*
*
*
volume
*
*
*
*
type
*
*
*
*
type_filling
*
*
*
*
price
*
*
 
 
sl
+
+
+
+
tp
+
+
+
+
deviation
+
+
 
 
magic
+
+
+
+
comment
+
+
+
+
Depending on server settings, it may be forbidden to fill in fields with protective sl and tp levels at the
moment of opening a position. This is often the case for exchange execution or market execution
modes, but the MQL5 API does not provide properties for clarifying this circumstance in advance. In
such cases, Stop Loss and Take Profit should be set by modifying an already open position. By the way,
this method can be recommended for all execution modes, since it is the only one that allows you to
accurately postpone the protective levels from the real position opening price. On the other hand,
creating and setting up a position in two moves can lead to a situation where the position is open, but
the second request to set protective levels failed for one reason or another.
Regardless of the trade direction (buy/sell), the Stop Loss order is always placed as a stop order
(ORDER_TYPE_BUY_STOP or ORDER_TYPE_SELL_STOP), and the Take Profit order is placed as a limit
order (ORDER_TYPE_BUY_LIMIT or ORDER_TYPE_SELL_LIMIT). Moreover, stop orders are always
controlled by the MetaTrader 5 server and only when the price reaches the specified level, they are
sent to the external trading system. In contrast, limit orders can be output directly to an external
trading system. Specifically, this is usually the case for exchange-traded instruments.
In order to simplify the coding of trading operations, not only buying and selling but also all others, we
will start from this section by developing classes, or rather structures that provide automatic and
correct filling of fields for trade requests, as well as a truly synchronous waiting for the result. The
latter is especially important, given that the OrderSend and OrderSendAsync functions return control to
the calling code before the trading action is completed in full. In particular, for market buy and sell, the
algorithm usually needs to know not the ticket number of the order created on the server, but whether

---

## Page 1216

Part 6. Trading automation
1 21 6
6.4 Creating Expert Advisors
the position is open or not. Depending on this, it can, for example, modify the position by setting Stop
Loss and Take Profit if it has opened or repeat attempts to open it if the order was rejected. 
A little later we will learn about the OnTrade and OnTradeTransaction trading events, which inform the
program about changes in the account state, including the status of orders, deals, and positions.
However, dividing the algorithm into two fragments – separately generating orders according to certain
signals or rules, and separately analyzing the situation in event handlers – makes the code less
understandable and maintainable.
In theory, the asynchronous programming paradigm is not inferior to the synchronous one either in
speed or in ease of coding. However, the ways of its implementation can be different, for example,
based on direct pointers to callback functions (a basic technique in Java, JavaScript, and many other
languages) or events (as in MQL5), which predetermines some features, which will be discussed in the
OnTradeTransaction section. Asynchronous mode allows you to speed up the sending of requests due to
deferred control over their execution. But this control will still need to be done sooner or later in the
same thread, so the average performance of the circuits is the same.
All new structures will be placed in the MqlTradeSync.mqh file. In order not to "reinvent the wheel",
let's take the built-in MQL5 structures as a starting point and describe our structures as child
structures. For example, to get query results, let's define MqlTradeResultSync, which is derived from
MqlTradeResult. Here we will add useful fields and methods, in particular, the position field to store an
open position ticket as a result of a market buy or sell operation.
struct MqlTradeResultSync: public MqlTradeResult
{
   ulong position;
   ...
};
The second important improvement will be a constructor that resets all fields (this saves us from having
to specify explicit initialization when describing variables of a structure type).
   MqlTradeResultSync()
   {
      ZeroMemory(this);
   }
Next, we will introduce a universal synchronization mechanism, i.e., waiting for the results of a request
(each type of request will have its own rules for checking readiness).
Let's define the type of the condition callback function. A function of this type must take the
MqlTradeResultSync structure parameter and return true if successful: the result of the operation is
received.
   typedef bool (*condition)(MqlTradeResultSync &ref);
Functions like this are meant to be passed to the wait method, which implements a cyclic check for the
readiness of the result during a predefined timeout in milliseconds.

---

## Page 1217

Part 6. Trading automation
1 21 7
6.4 Creating Expert Advisors
   bool wait(condition p, const ulong msc = 1000)
   {
      const ulong start = GetTickCount64();
      bool success;
      while(!(success = p(this)) && GetTickCount64() - start < msc);
      return success;
   }
Let's clarify right away that the timeout is the maximum waiting time: even if it is set to a very large
value, the loop will end immediately as soon as the result is received, which can happen instantly. Of
course, a meaningful timeout should last no more than a few seconds.
Let's see an example of a method that will be used to synchronously wait for an order to appear on the
server (it doesn't matter with what status: status analysis is the task of the calling code).
   static bool orderExist(MqlTradeResultSync &ref)
   {
      return OrderSelect(ref.order) || HistoryOrderSelect(ref.order);
   }
Two built-in MQL5 API functions are applied here, OrderSelect and HistoryOrderSelect: they search and
logically select an order by its ticket in the internal trading environment of the terminal. First, this
confirms the existence of an order (if one of the functions returned true), and second, it allows you to
read its properties using other functions, which is not important to us yet. We will cover all these
features in separate sections. The two functions are written in conjunction because a market order can
be filled so quickly that its active phase (falling into OrderSelect) will immediately flow into history
(HistoryOrderSelect).
Note that the method is declared static. This is due to the fact that MQL5 does not support pointers to
object methods. If this were the case, we could declare the method non-static while using the
prototype of the pointer to the condition callback functions without the parameter referencing to
MqlTradeResultSync (since all fields are present inside the this object).
The waiting mechanism can be started as follows:
   if(wait(orderExist))
   {
      // there is an order
   }
   else
   {
      // timeout
   }
Of course, this fragment must be executed after we receive a result from the server with the status
TRADE_RETCODE_DONE or TRADE_RETCODE_DONE_PARTIAL, and the order field in the
MqlTradeResultSync structure is guaranteed to contain an order ticket. Please note that due to the
system's distributed nature, an order from the server may not immediately be displayed in the terminal
environment. That's why you need waiting time.
As long as the orderExist function returns false into the wait method, the wait loop inside runs until the
timeout expires. Under normal circumstances, we will almost instantly find an order in the terminal
environment, and the loop will end with a sign of success (true).

---

## Page 1218

Part 6. Trading automation
1 21 8
6.4 Creating Expert Advisors
The positionExist function that checks the presence of an open position in a similar but a little more
complicated way. Since the previous orderExist function has completed checking the order, its ticket
contained in the field ref.order of the structure is confirmed as working.
   static bool positionExist(MqlTradeResultSync &ref)
   {
      ulong posid, ticket;
      if(HistoryOrderGetInteger(ref.order, ORDER_POSITION_ID, posid))
      {
         // in most cases, the position ID is equal to the ticket,
         // but not always: the full code implements getting a ticket by ID,
         // for which there are no built-in MQL5 tools
         ticket = posid;
         
         if(HistorySelectByPosition(posid))
         {
            ref.position = ticket;
            ...
            return true;
         }
      }
      return false;
   }
Using the built-in HistoryOrderGetInteger and HistorySelectByPosition functions, we get the ID and
ticket of the position based on the order.
Later we will see the use of orderExist and positionExist when verifying a buy/sell request, but now let's
turn to another structure: MqlTradeRequestSync. It is also inherited from the built-in one and contains
additional fields, in particular, a structure with a result (so as not to describe it in the calling code) and
a timeout for synchronous requests.
struct MqlTradeRequestSync: public MqlTradeRequest
{
   MqlTradeResultSync result;
   ulong timeout;
   ...
Since the inherited fields of the new structure are public, the MQL program can assign values to them
explicitly, just as it was done with the standard MqlTradeRequest structure. The methods that we will
add to perform trading operations will consider, check and, if necessary, correct these values for the
valid ones.
In the constructor, we reset all fields and set the symbol to the default value if the parameter is
omitted.
   MqlTradeRequestSync(const string s = NULL, const ulong t = 1000): timeout(t)
   {
      ZeroMemory(this);
      symbol = s == NULL ? _Symbol : s;
   }
In theory, due to the fact that all fields of the structure are public, they can technically be assigned
directly, but this is not recommended for those fields that require validation and for which we

---

## Page 1219

Part 6. Trading automation
1 21 9
6.4 Creating Expert Advisors
implement setter methods: they will be called before performing trading operations. The first of these
methods is setSymbol.
It fills the symbol field making sure the transmitted ticker exists and initiates the subsequent setting of
the volume filling mode.
   bool setSymbol(const string s)
   {
      if(s == NULL)
      {
         if(symbol == NULL)
         {
            Print("symbol is NULL, defaults to " + _Symbol);
            symbol = _Symbol;
            setFilling();
         }
         else
         {
            Print("new symbol is NULL, current used " + symbol);
         }
      }
      else
      {
         if(SymbolInfoDouble(s, SYMBOL_POINT) == 0)
         {
            Print("incorrect symbol " + s);
            return false;
         }
         if(symbol != s)
         {
            symbol = s;
            setFilling();
         }
      }
      return true;
   }
So, changing the symbol with setSymbol will automatically pick up the correct filling mode via a nested
call of setFilling.
The setFilling method provides the automatic specification of the volume filling method based on the
SYMBOL_FILLING_MODE and SYMBOL_TRADE_EXEMODE symbol properties (see the section Trading
conditions and order execution modes).

---

## Page 1220

Part 6. Trading automation
1 220
6.4 Creating Expert Advisors
private:
   void setFilling()
   {
      const int filling = (int)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
      const bool market = SymbolInfoInteger(symbol, SYMBOL_TRADE_EXEMODE)
         == SYMBOL_TRADE_EXECUTION_MARKET;
      
      // the field may already be filled
      // and bit match means a valid mode
      if(((type_filling + 1) & filling) != 0
         || (type_filling == ORDER_FILLING_RETURN && !market)) return;
      
      if((filling & SYMBOL_FILLING_FOK) != 0)
      {
         type_filling = ORDER_FILLING_FOK;
      }
      else if((filling & SYMBOL_FILLING_IOC) != 0)
      {
         type_filling = ORDER_FILLING_IOC;
      }
      else
      {
         type_filling = ORDER_FILLING_RETURN;
      }
   }
This method implicitly (without errors and messages) corrects the type_ filling field if the Expert Advisor
has set it incorrectly. If your algorithm requires a guaranteed specific fill method, without which trading
is impossible, make appropriate edits to interrupt the process.
For the set of structures being developed, it is assumed that, in addition to the type_ filling field, you
can directly set only optional fields without specific requirements for their content, such as magic
or comment.
In what follows, many of the methods are provided in a shorter form for the sake of simplicity. They
have parts for the types of operations we'll look at later, as well as branched error checking.
For the buy and sell operations, we need the price and volume fields; both these values should be
normalized and checked for the acceptable range. This is done by the setVolumePrices method.
   bool setVolumePrices(const double v, const double p,
      const double stop, const double take)
   {
      TU::SymbolMetrics sm(symbol);
      volume = sm.volume(v);
      
      if(p != 0) price = sm.price(p);
      else price = sm.price(TU::GetCurrentPrice(type, symbol));
      
      return setSLTP(stop, take);
   }
If the transaction price is not set (p == 0), the program will automatically take the current price of the
correct type, depending on the direction, which is read from the type field.

---

## Page 1221

Part 6. Trading automation
1 221 
6.4 Creating Expert Advisors
Although the Stop Loss and Take Profit levels are not required, they should also be normalized if
present, which is why they are added to the parameters of this method.
The abbreviation TU is already known to us. It stands for the namespace in the file TradeUtilits.mqh
with a lot of useful functions, including those for the normalization of prices and volumes.
Processing of sl and tp fields is performed by the separate setSLTP method because this is needed not
only in the buy and sell operations but also when modifying an existing position.
   bool setSLTP(const double stop, const double take)
   {
      TU::SymbolMetrics sm(symbol);
      TU::TradeDirection dir(type);
  
      if(stop != 0)
      {
         sl = sm.price(stop);
         if(!dir.worse(sl, price))
         {
            PrintFormat("wrong SL (%s) against price (%s)",
               TU::StringOf(sl), TU::StringOf(price));
            return false;
         }
      }
      else
      {
         sl = 0; // remove SL
      }
      
      if(take != 0)
      {
         tp = sm.price(take);
         if(!dir.better(tp, price))
         {
            PrintFormat("wrong TP (%s) against price (%s)",
               TU::StringOf(tp), TU::StringOf(price));
            return false;
         }
      }
      else
      {
         tp = 0; // remove TP
      }
      return true;
   }
In addition to normalizing and assigning values to sl and tp fields, this method checks the mutual
correct location of the levels relative to the price. For this purpose, the TradeDirection class is
described in the space TU.
Its constructors allow you to specify the analyzed direction of trade: buying or selling, in the context of
which it is easy to identify a profitable or unprofitable mutual arrangement of two prices. With this
class, the analysis is performed in a unified way and the checks in the code are reduced by 2 times

---

## Page 1222

Part 6. Trading automation
1 222
6.4 Creating Expert Advisors
since there is no need to separately process buy and sell operations. In particular, the worse method
has two price parameters p1 , p2, and returns true if the price p1  is placed worse, i.e., unprofitable, in
relation to the price p2. A similar method better represents reverse logic: it will return true if the price
p1  is better than price p2. For example, for a sale, the best price is placed lower because Take Profit is
below the current price.
TU::TradeDirection dir(ORDER_TYPE_SELL);
Print(dir.better(100, 200)); // true
Now, if an order is placed incorrectly, the setSLTP function logs a warning and aborts the verification
process without attempting to correct the values since the appropriate response may vary in different
programs. For example, from the two passed stop and take levels only one can be wrong, and then it
probably makes sense to use the second (correct) one.
You can change the behavior, for example, by skipping the assignment of invalid values (then the
protection levels simply will not be changed) or adding a field with an error flag to the structure (for
such a structure, an attempt to send a request should be suppressed so as not to load the server with
obviously impossible requests). Sending an invalid request will end with the retcode error code equal to
TRADE_RETCODE_INVALID_STOPS.
The setSLTP method also checks to make sure that the protective levels are not located closer to the
current price than the number of points in the SYMBOL_TRADE_STOPS_LEVEL property of the symbol
(if this property is set, i.e. greater than 0), and position modification is not requested when it is inside
the SYMBOL_TRADE_FREEZE_LEVEL freeze area (if it is set). These nuances are not shown here: they
can be found in the source code.
Now we are ready to implement a group of trading methods. For example, for buying and selling with
the most complete set of fields, we define buy and sell methods.
public:
   ulong buy(const string name, const double lot, const double p = 0,
      const double stop = 0, const double take = 0)
   {
      type = ORDER_TYPE_BUY;
      return _market(name, lot, p, stop, take);
   }
   ulong sell(const string name, const double lot, const double p = 0,
      const double stop = 0, const double take = 0)
   {
      type = ORDER_TYPE_SELL;
      return _market(name, lot, p, stop, take);
   }
As already mentioned, to set optional fields like deviation, comment, and magic should do a direct
assignment before calling buy/sell. This is all the more convenient since deviation and magic in most
cases are set once, and used in subsequent queries.
The methods return an order ticket, but below we will show in action the mechanism of "synchronous"
receipt of a position ticket, and this will be a ticket of a created or modified position (if position
increase or partial closing was done).
Methods buy and sell differ only in the type field value, while everything else is the same. This is why the
general part is framed as a separate method _ market. This is where we set action in
TRADE_ACTION_DEAL, and call setSymbol and setVolumePrices.

---

## Page 1223

Part 6. Trading automation
1 223
6.4 Creating Expert Advisors
private:
   ulong _market(const string name, const double lot, const double p = 0,
      const double stop = 0, const double take = 0)
   {
      action = TRADE_ACTION_DEAL;
      if(!setSymbol(name)) return 0;
      if(!setVolumePrices(lot, p, stop, take)) return 0;
      ...
Next, we could just call OrderSend, but given the possibility of requotes (price updates on the server
during the time the order was sent), let's wrap the call in a loop. Due to this, the method will be able to
retry several times, but no more than the preset number of times MAX_REQUOTES (the macro is
chosen to be 1 0 in the code).
      int count = 0;
      do
      {
         ZeroMemory(result);
         if(OrderSend(this, result)) return result.order;
         // automatic price selection means automatic processing of requotes
         if(result.retcode == TRADE_RETCODE_REQUOTE)
         {
            Print("Requote N" + (string)++count);
            if(p == 0)
            {
               price = TU::GetCurrentPrice(type, symbol);
            }
         }
      }
      while(p == 0 && result.retcode == TRADE_RETCODE_REQUOTE 
         && ++count < MAX_REQUOTES);
      return 0;
   }
Since the financial instrument is set in the structure constructor by default, we can provide a couple of
simplified overloads of buy/sell methods without the symbol parameter.
public:
   ulong buy(const double lot, const double p = 0,
      const double stop = 0, const double take = 0)
   {
      return buy(symbol, lot, p, stop, take);
   }
   
   ulong sell(const double lot, const double p = 0,
      const double stop = 0, const double take = 0)
   {
      return sell(symbol, lot, p, stop, take);
   }
Thus, in a minimal configuration, it will be enough for the program to call request.buy(1 .0) in order to
make a one-lot buy operation.

---

## Page 1224

Part 6. Trading automation
1 224
6.4 Creating Expert Advisors
Now let's get back to the problem of obtaining the final result of the request, which in the case of the
operation TRADE_ACTION_DEAL means the position ticket. In the MqlTradeRequestSync structure, this
problem is solved by the completed method: for each type of operation, it must ask for the nested
MqlTradeResultSync structure to wait for its filling in accordance with the type of operation.
   bool completed()
   {
      if(action == TRADE_ACTION_DEAL)
      {
         const bool success = result.opened(timeout);
         if(success) position = result.position;
         return success;
      }
      ...
      return false;
   }
Position opening is controlled by the opened method. Inside we will find a couple of calls to the wait
method described above: the first one is for orderExist, and the second one is for positionExist.

---

## Page 1225

Part 6. Trading automation
1 225
6.4 Creating Expert Advisors
   bool opened(const ulong msc = 1000)
   {
      if(retcode != TRADE_RETCODE_DONE
         && retcode != TRADE_RETCODE_DONE_PARTIAL)
      {
         return false;
      }
      
      if(!wait(orderExist, msc))
      {
         Print("Waiting for order: #" + (string)order);
      }
      
      if(deal != 0)
      {
         if(HistoryDealGetInteger(deal, DEAL_POSITION_ID, position))
         {
            return true;
         }
         Print("Waiting for position for deal D=" + (string)deal);
      }
      
      if(!wait(positionExist, msc))
      {
         Print("Timeout");
         return false;
      }
      position = result.position;
      
      return true;
   }
Of course, it makes sense to wait for an order and a position to appear only if the status of the retcode
indicates success. Other statuses refer to errors or cancellation of the operation, or to specific
intermediate codes (TRADE_RETCODE_PLACED, TRADE_RETCODE_TIMEOUT) that are not
accompanied by useful information in other fields. In both cases, this prevents further processing within
this "synchronous" framework.
It is important to note that we are using OrderSync and therefore we rely on the obligatory presence of
the order ticket in the structure received from the server.
In some cases, the system sends not only an order ticket but also a deal ticket at the same time. Then
from the deal, you can find the position faster. But even if there is information about the deal, the
trading environment of the terminal may temporarily not have information about the new position. That
is why you should wait for it with wait(positionExist).
Let's sum up for the intermediate result. The created structures allow you to write the following code
to buy 1  lot of the current symbol:

---

## Page 1226

Part 6. Trading automation
1 226
6.4 Creating Expert Advisors
   MqlTradeRequestSync request;
   if(request.buy(1.0) && request.completed())
   {
      Print("OK Position: P=", request.result.position);
   }
We get inside the block of the conditional operator only with a guaranteed open position, and we know
its ticket. If we used only buy/sell methods, they would receive an order ticket at their output and
would have to check the execution themselves. In case of an error, we will not get inside the if block,
and the server code will be contained in request.result.retcode.
When we implement methods for other trades in the following sections, they can be executed in a
similar "blocking" mode, for example, to modify stop levels:
  if(request.adjust(SL, TP) && request.completed())
  {
     Print("OK Adjust")
  }
Of course, you are not required to call completed if you don't want to check the result of the operation
in blocking mode. Instead, you can stick to the asynchronous paradigm and analyze the environment in
trading events handlers. But even in this case, the MqlTradeRequestAsync structure can be useful for
checking and normalizing operation parameters.
Let's write a test Expert Advisor MarketOrderSend.mq5 to put all this together. The input parameters
will provide input of values for the main and some optional fields of the trade request.
enum ENUM_ORDER_TYPE_MARKET
{
   MARKET_BUY = ORDER_TYPE_BUY,  // ORDER_TYPE_BUY
   MARKET_SELL = ORDER_TYPE_SELL // ORDER_TYPE_SELL
};
   
input string Symbol;         // Symbol (empty = current _Symbol)
input double Volume;         // Volume (0 = minimal lot)
input double Price;          // Price (0 = current Ask)
input ENUM_ORDER_TYPE_MARKET Type;
input string Comment;
input ulong Magic;
input ulong Deviation;
The ENUM_ORDER_TYPE_MARKET enumeration is a subset of the standard ENUM_ORDER_TYPE and is
introduced in order to limit the available types of operations to only two: market buy and sell.
The action will run once on a timer, in the same way as in the previous examples.
void OnInit()
{
   // scheduling a delayed start
   EventSetTimer(1);
}
In the timer handler, we disable the timer so that the request is executed only once. For the next
launch, you will need to change the Expert Advisor parameters.

---

## Page 1227

Part 6. Trading automation
1 227
6.4 Creating Expert Advisors
void OnTimer()
{
   EventKillTimer();
   ...
Let's describe a variable of type MqlTradeRequestSync and prepare the values for the main fields.
   const bool wantToBuy = Type == MARKET_BUY;
   const string symbol = StringLen(Symbol) == 0 ? _Symbol : Symbol;
   const double volume = Volume == 0 ?
      SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN) : Volume;
   MqlTradeRequestSync request(symbol);
   ...
Optional fields will be filled in directly.
   request.magic = Magic;
   request.deviation = Deviation;
   request.comment = Comment;
   ...
Among the optional fields, you can select the fill mode (type_ filling). By default, MqlTradeRequestSync
automatically writes to this field the first of the allowed modes ENUM_ORDER_TYPE_FILLING. Recall
that the structure has a special method setFilling for this.
Next, we call the buy or sell method with parameters, and if it returns an order ticket, we wait for an
open position to appear.
   ResetLastError();
   const ulong order = (wantToBuy ?
      request.buy(volume, Price) :
      request.sell(volume, Price));
   if(order != 0)
   {
      Print("OK Order: #=", order);
      if(request.completed()) // waiting for an open position
      {
         Print("OK Position: P=", request.result.position);
      }
   }
   Print(TU::StringOf(request));
   Print(TU::StringOf(request.result));
}
At the end of the function, the query and result structures are logged for reference.
If we run the Expert Advisor with the default parameters (buying the current symbol with the minimum
lot), we can get the following result for "XTIUSD".

---

## Page 1228

Part 6. Trading automation
1 228
6.4 Creating Expert Advisors
OK Order: #=218966930
Waiting for position for deal D=215494463
OK Position: P=218966930
TRADE_ACTION_DEAL, XTIUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 109.340, P=218966930
DONE, D=215494463, #=218966930, V=0.01, @ 109.35, Request executed, Req=8
Pay attention to the warning about the temporary absence of a position: it will always appear due to
the distributed processing of requests (the warnings themselves can be disabled by removing the
SHOW_WARNINGS macro in the Expert Advisor code, but the situation will remain). But thanks to the
developed new structures, the applied code is not diverted by these internal complexities and is written
in the form of a sequence of simple steps, where each next one is "confident" in the success of the
previous ones.
On a netting account, we can achieve an interesting effect of position reversal by subsequent selling
with a doubled minimum lot (0.02 in this case).
OK Order: #=218966932
Waiting for position for deal D=215494468
Position ticket <> id: 218966932, 218966930
OK Position: P=218966932
TRADE_ACTION_DEAL, XTIUSD, ORDER_TYPE_SELL, V=0.02, ORDER_FILLING_FOK, @ 109.390, P=218966932
DONE, D=215494468, #=218966932, V=0.02, @ 109.39, Request executed, Req=9
It is important to note that after the reversal, the position ticket ceases to be equal to the position
identifier: the identifier remains from the first order, and the ticket remains from the second. We
deliberately bypassed the task of finding the position ticket by its identifier in order to simplify the
presentation. In most cases, the ticket and ID are the same, but for precise control, use the
TU::PositionSelectById function. Those interested can study the attached source code.
Identifiers are constant as long as the position exists (until it closes to zero in terms of volume) and are
useful for analyzing the account history. Tickets describe positions while they are open (there is no
concept of a position ticket in history) and are used in some types of requests, in particular, to modify
protection levels or close with an opposite position. But there are nuances associated with pouring in
parts. We'll talk more about position properties in a separate section.
When making a buy or sell operation, our buy/sell methods allow you to immediately set the Stop Loss
and/or Take Profit levels. To do this, simply pass them as additional parameters obtained from input
variables or calculated using some formulas. For example,
input double SL;
input double TP;
...
void OnTimer()
{
   ...
   const ulong order = (wantToBuy ?
      request.buy(symbol, volume, Price, SL, TP) :
      request.sell(symbol, volume, Price, SL, TP));
   ...
All methods of the new structures provide automatic normalization of the passed parameters, so there
is no need to use NormalizeDouble or something else.
It has already been noted above that some server settings may prohibit the setting of protective levels
at the position opening time. In this case, you should set the sl and tp fields via a separate request.

---

## Page 1229

Part 6. Trading automation
1 229
6.4 Creating Expert Advisors
Exactly the same request is also used in those cases when it is required to modify already set levels, in
particular, to implement trailing stop or trailing profit.
In the next section, we will complete the current example with a delayed setting of sl and tp with the
second request after the successful opening of a position.
6.4.1 5 Modifying Stop Loss and/or Take Profit levels of a position
An MQL program can change protective Stop Loss and Take Profit price levels for an open position. The
TRADE_ACTION_SLTP element in the ENUM_TRADE_REQUEST_ACTIONS enumeration is intended for
this purpose, that is, when filling the MqlTradeRequest structure, we should write TRADE_ACTION_SLTP
in the action field.
This is the only required field. The need to fill in other fields is determined by the account operation
mode ENUM_ACCOUNT_MARGIN_MODE. On hedging accounts, you should fill in the symbol field, but
you can omit the position ticket. On hedging accounts, on the contrary, it is mandatory to indicate the
position position ticket, but you can omit the symbol. This is due to the specifics of position
identification on accounts of different types. During netting, only one position can exist for each
symbol. 
In order to unify the code, it is recommended to fill in both fields if information is available.
Protective price levels are set in the sl and tp fields. It is possible to est only one of the fields. To
remove protective levels, assign zero values to them.
The following table summarizes the requirements for filling in the fields depending on the counting
modes. Required fields are marked with an asterisk, optional fields are marked with a plus.
Field
Netting
Hedging
action
*
*
symbol
*
+
position
+
*
sl
+
+
tp
+
+
To perform the operation of modifying protective levels, we introduce several overloads of the adj ust
method in the MqlTradeRequestSync structure.

---

## Page 1230

Part 6. Trading automation
1 230
6.4 Creating Expert Advisors
struct MqlTradeRequestSync: public MqlTradeRequest
{
   ...
   bool adjust(const ulong pos, const double stop = 0, const double take = 0);
   bool adjust(const string name, const double stop = 0, const double take = 0);
   bool adjust(const double stop = 0, const double take = 0);
   ...
};
As we saw above, depending on the environment, modification can be done only by ticket or only by
position symbol. These options are taken into account in the first two prototypes.
In addition, since the structure may have already been used for previous requests, it may have filled
position and symbols fields. Then you can call the method with the last prototype.
We do not yet show the implementation of these three methods, because it is clear that it must have a
common body with sending a request. This part is framed as a private helper method _ adj ust with a full
set of options. Here its code is given with some abbreviations that do not affect the logic of work.
private:
   bool _adjust(const ulong pos, const string name,
      const double stop = 0, const double take = 0)
   {
      action = TRADE_ACTION_SLTP;
      position = pos;
      type = (ENUM_ORDER_TYPE)PositionGetInteger(POSITION_TYPE);
      if(!setSymbol(name)) return false;
      if(!setSLTP(stop, take)) return false;
      ZeroMemory(result);
      return OrderSend(this, result);
   }
We fill in all the fields of the structure according to the above rules, calling the previously described
setSymbol and setSLTP methods, and then send a request to the server. The result is a success status
(true) or errors (false).
Each of the overloaded adj ust methods separately prepares source parameters for the request. This is
how it is done in the presence of a position ticket.
public:
   bool adjust(const ulong pos, const double stop = 0, const double take = 0)
   {
      if(!PositionSelectByTicket(pos))
      {
         Print("No position: P=" + (string)pos);
         return false;
      }
      return _adjust(pos, PositionGetString(POSITION_SYMBOL), stop, take);
   }
Here, using the built-in PositionSelectByTicket function, we check for the presence of a position and its
selection in the trading environment of the terminal, which is necessary for the subsequent reading of
its properties, in this case, the symbol (PositionGetString(POSITION_ SYMBOL)). Then the universal
variant is called adj ust.

---

## Page 1231

Part 6. Trading automation
1 231 
6.4 Creating Expert Advisors
When modifying a position by symbol name (which is only available on a netting account), you can use
another option adj ust.
   bool adjust(const string name, const double stop = 0, const double take = 0)
   {
      if(!PositionSelect(name))
      {
         Print("No position: " + s);
         return false;
      }
      
      return _adjust(PositionGetInteger(POSITION_TICKET), name, stop, take);
   }
Here, position selection is done using the built-in PositionSelect function, and the ticket number is
obtained from its properties (PositionGetInteger(POSITION_ TICKET)).
All of these features will be discussed in detail in their respective sections on working with positions and
position properties.
The adj ust method version with the most minimalist set of parameters, i.e. with only stop and take
levels, is as follows.

---

## Page 1232

Part 6. Trading automation
1 232
6.4 Creating Expert Advisors
   bool adjust(const double stop = 0, const double take = 0)
   {
      if(position != 0)
      {
         if(!PositionSelectByTicket(position))
         {
            Print("No position with ticket P=" + (string)position);
            return false;
         }
         const string s = PositionGetString(POSITION_SYMBOL);
         if(symbol != NULL && symbol != s)
         {
            Print("Position symbol is adjusted from " + symbol + " to " + s);
         }
         symbol = s;
      }
      else if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)
         != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
         && StringLen(symbol) > 0)
      {
         if(!PositionSelect(symbol))
         {
            Print("Can't select position for " + symbol);
            return false;
         }
         position = PositionGetInteger(POSITION_TICKET);
      }
      else
      {
         Print("Neither position ticket nor symbol was provided");
         return false;
      }
      return _adjust(position, symbol, stop, take);
   }
This code ensures that the position and symbols fields are filled correctly in various modes or that it
exits early with an error message in the log. At the end, the private version of _ adj ust is called, which
sends the request via OrderSend.
Similar to buy/sell methods, the set of adj ust methods works "asynchronously": upon their completion,
only the request sending status is known, but there is no confirmation of the modification of the levels.
As we know, for the stock exchange, the Take Profit level can be forwarded as a limit order. Therefore,
in the MqlTradeResultSync structure, we should provide a "synchronous" wait until the changes take
effect.
The general wait mechanism formed as the MqlTradeResultSync::wait method is already ready and has
been used to wait for the opening of a position. The wait method receives as the first parameter a
pointer to another method with a predefined prototype condition to poll in a loop until the required
condition is met or a timeout occurs. In this case, this condition-compatible method should perform an
applied check of the stop levels in the position.
Let's add such a new method called adj usted.

---

## Page 1233

Part 6. Trading automation
1 233
6.4 Creating Expert Advisors
struct MqlTradeResultSync: public MqlTradeResult
{
   ...
   bool adjusted(const ulong msc = 1000)
   {
      if(retcode != TRADE_RETCODE_DONE || retcode != TRADE_RETCODE_PLACED)
      {
         return false;
      }
   
      if(!wait(checkSLTP, msc))
      {
         Print("SL/TP modification timeout: P=" + (string)position);
         return false;
      }
      
      return true;
   }
First of all, of course, we check the status in the field retcode. If there is a standard status, we
continue checking the levels themselves, passing to wait an auxiliary method checkSLTP.
struct MqlTradeResultSync: public MqlTradeResult
{
   ...
   static bool checkSLTP(MqlTradeResultSync &ref)
   {
      if(PositionSelectByTicket(ref.position))
      {
         return TU::Equal(PositionGetDouble(POSITION_SL), /*.?.*/)
            && TU::Equal(PositionGetDouble(POSITION_TP), /*.?.*/);
      }
      else
      {
         Print("PositionSelectByTicket failed: P=" + (string)ref.position);
      }
      return false;
   }
This code ensures that the position is selected by ticket in the trading environment of the terminal
using PositionSelectByTicket and reads the position properties POSITION_SL and POSITION_TP, which
should be compared with what was in the request. The problem is that here we don't have access to
the request object and we must somehow pass here a couple of values for the places marked with '.?.'.
Basically, since we are designing the MqlTradeResultSync structure, we can add sl and tp fields to it and
fill them with values from MqlTradeRequestSync before sending the request (the kernel does not "know"
about our added fields and will leave them untouched during theOrderSend call). But for simplicity, we
will use what is already available. The bid and ask fields in the MqlTradeResultSync structure are only
used to report requote prices (TRADE_RETCODE_REQUOTE status), which is not related to the
TRADE_ACTION_SLTP request, so we can store the sl and tp from the completed MqlTradeRequestSync
in them.

---

## Page 1234

Part 6. Trading automation
1 234
6.4 Creating Expert Advisors
It is logical to make this in the completed method of the MqlTradeRequestSync structure which starts a
blocking wait for the trading operation results with a predefined timeout. So far, its code has only had
one branch for the TRADE_ACTION_DEAL action. To continue, let's add a branch for
TRADE_ACTION_SLTP.
struct MqlTradeRequestSync: public MqlTradeRequest
{
   ...
   bool completed()
   {
      if(action == TRADE_ACTION_DEAL)
      {
         const bool success = result.opened(timeout);
         if(success) position = result.position;
         return success;
      }
      else if(action == TRADE_ACTION_SLTP)
      {
         // pass the original request data for comparison with the position properties,
         // by default they are not in the result structure
         result.position = position;
         result.bid = sl; // bid field is free in this result type, use under StopLoss
         result.ask = tp; // ask field is free in this type of result, we use it under TakeProfit
         return result.adjusted(timeout);
      }
      return false;
   }
As you can see, after setting the position ticket and price levels from the request, we call the adj usted
method discussed above which checks wait(checkSLTP). Now we can return to the helper method
checkSLTP in the MqlTradeResultSync structure and bring it to its final form.
struct MqlTradeResultSync: public MqlTradeResult
{
   ...
   static bool checkSLTP(MqlTradeResultSync &ref)
   {
      if(PositionSelectByTicket(ref.position))
      {
         return TU::Equal(PositionGetDouble(POSITION_SL), ref.bid) // sl from request
            && TU::Equal(PositionGetDouble(POSITION_TP), ref.ask); // tp from request
      }
      else
      {
         Print("PositionSelectByTicket failed: P=" + (string)ref.position);
      }
      return false;
   }
This completes the extension of the functionality of structures MqlTradeRequestSync and
MqlTradeResultSync for the of Stop Loss and Take Profit modification operation.

---

## Page 1235

Part 6. Trading automation
1 235
6.4 Creating Expert Advisors
With this in mind, let's continue with the example of the Expert Advisor MarketOrderSend.mq5 which we
started in the previous section. Let's add to it an input parameter Distance2SLTP, which allows you to
specify the distance in points to the levels Stop Loss and Take Profit.
input int Distance2SLTP = 0; // Distance to SL/TP in points (0 = no)
When it is zero, no guard levels will be set.
In the working code, after receiving confirmation of opening a position, we calculate the values of the
levels in the SL and TP variables and perform a synchronous modification: request.adj ust(SL, TP) &&
request.completed().
   ...
   const ulong order = (wantToBuy ?
      request.buy(symbol, volume, Price) :
      request.sell(symbol, volume, Price));
   if(order != 0)
   {
      Print("OK Order: #=", order);
      if(request.completed()) // waiting for position opening
      {
         Print("OK Position: P=", request.result.position);
         if(Distance2SLTP != 0)
         {
            // position "selected" in the trading environment of the terminal inside 'complete',
            // so it is not required to do this explicitly on the ticket
            // PositionSelectByTicket(request.result.position);
            
            // with the selected position, you can find out its properties, but we need the price,
            // to step back from it by a given number of points
            const double price = PositionGetDouble(POSITION_PRICE_OPEN);
            const double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
            // we count the levels using the auxiliary class TradeDirection
            TU::TradeDirection dir((ENUM_ORDER_TYPE)Type);
            // SL is always "worse" and TP is always "better" of the price: the code is the same for buying and selling
            const double SL = dir.negative(price, Distance2SLTP * point);
            const double TP = dir.positive(price, Distance2SLTP * point);
            if(request.adjust(SL, TP) && request.completed())
            {
               Print("OK Adjust");
            }
         }
      }
   }
   Print(TU::StringOf(request));
   Print(TU::StringOf(request.result));
}
In the first call of completed after a successful buy or sell operation, the position ticket is saved in the
position field of the request structure. Therefore, to modify stops, only price levels are sufficient, and
the symbol and ticket of the position are already present in request.

---

## Page 1236

Part 6. Trading automation
1 236
6.4 Creating Expert Advisors
Let's try to execute a buy operation using the Expert Advisor with default settings but with
Distance2SLTP set at 500 points.
OK Order: #=1273913958
Waiting for position for deal D=1256506526
OK Position: P=1273913958
OK Adjust
TRADE_ACTION_SLTP, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.10889, »
»  SL=1.10389, TP=1.11389, P=1273913958
DONE, Bid=1.10389, Ask=1.11389, Request executed, Req=26
The last two lines correspond to the debug output to the log of the contents of the request and
request.result structures, initiated at the end of the function. In these lines, it is interesting that the
fields store a symbiosis of values from two queries: first, a position was opened, and then it was
modified. In particular, the fields with volume (0.01 ) and price (1 .1 0889) in the request remained after
TRADE_ACTION_DEAL, but did not prevent the execution of TRADE_ACTION_SLTP. In theory, it is easy
to get rid of this by resetting the structure between two requests, however, we preferred to leave them
as they are, because among the filled fields there are also useful ones: the position field received the
ticket we need to request the modification. If we reset the structure, then we would need to introduce
a variable for intermediate storage of the ticket.
In general cases, of course, it is desirable to adhere to a strict data initialization policy, but knowing
how to use them in specific scenarios (such as two or more related requests of a predefined type)
allows you to optimize your code.
Also, one should not be surprised that in the structure with the result, we see the requested levels sl
and tp in the fields for the Bid and Ask prices: they were written there by the
MqlTradeRequestSync::completed method for the purpose of comparison with the actual position
changes. When executing the request, the system kernel filled only retcode (DONE), comment
("Request executed"), and request_ id (26) in the result structure.
Next, we will consider another example of level modification that implements the trailing stop.
6.4.1 6 Trailing stop
One of the most common tasks where the ability to change protective price levels is used is to
sequentially shift Stop Loss at a better price as the favorable trend continues. This is the trailing stop.
We implement it using new structures MqlTradeRequestSync and MqlTradeResultSync from previous
sections.
To be able to connect the mechanism to any Expert Advisor, let's declare it as the Trailing Stop class
(see the file TrailingStop.mqh). We will store the number of the controlled position, its symbol, and the
size of the price point, as well as the required distance of the stop loss level from the current price, and
the step of level changes in the personal variables of the class.

---

## Page 1237

Part 6. Trading automation
1 237
6.4 Creating Expert Advisors
#include <MQL5Book/MqlTradeSync.mqh>
   
class TrailingStop
{
   const ulong ticket;  // ticket of controlled position
   const string symbol; // position symbol
   const double point;  // symbol price pip size
   const uint distance; // distance to the stop in points
   const uint step;     // movement step (sensitivity) in points
   ...
The distance is only needed for the standard position tracking algorithm provided by the base class.
Derived classes will be able to move the protective level according to other principles, such as moving
averages, channels, the SAR indicator, and others. After getting acquainted with the base class, we will
give an example of a derived class with a moving average.
Let's create the level variable for the current stop price level. In the ok variable, we will maintain the
current status of the position: true if the position still exists and false if an error occurred and the
position was closed.
protected:
   double level;
   bool ok;
   virtual double detectLevel() 
   {
      return DBL_MAX;  
   }
A virtual method detectLevel is intended for overriding in descendant classes, where the stop price
should be calculated according to an arbitrary algorithm. In this implementation, a special value
DBL_MAX is returned, indicating the work according to the standard algorithm (see below).
In the constructor, fill in all the fields with the values of the corresponding parameters. The
PositionSelectByTicket function checks for the existence of a position with a given ticket and allocates
it in the program environment so that the subsequent call of PositionGetString returns its string
property with the symbol name.

---

## Page 1238

Part 6. Trading automation
1 238
6.4 Creating Expert Advisors
public:
   TrailingStop(const ulong t, const uint d, const uint s = 1) :
      ticket(t), distance(d), step(s),
      symbol(PositionSelectByTicket(t) ? PositionGetString(POSITION_SYMBOL) : NULL),
      point(SymbolInfoDouble(symbol, SYMBOL_POINT))
   {
      if(symbol == NULL)
      {
         Print("Position not found: " + (string)t);
         ok = false;
      }
      else
      {
         ok = true;
      }
   }
   
   bool isOK() const
   {
      return ok;
   }
Now let's consider the main public method of the trail class. The MQL program will need to call it on
every tick or by timer to keep track of the position. The method returns true while the position exists.
   virtual bool trail()
   {
      if(!PositionSelectByTicket(ticket))
      {
         ok = false;
         return false; // position closed
      }
   
      // find out prices for calculations: current quote and stop level
      const double current = PositionGetDouble(POSITION_PRICE_CURRENT);
      const double sl = PositionGetDouble(POSITION_SL);
      ...
Here and below we use the position properties reading functions. They will be discussed in detail in a
separate section. In particular, we need to find out the direction of trade – buying and selling – in order
to know in which direction the stop level should be set.
      // POSITION_TYPE_BUY  = 0 (false)
      // POSITION_TYPE_SELL = 1 (true)
      const bool sell = (bool)PositionGetInteger(POSITION_TYPE);
      TU::TradeDirection dir(sell);
      ...
For calculations and checks, we will use the helper class TU::TradeDirection and its object dir. For
example, its negative method allows you to calculate the price located at a specified distance from the
current price in a losing direction, regardless of the type of operation. This simplifies the code because
otherwise you would have to do "mirror" calculations for buys and sells.

---

## Page 1239

Part 6. Trading automation
1 239
6.4 Creating Expert Advisors
      level = detectLevel();
      // we can't trail without a level: removing the stop level must be done by the calling code
      if(level == 0) return true;
      // if there is a default value, make a standard offset from the current price
      if(level == DBL_MAX) level = dir.negative(current, point * distance);
      level = TU::NormalizePrice(level, symbol);
      
      if(!dir.better(current, level))
      {
         return true; // you can't set a stop level on the profitable side<
      }
      ...
The better method of the TU::TradeDirection class checks that the received stop level is located on the
right side of the price. Without this method, we would need to write the check twice again (for buys and
sells).
We may get an incorrect stop level value since the detectLevel method can be overridden in derived
classes. With the standard calculation, this problem is eliminated because the level is calculated by the
dir object.
Finally, when the level is calculated, it is necessary to apply it to the position. If the position does not
already have a stop loss, any valid level will do. If the stop loss has already been set, then the new
value should be better than the previous one and differ by more than the specified step.
      if(sl == 0)
      {
         PrintFormat("Initial SL: %f", level);
         move(level);
      }
      else
      {
         if(dir.better(level, sl) && fabs(level - sl) >= point * step)
         {
            PrintFormat("SL: %f -> %f", sl, level);
            move(level);
         }
      }
      
      return true; // success
   }
Sending of a position modification request is implemented in the move method which uses the familiar
adj ust method of the MqlTradeRequestSync structure (see the section Modifying Stop Loss and/or Take
Profit levels).

---

## Page 1240

Part 6. Trading automation
1 240
6.4 Creating Expert Advisors
   bool move(const double sl)
   {
      MqlTradeRequestSync request;
      request.position = ticket;
      if(request.adjust(sl, 0) && request.completed())
      {
         Print("OK Trailing: ", TU::StringOf(sl));
         return true;
      }
      return false;
   }
};
Now everything is ready to add trailing to the test Expert Advisor TrailingStop.mq5. In the input
parameters, you can specify the trading direction, the distance to the stop level in points, and the step
in points. The TrailingDistance parameter equals 0 by default, which means automatic calculation of the
daily range of quotes and using half of it as a distance.
#include <MQL5Book/MqlTradeSync.mqh>
#include <MQL5Book/TrailingStop.mqh>
   
enum ENUM_ORDER_TYPE_MARKET
{
   MARKET_BUY = ORDER_TYPE_BUY,   // ORDER_TYPE_BUY
   MARKET_SELL = ORDER_TYPE_SELL  // ORDER_TYPE_SELL
};
   
input int TrailingDistance = 0;   // Distance to Stop Loss in points (0 = autodetect)
input int TrailingStep = 10;      // Trailing Step in points
input ENUM_ORDER_TYPE_MARKET Type;
input string Comment;
input ulong Deviation;
input ulong Magic = 1234567890;
When launched, the Expert Advisor will find if there is a position on the current symbol with the
specified Magic number and will create it if it doesn't exist.
Trailing will be carried out by an object of the TrailingStop class wrapped in a smart pointer AutoPtr.
Thanks to the latter, we don't need to manually delete the old object when it needs a new tracking
object to replace it for the new position being created. When a new object is assigned to a smart
pointer, the old object is automatically deleted. Recall that dereferencing a smart pointer, i.e.,
accessing the work object stored inside, is done using the overloaded [] operator.
#include <MQL5Book/AutoPtr.mqh>
   
AutoPtr<TrailingStop> tr;
In the OnTick handler, we check if there is an object. If there is one, check whether a position exists
(the attribute is returned from the trail method). Immediately after the program starts, the object is
not there, and the pointer is NULL. In this case, you should either create a new position or find an
already open one and create a Trailing Stop object for it. This is done by the Setup function. On
subsequent calls of OnTick, the object starts and continues tracking, preventing the program from going
inside the if block while the position is "alive".

---

## Page 1241

Part 6. Trading automation
1 241 
6.4 Creating Expert Advisors
void OnTick()
{
   if(tr[] == NULL || !tr[].trail())
   {
      // if there is no trailing yet, create or find a suitable position
      Setup();
   }
}
And here is the Setup function.

---

## Page 1242

Part 6. Trading automation
1 242
6.4 Creating Expert Advisors
void Setup()
{
   int distance = 0;
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
   if(trailing distance == 0) // auto-detect the daily range of prices
   {
      distance = (int)((iHigh(_Symbol, PERIOD_D1, 1) - iLow(_Symbol, PERIOD_D1, 1))
         / point / 2);
      Print("Autodetected daily distance (points): ", distance);
   }
   else
   {
      distance = TrailingDistance;
   }
   
   // process only the position of the current symbol and our Magic
   if(GetMyPosition(_Symbol, Magic))
   {
      const ulong ticket = PositionGetInteger(POSITION_TICKET);
      Print("The next position found: ", ticket);
      tr = new TrailingStop(ticket, distance, TrailingStep);
   }
   else // there is no our position
   {
      Print("No positions found, lets open it...");
      const ulong ticket = OpenPosition();
      if(ticket)
      {
         tr = new TrailingStop(ticket, distance, TrailingStep);
      }
   }
   
   if(tr[] != NULL)
   {
      // Execute trailing for the first time immediately after creating or finding a position
      tr[].trail();
   }
}
The search for a suitable open position is implemented in the GetMyPosition function, and opening a
new position is done by the OpenPosition function. Both are presented below. In any case, we get a
position ticket and create a trailing object for it.

---

## Page 1243

Part 6. Trading automation
1 243
6.4 Creating Expert Advisors
bool GetMyPosition(const string s, const ulong m)
{
   for(int i = 0; i < PositionsTotal(); ++i)
   {
      if(PositionGetSymbol(i) == s && PositionGetInteger(POSITION_MAGIC) == m)
      {
         return true;
      }
   }
   return false;
}
The purpose and the general meaning of the algorithm should be clear from the names of the built-in
functions. In the loop through all open positions (PositionsTotal), we sequentially select each of them
using PositionGetSymbol and get its symbol. If the symbol matches the requested one, we read and
compare the position property POSITION_MAGIC with the passed "magic". All functions for working
with positions will be discussed in a separate section.
The function will return true as soon as the first matching position is found. At the same time, the
position will remain selected in the trading environment of the terminal which makes it possible for the
rest of the code to read its other properties if necessary.
We already know the algorithm for opening a position.
ulong OpenPosition()
{
   MqlTradeRequestSync request;
   
   // default values
   const bool wantToBuy = Type == MARKET_BUY;
   const double volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   // optional fields are filled directly in the structure
   request.magic = Magic;
   request.deviation = Deviation;
   request.comment = Comment;
   ResetLastError();
   // execute the selected trade operation and wait for its confirmation
   if((bool)(wantToBuy ? request.buy(volume) : request.sell(volume))
      && request.completed())
   {
      Print("OK Order/Deal/Position");
   }
   
   return request.position; // non-zero value - sign of success
}
For clarity, let's see how this program works in the tester, in visual mode.
After compilation, let's open the strategy tester panel in the terminal, on the Review tab, and choose
the first option: Single test.
In the Settings tab, select the following:
• in the drop-down list Expert Advisor: MQL5Book\p6\TralingStop

---

## Page 1244

Part 6. Trading automation
1 244
6.4 Creating Expert Advisors
• Symbol: EURUSD
• Timeframe: H1 
• Interval: last year, month, or custom
• Forward: No
• Delays: disabled
• Modeling: based on real or generated ticks
• Optimization: disabled
• Visual mode: enabled
Once you press Start, you will see something like this in a separate tester window:
Standard trailing stop in the tester
The log will show entries that look like this:

---

## Page 1245

Part 6. Trading automation
1 245
6.4 Creating Expert Advisors
2022.01.10 00:02:00   Autodetected daily distance (points): 373
2022.01.10 00:02:00   No positions found, let's open it...
2022.01.10 00:02:00   instant buy 0.01 EURUSD at 1.13612 (1.13550 / 1.13612 / 1.13550)
2022.01.10 00:02:00   deal #2 buy 0.01 EURUSD at 1.13612 done (based on order #2)
2022.01.10 00:02:00   deal performed [#2 buy 0.01 EURUSD at 1.13612]
2022.01.10 00:02:00   order performed buy 0.01 at 1.13612 [#2 buy 0.01 EURUSD at 1.13612]
2022.01.10 00:02:00   Waiting for position for deal D=2
2022.01.10 00:02:00   OK Order/Deal/Position
2022.01.10 00:02:00   Initial SL: 1.131770
2022.01.10 00:02:00   position modified [#2 buy 0.01 EURUSD 1.13612 sl: 1.13177]
2022.01.10 00:02:00   OK Trailing: 1.13177
2022.01.10 00:06:13   SL: 1.131770 -> 1.131880
2022.01.10 00:06:13   position modified [#2 buy 0.01 EURUSD 1.13612 sl: 1.13188]
2022.01.10 00:06:13   OK Trailing: 1.13188
2022.01.10 00:09:17   SL: 1.131880 -> 1.131990
2022.01.10 00:09:17   position modified [#2 buy 0.01 EURUSD 1.13612 sl: 1.13199]
2022.01.10 00:09:17   OK Trailing: 1.13199
2022.01.10 00:09:26   SL: 1.131990 -> 1.132110
2022.01.10 00:09:26   position modified [#2 buy 0.01 EURUSD 1.13612 sl: 1.13211]
2022.01.10 00:09:26   OK Trailing: 1.13211
2022.01.10 00:09:35   SL: 1.132110 -> 1.132240
2022.01.10 00:09:35   position modified [#2 buy 0.01 EURUSD 1.13612 sl: 1.13224]
2022.01.10 00:09:35   OK Trailing: 1.13224
2022.01.10 10:06:38   stop loss triggered #2 buy 0.01 EURUSD 1.13612 sl: 1.13224 [#3 sell 0.01 EURUSD at 1.13224]
2022.01.10 10:06:38   deal #3 sell 0.01 EURUSD at 1.13221 done (based on order #3)
2022.01.10 10:06:38   deal performed [#3 sell 0.01 EURUSD at 1.13221]
2022.01.10 10:06:38   order performed sell 0.01 at 1.13221 [#3 sell 0.01 EURUSD at 1.13224]
2022.01.10 10:06:38   Autodetected daily distance (points): 373
2022.01.10 10:06:38   No positions found, let's open it...
Look how the algorithm shifts the SL level up with a favorable price movement, up to the moment when
the position is closed by stop loss. Immediately after liquidating a position, the program opens a new
one.
To check the possibility of using non-standard tracking mechanisms, we implement an example of an
algorithm on a moving average. To do this, let's go back to the file TrailingStop.mqh and describe the
derived class TrailingStopByMA.

---

## Page 1246

Part 6. Trading automation
1 246
6.4 Creating Expert Advisors
class TrailingStopByMA: public TrailingStop
{
   int handle;
   
public:
   TrailingStopByMA(const ulong t, const int period,
      const int offset = 1,
      const ENUM_MA_METHOD method = MODE_SMA,
      const ENUM_APPLIED_PRICE type = PRICE_CLOSE): TrailingStop(t, 0, 1)
   {
      handle = iMA(_Symbol, PERIOD_CURRENT, period, offset, method, type);
   }
   
   virtual double detectLevel() override
   {
      double array[1];
      ResetLastError();
      if(CopyBuffer(handle, 0, 0, 1, array) != 1)
      {
         Print("CopyBuffer error: ", _LastError);
         return 0;
      }
      return array[0];
   }
};
It creates the iMA indicator instance in the constructor: the period, the averaging method, and the
price type are passed via parameters.
In the overridden detectLevel method, we read the value from the indicator buffer, and by default, this
is done with an offset of 1  bar, i.e., the bar is closed, and its readings do not change when ticks arrive.
Those who wish can take the value from the zero bar, but such signals are unstable for all price types,
except for PRICE_OPEN.
To use a new class in the same test Expert Advisor TrailingStop.mq5, let's add another input parameter
MATrailingPeriod with a moving period (we will leave other parameters of the indicator unchanged).
input int MATrailingPeriod = 0;   // Period for Trailing by MA (0 = disabled)
The value of 0 in this parameter disables the trailing moving average. If it is enabled, the distance
settings in the TrailingDistance parameter are ignored.
Depending on this parameter, we will create either a standard trailing object TrailingStop or the one
derivative from iMA –TrailingStopByMA.
      ...
      tr = MATrailingPeriod > 0 ?
         new TrailingStopByMA(ticket, MATrailingPeriod) :
         new TrailingStop(ticket, distance, TrailingStep);
      ...
Let's see how the updated program behaves in the tester. In the Expert Advisor settings, set a non-zero
period for MA, for example, 1 0.

---

## Page 1247

Part 6. Trading automation
1 247
6.4 Creating Expert Advisors
Trailing stop on the moving average in the tester
Please note that in those moments when the average comes close to the price, there is an effect of
frequent stop-loss triggering and closing the position. When the average is above the quotes, a
protective level is not set at all, because this is not correct for buying. This is a consequence of the
fact that our Expert Advisor does not have any strategy and always opens positions of the same type,
regardless of the situation on the market. For sales, the same paradoxical situation will occasionally
arise when the average goes below the price, which means the market is growing, and the robot
"stubbornly" gets into a short position.  
In working strategies, as a rule, the direction of the position is chosen taking into account the
movement of the market, and the moving average is located on the right side of the current price,
where placing a stop loss is allowed.
6.4.1 7 Closing a position: full and partial 
Technically, closing a position can be thought of as a trading operation that is opposite to the one used
to open it. For example, to exit a buy, you need to make a sell operation (ORDER_TYPE_SELL in the
type field) and to exit the sell one you need to buy (ORDER_TYPE_BUY in the type field).
The trading operation type in the action field of the MqlTradeTransaction structure remains the same:
TRADE_ACTION_DEAL.
On a hedging account, the position to be closed must be specified using a ticket in the position field.
For netting accounts, you can specify only the name of the symbol in the symbol field since only one
symbol position is possible on them. However, you can also close positions by ticket here.
In order to unify the code, it makes sense to fill in both position and symbol fields regardless of account
type.

---

## Page 1248

Part 6. Trading automation
1 248
6.4 Creating Expert Advisors
Also, be sure to set the volume in the volume field. If it is equal to the position volume, it will be closed
completely. However, by specifying a lower value, it is possible to close only part of the position.
In the following table, all mandatory structure fields are marked with an asterisk and optional fields are
marked with a plus.
Field
Netting
Hedging
action
*
*
symbol
*
+
position
+
*
type
*
*
type_filling
*
*
volume
*
*
price
*'
*'
deviation
±
±
magic
+
+
comment
+
+
The price field marked is with an asterisk with a tick because it is required only for symbols with the
Request and Instant execution modes), while for the Exchange and Market execution, the price in the
structure is not taken into account.
For a similar reason, the deviation field is marked with '±'. It has effect only for Instant and Request
modes.
To simplify the programmatic implementation of closing a position, let's return to our extended
structure MqlTradeRequestSync in the file MqlTradeSync.mqh. The method for closing a position by
ticket has the following code.

---

## Page 1249

Part 6. Trading automation
1 249
6.4 Creating Expert Advisors
struct MqlTradeRequestSync: public MqlTradeRequest
{
   double partial; // volume after partial closing
   ...
   bool close(const ulong ticket, const double lot = 0)
   {
      if(!PositionSelectByTicket(ticket)) return false;
      
      position = ticket;
      symbol = PositionGetString(POSITION_SYMBOL);
      type = (ENUM_ORDER_TYPE)(PositionGetInteger(POSITION_TYPE) ^ 1);
      price = 0; 
      ...
Here we first check for the existence of a position by calling the PositionSelectByTicket function.
Additionally, this call makes the position selected in the trading environment of the terminal, which
allows you to read its properties using the subsequent functions. In particular, we find out the symbol of
a position from the POSITION_SYMBOL property and "reverse" its type from POSITION_TYPE to the
opposite one in order to get the required order type.
The position types in the ENUM_POSITION_TYPE enum are POSITION_TYPE_BUY (value 0) and
POSITION_TYPE_SELL (value 1 ). In the enumeration of order types ENUM_ORDER_TYPE, exactly the
same values are occupied by market operations: ORDER_TYPE_BUY and ORDER_TYPE_SELL. That is
why we can bring the first enumeration to the second one, and to get the opposite direction of trading,
it is enough to switch the zero bit using the exclusive OR operation ('^'): we get 1  from 0, and 0 from
1 .
Zeroing the price field means automatic selection of the correct current price (Ask or Bid) before
sending the request: this is done a little later, inside the helper method setVolumePrices, which is called
further along the algorithm, from the market method.
The _ market method call occurs a couple of lines below. The _ market method generates a market
order for the full volume or a part, taking into account all the completed fields of the structure.
      const double total = lot == 0 ? PositionGetDouble(POSITION_VOLUME) : lot;
      partial = PositionGetDouble(POSITION_VOLUME) - total;
      return _market(symbol, total);
   }
This fragment is slightly simplified compared to the current source code. The full code contains the
handling of a rare but possible situation when the position volume exceeds the maximum allowed volume
in one order per symbol (SYMBOL_VOLUME_MAX property). In this case, the position has to be closed
in parts, via several orders.
Also note that since the position can be closed partially, we had to add a field to the partial structure,
where the planned balance of the volume after the operation is placed. Of course, for a complete
closure, this will be 0. This information will be required to further verify the completion of the operation.
For netting accounts, there is a version of the close method that identifies the position by symbol
name. It selects a position by symbol, gets its ticket, and then refers to the previous version of close.

---

## Page 1250

Part 6. Trading automation
1 250
6.4 Creating Expert Advisors
   bool close(const string name, const double lot = 0)
   {
      if(!PositionSelect(name)) return false;
      return close(PositionGetInteger(POSITION_TICKET), lot);
   }
In the MqlTradeRequestSync structure, we have the completed method that provides a synchronous
wait for the completion of the operation, if necessary. Now we need to supplement it to close positions,
in the branch where action equals TRADE_ACTION_DEAL. We will distinguish between opening a position
and closing by a zero value in the position field: it has no ticket when opening a position and has one
when closing.
   bool completed()
   {
      if(action == TRADE_ACTION_DEAL)
      {
         if(position == 0)
         {
            const bool success = result.opened(timeout);
            if(success) position = result.position;
            return success;
         }
         else
         {
            result.position = position;
            result.partial = partial;
            return result.closed(timeout);
         }
      }
To check the actual closing of a position, we have added the closed method into the
MqlTradeResultSync structure. Before calling it, we write the position ticket in the result.position field
so that the result structure can track the moment when the corresponding ticket disappears from the
trading environment of the terminal, or when the volume equals result.partial in case of partial closure.
Here is the closed method. It is built on a well-known principle: first checking the success of the server
return code, and then waiting with the wait method for some condition to fulfill.

---

## Page 1251

Part 6. Trading automation
1 251 
6.4 Creating Expert Advisors
struct MqlTradeResultSync: public MqlTradeResult
{
   ...
   bool closed(const ulong msc = 1000)
   {
      if(retcode != TRADE_RETCODE_DONE)
      {
         return false;
      }
      if(!wait(positionRemoved, msc))
      {
         Print("Position removal timeout: P=" + (string)position);
      }
      
      return true;
   }
In this case, to check the condition for the position to disappear, we had to implement a new function
positionRemoved.
   static bool positionRemoved(MqlTradeResultSync &ref)
   {
      if(ref.partial)
      {
         return PositionSelectByTicket(ref.position)
            && TU::Equal(PositionGetDouble(POSITION_VOLUME), ref.partial);
      }
      return !PositionSelectByTicket(ref.position);
   }
We will test the operation of closing positions using the Expert Advisor TradeClose.mq5, which
implements a simple trading strategy: enter the market if there are two consecutive bars in the same
direction, and as soon as the next bar closes in the opposite direction to the previous trend, we exit the
market. Repetitive signals during continuous trends will be ignored, that is, there will be a maximum of
one position (minimum lot) or none in the market.
The Expert Advisor will not have any adjustable parameters: only the (Deviation) and a unique number
(Magic). The implicit parameters are the timeframe and the working symbol of the chart.
To track the presence of an already open position, we use the GetMyPosition function from the previous
example TradeTrailing.mq5: it searches among positions by symbol and Expert Advisor number and
returns a logical true if a suitable position is found.
We also take the almost unchanged function OpenPosition: it opens a position according to the market
order type passed in the single parameter. Here, this parameter will come from the trend detection
algorithm, and earlier (in TrailingStop.mq5) the order type was set by the user through an input
variable.
A new function that implements closing a position is ClosePosition. Because the header file
MqlTradeSync.mqh took over the whole routine, we only need to call the request.close(ticket) method
for the submitted position ticket and wait for the deletion to complete by request.completed().
In theory, the latter can be avoided if the Expert Advisor analyzes the situation at each tick. In this
case, a potential problem with deleting the position will promptly reveal itself on the next tick, and the

---

## Page 1252

Part 6. Trading automation
1 252
6.4 Creating Expert Advisors
Expert Advisor can try to delete it again. However, this Expert Advisor has trading logic based on bars,
and therefore it makes no sense to analyze every tick. Next, we implement a special mechanism for
bar-by-bar work, and in this regard, we synchronously control the removal, otherwise, the position
would remain "hanging" for a whole bar.
ulong LastErrorCode = 0;
   
ulong ClosePosition(const ulong ticket)
{
   MqlTradeRequestSync request; // empty structure
   
   // optional fields are filled directly in the structure
   request.magic = Magic;
   request.deviation = Deviation;
   
   ResetLastError();
   // perform close and wait for confirmation
   if(request.close(ticket) && request.completed())
   {
      Print("OK Close Order/Deal/Position");
   }
   else // print diagnostics in case of problems
   {
      Print(TU::StringOf(request));
      Print(TU::StringOf(request.result));
      LastErrorCode = request.result.retcode;
      return 0; // error, code to parse in LastErrorCode
   }
   
   return request.position; // non-zero value - success
}
We could force the ClosePosition functions to return 0 in case of successful deletion of the position and
an error code otherwise. This seemingly efficient approach would make the behavior of the two
functions OpenPosition and ClosePosition different: in the calling code, it would be necessary to nest the
calls of these functions in logical expressions that are opposite in meaning, and this would introduce
confusion. In addition, we would require the global variable LastErrorCode in any case, in order to add
information about the error inside the OpenPosition function. Also, the if(condition) check is more
organically interpreted as success than if(!condition).
The function that generates trading signals according to the above strategy is called GetTradeDirection.

---

## Page 1253

Part 6. Trading automation
1 253
6.4 Creating Expert Advisors
ENUM_ORDER_TYPE GetTradeDirection()
{
   if(iClose(_Symbol, _Period, 1) > iClose(_Symbol, _Period, 2)
      && iClose(_Symbol, _Period, 2) > iClose(_Symbol, _Period, 3))
   {
      return ORDER_TYPE_BUY; // open a long position
   }
   
   if(iClose(_Symbol, _Period, 1) < iClose(_Symbol, _Period, 2)
      && iClose(_Symbol, _Period, 2) < iClose(_Symbol, _Period, 3))
   {
      return ORDER_TYPE_SELL; // open a short position
   }
   
   return (ENUM_ORDER_TYPE)-1; // close
}
The function returns a value of the ENUM_ORDER_TYPE type with two standard elements
(ORDER_TYPE_BUY and ORDER_TYPE_SELL) triggering buys and sells, respectively. The special value -
1  (not in the enumeration) will be used as a close signal.
To activate the Expert Advisor based on the trading algorithm, we use the OnTick handler. As we
remember, other options are suitable for other strategies, for example, a timer for trading on the news
or Depth of Market events for volume trading.
First, let's analyze the function in a simplified form, without handling potential errors. At the very
beginning, there is a block that ensures that the further algorithm is triggered only when a new bar is
opened.
void OnTick()
{
   static datetime lastBar = 0;
   if(iTime(_Symbol, _Period, 0) == lastBar) return;
   lastBar = iTime(_Symbol, _Period, 0);
   ...
Next, we get the current signal from the GetTradeDirection function.
   const ENUM_ORDER_TYPE type = GetTradeDirection();
If there is a position, we check whether a signal to close it has been received and call ClosePosition if
necessary. If there is no position yet and there is a signal to enter the market, we call OpenPosition.

---

## Page 1254

Part 6. Trading automation
1 254
6.4 Creating Expert Advisors
   if(GetMyPosition(_Symbol, Magic))
   {
      if(type != ORDER_TYPE_BUY && type != ORDER_TYPE_SELL)
      {
         ClosePosition(PositionGetInteger(POSITION_TICKET));
      }
   }
   else if(type == ORDER_TYPE_BUY || type == ORDER_TYPE_SELL)
   {
      OpenPosition(type);
   }
}
To analyze errors, you will need to enclose OpenPosition and ClosePosition calls into conditional
statements and take some action to restore the working state of the program. In the simplest case, it
is enough to repeat the request at the next tick, but it is desirable to do this a limited number of times.
Therefore, we will create static variables with a counter and an error limit.
void OnTick()
{
   static int errors = 0;
   static const int maxtrials = 10; // no more than 10 attempts per bar
   
   // expect a new bar to appear if there were no errors
   static datetime lastBar = 0;
   if(iTime(_Symbol, _Period, 0) == lastBar && errors == 0) return;
   lastBar = iTime(_Symbol, _Period, 0);
   ...
The bar-by-bar mechanism is temporarily disabled if errors appear since it is desirable to overcome
them as soon as possible.
Errors are counted in conditional statements around ClosePosition and OpenPosition.

---

## Page 1255

Part 6. Trading automation
1 255
6.4 Creating Expert Advisors
   const ENUM_ORDER_TYPE type = GetTradeDirection();
   
   if(GetMyPosition(_Symbol, Magic))
   {
      if(type != ORDER_TYPE_BUY && type != ORDER_TYPE_SELL)
      {
         if(!ClosePosition(PositionGetInteger(POSITION_TICKET)))
         {
            ++errors;
         }
         else
         {
            errors = 0;
         }
      }
   }
   else if(type == ORDER_TYPE_BUY || type == ORDER_TYPE_SELL)
   {
      if(!OpenPosition(type))
      {
         ++errors;
      }
      else
      {
         errors = 0;
      }
   }
 // too many errors per bar
   if(errors >= maxtrials) errors = 0;
 // error serious enough to pause
   if(IS_TANGIBLE(LastErrorCode)) errors = 0;
}
Setting the errors variable to 0 turns on the bar-by-bar mechanism again and stops attempts to repeat
the request until the next bar.
The macro IS_TANGIBLE is defined in TradeRetcode.mqh as:
#define IS_TANGIBLE(T) ((T) >= TRADE_RETCODE_ERROR)
Errors with smaller codes are operational, that is, normal in a sense. Large codes require analysis and
different actions, depending on the cause of the problem: incorrect request parameters, permanent or
temporary bans in the trading environment, lack of funds, and so on. We will present an improved error
classifier in the section Pending order modification.
Let's run the Expert Advisor in the tester on XAUUSD, H1  from the beginning of 2022, simulating real
ticks. The next collage shows a fragment of a chart with deals, as well as the balance curve.

---

## Page 1256

Part 6. Trading automation
1 256
6.4 Creating Expert Advisors
TradeClose testing results on XAUUSD, H1
Based on the report and the log, we can see that the combination of our simple trading logic and the
two operations of opening and closing positions is working properly.
In addition to simply closing a position, the platform supports the possibility of mutual closing of two
opposite positions on hedging accounts.
6.4.1 8 Closing opposite positions: full and partial (hedging)
On hedging accounts, it is allowed to open several positions at the same time, and in most cases, these
positions can be in the opposite direction. In some jurisdictions, hedging accounts are restricted: you
can only have positions in one direction at a time. In this case, you will receive the
TRADE_RETCODE_HEDGE_PROHIBITED error code when trying to execute an opposite trading
operation. Also, this restriction often correlates with the setting of the ACCOUNT_FIFO_CLOSE account
property to true.
When two opposite positions are opened at the same time, the platform supports the mechanism of
their simultaneous mutual closing using the TRADE_ACTION_CLOSE_BY operation. To perform this
action, you should fill two more fields in the MqlTradeTransaction structure in addition to the action
field: position and position_ by must contain the tickets of positions to be closed.
The availability of this feature depends on the SYMBOL_ORDER_MODE property of the financial
instrument: SYMBOL_ORDER_CLOSEBY (64) must be present in the allowed flags bitmask.
This operation not only simplifies closing (one operation instead of two) but also saves one spread.
As you know, any new position starts trading with a loss equal to the spread. For example, when buying
a financial instrument, a transaction is concluded at the Ask price, but for an exit deal, that is, a sale,

---

## Page 1257

Part 6. Trading automation
1 257
6.4 Creating Expert Advisors
the actual price is Bid. For a short position, the situation is reversed: immediately after entering at the
Bid price, we start tracking the price Ask for a potential exit.
If you close positions at the same time in a regular way, their exit prices will be at a distance of the
current spread from each other. However, if you use the TRADE_ACTION_CLOSE_BY operation, then
both positions will be closed without taking into account the current prices. The price at which positions
are offset is equal to the opening price of the position_ by position (in the request structure). It is
specified in the ORDER_TYPE_CLOSE_BY order generated by the TRADE_ACTION_CLOSE_BY request.
Unfortunately, in the reports in the context of deals and positions, the closing and opening prices of
opposite positions/deals are displayed in pairs of identical values, in a mirror direction, which gives the
impression of a double profit or loss. In fact, the financial result of the operation (the difference
between prices adjusted for the lot) is recorded only for the first position exit trade (the position field in
the request structure). The result of the second exit trade is always 0, regardless of the price
difference.
Another consequence of this asymmetry is that from changing the places of tickets in the position and
position_ by fields, the profit and loss statistics in the context of long and short trades changes in the
trading report, for example, profitable long trades can increase exactly as much as the number of
profitable short trades decreases. But this, in theory, should not affect the overall result, if we assume
that the delay in the execution of the order does not depend on the order of transfer of tickets.
The following diagram shows a graphical explanation of the process (spreads are intentionally
exaggerated).
Spread accounting when closing profitable positions
Here is a case of a profitable pair of positions. If the positions had opposite directions and were at a
loss, then when they were closed separately, the spread would be taken into account twice (in each).
Counter closing allows you to reduce the loss by one spread.

---

## Page 1258

Part 6. Trading automation
1 258
6.4 Creating Expert Advisors
Accounting for the spread when closing unprofitable positions
Reversed positions do not have to be of equal size. The opposite closing operation will work on the
minimum of the two volumes.
In the MqlTradeSync.mqh file, the close-by operation is implemented using the closeby method with two
parameters for position tickets.

---

## Page 1259

Part 6. Trading automation
1 259
6.4 Creating Expert Advisors
struct MqlTradeRequestSync: public MqlTradeRequest
{
   ...
   bool closeby(const ulong ticket1, const ulong ticket2)
   {
      if(!PositionSelectByTicket(ticket1)) return false;
      double volume1 = PositionGetDouble(POSITION_VOLUME);
      if(!PositionSelectByTicket(ticket2)) return false;
      double volume2 = PositionGetDouble(POSITION_VOLUME);
   
      action = TRADE_ACTION_CLOSE_BY;
      position = ticket1;
      position_by = ticket2;
      
      ZeroMemory(result);
      if(volume1 != volume2)
      {
         // remember which position should disappear
         if(volume1 < volume2)
            result.position = ticket1;
         else
            result.position = ticket2;
      }
      return OrderSend(this, result);
   }
To control the result of the closure, we store the ticket of a smaller position in the result.position
variable. Everything in the completed method and in the MqlTradeResultSync structure is ready for
synchronous position closing tracking: the same algorithm worked for a normal closing of a position.
struct MqlTradeRequestSync: public MqlTradeRequest
{
   ...
   bool completed()
   {
      ...
      else if(action == TRADE_ACTION_CLOSE_BY)
      {
         return result.closed(timeout);
      }
      return false;
   }
Opposite positions are usually used as a replacement for a stop order or an attempt to take profit on a
short-term correction while remaining in the market and following the main trend. The option of using a
pseudo-stop order allows you to postpone the decision to actually close positions for some time,
continuing the analysis of market movements expecting the price to reverse in the right direction.
However, it should be kept in mind that "locked" positions require increased deposits and are subject to
swaps. That is why it is difficult to imagine a trading strategy built on opposite positions in its pure
form, which could serve as an example for this section.
Let's develop the idea of the price-action bar-based strategy outlined in the previous example. The new
Expert Advisor is TradeCloseBy.mq5.

---

## Page 1260

Part 6. Trading automation
1 260
6.4 Creating Expert Advisors
We will use the previous signal to enter the market upon detection of two consecutive candles that
closed in the same direction. A function responsible for its formation is again GetTradeDirection.
However, let's allow re-entries if the trend continues. The total maximum allowed number of positions
will be set in the input variable PositionLimit, the default is 5.
The GetMyPositions function will undergo some changes: it will have two parameters, which will be
references to arrays that accept position tickets: buy and sell separately.
#define PUSH(A,V) (A[ArrayResize(A, ArraySize(A) + 1, ArraySize(A) * 2) - 1] = V)
   
int GetMyPositions(const string s, const ulong m,
   ulong &ticketsLong[], ulong &ticketsShort[])
{
   for(int i = 0; i < PositionsTotal(); ++i)
   {
      if(PositionGetSymbol(i) == s && PositionGetInteger(POSITION_MAGIC) == m)
      {
         if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
            PUSH(ticketsLong, PositionGetInteger(POSITION_TICKET));
         else
            PUSH(ticketsShort, PositionGetInteger(POSITION_TICKET));
      }
   }
   
   const int min = fmin(ArraySize(ticketsLong), ArraySize(ticketsShort));
   if(min == 0) return -fmax(ArraySize(ticketsLong), ArraySize(ticketsShort));
   return min;
}
The function returns the size of the smallest array of the two. When it is greater than zero, we have the
opportunity to close opposite positions.
If the minimum array is zero size, the function will return the size of another array, but with a minus
sign, just to let the calling code know that all positions are in the same direction.
If there are no positions in either direction, the function will return 0.
Opening positions will remain under the control of the function OpenPosition - no changes here.
Closing will be carried out only in the mode of two opposite positions in the new function
CloseByPosition. In other words, this Expert Advisor is not capable of closing positions one at a time, in
the usual way. Of course, in a real robot, such a principle is unlikely to occur, but as an example of an
oncoming closure, it fits very well. If we need to close a single position, it is enough to open an opposite
position for it (at this moment the floating profit or loss is fixed) and call CloseByPosition for two.

---

## Page 1261

Part 6. Trading automation
1 261 
6.4 Creating Expert Advisors
bool CloseByPosition(const ulong ticket1, const ulong ticket2)
{
   MqlTradeRequestSync request;
   request.magic = Magic;
   
   ResetLastError();
   // send a request and wait for it to complete
   if(request.closeby(ticket1, ticket2))
   {
      Print("Positions collapse initiated");
      if(request.completed())
      {
         Print("OK CloseBy Order/Deal/Position");
         return true; // success
      }
   }
   
   Print(TU::StringOf(request));
   Print(TU::StringOf(request.result));
   
   return false; // error
}
The code uses the request.closeby method described above. The position, and position_ by fields are
filled and OrderSend is called.
The trading logic is described in the OnTick handler which analyzes the price configuration only at the
moment of the formation of a new bar and receives a signal from the GetTradeDirection function.
void OnTick()
{
   static bool error = false;
   // waiting for the formation of a new bar, if there is no error
   static datetime lastBar = 0;
   if(iTime(_Symbol, _Period, 0) == lastBar && !error) return;
   lastBar = iTime(_Symbol, _Period, 0);
   
   const ENUM_ORDER_TYPE type = GetTradeDirection();
   ...
Next, we fill the ticketsLong and ticketsShort arrays with position tickets of the working symbol and
with the given Magic number. If the GetMyPositions function returns a value greater than zero, it gives
the number of formed pairs of opposite positions. They can be closed in a loop using the
CloseByPosition function. The combination of pairs in this case is chosen randomly (in the order of
positions in the terminal environment), however, in practice, it may be important to select pairs by
volume or in such a way that the most profitable ones are closed first.

---

## Page 1262

Part 6. Trading automation
1 262
6.4 Creating Expert Advisors
   ulong ticketsLong[], ticketsShort[];
   const int n = GetMyPositions(_Symbol, Magic, ticketsLong, ticketsShort);
   if(n > 0)
   {
      for(int i = 0; i < n; ++i)
      {
         error = !CloseByPosition(ticketsShort[i], ticketsLong[i]) && error;
      }
   }
   ...
For any other value of n, you should check if there is a signal (possibly repeated) to enter the market
and execute it by calling OpenPosition.
   else if(type == ORDER_TYPE_BUY || type == ORDER_TYPE_SELL)
   {
      error = !OpenPosition(type);
   }
   ...
Finally, if there are still open positions, but they are in the same direction, we check if their number has
reached the limit, in which case we form an opposite position in order to "collapse" two of them on the
next bar (thus closing one of any position from the old ones).
   else if(n < 0)
   {
      if(-n >= (int)PositionLimit)
      {
         if(ArraySize(ticketsLong) > 0)
         {
            error = !OpenPosition(ORDER_TYPE_SELL);
         }
         else // (ArraySize(ticketsShort) > 0)
         {
            error = !OpenPosition(ORDER_TYPE_BUY);
         }
      }
   }
}
Let's run the Expert Advisor in the tester on XAUUSD, H1  from the beginning of 2022, with default
settings. Below is the chart with positions in the process of the program, as well as the balance curve.

---

## Page 1263

Part 6. Trading automation
1 263
6.4 Creating Expert Advisors
TradeCloseBy test results on XAUUSD, H1
It is easy to find in the log the moments of time when one trend ends (buying with tickets from #2 to
#4), and transactions start being generated in the opposite direction (selling #5), after which a counter
close is triggered.

---

## Page 1264

Part 6. Trading automation
1 264
6.4 Creating Expert Advisors
2022.01.03 01:05:00   instant buy 0.01 XAUUSD at 1831.13 (1830.63 / 1831.13 / 1830.63)
2022.01.03 01:05:00   deal #2 buy 0.01 XAUUSD at 1831.13 done (based on order #2)
2022.01.03 01:05:00   deal performed [#2 buy 0.01 XAUUSD at 1831.13]
2022.01.03 01:05:00   order performed buy 0.01 at 1831.13 [#2 buy 0.01 XAUUSD at 1831.13]
2022.01.03 01:05:00   Waiting for position for deal D=2
2022.01.03 01:05:00   OK New Order/Deal/Position
2022.01.03 02:00:00   instant buy 0.01 XAUUSD at 1828.77 (1828.47 / 1828.77 / 1828.47)
2022.01.03 02:00:00   deal #3 buy 0.01 XAUUSD at 1828.77 done (based on order #3)
2022.01.03 02:00:00   deal performed [#3 buy 0.01 XAUUSD at 1828.77]
2022.01.03 02:00:00   order performed buy 0.01 at 1828.77 [#3 buy 0.01 XAUUSD at 1828.77]
2022.01.03 02:00:00   Waiting for position for deal D=3
2022.01.03 02:00:00   OK New Order/Deal/Position
2022.01.03 03:00:00   instant buy 0.01 XAUUSD at 1830.40 (1830.16 / 1830.40 / 1830.16)
2022.01.03 03:00:00   deal #4 buy 0.01 XAUUSD at 1830.40 done (based on order #4)
2022.01.03 03:00:00   deal performed [#4 buy 0.01 XAUUSD at 1830.40]
2022.01.03 03:00:00   order performed buy 0.01 at 1830.40 [#4 buy 0.01 XAUUSD at 1830.40]
2022.01.03 03:00:00   Waiting for position for deal D=4
2022.01.03 03:00:00   OK New Order/Deal/Position
2022.01.03 05:00:00   instant sell 0.01 XAUUSD at 1826.22 (1826.22 / 1826.45 / 1826.22)
2022.01.03 05:00:00   deal #5 sell 0.01 XAUUSD at 1826.22 done (based on order #5)
2022.01.03 05:00:00   deal performed [#5 sell 0.01 XAUUSD at 1826.22]
2022.01.03 05:00:00   order performed sell 0.01 at 1826.22 [#5 sell 0.01 XAUUSD at 1826.22]
2022.01.03 05:00:00   Waiting for position for deal D=5
2022.01.03 05:00:00   OK New Order/Deal/Position
2022.01.03 06:00:00   close position #5 sell 0.01 XAUUSD by position #2 buy 0.01 XAUUSD (1825.64 / 1825.86 / 1825.64)
2022.01.03 06:00:00   deal #6 buy 0.01 XAUUSD at 1831.13 done (based on order #6)
2022.01.03 06:00:00   deal #7 sell 0.01 XAUUSD at 1826.22 done (based on order #6)
2022.01.03 06:00:00   Positions collapse initiated
2022.01.03 06:00:00   OK CloseBy Order/Deal/Position
Transaction #3 is an interesting artifact. An attentive reader will notice that it opened lower than the
previous one, seemingly violating our strategy. In fact, there is no error here, and this is a consequence
of the fact that the conditions of the signals are written as simply as possible: only based on the closing
prices of the bars. Therefore, a bearish reversal candle (D), which opened with a gap up and closed
above the end of the previous bullish candle (C), generated a buy signal. This situation is illustrated in
the following screenshot.
Transactions on an uptrend at closing prices
All candles in sequence A, B, C, D, and E close higher than the previous one and encourage continued
buying. To exclude such artifacts, one should additionally analyze the direction of the bars themselves.

---

## Page 1265

Part 6. Trading automation
1 265
6.4 Creating Expert Advisors
The last thing to pay attention to in this example is the OnInit function. Since the Expert Advisor uses
the TRADE_ACTION_CLOSE_BY operation, checks are made here for the relevant account and working
symbol settings.
int OnInit()
{
   ...
   if(AccountInfoInteger(ACCOUNT_MARGIN_MODE) != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
   {
      Alert("An account with hedging is required for this EA!");
      return INIT_FAILED;
   }
   
   if((SymbolInfoInteger(_Symbol, SYMBOL_ORDER_MODE) & SYMBOL_ORDER_CLOSEBY) == 0)
   {
      Alert("'Close By' mode is not supported for ", _Symbol);
      return INIT_FAILED;
   }
   
   return INIT_SUCCEEDED;
}
If one of the properties does not support cross-closing, the Expert Advisor will not be able to continue
working. When creating working robots, these checks, as a rule, are carried out inside the trading
algorithm and switch the program to alternative modes, in particular, to a single closing of positions and
maintaining an aggregate position in case of netting.
6.4.1 9 Placing a pending order
In Types of orders, we theoretically considered all options for placing pending orders supported by the
platform. From a practical point of view, orders are created using OrderSend/OrderSendAsync
functions, for which the request structure MqlTradeRequest is prefilled according to special rules.
Specifically, the action field must contain the TRADE_ACTION_PENDING value from the
ENUM_TRADE_REQUEST_ACTIONS enumeration. With this in mind, the following fields are mandatory:
·action
·symbol
·volume
·price
·type (default value 0 corresponds to ORDER_TYPE_BUY)
·type_filling (default 0 corresponds to ORDER_FILLING_FOK)
·type_time (default value 0 corresponds to ORDER_TIME_GTC)
·expiration (default 0, not used for ORDER_TIME_GTC)
If zero defaults are suitable for the task, some of the last four fields can be skipped.
The stoplimit field is mandatory only for orders of types ORDER_TYPE_BUY_STOP_LIMIT and
ORDER_TYPE_SELL_STOP_LIMIT.
The following fields are optional:

---

## Page 1266

Part 6. Trading automation
1 266
6.4 Creating Expert Advisors
·sl
·tp
·magic
·comment
Zero values in sl and tp indicate the absence of protective levels.
Let's add the methods for checking values and filling fields into our structures in the MqlTradeSync.mqh
file. The principle of formation of all types of orders is the same, so let's consider a couple of special
cases of placing limit buy and sell orders. The remaining types will differ only in the value of the field
type. Public methods with a full set of required fields, as well as protective levels, are named according
to types: buyLimit and sellLimit.
   ulong buyLimit(const string name, const double lot, const double p,
      const double stop = 0, const double take = 0,
      ENUM_ORDER_TYPE_TIME duration = ORDER_TIME_GTC, datetime until = 0)
   {
      type = ORDER_TYPE_BUY_LIMIT;
      return _pending(name, lot, p, stop, take, duration, until);
   }
   
   ulong sellLimit(const string name, const double lot, const double p,
      const double stop = 0, const double take = 0,
      ENUM_ORDER_TYPE_TIME duration = ORDER_TIME_GTC, datetime until = 0)
   {
      type = ORDER_TYPE_SELL_LIMIT;
      return _pending(name, lot, p, stop, take, duration, until);
   }
Since the structure contains the symbol field which is optionally initialized in the constructor, there are
similar methods without the name parameter: they call the above methods by passing symbol as the
first parameter. Thus, to create an order with minimal effort, write the following:
MqlTradeRequestSync request; // by default uses the current chart symbol
request.buyLimit(volume, price);
The general part of the code for checking the passed values, normalizing them, saving them in
structure fields, and creating a pending order has been moved to the helper method _ pending. It
returns the order ticket on success or 0 on failure.

---

## Page 1267

Part 6. Trading automation
1 267
6.4 Creating Expert Advisors
   ulong _pending(const string name, const double lot, const double p,
      const double stop = 0, const double take = 0,
      ENUM_ORDER_TYPE_TIME duration = ORDER_TIME_GTC, datetime until = 0,
      const double origin = 0)
   {
      action = TRADE_ACTION_PENDING;
      if(!setSymbol(name)) return 0;
      if(!setVolumePrices(lot, p, stop, take, origin)) return 0;
      if(!setExpiration(duration, until)) return 0;
      if((SymbolInfoInteger(name, SYMBOL_ORDER_MODE) & (1 << (type / 2))) == 0)
      {
         Print(StringFormat("pending orders %s not allowed for %s",
            EnumToString(type), name));
         return 0;
      }
      ZeroMemory(result);
      if(OrderSend(this, result)) return result.order;
      return 0;
   }
We already know how to fill the action field and how to call the setSymbol and setVolumePrices methods
from previous trading operations.
The multi-string if operator ensures that the operation being prepared is present among the allowed
symbol operations specified in the SYMBOL_ORDER_MODE property. Integer type division type which
divides in half and shifts the resulting value by 1 , sets the correct bit in the mask of allowed order
types. This is due to the combination of constants in the ENUM_ORDER_TYPE enumeration and the
SYMBOL_ORDER_MODE property. For example, ORDER_TYPE_BUY_STOP and
ORDER_TYPE_SELL_STOP have the values 4 and 5, which when divided by 2 both give 2 (with decimals
removed). Operation 1  << 2 has a result 4 equal to SYMBOL_ORDER_STOP.
A special feature of pending orders is the processing of the expiration date. The setExpiration method
deals with it. In this method, it should be ensured that the specified expiration mode
ENUM_ORDER_TYPE_TIME of duration is allowed for the symbol and the date and time in until are filled
in correctly.

---

## Page 1268

Part 6. Trading automation
1 268
6.4 Creating Expert Advisors
   bool setExpiration(ENUM_ORDER_TYPE_TIME duration = ORDER_TIME_GTC, datetime until = 0)
   {
      const int modes = (int)SymbolInfoInteger(symbol, SYMBOL_EXPIRATION_MODE);
      if(((1 << duration) & modes) != 0)
      {
         type_time = duration;
         if((duration == ORDER_TIME_SPECIFIED || duration == ORDER_TIME_SPECIFIED_DAY)
            && until == 0)
         {
            Print(StringFormat("datetime is 0, "
               "but it's required for order expiration mode %s",
               EnumToString(duration)));
            return false;
         }
         if(until > 0 && until <= TimeTradeServer())
         {
            Print(StringFormat("expiration datetime %s is in past, server time is %s",
               TimeToString(until), TimeToString(TimeTradeServer())));
            return false;
         }
         expiration = until;
      }
      else
      {
         Print(StringFormat("order expiration mode %s is not allowed for %s",
            EnumToString(duration), symbol));
         return false;
      }
      return true;
   }
The bitmask of allowed modes is available in the SYMBOL_EXPIRATION_MODE property. The
combination of bits in the mask and the constants ENUM_ORDER_TYPE_TIME is such that we just need
to evaluate the expression 1  << duration and superimpose it on the mask: a non-zero value indicates
the presence of the mode.
For the ORDER_TIME_SPECIFIED and ORDER_TIME_SPECIFIED_DAY modes, the expiration field with
the specific datetime value cannot be empty. Also, the specified date and time cannot be in the past.
Since the _ pending method presented earlier sends a request to the server using OrderSend in the end,
our program must make sure that the order with the received ticket was actually created (this is
especially important for limit orders that can be output to an external trading system). Therefore, in
the completed method, which is used for "blocking" control of the result, we will add a branch for the
TRADE_ACTION_PENDING operation.

---

## Page 1269

Part 6. Trading automation
1 269
6.4 Creating Expert Advisors
   bool completed()
   {
      // old processing code
      // TRADE_ACTION_DEAL
      // TRADE_ACTION_SLTP
      // TRADE_ACTION_CLOSE_BY
      ...
      else if(action == TRADE_ACTION_PENDING)
      {
         return result.placed(timeout);
      }
      ...
      return false;
   }
In the MqlTradeResultSync structure, we add the placed method.
   bool placed(const ulong msc = 1000)
   {
      if(retcode != TRADE_RETCODE_DONE
         && retcode != TRADE_RETCODE_DONE_PARTIAL)
      {
         return false;
      }
      
      if(!wait(orderExist, msc))
      {
         Print("Waiting for order: #" + (string)order);
         return false;
      }
      return true;
   }
Its main task is to wait for the order to appear using the wait in the orderExist function: it has already
been used in the first stage of verification of position opening.
To test the new functionality, let's implement the Expert Advisor PendingOrderSend.mq5. It enables the
selection of the pending order type and all its attributes using input variables, after which a
confirmation request is executed.

---

## Page 1270

Part 6. Trading automation
1 270
6.4 Creating Expert Advisors
enum ENUM_ORDER_TYPE_PENDING
{                                                        // UI interface strings
   PENDING_BUY_STOP = ORDER_TYPE_BUY_STOP,               // ORDER_TYPE_BUY_STOP
   PENDING_SELL_STOP = ORDER_TYPE_SELL_STOP,             // ORDER_TYPE_SELL_STOP
   PENDING_BUY_LIMIT = ORDER_TYPE_BUY_LIMIT,             // ORDER_TYPE_BUY_LIMIT
   PENDING_SELL_LIMIT = ORDER_TYPE_SELL_LIMIT,           // ORDER_TYPE_SELL_LIMIT
   PENDING_BUY_STOP_LIMIT = ORDER_TYPE_BUY_STOP_LIMIT,   // ORDER_TYPE_BUY_STOP_LIMIT
   PENDING_SELL_STOP_LIMIT = ORDER_TYPE_SELL_STOP_LIMIT, // ORDER_TYPE_SELL_STOP_LIMIT
};
input string Symbol;             // Symbol (empty = current _Symbol)
input double Volume;             // Volume (0 = minimal lot)
input ENUM_ORDER_TYPE_PENDING Type = PENDING_BUY_STOP;
input int Distance2SLTP = 0;     // Distance to SL/TP in points (0 = no)
input ENUM_ORDER_TYPE_TIME Expiration = ORDER_TIME_GTC;
input datetime Until = 0;
input ulong Magic = 1234567890;
input string Comment;
The Expert Advisor will create a new order every time it is launched or parameters are changed.
Automatic order removal is not yet provided. We will discuss this operation type later. In this regard, do
not forget to delete orders manually.
A one-time order placement is performed, as in some previous examples, based on a timer (therefore,
you should first make sure that the market is open).
void OnTimer()
{
   // execute once and wait for the user to change the settings
   EventKillTimer();
   
   const string symbol = StringLen(Symbol) == 0 ? _Symbol : Symbol;
   if(PlaceOrder((ENUM_ORDER_TYPE)Type, symbol, Volume,
      Distance2SLTP, Expiration, Until, Magic, Comment))
   {
      Alert("Pending order placed - remove it manually, please");
   }
}
The PlaceOrder function accepts all settings as parameters, sends a request, and returns a success
indicator (non-zero ticket). Orders of all supported types are provided with pre-filled distances from the
current price which are calculated as part of the daily range of quotes.

---

## Page 1271

Part 6. Trading automation
1 271 
6.4 Creating Expert Advisors
ulong PlaceOrder(const ENUM_ORDER_TYPE type,
   const string symbol, const double lot,
   const int sltp, ENUM_ORDER_TYPE_TIME expiration, datetime until,
   const ulong magic = 0, const string comment = NULL)
{
   static double coefficients[] = // indexed by order type
   {
      0  ,   // ORDER_TYPE_BUY - not used
      0  ,   // ORDER_TYPE_SELL - not used
     -0.5,   // ORDER_TYPE_BUY_LIMIT - slightly below the price
     +0.5,   // ORDER_TYPE_SELL_LIMIT - slightly above the price
     +1.0,   // ORDER_TYPE_BUY_STOP - far above the price
     -1.0,   // ORDER_TYPE_SELL_STOP - far below the price
     +0.7,   // ORDER_TYPE_BUY_STOP_LIMIT - average above the price 
     -0.7,   // ORDER_TYPE_SELL_STOP_LIMIT - average below the price
      0  ,   // ORDER_TYPE_CLOSE_BY - not used
   };
   ...
For example, the coefficient of -0.5 for ORDER_TYPE_BUY_LIMIT means that the order will be placed
below the current price by half of the daily range (rebound inside the range), and the coefficient of
+1 .0 for ORDER_TYPE_BUY_STOP means that the order will be at the upper border of the range
(breakout).
The daily range itself is calculated as follows.
   const double range = iHigh(symbol, PERIOD_D1, 1) - iLow(symbol, PERIOD_D1, 1);
   Print("Autodetected daily range: ", (float)range);
   ...
We find the volume and point values that will be required below.
   const double volume = lot == 0 ? SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN) : lot;
   const double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
The price level for placing an order is calculated in the price variable based on the given coefficients
from the total range.
   const double price = TU::GetCurrentPrice(type, symbol) + range * coefficients[type];
The stoplimit field must be filled only for *_STOP_LIMIT orders. The values for it are stored in the origin
variable.
   const bool stopLimit =
      type == ORDER_TYPE_BUY_STOP_LIMIT ||
      type == ORDER_TYPE_SELL_STOP_LIMIT;
   const double origin = stopLimit ? TU::GetCurrentPrice(type, symbol) : 0;
When these two types of orders are triggered, a new pending order will be placed at the current price.
Indeed, in this scenario, the price moves from the current value to the price level, where the order is
activated, and therefore the "former current" price becomes the correct rebound level indicated by a
limit order. We will illustrate this situation below.
Protective levels are determined using the TU::TradeDirection object. For stop-limit orders, we
calculated starting from origin.

---

## Page 1272

Part 6. Trading automation
1 272
6.4 Creating Expert Advisors
   TU::TradeDirection dir(type);
   const double stop = sltp == 0 ? 0 :
      dir.negative(stopLimit ? origin : price, sltp * point);
   const double take = sltp == 0 ? 0 :
      dir.positive(stopLimit ? origin : price, sltp * point);
Next, the structure is described and the optional fields are filled in.
   MqlTradeRequestSync request(symbol);
   
   request.magic = magic;
   request.comment = comment;
   // request.type_filling = SYMBOL_FILLING_FOK;
Here you can select the fill mode. By default, MqlTradeRequestSync automatically selects the first of
the allowed modes, ENUM_ORDER_TYPE_FILLING.
Depending on the order type chosen by the user, we call one or another trading method.
   ResetLastError();
   // fill in and check the required fields, send the request
   ulong order = 0;
   switch(type)
   {
   case ORDER_TYPE_BUY_STOP:
      order = request.buyStop(volume, price, stop, take, expiration, until);
      break;
   case ORDER_TYPE_SELL_STOP:
      order = request.sellStop(volume, price, stop, take, expiration, until);
      break;
   case ORDER_TYPE_BUY_LIMIT:
      order = request.buyLimit(volume, price, stop, take, expiration, until);
      break;
   case ORDER_TYPE_SELL_LIMIT:
      order = request.sellLimit(volume, price, stop, take, expiration, until);
      break;
   case ORDER_TYPE_BUY_STOP_LIMIT:
      order = request.buyStopLimit(volume, price, origin, stop, take, expiration, until);
      break;
   case ORDER_TYPE_SELL_STOP_LIMIT:
      order = request.sellStopLimit(volume, price, origin, stop, take, expiration, until);
      break;
   }
   ...
If the ticket is received, we wait for it to appear in the trading environment of the terminal.

---

## Page 1273

Part 6. Trading automation
1 273
6.4 Creating Expert Advisors
   if(order != 0)
   {
      Print("OK order sent: #=", order);
      if(request.completed()) // expect result (order confirmation)
      {
         Print("OK order placed");
      }
   }
   Print(TU::StringOf(request));
   Print(TU::StringOf(request.result));
   return order;
}
Let's run the Expert Advisor on the EURUSD chart with default settings and additionally select the
distance to the protective levels of 1 000 points. We will see the following entries in the log (assuming
that the default settings match the permissions for EURUSD in your account).
Autodetected daily range: 0.01413
OK order sent: #=1282106395
OK order placed
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1.11248, SL=1.10248, TP=1.12248, ORDER_TIME_GTC, M=1234567890
DONE, #=1282106395, V=0.01, Request executed, Req=91
Alert: Pending order placed - remove it manually, please
Here is what it looks like on the chart:
Pending order ORDER_TYPE_BUY_STOP
Let's delete the order manually and change the order type to ORDER_TYPE_BUY_STOP_LIMIT. The
result is a more complex picture.

---

## Page 1274

Part 6. Trading automation
1 274
6.4 Creating Expert Advisors
Pending order ORDER_TYPE_BUY_STOP_LIMIT
The price where the upper pair of dash-dotted lines is located is the order trigger price, as a result of
which an ORDER_TYPE_BUY_LIMIT order will be placed at the current price level, with Stop Loss and
Take Profit values marked with red lines. The Take Profit level of the future ORDER_TYPE_BUY_LIMIT
order practically coincides with the activation level of the newly created preliminary order
ORDER_TYPE_BUY_STOP_LIMIT.
As an additional example for self-study, an Expert Advisor AllPendingsOrderSend.mq5 is included with
the book; the Expert Advisor sets 6 pending orders at once: one of each type.

---

## Page 1275

Part 6. Trading automation
1 275
6.4 Creating Expert Advisors
Pending orders of all types
As a result of running it with default settings, you may get log entries as follows:
Autodetected daily range: 0.01413
OK order placed: #=1282032135
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_BUY_LIMIT, V=0.01, ORDER_FILLING_FOK, »
  » @ 1.08824, ORDER_TIME_GTC, M=1234567890
DONE, #=1282032135, V=0.01, Request executed, Req=73
OK order placed: #=1282032136
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_SELL_LIMIT, V=0.01, ORDER_FILLING_FOK, »
  » @ 1.10238, ORDER_TIME_GTC, M=1234567890
DONE, #=1282032136, V=0.01, Request executed, Req=74
OK order placed: #=1282032138
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1.10944, ORDER_TIME_GTC, M=1234567890
DONE, #=1282032138, V=0.01, Request executed, Req=75
OK order placed: #=1282032141
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_SELL_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1.08118, ORDER_TIME_GTC, M=1234567890
DONE, #=1282032141, V=0.01, Request executed, Req=76
OK order placed: #=1282032142
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_BUY_STOP_LIMIT, V=0.01, ORDER_FILLING_FOK, »
  » @ 1.10520, X=1.09531, ORDER_TIME_GTC, M=1234567890
DONE, #=1282032142, V=0.01, Request executed, Req=77
OK order placed: #=1282032144
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_SELL_STOP_LIMIT, V=0.01, ORDER_FILLING_FOK, »
  » @ 1.08542, X=1.09531, ORDER_TIME_GTC, M=1234567890
DONE, #=1282032144, V=0.01, Request executed, Req=78
Alert: 6 pending orders placed - remove them manually, please

---

## Page 1276

Part 6. Trading automation
1 276
6.4 Creating Expert Advisors
6.4.20 Modifying a pending order
MetaTrader 5 allows you to modify certain properties of a pending order, including the activation price,
protection levels, and expiration date. The main properties such as order type or volume cannot be
changed. In such cases, you should delete the order and replace it with another one. The only case
where the order type can be changed by the server itself is the activation of a stop limit order, which
turns into the corresponding limit order.
Programmatic modification of orders is performed by the TRADE_ACTION_MODIFY operation: it is this
constant that needs to be written in the field action of the structure MqlTradeRequest before sending to
the server by the function OrderSend or OrderSendAsync. The ticket of the modified order is indicated
in the field order. Taking into account action and order, the full list of required fields for this operation
includes:
• action
• order
• price
• type_time (default value 0 corresponds to ORDER_TIME_GTC)
• expiration (default 0, not important for ORDER_TIME_GTC)
• type_filling (default 0 corresponds to ORDER_FILLING_FOK)
• stoplimit (only for orders of types ORDER_TYPE_BUY_STOP_LIMIT and
ORDER_TYPE_SELL_STOP_LIMIT)
Optional fields:
• sl
• tp
If protective levels have already been set for the order, they should be specified so they can be saved.
Zero values indicate deletion of Stop Loss and/or Take Profit.
In the MqlTradeRequestSync structure (MqlTradeSync.mqh), the implementation of order modification is
placed in the modify method.

---

## Page 1277

Part 6. Trading automation
1 277
6.4 Creating Expert Advisors
struct MqlTradeRequestSync: public MqlTradeRequest
{
   ...
   bool modify(const ulong ticket,
      const double p, const double stop = 0, const double take = 0,
      ENUM_ORDER_TYPE_TIME duration = ORDER_TIME_GTC, datetime until = 0,
      const double origin = 0)
   {
      if(!OrderSelect(ticket)) return false;
      
      action = TRADE_ACTION_MODIFY;
      order = ticket;
      
      // the following fields are needed for checks inside subfunctions
      type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
      symbol = OrderGetString(ORDER_SYMBOL);
      volume = OrderGetDouble(ORDER_VOLUME_CURRENT);
      
      if(!setVolumePrices(volume, p, stop, take, origin)) return false;
      if(!setExpiration(duration, until)) return false;
      ZeroMemory(result);
      return OrderSend(this, result);
   }
The actual execution of the request is again done in the completed method, in the dedicated branch of
the if operator.
   bool completed()
   {
      ...
      else if(action == TRADE_ACTION_MODIFY)
      {
         result.order = order;
         result.bid = sl;
         result.ask = tp;
         result.price = price;
         result.volume = stoplimit;
         return result.modified(timeout);
      }
      ...
   }
For the MqlTradeResultSync structure to know the new values of the properties of the edited order and
to be able to compare them with the result, we write them in free fields (they are not filled by the
server in this type of request). Further in the modified method, the result structure is waiting for the
modification to be applied.

---

## Page 1278

Part 6. Trading automation
1 278
6.4 Creating Expert Advisors
struct MqlTradeResultSync: public MqlTradeResult
{
   ...
   bool modified(const ulong msc = 1000)
   {
      if(retcode != TRADE_RETCODE_DONE && retcode != TRADE_RETCODE_PLACED)
      {
         return false;
      }
   
      if(!wait(orderModified, msc))
      {
         Print("Order not found in environment: #" + (string)order);
         return false;
      }
      return true;
   }
   
   static bool orderModified(MqlTradeResultSync &ref)
   {
      if(!(OrderSelect(ref.order) || HistoryOrderSelect(ref.order)))
      {
         Print("OrderSelect failed: #=" + (string)ref.order);
         return false;
      }
      return TU::Equal(ref.bid, OrderGetDouble(ORDER_SL))
         && TU::Equal(ref.ask, OrderGetDouble(ORDER_TP))
         && TU::Equal(ref.price, OrderGetDouble(ORDER_PRICE_OPEN))
         && TU::Equal(ref.volume, OrderGetDouble(ORDER_PRICE_STOPLIMIT));
   }
Here we see how the order properties are read using the OrderGetDouble function and compared with
the specified values. All this happens according to the already familiar procedure, in a loop inside the
wait function, within a certain timeout of msc (1 000 milliseconds by default).
As an example, let's use the Expert Advisor PendingOrderModify.mq5, while inheriting some code
fragments from PendingOrderSend.mq5. In particular, a set of input parameters and the PlaceOrder
function to create a new order. It is used at the first launch if there is no order for the given
combination of the symbol and Magic number, thus ensuring that the Expert Advisor has something to
modify.
A new function was required to find a suitable order: GetMyOrder. It is very similar to the GetMyPosition
function, which was used in the example with position tracking (TrailingStop.mq5) to find a suitable
position. The purpose of the built-in MQL5 API functions used inside GetMyOrder should be generally
clear from their names, and the technical description will be presented in separate sections.

---

## Page 1279

Part 6. Trading automation
1 279
6.4 Creating Expert Advisors
ulong GetMyOrder(const string name, const ulong magic)
{
   for(int i = 0; i < OrdersTotal(); ++i)
   {
      ulong t = OrderGetTicket(i);
      if(OrderGetInteger(ORDER_MAGIC) == magic
         && OrderGetString(ORDER_SYMBOL) == name)
      {
         return t;
      }
   }
   
   return 0;
}
The input parameter Distance2SLTP is now missing. Instead, the new Expert Advisor will automatically
calculate the daily range of prices and place protective levels at a distance of half of this range. At the
beginning of each day, the range and the new levels in the sl and tp fields will be recalculated. Order
modification requests will be generated based on the new values.
Those pending orders that trigger and turn into positions will be closed upon reaching Stop Loss or Take
Profit. The terminal can inform the MQL program about the activation of pending orders and the closing
of positions if you describe trading event handlers in it. This would allow, for example, to avoid the
creation of a new order if there is an open position. However, the current strategy can also be used.
So, we will deal with events later.
The main logic of the Expert Advisor is implemented in the OnTick handler.
void OnTick()
{
   static datetime lastDay = 0;
   static const uint DAYLONG = 60 * 60 * 24; // number of seconds in a day
   //discard the "fractional" part, i.e. time
   if(TimeTradeServer() / DAYLONG * DAYLONG == lastDay) return;
   ...
Two lines at the beginning of the function ensure that the algorithm runs once at the beginning of each
day. To do this, we calculate the current date without time and compare it with the value of the lastDay
variable which contains the last successful date. The success or error status of course becomes clear
at the end of the function, so we'll come back to it later.
Next, the price range for the previous day is calculated.
   const string symbol = StringLen(Symbol) == 0 ? _Symbol : Symbol;
   const double range = iHigh(symbol, PERIOD_D1, 1) - iLow(symbol, PERIOD_D1, 1);
   Print("Autodetected daily range: ", (float)range);
   ...
Depending on whether there is an order or not in the GetMyOrder function, we will either create a new
order via PlaceOrder or edit the existing one using ModifyOrder.

---

## Page 1280

Part 6. Trading automation
1 280
6.4 Creating Expert Advisors
   uint retcode = 0;
   ulong ticket = GetMyOrder(symbol, Magic);
   if(!ticket)
   {
      retcode = PlaceOrder((ENUM_ORDER_TYPE)Type, symbol, Volume,
         range, Expiration, Until, Magic);
   }
   else
   {
      retcode = ModifyOrder(ticket, range, Expiration, Until);
   }
   ...
Both functions, PlaceOrder and ModifyOrder, work on the basis of the Expert Advisor's input parameters
and the found price range. They return the status of the request, which will need to be analyzed in
some way to decide which action to take:
• Update the lastDay variable if the request is successful (the order has been updated and the Expert
Advisor sleeps until the beginning of the next day)
• Leave the old day in lastDay for some time to try again on the next ticks if there are temporary
problems (for example, the trading session has not started yet)
• Stop the Expert Advisor if serious problems are detected (for example, the selected order type or
trade direction is not allowed on the symbol)
   ...
   if(/* some kind of retcode analysis */)
   {
      lastDay = TimeTradeServer() / DAYLONG * DAYLONG;
   }
}
In the section Closing a position: full and partial, we used a simplified analysis with the IS_TANGIBLE
macro which gave an answer in the categories of "yes" and "no" to indicate whether there was an error
or not. Obviously, this approach needs to be improved, and we will return to this issue soon. For now,
we will focus on the main functionality of the Expert Advisor.
The source code of the PlaceOrder function remained virtually unchanged from the previous example.
ModifyOrder is shown below.
Recall that we determined the location of orders based on the daily range, to which the table of
coefficients was applied. The principle has not changed, however, since we now have two functions that
work with orders, PlaceOrder and ModifyOrder, the Coefficients table is placed in a global context. We
will not repeat it here and will go straight to the ModifyOrder function.
uint ModifyOrder(const ulong ticket, const double range,
   ENUM_ORDER_TYPE_TIME expiration, datetime until)
{
   // default values
   const string symbol = OrderGetString(ORDER_SYMBOL);
   const double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   ...
Price levels are calculated depending on the order type and the passed range.

---

## Page 1281

Part 6. Trading automation
1 281 
6.4 Creating Expert Advisors
   const ENUM_ORDER_TYPE type = (ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
   const double price = TU::GetCurrentPrice(type, symbol) + range * Coefficients[type];
   
   // origin is filled only for orders *_STOP_LIMIT
   const bool stopLimit =
      type == ORDER_TYPE_BUY_STOP_LIMIT ||
      type == ORDER_TYPE_SELL_STOP_LIMIT;
   const double origin = stopLimit ? TU::GetCurrentPrice(type, symbol) : 0; 
   
   TU::TradeDirection dir(type);
   const int sltp = (int)(range / 2 / point);
   const double stop = sltp == 0 ? 0 :
      dir.negative(stopLimit ? origin : price, sltp * point);
   const double take = sltp == 0 ? 0 :
      dir.positive(stopLimit ? origin : price, sltp * point);
   ...
After calculating all the values, we create an object of the MqlTradeRequestSync structure and execute
the request.
   MqlTradeRequestSync request(symbol);
   
   ResetLastError();
   // pass the data for the fields, send the order and wait for the result
   if(request.modify(ticket, price, stop, take, expiration, until, origin)
      && request.completed())
   {
      Print("OK order modified: #=", ticket);
   }
   
   Print(TU::StringOf(request));
   Print(TU::StringOf(request.result));
   return request.result.retcode;
}
To analyze retcode which we have to execute in the calling block inside OnTick, a new mechanism was
developed that supplemented the file TradeRetcode.mqh. All server return codes are divided into several
"severity" groups, described by the elements of the TRADE_RETCODE_SEVERITY enumeration.

---

## Page 1282

Part 6. Trading automation
1 282
6.4 Creating Expert Advisors
enum TRADE_RETCODE_SEVERITY
{
   SEVERITY_UNDEFINED,   // something non-standard - just output to the log
   SEVERITY_NORMAL,      // normal operation
   SEVERITY_RETRY,       // try updating environment/prices again (probably several times) 
   SEVERITY_TRY_LATER,   // we should wait and try again
   SEVERITY_REJECT,      // request denied, probably(!) you can try again
                         // 
   SEVERITY_INVALID,     // need to fix the request
   SEVERITY_LIMITS,      // need to check the limits and fix the request
   SEVERITY_PERMISSIONS, // it is required to notify the user and change the program/terminal settings
   SEVERITY_ERROR,       // stop, output information to the log and to the user
};
In a simplistic way, the first half corresponds to recoverable errors: it is usually enough to wait a while
and retry the request. The second half requires you to change the content of the request, check the
account or symbol settings, the permissions for the program, and in the worst case, stop trading.
Those who wish can draw a conditional separator line not after SEVERITY_REJECT, as it is visually
highlighted now, but before it.
The division of all codes into groups is performed by the TradeCodeSeverity function (given with
abbreviations).

---

## Page 1283

Part 6. Trading automation
1 283
6.4 Creating Expert Advisors
TRADE_RETCODE_SEVERITY TradeCodeSeverity(const uint retcode)
{
   static const TRADE_RETCODE_SEVERITY severities[] =
   {
      ...
      SEVERITY_RETRY,       // REQUOTE (10004)
      SEVERITY_UNDEFINED,     
      SEVERITY_REJECT,      // REJECT (10006)
      SEVERITY_NORMAL,      // CANCEL (10007)
      SEVERITY_NORMAL,      // PLACED (10008)
      SEVERITY_NORMAL,      // DONE (10009)
      SEVERITY_NORMAL,      // DONE_PARTIAL (10010)
      SEVERITY_ERROR,       // ERROR (10011)
      SEVERITY_RETRY,       // TIMEOUT (10012)
      SEVERITY_INVALID,     // INVALID (10013)
      SEVERITY_INVALID,     // INVALID_VOLUME (10014)
      SEVERITY_INVALID,     // INVALID_PRICE (10015)
      SEVERITY_INVALID,     // INVALID_STOPS (10016)
      SEVERITY_PERMISSIONS, // TRADE_DISABLED (10017)
      SEVERITY_TRY_LATER,   // MARKET_CLOSED (10018)
      SEVERITY_LIMITS,      // NO_MONEY (10019)
      ...
   };
   
   if(retcode == 0) return SEVERITY_NORMAL;
   if(retcode < 10000 || retcode > HEDGE_PROHIBITED) return SEVERITY_UNDEFINED;
   return severities[retcode - 10000];
}
Thanks to this functionality, the OnTick handler can be supplemented with "smart" error handling. A
static variable RetryFrequency stores the frequency with which the program will try to repeat the
request in case of non-critical errors. The last time such an attempt was made is stored in the
RetryRecordTime variable.
void OnTick()
{
   ...
   const static int DEFAULT_RETRY_TIMEOUT = 1; // seconds
   static int RetryFrequency = DEFAULT_RETRY_TIMEOUT;
   static datetime RetryRecordTime = 0;
   if(TimeTradeServer() - RetryRecordTime < RetryFrequency) return;
   ...
Once the PlaceOrder or ModifyOrder function returns the value of retcode, we learn how severe it is and,
based on the severity, we choose one of three alternatives: stopping the Expert Advisor, waiting for a
timeout, or regular operation (marking the successful modification of the order by the current day in
lastDay).

---

## Page 1284

Part 6. Trading automation
1 284
6.4 Creating Expert Advisors
   const TRADE_RETCODE_SEVERITY severity = TradeCodeSeverity(retcode);
   if(severity >= SEVERITY_INVALID)
   {
      Alert("Can't place/modify pending order, EA is stopped");
      RetryFrequency = INT_MAX;
   }
   else if(severity >= SEVERITY_RETRY)
   {
      RetryFrequency += (int)sqrt(RetryFrequency + 1);
      RetryRecordTime = TimeTradeServer();
      PrintFormat("Problems detected, waiting for better conditions "
         "(timeout enlarged to %d seconds)",
         RetryFrequency);
   }
   else
   {
      if(RetryFrequency > DEFAULT_RETRY_TIMEOUT)
      {
         RetryFrequency = DEFAULT_RETRY_TIMEOUT;
         PrintFormat("Timeout restored to %d second", RetryFrequency);
      }
      lastDay = TimeTradeServer() / DAYLONG * DAYLONG;
   }
In case of repeated problems that are classified as solvable, the RetryFrequency timeout gradually
increases with each subsequent error but resets to 1  second when the request is successfully
processed.
It should be noted that the methods of the applied structure MqlTradeRequestSync check a large
number of combinations of parameters for correctness and, if problems are found, interrupt the process
prior to the SendRequest call. This behavior is enabled by default, but it can be disabled by defining an
empty RETURN(X) macro before the directive #include with MqlTradeSync.mqh.
#define RETURN(X)
#include <MQL5Book/MqlTradeSync.mqh>
With this macro definition, checks will only print warnings to the log but will continue to execute
methods until the SendRequest call.
In any case, after calling one or another method of the MqlTradeResultSync structure, the error code
will be added to retcode. This will be done either by the server or by the MqlTradeRequestSync
structure's checking algorithms (here we utilize the fact that the MqlTradeResultSync instance is
included inside MqlTradeRequestSync). I do not provide here the description of the return of error codes
and the use of the RETURN macro in the MqlTradeRequestSync methods for the sake of brevity. Those
interested can see the full source code in the MqlTradeSync.mqh file.
Let's run the Expert Advisor PendingOrderModify.mq5 in the tester, with the visual mode enabled, using
the data of XAUUSD, H1  (all ticks or real ticks mode). With the default settings, the Expert Advisor will
place orders of the ORDER_TYPE_BUY_STOP type with a minimum lot. Let's make sure from the log
and trading history that the program places pending orders and modifies them at the beginning of each
day.

---

## Page 1285

Part 6. Trading automation
1 285
6.4 Creating Expert Advisors
2022.01.03 01:05:00   Autodetected daily range: 14.37
2022.01.03 01:05:00   buy stop 0.01 XAUUSD at 1845.73 sl: 1838.55 tp: 1852.91 (1830.63 / 1831.36)
2022.01.03 01:05:00   OK order placed: #=2
2022.01.03 01:05:00   TRADE_ACTION_PENDING, XAUUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1845.73, SL=1838.55, TP=1852.91, ORDER_TIME_GTC, M=1234567890
2022.01.03 01:05:00   DONE, #=2, V=0.01, Bid=1830.63, Ask=1831.36, Request executed
2022.01.04 01:05:00   Autodetected daily range: 33.5
2022.01.04 01:05:00   order modified [#2 buy stop 0.01 XAUUSD at 1836.56]
2022.01.04 01:05:00   OK order modified: #=2
2022.01.04 01:05:00   TRADE_ACTION_MODIFY, XAUUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, » 
  » @ 1836.56, SL=1819.81, TP=1853.31, ORDER_TIME_GTC, #=2
2022.01.04 01:05:00   DONE, #=2, @ 1836.56, Bid=1819.81, Ask=1853.31, Request executed, Req=1
2022.01.05 01:05:00   Autodetected daily range: 18.23
2022.01.05 01:05:00   order modified [#2 buy stop 0.01 XAUUSD at 1832.56]
2022.01.05 01:05:00   OK order modified: #=2
2022.01.05 01:05:00   TRADE_ACTION_MODIFY, XAUUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1832.56, SL=1823.45, TP=1841.67, ORDER_TIME_GTC, #=2
2022.01.05 01:05:00   DONE, #=2, @ 1832.56, Bid=1823.45, Ask=1841.67, Request executed, Req=2
...
2022.01.11 01:05:00   Autodetected daily range: 11.96
2022.01.11 01:05:00   order modified [#2 buy stop 0.01 XAUUSD at 1812.91]
2022.01.11 01:05:00   OK order modified: #=2
2022.01.11 01:05:00   TRADE_ACTION_MODIFY, XAUUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1812.91, SL=1806.93, TP=1818.89, ORDER_TIME_GTC, #=2
2022.01.11 01:05:00   DONE, #=2, @ 1812.91, Bid=1806.93, Ask=1818.89, Request executed, Req=6
2022.01.11 18:10:58   order [#2 buy stop 0.01 XAUUSD at 1812.91] triggered
2022.01.11 18:10:58   deal #2 buy 0.01 XAUUSD at 1812.91 done (based on order #2)
2022.01.11 18:10:58   deal performed [#2 buy 0.01 XAUUSD at 1812.91]
2022.01.11 18:10:58   order performed buy 0.01 at 1812.91 [#2 buy stop 0.01 XAUUSD at 1812.91]
2022.01.11 20:28:59   take profit triggered #2 buy 0.01 XAUUSD 1812.91 sl: 1806.93 tp: 1818.89 »
  » [#3 sell 0.01 XAUUSD at 1818.89]
2022.01.11 20:28:59   deal #3 sell 0.01 XAUUSD at 1818.91 done (based on order #3)
2022.01.11 20:28:59   deal performed [#3 sell 0.01 XAUUSD at 1818.91]
2022.01.11 20:28:59   order performed sell 0.01 at 1818.91 [#3 sell 0.01 XAUUSD at 1818.89]
2022.01.12 01:05:00   Autodetected daily range: 23.28
2022.01.12 01:05:00   buy stop 0.01 XAUUSD at 1843.77 sl: 1832.14 tp: 1855.40 (1820.14 / 1820.49)
2022.01.12 01:05:00   OK order placed: #=4
2022.01.12 01:05:00   TRADE_ACTION_PENDING, XAUUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1843.77, SL=1832.14, TP=1855.40, ORDER_TIME_GTC, M=1234567890
2022.01.12 01:05:00   DONE, #=4, V=0.01, Bid=1820.14, Ask=1820.49, Request executed, Req=7
The order can be triggered at any moment, after which the position is closed after some time by the
stop loss or take profit (as in the code above).
In some cases, a situation may arise when the position still exists at the beginning of the next day, and
then a new order will be created in addition to it, as in the screenshot below.

---

## Page 1286

Part 6. Trading automation
1 286
6.4 Creating Expert Advisors
The Expert Advisor with a trading strategy based on pending orders in the tester
Please note that due to the fact that we request quotes of the PERIOD_D1  timeframe to calculate the
daily range, the visual tester opens the corresponding chart, in addition to the current working one.
Such a service works not only for timeframes other than the working one but also for other symbols.
This will be useful, in particular, when developing multicurrency Expert Advisors.
To check how error handling works, try disabling trading for the Expert Advisor. The log will contain the
following:
Autodetected daily range: 34.48
TRADE_ACTION_PENDING, XAUUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
  » @ 1975.73, SL=1958.49, TP=1992.97, ORDER_TIME_GTC, M=1234567890
CLIENT_DISABLES_AT, AutoTrading disabled by client
Alert: Can't place/modify pending order, EA is stopped
This error is critical, and the Expert Advisor stops working.
To demonstrate one of the easier errors, we could use the OnTimer handler instead of OnTick. Then
launching the same Expert Advisor on symbols where trading sessions take only a part of a day would
periodically generate a sequence of non-critical errors about a closed market ("Market closed"). In this
case, the Expert Advisor would keep trying to start trading, constantly increasing the waiting time.
This, in particular, is easy to check in the tester, which allows you to set up arbitrary trading sessions
for any symbol. On the Settings tab, to the right of the Delays dropdown list, there is a button that
opens the Trade setup dialog. There, you should include the option Use your settings and on the Trade
tab add at least one record to the table Non-trading periods.

---

## Page 1287

Part 6. Trading automation
1 287
6.4 Creating Expert Advisors
Setting up non-trading periods in the tester
Please note that it is non-trading periods that are set here, not trading sessions, i.e., this setting acts
exactly the opposite in comparison with the symbol specification.
Many potential errors related to trade restrictions can be eliminated by preliminary analysis of the
environment using a class like Permissions presented in the section Restrictions and permissions for
account transactions.
6.4.21  Deleting a pending order
Deletion of a pending order is performed at the program level using the TRADE_ACTION_REMOVE
operation: this constant should be assigned to the action field of the MqlTradeRequest structure before
calling one of the versions of the OrderSend function. The only required field in addition to action is
order to specify the ticket of the order to be deleted.
The remove method in MqlTradeRequestSync application structure from the MqlTradeSync.mqh file is
pretty basic.

---

## Page 1288

Part 6. Trading automation
1 288
6.4 Creating Expert Advisors
struct MqlTradeRequestSync: public MqlTradeRequest
{
   ...
   bool remove(const ulong ticket)
   {
      if(!OrderSelect(ticket)) return false;
      action = TRADE_ACTION_REMOVE;
      order = ticket;
      ZeroMemory(result);
      return OrderSend(this, result);
   }
Checking the fact of deleting an order is traditionally done in the completed method.
   bool completed()
   {
      ...
      else if(action == TRADE_ACTION_REMOVE)
      {
         result.order = order;
         return result.removed(timeout);
      }
      ...
   }
Waiting for the actual removal of the order is performed in the removed method of the
MqlTradeResultSync structure.

---

## Page 1289

Part 6. Trading automation
1 289
6.4 Creating Expert Advisors
struct MqlTradeResultSync: public MqlTradeResult
{
   ...
   bool removed(const ulong msc = 1000)
   {
      if(retcode != TRADE_RETCODE_DONE)
      {
         return false;
      }
   
      if(!wait(orderRemoved, msc))
      {
         Print("Order removal timeout: #=" + (string)order);
         return false;
      }
      
      return true;
   }
   
   static bool orderRemoved(MqlTradeResultSync &ref)
   {
      return !OrderSelect(ref.order) && HistoryOrderSelect(ref.order);
   }
An example of the Expert Advisor (PendingOrderDelete.mq5) demonstrating the removal of an order we
will build almost entirely based on PendingOrderSend.mq5. This is due to the fact that it is easier to
guarantee the existence of an order before deletion. Thus, immediately after the launch, the Expert
Advisor will create a new order with the specified parameters. The order will then be deleted in the
OnDeinit handler. If you change the Expert Advisor input parameters, the symbol, or the chart
timeframe, the old order will also be deleted, and a new one will be created.
The OwnOrder global variable has been added to store the order ticket. It is filled as a result of the
PlaceOrder call (the function itself is unchanged).
ulong OwnOrder = 0;
   
void OnTimer()
{
   // execute the code once for the current parameters
   EventKillTimer();
   
   const string symbol = StringLen(Symbol) == 0 ? _Symbol : Symbol;
   OwnOrder = PlaceOrder((ENUM_ORDER_TYPE)Type, symbol, Volume,
      Distance2SLTP, Expiration, Until, Magic, Comment);
}
Here is a simple deletion function RemoveOrder, which creates the request object and sequentially calls
the remove and completed methods for it.

---

## Page 1290

Part 6. Trading automation
1 290
6.4 Creating Expert Advisors
void OnDeinit(const int)
{
   if(OwnOrder != 0)
   {
      RemoveOrder(OwnOrder);
   }
}
   
void RemoveOrder(const ulong ticket)
{
   MqlTradeRequestSync request;
   if(request.remove(ticket) && request.completed())
   {
      Print("OK order removed");
   }
   Print(TU::StringOf(request));
   Print(TU::StringOf(request.result));
}
The following log shows the entries that appeared as a result of placing the Expert Advisor on the
EURUSD chart, after which the symbol was switched to XAUUSD, and then the Expert Advisor was
deleted.
(EURUSD,H1)Autodetected daily range: 0.0094
(EURUSD,H1)OK order placed: #=1284920879
(EURUSD,H1)TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
                » @ 1.11011, ORDER_TIME_GTC, M=1234567890
(EURUSD,H1)DONE, #=1284920879, V=0.01, Request executed, Req=1
(EURUSD,H1)OK order removed
(EURUSD,H1)TRADE_ACTION_REMOVE, EURUSD, ORDER_TYPE_BUY, ORDER_FILLING_FOK, #=1284920879
(EURUSD,H1)DONE, #=1284920879, Request executed, Req=2
(XAUUSD,H1)Autodetected daily range: 47.45
(XAUUSD,H1)OK order placed: #=1284921672
(XAUUSD,H1)TRADE_ACTION_PENDING, XAUUSD, ORDER_TYPE_BUY_STOP, V=0.01, ORDER_FILLING_FOK, »
                » @ 1956.68, ORDER_TIME_GTC, M=1234567890
(XAUUSD,H1)DONE, #=1284921672, V=0.01, Request executed, Req=3
(XAUUSD,H1)OK order removed
(XAUUSD,H1)TRADE_ACTION_REMOVE, XAUUSD, ORDER_TYPE_BUY, ORDER_FILLING_FOK, #=1284921672
(XAUUSD,H1)DONE, #=1284921672, Request executed, Req=4
We will look at another example of deleting orders to implement the "One Cancel Other" (OCO)
strategy in the OnTrade events section.
6.4.22 Getting a list of active orders
Expert Advisor programs often need to enumerate existing active orders and analyze their properties. In
particular, in the section on pending order modifications, in the example PendingOrderModify.mq5, we
have created a special function GetMyOrder to find the orders belonging to the Expert Advisor in to
modify this order. There, the analysis was carried out by symbol name and Expert Advisor ID (Magic).
In theory, the same approach should have been applied in the example of deleting a pending order
PendingOrderDelete.mq5 from the previous section.

---

## Page 1291

Part 6. Trading automation
1 291 
6.4 Creating Expert Advisors
In the latter case, for simplicity, we created an order and stored its ticket in a global variable. But this
cannot be done in the general case because the Expert Advisor and the entire terminal can be stopped
or restarted at any time. Therefore, the Expert Advisor must contain an algorithm for restoring the
internal state, including the analysis of the entire trading environment, along with orders, deals,
positions, account balance, and so on.
In this section, we will study the MQL5 functions for obtaining a list of active orders and selecting any
of them in the trading environment, which makes it possible to read all its properties.
int OrdersTotal()
The OrdersTotal function returns the number of currently active orders. These include pending orders,
as well as market orders that have not yet been executed. As a rule, a market order is executed
promptly, and therefore it is not often possible to catch it in the active phase, but if there is not enough
liquidity in the market, this can happen. As soon as the order is executed (a deal is concluded), it is
transferred from the category of active ones to history. We will talk about working with order history in
a separate section.
Please note that only orders can be active and historical. This significantly distinguishes orders from
deals which are always created in history and from positions that exist only online. To restore the
history of positions, you should analyze the history of deals.
ulong OrderGetTicket(uint index)
The OrderGetTicket function returns the order ticket by its number in the list of orders in the terminal's
trading environment. The index parameter must be between 0 and the OrdersTotal()-1  value inclusive.
The way in which orders are organized is not regulated.
The OrderGetTicket function selects an order, that is, copies data about it to some internal cache so
that the MQL program can read all its properties using the subsequent calls of the OrderGetDouble,
OrderGetInteger, or OrderGetString function, which will be discussed in a separate section.
The presence of such a cache indicates that the data received from it can become obsolete: the order
may no longer exist or may have been modified (for example, it may have a different status, open price,
Stop Loss or Take Profit levels and expiration). Therefore, to guarantee the receipt of relevant data
about the order, it is recommended to call the OrderGetTicket function immediately prior to requesting
the data. Here is how this is done in the example of PendingOrderModify.mq5.
ulong GetMyOrder(const string name, const ulong magic)
{
   for(int i = 0; i < OrdersTotal(); ++i)
   {
      ulong t = OrderGetTicket(i);
      if(OrderGetInteger(ORDER_MAGIC) == magic
      && OrderGetString(ORDER_SYMBOL) == name)
      {
         return t;
      }
   }
   return 0;
}
Each MQL program maintains its own cache (trading environment context), which includes the selected
order. In the following sections, we will learn that in addition to orders, an MQL program can select
positions and history fragments with deals and orders into the active context.

---

## Page 1292

Part 6. Trading automation
1 292
6.4 Creating Expert Advisors
The OrderSelect function performs a similar selection of an order with copying of its data to the internal
cache.
bool OrderSelect(ulong ticket)
The function checks for the presence of an order and prepares the possibility of further reading its
properties. In this case, the order is specified not by a serial number but by a ticket which must be
received by the MQL program earlier in one way or another, in particular, as a result of executing
OrderSend/OrderSendAsync.
The function returns true in case of success. If false is received, it usually means that there is no order
with the specified ticket. The most common reason for this is when order status has changed from
active to history, for example, as a result of execution or cancellation (we will learn how to determine
the exact status later). Orders can be selected in history using the relevant functions.
Previously we used the OrderSelect function in the MqlTradeResultSync structure for tracking creation
and removal of pending orders.
6.4.23 Order properties (active and history)
In the sections related to trading operations, in particular to making buying/selling, closing a position,
and placing a pending order, we have seen that requests are sent to the server based on the filling of
specific fields of the MqlTradeRequest structure, most of which directly define the properties of the
resulting orders. The MQL5 API allows you to learn these and some other properties set by the trading
system itself, such as ticket, registration time, and status.
It is important to note that the list of order properties is common for both active and historical orders,
although, of course, the values of many properties will differ for them.
Order properties are grouped in MQL5 according to the principle already familiar to us based on the
type of values: integer (compatible with long/ulong), real (double), and strings. Each property group has
its own enumeration.
Integer properties are summarized in ENUM_ORDER_PROPERTY_INTEGER and are presented in the
following table.
Identifier
Description
Type
O R D E R _TYP E 
Order type
E N U M _O R D E R _TYP E 
O R D E R _TYP E _F IL L IN G 
Execution type by volume
E N U M _O R D E R _TYP E _F IL L IN G 
O R D E R _TYP E _TIM E 
Order lifetime (pending)
E N U M _O R D E R _TYP E _TIM E 
O R D E R _TIM E _E XP IR ATIO N 
Order expiration time (pending)
datetime
O R D E R _M AG IC
Arbitrary identifier set by the
Expert Advisor that placed the
order
ulong
O R D E R _TICK E T
Order ticket; a unique number
assigned by the server to each
order
ulong

---

## Page 1293

Part 6. Trading automation
1 293
6.4 Creating Expert Advisors
Identifier
Description
Type
O R D E R _S TATE 
Order status
ENUM_ORDER_STATE (see
below)
O R D E R _R E AS O N 
Reason or source for the order
ENUM_ORDER_REASON (see
below)
O R D E R _TIM E _S E TU P 
Order placement time
datetime
O R D E R _TIM E _D O N E 
Order execution or withdrawal
time
datetime
O R D E R _TIM E _S E TU P _M S C
Time of order placement for
execution in milliseconds
ulong
O R D E R _TIM E _D O N E _M S C
Order execution/withdrawal
time in milliseconds
ulong
O R D E R _P O S ITIO N _ID 
ID of the position that the order
generated or modified upon
execution
ulong
O R D E R _P O S ITIO N _B Y_ID 
Opposite position identifier for
orders of type
ORDER_TYPE_CLOSE_BY
ulong
Each executed order generates a deal that opens a new or changes an existing position. The ID of this
position is assigned to the executed order in the ORDER_POSITION_ID property.
The ENUM_ORDER_STATE enumeration contains elements that describe order statuses. See a
simplified scheme (state diagram) of orders below.

---

## Page 1294

Part 6. Trading automation
1 294
6.4 Creating Expert Advisors
Identifier
Description
ORDER_STATE_STARTED
The order has been checked for correctness but has not yet
been accepted by the server
ORDER_STATE_PLACED
The order has been accepted by the server
ORDER_STATE_CANCELED
The order has been canceled by the client (user or MQL
program)
ORDER_STATE_PARTIAL
The order has been partially executed
ORDER_STATE_FILLED
The order has been filled in full
ORDER_STATE_REJECTED
The order has been rejected by the server
ORDER_STATE_EXPIRED
The order has been canceled upon expiration
ORDER_STATE_REQUEST_ADD
The order is being registered (being placed in the trading
system)
ORDER_STATE_REQUEST_MODIFY
The order is being modified (its parameters are being
changed)
ORDER_STATE_REQUEST_CANCEL
The order is being deleted (removing from the trading
system)
Order status diagram
Changing the state is possible only for active orders. For historical orders (filled or canceled), the status
is fixed.

---

## Page 1295

Part 6. Trading automation
1 295
6.4 Creating Expert Advisors
You can cancel an order that has already been partially fulfilled, and then its status in the history will be
ORDER_STATE_CANCELED.
ORDER_STATE_PARTIAL occurs only for active orders. Executed (historical) orders always have the
status ORDER_STATE_FILLED.
The ENUM_ORDER_REASON enumeration specifies possible order source options.
Identifier
Description
ORDER_REASON_CLIENT
Order placed manually from the desktop terminal
ORDER_REASON_EXPERT
Order placed from the desktop terminal by an Expert Adviser
or a script
ORDER_REASON_MOBILE
Order placed from the mobile application
ORDER_REASON_WEB
Order placed from the web terminal (browser)
ORDER_REASON_SL
Order placed by the server as a result of Stop Loss triggering
ORDER_REASON_TP
Order placed by the server as a result of Take Profit
triggering
ORDER_REASON_SO
Order placed by the server as a result of the Stop Out event
Real properties are collected in the ENUM_ORDER_PROPERTY_DOUBLE enumeration.
Identifier
Description
ORDER_VOLUME_INITIAL
Initial volume when placing an order
ORDER_VOLUME_CURRENT
Current volume (initial or remaining after partial execution)
ORDER_PRICE_OPEN
The price indicated in the order
ORDER_PRICE_CURRENT
The current symbol price of an order that has not yet been
executed or the execution price
ORDER_SL
Stop Loss Level
ORDER_TP
Take Profit level
ORDER_PRICE_STOPLIMIT
The price for placing a Limit order when a StopLimit order is
triggered
The ORDER_PRICE_CURRENT property contains the current Ask price for active buy pending orders or
the Bid price for active sell pending orders. "Current" refers to the price known in the trading
environment at the time the order is selected using OrderSelect or OrderGetTicket. For executed orders
in the history, this property contains the execution price, which may differ from the one specified in the
order due to slippage.
The ORDER_VOLUME_INITIAL and ORDER_VOLUME_CURRENT properties are not equal to each other
only if the order status is ORDER_STATE_PARTIAL.

---

## Page 1296

Part 6. Trading automation
1 296
6.4 Creating Expert Advisors
If the order was filled in parts, then its ORDER_VOLUME_INITIAL property in history will be equal to the
size of the last filled part, and all other "fills" related to the original full volume will be executed as
separate orders (and deals).
String properties are described in the ENUM_ORDER_PROPERTY_STRING enumeration.
Identifier
Description
ORDER_SYMBOL
The symbol on which the order is placed
ORDER_COMMENT
Comment
ORDER_EXTERNAL_ID
Order ID in the external trading system (on the exchange)
To read all the above properties, there are two different sets of functions: for active orders and for
historical orders. First, we will consider the functions for active orders, and we will return to the
historical ones after we get acquainted with the principles of selecting the required period in history.
6.4.24 Functions for reading properties of active orders
The sets of functions that can be used to get the values of all order properties differ for active and
historical orders. This section describes the functions for reading the properties of active orders. For
the functions for accessing the properties of orders in the history, see the relevant section.
Integer properties can be read using the OrderGetInteger function, which has two forms: the first one
returns directly the value of the property, the second one returns a logical sign of success (true) or
error (false), and the second parameter passed by reference is filled with the value of the property.
long OrderGetInteger(ENUM_ORDER_PROPERTY_INTEGER property)
bool OrderGetInteger(ENUM_ORDER_PROPERTY_INTEGER property, long &value)
Both functions allow you to get the requested order property of an integer-compatible type (datetime,
long/ulong or listing). Although the prototype mentions long, from a technical point of view, the value is
stored as an 8-byte cell, which can be cast to compatible types without any conversion of the internal
representation, in particular, to ulong, which is used for all tickets.
A similar pair of functions is intended for properties of real type double.
double OrderGetDouble(ENUM_ORDER_PROPERTY_DOUBLE property)
bool OrderGetDouble(ENUM_ORDER_PROPERTY_DOUBLE property, double &value)
Finally, string properties are available through a pair of OrderGetString functions.
string OrderGetString(ENUM_ORDER_PROPERTY_STRING property)
bool OrderGetString(ENUM_ORDER_PROPERTY_STRING property, string &value)
As their first parameter, all functions take the identifier of the property we are interested in. This must
be an element of one of the enumerations – ENUM_ORDER_PROPERTY_INTEGER,
ENUM_ORDER_PROPERTY_DOUBLE, or ENUM_ORDER_PROPERTY_STRING – discussed in the previous
section.
Please note before calling any of the previous functions, you should first select an order using
OrderSelect or OrderGetTicket.

---

## Page 1297

Part 6. Trading automation
1 297
6.4 Creating Expert Advisors
To read all the properties of a specific order, we will develop the OrderMonitor class (OrderMonitor.mqh)
which operates on the same principle as the previously considered symbol (SymbolMonitor.mqh) and
trading account (AccountMonitor.mqh) monitors.
These and other monitor classes discussed in the book offer a unified way to analyze properties through
overloaded versions of virtual get methods.
Looking a little ahead, let's say that deals and positions have the same grouping of properties according
to the three main types of values, and we also need to implement monitors for them. In this regard, it
makes sense to separate the general algorithm into a base abstract class MonitorInterface
(TradeBaseMonitor.mqh). This is a template class with three parameters intended to specify the types
of specific enumerations, for integer (I), real (D), and string (S) property groups.
#include <MQL5Book/EnumToArray.mqh>
   
template<typename I,typename D,typename S>
class MonitorInterface
{
protected:
   bool ready;
public:
   MonitorInterface(): ready(false) { }
   
   bool isReady() const
   {
      return ready;
   }
   ...
Due to the fact that finding an order (deal or position) in the trading environment may fail for various
reasons, the class has a reserved variable ready in which derived classes will have to write a sign of
successful initialization, that is, the choice of an object to read its properties.
Several purely virtual methods declare access to properties of the corresponding types.
   virtual long get(const I property) const = 0;
   virtual double get(const D property) const = 0;
   virtual string get(const S property) const = 0;
   virtual long get(const int property, const long) const = 0;
   virtual double get(const int property, const double) const = 0;
   virtual string get(const int property, const string) const = 0;
   ...
In the first three methods, the property type is specified by one of the template parameters. In further
three methods, the type is specified by the second parameter of the method itself: this is required
because the last methods take not the constants of a particular enumeration but simply an integer as
the first parameter. On the one hand, this is convenient for the continuous numbering of identifiers (the
enumeration constants of the three types do not intersect). On the other hand, we need another
source for determining the type of value since the type returned by the function/method does not
participate in the process of choosing the appropriate overload.
This approach allows you to get properties based on various inputs available in the calling code. Next,
we will create classes based on OrderMonitor (as well as future DealMonitor and PositionMonitor) to
select objects according to a set of arbitrary conditions, and there all these methods will be in demand.

---

## Page 1298

Part 6. Trading automation
1 298
6.4 Creating Expert Advisors
Quite often, programs need to get a string representation of any properties, for example, for logging. In
the new monitors, this is implemented by the stringify methods. Obviously, they get the values of the
requested properties through get method calls mentioned above.
   virtual string stringify(const long v, const I property) const = 0;
   
   virtual string stringify(const I property) const
   {
      return stringify(get(property), property);
   }
   
   virtual string stringify(const D property, const string format = NULL) const
   {
      if(format == NULL) return (string)get(property);
      return StringFormat(format, get(property));
   }
   
   virtual string stringify(const S property) const
   {
      return get(property);
   }
   ...
The only method that has not received implementation is the first version of stringify for type long. This
is due to the fact that the group of integer properties, as we saw in the previous section, actually
contain different application types, including date and time, enumerations, and integers. Therefore,
only derived classes can provide their conversion to understandable strings. This situation is common
for all trading entities, not only orders but also deals and positions the properties of which we will
consider later.
When an integer property contains an enumeration element (for example, ENUM_ORDER_TYPE,
ORDER_TYPE_FILLING, etc.), you should use the EnumToString function to convert it to a string. This
task is fulfilled by a helper method enumstr. Soon we will see its widespread use in specific monitor
classes, starting with OrderMonitor after a couple of paragraphs.
   template<typename E>
   static string enumstr(const long v)
   {
      return EnumToString((E)v);
   }
To log all properties of a particular type, we have created the list2log method which uses stringify in a
loop.

---

## Page 1299

Part 6. Trading automation
1 299
6.4 Creating Expert Advisors
   template<typename E>
   void list2log() const
   {
      E e = (E)0; // suppress warning 'possible use of uninitialized variable'
      int array[];
      const int n = EnumToArray(e, array, 0, USHORT_MAX);
      Print(typename(E), " Count=", n);
      for(int i = 0; i < n; ++i)
      {
         e = (E)array[i];
         PrintFormat("% 3d %s=%s", i, EnumToString(e), stringify(e));
      }
   }
Finally, to make it easier to log the properties of all three groups, there is a method print which calls
list2log three times for each group of properties.
   virtual void print() const
   {
      if(!ready) return;
      
      Print(typename(this));
      list2log<I>();
      list2log<D>();
      list2log<S>();
   }
Having at our disposal a base template class MonitorInterface, we describe OrderMonitorInterface,
where we specify certain enumeration types for orders from the previous section and provide an
implementation of stringify for integer properties of orders.

---

## Page 1300

Part 6. Trading automation
1 300
6.4 Creating Expert Advisors
class OrderMonitorInterface:
   public MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER,
   ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>
{
public:
   // description of properties according to subtypes
   virtual string stringify(const long v,
      const ENUM_ORDER_PROPERTY_INTEGER property) const override
   {
      switch(property)
      {
         case ORDER_TYPE:
            return enumstr<ENUM_ORDER_TYPE>(v);
         case ORDER_STATE:
            return enumstr<ENUM_ORDER_STATE>(v);
         case ORDER_TYPE_FILLING:
            return enumstr<ENUM_ORDER_TYPE_FILLING>(v);
         case ORDER_TYPE_TIME:
            return enumstr<ENUM_ORDER_TYPE_TIME>(v);
         case ORDER_REASON:
            return enumstr<ENUM_ORDER_REASON>(v);
         
         case ORDER_TIME_SETUP:
         case ORDER_TIME_EXPIRATION:
         case ORDER_TIME_DONE:
            return TimeToString(v, TIME_DATE | TIME_SECONDS);
         
         case ORDER_TIME_SETUP_MSC:
         case ORDER_TIME_DONE_MSC:
            return STR_TIME_MSC(v);
      }
      
      return (string)v;
   }
};
The STR_TIME_MSC macro for displaying time in milliseconds is defined as follows:
#define STR_TIME_MSC(T) (TimeToString((T) / 1000, TIME_DATE | TIME_SECONDS) \
    + StringFormat("'%03d", (T) % 1000))
Now we are ready to describe the final class for reading the properties of any order: OrderMonitor
derived from OrderMonitorInterface. The order ticket is passed to the constructor, and it is selected in
the trading environment using OrderSelect.

---

## Page 1301

Part 6. Trading automation
1 301 
6.4 Creating Expert Advisors
class OrderMonitor: public OrderMonitorInterface
{
public:
   const ulong ticket;
   OrderMonitor(const long t): ticket(t)
   {
      if(!OrderSelect(ticket))
      {
         PrintFormat("Error: OrderSelect(%lld) failed: %s",
            ticket, E2S(_LastError));
      }
      else
      {
         ready = true;
      }
   }
   ...
The main working part of the monitor consists of redefinitions of virtual functions for reading properties.
Here we see the OrderGetInteger, OrderGetDouble, and OrderGetString function calls.

---

## Page 1302

Part 6. Trading automation
1 302
6.4 Creating Expert Advisors
   virtual long get(const ENUM_ORDER_PROPERTY_INTEGER property) const override
   {
      return OrderGetInteger(property);
   }
   
   virtual double get(const ENUM_ORDER_PROPERTY_DOUBLE property) const override
   {
      return OrderGetDouble(property);
   }
   
   virtual string get(const ENUM_ORDER_PROPERTY_STRING property) const override
   {
      return OrderGetString(property);
   }
   
   virtual long get(const int property, const long) const override
   {
      return OrderGetInteger((ENUM_ORDER_PROPERTY_INTEGER)property);
   }
   
   virtual double get(const int property, const double) const override
   {
      return OrderGetDouble((ENUM_ORDER_PROPERTY_DOUBLE)property);
   }
   
   virtual string get(const int property, const string)  const override
   {
      return OrderGetString((ENUM_ORDER_PROPERTY_STRING)property);
   }
};
This code fragment is presented in a short form: operators for working with orders in the history have
been removed from it. we will see the full code of OrderMonitor later when we explore this aspect in the
following sections.
It is important to note that the monitor object does not store copies of its properties. Therefore,
access to get methods must be carried out immediately after the creation of the object and,
accordingly, the call OrderSelect. To read the properties at a later period, you will need to allocate the
order again in the internal cache of the MQL program, for example, by calling the method refresh.
   void refresh()
   {
      ready = OrderSelect(ticket);
   }
Let's test the work of OrderMonitor by adding it to the Expert Advisor MarketOrderSend.mq5. A new
version named MarketOrderSendMonitor.mq5 connects the file OrderMonitor.mqh by the directive
#include, and in the body of the function OnTimer (in the block of successful confirmation of opening a
position on an order) creates a monitor object and calls its print method.

---

## Page 1303

Part 6. Trading automation
1 303
6.4 Creating Expert Advisors
#include <MQL5Book/OrderMonitor.mqh>
...
void OnTimer()
{
   ...
   const ulong order = (wantToBuy ?
      request.buy(volume, Price) :
      request.sell(volume, Price));
   if(order != 0)
   {
      Print("OK Order: #=", order);
      if(request.completed())
      {
         Print("OK Position: P=", request.result.position);
         
         OrderMonitor m(order);
         m.print();
         ...
      }
   }
}
In the log, we should see new lines containing all the properties of the order.

---

## Page 1304

Part 6. Trading automation
1 304
6.4 Creating Expert Advisors
OK Order: #=1287846602
Waiting for position for deal D=1270417032
OK Position: P=1287846602
MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER, »
   » ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>
ENUM_ORDER_PROPERTY_INTEGER Count=14
  0 ORDER_TIME_SETUP=2022.03.21 13:28:59
  1 ORDER_TIME_EXPIRATION=1970.01.01 00:00:00
  2 ORDER_TIME_DONE=2022.03.21 13:28:59
  3 ORDER_TYPE=ORDER_TYPE_BUY
  4 ORDER_TYPE_FILLING=ORDER_FILLING_FOK
  5 ORDER_TYPE_TIME=ORDER_TIME_GTC
  6 ORDER_STATE=ORDER_STATE_FILLED
  7 ORDER_MAGIC=1234567890
  8 ORDER_POSITION_ID=1287846602
  9 ORDER_TIME_SETUP_MSC=2022.03.21 13:28:59'572
 10 ORDER_TIME_DONE_MSC=2022.03.21 13:28:59'572
 11 ORDER_POSITION_BY_ID=0
 12 ORDER_TICKET=1287846602
 13 ORDER_REASON=ORDER_REASON_EXPERT
ENUM_ORDER_PROPERTY_DOUBLE Count=7
  0 ORDER_VOLUME_INITIAL=0.01
  1 ORDER_VOLUME_CURRENT=0.0
  2 ORDER_PRICE_OPEN=1.10275
  3 ORDER_PRICE_CURRENT=1.10275
  4 ORDER_PRICE_STOPLIMIT=0.0
  5 ORDER_SL=0.0
  6 ORDER_TP=0.0
ENUM_ORDER_PROPERTY_STRING Count=3
  0 ORDER_SYMBOL=EURUSD
  1 ORDER_COMMENT=
  2 ORDER_EXTERNAL_ID=
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, »
   » @ 1.10275, P=1287846602, M=1234567890
DONE, D=1270417032, #=1287846602, V=0.01, @ 1.10275, Bid=1.10275, Ask=1.10275, »
   » Request executed, Req=3
The fourth line starts the output from the print method which includes the full name of the monitor
object MonitorInterface together with parameter types (in this case, the triple
ENUM_ORDER_PROPERTY) and then all the properties of a particular order.
However, property printing is not the most interesting action a monitor can provide. The task of
selecting orders by conditions (values of arbitrary properties) is much more in demand among Experts
Advisors. Using the monitor as an auxiliary tool, we will create a mechanism for filtering orders similar
to what we have done for symbols: SymbolFilter.mqh.
6.4.25 Selecting orders by properties
In one of the sections on symbol properties, we introduced the SymbolFilter class to select financial
instruments with specified characteristics. Now we will apply the same approach for orders.

---

## Page 1305

Part 6. Trading automation
1 305
6.4 Creating Expert Advisors
Since we have to analyze not only orders but also deals and positions in a similar way, we will separate
the general part of the filtering algorithm into the base class TradeFilter (TradeFilter.mqh). It almost
exactly repeats the source code of SymbolFilter. Therefore, we will not explain it here again.
Those who wish can perform a contextual file comparison of SymbolFilter.mqh and TradeFilter.mqh to
see how similar they are and to localize minor edits.
The main difference is that the TradeFilter class is a template since it has to deal with the properties of
different objects: orders, deals, and positions.

---

## Page 1306

Part 6. Trading automation
1 306
6.4 Creating Expert Advisors
enum IS // supported comparison conditions in filters
{
   EQUAL,
   GREATER,
   NOT_EQUAL,
   LESS
};
   
enum ENUM_ANY // dummy enum to cast all enums to it
{
};
   
template<typename T,typename I,typename D,typename S>
class TradeFilter
{
protected:
   MapArray<ENUM_ANY,long> longs;
   MapArray<ENUM_ANY,double> doubles;
   MapArray<ENUM_ANY,string> strings;
   MapArray<ENUM_ANY,IS> conditions;
   ...
   
   template<typename V>
   static bool equal(const V v1, const V v2);
   
   template<typename V>
   static bool greater(const V v1, const V v2);
   
   template<typename V>
   bool match(const T &m, const MapArray<ENUM_ANY,V> &data) const;
   
public:
   // methods for adding conditions to the filter
   TradeFilter *let(const I property, const long value, const IS cmp = EQUAL);
   TradeFilter *let(const D property, const double value, const IS cmp = EQUAL);
   TradeFilter *let(const S property, const string value, const IS cmp = EQUAL);
   // methods for getting into arrays of records matching the filter
   template<typename E,typename V>
   bool select(const E property, ulong &tickets[], V &data[],
      const bool sort = false) const;
   template<typename E,typename V>
   bool select(const E &property[], ulong &tickets[], V &data[][],
      const bool sort = false) const
   bool select(ulong &tickets[]) const;
   ...
}
The template parameters I, D and S are enumerations for property groups of three main types (integer,
real, and string): for orders, they were described in previous sections, so for clarity, you can imagine
that I= E N U M _OR D E R _PR OPE R TY_IN TE GE R , D= E N U M _OR D E R _PR OPE R TY_D OU B L E ,
S=E N U M _OR D E R _PR OPE R TY_STR IN G.

---

## Page 1307

Part 6. Trading automation
1 307
6.4 Creating Expert Advisors
The T type is designed for specifying a monitor class. At the moment we have only one monitor ready,
OrderMonitor. Later we will implement DealMonitor and PositionMonitor.
Earlier, in the SymbolFilter class, we did not use template parameters because for symbols, all types of
property enumerations are invariably known, and there is a single class SymbolMonitor.
Recall the structure of the filter class. A group of let methods allows you to register a combination of
"property=value" pairs in the filter, which will then be used to select objects in select methods. The ID
property is specified in the property parameter, and the value is in the value parameter.
There are also several select methods. They allow the calling code to fill in an array with selected
tickets, as well as, if necessary, additional arrays with the values of the requested object properties.
The specific identifiers of the requested properties are set in the first parameter of the select method;
it can be one property or several. Depending on this, the receiving array must be one-dimensional or
two-dimensional.
The combination of property and value can be checked not only for equality (EQUAL) but also for
greater/less operations (GREATER/LESS). For string properties, it is acceptable to specify a search
pattern with the character "*" denoting any sequence of characters (for example, "*[tp]*" for the
ORDER_COMMENT property will match all comments in which "[tp]" occurs anywhere, although this is
only demonstration of the possibility – while to search for orders resulting from triggered Take Profit
you should analyze ORDER_REASON).
Since the algorithm requires the implementation of a loop though all objects and objects can be of
different types (so far these are orders, but then support for deals and positions will appear), we need
to describe two abstract methods in the TradeFilter class: total and get:
   virtual int total() const = 0;
   virtual ulong get(const int i) const = 0;
The first one returns the number of objects and the second one returns the order ticket by its number.
This should remind you of the pair of functions OrdersTotal and OrderGetTicket. Indeed, they are used
in specific implementations of methods for filtering orders.
Below is the OrderFilter class (OrderFilter.mqh) in full.

---

## Page 1308

Part 6. Trading automation
1 308
6.4 Creating Expert Advisors
#include <MQL5Book/OrderMonitor.mqh>
#include <MQL5Book/TradeFilter.mqh>
   
class OrderFilter: public TradeFilter<OrderMonitor,
   ENUM_ORDER_PROPERTY_INTEGER,
   ENUM_ORDER_PROPERTY_DOUBLE,
   ENUM_ORDER_PROPERTY_STRING>
{
protected:
   virtual int total() const override
   {
      return OrdersTotal();
   }
   virtual ulong get(const int i) const override
   {
      return OrderGetTicket(i);
   }
};
This simplicity is especially important given that similar filters will be created effortlessly for trades and
positions.
With the help of the new class, we can much more easily check the presence of orders belonging to our
Expert Advisor, i.e., replace any self-written versions of the GetMyOrder function used in the example
PendingOrderModify.mq5.
   OrderFilter filter;
   ulong tickets[];
   
   // set a condition for orders for the current symbol and our "magic" number
   filter.let(ORDER_SYMBOL, _Symbol).let(ORDER_MAGIC, Magic);
   // select suitable tickets in an array
   if(filter.select(tickets))
   {
      ArrayPrint(tickets);
   }
By "any versions" here we mean that thanks to the filter class, we can create arbitrary conditions for
selecting orders and changing them "on the go" (for example, at the direction of the user, not the
programmer).
As an example of how to utilize the filter, let's use an Expert Advisor that creates a grid of pending
orders for trading on a rebound from levels within a certain price range, that is, designed for a
fluctuating market. Starting from this section and over the next few, we will modify the Expert Advisor
in the context of the material being studied.
The first version of the Expert Advisor PendingOrderGrid1 .mq5 builds a grid of a given size from limit and
stop-limit orders. The parameters will be the number of price levels and the step in points between
them. The operation scheme is illustrated in the following chart.

---

## Page 1309

Part 6. Trading automation
1 309
6.4 Creating Expert Advisors
Grid of pending orders on 4 levels with a step of 200 points
At a certain initial time, which can be determined by the intraday schedule and can correspond, for
example, to the "night flat", the current price is rounded up to the size of the grid step, and a specified
number of levels is laid up from this level up and down.
At each upper level, we place a limit sell order and a stoplimit buy order with the price of the future
limit order one level lower. At each lower level, we place a limit buy order and a stoplimit sell order with
the price of the future limit order one level higher.
When the price touches one of the levels, the limit order standing there turns into a buy or sell
(position). At the same time, a stop-limit order of the same level is automatically converted by the
system into a limit order of the opposite direction at the next level.
For example, if the price breaks through the level while moving up, we will get a short position, and a
limit order to buy will be created at the step distance below it.
The Expert Advisor will monitor, that at each level there is a stop limit order paired with a limit order.
Therefore, after a new limit buy order is detected, the program will add a stop-limit sell order to it at
the same level, and the target price of the future limit order is the level next to the top, i.e., the one
where the position is opened.
Let's say the price turns down and activates a limit order to the level below – we will get a long
position. At the same time, the stop-limit order is converted into a limit order to sell at the next level
above. Now the Expert Advisor will again detect a "bare" limit order and create a stop-limit order to buy
as a pair to it, at the same level as the price of the future limit order one level lower.  
If there are opposite positions, we will close them. We will also provide for the setting of an intraday
period when the trading system is enabled, and for the rest of the time all orders and positions will be
deleted. This, in particular, is useful for the "night flat", when the return fluctuations of the market are
especially pronounced.

---

## Page 1310

Part 6. Trading automation
1 31 0
6.4 Creating Expert Advisors
Of course, this is just one of many potential implementations of the grid strategy which lacks many of
the customizations of grids, but we won't overcomplicate the example.
The Expert Advisor will analyze the situation on each bar (presumably H1  timeframe or less). In theory,
this Expert Advisor's operation logic needs to be improved by promptly responding to trading events but
we haven't explored them yet. Therefore, instead of constant tracking and instant "manual" restoration
of limit orders at vacant grid levels, we entrusted this work to the server through the use of stop-limit
orders. However, there is a nuance here.
The fact is that limit and stop-limit orders at each level are of opposite types (buy/sell) and therefore
are activated by different types of prices.
It turns out that if the market moved up to the next level in the upper half of the grid, the Ask price
may touch the level and activate a stop-limit buy order, but the Bid price will not reach the level, and
the sell limit order will remain as it is (it will not turn into a position). In the lower half of the grid, when
the market moves down, the situation is mirrored. Any level is first touched by the Bid price, and
activates a stop-limit order to sell, and only with a further decrease the level is also reached by the Ask
price. If there is no move, the buy limit order will remain as is.
This problem becomes critical as the spread increases. Therefore, the Expert Advisor will require
additional control over "extra" limit orders. In other words, the Expert Advisor will not generate a stop-
limit order that is missing at the level if there is already a limit order at its supposed target price
(adjacent level).
The source code is included in the file PendingOrderGrid1 .mq5. In the input parameters, you can set
the Volume of each trade (by default, if left equal to 0, the minimum lot of the chart symbol is taken),
the number of grid levels GridSize (must be even), and the step GridStep between levels in points. The
start and end times of the intraday segment on which the strategy is allowed to work are specified in
the parameters StartTime and StopTime: in both, only time is important.
#include <MQL5Book/MqlTradeSync.mqh>
#include <MQL5Book/OrderFilter.mqh>
#include <MQL5Book/MapArray.mqh>
   
input double Volume;                                       // Volume (0 = minimal lot)
input uint GridSize = 6;                                   // GridSize (even number of price levels)
input uint GridStep = 200;                                 // GridStep (points)
input ENUM_ORDER_TYPE_TIME Expiration = ORDER_TIME_GTC;
input ENUM_ORDER_TYPE_FILLING Filling = ORDER_FILLING_FOK;
input datetime StartTime = D'1970.01.01 00:00:00';         // StartTime (hh:mm:ss)
input datetime StopTime = D'1970.01.01 09:00:00';          // StopTime (hh:mm:ss)
input ulong Magic = 1234567890;
The segment of working time can be either within a day (StartTime < StopTime) or cross the boundary
of the day (StartTime > StopTime), for example, from 22:00 to 09:00. If the two times are equal,
round-the-clock trading is assumed.
Before proceeding with the implementation of the trading idea, let's simplify the task of setting up
queries and outputting diagnostic information to the log. To do this, we describe our own structure
MqlTradeRequestSyncLog, the derivative of MqlTradeRequestSync.

---

## Page 1311

Part 6. Trading automation
1 31 1 
6.4 Creating Expert Advisors
const ulong DAYLONG = 60 * 60 * 24; // length of the day in seconds
   
struct MqlTradeRequestSyncLog: public MqlTradeRequestSync
{
   MqlTradeRequestSyncLog()
   {
      magic = Magic;
      type_filling = Filling;
      type_time = Expiration;
      if(Expiration == ORDER_TIME_SPECIFIED)
      {
         expiration = (datetime)(TimeCurrent() / DAYLONG * DAYLONG
            + StopTime % DAYLONG);
         if(StartTime > StopTime)
         {
            expiration = (datetime)(expiration + DAYLONG);
         }
      }
   }
   ~MqlTradeRequestSyncLog()
   {
      Print(TU::StringOf(this));
      Print(TU::StringOf(this.result));
   }
};
In the constructor, we fill in all fields with unchanged values. In the destructor, we log meaningful query
and result fields. Obviously, the destructor of automatic objects will always be called at the moment of
exit from the code block where the order was formed and sent, that is, the sent and received data will
be printed.
In OnInit let's perform some checks for the correctness of the input variables, in particular, for an even
grid size.
int OnInit()
{
   if(GridSize < 2 || !!(GridSize % 2))
   {
      Alert("GridSize should be 2, 4, 6+ (even number)");
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}
The main entry point of the algorithm is the OnTick handler. In it, for brevity, we will omit the same
error handling mechanism based on TRADE_RETCODE_SEVERITY as in the example
PendingOrderModify.mq5.
For bar-by-bar work, the function has a static variable lastBar, in which we store the time of the last
successfully processed bar. All subsequent ticks on the same bar are skipped.

---

## Page 1312

Part 6. Trading automation
1 31 2
6.4 Creating Expert Advisors
void OnTick()
{
   static datetime lastBar = 0;
   if(iTime(_Symbol, _Period, 0) == lastBar) return;
   uint retcode = 0;
   
   ... // main algorithm (see further)
   
   const TRADE_RETCODE_SEVERITY severity = TradeCodeSeverity(retcode);
   if(severity < SEVERITY_RETRY)
   {
      lastBar = iTime(_Symbol, _Period, 0);
   }
}
Instead of an ellipsis, the main algorithm will follow, divided into several auxiliary functions for
systematization purposes. First of all, let's determine whether the working period of the day is set and,
if so, whether the strategy is currently enabled. This attribute is stored in the tradeScheduled variable.
   ...
   bool tradeScheduled = true;
   
   if(StartTime != StopTime)
   {
      const ulong now = TimeCurrent() % DAYLONG;
      
      if(StartTime < StopTime)
      {
         tradeScheduled = now >= StartTime && now < StopTime;
      }
      else
      {
         tradeScheduled = now >= StartTime || now < StopTime;
      }
   }
   ...
With trading enabled, first, check if there is already a network of orders using the CheckGrid function. If
there is no network, the function will return the GRID_EMPTY constant and we should create the
network by calling Setup Grid. If the network has already been built, it makes sense to check if there
are opposite positions to close: this is done by the CompactPositions function.

---

## Page 1313

Part 6. Trading automation
1 31 3
6.4 Creating Expert Advisors
   if(tradeScheduled)
   {
      retcode = CheckGrid();
      
      if(retcode == GRID_EMPTY)
      {
         retcode = SetupGrid();
      }
      else
      {
         retcode = CompactPositions();
      }
   }
   ...
As soon as the trading period ends, it is necessary to delete orders and close all positions (if any). This
is done, respectively, by the RemoveOrders and CompactPositions function, functions but with a boolean
flag (true): this single, optional argument instructs to apply a simple close for the remaining positions
after the opposite close.
   else
   {
      retcode = CompactPositions(true);
      if(!retcode) retcode = RemoveOrders();
   }
All functions return a server code, which is analyzed for success or failure with TradeCodeSeverity. The
special application codes GRID_EMPTY and GRID_OK are also considered standard according to
TRADE_RETCODE_SEVERITY.
#define GRID_OK    +1
#define GRID_EMPTY  0
Now let's take a look at the functions one by one.
The CheckGrid function uses the OrderFilter class presented at the beginning of this section. The filter
requests all pending orders for the current symbol and with "our" identification number, and the tickets
of found orders are stored in the array.
uint CheckGrid()
{
   OrderFilter filter;
   ulong tickets[];
   
   filter.let(ORDER_SYMBOL, _Symbol).let(ORDER_MAGIC, Magic)
      .let(ORDER_TYPE, ORDER_TYPE_SELL, IS::GREATER)
      .select(tickets);
   const int n = ArraySize(tickets);
   if(!n) return GRID_EMPTY;
   ...
The completeness of the grid is analyzed using the already familiar MapArray class that stores
"key=value" pairs. In this case, the key is the level (price converted into points), and the value is the

---

## Page 1314

Part 6. Trading automation
1 31 4
6.4 Creating Expert Advisors
bitmask (superposition) of order types at the given level. Also, limit and stop-limit orders are counted in
the limits and stops variables, respectively.
   // price levels => masks of types of orders existing there
   MapArray<ulong,uint> levels;
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int limits = 0;
   int stops = 0;
   
   for(int i = 0; i < n; ++i)
   {
      if(OrderSelect(tickets[i]))
      {
         const ulong level = (ulong)MathRound(OrderGetDouble(ORDER_PRICE_OPEN) / point);
         const ulong type = OrderGetInteger(ORDER_TYPE);
         if(type == ORDER_TYPE_BUY_LIMIT || type == ORDER_TYPE_SELL_LIMIT)
         {
            ++limits;
            levels.put(level, levels[level] | (1 << type));
         }
         else if(type == ORDER_TYPE_BUY_STOP_LIMIT 
            || type == ORDER_TYPE_SELL_STOP_LIMIT)
         {
            ++stops;
            levels.put(level, levels[level] | (1 << type));
         }
      }
   }
   ...
If the number of orders of each type matches and is equal to the specified grid size, then everything is
in order.
   if(limits == stops)
   {
      if(limits == GridSize) return GRID_OK; // complete grid
      
      Alert("Error: Order number does not match requested");
      return TRADE_RETCODE_ERROR;
   }
   ...
The situation when the number of limit orders is greater than the stop limit ones is normal: it means
that due to the price movement, one or more stop limit orders have turned into limit ones. The program
should then add stop-limit orders to the levels where there are not enough of them. A separate order of
a specific type for a specific level can be placed by the RepairGridLevel function.

---

## Page 1315

Part 6. Trading automation
1 31 5
6.4 Creating Expert Advisors
   if(limits > stops)
   {
      const uint stopmask = 
         (1 << ORDER_TYPE_BUY_STOP_LIMIT) | (1 << ORDER_TYPE_SELL_STOP_LIMIT);
      for(int i = 0; i < levels.getSize(); ++i)
      {
         if((levels[i] & stopmask) == 0) // there is no stop-limit order at this level
         {
            // the direction of the limit is required to set the reverse stop limit
            const bool buyLimit = (levels[i] & (1 << ORDER_TYPE_BUY_LIMIT));
            // checks for "extra" orders due to the spread are omitted here (see the source code)
            ...
            // create a stop-limit order in the desired direction
            const uint retcode = RepairGridLevel(levels.getKey(i), point, buyLimit);
            if(TradeCodeSeverity(retcode) > SEVERITY_NORMAL)
            {
               return retcode;
            }
         }
      }
      return GRID_OK;
   }
   ...
The situation when the number of stop-limit orders is greater than the limit ones is treated as an error
(probably the server skipped the price for some reason).
   Alert("Error: Orphaned Stop-Limit orders found");
   return TRADE_RETCODE_ERROR;
}
The function RepairGridLevel performs the following actions.

---

## Page 1316

Part 6. Trading automation
1 31 6
6.4 Creating Expert Advisors
uint RepairGridLevel(const ulong level, const double point, const bool buyLimit)
{
   const double price = level * point;
   const double volume = Volume == 0 ?
      SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) : Volume;
   
   MqlTradeRequestSyncLog request;
   
   request.comment = "repair";
   
   // if there is an unpaired buy-limit, set the sell-stop-limit to it
   // if there is an unpaired sell-limit, set buy-stop-limit to it
   const ulong order = (buyLimit ?
      request.sellStopLimit(volume, price, price + GridStep * point) :
      request.buyStopLimit(volume, price, price - GridStep * point));
   const bool result = (order != 0) && request.completed();
   if(!result) Alert("RepairGridLevel failed");
   return request.result.retcode;
}
Please note that we do not need to actually fill in the structure (except for a comment that can be
made more informative if necessary) since some of the fields are filled in automatically by the
constructor, and we pass the volume and price directly to the sellStopLimit or buyStopLimit method.
A similar approach is used in the SetupGrid function, which creates a new full network of orders. At the
beginning of the function, we prepare variables for calculations and describe the
MqlTradeRequestSyncLog array of structures.
uint SetupGrid()
{
   const double current = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   const double volume = Volume == 0 ?
      SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) : Volume;
   // central price of the range rounded to the nearest step,
   // from it up and down we identify the levels  
   const double base = ((ulong)MathRound(current / point / GridStep) * GridStep)
      * point;
   const string comment = "G[" + DoubleToString(base,
      (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) + "]";
   const static string message = "SetupGrid failed: ";
   MqlTradeRequestSyncLog request[][2]; // limit and stop-limit - one pair
   ArrayResize(request, GridSize);      // 2 pending orders per level
Next, we generate orders for the lower and upper half of the grid, diverging from the center to the
sides.

---

## Page 1317

Part 6. Trading automation
1 31 7
6.4 Creating Expert Advisors
   for(int i = 0; i < (int)GridSize / 2; ++i)
   {
      const int k = i + 1;
      
      // bottom half of the grid
      request[i][0].comment = comment;
      request[i][1].comment = comment;
      
      if(!(request[i][0].buyLimit(volume, base - k * GridStep * point)))
      {
         Alert(message + (string)i + "/BL");
         return request[i][0].result.retcode;
      }
      if(!(request[i][1].sellStopLimit(volume, base - k * GridStep * point,
         base - (k - 1) * GridStep * point)))
      {
         Alert(message + (string)i + "/SSL");
         return request[i][1].result.retcode;
      }
      
      // top half of the grid
      const int m = i + (int)GridSize / 2;
      
      request[m][0].comment = comment;
      request[m][1].comment = comment;
      
      if(!(request[m][0].sellLimit(volume, base + k * GridStep * point)))
      {
         Alert(message + (string)m + "/SL");
         return request[m][0].result.retcode;
      }
      if(!(request[m][1].buyStopLimit(volume, base + k * GridStep * point,
         base + (k - 1) * GridStep * point)))
      {
         Alert(message + (string)m + "/BSL");
         return request[m][1].result.retcode;
      }
   }
Then we check for readiness.

---

## Page 1318

Part 6. Trading automation
1 31 8
6.4 Creating Expert Advisors
   for(int i = 0; i < (int)GridSize; ++i)
   {
      for(int j = 0; j < 2; ++j)
      {
         if(!request[i][j].completed())
         {
            Alert(message + (string)i + "/" + (string)j + " post-check");
            return request[i][j].result.retcode;
         }
      }
   }
   return GRID_OK;
}
Although the check (call of completed) is spaced out with sending orders, our structure still uses the
synchronous form OrderSend internally. In fact, to speed up sending a batch of orders (as in our grid
Expert Advisor), it is better to use the asynchronous version OrderSendAsync. But then the order
execution status should be initiated from the event handler OnTradeTransaction. We will study this one
later.
An error when sending any order leads to an early exit from the loop and the return of the code from
the server. This testing Expert Advisor will simply stop its further work in case of an error. For a real
robot, it is desirable to provide an intellectual analysis of the meaning of the error and, if necessary,
delete all orders and close positions.
Positions that will be generated by pending orders are closed by the function CompactPositions.
uint CompactPositions(const bool cleanup = false)
The cleanup parameter equal to false by default means regular "cleaning" of positions within the trading
period, i.e., the closing of opposite positions (if any). Value cleanup=true is used to force close all
positions at the end of the trading period.
The function fills the ticketsLong and ticketsShort arrays with tickets of long and short positions using a
helper function GetMyPositions. We have already used the latter in the example TradeCloseBy.mq5 in
the section Closing opposite positions: full and partial. The CloseByPosition function from that example
has undergone minimal changes in the new Expert Advisor: it returns a code from the server instead of
a logical indicator of success or error.

---

## Page 1319

Part 6. Trading automation
1 31 9
6.4 Creating Expert Advisors
uint CompactPositions(const bool cleanup = false)
{
   uint retcode = 0;
   ulong ticketsLong[], ticketsShort[];
   const int n = GetMyPositions(_Symbol, Magic, ticketsLong, ticketsShort);
   if(n > 0)
   {
      Print("CompactPositions, pairs: ", n);
      for(int i = 0; i < n; ++i)
      {
         retcode = CloseByPosition(ticketsShort[i], ticketsLong[i]);
         if(retcode) return retcode;
      }
   }
   ...
The second part of CompactPositions only works when cleanup=true. It is far from perfect and will be
rewritten soon.
   if(cleanup)
   {
      if(ArraySize(ticketsLong) > ArraySize(ticketsShort))
      {
         retcode = CloseAllPositions(ticketsLong, ArraySize(ticketsShort));
      }
      else if(ArraySize(ticketsLong) < ArraySize(ticketsShort))
      {
         retcode = CloseAllPositions(ticketsShort, ArraySize(ticketsLong));
      }
   }
   
   return retcode;
}
For all found remaining positions, the usual closure is performed by calling CloseAllPositions.

---

## Page 1320

Part 6. Trading automation
1 320
6.4 Creating Expert Advisors
uint CloseAllPositions(const ulong &tickets[], const int start = 0)
{
   const int n = ArraySize(tickets);
   Print("CloseAllPositions ", n);
   for(int i = start; i < n; ++i)
   {
      MqlTradeRequestSyncLog request;
      request.comment = "close down " + (string)(i + 1 - start)
         + " of " + (string)(n - start);
      if(!(request.close(tickets[i]) && request.completed()))
      {
         Print("Error: position is not closed ", tickets[i]);
         return request.result.retcode;
      }
   }
   return 0; // success
}
Now we only need to consider the RemoveOrders function. It also uses the order filter to get a list of
them, and then calls the remove method in a loop.
uint RemoveOrders()
{
   OrderFilter filter;
   ulong tickets[];
   filter.let(ORDER_SYMBOL, _Symbol).let(ORDER_MAGIC, Magic)
      .select(tickets);
   const int n = ArraySize(tickets);
   for(int i = 0; i < n; ++i)
   {
      MqlTradeRequestSyncLog request;
      request.comment = "removal " + (string)(i + 1) + " of " + (string)n;
      if(!(request.remove(tickets[i]) && request.completed()))
      {
         Print("Error: order is not removed ", tickets[i]);
         return request.result.retcode;
      }
   }
   return 0;
}
Let's check how the Expert Advisor works in the tester with default settings (trading period from 00:00
to 09:00). Below is a screenshot for launching on EURUSD, H1 .

---

## Page 1321

Part 6. Trading automation
1 321 
6.4 Creating Expert Advisors
Grid strategy PendingOrderGrid1.mq5 in the tester
In the log, in addition to periodic entries about the batch creation of several orders (at the beginning of
the day) and their removal in the morning, we will regularly see the restoration of the network (adding
orders instead of triggered ones) and closing positions.
buy stop limit 0.01 EURUSD at 1.14200 (1.14000) (1.13923 / 1.13923)
TRADE_ACTION_PENDING, EURUSD, ORDER_TYPE_BUY_STOP_LIMIT, V=0.01, ORDER_FILLING_FOK, »
   » @ 1.14200, X=1.14000, ORDER_TIME_GTC, M=1234567890, repair
DONE, #=159, V=0.01, Bid=1.13923, Ask=1.13923, Request executed, Req=287
CompactPositions, pairs: 1
close position #152 sell 0.01 EURUSD by position #153 buy 0.01 EURUSD (1.13923 / 1.13923)
deal #18 buy 0.01 EURUSD at 1.13996 done (based on order #160)
deal #19 sell 0.01 EURUSD at 1.14202 done (based on order #160)
Positions collapse initiated
OK CloseBy Order/Deal/Position
TRADE_ACTION_CLOSE_BY, EURUSD, ORDER_TYPE_BUY, ORDER_FILLING_FOK, P=152, b=153, »
   » M=1234567890, compacting
DONE, D=18, #=160, Request executed, Req=288
Now it's time to study the MQL5 functions for working with positions and improve their selection and
analysis in our Expert Advisor. The following sections deal with this.
6.4.26 Getting the list of positions
In many examples of Expert Advisors, we have already used the MQL5 API functions designed to
analyze open trading positions. This section presents their formal description.

---

## Page 1322

Part 6. Trading automation
1 322
6.4 Creating Expert Advisors
It is important to note that the functions of this group are not able to create, modify, or delete
positions. As we saw earlier, all such actions are performed indirectly through the sending of orders. If
they are successfully executed, transactions are made, as a result of which positions are formed.
Another feature is that the functions are only applicable to online positions. To restore the history of
positions, it is necessary to analyze the history of trades.
The PositionsTotal function allows you to find out the total number of open positions on the account (for
all financial instruments).
int PositionsTotal()
With netting accounting of positions (ACCOUNT_MARGIN_MODE_RETAIL_NETTING and
ACCOUNT_MARGIN_MODE_EXCHANGE), there can be only one position for each symbol at any time.
This position can result from one or more deals.
With the independent representation of positions (ACCOUNT_MARGIN_MODE_RETAIL_HEDGING),
several positions can be opened simultaneously for each symbol, including multidirectional ones. Each
market entry trade creates a separate position, so a partial step-by-step execution of one order can
generate several positions.
The PositionGetSymbol function returns the symbol of a position by its number.
string PositionGetSymbol(int index)
The index must be between 0 and N-1 , where N is the value received by the pre-call of PositionsTotal.
The order of positions is not regulated.
If the position is not found, then an empty string will be returned, and the error code will be available in
_ LastError.
Examples of using these two functions were provided in several test Experts Advisors (TrailingStop.mq5,
TradeCloseBy.mq5, and others) in functions with names GetMyPosition/GetMyPositions.
An open position is characterized by a unique ticket which is the number that distinguishes it from
other positions, but may change during its life in some cases, such as a position reversal in netting
mode by one trade, or as a result of service operations on the server (reopening for swap accrual,
clearing).
To get a position ticket by its number, we use the PositionGetTicket function.
ulong PositionGetTicket(int index)
Additionally, the function highlights a position in the trading environment of the terminal, which then
allows you to read its properties using a group of special PositionGet functions. In other words, by
analogy with orders, the terminal maintains an internal cache for each MQL program to store the
properties of one position. To highlight a position, in addition to PositionGetTicket, there are two
functions: PositionSelect and PositionSelectByTicket which we will discuss below.
In case of an error, the PositionGetTicket function will return 0.
The ticket should not be confused with the identifier that is assigned to each position and never
changes. It is the identifiers that are used to link positions with orders and deals. We will talk about
this a little later. 
Tickets are needed to fulfill requests involving positions: the tickets are specified in the position and
position_ by fields of the MqlTradeRequest structure. Besides, by saving the ticket in a variable, the

---

## Page 1323

Part 6. Trading automation
1 323
6.4 Creating Expert Advisors
program can subsequently select a specific position using the PositionSelectByTicket function (see
below) and work with it without resorting to repeated enumeration of positions in the loop.
When a position is reversed on a netting account, POSITION_TICKET is changed to the ticket of the
order that initiated this operation. However, such a position can still be tracked using an ID. Position
reversal is not supported in hedging mode.
bool PositionSelect(const string symbol)
The function selects an open position by the name of the financial instrument.
With the independent representation of positions (ACCOUNT_MARGIN_MODE_RETAIL_HEDGING), there
can be several open positions for each symbol at the same time. In this case, PositionSelect will select
the position with the smallest ticket.
The returned result signals a successful (true) or unsuccessful (false) function execution.
The fact that the properties of the selected position are cached means that the position itself may no
longer exist, or it may be changed if the program reads its properties after some time. It is
recommended to call the PositionSelect function just before accessing the data.
bool PositionSelectByTicket(ulong ticket)
The function selects an open position for further work on the specified ticket.
We will look at examples of using functions later when studying properties and related PositionGet
functions.
When constructing algorithms using the PositionsTotal, OrdersTotal, and similar functions, the
asynchronous principles of the terminal operation should be taken into account. We have already
touched on this topic when writing the MqlTradeSync.mqh classes and implementing waiting for the
execution results from trade requests. However, this wait is not always possible on the client side. In
particular, if we place a pending order, then its transformation into a market order and subsequent
execution will take place on the server. At this moment, the order may cease to be listed among the
active ones (OrdersTotal will return 0), but the position is not displayed yet (PositionsTotal also equals
0). Therefore, an MQL program that has a condition for placing an order in the absence of a position
may erroneously initiate a new order, as a result of which the position will eventually double.
To solve this problem, an MQL program must analyze the trading environment more deeply than just
checking the number of orders and positions at once. For example, you can keep a snapshot of the last
correct state of the trading environment and not allow any entities to disappear without some kind of
confirmation. Only then can a new cast be formed. Thus, an order can be deleted only together with a
position change (creation, closing) or moved to history with a cancel status. One of the possible
solutions is proposed in the form of the TradeGuard class in the TradeGuard.mqh file. The book also
includes the demo script TradeGuardExample.mq5 which you can study additionally.
6.4.27 Position properties
All position properties are divided into three groups according to the type of values: integer and
compatible with them, real numbers, and strings. They are used to read PositionGet functions similar to
OrderGet functions. We will describe the functions themselves in the next section, and here we will give
the identifiers of all properties that are available for specifying in the first parameter of these functions.
Integer properties are provided in the ENUM_POSITION_PROPERTY_INTEGER enumeration.

---

## Page 1324

Part 6. Trading automation
1 324
6.4 Creating Expert Advisors
Identifier
Description
Type
P O S ITIO N _TICK E T
Position ticket
ulong
P O S ITIO N _TIM E 
Position opening time
datetime
P O S ITIO N _TIM E _M S C
Position opening time in
milliseconds
ulong
P O S ITIO N _TIM E _U P D ATE 
Position change (volume) time 
datetime
P O S ITIO N _TIM E _U P D ATE _M S C
Position change (volume) time in
milliseconds
ulong
P O S ITIO N _TYP E 
Position type
ENUM_POSITION_TYPE
P O S ITIO N _M AG IC
Position Magic number (based on
ORDER_MAGIC)
ulong
P O S ITIO N _ID E N TIF IE R 
Position identifier; a unique
number that is assigned to each
newly opened position and does
not change during its entire life.
ulong
P O S ITIO N _R E AS O N 
Reason for opening a position
ENUM_POSITION_REASON
As a rule, POSITION_IDENTIFIER corresponds to the ticket of the order that opened the position. The
position identifier is indicated in each order (ORDER_POSITION_ID) and deal (DEAL_POSITION_ID) that
opened, changed, or closed it. Therefore, it is convenient to use it to search for orders and deals
related to a position.
If the order is filled partially, then both the position and the active pending order for the remaining
volume with matching tickets can exist simultaneously. Moreover, such a position can be closed in time,
and at the next filling of the rest of the pending order, a position with the same ticket will appear again.
In netting mode, reversing a position with one trade is considered a position change, not a new one, so
the POSITION_IDENTIFIER is preserved. A new position on a symbol is possible only after closing the
previous one in zero volume.
The POSITION_TIME_UPDATE property only responds to volume changes (for example, as a result of
partial closing or position increase), but not other parameters like Stop Loss/Take Profit levels or swap
charges.
There are only two types of positions (ENUM_POSITION_TYPE).
Identifier
Description
POSITION_TYPE_BUY
Buy
POSITION_TYPE_SELL
Sell
Options for the origin of a position, that is, how the position was opened, are provided in the
ENUM_POSITION_REASON enumeration.

---

## Page 1325

Part 6. Trading automation
1 325
6.4 Creating Expert Advisors
Identifier
Description
POSITION_REASON_CLIENT
Triggering of an order placed from the desktop terminal
POSITION_REASON_MOBILE
Triggering of an order placed from a mobile application
POSITION_REASON_WEB
Triggering of an order placed from the web platform
(browser)
POSITION_REASON_EXPERT
Triggering of an order placed by an Expert Advisor or a script
Real properties are collected in ENUM_POSITION_PROPERTY_DOUBLE.
Identifier
Description
POSITION_VOLUME
Position volume
POSITION_PRICE_OPEN
Position price
POSITION_SL
Stop Loss Price
POSITION_TP
Take profit price
POSITION_PRICE_CURRENT
Current symbol price
POSITION_SWAP
Accumulated swap
POSITION_PROFIT
Current profit
The current price type corresponds to the position closing operation. For example, a long position must
be closed by selling, and therefore the Bid price for it is tracked in POSITION_PRICE_CURRENT.
Finally, the following string properties (ENUM_POSITION_PROPERTY_STRING) are supported for
positions.
Identifier
Description
POSITION_SYMBOL
The symbol on which the position is opened
POSITION_COMMENT
Position comment
POSITION_EXTERNAL_ID
Position ID in the external system (on the exchange)
After reviewing the list of position properties, we are ready to look at the functions for reading these
properties.
6.4.28 Functions for reading position properties
An MQL program can get position properties using several PositionGet functions depending on the type
of properties. In all functions, the specific property being requested is defined in the first parameter,
which takes the ID of one of the ENUM_POSITION_PROPERTY enumerations discussed in the previous
section.

---

## Page 1326

Part 6. Trading automation
1 326
6.4 Creating Expert Advisors
For each type of property, there is a short and long form of the function: the first returns the value of
the property directly, and the second writes it into the second parameter, passed by reference.
Integer properties and properties of compatible types (datetime, enumerations) can be obtained by the
PositionGetInteger function.
long PositionGetInteger(ENUM_POSITION_PROPERTY_INTEGER property)
bool PositionGetInteger(ENUM_POSITION_PROPERTY_INTEGER property, long &value)
If it fails, the function returns 0 or false.
The PositionGetDouble function is used to obtain real properties.
double PositionGetDouble(ENUM_POSITION_PROPERTY_DOUBLE property)
bool PositionGetDouble(ENUM_POSITION_PROPERTY_DOUBLE property, double &value)
Finally, the string properties are returned by the PositionGetString function.
string PositionGetString(ENUM_POSITION_PROPERTY_STRING property)
bool PositionGetString(ENUM_POSITION_PROPERTY_STRING property, string &value)
In case of failure, the first form of the function returns an empty string.
To read position properties, we already have a ready abstract interface MonitorInterface
(TradeBaseMonitor.mqh) which we used to write an order monitor. Now it will be easy to implement a
similar monitor for positions. The result is attached in the file PositionMonitor.mqh.
The PositionMonitorInterface class is inherited from MonitorInterface with assignment to the template
types I, D, and S of the considered ENUM_POSITION_PROPERTY enumerations, and overrides a couple
of stringify methods taking into account the specifics of position properties.

---

## Page 1327

Part 6. Trading automation
1 327
6.4 Creating Expert Advisors
class PositionMonitorInterface:
   public MonitorInterface<ENUM_POSITION_PROPERTY_INTEGER,
   ENUM_POSITION_PROPERTY_DOUBLE,ENUM_POSITION_PROPERTY_STRING>
{
public:
   virtual string stringify(const long v,
      const ENUM_POSITION_PROPERTY_INTEGER property) const override
   {
      switch(property)
      {
         case POSITION_TYPE:
            return enumstr<ENUM_POSITION_TYPE>(v);
         case POSITION_REASON:
            return enumstr<ENUM_POSITION_REASON>(v);
         
         case POSITION_TIME:
         case POSITION_TIME_UPDATE:
            return TimeToString(v, TIME_DATE | TIME_SECONDS);
         
         case POSITION_TIME_MSC:
         case POSITION_TIME_UPDATE_MSC:
            return STR_TIME_MSC(v);
      }
      
      return (string)v;
   }
   
   virtual string stringify(const ENUM_POSITION_PROPERTY_DOUBLE property,
      const string format = NULL) const override
   {
      if(format == NULL &&
         (property == POSITION_PRICE_OPEN || property == POSITION_PRICE_CURRENT
         || property == POSITION_SL || property == POSITION_TP))
      {
         const int digits = (int)SymbolInfoInteger(PositionGetString(POSITION_SYMBOL),
            SYMBOL_DIGITS);
         return DoubleToString(PositionGetDouble(property), digits);
      }
      return MonitorInterface<ENUM_POSITION_PROPERTY_INTEGER,
         ENUM_POSITION_PROPERTY_DOUBLE,ENUM_POSITION_PROPERTY_STRING>
         ::stringify(property, format);
   }
The specific monitor class, ready to view positions, is next in the inheritance chain and is based on
PositionGet functions. Selecting a position by ticket is done in the constructor.

---

## Page 1328

Part 6. Trading automation
1 328
6.4 Creating Expert Advisors
class PositionMonitor: public PositionMonitorInterface
{
public:
   const ulong ticket;
   PositionMonitor(const ulong t): ticket(t)
   {
      if(!PositionSelectByTicket(ticket))
      {
         PrintFormat("Error: PositionSelectByTicket(%lld) failed: %s",
            ticket, E2S(_LastError));
      }
      else
      {
         ready = true;
      }
   }
   
   virtual long get(const ENUM_POSITION_PROPERTY_INTEGER property) const override
   {
      return PositionGetInteger(property);
   }
   
   virtual double get(const ENUM_POSITION_PROPERTY_DOUBLE property) const override
   {
      return PositionGetDouble(property);
   }
   
   virtual string get(const ENUM_POSITION_PROPERTY_STRING property) const override
   {
      return PositionGetString(property);
   }
   ...
};
A simple script will allow you to log all the characteristics of the first position (if at least one is
available).
void OnStart()
{
   PositionMonitor pm(PositionGetTicket(0));
   pm.print();
}
In the log, we should get something like this.

---

## Page 1329

Part 6. Trading automation
1 329
6.4 Creating Expert Advisors
MonitorInterface<ENUM_POSITION_PROPERTY_INTEGER, »
   » ENUM_POSITION_PROPERTY_DOUBLE,ENUM_POSITION_PROPERTY_STRING>
ENUM_POSITION_PROPERTY_INTEGER Count=9
  0 POSITION_TIME=2022.03.24 23:09:45
  1 POSITION_TYPE=POSITION_TYPE_BUY
  2 POSITION_MAGIC=0
  3 POSITION_IDENTIFIER=1291755067
  4 POSITION_TIME_MSC=2022.03.24 23:09:45'261
  5 POSITION_TIME_UPDATE=2022.03.24 23:09:45
  6 POSITION_TIME_UPDATE_MSC=2022.03.24 23:09:45'261
  7 POSITION_TICKET=1291755067
  8 POSITION_REASON=POSITION_REASON_EXPERT
ENUM_POSITION_PROPERTY_DOUBLE Count=8
  0 POSITION_VOLUME=0.01
  1 POSITION_PRICE_OPEN=1.09977
  2 POSITION_PRICE_CURRENT=1.09965
  3 POSITION_SL=0.00000
  4 POSITION_TP=1.10500
  5 POSITION_COMMISSION=0.0
  6 POSITION_SWAP=0.0
  7 POSITION_PROFIT=-0.12
ENUM_POSITION_PROPERTY_STRING Count=3
  0 POSITION_SYMBOL=EURUSD
  1 POSITION_COMMENT=
  2 POSITION_EXTERNAL_ID=
If there are no open positions at the moment, we will see an error message.
Error: PositionSelectByTicket(0) failed: TRADE_POSITION_NOT_FOUND
However, the monitor is useful not only and not so much by outputting properties to the log. Based on
PositionMonitor, we create a class for selecting positions by conditions, similar to what we did for
orders (OrderFilter). The ultimate goal is to improve our grid Expert Advisor.
Thanks to OOP, creating a new filter class is almost effortless. Below is the complete source code (file
PositionFilter.mqh).

---

## Page 1330

Part 6. Trading automation
1 330
6.4 Creating Expert Advisors
class PositionFilter: public TradeFilter<PositionMonitor,
   ENUM_POSITION_PROPERTY_INTEGER,
   ENUM_POSITION_PROPERTY_DOUBLE,
   ENUM_POSITION_PROPERTY_STRING>
{
protected:
   virtual int total() const override
   {
      return PositionsTotal();
   }
   virtual ulong get(const int i) const override
   {
      return PositionGetTicket(i);
   }
};
Now we can write such a script for receiving specific profit on positions with the given magic number,
for example.
input ulong Magic;
   
void OnStart()
{
   PositionFilter filter;
   
   ENUM_POSITION_PROPERTY_DOUBLE properties[] =
      {POSITION_PROFIT, POSITION_VOLUME};
   
   double profits[][2];
   ulong tickets[];
   string symbols[];
   
   filter.let(POSITION_MAGIC, Magic).select(properties, tickets, profits);
   filter.select(POSITION_SYMBOL, tickets, symbols);
   
   for(int i = 0; i < ArraySize(symbols); ++i)
   {
      PrintFormat("%s[%lld]=%f",
         symbols[i], tickets[i], profits[i][0] / profits[i][1]);
   }
}
In this case, we had to call the select method twice, because the types of properties we are interested
in are different: real profit and lot, but the string name of the instrument. In one of the sections at the
beginning of the chapter, when we were developing the filter class for symbols, we described the
concept of tuples. In MQL5, we can implement it as structure templates with fields of arbitrary types.
Such tuples would come in very handy for finalizing the hierarchy of filter classes since then it would be
possible to describe the select method that fills an array of tuples with fields of any type.
The tuples are described in the file Tuples.mqh. All structures in it have a name TupleN<T1 ,...>, where N
is a number from 2 to 8, and it corresponds to the number of template parameters (Ti types). For
example, Tuple2:

---

## Page 1331

Part 6. Trading automation
1 331 
6.4 Creating Expert Advisors
template<typename T1,typename T2>
struct Tuple2
{
   T1 _1;
   T2 _2;
   
   static int size() { return 2; };
   
   // M – order, position, deal monitor class, any MonitorInterface<>
   template<typename M>
   void assign(const int &properties[], M &m)
   {
      if(ArraySize(properties) != size()) return;
      _1 = m.get(properties[0], _1);
      _2 = m.get(properties[1], _2);
   }
};
In the class TradeFilter (TradeFilter.mqh) let's add a version of the function select with tuples.

---

## Page 1332

Part 6. Trading automation
1 332
6.4 Creating Expert Advisors
template<typename T,typename I,typename D,typename S>
class TradeFilter
{
   ...
 template<typename U> // type U must be Tuple<>, e.g. Tuple3<T1,T2,T3>
   bool select(const int &property[], U &data[], const bool sort = false) const
   {
      const int q = ArraySize(property);
      static const U u;                 // PRB: U::size() does not compile
      if(q != u.size()) return false;   // required condition
      
      const int n = total();
      // cycle through orders/positions/deals
      for(int i = 0; i < n; ++i)
      {
         const ulong t = get(i);
         // access to properties via monitor T
         T m(t);
         // check all filter conditions for different types of properties
         if(match(m, longs)
         && match(m, doubles)
         && match(m, strings))
         {
            // for a suitable object, store the properties in an array of tuples
            const int k = EXPAND(data);
            data[k].assign(property, m);
         }
      }
      
      if(sort)
      {
         sortTuple(data, u._1);
      }
      
      return true;
   }
An array of tuples can optionally be sorted by the first field _1 , so you can additionally study the
sortTuple helper method.
With tuples, you can query a filter object for properties of three different types in one select call.
Below there are positions with some Magic number displayed, sorted by profit; for each a symbol and a
ticket are additionally obtained.

---

## Page 1333

Part 6. Trading automation
1 333
6.4 Creating Expert Advisors
 input ulong Magic;
   
   void OnStart()
   {
      int props[] = {POSITION_PROFIT, POSITION_SYMBOL, POSITION_TICKET};
      Tuple3<double,string,ulong> tuples[];
      PositionFilter filter;
      filter.let(POSITION_MAGIC, Magic).select(props, tuples, true);
      ArrayPrint(tuples);
   }
Of course, the parameter types in the description of the array of tuples (in this case,
Tuple3<double,string,ulong>) must match the requested property enumeration types
(POSITION_PROFIT, POSITION_SYMBOL, POSITION_TICKET).
Now we can slightly simplify the grid Expert Advisor (meaning not just a shorter, but also a more
understandable code). The new version is called PendingOrderGrid2.mq5. The changes will affect all
functions related to position management.
The GetMyPositions function populates the types4tickets array of tuples passed by reference. In each
Tuple2 tuple, it is supposed to store the type and ticket of the position. In this particular case, we could
manage just with a two-dimensional array ulong instead of tuples because both properties are of the
same base type. However, we use tuples to demonstrate how to work with them in the calling code.
#include <MQL5Book/Tuples.mqh>
#include <MQL5Book/PositionFilter.mqh>
   
int GetMyPositions(const string s, const ulong m,
   Tuple2<ulong,ulong> &types4tickets[])
{
   int props[] = {POSITION_TYPE, POSITION_TICKET};
   PositionFilter filter;
   filter.let(POSITION_SYMBOL, s).let(POSITION_MAGIC, m)
      .select(props, types4tickets, true);
   return ArraySize(types4tickets);
}
Note that the last, third parameter of the select method equals true, which instructs to sort the array
by the first field, i.e., the type of positions. Thus, we will have purchases at the beginning, and sales at
the end. This will be required for the counter closure.
The reincarnation of the CompactPositions method is as follows.

---

## Page 1334

Part 6. Trading automation
1 334
6.4 Creating Expert Advisors
uint CompactPositions(const bool cleanup = false)
{
   uint retcode = 0;
   Tuple2<ulong,ulong> types4tickets[];
   int i = 0, j = 0;
   int n = GetMyPositions(_Symbol, Magic, types4tickets);
   if(n > 0)
   {
      Print("CompactPositions: ", n);
      for(i = 0, j = n - 1; i < j; ++i, --j)
      {
         if(types4tickets[i]._1 != types4tickets[j]._1) // as long as the types are different
         {
            retcode = CloseByPosition(types4tickets[i]._2, types4tickets[j]._2);
            if(retcode) return retcode; // error
         }
         else
         {
            break;
         }
      }
   }
   
   if(cleanup && j < n)
   {
      retcode = CloseAllPositions(types4tickets, i, j + 1);
   }
   
   return retcode;
}
The CloseAllPositions function is almost the same:

---

## Page 1335

Part 6. Trading automation
1 335
6.4 Creating Expert Advisors
uint CloseAllPositions(const Tuple2<ulong,ulong> &types4tickets[],
   const int start = 0, const int end = 0)
{
   const int n = end == 0 ? ArraySize(types4tickets) : end;
   Print("CloseAllPositions ", n - start);
   for(int i = start; i < n; ++i)
   {
      MqlTradeRequestSyncLog request;
      request.comment = "close down " + (string)(i + 1 - start)
         + " of " + (string)(n - start);
      const ulong ticket = types4tickets[i]._2;
      if(!(request.close(ticket) && request.completed()))
      {
         Print("Error: position is not closed ", ticket);
         return request.result.retcode; // error
      }
   }
   return 0; // success 
}
You can compare the work of Expert Advisors PendingOrderGrid1 .mq5 and PendingOrderGrid2.mq5 in
the tester.
The reports will be slightly different, because if there are several positions, they are closed in opposite
combinations, due to which the closing of other, unpaired positions takes place with respect to their
individual spreads.
6.4.29 Deal properties
A deal is a reflection of the fact that a trading operation was performed on the basis of an order. One
order can generate several deals due to the execution in parts or the opposite closing of positions.
Deals are characterized by properties of three basic types: integer (and compatible with them), real,
and string. Each property is described by its own constant in one of the enumerations:
E N U M _D E AL _PR OPE R TY_IN TE GE R , E N U M _D E AL _PR OPE R TY_D OU B L E , E N U M _D E AL _PR OPE R TY_STR IN G.
To read deal properties, use the HistoryDealGet functions. All of them assume that the necessary
section of history was previously requested using special functions for the selection of orders and deals
from history.
Integer properties are described in the ENUM_DEAL_PROPERTY_INTEGER enumeration.

---

## Page 1336

Part 6. Trading automation
1 336
6.4 Creating Expert Advisors
Identifier
Description
Type
DEAL_TICKET
Deal ticket; a unique number that is
assigned to each transaction
ulong
DEAL_ORDER
The ticket of the order on the basis
of which the deal was executed
ulong
DEAL_TIME
Deal time
datetime
DEAL_TIME_MSC
Deal time in milliseconds
ulong
DEAL_TYPE
Deal type
ENUM_DEAL_TYPE (see
below)
DEAL_ENTRY
Deal direction; market entry,
market exit, or reversal
ENUM_DEAL_ENTRY (see
below)
DEAL_MAGIC
Magic number for the deal (based
on ORDER_MAGIC)
ulong
DEAL_REASON
Deal reason or source
ENUM_DEAL_REASON (see
below)
DEAL_POSITION_ID
Identifier of the position that was
opened, modified or closed by the
deal
ulong
Possible deal types are represented by the ENUM_DEAL_TYPE enumeration.
Identifier
Description
DEAL_TYPE_BUY
Buy
DEAL_TYPE_SELL
Sell
DEAL_TYPE_BALANCE
Balance accrued
DEAL_TYPE_CREDIT
Credit accrual
DEAL_TYPE_CHARGE
Additional charges
DEAL_TYPE_CORRECTION
Correction
DEAL_TYPE_BONUS
Bonuses
DEAL_TYPE_COMMISSION
Additional commission
DEAL_TYPE_COMMISSION_DAILY
Commission charged at the end of the trading day
DEAL_TYPE_COMMISSION_MONTHLY
Commission charged at the end of the month
DEAL_TYPE_COMMISSION_AGENT_DAILY
Agent commission charged at the end of the
trading day
DEAL_TYPE_COMMISSION_AGENT_MONTHLY
Agent commission charged at the end of the month

---

## Page 1337

Part 6. Trading automation
1 337
6.4 Creating Expert Advisors
Identifier
Description
DEAL_TYPE_INTEREST
Interest accrual on free funds
DEAL_TYPE_BUY_CANCELED
Canceled buy deal
DEAL_TYPE_SELL_CANCELED
Canceled sell deal
DEAL_DIVIDEND
Dividend accrual
DEAL_DIVIDEND_FRANKED
Accrual of a franked dividend (tax exempt)
DEAL_TAX
Tax accrual
The DEAL_TYPE_BUY_CANCELED and DEAL_TYPE_SELL_CANCELED options reflect the situation when
an earlier deal is canceled. In this case, the type of the previously executed deal (DEAL_TYPE_BUY or
DEAL_TYPE_SELL) is changed to DEAL_TYPE_BUY_CANCELED or DEAL_TYPE_SELL_CANCELED, and its
profit/loss is reset to zero. Previously received profit/loss is credited/debited from the account as a
separate balance operation.
Deals differ in the way the position is changed. This can be a simple opening of a position (entry to the
market), increasing the volume of a previously opened position, closing a position with a deal in the
opposite direction or position reversal when the opposite deal covers the volume of a previously opened
position. The latter operation is only supported on netting accounts.
All these situations are described by the elements of the ENUM_DEAL_ENTRY enumeration.
Identifier
Description
DEAL_ENTRY_IN
Market entry
DEAL_ENTRY_OUT
Market exit
DEAL_ENTRY_INOUT
Reversal
DEAL_ENTRY_OUT_BY
Closing by an opposite position
The reasons for the deal are summarized in the ENUM_DEAL_REASON enumeration.

---

## Page 1338

Part 6. Trading automation
1 338
6.4 Creating Expert Advisors
Identifier
Description
DEAL_REASON_CLIENT
Triggering of an order placed from the desktop
terminal
DEAL_REASON_MOBILE
Triggering of an order placed from a mobile
application
DEAL_REASON_WEB
Triggering of an order placed from the web
platform
DEAL_REASON_EXPERT
Triggering of an order placed by an Expert Advisor
or a script
DEAL_REASON_SL
Stop Loss order triggered
DEAL_REASON_TP
Take Profit order triggering
DEAL_REASON_SO
Stop Out event
DEAL_REASON_ROLLOVER
Position transfer to a new day
DEAL_REASON_VMARGIN
Add/deduct variation margin
DEAL_REASON_SPLIT
Split (lower price) the instrument on which there
was a position
Real type properties are represented by the ENUM_DEAL_PROPERTY_DOUBLE enumeration.
Identifier
Description
DEAL_VOLUME
Deal volume
DEAL_PRICE
Deal price
DEAL_COMMISSION
Deal commission
DEAL_SWAP
Accumulated swap at close
DEAL_PROFIT
Financial result of the deal
DEAL_FEE
Fee for the deal which is charged immediately after
the deal
DEAL_SL
Stop Loss Level
DEAL_TP
Take Profit level
The two last properties are filled as follows: for an entry or reversal deal, the Stop Loss/Take Profit
value is taken from the order by which the position was opened or expanded. For the exit deal, the Stop
Loss/Take Profit value is taken from the position at the time of its closing.
String deal properties are available via ENUM_DEAL_PROPERTY_STRING enumeration constants.

---

## Page 1339

Part 6. Trading automation
1 339
6.4 Creating Expert Advisors
Identifier
Description
DEAL_SYMBOL
The name of the symbol for which the deal was
made
DEAL_COMMENT
Deal comment
DEAL_EXTERNAL_ID
Deal identifier in the external trading system (on
the exchange)
We will test how to read the properties in the section on HistoryDealGet functions through the
DealMonitor and DealFilter classes.
6.4.30 Selecting orders and deals from history
MetaTrader 5 allows you to create a snapshot of history for a specific time period for an Expert Advisor
or a script. The snapshot is a list of orders and deals which can be further accessed through the
appropriate functions. In addition, history can be requested in relation to specific orders, deals or
positions.
Selecting the required period explicitly (by dates) is performed by the HistorySelect function. After that,
the size of the list of deals and the list of orders can be found using the HistoryDealsTotal and
HistoryOrdersTotal functions, respectively. The elements of the orders list can be checked using the
HistoryOrderGetTicket function; for elements of the deals list use HistoryDealGetTicket.
It is necessary to distinguish between active (working) orders and orders in history, i.e., those
executed, canceled or rejected. To analyze active orders, use the functions discussed in the
sections related to getting a list of active orders and reading their properties.
bool HistorySelect(datetime from, datetime to)
The function requests the history of deals and orders for the specified period of server time (from and
to inclusive, to >= from) and returns true in case of success.
Even if there are no orders and transactions in the requested period, the function will return true in the
absence of errors. An error can be, for example, a lack of memory for building a list of orders or deals.
Please note that orders have two times: set (ORDER_TIME_SETUP) and execution
(ORDER_TIME_DONE). The function HistorySelect selects orders by execution time.
To extract the entire account history, you can use the syntax HistorySelect(0, LONG_ MAX).
Another way to access a part of the history is by position ID.
bool HistorySelectByPosition(ulong positionID)
The function requests the history of deals and orders with the specified position ID in the
ORDER_POSITION_ID, DEAL_POSITION_ID properties.
Attention! The function does not select orders by the ID of the opposite position for Close By
operations. In other words, the ORDER_POSITION_BY_ID property is ignored, despite the fact that
the order data was involved in the formation of the position. 
For example, an Expert Advisor could complete a buy (order #1 ) and sell (order #2) on a hedging-
enabled account. This will then lead to the formation of positions #1  and #2. Opposite closing of

---

## Page 1340

Part 6. Trading automation
1 340
6.4 Creating Expert Advisors
positions requires the ORDER_TYPE_CLOSE_BY (#3) order. As a result, the
HistorySelectByPosition(#1 ) call will select orders #1  and #3, which is expected. However, the call
of HistorySelectByPosition(#2) will select only order #2 (despite the fact that order #3 has #2 in
the ORDER_POSITION_BY_ID property, and strictly speaking, order #3 participated in closing
position #2).
Upon successful execution of either of the two functions, HistorySelect or HistorySelectByPosition, the
terminal generates an internal list of orders and deals for the MQL program. You can also change the
historical context with the functions HistoryOrderSelect and HistoryDealSelect, for which you need to
know the ticket of the corresponding object in advance (for example, save it from the request result).
It is important to note that HistoryOrderSelect affects only the list of orders, and HistoryDealSelect is
only used for the list of deals.
All context selection functions return a bool value for success (true) or error (false). The error code can
be read in the built-in _ LastError variable.
bool HistoryOrderSelect(ulong ticket)
The HistoryOrderSelect function selects an order in the history by its ticket. The order is then used for
further operations with the deal (reading properties).
During the application of the HistoryOrderSelect function, if the search for an order by ticket was
successful, the new list of orders selected in the history will consist of the only order just found. In
other words, the previous list of selected orders (if any) is reset. However, the function does not reset
the previously selected transaction history, i.e., it does not select the transaction(s) associated with
the order.
bool HistoryDealSelect(ulong ticket)
The function HistoryDealSelect selects a deal in the history for further access to it through the
appropriate functions. The function does not reset the order history, i.e., it does not select the order
associated with the selected deal.
After a certain context is selected in the history by calling one of the above functions, the MQL
program can call the functions to iterate over the orders and deals that fall into this context and read
their properties.
int HistoryOrdersTotal()
The HistoryOrdersTotal function returns the number of orders in history (in the selection).
ulong HistoryOrderGetTicket(int index)
The HistoryOrderGetTicket function allows you to get an order ticket by its serial number in the
selected history context. The index must be between 0 and N-1 , where N is obtained from the
HistoryOrdersTotal function.
Knowing the order ticket, it is easy to get all the necessary properties of it using HistoryOrderGet
functions. The properties of historical orders are exactly the same as those of existing orders.
There is a similar pair of functions for working with deals.
int HistoryDealsTotal()
The HistoryDealsTotal function returns the number of deals in history (in the selection).

---

## Page 1341

Part 6. Trading automation
1 341 
6.4 Creating Expert Advisors
ulong HistoryDealGetTicket(int index)
The HistoryDealGetTicket function allows you to get a deal ticket by its serial number in the selected
history context. This is necessary for further processing of the deal using HistoryDealGet functions. The
list of deal properties accessible through these functions was described in the previous section.
We will consider an example of using functions after studying HistoryOrderGet and HistoryDealGet
functions.
6.4.31  Functions for reading order properties from history
The functions for reading properties of historical orders are divided into 3 groups according to the basic
type of property values, in accordance with the division of identifiers of available properties into three
enumerations: E N U M _OR D E R _PR OPE R TY_IN TE GE R , E N U M _OR D E R _PR OPE R TY_D OU B L E  and
E N U M _OR D E R _PR OPE R TY_STR IN G discussed earlier in a separate section when exploring active orders.
Before calling these functions, you need to somehow select the appropriate set of tickets in the history.
If you try to read the properties of an order or a deal having tickets outside the selected history
context, the environment may generate a WRONG_INTERNAL_PARAMETER (4002) error, which can be
analyzed via _ LastError.
For each base property type, there are two function forms: one directly returns the value of the
requested property, the second one writes it into a parameter passed by reference and returns a
success indicator (true) or errors (false).
For integer and compatible types (datetime, enums) of properties there is a dedicated function
HistoryOrderGetInteger.
long HistoryOrderGetInteger(ulong ticket, ENUM_ORDER_PROPERTY_INTEGER property)
bool HistoryOrderGetInteger(ulong ticket, ENUM_ORDER_PROPERTY_INTEGER property,
   long &value)
The function allows you to find out the order property from the selected history by its ticket number.
For real properties, the HistoryOrderGetDouble function is assigned.
double HistoryOrderGetDouble(ulong ticket, ENUM_ORDER_PROPERTY_DOUBLE property)
bool HistoryOrderGetDouble(ulong ticket, ENUM_ORDER_PROPERTY_DOUBLE property,
   double &value)
Finally, string properties can be read with HistoryOrderGetString.
string HistoryOrderGetString(ulong ticket, ENUM_ORDER_PROPERTY_STRING property)
bool HistoryOrderGetString(ulong ticket, ENUM_ORDER_PROPERTY_STRING property,
   string &value)
Now we can supplement the OrderMonitor class (OrderMonitor.mqh) for working with historical orders.
First, let's add a boolean variable to the history class, which we will fill in the constructor based on the
segment in which the order with the passed ticket was selected: among the active ones (OrderSelect)
or in history (HistoryOrderSelect).

---

## Page 1342

Part 6. Trading automation
1 342
6.4 Creating Expert Advisors
class OrderMonitor: public OrderMonitorInterface
{
   bool history;
   
public:
   const ulong ticket;
   OrderMonitor(const long t): ticket(t), history(!OrderSelect(t))
   {
      if(history && !HistoryOrderSelect(ticket))
      {
         PrintFormat("Error: OrderSelect(%lld) failed: %s", ticket, E2S(_LastError));
      }
      else
      {
         ResetLastError();
         ready = true;
      }
   }
   ...
We need to call the ResetLastError function in a successful if branch in order to reset the possible error
that could be set by the OrderSelect function (if the order is in history).
In fact, this version of the constructor contains a serious logical error, and we will return to it after a
few paragraphs.
To read properties in get methods, we now call different built-in functions, depending on the value of
the history variable.
   virtual long get(const ENUM_ORDER_PROPERTY_INTEGER property) const override
   {
      return history ? HistoryOrderGetInteger(ticket, property) : OrderGetInteger(property);
   }
   
   virtual double get(const ENUM_ORDER_PROPERTY_DOUBLE property) const override
   {
      return history ? HistoryOrderGetDouble(ticket, property) : OrderGetDouble(property);
   }
   
   virtual string get(const ENUM_ORDER_PROPERTY_STRING property) const override
   {
      return history ? HistoryOrderGetString(ticket, property) : OrderGetString(property);
   }
   ...
The main purpose of the OrderMonitor class is to supply data to other analytical classes. The
OrderMonitor objects are used to filter active orders in the OrderFilter class, and we need a similar class
for selecting orders by arbitrary conditions on the history: HistoryOrderFilter.
Let's write this class in the same file OrderFilter.mqh. It uses two new functions for working with
history: HistoryOrdersTotal and HistoryOrderGetTicket.

---

## Page 1343

Part 6. Trading automation
1 343
6.4 Creating Expert Advisors
class HistoryOrderFilter: public TradeFilter<OrderMonitor,
   ENUM_ORDER_PROPERTY_INTEGER,
   ENUM_ORDER_PROPERTY_DOUBLE,
   ENUM_ORDER_PROPERTY_STRING>
{
protected:
   virtual int total() const override
   {
      return HistoryOrdersTotal();
   }
   virtual ulong get(const int i) const override
   {
      return HistoryOrderGetTicket(i);
   }
};
This simple code inherits from the template class TradeFilter, where the class is passed as the first
parameter of the template OrderMonitor to read the properties of the corresponding objects (we saw an
analog for positions, and will soon create one for deals).
Here lies the problem with the OrderMonitor constructor. As we learned in the section Selecting orders
and deals from history, to analyze the account we must first set up the context with one of the
functions such as HistorySelect. So here in the source code HistoryOrderFilter it is assumed that the
MQL program has already selected the required history fragment. However, the new, intermediate
version of the OrderMonitor constructor uses the HistoryOrderSelect call to check the existence of a
ticket in history. Meanwhile, this function resets the previous context of historical orders and selects a
single order.
So we need a helper method historyOrderSelectWeak to validate the ticket in a "soft" way without
breaking the existing context. To do this, we can simply check if the ORDER_TICKET property is equal
to the passed ticket t: (HistoryOrderGetInteger(t, ORDER_ TICKET) == t). If such a ticket has already
been selected (available), the check will succeed, and the monitor does not need to manipulate the
history.

---

## Page 1344

Part 6. Trading automation
1 344
6.4 Creating Expert Advisors
class OrderMonitor: public OrderMonitorInterface
{
   bool historyOrderSelectWeak(const ulong t) const
   {
      return (((HistoryOrderGetInteger(t, ORDER_TICKET) == t) ||
         (HistorySelect(0, LONG_MAX) && (HistoryOrderGetInteger(t, ORDER_TICKET) == t))));
   }
   bool history;
   
public:
   const ulong ticket;
   OrderMonitor(const long t): ticket(t), history(!OrderSelect(t))
   {
      if(history && !historyOrderSelectWeak(ticket))
      {
         PrintFormat("Error: OrderSelect(%lld) failed: %s", ticket, E2S(_LastError));
      }
      else
      {
         ResetLastError();
         ready = true;
      }
   }
An example of applying order filtering on history will be considered in the next section after we prepare
a similar functionality for deals.
6.4.32 Functions for reading deal properties from history
To read deal properties, there are groups of functions organized by property type: integer, real and
string. Before calling functions, you need to select the desired period of history and thus ensure the
availability of deals with tickets that are passed in the first parameter (ticket) of all functions.
There are two forms for each type of property: returning a value directly and writing to a variable by
reference. The second form returns true to indicate success. The first form will simply return 0 on
error. The error code is in the _ LastError variable.
Integer and compatible property types (datetime, enumerations) can be obtained using the
HistoryDealGetInteger function.
long HistoryDealGetInteger(ulong ticket, ENUM_DEAL_PROPERTY_INTEGER property)
bool HistoryDealGetInteger(ulong ticket, ENUM_DEAL_PROPERTY_INTEGER property,
   long &value)
Real properties are read by the HistoryDealGetDouble function.
double HistoryDealGetDouble(ulong ticket, ENUM_DEAL_PROPERTY_DOUBLE property)
bool HistoryDealGetDouble(ulong ticket, ENUM_DEAL_PROPERTY_DOUBLE property,
   double &value)
For string properties there is the HistoryDealGetString function.

---

## Page 1345

Part 6. Trading automation
1 345
6.4 Creating Expert Advisors
string HistoryDealGetString(ulong ticket, ENUM_DEAL_PROPERTY_STRING property)
bool HistoryDealGetString(ulong ticket, ENUM_DEAL_PROPERTY_STRING property,
   string &value)
A unified reading of deal properties will be provided by the DealMonitor class (DealMonitor.mqh),
organized in exactly the same way as OrderMonitor and PositionMonitor. The base class is
DealMonitorInterface, inherited from the template MonitorInterface (we described it in the section
Functions for reading the properties of active orders). It is at this level that the specific types of
ENUM_DEAL_PROPERTY enumerations are specified as template parameters and the specific
implementation of the stringify method.
#include <MQL5Book/TradeBaseMonitor.mqh>
   
class DealMonitorInterface:
   public MonitorInterface<ENUM_DEAL_PROPERTY_INTEGER,
   ENUM_DEAL_PROPERTY_DOUBLE,ENUM_DEAL_PROPERTY_STRING>
{
public:
   // property descriptions taking into account integer subtypes
   virtual string stringify(const long v,
      const ENUM_DEAL_PROPERTY_INTEGER property) const override
   {
      switch(property)
      {
         case DEAL_TYPE:
            return enumstr<ENUM_DEAL_TYPE>(v);
         case DEAL_ENTRY:
            return enumstr<ENUM_DEAL_ENTRY>(v);
         case DEAL_REASON:
            return enumstr<ENUM_DEAL_REASON>(v);
         
         case DEAL_TIME:
            return TimeToString(v, TIME_DATE | TIME_SECONDS);
         
         case DEAL_TIME_MSC:
            return STR_TIME_MSC(v);
      }
      
      return (string)v;
   }
};
The DealMonitor class below is somewhat similar to a class recently modified to work with history
OrderMonitor. In addition to the application of HistoryDeal functions instead of HistoryOrder functions, it
should be noted that for deals there is no need to check the ticket in the online environment because
deals exist only in history.

---

## Page 1346

Part 6. Trading automation
1 346
6.4 Creating Expert Advisors
class DealMonitor: public DealMonitorInterface
{
   bool historyDealSelectWeak(const ulong t) const
   {
      return ((HistoryDealGetInteger(t, DEAL_TICKET) == t) ||
         (HistorySelect(0, LONG_MAX) && (HistoryDealGetInteger(t, DEAL_TICKET) == t)));
   }
public:
   const ulong ticket;
   DealMonitor(const long t): ticket(t)
   {
      if(!historyDealSelectWeak(ticket))
      {
         PrintFormat("Error: HistoryDealSelect(%lld) failed", ticket);
      }
      else
      {
         ready = true;
      }
   }
   
   virtual long get(const ENUM_DEAL_PROPERTY_INTEGER property) const override
   {
      return HistoryDealGetInteger(ticket, property);
   }
   
   virtual double get(const ENUM_DEAL_PROPERTY_DOUBLE property) const override
   {
      return HistoryDealGetDouble(ticket, property);
   }
   
   virtual string get(const ENUM_DEAL_PROPERTY_STRING property) const override
   {
      return HistoryDealGetString(ticket, property);
   }
   ...
};
Based on DealMonitor and TradeFilter it is easy to create a deal filter (DealFilter.mqh). Recall that
TradeFilter, as the base class for many entities, was described in the section Selecting orders by
properties.

---

## Page 1347

Part 6. Trading automation
1 347
6.4 Creating Expert Advisors
#include <MQL5Book/DealMonitor.mqh>
#include <MQL5Book/TradeFilter.mqh>
   
class DealFilter: public TradeFilter<DealMonitor,
   ENUM_DEAL_PROPERTY_INTEGER,
   ENUM_DEAL_PROPERTY_DOUBLE,
   ENUM_DEAL_PROPERTY_STRING>
{
protected:
   virtual int total() const override
   {
      return HistoryDealsTotal();
   }
   virtual ulong get(const int i) const override
   {
      return HistoryDealGetTicket(i);
   }
};
As a generalized example of working with histories, consider the position history recovery script
TradeHistoryPrint.mq5.
TradeHistoryPrint
The script will build a history for the current chart symbol.
We first need filters for deals and orders.
#include <MQL5Book/OrderFilter.mqh>
#include <MQL5Book/DealFilter.mqh>
From the deals, we will extract the position IDs and, based on them, we will request details about the
orders.
The history can be viewed in its entirety or for a specific position, for which we will provide a mode
selection and an input field for the identifier in the input variables.
enum SELECTOR_TYPE
{
   TOTAL,    // Whole history
   POSITION, // Position ID
};
   
input SELECTOR_TYPE Type = TOTAL;
input ulong PositionID = 0; // Position ID
It should be remembered that sampling a long account history can be an overhead, so it is desirable to
provide for caching of the obtained results of history processing in working Expert Advisors, along with
the last processing timestamp. With each subsequent analysis of history, you can start the process not
from the very beginning, but from a remembered moment.
To display information about history records with column alignment in a visually attractive way, it
makes sense to represent it as an array of structures. However, our filters already support querying
data stored in special structures - tuples. Therefore, we will apply a trick: we will describe our
application structures, observing the rules of tuples:

---

## Page 1348

Part 6. Trading automation
1 348
6.4 Creating Expert Advisors
• The first field must have the name _1 ; it is optionally used in the sorting algorithm.
• The size function returning the number of fields must be described in the structure.
• The structure should have a template method assign to populate fields from the properties of the
passed monitor object derived from MonitorInterface.
In standard tuples, the method assign is described like this:
   template<typename M> 
   void assign(const int &properties[], M &m);
As the first parameter, it receives an array with the property IDs corresponding to the fields we are
interested in. In fact, this is the array that is passed by the calling code to the select method of the
filter (TradeFilter::select), and then by reference it gets to assign. But since we will now create not
some standard tuples but our own structures that "know" about the applied nature of their fields, we
can leave the array with property identifiers inside the structure itself and not "drive" it into the filter
and back to the assign method of the same structure.
In particular, to request deals, we describe the DealTuple structure with 8 fields. Their identifiers will be
specified in the fields static array.
struct DealTuple
{
   datetime _1;   // deal time
   ulong deal;    // deal ticket
   ulong order;   // order ticket
   string type;   // ENUM_DEAL_TYPE as string 
   string in_out; // ENUM_DEAL_ENTRY as string 
   double volume;
   double price;
   double profit;
   
   static int size() { return 8; }; // number of properties 
   static const int fields[]; // identifiers of the requested deal properties
   ...
};
   
static const int DealTuple::fields[] =
{
   DEAL_TIME, DEAL_TICKET, DEAL_ORDER, DEAL_TYPE,
   DEAL_ENTRY, DEAL_VOLUME, DEAL_PRICE, DEAL_PROFIT
};
This approach brings together identifiers and fields to store the corresponding values in a single place,
which makes it easier to understand and maintain the source code.
Filling fields with property values will require a slightly modified (simplified) version of the assign method
which takes the IDs from the fields array and not from the input parameter.

---

## Page 1349

Part 6. Trading automation
1 349
6.4 Creating Expert Advisors
struct DealTuple
{
   ...
   template<typename M> // M is derived from MonitorInterface<>
   void assign(M &m)
   {
      static const int DEAL_TYPE_ = StringLen("DEAL_TYPE_");
      static const int DEAL_ENTRY_ = StringLen("DEAL_ENTRY_");
      static const ulong L = 0; // default type declaration (dummy)
      
      _1 = (datetime)m.get(fields[0], L);
      deal = m.get(fields[1], deal);
      order = m.get(fields[2], order);
      const ENUM_DEAL_TYPE t = (ENUM_DEAL_TYPE)m.get(fields[3], L);
      type = StringSubstr(EnumToString(t), DEAL_TYPE_);
      const ENUM_DEAL_ENTRY e = (ENUM_DEAL_ENTRY)m.get(fields[4], L);
      in_out = StringSubstr(EnumToString(e), DEAL_ENTRY_);
      volume = m.get(fields[5], volume);
      price = m.get(fields[6], price);
      profit = m.get(fields[7], profit);
   }
};
At the same time, we convert the numeric elements of the ENUM_DEAL_TYPE and ENUM_DEAL_ENTRY
enumerations into user-friendly strings. Of course, this is only needed for logging. For programmatic
analysis, the types should be left as they are.
Since we have invented a new version of the assign method in their tuples, you need to add a new
version of the select method for it in the TradeFilter class. The innovation will certainly be useful for
other programs, and therefore we will introduce it directly into TradeFilter, not into some new derived
class.

---

## Page 1350

Part 6. Trading automation
1 350
6.4 Creating Expert Advisors
template<typename T,typename I,typename D,typename S>
class TradeFilter
{
   ...
   template<typename U> // U must have first field _1 and method assign(T)
   bool select(U &data[], const bool sort = false) const
   {
      const int n = total();
      // loop through the elements
      for(int i = 0; i < n; ++i)
      {
         const ulong t = get(i);
         // read properties through the monitor object
         T m(t);
         // check all filtering conditions
         if(match(m, longs)
         && match(m, doubles)
         && match(m, strings))
         {
            // for a suitable object, add its properties to an array
            const int k = EXPAND(data);
            data[k].assign(m);
         }
      }
      
      if(sort)
      {
         static const U u;
         sortTuple(data, u._1);
      }
      
      return true;
   }
Recall that all template methods are not implemented by the compiler until they are called in code with
a specific type. Therefore, the presence of such patterns in TradeFilter does not oblige you to include
any tuple header files or describe similar structures if you don't use them.
So, if earlier, to select transactions using a standard tuple, we would have to write like this:
#include <MQL5Book/Tuples.mqh>
...
DealFilter filter;
int properties[] =
{
   DEAL_TIME, DEAL_TICKET, DEAL_ORDER, DEAL_TYPE,
   DEAL_ENTRY, DEAL_VOLUME, DEAL_PRICE, DEAL_PROFIT
};
Tuple8<ulong,ulong,ulong,ulong,ulong,double,double,double> tuples[];
filter.let(DEAL_SYMBOL, _Symbol).select(properties, tuples);
Then with a customized structure, everything is much simpler:

---

## Page 1351

Part 6. Trading automation
1 351 
6.4 Creating Expert Advisors
DealFilter filter;
DealTuple tuples[];
filter.let(DEAL_SYMBOL, _Symbol).select(tuples);
Similar to the DealTuple structure, let's describe the 1 0-field structure for orders OrderTuple.
struct OrderTuple
{
   ulong _1;       // ticket (also used as 'ulong' prototype)
   datetime setup;
   datetime done;
   string type;
   double volume;
   double open;
   double current;
   double sl;
   double tp;
   string comment;
   
   static int size() { return 10; }; // number of properties
   static const int fields[]; // identifiers of requested order properties
   
   template<typename M> // M is derived from MonitorInterface<>
   void assign(M &m)
   {
      static const int ORDER_TYPE_ = StringLen("ORDER_TYPE_");
      
      _1 = m.get(fields[0], _1);
      setup = (datetime)m.get(fields[1], _1);
      done = (datetime)m.get(fields[2], _1);
      const ENUM_ORDER_TYPE t = (ENUM_ORDER_TYPE)m.get(fields[3], _1);
      type = StringSubstr(EnumToString(t), ORDER_TYPE_);
      volume = m.get(fields[4], volume);
      open = m.get(fields[5], open);
      current = m.get(fields[6], current);
      sl = m.get(fields[7], sl);
      tp = m.get(fields[8], tp);
      comment = m.get(fields[9], comment);
   }
};
   
static const int OrderTuple::fields[] =
{
   ORDER_TICKET, ORDER_TIME_SETUP, ORDER_TIME_DONE, ORDER_TYPE, ORDER_VOLUME_INITIAL,
   ORDER_PRICE_OPEN, ORDER_PRICE_CURRENT, ORDER_SL, ORDER_TP, ORDER_COMMENT
};
Now everything is ready to implement the main function of the script – OnStart. At the very beginning,
we will describe the objects of filters for deals and orders.

---

## Page 1352

Part 6. Trading automation
1 352
6.4 Creating Expert Advisors
void OnStart()
{
   DealFilter filter;
   HistoryOrderFilter subfilter;
   ...
Depending on the input variables, we choose either the entire history or a specific position.
   if(PositionID == 0 || Type == TOTAL)
   {
      HistorySelect(0, LONG_MAX);
   }
   else if(Type == POSITION)
   {
      HistorySelectByPosition(PositionID);
   }
   ...
Next, we will collect all position identifiers in an array, or leave one specified by the user.
   ulong positions[];
   if(PositionID == 0)
   {
      ulong tickets[];
      filter.let(DEAL_SYMBOL, _Symbol)
         .select(DEAL_POSITION_ID, tickets, positions, true); // true - sorting
      ArrayUnique(positions);
   }
   else
   {
      PUSH(positions, PositionID);
   }
   
   const int n = ArraySize(positions);
   Print("Positions total: ", n);
   if(n == 0) return;
   ...
The helper function ArrayUnique leaves non-repeating elements in the array. It requires the source
array to be sorted for it to work.
Further, in a loop through positions, we request deals and orders related to each of them. Deals are
sorted by the first field of the DealTuple structure, i.e., by time. Perhaps the most interesting is the
calculation of profit/loss on a position. To do this, we sum the values of the profit field of all deals.

---

## Page 1353

Part 6. Trading automation
1 353
6.4 Creating Expert Advisors
   for(int i = 0; i < n; ++i)
   {
      DealTuple deals[];
      filter.let(DEAL_POSITION_ID, positions[i]).select(deals, true);
      const int m = ArraySize(deals);
      if(m == 0)
      {
         Print("Wrong position ID: ", positions[i]);
         break; // invalid id set by user
      }
      double profit = 0; // TODO: need to take into account commissions, swaps and fees
      for(int j = 0; j < m; ++j) profit += deals[j].profit;
      PrintFormat("Position: % 8d %16lld Profit:%f", i + 1, positions[i], (profit));
      ArrayPrint(deals);
      
      Print("Order details:");
      OrderTuple orders[];
      subfilter.let(ORDER_POSITION_ID, positions[i], IS::OR_EQUAL)
         .let(ORDER_POSITION_BY_ID, positions[i], IS::OR_EQUAL)
         .select(orders);
      ArrayPrint(orders);
   }
}
This code does not analyze commissions (DEAL_COMMISSION), swaps (DEAL_SWAP), and fees
(DEAL_FEE) in deal properties. In real Expert Advisors, this should probably be done (depending on the
requirements of the strategy). We will look at another example of trading history analysis in the section
on testing multicurrency Expert Advisors, and there we will take into account this moment.
You can compare the results of the script with the table on the History tab in the terminal: its Profit
column shows the net profit for each position (swaps, commissions, and fees are in adjacent columns,
but they need to be included).
It is important to note that an order of the ORDER_TYPE_CLOSE_BY type will be displayed in both
positions only if the entire history is selected in the settings. If a specific position was selected, the
system will include such an order only in one of them (the one that was specified in the trade request
first, in the position field) but not the second one (which was specified in position_ by).
Below is an example of the result of the script for a symbol with a small history.

---

## Page 1354

Part 6. Trading automation
1 354
6.4 Creating Expert Advisors
Positions total: 3
Position:        1       1253500309 Profit:238.150000
                   [_1]     [deal]    [order] [type] [in_out] [volume]  [price]  [profit]
[0] 2022.02.04 17:34:57 1236049891 1253500309 "BUY"  "IN"      1.00000 76.23900   0.00000
[1] 2022.02.14 16:28:41 1242295527 1259788704 "SELL" "OUT"     1.00000 76.42100 238.15000
Order details:
          [_1]             [setup]              [done] [type] [volume]   [open] [current] »
   » [sl] [tp] [comment]
[0] 1253500309 2022.02.04 17:34:57 2022.02.04 17:34:57 "BUY"   1.00000 76.23900  76.23900 »
   » 0.00 0.00 ""       
[1] 1259788704 2022.02.14 16:28:41 2022.02.14 16:28:41 "SELL"  1.00000 76.42100  76.42100 »
   » 0.00 0.00 ""       
Position:        2       1253526613 Profit:878.030000
                   [_1]     [deal]    [order] [type] [in_out] [volume]  [price]  [profit]
[0] 2022.02.07 10:00:00 1236611994 1253526613 "BUY"  "IN"      1.00000 75.75000   0.00000
[1] 2022.02.14 16:28:40 1242295517 1259788693 "SELL" "OUT"     1.00000 76.42100 878.03000
Order details:
          [_1]             [setup]              [done]      [type] [volume]   [open] [current] »
   » [sl] [tp] [comment]
[0] 1253526613 2022.02.04 17:55:18 2022.02.07 10:00:00 "BUY_LIMIT"  1.00000 75.75000  75.67000 »
   » 0.00 0.00 ""       
[1] 1259788693 2022.02.14 16:28:40 2022.02.14 16:28:40 "SELL"       1.00000 76.42100  76.42100 »
   » 0.00 0.00 ""       
Position:        3       1256280710 Profit:4449.040000
                   [_1]     [deal]    [order] [type] [in_out] [volume]  [price]   [profit]
[0] 2022.02.09 13:17:52 1238797056 1256280710 "BUY"  "IN"      2.00000 74.72100    0.00000
[1] 2022.02.14 16:28:39 1242295509 1259788685 "SELL" "OUT"     2.00000 76.42100 4449.04000
Order details:
          [_1]             [setup]              [done] [type] [volume]   [open] [current] »
   » [sl] [tp] [comment]
[0] 1256280710 2022.02.09 13:17:52 2022.02.09 13:17:52 "BUY"   2.00000 74.72100  74.72100 »
   » 0.00 0.00 ""       
[1] 1259788685 2022.02.14 16:28:39 2022.02.14 16:28:39 "SELL"  2.00000 76.42100  76.42100 »
   » 0.00 0.00 ""       
The case of increasing a position (two "IN" deals) and its reversal (an "INOUT" deal of a larger volume)
on a netting account is shown in the following fragment.

---

## Page 1355

Part 6. Trading automation
1 355
6.4 Creating Expert Advisors
Position:        5        219087383 Profit:0.170000
                   [_1]    [deal]   [order] [type] [in_out] [volume] [price] [profit]
[0] 2022.03.29 08:03:33 215612450 219087383 "BUY"  "IN"      0.01000 1.10011  0.00000
[1] 2022.03.29 08:04:05 215612451 219087393 "BUY"  "IN"      0.01000 1.10009  0.00000
[2] 2022.03.29 08:04:29 215612457 219087400 "SELL" "INOUT"   0.03000 1.10018  0.16000
[3] 2022.03.29 08:04:34 215612460 219087403 "BUY"  "OUT"     0.01000 1.10017  0.01000
Order details:
         [_1]             [setup]              [done] [type] [volume] [open] [current] »
   » [sl] [tp] [comment]
[0] 219087383 2022.03.29 08:03:33 2022.03.29 08:03:33 "BUY"   0.01000 0.0000   1.10011 »
   » 0.00 0.00 ""       
[1] 219087393 2022.03.29 08:04:05 2022.03.29 08:04:05 "BUY"   0.01000 0.0000   1.10009 »
   » 0.00 0.00 ""       
[2] 219087400 2022.03.29 08:04:29 2022.03.29 08:04:29 "SELL"  0.03000 0.0000   1.10018 »
   » 0.00 0.00 ""       
[3] 219087403 2022.03.29 08:04:34 2022.03.29 08:04:34 "BUY"   0.01000 0.0000   1.10017 »
   » 0.00 0.00 ""       
We will consider a partial history using the example of specific positions for the case of an opposite
closure on a hedging account. First, you can view the first position separately:
PositionID=1 2761 09280. It will be shown in full regardless of the input parameter Type.
Positions total: 1
Position:        1       1276109280 Profit:-0.040000
                   [_1]     [deal]    [order] [type] [in_out] [volume] [price] [profit]
[0] 2022.03.07 12:20:53 1258725455 1276109280 "BUY"  "IN"      0.01000 1.08344  0.00000
[1] 2022.03.07 12:20:58 1258725503 1276109328 "SELL" "OUT_BY"  0.01000 1.08340 -0.04000
Order details:
          [_1]             [setup]              [done]     [type] [volume]  [open] [current] »
   » [sl] [tp]                    [comment]
[0] 1276109280 2022.03.07 12:20:53 2022.03.07 12:20:53 "BUY"       0.01000 1.08344   1.08344 »
   » 0.00 0.00 ""                          
[1] 1276109328 2022.03.07 12:20:58 2022.03.07 12:20:58 "CLOSE_BY"  0.01000 1.08340   1.08340 »
   » 0.00 0.00 "#1276109280 by #1276109283"
You can also see the second one: PositionID=1 2761 09283. However, if Type equals "position", to
select a fragment of history, the function HistorySelectByPosition is used, and as a result there will be
only one exit order (despite the fact that there are two deals).
Positions total: 1
Position:        1       1276109283 Profit:0.000000
                   [_1]     [deal]    [order] [type] [in_out] [volume] [price] [profit]
[0] 2022.03.07 12:20:53 1258725458 1276109283 "SELL" "IN"      0.01000 1.08340  0.00000
[1] 2022.03.07 12:20:58 1258725504 1276109328 "BUY"  "OUT_BY"  0.01000 1.08344  0.00000
Order details:
          [_1]             [setup]              [done] [type] [volume]  [open] [current] »
   » [sl] [tp] [comment]
[0] 1276109283 2022.03.07 12:20:53 2022.03.07 12:20:53 "SELL"  0.01000 1.08340   1.08340 »
   » 0.00 0.00 ""       
If we set Type to the "whole history", a "CLOSE_BY" order will appear.

---

## Page 1356

Part 6. Trading automation
1 356
6.4 Creating Expert Advisors
Positions total: 1
Position:        1       1276109283 Profit:0.000000
                   [_1]     [deal]    [order] [type] [in_out] [volume] [price] [profit]
[0] 2022.03.07 12:20:53 1258725458 1276109283 "SELL" "IN"      0.01000 1.08340  0.00000
[1] 2022.03.07 12:20:58 1258725504 1276109328 "BUY"  "OUT_BY"  0.01000 1.08344  0.00000
Order details:
          [_1]             [setup]              [done]     [type] [volume]  [open] [current] »
   » [sl] [tp]                    [comment]
[0] 1276109283 2022.03.07 12:20:53 2022.03.07 12:20:53 "SELL"      0.01000 1.08340   1.08340 »
   » 0.00 0.00 ""                          
[1] 1276109328 2022.03.07 12:20:58 2022.03.07 12:20:58 "CLOSE_BY"  0.01000 1.08340   1.08340 »
   » 0.00 0.00 "#1276109280 by #1276109283"
With such settings, the history is selected completely, but the filter leaves only those orders, in which
the identifier of the specified position is found in the ORDER_POSITION_ID or ORDER_POSITION_BY_ID
properties. For composing conditions with a logical OR, the IS::OR_EQUAL element has been added to
the TradeFilter class. You can additionally study it.
6.4.33 Types of trading transactions
In addition to performing trading operations, MQL programs can respond to trading events. It is
important to note that such events occur not only as a result of the actions of programs, but also for
other reasons, for example, when manually managed by the user or performing automatic actions on
the server (activation of a pending order, Stop Loss, Take Profit, Stop Out, position transfer to a new
day, depositing or withdrawing funds from the account, and much more).
Regardless of the initiator of the actions, they result in the execution of trading transactions on the
account. Trading transactions are indivisible steps that include:
·Processing a trade request
·Changing the list of active orders (including adding a new order, executing and deleting a triggered
order)
·Changing the history of orders
·Changing the history of deals
·Changing positions
Depending on the nature of the operation, some steps may be optional. For example, modifying the
protective levels of a position will miss three middle points. And when a buy order is sent, the market
will go through a full cycle: the request is processed, a corresponding order is created for the account,
the order is executed, it is removed from the active list, added to the order history, then the
corresponding deal is added to the history and a new position is created. All these actions are trading
transactions.
To receive notifications about such events, the special OnTradeTransaction handler function should be
described in an Expert Advisor or an indicator. We will look at it in detail in the next section. The fact is
that one of its parameters, the first and most important, has the type of a predefined structure
MqlTradeTransaction. So let's first talk about transactions as such.

---

## Page 1357

Part 6. Trading automation
1 357
6.4 Creating Expert Advisors
struct MqlTradeTransaction
{ 
   ulong                         deal;             // Deal ticket 
   ulong                         order;            // Order ticket 
   string                        symbol;           // Name of the trading instrument 
   ENUM_TRADE_TRANSACTION_TYPE   type;             // Trade transaction type 
   ENUM_ORDER_TYPE               order_type;       // Order type
   ENUM_ORDER_STATE              order_state;      // Order state 
   ENUM_DEAL_TYPE                deal_type;        // Deal type 
   ENUM_ORDER_TYPE_TIME          time_type;        // Order type by duration
   datetime                      time_expiration;  // Order expiration date 
   double                        price;            // Price 
   double                        price_trigger;    // Stop limit order trigger price 
   double                        price_sl;         // Stop Loss Level 
   double                        price_tp;         // Take Profit Level 
   double                        volume;           // Volume in lots 
   ulong                         position;         // Position ticket 
   ulong                         position_by;      // Opposite position ticket 
};
The following table describes each structure field.
Field
Description
deal
Deal ticket
order
Order ticket
symbol
The name of the trading instrument on which the transaction was made
type
Trade transaction type ENUM_TRADE_TRANSACTION_TYPE (see below)
order_type
Order type ENUM_ORDER_TYPE
order_state
Order status ENUM_ORDER_STATE
deal_type
Deal type ENUM_DEAL_TYPE
time_type
Order type by expiration ENUM_ORDER_TYPE_TIME
time_expiration
Pending order expiration date
price
The price of an order, deal or position, depending on the transaction
price_trigger
Stop price (trigger price) of a stop limit order
price_sl
Stop Loss price; it may refer to an order, deal, or position, depending on
the transaction
price_tp
Take Profit price; it may refer to an order, deal, or position, depending
on the transaction
volume
Volume in lots; it may indicate the current volume of the order, deal, or
position, depending on the transaction

---

## Page 1358

Part 6. Trading automation
1 358
6.4 Creating Expert Advisors
Field
Description
position
Ticket of the position affected by the transaction
position_by
Opposite position ticket
Some fields only make sense in certain cases. In particular, the time_ expiration field is filled for orders
with time_ type equal to the ORDER_TIME_SPECIFIED or ORDER_TIME_SPECIFIED_DAY  expiration
type. The price_ trigger field is reserved for stop-limit orders only (ORDER_TYPE_BUY_STOP_LIMIT and
ORDER_TYPE_SELL_STOP_LIMIT).
It is also obvious that position modifications operate on the position ticket (field position), but do not
use order or deal tickets. In addition, the position_ by field is reserved exclusively for closing a counter
position, that is, the one opened for the same instrument but in the opposite direction.
The defining characteristic for the analysis of a transaction is its type (field type). To describe it, the
MQL5 API introduces a special enumeration ENUM_TRADE_TRANSACTION_TYPE, which contains all
possible types of transactions.
Identifier
Description
TR AD E _TR AN S ACTIO N _O R D E R _AD D 
Adding a new order
TR AD E _TR AN S ACTIO N _O R D E R _U P D ATE 
Changing an active order
TR AD E _TR AN S ACTIO N _O R D E R _D E L E TE 
Deleting an active order
TR AD E _TR AN S ACTIO N _D E AL _AD D 
Adding a deal to history
TR AD E _TR AN S ACTIO N _D E AL _U P D ATE 
Changing a deal in history
TR AD E _TR AN S ACTIO N _D E AL _D E L E TE 
Deleting a deal from history
TR AD E _TR AN S ACTIO N _H IS TO R Y_AD D 
Adding an order to history as a result of execution or
cancellation
TR AD E _TR AN S ACTIO N _H IS TO R Y_U P D ATE 
Changing an order in history
TR AD E _TR AN S ACTIO N _H IS TO R Y_D E L E TE 
Deleting an order from history
TR AD E _TR AN S ACTIO N _P O S ITIO N 
Change a position
TR AD E _TR AN S ACTIO N _R E Q U E S T
Notification that a trade request has been processed by the
server and the result of its processing has been received
Let's provide some explanations.
In a transaction of the TRADE_TRANSACTION_ORDER_UPDATE type, order changes include not only
explicit changes on the part of the client terminal or trade server but also changes in its state (for
example, transition from the ORDER_STATE_STARTED state to ORDER_STATE_PLACED or from
ORDER_STATE_PLACED to ORDER_STATE_PARTIAL, etc.).

---

## Page 1359

Part 6. Trading automation
1 359
6.4 Creating Expert Advisors
During the TRADE_TRANSACTION_ORDER_DELETE transaction, an order can be deleted as a result of a
corresponding explicit request or execution (fill) on the server. In both cases, it will be transferred to
history and the transaction TRADE_TRANSACTION_HISTORY_ADD must also occur.
The TRADE_TRANSACTION_DEAL_ADD transaction is carried out not only as a result of order execution
but also as a result of transactions with the account balance.
Some transactions, such as TRADE_TRANSACTION_DEAL_UPDATE,
TRADE_TRANSACTION_DEAL_DELETE, TRADE_TRANSACTION_HISTORY_DELETE are quite rare
because they describe situations when a deal or order in the history is changed or deleted on the server
retroactively. This, as a rule, is a consequence of synchronization with an external trading system
(exchange).
It is important to note that adding or liquidating a position does not entail the appearance of the
TRADE_TRANSACTION_POSITION transaction. This type of transaction informs that the position has
been changed on the side of the trade server, programmatically or manually by the user. In particular,
a position can experience changes of the volume (partial opposite closing, reversal), opening price, as
well as Stop Loss and Take Profit levels. Some actions, such as refills, do not trigger this event.
All trade requests issued by MQL programs are reflected in TRADE_TRANSACTION_REQUEST
transactions, which allows analyzing their execution in a deferred way. This is especially important
when using the function OrderSendAsync, which immediately returns control to the calling code, so the
result is not known. At the same time, transactions are generated in the same way when using the
synchronous OrderSend function.
In addition, using the TRADE_TRANSACTION_REQUEST transactions, you can analyze the user's
trading actions from the terminal interface.
6.4.34 OnTradeTransaction event
Expert Advisors and indicators can receive notifications about trading events if their code contains a
special processing function OnTradeTransaction.
void OnTradeTransaction(const MqlTradeTransaction &trans,
   const MqlTradeRequest &request, const MqlTradeResult &result)
The first parameter is the MqlTradeTransaction structure described in the previous section. The second
and third parameters are structures MqlTradeRequest and MqlTradeResult, which were presented
earlier in the relevant sections.
The MqlTradeTransaction structure that describes the trade transaction is filled in differently depending
on the type of transaction specified in the type field. For example, for transactions of the
TRADE_TRANSACTION_REQUEST type, all other fields are not important, and to obtain additional
information, it is necessary to analyze the second and third parameters of the function (request and
result). Conversely, for all other types of transactions, the last two parameters of the function should
be ignored.
In case of TRADE_TRANSACTION_REQUEST, the request_ id field in the result variable contains an
identifier (through serial number), under which the trade request is registered in the terminal. This
number has nothing to do with order and deal tickets, as well as position identifiers. During each session
with the terminal, the numbering starts from the beginning (1 ). The presence of a request identifier
allows you to associate the performed action (calling OrderSend or OrderSendAsync functions) with the
result of this action passed to OnTradeTransaction. We'll look at examples later.

---

## Page 1360

Part 6. Trading automation
1 360
6.4 Creating Expert Advisors
For trading transactions related to active orders (TR AD E _TR AN SAC TION _OR D E R _AD D ,
TR AD E _TR AN SAC TION _OR D E R _U PD ATE  and TR AD E _TR AN SAC TION _OR D E R _D E L E TE ) and order history
(TR AD E _TR AN SAC TION _H ISTOR Y_AD D , TR AD E _TR AN SAC TION _H ISTOR Y_U PD ATE ,
TR AD E _TR AN SAC TION _H ISTOR Y_D E L E TE ), the following fields are filled in the MqlTradeTransaction
structure:
·order — order ticket
·symbol — name of the financial instrument in the order
·type — trade transaction type
·order_type — order type
·orders_state — current order state
·time_type — order expiration type
·time_expiration — order expiration time (for orders with ORDER_TIME_SPECIFIED and
ORDER_TIME_SPECIFIED_DAY expiration types)
·price — order price specified by the client/program
·price_trigger — stop price for triggering a stop-limit order (only for
ORDER_TYPE_BUY_STOP_LIMIT and ORDER_TYPE_SELL_STOP_LIMIT)
·price_sl — Stop Loss order price (filled if specified in the order)
·price_tp — Take Profit order price (filled if specified in the order)
·volume — current order volume (not executed), the initial order volume can be found from the
order history
·position — ticket of an open, modified, or closed position
·position_by — opposite position ticket (only for orders to close with opposite position)
For trading transactions related to deals ( TR AD E _TR AN SAC TION _D E AL _AD D ,
TR AD E _TR AN SAC TION _D E AL _U PD ATE  and TR AD E _TR AN SAC TION _D E AL _D E L E TE ), the following fields are filled
in the MqlTradeTransaction structure:
·deal — deal ticket
·order — order ticket on the basis of which the deal was made
·symbol — name of the financial instrument in the deal
·type — trade transaction type
·deal_type — deal type
·price — deal price
·price_sl — Stop Loss price (filled if specified in the order on the basis of which the deal was made)
·price_tp — Take Profit price (filled if specified in the order on the basis of which the deal was made)
·volume — deal volume
·position — ticket of an open, modified, or closed position
·position_by — opposite position ticket (for deals to close with opposite position)
For trading transactions related to position changes (TR AD E _TR AN SAC TION _POSITION ), the following
fields are filled in the MqlTradeTransaction structure:
·symbol — name of the financial instrument of the position
·type — trade transaction type

---

## Page 1361

Part 6. Trading automation
1 361 
6.4 Creating Expert Advisors
·deal_type — position type (DEAL_TYPE_BUY or DEAL_TYPE_SELL)
·price — weighted average position opening price
·price_sl — Stop Loss price
·price_tp — Take Profit price
·volume — position volume in lots
·position — position ticket
Not all available information on orders, deals and positions (for example, a comment) is transmitted in
the description of a trading transaction. For more information use the relevant functions: OrderGet,
HistoryOrderGet, HistoryDealGet and PositionGet.
One trade request sent from the terminal manually or through the trading functions
OrderSend/OrderSendAsync can generate several consecutive trade transactions on the trade server.
At the same time, the order in which notifications about these transactions arrive at the terminal is not
guaranteed, so you cannot build your trading algorithm on waiting for some trading transactions after
others.
Trading events are processed asynchronously, that is, delayed (in time) relative to the moment of
generation. Each trade event is sent to the queue of the MQL program, and the program sequentially
picks them up in the order of the queue.
When an Expert Advisor is processing trade transactions inside the OnTradeTransaction processor, the
terminal continues to accept incoming trade transactions. Thus, the state of the trading account may
change while OnTradeTransaction is running. In the future, the program will be notified of all these
events in the order the appear.
The length of the transaction queue is 1 024 elements. If OnTradeTransaction processes the next
transaction for too long, old transactions in the queue may be ousted by newer ones.
Due to parallel multi-threaded operation of the terminal with trading objects, by the time the
OnTradeTransaction handler is called, all the entities mentioned in it, including orders, deals, and
positions, may already be in a different state than that specified in the transaction properties. To get
their current state, you should select them in the current environment or in the history and request
their properties using the appropriate MQL5 functions.
Let's start with a simple Expert Advisor example TradeTransactions.mq5, which logs all
OnTradeTransaction trading events. Its only parameter DetailedLog allows you to optionally use classes
OrderMonitor, DealMonitor, PositionMonitor to display all properties. By default, the Expert Advisor
displays only the contents of the filled fields of the MqlTradeTransaction, MqlTradeRequest and
MqlTradeResult structures, coming to the handler in the form of parameters; at the same time request
and result are processed only for TRADE_TRANSACTION_REQUEST transactions.

---

## Page 1362

Part 6. Trading automation
1 362
6.4 Creating Expert Advisors
input bool DetailedLog = false; // DetailedLog ('true' shows order/deal/position details)
   
void OnTradeTransaction(const MqlTradeTransaction &transaction,
   const MqlTradeRequest &request,
   const MqlTradeResult &result)
{
   static ulong count = 0;
   PrintFormat(">>>% 6d", ++count);
   Print(TU::StringOf(transaction));
   
   if(transaction.type == TRADE_TRANSACTION_REQUEST)
   {
      Print(TU::StringOf(request));
      Print(TU::StringOf(result));
   }
   
   if(DetailedLog)
   {
      if(transaction.order != 0)
      {
         OrderMonitor m(transaction.order);
         m.print();
      }
      if(transaction.deal != 0)
      {
         DealMonitor m(transaction.deal);
         m.print();
      }
      if(transaction.position != 0)
      {
         PositionMonitor m(transaction.position);
         m.print();
      }
   }
}
Let's run it on the EURUSD chart and perform several actions manually, and the corresponding entries
will appear in the log (for the purity of the experiment, it is assumed that no one and nothing else
performs operations on the trading account, in particular, no other Expert Advisors are running).
Let's open a long position with a minimum lot.

---

## Page 1363

Part 6. Trading automation
1 363
6.4 Creating Expert Advisors
>>>      1
TRADE_TRANSACTION_ORDER_ADD, #=1296991463(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, »
   » @ 1.10947, V=0.01
>>>      2
TRADE_TRANSACTION_DEAL_ADD, D=1279627746(DEAL_TYPE_BUY), »
   » #=1296991463(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10947, V=0.01, P=1296991463
>>>      3
TRADE_TRANSACTION_ORDER_DELETE, #=1296991463(ORDER_TYPE_BUY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10947, P=1296991463
>>>      4
TRADE_TRANSACTION_HISTORY_ADD, #=1296991463(ORDER_TYPE_BUY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10947, P=1296991463
>>>      5
TRADE_TRANSACTION_REQUEST
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.10947, #=1296991463
DONE, D=1279627746, #=1296991463, V=0.01, @ 1.10947, Bid=1.10947, Ask=1.10947, Req=7
We will sell double the minimum lot.
>>>      6
TRADE_TRANSACTION_ORDER_ADD, #=1296992157(ORDER_TYPE_SELL/ORDER_STATE_STARTED), EURUSD, »
   » @ 1.10964, V=0.02
>>>      7
TRADE_TRANSACTION_DEAL_ADD, D=1279628463(DEAL_TYPE_SELL), »
   » #=1296992157(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10964, V=0.02, P=1296992157
>>>      8
TRADE_TRANSACTION_ORDER_DELETE, #=1296992157(ORDER_TYPE_SELL/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10964, P=1296992157
>>>      9
TRADE_TRANSACTION_HISTORY_ADD, #=1296992157(ORDER_TYPE_SELL/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10964, P=1296992157
>>>     10
TRADE_TRANSACTION_REQUEST
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_SELL, V=0.02, ORDER_FILLING_FOK, @ 1.10964, #=1296992157
DONE, D=1279628463, #=1296992157, V=0.02, @ 1.10964, Bid=1.10964, Ask=1.10964, Req=8
Let's perform the counter closing operation.

---

## Page 1364

Part 6. Trading automation
1 364
6.4 Creating Expert Advisors
>>>     11
TRADE_TRANSACTION_ORDER_ADD, #=1296992548(ORDER_TYPE_CLOSE_BY/ORDER_STATE_STARTED), EURUSD, »
   » @ 1.10964, V=0.01, P=1296991463, b=1296992157
>>>     12
TRADE_TRANSACTION_DEAL_ADD, D=1279628878(DEAL_TYPE_SELL), »
   » #=1296992548(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10964, V=0.01, P=1296991463
>>>     13
TRADE_TRANSACTION_POSITION, EURUSD, @ 1.10947, P=1296991463
>>>     14
TRADE_TRANSACTION_DEAL_ADD, D=1279628879(DEAL_TYPE_BUY), »
   » #=1296992548(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10947, V=0.01, P=1296992157
>>>     15
TRADE_TRANSACTION_ORDER_DELETE, #=1296992548(ORDER_TYPE_CLOSE_BY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10964, P=1296991463, b=1296992157
>>>     16
TRADE_TRANSACTION_HISTORY_ADD, #=1296992548(ORDER_TYPE_CLOSE_BY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10964, P=1296991463, b=1296992157
>>>     17
TRADE_TRANSACTION_REQUEST
TRADE_ACTION_CLOSE_BY, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, #=1296992548, »
   » P=1296991463, b=1296992157
DONE, D=1279628878, #=1296992548, V=0.01, @ 1.10964, Bid=1.10961, Ask=1.10965, Req=9
We still have a short position of the minimum lot. Let's close it.
>>>     18
TRADE_TRANSACTION_ORDER_ADD, #=1297002683(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, »
   » @ 1.10964, V=0.01, P=1296992157
>>>     19
TRADE_TRANSACTION_ORDER_DELETE, #=1297002683(ORDER_TYPE_BUY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10964, P=1296992157
>>>     20
TRADE_TRANSACTION_HISTORY_ADD, #=1297002683(ORDER_TYPE_BUY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10964, P=1296992157
>>>     21
TRADE_TRANSACTION_DEAL_ADD, D=1279639132(DEAL_TYPE_BUY), »
   » #=1297002683(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10964, V=0.01, P=1296992157
>>>     22
TRADE_TRANSACTION_REQUEST
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.10964, #=1297002683, »
   » P=1296992157
DONE, D=1279639132, #=1297002683, V=0.01, @ 1.10964, Bid=1.10964, Ask=1.10964, Req=10
If you wish, you can enable the DetailedLog option to log all properties of trading objects at the
moment of event processing. In a detailed log, you can notice discrepancies between the state of
objects stored in the transaction structure (at the time of its initiation) and the current state. For
example, when adding an order to close a position (opposite or normal), a ticket is specified in the
transaction, according to which the monitor object will no longer be able to read anything, since the
position has been deleted. As a result, we will see lines like this in the log:

---

## Page 1365

Part 6. Trading automation
1 365
6.4 Creating Expert Advisors
TRADE_TRANSACTION_ORDER_ADD, #=1297777749(ORDER_TYPE_CLOSE_BY/ORDER_STATE_STARTED), EURUSD, »
   » @ 1.10953, V=0.01, P=1297774881, b=1297776850
...
Error: PositionSelectByTicket(1297774881) failed: TRADE_POSITION_NOT_FOUND
Let's restart the Expert Advisor TradeTransaction.mq5 to reset the logged events for the next test. This
time we will use default settings (no details).
Now let's try to perform trading actions programmatically in the new Expert Advisor
OrderSendTransaction1 .mq5, and at the same time describe our OnTradeTransaction handler in it (same
as in the previous example).
This Expert Advisor allows you to select the trade direction and volume: if you leave it at zero, the
minimum lot of the current symbol is used by default. Also in the parameters there is a distance to the
protective levels in points. The market is entered with the specified parameters, there is a 5 second
pause between the setting of Stop Loss and Take Profit, and then closing the position, so that the user
can intervene (for example, edit the stop loss manually), although this is not necessary, since we have
already made sure that manual operations are intercepted by the program.
enum ENUM_ORDER_TYPE_MARKET
{
   MARKET_BUY = ORDER_TYPE_BUY,    // ORDER_TYPE_BUY
   MARKET_SELL = ORDER_TYPE_SELL   // ORDER_TYPE_SELL
};
   
input ENUM_ORDER_TYPE_MARKET Type;
input double Volume;               // Volume (0 - minimal lot)
input uint Distance2SLTP = 1000;
The strategy is launched once, for which a 1 -second timer is used, which is turned off in its own
handler.
int OnInit()
{
   EventSetTimer(1);
   return INIT_SUCCEEDED;
}
   
void OnTimer()
{
   EventKillTimer();
   ...
All actions are performed through an already familiar MqlTradeRequestSync structure with advanced
features (MqlTradeSync.mqh): implicit initialization of fields with correct values, buy/sell methods for
market orders, adj ust for protective levels, and close for closing the position.
Step 1 :

---

## Page 1366

Part 6. Trading automation
1 366
6.4 Creating Expert Advisors
   MqlTradeRequestSync request;
   
   const double volume = Volume == 0 ?
      SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) : Volume;
   
   Print("Start trade");
   const ulong order = (Type == MARKET_BUY ? request.buy(volume) : request.sell(volume));
   if(order == 0 || !request.completed())
   {
      Print("Failed Open");
      return;
   }
   
   Print("OK Open");
Step 2:
   Sleep(5000); // wait 5 seconds (user can edit position)
   Print("SL/TP modification");
   const double price = PositionGetDouble(POSITION_PRICE_OPEN);
   const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   TU::TradeDirection dir((ENUM_ORDER_TYPE)Type);
   const double SL = dir.negative(price, Distance2SLTP * point);
   const double TP = dir.positive(price, Distance2SLTP * point);
   if(request.adjust(SL, TP) && request.completed())
   {
      Print("OK Adjust");
   }
   else
   {
      Print("Failed Adjust");
   }
Step 3:
   Sleep(5000); // wait another 5 seconds
   Print("Close down");
   if(request.close(request.result.position) && request.completed())
   {
      Print("Finish");
   }
   else
   {
      Print("Failed Close");
   }
}
Intermediate waits not only make it possible to have time to consider the process, but also
demonstrate an important aspect of MQL5 programming, which is single-threading. While our trading
Expert Advisor is inside OnTimer, trading events generated by the terminal are accumulated in its queue
and will be forwarded to the internal OnTradeTransaction handler in a deferred style, only after the exit
from OnTimer.

---

## Page 1367

Part 6. Trading automation
1 367
6.4 Creating Expert Advisors
At the same time, the TradeTransactions Expert Advisor running in parallel is not busy with any
calculations and will receive trading events as quickly as possible.
The result of the execution of two Expert Advisors is presented in the following log with timing (for
brevity OrderSendTransaction1  tagged as OS1 , and Trade Transactions tagged as TTs).

---

## Page 1368

Part 6. Trading automation
1 368
6.4 Creating Expert Advisors
19:09:08.078  OS1  Start trade
19:09:08.109  TTs  >>>     1
19:09:08.125  TTs  TRADE_TRANSACTION_ORDER_ADD, #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_STARTED), »
                   EURUSD, @ 1.10913, V=0.01
19:09:08.125  TTs  >>>     2
19:09:08.125  TTs  TRADE_TRANSACTION_DEAL_ADD, D=1280661362(DEAL_TYPE_BUY), »
                   #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10913, V=0.01, »
                   P=1298021794
19:09:08.125  TTs  >>>     3
19:09:08.125  TTs  TRADE_TRANSACTION_ORDER_DELETE, #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10913, P=1298021794
19:09:08.125  TTs  >>>     4
19:09:08.125  TTs  TRADE_TRANSACTION_HISTORY_ADD, #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10913, P=1298021794
19:09:08.125  TTs  >>>     5
19:09:08.125  TTs  TRADE_TRANSACTION_REQUEST
19:09:08.125  TTs  TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.10913, »
                   D=10, #=1298021794, M=1234567890
19:09:08.125  TTs  DONE, D=1280661362, #=1298021794, V=0.01, @ 1.10913, Bid=1.10913, Ask=1.10913, »
                   Req=9
19:09:08.125  OS1  Waiting for position for deal D=1280661362
19:09:08.125  OS1  OK Open
19:09:13.133  OS1  SL/TP modification
19:09:13.164  TTs  >>>     6
19:09:13.164  TTs  TRADE_TRANSACTION_POSITION, EURUSD, @ 1.10913, SL=1.09913, TP=1.11913, V=0.01, »
                   P=1298021794
19:09:13.164  OS1  OK Adjust
19:09:13.164  TTs  >>>     7
19:09:13.164  TTs  TRADE_TRANSACTION_REQUEST
19:09:13.164  TTs  TRADE_ACTION_SLTP, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, SL=1.09913, »
                   TP=1.11913, D=10, P=1298021794, M=1234567890
19:09:13.164  TTs  DONE, Req=10
19:09:18.171  OS1  Close down
19:09:18.187  OS1  Finish
19:09:18.218  TTs  >>>     8
19:09:18.218  TTs  TRADE_TRANSACTION_ORDER_ADD, #=1298022443(ORDER_TYPE_SELL/ORDER_STATE_STARTED), »
                   EURUSD, @ 1.10901, V=0.01, P=1298021794
19:09:18.218  TTs  >>>     9
19:09:18.218  TTs  TRADE_TRANSACTION_DEAL_ADD, D=1280661967(DEAL_TYPE_SELL), »
                   #=1298022443(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10901, »
                   SL=1.09913, TP=1.11913, V=0.01, P=1298021794
19:09:18.218  TTs  >>>    10
19:09:18.218  TTs  TRADE_TRANSACTION_ORDER_DELETE, #=1298022443(ORDER_TYPE_SELL/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10901, P=1298021794
19:09:18.218  TTs  >>>    11
19:09:18.218  TTs  TRADE_TRANSACTION_HISTORY_ADD, #=1298022443(ORDER_TYPE_SELL/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10901, P=1298021794
19:09:18.218  TTs  >>>    12
19:09:18.218  TTs  TRADE_TRANSACTION_REQUEST
19:09:18.218  TTs  TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_SELL, V=0.01, ORDER_FILLING_FOK, @ 1.10901, »
                   D=10, #=1298022443, P=1298021794, M=1234567890
19:09:18.218  TTs  DONE, D=1280661967, #=1298022443, V=0.01, @ 1.10901, Bid=1.10901, Ask=1.10901, »
                   Req=11
19:09:18.218  OS1  >>>     1

---

## Page 1369

Part 6. Trading automation
1 369
6.4 Creating Expert Advisors
19:09:18.218  OS1  TRADE_TRANSACTION_ORDER_ADD, #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_STARTED), »
                   EURUSD, @ 1.10913, V=0.01
19:09:18.218  OS1  >>>     2
19:09:18.218  OS1  TRADE_TRANSACTION_DEAL_ADD, D=1280661362(DEAL_TYPE_BUY), »
                   #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, »
                   @ 1.10913, V=0.01, P=1298021794
19:09:18.218  OS1  >>>     3
19:09:18.218  OS1  TRADE_TRANSACTION_ORDER_DELETE, #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10913, P=1298021794
19:09:18.218  OS1  >>>     4
19:09:18.218  OS1  TRADE_TRANSACTION_HISTORY_ADD, #=1298021794(ORDER_TYPE_BUY/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10913, P=1298021794
19:09:18.218  OS1  >>>     5
19:09:18.218  OS1  TRADE_TRANSACTION_REQUEST
19:09:18.218  OS1  TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.10913, »
                   D=10, #=1298021794, M=1234567890
19:09:18.218  OS1  DONE, D=1280661362, #=1298021794, V=0.01, @ 1.10913, Bid=1.10913, Ask=1.10913, »
                   Req=9
19:09:18.218  OS1  >>>     6
19:09:18.218  OS1  TRADE_TRANSACTION_POSITION, EURUSD, @ 1.10913, SL=1.09913, TP=1.11913, V=0.01, »
                   P=1298021794
19:09:18.218  OS1  >>>     7
19:09:18.218  OS1  TRADE_TRANSACTION_REQUEST
19:09:18.218  OS1  TRADE_ACTION_SLTP, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, »
                   SL=1.09913, TP=1.11913, D=10, P=1298021794, M=1234567890
19:09:18.218  OS1  DONE, Req=10
19:09:18.218  OS1  >>>     8
19:09:18.218  OS1  TRADE_TRANSACTION_ORDER_ADD, #=1298022443(ORDER_TYPE_SELL/ORDER_STATE_STARTED), »
                   EURUSD, @ 1.10901, V=0.01, P=1298021794
19:09:18.218  OS1  >>>     9
19:09:18.218  OS1  TRADE_TRANSACTION_DEAL_ADD, D=1280661967(DEAL_TYPE_SELL), »
                   #=1298022443(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10901, »
                   SL=1.09913, TP=1.11913, V=0.01, P=1298021794
19:09:18.218  OS1  >>>    10
19:09:18.218  OS1  TRADE_TRANSACTION_ORDER_DELETE, #=1298022443(ORDER_TYPE_SELL/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10901, P=1298021794
19:09:18.218  OS1  >>>    11
19:09:18.218  OS1  TRADE_TRANSACTION_HISTORY_ADD, #=1298022443(ORDER_TYPE_SELL/ORDER_STATE_FILLED), »
                   EURUSD, @ 1.10901, P=1298021794
19:09:18.218  OS1  >>>    12
19:09:18.218  OS1  TRADE_TRANSACTION_REQUEST
19:09:18.218  OS1  TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_SELL, V=0.01, ORDER_FILLING_FOK, @ 1.10901, »
                   D=10, #=1298022443, P=1298021794, M=1234567890
19:09:18.218  OS1  DONE, D=1280661967, #=1298022443, V=0.01, @ 1.10901, Bid=1.10901, Ask=1.10901, »
                   Req=11
The numbering of events in the programs is the same (provided that they are started cleanly, as
recommended). Note that the same event is printed first from TTs immediately after the request is
executed, and the second time only at the end of the test, where, in fact, all events are output from
the queue to the OS1 .
If we remove artificial delays, the script will, of course, run faster, but still the OnTradeTransaction
handler will receive notifications (multiple times) after all three steps, not after each respective
request. How critical it is?

---

## Page 1370

Part 6. Trading automation
1 370
6.4 Creating Expert Advisors
Now the examples use our modification of the structure MqlTradeRequestSync, purposefully using the
synchronous option OrderSend, which also implements a universal completed method which checks if
the request completed successfully. With this control, we can set protective levels for a position,
because we know how to wait for its ticket to appear. Within the framework of such a synchronous
concept (adopted for the sake of convenience), we do not need to analyze query results in
OnTradeTransaction. However, this is not always the case.
When an Expert Advisor needs to send many requests at once, as in the case of the example with
setting a grid of orders PendingOrderGrid2.mq5 discussed in the section on position properties, waiting
for each position or order to be "ready" may reduce the overall performance of the Expert Advisor. In
such cases, it is recommended to use the OrderSendAsync function. But if successful, it fills only the
request_ id field in the MqlTradeResult, with which you then need to track the appearance of orders,
deals and positions in OnTradeTransaction.
One of the most obvious but not particularly elegant tricks for implementing this scheme is to store the
identifiers of requests or entire structures of the requests being sent in an array, in the global context.
These identifiers can then be looked up in incoming transactions in OnTradeTransaction, the tickets can
be found in the MqlTradeResult parameter and further actions can be taken. As a result, the trading
logic is separated into different functions. For example, in the context of the last Expert Advisor
OrderSendTransaction1 .mq5 this "diversification" lies in the fact that after sending the first order, the
code fragments must be transferred to OnTradeTransaction and checked for the following:
·transaction type in MqlTradeTransaction (transaction type);
·request type in MqlTradeRequest (request action);
·request id in MqlTradeResult (result.request_ id);
All this should be supplemented with specific applied logic (for example, checking for the existence of a
position), which provides branching by trading strategy states. A little later we will make a similar
modification of the OrderSendTransaction Expert Advisor under a different number to visually show the
amount of additional source code. And then we will offer a way to organize the program more linearly,
but without abandoning transactional events.
For now, we only note that the developer should choose whether to build an algorithm around
OnTradeTransaction or without it. In many cases, when bulk sending of orders is not needed, it is
possible to stay in the synchronous programming paradigm. However, OnTradeTransaction is the most
practical way to control the triggering of pending orders and protective levels, as well as other events
generated by the server. After a little preparation, we will present two relevant examples: the final
modification of the grid Expert Advvisor and the implementation of the popular setup of two OCO (One
Cancels Other) orders (see the section On Trade).
An alternative to application of OnTradeTransaction consists in periodic analysis of the trading
environment, that is, in fact, in remembering the number of orders and positions and looking for
changes among them. This approach is suitable for strategies based on schedules or allowing certain
time delays.
We emphasize again that the use of OnTradeTransaction does not mean that the program must
necessarily switch from OrderSend on OrderSendAsync: You can use either variety or both. Recall that
the OrderSend function is also not quite synchronous, as it returns, at best, the ticket of the order and
the deal but not the position. Soon we will be able to measure the execution time of a batch of orders
within the same grid strategy using both variants of the function: OrderSend and OrderSendAsync.
To unify the development of synchronous and asynchronous programs, it would be great to support
OrderSendAsync in our structure MqlTradeRequestSync (despite its name). This can be done with just a

---

## Page 1371

Part 6. Trading automation
1 371 
6.4 Creating Expert Advisors
couple of corrections. First, you need to replace all currently existing calls OrderSend to your own
method orderSend, and in it switch the call to OrderSend or OrderSendAsync depending on a flag.
struct MqlTradeRequestSync: public MqlTradeRequest
{
   ...
   static bool AsyncEnabled;
   ...
private:
   bool orderSend(const MqlTradeRequest &req, MqlTradeResult &res)
   {
      return AsyncEnabled ? ::OrderSendAsync(req, res) : ::OrderSend(req, res);
   }
};
By setting the AsyncEnabled public variable to true or false, you can switch from one mode to another,
for example, in the code fragment where mass orders are sent. 
Second, those methods of the structure that returned a ticket (for example, for entering the market)
you should return the request_ id field instead of order. For example, inside the methods _ pending and
_ market we had the following operator:
if(OrderSend(this, result)) return result.order;
Now it is replaced by:
if(orderSend(this, result)) return result.order ? result.order :
   (result.retcode == TRADE_RETCODE_PLACED ? result.request_id : 0);
Of course, when asynchronous mode is enabled, we can no longer use the completed method to wait for
the query results to be ready immediately after it is sent. But this method is, basically, optional: you
can just drop it even when working through OrderSend.
So, taking into account the new modification of the MqlTradeSync.mqh file, let's create
OrderSendTransaction2.mq5.
This Expert Advisor will send the initial request as before from OnTimer, while setting protective levels
and closing a position in OnTradeTransaction step by step. Although we will not have an artificial delay
between the stages this time, the sequence of states itself is standard for many Expert Advisors:
opened a position, modified, closed (if certain market conditions are met, which are left behind the
scenes here).
Two global variables will allow you to track the state: RequestID with the id of the last request sent (the
result of which we expect) and Position Ticket with an open position ticket. When there the position did
not appear yet, or no longer exists, the ticket is equal to 0.
uint RequestID = 0;
ulong PositionTicket = 0;
Asynchronous mode is enabled in the OnInit handler.

---

## Page 1372

Part 6. Trading automation
1 372
6.4 Creating Expert Advisors
int OnInit()
{
   ...
   MqlTradeRequestSync::AsyncEnabled = true;
   ...
}
The OnTimer function is now much shorter.
void OnTimer()
{
   ...
   // send a request TRADE_ACTION_DEAL (asynchronously!)
   const ulong order = (Type == MARKET_BUY ? request.buy(volume) : request.sell(volume));
   if(order) // in asynchronous mode this is now request_id
   {
      Print("OK Open?");
      RequestID = request.result.request_id; // same as order
   }
   else
   {
      Print("Failed Open");
   }
}
On successful completion of the request, we get only request_ id and store it in the RequestID variable.
The status print now contains a question mark, like "OK Open?", because the actual result is not yet
known.
OnTradeTransaction became significantly more complicated due to the verification of the results and
the execution of subsequent trading orders according to the conditions. Let's consider it gradually.
In this case, the entire trading logic has moved into the branch for transactions of the
TRADE_TRANSACTION_REQUEST type. Of course, the developer can use other types if desired, but we
use this one because it contains information in the form of a familiar structure MqlTradeResult, i.e., this
sort of represents a delayed ending of an asynchronous call OrderSendAsync.

---

## Page 1373

Part 6. Trading automation
1 373
6.4 Creating Expert Advisors
void OnTradeTransaction(const MqlTradeTransaction &transaction,
   const MqlTradeRequest &request,
   const MqlTradeResult &result)
{
   static ulong count = 0;
   PrintFormat(">>>% 6d", ++count);
   Print(TU::StringOf(transaction));
   
   if(transaction.type == TRADE_TRANSACTION_REQUEST)
   {
      Print(TU::StringOf(request));
      Print(TU::StringOf(result));
      
      ...
      // here is the whole algorithm
   }
}
We should only be interested in requests with the ID we expect. So the next statement will be nested if.
In its block, we describe the MqlTradeRequestSync object in advance, because it will be necessary to
send regular trade requests according to the plan.
      if(result.request_id == RequestID)
      {
         MqlTradeRequestSync next;
         next.magic = Magic;
         next.deviation = Deviation;
         ...
      }
We have only two working request types, so we add one more nested if one for them.
         if(request.action == TRADE_ACTION_DEAL)
         {
            ... // here is the reaction to opening and closing a position
         }
         else if(request.action == TRADE_ACTION_SLTP)
         {
            ... // here is the reaction to setting SLTP for an open position
         }
Please note that TRADE_ACTION_DEAL is used for both opening and closing a position, and therefore
one more if is required, in which we will distinguish between these two states depending on the value of
the PositionTicket variable.

---

## Page 1374

Part 6. Trading automation
1 374
6.4 Creating Expert Advisors
            if(PositionTicket == 0)
            {
               ... // there is no position, so this is an opening notification 
            }
            else
            {
               ... // there is a position, so this is a closure
            }
There are no position increases (for netting) or multiple positions (for hedging) in the trading strategy
under consideration, which is why this part is logically simple. Real Expert Advisors will require much
more different estimates of intermediate states.
In the case of a position opening notification, the block of code looks like this:
            if(PositionTicket == 0)
            {
               // trying to get results from the transaction: select an order by ticket
               if(!HistoryOrderSelect(result.order))
               {
                  Print("Can't select order in history");
                  RequestID = 0;
                  return;
               }
               // get position ID and ticket
               const ulong posid = HistoryOrderGetInteger(result.order, ORDER_POSITION_ID);
               PositionTicket = TU::PositionSelectById(posid);
               ...
For simplicity, we have omitted error and requote checking here. You can see an example of their
handling in the attached source code. Recall that all these checks have already been implemented in
the methods of the MqlTradeRequestSync structure, but they only work in synchronous mode, and
therefore we have to repeat them explicitly.
The next code fragment for setting protective levels has not changed much.

---

## Page 1375

Part 6. Trading automation
1 375
6.4 Creating Expert Advisors
            if(PositionTicket == 0)
            {
               ...
               const double price = PositionGetDouble(POSITION_PRICE_OPEN);
               const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
               TU::TradeDirection dir((ENUM_ORDER_TYPE)Type);
               const double SL = dir.negative(price, Distance2SLTP * point);
               const double TP = dir.positive(price, Distance2SLTP * point);
               // sending TRADE_ACTION_SLTP request (asynchronously!)
               if(next.adjust(PositionTicket, SL, TP))
               {
                  Print("OK Adjust?");
                  RequestID = next.result.request_id;
               }
               else
               {
                  Print("Failed Adjust");
                  RequestID = 0;
               }
            }
The only difference here is: we fill the RequestID variable with ID of the new TRADE_ACTION_SLTP
request.
Receiving a notification about a deal with a non-zero PositionTicket implies that the position has been
closed.
            if(PositionTicket == 0)
            {
               ... // see above
            }
            else
            {
               if(!PositionSelectByTicket(PositionTicket))
               {
                  Print("Finish");
                  RequestID = 0;
                  PositionTicket = 0;
               }
            }
In case of successful deletion, the position cannot be selected using PositionSelectByTicket, so we reset
RequestID and PositionTicket. The Expert Advisor then returns to its initial state and is ready to make
the next buy/sell-modify-close cycle.
It remains for us to consider sending a request to close the position. In our simplified to a minimum
strategy, this happens immediately after the successful modification of the protective levels.

---

## Page 1376

Part 6. Trading automation
1 376
6.4 Creating Expert Advisors
         if(request.action == TRADE_ACTION_DEAL)
         {
            ... // see above
         }
         else if(request.action == TRADE_ACTION_SLTP)
         {
            // send a TRADE_ACTION_DEAL request to close (asynchronously!)
            if(next.close(PositionTicket))
            {
               Print("OK Close?");
               RequestID = next.result.request_id;
            }
            else
            {
               PrintFormat("Failed Close %lld", PositionTicket);
            }
         }
That's the whole function OnTradeTransaction. The Expert Advisor is ready.
Let's run OrderSendTransaction2.mq5 with default settings on EURUSD. Below is an example log.

---

## Page 1377

Part 6. Trading automation
1 377
6.4 Creating Expert Advisors
Start trade
OK Open?
>>>     1
TRADE_TRANSACTION_ORDER_ADD, #=1299508203(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, »
   » @ 1.10640, V=0.01
>>>     2
TRADE_TRANSACTION_DEAL_ADD, D=1282135720(DEAL_TYPE_BUY), »
   » #=1299508203(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10640, V=0.01, P=1299508203
>>>     3
TRADE_TRANSACTION_ORDER_DELETE, #=1299508203(ORDER_TYPE_BUY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10640, P=1299508203
>>>     4
TRADE_TRANSACTION_HISTORY_ADD, #=1299508203(ORDER_TYPE_BUY/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10640, P=1299508203
>>>     5
TRADE_TRANSACTION_REQUEST
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, @ 1.10640, D=10, »
   » #=1299508203, M=1234567890
DONE, D=1282135720, #=1299508203, V=0.01, @ 1.1064, Bid=1.1064, Ask=1.1064, Req=7
OK Adjust?
>>>     6
TRADE_TRANSACTION_POSITION, EURUSD, @ 1.10640, SL=1.09640, TP=1.11640, V=0.01, P=1299508203
>>>     7
TRADE_TRANSACTION_REQUEST
TRADE_ACTION_SLTP, EURUSD, ORDER_TYPE_BUY, V=0.01, ORDER_FILLING_FOK, SL=1.09640, TP=1.11640, »
   » D=10, P=1299508203, M=1234567890
DONE, Req=8
OK Close?
>>>     8
TRADE_TRANSACTION_ORDER_ADD, #=1299508215(ORDER_TYPE_SELL/ORDER_STATE_STARTED), EURUSD, »
   » @ 1.10638, V=0.01, P=1299508203
>>>     9
TRADE_TRANSACTION_ORDER_DELETE, #=1299508215(ORDER_TYPE_SELL/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10638, P=1299508203
>>>    10
TRADE_TRANSACTION_HISTORY_ADD, #=1299508215(ORDER_TYPE_SELL/ORDER_STATE_FILLED), EURUSD, »
   » @ 1.10638, P=1299508203
>>>    11
TRADE_TRANSACTION_DEAL_ADD, D=1282135730(DEAL_TYPE_SELL), »
   » #=1299508215(ORDER_TYPE_BUY/ORDER_STATE_STARTED), EURUSD, @ 1.10638, »
   » SL=1.09640, TP=1.11640, V=0.01, P=1299508203
>>>    12
TRADE_TRANSACTION_REQUEST
TRADE_ACTION_DEAL, EURUSD, ORDER_TYPE_SELL, V=0.01, ORDER_FILLING_FOK, @ 1.10638, D=10, »
   » #=1299508215, P=1299508203, M=1234567890
DONE, D=1282135730, #=1299508215, V=0.01, @ 1.10638, Bid=1.10638, Ask=1.10638, Req=9
Finish
The trading logic is working as expected, and transaction events arrive strictly after each next order is
sent. If we now run our new Expert Advisor and the transactions interceptor TradeTransactions.mq5 in
parallel, log messages from two Expert Advisors will appear synchronously.
However, a remake from the first straight version OrderSendTransaction1 .mq5 to an asynchronous
second version OrderSendTransaction2.mq5 required significantly more sophisticated code. The

---

## Page 1378

Part 6. Trading automation
1 378
6.4 Creating Expert Advisors
question arises: is it possible to somehow combine the principles of sequential description of trading
logic (code transparency) and parallel processing (speed)?
In theory, this is possible, but it will require at some point to spend time to work on creating some kind
of auxiliary mechanism.
6.4.35 Synchronous and asynchronous requests
Before going into details, let's remind you that each MQL program is executed in its own thread, and
therefore parallel asynchronous processing of transactions (and other events) is only possible due to
the fact that another MQL program would be doing it. At the same time, it is necessary to ensure
information exchange between programs. We already know a couple of ways to do this: global variables
of the terminal and files. In Part 7 of the book, we will explore other features such as graphical
resources and databases.
Indeed, imagine that an Expert Advisor similar to TradeTransactions.mq5 runs in parallel with the
trading Expert Advisor and saves the received transactions (not necessarily all fields, but only selective
ones that affect decision-making) in global variables. Then the Expert Advisor could check the global
variables immediately after sending the next request and read the results from them without leaving the
current function. Moreover, it does not need its own OnTradeTransaction handler.
However, it is not easy to organize the running of a third-party Expert Advisor. From the technical point
of view, this could be done by creating a chart object and applying a template with a predefined
transaction monitor Expert Advisor. But there is an easier way. The point is that events of
OnTradeTransaction are translated not only into Expert Advisor but also into indicators. In turn, an
indicator is the most easily launched type of MQL program: it is enough to call iCustom.
In addition, the use of the indicator gives one more nice bonus: it can describe the indicator buffer
available from external programs via CopyBuffer, and arrange a ring buffer in it for storing transactions
coming from the terminal (request results). Thus, there is no need to mess with global variables.
Attention! The OnTradeTransaction event is not generated for indicators in the tester, so you can
only check the operation of the Expert Advisor-indicator pair online.
Let's call this indicator TradeTransactionRelay.mq5 and describe one buffer in it. It could be made
invisible because it will write data that cannot be rendered, but we left it visible to prove the concept.
#property indicator_chart_window
#property indicator_buffers 1
#property indicator_plots   1
   
double Buffer[];
   
void OnInit()
{
   SetIndexBuffer(0, Buffer, INDICATOR_DATA);
}
The OnCalculate handler is empty.

---

## Page 1379

Part 6. Trading automation
1 379
6.4 Creating Expert Advisors
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
   return rates_total;
}
In the code, we need a ready converter from double to ulong and vice versa, since buffer cells can
corrupt large ulong values if they are written there using a simple typecast (see Real numbers).
#include <MQL5Book/ConverterT.mqh>
Converter<ulong,double> cnv;
Here is the OnTradeTransaction function.
#defineFIELD_NUM6// the most important fields in MqlTradeResult
   
void OnTradeTransaction(const MqlTradeTransaction &transaction,
   const MqlTradeRequest &request,
   const MqlTradeResult &result)
{
   if(transaction.type == TRADE_TRANSACTION_REQUEST)
   {
      ArraySetAsSeries(Buffer, true);
      
      // store FIELD_NUM result fields into consecutive buffer cells
      const int offset = (int)((result.request_id * FIELD_NUM)
         % (Bars(_Symbol, _Period) / FIELD_NUM * FIELD_NUM));
      Buffer[offset + 1] = result.retcode;
      Buffer[offset + 2] = cnv[result.deal];
      Buffer[offset + 3] = cnv[result.order];
      Buffer[offset + 4] = result.volume;
      Buffer[offset + 5] = result.price;
      // this assignment must come last,
      // because it is the result ready flag
      Buffer[offset + 0] = result.request_id;
   }
}
We decided to keep only the six most important fields of the MqlTradeResult structure. If desired, you
can extend the mechanism to the entire structure, but to transfer the string field comment you will
need an array of characters for which you will have to reserve quite a lot of elements.
Thus, each result now occupies six consecutive buffer cells. The index of the first cell of these six is
determined based on the request ID: this number is simply multiplied by 6. Since there can be many
requests, the entry works on the principle of a ring buffer, i.e., the resulting index is normalized by
dividing with remainder ('%') by the size of the indicator buffer, which is the number of bars rounded up
to 6. When the request numbers exceed the size, the record will go in a circle from the initial elements.
Since the numbering of bars is affected by the formation of new bars, it is recommended to put the
indicator on large timeframes, such as D1 . Then only at the beginning of the day is it likely (yet rather
unlikely) the situation when the numbering of bars in the indicator will shift directly during the

---

## Page 1380

Part 6. Trading automation
1 380
6.4 Creating Expert Advisors
processing of the next transaction, and then the results recorded by the indicator will not be read by
the Expert Advisor (one transaction may be missed).
The indicator is ready. Now let's start implementing a new modification of the test Expert Advisor
OrderSendTransaction3.mq5 (hooray, this is its latest version). Let's describe the handle variable for the
indicator handle and create the indicator in OnInit.
int handle = 0;
   
int OnInit()
{
   ...
   const static string indicator = "MQL5Book/p6/TradeTransactionRelay";
   handle = iCustom(_Symbol, PERIOD_D1, indicator);
   if(handle == INVALID_HANDLE)
   {
      Alert("Can't start indicator ", indicator);
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}
To read query results from the indicator buffer, let's prepare a helper function AwaitAsync. As its first
parameter, it receives a reference to the MqlTradeRequestSync structure. If successful, the results
obtained from the indicator buffer with handle will be written to this structure. The identifier of the
request we are interested in should already be in the nested structure, in the result.request_ id field. Of
course, here we must read the data according to the same principle, that is, in six bars.

---

## Page 1381

Part 6. Trading automation
1 381 
6.4 Creating Expert Advisors
#define FIELD_NUM   6  // the most important fields in MqlTradeResult
#define TIMEOUT  1000  // 1 second
   
bool AwaitAsync(MqlTradeRequestSync &r, const int _handle)
{
   Converter<ulong,double> cnv;
   const int offset = (int)((r.result.request_id * FIELD_NUM)
      % (Bars(_Symbol, _Period) / FIELD_NUM * FIELD_NUM));
   const uint start = GetTickCount();
   // wait for results or timeout
   while(!IsStopped() && GetTickCount() - start < TIMEOUT)
   {
      double array[];
      if((CopyBuffer(_handle, 0, offset, FIELD_NUM, array)) == FIELD_NUM)
      {
         ArraySetAsSeries(array, true);
         // when request_id is found, fill other fields with results
         if((uint)MathRound(array[0]) == r.result.request_id)
         {
            r.result.retcode = (uint)MathRound(array[1]);
            r.result.deal = cnv[array[2]];
            r.result.order = cnv[array[3]];
            r.result.volume = array[4];
            r.result.price = array[5];
            PrintFormat("Got Req=%d at %d ms",
               r.result.request_id, GetTickCount() - start);
            Print(TU::StringOf(r.result));
            return true;
         }
      }
   }
   Print("Timeout for: ");
   Print(TU::StringOf(r));
   return false;
}
Now that we have this function, let's write a trading algorithm in an asynchronous-synchronous style:
as a direct sequence of steps, each of which waits for the previous one to be ready due to notifications
from the parallel indicator program while remaining inside one function.