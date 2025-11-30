# t5copier.mq5

Source: https://www.mql5.com/en/code/download/32614/t5copier.mq5

CTrade m\_trade;
CPosition m\_position;
#else // Use standard libraries that come with the default terminal installation
#include 
CTrade m\_trade;
CPositionInfo m\_position;
#endif
//--- program running modes.
enum Mode {Server, Client};
//---
sinput group "=== settings ===";
sinput Mode program\_mode = Server; // Application program\_mode.
sinput string server\_id = "LetCashFlow"; // Server identifier.(Unique per server instance)
sinput double copy\_factor = 100; // Copy factor (100% means copy as is)
sinput int max\_slippage = 20; // Maximum slippage
sinput bool no\_try = true; // Do not open m\_trade if slippage is large?
//--- store corresponding received ticket with opened one, to track closing
struct pair\_ticket {
ulong sr\_ticket, cl\_ticket;
pair\_ticket(void):
sr\_ticket(0), cl\_ticket(0) {};
};
//---
static int fhandle = INVALID\_HANDLE;
const string fname = "Server++/" + server\_id;
ulong hash\_at\_connection = NULL;
ulong open\_tickets[], expired\_tickets[];
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
int OnInit() {
if(program\_mode == Server) {
if(FileIsExist("Server++/delete", FILE\_COMMON))
if(!FileDelete("Server++/delete", FILE\_COMMON)) {
Alert("Server error!");
return 97;
}
if(FileIsExist(fname, FILE\_COMMON))
if(!FileDelete(fname, FILE\_COMMON)) {
Alert(StringFormat("Please terminate the server instance at [%s] and try again!", server\_id));
return 98;
}
//---
fhandle = FileOpen(fname, server\_flags);
if(fhandle == INVALID\_HANDLE) {
Alert(StringFormat("Failed setting up server at address [%s]", server\_id));
return 99;
}
} else {
if(FileIsExist("Server++/delete", FILE\_COMMON))
if(!FileDelete("Server++/delete", FILE\_COMMON)) {
Alert("Server error!");
return 100;
}
int error = 0;
if(!reconnect(server\_id, error))
return error;
}
Alert(StringFormat("%s launched successfully!", EnumToString(program\_mode)));
if(program\_mode == Client)
EventSetTimer(1);
//---
return 0;
}
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
Comment("");
FileClose(fhandle);
Print(StringFormat("%s shutting down...", EnumToString(program\_mode)));
if(program\_mode == Server)
if(!FileDelete(fname, FILE\_COMMON)) {
if(reason == REASON\_INITFAILED)
FileDelete(fname, FILE\_COMMON);
int del = FileOpen("Server++/delete", FILE\_BIN | FILE\_COMMON | FILE\_WRITE);
if(del == INVALID\_HANDLE) {
Alert("Server abnormal shutdown!\nClear common folder.");
return;
}
FileWriteDouble(del, 3.142);
FileFlush(del);
FileClose(del);
printf("Preparing termination of clients...");
printf("%s shutdown successfull.", EnumToString(program\_mode));
return;
}
if(program\_mode == Client) {
if(reason == REASON\_INITFAILED)
return;
FileDelete(fname, FILE\_COMMON);
printf("%s shutdown successfull.", EnumToString(program\_mode));
}
return;
}
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
struct Data {
int type;
double lotsize, sl, tp;
ulong ticket, leverage, hash;
double open\_price, equity, close\_price;
uchar symbol[8];
//---
Data(void): ticket(0), lotsize(0.0), sl(0.0),
tp(0.0), open\_price(0.0), leverage(0), equity(0.0),
close\_price(0.0) {
ArrayInitialize(symbol, NULL);
}
};
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
void OnTick(void) {
if(program\_mode == Server)
return;
if(!TradeAllowed())
return;
//---
if(len(to\_open) > 0 || len(to\_close) > 0)
retry();
//--- check if server is active
int error = NULL;
if(program\_mode == Client)
if(FileIsExist("Server++/delete", FILE\_COMMON)) {
Alert("Server disconnected!");
ExpertRemove();
return;
}
//--- run client loop
if(program\_mode == Server)
return;
//---
Data dt;
int data\_size = sizeof(Data);
FileReadStruct(fhandle, dt, data\_size);
const string sym = CharArrayToString(dt.symbol, 0, 8) + StringSubstr(Symbol(), 6);
if(len(sym) >= 6)
if(!SymbolSelect(sym, true))
Alert(StringFormat("Symbol [%s] not selected in marketwatch!", sym));
if(dt.ticket == 0)
return;
//---
#ifdef \_debug
string open\_msg, close\_msg;
open\_msg = StringFormat("Server: [%s]\nLeverage: %d\nTicket: "
"%d\nSymbol: %s\nVolume: %f\nType: %s\n"
"SL: %f\nTP: %f\nOpenPrice: %f",
server\_id, dt.leverage, dt.ticket, sym,
dt.lotsize, dt.type == 1 ? "Buy" : dt.type == -1 ?
"Sell" : "Close", dt.sl, dt.tp, dt.open\_price);
close\_msg = StringFormat("Server: [%s]\nType: %s\nTicket: %d\nPrice: %f",
server\_id, "\*\*\*Close\*\*\*", dt.ticket, dt.close\_price);
if(dt.type != 0)
Comment(open\_msg);
else
Comment(close\_msg);
#endif
//---
static bool init = false;
if(!init) {
m\_trade.SetDeviationInPoints(max\_slippage \* 2);
m\_trade.SetTypeFilling(ORDER\_FILLING\_RETURN);
init = true;
}
//---
MqlTick tick;
tick.ask = 0.0;
tick.bid = 0.0;
if(len(sym) >= 6)
if(!SymbolInfoTick(sym, tick))
Alert("Price refresh error!");
static pair\_ticket p[];
double lots\_normalized = 0.0;
if(dt.type != 0)
lots\_normalized = NormalizeDouble(((AccountInfoDouble(ACCOUNT\_EQUITY)
\* dt.lotsize) / dt.equity) \* (copy\_factor > 0 ? copy\_factor / 100 : 1.0), 2);
lots\_normalized = VerifyLots(lots\_normalized, sym);
//---
switch(dt.type) {
case 1: {
if(!m\_trade.Buy(lots\_normalized, sym, tick.ask))
if(m\_trade.ResultRetcode() == TRADE\_RETCODE\_REQUOTE)
if(no\_try) {
Alert(StringFormat("Failed buy ticket %d [requote]", dt.ticket));
return;
} else {
dt.lotsize = lots\_normalized;
append(to\_open, dt);
return;
}
pair\_ticket d;
d.sr\_ticket = dt.ticket;
d.cl\_ticket = m\_trade.ResultOrder();
append(open\_tickets, d.cl\_ticket);
append(p, d);
return;
}
case -1: {
if(!m\_trade.Sell(lots\_normalized, sym, tick.bid))
if(m\_trade.ResultRetcode() == TRADE\_RETCODE\_REQUOTE)
if(no\_try) {
Alert(StringFormat("Failed sell ticket %d [requote]", dt.ticket));
return;
} else {
dt.lotsize = lots\_normalized;
append(to\_open, dt);
return;
}
pair\_ticket d;
d.sr\_ticket = dt.ticket;
d.cl\_ticket = m\_trade.ResultOrder();
append(open\_tickets, d.cl\_ticket);
append(p, d);
return;
}
default: {
ulong ticket = 0;
for(int i = 0; i < len(p); i++)
if(p[i].sr\_ticket == dt.ticket) {
ticket = p[i].cl\_ticket;
if(ticket == 0)
return;
ArrayRemove(p, i, 1);
if(!m\_trade.PositionClose(ticket) ||
m\_trade.ResultRetcode() != TRADE\_RETCODE\_DONE)
append(to\_close, ticket);
}
return;
}
}
}
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
Data to\_open[];
ulong to\_close[];
//---
void retry(void) {
if(len(to\_close))
while(len(to\_close))
if(m\_trade.PositionClose(to\_close[len(to\_close) - 1], max\_slippage \* 10))
if(m\_trade.ResultRetcode() == TRADE\_RETCODE\_DONE)
pop(to\_close, to\_close[len(to\_close) - 1]);
if(len(to\_open))
while(len(to\_open)) {
int idx = len(to\_open) - 1;
string sym = CharArrayToString(to\_open[idx].symbol, 0, 8);
MqlTick tick;
tick.ask = 0.0;
tick.bid = 0.0;
if(len(sym) >= 6)
if(!SymbolInfoTick(sym, tick))
Alert("Prices refresh error");
switch(to\_open[idx].type) {
case 1: {
if(m\_trade.Buy(to\_open[idx].lotsize, sym, tick.ask, to\_open[idx].sl, to\_open[idx].tp))
if(m\_trade.ResultRetcode() == TRADE\_RETCODE\_DONE) {
ulong ticket = m\_trade.ResultOrder();
append(open\_tickets, ticket);
ArrayRemove(to\_open, idx, 1);
}
break;
}
case -1: {
if(m\_trade.Sell(to\_open[idx].lotsize, sym, tick.bid, to\_open[idx].sl, to\_open[idx].tp))
if(m\_trade.ResultRetcode() == TRADE\_RETCODE\_DONE) {
ulong ticket = m\_trade.ResultOrder();
append(open\_tickets, ticket);
ArrayRemove(to\_open, idx, 1);
}
break;
}
}
}
//---
return;
}
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
void OnTimer() {
if(TradeAllowed())
OnTick();
}
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
const MqlTradeRequest& request,
const MqlTradeResult& result) {
if(program\_mode == Client)
return;
if(result.order == 0 || request.type > 1)
return;
if(request.position == 0) {
if(!m\_position.SelectByTicket(request.order))
return;
Data s;
//---
s.lotsize = request.volume;
s.open\_price = result.price;
s.sl = request.sl;
s.ticket = request.order;
s.tp = request.tp;
s.leverage = AccountInfoInteger(ACCOUNT\_LEVERAGE);
s.equity = AccountInfoDouble(ACCOUNT\_EQUITY);
s.type = request.type == ORDER\_TYPE\_BUY ? 1 : -1;
StringToCharArray(request.symbol, s.symbol, 0, 8);
//---
if(FileWriteStruct(fhandle, s, sizeof(Data)) < sizeof(Data))
Alert("Server write error!");
FileFlush(fhandle);
}
if(request.position != 0) {
Data s;
s.type = 0;
s.ticket = request.position;
s.close\_price = result.price;
if(FileWriteStruct(fhandle, s, sizeof(Data)) < sizeof(Data))
Alert("Server write error!");
FileFlush(fhandle);
}
}
//+------------------------------------------------------------------+
//| |
//+------------------------------------------------------------------+
//| Utility Tools |
//+------------------------------------------------------------------+
template 
bool append(t &array[], const t &element) {
int size = len(array);
int reserved = 0;
if(size % 10 == 0)
reserved = 10;
if(ArrayResize(array, size + 1, reserved) <= 0)
return false;
array[size] = element;
return true;
}
//---
template 
bool pop(t &array[], const t element) {
int counter = len(array);
while(counter) {
int idx = present(array, element);
if(idx == -1)
break;
ArrayRemove(array, idx, 1);
--counter;
}
return present(array, element) == -1;
}
//---
template 
int present(const t &array[], const t element) {
for(int i = 0; i < len(array); i++)
if(array[i] == element)
return i;
return -1;
}
//---
template 
int len(const t &array[]) {
return ArraySize(array);
}
//---
int len(const string var) {
return StringLen(var);
}
//---
double VerifyLots(const double variable, const string &sym) {
double volume = NormalizeDouble(variable, 2);
double lots\_step = SymbolInfoDouble(sym, SYMBOL\_VOLUME\_STEP);
if(lots\_step > 0.0)
volume = lots\_step \* MathFloor(volume / lots\_step);
double lots\_min = SymbolInfoDouble(sym, SYMBOL\_VOLUME\_MIN);
if(volume < lots\_min)
volume = lots\_min;
double lots\_max = SymbolInfoDouble(sym, SYMBOL\_VOLUME\_MAX);
if(volume > lots\_max)
volume = lots\_max;
return(volume);
}
//---
bool TradeAllowed(void) {
return AccountInfoInteger(ACCOUNT\_TRADE\_ALLOWED) &&
AccountInfoInteger(ACCOUNT\_TRADE\_EXPERT) &&
TerminalInfoInteger(TERMINAL\_TRADE\_ALLOWED) &&
TerminalInfoInteger(TERMINAL\_CONNECTED);
}
//---
bool reconnect(const string sr\_id, int &err) {
if(program\_mode != Client)
return false;
if(!FileIsExist(fname, FILE\_COMMON)) {
Alert(StringFormat("No server at specified address [%s]!", server\_id));
err = 100;
return false;
}
if(FileIsExist("Server++/delete", FILE\_COMMON))
return false;
fhandle = FileOpen(fname, client\_flags);
if(fhandle == INVALID\_HANDLE) {
Alert(StringFormat("Failed setting up client at address [%s]", server\_id));
err = 101;
return false;
}
if(FileSize(fhandle) > 0)
if(!FileSeek(fhandle, 0, SEEK\_END)) {
Alert("Server error!");
err = 102;
return false;
}
return true;
}
//+------------------------------------------------------------------+