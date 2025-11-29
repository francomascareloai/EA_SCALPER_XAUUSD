# MQL5 Book - Part 8 (Pages 1401-1600)

## Page 1401

Part 6. Trading automation
1 401 
6.4 Creating Expert Advisors
void OnStart()
{
   process((ENUM_POSITION_PROPERTY_INTEGER)0);
   process((ENUM_POSITION_PROPERTY_DOUBLE)0);
   process((ENUM_POSITION_PROPERTY_STRING)0);
}
As a result of its execution, we get the following log. The left column contains the numbering inside the
enumerations, and the values on the right (after the '=' sign) are the built-in constants (identifiers) of
the elements.
ENUM_POSITION_PROPERTY_INTEGER Count=9
0 POSITION_TIME=1
1 POSITION_TYPE=2
2 POSITION_MAGIC=12
3 POSITION_IDENTIFIER=13
4 POSITION_TIME_MSC=14
5 POSITION_TIME_UPDATE=15
6 POSITION_TIME_UPDATE_MSC=16
7 POSITION_TICKET=17
8 POSITION_REASON=18
ENUM_POSITION_PROPERTY_DOUBLE Count=8
0 POSITION_VOLUME=3
1 POSITION_PRICE_OPEN=4
2 POSITION_PRICE_CURRENT=5
3 POSITION_SL=6
4 POSITION_TP=7
5 POSITION_COMMISSION=8
6 POSITION_SWAP=9
7 POSITION_PROFIT=10
ENUM_POSITION_PROPERTY_STRING Count=3
0 POSITION_SYMBOL=0
1 POSITION_COMMENT=11
2 POSITION_EXTERNAL_ID=19
For example, the property with a constant of 0 is a string POSITION_SYMBOL, the properties with
constants 1  and 2 are integers POSITION_TIME and POSITION_TYPE, the property with a constant of
3 is a real POSITION_VOLUME, and so on.
Thus, constants are a system of end-to-end indexes on properties of all types, and we can use the
same algorithm (based on EnumToArray.mqh) to get them.
For each property, you need to remember its type (which determines which of the three arrays will
store the value) and the serial number among the properties of the same type (this will be the index of
the element in the corresponding array). For example, we see that positions have only 3 string
properties, so the strings array in the snapshot of one position will have to have the same size, and
POSITION_SYMBOL (0), POSITION_COMMENT (1 1 ), and POSITION_EXTERNAL_ID (1 9) will be written
to its indexes 0, 1 , and 2.
The conversion of end-to-end indexes of properties into their type (one of PROP_TYPE) and into an
ordinal number in an array of the corresponding type can be done once at the start of the program
since enumerations with properties are constant (built into the system). We write the resulting indirect
addressing table into a static two-dimensional indices array. Its size in the first dimension will be

---

## Page 1402

Part 6. Trading automation
1 402
6.4 Creating Expert Advisors
dynamically determined as the total number of properties (of all 3 types). We will write the size into the
limit static variable. A couple of cells are allocated for the second dimension: indices[i][0] – type
PROP_TYPE, indices[i][1 ] – index in one of the arrays ulongs, doubles, or strings (depending on
indices[i][0]).
   class TradeState
   {
      ...
      static int indices[][2];
      static int j, d, s;
   public:
      const static int limit;
      
      static PROP_TYPE type(const int i)
      {
         return (PROP_TYPE)indices[i][0];
      }
      
      static int offset(const int i)
      {
         return indices[i][1];
      }
      ...
Variables j, d, and s will be used to sequentially index properties within each of the 3 different types.
Here's how it's done in the static method calcIndices.

---

## Page 1403

Part 6. Trading automation
1 403
6.4 Creating Expert Advisors
      static int calcIndices()
      {
         const int size = fmax(boundary<I>(),
            fmax(boundary<D>(), boundary<S>())) + 1;
         ArrayResize(indices, size);
         j = d = s = 0;
         for(int i = 0; i < size; ++i)
         {
            if(detect<I>(i))
            {
               indices[i][0] = PROP_TYPE_INTEGER;
               indices[i][1] = j++;
            }
            else if(detect<D>(i))
            {
               indices[i][0] = PROP_TYPE_DOUBLE;
               indices[i][1] = d++;
            }
            else if(detect<S>(i))
            {
               indices[i][0] = PROP_TYPE_STRING;
               indices[i][1] = s++;
            }
            else
            {
               Print("Unresolved int value as enum: ", i, " ", typename(TradeState));
            }
         }
         return size;
      }
The boundary method returns the maximum constant among all elements of the given enumeration E.
   template<typename E>
   static int boundary(const E dummy = (E)NULL)
   {
      int values[];
      const int n = EnumToArray(dummy, values, 0, 1000);
      ArraySort(values);
      return values[n - 1];
   }
The largest value of all three types of enumerations determines the range of integers that should be
sorted in accordance with the property type to which they belong.
Here we use the detect method which returns true if the integer is an element of an enumeration.

---

## Page 1404

Part 6. Trading automation
1 404
6.4 Creating Expert Advisors
   template<typename E>
   static bool detect(const int v)
   {
      ResetLastError();
      const string s = EnumToString((E)v); // result is not used 
      if(_LastError == 0) // only the absence of an error is important
      {
         return true;
      }
      return false;
   }
The last question is how to run this calculation when the program starts. This is achieved by utilizing
the static nature of the variables and the method.
template<typename I,typename D,typename S>
static int MonitorInterface::TradeState::indices[][2];
template<typename I,typename D,typename S>
static int MonitorInterface::TradeState::j,
   MonitorInterface::TradeState::d,
   MonitorInterface::TradeState::s;
template<typename I,typename D,typename S>
const static int MonitorInterface::TradeState::limit =
   MonitorInterface::TradeState::calcIndices();
Note that limit is initialized by the result of calling our calcIndices function.
Having a table with indexes, we implement the filling of arrays with property values in the cache
method.

---

## Page 1405

Part 6. Trading automation
1 405
6.4 Creating Expert Advisors
   class TradeState
   {
      ...
      TradeState(const MonitorInterface *ptr) : owner(ptr)
      {
         cache(); // when creating an object, immediately cache the properties
      }
      
      template<typename T>
      void _get(const int e, T &value) const // overload with record by reference
      {
         value = owner.get(e, value);
      }
      
      void cache()
      {
         ArrayResize(ulongs, j);
         ArrayResize(doubles, d);
         ArrayResize(strings, s);
         for(int i = 0; i < limit; ++i)
         {
            switch(indices[i][0])
            {
            case PROP_TYPE_INTEGER: _get(i, ulongs[indices[i][1]]); break;
            case PROP_TYPE_DOUBLE: _get(i, doubles[indices[i][1]]); break;
            case PROP_TYPE_STRING: _get(i, strings[indices[i][1]]); break;
            }
         }
      }
   };
We loop through the entire range of properties from 0 to limit and, depending on the property type in
indices[i][0], write its value to the element of the ulongs, doubles, or strings array under the number
indices[i][1 ] (the corresponding element of the array is passed by reference to the _ get method).
A call of owner.get(e, value) refers to one of the standard methods of the monitor class (here it is visible
as an abstract pointer MonitorInterface). In particular, for positions in the PositionMonitor class, this
will lead to PositionGetInteger, PositionGetDouble, or PositionGetString calls. The compiler will choose
the correct type. Order and deal monitors have their own similar implementations, which are
automatically included by this base code.
It is logical to inherit the description of a snapshot of one trading object from the monitor class. Since
we have to cache orders, deals, and positions, it makes sense to make the new class a template and
collect all common algorithms suitable for all objects in it. Let's call it TradeBaseState
(fileTradeState.mqh).

---

## Page 1406

Part 6. Trading automation
1 406
6.4 Creating Expert Advisors
template<typename M,typename I,typename D,typename S>
class TradeBaseState: public M
{
   M::TradeState state;
   bool cached;
   
public:
   TradeBaseState(const ulong t) : M(t), state(&this), cached(ready)
   {
   }
   
   void passthrough(const bool b)   // enable/disable cache as desired
   {
      cached = b;
   }
   ...
One of the specific monitor classes described earlier is hidden under the letter M (OrderMonitor.mqh,
PositionMonitor.mqh, DealMonitor.mqh). The basis is the state caching object of the newly introduced
M::TradeState class. Depending on M, a specific index table will be formed inside (one for class M) and
arrays of properties will be distributed (own for each instance of M, that is, for each order, deal,
position).
The cached variable contains a sign of whether the arrays in the state are filled with property values,
and whether to query properties on an object to return values from the cache. This will be required
later to compare the saved and current states.
In other words, when cached is set to false, the object will behave like a regular monitor, reading
properties from the trading environment. When cached equals true, the object will return previously
stored values from internal arrays.
   virtual long get(const I property) const override
   {
      return cached ? state.ulongs[M::TradeState::offset(property)] : M::get(property);
   }
   
   virtual double get(const D property) const override
   {
      return cached ? state.doubles[M::TradeState::offset(property)] : M::get(property);
   }
   
   virtual string get(const S property) const override
   {
      return cached ? state.strings[M::TradeState::offset(property)] : M::get(property);
   }
   ...
By default, caching is, of course, enabled.
We must also provide a method that directly performs caching (filling arrays). To do this, just call the
cache method for the state object.

---

## Page 1407

Part 6. Trading automation
1 407
6.4 Creating Expert Advisors
   bool update()
   {
      if(refresh())
      {
         cached = false; // disable reading from the cache
         state.cache();  // read real properties and write to cache
         cached = true;  // enable external cache access back 
         return true;
      }
      return false;
   }
What is the refresh method?
So far, we have been using monitor objects in simple mode: creating, reading properties, and deleting
them. At the same time, property reading assumes that the corresponding order, deal, or position was
selected in the trading context (inside the constructor). Since we are now improving monitors to
support the internal state, it is necessary to ensure that the desired element is re-allocated in order to
read the properties even after an indefinite time (of course, with a check that the element still exists).
To implement this, we have added the refresh virtual method to the template MonitorInterface class.
// TradeBaseMonitor.mqh
template<typename I,typename D,typename S>
class MonitorInterface
{
   ...
   virtual bool refresh() = 0;
It must return true upon successful allocation of an order, deal, or position. If the result is false, one of
the following errors should be contained in the built-in _ LastError variable:
• 4753 ERR_TRADE_POSITION_NOT_FOUND;
• 4754 ERR_TRADE_ORDER_NOT_FOUND;
• 4755 ERR_TRADE_DEAL_NOT_FOUND;
In this case, the ready member variable, which signals the availability of the object, must be reset to
false in implementations of this method in derived classes.
For example, in the PositionMonitor constructor, we had and still have such an initialization. The
situation is similar to order and deal monitors.

---

## Page 1408

Part 6. Trading automation
1 408
6.4 Creating Expert Advisors
// PositionMonitor.mqh
   const ulong ticket;
   PositionMonitor(const ulong t): ticket(t)
   {
      if(!PositionSelectByTicket(ticket))
      {
         PrintFormat("Error: PositionSelectByTicket(%lld) failed: %s", ticket,
            E2S(_LastError));
      }
      else
      {
         ready = true;
      }
   }
   ...
Now we will add the refresh method to all specific classes of this kind (see example PositionMonitor):
// PositionMonitor.mqh
   virtual bool refresh() override
   {
      ready = PositionSelectByTicket(ticket);
      return ready;
   }
But populating cache arrays with property values is only half the battle. The second half is to compare
these values with the actual state of the order, deal, or position.
To identify differences and write indexes of changed properties to the changes array, the generated
TradeBaseState class provides the getChanges method. The method returns true when changes are
detected.

---

## Page 1409

Part 6. Trading automation
1 409
6.4 Creating Expert Advisors
template<typename M,typename I,typename D,typename S>
class TradeBaseState: public M
{
   ...
   bool getChanges(int &changes[])
   {
      const bool previous = ready;
      if(refresh())
      {
         // element is selected in the trading environment = properties can be read and compared
         cached = false;    // read directly
         const bool result = M::diff(state, changes);
         cached = true;     // turn cache back on by default
         return result;
      }
      // no longer "ready" = most likely deleted
      return previous != ready; // if just deleted, this is also a change 
   }
As you can see, the main work is entrusted to a certain method diff in class M. This is a new method:
we need to write it. Fortunately, thanks to OOP, you can do this once in the base template
MonitorInterface, and the method will appear immediately for orders, deals, and positions.

---

## Page 1410

Part 6. Trading automation
1 41 0
6.4 Creating Expert Advisors
// TradeBaseMonitor.mqh
template<typename I,typename D,typename S>
class MonitorInterface
{
   ...
   bool diff(const TradeState &that, int &changes[])
   {
      ArrayResize(changes, 0);
      for(int i = 0; i < TradeState::limit; ++i)
      {
         switch(TradeState::indices[i][0])
         {
         case PROP_TYPE_INTEGER:
            if(this.get((I)i) != that.ulongs[TradeState::offset(i)])
            {
               PUSH(changes, i);
            }
            break;
         case PROP_TYPE_DOUBLE:
            if(!TU::Equal(this.get((D)i), that.doubles[TradeState::offset(i)]))
            {
               PUSH(changes, i);
            }
            break;
         case PROP_TYPE_STRING:
            if(this.get((S)i) != that.strings[TradeState::offset(i)])
            {
               PUSH(changes, i);
            }
            break;
         }
      }
      return ArraySize(changes) > 0;
   }
So, everything is ready to form specific caching classes for orders, deals, and positions. For example,
positions will be stored in the extended monitor PositionState on the base of PositionMonitor.
class PositionState: public TradeBaseState<PositionMonitor,
   ENUM_POSITION_PROPERTY_INTEGER,
   ENUM_POSITION_PROPERTY_DOUBLE,
   ENUM_POSITION_PROPERTY_STRING>
{
public:
   PositionState(const long t): TradeBaseState(t) { }
};
Similarly, a caching class for deals is defined in the file TradeState.mqh.

---

## Page 1411

Part 6. Trading automation
1 41 1 
6.4 Creating Expert Advisors
class DealState: public TradeBaseState<DealMonitor,
   ENUM_DEAL_PROPERTY_INTEGER,
   ENUM_DEAL_PROPERTY_DOUBLE,
   ENUM_DEAL_PROPERTY_STRING>
{
public:
   DealState(const long t): TradeBaseState(t) { }
};
With orders, things are a little more complicated, because they can be active and historical. So far we
have had one generic monitor class for orders, OrderMonitor. It tries to find the submitted order ticket
both among the active orders and in the history. This approach is not suitable for caching, because
Expert Advisors need to track the transition of an order from one state to another.
For this reason, we add 2 more specific classes to the OrderMonitor.mqh file: ActiveOrderMonitor and
HistoryOrderMonitor.
// OrderMonitor.mqh
class ActiveOrderMonitor: public OrderMonitor
{
public:
   ActiveOrderMonitor(const ulong t): OrderMonitor(t)
   {
      if(history) // if the order is in history, then it is already inactive
      {
         ready = false;   // reset ready flag
         history = false; // this object is only for active orders by definition
      }
   }
   
   virtual bool refresh() override
   {
      ready = OrderSelect(ticket);
      return ready;
   }
};
   
class HistoryOrderMonitor: public OrderMonitor
{
public:
   HistoryOrderMonitor(const ulong t): OrderMonitor(t) { }
   
   virtual bool refresh() override
   {
      history = true; // work only with history
      ready = historyOrderSelectWeak(ticket);
      return ready; // readiness is determined by the presence of a ticket in the history
   }
};
Each of them searches for a ticket only in their area. Based on these monitors, you can already create
caching classes.

---

## Page 1412

Part 6. Trading automation
1 41 2
6.4 Creating Expert Advisors
// TradeState.mqh
class OrderState: public TradeBaseState<ActiveOrderMonitor,
   ENUM_ORDER_PROPERTY_INTEGER,
   ENUM_ORDER_PROPERTY_DOUBLE,
   ENUM_ORDER_PROPERTY_STRING>
{
public:
   OrderState(const long t): TradeBaseState(t) { }
};
   
class HistoryOrderState: public TradeBaseState<HistoryOrderMonitor,
   ENUM_ORDER_PROPERTY_INTEGER,
   ENUM_ORDER_PROPERTY_DOUBLE,
   ENUM_ORDER_PROPERTY_STRING>
{
public:
   HistoryOrderState(const long t): TradeBaseState(t) { }
};
The final touch that we will add to the TradeBaseState class for convenience is a special method for
converting a property value to a string. Although there are several versions of the stringify methods in
the monitor, they will all "print" either values from the cache (if the member variable cached equals
true) or values from the original object of the trading environment (if cached equals false). To visualize
the differences between the cache and the changed object (when these differences are found), we need
to simultaneously read the value from the cache and bypass the cache. In this regard, we add the
stringifyRaw method which always works with the property directly (due to the fact that the cached
variable is temporarily reset and reinstalled).
   // get the string representation of the property 'i' bypassing the cache
   string stringifyRaw(const int i)
   {
      const bool previous = cached;
      cached = false;
      const string s = stringify(i);
      cached = previous;
   }
Let's check the performance of the caching monitor using a simple example of an Expert Advisor that
monitors the status of an active order (OrderSnapshot.mq5). Later we will develop this idea for caching
any set of orders, deals, or positions, that is, we will create a full-fledged cache.
The Expert Advisor will try to find the last one in the list of active orders and create the OrderState
object for it. If there are no orders, the user will be prompted to create an order or open a position (the
latter is associated with placing and executing an order on the market). As soon as an order is found,
we check if the order state has changed. This check is performed in the OnTrade handler. The Expert
Advisor will continue to monitor this order until it is unloaded.

---

## Page 1413

Part 6. Trading automation
1 41 3
6.4 Creating Expert Advisors
int OnInit()
{
   if(OrdersTotal() == 0)
   {
      Alert("Please, create a pending order or open/close a position");
   }
   else
   {
      OnTrade(); // self-invocation
   }
   return INIT_SUCCEEDED;
}
   
void OnTrade()
{
   static int count = 0;
   // object pointer is stored in static AutoPtr
   static AutoPtr<OrderState> auto;
   // get a "clean" pointer (so as not to dereference auto[] everywhere)
   OrderState *state = auto[];
   
   PrintFormat(">>> OnTrade(%d)", count++);
   
   if(OrdersTotal() > 0 && state == NULL)
   {
      const ulong ticket = OrderGetTicket(OrdersTotal() - 1);
      auto = new OrderState(ticket);
      PrintFormat("Order picked up: %lld %s", ticket,
         auto[].isReady() ? "true" : "false");
      auto[].print(); // initial state at the time of "capturing" the order
   }
   else if(state)
   {
      int changes[];
      if(state.getChanges(changes))
      {
         Print("Order properties changed:");
         ArrayPrint(changes);
         ...
      }
      if(_LastError != 0) Print(E2S(_LastError));
   }
}
In addition to displaying an array of changed properties, it would be nice to display the changes
themselves. Therefore, instead of an ellipsis, we will add such a fragment (it will be useful to us in future
classes of full-fledged caches).

---

## Page 1414

Part 6. Trading automation
1 41 4
6.4 Creating Expert Advisors
         for(int k = 0; k < ArraySize(changes); ++k)
         {
            switch(OrderState::TradeState::type(changes[k]))
            {
            case PROP_TYPE_INTEGER:
               Print(EnumToString((ENUM_ORDER_PROPERTY_INTEGER)changes[k]), ": ",
                  state.stringify(changes[k]), " -> ",
                  state.stringifyRaw(changes[k]));
                  break;
            case PROP_TYPE_DOUBLE:
               Print(EnumToString((ENUM_ORDER_PROPERTY_DOUBLE)changes[k]), ": ",
                  state.stringify(changes[k]), " -> ",
                  state.stringifyRaw(changes[k]));
                  break;
            case PROP_TYPE_STRING:
               Print(EnumToString((ENUM_ORDER_PROPERTY_STRING)changes[k]), ": ",
                  state.stringify(changes[k]), " -> ",
                  state.stringifyRaw(changes[k]));
                  break;
            }
         }
Here we use the new stringifyRaw method. After displaying the changes, do not forget to update the
cache state.
         state.update();
If you run the Expert Advisor on an account with no active orders and place a new one, you will see the
following entries in the log (here buy limit for EURUSD is created below the current market price).

---

## Page 1415

Part 6. Trading automation
1 41 5
6.4 Creating Expert Advisors
Alert: Please, create a pending order or open/close a position
>>> OnTrade(0)
Order picked up: 1311736135 true
MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER,ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>
ENUM_ORDER_PROPERTY_INTEGER Count=14
  0 ORDER_TIME_SETUP=2022.04.11 11:42:39
  1 ORDER_TIME_EXPIRATION=1970.01.01 00:00:00
  2 ORDER_TIME_DONE=1970.01.01 00:00:00
  3 ORDER_TYPE=ORDER_TYPE_BUY_LIMIT
  4 ORDER_TYPE_FILLING=ORDER_FILLING_RETURN
  5 ORDER_TYPE_TIME=ORDER_TIME_GTC
  6 ORDER_STATE=ORDER_STATE_STARTED
  7 ORDER_MAGIC=0
  8 ORDER_POSITION_ID=0
  9 ORDER_TIME_SETUP_MSC=2022.04.11 11:42:39'729
 10 ORDER_TIME_DONE_MSC=1970.01.01 00:00:00'000
 11 ORDER_POSITION_BY_ID=0
 12 ORDER_TICKET=1311736135
 13 ORDER_REASON=ORDER_REASON_CLIENT
ENUM_ORDER_PROPERTY_DOUBLE Count=7
  0 ORDER_VOLUME_INITIAL=0.01
  1 ORDER_VOLUME_CURRENT=0.01
  2 ORDER_PRICE_OPEN=1.087
  3 ORDER_PRICE_CURRENT=1.087
  4 ORDER_PRICE_STOPLIMIT=0.0
  5 ORDER_SL=0.0
  6 ORDER_TP=0.0
ENUM_ORDER_PROPERTY_STRING Count=3
  0 ORDER_SYMBOL=EURUSD
  1 ORDER_COMMENT=
  2 ORDER_EXTERNAL_ID=
>>> OnTrade(1)
Order properties changed:
10 14
ORDER_PRICE_CURRENT: 1.087 -> 1.09073
ORDER_STATE: ORDER_STATE_STARTED -> ORDER_STATE_PLACED
>>> OnTrade(2)
>>> OnTrade(3)
>>> OnTrade(4)
Here you can see how the status of the order changed from STARTED to PLACED. If, instead of a
pending order, we opened on the market with a small volume, we might not have time to receive these
changes, because such orders, as a rule, are set very quickly, and their observed status changes from
STARTED immediately to FILLED. And the latter already means that the order has been moved to
history. Therefore, parallel history monitoring is required to track them. We will show this in the next
example.
Please note that there may be many OnTrade events but not all of them are related to our order.
Let's try to set the Take Profit level and check the log.

---

## Page 1416

Part 6. Trading automation
1 41 6
6.4 Creating Expert Advisors
>>> OnTrade(5)
Order properties changed:
10 13
ORDER_PRICE_CURRENT: 1.09073 -> 1.09079
ORDER_TP: 0.0 -> 1.097
>>> OnTrade(6)
>>> OnTrade(7)
Next, change the expiration date: from GTC to one day.
>>> OnTrade(8)
Order properties changed:
10
ORDER_PRICE_CURRENT: 1.09079 -> 1.09082
>>> OnTrade(9)
>>> OnTrade(10)
Order properties changed:
2 6
ORDER_TIME_EXPIRATION: 1970.01.01 00:00:00 -> 2022.04.11 00:00:00
ORDER_TYPE_TIME: ORDER_TIME_GTC -> ORDER_TIME_DAY
>>> OnTrade(11)
Here, in the process of changing our order, the price had enough time to change, and therefore we
"hooked" an intermediate notification about the new value in ORDER_PRICE_CURRENT. And only after
that, the expected changes in ORDER_TYPE_TIME and ORDER_TIME_EXPIRATION got into the log.
Next, we removed the order.
>>> OnTrade(12)
TRADE_ORDER_NOT_FOUND
Now for any actions with the account that lead to OnTrade events, our Expert Advisor will output
TRADE_ORDER_NOT_FOUND, because it is designed to track a single order. If the Expert Advisor is
restarted, it will "catch" another order if there is one. But we will stop the Expert Advisor and start
preparing for a more urgent task.
As a rule, caching and controlling changes is required not for a single order or position, but for all or a
set of them, selected according to certain conditions. For these purposes, we will develop a base
template class TradeCache (TradeCache.mqh) and, based on it, we will create applied classes for lists of
orders, deals, and positions.

---

## Page 1417

Part 6. Trading automation
1 41 7
6.4 Creating Expert Advisors
template<typename T,typename F,typename E>
class TradeCache
{
   AutoPtr<T> data[];
   const E property;
   const int NOT_FOUND_ERROR;
   
public:
   TradeCache(const E id, const int error): property(id), NOT_FOUND_ERROR(error) { }
   
   virtual string rtti() const
   {
      return typename(this); // will be redefined in derived classes for visual output to the log
   }
   ...
In this template, the letter T denotes one of the classes of the TradeState family. As you can see, an
array of such objects in the form of auto-pointers is reserved under the name data.
The letter F describes the type of one of the filter classes (OrderFilter.mqh, including HistoryOrderFilter,
DealFilter.mqh, PositionFilter.mqh) used to select cached items. In the simplest case, when the filter
does not contain let conditions, all elements will be cached (with respect to sampling history for objects
from history).
The letter E corresponds to the enumeration in which the property identifying the objects is located.
Since this property is usually SOME_TICKET, the enumeration is assumed to be an integer
ENUM_SOMETHING_PROPERTY_INTEGER.
The NOT_FOUND_ERROR variable is intended for the error code that occurs when trying to allocate a
non-existent object for reading, for example, ERR_TRADE_POSITION_NOT_FOUND for positions.
In parameters, the main class method scan receives a reference to the configured filter (it should be
configured by the calling code).
   void scan(F &f)
   {
      const int existedBefore = ArraySize(data);
      
      ulong tickets[];
      ArrayResize(tickets, existedBefore);
      for(int i = 0; i < existedBefore; ++i)
      {
         tickets[i] = data[i][].get(property);
      }
      ...
At the beginning of the method, we collect the identifiers of already cached objects into the tickets
array. Obviously, on the first run, it will be empty.
Next, we fill the obj ects array with tickets of relevant objects using a filter. For each new ticket, we
create a caching monitor object T and add it to the data array. For old objects, we analyze the
presence of changes by calling data[j ][].getChanges(changes) and then update the cache by calling
data[j ][].update().

---

## Page 1418

Part 6. Trading automation
1 41 8
6.4 Creating Expert Advisors
      ulong objects[];
      f.select(objects);
      for(int i = 0, j; i < ArraySize(objects); ++i)
      {
         const ulong ticket = objects[i];
         for(j = 0; j < existedBefore; ++j)
         {
            if(tickets[j] == ticket)
            {
               tickets[j] = 0; // mark as found
               break;
            }
         }
         
         if(j == existedBefore) // this is not in the cache, you need to add
         {
            const T *ptr = new T(ticket);
            PUSH(data, ptr);
            onAdded(*ptr);
         }
         else
         {
            ResetLastError();
            int changes[];
            if(data[j][].getChanges(changes))
            {
               onUpdated(data[j][], changes);
               data[j][].update();
            }
            if(_LastError) PrintFormat("%s: %lld (%s)", rtti(), ticket, E2S(_LastError));
         }
      }
      ...
As you can see, in each phase of the change, that is, when an object is added or after it is changed, the
onAdded and onUpdated methods are called. These are virtual stub methods that the scan can use to
notify the program of the appropriate events. Application code is expected to implement a derived class
with overridden versions of these methods. We will touch on this issue a little later, but for now, we will
continue to consider the method scan.
In the above loop, all found tickets in the tickets array are set to zero, and therefore the remaining
elements correspond to the missing objects of the trading environment. Next, they are checked by
calling getChanges and comparing the error code with NOT_FOUND_ERROR. If this is true, the
onRemoved virtual method is called. It returns a boolean flag (provided by your application code) saying
whether the item should be removed from the cache.

---

## Page 1419

Part 6. Trading automation
1 41 9
6.4 Creating Expert Advisors
      for(int j = 0; j < existedBefore; ++j)
      {
         if(tickets[j] == 0) continue; // skip processed elements
         
         // this ticket was not found, most likely deleted
         int changes[];
         ResetLastError();
         if(data[j][].getChanges(changes))
         {
            if(_LastError == NOT_FOUND_ERROR) // for example, ERR_TRADE_POSITION_NOT_FOUND
            {
               if(onRemoved(data[j][]))
               {
                  data[j] = NULL;             // release the object and array element
               }
               continue;
            }
            
            // NB! Usually we shouldn't fall here
            PrintFormat("Unexpected ticket: %lld (%s) %s", tickets[j],
               E2S(_LastError), rtti());
            onUpdated(data[j][], changes, true);
            data[j][].update();
         }
         else
         {
            PrintFormat("Orphaned element: %lld (%s) %s", tickets[j],
               E2S(_LastError), rtti());
         }
      }
   }
At the very end of the scan method, the data array is cleared of null elements but this fragment is
omitted here for brevity.
The base class provides standard implementations of the onAdded, onRemoved, and onUpdated
methods which display the essence of events in the log. By defining the PRINT_DETAILS macro in your
code before including the header file TradeCache.mqh, you can order a printout of all the properties of
each new object.

---

## Page 1420

Part 6. Trading automation
1 420
6.4 Creating Expert Advisors
   virtual void onAdded(const T &state)
   {
      Print(rtti(), " added: ", state.get(property));
      #ifdef PRINT_DETAILS
      state.print();
      #endif
   }
   
   virtual bool onRemoved(const T &state)
   {
      Print(rtti(), " removed: ", state.get(property));
      return true; // allow the object to be removed from the cache (false to save)
   }
   
   virtual void onUpdated(T &state, const int &changes[],
      const bool unexpected = false)
   {
      ...
   }
We will not present the onUpdated method, because it practically repeats the code for outputting
changes from the Expert Advisor OrderSnapshot.mq5 shown above.
Of course, the base class has facilities for getting the size of the cache and accessing a specific object
by number.
   int size() const
   {
      return ArraySize(data);
   }
   
   T *operator[](int i) const
   {
      return data[i][]; // return pointer (T*) from AutoPtr object
   }
Based on the base TradeCache class, we can easily create certain classes for caching lists of positions,
active orders, and orders from history. Deal caching is left as an independent task.

---

## Page 1421

Part 6. Trading automation
1 421 
6.4 Creating Expert Advisors
class PositionCache: public TradeCache<PositionState,PositionFilter,
   ENUM_POSITION_PROPERTY_INTEGER>
{
public:
   PositionCache(const ENUM_POSITION_PROPERTY_INTEGER selector = POSITION_TICKET,
      const int error = ERR_TRADE_POSITION_NOT_FOUND): TradeCache(selector, error) { }
};
   
class OrderCache: public TradeCache<OrderState,OrderFilter,
   ENUM_ORDER_PROPERTY_INTEGER>
{
public:
   OrderCache(const ENUM_ORDER_PROPERTY_INTEGER selector = ORDER_TICKET,
      const int error = ERR_TRADE_ORDER_NOT_FOUND): TradeCache(selector, error) { }
};
   
class HistoryOrderCache: public TradeCache<HistoryOrderState,HistoryOrderFilter,
   ENUM_ORDER_PROPERTY_INTEGER>
{
public:
   HistoryOrderCache(const ENUM_ORDER_PROPERTY_INTEGER selector = ORDER_TICKET,
      const int error = ERR_TRADE_ORDER_NOT_FOUND): TradeCache(selector, error) { }
};
To summarize the process of developing the presented functionality, we present a diagram of the main
classes. This is a simplified version of UML diagrams which can be useful when designing complex
programs in MQL5.

---

## Page 1422

Part 6. Trading automation
1 422
6.4 Creating Expert Advisors
Class diagram of monitors, filters, and caches of trading objects
Templates are marked in yellow, abstract classes are left in white, and certain implementations are
shown in color. Solid arrows with filled tips indicate inheritance, and dotted arrows with hollow tips
indicate template typing. Dotted arrows with open tips indicate the use of the specified methods of
each other by classes. Connections with diamonds are a composition (inclusion of some objects into
others).
As an example of using the cache, let's create an Expert Advisor TradeSnapshot.mq5, which will
respond to any changes in the trading environment from the OnTrade handler. For filtering and caching,
the code describes 6 objects, 2 (filter and cache) for each type of element: positions, active orders,
and historical orders.

---

## Page 1423

Part 6. Trading automation
1 423
6.4 Creating Expert Advisors
PositionFilter filter0;
PositionCache positions;
   
OrderFilter filter1;
OrderCache orders;
   
HistoryOrderFilter filter2;
HistoryOrderCache history;
No conditions are set for filters through the let method calls so that all discovered online objects will
get into the cache. There is an additional setting for orders from the history.
Optionally, at startup, you can load past orders to the cache at a given history depth. This can be done
via the HistoryLookup input variable. In this variable, you can select the last day, last week (by
duration, not calendar), month (30 days), or year (360 days). By default, the past history is not loaded
(more precisely, it is loaded only in 1  second). Since the macro PRINT_DETAILS is defined in the
Expert Advisor, be careful with accounts with a large history: they can generate a large log if the period
is not limited.
enum ENUM_HISTORY_LOOKUP
{
   LOOKUP_NONE = 1,
   LOOKUP_DAY = 86400,
   LOOKUP_WEEK = 604800,
   LOOKUP_MONTH = 2419200,
   LOOKUP_YEAR = 29030400,
   LOOKUP_ALL = 0,
};
   
input ENUM_HISTORY_LOOKUP HistoryLookup = LOOKUP_NONE;
   
datetime origin;
In the OnInit handler, we reset the caches (in case the Expert Advisor is restarted with new
parameters), calculate the start date of the history in the origin variable, and call OnTrade for the first
time.
int OnInit()
{
   positions.reset();
   orders.reset();
   history.reset();
   origin = HistoryLookup ? TimeCurrent() - HistoryLookup : 0;
   
   OnTrade(); // self start
   return INIT_SUCCEEDED;
}
The OnTrade handler is minimalistic as all the complexities are now hidden inside the classes.

---

## Page 1424

Part 6. Trading automation
1 424
6.4 Creating Expert Advisors
void OnTrade()
{
   static int count = 0;
   
   PrintFormat(">>> OnTrade(%d)", count++);
   positions.scan(filter0);
   orders.scan(filter1);
   // make a history selection just before using the filter
   // inside the 'scan' method
   HistorySelect(origin, LONG_MAX);
   history.scan(filter2);
   PrintFormat(">>> positions: %d, orders: %d, history: %d",
      positions.size(), orders.size(), history.size());
}
Immediately after launching the Expert Advisor on a clean account, we will see the following message:
>>> OnTrade(0)
>>> positions: 0, orders: 0, history: 0
Let's try to execute the simplest test case: let's buy or sell on an "empty" account which has no open
positions and pending orders. The log will record the following events (occurring almost instantly).
First, an active order will be detected.

---

## Page 1425

Part 6. Trading automation
1 425
6.4 Creating Expert Advisors
>>> OnTrade(1)
OrderCache added: 1311792104
MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER,ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>
ENUM_ORDER_PROPERTY_INTEGER Count=14
  0 ORDER_TIME_SETUP=2022.04.11 12:34:51
  1 ORDER_TIME_EXPIRATION=1970.01.01 00:00:00
  2 ORDER_TIME_DONE=1970.01.01 00:00:00
  3 ORDER_TYPE=ORDER_TYPE_BUY
  4 ORDER_TYPE_FILLING=ORDER_FILLING_FOK
  5 ORDER_TYPE_TIME=ORDER_TIME_GTC
  6 ORDER_STATE=ORDER_STATE_STARTED
  7 ORDER_MAGIC=0
  8 ORDER_POSITION_ID=0
  9 ORDER_TIME_SETUP_MSC=2022.04.11 12:34:51'096
 10 ORDER_TIME_DONE_MSC=1970.01.01 00:00:00'000
 11 ORDER_POSITION_BY_ID=0
 12 ORDER_TICKET=1311792104
 13 ORDER_REASON=ORDER_REASON_CLIENT
ENUM_ORDER_PROPERTY_DOUBLE Count=7
  0 ORDER_VOLUME_INITIAL=0.01
  1 ORDER_VOLUME_CURRENT=0.01
  2 ORDER_PRICE_OPEN=1.09218
  3 ORDER_PRICE_CURRENT=1.09218
  4 ORDER_PRICE_STOPLIMIT=0.0
  5 ORDER_SL=0.0
  6 ORDER_TP=0.0
ENUM_ORDER_PROPERTY_STRING Count=3
  0 ORDER_SYMBOL=EURUSD
  1 ORDER_COMMENT=
  2 ORDER_EXTERNAL_ID=
Then this order will be moved to the history (at the same time, at least the status, execution time and
position ID will change).

---

## Page 1426

Part 6. Trading automation
1 426
6.4 Creating Expert Advisors
HistoryOrderCache added: 1311792104
MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER,ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>
ENUM_ORDER_PROPERTY_INTEGER Count=14
  0 ORDER_TIME_SETUP=2022.04.11 12:34:51
  1 ORDER_TIME_EXPIRATION=1970.01.01 00:00:00
  2 ORDER_TIME_DONE=2022.04.11 12:34:51
  3 ORDER_TYPE=ORDER_TYPE_BUY
  4 ORDER_TYPE_FILLING=ORDER_FILLING_FOK
  5 ORDER_TYPE_TIME=ORDER_TIME_GTC
  6 ORDER_STATE=ORDER_STATE_FILLED
  7 ORDER_MAGIC=0
  8 ORDER_POSITION_ID=1311792104
  9 ORDER_TIME_SETUP_MSC=2022.04.11 12:34:51'096
 10 ORDER_TIME_DONE_MSC=2022.04.11 12:34:51'097
 11 ORDER_POSITION_BY_ID=0
 12 ORDER_TICKET=1311792104
 13 ORDER_REASON=ORDER_REASON_CLIENT
ENUM_ORDER_PROPERTY_DOUBLE Count=7
  0 ORDER_VOLUME_INITIAL=0.01
  1 ORDER_VOLUME_CURRENT=0.0
  2 ORDER_PRICE_OPEN=1.09218
  3 ORDER_PRICE_CURRENT=1.09218
  4 ORDER_PRICE_STOPLIMIT=0.0
  5 ORDER_SL=0.0
  6 ORDER_TP=0.0
ENUM_ORDER_PROPERTY_STRING Count=3
  0 ORDER_SYMBOL=EURUSD
  1 ORDER_COMMENT=
  2 ORDER_EXTERNAL_ID=
>>> positions: 0, orders: 1, history: 1
Note that these modifications happened within the same call of OnTrade. In other words, while our
program was analyzing the properties of the new order (by calling orders.scan), the order was
processed by the terminal in parallel, and by the time the history was checked (by calling history.scan),
it has already gone down in history. That is why it is listed both here and there according to the last
line of this log fragment. This behavior is normal for multithreaded programs and should be taken into
account when designing them. But it doesn't always have to be. Here we are simply drawing attention
to it. When executing an MQL program quickly, this situation usually does not occur.
If we were to check the history first, and then the online orders, then at the first stage we could find
that the order is not yet in the history, and at the second stage that the order is no longer online. That
is, it could theoretically get lost for a moment. A more realistic situation is to skip an order in its active
phase due to history synchronization, i.e., right away fix it for the first time in history.
Recall that MQL5 does not allow you to synchronize the trading environment as a whole, but only in
parts:
• Among active orders, information is relevant for the order for which the OrderSelect or
OrderGetTicket function has just been called
• Among the positions, the information is relevant for the position for which the function
PositionSelect, PositionSelectByTicket, or PositionGetTicket has just been called
• For orders and transactions in the history, information is available in the context of the last call of
HistorySelect, HistorySelectByPosition, HistoryOrderSelect, HistoryDealSelect

---

## Page 1427

Part 6. Trading automation
1 427
6.4 Creating Expert Advisors
In addition, let's remind you that trade events (like any MQL5 events) are messages about changes
that have occurred, placed in the queue, and retrieved from the queue in a delayed way, and not
immediately at the time of the changes. Moreover, the OnTrade event occurs after the relevant
OnTradeTransaction events.
Try different program configurations, debug, and generate detailed logs to choose the most reliable
algorithm for your trading system.
Let's return to our log. On the next triggering of OnTrade, the situation has already been fixed: the
cache of active orders has detected the deletion of the order. Along the way, the position cache saw an
open position.
>>> OnTrade(2)
PositionCache added: 1311792104
MonitorInterface<ENUM_POSITION_PROPERTY_INTEGER,ENUM_POSITION_PROPERTY_DOUBLE,ENUM_POSITION_PROPERTY_STRING>
ENUM_POSITION_PROPERTY_INTEGER Count=9
  0 POSITION_TIME=2022.04.11 12:34:51
  1 POSITION_TYPE=POSITION_TYPE_BUY
  2 POSITION_MAGIC=0
  3 POSITION_IDENTIFIER=1311792104
  4 POSITION_TIME_MSC=2022.04.11 12:34:51'097
  5 POSITION_TIME_UPDATE=2022.04.11 12:34:51
  6 POSITION_TIME_UPDATE_MSC=2022.04.11 12:34:51'097
  7 POSITION_TICKET=1311792104
  8 POSITION_REASON=POSITION_REASON_CLIENT
ENUM_POSITION_PROPERTY_DOUBLE Count=8
  0 POSITION_VOLUME=0.01
  1 POSITION_PRICE_OPEN=1.09218
  2 POSITION_PRICE_CURRENT=1.09214
  3 POSITION_SL=0.00000
  4 POSITION_TP=0.00000
  5 POSITION_COMMISSION=0.0
  6 POSITION_SWAP=0.00
  7 POSITION_PROFIT=-0.04
ENUM_POSITION_PROPERTY_STRING Count=3
  0 POSITION_SYMBOL=EURUSD
  1 POSITION_COMMENT=
  2 POSITION_EXTERNAL_ID=
OrderCache removed: 1311792104
>>> positions: 1, orders: 0, history: 1
After some time, we close the position. Since in our code the position cache is checked first
(positions.scan), changes to the closed position are included in the log.
>>> OnTrade(8)
PositionCache changed: 1311792104
POSITION_PRICE_CURRENT: 1.09214 -> 1.09222
POSITION_PROFIT: -0.04 -> 0.04
Further in the same call of OnTrade, we detect the appearance of a closing order and its instantaneous
transfer to history (again, due to its fast parallel processing by the terminal).

---

## Page 1428

Part 6. Trading automation
1 428
6.4 Creating Expert Advisors
OrderCache added: 1311796883
MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER,ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>
ENUM_ORDER_PROPERTY_INTEGER Count=14
  0 ORDER_TIME_SETUP=2022.04.11 12:39:55
  1 ORDER_TIME_EXPIRATION=1970.01.01 00:00:00
  2 ORDER_TIME_DONE=1970.01.01 00:00:00
  3 ORDER_TYPE=ORDER_TYPE_SELL
  4 ORDER_TYPE_FILLING=ORDER_FILLING_FOK
  5 ORDER_TYPE_TIME=ORDER_TIME_GTC
  6 ORDER_STATE=ORDER_STATE_STARTED
  7 ORDER_MAGIC=0
  8 ORDER_POSITION_ID=1311792104
  9 ORDER_TIME_SETUP_MSC=2022.04.11 12:39:55'710
 10 ORDER_TIME_DONE_MSC=1970.01.01 00:00:00'000
 11 ORDER_POSITION_BY_ID=0
 12 ORDER_TICKET=1311796883
 13 ORDER_REASON=ORDER_REASON_CLIENT
ENUM_ORDER_PROPERTY_DOUBLE Count=7
  0 ORDER_VOLUME_INITIAL=0.01
  1 ORDER_VOLUME_CURRENT=0.01
  2 ORDER_PRICE_OPEN=1.09222
  3 ORDER_PRICE_CURRENT=1.09222
  4 ORDER_PRICE_STOPLIMIT=0.0
  5 ORDER_SL=0.0
  6 ORDER_TP=0.0
ENUM_ORDER_PROPERTY_STRING Count=3
  0 ORDER_SYMBOL=EURUSD
  1 ORDER_COMMENT=
  2 ORDER_EXTERNAL_ID=
HistoryOrderCache added: 1311796883
MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER,ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>
ENUM_ORDER_PROPERTY_INTEGER Count=14
  0 ORDER_TIME_SETUP=2022.04.11 12:39:55
  1 ORDER_TIME_EXPIRATION=1970.01.01 00:00:00
  2 ORDER_TIME_DONE=2022.04.11 12:39:55
  3 ORDER_TYPE=ORDER_TYPE_SELL
  4 ORDER_TYPE_FILLING=ORDER_FILLING_FOK
  5 ORDER_TYPE_TIME=ORDER_TIME_GTC
  6 ORDER_STATE=ORDER_STATE_FILLED
  7 ORDER_MAGIC=0
  8 ORDER_POSITION_ID=1311792104
  9 ORDER_TIME_SETUP_MSC=2022.04.11 12:39:55'710
 10 ORDER_TIME_DONE_MSC=2022.04.11 12:39:55'711
 11 ORDER_POSITION_BY_ID=0
 12 ORDER_TICKET=1311796883
 13 ORDER_REASON=ORDER_REASON_CLIENT
ENUM_ORDER_PROPERTY_DOUBLE Count=7
  0 ORDER_VOLUME_INITIAL=0.01
  1 ORDER_VOLUME_CURRENT=0.0
  2 ORDER_PRICE_OPEN=1.09222
  3 ORDER_PRICE_CURRENT=1.09222
  4 ORDER_PRICE_STOPLIMIT=0.0
  5 ORDER_SL=0.0
  6 ORDER_TP=0.0

---

## Page 1429

Part 6. Trading automation
1 429
6.4 Creating Expert Advisors
ENUM_ORDER_PROPERTY_STRING Count=3
  0 ORDER_SYMBOL=EURUSD
  1 ORDER_COMMENT=
  2 ORDER_EXTERNAL_ID=
>>> positions: 1, orders: 1, history: 2
There are already 2 orders in the history cache, but the position and active order caches that were
analyzed before the history cache have not yet applied these changes.
But in the next OnTrade event, we see that the position is closed, and the market order has
disappeared.
>>> OnTrade(9)
PositionCache removed: 1311792104
OrderCache removed: 1311796883
>>> positions: 0, orders: 0, history: 2
If we monitor caches on every tick (or once per second, but not only for OnTrade events), we will see
changes in the ORDER_PRICE_CURRENT and POSITION_PRICE_CURRENT properties on the go.
POSITION_PROFIT will also change.
Our classes do not have persistence, that is, they live only in RAM and do not know how to save and
restore their state in any long-term storage, such as files. This means that the program may miss a
change that happened between terminal sessions. If you need such functionality, you should implement
it yourself. In the future, in Part 7 of the book, we will look at the built-in SQLite database support in
MQL5, which provides the most efficient and convenient way to store the trading environment cache
and similar tabular data.
6.4.38 Creating multi-symbol Expert Advisors
Until now, within the framework of the book, we have mainly analyzed examples of Expert Advisors
trading on the current working symbol of the chart. However, MQL5 allows you to generate trade
orders for any symbols of Market Watch, regardless of the working symbol of the chart.
In fact, many of the examples in the previous sections had an input parameter symbol, in which you can
specify an arbitrary symbol. By default, there is an empty string, which is treated as the current
symbol of the chart. So, we have already considered the following examples:
• CustomOrderSend.mq5 in Sending a trade request
• MarketOrderSend.mq5 in Buying and selling operations
• MarketOrderSendMonitor.mq5 in Functions for reading the properties of active orders
• PendingOrderSend.mq5 in Setting a pending order
• PendingOrderModify.mq5 in Modifying a pending order
• PendingOrderDelete.mq5 in Deleting a pending order
You can try to run these examples with a different symbol and make sure that trading operations are
performed exactly the same as with the native one.
Moreover, as we saw in the description of OnBookEven and OnTradeTransaction events, they are
universal and inform about changes in the trading environment concerning arbitrary symbols. But this is
not true for the OnTick event which is only generated when there is a change in the new prices of the
current symbol. Usually, this is not a problem, but high-frequency multicurrency trading requires some

---

## Page 1430

Part 6. Trading automation
1 430
6.4 Creating Expert Advisors
additional technical steps to be taken, such as subscribing to OnBookEvent events for other symbols or
setting a high-frequency timer. Another option to bypass this limitation in the form of a spy indicator
EventTickSpy.mq5 was presented in the section Generating custom events.
In the context of talking about the support of multi-symbol trading, it should be noted that a similar
concept of multi-timeframe Expert Advisors is not entirely correct. Trading at new bar opening times is
only a special case of grouping ticks by arbitrary periods, not necessarily standard ones. Of course, the
analysis of the emergence of a new bar on a specific timeframe is simplified by the system core due to
functions like iTime(_ Symbol, PERIOD_ XX, 0), but this analysis is based on ticks anyway.
You can build virtual bars inside your Expert Advisor by the number of ticks (equivolume), by the range
of prices (renko, range) and so on. In some cases, including for clarity, it makes sense to generate
such "timeframes" explicitly outside the Expert Advisor, in the form of custom symbols. But this
approach has its limitations: we will talk about them in the next part of the book.
However, if the trading system still requires analysis of quotes based on the opening of bars or uses a
multi-currency indicator, one should somehow wait for the synchronization of bars on all involved
instruments. We provided an example of a class that performs this task in the section Tracking bar
formation.
When developing a multi-symbol Expert Advisor, the imperative task involves segregating a universal
trading algorithm into distinct blocks. These blocks can subsequently be applied to various symbols with
differing settings. The most logical approach to achieve this is to articulate one or more classes within
the framework of the Object-Oriented Programming (OOP) concept.
Let's illustrate this methodology using an example of an Expert Advisor employing the well-known
martingale strategy. As is commonly understood, the martingale strategy is inherently risky, given its
practice of doubling lots after each losing trade in anticipation of recovering previous losses. Mitigating
this risk is essential, and one effective approach is to simultaneously trade multiple symbols, preferably
those with weak correlations. This way, temporary drawdowns on one instrument can potentially be
offset by gains on others.
The incorporation of a variety of instruments (or diverse settings within a single trading system, or even
distinct trading systems) within the Expert Advisor serves to diminish the overall impact of individual
component failures. In essence, the greater the diversity in instruments or systems, the less the final
result is contingent on the isolated setbacks of its constituent parts.
Let's call a new Expert Advisor MultiMartingale.mq5. Trading algorithm settings include:
• UseTime – logical flag for enabling/disabling scheduled trading
• HourStart and Hour End – the range of hours within which trading is allowed, if UseTime equals true
• Lots – the volume of the first deal in the series
• Factor – coefficient of increase in volume for subsequent transactions after a loss
• Limit – the maximum number of trades in a losing series with multiplication of volumes (after it,
return to the initial lot)
• Stop Loss and Take Profit – distance to protective levels in points
• StartType – type of the first deal (purchase or sale)
• Trailing – indication of a stop loss trailing
In the source code, they are described in this way.

---

## Page 1431

Part 6. Trading automation
1 431 
6.4 Creating Expert Advisors
input bool UseTime = true;      // UseTime (hourStart and hourEnd)
input uint HourStart = 2;       // HourStart (0...23)
input uint HourEnd = 22;        // HourEnd (0...23)
input double Lots = 0.01;       // Lots (initial)
input double Factor = 2.0;      // Factor (lot multiplication)
input uint Limit = 5;           // Limit (max number of multiplications)
input uint StopLoss = 500;      // StopLoss (points)
input uint TakeProfit = 500;    // TakeProfit (points)
input ENUM_POSITION_TYPE StartType = 0; // StartType (first order type: BUY or SELL)
input bool Trailing = true;     // Trailing
In theory, it is logical to establish protective levels not in points but in terms of shares of the Average
True Range indicator (ATR). However, at present, this is not a primary task.
Additionally, the Expert Advisor incorporates a mechanism to temporarily halt trading operations for a
user-specified duration (controlled by the parameter SkipTimeOnError) in case of errors. We will omit a
detailed discussion of this aspect here, as it can be referenced in the source codes.
To consolidate the entire set of configurations into a unified entity, a structure named Settings is
defined. This structure has fields that mirror input variables. Furthermore, the structure includes the
symbol field, addressing the strategy's multicurrency nature. In other words, the symbol can be
arbitrary and differs from the working symbol on the chart.
struct Settings
{
   bool useTime;
   uint hourStart;
   uint hourEnd;
   double lots;
   double factor;
   uint limit;
   uint stopLoss;
   uint takeProfit;
   ENUM_POSITION_TYPE startType;
   ulong magic;
   bool trailing;
   string symbol;
   ...
};
In the initial development phase, we populate the structure with input variables. Nevertheless, this is
only sufficient for trading on a single symbol. Subsequently, as we expand the algorithm to encompass
multiple symbols, we'll be required to read various sets of settings (using a different approach) and
append them to an array of structures.
The structure also encompasses several beneficial methods. Specifically, the validate method verifies
the correctness of the settings, confirming the existence of the specified symbol, and returns a success
indicator (true).

---

## Page 1432

Part 6. Trading automation
1 432
6.4 Creating Expert Advisors
struct Settings
{
   ...
   bool validate()
   {
 ...// checking the lot size and protective levels (see the source code)
      
      double rates[1];
      const bool success = CopyClose(symbol, PERIOD_CURRENT, 0, 1, rates) > -1;
      if(!success)
      {
         Print("Unknown symbol: ", symbol);
      }
      return success;
   }
   ...
};
Calling CopyClose not only checks if the symbol is online in the Market Watch but also initiates the
loading of its quotes (of the desired timeframe) and ticks in the tester. If this is not done, only quotes
and ticks (in real ticks mode) of the currently selected instrument and timeframe are available in the
tester by default. Since we are writing a multi-currency Expert Advisor, we will need third-party quotes
and ticks.
struct Settings
{
   ...
   void print() const
   {
      Print(symbol, (startType == POSITION_TYPE_BUY ? "+" : "-"), (float)lots,
        "*", (float)factor,
        "^", limit,
        "(", stopLoss, ",", takeProfit, ")",
        useTime ? "[" + (string)hourStart + "," + (string)hourEnd + "]": "");
   }
};
The print method outputs all fields to the log in abbreviated form in one line. For example,

---

## Page 1433

Part 6. Trading automation
1 433
6.4 Creating Expert Advisors
EURUSD+0.01*2.0^5(500,1000)[2,22]
|     | |   |   |  |    |   |  |
|     | |   |   |  |    |   |  `until this hour trading is allowed
|     | |   |   |  |    |   `from this hour trading is allowed
|     | |   |   |  |    `take profit in points
|     | |   |   |  `stop loss in points
|     | |   |   `maximum size of a series of losing trades (after '^')
|     | |   `lot multiplication factor (after '*')
|     | `initial lot in series
|     `+ start with Buy
|     `- start with Sell
`instrument
We will need other methods in the Settings structure when we move to multicurrency. For now, let's
imagine a simplified version of what the handler OnInit of the Expert Advisor trading on one symbol
might look like.
int OnInit()
{
   Settings settings =
   {
      UseTime, HourStart, HourEnd,
      Lots, Factor, Limit,
      StopLoss, TakeProfit,
      StartType, Magic, SkipTimeOnError, Trailing, _Symbol
   };
   
   if(settings.validate())
   {
      settings.print();
      ...
      // here you will need to initialize the trading algorithm with these settings
   }
   ...
}
Adhering to the OOP, the trading system in a generalized form should be described as a software
interface. Again, in order to simplify the example, we will only use one method in this interface: trade.
interface TradingStrategy
{
   virtual bool trade(void);
};
Well, the main task of the algorithm is to trade, and it doesn’t even matter where we decide to call this
method from: on each tick from OnTick, at the bar opening, or possibly on timer.
Your working Expert Advisors will most likely need additional interface methods to set up and support
various modes. But they are not needed in this example.
Let's start creating a class of a specific trading system based on the interface. In our case, all
instances will be of the class SimpleMartingale. However, it is also possible to implement many different
classes that inherit the interface within one Expert Advisor and then use them in a uniform way in an

---

## Page 1434

Part 6. Trading automation
1 434
6.4 Creating Expert Advisors
arbitrary combination. A portfolio of strategies (preferably very different in nature) is usually
characterized by increased stability of financial performance.
class SimpleMartingale: public TradingStrategy
{
protected:
   Settings settings;
   SymbolMonitor symbol;
   AutoPtr<PositionState> position;
   AutoPtr<TrailingStop> trailing;
   ...
};
Inside the class, we see a familiar Settings structure and the working symbol monitor SymbolMonitor.
In addition, we will need to control the presence of positions and follow the stop-loss level for them, for
which we have introduced variables with auto-pointers to objects PositionState and TrailingStop. Auto-
pointers allow us in our code not to worry about the explicit deletion of objects as this will be done
automatically when the control exits the scope, or when a new pointer is assigned to the auto-pointer.
The class TrailingStop is a base class, with the simplest implementation of price tracking, from
which you can inherit a lot of more complex algorithms, an example of which we considered as a
derivative TrailingStopByMA. Therefore, in order to give the program flexibility in the future, it is
desirable to ensure that the calling code can pass its own specific, customized trailing object,
derived from TrailingStop. This can be done, for example, by passing a pointer to the constructor or
by turning SimpleMartingale into a template class (then the trail class will be set by the template
parameter). 
This principle of OOP is called dependency inj ection and is widely used along with many others that
we briefly mentioned in the section Theoretical foundations of OOP: composition.
The settings are passed to the strategy class as a constructor parameter. Based on them, we assign all
internal variables.

---

## Page 1435

Part 6. Trading automation
1 435
6.4 Creating Expert Advisors
class SimpleMartingale: public TradingStrategy
{
   ...
   double lotsStep;
   double lotsLimit;
   double takeProfit, stopLoss;
public:
   SimpleMartingale(const Settings &state) : symbol(state.symbol)
   {
      settings = state;
      const double point = symbol.get(SYMBOL_POINT);
      takeProfit = settings.takeProfit * point;
      stopLoss = settings.stopLoss * point;
      lotsLimit = settings.lots;
      lotsStep = symbol.get(SYMBOL_VOLUME_STEP);
      
      // calculate the maximum lot in the series (after a given number of multiplications)
      for(int pos = 0; pos < (int)settings.limit; pos++)
      {
         lotsLimit = MathFloor((lotsLimit * settings.factor) / lotsStep) * lotsStep;
      }
      
      double maxLot = symbol.get(SYMBOL_VOLUME_MAX);
      if(lotsLimit > maxLot)
      {
         lotsLimit = maxLot;
      }
      ...
Next, we use the object PositionFilter to search for existing "own" positions (by the magic number and
symbol). If such a position is found, we create the PositionState object and, if necessary, the
TrailingStop object for it.

---

## Page 1436

Part 6. Trading automation
1 436
6.4 Creating Expert Advisors
      PositionFilter positions;
      ulong tickets[];
      positions.let(POSITION_MAGIC, settings.magic).let(POSITION_SYMBOL, settings.symbol)
         .select(tickets);
      const int n = ArraySize(tickets);
      if(n > 1)
      {
         Alert(StringFormat("Too many positions: %d", n));
      }
      else if(n > 0)
      {
         position = new PositionState(tickets[0]);
         if(settings.stopLoss && settings.trailing)
         {
           trailing = new TrailingStop(tickets[0], settings.stopLoss,
              ((int)symbol.get(SYMBOL_SPREAD) + 1) * 2);
         }
      }
   }
Schedule operations will be left "behind the scene" in the trade method for now (useTime, hourStart,
and hourEnd parameter fields). Let's proceed directly to the trading algorithm.
If there are no and have not been any positions yet, the PositionState pointer will be zero, and we need
to open a long or short position in accordance with the selected direction startType.
   virtual bool trade() override
   {
      ...
      ulong ticket = 0;
      
      if(position[] == NULL)
      {
         if(settings.startType == POSITION_TYPE_BUY)
         {
            ticket = openBuy(settings.lots);
         }
         else
         {
            ticket = openSell(settings.lots);
         }
      }
      ...
Helper methods openBuy and openSell are used here. We'll get to them in a couple of paragraphs. For
now, we only need to know that they return the ticket number on success or 0 on failure.
If the position object already contains information about the tracked position, we check whether it is
live by calling refresh. In case of success (true), update position information by calling update and also
trail the stop loss, if it was requested by the settings.

---

## Page 1437

Part 6. Trading automation
1 437
6.4 Creating Expert Advisors
      else // position[] != NULL
      {
         if(position[].refresh()) // does position still exists?
         {
            position[].update();
            if(trailing[]) trailing[].trail();
         }
         ...
If the position is closed, refresh will return false, and we will be in another if branch to open a new
position: either in the same direction, if a profit was fixed, or in the opposite direction, if a loss
occurred. Please note that we still have a snapshot of the previous position in the cache.
         else // the position is closed - you need to open a new one
         {
            if(position[].get(POSITION_PROFIT) >= 0.0) 
            {
              // keep the same direction:
              // BUY in case of profitable previous BUY
              // SELL in case of profitable previous SELL
               if(position[].get(POSITION_TYPE) == POSITION_TYPE_BUY)
                  ticket = openBuy(settings.lots);
               else
                  ticket = openSell(settings.lots);
            }
            else
            {
               // increase the lot within the specified limits
               double lots = MathFloor((position[].get(POSITION_VOLUME) * settings.factor) / lotsStep) * lotsStep;
   
               if(lotsLimit < lots)
               {
                  lots = settings.lots;
               }
             
               // change the trade direction:
               // SELL in case of previous unprofitable BUY
               // BUY in case of previous unprofitable SELL
               if(position[].get(POSITION_TYPE) == POSITION_TYPE_BUY)
                  ticket = openSell(lots);
               else
                  ticket = openBuy(lots);
            }
         }
      }
      ...
The presence of a non-zero ticket at this final stage means that we must start controlling it with new
objects PositionState and TrailingStop.

---

## Page 1438

Part 6. Trading automation
1 438
6.4 Creating Expert Advisors
      if(ticket > 0)
      {
         position = new PositionState(ticket);
         if(settings.stopLoss && settings.trailing)
         {
            trailing = new TrailingStop(ticket, settings.stopLoss,
               ((int)symbol.get(SYMBOL_SPREAD) + 1) * 2);
         }
      }
  
      return true;
    }
We now present, with some abbreviations, the openBuy method (openSell is all the same). It has three
steps:
• Preparing the MqlTradeRequestSync structure using the prepare method (not shown here, it fills
deviation and magic)
• Sending an order using a request.buy method call
• Checking the result with the postprocess method (not shown here, it calls request.completed and in
case of an error, the period of suspension of trading begins in anticipation of better conditions);
   ulong openBuy(double lots)
   {
      const double price = symbol.get(SYMBOL_ASK);
      
      MqlTradeRequestSync request;
      prepare(request);
      if(request.buy(settings.symbol, lots, price,
         stopLoss ? price - stopLoss : 0,
         takeProfit ? price + takeProfit : 0))
      {
         return postprocess(request);
      }
      return 0;
   }
Usually, positions will be closed by stop loss or take profit. However, we support scheduled operations
that may result in closures. Let's go back to the beginning of the trade method for scheduling work.

---

## Page 1439

Part 6. Trading automation
1 439
6.4 Creating Expert Advisors
   virtual bool trade() override
   {
      if(settings.useTime && !scheduled(TimeCurrent())) // time out of schedule?
      {
         // if there is an open position, close it
         if(position[] && position[].isReady())
         {
            if(close(position[].get(POSITION_TICKET)))
            {
                                // at the request of the designer:
               position = NULL; // clear the cache or we could...
               // do not do this zeroing, that is, save the position in the cache,
               // to transfer the direction and lot of the next trade to a new series
            }
            else
            {
               position[].refresh(); // guaranteeing reset of the 'ready' flag
            }
         }
         return false;
      }
      ...// opening positions (given above)
   }
Working method close is largely similar to openBuy so we will not consider it here. Another method,
scheduled, just returns true or false, depending on whether the current time falls within the specified
working hours range (hourStart, hourEnd).
So, the trading class is ready. But for multi-currency work, you will need to create several copies of it.
The TradingStrategyPool class will manage them, in which we describe an array of pointers to
TradingStrategy and methods for replenishing it: parametric constructor and push.

---

## Page 1440

Part 6. Trading automation
1 440
6.4 Creating Expert Advisors
class TradingStrategyPool: public TradingStrategy
{
private:
   AutoPtr<TradingStrategy> pool[];
public:
   TradingStrategyPool(const int reserve = 0)
   {
      ArrayResize(pool, 0, reserve);
   }
   
   TradingStrategyPool(TradingStrategy *instance)
   {
      push(instance);
   }
   
   void push(TradingStrategy *instance)
   {
      int n = ArraySize(pool);
      ArrayResize(pool, n + 1);
      pool[n] = instance;
   }
   
   virtual bool trade() override
   {
      for(int i = 0; i < ArraySize(pool); i++)
      {
         pool[i][].trade();
      }
      return true;
   }
};
It is not necessary to make the pool derived from the interface TradingStrategy interface, but if we do
so, this allows future packing of strategy pools into other larger strategy pools, and so on. The trade
method simply calls the same method on all array objects.
In the global context, let's add an autopointer to the trading pool, and in the OnInit handler we will
ensure its filling. We can start with one single strategy (we will deal with multicurrency a bit later).

---

## Page 1441

Part 6. Trading automation
1 441 
6.4 Creating Expert Advisors
AutoPtr<TradingStrategyPool> pool;
   
int OnInit()
{
   ... // settings initialization was given earlier
   if(settings.validate())
   {
      settings.print();
      pool = new TradingStrategyPool(new SimpleMartingale(settings));
      return INIT_SUCCEEDED;
   }
   else
   {
      return INIT_FAILED;
   }
   ...
}
To start trading, we just need to write the following small handler OnTick.
void OnTick()
{
   if(pool[] != NULL)
   {
      pool[].trade();
   }
}
But what about multicurrency support?
The current set of input parameters is designed for only one instrument. We can use this to test and
optimize the Expert Advisor on a single symbol, but after the optimal settings are found for all symbols,
they need to be somehow combined and passed to the algorithm.
In this case, we apply the simplest solution. The code above contained a line with the settings formed
by the print method generated by the Settings structures. We implement the method in the parse
structure which does the reverse operation: restores the state of the fields by the line description. Also,
since we need to concatenate several settings for different characters, we will agree that they can be
concatenated into a single long string through a special delimiter character, for example ';'. Then it is
easy to write the parseAll static method to read the merged set of settings, which will call parse to fill
the array of Settings structures passed by reference. The full source code of the methods can be found
in the attached file.
struct Settings
{
   ...
   bool parse(const string &line);
   void static parseAll(const string &line, Settings &settings[])
   ...
};  
For example, the following concatenated string contains settings for three symbols.

---

## Page 1442

Part 6. Trading automation
1 442
6.4 Creating Expert Advisors
EURUSD+0.01*2.0^7(500,500)[2,22];AUDJPY+0.01*2.0^8(300,500)[2,22];GBPCHF+0.01*1.7^8(1000,2000)[2,22]
It is lines of this kind that the method parseAll can parse. To enter such a string into the Expert
Advisor, we describe the input WorkSymbols variable.
input string WorkSymbols = ""; // WorkSymbols (name±lots*factor^limit(sl,tp)[start,stop];...)
If it is empty, the Expert Advisor will work with the settings from the individual input variables presented
earlier. If the string is specified, the OnInit handler will fill the pool of trading systems based on the
results of parsing this line.
int OnInit()
{
   if(WorkSymbols == "")
   {
      ...// work with the current single character, as before
   }
   else
   {
      Print("Parsed settings:");
      Settings settings[];
      Settings::parseAll(WorkSymbols, settings);
      const int n = ArraySize(settings);
      pool = new TradingStrategyPool(n);
      for(int i = 0; i < n; i++)
      {
         settings[i].trailing = Trailing;
         // support multiple systems on one symbol for hedging accounts
         settings[i].magic = Magic + i;  // different magic numbers for each subsystem
         pool[].push(new SimpleMartingale(settings[i]));
      }
   }
   return INIT_SUCCEEDED;
}
It's important to note that in MQL5, the length of the input string is restricted to 250 characters.
Additionally, during optimization in the tester, strings are further truncated to a maximum of 63
characters. Consequently, to optimize concurrent trading across numerous symbols, it becomes
imperative to devise an alternative method for loading settings, such as retrieving them from a text file.
This can be easily accomplished by utilizing the same input variable, provided it is designated with a file
name rather than a string containing settings.
This approach is implemented in the mentioned Settings::parseAll method. The name of the text file in
which an input string will be passed to the Expert Advisor without length limitation is set according to
the universal principle suitable for all similar cases: the file name begins with the name of the Expert
Advisor, and then, after the hyphen, there must be the name of the variable whose data the file
contains. For example, in our case, in the WorkSymbols input variable, you can optionally specify the
file name "MultiMartingale-WorkSymbols.txt". Then the parseAll method will try to read the text from
the file (it should be in the standard MQL5/Files sandbox).
Passing file names in input parameters requires additional steps to be taken for further testing and
optimization of such an Expert Advisor: the #property tester_ file "MultiMartingale-WorkSymbols.txt"
directive should be added to the source code. This will be discussed in detail in the section Tester

---

## Page 1443

Part 6. Trading automation
1 443
6.4 Creating Expert Advisors
preprocessor directives. When this directive is added, the Expert Advisor will require the presence of
the file and will not start without it in the tester!
The Expert Advisor is ready. We can test it on different symbols separately, choose the best settings
for each and build a trading portfolio. In the next chapter, we will study the tester API, including
optimization, and this Expert Advisor will come in handy. In the meantime, let's check its multi-
currency operation.
WorkSymbols=EURUSD+0.01*1.2^4(300,600)[9,11];GBPCHF+0.01*2.0^7(300,400)[14,16];AUDJPY+0.01*2.0^6(500,800)[18,16]
In the first quarter of 2022, we will receive the following report (MetaTrader 5 reports do not provide
statistics broken down by symbols, so it is possible to distinguish a single-currency report from a multi-
currency one only by the table of deals/orders/positions).
Tester's report for a multi-currency Martingale strategy Expert Advisor
It should be noted that due to the fact that the strategy is launched from the OnTick handler, the runs
on different main symbols (that is, those selected in the tester's settings drop-down list) will give
slightly different results. In our test, we simply used EURUSD as the most liquid and most frequently
ticked instrument, which is sufficient for most applications. However, if you want to react to ticks of all
instruments, you can use an indicator like EventTickSpy.mq5. Optionally, you can run the trading logic
on a timer without being tied to the ticks of a specific instrument.
And here is what the trading strategy looks like for a single symbol, in this case AUDJPY.

---

## Page 1444

Part 6. Trading automation
1 444
6.4 Creating Expert Advisors
Chart with a test of a multi-currency Martingale strategy Expert Advisor
By the way, for all multicurrency Expert Advisors, there is another important issue that is left
unattended here. We are talking about the method of selecting the lot size, for example, based on the
loading of the deposit or risk. Earlier, we showed examples of such calculations in a non-trading Expert
Advisor LotMarginExposureTable.mq5. In MultiMartingale.mq5, we have simplified the task by choosing a
fixed lot and displaying it in the settings for each symbol. However, in operational multicurrency Expert
Advisors, it makes sense to choose lots in proportion to the value of the instruments (by margin or
volatility).
In conclusion, I would like to note that multi-currency strategies may require different optimization
principles. The considered strategy makes it possible to separately find parameters for symbols and
then combine them. However, some arbitrage and cluster strategies (for example, pair trading) are
based on the simultaneous analysis of all tools for making trading decisions. In this case, the settings
associated with all symbols should be separately included in the input parameters.
6.4.39 Limitations and benefits of Expert Advisors
Due to their specific operation, Expert Advisors have some limitations, as well as advantages over other
types of MQL programs. In particular, all functions intended for indicators are banned in Expert
Advisors:
·SetIndexBuffer
·IndicatorSetDouble
·IndicatorSetInteger
·IndicatorSetString
·PlotIndexSetDouble
·PlotIndexSetInteger

---

## Page 1445

Part 6. Trading automation
1 445
6.4 Creating Expert Advisors
·PlotIndexSetString
·PlotIndexGetInteger
Also, Expert Advisors should not describe event handlers that are typical for other types of programs:
OnStart (scripts and services) and OnCalculate (indicators).
Unlike indicators, only one Expert Advisor can be placed on each chart.
At the same time, Expert Advisors are the only type of MQL programs that in addition to testing (which
we have already done for both indicators and Expert Advisors), can also be optimized. The optimizer
allows finding the best input parameters according to various criteria, both trading and abstract
mathematical ones. For these purposes, the API includes additional functions and several specific event
handlers. We will study this material in the next chapter.
In addition, groups of built-in MQL5 functions for working with the network at the socket level and
various Internet protocols (HTTP, FTP, SMTP) are available in Expert Advisors (as well as in scripts and
services, that is, in all types of programs except indicators). We will consider them in the seventh part
of the book.
6.4.40 Creating Expert Advisors in the MQL Wizard
So, we are completing the study of trading APIs for developing Expert Advisors. Throughout this
chapter, we have considered various examples, which you can use as a starting point for your own
project. However, if you want to start an Expert Advisor from scratch, you don't have to do it literally
"from scratch". The MetaEditor provides the built-in MQL Wizard which, among other things, allows the
creation of Expert Advisor templates. Moreover, in the case of Expert Advisor, this Wizard offers two
different ways to generate source code.
We already got acquainted with the first step of the Wizard in the section MQL Wizard and program
draft. Obviously, in the first step, we select the type of project to be created. In the previously
mentioned chapter, we created a script template. Later, in the chapter on indicators, we took a tour of
creating an indicator template. Now we will consider the following two options:
• Expert Advisor (template)
• Expert Advisor (generate)
The first one is more simple. You can select a name, input parameters, and required event handlers, as
shown in screenshots below, but there will be no trading logic and ready-made algorithms in the
resulting source file.
The second option is more complicated. It will result in a ready-made Expert Advisor based on the
standard library that provides a set of classes in header files available in the standard MetaTrader 5
package. Files are located in folders MQL5/Include/Expert/, MQL5/Include/Trade,
MQL5/Include/Indicators, and several others. The library classes implement the most popular indicator
signals, mechanisms for performing trading operations based on combinations of signals, as well as
money management and trailing stop algorithms. The detailed study of the standard library is beyond
the scope of this book.
Regardless of which options you select, at the second step of the Wizard, you need to enter the Expert
Advisor name and input parameters. The appearance of this step is similar to what was also already
shown in the section MQL Wizard and program draft. The only caveat is that Expert Advisors based on
the standard library must have two mandatory (non-removable) parameters: Symbol and TimeFrame.

---

## Page 1446

Part 6. Trading automation
1 446
6.4 Creating Expert Advisors
For a simple template, at the 3rd step, it is proposed to select additional event handlers that will be
added to the source code, in addition to OnTick (OnTick always inserted).
Creation of an Expert Advisor template. Step 3. Additional event handlers
The final fourth step allows you to specify one or more optional event handlers for the tester. Those will
be discussed in the next chapter.


---

## Page 1447

Part 6. Trading automation
1 447
6.4 Creating Expert Advisors
Creation of an Expert Advisor template. Step 4. Tester event handlers
If the user chooses to generate a program based on the standard library at the first step of the Wizard,
then the 3rd step is to set up trading signals.
Generation of a ready Expert Advisor. Step 3. Setting up trading signals
You can read more about it in the documentation.
Steps 4 and 5 are designed to include trailing in the Expert Advisor and automatically select lots
according to one of the predefined methods.

---

## Page 1448

Part 6. Trading automation
1 448
6.4 Creating Expert Advisors
Generation of a ready Expert Advisor. Step 4. Choosing a trailing stop method
Generation of a ready Expert Advisor. Step 5. Selection of lots

---

## Page 1449

Part 6. Trading automation
1 449
6.4 Creating Expert Advisors
The Wizard, of course, is not a universal tool, and the resulting program prototype, as a rule, needs to
be improved. However, the knowledge gained in this chapter will give you more confidence in the
generated source codes and extend them as needed.
6.5 Testing and optimization of Expert Advisors
Development of Expert Advisors implies not only and not so much the implementation of a trading
strategy in MQL5 but to a greater extent testing its financial performance, finding optimal settings, and
debugging (searching for and correcting errors) in various situations. All this can be done in the
integrated MetaTrader 5 tester.
The tester works for various currencies and supports several tick generation modes: based on opening
prices of the selected timeframe, on OHLC prices of the M1  timeframe, on artificially generated ticks,
and on the real tick history. This way you can choose the optimal ratio of speed and accuracy of
trading simulation.
The tester settings allow you to set the testing time interval in the past, the size of the deposit, and the
leverage; they are used to emulate requotes and specific account features (including the size of
commissions, margins, session schedules, limiting the number of lots). All the details of working with
the tester from the user's point of view can be found in terminal documentation.
Earlier, we already briefly discussed working with the tester, in particular, in the section Testing
indicators. Let's recall that the tester control functions and their optimization are not available for
indicators, unlike for Expert Advisors. However, personally, I would like to see an option of adaptive self-
tuning of indicators: all that is needed is to support the OnTester handler in them, which we will present
in a separate section.
As you know, various modes are available for optimization, such as direct enumeration of combinations
of Expert Advisor input parameters, accelerated genetic algorithm, mathematical calculations, or
sequential runs through symbols in Market Watch. As an optimization criterion, you can use both well-
known metrics such as profitability, Sharpe ratio, recovery factor, and expected payoff, as well as
"custom" variables embedded in the source code by the developer of the Expert Advisor. In the context
of this book, it is assumed that the reader is already familiar with the principles of setting up, running,
and interpreting optimization results because in this chapter we will begin to study the tester control
API. Those interested can refresh their knowledge with the help of the relevant section of
documentation.
A particularly important function of the tester is multi-threaded optimization, which can be performed
using local and distributed (network) agent programs, including those in the MQL5 Cloud Network. A
single testing run (with specific input parameters) launched manually by the user, or one of the many
runs called during optimization (when we implement enumeration of parameter values in given ranges)
is performed in a separate program – the agent. Technically, this is a metatester64.exe file, and the
copies of its processes can be seen in the Windows Task Manager during testing and optimization. It is
due to this that the tester is multi-threaded.
The terminal is a dispatcher that distributes tasks to local and remote agents. It launches local agents
if necessary. When optimizing, by default, several agents are launched; their quantity corresponds to
the number of processor cores. After executing the next task for testing an Expert Advisor with the
specified parameters, the agent returns the results to the terminal.
Each agent creates its own trading and software environment. All agents are isolated from each other
and from the client terminal.

---

## Page 1450

Part 6. Trading automation
1 450
6.5 Testing and optimization of Expert Advisors
In particular, the agent has its own global variables and its own file sandbox, including the folder where
detailed agent logs are written: Tester/Agent-IPaddress-Port/Logs. Here Tester is the tester installation
directory (during a standard installation together with MetaTrader 5, this is the subfolder where the
terminal is installed). The name of the directory Agent-IPaddress-Port, instead of IPaddress and Port,
will contain the specific network address and port values that are used to communicate with the
terminal. For local agents, this is the address 1 27.0.0.1  and the range of ports, by default, starting
from 3000 (for example, on a computer with 4 cores, we will see agents on ports 3000, 3001 , 3002,
3003).
When testing an Expert Advisor, all file operations are performed in the Tester/Agent-IPaddress-
Port/MQL5/Files folder. However, it is possible to implement interaction between local agents and the
client terminal (as well as between different copies of the terminal on the same computer) via a shared
folder. For this, when opening a file with the FileOpen function, the FILE_COMMON flag must be
specified. Another way to transfer data from agents to the terminal is provided by the frames
mechanism.
The agent's local sandbox is automatically cleared before each test due to security reasons (to prevent
different Expert Advisors from reading each other's data).
A folder with the quotes history is created next to the file sandbox for each agent: Tester/Agent-
IPaddress-Port/bases/ServerName/Symbol/. In the next section, we briefly remind you how it is
formed.
The results of individual test runs and optimizations are stored by the terminal in a special cache which
can be found in the installation directory, in the subfolder Tester/cache/. Test results are stored in files
with the extension tst, and the optimization results are stored in opt files. Both formats are open-
sourced by MetaQuotes developers, so you can implement your own batch analytical data processing,
or use ready-made source codes from the codebase on the mql5.com website.
In this chapter, first, we will consider the basic principles of how MQL programs work in the tester, and
then we will learn how to interact with it in practice.
6.5.1  Generating ticks in tester
The presence of the OnTick handler in the Expert Advisor is not mandatory for it to be tested in the
tester. The Expert Advisor can use one or more of the other familiar functions:
• OnTick – event handler for the arrival of a new tick
• OnTrade – trade event handler
• OnTradeTransaction – trade transaction handler
• OnTimer – timer signal handler
• OnChartEvent – event handler on the chart, including custom charts
At the same time, inside the tester, the main equivalent of the time course is a thread of ticks, which
contain not only price changes but also time accurate to milliseconds. Therefore, to test Expert
Advisors, it is necessary to generate tick sequences. The MetaTrader 5 tester has 4 tick generation
modes:
• Real ticks (if their history is provided by the broker)
• Every tick (emulation based on available M1  timeframe quotes)
• OHLC prices from minute bars (1  Minute OHLC)

---

## Page 1451

Part 6. Trading automation
1 451 
6.5 Testing and optimization of Expert Advisors
• Open prices only (1  tick per bar)
Another mode of operation – mathematical calculations – we will analyze later since it is not related to
quotes and ticks.
Whichever of the 4 modes the user chooses, the terminal loads the available historical data for testing.
If the mode of real ticks was selected, and the broker does not have them for this instrument, then the
"All ticks" mode is used. The tester indicates the nature of tick generation in its report graphically and
as a percentage (where 1 00% means all ticks are real).
The history of the instrument selected in the tester settings is synchronized and downloaded by the
terminal from the trading server before starting the testing process. At the same time, for the first
time, the terminal downloads the history from the trading server to the required depth (with a certain
margin, depending on the timeframe, at least 1  year before the start of the test), so as not to apply for
it later. In the future, only the download of new data will occur. All this is accompanied by
corresponding messages in the tester's log.
The testing agent receives the history of the tested instrument from the client terminal immediately
after testing is started. If the testing process uses data on other instruments (for example, this is a
multicurrency Expert Advisor), then in this case the testing agent requests the required history from
the client terminal at the first call. If historical data is available on the terminal, they are immediately
transferred to testing agents. If the data is missing, the terminal will request and download it from the
server, and then transfer it to the testing agents.
Additional instruments are also used when the cross-rate price is calculated during trading operations.
For example, when testing a strategy on EURCHF with a deposit currency in US dollars, before
processing the first trading operation, the testing agent will request the history of EURUSD and
USDCHF from the client terminal, although the strategy does not directly refer to these instruments.
In this regard, before testing a multicurrency strategy, it is recommended that you first download all
the necessary historical data into the client terminal. This will assist in avoiding testing/optimization
delays associated with resuming data. You can download the history, for example, by opening the
corresponding charts and scrolling them to the beginning of the history.
Now let's look at tick generation modes in more detail.
Real ticks from history
Testing and optimization on real ticks are as close to real conditions as possible. These are ticks from
exchanges and liquidity providers.
If there is a minute bar in the symbol's history, but no tick data for that minute, the tester will
generate ticks in the "Every tick" mode (see further). This allows you to build the correct chart in
the tester in case of incomplete tick data from the broker. Moreover, tick data may not match
minute bars for various reasons. For example, due to disconnections or other failures in the
transmission of data from the source to the client terminal. When testing, minute data is considered
more reliable.
Ticks are stored in the symbol cache in the strategy tester. The cache size is no more than 1 28,000
ticks. When new ticks arrive, the oldest data is pushed out of it. However, using the CopyTicks function,
you can get ticks outside the cache (only when testing using real ticks). In this case, the data will be
requested from the tester's tick database, which fully corresponds to the similar database of the client
terminal. No adjustments by minute bars are made to this base. Therefore, the ticks in it may differ
from the ticks in the cache.

---

## Page 1452

Part 6. Trading automation
1 452
6.5 Testing and optimization of Expert Advisors
Every tick (emulation)
If the real tick history is not available or if you need to minimize network traffic (because the archive of
real ticks can consume significant resources), you can choose to artificially generate ticks based on
the available quotes of the M1  timeframe.
The history of quotes for financial instruments is transmitted from the trading server to the MetaTrader
5 client terminal in the form of tightly packed blocks of minute bars. The history query procedure and
the constructions of the required timeframes were considered in detail in the section Technical features
of organization and storage of timeseries.
The minimum element of the price history is a minute bar, from which you can get information about
four OHLC price values: Open, High, Low, and Close.
A new minute bar opens not at the moment when a new minute begins (the number of seconds
becomes 0) but when a tick – a price change of at least one point – occurs. Similarly, we cannot
determine from the bar with accuracy of a second when the tick corresponding to the closing price of
this minute bar arrived: we only know the last price of the one-minute bar, which was recorded as the
Close price.
Thus, for each minute bar, we know 4 control points, which we can say for sure that the price has been
there. If the bar has only 4 ticks, then this information is enough for testing, but usually, the tick
volume is more than 4. This means that it is necessary to generate additional checkpoints for ticks that
came between prices Open, High, Low, and Close. The basics of generating ticks in the "Every tick"
mode are described in the documentation.
When testing in the "Every tick" mode, the OnTick function of the Expert Advisor will be called on every
generated tick. The Expert Advisor will receive time and Ask/Bid/Last prices the same way as when
working online.
The "Every tick" testing mode is the most accurate (after the real ticks mode), but also the most time-
consuming. For the primary evaluation of most trading strategies, it is usually sufficient to use one of
two simplified testing modes: at OHLC M1  prices or at the opening of bars of the selected timeframe.
1  minute OHLC
In the "1  minute OHLC" mode, the tick sequence is built only by OHLC prices of minute bars, the
number of the OnTick function calls is significantly reduced; hence, the testing time is also reduced.
This is a very efficient, useful mode that offers a compromise between testing accuracy and speed.
However, you need to be careful with it when it comes to someone else's Expert Advisor.
Refusal to generate additional intermediate ticks between prices Open, High, Low, and Close leads to
the appearance of rigid determinism in the development of prices from the moment the Open price is
defined. This makes it possible to create a "Testing Grail" that shows a nice upward trending balance
chart when testing.
For a minute bar, 4 prices are known, of which the first one is Open, and the last one is Close.
Prices registered between them are High and Low, and information about the order of their
occurrence is lost, but we know that the High price is greater than or equal to Open, and Low is less
than or equal to Open. 
After receiving the Open price, we need to analyze only the next tick to determine whether it is High
or Low. If the price is below Open, this is Low – buy on this tick, as the next tick will correspond to
the High price, on which we close the buy trade and open a sell one. The next tick is the last on the
bar, Close, on which we close sell. 

---

## Page 1453

Part 6. Trading automation
1 453
6.5 Testing and optimization of Expert Advisors
If a tick with a price higher than the opening price comes after our price, then the sequence of
transactions is reversed. Seemingly, one could trade on every bar in this mode. When testing such
an Expert Advisor on history, everything goes perfectly, but online it will fail.
A similar effect can happen unintentionally, due to a combination of features of the calculation
algorithm (for example, statistics calculation) and tick generation.
Thus, it is always important to test it in the "Every tick" mode or, better, based on real ticks after
finding the optimal Expert Advisor settings on rough testing modes ("1  minute OHLC" and "Only Open
Prices").
Open prices only
In this mode, ticks are generated using the OHLC prices of the timeframe selected for testing. In this
case, the OnTick function runs only once, at the beginning of each bar. Because of this feature, stop
levels and pending orders may be triggered at a price different from the requested one (especially when
testing on higher timeframes). In exchange for this, we get the opportunity to quickly conduct
evaluation testing of the Expert Advisor.
For example, the Expert Advisor is tested on EURUSD H1  in the "Open price only" mode. In this case,
the total number of ticks (control points) will be 4 times more than the number of hourly bars that fall
within the tested interval. But in this case, the OnTick handler will only be called at the opening of
hourly bars. For the rest ticks ("hidden" from the Expert Advisor), the following checks required for
correct testing are performed:
• calculation of margin requirements
• triggering of Stop Loss and Take Profit
• triggering pending orders
• deleting pending orders upon expiration
If there are no open positions or pending orders, then there is no need for these checks on hidden
ticks, and the speed increase can be significant.
An exception when generating ticks in the "Open prices only" mode are the W1  and MN1  periods: for
these timeframes, ticks are generated for the OHLC prices of each day, not weekly or monthly,
respectively.
This "Open prices only" mode is well suited for testing strategies that perform trades only at the
opening of the bar and do not use pending orders, and do not use Stop Loss and Take Profit levels. For
the class of such strategies, all the necessary testing accuracy is preserved.
The MQL5 API does not allow the program to find out in which mode it is running in the tester. At the
same time, this may be important for Expert Advisors or the indicators they use, which are not
designed, for example, to work correctly at opening prices or OHLC. In this regard, we implement a
simple mode detection mechanism. The source code is attached in the file TickModel.mqh.
Let's declare our enumeration with the existing modes.

---

## Page 1454

Part 6. Trading automation
1 454
6.5 Testing and optimization of Expert Advisors
enum TICK_MODEL
{
   TICK_MODEL_UNKNOWN = -1,    /*Unknown (any)*/    // unknown/not yet defined
   TICK_MODEL_REAL = 0,        /*Real ticks*/       // best quality
   TICK_MODEL_GENERATED = 1,   /*Generated ticks*/  // good quality
   TICK_MODEL_OHLC_M1 = 2,     /*OHLC M1*/          // acceptable quality and fast
   TICK_MODEL_OPEN_PRICES = 3, /*Open prices*/      // worse quality, but very fast
   TICK_MODEL_MATH_CALC = 4,   /*Math calculations*/// no ticks (not defined)
};
Except the first element, which is reserved for the case when the mode has not yet been determined or
cannot be determined for some reason, all other elements are arranged in descending order of
simulation quality, starting from real and ending with opening prices (for them, the developer must
check the compatibility strategy with the fact that its trading is carried out only at the opening of a
new bar). The last mode TICK_MODEL_MATH_CALC operates without ticks altogether; we will consider
it separately.
The mode detection principle is based on the check of the availability of ticks and their times on the
first two ticks when starting the test. The check itself is wrapped in the getTickModel function, which
the Expert Advisor should call from the OnTick handler. Since the check is done once, the static variable
model is described inside the function initially set to TICK_MODEL_UNKNOWN. It will store and switch
the current state of the check, which will be required to distinguish between OHLC modes and opening
prices.
TICK_MODEL getTickModel()
{
   static TICK_MODEL model = TICK_MODEL_UNKNOWN;
   ...
On the first analyzed tick, the model is equal to TICK_MODEL_UNKNOWN, and an attempt is made to
get real ticks by calling CopyTicks.

---

## Page 1455

Part 6. Trading automation
1 455
6.5 Testing and optimization of Expert Advisors
   if(model == TICK_MODEL_UNKNOWN)
   {
      MqlTick ticks[];
      const int n = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, 10);
      if(n == -1)
      {
         switch(_LastError)
         {
         case ERR_NOT_ENOUGH_MEMORY:    // emulate ticks
            model = TICK_MODEL_GENERATED;
            break;
            
         case ERR_FUNCTION_NOT_ALLOWED: // prices of opening and OHLC
            if(TimeCurrent() != iTime(_Symbol, _Period, 0))
            {
               model = TICK_MODEL_OHLC_M1;
            }
            else if(model == TICK_MODEL_UNKNOWN)
            {
               model = TICK_MODEL_OPEN_PRICES;
            }
            break;
         }
         
         Print(E2S(_LastError));
      }
      else
      {
         model = TICK_MODEL_REAL;
      }
   }
   ...
If it succeeds, the detection immediately ends with setting the model to TICK_MODEL_REAL. If real
ticks are not available, the system will return a certain error code, according to which we can draw the
following conclusions. The error code ERR_NOT_ENOUGH_MEMORY corresponds to the tick emulation
mode. Why the code is this way is not entirely clear, but this is a characteristic feature, and we use it
here. In the other two tick generation modes, we will get the ERR_FUNCTION_NOT_ALLOWED error.
You can distinguish one mode from the other by the tick time. If it turns out to be a non-multiple of the
timeframe for a tick, then we are talking about the OHLC mode. However, the problem here is that the
first tick in both modes can be aligned with the bar opening time. Thus, we will get the value
TICK_MODEL_OPEN_PRICES, but it needs to be specified. Therefore, for the final conclusion, one more
tick should be analyzed (call the function on it again if TICK_MODEL_OPEN_PRICES was received
earlier). For this case, the following if branch is provided inside the function.

---

## Page 1456

Part 6. Trading automation
1 456
6.5 Testing and optimization of Expert Advisors
   else if(model == TICK_MODEL_OPEN_PRICES)
   {
      if(TimeCurrent() != iTime(_Symbol, _Period, 0))
      {
         model = TICK_MODEL_OHLC_M1;
      }
   }
   return model;
}
Let's check the operation of the detector in a simple Expert Advisor TickModel.mq5. In the TickCount
input parameter, we specify the maximum number of analyzed ticks, that is, how many times the
getTickModel function will be called. We know that two is enough, but in order to make sure that the
model does not change afterward, 5 ticks are suggested by default. We also provide the
RequireTickModel parameter which instructs the Expert Advisor to terminate operation if the simulation
level is lower than the requested one. By default, its value is TICK_MODEL_UNKNOWN, which means no
mode restriction.
input int TickCount = 5;
input TICK_MODEL RequireTickModel = TICK_MODEL_UNKNOWN;
In the OnTick handler, we run our code only if it works in the tester.

---

## Page 1457

Part 6. Trading automation
1 457
6.5 Testing and optimization of Expert Advisors
void OnTick()
{
   if(MQLInfoInteger(MQL_TESTER))
   {
      static int count = 0;
      if(count++ < TickCount)
      {
         // output tick information for reference
         static MqlTick tick[1];
         SymbolInfoTick(_Symbol, tick[0]);
         ArrayPrint(tick);
         // define and display the model (preliminarily)
         const TICK_MODEL model = getTickModel();
         PrintFormat("%d %s", count, EnumToString(model));
         // if the tick counter is 2+, the conclusion is final and we act based on it
         if(count >= 2)
         {
            if(RequireTickModel != TICK_MODEL_UNKNOWN
            && RequireTickModel < model) // quality less than requested
            {
               PrintFormat("Tick model is incorrect (%s %sis required), terminating",
                  EnumToString(RequireTickModel),
                  (RequireTickModel != TICK_MODEL_REAL ? "or better " : ""));
               ExpertRemove(); // end operation
            }
         }
      }
   }
}
Let's try to run the Expert Advisor in the tester with different tick generation modes by choosing a
common combination of EURUSD H1 .
The RequireTickModel parameter in the Expert Advisor is set to OHLC M1 . If the tester mode is "Every
tick", we will receive a corresponding message in the log, and the Expert Advisor will continue working.
                 [time]   [bid]   [ask]  [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 00:00:30 1.10656 1.10679 1.10656        0 1648771230000      14       0.00000
NOT_ENOUGH_MEMORY
1 TICK_MODEL_GENERATED
                 [time]   [bid]   [ask]  [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 00:01:00 1.10656 1.10680 1.10656        0 1648771260000      12       0.00000
2 TICK_MODEL_GENERATED
                 [time]   [bid]   [ask]  [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 00:01:30 1.10608 1.10632 1.10608        0 1648771290000      14       0.00000
3 TICK_MODEL_GENERATED
The OHLC M1  and real ticks modes are also suitable, and in the latter case, there will be no error code.

---

## Page 1458

Part 6. Trading automation
1 458
6.5 Testing and optimization of Expert Advisors
                 [time]   [bid]   [ask] [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 00:00:00 1.10656 1.10687 0.0000        0 1648771200122     134       0.00000
1 TICK_MODEL_REAL
                 [time]   [bid]   [ask] [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 00:00:00 1.10656 1.10694 0.0000        0 1648771200417       4       0.00000
2 TICK_MODEL_REAL
                 [time]   [bid]   [ask] [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 00:00:00 1.10656 1.10691 0.0000        0 1648771200816       4       0.00000
3 TICK_MODEL_REAL
However, if you change the mode in the tester to "Open prices only", the Expert Advisor will stop after
the second tick.
                 [time]   [bid]   [ask]  [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 00:00:00 1.10656 1.10679 1.10656        0 1648771200000      14       0.00000
FUNCTION_NOT_ALLOWED
1 TICK_MODEL_OPEN_PRICES
                 [time]   [bid]   [ask]  [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.04.01 01:00:00 1.10660 1.10679 1.10660        0 1648774800000      14       0.00000
2 TICK_MODEL_OPEN_PRICES
Tick model is incorrect (TICK_MODEL_OHLC_M1 or better is required), terminating
ExpertRemove() function called
This method requires running a test and waiting for a couple of ticks in order to determine the mode. In
other words, we cannot stop the test early by returning an error from OnInit. Even more, when starting
an optimization with the wrong type of tick generation, we will not be able to stop the optimization,
which can only be done from the OnTesterInit function. Thus, the tester will try to complete all passes
during the optimization, although they will be stopped at the very beginning. This is the current
platform limitation.
6.5.2 Time management in the tester: timer, Sleep, GMT
When developing Expert Advisors, it should be taken into account that the tester has some specifics of
simulating the passage of time based on generated ticks and operation of time-related functions.
When testing, the local time returned by the TimeLocal function is always equal to the server time
according to TimeTradeServer. In turn, server time is always equal to GMT TimeGMT. Thus, all these
functions, when tested, give the same time. This is a technical feature of the platform, which occurs
because it was decided not to store information about the server time locally, but always take it from
the server, with which there may be no connection at a particular moment.
This feature creates difficulties in the implementation of strategies related to global time, in particular,
with reference to news releases. In such cases, it is necessary to specify the time zone of quotes in the
settings of the Expert Advisor being tested or to invent methods for auto-detection of the time zone
(see section Daylight saving time).
Let's turn now to other functions for working with time.
As we know, it is possible to process timer events in MQL5. The OnTimer handler is called regardless of
the testing mode. This means that if testing is launched in the "Open prices only" mode on the H4
period, and a timer is set inside the Expert Advisor with a call coming every second, then the OnTick
handler will be called once at the opening of each H4 bar and then, within the bar, the OnTimer handler
will be called 1 4400 times (3600 seconds * 4 hours). The extent to which the Expert Advisor testing
time will increase in this case depends on its algorithm.

---

## Page 1459

Part 6. Trading automation
1 459
6.5 Testing and optimization of Expert Advisors
Another function that influences the course of time within a program is the Sleep function. It allows you
to suspend the execution of an Expert Advisor for some time. This may be necessary when requesting
any data that is not yet ready at the time of the request, and it is necessary to wait until it is ready.
It is important to understand that Sleep affects only the program that calls it and does not delay the
testing process. In fact, when calling Sleep, the generated ticks are "played" within the specified delay,
as a result of which pending orders, stop levels, etc. can be triggered. After calling Sleep, the time
simulated in the tester is increased by the interval specified in the function parameter.
Later, in the section on testing multi-currency Expert Advisors, we will show how you can use the timer
and the Sleep function to synchronize bars.
6.5.3 Testing visualization: chart, objects, indicators
The tester allows testing in two different ways: with and without visualization. The method is selected
by choosing a corresponding option on the main settings tab of the tester. 
When visualization is enabled, the tester opens a separate window in which it reproduces trading
operations and displays indicators and objects. Though it is visual, we don't need to see it for every
case, but only for programs with a user interface (for example, trading panels or controlled markup
made by graphical objects). For other Expert Advisors, only the execution of the algorithm according to
the established strategy is important. This can be checked without visualization, which can significantly
speed up the process. By the way, it is in this mode that test runs are made during optimization.
During such "background" testing and optimization, no graphical objects are built. Therefore, when
accessing the properties of objects, the Expert Advisor will receive zero values. Thus, you can check
the work with objects and the chart only when testing in the visual mode.
Previously, in the Testing indicators section, we have seen the specific behavior of indicators in the
tester. To increase the efficiency of non-visual testing and optimization of Expert Advisors (using
indicators), indicators can be calculated not on every tick, but only when we request data from them.
Recalculation on each tick occurs only if there are EventChartCustom, OnChartEvent, OnTimer functions
or tester_ everytick_ calculate directives in the indicator (see Preprocessor directives for the tester). In
the visual tester window online indicators always receive OnCalculate events on every tick.
If testing is carried out in a non-visual mode, after its completion, the symbol chart automatically
opens in the terminal, which displays completed deals and indicators that were used in the Expert
Advisor. This helps to correlate the market entry and exit moments with indicator values. However,
here we mean only indicators that work on the symbol and timeframe of testing. If the Expert Advisor
created indicators on other symbols or timeframes, they will not be shown.
It is important to note that the indicators displayed on the chart automatically opened after testing
is completed are recalculated after the end of testing. This happens even if these indicators were
used in the tested Expert Advisor and were previously calculated "on the go", as the bars were
forming.
In some cases, the programmer may need to hide information about which indicators are used in the
trading algorithm, and therefore their visualization on the chart is undesirable. The IndicatorRelease
function can be used for this.
The IndicatorRelease function is originally intended to release the calculated part of the indicator if it is
no longer needed. This saves memory and processor resources. Its second purpose is to prohibit the
display of the indicator on the testing chart after completing a single run.

---

## Page 1460

Part 6. Trading automation
1 460
6.5 Testing and optimization of Expert Advisors
To disable the display of the indicator on the chart at the end of testing, just call IndicatorRelease with
the indicator handle in the OnDeinit handler. The OnDeinit function is always called in Expert Advisors
after completion and before displaying the test chart. Neither OnDeinit nor the destructors of global
and static objects are called in the indicators themselves in the tester – this is what the developers of
MetaTrader 5 agreed on.
In addition, the MQL5 API includes a special function TesterHideIndicators with a similar purpose, which
we will consider later.
At the same time, it should be taken into account that tpl templates (if they are created) can
additionally influence the external representation of the testing graph.
So if there is a tester.tpl template in the MQL5/Profiles/Templates directory, it will be applied to the
opened chart. If the Expert Advisor used other indicators in its work and did not prohibit their display,
then the indicators from the template and from the Expert Advisor will be combined on the chart.
When tester.tpl is absent, the default template (default.tpl) is applied.
If the MQL5/Profiles/Templates folder contains a tpl template with the same name as the Expert Advisor
(for example, ExpertMACD.tpl), then during visual testing or on the chart opened after testing, only
indicators from this template will be shown. In this case, no indicators used in the tested Expert Advisor
will be shown.
6.5.4 Multicurrency testing
As you know, the MetaTrader 5 tester allows you to test strategies that trade multiple financial
instruments. Purely technically, subject to the computer hardware resources, it is possible to simulate
simultaneous trading for all available instruments.
Testing such strategies imposes several additional technical requirements on the tester:
• Generation of tick sequences for all instruments
• Calculation of indicators for all instruments
• Calculation of margin requirements and emulation of other trading conditions for all instruments
The tester automatically downloads the history of required instruments from the terminal when
accessing the history for the first time. If the terminal does not contain the required history, it will in
turn request it from the trade server. Therefore, before testing a multicurrency Expert Advisor, it is
recommended to select the required instruments in the terminal's Market Watch and download the
desired amount of data.
The agent uploads the missing history with a small margin to provide the necessary data for calculating
indicators or copying by the Expert Advisor at the time of testing. The minimum amount of history
downloaded from the trading server depends on the timeframe. For example, for D1  timeframes and
less, it is one year. In other words, the preliminary history is downloaded from the beginning of the
previous year relative to the tester start date. This gives at least 1  year of history if testing is
requested from January 1 st and a maximum of almost two years if testing is ordered from December.
For a weekly timeframe, a history of 1 00 bars is requested, that is, approximately two years (there are
52 weeks in a year). For testing on a monthly timeframe, the agent will request 1 00 months (equal to
the history of about 8 years: 1 2 months * 8 years = 96). In any case, on timeframes lower than the
working one, a proportionally larger number of bars will be available. If the existing data is not enough
for the predefined depth of the preliminary history, this fact will be recorded in the test log.

---

## Page 1461

Part 6. Trading automation
1 461 
6.5 Testing and optimization of Expert Advisors
You cannot configure (change) this behavior. Therefore, if you need to provide a specified number of
historical bars of the current timeframe from the very beginning, you should set an earlier start date for
the test and then "wait" in the Expert Advisor code for the required trading start date or a sufficient
number of bars. Before that, you should skip all events. 
The tester also emulates its own Market Watch, from which the program can obtain information on
instruments. By default, at the beginning of testing, the tester Market Watch contains only one symbol:
the symbol on which testing is started. All additional symbols are added to the tester Market Watch
automatically when accessing them through the API functions. At the first access to a "third-party"
symbol from an MQL program, the testing agent will synchronize the symbol data with the terminal.
The data of additional symbols can be accessed in the following cases:
• Using technical indicators, iCustom, or IndicatorCreate for the symbol/timeframe pair
• Querying another symbol's Market Watch:
• SeriesInfoInteger
• Bars
• SymbolSelect
• SymbolIsSynchronized
• SymbolInfoDouble
• SymbolInfoInteger
• SymbolInfoString
• SymbolInfoTick
• SymbolInfoSessionQuote
• SymbolInfoSessionTrade
• MarketBookAdd
• MarketBookGet
• Querying the symbol/timeframe pair timeseries using the following functions:
• CopyBuffer
• CopyRates
• CopyTime
• CopyOpen
• CopyHigh
• CopyLow
• CopyClose
• CopyTickVolume
• CopyRealVolume
• CopySpread
In addition, you can explicitly request the history for the desired symbols by calling the SymbolSelect
function in the OnInit handler. The history will be loaded in advance before the Expert Advisor testing
starts.

---

## Page 1462

Part 6. Trading automation
1 462
6.5 Testing and optimization of Expert Advisors
At the moment when another symbol is accessed for the first time, the testing process stops and the
symbol/period pair history is downloaded from the terminal into the testing agent. The tick sequence
generation is also enabled at this moment.
Each instrument generates its own tick sequence according to the set tick generation mode.
Synchronization of bars of different symbols is of particular importance when implementing
multicurrency Expert Advisors since the correctness of calculations depends on this. A state is
considered synchronized when the last bars of all used symbols have the same opening time.
The tester generates and plays its tick sequence for each instrument. At the same time, a new bar on
each instrument is opened regardless of how bars are opened on other instruments. This means that
when testing a multicurrency Expert Advisor, a situation is possible (and most often it happens) when a
new bar has already opened on one instrument, but not yet on another.
For example, if we are testing an Expert Advisor using EURUSD symbol data and a new hourly
candlestick has opened for this symbol, we will receive the OnTick event. But at the same time, there is
no guarantee that a new candlestick has opened on GBPUSD, which we might also be using.
Thus, the synchronization algorithm implies that you need to check the quotes of all instruments and
wait for the equality of the opening times of the last bars.
This does not raise any questions for as long as the real tick, emulation of all ticks, or OHLC M1  testing
modes are used. With these modes, a sufficient number of ticks are generated within one candlestick to
wait for the moment of synchronization of bars from different symbols. Just complete the OnTick
function and check the appearance of a new bar on GBPUSD on the next tick. But when testing in the
"Open prices only" mode, there will be no other tick, since the Expert Advisor is called only once per
bar, and it may seem that this mode is not suitable for testing multicurrency Expert Advisors. In fact,
the tester allows you to detect the moment when a new bar opens on another symbol using the Sleep
function (in a loop) or a timer.
First, let's consider an example of an Expert Advisor SyncBarsBySleep.mq5, which demonstrates the
synchronization of bars through Sleep.
A pair of input parameters allows you to set the Pause size in seconds to wait for other symbol's bars,
as well as the name of that other symbol (OtherSymbol), which must be different from the chart
symbol.
input uint Pause = 1;                   // Pause (seconds)
input string OtherSymbol = "USDJPY";
To identify patterns in the delay of bar opening times, we describe a simple class BarTimeStatistics. It
contains a field for counting the total number of bars (total) and the number of bars on which there was
no synchronization initially (late), that is, the other symbol was late.

---

## Page 1463

Part 6. Trading automation
1 463
6.5 Testing and optimization of Expert Advisors
class BarTimeStatistics
{
public:
   int total;
   int late;
   
   BarTimeStatistics(): total(0), late(0) { }
   
   ~BarTimeStatistics()
   {
      PrintFormat("%d bars on %s was late among %d total bars on %s (%2.1f%%)",
         late, OtherSymbol, total, _Symbol, late * 100.0 / total);
   }
};
The object of this class prints the received statistics in its destructor. Since we are going to make this
object static, the report will be printed at the very end of the test.
If the tick generation mode selected in the tester differs from the opening prices, we will detect this
using the previously considered getTickModel function and will return a warning.
void OnTick()
{
   const TICK_MODEL model = getTickModel();
   if(model != TICK_MODEL_OPEN_PRICES)
   {
      static bool shownOnce = false;
      if(!shownOnce)
      {
         Print("This Expert Advisor is intended to run in \"Open Prices\" mode");
         shownOnce = true;
      }
   }
Next, OnTick provides the working synchronization algorithm.

---

## Page 1464

Part 6. Trading automation
1 464
6.5 Testing and optimization of Expert Advisors
   // time of the last known bar for _Symbol
   static datetime lastBarTime = 0;
   // attribute of synchronization
   static bool synchronized = false;
   // bar counters
   static BarTimeStatistics stats;
   
   const datetime currentTime = iTime(_Symbol, _Period, 0);
   
   // if it is executed for the first time or the bar has changed, save the bar
   if(lastBarTime != currentTime)
   {
      stats.total++;
      lastBarTime = currentTime;
      PrintFormat("Last bar on %s is %s", _Symbol, TimeToString(lastBarTime));
      synchronized = false;
   }
   
   // time of the last known bar for another symbol
   datetime otherTime;
   bool late = false;
   
   // wait until the times of two bars become the same
   while(currentTime != (otherTime = iTime(OtherSymbol, _Period, 0)))
   {
      late = true;
      PrintFormat("Wait %d seconds...", Pause);
      Sleep(Pause * 1000);
   }
   if(late) stats.late++;
   
   // here we are after synchronization, save the new status
   if(!synchronized)
   {
      // use TimeTradeServer() because TimeCurrent() does not change in the absence of ticks
      Print("Bars are in sync at ", TimeToString(TimeTradeServer(),
         TIME_DATE | TIME_SECONDS));
      // no longer print a message until the next out of sync
      synchronized = true;
   }
   // here is your synchronous algorithm
   // ...
}
Let's set up the tester to run the Expert Advisor on EURUSD, H1 , which is the most liquid instrument.
Let's use the default Expert Advisor parameters, that is, USDJPY will be the "other" symbol.
As a result of the test, the log will contain the following entries (we intentionally show the logs related
to the downloading of the USDJPY history, which occurred during the first iTime call).

---

## Page 1465

Part 6. Trading automation
1 465
6.5 Testing and optimization of Expert Advisors
2022.04.15 00:00:00   Last bar on EURUSD is 2022.04.15 00:00
USDJPY: load 27 bytes of history data to synchronize in 0:00:00.001
USDJPY: history synchronized from 2020.01.02 to 2022.04.20
USDJPY,H1: history cache allocated for 8109 bars and contains 8006 bars from 2021.01.04 00:00 to 2022.04.14 23:00
USDJPY,H1: 1 bar from 2022.04.15 00:00 added
USDJPY,H1: history begins from 2021.01.04 00:00
2022.04.15 00:00:00   Bars are in sync at 2022.04.15 00:00:00
2022.04.15 01:00:00   Last bar on EURUSD is 2022.04.15 01:00
2022.04.15 01:00:00   Wait 1 seconds...
2022.04.15 01:00:01   Bars are in sync at 2022.04.15 01:00:01
2022.04.15 02:00:00   Last bar on EURUSD is 2022.04.15 02:00
2022.04.15 02:00:00   Wait 1 seconds...
2022.04.15 02:00:01   Bars are in sync at 2022.04.15 02:00:01
...
2022.04.20 23:59:59   95 bars on USDJPY was late among 96 total bars on EURUSD (99.0%)
You can see that the USDJPY bars are delayed regularly. If you select USDJPY, H1  in the tester
settings and EURUSD in the Expert Advisor parameters, you will get the opposite picture.
2022.04.15 00:00:00   Last bar on USDJPY is 2022.04.15 00:00
EURUSD: load 27 bytes of history data to synchronize in 0:00:00.002
EURUSD: history synchronized from 2018.01.02 to 2022.04.20
EURUSD,H1: history cache allocated for 8109 bars and contains 8006 bars from 2021.01.04 00:00 to 2022.04.14 23:00
EURUSD,H1: 1 bar from 2022.04.15 00:00 added
EURUSD,H1: history begins from 2021.01.04 00:00
2022.04.15 00:00:00   Bars are in sync at 2022.04.15 00:00:00
2022.04.15 01:00:00   Last bar on USDJPY is 2022.04.15 01:00
2022.04.15 01:00:00   Wait 1 seconds...
2022.04.15 01:00:01   Bars are in sync at 2022.04.15 01:00:01
2022.04.15 02:00:00   Last bar on USDJPY is 2022.04.15 02:00
2022.04.15 02:00:00   Wait 1 seconds...
2022.04.15 02:00:01   Bars are in sync at 2022.04.15 02:00:01
...
2022.04.20 23:59:59   23 bars on EURUSD was late among 96 total bars on USDJPY (24.0%)
Here, in most cases, there was no need to wait: the EURUSD bars already existed at the time the
USDJPY bar was formed.
There is another way to synchronize bars: using a timer. An example of such an Expert Advisor,
SyncBarsByTimer.mq5, is included in the book. Please note that the timer events, as a rule, occur inside
the bar (because the probability of hitting exactly the beginning is very low). Because of this, the bars
are almost always synchronized.
We could also remind you about the possibility of synchronizing bars using the spy indicator
EventTickSpy.mq5, but it's based on custom events that only work when testing visually. In addition, for
such indicators that require a response to each tick, it is important to use the #property
tester_ everytick_ calculate directive. We have already talked about it in the Testing indicators section,
and we will remind you about it once again in the section on specific tester directives.
6.5.5 Optimization criteria
An optimization criterion is a certain metric that defines the quality of the tested set of input
parameters. The greater the value of the optimization criterion, the better the test result with a given

---

## Page 1466

Part 6. Trading automation
1 466
6.5 Testing and optimization of Expert Advisors
set of parameters is estimated. The parameter is selected on the "Settings" tab to the right of the
"Optimization" field.
The criterion is important not only for the user to be able to compare the results. Without an
optimization criterion, it is impossible to use a genetic algorithm, since on the basis of the criterion it
"decides" how to select candidates for new generations. The criterion is not used during full
optimization with a complete iteration of all possible variants.
The following built-in optimization criteria are available in the tester:
·Maximum balance
·Maximum profitability
·Maximum expected win (average profit/loss per trade)
·Minimum drawdown as a percentage of equity
·Maximum recovery factor
·Maximum Sharpe ratio
·Custom optimization criterion
When choosing the latter option, the value of the OnTester function implemented in the Expert Advisor
will be taken into account as an optimization criterion – we will consider it later. This parameter allows
the programmer to use any custom index for optimization.
A special "complex criterion" is also available in MetaTrader 5. This is an integral metric of the quality
of the testing pass, which takes into account several parameters at once:
·Number of deals
·Drawdown
·Recovery factor
·Mathematical expectation of winning
·Sharpe ratio
The formula is not disclosed by the developers, but it is known that possible values range from 0 to
1 00. It is important that the values of the complex parameter affect the color of the cells of the Result
column in the optimization table regardless of the criterion, i.e., highlighting following this scheme
works even when another criterion is chosen for display in the Result column. Weak combinations with
values below 20 are highlighted in red, strong combinations above 80 are highlighted in dark green.
The search for a universal criterion of the trading system quality factor is an urgent and difficult task
for most traders, since the choice of settings based on the maximum value of one criterion (for
example, profit) is, as a rule, far from the best option in terms of stable and predictable behavior of the
Expert Advisor in the foreseeable future.
The presence of a complex indicator allows you to level the weaknesses of each individual metric (and
they are necessarily available and widely known) and provides a guideline when developing your own
custom variables for calculation in OnTester. We will deal with this soon.
6.5.6 Getting testing financial statistics: TesterStatistics
We usually evaluate the quality of an Expert Advisor based on a trading report, which is similar to a
testing report when dealing with a tester. It contains a large number of variables that characterize the
trading style, stability and, of course, profitability. All these metrics, with some exceptions, are

---

## Page 1467

Part 6. Trading automation
1 467
6.5 Testing and optimization of Expert Advisors
available to the MQL program through a special function TesterStatistics. Thus, the Expert Advisor
developer has the ability to analyze individual variables in the code and construct their own combined
optimization quality criteria from them.
double TesterStatistics(ENUM_STATISTICS statistic)
The TesterStatistics function returns the value of the specified statistical variable, calculated based on
the results of a separate run of the Expert Advisor in the tester. A function can be called in the OnDeinit
or OnTester handler, which is yet to be discussed.
All available statistical variables are summarized in the ENUM_STATISTICS enumeration. Some of them
serve as qualitative characteristics, that is, real numbers (usually total profits, drawdowns, ratios, and
so on), and the other part is quantitative, that is, integers (for example, the number of transactions).
However, both groups are controlled by the same function with the double result.
The following table shows real indicators (monetary amounts and coefficients). All monetary amounts
are expressed in the deposit currency.
Identifier
Description
S TAT_IN ITIAL _D E P O S IT
Initial deposit
S TAT_W ITH D R AW AL 
The amount of funds withdrawn from the account
S TAT_P R O F IT
Net profit or loss at the end of testing, the sum of
STAT_GROSS_PROFIT and STAT_GROSS_LOSS
S TAT_G R O S S _P R O F IT
Total profit, the sum of all profitable trades (greater than or equal
to zero)
S TAT_G R O S S _L O S S 
Total loss, the sum of all losing trades (less than or equal to zero)
S TAT_M AX_P R O F ITTR AD E 
Maximum profit: the largest value among all profitable trades
(greater than or equal to zero)
S TAT_M AX_L O S S TR AD E 
Maximum loss: the smallest value among all losing trades (less than
or equal to zero)
S TAT_CO N P R O F ITM AX
Total maximum profit in a series of profitable trades (greater than
or equal to zero)
S TAT_M AX_CO N W IN S 
Total profit in the longest series of profitable trades
S TAT_CO N L O S S M AX
Total maximum loss in a series of losing trades (less than or equal
to zero)
S TAT_M AX_CO N L O S S E S 
Total loss in the longest series of losing trades
S TAT_B AL AN CE M IN 
Minimum balance value
S TAT_B AL AN CE _D D 
Maximum balance drawdown in money
S TAT_B AL AN CE D D _P E R CE N T
Balance drawdown in percent, which was recorded at the time of
the maximum balance drawdown in money (STAT_BALANCE_DD)
S TAT_B AL AN CE _D D R E L _P E R CE N T
Maximum balance drawdown in percent

---

## Page 1468

Part 6. Trading automation
1 468
6.5 Testing and optimization of Expert Advisors
Identifier
Description
S TAT_B AL AN CE _D D _R E L ATIVE 
Balance drawdown in money equivalent, which was recorded at the
moment of the maximum balance drawdown in percent
(STAT_BALANCE_DDREL_PERCENT)
S TAT_E Q U ITYM IN 
Minimum equity value
S TAT_E Q U ITY_D D 
Maximum drawdown in money
S TAT_E Q U ITYD D _P E R CE N T
Drawdown in percent, which was recorded at the time of the
maximum drawdown of funds in the money (STAT_EQUITY_DD)
S TAT_E Q U ITY_D D R E L _P E R CE N T
Maximum drawdown in percent
S TAT_E Q U ITY_D D _R E L ATIVE 
Drawdown in money that was recorded at the time of the maximum
drawdown in percent (STAT_EQUITY_DDREL_PERCENT)
S TAT_E XP E CTE D _P AYO F F 
Mathematical expectation of winnings (arithmetic mean of the total
profit and the number of transactions)
S TAT_P R O F IT_F ACTO R 
Profitability, which is the ratio
STAT_GROSS_PROFIT/STAT_GROSS_LOSS (if STAT_GROSS_LOSS
= 0; profitability takes the value DBL_MAX)
S TAT_R E CO VE R Y_F ACTO R 
Recovery factor: the ratio of STAT_PROFIT/STAT_BALANCE_DD
S TAT_S H AR P E _R ATIO 
Sharpe ratio
S TAT_M IN _M AR G IN L E VE L 
Minimum margin level reached
S TAT_CU S TO M _O N TE S TE R 
The value of the custom optimization criterion returned by the
OnTester function
The following table shows integer indicators (amounts).
Identifier
Description
S TAT_D E AL S 
Total number of completed transactions
S TAT_TR AD E S 
Number of trades (deals to exit the market)
S TAT_P R O F IT_TR AD E S 
Profitable trades
S TAT_L O S S _TR AD E S 
Losing trades
S TAT_S H O R T_TR AD E S 
Short trades
S TAT_L O N G _TR AD E S 
Long trades
S TAT_P R O F IT_S H O R TTR AD E S 
Short profitable trades
S TAT_P R O F IT_L O N G TR AD E S 
Long profitable trades
S TAT_P R O F ITTR AD E S _AVG CO N 
Average length of a profitable series of trades

---

## Page 1469

Part 6. Trading automation
1 469
6.5 Testing and optimization of Expert Advisors
Identifier
Description
S TAT_L O S S TR AD E S _AVG CO N 
Average length of a losing series of trades
S TAT_CO N P R O F ITM AX_TR AD E S 
Number of trades that formed STAT_CONPROFITMAX (maximum
profit in the sequence of profitable trades)
S TAT_M AX_CO N P R O F IT_TR AD E S 
Number of trades in the longest series of profitable trades
STAT_MAX_CONWINS
S TAT_CO N L O S S M AX_TR AD E S 
Number of trades that formed STAT_CONLOSSMAX (maximum loss
in the sequence of losing trades)
S TAT_M AX_CO N L O S S _TR AD E S 
Number of trades in the longest series of losing trades
STAT_MAX_CONLOSSES
Let's try to use the presented metrics to create our own complex Expert Advisor quality criterion. To
do this, we need some kind of "experimental" example of an MQL program. Let's take the Expert
Advisor MultiMartingale.mq5 as a starting point, but we will simplify it: we will remove multicurrency,
built-in error handling, and scheduling. Moreover, we will choose a signal trading strategy for it with a
single calculation on the bar, i.e., at the opening prices. This will speed up optimization and expand the
field for experiments.
The strategy will be based on the overbought and oversold conditions determined by the OsMA
indicator. The Bollinger Bands indicator superimposed on OsMA will help you dynamically find the
boundaries of excess volatility, which means trading signals.
When OsMA returns inside the corridor, crossing the lower border from the bottom up, we will open a
buy trade. When OsMA crosses the upper boundary in the same way from top to bottom, we will sell. To
exit positions, we use the moving average, also applied to OsMA. If OsMA shows a reverse movement
(down for a long position or up for a short position) and touches the MA, the position will be closed. This
strategy is illustrated in the following screenshot.

---

## Page 1470

Part 6. Trading automation
1 470
6.5 Testing and optimization of Expert Advisors
Trading strategy based on OsMA, BBands and MA indicators
The blue vertical line corresponds to the bar where the buy is opened, since on the two previous bars
the lower Bollinger band was crossed by the OsMA histogram from the bottom up (this place is marked
with a hollow blue arrow in the subwindow). The red vertical line is the location of the reverse signal, so
the buy was closed and the sell was opened. In the subwindow, in this place (or rather, on the two
previous bars, where the hollow red arrow is located), the OsMA histogram crosses the upper Bollinger
band from top to bottom. Finally, the green line indicates the closing of the sale, due to the fact that
the histogram began to rise above the red MA.
Let's name the Expert Advisor BandOsMA.mq5. The general settings will include a magic number, a fixed
lot, and a stop loss distance in points. For the stop loss, we will use TrailingStop from the previous
example. Take profit is not used here.
input group "C O M M O N   S E T T I N G S"
sinput ulong Magic = 1234567890;
input double Lots = 0.01;
input int StopLoss = 1000;
Three groups of settings are intended for indicators.

---

## Page 1471

Part 6. Trading automation
1 471 
6.5 Testing and optimization of Expert Advisors
input group "O S M A   S E T T I N G S"
input int FastOsMA = 12;
input int SlowOsMA = 26;
input int SignalOsMA = 9;
input ENUM_APPLIED_PRICE PriceOsMA = PRICE_TYPICAL;
   
input group "B B A N D S   S E T T I N G S"
input int BandsMA = 26;
input int BandsShift = 0;
input double BandsDeviation = 2.0;
   
input group "M A   S E T T I N G S"
input int PeriodMA = 10;
input int ShiftMA = 0;
input ENUM_MA_METHOD MethodMA = MODE_SMA;
In the MultiMartingale.mq5 Expert Advisor, we had no trading signals, while the opening direction was
set by the user. Here we have trading signals, and it makes sense to arrange them as a separate class.
First, let's describe the abstract interface TradingSignal.
interface TradingSignal
{
   virtual int signal(void);
};
It is as simple as our other interface TradingStrategy. And this is good. The simpler the interfaces and
objects, the more likely they are to do one single thing, which is a good programming style because it
minimizes bugs and makes large software projects more understandable. Due to abstraction in any
program that uses TradingSignal, it will be possible to replace one signal with another. We can also
replace the strategy. Our strategies are now responsible for preparing and sending orders, and signals
initiate them based on market analysis.
In our case, let's pack the specific implementation of TradingSignal into the BandOsMaSignal class. Of
course, we need variables to store the descriptors of the 3 indicators. Indicator instances are created
and deleted in the constructor and destructor, respectively. All parameters will be passed from input
variables. Note that iBands and iMA are built based on the hOsMA handler.

---

## Page 1472

Part 6. Trading automation
1 472
6.5 Testing and optimization of Expert Advisors
class BandOsMaSignal: public TradingSignal
{
   int hOsMA, hBands, hMA;
   int direction;
public:
   BandOsMaSignal(const int fast, const int slow, const int signal,
      const ENUM_APPLIED_PRICE price,
      const int bands, const int shift, const double deviation,
      const int period, const int x, ENUM_MA_METHOD method)
   {
      hOsMA = iOsMA(_Symbol, _Period, fast, slow, signal, price);
      hBands = iBands(_Symbol, _Period, bands, shift, deviation, hOsMA);
      hMA = iMA(_Symbol, _Period, period, x, method, hOsMA);
      direction = 0;
   }
   
   ~BandOsMaSignal()
   {
      IndicatorRelease(hMA);
      IndicatorRelease(hBands);
      IndicatorRelease(hOsMA);
   }
   ...
The direction of the current trading signal is placed in the variable direction: 0 – no signals (undefined
situation), +1  – buy, -1  – sell. We will fill in this variable in the signal method. Its code repeats the
above verbal description of signals in MQL5.

---

## Page 1473

Part 6. Trading automation
1 473
6.5 Testing and optimization of Expert Advisors
   virtual int signal(void) override
   {
      double osma[2], upper[2], lower[2], ma[2];
      // get two values of each indicator on bars 1 and 2
      if(CopyBuffer(hOsMA, 0, 1, 2, osma) != 2) return 0;
      if(CopyBuffer(hBands, UPPER_BAND, 1, 2, upper) != 2) return 0;
      if(CopyBuffer(hBands, LOWER_BAND, 1, 2, lower) != 2) return 0;
      if(CopyBuffer(hMA, 0, 1, 2, ma) != 2) return 0;
      
      // if there was a signal already, check if it has ended
      if(direction != 0)
      {
         if(direction > 0)
         {
            if(osma[0] >= ma[0] && osma[1] < ma[1])
            {
               direction = 0;
            }
         }
         else
         {
            if(osma[0] <= ma[0] && osma[1] > ma[1])
            {
               direction = 0;
            }
         }
      }
      
      // in any case, check if there is a new signal
      if(osma[0] <= lower[0] && osma[1] > lower[1])
      {
         direction = +1;
      }
      else if(osma[0] >= upper[0] && osma[1] < upper[1])
      {
         direction = -1;
      }
      
      return direction;
   }
};
As you can see, the indicator values are read for bars 1  and 2, since we will work on opening a bar, and
the 0th bar has just opened by the time we we call the signal method.
The new class that implements the TradingStrategy interface will be called SimpleStrategy.
The class provides some new features while also using some previously existing parts. In particular, it
retained autopointers for PositionState and TrailingStop and has a new autopointer to the TradingSignal
signal. Also, since we are going to trade only on the opening of bars, we needed the lastBar variable,
which will store the time of the last processed bar.

---

## Page 1474

Part 6. Trading automation
1 474
6.5 Testing and optimization of Expert Advisors
class SimpleStrategy: public TradingStrategy
{
protected:
   AutoPtr<PositionState> position;
   AutoPtr<TrailingStop> trailing;
   AutoPtr<TradingSignal> command;
   
   const int stopLoss;
   const ulong magic;
   const double lots;
   
   datetime lastBar;
   ...
The global parameters are passed to the SimpleStrategy constructor. We also pass a pointer to the
TradingSignal object: in this case, it will be BandOsMaSignal which will have to be created by the calling
code. Next, the constructor tries to find among the existing positions those that have the required
magic number and symbol, and if successful, adds a trailing stop. This will be useful if the Expert Advisor
has a break for one reason or another, and the position has already been opened.
public:
   SimpleStrategy(TradingSignal *signal, const ulong m, const int sl, const double v):
      command(signal), magic(m), stopLoss(sl), lots(v), lastBar(0)
   {
 // select "our" position among the existing ones (if there is a suitable one)
      PositionFilter positions;
      ulong tickets[];
      positions.let(POSITION_MAGIC, magic).let(POSITION_SYMBOL, _Symbol).select(tickets);
      const int n = ArraySize(tickets);
      if(n > 1)
      {
         Alert(StringFormat("Too many positions: %d", n));
 // TODO: close extra positions - this is not allowed by the strategy
      }
      else if(n > 0)
      {
         position = new PositionState(tickets[0]);
         if(stopLoss)
         {
           trailing = new TrailingStop(tickets[0], stopLoss, stopLoss / 50);
         }
      }
   }
The implementation of the trade method is similar to the martingale example. However, we have
removed lot multiplications and added the signal method call.

---

## Page 1475

Part 6. Trading automation
1 475
6.5 Testing and optimization of Expert Advisors
   virtual bool trade() override
   {
      // we work only once when a new bar appears
      if(lastBar == iTime(_Symbol, _Period, 0)) return false;
      
      int s = command[].signal(); // getting a signal
      
      ulong ticket = 0;
      
      if(position[] != NULL)
      {
         if(position[].refresh()) // position exists
         {
            // the signal has changed to the opposite or disappeared
            if((position[].get(POSITION_TYPE) == POSITION_TYPE_BUY && s != +1)
            || (position[].get(POSITION_TYPE) == POSITION_TYPE_SELL && s != -1))
            {
               PrintFormat("Signal lost: %d for position %d %lld",
                  s, position[].get(POSITION_TYPE), position[].get(POSITION_TICKET));
               if(close(position[].get(POSITION_TICKET)))
               {
                  position = NULL;
               }
               else
               {
                 // update internal flag 'ready'
                 // according to whether or not there was a closure
                  position[].refresh();
               }
            }
            else
            {
               position[].update();
               if(trailing[]) trailing[].trail();
            }
         }
         else // position is closed
         {
            position = NULL;
         }
      }
      
      if(position[] == NULL && s != 0)
      {
         ticket = (s == +1) ? openBuy() : openSell();
      }
      
      if(ticket > 0) // new position just opened
      {
         position = new PositionState(ticket);
         if(stopLoss)

---

## Page 1476

Part 6. Trading automation
1 476
6.5 Testing and optimization of Expert Advisors
         {
            trailing = new TrailingStop(ticket, stopLoss, stopLoss / 50);
         }
      }
      // store the current bar
      lastBar = iTime(_Symbol, _Period, 0);
      
      return true;
   }
Auxiliary methods openBuy, openSell, and others have undergone minimal changes, so we will not list
them (the full source code is attached).
Since we always have only one strategy in this Expert Advisor, in contrast to the multi-currency
martingale in which each symbol required its own settings, let's exclude the strategy pool and manage
the strategy object directly.
AutoPtr<TradingStrategy> strategy;
   
int OnInit()
{
   if(FastOsMA >= SlowOsMA) return INIT_PARAMETERS_INCORRECT;
   strategy = new SimpleStrategy(
      new BandOsMaSignal(FastOsMA, SlowOsMA, SignalOsMA, PriceOsMA,
         BandsMA, BandsShift, BandsDeviation,
         PeriodMA, ShiftMA, MethodMA),
         Magic, StopLoss, Lots);
   return INIT_SUCCEEDED;
}
   
void OnTick()
{
   if(strategy[] != NULL)
   {
      strategy[].trade();
   }
}
We now have a ready Expert Advisor which we can use as a tool for studying the tester. First, let's
create an auxiliary structure TesterRecord for querying and storing all statistical data.

---

## Page 1477

Part 6. Trading automation
1 477
6.5 Testing and optimization of Expert Advisors
struct TesterRecord
{
   string feature;
   double value;
   
   static void fill(TesterRecord &stats[])
   {
      ResetLastError();
      for(int i = 0; ; ++i)
      {
         const double v = TesterStatistics((ENUM_STATISTICS)i);
         if(_LastError) return;
         TesterRecord t = {EnumToString((ENUM_STATISTICS)i), v};
         PUSH(stats, t);
      }
   }
};
In this case, the feature string field is needed only for informative log output. To save all indicators (for
example, to be able to generate your own report form later), a simple array of type double of
appropriate length is enough.
Using the structure in the OnDeinit handler, we make sure that the MQL5 API returns the same values
as the tester's report.
void OnDeinit(const int)
{
   TesterRecord stats[];
   TesterRecord::fill(stats);
   ArrayPrint(stats, 2);
}
For example, when running the Expert Advisor on EURUSD, H1  with a deposit of 1 0000 and without any
optimizations (with default settings), we will get approximately the following values for 2021 
(fragment):

---

## Page 1478

Part 6. Trading automation
1 478
6.5 Testing and optimization of Expert Advisors
                        [feature]  [value]
[ 0] "STAT_INITIAL_DEPOSIT"       10000.00
[ 1] "STAT_WITHDRAWAL"                0.00
[ 2] "STAT_PROFIT"                    6.01
[ 3] "STAT_GROSS_PROFIT"            303.63
[ 4] "STAT_GROSS_LOSS"             -297.62
[ 5] "STAT_MAX_PROFITTRADE"          15.15
[ 6] "STAT_MAX_LOSSTRADE"           -10.00
...
[27] "STAT_DEALS"                   476.00
[28] "STAT_TRADES"                  238.00
...
[37] "STAT_CONLOSSMAX_TRADES"         8.00
[38] "STAT_MAX_CONLOSS_TRADES"        8.00
[39] "STAT_PROFITTRADES_AVGCON"       2.00
[40] "STAT_LOSSTRADES_AVGCON"         2.00
Knowing all these values, we can invent our own formula for the combined metric of the Expert Advisor
quality and, at the same time, the objective optimization function. But the value of this indicator in any
case will need to be reported to the tester. And that's what the OnTester function does.
6.5.7 OnTester event
The OnTester event is generated upon the completion of Expert Advisor testing on historical data (both
a separate tester run initiated by the user and one of the multiple runs automatically launched by the
tester during optimization). To handle the OnTester event, an MQL program must have a corresponding
function in its source code, but this is not necessary. Even without the OnTester function, Expert
Advisors can be successfully optimized based on standard criteria.
The function can only be used in Expert Advisors.
double OnTester()
The function is designed to calculate some value of type double, used as a custom optimization
criterion (Custom max). Criterion selection is important primarily for successful genetic optimization,
while it also allows the user to evaluate and compare the effects of different settings.
In genetic optimization, the results are sorted within one generation in the criterion descending order.
That is, the results with the highest value are considered the best in terms of the optimization criterion.
The worst values in this sorting are subsequently discarded and do not take part in the formation of the
next generation.
Please note that the values returned by the OnTester function are taken into account only when a
custom criterion is selected in the tester settings. The availability of the OnTester function does not
automatically mean its use by the genetic algorithm. 
The MQL5 API does not provide the means to programmatically find out which optimization
criterion the user has selected in the tester settings. Sometimes it is very important to know this in
order to implement your own analytical algorithms to post-process optimization results.
The function is called by the kernel only in the tester, just before the call of the OnDeinit function.
To calculate the return value, we can use both the standard statistics available through the
TesterStatistics function and their arbitrary calculations.

---

## Page 1479

Part 6. Trading automation
1 479
6.5 Testing and optimization of Expert Advisors
In the BandOsMA.mq5 Expert Advisor, we create the OnTester handler which takes into account several
metrics: profit, profitability, the number of trades, and the Sharpe ratio. Next, we multiply all the
metrics after taking the square root of each. Of course, each developer may have their own
preferences and ideas for constructing such generalized quality criteria.
double sign(const double x)
{
   return x > 0 ? +1 : (x < 0 ? -1 : 0);
}
   
double OnTester()
{
   const double profit = TesterStatistics(STAT_PROFIT);
   return sign(profit) * sqrt(fabs(profit))
      * sqrt(TesterStatistics(STAT_PROFIT_FACTOR))
      * sqrt(TesterStatistics(STAT_TRADES))
      * sqrt(fabs(TesterStatistics(STAT_SHARPE_RATIO)));
}
The unit test log displays a line with the value of the OnTester function.
Let's launch the genetic optimization of the Expert Advisor for 2021  on EURUSD, H1  with the selection
of indicator parameters and stop loss size (the file MQL5/Presets/MQL5Book/BandOsMA.set is provided
with the book). To check the quality of optimization, we will also include forward tests from the
beginning of 2022 (5 months).
First, let's optimize according to our criterion.
As you know, MetaTrader 5 saves all standard criteria in the optimization results in addition to the
current one used during optimization. This allows, upon completion of the optimization, to analyze the
results from different points by selecting certain criteria from the drop-down list in the upper right
corner of the panel with the table. Thus, although we did optimization according to our own criterion,
the most interesting built-in complex criterion is also available to us.
We can export the optimization table to an XML file, first with our criteria selected, and then with a
complex criterion giving the file a new name (unfortunately, only one criterion is written to the export
file; it is important not to change the sorting between two exports). This makes it possible to combine
two tables in an external program and build a diagram on which two criteria are plotted along the axes;
each point there indicates a combination of criteria in one run.

---

## Page 1480

Part 6. Trading automation
1 480
6.5 Testing and optimization of Expert Advisors
Comparison of custom and complex optimization criteria
In a complex criterion, we observe a multi-level structure, since it is calculated according to a formula
with conditions: somewhere one branch works, and somewhere else another one operates. Our custom
criteria are always calculated using the same formula. We also note the presence of negative values in
our criterion (this is expected) and the declared range of 0-1 00 for the complex criterion.
Let's check how good our criterion is by analyzing its values for the forward period.

---

## Page 1481

Part 6. Trading automation
1 481 
6.5 Testing and optimization of Expert Advisors
Values of the custom criterion on periods of optimization and forward tests
As expected, only a part of the good optimization indicators remained on the forward. But we are more
interested not in the criterion, but in profit. Let's look at its distribution in the optimization-forward link.
Profit on periods of optimization and forward tests

---

## Page 1482

Part 6. Trading automation
1 482
6.5 Testing and optimization of Expert Advisors
The picture here is similar. Of the 6850 passes with a profit in the optimization period, 31 23 turned out
to be profitable in the forward as well (45%). And out of the first 1 000 best, only 323 were profitable,
which is not good enough. Therefore, this Expert Advisor will need a lot of work to identify stable
profitable settings. But maybe it's the optimization criteria problem?
Let's repeat the optimization, this time using the built-in complex criterion.
Attention! MetaTrader 5 generates optimization caches during optimizations: opt files at
Tester/cache. When starting the next optimization, it looks for suitable caches to continue the
optimization. If there is a cache file with the previous settings, the process does not start from the
very beginning, but it takes into account previous results. This allows you to build genetic
optimizations in chains, assuming that you find the best results (after all, each genetic optimization
is a random process). 
MetaTrader 5 does not take into account the optimization criterion as a distinguishing factor in the
settings. This may be useful in some cases, based on the foregoing, but it will interfere with our
current task. To conduct a pure experiment, we need optimization from scratch. Therefore,
immediately after the first optimization using our criterion, we cannot launch the second one using
the complex criterion. 
There is no way to disable the current behavior from the terminal interface. Therefore, you should
either delete or rename (change the extension) the previous opt-file manually in any file manager. A
little later we will get acquainted with the preprocessor directive for the tester tester_ no_ cache,
which can be specified in the source code of a particular Expert Advisor, allowing you to disable the
cache reading.
Comparison of the values of the complex criterion on the periods of optimization and the forward period
takes the following form.
Complex criterion for periods of optimization and forward tests

---

## Page 1483

Part 6. Trading automation
1 483
6.5 Testing and optimization of Expert Advisors
Here's the stability of profits on forwards.
Profit on periods of optimization and forward tests
Of the 5952 positive results in history, only 2655 (also about 45%) remained in the black. But out of
the first 1 000, 581  turned out to be successful on the forward.
So, we have seen that it is quite simple to use OnTester from the technical point of view, but our
criterion works worse than the built-in one (ceteris paribus), although it is far from ideal. Thus, from the
point of view of the search for the formula of the criterion itself, and the subsequent reasonable choice
of parameters without looking into the future, there are more questions about the content of OnTester,
than there are answers.
Here, programming smoothly flows into research and scientific activity, and is beyond the scope of this
book. But we will give one example of a criterion calculated on our own metric, and not on ready-made
metrics: TesterStatistics. We will talk about the criterion R2, also known as the coefficient of
determination (RSquared.mqh).
Let's create a function to calculate R2 from the balance curve. It is known that when trading with a
permanent lot, an ideal trading system should show the balance in the form of a straight line. We are
now using a permanent lot, and therefore it will suit us. As for R2 in the case of variable lots, we will
deal with it a little later.
In the end, R2 is an inverse measure of the variance of the data relative to the linear regression built on
them. The range of R2 values lies from minus infinity to +1  (although large negative values are very
unlikely in our case). It is obvious that the found line is simultaneously characterized by a slope,
therefore, in order to universalize the code, we will save both R2 and the tangent of the angle in the
R2A structure as an intermediate result.

---

## Page 1484

Part 6. Trading automation
1 484
6.5 Testing and optimization of Expert Advisors
struct R2A
{
   double r2;    // square of correlation coefficient
   double angle; // tangent of the slope
   R2A(): r2(0), angle(0) { }
};
Calculation of indicators is performed in the RSquared function which takes an array of data as input
and returns an R2A structure.
R2A RSquared(const double &data[])
{
   int size = ArraySize(data);
   if(size <= 2) return R2A();
   double x, y, div;
   int k = 0;
   double Sx = 0, Sy = 0, Sxy = 0, Sx2 = 0, Sy2 = 0;
   for(int i = 0; i < size; ++i)
   {
      if(data[i] == EMPTY_VALUE
      || !MathIsValidNumber(data[i])) continue;
      x = i + 1;
      y = data[i];
      Sx  += x;
      Sy  += y;
      Sxy += x * y;
      Sx2 += x * x;
      Sy2 += y * y;
      ++k;
   }
   size = k;
   const double Sx22 = Sx * Sx / size;
   const double Sy22 = Sy * Sy / size;
   const double SxSy = Sx * Sy / size;
   div = (Sx2 - Sx22) * (Sy2 - Sy22);
   if(fabs(div) < DBL_EPSILON) return R2A();
   R2A result;
   result.r2 = (Sxy - SxSy) * (Sxy - SxSy) / div;
   result.angle = (Sxy - SxSy) / (Sx2 - Sx22);
   return result;
}
For optimization, we need one criterion value, and here the angle is important because a smooth falling
balance curve with a negative slope can also get a good R2 estimate. Therefore, we will write one more
function that will "add minus" to any estimates of R2 with a negative angle. We take the value of R2
modulo because it can itself be negative in the case of very bad (scattered) data that do not fit into our
linear model. Thus, we must prevent a situation where a minus times minus gives a plus.

---

## Page 1485

Part 6. Trading automation
1 485
6.5 Testing and optimization of Expert Advisors
double RSquaredTest(const double &data[])
{
   const R2A result = RSquared(data);
   const double weight = 1.0 - 1.0 / sqrt(ArraySize(data) + 1);
   if(result.angle < 0) return -fabs(result.r2) * weight;
   return result.r2 * weight;
}
Additionally, our criterion takes into account the size of the series, which corresponds to the number of
trades. Due to this, an increase in the number of transactions will increase the indicator.
Having this tool at our disposal, we will implement the function of calculating the balance line in the
Expert Advisor and find R2 for it. At the end, we multiply the value by 1 00, thereby converting the scale
to the range of the built-in complex criterion.

---

## Page 1486

Part 6. Trading automation
1 486
6.5 Testing and optimization of Expert Advisors
#define STAT_PROPS 4
   
double GetR2onBalanceCurve()
{
   HistorySelect(0, LONG_MAX);
   
   const ENUM_DEAL_PROPERTY_DOUBLE props[STAT_PROPS] =
   {
      DEAL_PROFIT, DEAL_SWAP, DEAL_COMMISSION, DEAL_FEE
   };
   double expenses[][STAT_PROPS];
   ulong tickets[]; // only needed because of the 'select' prototype, but useful for debugging
   
   DealFilter filter;
   filter.let(DEAL_TYPE, (1 << DEAL_TYPE_BUY) | (1 << DEAL_TYPE_SELL), IS::OR_BITWISE)
      .let(DEAL_ENTRY,
      (1 << DEAL_ENTRY_OUT) | (1 << DEAL_ENTRY_INOUT) | (1 << DEAL_ENTRY_OUT_BY),
      IS::OR_BITWISE)
      .select(props, tickets, expenses);
   
   const int n = ArraySize(tickets);
   
   double balance[];
   
   ArrayResize(balance, n + 1);
   balance[0] = TesterStatistics(STAT_INITIAL_DEPOSIT);
   
   for(int i = 0; i < n; ++i)
   {
      double result = 0;
      for(int j = 0; j < STAT_PROPS; ++j)
      {
         result += expenses[i][j];
      }
      balance[i + 1] = result + balance[i];
   }
   const double r2 = RSquaredTest(balance);
   return r2 * 100;
}
In the OnTester handler, we will use the new criterion under the conditional compilation directive, so we
need to uncomment the directive #define USE_ R2_ CRITERION at the beginning of the source code.

---

## Page 1487

Part 6. Trading automation
1 487
6.5 Testing and optimization of Expert Advisors
double OnTester()
{
#ifdef USE_R2_CRITERION
   return GetR2onBalanceCurve();
#else
   const double profit = TesterStatistics(STAT_PROFIT);
   return sign(profit) * sqrt(fabs(profit))
      * sqrt(TesterStatistics(STAT_PROFIT_FACTOR))
      * sqrt(TesterStatistics(STAT_TRADES))
      * sqrt(fabs(TesterStatistics(STAT_SHARPE_RATIO)));
#endif      
}
Let's delete the previous results of optimizations (opt-files with cache) and launch a new optimization of
the Expert Advisor: by the R2 criterion.
When comparing the values of the R2 criterion with the complex criterion, we can say that the
"convergence" between them has increased.
Comparison of custom criterion R2 and complex built-in criterion
The values of the R2 criterion in the optimization window and on the forward period for the
corresponding sets of parameters look as follows.

---

## Page 1488

Part 6. Trading automation
1 488
6.5 Testing and optimization of Expert Advisors
Criterion R2 on periods of optimization and forward tests
And here is how the profits in the past and in the future are combined.
Profit on periods of optimization and forward tests for R2

---

## Page 1489

Part 6. Trading automation
1 489
6.5 Testing and optimization of Expert Advisors
The statistics are as follows: out of the last 5582 profitable passes, 2638 (47%) remained profitable,
and out of the first 1 000 most profitable passes there are 566 that remained profitable, which is
comparable to the built-in complex criterion.
As mentioned above, the statistics provide raw source material for the next, more intelligent
optimization stages, which is more than just a programming task. We will concentrate on other, purely
programmatic aspects of optimization.
6.5.8 Auto-tuning: ParameterGetRange and ParameterSetRange
In the previous section, we learned how to pass an optimization criterion to the tester. However, we
missed one important point. If you look into our optimization logs, you can see a lot of error messages
there, like the ones below.
...
Best result 90.61004580175876 produced at generation 25. Next generation 26
genetic pass (26, 388) tested with error "incorrect input parameters" in 0:00:00.021
genetic pass (26, 436) tested with error "incorrect input parameters" in 0:00:00.007
genetic pass (26, 439) tested with error "incorrect input parameters" in 0:00:00.007
genetic pass (26, 363) tested with error "incorrect input parameters" in 0:00:00.008
genetic pass (26, 365) tested with error "incorrect input parameters" in 0:00:00.008
...
In other words, every few test passes, something is wrong with the input parameters, and such a pass
is not performed. The OnInit handler contains the following check:
   if(FastOsMA >= SlowOsMA) return INIT_PARAMETERS_INCORRECT;
On our part, it is quite logical to impose such a restriction that the period of the slow MA should be
greater than the period of the fast one. However, the tester does not know such things about our
algorithm therefore tries to sort through a variety of combinations of periods, including incorrect ones.
This might be a common situation for optimization which, however, has a negative consequence.
Since we apply genetic optimization, there are several rejected samples in each generation that do not
participate in further mutations. The MetaTrader 5 optimizer does not make up for these losses, i.e., it
does not generate a replacement for them. Then, a smaller population size can negatively affect
quality. Thus, it is necessary to come up with a way to ensure that the input settings are enumerated
only in the correct combinations. And here two MQL5 API functions come to our aid:
ParameterGetRange and ParameterSetRange.
Both functions have two overloaded prototypes that differ in parameter types: long and double. This is
how the two variants of the ParameterGetRange function are described.
bool ParameterGetRange(const string name, bool &enable, long &value, long &start, long &step, long
&stop)
bool ParameterGetRange(const string name, bool &enable, double &value, double &start, double
&step, double &stop)
For the input variable specified by name, the function receives information about its current value
(value), range of values (start, stop), and change step (step) during optimization. In addition, an
attribute is written to the enable variable of whether the optimization is enabled for the input variable
named 'name'.
The function returns an indication of success (true) or error (false).

---

## Page 1490

Part 6. Trading automation
1 490
6.5 Testing and optimization of Expert Advisors
The function can only be called from three special optimization-related handlers: OnTesterInit,
OnTesterPass, and OnTesterDeinit. We will talk about them in the next section. As you can guess from
the names, OnTesterInit is called before optimization starts, OnTesterDeinit – after completion of
optimization, and OnTesterPass – after each pass in the optimization process. For now, we are only
interested in OnTesterInit. Just like the other two functions, it has no parameters and can be declared
with the type void, i.e., it returns nothing.
Two versions of the ParameterSetRange function have similar prototypes and perform the opposite
action: they set the optimization properties of the Expert Advisor's input parameter.
bool ParameterSetRange(const string name, bool enable, long value, long start, long step, long stop)
bool ParameterSetRange(const string name, bool enable, double value, double start, double step,
double stop)
The function sets the modification rules of the input variable with the name name when optimizing:
value, change step, start and end values.
This function can only be called from the OnTesterInit handler when starting optimization in the
strategy tester.
Thus, using the ParameterGetRange and ParameterSetRange functions, you can analyze and set new
range and step values, as well as completely exclude, or, vice versa, include certain parameters from
optimization, despite the settings in the strategy tester. This allows you to create your own scripts to
manage the space of input parameters during optimization.
The function allows you to use in optimization even those variables that are declared with the sinput
modifier (they are not available for inclusion in the optimization by the user).
Attention! After the call of ParameterSetRange with a change in the settings of a specific input
variable, subsequent calls of ParameterGetRange will not "see" these changes and will still return to
the original settings. This makes it impossible to use functions together in complex software
products, where settings can be handled by different classes and libraries from independent
developers.
Let's improve the BandOsMA Expert Advisor using the new functions. The updated version is named
BandOsMApro.mq5 ("pro" can be conditionally decoded as "parameter range optimization").
So, we have the OnTesterInit handler, in which we read the settings for the FastOsMA and SlowOsMA
parameters, and check if they are included in the optimization. If so, you need to turn them off and
offer something in return.

---

## Page 1491

Part 6. Trading automation
1 491 
6.5 Testing and optimization of Expert Advisors
void OnTesterInit()
{
   bool enabled1, enabled2;
   long value1, start1, step1, stop1;
   long value2, start2, step2, stop2;
   if(ParameterGetRange("FastOsMA", enabled1, value1, start1, step1, stop1)
   && ParameterGetRange("SlowOsMA", enabled2, value2, start2, step2, stop2))
   {
      if(enabled1 && enabled2)
      {
         if(!ParameterSetRange("FastOsMA", false, value1, start1, step1, stop1)
         || !ParameterSetRange("SlowOsMA", false, value2, start2, step2, stop2))
         {
            Print("Can't disable optimization by FastOsMA and SlowOsMA: ",
               E2S(_LastError));
            return;
         }
         ...
      }
   }
   else
   {
      Print("Can't adjust optimization by FastOsMA and SlowOsMA: ", E2S(_LastError));
   }
}
Unfortunately, due to the addition of OnTesterInit, the compiler also requires you to add
OnTesterDeinit, although we do not need this function. But we are forced to agree and add an empty
handler.
void OnTesterDeinit()
{
}
The presence of the OnTesterInit/OnTesterDeinit functions in the code will lead to the fact that when
the optimization is started, an additional chart will open in the terminal with a copy of our Expert
Advisor running on it. It works in a special mode that allows you to receive additional data (the so-
called frames) from tested copies on agents, but we will explore this possibility later. For now, it is
important for us to note that all operations with files, logs, charts, and objects work in this auxiliary
copy of the Expert Advisor directly in the terminal, as usual (and not on the agent). In particular, all
error messages and Print calls will be displayed in the log on the Experts tab of the terminal.
We have information about the change ranges and steps of these parameters, we can literally
recalculate all the correct combinations. This task is assigned to a separate Iterate function because a
similar operation will have to be reproduced by copies of the Expert Advisor on agents, in the OnInit
handler.
In the Iterate function, we have two nested loops over the periods of fast and slow MA in which we
count the number of valid combinations, i.e. when the i period is less than j. We need the optional find
parameter when calling Iterate from OnInit to return the pair by the sequence number of the
combination i and j. Since it is required to return 2 numbers, we declared the PairOfPeriods structure
for them.

---

## Page 1492

Part 6. Trading automation
1 492
6.5 Testing and optimization of Expert Advisors
struct PairOfPeriods
{
   int fast;
   int slow;
};
   
PairOfPeriods Iterate(const long start1, const long stop1, const long step1,
   const long start2, const long stop2, const long step2,
   const long find = -1)
{
   int count = 0;
   for(int i = (int)start1; i <= (int)stop1; i += (int)step1)
   {
      for(int j = (int)start2; j <= (int)stop2; j += (int)step2)
      {
         if(i < j)
         {
            if(count == find)
            {
               PairOfPeriods p = {i, j};
               return p;
            }
            ++count;
         }
      }
   }
   PairOfPeriods p = {count, 0};
   return p;
}
When calling Iterate from OnTesterInit, we don't use the find parameter and keep counting until the
very end, and return the resulting amount in the first field of the structure. This will be the range of
values of some new shadow parameter, for which we must enable optimization. Let's call it
FastSlowCombo4Optimization and add to the new group of auxiliary input parameters. More will be
added here soon.
input group "A U X I L I A R Y"
sinput int FastSlowCombo4Optimization = 0;   // (reserved for optimization)
...
Let's go back to OnTesterInit and organize an MQL5 optimization by the FastSlowCombo4Optimization
parameter in the desired range using ParameterSetRange.

---

## Page 1493

Part 6. Trading automation
1 493
6.5 Testing and optimization of Expert Advisors
void OnTesterInit()
{
   ...
         PairOfPeriods p = Iterate(start1, stop1, step1, start2, stop2, step2);
         const int count = p.fast;
         ParameterSetRange("FastSlowCombo4Optimization", true, 0, 0, 1, count);
         PrintFormat("Parameter FastSlowCombo4Optimization is enabled with maximum: %d",
            count);
   ...
}
Please note that the resulting number of iterations for the new parameter should be displayed in the
terminal log.
When testing on the agent, use the number in FastSlowCombo4Optimization to get a couple of periods
by calling Iterate again, this time with the filled find parameter. But the problem is that for this
operation, it is required to know the initial ranges and the FastOsMA and SlowOsMA parameter change
step. This information is present only in the terminal. So, we need to somehow transfer it to the agent.
Now we will apply the only solution we know so far: we will add 3 more shadow optimization parameters
and set some values for them. In the future, we will get acquainted with the technology of transferring
files to agents (see Preprocessor directives for the tester). Then we will be able to write to the file the
entire array of indexes calculated by the Iterate function and send it to agents. This will avoid three
extra shadow optimization parameters.
So, let's add three input parameters:
sinput ulong FastShadow4Optimization = 0;    // (reserved for optimization)
sinput ulong SlowShadow4Optimization = 0;    // (reserved for optimization)
sinput ulong StepsShadow4Optimization = 0;   // (reserved for optimization)
We use the ulong type to be more economical: to pack 2 int numbers into each value. This is how they
are filled in OnTesterInit.
void OnTesterInit()
{
   ...
         const ulong fast = start1 | (stop1 << 16);
         const ulong slow = start2 | (stop2 << 16);
         const ulong step = step1 | (step2 << 16);
         ParameterSetRange("FastShadow4Optimization", false, fast, fast, 1, fast);
         ParameterSetRange("SlowShadow4Optimization", false, slow, slow, 1, slow);
         ParameterSetRange("StepsShadow4Optimization", false, step, step, 1, step);
   ...
}
All 3 parameters are non-optimizable (false in the second argument).
This concludes our operations with the OnTesterInit function. Let's move to the receiving side: the
OnInit handler.

---

## Page 1494

Part 6. Trading automation
1 494
6.5 Testing and optimization of Expert Advisors
int OnInit()
{
   // keep the check for single tests
   if(FastOsMA >= SlowOsMA) return INIT_PARAMETERS_INCORRECT;
   
   // when optimizing, we require the presence of shadow parameters
   if(MQLInfoInteger(MQL_OPTIMIZATION) && StepsShadow4Optimization == 0)
   {
      return INIT_PARAMETERS_INCORRECT;
   }
   
   PairOfPeriods p = {FastOsMA, SlowOsMA}; // by default we work with normal parameters
   if(FastShadow4Optimization && SlowShadow4Optimization && StepsShadow4Optimization)
   {
      // if the shadow parameters are full, decode them into periods
      int FastStart = (int)(FastShadow4Optimization & 0xFFFF);
      int FastStop = (int)((FastShadow4Optimization >> 16) & 0xFFFF);
      int SlowStart = (int)(SlowShadow4Optimization & 0xFFFF);
      int SlowStop = (int)((SlowShadow4Optimization >> 16) & 0xFFFF);
      int FastStep = (int)(StepsShadow4Optimization & 0xFFFF);
      int SlowStep = (int)((StepsShadow4Optimization >> 16) & 0xFFFF);
      
      p = Iterate(FastStart, FastStop, FastStep,
         SlowStart, SlowStop, SlowStep, FastSlowCombo4Optimization);
      PrintFormat("MA periods are restored from shadow: FastOsMA=%d SlowOsMA=%d",
         p.fast, p.slow);
   }
   
   strategy = new SimpleStrategy(
      new BandOsMaSignal(p.fast, p.slow, SignalOsMA, PriceOsMA,
         BandsMA, BandsShift, BandsDeviation,
         PeriodMA, ShiftMA, MethodMA),
         Magic, StopLoss, Lots);
   return INIT_SUCCEEDED;
}
Using the MQLInfoInteger function, we can determine all Expert Advisor modes, including those related
to the tester and optimization. Having specified one of the elements of the ENUM_MQL_INFO_INTEGER
enumeration as a parameter, we will get a logical sign as a result (true/false):
• MQL_TESTER – the program works in the tester
• MQL_VISUAL_MODE – the tester is running in the visual mode
• MQL_OPTIMIZATION – the test pass is performed during optimization (not separately)
• MQL_FORWARD – the test pass is performed on the forward period after optimization (if specified
by optimization settings)
• MQL_FRAME_MODE – the Expert Advisor is running in a special service mode on the terminal chart
(and not on the agent) to control optimization (more on this in the next section)

---

## Page 1495

Part 6. Trading automation
1 495
6.5 Testing and optimization of Expert Advisors
Tester modes of MQL programs
Everything is ready to start optimization. As soon as it starts, with the mentioned settings
Presets/MQL5Book/BandOsMA.set, we will see a message in the Experts log in the terminal:
Parameter FastSlowCombo4Optimization is enabled with maximum: 698
This time there should be no errors in the optimization log and all generations are generated without
crashing.
...
Best result 91.02452934181422 produced at generation 39. Next generation 42
Best result 91.56338892567393 produced at generation 42. Next generation 43
Best result 91.71026391877101 produced at generation 43. Next generation 44
Best result 91.71026391877101 produced at generation 43. Next generation 45
Best result 92.48460871443507 produced at generation 45. Next generation 46
...
This can be determined even by the increased overall optimization time: earlier, some passes were
rejected at an early stage, and now they are all processed in full.
But our solution has one drawback. Now the working settings of the Expert Advisor include not just a
couple of periods in the FastOsMA and SlowOsMA parameters, but also the ordinal number of their
combination among all possible (FastSlowCombo4Optimization). The only thing we can do is output the
periods decoded in the OnInit function, which was demonstrated above.
Thus, having found good settings with the help of optimization, the user, as usual, will perform a single
run to refine the behavior of the trading system. At the beginning of the test log, an inscription of the
following form should appear:

---

## Page 1496

Part 6. Trading automation
1 496
6.5 Testing and optimization of Expert Advisors
MA periods are restored from shadow: FastOsMA=27 SlowOsMA=175
Then you can enter the specified periods in the parameters of the same name, and reset all shadow
parameters.
6.5.9 Group of OnTester events for optimization control
There are three special events in MQL5 to manage the optimization process and transfer arbitrary
applied results (in addition to trading indicators) from agents to the terminal: OnTesterInit,
OnTesterDeinit, and OnTesterPass. Having described the handlers for them in the code, the programmer
will be able to perform the actions they need before starting the optimization, after the optimization is
completed, and at the end of each of the individual optimization passes (if application data has been
received from the agent, more on that below).
All handlers are optional. As we have seen, optimization works without them. It should also be
understood that all three events work only during optimization, but not in a single test.
The Expert Advisor with these handlers is automatically loaded on a separate chart of the terminal with
the symbol and period specified in the tester. This Expert instance Advisor does not trade, but only
performs service actions. All other event handlers, such as OnInit, OnDeinit, and OnTick do not work in
it.
To find out whether an Expert Advisor is executed in the regular trading mode on the agent or in the
service mode in the terminal, call the function MQLInfoInteger(MQL_ FRAME_ MODE) in its code and get
true or false. This service mode is also referred to as the "frames" mode which applies to data packets
that can be sent to the terminal from Expert Advisor instances on agents. We will see a little later how
it is done.
During optimization, only one Expert Advisor instance works in the terminal and, if necessary, receives
incoming frames. Don't forget that such an instance is launched only if the Expert Advisor code
contains one of the three described event handlers.
The OnTesterInit event is generated when optimization is launched in the strategy tester before the
very first pass. The handler has two versions: with return type int and void.
int OnTesterInit(void)
void OnTesterInit(void)
In the int return version, a zero value (INIT_SUCCEEDED) means successful initialization of the Expert
Advisor launched on the chart in the terminal, which allows starting optimization. Any other value
means an error code, and optimization will not start.
The second version of the function always implies successful preparation of the Expert Advisor for
optimization.
A limited time is provided for the execution of OnTesterInit, after which the Expert Advisor will be forced
to terminate, and the optimization itself will be canceled. In this case, a corresponding message will be
displayed in the tester's log.
In the previous section, we saw an example of how the OnTesterInit handler was used to modify the
optimization parameters using the ParameterGetRange/ParameterSetRange functions.
void OnTesterDeinit(void)
The OnTesterDeinit function is called upon completion of the Expert Advisor optimization.

---

## Page 1497

Part 6. Trading automation
1 497
6.5 Testing and optimization of Expert Advisors
The function is intended for the final processing of applied optimization results. For example, if a file
was opened in OnTesterInit to write the contents of frames, then it needs to be closed in
OnTesterDeinit.
void OnTesterPass(void)
The OnTesterPass event is automatically generated when a data frame arrives during optimization. The
function allows the processing of application data received from Expert Advisor instances running on
agents during optimization. A frame from the testing agent must be sent from the OnTester handler
using the FrameAdd function.
The diagram shows the sequence of events when optimizing Expert Advisors
A standard set of financial statistics about each test pass is sent from the agents to the terminal
automatically. The Expert Advisor is not required to send anything using FrameAdd if it doesn't need
it. If frames are not used, the OnTesterPass handler will not be called.
By using OnTesterPass, you can dynamically process the optimization results "on the go", for example,
display them on a chart in the terminal or add them to a file for subsequent batch processing.
To demonstrate the capabilities of OnTester event handlers, we first need to learn the functions for
working with frames. They are presented in the following sections.
6.5.1 0 Sending data frames from agents to the terminal
MQL5 provides a group of functions for organizing the transfer and processing of your own (applied)
optimization results, in addition to standard financial indicators and statistics. One of them, FrameAdd,
is designed to send data from testing agents. Other functions are intended to receive data in the
terminal.

---

## Page 1498

Part 6. Trading automation
1 498
6.5 Testing and optimization of Expert Advisors
The data exchange format is based on frames. This is a special internal structure that an Expert Advisor
can fill in the tester based on an array of a simple type (which does not contain strings, class objects,
or dynamic arrays) or using a file with a specified name (the file must first be created in the agent's
sandbox). By calling the FrameAdd function multiple times, the Expert Advisor can send a series of
frames to the terminal. There are no limits on the number of frames.
There are two versions of the FrameAdd function.
bool FrameAdd(const string name, ulong id, double value, const string filename)
bool FrameAdd(const string name, ulong id, double value, const void &data[])
The function adds a data frame to the buffer to be sent to the terminal. The name and id parameters
are public labels that can be used to filter frames in the FrameFilter function. The value parameter
allows you to pass an arbitrary numeric value that can be used when one value is enough. More bulky
data is indicated either in the data array (may be an array of simple structures) or in a file named
filename.
If there is no bulk data to transfer (for example, you only need to transfer the status of the process),
use the first form of the function and specify NULL instead of a string with the file name or the second
form with a dummy array of zero size.
The function returns true in case of success.
The function can only be called in the OnTester handler.
The function has no effect when called during a simple test, that is, outside of optimization.
You can send data only from agents to the terminal. There are no mechanisms in MQL5 for sending
data in the opposite direction during optimization. All data that the Expert Advisor wants to send to
agents must be prepared and available (in the form of input parameters or files connected by
directives) before starting the optimization.
We will look at an example of using FrameAdd after we get familiar with the functions of the host in the
next section.
6.5.1 1  Getting data frames in terminal
Frames sent from testing agents by the FrameAdd function are delivered into the terminal and written
in the order of receipt to an mqd file having the name of the Expert Advisor into the folder
terminal_ directory/MQL5/Files/Tester. The arrival of one or more frames at once generates the
OnTesterPass event.
The MQL5 API provides 4 functions for analyzing and reading frames: FrameFirst, FrameFilter,
FrameNext, and FrameInputs. All functions return a boolean value with an indication of success (true)
or error (false).
To access existing frames, the kernel maintains the metaphor of an internal pointer to the current
frame. The pointer automatically moves forward when the next frame is read by the FrameNext
function, but it can be returned to the beginning of all frames with FrameFirst or FrameFilter. Thus, an
MQL program can organize the iteration of frames in a loop until it has looked through all the frames.
This process can be repeated if necessary, for example, by applying different filters in OnTesterDeinit.

---

## Page 1499

Part 6. Trading automation
1 499
6.5 Testing and optimization of Expert Advisors
bool FrameFirst()
The FrameFirst function sets the internal frame reading pointer to the beginning and resets the filter (if
it was previously set using the FrameFilter function).
In theory, for a single reception and processing of all frames, it is not necessary to call FrameFirst,
since the pointer is already at the beginning when the optimization starts.
bool FrameFilter(const string name, ulong id)
It sets the frame reading filter and sets the internal frame pointer to the beginning. The filter will affect
which frames are included in subsequent calls of FrameNext.
If an empty string is passed as the first parameter, the filter will work only by a numeric parameter,
that is, all frames with the specified id. If the value of the second parameter is equal to ULONG_MAX,
then only the text filter works.
Calling FrameFilter("", ULONG_ MAX) is equivalent to calling FrameFirst(), which is equivalent to the
absence of a filter.
If you call FrameFirst or FrameFilter in OnTesterPass, make sure this is really what you need: the
code probably contains a logical error as it is possible to loop, read the same frame, or increase the
computational load exponentially.
bool FrameNext(ulong &pass, string &name, ulong &id, double &value)
bool FrameNext(ulong &pass, string &name, ulong &id, double &value, void &data[])
The FrameNext function reads one frame and moves the pointer to the next one. The pass parameter
will have the optimization pass number recorded in it. The name, id, and value parameters will receive
the values passed in the corresponding parameters of the FrameAdd function.
It is important to note that the function can return false while operating normally when there are no
more frames to read. In this case, the built-in variable _ LastError contains the value 4000 (it has no
built-in notation).
No matter which form of the FrameAdd function was used to send data, the contents of the file or array
will be placed in the receiving data array. The type of the receiving array must match the type of the
sent array, and there are certain nuances in the case of sending a file.
A binary file (FILE_BIN) should preferably be accepted in a byte array uchar to ensure compatibility
with any size (because other larger types may not be a multiple of the file size). If the file size (in fact,
the size of the data block in the received frame) is not a multiple of the size of the receiving array type,
the FrameNext function will not read the data and will return an INVALID_ARRAY (4006) error.
A Unicode text file (FILE_TXT or FILE_CSV without FILE_ANSI modifier) should be accepted into an
array of ushort type and then converted to a string by calling ShortArrayToString. An ANSI text file
should be received in a uchar array and converted using CharArrayToString.
bool FrameInputs(ulong pass, string &parameters[], uint &count)
The FrameInputs function allows you to get descriptions and values of Expert Advisor input parameters
on which the pass with the specified pass number is formed. The parameters string array will be filled
with lines like "ParameterNameN=ValueParameterN". The count parameter will be filled with the
number of elements in the parameters array.
The calls of these four functions are only allowed inside the OnTesterPass and OnTesterDeinit handlers.

---

## Page 1500

Part 6. Trading automation
1 500
6.5 Testing and optimization of Expert Advisors
Frames can arrive to the terminal in batches, in which case it takes time to deliver them. So, it is not
necessary that all of them have time to generate the OnTesterPass event and will be processed until
the end of the optimization. In this regard, in order to guarantee the receipt of all late frames, it is
necessary to place a block of code with their processing (using the FrameNext function) in
OnTesterDeinit.
Consider a simple example FrameTransfer.mq5.
The Expert Advisor has four test parameters. All of them, except for the last string, can be included in
the optimization.
input bool Parameter0;
input long Parameter1;
input double Parameter2;
input string Parameter3;
However, to simplify the example, the number of steps for parameters Parameter1  and Parameter2 is
limited to 1 0 (for each). Thus, if you do not use Parameter0, the maximum number of passes is 1 21 .
Parameter3 is an example of a parameter that cannot be included in the optimization.
The Expert Advisor does not trade but generates random data that mimics arbitrary application data.
Do not use randomization like this in your work projects: it is only suitable for demonstration.
ulong startup; // track the time of one run (just like demo data)
   
int OnInit()
{
   startup = GetMicrosecondCount();
   MathSrand((int)startup);
   return INIT_SUCCEEDED;
}
Data is sent in two types of frames: from a file and from an array. Each type has its own identifier.

---

## Page 1501

Part 6. Trading automation
1 501 
6.5 Testing and optimization of Expert Advisors
#define MY_FILE_ID 100
#define MY_TIME_ID 101
   
double OnTester()
{
 // send file in one frame
   const static string filename = "binfile";
   int h = FileOpen(filename, FILE_WRITE | FILE_BIN | FILE_ANSI);
   FileWriteString(h, StringFormat("Random: %d", MathRand()));
   FileClose(h);
   FrameAdd(filename, MY_FILE_ID, MathRand(), filename);
   
 // send array in another frame
   ulong dummy[1];
   dummy[0] = GetMicrosecondCount() - startup;
   FrameAdd("timing", MY_TIME_ID, 0, dummy);
   
   return (Parameter2 + 1) * (Parameter1 + 2);
}
The file is written as binary, with simple strings. The result (criterion) of OnTester is a simple arithmetic
expression involving Parameter1  and Parameter2.
On the receiving side, in the Expert Advisor instance running in the service mode on the terminal chart,
we collect data from all frames with files and put them into a common CSV file. The file is opened in the
handler OnTesterInit.
int handle; // file for collecting applied results
void OnTesterInit()
{
   handle = FileOpen("output.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
}
As mentioned earlier, all frames may not have time to get into the handler OnTesterPass, and they need
to be additionally checked in OnTesterDeinit. Therefore, we have implemented one helper function
ProcessFileFrames, which we will call from OnTesterPass, and from OnTesterDeinit.
Inside ProcessFileFrames we keep our internal counter of processed frames, framecount. Using it as an
example, we will make sure that the order of arrival of frames and the numbering of test passes often
do not match.
void ProcessFileFrames()
{
   static ulong framecount = 0;
   ...
To receive frames in the function, the variables necessary according to the prototype FrameNext are
described. The receiving data array is described here as uchar. If we were to write some structures to
our binary file, we could take them directly into an array of structures of the same type.

---

## Page 1502

Part 6. Trading automation
1 502
6.5 Testing and optimization of Expert Advisors
   ulong   pass;
   string  name;
   long    id;
   double  value;
   uchar   data[];
   ...
The following describes the variables for obtaining the Expert Advisor inputs for the current pass to
which the frame belongs.
   string  params[];
   uint    count;
   ...
We then read frames in a loop with FrameNext. Recall that several frames can enter the handler at
once, so a loop is needed. For each frame, we output to the terminal log the pass number, the name of
the frame, and the resulting double value. We skip frames with an ID other than MY_FILE_ID and will
process them later.
   ResetLastError();
   
   while(FrameNext(pass, name, id, value, data))
   {
      PrintFormat("Pass: %lld Frame: %s Value:%f", pass, name, value);
      if(id != MY_FILE_ID) continue;
      ...
   }
   
   if(_LastError != 4000 && _LastError != 0)
   {
      Print("Error: ", E2S(_LastError));
   }
}
For frames with MY_FILE_ID, we do the following: query the input variables, find out which ones are
included in the optimization, and save their values to a common CSV file along with the information
from the frame. When the frame count is 0, we form the header of the CSV file in the header variable.
In all frames, the current (new) record for the CSV file is formed in the record variable.

---

## Page 1503

Part 6. Trading automation
1 503
6.5 Testing and optimization of Expert Advisors
void ProcessFileFrames()
{
   ...
      if(FrameInputs(pass, params, count))
      {
         string header, record;
         if(framecount == 0) // prepare CSV header
         {
            header = "Counter,Pass ID,";
         }
         record = (string)framecount + "," + (string)pass + ",";
         // collect optimized parameters and their values
         for(uint i = 0; i < count; i++)
         {
            string name2value[];
            int n = StringSplit(params[i], '=', name2value);
            if(n == 2)
            {
               long pvalue, pstart, pstep, pstop;
               bool enabled = false;
               if(ParameterGetRange(name2value[0],
                  enabled, pvalue, pstart, pstep, pstop))
               {
                  if(enabled)
                  {
                     if(framecount == 0) // prepare CSV header
                     {
                        header += name2value[0] + ",";
                     }
                     record += name2value[1] + ","; // data field
                  }
               }
            }
         }
         if(framecount == 0) // prepare CSV header
         {
            FileWriteString(handle, header + "Value,File Content\n");
         }
         // write data to CSV
         FileWriteString(handle, record + DoubleToString(value) + ","
            + CharArrayToString(data) + "\n");
      }
      framecount++;
   ...
}
Calling ParameterGetRange could also be done more efficiently, only with a zero value of framecount.
You can try to do so.
In the OnTesterPass handler, we just call ProcessFileFrames.

---

## Page 1504

Part 6. Trading automation
1 504
6.5 Testing and optimization of Expert Advisors
void OnTesterPass()
{
   ProcessFileFrames(); // standard processing of frames on the go
}
Additionally, we call the same function from OnTesterDeinit and close the CSV file.
void OnTesterDeinit()
{
   ProcessFileFrames(); // pick up late frames
   FileClose(handle);   // close the CSV file
   ..
}
In OnTesterDeinit, we process frames with MY_TIME_ID. The durations of test passes is delivered in
these frames, and the average duration of one pass is calculated here. In theory, it makes sense to do
this only for analysis in your program, since for the user the duration of the passes is already displayed
by the tester in the log.
void OnTesterDeinit()
{
   ...
   ulong   pass;
   string  name;
   long    id;
   double  value;
   ulong   data[]; // same array type as sent
   
   FrameFilter("timing", MY_TIME_ID); // rewind to the first frame
   
   ulong count = 0;
   ulong total = 0;
   // cycle through 'timing' frames only
   while(FrameNext(pass, name, id, value, data))
   {
      if(ArraySize(data) == 1)
      {
         total += data[0];
      }
      else
      {
         total += (ulong)value;
      }
      ++count;
   }
   if(count > 0)
   {
      PrintFormat("Average timing: %lld", total / count);
   }
}
The Expert Advisor is ready. Let's enable the complete optimization for it (because the total number of
options is artificially limited and is too small for the genetic algorithm). We can choose open prices only

---

## Page 1505

Part 6. Trading automation
1 505
6.5 Testing and optimization of Expert Advisors
since the Expert Advisor does not trade. Because of this, you should choose a custom criterion (all
other criteria will give 0). For example, let's set the range Parameter1  from 1  to 1 0 in single steps, and
Parameter2 is set from -0.5 to +0.5 in steps of 0.1 .
Let's run the optimization. In the expert log in the terminal, we will see entries about received frames
of the form:
Pass: 0 Frame: binfile Value:5105.000000
Pass: 0 Frame: timing Value:0.000000
Pass: 1 Frame: binfile Value:28170.000000
Pass: 1 Frame: timing Value:0.000000
Pass: 2 Frame: binfile Value:17422.000000
Pass: 2 Frame: timing Value:0.000000
...
Average timing: 1811
The corresponding lines with pass numbers, parameter values and frame contents will appear in the
output.csv file:
Counter,Pass ID,Parameter1,Parameter2,Value,File Content
0,0,0,-0.5,5105.00000000,Random: 87
1,1,1,-0.5,28170.00000000,Random: 64
2,2,2,-0.5,17422.00000000,Random: 61
...
37,35,2,-0.2,6151.00000000,Random: 68
38,62,7,0.0,17422.00000000,Random: 61
39,36,3,-0.2,16899.00000000,Random: 71
40,63,8,0.0,17422.00000000,Random: 61
...
117,116,6,0.5,27648.00000000,Random: 74
118,117,7,0.5,16899.00000000,Random: 71
119,118,8,0.5,17422.00000000,Random: 61
120,119,9,0.5,28170.00000000,Random: 64
Obviously, our internal numbering (column Count) goes in order, and the pass numbers Pass ID can be
mixed (this depends on many factors of parallel processing of job batches by agents). In particular, the
batch of tasks can be the first to finish the agent to which the tasks with higher sequence numbers
were assigned: in this case, the numbering in the file will start from the higher passes.
In the tester's log, you can check service statistics by frames.
242 frames (42.78 Kb total, 181 bytes per frame) received
local 121 tasks (100%), remote 0 tasks (0%), cloud 0 tasks (0%)
121 new records saved to cache file 'tester\cache\FrameTransfer.EURUSD.H1. »
  » 20220101.20220201.20.9E2DE099D4744A064644F6BB39711DE8.opt'
It is important to note that during genetic optimization, run numbers are presented in the optimization
report as a pair (generation number, copy number), while the pass number obtained in the FrameNext
function is ulong. In fact, it is the pass number in batch jobs in the context of the current optimization
run. MQL5 does not provide a means to match pass numbering with a genetic report. For this purpose,
the checksums of the input parameters of each pass should be calculated. Opt files with an
optimization cache already contain such a field with an MD5 hash.

---

## Page 1506

Part 6. Trading automation
1 506
6.5 Testing and optimization of Expert Advisors
6.5.1 2 Preprocessor directives for the tester
In the section on General properties of programs, we first become acquainted with #property directives
in MQL programs. Then we met directives intended for scripts, services, and indicators. There is also a
group of directives for the tester. We have already mentioned some of them. For example,
tester_ everytick_ calculate affects the calculation of indicators.
The following table lists all tester directives with explanations.
Directive
Description
tester_indicator "string"
The name of the custom indicator in the format "indicator_name.ex5"
tester_file "string"
File name in the format "file_name.extension" with the initial data
required for the program test
tester_library "string"
Library name with an extension such as "library.ex5" or "library.dll"
tester_set "string"
File name in the format "file_name.set" with settings for values and
ranges of optimization of program input parameters
tester_no_cache
Disabling reading the existing cache of previous optimizations (opt
files)
tester_everytick_calculate
Disabling the resource-saving mode for calculating indicators in the
tester
The last two directives have no arguments. All others expect a double-quoted string with the name of a
file of one type or another. It also follows from this that directives can be repeated with different files,
i.e., you can include several settings files or several indicators.
The tester_ indicator directive is required to connect to the testing process those indicators that are
not mentioned in the source code of the program under test in the form of constant strings (literals).
As a rule, the required indicator can be determined automatically by the compiler from iCustom calls if
its name is explicitly specified in the corresponding parameter, for example, iCustom(symbol, period,
"indicator_ name",...). However, this is not always the case.
Let's say we are writing a universal Expert Advisor that can use different moving average indicators, not
just the standard built-in ones. Then we can create an input variable to specify the name of the
indicator by the user. Then, the iCustom call will turn into iCustom(symbol, period,
CustomIndicatorName,...), where CustomIndicatorName is an input variable of the Expert Advisor, the
content of which is not known at the time of compilation. Moreover, the developer in this case is likely
to apply IndicatorCreate instead of iCustom, since the number and types of indicator parameters must
also be configured. In such cases, to debug the program or demonstrate it with a specific indicator, we
should provide the name to the tester using the tester_ indicator directive.
The need to report indicator names in the source code significantly limits the ability to test such
universal programs that can connect various indicators online.
Without the tester_ indicator directive, the terminal will not be able to send an indicator to the agent
that is not explicitly declared in the source code, as a result of which the dependent program will lose
part or all of its functionality.

---

## Page 1507

Part 6. Trading automation
1 507
6.5 Testing and optimization of Expert Advisors
The tester_ file directive allows you to specify a file that will be transferred to the agents and placed in
the sandbox before testing. The content and type of the file is not regulated. For example, these can be
the weights of a pre-trained neural network, pre-collected Depth of Market data (because such data
cannot be reproduced by the tester), and so on.
Note, that the file from the tester_ file directive is only read if it existed at compile time. If the
source code was compiled when there was no corresponding file, then its appearance in the future
will no longer help: the compiled program will be sent to the agent without an auxiliary file.
Therefore, for example, if the file specified in tester_ file is generated in OnTesterInit, you should
make sure that the file with the given name already existed at the time of compilation, even if it
was empty. We will demonstrate this below.
Please note that the compiler does not generate warnings if the file specified in the tester_ file directive
does not exist.
The connected files must be in the terminal's sandbox" MQL5/Files/.
The tester_ library directive informs the tester about the need to transfer the library, which is an
auxiliary program that can only work in the context of another MQL program, to the agents. We will talk
about libraries in detail in a separate section.
The libraries required for testing are determined automatically by the #import directives in the source
code. However, if any library is used by an external indicator, then this property must be enabled. The
library can be both with the dll extension, as well as with the ex5 extension.
The tester_ set directive operates with set files with MQL program settings. The file specified in the
directive will become available from the context menu of the tester and will allow the user to quickly
apply the settings.
If the name is specified without a path, the set file must be in the same directory as the Expert Advisor.
This is somewhat unexpected, because the default directory for set files is Presets, and this is where
they are saved by commands from the terminal interface. To connect the set file from the given
directory, you must explicitly specify it in the directive and precede it with a slash, which indicates the
absolute path inside the MQL5 folder.
#property tester_set "/Presets/xyz.set"
When there is no leading slash, the path is relative to where the source text was placed.
Immediately after adding the file and recompiling the program, you need to reselect the Expert
Advisor in the tester; otherwise, the file will not be picked up!
If you specify the Expert Advisor name and version number as "<expert_ name> _ <number> .set"  in
the name of the set file, then it will automatically be added to the parameter version download menu
under the version number <number>. For example, the name "MACD Sample_ 4.set" means that it is a
set file for the Expert Advisor "MACD Sample.mq5" with version number 4.
Those interested can study the format of set files: to do this, manually save the testing/optimization
settings in the strategy tester and then open the file created in this way in a text editor.
Now let's look at the directive tester_ no_ cache. When performing optimization, the strategy tester
saves all the results of the performed passes to the optimization cache (files with the extension opt), in
which the test result is stored for each set of input parameters. This allows, when re-optimizing on the
same parameters, to take ready-made results without re-calculation and time wasting.

---

## Page 1508

Part 6. Trading automation
1 508
6.5 Testing and optimization of Expert Advisors
However, for some tasks, such as mathematical calculations, it may be necessary to perform
calculations regardless of the presence of ready-made results in the optimization cache. In this case, in
the source code, you must include the property tester_ no_ cache. At the same time, the test results
themselves will still be stored in the cache so that you can see all the data on the completed passes in
the strategy tester.
The directive tester_ everytick_ calculate is designed to enable the indicator calculation mode on each
tick in the tester.
By default, indicators are calculated in the tester only when they are accessed for data, i.e., when the
values of indicator buffers are requested. This gives a significant speed-up in testing and optimization if
you do not need to get the indicator values at each tick.
However, some programs may require indicators to be recalculated on every tick. It is in such cases
that the property tester_ everytick_ calculate is useful.
Indicators in the strategy tester are also forced to be calculated on each tick in the following cases:
• when testing in visual mode
• if there are the EventChartCustom, OnChartEvent, or OnTimer functions in the indicator
This property applies only to operations in the strategy tester. In the terminal, indicators are always
calculated on each incoming tick.
The directive has actually been used in the FrameTransfer.mq5 Expert Advisor:
#property tester_set "FrameTransfer.set"
We just didn't focus on it. The file "FrameTransfer.set" is located next to the source code. In the same
Expert Advisor, we also needed another directive from the above table:
#property tester_no_cache
In addition, let's consider an example of a directive tester_ file. Earlier in the section on auto-tuning of
Expert Advisor parameters when optimizing, we introduced BandOsMApro.mq5, in which it was
necessary to introduce several shadow parameters to pass optimization ranges to our source code
running on agents.
The tester_ file directive will allow us to get rid of these extra parameters. Let's name the new version
BandOsMAprofile.mq5.
Since we are now familiar with the directive tester_ set, let's add to the new version the previously
mentioned file /Presets/MQL5Book/BandOsMA.set.
#property tester_set "/Presets/MQL5Book/BandOsMA.set"
Information about the range and step of changing periods of FastOsMA and SlowOsMA will be saved to
file BandOsMAprofile.csv" instead of three additional input parameters FastShadow4Optimization,
SlowShadow4Optimization, StepsShadow4Optimization.
#define SETTINGS_FILE "BandOsMAprofile.csv"
#property tester_file SETTINGS_FILE
   
const string SettingsFile = SETTINGS_FILE;
Shadow setting FastSlowCombo4Optimization is still needed for a complete enumeration of allowed
combinations of periods.

---

## Page 1509

Part 6. Trading automation
1 509
6.5 Testing and optimization of Expert Advisors
input group "A U X I L I A R Y"
sinput int FastSlowCombo4Optimization = 0;   // (reserved for optimization)
Recall that we find its range for optimization in the Iterate function. The first time we call it in
OnTesterInit with a complete enumeration of combinations of fast and slow periods.
Basically, we could store all valid combinations in the array of structures PairOfPeriods and write it to a
binary file for transmission to agents. Then, on the agents, our Expert Advisor could read the ready
array from the file and by the FastSlowCombo4Optimization index extract the corresponding pair of
FastOsMA and SlowOsMA from the array.
Instead, we will focus on a minimal change in the working logic of the program: we will continue to
restore a couple of periods due to the second call Iterate in the OnInit handler. This time, we will get
the range and step of enumeration of period values not from the shadow parameters, but from the CSV
file.
Here are the changes to OnTesterInit.
int OnTesterInit()
{
   ...
        // check if the file already exists before compiling
        // - if not, the tester will not be able to send it to agents
         const bool preExisted = FileIsExist(SettingsFile);
         
         // write the settings to a file for transfer to copy programs on agents
         int handle = FileOpen(SettingsFile, FILE_WRITE | FILE_CSV | FILE_ANSI, ",");
         FileWrite(handle, "FastOsMA", start1, step1, stop1);
         FileWrite(handle, "SlowOsMA", start2, step2, stop2);
         FileClose(handle);
         
         if(!preExisted)
         {
            PrintFormat("Required file %s is missing. It has been just created."
               " Please restart again.",
               SettingsFile);
            ChartClose();
            return INIT_FAILED;
         }
   ...
   return INIT_SUCCEEDED;
}
Note that we have made the OnTesterInit handler with the return type int, which makes it possible to
cancel optimization if the file does not exist. However, in any case, the actual data is written to the file,
so if it did not exist, it is now created, and the subsequent start of the optimization will definitely be
successful.
If you want to skip this step, you can create an empty file MQL5/Files/BandOsMAprofile.csv beforehand.
The OnInit handler has been changed as follows.

---

## Page 1510

Part 6. Trading automation
1 51 0
6.5 Testing and optimization of Expert Advisors
int OnInit()
{
   if(FastOsMA >= SlowOsMA) return INIT_PARAMETERS_INCORRECT;
   
   PairOfPeriods p = {FastOsMA, SlowOsMA}; // default initial parameters
   int handle = FileOpen(SettingsFile, FILE_READ | FILE_TXT | FILE_ANSI);
   
   // during optimization, a file with shadow parameters is needed
   if(MQLInfoInteger(MQL_OPTIMIZATION) && handle == INVALID_HANDLE)
   {
      return INIT_PARAMETERS_INCORRECT;
   }
   
   if(handle != INVALID_HANDLE)
   {
      if(FastSlowCombo4Optimization != -1)
      {
         // if there is a shadow copy, read the period values from it
         const string line1 = FileReadString(handle);
         string settings[];
         if(StringSplit(line1, ',', settings) == 4)
         {
            int FastStart = (int)StringToInteger(settings[1]);
            int FastStep = (int)StringToInteger(settings[2]);
            int FastStop = (int)StringToInteger(settings[3]);
            const string line2 = FileReadString(handle);
            if(StringSplit(line2, ',', settings) == 4)
            {
               int SlowStart = (int)StringToInteger(settings[1]);
               int SlowStep = (int)StringToInteger(settings[2]);
               int SlowStop = (int)StringToInteger(settings[3]);
               p = Iterate(FastStart, FastStop, FastStep,
                  SlowStart, SlowStop, SlowStep, FastSlowCombo4Optimization);
               PrintFormat("MA periods are restored from shadow: FastOsMA=%d SlowOsMA=%d",
                  p.fast, p.slow);
            }
         }
      }
      FileClose(handle);
   }
When running single tests after optimization, we will see decoded period values in the log FastOsMA and
SlowOsMA based on the optimized value FastSlowCombo4Optimization. In the future, we can substitute
these values in the period parameters, and delete the csv file. We also provided that the file will not be
taken into account if FastSlowCombo4Optimization is set to -1 .
6.5.1 3 Managing indicator visibility: TesterHideIndicators
By default, the visual testing chart shows all the indicators that are created in the Expert Advisor being
tested. Also, these indicators are shown on the chart, which automatically opens at the end of testing.

---

## Page 1511

Part 6. Trading automation
1 51 1 
6.5 Testing and optimization of Expert Advisors
All this applies only to those indicators that are directly created in your code: nested indicators that
can be used in the calculation of the main indicators do not apply here.
The visibility of indicators is not always desirable from the developer's point of view, who may want to
hide the implementation details of an Expert Advisor. In such cases, the function TesterHideIndicators
will disable the display of the used indicators on the chart.
void TesterHideIndicators(bool hide)
Boolean parameter hide instructs either to hide (by value true) or display (by value false) indicators.
The set state is remembered by the MQL program execution environment until it is changed by calling
the function again with the inverse parameter value. The current state of this setting affects all newly
created indicators.
In other words, the function TesterHideIndicators with the required flag value hide should be called
before creating descriptors of the corresponding indicators. In particular, after calling the function with
the true parameter, new indicators will be marked with a hidden flag and will not be shown during visual
testing and on the chart, which is automatically opened when testing is completed.
To disable the mode of hiding newly created indicators, call TesterHideIndicators with false.
The function is applicable only in the tester.
The function has some specifics related to its performance, provided that special tpl templates are
created for the tester or Expert Advisor in the folder /MQL5/Profiles/Templates.
If there is a special template in the folder <expert_ name>.tpl, then during visual testing and on the
testing chart, only indicators from this template will be shown. In this case, no indicators used in the
tested Expert Advisor will be displayed, even if the function was called in the Expert Advisor code
TesterHideIndicators with false.
If there is a template in the tester.tpl folder, then during visual testing and on the testing chart,
indicators from the tester.tpl template will be shown, plus those indicators from the Expert Advisor that
are not prohibited by the TesterHideIndicators call. The TesterHideIndicators function does not affect
the indicators in the template.
If there is no template tester.tpl, but there is a template default.tpl, then the indicators from it are
processed according to a similar principle.
We will demonstrate how the function works in the Big Expert Advisor example a little later.
6.5.1 4 Emulation of deposits and withdrawals
The MetaTrader 5 tester allows you to emulate deposit and withdrawal operations. This allows you to
experiment with some money management systems.
bool TesterDeposit(double money)
The TesterDeposit function replenishes the account in the process of testing for the size of the
deposited amount in the money parameter. The amount is indicated in the test deposit currency.
bool TesterWithdrawal(double money)
The TesterWithdrawal function makes withdrawals equal to money.
Both functions return true as a sign of success.

---

## Page 1512

Part 6. Trading automation
1 51 2
6.5 Testing and optimization of Expert Advisors
As an example, let's consider an Expert Advisor based on the "carry trade" strategy. For it, we need to
select a symbol with large positive swaps in one of the trading directions, for example, buying AUDUSD.
The Expert Advisor will open one or more positions in the specified direction. Unprofitable positions will
be held for the sake of accumulating swaps on them. Profitable positions will be closed upon reaching a
predetermined amount of profit per lot. Earned swaps will be withdrawn from the account. The source
code is available in the CrazyCarryTrade.mq5 file.
In the input parameters, the user can select the direction of trade, the size of one trade (0 by default,
which means the minimum lot), and the minimum profit per lot, at which a profitable position will be
closed.
enum ENUM_ORDER_TYPE_MARKET
{
   MARKET_BUY = ORDER_TYPE_BUY,
   MARKET_SELL = ORDER_TYPE_SELL
};
   
input ENUM_ORDER_TYPE_MARKET Type;
input double Volume;
input double MinProfitPerLot = 1000;
First, let's test in the handler OnInit the performance of functions TesterWithdrawal and TesterDeposit.
In particular, an attempt to withdraw a double balance will result in error 1 001 9.
int OnInit()
{
   PRTF(TesterWithdrawal(AccountInfoDouble(ACCOUNT_BALANCE) * 2));
   /*
   not enough money for 20 000.00 withdrawal (free margin: 10 000.00)
   TesterWithdrawal(AccountInfoDouble(ACCOUNT_BALANCE)*2)=false / MQL_ERROR::10019(10019)
   */
   ...
But the subsequent withdrawals and crediting back of 1 00 units of the account currency will be
successful.
   PRTF(TesterWithdrawal(100));
   /*
   deal #2 balance -100.00 [withdrawal] done
   TesterWithdrawal(100)=true / ok
   */
   PRTF(TesterDeposit(100)); // return the money 
   /*
   deal #3 balance 100.00 [deposit] done
   TesterDeposit(100)=true / ok
   */
   return INIT_SUCCEEDED;
}
In the OnTick handler, let's check the availability of positions using PositionFilter and fill the values
array with their current profit/loss and accumulated swaps.

---

## Page 1513

Part 6. Trading automation
1 51 3
6.5 Testing and optimization of Expert Advisors
void OnTick()
{
   const double volume = Volume == 0 ?
      SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) : Volume;
   ENUM_POSITION_PROPERTY_DOUBLE props[] = {POSITION_PROFIT, POSITION_SWAP};
   double values[][2];
   ulong tickets[];
   PositionFilter pf;
   pf.select(props, tickets, values, true);
   ...
When there are no positions, we open one in a predefined direction.
   if(ArraySize(tickets) == 0) // no positions 
   {
      MqlTradeRequestSync request1;
      (Type == MARKET_BUY ? request1.buy(volume) : request1.sell(volume));
   }
   else
   {
      ... // there are positions - see the next box
   }
When there are positions, we go through them in a cycle and close those for which there is sufficient
profit (adjusted for swaps). While doing so, we also sum up the swaps of closed positions and total
losses. Since swaps grow in proportion to time, we use them as an amplifying factor for closing "old"
positions. Thus, it is possible to close with a loss.
      double loss = 0, swaps = 0;
      for(int i = 0; i < ArraySize(tickets); ++i)
      {
         if(values[i][0] + values[i][1] * values[i][1] >= MinProfitPerLot * volume)
         {
            MqlTradeRequestSync request0;
            if(request0.close(tickets[i]) && request0.completed())
            {
               swaps += values[i][1];
            }
         }
         else
         {
            loss += values[i][0];
         }
      }
      ...
If the total losses increase, we periodically open additional positions, but we do it less often when there
are more positions, in order to somehow control the risks.

---

## Page 1514

Part 6. Trading automation
1 51 4
6.5 Testing and optimization of Expert Advisors
      if(loss / ArraySize(tickets) <= -MinProfitPerLot * volume * sqrt(ArraySize(tickets)))
      {
         MqlTradeRequestSync request1;
         (Type == MARKET_BUY ? request1.buy(volume) : request1.sell(volume));
      }
      ...
Finally, we remove swaps from the account.
      if(swaps >= 0)
      {
         TesterWithdrawal(swaps);
      }
In the OnDeinit handler, we display statistics on deductions.
void OnDeinit(const int)
{
   PrintFormat("Deposit: %.2f Withdrawals: %.2f",
      TesterStatistics(STAT_INITIAL_DEPOSIT),
      TesterStatistics(STAT_WITHDRAWAL));
}
For example, when running the Expert Advisor with default settings for the period from 2021  to the
beginning of 2022, we get the following result for AUDUSD:
   final balance 10091.19 USD
   Deposit: 10000.00 Withdrawals: 197.42
Here is what the report and graph look like.
Expert report with withdrawals from the account

---

## Page 1515

Part 6. Trading automation
1 51 5
6.5 Testing and optimization of Expert Advisors
Thus, when trading a minimum lot and loading a deposit of no more than 1 % for a little over a year, we
managed to withdraw about 200 USD.
6.5.1 5 Forced test stop: TesterStop
If necessary, depending on the conditions observed, the developer can stop testing the Expert Advisor
earlier. For example, this can be done when a specified number of losing deals or a drawdown level is
reached. For this purpose, the API provides the TesterStop function.
void TesterStop()
The function gives a command to terminate the tester, i.e., the stop will occur only after the program
returns control to the execution environment.
Calling TesterStop is considered a normal end of testing, and so this will call the OnTester function and
return all accumulated trading statistics and the value of the optimization criterion to the strategy
tester.
There is also an alternative regular way to interrupt testing: using the previously considered
ExpertRemove function. The call of ExpertRemove also returns trading statistics collected by the time
the function is called. However, there are some differences.
As a result of the ExpertRemove call, the Expert Advisor is unloaded from the agent's memory.
Therefore, if you need to run a new pass with a new set of parameters, some time will be taken to
reload the MQL program. When using TesterStop, this does not happen, and this method is preferable in
terms of performance.
On the other hand, the ExpertRemove call sets the _ IsStopped stop flag in the MQL program, which can
be used in a standard way in different parts of the program for finalizing ("cleaning up" resources). But
calling TesterStop does not set this flag, and therefore the developer may need to introduce their own
global variable to indicate early termination and handle it in a specific way.
It is important to note that TesterStop is designed to stop only one pass of the tester. 
MQL5 does not provide functions for the early termination of optimization. Therefore, for example, if
your Expert Advisor detects that the optimization has been launched on the wrong tick generation
model, and this can be detected only after the optimization has been launched (OnTesterInit does
not help here), then the TesterStop or ExpertRemove calls will interrupt new passes, but the passes
themselves will continue to be initiated, generating mass null results. We will see it in the section
Big Expert Advisor example, which will use protection from launching at open prices. 
It could be assumed that the ExpertRemove call in the Expert Advisor instance running in the
terminal and actually serving an optimization manager would stop the optimization. But this is not
the case. Even closing the chart with this Expert Advisor working in the frame mode does not stop
the optimization.
It is suggested that you try these functions in action yourself.
6.5.1 6 Big Expert Advisor example
To generalize and consolidate knowledge about the capabilities of the tester, let's consider a large
example of an Expert Advisor step by step. In this example, we will summarize the following aspects:
• Using multiple symbols, including the synchronization of bars

---

## Page 1516

Part 6. Trading automation
1 51 6
6.5 Testing and optimization of Expert Advisors
• Using an indicator from an Expert Advisor
• Using events
• Independent calculation of the main trading statistics
• Calculation of the R2 custom optimization criterion adjusted for variable lots
• Sending and processing frames with application data (trade reports broken down by symbols)
We will use MultiMartingale.mq5 as the technical base for the Expert Advisor but we will make it less
risky by switching to trading multi-currency overbought/oversold signals and increasing lots only as an
optional addition. Previously, in BandOsMA.mq5, we have already seen how to operate based on
indicator trading signals. This time we will use UseUnityPercentPro.mq5 as the signal indicator.
However, we need to modify it first. Let's call the new version UnityPercentEvent.mq5.
UnityPercentEvent.mq5
Recall the essence of the Unity indicator. It calculates the relative strength of currencies or tickers
included in a set of given instruments (it is assumed that all instruments have a common currency
through which conversion is possible). On each bar, readings are formed for all currencies: some will be
more expensive, some will be cheaper, and the two extreme elements are in a borderline state. Further
along, two essentially opposite strategies can be considered for them:
• Further breakdown (confirmation and continuation of a strong movement to the sides)
• Pullback (reversal of movement towards the center due to overbought and oversold)
To trade any of these signals, we must make a working symbol of two currencies (or tickers in
general), if there is something suitable for this combination in the Market Watch. For example, if the
upper line of the indicator belongs to EUR and the lower line belongs to USD, they correspond to the
EURUSD pair, and according to the breakout strategy we should buy it but according to the rebound
strategy, we should sell it.
In a more general case, for example, when CFDs or commodities with a common quote currency
are indicated in the indicator's basket of working instruments, it is not always possible to create a
real instrument. For such cases, it would be necessary to make the Expert Advisor more
complicated by introducing trading synthetics (compound positions), but we will not do this here
and will limit ourselves to the Forex market, where almost all cross rates are usually available.
Thus, the Expert Advisor must not only read all indicator buffers but also find out the names of
currencies, which correspond to the maximum and minimum values. And here we have a small obstacle.
MQL5 does not allow reading the names of third-party indicator buffers and in general, any line
properties other than integer ones. There are three functions for setting properties:
PlotIndexSetInteger, PlotIndexSetDouble, and PlotIndexSetString, but there is only one function for
reading them: PlotIndexGetInteger.
In theory, when MQL programs compiled into a single trading complex are created by the same
developer, this is not a big problem. In particular, we could separate a part of the indicator's source
code into a header file and include it not only in the indicator but also in the Expert Advisor. Then in the
Expert Advisor, it would be possible to repeat the analysis of the indicator's input parameters and
restore the list of currencies, completely similar to that created by the indicator. Duplicating
calculations is not very pretty, but it would work. However, a more universal solution is also required
when the indicator has a different developer, and they do not want to disclose the algorithm or plan to
change it in the future (then the compiled versions of the indicator and the Expert Advisor will become
incompatible). Such a "docking" of other people's indicators with one's own, or an Expert Advisor

---

## Page 1517

Part 6. Trading automation
1 51 7
6.5 Testing and optimization of Expert Advisors
ordered from a freelance service is a very common practice. Therefore, the indicator developer should
make it as integration-friendly as possible.
One of the possible solutions is for the indicator to send messages with the numbers and names of
buffers after initialization.
This is how it's done in the OnInit handler of the UnityPercentEvent.mq5 indicator (the code below is
shown in a shorted form since almost nothing has changed).
int OnInit()
{
   // find the common currency for all pairs
   const string common = InitSymbols();
   ...
   // set up the displayed lines in the currency cycle
   int replaceIndex = -1;
   for(int i = 0; i <= SymbolCount; i++)
   {
      string name;
      // change the order so that the base (common) currency goes under index 0,
      // the rest depends on the order in which the pairs are entered by the user
      if(i == 0)
      {
         name = common;
         if(name != workCurrencies.getKey(i))
         {
            replaceIndex = i;
         }
      }
      else
      {
         if(common == workCurrencies.getKey(i) && replaceIndex > -1)
         {
            name = workCurrencies.getKey(replaceIndex);
         }
         else
         {
            name = workCurrencies.getKey(i);
         }
      }
    
      // set up rendering of buffers
      PlotIndexSetString(i, PLOT_LABEL, name);
      ...
      // send indexes and buffer names to programs where they are needed
      EventChartCustom(0, (ushort)BarLimit, i, SymbolCount + 1, name);
   }
   ...
}
Compared to the original version, only one line has been added here. It contains the EventChartCustom
call. The input variable BarLimit is used as the identifier of the indicator copy (of which there may

---

## Page 1518

Part 6. Trading automation
1 51 8
6.5 Testing and optimization of Expert Advisors
potentially be several). Since the indicator will be called from the Expert Advisor and will not be
displayed to the user, it is enough to indicate a small positive number, at least 1 , but we will have, for
example, 1 0.
Now the indicator is ready and its signals can be used in third-party Expert Advisors. Let's start
developing the Expert Advisor UnityMartingale.mq5. To simplify the presentation, we will divide it into 4
stages, gradually adding new blocks. We will have three preliminary versions and one final version.
UnityMartingaleDraft1 .mq5
In the first stage, for the version UnityMartingaleDraft1 .mq5, let's use MultiMartingale.mq5 as the basis
and modify it.
We will rename the former input variable StartType which determined the direction of the first deal in
the series into SignalType. It will be used to choose between the considered strategies BREAKOUT and
PULLBACK.
enum SIGNAL_TYPE
{
   BREAKOUT,
   PULLBACK
};
...
input SIGNAL_TYPE StartType = 0; // SignalType
To set up the indicator, we need a separate group of input variables.
input group "U N I T Y   S E T T I N G S"
input string UnitySymbols = "EURUSD,GBPUSD,USDCHF,USDJPY,AUDUSD,USDCAD,NZDUSD";
input int UnityBarLimit = 10;
input ENUM_APPLIED_PRICE UnityPriceType = PRICE_CLOSE;
input ENUM_MA_METHOD UnityPriceMethod = MODE_EMA;
input int UnityPricePeriod = 1;
Please note that the UnitySymbols parameter contains a list of cluster instruments for building an
indicator, and usually differs from the list of working instruments that we want to trade. Traded
instruments are still set in the WorkSymbols parameter.
For example, by default, we pass a set of major Forex currency pairs to the indicator, and therefore we
can indicate as trading not only the main pairs but also any crosses. It usually makes sense to limit this
set to instruments with the best trading conditions (in particular, small or moderate spreads). In
addition, it is desirable to avoid distortions, i.e., to keep an equal amount of each currency in all pairs,
thereby statistically neutralizing the potential risks of choosing an unsuccessful direction for one of the
currencies.
Next, we wrap the indicator control in the UnityController class. In addition to the indicator handle, the
class fields store the following data:
• The number of indicator buffers, which will be received from messages from the indicator after its
initialization
• The bar number from which the data is being read (usually the current incomplete is 0, or the last
completed is 1 )
• The data array with values read from indicator buffers on the specified bar
• The last read time lastRead

---

## Page 1519

Part 6. Trading automation
1 51 9
6.5 Testing and optimization of Expert Advisors
• Flag of operation by ticks or bars tickwise
In addition, the class uses the MultiSymbolMonitor object to synchronize the bars of all involved
symbols.
class UnityController
{
   int handle;
   int buffers;
   const int bar;
   double data[];
   datetime lastRead;
   const bool tickwise;
   MultiSymbolMonitor sync;
   ...
In the constructor, which accepts all parameters for the indicator through arguments, we create the
indicator and set up the sync object.
public:
   UnityController(const string symbolList, const int offset, const int limit,
      const ENUM_APPLIED_PRICE type, const ENUM_MA_METHOD method, const int period):
      bar(offset), tickwise(!offset)
   {
      handle = iCustom(_Symbol, _Period, "MQL5Book/p6/UnityPercentEvent",
         symbolList, limit, type, method, period);
      lastRead = 0;
      
      string symbols[];
      const int n = StringSplit(symbolList, ',', symbols);
      for(int i = 0; i < n; ++i)
      {
         sync.attach(symbols[i]);
      }
   }
   
   ~UnityController()
   {
      IndicatorRelease(handle);
   }
   ...
The number of buffers is set by the attached method. We will call it upon receiving a message from the
indicator.
   void attached(const int b)
   {
      buffers = b;
      ArrayResize(data, buffers);
   }
A special method isReady returns true when the last bars of all symbols have the same time. Only in the
state of such synchronization will we get the correct values of the indicator. It should be noted that the

---

## Page 1520

Part 6. Trading automation
1 520
6.5 Testing and optimization of Expert Advisors
same schedule of trading sessions for all instruments is assumed here. If this is not the case, the timing
analysis needs to be changed.
   bool isReady()
   {
      return sync.check(true) == 0;
   }
We define the current time in different ways depending on the indicator operation mode: when
recalculating on each tick (tickwise equals true), we use the server time, and when recalculated once
per bar, we use the opening time of the last bar.
   datetime lastTime() const
   {
      return tickwise ? TimeTradeServer() : iTime(_Symbol, _Period, 0);
   }
The presence of this method will allow us to exclude reading the indicator if the current time has not
changed and, accordingly, the last read data stored in the data buffer is still relevant. And this is how
the reading of indicator buffers is organized in the read method. We only need one value of each buffer
for the bar with the bar index.
   bool read()
   {
      if(!buffers) return false;
      for(int i = 0; i < buffers; ++i)
      {
         double temp[1];
         if(CopyBuffer(handle, i, bar, 1, temp) == 1)
         {
            data[i] = temp[0];
         }
         else
         {
            return false;
         }
      }
      lastRead = lastTime();
      return true;
   }
In the end, we just save the reading time into the lastRead variable. If it is empty or not equal to the
new current time, accessing the controller data in the following methods will cause the indicator buffers
to be read using read.
The main external methods of the controller are getOuterIndices to get the indexes of the maximum
and minimum values and the operator '[]' to read the values.

---

## Page 1521

Part 6. Trading automation
1 521 
6.5 Testing and optimization of Expert Advisors
   bool isNewTime() const
   {
      return lastRead != lastTime();
   }
   
   bool getOuterIndices(int &min, int &max)
   {
      if(isNewTime())
      {
         if(!read()) return false;
      }
      max = ArrayMaximum(data);
      min = ArrayMinimum(data);
      return true;
   }
   
   double operator[](const int buffer)
   {
      if(isNewTime())
      {
         if(!read())
         {
            return EMPTY_VALUE;
         }
      }
      return data[buffer];
   }
};
Previously, the Expert Advisor BandOsMA.mq5 introduced the concept of the TradingSignal interface.
interface TradingSignal
{
   virtual int signal(void);
};
Based on it, we will describe the implementation of the signal using the UnityPercentEvent indicator.
The controller object UnityController is passed to the constructor. It also indicates the indexes of
currencies (buffers), the signals for which we want to track. We will be able to create an arbitrary set
of different signals for the selected working symbols.

---

## Page 1522

Part 6. Trading automation
1 522
6.5 Testing and optimization of Expert Advisors
class UnitySignal: public TradingSignal
{
   UnityController *controller;
   const int currency1;
   const int currency2;
   
public:
   UnitySignal(UnityController *parent, const int c1, const int c2):
      controller(parent), currency1(c1), currency2(c2) { }
   
   virtual int signal(void) override
   {
      if(!controller.isReady()) return 0; // waiting for bars synchronization
      if(!controller.isNewTime()) return 0; // waitng for time to change
      
      int min, max;
      if(!controller.getOuterIndices(min, max)) return 0;
      
      // overbought
      if(currency1 == max && currency2 == min) return +1;
      // oversold
      if(currency2 == max && currency1 == min) return -1;
      return 0;
   }
};
The signal method returns 0 in an uncertain situation and +1  or -1  in overbought and oversold states of
two specific currencies.
To formalize trading strategies, we used the TradingStrategy interface.
interface TradingStrategy
{
   virtual bool trade(void);
};
In this case, the UnityMartingale class is created on its basis, which largely coincides with
SimpleMartingale from MultiMartingale.mq5. We will only show the differences.

---

## Page 1523

Part 6. Trading automation
1 523
6.5 Testing and optimization of Expert Advisors
class UnityMartingale: public TradingStrategy
{
protected:
   ...
   AutoPtr<TradingSignal> command;
   
public:
   UnityMartingale(const Settings &state, TradingSignal *signal)
   {
      ...
      command = signal;
   }
   virtual bool trade() override
   {
      ...
      int s = command[].signal(); // get controller signal
      if(s != 0)
      {
         if(settings.startType == PULLBACK) s *= -1; // reverse logic for bounce
      }
      ulong ticket = 0;
      if(position[] == NULL) // clean start - there were (and is) no positions
      {
         if(s == +1)
         {
            ticket = openBuy(settings.lots);
         }
         else if(s == -1)
         {
            ticket = openSell(settings.lots);
         }
      }
      else
      {
         if(position[].refresh()) // position exists
         {
            if((position[].get(POSITION_TYPE) == POSITION_TYPE_BUY && s == -1)
            || (position[].get(POSITION_TYPE) == POSITION_TYPE_SELL && s == +1))
            {
               // signal in the other direction - we need to close
               PrintFormat("Opposite signal: %d for position %d %lld",
                  s, position[].get(POSITION_TYPE), position[].get(POSITION_TICKET));
               if(close(position[].get(POSITION_TICKET)))
               {
                  // position = NULL; - save the position in the cache
               }
               else
               {
                  position[].refresh(); // control possible closing errors
               }
            }

---

## Page 1524

Part 6. Trading automation
1 524
6.5 Testing and optimization of Expert Advisors
            else
            {
               // the signal is the same or absent - "trailing"
               position[].update();
               if(trailing[]) trailing[].trail();
            }
         }
         else // no position - open a new one
         {
            if(s == 0) // no signals
            {
               // here is the full logic of the old Expert Advisor:
               // - reversal for martingale loss
               // - continuation by the initial lot in a profitable direction
               ...
            }
            else // there is a signal
            {
               double lots;
               if(position[].get(POSITION_PROFIT) >= 0.0)
               {
                  lots = settings.lots; // initial lot after profit
               }
               else // increase the lot after the loss
               {
                  lots = MathFloor((position[].get(POSITION_VOLUME) * settings.factor) / lotsStep) * lotsStep;
      
                  if(lotsLimit < lots)
                  {
                     lots = settings.lots;
                  }               
               }
               
               ticket = (s == +1) ? openBuy(lots) : openSell(lots);
            }
         }
      }
   }
   ...
}
The trading part is ready. It remains to consider the initialization. An autopointer to the UnityController
object and the array with currency names are described at the global level. The pool of trading systems
is completely similar to the previous developments.

---

## Page 1525

Part 6. Trading automation
1 525
6.5 Testing and optimization of Expert Advisors
AutoPtr<TradingStrategyPool> pool;
AutoPtr<UnityController> controller;
   
int currenciesCount;
string currencies[];
In the OnInit handler, we create the UnityController object and wait for the indicator to send the
distribution of currencies by buffer indexes.
int OnInit()
{
   currenciesCount = 0;
   ArrayResize(currencies, 0);
   
   if(!StartUp(true)) return INIT_PARAMETERS_INCORRECT;
   
   const bool barwise = UnityPriceType == PRICE_CLOSE && UnityPricePeriod == 1;
   controller = new UnityController(UnitySymbols, barwise,
      UnityBarLimit, UnityPriceType, UnityPriceMethod, UnityPricePeriod);
   // waiting for messages from the indicator on currencies in buffers
   return INIT_SUCCEEDED;
}
If the price type PRICE_CLOSE and a single period are selected in the indicator input parameters, the
calculation in the controller will be performed once per bar. In all other cases, the signals will be
updated by ticks, but not more often than once per second (recall the implementation of the lastTime
method in the controller).
The helper method StartUp generally does the same thing as the old OnInit handler in the Expert
Advisor MultiMartingale. It fills the Settings structure with settings, checking them for correctness and
creating a pool of trading systems TradingStrategyPool, consisting of objects of the UnityMartingale
class for different trading symbols WorkSymbols. However, now this process is divided into two stages
due to the fact that we need to wait for information about the distribution of currencies among buffers.
Therefore, the StartUp function has an input parameter denoting a call from OnInit and later from
OnChartEvent.
When analyzing the source code of StartUp, it is important to remember that the initialization is
different for the cases when we trade only one instrument that matches the current chart and when a
basket of instruments is specified. The first mode is active when WorkSymbols is an empty line. It is
convenient for optimizing an Expert Advisor for a specific instrument. Having found the settings for
several instruments, we can combine them in WorkSymbols.

---

## Page 1526

Part 6. Trading automation
1 526
6.5 Testing and optimization of Expert Advisors
bool StartUp(const bool init = false)
{
   if(WorkSymbols == "")
   {
      Settings settings =
      {
         UseTime, HourStart, HourEnd,
         Lots, Factor, Limit,
         StopLoss, TakeProfit,
         StartType, Magic, SkipTimeOnError, Trailing, _Symbol
      };
      
      if(settings.validate())
      {
         if(init)
         {
            Print("Input settings:");
            settings.print();
         }
      }
      else
      {
         if(init) Print("Wrong settings, please fix");
         return false;
      }
      if(!init)
      {
         ...// creating a trading system based on the indicator
      }
   }
   else
   {
      Print("Parsed settings:");
      Settings settings[];
      if(!Settings::parseAll(WorkSymbols, settings))
      {
         if(init) Print("Settings are incorrect, can't start up");
         return false;
      }
      if(!init)
      {
         ...// creating a trading system based on the indicator
      }
   }
   return true;
}
The StartUp function in OnInit is called with the true parameter, which means only checking the
correctness of the settings. The creation of a trading system object is delayed until a message is
received from the indicator in OnChartEvent.

---

## Page 1527

Part 6. Trading automation
1 527
6.5 Testing and optimization of Expert Advisors
void OnChartEvent(const int id,
   const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_CUSTOM + UnityBarLimit)
   {
      PrintFormat("%lld %f '%s'", lparam, dparam, sparam);
      if(lparam == 0) ArrayResize(currencies, 0);
      currenciesCount = (int)MathRound(dparam);
      PUSH(currencies, sparam);
      if(ArraySize(currencies) == currenciesCount)
      {
         if(pool[] == NULL)
         {
            start up(); // indicator readiness confirmation
         }
         else
         {
            Alert("Repeated initialization!");
         }
      }
   }
}
Here we remember the number of currencies in the global variable currenciesCount and store them in
the currencies array, after which we call StartUp with the false parameter (default value, therefore
omitted). Messages arrive from the queue in the order in which they exist in the indicator's buffers.
Thus, we get a match between the index and the name of the currency.
When StartUp is called again, an additional code is executed:

---

## Page 1528

Part 6. Trading automation
1 528
6.5 Testing and optimization of Expert Advisors
bool StartUp(const bool init = false)
{
   if(WorkSymbols == "") // one current symbol
   {
      ...
      if(!init) // final initialization after OnInit
      {
         controller[].attached(currenciesCount);
         // split _Symbol into 2 currencies from the currencies array [] 
         int first, second;
         if(!SplitSymbolToCurrencyIndices(_Symbol, first, second))
         {
            PrintFormat("Can't find currencies (%s %s) for %s",
               (first == -1 ? "base" : ""), (second == -1 ? "profit" : ""), _Symbol);
            return false;
         }
         // create a pool from a single strategy
         pool = new TradingStrategyPool(new UnityMartingale(settings,
            new UnitySignal(controller[], first, second)));
      }
   }
   else // symbol basket
   {
      ...
      if(!init) // final initialization after OnInit
      {
         controller[].attached(currenciesCount);
      
         const int n = ArraySize(settings);
         pool = new TradingStrategyPool(n);
         for(int i = 0; i < n; i++)
         {
            ...
            // split settings[i].symbol into 2 currencies from currencies[]
            int first, second;
            if(!SplitSymbolToCurrencyIndices(settings[i].symbol, first, second))
            {
               PrintFormat("Can't find currencies (%s %s) for %s",
                  (first == -1 ? "base" : ""), (second == -1 ? "profit" : ""),
                  settings[i].symbol);
            }
            else
            {
               // add a strategy to the pool on the next trading symbol
               pool[].push(new UnityMartingale(settings[i],
                  new UnitySignal(controller[], first, second)));
            }
         }
      }
   }

---

## Page 1529

Part 6. Trading automation
1 529
6.5 Testing and optimization of Expert Advisors
The helper function SplitSymbolToCurrencyIndices selects the base currency and profit currency of the
passed symbol and finds their indexes in the currencies array. Thus, we get the reference data for
generating signals in UnitySignal objects. Each of them will have its own pair of currency indexes.
bool SplitSymbolToCurrencyIndices(const string symbol, int &first, int &second)
{
   const string s1 = SymbolInfoString(symbol, SYMBOL_CURRENCY_BASE);
   const string s2 = SymbolInfoString(symbol, SYMBOL_CURRENCY_PROFIT);
   first = second = -1;
   for(int i = 0; i < ArraySize(currencies); ++i)
   {
      if(currencies[i] == s1) first = i;
      else if(currencies[i] == s2) second = i;
   }
   
   return first != -1 && second != -1;
}
In general, the Expert Advisor is ready.
You can see that in the last examples of Expert Advisors we have classes of strategies and classes
of trading signals. We deliberately made them descendants of generic interfaces TradingStrategy
and TradingSignal in order to subsequently be able to collect collections of compatible but different
implementations that can be combined in the development of future Expert Advisors. Such unified
concrete classes should usually be separated into separate header files. In our examples, we did not
do this for the sake of simplifying the step-by-step modification. 
However, the described approach is standard for OOP. In particular, as we mentioned in the section
on creating Expert Advisor drafts, along with MetaTrader 5 comes a framework of header files with
standard classes of trading operations, signal indicators, and money management, which are used
in the MQL Wizard. Other similar solutions are published on the mql5.com site in the articles and the
Code Base section. 
You can use the ready-made class hierarchies as the basis for your projects, provided they are
suitable in terms of capabilities and ease of use.
To complete the picture, we wanted to introduce our own R2-based optimization criterion in the Expert
Advisor. To avoid the contradiction between the linear regression in the R2 calculation formula and the
variable lots that are included in our strategy, we will calculate the coefficient not for the usual balance
line but for its cumulative increments normalized by lot sizes in each trade.
To do this, in the OnTester handler, we select deals with the types DEAL_TYPE_BUY and
DEAL_TYPE_SELL and with the direction OUT. We will request all deal properties that form the financial
result (profit/loss), i.e., DEAL_PROFIT, DEAL_SWAP, DEAL_COMMISSION, DEAL_FEE, as well as their
DEAL_VOLUME volume.

---

## Page 1530

Part 6. Trading automation
1 530
6.5 Testing and optimization of Expert Advisors
#define STAT_PROPS 5 // number of requested deal properties
   
double OnTester()
{
   HistorySelect(0, LONG_MAX);
   
   const ENUM_DEAL_PROPERTY_DOUBLE props[STAT_PROPS] =
   {
      DEAL_PROFIT, DEAL_SWAP, DEAL_COMMISSION, DEAL_FEE, DEAL_VOLUME
   };
   double expenses[][STAT_PROPS];
   ulong tickets[]; // needed because of 'select' method prototype, but useful for debugging
   
   DealFilter filter;
   filter.let(DEAL_TYPE, (1 << DEAL_TYPE_BUY) | (1 << DEAL_TYPE_SELL), IS::OR_BITWISE)
      .let(DEAL_ENTRY, (1 << DEAL_ENTRY_OUT) | (1 << DEAL_ENTRY_INOUT) | (1 << DEAL_ENTRY_OUT_BY),
      IS::OR_BITWISE)
      .select(props, tickets, expenses);
   ...
Next, in the balance array, we accumulate profits/losses normalized by trading volumes and calculate
the criterion R2 for it.
   const int n = ArraySize(tickets);
   double balance[];
   ArrayResize(balance, n + 1);
   balance[0] = TesterStatistics(STAT_INITIAL_DEPOSIT);
   
   for(int i = 0; i < n; ++i)
   {
      double result = 0;
      for(int j = 0; j < STAT_PROPS - 1; ++j)
      {
         result += expenses[i][j];
      }
      result /= expenses[i][STAT_PROPS - 1]; // normalize by volume
      balance[i + 1] = result + balance[i];
   }
   const double r2 = RSquaredTest(balance);
   return r2 * 100;
}
The first version of the Expert Advisor is basically ready. We have not included the check for the tick
model using TickModel.mqh. It is assumed that the Expert Advisor will be tested when generating ticks
in the OHLC M1  mode or better. When the "open prices only" model is detected, the Expert Advisor will
send a special frame with an error status to the terminal and unload itself from the tester.
Unfortunately, this will only stop this pass, but the optimization will continue. Therefore, the copy of the
Expert Advisor that runs in the terminal issues an "alert" for the user to interrupt the optimization
manually.

---

## Page 1531

Part 6. Trading automation
1 531 
6.5 Testing and optimization of Expert Advisors
void OnTesterPass()
{
   ulong   pass;
   string  name;
   long    id;
   double  value;
   uchar   data[];
   while(FrameNext(pass, name, id, value, data))
   {
      if(name == "status" && id == 1)
      {
         Alert("Please stop optimization!");
         Alert("Tick model is incorrect: OHLC M1 or better is required");
         // it would be logical if the next call would stop all optimization,
         // but it is not
         ExpertRemove();
      }
   }
}
You can optimize SYMBOL SETTINGS parameters for any symbol and repeat the optimization for
different symbols. At the same time, the COMMON SETTINGS and UNITY SETTINGS groups should
always contain the same settings, because they apply to all symbols and instances of trading systems.
For example, Trailing must be either enabled or disabled for all optimizations. Also note that the input
variables for a single symbol (i.e. the SYMBOL SETTINGS group) have an effect only while WorkSymbols
contains an empty string. Therefore, at the optimization stage, you should keep it empty.
For example, to diversify risks, you can consistently optimize an Expert Advisor on completely
independent pairs: EURUSD, AUDJPY, GBPCHF, NZDCAD, or in other combinations. Three set files with
examples of private settings are connected to the source code.
#property tester_set "UnityMartingale-eurusd.set"
#property tester_set "UnityMartingale-gbpchf.set"
#property tester_set "UnityMartingale-audjpy.set"
In order to trade on three symbols at once, these settings should be "packed" into a common
parameter WorkSymbols:
EURUSD+0.01*1.6^5(200,200)[17,21];GBPCHF+0.01*1.2^8(600,800)[7,20];AUDJPY+0.01*1.2^8(600,800)[7,20]
This setting is also included in a separate file.
#property tester_set "UnityMartingale-combo.set"
One of the problems with the current version of the Expert Advisor is that the tester report will provide
general statistics for all symbols (more precisely, for all trading strategies, since we can include
different classes in the pool), while it would be interesting for us to monitor and evaluate each
component of the system separately.
To do this, you need to learn how to independently calculate the main financial indicators of trading, by
analogy with how the tester does it for us. We will deal with this at the second stage of the Expert
Advisor development.

---

## Page 1532

Part 6. Trading automation
1 532
6.5 Testing and optimization of Expert Advisors
UnityMartingaleDraft2.mq5
Statistics calculation might be needed quite frequently, so we will implement it in a separate header file
TradeReport.mqh, where we organize the source code into the appropriate classes.
Let's call the main class TradeReport. Many trading variables depend on balance and free margin
(equity) curves. Therefore, the class contains variables for tracking the current balance and profit, as
well as a constantly updated array with the balance history. We will not store the history of equity,
because it can change on every tick, and it is better to calculate it right on the go. We will see a little
later the reason for having the balance curve.
class TradeReport
{
   double balance;     // current balance
   double floating;    // current floating profit
   double data[];      // full balance curve - prices
   datetime moments[]; // and date/time
   ...
Changing and reading class fields is done using methods, including the constructor, in which the balance
is initialized by the ACCOUNT_BALANCE property.
   TradeReport()
   {
      balance = AccountInfoDouble(ACCOUNT_BALANCE);
   }
   
   void resetFloatingPL()
   {
      floating = 0;
   }
   
   void addFloatingPL(const double pl)
   {
      floating += pl;
   }
   
   void addBalance(const double pl)
   {
      balance += pl;
   }
   
   double getCurrent() const
   {
      return balance + floating;
   }
   ...
These methods will be needed to iteratively calculate equity drawdown (on the fly). The data balance
array will be required for a one-time calculation of the balance drawdown (we will do this at the end of
the test).
Based on the fluctuations of the curve (it does not matter, balance or equity), absolute and relative
drawdown should be calculated using the same algorithm. Therefore, this algorithm and the internal

---

## Page 1533

Part 6. Trading automation
1 533
6.5 Testing and optimization of Expert Advisors
variables necessary for it, which store intermediate states, are implemented in the nested structure
DrawDown. The below code shows its main methods and properties.
   struct DrawDown
   {
      double
      series_start,
      series_min,
      series_dd,
      series_dd_percent,
      series_dd_relative_percent,
      series_dd_relative;
      ...
      void reset();
      void calcDrawdown(const double &data[]);
      void calcDrawdown(const double amount);
      void print() const;
   };
The first calcDrawdown method calculates drawdowns when we know the entire array and this will be
used for balance. The second calcDrawdown method calculates the drawdown iteratively: each time it is
called, it is told the next value of the series, and this will be used for equity.
In addition to the drawdown, as we know, there are a large number of standard statistics for reports,
but we will support only a few of them to begin with. To do this, we describe the corresponding fields in
another nested structure, GenericStats. It is inherited from DrawDown because we still need the
drawdown in the report.
   struct GenericStats: public DrawDown
   {
      long deals;
      long trades;
      long buy_trades;
      long wins;
      long buy_wins;
      long sell_wins;
      
      double profits;
      double losses;
      double net;
      double pf;
      double average_trade;
      double recovery;
      double max_profit;
      double max_loss;
      double sharpe;
      ...
By the names of the variables, it is easy to guess what standard metrics they correspond to. Some
metrics are redundant and therefore omitted. For example, given the total number of trades (trades)
and the number of buy ones among them (buy_ trades), we can easily find the number of sell trades
(trades - sell_ trades). The same goes for complementary win/loss statistics. Winning and losing streaks
are not counted. Those who wish can supplement our report with these indicators.

---

## Page 1534

Part 6. Trading automation
1 534
6.5 Testing and optimization of Expert Advisors
For unification with the general statistics of the tester, there is the fillByTester method which fills all
fields through the TesterStatistics function. We will use it later.
      void fillByTester()
      {
         deals = (long)TesterStatistics(STAT_DEALS);
         trades = (long)TesterStatistics(STAT_TRADES);
         buy_trades = (long)TesterStatistics(STAT_LONG_TRADES);
         wins = (long)TesterStatistics(STAT_PROFIT_TRADES);
         buy_wins = (long)TesterStatistics(STAT_PROFIT_LONGTRADES);
         sell_wins = (long)TesterStatistics(STAT_PROFIT_SHORTTRADES);
         
         profits = TesterStatistics(STAT_GROSS_PROFIT);
         losses = TesterStatistics(STAT_GROSS_LOSS);
         net = TesterStatistics(STAT_PROFIT);
         pf = TesterStatistics(STAT_PROFIT_FACTOR);
         average_trade = TesterStatistics(STAT_EXPECTED_PAYOFF);
         recovery = TesterStatistics(STAT_RECOVERY_FACTOR);
         sharpe = TesterStatistics(STAT_SHARPE_RATIO);
         max_profit = TesterStatistics(STAT_MAX_PROFITTRADE);
         max_loss = TesterStatistics(STAT_MAX_LOSSTRADE);
         
         series_start = TesterStatistics(STAT_INITIAL_DEPOSIT);
         series_min = TesterStatistics(STAT_EQUITYMIN);
         series_dd = TesterStatistics(STAT_EQUITY_DD);
         series_dd_percent = TesterStatistics(STAT_EQUITYDD_PERCENT);
         series_dd_relative_percent = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
         series_dd_relative = TesterStatistics(STAT_EQUITY_DD_RELATIVE);
      }
   };
Of course, we need to implement our own calculation for those separate balances and equity of trading
systems that the tester cannot calculate. Prototypes of calcDrawdown methods have been presented
above. During operation, they fill in the last group of fields with the "series_dd" prefix. Also, the
TradeReport class contains a method for calculating the Sharpe ratio. As input, it takes a series of
numbers and a risk-free funding rate. The complete source code can be found in the attached file.
   static double calcSharpe(const double &data[], const double riskFreeRate = 0);
As you might guess, when calling this method, the relevant member array of the TradeReport class with
balances will be passed in the data parameter. The process of filling this array and calling the above
methods for specific indicators occurs in the calcStatistics method (see below). An object filter of deals
is passed to it as input (filter), initial deposit (start), and time (origin). It is assumed that the calling
code will set up the filter in such a way that only trades of the trading system we are interested in fall
under it.
The method returns a filled structure GenericStats, and in addition, it fills two arrays inside the
TradeReport object, data, and moments, with balance values and time references of changes,
respectively. We will need it in the final version of the Expert Advisor.

---

## Page 1535

Part 6. Trading automation
1 535
6.5 Testing and optimization of Expert Advisors
   GenericStats calcStatistics(DealFilter &filter,
      const double start = 0, const datetime origin = 0,
      const double riskFreeRate = 0)
   {
      GenericStats stats;
      ArrayResize(data, 0);
      ArrayResize(moments, 0);
      ulong tickets[];
      if(!filter.select(tickets)) return stats;
      
      balance = start;
      PUSH(data, balance);
      PUSH(moments, origin);
      
      for(int i = 0; i < ArraySize(tickets); ++i)
      {
         DealMonitor m(tickets[i]);
         if(m.get(DEAL_TYPE) == DEAL_TYPE_BALANCE) //deposit/withdrawal
         {
            balance += m.get(DEAL_PROFIT);
            PUSH(data, balance);
            PUSH(moments, (datetime)m.get(DEAL_TIME));
         }
         else if(m.get(DEAL_TYPE) == DEAL_TYPE_BUY 
            || m.get(DEAL_TYPE) == DEAL_TYPE_SELL)
         {
            const double profit = m.get(DEAL_PROFIT) + m.get(DEAL_SWAP)
               + m.get(DEAL_COMMISSION) + m.get(DEAL_FEE);
            balance += profit;
            
            stats.deals++;
            if(m.get(DEAL_ENTRY) == DEAL_ENTRY_OUT 
               || m.get(DEAL_ENTRY) == DEAL_ENTRY_INOUT
               || m.get(DEAL_ENTRY) == DEAL_ENTRY_OUT_BY)
            {
               PUSH(data, balance);
               PUSH(moments, (datetime)m.get(DEAL_TIME));
               stats.trades++;        // trades are counted by exit deals
               if(m.get(DEAL_TYPE) == DEAL_TYPE_SELL)
               {
                  stats.buy_trades++; // closing with a deal in the opposite direction
               }
               if(profit >= 0)
               {
                  stats.wins++;
                  if(m.get(DEAL_TYPE) == DEAL_TYPE_BUY)
                  {
                     stats.sell_wins++; // closing with a deal in the opposite direction
                  }
                  else
                  {

---

## Page 1536

Part 6. Trading automation
1 536
6.5 Testing and optimization of Expert Advisors
                     stats.buy_wins++;
                  }
               }
            }
            else if(!TU::Equal(profit, 0))
            {
               PUSH(data, balance); // entry fee (if any)
               PUSH(moments, (datetime)m.get(DEAL_TIME));
            }
            
            if(profit >= 0)
            {
               stats.profits += profit;
               stats.max_profit = fmax(profit, stats.max_profit);
            }
            else
            {
               stats.losses += profit;
               stats.max_loss = fmin(profit, stats.max_loss);
            }
         }
      }
      
      if(stats.trades > 0)
      {
         stats.net = stats.profits + stats.losses;
         stats.pf = -stats.losses > DBL_EPSILON ?
            stats.profits / -stats.losses : MathExp(10000.0); // NaN(+inf)
         stats.average_trade = stats.net / stats.trades;
         stats.sharpe = calcSharpe(data, riskFreeRate);
         stats.calcDrawdown(data);     // fill in all fields of the DrawDown substructure
         stats.recovery = stats.series_dd > DBL_EPSILON ?
            stats.net / stats.series_dd : MathExp(10000.0);
      }
      return stats;
   }
};
Here you can see how we call calcSharpe and calcDrawdown to get the corresponding indicators on the
array data. The remaining indicators are calculated directly in the loop inside calcStatistics.
The TradeReport class is ready, and we can expand the functionality of the Expert Advisor to the
version UnityMartingaleDraft2.mq5.
Let's add new members to the UnityMartingale class.

---

## Page 1537

Part 6. Trading automation
1 537
6.5 Testing and optimization of Expert Advisors
class UnityMartingale: public TradingStrategy
{
protected:
   ...
   TradeReport report;
   TradeReport::DrawDown equity;
   const double deposit;
   const datetime epoch;
   ...
We need the report object in order to call calcStatistics, where the balance drawdown will be included.
The equity object is required for an independent calculation of equity drawdown. The initial balance and
date, as well as the beginning of the equity drawdown calculation, are set in the constructor.
public:
   UnityMartingale(const Settings &state, TradingSignal *signal):
      symbol(state.symbol), deposit(AccountInfoDouble(ACCOUNT_BALANCE)),
      epoch(TimeCurrent())
   {
      ...
      equity.calcDrawdown(deposit);
      ...
   }
Continuation of the calculation of drawdown by equity is done on the go, with each call to the trade
method.
   virtual bool trade() override
   {
      ...
      if(MQLInfoInteger(MQL_TESTER))
      {
         if(position[])
         {
            report.resetFloatingPL();
            // after reset, sum all floating profits
            // why we call addFloatingPL for each existing position,
            // but this strategy has a maximum of 1 position at a time
            report.addFloatingPL(position[].get(POSITION_PROFIT)
               + position[].get(POSITION_SWAP));
            // after taking into account all the amounts - update the drawdown
            equity.calcDrawdown(report.getCurrent());
         }
      }
      ...
   }
This is not all that is needed for a correct calculation. We should take into account the floating profit or
loss on top of the balance. The above code part only shows the addFloatingPL call, but the TradeReport
class has also a method for modifying the balance: addBalance. However, the balance changes only
when the position is closed.

---

## Page 1538

Part 6. Trading automation
1 538
6.5 Testing and optimization of Expert Advisors
Thanks to the OOP concept, closing a position in our situation corresponds to deleting the position
object of the PositionState class. So why can't we intercept it?
The PositionState class does not provide any means for this, but we can declare a derived class
PositionStateWithEquity with a special constructor and destructor.
When creating an object, not only the position identifier is passed to the constructor, but also a pointer
to the report object to which information will need to be sent.
class PositionStateWithEquity: public PositionState
{
   TradeReport *report;
   
public:
   PositionStateWithEquity(const long t, TradeReport *r):
      PositionState(t), report(r) { }
   ...
In the destructor, we find all deals by the closed position ID, calculate the total financial result
(together with commissions and other deductions), and then call addBalance for related the report
object.
   ~PositionStateWithEquity()
   {
      if(HistorySelectByPosition(get(POSITION_IDENTIFIER)))
      {
         double result = 0;
         DealFilter filter;
         int props[] = {DEAL_PROFIT, DEAL_SWAP, DEAL_COMMISSION, DEAL_FEE};
         Tuple4<double, double, double, double> overheads[];
         if(filter.select(props, overheads))
         {
            for(int i = 0; i < ArraySize(overheads); ++i)
            {
               result += NormalizeDouble(overheads[i]._1, 2)
                  + NormalizeDouble(overheads[i]._2, 2)
                  + NormalizeDouble(overheads[i]._3, 2)
                  + NormalizeDouble(overheads[i]._4, 2);
            }
         }
         if(CheckPointer(report) != POINTER_INVALID) report.addBalance(result);
      }
   }
};
It remains to clarify one point – how to create PositionStateWithEquity class objects for positions
instead of PositionState. To do this, it is enough to change the new operator in a couple of places
where it is called in the TradingStrategy class.

---

## Page 1539

Part 6. Trading automation
1 539
6.5 Testing and optimization of Expert Advisors
   position = MQLInfoInteger(MQL_TESTER) ?
      new PositionStateWithEquity(tickets[0], &report) : new PositionState(tickets[0]); 
Thus, we have implemented the collection of data. Now we need to directly generate a report, that is,
to call calcStatistics. Here we need to expand our TradingStrategy interface: we add the statement
method to it.
interface TradingStrategy
{
   virtual bool trade(void);
   virtual bool statement();
};
Then, in this current implementation, intended for our strategy, we will be able to bring the work to its
logical conclusion.
class UnityMartingale: public TradingStrategy
{
   ...
   virtual bool statement() override
   {
      if(MQLInfoInteger(MQL_TESTER))
      {
         Print("Separate trade report for ", settings.symbol);
         // equity drawdown should already be calculated on the fly
         Print("Equity DD:");
         equity.print();
         
         // balance drawdown is calculated in the resulting report
         Print("Trade Statistics (with Balance DD):");
         // configure the filter for a specific strategy
         DealFilter filter;
         filter.let(DEAL_SYMBOL, settings.symbol)
            .let(DEAL_MAGIC, settings.magic, IS::EQUAL_OR_ZERO);
           // zero "magic" number is needed for the last exit deal
           // - it is done by the tester itself
         HistorySelect(0, LONG_MAX);
         TradeReport::GenericStats stats =
            report.calcStatistics(filter, deposit, epoch);
         stats.print();
      }
      return false;
   }
   ...
The new method will simply print out all the calculated indicators in the log. By forwarding the same
method through the pool of trading systems TradingStrategyPool, let's request separate reports for all
symbols from the handler OnTester.

---

## Page 1540

Part 6. Trading automation
1 540
6.5 Testing and optimization of Expert Advisors
double OnTester()
{
   ...
   if(pool[] != NULL)
   {
      pool[].statement(); // ask all trading systems to display their results
   }
   ...
}
Let's check the correctness of our report. To do this, let's run the Expert Advisor in the tester, one
symbol at a time, and compare the standard report with our calculations. For example, to set up
UnityMartingale-eurusd.set, trading on EURUSD H1  we will get such indicators for 2021 .
Tester report for 2021, EURUSD H1
In the log, our version is displayed as two structures: DrawDown with equity drawdown and
GenericStats with balance drawdown indicators and other statistics.

---

## Page 1541

Part 6. Trading automation
1 541 
6.5 Testing and optimization of Expert Advisors
Separate trade report for EURUSD
Equity DD:
    [maxpeak] [minpeak] [series_start] [series_min] [series_dd] [series_dd_percent] »
[0]  10022.48  10017.03       10000.00      9998.20        6.23                0.06 »
» [series_dd_relative_percent] [series_dd_relative]
»                         0.06                 6.23
Trade Statistics (with Balance DD):
    [maxpeak] [minpeak] [series_start] [series_min] [series_dd] [series_dd_percent] »
[0]  10022.40  10017.63       10000.00      9998.51        5.73                0.06 »
» [series_dd_relative_percent] [series_dd_relative] »
»                         0.06                 5.73 »
» [deals] [trades] [buy_trades] [wins] [buy_wins] [sell_wins] [profits] [losses] [net] [pf] »
»     194       97           43     42         19          23     57.97   -39.62 18.35 1.46 »
» [average_trade] [recovery] [max_profit] [max_loss] [sharpe]
»            0.19       3.20         2.00      -2.01     0.15
It is easy to verify that these numbers match with the tester's report.
Now let's start trading on the same period for three symbols at once (setting UnityMartingale-
combo.set).
In addition to EURUSD entries, structures for GBPCHF and AUDJPY will appear in the journal.

---

## Page 1542

Part 6. Trading automation
1 542
6.5 Testing and optimization of Expert Advisors
Separate trade report for GBPCHF
Equity DD:
    [maxpeak] [minpeak] [series_start] [series_min] [series_dd] [series_dd_percent] »
[0]  10029.50  10000.19       10000.00      9963.65       62.90                0.63 »
» [series_dd_relative_percent] [series_dd_relative]
»                         0.63                62.90
Trade Statistics (with Balance DD):
    [maxpeak] [minpeak] [series_start] [series_min] [series_dd] [series_dd_percent] »
[0]  10023.68   9964.28       10000.00      9964.28       59.40                0.59 »
» [series_dd_relative_percent] [series_dd_relative] »
»                         0.59                59.40 »
» [deals] [trades] [buy_trades] [wins] [buy_wins] [sell_wins] [profits] [losses] [net] [pf] »
»     600      300          154    141         63          78    394.53  -389.33  5.20 1.01 »
» [average_trade] [recovery] [max_profit] [max_loss] [sharpe]
»            0.02       0.09         9.10      -6.73     0.01
Separate trade report for AUDJPY
Equity DD:
    [maxpeak] [minpeak] [series_start] [series_min] [series_dd] [series_dd_percent] »
[0]  10047.14  10041.53       10000.00      9961.62       48.20                0.48 »
» [series_dd_relative_percent] [series_dd_relative]
»                         0.48                48.20
Trade Statistics (with Balance DD):
    [maxpeak] [minpeak] [series_start] [series_min] [series_dd] [series_dd_percent] »
[0]  10045.21  10042.75       10000.00      9963.62       44.21                0.44 »
» [series_dd_relative_percent] [series_dd_relative] »
»                         0.44                44.21 »
» [deals] [trades] [buy_trades] [wins] [buy_wins] [sell_wins] [profits] [losses] [net] [pf] »
»     332      166           91     89         54          35    214.79  -170.20 44.59 1.26 »
» [average_trade] [recovery] [max_profit] [max_loss] [sharpe]
»            0.27       1.01         7.58      -5.17     0.09
The tester report in this case will contain generalized data, so thanks to our classes, we have received
previously inaccessible details.
However, looking at a pseudo-report in a log is not very convenient. Moreover, I would like to see a
graphic representation of the balance line at the very least as its appearance often says more about
the suitability of the system than dry statistics.
Let's improve the Expert Advisor by giving it the ability to generate visual reports in HTML format: after
all, the tester's reports can also be exported to HTML, saved, and compared over time. In addition, in
the future, such reports can be transmitted in frames to the terminal right during optimization, and the
user will be able to start studying the reports of specific passes even before the completion of the
entire process.
This will be the penultimate version of the example UnityMartingaleDraft3.mq5.
UnityMartingaleDraft3.mq5
Visualization of the trading report includes a balance line and a table with statistical indicators. We will
not generate a complete report similar to the tester's report but will limit ourselves to the selected
most important values. Our purpose is to implement a working mechanism that can then be customized
in accordance with personal requirements.

---

## Page 1543

Part 6. Trading automation
1 543
6.5 Testing and optimization of Expert Advisors
We will arrange the basis of the algorithm in the form of the TradeReportWriter class
(TradeReportWriter.mqh). The class will be able to store an arbitrary number of reports from different
trading systems: each in a separate object DataHolder, which includes arrays of balance values and
timestamps (data and when, respectively), the stats structure with statistics, as well as the title, color,
and width of the line to display.
class TradeReportWriter
{
protected:
   class DataHolder
   {
   public:
      double data[];                   // balance changes
      datetime when[];                 // balance timestamps
      string name;                     // description
      color clr;                       // color
      int width;                       // line width
      TradeReport::GenericStats stats; // trading indicators
   };
   ...
We have an array of autopointers curves allocated for the objects of the DataHolder class. In addition,
we will need common limits on amounts and terms to match the lines of all trading systems in the
picture. This will be provided by the variables lower, upper, start, and stop.
   AutoPtr<DataHolder> curves[];
   double lower, upper;
   datetime start, stop;
   
public:
   TradeReportWriter(): lower(DBL_MAX), upper(-DBL_MAX), start(0), stop(0) { }
   ...
The addCurve method adds a balance line.

---

## Page 1544

Part 6. Trading automation
1 544
6.5 Testing and optimization of Expert Advisors
   virtual bool addCurve(double &data[], datetime &when[], const string name,
      const color clr = clrNONE, const int width = 1)
   {
      if(ArraySize(data) == 0 || ArraySize(when) == 0) return false;
      if(ArraySize(data) != ArraySize(when)) return false;
      DataHolder *c = new DataHolder();
      if(!ArraySwap(data, c.data) || !ArraySwap(when, c.when))
      {
         delete c;
         return false;
      }
   
      const double max = c.data[ArrayMaximum(c.data)];
      const double min = c.data[ArrayMinimum(c.data)];
      
      lower = fmin(min, lower);
      upper = fmax(max, upper);
      if(start == 0) start = c.when[0];
      else if(c.when[0] != 0) start = fmin(c.when[0], start);
      stop = fmax(c.when[ArraySize(c.when) - 1], stop);
      
      c.name = name;
      c.clr = clr;
      c.width = width;
      ZeroMemory(c.stats); // no statistics by default
      PUSH(curves, c);
      return true;
   }
The second version of the addCurve method adds not only a balance line but also a set of financial
variables in the GenericStats structure.
   virtual bool addCurve(TradeReport::GenericStats &stats,
      double &data[], datetime &when[], const string name,
      const color clr = clrNONE, const int width = 1)
   {
      if(addCurve(data, when, name, clr, width))
      {
         curves[ArraySize(curves) - 1][].stats = stats;
         return true;
      }
      return false;
   }
The most important class method which visualizes the report is made abstract.
   virtual void render() = 0;
This makes it possible to implement many ways of displaying reports, for example, both, with recording
to files of different formats, and with drawing directly on the chart. We will now restrict ourselves to the
formation of HTML files since this is the most technologically advanced and widespread method.

---

## Page 1545

Part 6. Trading automation
1 545
6.5 Testing and optimization of Expert Advisors
The new class HTMLReportWriter has a constructor, the parameters of which specify the name of the
file, as well as the size of the picture with balance curves. We will generate the image itself in the well-
known SVG vector graphics format: it is ideal in this case since it is a subset of the XML language,
which is HTML itself.
class HTMLReportWriter: public TradeReportWriter
{
   int handle;
   int width, height;
   
public:
   HTMLReportWriter(const string name, const int w = 600, const int h = 400):
      width(w), height(h)
   {
      handle = FileOpen(name,
         FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_REWRITE);
   }
   
   ~HTMLReportWriter()
   {
      if(handle != 0) FileClose(handle);
   }
   
   void close()
   {
      if(handle != 0) FileClose(handle);
      handle = 0;
   }
   ...
Before turning to the main public render method, it is necessary to introduce the reader to one
technology, which will be described in detail in the final Part 7 of the book. We are talking about
resources: files and arrays of arbitrary data connected to an MQL program for working with multimedia
(sound and images), embedding compiled indicators, or simply as a repository of application
information. It is the latter option that we will now use.
The point is that it is better to generate an HTML page not entirely from MQL code, but based on a
template (page template), into which the MQL code will only insert the values of some variables. This is
a well-known technique in programming that allows you to separate the algorithm and the external
representation of the program (or the result of its work). Due to this, we can separately experiment
with the HTML template and MQL code, working with each of the components in a familiar environment.
Specifically, MetaEditor is still not very suitable for editing web pages and viewing them, just like a
standard browser does not know anything about MQL5 (although this can be fixed).
We will store HTML report templates in text files connected to the MQL5 source code as resources.
The connection is made using a special directive #resource. For example, there is the following line in
the file TradeReportWriter.mqh.
#resource "TradeReportPage.htm" as string ReportPageTemplate
It means that next to the source code there should be the file TradeReportPage.htm, which will become
available in the MQL code as a string ReportPageTemplate. By extension, you can understand that the
file is a web page. Here are the contents of this file with abbreviations (we do not have the task of

---

## Page 1546

Part 6. Trading automation
1 546
6.5 Testing and optimization of Expert Advisors
teaching the reader about web development, although, apparently, knowledge in this area can be useful
for a trader as well). Indents are added to visually represent the nesting hierarchy of HTML tags; there
are no indents in the file.
<!DOCTYPE html>
<html>
   <head>
      <title>Trade Report</title>
      <style>
         *{font: 9pt "Segoe UI";}
         .center{width:fit-content;margin:0 auto;}
         ...
      </style>
   </head>
   <body>
      <div class="center">
         <h1>Trade Report</h1>
         ~
      </div>
   </body>
   <script>
   ...
   </script>
</html>
The basics of the templates are chosen by the developer. There are a large number of ready-made
HTML template systems, but they provide a lot of redundant features and are therefore too complex for
our example. We will develop our own concept.
To begin with, let's note that most web pages have an initial part (header), a final part (footer), and
useful information is located between them. The above draft report is no exception in this sense. It uses
the tilde character '~' to indicate useful content. Instead, the MQL code will have to insert a balance
image and a table with indicators. But the presence of '~' is not necessary, since the page can be a
single whole, that is, the very useful middle part: after all, the MQL code can, if necessary, insert the
result of processing one template into another.
To complete the digression regarding HTML templates, let's pay attention to one more thing. In
theory, a web page consists of tags that perform essentially different functions. Standard HTML
tags tell the browser what to display. In addition to them, there are cascading styles (CSS), which
describe how to display it. Finally, the page can have a dynamic component in the form of
JavaScript scripts that interactively control both the first and second. 
Usually, these three components are templatized independently, i.e., for example, an HTML
template, strictly speaking, should contain only HTML but not CSS or JavaScript. This allows
"unbinding" the content, appearance, and behavior of the web page, which facilitates
development (it is recommended to follow the same approach in MQL5!). 
However, in our example, we have included all the components in the template. In particular, in the
above template, we see the tag <style> with styles CSS and tag <script> with some JavaScript
functions, which are omitted. This is done to simplify the example, with an emphasis on MQL5
features rather than web development.

---

## Page 1547

Part 6. Trading automation
1 547
6.5 Testing and optimization of Expert Advisors
Having a web page template in the ReportPageTemplate variable connected as a resource, we can write
the render method.
   virtual void render() override
   {
      string headerAndFooter[2];
      StringSplit(ReportPageTemplate, '~', headerAndFooter);
      FileWriteString(handle, headerAndFooter[0]);
      renderContent();
      FileWriteString(handle, headerAndFooter[1]);
   }
   ...
It actually splits the page into upper and lower halves by the '~' character, displays them as is, and
calls a helper method renderContent between them.
We have already described that the report will consist of a general picture with balance curves and
tables with indicators of trading systems, so the implementation renderContent is natural.
private:
   void renderContent()
   {
      renderSVG();
      renderTables();
   }
Image generation inside renderSVG is based on yet another template file TradeReportSVG.htm, which
binds to a string variable SVGBoxTemplate:
#resource "TradeReportSVG.htm" as string SVGBoxTemplate
The content of this template is the last one we list here. Those who wish can look into the source codes
of the rest of the templates themselves.
<span id="params" style="display:block;width:%WIDTH%px;text-align:center;"></span>
<a id="main" style="display:block;text-align:center;">
   <svg width="%WIDTH%" height="%HEIGHT%" xmlns="http://www.w3.org/2000/svg">
      <style>.legend {font: bold 11px Consolas;}</style>
      <rect x="0" y="0" width="%WIDTH%" height="%HEIGHT%"
         style="fill:none; stroke-width:1; stroke: black;"/>
      ~
   </svg>
</a>
In the code of the renderSVG method, we'll see the familiar trick of splitting the content into two
blocks "before" and "after" the tilde, but there's something new here.

---

## Page 1548

Part 6. Trading automation
1 548
6.5 Testing and optimization of Expert Advisors
   void renderSVG()
   {
      string headerAndFooter[2];
      if(StringSplit(SVGBoxTemplate, '~', headerAndFooter) != 2) return;
      StringReplace(headerAndFooter[0], "%WIDTH%", (string)width);
      StringReplace(headerAndFooter[0], "%HEIGHT%", (string)height);
      FileWriteString(handle, headerAndFooter[0]);
      
      for(int i = 0; i < ArraySize(curves); ++i)      
      {
         renderCurve(i, curves[i][].data, curves[i][].when,
            curves[i][].name, curves[i][].clr, curves[i][].width);
      }
      
      FileWriteString(handle, headerAndFooter[1]);
   }
At the top of the page, in the string headerAndFooter[0], we are looking for substrings of the special
form "%WIDTH%" and "%HEIGHT%", and replacing them with the required width and height of the
image. It is by this principle that value substitution works in our templates. For example, in this
template, these substrings actually occur in the rect tag:
<rect x="0" y="0" width="%WIDTH%" height="%HEIGHT%" style="fill:none; stroke-width:1; stroke: black;"/>
Thus, if the report is ordered with a size of 600 by 400, the line will be converted to the following:
<rect x="0" y="0" width="600" height="400" style="fill:none; stroke-width:1; stroke: black;"/>
This will display a 1 -pixel thick black border of the specified dimensions in the browser.
The generation of tags for drawing specific balance lines is handled by the renderCurve method, to
which we pass all the necessary arrays and other settings (name, color, and thickness). We will leave
this method and other highly specialized methods (renderTables, renderTable) for independent study.
Let's return to the main module of the UnityMartingaleDraft3.mq5 Expert Advisor. Set the size of the
image of the balance graphs and connect TradeReportWriter.mqh.
#define MINIWIDTH  400
#define MINIHEIGHT 200
   
#include <MQL5Book/TradeReportWriter.mqh>
In order to "connect" the strategies with the report builder, you will need to modify the statement
method in the TradingStrategy interface: pass a pointer to the TradeReportWriter object, which the
calling code can create and configure.
interface TradingStrategy
{
   virtual bool trade(void);
   virtual bool statement(TradeReportWriter *writer = NULL);
};
Now let's add some lines in the specific implementation of this method in our UnityMartingale strategy
class.

---

## Page 1549

Part 6. Trading automation
1 549
6.5 Testing and optimization of Expert Advisors
class UnityMartingale: public TradingStrategy
{
   ...
   TradeReport report;
   ...
   virtual bool statement(TradeReportWriter *writer = NULL) override
   {
      if(MQLInfoInteger(MQL_TESTER))
      {
         ...
         // it's already been done
         DealFilter filter;
         filter.let(DEAL_SYMBOL, settings.symbol)
            .let(DEAL_MAGIC, settings.magic, IS::EQUAL_OR_ZERO);
         HistorySelect(0, LONG_MAX);
         TradeReport::GenericStats stats =
            report.calcStatistics(filter, deposit, epoch);
         ...
         // adding this
         if(CheckPointer(writer) != POINTER_INVALID)
         {
            double data[];               // balance values
            datetime time[];             // balance points time to synchronize curves
            report.getCurve(data, time); // fill in the arrays and transfer to write to the file
            return writer.addCurve(stats, data, time, settings.symbol);
         }
         return true;
      }
      return false;
   }
It all comes down to getting an array of balance and a structure with indicators from the report object
(class TradeReport) and passing to the TradeReportWriter object, calling addCurve.
Of course, the pool of trading strategies ensures the transfer of the same object TradeReportWriter to
all strategies to generate a combined report.
class TradingStrategyPool: public TradingStrategy
{
   ...
   virtual bool statement(TradeReportWriter *writer = NULL) override
   {
      bool result = false;
      for(int i = 0; i < ArraySize(pool); i++)
      {
         result = pool[i][].statement(writer) || result;
      }
      return result;
   }
Finally, the OnTester handler has undergone the largest modification. The following lines would suffice
to generate an HTML report of trading strategies.

---

## Page 1550

Part 6. Trading automation
1 550
6.5 Testing and optimization of Expert Advisors
double OnTester()
{
   ...
   const static string tempfile = "temp.html";
   HTMLReportWriter writer(tempfile, MINIWIDTH, MINIHEIGHT);
   if(pool[] != NULL)
   {
      pool[].statement(&writer); // ask strategies to report their results
   }
   writer.render(); // write the received data to a file
   writer.close();
}
However, for clarity and user convenience, it would be great to add to the report a general balance
curve, as well as a table with general indicators. It makes sense to output them only when several
symbols are specified in the Expert Advisor settings because otherwise, the report of one strategy
coincides with the general one in the file.
This required a little more code.

---

## Page 1551

Part 6. Trading automation
1 551 
6.5 Testing and optimization of Expert Advisors
double OnTester()
{
   ...
   // had it before
   DealFilter filter;
   // set up the filter and fill in the array of deals based on it tickets
   ...
   const int n = ArraySize(tickets);
   
   // add this
   const bool singleSymbol = WorkSymbols == "";
   double curve[];    // total balance curve
   datetime stamps[]; // date and time of total balance points
   
   if(!singleSymbol) // the total balance is displayed only if there are several symbols/strategies
   {
      ArrayResize(curve, n + 1);
      ArrayResize(stamps, n + 1);
      curve[0] = TesterStatistics(STAT_INITIAL_DEPOSIT);
      
      // MQL5 does not allow to know the test start time,
      // this could be found out from the first transaction,
      // but it is outside the filter conditions of a specific system,
      // so let's just agree to skip time 0 in calculations
      stamps[0] = 0;
   }
   
   for(int i = 0; i < n; ++i) // deal cycle
   {
      double result = 0;
      for(int j = 0; j < STAT_PROPS - 1; ++j)
      {
         result += expenses[i][j];
      }
      if(!singleSymbol)
      {
         curve[i + 1] = result + curve[i];
         stamps[i + 1] = (datetime)HistoryDealGetInteger(tickets[i], DEAL_TIME);
      }
      ...
   }
   if(!singleSymbol) // send the tester's statistics and the overall curve to the report 
   {
      TradeReport::GenericStats stats;
      stats.fillByTester();
      writer.addCurve(stats, curve, stamps, "Overall", clrBlack, 3);
   }
   ...
}

---

## Page 1552

Part 6. Trading automation
1 552
6.5 Testing and optimization of Expert Advisors
Let's see what we got. If we run the Expert Advisor with settings UnityMartingale-combo.set, we will
have the temp.html file in the MQL5/Files folder of one of the agents. Here's what it looks like in the
browser.
HTML report for Expert Advisor with multiple trading strategies/symbols
Now that we know how to generate reports on one test pass, we can send them to the terminal during
optimization, select the best ones on the go, and present them to the user before the end of the whole
process. All reports will be put in a separate folder inside MQL5/Files of the terminal. The folder will

---

## Page 1553

Part 6. Trading automation
1 553
6.5 Testing and optimization of Expert Advisors
receive a name containing the symbol and timeframe from the tester's settings, as well as the name of
the Expert Advisor.
UnityMartingale.mq5
As we know, to send a file to the terminal, it is enough to call the function FrameAdd. We have already
generated the file within the framework of the previous version.
double OnTester()
{
   ...
   if(MQLInfoInteger(MQL_OPTIMIZATION))
   {
      FrameAdd(tempfile, 0, r2 * 100, tempfile);
   }
}
In the receiving Expert Advisor instance, we will perform the necessary preparation. Let's describe the
structure Pass with the main parameters of each optimization pass.
struct Pass
{
   ulong id;          // pass number
   double value;      // optimization criterion value
   string parameters; // optimized parameters as list 'name=value'
   string preset;     // text to generate set-file (with all parameters)
};
In the parameters strings, "name=value" pairs are connected with the '&' symbol. This will be useful
for the interaction of web pages of reports in the future (the '&' symbol is the standard for combining
parameters in web addresses). We did not describe the format of set files, but the following source
code that forms the preset string allows you to study this issue in practice.
As frames arrive, we will write improvements according to the optimization criterion to the TopPasses
array. The current best pass will always be the last pass in the array and is also available in the
BestPass variable.
Pass TopPasses[];     // stack of constantly improving passes (last one is best)
Pass BestPass;        // current best pass
string ReportPath;    // dedicated folder for all html files of this optimization
In the handler OnTesterInit let's create a folder name.
void OnTesterInit()
{
   BestPass.value = -DBL_MAX;
   ReportPath = _Symbol + "-" + PeriodToString(_Period) + "-"
      + MQLInfoString(MQL_PROGRAM_NAME) + "/";
}
In the OnTesterPass handler, we will sequentially select only those frames in which the indicator has
improved, find for them the values of optimized and other parameters, and add all this information to
the Pass array of structures.

---

## Page 1554

Part 6. Trading automation
1 554
6.5 Testing and optimization of Expert Advisors
void OnTesterPass()
{
   ulong   pass;
   string  name;
   long    id;
   double  value;
   uchar   data[];
   
   // input parameters for the pass corresponding to the current frame
   string  params[];
   uint    count;
   
   while(FrameNext(pass, name, id, value, data))
   {
      // collect passes with improved stats
      if(value > BestPass.value && FrameInputs(pass, params, count))
      {
         BestPass.preset = "";
         BestPass.parameters = "";
         // get optimized and other parameters for generating a set-file
         for(uint i = 0; i < count; i++)
         {
            string name2value[];
            int n = StringSplit(params[i], '=', name2value);
            if(n == 2)
            {
               long pvalue, pstart, pstep, pstop;
               bool enabled = false;
               if(ParameterGetRange(name2value[0], enabled, pvalue, pstart, pstep, pstop))
               {
                  if(enabled)
                  {
                     if(StringLen(BestPass.parameters)) BestPass.parameters += "&";
                     BestPass.parameters += params[i];
                  }
                  
                  BestPass.preset += params[i] + "||" + (string)pstart + "||"
                    + (string)pstep + "||" + (string)pstop + "||"
                    + (enabled ? "Y" : "N") + "<br>\n";
               }
               else
               {
                  BestPass.preset += params[i] + "<br>\n";
               }
            }
         }
      
         BestPass.value = value;
         BestPass.id = pass;
         PUSH(TopPasses, BestPass);
         // write the frame with the report to the HTML file

---

## Page 1555

Part 6. Trading automation
1 555
6.5 Testing and optimization of Expert Advisors
         const string text = CharArrayToString(data);
         int handle = FileOpen(StringFormat(ReportPath + "%06.3f-%lld.htm", value, pass),
            FILE_WRITE | FILE_TXT | FILE_ANSI);
         FileWriteString(handle, text);
         FileClose(handle);
      }
   }
}
The resulting reports with improvements are saved in files with names that include the value of the
optimization criterion and the pass number.
Now comes the most interesting. In the OnTesterDeinit handler, we can form a common HTML file
(overall.htm), which allows you to see all the reports at once (or, say, the top 1 00). It uses the same
scheme with templates that we covered earlier.
#resource "OptReportPage.htm" as string OptReportPageTemplate
#resource "OptReportElement.htm" as string OptReportElementTemplate
   
void OnTesterDeinit()
{
   int handle = FileOpen(ReportPath + "overall.htm",
      FILE_WRITE | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
   string headerAndFooter[2];
   StringSplit(OptReportPageTemplate, '~', headerAndFooter);
   StringReplace(headerAndFooter[0], "%MINIWIDTH%", (string)MINIWIDTH);
   StringReplace(headerAndFooter[0], "%MINIHEIGHT%", (string)MINIHEIGHT);
   FileWriteString(handle, headerAndFooter[0]);
   // read no more than 100 best records from TopPasses
   for(int i = ArraySize(TopPasses) - 1, k = 0; i >= 0 && k < 100; --i, ++k)
   {
      string p = TopPasses[i].parameters;
      StringReplace(p, "&", " ");
      const string filename = StringFormat("%06.3f-%lld.htm",
         TopPasses[i].value, TopPasses[i].id);
      string element = OptReportElementTemplate;
      StringReplace(element, "%FILENAME%", filename);
      StringReplace(element, "%PARAMETERS%", TopPasses[i].parameters);
      StringReplace(element, "%PARAMETERS_SPACED%", p);
      StringReplace(element, "%PASS%", IntegerToString(TopPasses[i].id));
      StringReplace(element, "%PRESET%", TopPasses[i].preset);
      StringReplace(element, "%MINIWIDTH%", (string)MINIWIDTH);
      StringReplace(element, "%MINIHEIGHT%", (string)MINIHEIGHT);
      FileWriteString(handle, element);
   }
   FileWriteString(handle, headerAndFooter[1]);
   FileClose(handle);
}
The following image shows what the overview web page looks like after optimizing UnityMartingale.mq5
by UnityPricePeriod parameter in multicurrency mode.

---

## Page 1556

Part 6. Trading automation
1 556
6.5 Testing and optimization of Expert Advisors
Overview web page with trading reports of the best optimization passes
For each report, we display only the upper part, where the balance chart falls. This part is the most
convenient to get an estimate by just looking at it.
Lists of optimized parameters ("name=value&name=value...") are displayed above each graph.
Pressing on a line opens a block with the text for the set file of all the settings of this pass. If you click
inside a block, its contents will be copied to the clipboard. It can be saved in a text editor and thus get
a ready-made set file.
Clicking on the chart will take you to the specific report page, along with the scorecards (given above).
At the end of the section, we touch on one more question. Earlier we promised to demonstrate the
effect of the TesterHideIndicators function. The UnityMartingale.mq5 Expert Advisor currently uses the
UnityPercentEvent.mq5 indicator. After any test, the indicator is displayed on the opening chart. Let's
suppose that we want to hide from the user the mechanism of the Expert Advisor's work and from
where it takes signals. Then you can call the function TesterHideIndicators (with the true parameter) in
the handler OnInit, before creating the object UnityController, in which the descriptor is received
through iCustom.

---

## Page 1557

Part 6. Trading automation
1 557
6.5 Testing and optimization of Expert Advisors
int OnInit()
{
   ...
   TesterHideIndicators(true);
   ...
   controller = new UnityController(UnitySymbols, barwise,
      UnityBarLimit, UnityPriceType, UnityPriceMethod, UnityPricePeriod);
   return INIT_SUCCEEDED;
}
This version of the Expert Advisor will no longer display the indicator on the chart. However, it is not
very well hidden. If we look into the tester's log, we will see lines about loaded programs among a lot of
useful information: first, a message about loading the Expert Advisor itself, and a little later, about
loading the indicator.
...
expert file added: Experts\MQL5Book\p6\UnityMartingale.ex5.
...
program file added: \Indicators\MQL5Book\p6\UnityPercentEvent.ex5. 
...
Thus, a meticulous user can find out the name of the indicator. This possibility can be eliminated by the
resource mechanism, which we have already mentioned in passing in the context of web page blanks. It
turns out that the compiled indicator can also be embedded into an MQL program (in an Expert Advisor
or another indicator) as a resource. And such resource programs are no longer mentioned in the
tester's log. We will study the resources in detail in the 7th Part of the book, and now we will show the
lines associated with them in the final version of our Expert Advisor.
First of all, let's describe the resource with the #resource indicator directive. In fact, it simply contains
the path to the compiled indicator file (obviously, it must already be compiled beforehand), and here it
is mandatory to use double backslashes as delimiters as forward single slashes in resource paths are
not supported.
#resource "\\Indicators\\MQL5Book\\p6\\UnityPercentEvent.ex5"
Then, in the lines with the iCustom call, we replace the previous operator:
   UnityController(const string symbolList, const int offset, const int limit,
      const ENUM_APPLIED_PRICE type, const ENUM_MA_METHOD method, const int period):
      bar(offset), tickwise(!offset)
   {
      handle = iCustom(_Symbol, _Period,
         "MQL5Book/p6/UnityPercentEvent",                      // <---
         symbolList, limit, type, method, period);
      ...
By exactly the same, but with a link to the resource (note the syntax with a leading pair of colons '::'
which is necessary to distinguish between ordinary paths in the file system and paths within resources).

---

## Page 1558

Part 6. Trading automation
1 558
6.5 Testing and optimization of Expert Advisors
   UnityController(const string symbolList, const int offset, const int limit,
      const ENUM_APPLIED_PRICE type, const ENUM_MA_METHOD method, const int period):
      bar(offset), tickwise(!offset)
   {
      handle = iCustom(_Symbol, _Period,
         "::Indicators\\MQL5Book\\p6\\UnityPercentEvent.ex5",  // <---
         symbolList, limit, type, method, period);
      ...
Now the compiled version of the Expert Advisor can be delivered to users on its own, without a separate
indicator, since it is hidden inside the Expert Advisor. This does not affect its performance in any way,
but taking into account the TesterHideIndicators challenge, the internal device is hidden. It should be
remembered that if the indicator is then updated, the Expert Advisor will also need to be recompiled.
6.5.1 7 Mathematical calculations
The tester in the MetaTrader 5 terminal can be used not only to test trading strategies but also for
mathematical calculations. To do this, select the appropriate mode in the tester settings, in the
Simulation drop-down list. This is the same list where we select the tick generation method, but in this
case, the tester will not generate ticks or quotes, or even connect the trading environment (trading
account and symbols).
The choice between the full enumeration of parameters and a genetic algorithm depends on the size of
the search space. For the optimization criterion, select "Custom max". Other input fields in the tester
settings (such as date range or delays) are not important and are therefore automatically disabled.
In the "Mathematical calculations" mode, each test agent run is performed with a call of only three
functions: OnInit, OnTester, OnDeinit.
A typical mathematical problem for solving in the MetaTrader 5 tester is finding an extremum for a
function of many variables. To solve it, it is necessary to declare the function parameters in the form of
input variables and place the block for calculating its values in OnTester.
The value of the function for a specific set of input variables is returned as an output value of OnTester.
Do not use any built-in functions other than math functions in calculations.
It must be remembered that when optimizing, the maximum value of the OnTester function is always
sought. Therefore, if you need to find the minimum, you should return the inverse values or the values
multiplied by -1  values.
To understand how this works, let's take as an example a relatively simple function of two variables
with one maximum. Let's describe it in the MathCalc.mq5 Expert Advisor algorithm.
It is usually assumed that we do not know the representation of the function in an analytical form,
otherwise, it would be possible to calculate its extrema. But now let's take a well-known formula to
make sure the answer is correct.

---

## Page 1559

Part 6. Trading automation
1 559
6.5 Testing and optimization of Expert Advisors
input double X1;
input double X2;
   
double OnTester()
{
   const double r = 1 + sqrt(X1 * X1 + X2 * X2);
   return sin(r) / r;
}
The Expert Advisor is accompanied by the MathCalc.set file with parameters for optimization:
arguments X1  and X2 are iterated in the ranges [-1 5, +1 5] with a step of 0.5.
Let's run the optimization and see the solution in the optimization table. The best pass gives the
correct result:
  X1=0.0
  X2=0.0
OnTester result 0.8414709848078965
On the optimization chart, you can turn on the 3D mode and access the shape of the surface visually.
The result of optimization (maximization) of a function in the mathematical calculations mode
At the same time, the use of the tester in the mathematical calculations mode is not limited to purely
scientific research. On its basis, in particular, it is possible to organize the optimization of trading
systems using alternative well-known optimization methods, such as the "particle swarm" or "simulated
annealing" method. Of course, to do this, you will need to upload the history of quotes or ticks to files
and connect them to the tested Expert Advisor, as well as emulate the execution of trades, accounting
for positions and funds. This routine work can be attractive due to the fact that you can freely
customize the optimization process (as opposed to the built-in "black box" with a genetic algorithm)
and control resources (primarily RAM).

---

## Page 1560

Part 6. Trading automation
1 560
6.5 Testing and optimization of Expert Advisors
6.5.1 8 Debugging and profiling
The MetaTrader 5 tester is useful not only for testing the profitability of trading strategies but also for
debugging MQL programs. Error detection is primarily associated with the ability to reproduce the
problem situation. If we could only run MQL programs online, debugging and analyzing source code
execution would require an unrealistic amount of effort. However, the tester allows you to "run"
programs on arbitrary sections of history, change account settings and trading symbols.
Recall that in MetaEditor there are 2 commands in the Debug menu:
·Start/Continue on real data (F5)
·Start/Continue on historical data (Ctrl-F5)
In both cases, the program is promptly recompiled in a special way with additional debugging
information in the ex5 file and then launched directly in the terminal (first option) or in the tester
(second option).
When debugging in the tester, you can use both quick (background) mode and visual mode. This setting
is provided in the Setting dialog on the Debug/Profile tab: enable or disable the flag Use visual mode for
debugging on history. The environment and settings of the program being debugged can be taken
directly from the tester (as they were last set for this program) or in the same dialog in the input fields
under the flag Use specified settings (for them to work, the flag must be enabled).
You can pre-set breakpoints (F9) on operators in the part where something is supposedly starting to
work wrong. The tester will pause the process when it reaches the specified location in the source
code.
Please note that in the tester, the number of history bars loaded at startup depends on different
factors (including timeframe, day number within a year, etc.) and can vary significantly. If necessary,
move the start time of the test back in time.
In addition to obvious bugs that cause the program to stop or explicitly malfunction, there is a class of
subtle bugs that negatively affect performance. As a rule, they are not so obvious, but turn into
problems as the amount of data processed increases, for example, on trading accounts with a very long
history, or on charts with a large number of markup objects.
To find "bottlenecks" in terms of performance, the debugger provides a source code profiling
mechanism. It can also be performed online or in the tester, and the latter is especially valuable, as it
allows you to significantly compress the time. The corresponding commands are also available in the
debug menu.
·Start profiling on real data
·Start profiling on historical data
For profiling, the program is also pre-compiled with special settings, so don't forget to compile the
program again in normal mode after debugging or profiling is complete (especially if you plan to send it
to a client or upload it to the MQL5 Market).
As a result of profiling in MetaEditor, you will receive time statistics of your code execution, broken
down by lines and functions (methods). As a result, it will become clear what exactly slows down the
program. The next stage of development is usually source code refactoring, i.e., its rewriting using
improved algorithms, data structures, or other principles of the constructive organization of modules
(components). Unfortunately, a significant part of the time in programming is spent on rewriting
existing code, finding and fixing errors.

---

## Page 1561

Part 6. Trading automation
1 561 
6.5 Testing and optimization of Expert Advisors
The program itself can, if necessary, find out its mode of operation and adapt its behavior to the
environment (for example, when run in the tester, it will not try to download data from the Internet,
since this feature is disabled, but will read them from a certain file).
At the compilation stage, the debug and production versions of the program can be formed differently
due to preprocessor macros _DEBUG and _RELEASE.
At the program execution stage, its modes can be distinguished using the MQLInfoInteger function
options.
The following table summarizes all available combinations that affect runtime specifics.
Runtime\ flags
MQL_DEBUG
MQL_PROFILER
Normal(release)
Online
+
+
+
Tester
(MQL_TESTER)
+
+
+
Tester
(M Q L _TE S TE R + M Q L _VIS U AL _M O D E )
+
-
+
Profiling in the tester is only possible without the visual mode, so you should measure operations with
charts and objects online.
Debugging is not allowed during the optimization process, including special handlers OnTesterInit,
OnTesterDeinit, and OnTesterPass. If you need to check their performance, consider calling their code
under other conditions.
6.5.1 9 Limitations of functions in the tester
When using the tester, you should take into account some restrictions imposed on built-in functions.
Some of the MQL5 API functions are never executed in the strategy tester and some work only in
single passes but not during optimization.
So, to increase performance when optimizing Expert Advisors, the Comment, Print, and PrintFormat
functions are not executed.
The exception is the use of these functions inside the OnInit handler which is done to make it easier to
find possible causes of initialization errors.
Functions that provide interaction with the "world" are not executed in the strategy tester. These
include MessageBox, PlaySound, SendFTP, SendMail, SendNotification, WebRequest, and functions for
working with sockets.
In addition, many functions for working with charts and objects have no effect. In particular, you will
not be able to change the symbol or period of the current chart by calling ChartSetSymbolPeriod, list all
indicators (including subordinate ones) with ChartIndicatorGet, work with templates
ChartSaveTemplate, and so on.
In the tester, even in the visual mode, interactive chart, object, keyboard and mouse events are not
generated for the OnChartEvent handler.

---

## Page 1562

Part 7. Advanced language tools
1 562
 
Part 7. Advanced MQL5 Tools
In this part of the book, we will learn about additional MQL5 API features in various areas that may be
required when developing programs for the MetaTrader 5 environment. Some of them are of an applied
trading nature, for example, custom financial instruments or the built-in economic calendar. Others
represent universal technologies that can be useful everywhere: network functions, databases,
cryptography, etc.
In addition, we will consider extending MQL programs using resources which are files of an arbitrary
type that can be embedded in the code and contain multimedia, "heavy" settings from external
programs (for example, ready-made machine learning models or neural network configurations) or
other MQL programs (indicators) in a compiled form.
A couple of chapters will be devoted to the modular development of MQL programs. In this context, we
will consider a special program type – libraries, which can be connected to other MQL programs to
provide ready-made sets of specific APIs in closed form but which cannot be used standalone. We will
also explore the possibilities for organizing the process of developing software complexes and combining
logically interrelated programs into projects.
Finally, we will present integration with other software environments, in particular, with Python.
The book does not cover some highly specialized topics that may be of interest to advanced users,
such as hardware capabilities for parallel computing using OpenCL, as well as 2D and 3D graphics based
on DirectX. It is suggested that you familiarize yourself with these technologies using the
documentation and articles on the mql5.com website.
MQL5 Programming for Traders – Source Codes from the Book. Part 7
Examples from the book are also available in the public project \MQL5\Shared Projects\MQL5Book
7.1  Resources
The operation of MQL programs may require many auxiliary resources, which are arrays of application
data or files of various types, including images, sounds, and fonts. The MQL development environment
allows you to include all such resources in the executable file at the compilation stage. This eliminates
the need for their parallel transfer and installation along with the main program and makes it a
complete self-sufficient product that is convenient for the end user.
In this chapter, we will learn how to describe different types of resources and built-in functions for
subsequent operations with connected resources.
Raster images, represented as arrays of points (pixels) in the widely recognized BMP format, hold a
unique position among resources. The MQL5 API allows the creation, manipulation, and dynamic display
of these graphic resources on charts.
Earlier, we already discussed graphical objects and, in particular, objects of types OBJ_BITMAP and
OBJ_BITMAP_LABEL that are useful for designing user interfaces. For these objects, there is the
OBJPROP_BMPFILE property that specifies the image as a file or resource. Previously, we only
considered examples with files. Now we will learn how to work with resource images.

---

## Page 1563

Part 7. Advanced language tools
1 563
7.1  Resources
7.1 .1  Describing resources using the #resource directive
To include a resource file in the compiled program version, use the #resource directive in the source
code. The directive has different forms depending on the file type. In any case, the directive contains
the #resource keyword followed by a constant string.
#resource "path_file_name"
The #resource command instructs the compiler to include (in binary format ex5) a file with the
specified name and, optionally, location (at the time of compilation) into the executable program being
generated. The path is optional: if the string contains only the file name, it is searched in the directory
next to the compiled source code. If there is a path in the string, the rules described below apply.
The compiler looks for the resource at the specified path in the following sequence:
• If the path is preceded by a backslash '\\' (it must be doubled, since a single backslash is a control
character; in particular, '\' is used for newlines '\r', '\n' and tabs '\t'), then the resource is
searched starting from the MQL5 folder inside the terminal data directory.
• If there is no backslash, then the resource is searched relative to the location of the source file in
which this resource is registered.
Note that in constant strings with resource paths, you must use double backslashes as separators.
Forward single slashes are not supported here, unlike paths in the file system.
For example:
#resource "\\Images\\euro.bmp" // euro.bmp is in /MQL5/Images/
#resource "picture.bmp"        // picture.bmp is in the same directory,
                               // where the source file is (mq5 or mqh)
#resource "Resource\\map.bmp"  // map.bmp is in the Resource subfolder of the directory
                               // where the source file is (mq5 or mqh)
If the resource is declared with a relative path in the mqh header file, the path is considered relative to
this mqh file and not to the mq5 file of the program being compiled.
The substrings "..\\" and ":\\" are not allowed in the resource path.
Using a few directives, you can, for example, put all the necessary pictures and sounds directly into the
ex5 file. Then, to run such a program in another terminal, you do not need to transfer them separately.
We will consider programmatic ways of accessing resources from MQL5 in the following sections.
The length of the constant string "path_file_name" must not exceed 63 characters. The resource file
size cannot be more than 1 28 Mb. Resource files are automatically compressed before being included
in the executable.
After the resource is declared by the #resource directive, it can be used in any part of the program.
The name of the resource becomes the constant string specified in the directive without a slash at the
beginning (if any), and a special sign of the resource (two colons, "::") should be added before the
contents of the string.
Below we present examples of resources, with their names in the comments.

---

## Page 1564

Part 7. Advanced language tools
1 564
7.1  Resources
#resource "\\Images\\euro.bmp"          // resource name - ::Images\\euro.bmp
#resource "picture.bmp"                 // resource name - ::picture.bmp
#resource "Resource\\map.bmp"           // resource name - ::Resource\\map.bmp
#resource "\\Files\\Pictures\\good.bmp" // resource name - ::Files\\Pictures\\good.bmp
#resource "\\Files\\demo.wav";          // resource name - ::Files\\demo.wav"
#resource "\\Sounds\\thrill.wav";       // resource name - ::Sounds\\thrill.wav"
Further in the MQL code, you can refer to these resources as follows (here, only the Obj ectSetString
and PlaySound functions are already known to us, but there are other options like ResourceReadImage,
which will be described in the following sections).
ObjectSetString(0, bitmap_name, OBJPROP_BMPFILE, 0, "::Images\\euro.bmp");
...
ObjectSetString(0, my_bitmap, OBJPROP_BMPFILE, 0, "::picture.bmp");
...
ObjectSetString(0, bitmap_label, OBJPROP_BMPFILE, 0, "::Resource\\map.bmp");
ObjectSetString(0, bitmap_label, OBJPROP_BMPFILE, 1, "::Files\\Pictures\\good.bmp");
...
PlaySound("::Files\\demo.wav");
...
PlaySound("::Sounds\\thrill.wav");
It should be noted that when setting an image from a resource to OBJ_BITMAP and
OBJ_BITMAP_LABEL objects, the value of the OBJPROP_BMPFILE property cannot be changed
manually (in the object's properties dialog).
Note that wav files are set by default for the PlaySound function relative to the Sounds folder (or its
subfolders) located in the terminal's data directory. At the same time, resources (including sound
ones), if they are described with a leading slash in the path, are searched inside the MQL5 directory.
Therefore, in the example above, the "\\Sounds\\thrill.wav" string refers to the file
MQL5/Sounds/thrill.wav and not to Sounds/thrill.wav relative to the data directory (there is indeed the
Sounds directory with standard terminal sounds).
The simple syntax of the #resource directive discussed above allows the description of only image
resources (BMP format) and sound resources (WAV format). Attempting to describe a file of a different
type as a resource will result in an "unknown resource type" error.
As a result of #resource directive processing, the files in fact become embedded into the executable
binary program and become accessible by the resource name. Moreover, you should pay attention to a
special property of such resources which is their public availability from other programs (more on this in
the next section).
MQL5 also supports another way of embedding a file in a program:  in the form of a resource variable.
This method uses extended syntax of the #resource directive and allows you to connect not only BMP
or WAV files but also others, for example, text or an array of structures.
We will analyze a practical example of connecting resources in a couple of sections.
7.1 .2 Shared use of resources of different MQL programs
The resource name is unique throughout the terminal. Later we will learn how to create resources not
at the compilation stage (by the #resource directive) but dynamically, using the ResourceCreate
function. In any case, the resource is declared in the context of the program that creates it, so that

---

## Page 1565

Part 7. Advanced language tools
1 565
7.1  Resources
the uniqueness of the full name is provided automatically by binding to the file system (path and name
of a specific file ex5).
In addition to containing and using resources, an MQL program can also access the resources of
another compiled program (ex5 file). This is possible provided that the program using the resource
knows the location path and the name of another program containing the required resource, as well as
the name of this resource.
Thus, the terminal provides an important property of resources which is their shared use: resources
from one ex5 file can be used in many other programs.
In order to use a resource from a third-party ex5 file, it must be specified in the form
"path_file_name.ex5::resource_name". For example, let's say the script DrawingScript.mq5 refers to a
specified image resource in the file triangle.bmp:
#resource "\\Files\\triangle.bmp"
Then its name for use in the actual script will look like "::Files\\triangle.bmp".
To use the same resource from another program, for example, an Expert Advisor, the resource name
should be preceded by the path of the ex5 script file relative to the MQL5 folder in the terminal data
directory, as well as the name of the script itself (in the compiled form, DrawingScript.ex5). Let the
script be in the standard MQL5/Scripts/ folder. In this case, the image should be accessed using the "\
\Scripts\\DrawingScript.ex5::Files\\triangle.bmp" string. The ".ex5" extension is optional.
If, when accessing the resource of another ex5 file, the path to this file is not specified, then such a file
is searched in the same folder where the program requesting the resource is located. For example, if
we assume that the same Expert Advisor is in the standard MQL5/Experts/ folder, and it queries a
resource without specifying the path (for example, "DrawingScript.ex5::Files\\triangle.bmp"), then
DrawingScript.ex5 will be searched in the MQL5/Experts/ folder.
Due to the shared use of resources, their dynamic creation and updating can be used to exchange data
between MQL programs. This happens right in memory and is therefore a good alternative to files or
global variables.
Please note that to load a resource from an MQL program, you do not need to run it: to read resources,
it is enough to have an ex5 file with resources.
An important exception during which report sharing is not possible is when a resource is described in
the form of a resource variable.
7.1 .3 Resource variables
The #resource directive has a special form with which external files can be declared as resource
variables and accessed within the program as normal variables of the corresponding type. The
declaration format is:
#resource "path_file_name" as resource_variable_type resource_variable_name
Here are some examples of declarations:

---

## Page 1566

Part 7. Advanced language tools
1 566
7.1  Resources
#resource "data.bin" as int Data[]           //array of int type with data from the file data.bin 
#resource "rates.dat" as MqlRates Rates[]    // array of MqlRates structures from the file rates.dat
#resource "data.txt" as string Message       // line with the contents of the file data.txt
#resource "image.bmp" as bitmap Bitmap1[]    // one-dimensional array with image pixels
                                             // from file image.bmp
#resource "image.bmp" as bitmap Bitmap2[][]  // two-dimensional array with the same image
Let's give some explanations. Resource variables are constants (they cannot be modified in MQL5
code). For example, to edit images before displaying on the screen, you should create copies of
resource array variables.
For text files (resources of type string) the encoding is automatically determined by the presence of a
BOM header. If there is no BOM, then the encoding is determined by the contents of the file. ANSI,
UTF-8, and UTF-1 6 encodings are supported. When reading data from files, all strings are converted to
Unicode.
The use of resource string variables can greatly facilitate the writing of programs based not only on
pure MQL5 but also on additional technologies. For example, you can write OpenCL code (which is
supported in MQL5 as an extension) in a separate file and then include it as a string in the resources of
an MQL program. In the big Expert Advisor example, we've already used resource strings to include
HTML templates.
For images, a special bitmap type has been introduced; this type has several features.
The bitmap type describes a single dot or pixel in an image and is represented by a 4-byte unsigned
integer (uint). The pixel contains 4 bytes that correspond to the color components in ARGB or XRGB
format (one letter = one byte), where R is red, G is green, B is blue, A is transparency (alpha channel),
X is an is ignored byte (no transparency). Transparency can be used for various effects when overlaying
images on a chart and on top of each other.
We will study the definition of ARGB and XRGB formats in the section on dynamic creation of graphic
resources (see ResourceCreate). For example, for ARGB, the hexadecimal number 0xFFFF0000
specifies a fully opaque pixel (highest byte is 0xFF) of red color (the next byte is also 0xFF), and the
next bytes for the green and blue components are zero.
It is important to note that the pixel color encoding is different from the byte representation of type
color. Let's recall that the value of type color can be written in hexadecimal form as follows:
0x00BBGGRR, where BB, GG, RR are the blue, green and red components, respectively (in each byte,
the value 255 gives the maximum intensity of the component). With a similar record of a pixel, there is
a reverse byte order: 0xAARRGGBB. Full transparency is obtained when the high byte (here denoted
AA) is 0 and the value 255 is a solid color. The ColorToARGB function can be used to convert color to
ARGB.
BMP files can have various encoding methods (if you create or edit them in any editor, check this issue
in the documentation of this program). MQL5 resources do not support all existing encoding methods.
You can check if a particular file is supported using the ResourceCreate function. Specifying an
unsupported BMP format file in the directive will result in a compilation error.
When loading a file with 24-bit color encoding, all pixels of the alpha channel component are set to 255
(opaque). When loading a file with a 32-bit color encoding without an alpha channel, it also implies no
transparency, that is, for all image pixels, the alpha channel component is set to 255. When loading a
32-bit color-coded file with an alpha channel, no pixel manipulation takes place.

---

## Page 1567

Part 7. Advanced language tools
1 567
7.1  Resources
Images can be described by both one-dimensional and two-dimensional arrays. This only affects the
addressing method, while the amount of memory occupied will be the same. In both cases, the array
sizes are automatically set based on the data from the BMP file. The size of a one-dimensional array
will be equal to the product of the height and the width of the image (height * width), and a two-
dimensional array will get separate dimensions [height][width]: the first index is the line number, the
second is a dot in the line.
Attention! When declaring a resource linked to a resource variable, the only way to access the
resource is through that variable, and the standard way of reading through the name
"::resource_name" (or more generally "path_file_name.ex5::resource_name") no longer works. This
also means that such resources cannot be used as shared resources from other programs.
Let's consider two indicators as an example; both are bufferless. This MQL program type was chosen
only for reasons of convenience because it can be applied to the chart without conflict in addition to
other indicators while an Expert Advisor would require a chart without another Expert Advisor. In
addition, they remain on the chart and are available for subsequent settings changes, unlike scripts.
The BmpOwner.mq5 indicator contains a description of three resources:
• An image "search1 .bmp" with a simple #resource directive which is accessible from other programs
• An image "search2.bmp" as a resource array variable of type bitmap, inaccessible from the outside
• A text file "message.txt" as a resource string for displaying a warning to the user
Both images are not used in any way within this indicator. The warning line is required in the OnInit
function to call Alert since the indicator is not intended for independent use but only acts as a provider
of an image resource.
If the resource variable is not used in the source code, the compiler may not include the resource
at all in the binary code of the program, but this does not apply to images.
#resource "search1.bmp"
#resource "search2.bmp" as bitmap image[]
#resource "message.txt" as string Message
All three files are located in the same directory where the indicator source is located:
MQL5/Indicators/MQL5Book/p7/.
If the user tries to run the indicator, it displays a warning and immediately stops working. The warning
is contained in the Message resource string variable.
int OnInit()
{
   Alert(Message); // equivalent to the following line of the code
   // Alert("This indicator is not intended to run, it holds a bitmap resource");
   
   // remove the indicator explicitly, because otherwise it remains "hanging" on the chart uninitialized
   ChartIndicatorDelete(0, 0, MQLInfoString(MQL_PROGRAM_NAME));
   return INIT_FAILED;
}
In the second indicator BmpUser.mq5, we will try to use the external resources specified in the input
variables ResourceOff and ResourceOn, to display in the OBJ_BITMAP_LABEL object.

---

## Page 1568

Part 7. Advanced language tools
1 568
7.1  Resources
input string ResourceOff = "BmpOwner.ex5::search1.bmp";
input string ResourceOn = "BmpOwner.ex5::search2.bmp";
By default, the state of the object is disabled/released ("Off"), and the image for it is taken from the
previous indicator "BmpOwner.ex5::search1 .bmp". This path and resource name are similar to the full
notation "\\Indicators\\MQL5Book\\p7\\BmpOwner.ex5::search1 .bmp". The short form is acceptable
here, given that the indicators are located next to each other. If you subsequently open the object
properties dialog, you will see the full notation in the Bitmap file (On/Off) fields.
For the pressed state, in ResourceOn we should read the resource "BmpOwner.ex5::search2.bmp" (let's
see what happens).
In other input variables, you can select the corner of the chart, relative to which the positioning of the
image is set, and the horizontal and vertical indents.
input int X = 25;
input int Y = 25;
input ENUM_BASE_CORNER Corner = CORNER_RIGHT_LOWER;
The creation of the OBJ_BITMAP_LABEL object and the setting of its properties, including the resource
name as a picture for OBJPROP_BMPFILE, are performed in OnInit.
const string Prefix = "BMP_";
const ENUM_ANCHOR_POINT Anchors[] =
{
   ANCHOR_LEFT_UPPER,
   ANCHOR_LEFT_LOWER,
   ANCHOR_RIGHT_LOWER,
   ANCHOR_RIGHT_UPPER
};
   
void OnInit()
{
   const string name = Prefix + "search";
   ObjectCreate(0, name, OBJ_BITMAP_LABEL, 0, 0, 0);
   
   ObjectSetString(0, name, OBJPROP_BMPFILE, 0, ResourceOn);
   ObjectSetString(0, name, OBJPROP_BMPFILE, 1, ResourceOff);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, X);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, Y);
   ObjectSetInteger(0, name, OBJPROP_CORNER, Corner);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, Anchors[(int)Corner]);
}
Recall that when specifying images in OBJPROP_BMPFILE, the pressed state is indicated by modifier 0,
and the released (unpressed) state (by default) is indicated by modifier 1 , which is somewhat
unexpected.
The OnDeinit handler deletes the object when unloading the indicator.

---

## Page 1569

Part 7. Advanced language tools
1 569
7.1  Resources
void OnDeinit(const int)
{
   ObjectsDeleteAll(0, Prefix);
}
Let's compile both indicators and run BmpUser.ex5 with default settings. The image of the graphic file
search1 .bmp should appear on the chart (see on the left).
N ormal (left) and wrong (right) display of graphic resources in an object on a chart
If you click on the picture, that is, switch it to the pressed state, the program will try to access the
"BmpOwner.ex5::search2.bmp" resource (which is unavailable due to the resource array bitmap
attached to it). As a result, we will see a red square, indicating an empty object without a picture (see
above, on the right). A similar situation will always occur if the input parameter specifies a path or
name with a knowingly non-existent or non-shared resource. You can create your own program,
describe in it a resource that refers to some existing bmp file, and then specify in the indicator input
parameters BmpUser. In this case, the indicator will be able to display the picture on the chart.
7.1 .4 Connecting custom indicators as resources
For operation, MQL programs may require one or more custom indicators. All of these can be included
as resources in the ex5 executable, making it easy to distribute and install.
The #resource directive with the description of the nested indicator has the following format:
#resource "path_indicator_name.ex5"
The rules for setting and searching for the specified file are the same as for all resources generally.
We have already used this feature in the big Expert Advisor example, in the final version of
UnityMartingale.mq5.
#resource "\\Indicators\\MQL5Book\\p6\\UnityPercentEvent.ex5"
In that Expert Advisor, instead of the indicator name, this resource was passed to the iCustom function:
"::Indicators\\MQL5Book\\p6\\UnityPercentEvent.ex5".
The case when a custom indicator in the OnInit function creates one or more instances of itself
requires separate consideration (if this technical solution itself seems strange, we will give a practical
example after the introductory examples).
As we know, to use a resource from an MQL program, it must be specified in the following form:
path_file_name.ex5::resource_name. For example, if the EmbeddedIndicator.ex5 indicator is included as

---

## Page 1570

Part 7. Advanced language tools
1 570
7.1  Resources
a resource in another indicator MainIndicator.mq5 (more precisely, in its binary image
MainIndicator.ex5), then the name specified when calling itself via iCustom can no longer be short,
without a path, and the path must include the location of the "parent" indicator inside the MQL5 folder.
Otherwise, the system will not be able to find the nested indicator.
Indeed, under normal circumstances, an indicator can call itself using, for example, the operator
iCustom(_ Symbol, _ Period, myself,...), where myself is a string equal to either
MQLInfoString(MQL_ PROGRAM_ NAME) or the name that was previously assigned to the
INDICATOR_SHORTNAME property in the code. But when the indicator is located inside another MQL
program as a resource, the name no longer refers to the corresponding file because the file that served
as a prototype for the resource remained on the computer where the compilation was performed, and
on the user's computer there is only the file MainIndicator.ex5. This will require some analysis of the
program environment when starting the program.
Let's see this in practice.
To begin with, let's create an indicator NonEmbeddedIndicator.mq5. It is important to note that it is
located in the folder MQL5/Indicators/MQL5Book/p7/SubFolder/, i.e. in a SubFolder relative to the
folder p7 allocated for all indicators of this Part of the book. This is done intentionally to emulate a
situation where the compiled file is not present on the user's computer. Now we will see how it works
(or rather, demonstrates the problem).
The indicator has a single input parameter Reference. Its purpose is to count the number of copies of
itself: when first created, the parameter equals 0, and the indicator will create its own copy with the
parameter value of 1 . The second copy, after "seeing" the value 1 , will no longer create another copy
(otherwise we would quickly run out of resources without the boundary condition for stopping
reproduction).
input int Reference = 0;
The handle variable is reserved for the handle of the copy indicator.
int handle = 0;
In the handler OnInit, for clarity, we first display the name and path of the MQL program.
int OnInit()
{
   const string name = MQLInfoString(MQL_PROGRAM_NAME);
   const string path = MQLInfoString(MQL_PROGRAM_PATH);
   Print(Reference);
   Print("Name: " + name);
   Print("Full path: " + path);
   ...
Next comes the code suitable for self-launching a separate indicator (existing in the form of the familiar
file NonEmbeddedIndicator.ex5).

---

## Page 1571

Part 7. Advanced language tools
1 571 
7.1  Resources
   if(Reference == 0)
   {
      handle = iCustom(_Symbol, _Period, name, 1);
      if(handle == INVALID_HANDLE)
      {
         return INIT_FAILED;
      }
   }
   Print("Success");
   return INIT_SUCCEEDED;
}
We could successfully place such an indicator on the chart and receive entries of the following kind in
the log (you will have your own file system paths):
0
Name: NonEmbeddedIndicator
Full path: C:\Program Files\MT5East\MQL5\Indicators\MQL5Book\p7\SubFolder\NonEmbeddedIndicator.ex5
Success
1
Name: NonEmbeddedIndicator
Full path: C:\Program Files\MT5East\MQL5\Indicators\MQL5Book\p7\SubFolder\NonEmbeddedIndicator.ex5
Success
The copy started successfully just by using the name "NonEmbeddedIndicator".
Let's leave this indicator for now and create a second one, FaultyIndicator.mq5, into which we will
include the first indicator as a resource (pay attention to the specification of subfolder in the relative
path of the resource; this is necessary because the FaultyIndicator.mq5 indicator is located in the
folder one level up: MQL5/Indicators/MQL5Book/p7/).
// FaultyIndicator.mq5
#resource "SubFolder\\NonEmbeddedIndicator.ex5"
   
int handle;
   
int OnInit()
{
   handle = iCustom(_Symbol, _Period, "::SubFolder\\NonEmbeddedIndicator.ex5");
   if(handle == INVALID_HANDLE)
   {
      return INIT_FAILED;
   }
   return INIT_SUCCEEDED;
}
If you try to run the compiled FaultyIndicator.ex5, an error will occur:

---

## Page 1572

Part 7. Advanced language tools
1 572
7.1  Resources
0
Name: NonEmbeddedIndicator
Full path: C:\Program Files\MT5East\MQL5\Indicators\MQL5Book\p7\FaultyIndicator.ex5 »
»  ::SubFolder\NonEmbeddedIndicator.ex5
cannot load custom indicator 'NonEmbeddedIndicator' [4802]
When a copy of a nested indicator is launched, it is searched for in the folder of the main indicator, in
which the resource is described. But there is no file NonEmbeddedIndicator.ex5 because the required
resource is inside FaultyIndicator.ex5.
To solve the problem, we modify NonEmbeddedIndicator.mq5. First of all, let's give it another, more
appropriate name, EmbeddedIndicator.mq5. In the source code, we need to add a helper function
GetMQL5Path, which can isolate the relative part inside the MQL5 folder from the general path of the
launched MQL program (this part will also contain the name of the resource if the indicator is launched
from a resource).
// EmbeddedIndicator.mq5
string GetMQL5Path()
{
   static const string MQL5 = "\\MQL5\\";
   static const int length = StringLen(MQL5) - 1;
   static const string path = MQLInfoString(MQL_PROGRAM_PATH);
   const int start = StringFind(path, MQL5);
   if(start != -1)
   {
      return StringSubstr(path, start + length);
   }
   return path;
}
Taking into account the new function, we will change the iCustom call in the OnInit handler.
int OnInit()
{
   ...
   const string location = GetMQL5Path();
   Print("Location in MQL5:" + location);
   if(Reference == 0)
   {
      handle = iCustom(_Symbol, _Period, location, 1);
      if(handle == INVALID_HANDLE)
      {
         return INIT_FAILED;
      }
   }
   return INIT_SUCCEEDED;
}
Let's make sure that this edit did not break the launch of the indicator. Overlaying on a chart results in
the expected lines appearing in the log:

---

## Page 1573

Part 7. Advanced language tools
1 573
7.1  Resources
0
Name: EmbeddedIndicator
Full path: C:\Program Files\MT5East\MQL5\Indicators\MQL5Book\p7\SubFolder\EmbeddedIndicator.ex5
Location in MQL5:\Indicators\MQL5Book\p7\SubFolder\EmbeddedIndicator.ex5
Success
1
Name: EmbeddedIndicator
Full path: C:\Program Files\MT5East\MQL5\Indicators\MQL5Book\p7\SubFolder\EmbeddedIndicator.ex5
Location in MQL5:\Indicators\MQL5Book\p7\SubFolder\EmbeddedIndicator.ex5
Success
Here we added debug output of the relative path that the GetMQL5Path function received. This line is
now used in iCustom, and it works in this mode: a copy has been created.
Now let's embed this indicator as a resource into another indicator in the MQL5Book/p7 folder with the
name MainIndicator.mq5. MainIndicator.mq5 is completely identical to FaultyIndicator.mq5 except for
the connected resource.
// MainIndicator.mq5
#resource "SubFolder\\EmbeddedIndicator.ex5"
...
int OnInit()
{
   handle = iCustom(_Symbol, _Period, "::SubFolder\\EmbeddedIndicator.ex5");
   ...
}
Let's compile and run it. Entries appear in the log with a new relative path that includes the nested
resource.
0
Name: EmbeddedIndicator
Full path: C:\Program Files\MT5East\MQL5\Indicators\MQL5Book\p7\MainIndicator.ex5 »
»  ::SubFolder\EmbeddedIndicator.ex5
Location in MQL5:\Indicators\MQL5Book\p7\MainIndicator.ex5::SubFolder\EmbeddedIndicator.ex5
Success
1
Name: EmbeddedIndicator
Full path: C:\Program Files\MT5East\MQL5\Indicators\MQL5Book\p7\MainIndicator.ex5 »
»  ::SubFolder\EmbeddedIndicator.ex5
Location in MQL5:\Indicators\MQL5Book\p7\MainIndicator.ex5::SubFolder\EmbeddedIndicator.ex5
Success
As we can see, this time the nested indicator successfully created a copy of itself, as it used a qualified
name with a relative path and a resource name "\\Indicators\\MQL5Book\\p7\
\MainIndicator.ex5::SubFolder\\EmbeddedIndicator.ex5".
During multiple experiments with launching this indicator, please note that nested copies are not
immediately unloaded from the chart after the main indicator is removed. Therefore, restarts
should be performed only after we waited for unloading to happen: otherwise, copies still running will
be reused, and the above initialization lines will not appear in the log. To control the unloading, a
printout of the Reference value has been added to the OnDeinit handler.
We promised to show that creating a copy of the indicator is not something extraordinary. As an
applied demonstration of this technique, we use the indicator DeltaPrice.mq5 which calculates the

---

## Page 1574

Part 7. Advanced language tools
1 574
7.1  Resources
difference in price increments of a given order. Order 0 means no differentiation (only to check the
original time series), 1  means single differentiation, 2 means double differentiation, and so on.
The order is specified in the input parameter Differentiating.
input int Differencing = 1;
The difference series will be displayed in a single buffer in the subwindow.
#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1
   
#property indicator_type1 DRAW_LINE
#property indicator_color1 clrDodgerBlue
#property indicator_width1 2
#property indicator_style1 STYLE_SOLID
   
double Buffer[];
In the OnInit, handler we set up the buffer and create the same indicator, passing the value reduced by
1  in the input parameter.
#include <MQL5Book/AppliedTo.mqh> // APPLIED_TO_STR macro
int handle = 0;
   
int OnInit()
{
   const string label = "DeltaPrice (" + (string)Differencing + "/"
      + APPLIED_TO_STR() + ")";
   IndicatorSetString(INDICATOR_SHORTNAME, label);
   PlotIndexSetString(0, PLOT_LABEL, label);
   
   SetIndexBuffer(0, Buffer);
   if(Differencing > 1)
   {
      handle = iCustom(_Symbol, _Period, GetMQL5Path(), Differencing - 1);
      if(handle == INVALID_HANDLE)
      {
         return INIT_FAILED;
      }
   }
   return INIT_SUCCEEDED;
}
To avoid potential problems with embedding the indicator as a resource, we use the already proven
function GetMQL5Path.
In the OnCalculate function, we perform the operation of subtracting neighboring values of the time
series. When Differentiating equals 1 , the operands are elements of the price array. With a larger value
of Differentiating, we read the buffer of the indicator copy created for the previous order.

---

## Page 1575

Part 7. Advanced language tools
1 575
7.1  Resources
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
   for(int i = fmax(prev_calculated - 1, 1); i < rates_total; ++i)
   {
      if(Differencing > 1)
      {
         static double value[2];
         CopyBuffer(handle, 0, rates_total - i - 1, 2, value);
         Buffer[i] = value[1] - value[0];
      }
      else if(Differencing == 1)
      {
         Buffer[i] = price[i] - price[i - 1];
      }
      else
      {
         Buffer[i] = price[i];
      }
   }
   return rates_total;
}
The initial type of differentiated price is set in the indicator settings dialog in the Apply to drop-down
list. By default, this is the Close price.
This is how several copies of the indicator look on the chart with different orders of differentiation.


---

## Page 1576

Part 7. Advanced language tools
1 576
7.1  Resources
Difference in Close prices of different orders
7.1 .5 Dynamic resource creation: ResourceCreate
The #resource directives embed resources into the program at the compilation stage, and therefore
they can be called static. However, it often becomes necessary to generate resources (create
completely new or modify existing ones) at the stage of program execution. For these purposes, MQL5
provides the ResourceCreate function. The resources created with the help of this function will be called
dynamic.
The function has two forms: the first one allows you to load pictures and sounds from files, and the
second one is designed to create bitmap images based on an array of pixels prepared in memory.
bool ResourceCreate(const string resource, const string filepath)
The function loads the resource named resource from a file located at filepath. If the path starts with a
backslash '\' (in constant strings it should be doubled: "\\path\\name.ext"), then the file is searched at
this path relative to the MQL5 folder in the terminal data directory (for example, "\\Files \
\CustomSounds\\Hello.wav" refers to MQL5/Files/CustomSounds/Hello.wav). If there is no backslash,
then the resource is searched starting from the folder where the executable file from which we call the
function is located.
The path can point to a static resource hardwired into a third-party or current MQL program. For
example, a certain script is able to create a resource based on a picture from the indicator
BmpOwner.mq5 discussed in the section on Resource variables.
ResourceCreate("::MyImage", "\\Indicators\\MQL5Book\\p7\\BmpOwner.ex5::search1.bmp");
The resource name in the resource parameter may contain an initial double colon (although this is not
required, because if it is not present, the "::" prefix will be added to the name automatically). This
ensures the unification of the use of one line for declaring a resource in the ResourceCreate call, as well
as for subsequent access to it (for example, when setting the OBJPROP_BMPFILE property).
Of course, the above statement for creating a dynamic resource is redundant if we just want to load a
third-party image resource into our object on the chart, since it is enough to directly assign the string
"\\Indicators\\MQL5Book\\p7\\BmpOwner.ex5:" to the OBJPROP_BMPFILE property: search1 .bmp".
However, if you need to edit an image, a dynamic resource is indispensable. Next, we will show an
example in the section Reading and modifying resource data.
Dynamic resources are publicly available from other MQL programs by their full name, which includes
the path and name of the program that created the resource. For example, if the previous
ResourceCreate call was produced by the script MQL5/Scripts/MyExample.ex5, then another MQL
program can access the same resource using the full link "\\Scripts\\MyExample.ex5::MyImage", and
any other script in the same folder can access the shorthand "MyExample.ex5::MyImage" (here the
relative path is simply degenerate). The rules for writing full (from the MQL5 root folder) and relative
paths were given above.
The ResourceCreate function returns a boolean indicator of success (true) or error (false) as a result of
execution. The error code, as usual, can be found in the _ LastError variable. Specifically, you are likely
to receive the following errors:
• ERR_RESOURCE_NAME_DUPLICATED (401 5) – matching names of the dynamic and static
resources

---

## Page 1577

Part 7. Advanced language tools
1 577
7.1  Resources
• ERR_RESOURCE_NOT_FOUND (401 6) – the given resource/file from the filepath parameter is not
found
• ERR_RESOURCE_UNSUPPOTED_TYPE (401 7) – unsupported resource type or size more than 2 GB
• ERR_RESOURCE_NAME_IS_TOO_LONG (401 8) – resource name exceeds 63 characters
All this applies not only to the first form of the function but also to the second.
bool ResourceCreate(const string resource, const uint &data[], uint img_width, uint img_height, uint
data_xoffset, uint data_yoffset, uint data_width, ENUM_COLOR_FORMAT color_format)
The resource parameter still means the name of the new resource, and the content of the image is
given by the rest of the parameters.
The data array may be one-dimensional (data[]) or two-dimensional (data[][]): it passes dots (pixels) of
the raster. The parameters img_ width and img_ height set the dimensions of the displayed image (in
pixels). These sizes may be less than the physical size of the image in the data array, due to which the
effect of framing is achieved when only a part of the original image is output. The data_ xoffset and
data_ yoffset parameters determine the coordinate of the upper left corner of the "frame".
The data_ width parameter means the full width of the original image (in the data array). A value of 0
implies that this width is the same as img_ width. The data_ width parameter makes sense only when
specifying a one-dimensional array in the data parameter, since for a two-dimensional array its
dimensions are known in both dimensions (in this case, the data_ width parameter is ignored and is
assumed equal to the second dimension of the data[][] array).
In the most common case, when you want to display the image in full ("as is"), use the following
syntax:
ResourceCreate(name, data, width, height, 0, 0, 0, ...);
For example, if the program has a static resource described as a two-dimensional bitmap array:
#resource "static.bmp" as bitmap data[][]
Then the creation of a dynamic resource based on it can be performed in the following way:
ResourceCreate("dynamic", data, ArrayRange(data, 1), ArrayRange(data, 0), 0, 0, 0, ...);
Creating a dynamic resource based on a static one is in demand not only when direct editing is
required, but also to control how colors are processed when displaying a resource. This mode is
selected using the last parameter of the function: color_ format. It uses the ENUM_COLOR_FORMAT
enumeration.
Identifier
Description
COLOR_FORMAT_XRGB_NOALPHA
The alpha channel component (transparency) is ignored
COLOR_FORMAT_ARGB_RAW
Color components are not processed by the terminal
COLOR_FORMAT_ARGB_NORMALIZE
Color components are processed by the terminal (see below)
In the COLOR_FORMAT_XRGB_NOALPHA mode, the image is displayed without effects: each point is
displayed in a solid color (this is the fastest way to draw). The other two modes display pixels taking
into account the transparency in the high byte of each pixel but have different effects. In the case of

---

## Page 1578

Part 7. Advanced language tools
1 578
7.1  Resources
COLOR_FORMAT_ARGB_NORMALIZE, the terminal performs the following transformations of the color
components of each point when preparing the raster at the time of the ResourceCreate call:
R = R * A / 255
G = G * A / 255
B = B * A / 255
A = A
Static image resources in #resource directives are connected with the help of
COLOR_FORMAT_ARGB_NORMALIZE.
In a dynamic resource, the array size is limited by the value of INT_MAX bytes (21 47483647, 2 Gb),
which significantly exceeds the limit imposed by the compiler when processing the static directive
#resource: the file size cannot exceed 1 28 Mb.
If the second version of the function is called to create a resource with the same name, but with
changing other parameters (the contents of the pixel array, width, height, or offset), then the new
resource is not recreated, but the existing one is simply updated. Only the program owning the
resource (the program that created it in the first place) can modify a resource in this way.
If, when creating dynamic resources from different copies of the program running on different
charts, you need your own resource in each copy, you should add ChartID to the name of the
resource.
To demonstrate the dynamic creation of images in various color schemes, we propose to disassemble
the script ARGBbitmap.mq5.
The image "argb.bmp" is statically attached to it.
#resource "argb.bmp" as bitmap Data[][]
The user selects the color formatting method by the ColorFormat parameter.
input ENUM_COLOR_FORMAT ColorFormat = COLOR_FORMAT_XRGB_NOALPHA;
The name of the object in which the image will be displayed and the name of the dynamic resource are
described by the variables BitmapObj ect and ResName.
const string BitmapObject = "BitmapObject";
const string ResName = "::image";
Below is the main function of the script.

---

## Page 1579

Part 7. Advanced language tools
1 579
7.1  Resources
void OnStart()
{
   ResourceCreate(ResName, Data, ArrayRange(Data, 1), ArrayRange(Data, 0),
      0, 0, 0, ColorFormat);
   
   ObjectCreate(0, BitmapObject, OBJ_BITMAP_LABEL, 0, 0, 0);
   ObjectSetInteger(0, BitmapObject, OBJPROP_XDISTANCE, 50);
   ObjectSetInteger(0, BitmapObject, OBJPROP_YDISTANCE, 50);
   ObjectSetString(0, BitmapObject, OBJPROP_BMPFILE, ResName);
   
   Comment("Press ESC to stop the demo");
   const ulong start = TerminalInfoInteger(TERMINAL_KEYSTATE_ESCAPE);
   while(!IsStopped()  // waiting for the user's command to end the demo
   && TerminalInfoInteger(TERMINAL_KEYSTATE_ESCAPE) == start)
   {
      Sleep(1000);
   }
   
   Comment("");
   ObjectDelete(0, BitmapObject);
   ResourceFree(ResName);
}
The script creates a new resource in the specified color mode and assigns it to the OBJPROP_BMPFILE
property of an object of type OBJ_BITMAP_LABEL. Next, the script waits for the user to explicitly stop
the script or press Esc and then deletes the object (by calling Obj ectDelete) and the resource using the
ResourceFree function. Note that deleting an object does not automatically delete the resource. That is
why we need the ResourceFree function which we will discuss in the next section.
If we don't call ResourceFree, then dynamic resources remain in the terminal's memory even after the
MQL program terminates, right up until the terminal is closed. This makes it possible to use them as
repositories or a means for exchanging information between MQL programs.
A dynamic resource created using the second form of ResourceCreate does not have to carry an image.
The data array may contain arbitrary data if we don't use it for rendering. In this case, it is important
to set the COLOR_FORMAT_XRGB_NOALPHA scheme. We will show such an example at some point.
In the meantime, let's check how the ARGBbitmap.mq5 script works.
The above picture "argb.bmp" contains information about transparency: the upper left corner has a
completely transparent background, and the transparency fades out diagonally towards the lower right
corner.
The following images show the results of running the script in three different modes.

---

## Page 1580

Part 7. Advanced language tools
1 580
7.1  Resources
Image output in color format COLOR_FORMAT_XRGB_N OALPHA
Image output in color format COLOR_FORMAT_ARGB_RAW

---

## Page 1581

Part 7. Advanced language tools
1 581 
7.1  Resources
Image output in color format COLOR_FORMAT_ARGB_N ORMALIZE
7.1 .6 Deleting dynamic resources: ResourceFree
The ResourceFree function removes the previously created dynamic resource and frees the memory it
occupies. If you don't call ResourceFree, the dynamic resource will remain in memory until the end of
the current terminal session. This can be used as a convenient way to store data, but for regular work
with images, it is recommended to release them when the need for them disappears.
Graphical objects attached to the resource being deleted will be displayed correctly even after its
deletion. However, newly created graphical objects (OBJ_BITMAP and OBJ_BITMAP_LABEL) will no
longer be able to use the deleted resource.
bool ResourceFree(const string resource)
The resource name is set in the resource parameter and must start with "::".
The function returns an indicator of success (true) or error (false).
The function deletes only dynamic resources created by the given MQL program, but not "third-party"
ones.
In the previous section, we saw an example of the script ARGBbitmap.mq5, which called ResourceFree
upon completion of its operation.
7.1 .7 Reading and modifying resource data: ResourceReadImage
The ResourceReadImage function allows reading the data of the resource created by the
ResourceCreate function or embedded into the executable at compile time according to the #resource

---

## Page 1582

Part 7. Advanced language tools
1 582
7.1  Resources
directive. Despite the suffix "Image" in the name, the function works with any data arrays, including
custom ones (see the example of Reservoir.mq5 below).
bool ResourceReadImage(const string resource, uint &data[], uint &width, uint &height)
The name of the resource is specified in the resource parameter. To access your own resources, the
short form "::resource_name" is sufficient. To read a resource from another compiled file, you need the
full name followed by the path according to the path resolution rules described in the section on
resources. In particular, a path starting with a backslash means the path from the MQL5 root folder
(this way "\\path\\filename.ex5::resource_name" is searched for in the file /MQL5/path/filename.ex5
under the name "resource_name"), and the path without this leading character means the path relative
to the folder where the executed program is located.
The internal information of the resource will be written into the receiving data array, and the width and
height parameters will receive, respectively, the width and height, that is, the size of the array
(width*height) indirectly. Separately, width and height are only relevant if the image is stored in the
resource. The array must be dynamic or fixed, but of sufficient size. Otherwise, we will get a
SMALL_ARRAY (5052) error.
If in the future you want to create a graphic resource based on the data array, then the source
resource should use the COLOR_FORMAT_ARGB_NORMALIZE or COLOR_FORMAT_XRGB_NOALPHA
color format. If the data array contains arbitrary application data, use
COLOR_FORMAT_XRGB_NOALPHA.
As a first example, let's consider the script ResourceReadImage.mq5. It demonstrates several aspects
of working with graphic resources:
• Creating an image resource from an external file
• Reading and modifying the data of this image in another dynamically created resource
• Preserving created resources in the terminal memory between script launches
• Using resources in objects on the chart
• Deleting an object and resources
Image modifying in this particular case means the inversion of all colors (as the most visual).
All of the above methods of work are performed in three stages: each stage is performed in one run of
the script. The script determines the current stage by analyzing the available resources and the object:
1 .In the absence of the required graphic resources, the script will create them (one original image
and one inverted image).
2. If there are resources but there is no graphic object, the script will create an object with two
images from the first step for on/off states (they can be switched by mouse click).
3. If there is an object, the script will delete the object and resources.
The main function of the script starts by defining the names of the resources and of the object on the
chart.

---

## Page 1583

Part 7. Advanced language tools
1 583
7.1  Resources
void OnStart()
{
   const static string resource = "::Images\\pseudo.bmp";
   const static string inverted = resource + "_inv";
   const static string object = "object";
   ...
Note that we have chosen a name for the original resource that looks like the location of the bmp file in
the standard Images folder, but there is no such file. This emphasizes the virtual nature of resources
and allows you to make substitutions to meet technical requirements or to make it difficult to reverse
engineer your programs.
The next ResourceReadImage call is used to check if the resource already exists. In the initial state (on
the first run), we will get a negative result (false) and start the first step: we create the original
resource from the file "\\Images\\dollar.bmp", and then invert it in a new resource with the "_inv"
suffix.
   uint data[], width, height;
   // check for resource existence
   if(!PRTF(ResourceReadImage(resource, data, width, height)))
   {
      Print("Initial state: Creating 2 bitmaps");
      PRTF(ResourceCreate(resource, "\\Images\\dollar.bmp")); // try "argb.bmp"
      ResourceCreateInverted(resource, inverted);
   }
   ...
The source code of the helper function ResourceCreateInverted will be presented below.
If the resource is found (second run), the script checks for the existence of the object and, if
necessary, creates it, including setting properties with image resources in the ShowBitmap function
(see below).
   else
   {
      Print("Resources (bitmaps) are detected");
      if(PRTF(ObjectFind(0, object) < 0))
      {
         Print("Active state: Creating object to draw 2 bitmaps");
         ShowBitmap(object, resource, inverted);
      }
      ...
If both the resources and the object are already on the chart, then we are at the final stage and must
remove all resources.

---

## Page 1584

Part 7. Advanced language tools
1 584
7.1  Resources
      else
      {
         Print("Cleanup state: Removing object and resources");
         PRTF(ObjectDelete(0, object));
         PRTF(ResourceFree(resource));
         PRTF(ResourceFree(inverted));
      }
   }
}
The ResourceCreateInverted function uses the ResourceReadImage call to get an array of pixels and
then inverts the color into them using the '^' (XOR) operator and an operand with all singular bits in the
color components.
bool ResourceCreateInverted(const string resource, const string inverted)
{
   uint data[], width, height;
   PRTF(ResourceReadImage(resource, data, width, height));
   for(int i = 0; i < ArraySize(data); ++i)
   {
      data[i] = data[i] ^ 0x00FFFFFF;
   }
   return PRTF(ResourceCreate(inverted, data, width, height, 0, 0, 0,
      COLOR_FORMAT_ARGB_NORMALIZE));
}
The new array data is transferred to ResourceCreate to create the second image.
The ShowBitmap function creates a graphic object in the usual way (in the lower right corner of the
graph) and sets its properties for on and off states to the original and inverted images, respectively.
void ShowBitmap(const string name, const string resourceOn, const string resourceOff = NULL)
{
   ObjectCreate(0, name, OBJ_BITMAP_LABEL, 0, 0, 0);
   
   ObjectSetString(0, name, OBJPROP_BMPFILE, 0, resourceOn);
   if(resourceOff != NULL) ObjectSetString(0, name, OBJPROP_BMPFILE, 1, resourceOff);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, 50);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 50);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_RIGHT_LOWER);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_RIGHT_LOWER);
}
Since the newly created object is off by default, we will first see the inverted image and we can switch
it to the original one on a mouse click. But let's remind you that our script performs actions step by
step, and therefore, before the image appears on the chart, the script must be run twice. At all stages,
the current status and actions performed (along with a success or error indication) are logged.
After the first launch, the following entries will appear in the log:

---

## Page 1585

Part 7. Advanced language tools
1 585
7.1  Resources
ResourceReadImage(resource,data,width,height)=false / RESOURCE_NOT_FOUND(4016)
Initial state: Creating 2 bitmaps
ResourceCreate(resource,\Images\dollar.bmp)=true / ok
ResourceReadImage(resource,data,width,height)=true / ok
ResourceCreate(inverted,data,width,height,0,0,0,COLOR_FORMAT_XRGB_NOALPHA)=true / ok
The logs indicate that the resources have not been found and that's why the script has created them.
After the second run, the log will say that resources have been found (which were left in memory from
the previous run of the script) but the object is not there yet, and the script will create it based on the
resources.
ResourceReadImage(resource,data,width,height)=true / ok
Resources (bitmaps) are detected
ObjectFind(0,object)<0=true / OBJECT_NOT_FOUND(4202)
Active state: Creating object to draw 2 bitmaps
We will see an object and an image on the chart. Switching states is available by mouse click (events
about changes of the state are not handled here).
Inverted and original images in an object on a chart
Finally, during the third run, the script will detect the object and delete all its developments.
ResourceReadImage(resource,data,width,height)=true / ok
Resources (bitmaps) are detected
ObjectFind(0,object)<0=false / ok
Cleanup state: Removing object and resources
ObjectDelete(0,object)=true / ok
ResourceFree(resource)=true / ok
ResourceFree(inverted)=true / ok
Then you can repeat the cycle.
The second example of the section will consider the use of resources for storing arbitrary application
data, that is, a kind of clipboard inside the terminal (in theory, there can be any number of such
buffers, since each of them is a separate named resource). Due to the universality of the problem, we
will create the Reservoir class with the main functionality (in the file Reservoir.mqh), and on its basis we
will write a demo script (Reservoir.mq5).
Before "diving" directly into Reservoir, let's introduce an auxiliary union ByteOverlay which will be
required often. A union will allow any simple built-in type (including simple structures) to be converted
to a byte array and vice versa. By "simple" we mean all built-in numeric types, date and time,
enumerations, color, and boolean flags. However, objects and dynamic arrays are no longer simple and

---

## Page 1586

Part 7. Advanced language tools
1 586
7.1  Resources
will not be supported by our new storage (due to technical limitations of the platform). Strings are also
not considered simple but for them, we will make an exception and will process them in a special way.
template<typename T>
union ByteOverlay
{
   uchar buffer[sizeof(T)];
   T value;
   
   ByteOverlay(const T &v)
   {
      value = v;
   }
   
   ByteOverlay(const uchar &bytes[], const int offset = 0)
   {
      ArrayCopy(buffer, bytes, 0, offset, sizeof(T));
   }
};
As we know, resources are built on the basis of arrays of type uint, so we describe such an array
(storage) in the Reservoir class. There we will add all the data to be subsequently written to the
resource. The current position in the array where data is written or read from is stored in the offset
field.
class Reservoir
{
   uint storage[];
   int offset;
public:
   Reservoir(): offset(0) { }
   ...
To place an array of data of arbitrary type into storage, you can use the template method packArray.
In the first half of it, we convert the passed array into a byte array using ByteOverlay.
   template<typename T>
   int packArray(const T &data[])
   {
      const int bytesize = ArraySize(data) * sizeof(T); // TODO: check for overflow
      uchar buffer[];
      ArrayResize(buffer, bytesize);
      for(int i = 0; i < ArraySize(data); ++i)
      {
         ByteOverlay<T> overlay(data[i]);
         ArrayCopy(buffer, overlay.buffer, i * sizeof(T));
      }
      ...
In the second half, we convert the byte array into a sequence of uint values, which are written in
storage with an offset. The number of required elements uint is determined by taking into account
whether there is a remainder after dividing the size of the data in bytes by the size of uint: optionally
we add one additional element.

---

## Page 1587

Part 7. Advanced language tools
1 587
7.1  Resources
      const int size = bytesize / sizeof(uint) + (bool)(bytesize % sizeof(uint));
      ArrayResize(storage, offset + size + 1);
      storage[offset] = bytesize;       // write the size of the data before the data
      for(int i = 0; i < size; ++i)
      {
         ByteOverlay<uint> word(buffer, i * sizeof(uint));
         storage[offset + i + 1] = word.value;
      }
      
      offset = ArraySize(storage);
      
      return offset;
   }
Before the data itself, we write the size of the data in bytes: this is the smallest possible protocol for
error checking when recovering data. In the future, it would be possible to write the typename(T) data
in the storage as well.
The method returns the current position in the storage after writing.
Based on packArray, it's easy to implement a method to save strings:
   int packString(const string text)
   {
      uchar data[];
      StringToCharArray(text, data, 0, -1, CP_UTF8);
      return packArray(data);
   }
There is also an option to store a separate number:
   template<typename T>
   int packNumber(const T number)
   {
      T array[1] = {number};
      return packArray(array);
   }
A method for restoring an array of arbitrary type T from the storage of type uint "loses" all operations
in the opposite direction. If inconsistencies are found in the readable type and amount of data with the
storage, the method returns 0 (an error sign). In normal mode, the current position in the array
storage is returned (it is always greater than 0 if something was successfully read).

---

## Page 1588

Part 7. Advanced language tools
1 588
7.1  Resources
   template<typename T>
   int unpackArray(T &output[])
   {
      if(offset >= ArraySize(storage)) return 0; // out of array bounds
      const int bytesize = (int)storage[offset];
      if(bytesize % sizeof(T) != 0) return 0;    // wrong data type
      if(bytesize > (ArraySize(storage) - offset) * sizeof(uint)) return 0;
      
      uchar buffer[];
      ArrayResize(buffer, bytesize);
      for(int i = 0, k = 0; i < ArraySize(storage) - 1 - offset
         && k < bytesize; ++i, k += sizeof(uint))
      {
         ByteOverlay<uint> word(storage[i + 1 + offset]);
         ArrayCopy(buffer, word.buffer, k);
      }
      
      int n = bytesize / sizeof(T);
      n = ArrayResize(output, n);
      for(int i = 0; i < n; ++i)
      {
         ByteOverlay<T> overlay(buffer, i * sizeof(T));
         output[i] = overlay.value;
      }
      
      offset += 1 + bytesize / sizeof(uint) + (bool)(bytesize % sizeof(uint));
      
      return offset;
   }
Unpacking strings and numbers is done by calling unpackArray.

---

## Page 1589

Part 7. Advanced language tools
1 589
7.1  Resources
   int unpackString(string &output)
   {
      uchar bytes[];
      const int p = unpackArray(bytes);
      if(p == offset)
      {
         output = CharArrayToString(bytes, 0, -1, CP_UTF8);
      }
      return p;
   }
   
   template<typename T>
   int unpackNumber(T &number)
   {
      T array[1] = {};
      const int p = unpackArray(array);
      number = array[0];
      return p;
   }
Simple helper methods allow you to find out the size of the storage and the current position in it, as
well as clear it.
   int size() const
   {
      return ArraySize(storage);
   }
   
   int cursor() const
   {
      return offset;
   }
   
   void clear()
   {
      ArrayFree(storage);
      offset = 0;
   }
Now we come to the most interesting: interaction with resources.
Having filled the storage array with application data, it is easy to "move" it to a provided resource.
   bool submit(const string resource)
   {
      return ResourceCreate(resource, storage, ArraySize(storage), 1,
         0, 0, 0, COLOR_FORMAT_XRGB_NOALPHA);
   }
Also, we can just read data from a resource into an internal array storage.

---

## Page 1590

Part 7. Advanced language tools
1 590
7.1  Resources
   bool acquire(const string resource)
   {
      uint width, height;
      if(ResourceReadImage(resource, storage, width, height))
      {
         return true;
      }
      return false;
   }
We will show in the script Reservoir.mq5, how to use it.
In the first half of OnStart, we describe the name for the storage resource and the class object
Reservoir, and then sequentially "pack" into this object a string, structure MqlTick, and number double.
The structure is "wrapped" in an array of one element to explicitly demonstrate the packArray method.
In addition, we will then need to compare the restored data with the original ones, and MQL5 does not
provide the '==' operator for structures. Therefore it will be more convenient to use the ArrayCompare
function.
#include <MQL5Book/Reservoir.mqh>
#include <MQL5Book/PRTF.mqh>
   
void OnStart()
{
   const string resource = "::reservoir";
   
   Reservoir res1;
   string message = "message1";     // string to write to the resource
   PRTF(res1.packString(message));
   
   MqlTick tick1[1];                // add a simple structure
   SymbolInfoTick(_Symbol, tick1[0]);
   PRTF(res1.packArray(tick1));
   PRTF(res1.packNumber(DBL_MAX));  // real number
   ...
When all the necessary data is "packed" into the object, write it to the resource and clear the object.
   res1.submit(resource);           // create a resource with storage data
   res1.clear();                    // clear the object, but not the resource
In the second half of OnStart let's perform the reverse operations of reading data from the resource.

---

## Page 1591

Part 7. Advanced language tools
1 591 
7.1  Resources
   string reply;                    // new variable for message
   MqlTick tick2[1];                // new structure for tick
   double result;                   // new variable for number
   
   PRTF(res1.acquire(resource));    // connect the object to the given resource
   PRTF(res1.unpackString(reply));  // read line
   PRTF(res1.unpackArray(tick2));   // read simple structure
   PRTF(res1.unpackNumber(result)); // read number
   
   // output and compare data element by element
   PRTF(reply);
   PRTF(ArrayCompare(tick1, tick2));
   ArrayPrint(tick2);
   PRTF(result == DBL_MAX);
   
   // make sure the storage is read completely
   PRTF(res1.size());
   PRTF(res1.cursor());
   ...
In the end, we clean up the resource, since this is a test. In practical tasks, an MQL program will most
likely leave the created resource in memory so that it can be read by other programs. In the naming
hierarchy, resources are declared nested in the program that created them. Therefore, for access from
other programs, you must specify the name of the resource along with the name of the program and
optionally the path (if the program-creator and the program-reader are in different folders). For
example, to read a newly created resource from outside, the full path "\\Scripts\\MQL5Book\\p7\
\Reservoir.ex5::reservoir" will do the job.
   PrintFormat("Cleaning up local storage '%s'", resource);
   ResourceFree(resource);
}
Since all major method calls are controlled by the PRTF macro, when we run the script, we will see a
detailed progress "report" in the log.
res1.packString(message)=4 / ok
res1.packArray(tick1)=20 / ok
res1.packNumber(DBL_MAX)=23 / ok
res1.acquire(resource)=true / ok
res1.unpackString(reply)=4 / ok
res1.unpackArray(tick2)=20 / ok
res1.unpackNumber(result)=23 / ok
reply=message1 / ok
ArrayCompare(tick1,tick2)=0 / ok
                 [time]   [bid]   [ask] [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.05.19 23:09:32 1.05867 1.05873 0.0000        0 1653001772050       6       0.00000
result==DBL_MAX=true / ok
res1.size()=23 / ok
res1.cursor()=23 / ok
Cleaning up local storage '::reservoir'
The data was successfully copied to the resource and then restored from there.

---

## Page 1592

Part 7. Advanced language tools
1 592
7.1  Resources
Programs can use this approach to exchange bulky data that does not fit in custom messages
(eventsCHARTEVENT_CUSTOM+). It is enough to send in a string parameter sparam the name of the
resource to read. To post back data, create your own resource with it and send a response message.
7.1 .8 Saving images to a file: ResourceSave
The MQL5 API allows you to write a resource to a BMP file using the ResourceSave function. The
framework currently only supports image resources.
bool ResourceSave(const string resource, const string filename)
The resource and filename parameters specify the name of the resource and file, respectively. The
resource name must start with "::". The file name may contain a path relative to the folder MQL5/Files.
If necessary, the function will create all intermediate subdirectories. If the specified file exists, it will be
overwritten.
The function returns true in case of success.
To test the operation of this function, it is desirable to create an original image. We have exactly the
right image for this.
As part of the study of OOP, in the chapter Classes and interfaces, we started a series of examples
about graphic shapes: from the very first version Shapes1 .mq5 in the section about Class definition to
the last version Shapes6.mq5 in the section about Nested types. Drawing was not available to us then,
until the chapter on graphical objects, where we were able to implement visualization in the script
Obj ectShapesDraw.mq5. Now, after studying the graphical resources, it's time for another "upgrade".
In the new version of the script ResourceShapesDraw.mq5 we will draw the shapes. To make it easier to
analyze the changes compared to the previous version, we will keep the same set of shapes: rectangle,
square, oval, circle, and triangle. This is done to give an example, and not because something limits us
in drawing: on the contrary, there is a potential for expanding the set of shapes, visual effects and
labeling. We'll look at the features in a few examples, starting with the current one. However, please
note that it is not possible to demonstrate the full range of applications within the scope of this book.
After the shapes are generated and drawn, we save the resulting resource to a file.
The basis of the shape class hierarchy is the Shape class which had a draw method.
class Shape
{
public:
   ...
   virtual void draw() = 0;
   ...
}
In derived classes, it was implemented on the basis of graphic objects, with calls to Obj ectCreate and
subsequent setup of objects using Obj ectSet functions. The shared canvas of such a drawing was the
chart itself.
Now we need to paint pixels in some shared resource according to the particular shape. It is desirable
to allocate a common resource and methods for modifying pixels in it into a separate class or, better,
an interface.

---

## Page 1593

Part 7. Advanced language tools
1 593
7.1  Resources
An abstract entity will allow us not to make links with the method of creating and configuring the
resource. In particular, our next implementation will place the resource in an OBJ_BITMAP_LABEL
object (as we have already done in this chapter), and for some, it may be enough to generate images in
memory and save to disk without plotting (as many traders like to periodically capture states charts).
Let's call the interface Drawing.
interface Drawing
{
   void point(const float x1, const float y1, const uint pixel);
   void line(const int x1, const int y1, const int x2, const int y2, const color clr);
   void rect(const int x1, const int y1, const int x2, const int y2, const color clr);
};
Here are just three of the most basic methods for drawing, which are enough for this case.
The point method is public (which makes it possible to put a separate point), but in a sense, it is low-
level since all the others will be implemented through it. That is why the coordinates in it are real, and
the content of the pixel is a ready-made value of the uint type. This will allow, if necessary, to apply
various anti-aliasing algorithms so that the shapes do not look stepped due to pixelation. Here we will
not touch on this issue.
Taking into account an interface, the Shape::draw method turns into the following one:
virtual void draw(Drawing *drawing) = 0;
Then, in the Rectangle class, it's very easy to delegate the drawing of the rectangle to a new interface.
class Rectangle : public Shape
{
protected:
   int dx, dy; // size (width, height)
   ...
public:
   void draw(Drawing *drawing) override
   {
 // x, y - anchor point (center) in Shape
      drawing.rect(x – dx / 2, y – dy / 2, x + dx / 2, y + dy / 2, backgroundColor);
   }
};
More efforts are required to draw an ellipse.

---

## Page 1594

Part 7. Advanced language tools
1 594
7.1  Resources
class Ellipse : public Shape
{
protected:
   int dx, dy; // large and small radii
   ...
public:
   void draw(Drawing *drawing) override
   {
      // (x, y) - center
      const int hh = dy * dy;
      const int ww = dx * dx;
      const int hhww = hh * ww;
      int x0 = dx;
      int step = 0;
      
      // main horizontal diameter
      drawing.line(x - dx, y, x + dx, y, backgroundColor);
      
      // horizontal lines in the upper and lower half, symmetrically decreasing in length
      for(int j = 1; j <= dy; j++)
      {
         for(int x1 = x0 - (step - 1); x1 > 0; --x1)
         {
            if(x1 * x1 * hh + j * j * ww <= hhww)
            {
               step = x0 - x1;
               break;
            }
         }
         x0 -= step;
         drawing.line(x - x0, y - j, x + x0, y - j, backgroundColor);
         drawing.line(x - x0, y + j, x + x0, y + j, backgroundColor);
      }
   }
};
Finally, for the triangle, the rendering is implemented as follows.

---

## Page 1595

Part 7. Advanced language tools
1 595
7.1  Resources
class Triangle: public Shape
{
protected:
   int dx;  // one size, because triangles are equilateral 
   ...
public:
   virtual void draw(Drawing *drawing) override
   {
      // (x, y) - center
      // R = a * sqrt(3) / 3
      // p0: x, y + R
      // p1: x - R * cos(30), y - R * sin(30)
      // p2: x + R * cos(30), y - R * sin(30)
      // Pythagorean height: dx * dx = dx * dx / 4 + h * h
      // sqrt(dx * dx * 3/4) = h
      const double R = dx * sqrt(3) / 3;
      const double H = sqrt(dx * dx * 3 / 4);
      const double angle = H / (dx / 2);
      
      // main vertical line (triangle height)
      const int base = y + (int)(R - H);
      drawing.line(x, y + (int)R, x, base, backgroundColor);
      
      // smaller vertical lines left and right, symmetrical
      for(int j = 1; j <= dx / 2; ++j)
      {
         drawing.line(x - j, y + (int)(R - angle * j), x - j, base, backgroundColor);
         drawing.line(x + j, y + (int)(R - angle * j), x + j, base, backgroundColor);
      }
   }
};
Now let's turn to the MyDrawing class which is derived from the Drawing interface. This is MyDrawing
that must, guided by calls to interface methods in shapes, ensure that a certain resource is displayed in
a bitmap. Therefore the class describes variables for the names of the graphical object (obj ect) and
resource (sheet), as well as the data array of type uint to store the image. In addition, we moved the
shapes array of shapes, which was previously declared in the OnStart handler. Since MyDrawing is
responsible for drawing all shapes, it is better to manage their set here.
class MyDrawing: public Drawing
{
   const string object;     // object with bitmap
   const string sheet;      // resource
   uint data[];             // pixels
   int width, height;       // dimensions
   AutoPtr<Shape> shapes[]; // figures/shapes
   const uint bg;           // background color
   ...
In the constructor, we create a graphical object for the size of the entire chart and allocate memory
for the data array. The canvas is filled with zeros (meaning "black transparency") or whatever value is
passed in the background parameter, after which a resource is created based on it. By default, the

---

## Page 1596

Part 7. Advanced language tools
1 596
7.1  Resources
resource name starts with the letter 'D' and includes the ID of the current chart, but you can specify
something else.
public:
   MyDrawing(const uint background = 0, const string s = NULL) :
      object((s == NULL ? "Drawing" : s)),
      sheet("::" + (s == NULL ? "D" + (string)ChartID() : s)), bg(background)
   {
      width = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
      height = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
      ArrayResize(data, width * height);
      ArrayInitialize(data, background);
   
      ResourceCreate(sheet, data, width, height, 0, 0, width, COLOR_FORMAT_ARGB_NORMALIZE);
      
      ObjectCreate(0, object, OBJ_BITMAP_LABEL, 0, 0, 0);
      ObjectSetInteger(0, object, OBJPROP_XDISTANCE, 0);
      ObjectSetInteger(0, object, OBJPROP_YDISTANCE, 0);
      ObjectSetInteger(0, object, OBJPROP_XSIZE, width);
      ObjectSetInteger(0, object, OBJPROP_YSIZE, height);
      ObjectSetString(0, object, OBJPROP_BMPFILE, sheet);
   }
The calling code can find out the name of the resource using the resource method.
   string resource() const
   {
      return sheet;
   }
The resource and object are removed in the destructor.
   ~MyDrawing()
   {
      ResourceFree(sheet);
      ObjectDelete(0, object);
   }
The push method fills the array of shapes.
   Shape *push(Shape *shape)
   {
      shapes[EXPAND(shapes)] = shape;
      return shape;
   }
The draw method draws the shapes. It simply calls the draw method of each shape in the loop and then
updates the resource and the chart.

---

## Page 1597

Part 7. Advanced language tools
1 597
7.1  Resources
   void draw()
   {
      for(int i = 0; i < ArraySize(shapes); ++i)
      {
         shapes[i][].draw(&this);
      }
      ResourceCreate(sheet, data, width, height, 0, 0, width, COLOR_FORMAT_ARGB_NORMALIZE);
      ChartRedraw();
   }
Below are the most important methods which are the methods of the Drawing interface and which
actually implement drawing.
Let's start with the point method, which we present in a simplified form for now (we will deal with the
improvements later).
   virtual void point(const float x1, const float y1, const uint pixel) override
   {
      const int x_main = (int)MathRound(x1);
      const int y_main = (int)MathRound(y1);
      const int index = y_main * width + x_main;
      if(index >= 0 && index < ArraySize(data))
      {
         data[index] = pixel;
      }
   }
Based on point, it is easy to implement line drawing. When the coordinates of the start and end points
match in one of the dimensions, we use the rect method to draw since a straight line is a degenerate
case of a rectangle of unit thickness.

---

## Page 1598

Part 7. Advanced language tools
1 598
7.1  Resources
   virtual void line(const int x1, const int y1, const int x2, const int y2, const color clr) override
   {
      if(x1 == x2) rect(x1, y1, x1, y2, clr);
      else if(y1 == y2) rect(x1, y1, x2, y1, clr);
      else
      {
         const uint pixel = ColorToARGB(clr);
         double angle = 1.0 * (y2 - y1) / (x2 - x1);
         if(fabs(angle) < 1) // step along the axis with the largest distance, x
         {
            const int sign = x2 > x1 ? +1 : -1;
            for(int i = 0; i <= fabs(x2 - x1); ++i)
            {
               const float p = (float)(y1 + sign * i * angle);
               point(x1 + sign * i, p, pixel);
            }
         }
         else // or y-step
         {
            const int sign = y2 > y1 ? +1 : -1;
            for(int i = 0; i <= fabs(y2 - y1); ++i)
            {
               const float p = (float)(x1 + sign * i / angle);
               point(p, y1 + sign * i, pixel);
            }
         }
      }
   }
And here is the rect method.
   virtual void rect(const int x1, const int y1, const int x2, const int y2, const color clr) override
   {
      const uint pixel = ColorToARGB(clr);
      for(int i = fmin(x1, x2); i <= fmax(x1, x2); ++i)
      {
         for(int j = fmin(y1, y2); j <= fmax(y1, y2); ++j)
         {
            point(i, j, pixel);
         }
      }
   }
Now we need to modify the OnStart handler, and the script will be ready.
First, we set up the chart (hide all elements). In theory, this is not necessary: it is left to match with
the prototype script.

---

## Page 1599

Part 7. Advanced language tools
1 599
7.1  Resources
void OnStart()
{
   ChartSetInteger(0, CHART_SHOW, false);
   ...
Next, we describe the object of the MyDrawing class, generate a predefined number of random shapes
(everything remains unchanged here, including the addRandomShape generator and the FIGURES
macro equal to 21 ), draw them in the resource, and display them in the object on the chart.
   MyDrawing raster;
   
   for(int i = 0; i < FIGURES; ++i)
   {
      raster.push(addRandomShape());
   }
   
   raster.draw(); // display the initial state
   ...
In the example Obj ectShapesDraw.mq5, we started an endless loop in which we moved the pieces
randomly. Let's repeat this trick here. Here we will need to add the MyDrawing class since the array of
shapes is stored inside it. Let's write a simple method shake.
class MyDrawing: public Drawing
{
public:
   ...
   void shake()
   {
      ArrayInitialize(data, bg);
      for(int i = 0; i < ArraySize(shapes); ++i)
      {
         shapes[i][].move(random(20) - 10, random(20) - 10);
      }
   }
   ...
};
Then, in OnStart, we can use the new method in a loop until the user stops the animation.
void OnStart()
{
   ...
   while(!IsStopped())
   {
      Sleep(250);
      raster.shake();
      raster.draw();
   }
   ...
}
At this point, the functionality of the previous example is virtually repeated. But we need to add image
saving to a file. So let's add an input parameter SaveImage.

---

## Page 1600

Part 7. Advanced language tools
1 600
7.1  Resources
input bool SaveImage = false;
When it is set to true, check the performance of the ResourceSave function.
void OnStart()
{
   ...
   if(SaveImage)
   {
      const string filename = "temp.bmp";
      if(ResourceSave(raster.resource(), filename))
      {
         Print("Bitmap image saved: ", filename);
      }
      else
      {
         Print("Can't save image ", filename, ", ", E2S(_LastError));
      }
   }
}
Also, since we are talking about input variables, let the user select a background and pass the resulting
value to the MyDrawing constructor.
input color BackgroundColor = clrNONE;
void OnStart()
{
   ...
   MyDrawing raster(BackgroundColor != clrNONE ? ColorToARGB(BackgroundColor) : 0);
   ...
}
So, everything is ready for the first test.
If you run the script ResourceShapesDraw.mq5, the chart will form an image like the following.

---

