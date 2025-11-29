---
title: "Trade Functions"
url: "https://www.mql5.com/en/docs/trading"
hierarchy: []
scraped_at: "2025-11-28 09:31:22"
---

# Trade Functions

[MQL5 Reference](/en/docs "MQL5 Reference")Trade Functions

* [OrderCalcMargin](/en/docs/trading/ordercalcmargin "OrderCalcMargin")
* [OrderCalcProfit](/en/docs/trading/ordercalcprofit "OrderCalcProfit")
* [OrderCheck](/en/docs/trading/ordercheck "OrderCheck")
* [OrderSend](/en/docs/trading/ordersend "OrderSend")
* [OrderSendAsync](/en/docs/trading/ordersendasync "OrderSendAsync")
* [PositionsTotal](/en/docs/trading/positionstotal "PositionsTotal")
* [PositionGetSymbol](/en/docs/trading/positiongetsymbol "PositionGetSymbol")
* [PositionSelect](/en/docs/trading/positionselect "PositionSelect")
* [PositionSelectByTicket](/en/docs/trading/positionselectbyticket "PositionSelectByTicket")
* [PositionGetDouble](/en/docs/trading/positiongetdouble "PositionGetDouble")
* [PositionGetInteger](/en/docs/trading/positiongetinteger "PositionGetInteger")
* [PositionGetString](/en/docs/trading/positiongetstring "PositionGetString")
* [PositionGetTicket](/en/docs/trading/positiongetticket "PositionGetTicket")
* [OrdersTotal](/en/docs/trading/orderstotal "OrdersTotal")
* [OrderGetTicket](/en/docs/trading/ordergetticket "OrderGetTicket")
* [OrderSelect](/en/docs/trading/orderselect "OrderSelect")
* [OrderGetDouble](/en/docs/trading/ordergetdouble "OrderGetDouble")
* [OrderGetInteger](/en/docs/trading/ordergetinteger "OrderGetInteger")
* [OrderGetString](/en/docs/trading/ordergetstring "OrderGetString")
* [HistorySelect](/en/docs/trading/historyselect "HistorySelect")
* [HistorySelectByPosition](/en/docs/trading/historyselectbyposition "HistorySelectByPosition")
* [HistoryOrderSelect](/en/docs/trading/historyorderselect "HistoryOrderSelect")
* [HistoryOrdersTotal](/en/docs/trading/historyorderstotal "HistoryOrdersTotal")
* [HistoryOrderGetTicket](/en/docs/trading/historyordergetticket "HistoryOrderGetTicket")
* [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble "HistoryOrderGetDouble")
* [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger "HistoryOrderGetInteger")
* [HistoryOrderGetString](/en/docs/trading/historyordergetstring "HistoryOrderGetString")
* [HistoryDealSelect](/en/docs/trading/historydealselect "HistoryDealSelect")
* [HistoryDealsTotal](/en/docs/trading/historydealstotal "HistoryDealsTotal")
* [HistoryDealGetTicket](/en/docs/trading/historydealgetticket "HistoryDealGetTicket")
* [HistoryDealGetDouble](/en/docs/trading/historydealgetdouble "HistoryDealGetDouble")
* [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger "HistoryDealGetInteger")
* [HistoryDealGetString](/en/docs/trading/historydealgetstring "HistoryDealGetString")

# Trade Functions

This is the group of functions intended for managing trading activities.

Before you proceed to study the trade functions of the platform, you must have a clear understanding of the basic terms: order, deal and position:

* An order is an instruction given to a broker to buy or sell a financial instrument. There are two main types of orders: Market and Pending. In addition, there are special Take Profit and Stop Loss levels.
* A deal is the commercial exchange (buying or selling) of a financial security. Buying is executed at the demand price (Ask), and Sell is performed at the supply price (Bid). A deal can be opened as a result of market order execution or pending order triggering. Note that in some cases, execution of an order can result in several deals.
* A position is a trade obligation, i.e. the number of bought or sold contracts of a financial instrument. A long position is financial security bought expecting the security price go higher. A short position is an obligation to supply a security expecting the price will fall in future.

General information about trading operations is available in the [client terminal help](https://www.metatrader5.com/en/terminal/help/trading/general_concept "MetaTrader 5 Help").

Trading functions can be used in Expert Advisors and scripts. Trading functions can be called only if in the properties of the Expert Advisor or script the "Allow live trading" checkbox is enabled.

Trading can be allowed or prohibited depending on various factors described in the [Trade Permission](/en/docs/runtime/tradepermission) section.

| Function | Action |
| --- | --- |
| [OrderCalcMargin](/en/docs/trading/ordercalcmargin) | Calculates the margin required for the specified order type, in the deposit currency |
| [OrderCalcProfit](/en/docs/trading/ordercalcprofit) | Calculates the profit based on the parameters passed, in the deposit currency |
| [OrderCheck](/en/docs/trading/ordercheck) | Checks if there are enough funds to execute the required [trade operation](/en/docs/constants/tradingconstants/enum_trade_request_actions). |
| [OrderSend](/en/docs/trading/ordersend) | Sends [trade requests](/en/docs/constants/structures/mqltraderequest) to a server |
| [OrderSendAsync](/en/docs/trading/ordersendasync) | Asynchronously sends [trade requests](/en/docs/constants/tradingconstants/enum_trade_request_actions) without waiting for the trade response of the trade server |
| [PositionsTotal](/en/docs/trading/positionstotal) | Returns the number of open positions |
| [PositionGetSymbol](/en/docs/trading/positiongetsymbol) | Returns the symbol corresponding to the open position |
| [PositionSelect](/en/docs/trading/positionselect) | Chooses an open position for further working with it |
| [PositionSelectByTicket](/en/docs/trading/positionselectbyticket) | Selects a position to work with by the ticket number specified in it |
| [PositionGetDouble](/en/docs/trading/positiongetdouble) | Returns the requested property of an open position (double) |
| [PositionGetInteger](/en/docs/trading/positiongetinteger) | Returns the requested property of an open position (datetime or int) |
| [PositionGetString](/en/docs/trading/positiongetstring) | Returns the requested property of an open position (string) |
| [PositionGetTicket](/en/docs/trading/positiongetticket) | Returns the ticket of the position with the specified index in the list of open positions |
| [OrdersTotal](/en/docs/trading/orderstotal) | Returns the number of orders |
| [OrderGetTicket](/en/docs/trading/ordergetticket) | Return the ticket of a corresponding order |
| [OrderSelect](/en/docs/trading/orderselect) | Selects a order for further working with it |
| [OrderGetDouble](/en/docs/trading/ordergetdouble) | Returns the requested property of the order (double) |
| [OrderGetInteger](/en/docs/trading/ordergetinteger) | Returns the requested property of the order (datetime or int) |
| [OrderGetString](/en/docs/trading/ordergetstring) | Returns the requested property of the order (string) |
| [HistorySelect](/en/docs/trading/historyselect) | Retrieves the history of transactions and orders for the specified period of the server time |
| [HistorySelectByPosition](/en/docs/trading/historyselectbyposition) | Requests the history of deals with a specified [position identifier](/en/docs/constants/tradingconstants/positionproperties#enum_position_property_integer). |
| [HistoryOrderSelect](/en/docs/trading/historyorderselect) | Selects an order in the history for further working with it |
| [HistoryOrdersTotal](/en/docs/trading/historyorderstotal) | Returns the number of orders in the history |
| [HistoryOrderGetTicket](/en/docs/trading/historyordergetticket) | Return order ticket of a corresponding order in the history |
| [HistoryOrderGetDouble](/en/docs/trading/historyordergetdouble) | Returns the requested property of an order in the history (double) |
| [HistoryOrderGetInteger](/en/docs/trading/historyordergetinteger) | Returns the requested property of an order in the history (datetime or int) |
| [HistoryOrderGetString](/en/docs/trading/historyordergetstring) | Returns the requested property of an order in the history (string) |
| [HistoryDealSelect](/en/docs/trading/historydealselect) | Selects a deal in the history for further calling it through appropriate functions |
| [HistoryDealsTotal](/en/docs/trading/historydealstotal) | Returns the number of deals in the history |
| [HistoryDealGetTicket](/en/docs/trading/historydealgetticket) | Returns a ticket of a corresponding deal in the history |
| [HistoryDealGetDouble](/en/docs/trading/historydealgetdouble) | Returns the requested property of a deal in the history (double) |
| [HistoryDealGetInteger](/en/docs/trading/historydealgetinteger) | Returns the requested property of a deal in the history (datetime or int) |
| [HistoryDealGetString](/en/docs/trading/historydealgetstring) | Returns the requested property of a deal in the history (string) |

[ChartScreenShot](/en/docs/chart_operations/chartscreenshot "ChartScreenShot")

[OrderCalcMargin](/en/docs/trading/ordercalcmargin "OrderCalcMargin")