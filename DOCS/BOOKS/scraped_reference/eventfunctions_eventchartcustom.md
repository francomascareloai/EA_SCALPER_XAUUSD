---
title: "EventChartCustom"
url: "https://www.mql5.com/en/docs/eventfunctions/eventchartcustom"
hierarchy: []
scraped_at: "2025-11-28 09:31:08"
---

# EventChartCustom

[MQL5 Reference](/en/docs "MQL5 Reference")[Working with Events](/en/docs/eventfunctions "Working with Events")EventChartCustom

* [EventSetMillisecondTimer](/en/docs/eventfunctions/eventsetmillisecondtimer "EventSetMillisecondTimer")
* [EventSetTimer](/en/docs/eventfunctions/eventsettimer "EventSetTimer")
* [EventKillTimer](/en/docs/eventfunctions/eventkilltimer "EventKillTimer")
* EventChartCustom

# EventChartCustom

The function generates a custom event for the specified chart.

| |
| --- |
| bool  EventChartCustom(    long    chart\_id,            // identifier of the event receiving chart    ushort  custom\_event\_id,     // event identifier    long    lparam,              // parameter of type long    double  dparam,              // parameter of type double    string  sparam               // string parameter of the event    ); |

Parameters

chart\_id

[in] Chart identifier. 0 means the current chart.

custom\_event\_id

[in] ID of the user events. This identifier is automatically added to the value [CHARTEVENT\_CUSTOM](/en/docs/constants/chartconstants/enum_chartevents) and converted to the integer type.

lparam

[in] Event parameter of the long type passed to the [OnChartEvent](/en/docs/event_handlers/onchartevent) function.

dparam

[in] Event parameter of the double type passed to the [OnChartEvent](/en/docs/event_handlers/onchartevent) function.

sparam

[in] Event parameter of the string type passed to the [OnChartEvent](/en/docs/event_handlers/onchartevent) function. If the string is longer than 63 characters, it is truncated.

Return Value

Returns true if a custom event has been successfully placed in the events queue of the chart that receives the events. In case of failure, it returns false. Use [GetLastError()](/en/docs/check/getlasterror) to get an error code.

Note

An Expert Advisor or indicator attached to the specified chart handles the event using the function [OnChartEvent](/en/docs/event_handlers/onchartevent)(int event\_id, long& lparam, double& dparam, string& sparam).

For each type of event, the input parameters of the OnChartEvent() function have definite values that are required for the processing of this event. The events and values passed through this parameters are listed in the below table.

| Event | Value of the id parameter | Value of the lparam parameter | Value of the dparam parameter | Value of the sparam parameter |
| --- | --- | --- | --- | --- |
| Event of a keystroke | CHARTEVENT\_KEYDOWN | code of a pressed key | Repeat count (the number of times the keystroke is repeated as a result of the user holding down the key) | The string value of a bit mask describing the status of keyboard buttons |
| Mouse event (if property [CHART\_EVENT\_MOUSE\_MOVE](/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer)=true is set for the chart) | CHARTEVENT\_MOUSE\_MOVE | the X coordinate | the Y coordinate | The string value of a bit mask describing the status of mouse buttons |
| Event of graphical object creation (if [CHART\_EVENT\_OBJECT\_CREATE](/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer)=true is set for the chart) | CHARTEVENT\_OBJECT\_CREATE | — | — | Name of the created graphical object |
| Event of change of an object property via the properties dialog | CHARTEVENT\_OBJECT\_CHANGE | — | — | Name of the modified graphical object |
| Event of graphical object deletion (if [CHART\_EVENT\_OBJECT\_DELETE](/en/docs/constants/chartconstants/enum_chart_property#enum_chart_property_integer)=true is set for the chart) | CHARTEVENT\_OBJECT\_DELETE | — | — | Name of the deleted graphical object |
| Event of a mouse click on the chart | CHARTEVENT\_CLICK | the X coordinate | the Y coordinate | — |
| Event of a mouse click in a graphical object belonging to the chart | CHARTEVENT\_OBJECT\_CLICK | the X coordinate | the Y coordinate | Name of the graphical object, on which the event occurred |
| Event of a graphical object dragging using the mouse | CHARTEVENT\_OBJECT\_DRAG | — | — | Name of the moved graphical object |
| Event of the finished text editing in the entry box of the LabelEdit graphical object | CHARTEVENT\_OBJECT\_ENDEDIT | — | — | Name of the LabelEdit graphical object, in which text editing has completed |
| Event of changes on a chart | CHARTEVENT\_CHART\_CHANGE | — | — | — |
| ID of the user event under the N number | CHARTEVENT\_CUSTOM+N | Value set by the EventChartCustom() function | Value set by the EventChartCustom() function | Value set by the EventChartCustom() function |

Example:

| |
| --- |
| //+------------------------------------------------------------------+ //|                                            ButtonClickExpert.mq5 | //|                        Copyright 2009, MetaQuotes Software Corp. | //|                                              https://www.mql5.com | //+------------------------------------------------------------------+ #property copyright "2009, MetaQuotes Software Corp." #property link      "https://www.mql5.com" #property version   "1.00"   string buttonID="Button"; string labelID="Info"; int broadcastEventID=5000; //+------------------------------------------------------------------+ //| Expert initialization function                                   | //+------------------------------------------------------------------+ int OnInit()   { //--- Create a button to send custom events    ObjectCreate(0,buttonID,OBJ\_BUTTON,0,100,100);    ObjectSetInteger(0,buttonID,OBJPROP\_COLOR,clrWhite);    ObjectSetInteger(0,buttonID,OBJPROP\_BGCOLOR,clrGray);    ObjectSetInteger(0,buttonID,OBJPROP\_XDISTANCE,100);    ObjectSetInteger(0,buttonID,OBJPROP\_YDISTANCE,100);    ObjectSetInteger(0,buttonID,OBJPROP\_XSIZE,200);    ObjectSetInteger(0,buttonID,OBJPROP\_YSIZE,50);    ObjectSetString(0,buttonID,OBJPROP\_FONT,"Arial");    ObjectSetString(0,buttonID,OBJPROP\_TEXT,"Button");    ObjectSetInteger(0,buttonID,OBJPROP\_FONTSIZE,10);    ObjectSetInteger(0,buttonID,OBJPROP\_SELECTABLE,0);   //--- Create a label for displaying information    ObjectCreate(0,labelID,OBJ\_LABEL,0,100,100);    ObjectSetInteger(0,labelID,OBJPROP\_COLOR,clrRed);    ObjectSetInteger(0,labelID,OBJPROP\_XDISTANCE,100);    ObjectSetInteger(0,labelID,OBJPROP\_YDISTANCE,50);    ObjectSetString(0,labelID,OBJPROP\_FONT,"Trebuchet MS");    ObjectSetString(0,labelID,OBJPROP\_TEXT,"No information");    ObjectSetInteger(0,labelID,OBJPROP\_FONTSIZE,20);    ObjectSetInteger(0,labelID,OBJPROP\_SELECTABLE,0);   //---    return(INIT\_SUCCEEDED);   } //+------------------------------------------------------------------+ //| Expert deinitialization function                                 | //+------------------------------------------------------------------+ void OnDeinit(const int reason)   { //---    ObjectDelete(0,buttonID);    ObjectDelete(0,labelID);   } //+------------------------------------------------------------------+ //| Expert tick function                                             | //+------------------------------------------------------------------+ void OnTick()   { //---     } //+------------------------------------------------------------------+ void OnChartEvent(const int id,                   const long &lparam,                   const double &dparam,                   const string &sparam)   { //--- Check the event by pressing a mouse button    if(id==CHARTEVENT\_OBJECT\_CLICK)      {       string clickedChartObject=sparam;       //--- If you click on the object with the name buttonID       if(clickedChartObject==buttonID)         {          //--- State of the button - pressed or not          bool selected=ObjectGetInteger(0,buttonID,OBJPROP\_STATE);          //--- log a debug message          Print("Button pressed = ",selected);          int customEventID; // Number of the custom event to send          string message;    // Message to be sent in the event          //--- If the button is pressed          if(selected)            {             message="Button pressed";             customEventID=CHARTEVENT\_CUSTOM+1;            }          else // Button is not pressed            {             message="Button in not pressed";             customEventID=CHARTEVENT\_CUSTOM+999;            }          //--- Send a custom event "our" chart          EventChartCustom(0,customEventID-CHARTEVENT\_CUSTOM,0,0,message);          ///--- Send a message to all open charts          BroadcastEvent(ChartID(),0,"Broadcast Message");          //--- Debug message          Print("Sent an event with ID = ",customEventID);         }       ChartRedraw();// Forced redraw all chart objects      }   //--- Check the event belongs to the user events    if(id>CHARTEVENT\_CUSTOM)      {       if(id==broadcastEventID)         {          Print("Got broadcast message from a chart with id = "+lparam);         }       else         {          //--- We read a text message in the event          string info=sparam;          Print("Handle the user event with the ID = ",id);          //--- Display a message in a label          ObjectSetString(0,labelID,OBJPROP\_TEXT,sparam);          ChartRedraw();// Forced redraw all chart objects         }      }   } //+------------------------------------------------------------------+ //| sends broadcast event to all open charts                         | //+------------------------------------------------------------------+ void BroadcastEvent(long lparam,double dparam,string sparam)   {    int eventID=broadcastEventID-CHARTEVENT\_CUSTOM;    long currChart=ChartFirst();    int i=0;    while(i<CHARTS\_MAX)                 // We have certainly no more than CHARTS\_MAX open charts      {       EventChartCustom(currChart,eventID,lparam,dparam,sparam);       currChart=ChartNext(currChart); // We have received a new chart from the previous       if(currChart==-1) break;        // Reached the end of the charts list       i++;// Do not forget to increase the counter      }   } //+------------------------------------------------------------------+ |

See also

[Events of the client terminal](/en/docs/runtime/event_fire), [Event handler functions](/en/docs/basis/function/events)

[EventKillTimer](/en/docs/eventfunctions/eventkilltimer "EventKillTimer")

[Working with OpenCL](/en/docs/opencl "Working with OpenCL")