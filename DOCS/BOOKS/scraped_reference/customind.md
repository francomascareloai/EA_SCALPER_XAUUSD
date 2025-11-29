---
title: "Custom Indicators"
url: "https://www.mql5.com/en/docs/customind"
hierarchy: []
scraped_at: "2025-11-28 09:30:18"
---

# Custom Indicators

[MQL5 Reference](/en/docs "MQL5 Reference")Custom Indicators

* [Indicator Styles in Examples](/en/docs/customind/indicators_examples "Indicator Styles in Examples")
* [Connection between Indicator Properties and Functions](/en/docs/customind/propertiesandfunctions "Connection between Indicator Properties and Functions")
* [SetIndexBuffer](/en/docs/customind/setindexbuffer "SetIndexBuffer")
* [IndicatorSetDouble](/en/docs/customind/indicatorsetdouble "IndicatorSetDouble")
* [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger "IndicatorSetInteger")
* [IndicatorSetString](/en/docs/customind/indicatorsetstring "IndicatorSetString")
* [PlotIndexSetDouble](/en/docs/customind/plotindexsetdouble "PlotIndexSetDouble")
* [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger "PlotIndexSetInteger")
* [PlotIndexSetString](/en/docs/customind/plotindexsetstring "PlotIndexSetString")
* [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger "PlotIndexGetInteger")

# Custom Indicators

This is the group functions used in the creation of custom indicators. These functions can't be used when writing Expert Advisors and Scripts.

| Function | Action |
| --- | --- |
| [SetIndexBuffer](/en/docs/customind/setindexbuffer) | Binds the specified indicator buffer with one-dimensional dynamic [array](/en/docs/basis/variables#array_define) of the [double](/en/docs/basis/types/double) type |
| [IndicatorSetDouble](/en/docs/customind/indicatorsetdouble) | Sets the value of an indicator property of the [double](/en/docs/basis/types/double) type |
| [IndicatorSetInteger](/en/docs/customind/indicatorsetinteger) | Sets the value of an indicator property of the [int](/en/docs/basis/types/integer/integertypes) type |
| [IndicatorSetString](/en/docs/customind/indicatorsetstring) | Sets the value of an indicator property of the [string](/en/docs/basis/types/stringconst) type |
| [PlotIndexSetDouble](/en/docs/customind/plotindexsetdouble) | Sets the value of an indicator line property of the type [double](/en/docs/basis/types/double) |
| [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger) | Sets the value of an indicator line property of the [int](/en/docs/basis/types/integer/integertypes) type |
| [PlotIndexSetString](/en/docs/customind/plotindexsetstring) | Sets the value of an indicator line property of the [string](/en/docs/basis/types/stringconst) type |
| [PlotIndexGetInteger](/en/docs/customind/plotindexgetinteger) | Returns the value of an indicator line property of the [integer](/en/docs/basis/types/integer/integertypes) type |

[Indicator properties](/en/docs/customind/propertiesandfunctions) can be set using the compiler directives or using functions. To better understand this, it is recommended that you study [indicator styles in examples](/en/docs/customind/indicators_examples).

All the necessary calculations of a custom indicator must be placed in the predetermined function [OnCalculate()](/en/docs/event_handlers/oncalculate). If you use a short form of the OnCalculate() function call, like

| |
| --- |
| int OnCalculate (const int rates\_total, const int prev\_calculated, const int begin, const double& price[]) |

then the rates\_total variable contains the value of the total number of elements of the price[] array, passed as an input parameter for calculating indicator values.

Parameter prev\_calculated is the result of the execution of OnCalculate() at the previous call; it allows organizing a saving algorithm for calculating indicator values. For example, if the current value rates\_total = 1000, prev\_calculated = 999, then perhaps it's enough to make calculations only for one value of each indicator buffer.

If the information about the size of the input array price would have been unavailable, then it would lead to the necessity to make calculations for 1000 values of each indicator buffer. At the first call of OnCalculate() value prev\_calculated = 0. If the price[] array has changed somehow, then in this case prev\_calculated is also equal to 0.

The begin parameter shows the number of initial values of the price array, which don't contain data for calculation. For example, if values of Accelerator Oscillator (for which the first 37 values aren't calculated) were used as an input parameter, then begin = 37. For example, let's consider a simple indicator:

| |
| --- |
| #property indicator\_chart\_window #property indicator\_buffers 1 #property indicator\_plots   1 //---- plot Label1 #property indicator\_label1  "Label1" #property indicator\_type1   DRAW\_LINE #property indicator\_color1  clrRed #property indicator\_style1  STYLE\_SOLID #property indicator\_width1  1 //--- indicator buffers double         Label1Buffer[]; //+------------------------------------------------------------------+ //| Custom indicator initialization function                         | //+------------------------------------------------------------------+ void OnInit()   { //--- indicator buffers mapping    SetIndexBuffer(0,Label1Buffer,INDICATOR\_DATA); //---   } //+------------------------------------------------------------------+ //| Custom indicator iteration function                              | //+------------------------------------------------------------------+ int OnCalculate(const int rates\_total,                 const int prev\_calculated,                 const int begin,                 const double &price[])     { //---    Print("begin = ",begin,"  prev\_calculated = ",prev\_calculated,"  rates\_total = ",rates\_total); //--- return value of prev\_calculated for next call    return(rates\_total);   } |

Drag it from the "Navigator" window to the window of the Accelerator Oscillator indicator and we indicate that calculations will be made based on the values of the previous indicator:

![Calculating an indicator on values of the previously attached indicator](/en/docs/img/previousindicatorsdata.png "Calculating an indicator on values of the previously attached indicator")

As a result, the first call of OnCalculate() the value of prev\_calculated will be equal to zero, and in further calls it will be equal to the rates\_total value (until the number of bars on the price chart increases).

![The begin parameter shows the number of initial bars, on which values are omitted](/en/docs/img/beginparameteronprevindabsent.png "The begin parameter shows the number of initial bars, on which values are omitted")

The value of the begin parameter will be exactly equal to the number of initial bars, for which the values of the Accelerator indicator aren't calculated according to the logic of this indicator. If we look at the source code of the custom indicator Accelerator.mq5, we'll see the following lines in the [OnInit()](/en/docs/event_handlers/oninit) function:

| |
| --- |
| //--- sets first bar from which index will be drawn    PlotIndexSetInteger(0,PLOT\_DRAW\_BEGIN,37); |

Using the function [PlotIndexSetInteger](/en/docs/customind/plotindexsetinteger)(0, [PLOT\_DRAW\_BEGIN](/en/docs/constants/indicatorconstants/drawstyles#enum_plot_property_integer), empty\_first\_values), we set the number of non-existing first values in the zero indicator array of a custom indicator, which we don't need to accept for calculation (empty\_first\_values). Thus, we have mechanisms to:

1. set the number of initial values of an indicator, which shouldn't be used for calculations in another custom indicator;
2. get information on the number of first values to be ignored when you call another custom indicator, without going into the logic of its calculations.

[FolderClean](/en/docs/files/folderclean "FolderClean")

[Indicator Styles in Examples](/en/docs/customind/indicators_examples "Indicator Styles in Examples")