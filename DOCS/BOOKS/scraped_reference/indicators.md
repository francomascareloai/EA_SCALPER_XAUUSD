---
title: "Technical Indicator Functions"
url: "https://www.mql5.com/en/docs/indicators"
hierarchy: []
scraped_at: "2025-11-28 09:31:12"
---

# Technical Indicator Functions

[MQL5 Reference](/en/docs "MQL5 Reference")Technical Indicators

* [iAC](/en/docs/indicators/iac "iAC")
* [iAD](/en/docs/indicators/iad "iAD")
* [iADX](/en/docs/indicators/iadx "iADX")
* [iADXWilder](/en/docs/indicators/iadxwilder "iADXWilder")
* [iAlligator](/en/docs/indicators/ialligator "iAlligator")
* [iAMA](/en/docs/indicators/iama "iAMA")
* [iAO](/en/docs/indicators/iao "iAO")
* [iATR](/en/docs/indicators/iatr "iATR")
* [iBearsPower](/en/docs/indicators/ibearspower "iBearsPower")
* [iBands](/en/docs/indicators/ibands "iBands")
* [iBullsPower](/en/docs/indicators/ibullspower "iBullsPower")
* [iCCI](/en/docs/indicators/icci "iCCI")
* [iChaikin](/en/docs/indicators/ichaikin "iChaikin")
* [iCustom](/en/docs/indicators/icustom "iCustom")
* [iDEMA](/en/docs/indicators/idema "iDEMA")
* [iDeMarker](/en/docs/indicators/idemarker "iDeMarker")
* [iEnvelopes](/en/docs/indicators/ienvelopes "iEnvelopes")
* [iForce](/en/docs/indicators/iforce "iForce")
* [iFractals](/en/docs/indicators/ifractals "iFractals")
* [iFrAMA](/en/docs/indicators/iframa "iFrAMA")
* [iGator](/en/docs/indicators/igator "iGator")
* [iIchimoku](/en/docs/indicators/iichimoku "iIchimoku")
* [iBWMFI](/en/docs/indicators/ibwmfi "iBWMFI")
* [iMomentum](/en/docs/indicators/imomentum "iMomentum")
* [iMFI](/en/docs/indicators/imfi "iMFI")
* [iMA](/en/docs/indicators/ima "iMA")
* [iOsMA](/en/docs/indicators/iosma "iOsMA")
* [iMACD](/en/docs/indicators/imacd "iMACD")
* [iOBV](/en/docs/indicators/iobv "iOBV")
* [iSAR](/en/docs/indicators/isar "iSAR")
* [iRSI](/en/docs/indicators/irsi "iRSI")
* [iRVI](/en/docs/indicators/irvi "iRVI")
* [iStdDev](/en/docs/indicators/istddev "iStdDev")
* [iStochastic](/en/docs/indicators/istochastic "iStochastic")
* [iTEMA](/en/docs/indicators/itema "iTEMA")
* [iTriX](/en/docs/indicators/itrix "iTriX")
* [iWPR](/en/docs/indicators/iwpr "iWPR")
* [iVIDyA](/en/docs/indicators/ividya "iVIDyA")
* [iVolumes](/en/docs/indicators/ivolumes "iVolumes")

# Technical Indicator Functions

All functions like iMA, iAC, iMACD, iIchimoku etc. create a copy of the corresponding technical indicator in the global cache of the client terminal. If a copy of the indicator with such parameters already exists, the new copy is not created, and the counter of references to the existing copy increases.

These functions return the handle of the appropriate copy of the indicator. Further, using this handle, you can receive data calculated by the corresponding indicator. The corresponding buffer data (technical indicators contain calculated data in their internal buffers, which can vary from 1 to 5, depending on the indicator) can be copied to a mql5-program using the [CopyBuffer()](/en/docs/series/copybuffer) function.

You can't refer to the indicator data right after it has been created, because calculation of indicator values requires some time, so it's better to create indicator handles in OnInit(). Function [iCustom()](/en/docs/indicators/icustom) creates the corresponding custom indicator, and returns its handle in case it is successfully create. Custom indicators can contain up to 512 indicator buffers, the contents of which can also be obtained by the [CopyBuffer()](/en/docs/series/copybuffer) function, using the obtained handle.

There is a universal method for creating any technical indicator using the [IndicatorCreate()](/en/docs/series/indicatorcreate) function. This function accepts the following data as input parameters:

* symbol name;
* timeframe;
* type of the indicator to create;
* number of input parameters of the indicator;
* an array of [MqlParam](/en/docs/constants/structures/mqlparam) type containing all the necessary input parameters.

The computer memory can be freed from an indicator that is no more utilized, using the [IndicatorRelease()](/en/docs/series/indicatorrelease) function, to which the indicator handle is passed.

Note. Repeated call of the indicator function with the same parameters within one mql5-program does not lead to a multiple increase of the reference counter; the counter will be increased only once by 1. However, it's recommended to get the indicators handles in function [OnInit()](/en/docs/event_handlers/oninit) or in the class constructor, and further use these handles in other functions. The reference counter decreases when a mql5-program is deinitialized.

All indicator functions have at least 2 parameters - symbol and period. The [NULL](/en/docs/basis/types/void) value of the symbol means the current symbol, the 0 value of the period means the current [timeframe](/en/docs/constants/chartconstants/enum_timeframes).

| Function | Returns the handle of the indicator: |
| --- | --- |
| [iAC](/en/docs/indicators/iac) | Accelerator Oscillator |
| [iAD](/en/docs/indicators/iad) | Accumulation/Distribution |
| [iADX](/en/docs/indicators/iadx) | Average Directional Index |
| [iADXWilder](/en/docs/indicators/iadxwilder) | Average Directional Index by Welles Wilder |
| [iAlligator](/en/docs/indicators/ialligator) | Alligator |
| [iAMA](/en/docs/indicators/iama) | Adaptive Moving Average |
| [iAO](/en/docs/indicators/iao) | Awesome Oscillator |
| [iATR](/en/docs/indicators/iatr) | Average True Range |
| [iBearsPower](/en/docs/indicators/ibearspower) | Bears Power |
| [iBands](/en/docs/indicators/ibands) | Bollinger Bands® |
| [iBullsPower](/en/docs/indicators/ibullspower) | Bulls Power |
| [iCCI](/en/docs/indicators/icci) | Commodity Channel Index |
| [iChaikin](/en/docs/indicators/ichaikin) | Chaikin Oscillator |
| [iCustom](/en/docs/indicators/icustom) | Custom indicator |
| [iDEMA](/en/docs/indicators/idema) | Double Exponential Moving Average |
| [iDeMarker](/en/docs/indicators/idemarker) | DeMarker |
| [iEnvelopes](/en/docs/indicators/ienvelopes) | Envelopes |
| [iForce](/en/docs/indicators/iforce) | Force Index |
| [iFractals](/en/docs/indicators/ifractals) | Fractals |
| [iFrAMA](/en/docs/indicators/iframa) | Fractal Adaptive Moving Average |
| [iGator](/en/docs/indicators/igator) | Gator Oscillator |
| [iIchimoku](/en/docs/indicators/iichimoku) | Ichimoku Kinko Hyo |
| [iBWMFI](/en/docs/indicators/ibwmfi) | Market Facilitation Index by Bill Williams |
| [iMomentum](/en/docs/indicators/imomentum) | Momentum |
| [iMFI](/en/docs/indicators/imfi) | Money Flow Index |
| [iMA](/en/docs/indicators/ima) | Moving Average |
| [iOsMA](/en/docs/indicators/iosma) | Moving Average of Oscillator (MACD histogram) |
| [iMACD](/en/docs/indicators/imacd) | Moving Averages Convergence-Divergence |
| [iOBV](/en/docs/indicators/iobv) | On Balance Volume |
| [iSAR](/en/docs/indicators/isar) | Parabolic Stop And Reverse System |
| [iRSI](/en/docs/indicators/irsi) | Relative Strength Index |
| [iRVI](/en/docs/indicators/irvi) | Relative Vigor Index |
| [iStdDev](/en/docs/indicators/istddev) | Standard Deviation |
| [iStochastic](/en/docs/indicators/istochastic) | Stochastic Oscillator |
| [iTEMA](/en/docs/indicators/itema) | Triple Exponential Moving Average |
| [iTriX](/en/docs/indicators/itrix) | Triple Exponential Moving Averages Oscillator |
| [iWPR](/en/docs/indicators/iwpr) | Williams' Percent Range |
| [iVIDyA](/en/docs/indicators/ividya) | Variable Index Dynamic Average |
| [iVolumes](/en/docs/indicators/ivolumes) | Volumes |

[TextGetSize](/en/docs/objects/textgetsize "TextGetSize")

[iAC](/en/docs/indicators/iac "iAC")