# MQL5 Source Codes of Technical Indicators for MetaTrader 5
           - 61

Source: https://www.mql5.com/en/code/mt5/indicators/page61

* [![](https://c.mql5.com/i/sidebar/mt.svg)MetaTrader 5](/en/code/mt5)
  + [![](https://c.mql5.com/i/sidebar/expert.svg)Experts](/en/code/mt5/experts)
  + [![](https://c.mql5.com/i/sidebar/indicator.svg)Indicators](/en/code/mt5/indicators)
  + [![](https://c.mql5.com/i/sidebar/scripts.svg)Scripts](/en/code/mt5/scripts)
  + [![](https://c.mql5.com/i/sidebar/library.svg)Libraries](/en/code/mt5/libraries)
* [![](https://c.mql5.com/i/sidebar/mt.svg)MetaTrader 4](/en/code/mt4)
  + [![](https://c.mql5.com/i/sidebar/expert.svg)Experts](/en/code/mt4/experts)
  + [![](https://c.mql5.com/i/sidebar/indicator.svg)Indicators](/en/code/mt4/indicators)
  + [![](https://c.mql5.com/i/sidebar/scripts.svg)Scripts](/en/code/mt4/scripts)
  + [![](https://c.mql5.com/i/sidebar/library.svg)Libraries](/en/code/mt4/libraries)

* [![](https://c.mql5.com/i/sidebar/storage.svg)Storage](/en/code/storage)

Watch [how to download](https://youtu.be/rloNyFVtHuA?list=PLltlMLQ7OLeKwyQwC8FhiKwjl9syKhOCK) trading robots for free

Find us on [Twitter](https://x.com/mql5com)!  
 Join our fan page

Access the CodeBase from your [MetaTrader 5](https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/MetaTrader5.pkg.zip?utm_source=www.mql5.com&utm_campaign=download) terminal

Couldn't find the right code? Order it in the [Freelance](https://www.mql5.com/en/job) section

[How to Write](https://www.mql5.com/en/articles/100) an Expert Advisor or an Indicator

# MQL5 Source Codes of Technical Indicators for MetaTrader 5 - 61

![icon](https://c.mql5.com/i/code/code-icon.png)

MQL5 technical indicators analyze MetaTrader 5 price charts on Forex, as well as stock and commodity markets. Indicators define trend direction and power, overbought and oversold states, support and resistance levels. Underlying mathematical models provide objective assessment of the current market state allowing traders to accept or reject trading system's signals.

You can download and launch offered indicators in MetaTrader 5. The library of indicators is also available directly from MetaTrader 5 platform and MetaEditor development environment.

[Submit your code](/en/code/new)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

* Free trading apps
* Over 8,000 signals for copying
* Economic news for exploring financial markets

Registration
Log in

latin characters without spaces

a password will be sent to this email

An error occurred

* [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](/en/about/privacy) and [terms of use](/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

 

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

* [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)

[1](/en/code/mt5/indicators)[...](/en/code/mt5/indicators/page27)[54](/en/code/mt5/indicators/page54)[55](/en/code/mt5/indicators/page55)[56](/en/code/mt5/indicators/page56)[57](/en/code/mt5/indicators/page57)[58](/en/code/mt5/indicators/page58)[59](/en/code/mt5/indicators/page59)[60](/en/code/mt5/indicators/page60)[61](/en/code/mt5/indicators/page61)[62](/en/code/mt5/indicators/page62)[63](/en/code/mt5/indicators/page63)[64](/en/code/mt5/indicators/page64)[65](/en/code/mt5/indicators/page65)[66](/en/code/mt5/indicators/page66)[67](/en/code/mt5/indicators/page67)[68](/en/code/mt5/indicators/page68)[...](/en/code/mt5/indicators/page104)[139](/en/code/mt5/indicators/page139)

latest |

[best](/en/code/mt5/indicators/best)

[EMA levels](/en/code/20037 "EMA levels")

Instead of being in a hurry to follow the price with the stop loss, this indicator is adjusting a proposed stop loss only when it notices a trend change in the opposite direction. That way, it keeps the level intact while trending, and adjusts it when it "sees" that there is a possible trend change) and gives bigger "room" for an order to survive possible whipsaws in trends. Also, since it is estimating a trend, this indicator displays the periods when there is a trend in different color(s) in order to make it easier to decided what to do.

[DSL - extended stochastic](/en/code/20036 "DSL - extended stochastic")

The usual average that is used for stochastic calculation is simple Moving Average (SMA). This (extended) version allows you to use any of the 4 basic types of averages (default is SMA, but you can use EMA, SMMA or LWMA too) - some are "faster" then the default version (like EMA and LWMA versions) and SMMA is a bit "slower", but this way you can fine tune the "speed" to signals ratio.

[DSL - stochastic](/en/code/20035 "DSL - stochastic")

The DSL (Discontinued Signal Line) version of Stochastic does not use a moving average in a classical way for signals, but is instead calculating the signal lines depending on the value(s) of the stochastic. Thus, we are having two things : a signal line and a sort of levels that can be used for overbought and oversold estimation.

[DSL - WPR](/en/code/20034 "DSL - WPR")

The DSL version of Williams' Percent Range does not use fixed levels for oversold and overbought levels, but is having a sort of dynamic (discontinued signal lines) calculated to identify those levels. That makes it a bit more responsive to market changes and volatile markets.

[Nonlinear Kalman filter](/en/code/20028 "Nonlinear Kalman filter")

One more from the creations of John Ehlers - nonlinear Kalman filter.

[Kalman bands](/en/code/20027 "Kalman bands")

This is a conversion of Kalman bands originally developed by Igor Durkin. Values are the same as MetaTrader 4 version except that we are using possibilities that MetaTrader 4 does not have to make the indicator easier to use.

[DSL - TEMA MACD](/en/code/20019 "DSL - TEMA MACD")

Variation of a long known and useful MACD indicator using TEMA (Triple Exponential Moving Average) instead of using EMA (Exponential Moving Average) for MACD calculation, and DSL (Discontinued Signal Lines) and instead of using one signal line uses two.

[DSL - DEMA MACD](/en/code/20018 "DSL - DEMA MACD")

Variation of a long known and useful MACD indicator using DEMA (Double Exponential Moving Average) instead of using EMA (Exponential Moving Average) for MACD calculation, and DSL (Discontinued Signal Lines) and instead of using one signal line uses two. That way it sort of introduces levels as well as signal lines and, judging from tests, it seems to be better in avoiding false signals and it can be used in (short term) reversals detection.

[Stochastic RVI](/en/code/20011 "Stochastic RVI")

Stochastic and RVI (Relative Vigor Index) - both indicators measure overbought and oversold area of the market movement. This indicator combines them both in one single indicator - Stochastic of Relative Vigor Index.

[Inverse Fisher RVI](/en/code/20004 "Inverse Fisher RVI")

The Inverse Fisher Transform normalizes the values in the desired range (-1 to +1 in this case) which helps in assessing the overbought and oversold market conditions.

[Fisher RVI](/en/code/20003 "Fisher RVI")

This indicator has an addition of Fisher Transform to the RVI. The Fisher Transform enables traders to create a nearly Gaussian probability density function by normalizing prices. In essence, the transformation makes peak swings relatively rare events and unambiguously identifies price reversals on a chart. The technical indicator is commonly used by traders looking for extremely timely signals rather than lagging indicators.

[Stochastic Extended](/en/code/19992 "Stochastic Extended")

This version of Stochastic Oscillator allows you to use any of the 4 basic types of averages (default is SMA, but you can use EMA, SMMA or LWMA too) - some are "faster" then the default version (like EMA and LWMA versions) and SMMA is a bit "slower" but this way you can fine tune the "speed" to signals ratio.

[T3 Stochastic Momentum Index](/en/code/19988 "T3 Stochastic Momentum Index")

This version is doing the calculation in the same way as the original Stochastic Momentum Index, except in one very important part: instead of using EMA (Exponential Moving Average) for calculation, it is using T3. That produces a smoother result without adding any lag.

[Stochastic Momentum Index](/en/code/19986 "Stochastic Momentum Index")

The Stochastic Momentum Index (SMI) was developed by William Blau and was introduced in the January 1993 issue of Technical Analysis of Stocks & Commodities magazine. It incorporates an interesting twist on the popular Stochastic Oscillator. While the Stochastic Oscillator provides you with a value showing the distance the current close is relative to the recent x-period high/low range, the SMI shows you where the close is relative to the midpoint of the recent x-period high/low range.

[Directional Efficiency Ratio](/en/code/19985 "Directional Efficiency Ratio")

The Efficiency Ratio (ER) was first presented by Perry Kaufman in his 1995 book "Smarter Trading". It is calculated by dividing the price change over a period by the absolute sum of the price movements that occurred to achieve that change. The resulting ratio ranges between 0 and 1 with higher values representing a more efficient or trending market.

[TTM trend](/en/code/19956 "TTM trend")

The TTM (Trade The Markets) Trend is basically an easier way to look at candlesticks. It is the The Heikin-Ashi method. Literally translated Heikin is "average" or "balance,", while Ashi means "foot" or "bar." The TTM trend is a visual technique that eliminates the irregularities from a normal candlestick chart and offers a better picture of trends and consolidations.

[MACD TEMA](/en/code/19970 "MACD TEMA")

MACD TEMA is even a bit more "faster" than MACD DEMA so, depending on the parameters, in scalping mode (short calculating periods) or trending mode (when longer periods are used. Never forget that MACD is primarily a momentum indicator and that it is the main goal of MACD.

[MACD DEMA](/en/code/19969 "MACD DEMA")

MACD that is using DEMA fo calculation.

[Smoothed Kijun-Sen](/en/code/19946 "Smoothed Kijun-Sen")

The Kijun-Sen is a major indicator line and component of the Ichimoku Kinko Hyo indicator, also known as the Ichimoku cloud. It is generally used as a metric for medium-term momentum.

[CCI alternative](/en/code/19942 "CCI alternative")

Commodity Channel Index (CCI) is a versatile indicator that can be used to identify a new trend or warn of extreme conditions. Donald Lambert originally developed CCI to identify cyclical turns in commodities, but the indicator can be successfully applied to indices, ETFs, stocks, and other securities.

[Volume Rate of Change](/en/code/19922 "Volume Rate of Change")

The Volume Rate of Change indicator (VROC) measures the rate of change in volume over the past "n" sessions. In other words, the VROC measures the current volume by comparing it to the volume "n" periods or sessions ago.

[DevStops](/en/code/19927 "DevStops")

A variation of Deviation Stops (DevStops) indicator. Some are wrongly calling this version a Kase DevStops (which it is not - Kase DevStops indicator is calculated in a quite different way), but this version has its good points too and can be used in regular support/resistance mode. Additionally each DevStop value is colored according to the slope (trend) of the line - when all are aligned in the same direction, it can be treated as a confirmed trend change.

[Kase DevStops](/en/code/19926 "Kase DevStops")

Kase DevStops. What all of this boils down to is that we need to take variance and skew into consideration when we are establishing a system for setting stops. Three steps that we can take in order to both better define and to minimize the threshold of uncertainty in setting stops are: 1. Consideration of the variance or the standard deviation of range. 2. Consideration of the skew, or more simply, the amount at which range can spike in the opposite direction of the trend. 3. Reformation of our data to be more consistent (this step is examined in detail in Chapter 81, while minimizing the degree of uncertainty as much as possible).

[Smoothed Rate of Change](/en/code/19925 "Smoothed Rate of Change")

Smoothed Rate of Change (Smoothed-RoC) is a refinement of Rate of Change (RoC) indicator that was developed by Fred G Schutzman. It differs from the RoC in that it based on Exponential Moving Averages (EMAs) rather than on price closes. Like the RoC, Smoothed RoC is a leading Momentum indicator that can be used to determine the strength of a trend by determining if the trend is accelerating or decelerating. The Smoothed RoC does this by comparing the current EMA to value that the EMA was a specified periods ago. The use of EMAs rather than the price close eliminates the erratic tendencies of the RoC.

[Percentage Price Oscillator Extended](/en/code/19924 "Percentage Price Oscillator Extended")

The Percentage Price Oscillator Extended (PPO) is a technical Momentum indicator showing the relationship between two Moving Averages. To calculate the PPO, subtract the 26-day Exponential Moving Average (EMA) from the nine-day EMA, and then divide this difference by the 26-day EMA. The end result is a percentage that tells the trader where the short-term average is relative to the longer-term average.

[Percentage Price Oscillator](/en/code/19923 "Percentage Price Oscillator")

The Percentage Price Oscillator (PPO) is a technical Momentum indicator showing the relationship between two Moving Averages. To calculate the PPO, subtract the 26-day Exponential Moving Average (EMA) from the nine-day EMA, and then divide this difference by the 26-day EMA. The end result is a percentage that tells the trader where the short-term average is relative to the longer-term average.

[Woodies CCI](/en/code/19912 "Woodies CCI")

Woodies CCI is a momentum indicator that was developed by Ken Woods. It's based on a 14 period Commodity Channel Index (CCI).

[ATR Probability Levels](/en/code/19905 "ATR Probability Levels")

Probability levels based on ATR. "Probability" is calculated based on the projected Average True Range and previous period Close.

[HOPS and LOPS](/en/code/19881 "HOPS and LOPS")

HOPS and LOPS indicator. The "HOPS" and "LOPS" stand for High Of the Previous Session and Low Of the Previous Sessions.

[Rsi(var) with averages](/en/code/19865 "Rsi(var) with averages")

Rsi(var) with averages.

[Rsi(var)](/en/code/19860 "Rsi(var)")

RSI variation.

[Chandelier exit](/en/code/19875 "Chandelier exit")

Chandelier exit indicator is designed to keep traders in a trend and prevent an early exit as long as the trend extends. Typically, the Chandelier Exit will be above prices during a downtrend and below prices during an uptrend.

[Lot calculator - risk management tool](/en/code/19870 "Lot calculator - risk management tool")

This tool allows you to calculate the correct lot size of the next trade by following some simple money management rules.

[RAVI iFish](/en/code/19867 "RAVI iFish")

Range Action Verification Index (RAVI) with inverse Fisher transform.

[Chandes Quick Stick (Qstick)](/en/code/19858 "Chandes Quick Stick (Qstick)")

Chandes Quick Stick (Qstick)

[Ulcer Index](/en/code/19813 "Ulcer Index")

This Ulcer Index indicator was derived from the stock risk indicator by Peter Martin in the 1987 book "The Investors Guide to Fidelity Funds".

[Vertical Horizontal Filter](/en/code/19847 "Vertical Horizontal Filter")

The Vertical Horizontal Filter ("VHF") determines whether prices are in a trending phase or a congestion phase. The VHF was first presented by Adam White in an article published in the August, 1991 issue of Futures Magazine.

[Relative Momentum Index](/en/code/19851 "Relative Momentum Index")

Relative Momentum Index (RMI) is a variation of the RSI indicator. The RMI counts up and down days from the Close relative to the Close X days ago (where X is not limited to 1 as is required by the RSI) instead of counting up and down days from Close to Close as the RSI does.

[Moving slope rate of change - Extended](/en/code/19842 "Moving slope rate of change - Extended")

Extended version of MSROC indicator.

[Moving slope rate of change](/en/code/19840 "Moving slope rate of change")

Moving slope rate of change.

[1](/en/code/mt5/indicators)[...](/en/code/mt5/indicators/page27)[54](/en/code/mt5/indicators/page54)[55](/en/code/mt5/indicators/page55)[56](/en/code/mt5/indicators/page56)[57](/en/code/mt5/indicators/page57)[58](/en/code/mt5/indicators/page58)[59](/en/code/mt5/indicators/page59)[60](/en/code/mt5/indicators/page60)[61](/en/code/mt5/indicators/page61)[62](/en/code/mt5/indicators/page62)[63](/en/code/mt5/indicators/page63)[64](/en/code/mt5/indicators/page64)[65](/en/code/mt5/indicators/page65)[66](/en/code/mt5/indicators/page66)[67](/en/code/mt5/indicators/page67)[68](/en/code/mt5/indicators/page68)[...](/en/code/mt5/indicators/page104)[139](/en/code/mt5/indicators/page139)