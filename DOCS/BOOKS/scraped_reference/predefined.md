---
title: "The predefined Variables"
url: "https://www.mql5.com/en/docs/predefined"
hierarchy: []
scraped_at: "2025-11-28 09:31:26"
---

# The predefined Variables

[MQL5 Reference](/en/docs "MQL5 Reference")Predefined Variables

* [\_AppliedTo](/en/docs/predefined/_appliedto "_AppliedTo")
* [\_Digits](/en/docs/predefined/_digits "_Digits")
* [\_Point](/en/docs/predefined/_point "_Point")
* [\_LastError](/en/docs/predefined/_lasterror "_LastError")
* [\_Period](/en/docs/predefined/_period "_Period")
* [\_RandomSeed](/en/docs/predefined/_randomseed "_RandomSeed")
* [\_StopFlag](/en/docs/predefined/_stopflag "_StopFlag")
* [\_Symbol](/en/docs/predefined/_symbol "_Symbol")
* [\_UninitReason](/en/docs/predefined/_uninitreason "_UninitReason")
* [\_IsX64](/en/docs/predefined/_isx64 "_IsX64")

# The predefined Variables

For each executable mql5-program a set of predefined variables is supported, which reflect the state of the current price chart by the moment a mql5-program (Expert Advisor, script or custom indicator) is started.

Values of predefined variables are set by the client terminal before a mql5-program is started. Predefined variables are constant and cannot be changed from a mql5-program. As exception, there is a special variable \_LastError, which can be reset to 0 by the [ResetLastError](/en/docs/common/resetlasterror) function.

| Variable | Value |
| --- | --- |
| [\_AppliedTo](/en/docs/predefined/_appliedto) | The \_AppliedTo variable allows finding out the type of data, used for indicator calculation |
| [\_Digits](/en/docs/predefined/_digits) | Number of decimal places |
| [\_Point](/en/docs/predefined/_point) | Size of the current symbol point in the quote currency |
| [\_LastError](/en/docs/predefined/_lasterror) | The last error code |
| [\_Period](/en/docs/predefined/_period) | Timeframe of the current chart |
| [\_RandomSeed](/en/docs/predefined/_randomseed) | Current status of the generator of pseudo-random integers |
| [\_StopFlag](/en/docs/predefined/_stopflag) | Program stop flag |
| [\_Symbol](/en/docs/predefined/_symbol) | Symbol name of the current chart |
| [\_UninitReason](/en/docs/predefined/_uninitreason) | Uninitialization reason code |
| [\_IsX64](/en/docs/predefined/_isx64) | The \_IsX64 variable allows finding out the bit version of the terminal, in which an MQL5 application is running |

Predefined variables cannot be defined in a library. A library uses such variables that are defined in program from which this library is called.

[Testing Trading Strategies](/en/docs/runtime/testing "Testing Trading Strategies")

[\_AppliedTo](/en/docs/predefined/_appliedto "_AppliedTo")