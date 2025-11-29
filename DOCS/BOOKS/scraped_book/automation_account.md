---
title: "Trading account information"
url: "https://www.mql5.com/en/book/automation/account"
hierarchy: []
scraped_at: "2025-11-28 09:48:01"
---

# Trading account information

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Trading automation](/en/book/automation "Trading automation")Trading account information

* [Overview of functions for getting account properties](/en/book/automation/account/account_info_overview "Overview of functions for getting account properties")
* [Identifying the account, client, server, and broker](/en/book/automation/account/account_number_identity "Identifying the account, client, server, and broker")
* [Account type: real, demo or contest](/en/book/automation/account/account_real_demo_contest "Account type: real, demo or contest")
* [Account currency](/en/book/automation/account/account_currency "Account currency")
* [Account type: netting or hedging](/en/book/automation/account/account_netting_hedge "Account type: netting or hedging")
* [Restrictions and permissions for account operations](/en/book/automation/account/account_limits_and_restrictions "Restrictions and permissions for account operations")
* [Account margin settings](/en/book/automation/account/account_margin "Account margin settings")
* [Current financial performance of the account](/en/book/automation/account/account_state "Current financial performance of the account")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Trading account information

In this chapter, we will study the last important aspect of the trading environment of MQL programs and, specifically, Expert Advisors, which we will develop in detail in the next few chapters. Let's talk about a trading account.

Having a valid account and an active connection to it are a necessary condition for the functioning of most MQL programs. Until now, we have not focused on this, but getting quotes, ticks, and, in general, the ability to open a workable chart implies a successful connection to a trading account.

In the context of Expert Advisors, an account additionally reflects the financial condition of the client, accumulates the trading history and determines the specific modes allowed for trading.

The MQL5 API allows you to get the properties of an account, starting with its number and ending with the current profit. All of them are read-only in the terminal and are installed by the broker on the server.

The terminal can only be connected to one account at a time. All MQL programs work with this account. As we have already noted in the section [Features of starting and stopping programs of various types](/en/book/applications/runtime/runtime_lifecycle), switching an account initiates a reload of the indicators and Expert Advisors attached to the charts. However, in the OnDeinit handler, the program can find the reason for deinitialization, which, when switching the account, will be equal to [REASON\_ACCOUNT](/en/book/applications/runtime/runtime_oninit_ondeinit).

[Using Depth of Market data in applied algorithms](/en/book/automation/marketbook/marketbook_application "Using Depth of Market data in applied algorithms")

[Overview of functions for getting account properties](/en/book/automation/account/account_info_overview "Overview of functions for getting account properties")