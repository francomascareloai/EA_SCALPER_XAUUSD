---
title: "Trading automation"
url: "https://www.mql5.com/en/book/automation"
hierarchy: []
scraped_at: "2025-11-28 10:14:19"
---

# Trading automation

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")Trading automation

* [Financial instruments and Market Watch](/en/book/automation/symbols "Financial instruments and Market Watch")
* [Depth of Market](/en/book/automation/marketbook "Depth of Market")
* [Trading account information](/en/book/automation/account "Trading account information")
* [Creating Expert Advisors](/en/book/automation/experts "Creating Expert Advisors")
* [Testing and optimization of Expert Advisors](/en/book/automation/tester "Testing and optimization of Expert Advisors")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Trading automation

In this part, we will study the most complex and important component of the MQL5 API which allows the automation of trading actions.

We will start by describing the entities without which it is impossible to write a proper Expert Advisor. These include [financial symbols](/en/book/automation/symbols) and [trading account](/en/book/automation/account) settings.

Then we will look at built-in [trading functions](/en/book/automation/experts) and data structures, along with robot-specific [events](/en/book/automation/experts/experts_ontick) and operating modes. In particular, the key feature of Expert Advisors is integration with the [tester](/en/book/automation/tester), which allows users to evaluate financial performance and optimize trading strategies. We will consider the internal optimization mechanisms and optimization management through the API.

The strategy tester is an essential tool for developing MQL programs since it provides the ability to debug programs in various modes, including bars and ticks, based on modeled or real ticks, with or without visualization of the price stream.

We've already tried to [test indicators](/en/book/applications/indicators_make/indicators_test) in visual mode. However, the set of testing parameters is limited for indicators. When developing Expert Advisors, we will have access to the full range of tester capabilities.

In addition, we will be introduced to a new form of market information: the [Depth of Market](/en/book/automation/marketbook) and its software interface.

 

| | |
| --- | --- |
| MQL5 Programming for Traders — Source Codes from the Book. Part 6 | [MQL5 Programming for Traders — Source Codes from the Book. Part 6](https://www.mql5.com/en/code/45595) |
| Примеры из книги также доступны в публичном проекте \MQL5\Shared Projects\MQL5Book. | Examples from the book are also available in the [public project](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#public) \MQL5\Shared Projects\MQL5Book |

[Generation of custom events](/en/book/applications/events/events_custom "Generation of custom events")

[Financial instruments and Market Watch](/en/book/automation/symbols "Financial instruments and Market Watch")