---
title: "Scripts and services"
url: "https://www.mql5.com/en/book/applications/script_service"
hierarchy: []
scraped_at: "2025-11-28 09:48:14"
---

# Scripts and services

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")Scripts and services

* [Scripts](/en/book/applications/script_service/scripts "Scripts")
* [Services](/en/book/applications/script_service/services "Services")
* [Restrictions for scripts and services](/en/book/applications/script_service/script_service_limitations "Restrictions for scripts and services")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Scripts and services

In this chapter, we will summarize and present the full technical information about the scripts and services that we have already started to get acquainted with in the previous parts of the book.

Scripts and services have the same principles for organizing and executing program code. As we know, their main function [OnStart](/en/book/applications/runtime/runtime_onstart) is also the only one. Scripts and services cannot process [other events](/en/book/applications/runtime/runtime_events_overview).

However, there are a couple of significant differences. Scripts are executed in the context of a chart and have direct access to its properties through built-in variables such as \_Symbol, \_Period, \_Point, and others. We will study them in the section [Chart properties](/en/book/applications/charts/charts_main_properties). Services, on the other hand, work on their own, not tied to any windows, although they have the ability to analyze all charts using special functions (the same [Chart functions](/en/book/applications/charts/charts_properties_overview) can be used in other types of programs: scripts, indicators, and Expert Advisors).

On the other hand, the created instances of the service are automatically restored by the terminal in the next sessions. In other words, the service, once started, always remains running until the user stops it. In contrast, the script is deleted when the terminal is turned off or the chart is closed.

Please note that the service is executed in the terminal, like all other types of MQL programs, and therefore closing the terminal also stops the service. The active service will resume the next time you start the terminal. Uninterrupted operation of MQL programs can only be ensured by a constantly running terminal, for example, on a VPS.

In scripts and services, you can set [General properties of programs](/en/book/basis/preprocessor/preprocessor_properties) using #property directives. In addition to them, there are properties that are specific to scripts and services; we will discuss them in the next two sections.

The scripts that are currently running on the charts are listed in the same list that shows running Expert Advisors — in the Experts dialog opened with the Expert List command of the chart context menu. From there, they can be forcibly removed from the chart.

Services can only be managed from the Navigator window.

[Programmatic removal of Expert Advisors and scripts: ExpertRemove](/en/book/applications/runtime/runtime_remove "Programmatic removal of Expert Advisors and scripts: ExpertRemove")

[Scripts](/en/book/applications/script_service/scripts "Scripts")