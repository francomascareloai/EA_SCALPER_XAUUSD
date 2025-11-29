---
title: "Economic calendar functions"
url: "https://www.mql5.com/en/docs/calendar"
hierarchy: []
scraped_at: "2025-11-28 09:30:40"
---

# Economic calendar functions

[MQL5 Reference](/en/docs "MQL5 Reference")Economic Calendar

* [CalendarCountryById](/en/docs/calendar/calendarcountrybyid "CalendarCountryById")
* [CalendarEventById](/en/docs/calendar/calendareventbyid "CalendarEventById")
* [CalendarValueById](/en/docs/calendar/calendarvaluebyid "CalendarValueById")
* [CalendarCountries](/en/docs/calendar/calendarcountries "CalendarCountries")
* [CalendarEventByCountry](/en/docs/calendar/calendareventbycountry "CalendarEventByCountry")
* [CalendarEventByCurrency](/en/docs/calendar/calendareventbycurrency "CalendarEventByCurrency")
* [CalendarValueHistoryByEvent](/en/docs/calendar/calendarvaluehistorybyevent "CalendarValueHistoryByEvent")
* [CalendarValueHistory](/en/docs/calendar/calendarvaluehistory "CalendarValueHistory")
* [CalendarValueLastByEvent](/en/docs/calendar/calendarvaluelastbyevent "CalendarValueLastByEvent")
* [CalendarValueLast](/en/docs/calendar/calendarvaluelast "CalendarValueLast")

# Economic calendar functions

This section describes the functions for working with the [economic calendar](https://www.metatrader5.com/en/terminal/help/charts_analysis/fundamental) available directly in the MetaTrader platform. The economic calendar is a ready-made encyclopedia featuring descriptions of macroeconomic indicators, their release dates and degrees of importance. Relevant values of macroeconomic indicators are sent to the MetaTrader platform right at the moment of publication and are displayed on a chart as tags allowing you to visually track the required indicators by countries, currencies and importance.

All functions for working with the economic calendar use the trade server time ([TimeTradeServer](/en/docs/dateandtime/timetradeserver)). This means that the time in the [MqlCalendarValue](/en/docs/constants/structures/mqlcalendar#mqlcalendarvalue) structure and the time inputs in the [CalendarValueHistoryByEvent](/en/docs/calendar/calendarvaluehistorybyevent)/[CalendarValueHistory](/en/docs/calendar/calendarvaluehistory) functions are set in a trade server timezone, rather than a user's local time.

[Economic calendar functions](/en/docs/calendar) allow conducting the auto analysis of incoming events according to custom importance criteria from a perspective of necessary countries/currencies.

| Function | Action |
| --- | --- |
| [CalendarCountryById](/en/docs/calendar/calendarcountrybyid) | Get a country description by its ID |
| [CalendarEventById](/en/docs/calendar/calendareventbyid) | Get an event description by its ID |
| [CalendarValueById](/en/docs/calendar/calendarvaluebyid) | Get an event value description by its ID |
| [CalendarCountries](/en/docs/calendar/calendarcountries) | Get the array of country names available in the calendar |
| [CalendarEventByCountry](/en/docs/calendar/calendareventbycountry) | Get the array of descriptions of all events available in the calendar by a specified country code |
| [CalendarEventByCurrency](/en/docs/calendar/calendareventbycurrency) | Get the array of descriptions of all events available in the calendar by a specified currency |
| [CalendarValueHistoryByEvent](/en/docs/calendar/calendarvaluehistorybyevent) | Get the array of values for all events in a specified time range by an event ID |
| [CalendarValueHistory](/en/docs/calendar/calendarvaluehistory) | Get the array of values for all events in a specified time range with the ability to sort by country and/or currency |
| [CalendarValueLastByEvent](/en/docs/calendar/calendarvaluelastbyevent) | Get the array of event values by its ID since the calendar database status with a specified change\_id |
| [CalendarValueLast](/en/docs/calendar/calendarvaluelast) | Get the array of values for all events with the ability to sort by country and/or currency since the calendar database status with a specified change\_id |

[MarketBookGet](/en/docs/marketinformation/marketbookget "MarketBookGet")

[CalendarCountryById](/en/docs/calendar/calendarcountrybyid "CalendarCountryById")