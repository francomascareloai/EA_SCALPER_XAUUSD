---
title: "Scripts"
url: "https://www.mql5.com/en/book/applications/script_service/scripts"
hierarchy: []
scraped_at: "2025-11-28 09:49:32"
---

# Scripts

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")[Scripts and services](/en/book/applications/script_service "Scripts and services")Scripts

* Scripts
* [Services](/en/book/applications/script_service/services "Services")
* [Restrictions for scripts and services](/en/book/applications/script_service/script_service_limitations "Restrictions for scripts and services")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Scripts

A script is an MQL program with the only handler OnStart, provided there is no #property servicedirective  (otherwise you get a service, see the next section).

By default, the script immediately starts executing when it is placed on the chart. The developer can ask the user to confirm the start by adding the #property script\_show\_confirm directive to the beginning of the file. In this case, the terminal will show a message with the question "Are you sure you want to run 'program' on chart 'symbol, timeframe'?" and buttons Yes and No.

Scripts, like other programs, can have [input variables](/en/book/basis/variables/input_variables). However, for scripts, the parameter input dialog is not shown by default, even if the script defines inputs. To ensure that the properties dialog opens before running the script, the #property script\_show\_inputs directive should be applied. It takes precedence over script\_show\_confirm, that is, the output of the dialog disables the confirmation request (since the dialog itself acts in a similar role). The directive calls a dialog even if there are no input variables. It can be used to show the product description and version (they are displayed on the Common tab) to the user.

The following table shows combination options for the #property directive and their effect on the program.

| Directive Effect | script\_show\_confirm | script\_show\_inputs |
| --- | --- | --- |
| Immediate launch | No | No |
| Confirmation request | Yes | No |
| Opening the properties dialog | irrelevant | Yes |

A simple example of a script with directives is in the file ScriptNoComment.mq5. The purpose of the script is as follows. Sometimes MQL programs leave behind unnecessary comments in the upper left corner of the chart. Comments are stored in chr-files along with the chart, so even after restarting the terminal they are restored. This script allows you to clear a comment or set it to an arbitrary value. If you Assign hotkey to a script using the Navigator context menu command, it will be possible to clean the comment of the current chart with one click.

Originally, directives script\_show\_confirm and script\_show\_inputs are disabled by becoming inline comments. You can experiment with different combinations of directives by uncommenting them one at a time or at the same time.

| |
| --- |
| //#property script\_show\_confirm //#property script\_show\_inputs     input string Text = "";     void OnStart() {    Comment(""); // clean up the comment } |

[Scripts and services](/en/book/applications/script_service "Scripts and services")

[Services](/en/book/applications/script_service/services "Services")