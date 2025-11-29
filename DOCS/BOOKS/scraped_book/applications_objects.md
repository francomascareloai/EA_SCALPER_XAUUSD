---
title: "Graphical objects"
url: "https://www.mql5.com/en/book/applications/objects"
hierarchy: []
scraped_at: "2025-11-28 09:48:22"
---

# Graphical objects

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Creating application programs](/en/book/applications "Creating application programs")Graphical objects

* [Object types and features of specifying their coordinates](/en/book/applications/objects/objects_main_characteristics "Object types and features of specifying their coordinates")
* [Time and price bound objects](/en/book/applications/objects/objects_time_price "Time and price bound objects")
* [Objects bound to screen coordinates](/en/book/applications/objects/objects_screen_coordinates "Objects bound to screen coordinates")
* [Creating objects](/en/book/applications/objects/objects_create "Creating objects")
* [Deleting objects](/en/book/applications/objects/objects_delete "Deleting objects")
* [Finding objects](/en/book/applications/objects/objects_find "Finding objects")
* [Overview of object property access functions](/en/book/applications/objects/objects_properties_get_set "Overview of object property access functions")
* [Main object properties](/en/book/applications/objects/objects_properties_main "Main object properties")
* [Price and time coordinates](/en/book/applications/objects/objects_time_price_coordinates "Price and time coordinates")
* [Anchor window corner and screen coordinates](/en/book/applications/objects/objects_corner_x_y "Anchor window corner and screen coordinates")
* [Defining anchor point on the object](/en/book/applications/objects/objects_anchor "Defining anchor point on the object")
* [Managing the object state](/en/book/applications/objects/objects_state "Managing the object state")
* [Priority of objects (Z-Order)](/en/book/applications/objects/objects_z_order "Priority of objects (Z-Order)")
* [Object display settings: color, style, and frame](/en/book/applications/objects/objects_color_style "Object display settings: color, style, and frame")
* [Font settings](/en/book/applications/objects/objects_font "Font settings")
* [Rotating text at an arbitrary angle](/en/book/applications/objects/objects_angle "Rotating text at an arbitrary angle")
* [Determining object width and height](/en/book/applications/objects/objects_width_height "Determining object width and height")
* [Visibility of objects in the context of timeframes](/en/book/applications/objects/objects_timeframes "Visibility of objects in the context of timeframes")
* [Assigning a character code to a label](/en/book/applications/objects/objects_arrow_codes "Assigning a character code to a label")
* [Ray properties for objects with straight lines](/en/book/applications/objects/objects_rays "Ray properties for objects with straight lines")
* [Managing object pressed state](/en/book/applications/objects/objects_pressed_state "Managing object pressed state")
* [Adjusting images in bitmap objects](/en/book/applications/objects/objects_bitmap "Adjusting images in bitmap objects")
* [Cropping (outputting part) of an image](/en/book/applications/objects/objects_bitmap_offset "Cropping (outputting part) of an image")
* [Input field properties: alignment and read-only](/en/book/applications/objects/objects_edit "Input field properties: alignment and read-only")
* [Standard deviation channel width](/en/book/applications/objects/objects_stddev_channel "Standard deviation channel width")
* [Setting levels in level objects](/en/book/applications/objects/objects_levels "Setting levels in level objects")
* [Additional properties of Gann, Fibonacci, and Elliot objects](/en/book/applications/objects/objects_gann_fibo_elliott "Additional properties of Gann, Fibonacci, and Elliot objects")
* [Chart object](/en/book/applications/objects/objects_chart "Chart object")
* [Moving objects](/en/book/applications/objects/objects_move "Moving objects")
* [Getting time or price at the specified line points](/en/book/applications/objects/objects_get_time_value "Getting time or price at the specified line points")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Graphical objects

MetaTrader 5 users are well aware of the concept of graphical objects: trend lines, price labels, channels, Fibonacci levels, geometric shapes, and many other visual elements that are used for the analytical chart markup. The MQL5 language allows you to create, edit, and delete graphical objects programmatically. This can be useful, for example, when it is desirable to display certain data simultaneously in a subwindow and the main window of an indicator. Since the platform only supports the output of indicator buffers in one window, we can generate objects in the other window. With the markup created from graphical objects, it is easy to organize semi-automated trading using [Expert Advisors](/en/book/automation/experts). Additionally, objects are often used to build custom graphical interfaces for MQL programs, such as buttons, input fields, and flags. These programs can be controlled without opening the properties dialog, and the panels created in MQL can offer much greater flexibility than standard input variables.  

Each object exists in the context of a particular chart. That's why the functions we will discuss in this chapter share a common characteristic: the first parameter specifies the [chart ID](/en/book/applications/charts/charts_id). In addition, each graphical object is characterized by a name that is unique within one chart, including all subwindows. Changing the name of a graphical object involves deleting the object with the old name and creating the same object with a new name. You cannot create two objects with the same name.

The functions that define the properties of graphical objects, as well as the operations of creating ([ObjectCreate](/en/book/applications/objects/objects_create)) and moving ([ObjectMove](/en/book/applications/objects/objects_move)) objects on the chart, essentially serve to send asynchronous commands to the chart. If these functions are successfully executed, the command enters the shared event queue of the chart. The visual modification of the properties of graphical objects occurs during the processing of the event queue for that particular chart. Therefore, the external representation of the chart may reflect the changed state of objects with some delay after the function calls.

In general, the update of graphical objects on the chart is done automatically by the terminal in response to chart-related events such as receiving a new quote, resizing the window, and so on. To force the update of graphical objects, you can use the function for requesting chart redraw ([ChartRedraw](/en/book/applications/charts/charts_redraw)). This is particularly important after mass creation or modification of objects.

Objects serve as a source of programmatic events, such as creation, deletion, modification of their properties, and mouse clicks. All aspects of event occurrence and handling are discussed in a separate [chapter](/en/book/applications/events), along with events in the general window context.

We will begin with the theoretical foundations and gradually move on to practical aspects.

[Saving a chart image](/en/book/applications/charts/charts_screenshot "Saving a chart image")

[Object types and features of specifying their coordinates](/en/book/applications/objects/objects_main_characteristics "Object types and features of specifying their coordinates")