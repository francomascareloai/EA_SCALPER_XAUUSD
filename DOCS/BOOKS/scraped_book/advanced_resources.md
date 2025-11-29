---
title: "Resources"
url: "https://www.mql5.com/en/book/advanced/resources"
hierarchy: []
scraped_at: "2025-11-28 09:48:55"
---

# Resources

[MQL5 Programming for Traders](/en/book "MQL5 Programming for Traders")[Advanced language tools](/en/book/advanced "Advanced language tools")Resources

* [Describing resources using the #resource directive](/en/book/advanced/resources/resources_directive "Describing resources using the #resource directive")
* [Shared use of resources of different MQL programs](/en/book/advanced/resources/resources_sharing "Shared use of resources of different MQL programs")
* [Resource variables](/en/book/advanced/resources/resources_variables "Resource variables")
* [Connecting custom indicators as resources](/en/book/advanced/resources/resources_indicators "Connecting custom indicators as resources")
* [Dynamic resource creation: ResourceCreate](/en/book/advanced/resources/resources_resourcecreate "Dynamic resource creation: ResourceCreate")
* [Deleting dynamic resources: ResourceFree](/en/book/advanced/resources/resources_resourcefree "Deleting dynamic resources: ResourceFree")
* [Reading and modifying resource data: ResourceReadImage](/en/book/advanced/resources/resources_resourcereadimage "Reading and modifying resource data: ResourceReadImage")
* [Saving images to a file: ResourceSave](/en/book/advanced/resources/resources_resourcesave "Saving images to a file: ResourceSave")
* [Fonts and text output to graphic resources](/en/book/advanced/resources/resources_textout "Fonts and text output to graphic resources")
* [Application of graphic resources in trading](/en/book/advanced/resources/resources_applied_usecase "Application of graphic resources in trading")

Download in one file:

[MQL5 Algo Book in PDF](https://www.mql5.com/files/book/mql5book.pdf "mql5book.pdf")

[MQL5 Algo Book in CHM](https://www.mql5.com/files/book/mql5book.chm?v=2 "mql5book.chm")

# Resources

The operation of MQL programs may require many auxiliary resources, which are arrays of application data or files of various types, including images, sounds, and fonts. The MQL development environment allows you to include all such resources in the executable file at the compilation stage. This eliminates the need for their parallel transfer and installation along with the main program and makes it a complete self-sufficient product that is convenient for the end user.

In this chapter, we will learn how to describe different types of resources and built-in functions for subsequent operations with connected resources.

Raster images, represented as arrays of points (pixels) in the widely recognized BMP format, hold a unique position among resources. The MQL5 API allows the creation, manipulation, and dynamic display of these graphic resources on charts.

Earlier, we already discussed graphical objects and, in particular, objects of types [OBJ\_BITMAP](/en/book/applications/objects/objects_time_price) and [OBJ\_BITMAP\_LABEL](/en/book/applications/objects/objects_screen_coordinates) that are useful for designing user interfaces. For these objects, there is the [OBJPROP\_BMPFILE](/en/book/applications/objects/objects_bitmap) property that specifies the image as a file or resource. Previously, we only considered examples with files. Now we will learn how to work with resource images.

[Advanced language tools](/en/book/advanced "Advanced language tools")

[Describing resources using the #resource directive](/en/book/advanced/resources/resources_directive "Describing resources using the #resource directive")