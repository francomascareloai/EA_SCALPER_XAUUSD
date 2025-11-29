# MQL5 Book - Part 9 (Pages 1601-1800)

## Page 1601

Part 7. Advanced language tools
1 601 
7.1  Resources
Bitmap of a resource with a set of random shapes
When comparing this image with what we saw in the example Obj ectShapesDraw.mq5, it turns out that
our new way of rendering is somewhat different from how the terminal displays objects. Although the
shapes and colors are correct, the places where the shapes overlap are indicated in different ways.
Our script paints the shapes with the specified color, superimposing them on top of each other in the
order they appear in the array. Later shapes overlap the earlier ones. The terminal, on the other hand,
applies some kind of color mixing (inversion) in places of overlap.
Both methods have the right to exist, there are no errors here. However, is it possible to achieve a
similar effect when drawing?
We have full control over the drawing process, so any effects can be applied to it not only the one from
the terminal.
In addition to the original, simple way of drawing, let's implement a few more modes. All of them are
summarized in the COLOR_EFFECT enumeration.
enum COLOR_EFFECT
{
   PLAIN,         // simple drawing with overlap (default)
   COMPLEMENT,    // draw with a complementary color (like in the terminal) 
   BLENDING_XOR,  // mixing colors with XOR '^'
   DIMMING_SUM,   // "darken" colors with '+'
   LIGHTEN_OR,    // "lighten" colors with '|'
};
Let's add an input variable to select the mode.

---

## Page 1602

Part 7. Advanced language tools
1 602
7.1  Resources
input COLOR_EFFECT ColorEffect = PLAIN;
Let's support modes in the MyDrawing class. First, let's describe the corresponding field and method.
class MyDrawing: public Drawing
{
   ...
   COLOR_EFFECT xormode;
   ...
public:
   void setColorEffect(const COLOR_EFFECT x)
   {
      xormode = x;
   }
   ...
Then we improve the point method.
   virtual void point(const float x1, const float y1, const uint pixel) override
   {
      ...
      if(index >= 0 && index < ArraySize(data))
      {
         switch(xormode)
         {
         case COMPLEMENT:
            data[index] = (pixel ^ (1 - data[index])); // blending with complementary color
            break;
         case BLENDING_XOR:
            data[index] = (pixel & 0xFF000000) | (pixel ^ data[index]); // direct mixing (XOR)
            break;
         case DIMMING_SUM:
            data[index] =  (pixel + data[index]); // "darkening" (SUM)
            break;
         case LIGHTEN_OR:
            data[index] =  (pixel & 0xFF000000) | (pixel | data[index]); // "lightening" (OR)
            break;
         case PLAIN:
         default:
            data[index] = pixel;
         }
      }
   }
You can try running the script in different modes and compare the results. Don't forget about the ability
to customize the background. Here is an example of what lightening looks like.

---

## Page 1603

Part 7. Advanced language tools
1 603
7.1  Resources
Image of shapes with lightening color mixing
To visually see the difference in effects, you can turn off color randomization and shape movement. The
standard way of overlapping objects corresponds to the COMPLEMENT constant.
As a final experiment, enable the SaveImage option. In the OnStart handler, when generating the name
of the file with the image, we now use the name of the current mode. We need to get a copy of the
image on the chart in the file.
   ...
   if(SaveImage)
   {
      const string filename = EnumToString(ColorEffect) + ".bmp";
      if(ResourceSave(raster.resource(), filename))
      ...
For more sophisticated graphic constructions of our interface, Drawing may not be enough. Therefore,
you can use ready-made drawing classes supplied with MetaTrader 5 or available in the mql5.com
codebase. In particular, take a look at the file MQL5/Include/Canvas/Canvas.mqh.
7.1 .9 Fonts and text output to graphic resources
In addition to rendering individual pixels in an array of a graphic resource, we can use built-in functions
for displaying text. Functions allow you to change the current font and its characteristics
(TextSetFont), get the dimensions of the rectangle in which the given string can be inscribed
(TextGetSize), as well as directly insert the caption into the generated image (TextOut).

---

## Page 1604

Part 7. Advanced language tools
1 604
7.1  Resources
bool TextSetFont(const string name, int size, uint flags, int orientation = 0)
The function sets the font and its characteristics for subsequent drawing of text in the image buffer
using the TextOut function (see further). The name parameter may contain the name of a built-in
Windows font or a ttf font file (TrueType Font) connected by the resource directive (if the name starts
with "::").
Size (size) can be specified in points (a typographic unit of measurement) or pixels (screen points).
Positive values mean that the unit of measurement is a pixel, and negative values are measured in
tenths of a point. Height in pixels will look different to users depending on the technical capabilities and
settings of their monitors. The height in points will be approximately ("judging by eye") the same for
everyone.
A typographical point is a physical unit of length, traditionally equal to 1 /72nd of an inch. Hence, 1 
point is equal to 0.352778 millimeters. A pixel on the screen is a virtual measure of length. Its
physical size depends on the hardware resolution of the screen. For example, with a screen density
of 96 DPI (dots per inch), 1  pixel will take 0.264583 millimeters or 0.75 points. However, most
modern displays have much higher DPI values and therefore smaller pixels. Because of this,
operating systems, including Windows, have long had settings to increase the visible scale of
interface elements. Thus, if you specify a size in points (negative values), the size of the text in
pixels will depend on the display and scale settings in the operating system (for example, "standard"
1 00%, "medium" 1 25%, or "large" 1 50%). 
Zooming in causes the displayed pixels to be artificially enlarged by the system. This is equivalent to
reducing the screen size in pixels, and the system applies the effective DPI to achieve the same
physical size. If scaling is enabled, then it is the effective DPI that is reported to programs,
including the terminal and then MQL programs. If necessary, you can find out the DPI of the screen
from the TERMINAL_SCREEN_DPI property (see Screen specifications). However in reality, by
setting the font size in points, we are relieved of the need to recalculate its size depending on the
DPI, since the system will do it for us.
The default font is Arial and the default size is -1 20 (1 2 pt). Controls, in particular, built-in objects on
charts also operate with font sizes in points. For example, if in an MQL program, you want to draw text
of the same size as the text in the OBJ_LABEL object, which has a size of 1 0 points, you should use the
parameter size equal to -1 00.
The flags parameter sets a combination of flags describing the style of the font. The combination is
made up of a bitmask using the bitwise operator OR ('| '). Flags are divided into two groups: style flags
and boldness flags.
The following table lists the style flags. They can be mixed.
Flag
Description
FONT_ITALIC
Italics
FONT_UNDERLINE
Underline
FONT_STRIKEOUT
Strikethrough
Boldness flags have relative weights corresponding to them (given to compare expected effects).

---

## Page 1605

Part 7. Advanced language tools
1 605
7.1  Resources
Flag
Description
FW_DONTCARE
0 (system default will be applied)
FW_THIN
1 00
FW_EXTRALIGHT, FW_ULTRALIGHT
200
FW_LIGHT
300
FW_NORMAL, FW_REGULAR
400
FW_MEDIUM
500
FW_SEMIBOLD, FW_DEMIBOLD
600
FW_BOLD
700
FW_EXTRABOLD, FW_ULTRABOLD
800
FW_HEAVY, FW_BLACK
900
Use only one of these values in a combination of flags.
The orientation parameter specifies the angle of the text in relation to the horizontal, in tenths of a
degree. For example, orientation = 0 means normal text output, while orientation = 450 will result in a
45-degree tilt (counterclockwise).
Note that settings made in one TextSetFont call will affect all subsequent TextOut calls until they are
changed.
The function returns true if successful or false if problems occur (for example, if the font is not found).
We will consider an example of using this function, as well as the two others after describing all of them.
bool TextGetSize(const string text, uint &width, uint &height)
The function returns the width and height of the line at the current font settings (this can be the
default font or the one specified in the previous call to TextSetFont).
The text parameter passes a string in which length and width in pixels are required. Dimension values
are written by the function based on references in the width and height parameters.
It should be noted that the rotation (skew) of the displayed text specified by the orientation parameter
when TextSetFont is called does not affect the sizing in any way. In other words, if the text is supposed
to be rotated by 45 degrees, then the MQL program itself must calculate the minimum square in which
the text can fit. The TextGetSize function calculates the text size in a standard (horizontal) position.
bool TextOut(const string text, int x, int y, uint anchor, uint &data[], uint width, uint height, uint color,
ENUM_COLOR_FORMAT color_format)
The function draws text in the graphic buffer at the specified coordinates taking into account the color,
format, and previous settings (font, style, and orientation).
The text is passed in the text parameter and must be in the form of one line.

---

## Page 1606

Part 7. Advanced language tools
1 606
7.1  Resources
The x and y coordinates specified in pixels define the point in the graphics buffer where text is
displayed. Which place of the generated inscription will be at the point (x, y) depends on the binding
method in the anchor parameter (see further).
The buffer is represented by the data array, and although the array is one-dimensional, it stores a two-
dimensional "canvas" with dimensions of width x height points. This array can be obtained from the
ResourceReadImage function, or allocated by an MQL program. After all editing operations are
completed, including text output, you should create a new resource based on this buffer or apply it to
an already existing resource. In both cases, you should call ResourceCreate.
The color of the text and the way the color is handled are set by the parameters color and color_ format
(see ENUM_COLOR_FORMAT). Note that the type used for color is uint, i.e., to convey the color, it
should be converted using ColorToARGB.
The anchoring method specified by the anchor parameter is a combination of two text positioning flags:
vertical and horizontal.
Horizontal text position flags are:
• TA_LEFT – anchor to the left side of the bounding box
• TA_CENTER – anchor to the middle between the left and right sides of the rectangle
• TA_RIGHT – anchor to the right side of the bounding box
Vertical text position flags are:
• TA_TOP – anchor to the top side of the bounding box
• TA_VCENTER – anchor to the middle between the top and bottom side of the rectangle
• TA_BOTTOM – anchor to the bottom side of the bounding box
In total, there are 9 valid combinations of flags to describe the anchoring method.
The position of the output text relative to the anchor point
Here, the center of the picture contains a deliberately exaggerated large point in the generated image
with coordinates (x, y). Depending on the flags, the text appears relative to this point at the specified
positions (the content of the text corresponds to the applied anchoring method).
For ease of reference, all the inscriptions are made in the standard horizontal position. However, note
that an angle could also be applied to any of them (orientation), and then the corresponding inscription
would be rotated around the point. In this image, only the label centered on both dimensions is rotated.

---

## Page 1607

Part 7. Advanced language tools
1 607
7.1  Resources
These flags should not be confused with text alignment. The bounding box is always sized to fit the text,
and its position relative to the anchor point is, in a sense, the opposite of the flag names.
Let's look at some examples using three functions.
To begin with, let's check the simplest options of setting the font boldness and style. The
ResourceText.mq5 script allows you to select the name of the font, its size, as well as the colors of the
background and text in the input variables. The labels will be displayed on the chart for the specified
number of seconds.
input string Font = "Arial";             // Font Name
input int    Size = -240;                // Size
input color  Color = clrBlue;            // Font Color
input color  Background = clrNONE;       // Background Color
input uint   Seconds = 10;               // Demo Time (seconds)
The name of each gradation of boldness will be displayed in the label text, so to simplify the process (by
using EnumToString) the ENUM_FONT_WEIGHTS enumeration is declared.
enum ENUM_FONT_WEIGHTS
{
   _DONTCARE = FW_DONTCARE,
   _THIN = FW_THIN,
   _EXTRALIGHT = FW_EXTRALIGHT,
   _LIGHT = FW_LIGHT,
   _NORMAL = FW_NORMAL,
   _MEDIUM = FW_MEDIUM,
   _SEMIBOLD = FW_SEMIBOLD,
   _BOLD = FW_BOLD,
   _EXTRABOLD = FW_EXTRABOLD,
   _HEAVY = FW_HEAVY,
};
const int nw = 10; // number of different weights
The inscription flags are collected in the rendering array and random combinations are selected from it.
   const uint rendering[] =
   {
      FONT_ITALIC,
      FONT_UNDERLINE,
      FONT_STRIKEOUT
   };
   const int nr = sizeof(rendering) / sizeof(uint);
To get a random number in a range, there is an auxiliary function Random.
int Random(const int limit)
{
   return rand() % limit;
}
In the main function of the script, we find the size of the chart and create an OBJ_BITMAP_LABEL
object that spans the entire space.

---

## Page 1608

Part 7. Advanced language tools
1 608
7.1  Resources
void OnStart()
{
   ...
   const string name = "FONT";
   const int w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   const int h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   
   // object for a resource with a picture filling the whole window
   ObjectCreate(0, name, OBJ_BITMAP_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ...
Next, we allocate memory for the image buffer, fill it with the specified background color (or leave it
transparent, by default), create a resource based on the buffer, and bind it to the object.
   uint data[];
   ArrayResize(data, w * h);
   ArrayInitialize(data, Background == clrNONE ? 0 : ColorToARGB(Background));
   ResourceCreate(name, data, w, h, 0, 0, w, COLOR_FORMAT_ARGB_RAW);
   ObjectSetString(0, name, OBJPROP_BMPFILE, "::" + name);
   ...
Just in case, note that we can set the OBJPROP_BMPFILE property without a modifier (0 or 1 ) in the
Obj ectSetString call unless the object is supposed to switch between two states.
All font weights are listed in the weights array.
   const uint weights[] =
   {
      FW_DONTCARE,
      FW_THIN,
      FW_EXTRALIGHT, // FW_ULTRALIGHT,
      FW_LIGHT,
      FW_NORMAL,     // FW_REGULAR,
      FW_MEDIUM,
      FW_SEMIBOLD,   // FW_DEMIBOLD,
      FW_BOLD,
      FW_EXTRABOLD,  // FW_ULTRABOLD,
      FW_HEAVY,      // FW_BLACK
   };
   const int nw = sizeof(weights) / sizeof(uint);
In the loop, in order, we set the next gradation of boldness for each line using TextSetFont, preselecting
a random style. A description of the font, including its name and weight, is drawn in the buffer using
TextOut.

---

## Page 1609

Part 7. Advanced language tools
1 609
7.1  Resources
   const int step = h / (nw + 2);
   int cursor = 0;    // Y coordinate of the current "text line"
   
   for(int weight = 0; weight < nw; ++weight)
   {
      // apply random style
      const int r = Random(8);
      uint render = 0;
      for(int j = 0; j < 3; ++j)
      {
         if((bool)(r & (1 << j))) render |= rendering[j];
      }
      TextSetFont(Font, Size, weights[weight] | render);
      
      // generate font description
      const string text = Font + EnumToString((ENUM_FONT_WEIGHTS)weights[weight]);
      
      // draw text on a separate "line"
      cursor += step;
      TextOut(text, w / 2, cursor, TA_CENTER | TA_TOP, data, w, h,
         ColorToARGB(Color), COLOR_FORMAT_ARGB_RAW);
   }
   ...
Now update the resource and chart.
   ResourceCreate(name, data, w, h, 0, 0, w, COLOR_FORMAT_ARGB_RAW);
   ChartRedraw();
   ...
The user can stop the demonstration in advance.
   const uint timeout = GetTickCount() + Seconds * 1000;
   while(!IsStopped() && GetTickCount() < timeout)
   {
      Sleep(1000);
   }
Finally, the script deletes the resource and the object.
   ObjectDelete(0, name);
   ResourceFree("::" + name);
}
The result of the script is shown in the following image.

---

## Page 1610

Part 7. Advanced language tools
1 61 0
7.1  Resources
Drawing text in different weights and styles
In the second example of ResourceFont.mq5, we will make the task more difficult by including a custom
font as a resource and using text rotation in 90-degree increments.
The font file is located next to the script.
#resource "a_LCDNova3DCmObl.ttf"
The message can be changed in the input parameter.
input string Message = "Hello world!";   // Message
This time, the OBJ_BITMAP_LABEL will not occupy the entire window and is therefore centered both
horizontally and vertically.
void OnStart()
{
   const string name = "FONT";
   const int w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   const int h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   
   // object for a resource with a picture
   ObjectCreate(0, name, OBJ_BITMAP_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, w / 2);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, h / 2);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_CENTER);
   ...
To begin with, the buffer of the minimum size is allocated, just to complete resource creation. Later we
will expand it to fit the dimensions of the inscription, for which there are reserved variables width and
height.

---

## Page 1611

Part 7. Advanced language tools
1 61 1 
7.1  Resources
   uint data[], width, height;
   ArrayResize(data, 1);
   ResourceCreate(name, data, 1, 1, 0, 0, 1, COLOR_FORMAT_ARGB_RAW);
   ObjectSetString(0, name, OBJPROP_BMPFILE, "::" + name);
   ...
In a loop with the test time countdown, we need to change the orientation of the inscription, for which
there is the angle variable (degrees will be scrolled in it). The orientation will change once per second,
the count is in the remain variable.
   const uint timeout = GetTickCount() + Seconds * 1000;
   int angle = 0;
   int remain = 10;
   ...
In the loop, we constantly change the rotation of the text, and in the text itself, we display a
countdown counter of seconds. For each new inscription, its size is calculated using TextGetSize, based
on which the buffer is reallocated.

---

## Page 1612

Part 7. Advanced language tools
1 61 2
7.1  Resources
   while(!IsStopped() && GetTickCount() < timeout)
   {
      // apply new angle
      TextSetFont("::a_LCDNova3DCmObl.ttf", -240, 0, angle * 10);
      
      // form the text
      const string text = Message + " (" + (string)remain-- + ")";
      
      // get the text size, allocate the array
      TextGetSize(text, width, height);
      ArrayResize(data, width * height);
      ArrayInitialize(data, 0);            // transparency
      
      // for vertical orientation, swap sizes
      if((bool)(angle / 90 & 1))
      {
         const uint t = width;
         width = height;
         height = t;
      }
      
      // adjust the size of the object
      ObjectSetInteger(0, name, OBJPROP_XSIZE, width);
      ObjectSetInteger(0, name, OBJPROP_YSIZE, height);
      
      // draw text
      TextOut(text, width / 2, height / 2, TA_CENTER | TA_VCENTER, data, width, height,
         ColorToARGB(clrBlue), COLOR_FORMAT_ARGB_RAW);
      
      // update resource and chart
      ResourceCreate(name, data, width, height, 0, 0, width, COLOR_FORMAT_ARGB_RAW);
      ChartRedraw();
      
      // change angle
      angle += 90;
      
      Sleep(100);
   }
   ...
Note that if the text is vertical, the dimensions need to be swapped. More generally, with text rotated
to an arbitrary angle, it took more math to get the buffer size to fit the entire text.
At the end, we also delete the object and resource.
   ObjectDelete(0, name);
   ResourceFree("::" + name);
}
One of the moments of the script execution is shown in the following screenshot.

---

## Page 1613

Part 7. Advanced language tools
1 61 3
7.1  Resources
Inscription with custom font
As a final example, let's take a look at the script ResourceTextAnchOrientation.mq5 showing various
rotations and anchor points of the text.
The script generates the specified number of labels (ExampleCount) using the specified font.
input string Font = "Arial";             // Font Name
input int    Size = -150;                // Size
input int    ExampleCount = 11;          // Number of examples
Anchor points and rotations are chosen randomly.
To specify the names of anchor points in labels, there is the ENUM_TEXT_ANCHOR enumeration with all
valid options declared. So, we can simply call EnumToString any randomly selected element.
enum ENUM_TEXT_ANCHOR
{
   LEFT_TOP = TA_LEFT | TA_TOP,
   LEFT_VCENTER = TA_LEFT | TA_VCENTER,
   LEFT_BOTTOM = TA_LEFT | TA_BOTTOM,
   CENTER_TOP = TA_CENTER | TA_TOP,
   CENTER_VCENTER = TA_CENTER | TA_VCENTER,
   CENTER_BOTTOM = TA_CENTER | TA_BOTTOM,
   RIGHT_TOP = TA_RIGHT | TA_TOP,
   RIGHT_VCENTER = TA_RIGHT | TA_VCENTER,
   RIGHT_BOTTOM = TA_RIGHT | TA_BOTTOM,
};
An array of these new constants is declared in the OnStart handler.

---

## Page 1614

Part 7. Advanced language tools
1 61 4
7.1  Resources
void OnStart()
{
   const ENUM_TEXT_ANCHOR anchors[] =
   {
      LEFT_TOP,
      LEFT_VCENTER,
      LEFT_BOTTOM,
      CENTER_TOP,
      CENTER_VCENTER,
      CENTER_BOTTOM,
      RIGHT_TOP,
      RIGHT_VCENTER,
      RIGHT_BOTTOM,
   };
   const int na = sizeof(anchors) / sizeof(uint);
   ...
Initial object and resource creation are similar to the example with ResourceText.mq5, so let's leave
them out here. The most interesting thing happens in the loop.
   for(int i = 0; i < ExampleCount; ++i)
   {
      // apply a random angle
      const int angle = Random(360);
      TextSetFont(Font, Size, 0, angle * 10);
      
      // take random coordinates and an anchor point
      const ENUM_TEXT_ANCHOR anchor = anchors[Random(na)];
      const int x = Random(w / 2) + w / 4;
      const int y = Random(h / 2) + h / 4;
      const color clr = ColorMix::HSVtoRGB(angle);
      
     // draw a circle directly in that place of the image,
     // where the anchor point goes
      TextOut(ShortToString(0x2022), x, y, TA_CENTER | TA_VCENTER, data, w, h,
         ColorToARGB(clr), COLOR_FORMAT_ARGB_NORMALIZE);
      
      // form the text describing the anchor type and angle
      const string text =  EnumToString(anchor) +
         "(" + (string)angle + CharToString(0xB0) + ")";
   
      // draw text
      TextOut(text, x, y, anchor, data, w, h,
         ColorToARGB(clr), COLOR_FORMAT_ARGB_NORMALIZE);
   }
   ...
It remains only to update the picture and chart, and then wait for the user's command and free up
resources.

---

## Page 1615

Part 7. Advanced language tools
1 61 5
7.1  Resources
   ResourceCreate(name, data, w, h, 0, 0, w, COLOR_FORMAT_ARGB_NORMALIZE);
   ChartRedraw();
   
   const uint timeout = GetTickCount() + Seconds * 1000;
   while(!IsStopped() && GetTickCount() < timeout)
   {
      Sleep(1000);
   }
   
   ObjectDelete(0, name);
   ResourceFree("::" + name);
}
Here's what we get as a result.
Text output with random coordinates, anchor points, and angles
Additionally, for an independent study, the book provides a toy graphics editor SimpleDrawing.mq5. It is
designed as a bufferless indicator and uses in its work the classes of shapes considered earlier (see the
example with ResourceShapesDraw.mq5). They are put in the header file ShapesDrawing.mqh almost
unchanged. Previously, the shapes were randomly generated by the script. Now the user can select and
plot them on the chart. For this purpose, an interface with a color palette and a button bar has been
implemented according to the number of registered shape classes. The interface is implemented by the
SimpleDrawing class (SimpleDrawing.mqh).

---

## Page 1616

Part 7. Advanced language tools
1 61 6
7.1  Resources
Simple graphic editor
The panel and palette can be positioned along any border of the chart, demonstrating the ability to
rotate labels.
Selecting the next shape to draw is done by pressing the button in the panel: the button "sticks" in the
pressed state, and its background color indicates the selected drawing color. To change the color, click
anywhere on the palette.
When one of the shape types is selected in the panel (one of the buttons is "active"), clicking in the
drawing area (the rest of the chart, indicated by shading) draws a shape of predefined size at that
location. At this point, the button "switches off". In this state, when all buttons are inactive, you can
move the shapes around the workspace using the mouse. If we keep the key Ctrl pressed, instead of
moving, the shape gets resized. The "hot spot" is located in the center of each shape (the size of the
sensitive area is set by a macro in the source code and will probably need to be increased for very high
DPI displays).
Note that the editor includes the plot ID (ChartID) in the names of the generated resources. This allows
to run the editor in parallel on several charts.
7.1 .1 0 Application of graphic resources in trading
Of course, beautifying is not the primary purpose of the resources. Let's see how to create a useful tool
based on them. We will also eliminate one more omission: so far we have used resources only inside
OBJ_BITMAP_LABEL objects, which are positioned in screen coordinates. However, graphic resources
can also be embedded in OBJ_BITMAP objects with reference to quote coordinates: prices and time.
Earlier in the book, we have seen the IndDeltaVolume.mq5 indicator which calculates the delta volume
(tick or real) for each bar. In addition to this representation of the delta volume, there is another one
that is no less popular with users: the market profile. This is the distribution of volumes in the context

---

## Page 1617

Part 7. Advanced language tools
1 61 7
7.1  Resources
of price levels. Such a histogram can be built for the entire window, for a given depth (for example,
within a day), or for a single bar.
It is the last option that we implement in the form of a new indicator DeltaVolumeProfile.mq5. We have
already considered the main technical details of the tick history request within the framework of the
above indicator, so now we will focus mainly on the graphical component.
Flag ShowSplittedDelta in the input variable will control how volumes are displayed: broken down by
buy/sell directions or collapsed.
input bool ShowSplittedDelta = true;
There will be no buffers in the indicator. It will calculate and display a histogram for a specific bar at
the user's request, and specifically, by clicking on this bar. Thus, we will use the OnChartEvent handler.
In this handler, we get screen coordinates, recalculate them into price and time, and call some helper
function RequestData, which starts the calculation.
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_CLICK)
   {
      datetime time;
      double price;
      int window;
      ChartXYToTimePrice(0, (int)lparam, (int)dparam, window, time, price);
      time += PeriodSeconds() / 2;
      const int b = iBarShift(_Symbol, _Period, time, true);
      if(b != -1 && window == 0)
      {
         RequestData(b, iTime(_Symbol, _Period, b));
      }
   }
   ...
}
To fill it, we need the DeltaVolumeProfile class, which is built to be similar to the class CalcDeltaVolume
from IndDeltaVolume.mq5.
The new class describes variables that take into account the volume calculation method (tickType), the
type of price on which the chart is built (barType), mode from the ShowSplittedDelta input variable (will
be placed in a member variable delta), as well as a prefix for generated objects on the chart.

---

## Page 1618

Part 7. Advanced language tools
1 61 8
7.1  Resources
class DeltaVolumeProfile
{
   const COPY_TICKS tickType;
   const ENUM_SYMBOL_CHART_MODE barType;
   const bool delta;
   
   static const string prefix;
   ...
public:
   DeltaVolumeProfile(const COPY_TICKS type, const bool d) :
      tickType(type), delta(d),
      barType((ENUM_SYMBOL_CHART_MODE)SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE))
   {
   }
   
   ~DeltaVolumeProfile()
   {
      ObjectsDeleteAll(0, prefix, 0); // TODO: delete resources
   }
   ...
};
   
static const string DeltaVolumeProfile::prefix = "DVP";
   
DeltaVolumeProfile deltas(TickType, ShowSplittedDelta);
The tick type can be changed to the TRADE_TICKS value only for trading instruments for which real
volumes are available. By default, the INFO_TICKS mode is enabled, which works on all instruments.
Ticks for a particular bar are requested by the createProfileBar method.

---

## Page 1619

Part 7. Advanced language tools
1 61 9
7.1  Resources
   int createProfileBar(const int i)
   {
      MqlTick ticks[];
      const datetime time = iTime(_Symbol, _Period, i);
      // prev and next - time limits of the bar
      const datetime prev = time;
      const datetime next = prev + PeriodSeconds();
      ResetLastError();
      const int n = CopyTicksRange(_Symbol, ticks, COPY_TICKS_ALL,
         prev * 1000, next * 1000 - 1);
      if(n > -1 && _LastError == 0)
      {
         calcProfile(i, time, ticks);
      }
      else
      {
         return -_LastError;
      }
      return n;
   }
Direct analysis of ticks and calculation of volumes is performed in the protected method calcProfile. In
it, first of all, we find out the price range of the bar and its size in pixels.
   void calcProfile(const int b, const datetime time, const MqlTick &ticks[])
   {
      const string name = prefix + (string)(ulong)time;
      const double high = iHigh(_Symbol, _Period, b);
      const double low = iLow(_Symbol, _Period, b);
      const double range = high - low;
      
      ObjectCreate(0, name, OBJ_BITMAP, 0, time, high);
      
      int x1, y1, x2, y2;
      ChartTimePriceToXY(0, 0, time, high, x1, y1);
      ChartTimePriceToXY(0, 0, time, low, x2, y2);
      
      const int h = y2 - y1 + 1;
      const int w = (int)(ChartGetInteger(0, CHART_WIDTH_IN_PIXELS)
         / ChartGetInteger(0, CHART_WIDTH_IN_BARS));
      ...
Based on this information, we create an OBJ_BITMAP object, allocate an array for the image, and
create a resource. The background of the whole picture is empty (transparent). Each object is
anchored by the upper midpoint to the High price of its bar and has a width of one bar.

---

## Page 1620

Part 7. Advanced language tools
1 620
7.1  Resources
      uint data[];
      ArrayResize(data, w * h);
      ArrayInitialize(data, 0);
      ResourceCreate(name + (string)ChartID(), data, w, h, 0, 0, w, COLOR_FORMAT_ARGB_NORMALIZE);
         
      ObjectSetString(0, name, OBJPROP_BMPFILE, "::" + name + (string)ChartID());
      ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
      ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_UPPER);
      ...
This is followed by the calculation of volumes in ticks of the passed array. The number of price levels is
equal to the height of the bar in pixels (h). Usually, it is less than the price range in points, and
therefore the pixels act as a kind of basket for calculating statistics. If on a small timeframe, the range
of points is smaller than the size in pixels, the histogram will be visually sparse. Volumes of purchases
and sales are accumulated separately in plus and minus arrays.

---

## Page 1621

Part 7. Advanced language tools
1 621 
7.1  Resources
      long plus[], minus[], max = 0;
      ArrayResize(plus, h);
      ArrayResize(minus, h);
      ArrayInitialize(plus, 0);
      ArrayInitialize(minus, 0);
      
      const int n = ArraySize(ticks);
      for(int j = 0; j < n; ++j)
      {
         const double p1 = price(ticks[j]); // returns Bid or Last
         const int index = (int)((high - p1) / range * (h - 1));
         if(tickType == TRADE_TICKS)
         {
            // if real volumes are available, we can take them into account
            if((ticks[j].flags & TICK_FLAG_BUY) != 0)
            {
               plus[index] += (long)ticks[j].volume;
            }
            if((ticks[j].flags & TICK_FLAG_SELL) != 0)
            {
               minus[index] += (long)ticks[j].volume;
            }
         }
         else // tickType == INFO_TICKS or tickType == ALL_TICKS
         if(j > 0)
         {
           // if there are no real volumes,
           // price movement up/down is an estimate of the volume type
            if((ticks[j].flags & (TICK_FLAG_ASK | TICK_FLAG_BID)) != 0)
            {
               const double d = (((ticks[j].ask + ticks[j].bid)
                              - (ticks[j - 1].ask + ticks[j - 1].bid)) / _Point);
               if(d > 0) plus[index] += (long)d;
               else minus[index] -= (long)d;
            }
         }
         ...
To normalize the histogram, we find the maximum value.

---

## Page 1622

Part 7. Advanced language tools
1 622
7.1  Resources
         if(delta)
         {
            if(plus[index] > max) max = plus[index];
            if(minus[index] > max) max = minus[index];
         }
         else
         {
            if(fabs(plus[index] - minus[index]) > max)
               max = fabs(plus[index] - minus[index]);
         }
      }
      ...
Finally, the resulting statistics are output to the graphics buffer data and sent to the resource. Buy
volumes are displayed in blue, and sell volumes are shown in red. If the net mode is enabled, then the
amount is displayed in green.
      for(int i = 0; i < h; i++)
      {
         if(delta)
         {
            const int dp = (int)(plus[i] * w / 2 / max);
            const int dm = (int)(minus[i] * w / 2 / max);
            for(int j = 0; j < dp; j++)
            {
               data[i * w + w / 2 + j] = ColorToARGB(clrBlue);
            }
            for(int j = 0; j < dm; j++)
            {
               data[i * w + w / 2 - j] = ColorToARGB(clrRed);
            }
         }
         else
         {
            const int d = (int)((plus[i] - minus[i]) * w / 2 / max);
            const int sign = d > 0 ? +1 : -1;
            for(int j = 0; j < fabs(d); j++)
            {
               data[i * w + w / 2 + j * sign] = ColorToARGB(clrGreen);
            }
         }
      }
      ResourceCreate(name + (string)ChartID(), data, w, h, 0, 0, w, COLOR_FORMAT_ARGB_NORMALIZE);
   }
Now we can return to the RequestData function: its task is to call the createProfileBar method and
handle errors (if any).

---

## Page 1623

Part 7. Advanced language tools
1 623
7.1  Resources
void RequestData(const int b, const datetime time, const int count = 0)
{
   Comment("Requesting ticks for ", time);
   if(deltas.createProfileBar(b) <= 0)
   {
      Print("No data on bar ", b, ", at ", TimeToString(time),
         ". Sending event for refresh...");
      ChartSetSymbolPeriod(0, _Symbol, _Period); // request to update the chart
      EventChartCustom(0, TRY_AGAIN, b, count + 1, NULL);
   }
   Comment("");
}
The only error-handling strategy is to try requesting the ticks again because they might not have had
time to load. For this purpose, the function sends a custom TRY_AGAIN message to the chart and
processes it itself.
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   ...
   else if(id == CHARTEVENT_CUSTOM + TRY_AGAIN)
   {
      Print("Refreshing... ", (int)dparam);
      const int b = (int)lparam;
      if((int)dparam < 5)
      {
         RequestData(b, iTime(_Symbol, _Period, b), (int)dparam);
      }
      else
      {
         Print("Give up. Check tick history manually, please, then click the bar again");
      }
   }
}
We repeat this process no more than 5 times, because the tick history can have a limited depth, and it
makes no sense to load the computer for no reason.
The DeltaVolumeProfile class also has the mechanism for processing the message
CHARTEVENT_CHART_CHANGE in order to redraw existing objects in case of changing the size or scale
of the chart. Details can be found in the source code.
The result of the indicator is shown in the following image.

---

## Page 1624

Part 7. Advanced language tools
1 624
7.1  Resources
Displaying per-bar histograms of separate volumes in graphic resources
Note that the histograms are not displayed immediately after drawing the indicator: you have to click
on the bar to calculate its histogram.
7.2 Custom symbols
One of the interesting technical features of MetaTrader 5 is the support for custom financial
instruments. These are the symbols that are defined not by the broker on the server side but by the
trader directly in the terminal.
Custom symbols can be added to the Market Watch list along with standard symbols. The charts of
such symbols with them can be used in a usual way.
The easiest way to create a custom symbol is to specify its calculation formula in the corresponding
property. To do this, from the terminal interface, call the context menu in the Market Watch window,
execute the Symbols command, go to the symbol hierarchy and its Custom branch, and press the
Create symbol button. As a result, a dialog for setting the properties of the new symbol will open. At the
same place, you can import external tick history (tab Ticks) or quotes (tab Bars) into similar tools, from
files. This is discussed in detail in the MetaTrader 5 documentation.
However, the MQL5 API provides the most complete control over custom symbols.
For custom symbols, the API provides a group of functions working with Financial instruments and
Market Watch. In particular, such symbols can be listed from the program using standard functions
such as SymbolsTotal, SymbolName, and SymbolInfo. We have already briefly touched on this possibility
and provided an example in the section on Custom symbol properties. A distinctive feature of a custom
symbol is the enabled flag (property) SYMBOL_CUSTOM.

---

## Page 1625

Part 7. Advanced language tools
1 625
7.2 Custom symbols
Using the built-in functions, you can splice Futures, generate random time series with specified
characteristics, emulate renko, equal-range bars, equivolume, and other non-standard types of charts
(for example, second timeframes). Also, unlike importing static files, software-controlled custom
symbols can be generated in realt-time based on the data from web services such as cryptocurrency
exchanges. The conversation on integrating MQL programs with the web is still ahead, but this
possibility cannot be ignored.
A custom symbol can be easily used to test strategies in the tester or as an additional method of
technical analysis. However, this technology has its limitations.
Due to the fact that custom symbols are defined in the terminal and not on the server, they cannot be
traded online. In particular, if you create a renko chart, trading strategies based on it will need to be
adapted in one way or another so that trading signals and trades are actually separated by different
symbols: artificial user and real brokerage. We will look at a couple of solutions to the problem.
In addition, since the duration of all bars of one timeframe is the same in the platform, any emulation of
bars with different periods (Renko, equivolume, etc.) is usually based on the smaller of the available M1 
timeframes and does not provide a full time synchronization with reality. In other words, ticks belonging
to such a bar are forced to have an artificial time within 60 seconds, even if a renko "brick" or a bar of
a given volume actually required much more time to form. Otherwise, if we put ticks in real time, they
would form the next M1  bars, violating the rules of renko or equivolume. Moreover, there are situations
when a renko "brick" or other artificial bar should be created with a time interval smaller than 1  minute
from the previous bar (for example, when there is increased fast volatility). In such cases, it will be
necessary to change the time of historical bars in quotes of the custom instrument (shift them to the
left "retroactively") or put future times on new bars (which is highly undesirable). This problem cannot
be solved in a general way within the framework of user-defined symbols technology.
7.2.1  Creating and deleting custom symbols
The first two functions you need in order to work with custom symbols are CustomSymbolCreate and
CustomSymbolDelete.
bool CustomSymbolCreate(const string name, const string path = "", const string origin = NULL)
The function creates a custom symbol with the specified name (name) in the specified group (path)
and, if necessary, with the properties of an exemplary symbol – its name can be specified in the
parameter origin.
The name parameter should be a simple identifier, without hierarchy. If necessary, one or more
required levels of groups (subfolders) should be specified in the parameter path, with the delimiter
character being a backslash '\' (the forward slash is not supported here, unlike the file system). The
backslash must be doubled in literal strings ("\\").
By default, if the path string is empty ("" or NULL), the symbol is created directly in the Custom folder,
which is allocated in the general hierarchy of symbols for user symbols. If the path is filled, it is created
inside the Custom folder to the full depth (if there are no corresponding folders yet).
The name of a symbol, as well as the name of a group of any level, can contain Latin letters and
numbers, without punctuation marks, spaces, and special characters. Additionally, only '.', '_', '&', and
'#' are allowed.
The name must be unique in the entire symbol hierarchy, regardless of which group the symbol is
supposed to be created in. If a symbol with the same name already exists, the function will return false

---

## Page 1626

Part 7. Advanced language tools
1 626
7.2 Custom symbols
and will set the error code 5300 (ERR_NOT_CUSTOM_SYMBOL) or 5304
(ERR_CUSTOM_SYMBOL_EXIST) in _ LastError.
Note that if the last (or even the only) element of the hierarchy in the path string exactly matches the
name (case sensitive), then it is treated as a symbol name that is part of the path and not as a folder.
For example, if the name and path contain the strings "Example" and "MQL5Book\\Example",
respectively, then the symbol "Example" will be created in the "Custom\\MQL5Book\\" folder. At the
same time, if we change the name to "example", we will get the "example" symbol in the "Custom\
\MQL5Book\\Example" folder.
This feature has another consequence. The SYMBOL_PATH property returns the path along with the
symbol name at the end. Therefore, if we transfer its value without changes from some exemplary
symbol to a newly created one, we will get the following effect: a folder with the name of the old symbol
will be created, inside which a new symbol will appear. Thus, if you want to create a custom symbol in
the same group as the original symbol, you must strip the name of the original symbol from the string
obtained from the SYMBOL_PATH property.
We will demonstrate the side effect of copying the SYMBOL_PATH property in an example in the next
section. However, this effect can also be used as a positive one. In particular, by creating several of its
symbols based on one original symbol, copying SYMBOL_PATH will ensure that all new symbols are
placed in the folder with the name of the original, i.e., it will group the symbols according to their
prototype symbol.
The SYMBOL_PATH property for custom symbols always starts with the "Custom\\" folder (this prefix
is added automatically).
Name length is limited to 31  characters. When the limit is exceeded, CustomSymbolCreate will return
false and set error code 5302 (ERR_CUSTOM_SYMBOL_NAME_LONG).
The maximum length of the parameter path is 1 27 characters, including "Custom\\", group separators
"\\", and the symbol name, if it is specified at the end.
The origin parameter allows you to optionally specify the name of the symbol from which the properties
of the created custom symbol will be copied. After creating a custom symbol, you can change any of
its properties to the desired value using the appropriate functions (see CustomSymbolSet functions).
If a non-existent symbol is given as the origin parameter, then the custom symbol will be created
"empty", as if the parameter origin was not specified. This will raise error 4301 
(ERR_MARKET_UNKNOWN_SYMBOL).
In a new symbol created "blank", all properties are set to their default values. For example, the
contract size is 1 00000, the number of digits in the price is 4, the margin calculation is carried out
according to Forex rules, and charting is based on the Bid prices.
When you specify origin, only settings are transferred from this symbol to the new symbol but not
quotes or ticks as they should be generated separately. This will be discussed in the following sections.
Creating a symbol does not automatically add it to Market Watch. So, this must be done explicitly
(manually or programmatically). Without quotes, the chart window will be empty.
bool CustomSymbolDelete(const string name)
The function deletes a custom symbol with the specified name. Not only settings are deleted, but also
all data on the symbol (quotes and ticks). It is worth noting, that the history is not deleted
immediately, but only after some delay, which can be a source of problems if you intend to recreate a

---

## Page 1627

Part 7. Advanced language tools
1 627
7.2 Custom symbols
symbol with the same name (we will touch on this point in the example of the section Adding, replacing,
and deleting quotes).
Only a custom symbol can be deleted. Also, you cannot delete a symbol selected in Market Watch or a
symbol having an open chart. Please note that a symbol can also be selected implicitly, without
displaying in the visible list (in such cases, the SYMBOL_VISIBLE property is false, and the
SYMBOL_SELECT property is true). Such a symbol first must be "hidden" by calling
SymbolSelect("name", false) before attempting to delete: otherwise, we get a
CUSTOM_SYMBOL_SELECTED (5306) error.
If deleting a symbol leaves an empty folder (or folder hierarchy), it is also deleted.
For example, let's create a simple script CustomSymbolCreateDelete.mq5. In the input parameters, you
can specify a name, a path, and an exemplary symbol.
input string CustomSymbol = "Dummy";         // Custom Symbol Name
input string CustomPath = "MQL5Book\\Part7"; // Custom Symbol Folder
input string Origin;
In the OnStart handler, let's check if there is already a symbol with the given name. If not, then after
the confirmation from the user, we will create such a symbol. If the symbol is already there and it's a
custom symbol, let's delete it with the user's permission (this will make it easier to clean up after the
experiment is over).
void OnStart()
{
   bool custom = false;
   if(!PRTF(SymbolExist(CustomSymbol, custom)))
   {
      if(IDYES == MessageBox("Create new custom symbol?", "Please, confirm", MB_YESNO))
      {
         PRTF(CustomSymbolCreate(CustomSymbol, CustomPath, Origin));
      }
   }
   else
   {
      if(custom)
      {
         if(IDYES == MessageBox("Delete existing custom symbol?", "Please, confirm", MB_YESNO))
         {
            PRTF(CustomSymbolDelete(CustomSymbol));
         }
      }
      else
      {
         Print("Can't delete non-custom symbol");
      }
   }
}
Two consecutive runs with default options should result in the following log entries.

---

## Page 1628

Part 7. Advanced language tools
1 628
7.2 Custom symbols
SymbolExist(CustomSymbol,custom)=false / ok
Create new custom symbol?
CustomSymbolCreate(CustomSymbol,CustomPath,Origin)=true / ok
   
SymbolExist(CustomSymbol,custom)=true / ok
Delete existing custom symbol?
CustomSymbolDelete(CustomSymbol)=true / ok
Between runs, you can open the symbol dialog in the terminal and check that the corresponding
custom symbol has appeared in the symbol hierarchy.
7.2.2 Custom symbol properties
Custom symbols have the same properties as broker-provided symbols. The properties are read by the
standard functions discussed in the chapter on financial instruments.
The properties of custom symbols can be set by a special group of CustomSymbolSet functions, one
function for each fundamental type (integer, real, string).
bool CustomSymbolSetInteger(const string name, ENUM_SYMBOL_INFO_INTEGER property, long
value)
bool CustomSymbolSetDouble(const string name, ENUM_SYMBOL_INFO_DOUBLE property, double
value)
bool CustomSymbolSetString(const string name, ENUM_SYMBOL_INFO_STRING property, string
value)
The functions set for a custom symbol named name a value of property to value. All existing properties
are grouped into enumerations ENUM_SYMBOL_INFO_INTEGER, ENUM_SYMBOL_INFO_DOUBLE,
ENUM_SYMBOL_INFO_STRING, which were considered element by element in the sections of the
aforementioned chapter.
The functions return an indication of success (true) or error (false). One possible problem for errors is
that not all properties are allowed to change. When trying to set a read-only property, we get the error
CUSTOM_SYMBOL_PROPERTY_WRONG (5307). If you try to write an invalid value to the property, you
will get a CUSTOM_SYMBOL_PARAMETER_ERROR (5308) error.
Please note that the minute and tick history of a custom symbol is completely deleted if any of the
following properties are changed in the symbol specification:
• SYMBOL_CHART_MODE – price type used to build bars (Bid or Last)
• SYMBOL_DIGITS – number of decimal places in price values
• SYMBOL_POINT – value of one point
• SYMBOL_TRADE_TICK_SIZE – the value of one tick, the minimum allowable price change
• SYMBOL_TRADE_TICK_VALUE – price change cost per tick (see also
SYMBOL_TRADE_TICK_VALUE_PROFIT, SYMBOL_TRADE_TICK_VALUE_LOSS)
• SYMBOL_FORMULA – formula for price calculation
If a custom symbol is calculated by a formula, then after deleting its history, the terminal will
automatically try to create a new history using the updated properties. However, for programmatically
generated symbols, the MQL program itself must take care of the recalculation.

---

## Page 1629

Part 7. Advanced language tools
1 629
7.2 Custom symbols
Editing individual properties is most in demand for modifying custom symbols created earlier (after
specifying the third parameter origin in the CustomSymbolCreate function).
In other cases, changing properties in bulk can cause subtle effects. The point is that properties are
internally linked and changing one of them may require a certain state of other properties in order for
the operation to complete successfully. Moreover, setting some properties leads to automatic changes
in others.
In the simplest example, after setting the SYMBOL_DIGITS property, you will find that the
SYMBOL_POINT property has changed as well. Here is the less obvious case: assigning
SYMBOL_CURRENCY_MARGIN or SYMBOL_CURRENCY_PROFIT has no effect on Forex symbols, since
the system assumes currency names to occupy the first 3 and next 3 letters of the name
("XXXYYY[suffix]"), respectively. Please note that immediately after the creation of an "empty" symbol,
it is by default considered a Forex symbol, and therefore these properties cannot be set for it without
first changing the market.
When copying or setting symbol properties, be aware that the platform implies some specifics. In
particular, the property SYMBOL_TRADE_CALC_MODE has a default value of 0 (immediately after the
symbol is created, but before any property is set), while 0 in the ENUM_SYMBOL_CALC_MODE
enumeration corresponds to the SYMBOL_CALC_MODE_FOREX member. At the same time, special
naming rules are implied for Forex symbols in the form XXXYYY (where XXX and YYY are currency
codes) plus an optional suffix. Therefore, if you do not change SYMBOL_TRADE_CALC_MODE to another
required mode in advance, substrings of the specified symbol name (the first and second triple of
symbols) will automatically fall into the properties of the base currency (SYMBOL_CURRENCY_BASE)
and profit currency (SYMBOL_CURRENCY_PROFIT). For example, if you specify the name "Dummy", it
will be split into 2 pseudo-currencies "Dum" and "my".
Another nuance is that before setting the value of SYMBOL_POINT with an accuracy of N decimal
places, you need to ensure that SYMBOL_DIGITS is at least N.
The book comes with the script CustomSymbolProperties.mq5, which allows you to experiment with
creating copies of the symbol of the current chart and study the resulting effects in practice. In
particular, you can choose the name of the symbol, its path, and the direction of bypassing (setting) all
supported properties, direct or reverse in terms of property numbering in the language. The script uses
a special class CustomSymbolMonitor, which is a wrapper for the above built-in functions: we will
describe it later.
7.2.3 Setting margin rates
Previously, we studied the SymbolInfoMarginRate function, which returns the margin rates per symbol
set by the broker. For a custom symbol, we are free to set these rates using the function
CustomSymbolSetMarginRate.
bool CustomSymbolSetMarginRate(const string name, ENUM_ORDER_TYPE orderType, double initial,
double maintenance)
The function sets margin rates depending on the type and direction of the order (according to the
orderType value from the ENUM_ORDER_TYPE enumeration). The rates for calculating the initial and
maintenance margin (collateral for each lot of an opened or existing position) are transmitted,
respectively, in the initial and maintenance parameters.
The final margin amounts are determined based on several symbol properties
(SYMBOL_TRADE_CALC_MODE, SYMBOL_MARGIN_INITIAL, SYMBOL_MARGIN_MAINTENANCE, and

---

## Page 1630

Part 7. Advanced language tools
1 630
7.2 Custom symbols
others) described in the section Margin requirements, so they should also be set on the custom symbol
if needed.
The function will return an indicator of success (true) or error (false).
With the help of this function and the properties related to margin calculation, you can emulate trading
conditions of servers that are unavailable for one reason or another, and debug your MQL programs in
the tester.
7.2.4 Configuring quoting and trading sessions
Two API functions allow setting quoting and trading sessions of a custom instrument. These two
concepts were discussed in the section Schedules of trading and quoting sessions.
bool CustomSymbolSetSessionQuote(const string name, ENUM_DAY_OF_WEEK dayOfWeek,
   uint sessionIndex, datetime from, datetime to)
bool CustomSymbolSetSessionTrade(const string name, ENUM_DAY_OF_WEEK dayOfWeek,
   uint sessionIndex, datetime from, datetime to)
CustomSymbolSetSessionQuote sets the start and end time of the quoting session specified by number
(sessionIndex) for a specific day of the week (dayOfWeek). CustomSymbolSetSessionTrade does the
same for trading sessions.
Session numbering starts from 0.
Sessions can only be added sequentially, that is, a session with index 1  can only be added if there
already exists a session with index 0. If this rule is violated, a new session will not be created, and the
function will return false.
Date values in the from and to parameters are measured in seconds, and from should be less than to.
The range is limited to two days, from 0 (00 hours 00 minutes 00 seconds) to 1 72800 (23 hours 59
minutes 59 seconds the next day). The day change was required in order to be able to specify sessions
that begin before midnight and end after midnight. This situation often occurs when the exchange is
located on the other side of the world relative to the broker (dealer) servers.  
If zero start and end parameters (from = 0 and to = 0) are passed for the sessionIndex session, then it
is deleted, and the numbering of the next sessions (if any) is shifted down.
Trading sessions cannot go beyond quoting ones.
For example, we can create a copy of an instrument for a different time zone by shifting the intraday
quote time and session schedule for debugging the robot in different conditions, like with any exotic
brokers.
7.2.5 Adding, replacing, and deleting quotes
A custom symbol is populated quotes by two built-in functions: CustomRatesUpdate and
CustomRatesReplace. At the input, in addition to the name of the symbol, both expect an array of
structures MqlRates for the M1  timeframe (higher timeframes are completed automatically from M1 ).
CustomRatesReplace has an additional pair of parameters (from and to) that define the time range to
which history editing is limited.

---

## Page 1631

Part 7. Advanced language tools
1 631 
7.2 Custom symbols
int CustomRatesUpdate(const string symbol, const MqlRates &rates[], uint count = WHOLE_ARRAY)
int CustomRatesReplace(const string symbol, datetime from, datetime to, const MqlRates &rates[],
uint count = WHOLE_ARRAY)
CustomRatesUpdate adds missing bars to the history and replaces existing matching bars with data
from the array.
CustomRatesReplace completely replaces the history in the specified time interval with the data from
the array.
The difference between the functions is due to different scenarios of the intended application. The
differences are listed in more detail in the following table.
CustomRatesUpdate
CustomRatesReplace
Applies the elements of the passed MqlRates array
to the history, regardless of their timestamps
Applies only those elements of the passed
MqlRates array that fall within the specified range
Leaves untouched in the history those M1  bars
that were already there before the function call
and do not coincide in time with the bars in the
array
Leaves untouched all history out of range
Replaces existing history bars with the bars from
the array when timestamps match
Completely deletes existing history bars in the
specified range
Inserts elements from the array as "new" bars if
there are no matches with the old bars
Inserts the bars from the array that fall within the
relevant range into the specified history range
Data in the rates array must be represented by valid OHLC prices, and bar opening times must not
contain seconds.
An interval within from and to is set inclusive: from is equal to the time of the first bar to be processed
and to is equal to the time of the last.
The following diagram illustrates these rules more clearly. Each unique timestamp for a bar is
designated by its own Latin letter. Available bars in the history are shown in capital letters, while bars in
the array are shown in lowercase. The character '-' is a gap in the history or in the array for the
corresponding time.
History                        ABC-EFGHIJKLMN-PQRST------    B
Array                          -------hijk--nopqrstuvwxyz    A
Result of CustomRatesUpdate    ABC-EFGhijkLMnopqrstuvwxyz    R
Result of CustomRatesReplace   ABC-E--hijk--nopqrstuvw---    S
                                    ^                ^
                                    |from          to|    TIME
The optional parameter count sets the number of elements in the rates array that should be used
(others will be ignored). This allows you to partially process the passed array. The default value
WHOLE_ARRAY means the entire array.
The quotes history of a custom symbol can be deleted entirely or partially using the
CustomRatesDelete function.

---

## Page 1632

Part 7. Advanced language tools
1 632
7.2 Custom symbols
int CustomRatesDelete(const string symbol, datetime from, datetime to)
Here, the parameters from and to also set the time range of removed bars. To cover the entire history,
specify 0 and LONG_MAX.
All three functions return the number of processed bars: updated or deleted. In case of an error, the
result is -1 .
It should be noted that quotes of a custom symbol can be formed not only by adding ready-made bars
but also by arrays of ticks or even a sequence of individual ticks. The relevant functions will be
presented in the next section. When adding ticks, the terminal will automatically calculate bars based
on them. The difference between these methods is that the custom tick history allows you to test MQL
programs in the "real" ticks mode, while the history of bars only will force you to either limit yourself to
the OHLC M1  or open price modes or rely on the tick emulation implemented by the tester.
In addition, adding ticks one at a time allows you to simulate standard events OnTick and OnCalculate
on the chart of a custom symbol, which "animates" the chart similar to tools available online, and
launches the corresponding handler functions in MQL programs if they are plotted on the chart. But we
will talk about this in the next section.
As an example of using new functions, let's consider the script CustomSymbolRandomRates.mq5. It is
designed to generate random quotes on the principle of "random walk" or noise existing quotes. The
latter can be used to assess the stability of an Expert Advisor.
To check the correctness of the formation of quotes, we will also support the mode in which a complete
copy of the original instrument is created, on the chart of which the script was launched.
All modes are collected in the RANDOMIZATION enumeration.
enum RANDOMIZATION
{
   ORIGINAL,
   RANDOM_WALK,
   FUZZY_WEAK,
   FUZZY_STRONG,
};
We implement quotes noise with two levels of intensity: weak and strong.
In the input parameters, you can choose, in addition to the mode, a folder in the symbol hierarchy, a
date range, and a number to initialize the random generator (to be able to reproduce the results).
input string CustomPath = "MQL5Book\\Part7";    // Custom Symbol Folder
input RANDOMIZATION RandomFactor = RANDOM_WALK;
input datetime _From;                           // From (default: 120 days ago)
input datetime _To;                             // To (default: current time)
input uint RandomSeed = 0;
By default, when no dates are specified, the script generates quotes for the last 1 20 days. The value 0
in the RandomSeed parameter means random initialization.
The name of the symbol is generated based on the symbol of the current chart and the selected
settings.

---

## Page 1633

Part 7. Advanced language tools
1 633
7.2 Custom symbols
const string CustomSymbol = _Symbol + "." + EnumToString(RandomFactor)
   + (RandomSeed ? "_" + (string)RandomSeed : "");
At the beginning of OnStart we will prepare and check the data.
datetime From;
datetime To;
   
void OnStart()
{
   From = _From == 0 ? TimeCurrent() - 60 * 60 * 24 * 120 : _From;
   To = _To == 0 ? TimeCurrent() / 60 * 60 : _To;
   if(From > To)
   {
      Alert("Date range must include From <= To");
      return;
   }
   
   if(RandomSeed != 0) MathSrand(RandomSeed);
   ...
Since the script will most likely need to be run several times, we will provide the ability to delete the
custom symbol created earlier, with a preliminary confirmation request from the user.
   bool custom = false;
   if(PRTF(SymbolExist(CustomSymbol, custom)) && custom)
   {
      if(IDYES == MessageBox(StringFormat("Delete custom symbol '%s'?", CustomSymbol),
         "Please, confirm", MB_YESNO))
      {
         if(CloseChartsForSymbol(CustomSymbol))
         {
            Sleep(500); // wait for the changes to take effect (opportunistically)
            PRTF(CustomRatesDelete(CustomSymbol, 0, LONG_MAX));
            PRTF(SymbolSelect(CustomSymbol, false));
            PRTF(CustomSymbolDelete(CustomSymbol));
         }
      }
   }
   ...
The helper function CloseChartsForSymbol is not shown here (those who wish can look at the attached
source code): its purpose is to view the list of open charts and close those where the working symbol is
the custom symbol being deleted (without this, the deletion will not work).
More important is to pay attention to calling CustomRatesDelete with a full range of dates. If it is not
done, the data of the previous user symbol will remain on the disk for a while in the history database
(folder bases/Custom/history/<symbol-name>). In other words, the CustomSymbolDelete call, which is
shown in the last line above, is not enough to actually clear the custom symbol from the terminal.
If the user decides to immediately create a symbol with the same name again (and we provide such an
opportunity in the code below), then the old quotes can be mixed into the new ones.

---

## Page 1634

Part 7. Advanced language tools
1 634
7.2 Custom symbols
Further, upon the user's confirmation, the process of generating quotes is launched. This is done by the
GenerateQuotes function (see further).
   if(IDYES == MessageBox(StringFormat("Create new custom symbol '%s'?", CustomSymbol),
      "Please, confirm", MB_YESNO))
   {
      if(PRTF(CustomSymbolCreate(CustomSymbol, CustomPath, _Symbol)))
      {
         if(RandomFactor == RANDOM_WALK)
         {
            CustomSymbolSetInteger(CustomSymbol, SYMBOL_DIGITS, 8);
         }
         
         CustomSymbolSetString(CustomSymbol, SYMBOL_DESCRIPTION, "Randomized quotes");
      
         const int n = GenerateQuotes();
         Print("Bars M1 generated: ", n);
         if(n > 0)
         {
            SymbolSelect(CustomSymbol, true);
            ChartOpen(CustomSymbol, PERIOD_M1);
         }
      }
   }
If successful, the newly created symbol is selected in Market Watch and a chart opens for it. Along the
way, setting a pair of properties is demonstrated here: SYMBOL_DIGITS and SYMBOL_DESCRIPTION.
In the function GenerateQuotes it is required to request quotes of the original symbol for all modes
except RANDOM_WALK.
int GenerateQuotes()
{
   MqlRates rates[];
   MqlRates zero = {};
   datetime start;     // time of the current bar
   double price;       // last closing price
   
   if(RandomFactor != RANDOM_WALK)
   {
      if(PRTF(CopyRates(_Symbol, PERIOD_M1, From, To, rates)) <= 0)
      {
         return 0; // error
      }
      if(RandomFactor == ORIGINAL)
      {
         return PRTF(CustomRatesReplace(CustomSymbol, From, To, rates));
      }
      ...
It is important to recall that CopyRates is affected by the limit on the number of bars on the chart,
which is set in the terminal settings, affects.

---

## Page 1635

Part 7. Advanced language tools
1 635
7.2 Custom symbols
In the case of ORIGINAL mode, we simply forward the resulting array rates into the
CustomRatesReplace function. For noise modes, we set the specially selected price and start variables
to the initial values of price and time from the first bar.
      price = rates[0].open;
      start = rates[0].time;
   }
   ...
In random walk mode, quotes are not needed, so we just allocate the rates array for future random M1 
bars.
   else
   {
      ArrayResize(rates, (int)((To - From) / 60) + 1);
      price = 1.0;
      start = From;
   }
   ...
Further in the loop through the rates array, random values are added either to the noisy prices of the
original symbol or "as is". In the RANDOM_WALK mode, we ourselves are responsible for increasing the
time in the variable start. In other modes, the time is already in the initial quotes.

---

## Page 1636

Part 7. Advanced language tools
1 636
7.2 Custom symbols
   const int size = ArraySize(rates);
   
   double hlc[3]; // future High Low Close (in unknown order)
   for(int i = 0; i < size; ++i)
   {
      if(RandomFactor == RANDOM_WALK)
      {
         rates[i] = zero;             // zeroing the structure
         rates[i].time = start += 60; // plus a minute to the last bar
         rates[i].open = price;       // start from the last price
         hlc[0] = RandomWalk(price);
         hlc[1] = RandomWalk(price);
         hlc[2] = RandomWalk(price);
      }
      else
      {
         double delta = 0;
         if(i > 0)
         {
            delta = rates[i].open - price; // cumulative correction
         }
         rates[i].open = price;
         hlc[0] = RandomWalk(rates[i].high - delta);
         hlc[1] = RandomWalk(rates[i].low - delta);
         hlc[2] = RandomWalk(rates[i].close - delta);
      }
      ArraySort(hlc);
      
      rates[i].high = fmax(hlc[2], rates[i].open);
      rates[i].low = fmin(hlc[0], rates[i].open);
      rates[i].close = price = hlc[1];
      rates[i].tick_volume = 4;
   }
   ...
Based on the closing price of the last bar, 3 random values are generated (using the RandomWalk
function). The maximum and minimum of them become, respectively, the prices High and Low of a new
bar. The average is the price Close.
At the end of the loop, we pass the array to CustomRatesReplace.
   return PRTF(CustomRatesReplace(CustomSymbol, From, To, rates));
}
In the RandomWalk function, an attempt was made to simulate a distribution with wide tails, which is
typical for real quotes.

---

## Page 1637

Part 7. Advanced language tools
1 637
7.2 Custom symbols
double RandomWalk(const double p)
{
   const static double factor[] = {0.0, 0.1, 0.01, 0.05};
   const static double f = factor[RandomFactor] / 100;
   const double r = (rand() - 16383.0) / 16384.0; // [-1,+1]
   const int sign = r >= 0 ? +1 : -1;
   if(r != 0)
   {
      return p + p * sign * f * sqrt(-log(sqrt(fabs(r))));
   }
   return p;
}
The scatter coefficients of random variables depend on the mode. For example, weak noise adds (or
subtracts) a maximum of 1  hundredth of a percent, and strong noise adds 5 hundredths of a percent of
the price.
While running, the script outputs a detailed log like this one:
Create new custom symbol 'GBPUSD.RANDOM_WALK'?
CustomSymbolCreate(CustomSymbol,CustomPath,_Symbol)=true / ok
CustomRatesReplace(CustomSymbol,From,To,rates)=171416 / ok
Bars M1 generated: 171416
Let's see what we get as a result.
The following image shows several implementations of a random walk (the visual overlay is done in a
graphical editor, in reality, each custom symbol opens in a separate window as usual).
Quote options for custom symbols with random walk

---

## Page 1638

Part 7. Advanced language tools
1 638
7.2 Custom symbols
And here is how noisy GBPUSD quotes look like (original in black, color with noise). First, in a weak
version.
GBPUSD quotes with low noise
And then with strong noise.


---

## Page 1639

Part 7. Advanced language tools
1 639
7.2 Custom symbols
GBPUSD quotes with strong noise
Larger discrepancies are obvious, though with the preservation of local features.
7.2.6 Adding, replacing, and removing ticks
The MQL5 API allows you to generate the history of a custom symbol not only at the bar level but also
at the tick level. Thus, it is possible to achieve greater realism when testing and optimizing Expert
Advisors, as well as to emulate real-time updating of charts of custom symbols, broadcasting your ticks
to them. The set of ticks transferred to the system is automatically taken into account when forming
bars. In other words, there is no need to call the functions from the previous section that operate on
structures MqlRates, if more detailed information about price changes for the same period is provided in
the form of ticks, namely the MqlTick arrays of structures. The only advantage of per-bar MqlRates
quotes is the performance and memory efficiency.
There are two functions for adding ticks: CustomTicksAdd and CustomTicksReplace. The first one adds
interactive ticks that arrive at the Market Watch window (from which they are automatically transferred
by the terminal to the tick database) and that generate corresponding events in MQL programs. The
second one writes ticks directly to the tick database.
int CustomTicksAdd(const string symbol, const MqlTick &ticks[], uint count = WHOLE_ARRAY)
The CustomTicksAdd function adds data from the ticks array to the price history of a custom symbol
specified in symbol. By default, if the count setting is equal to WHOLE_ARRAY, the entire array is
added. If necessary, you can specify a smaller number and download only a part of the ticks.
Please note that the custom symbol must be selected in the Market Watch window by the time of the
function call. For symbols not selected in Market Watch, you need to use the CustomTicksReplace
function (see further).
The array of tick data must be sorted by time in ascending order, i.e. it is required that the following
conditions are met: ticks[i].time_ msc <= ticks[j ].time_ msc for all i < j .
The function returns the number of added ticks or -1  in case of an error.
The CustomTicksAdd function broadcasts ticks to the chart in the same way as if they came from the
broker's server. Usually, the function is applied for one or more ticks. In this case, they are "played" in
the Market Watch window, from which they are saved in the tick database.
However, when a large amount of data is transferred in one call, the function changes its behavior to
save resources. If more than 256 ticks are transmitted, they are divided into two parts. The first part
(large) is immediately written directly to the tick database (as does CustomTicksReplace). The second
part, consisting of the last (most recent) 1 28 ticks, is passed to the Market Watch window, and after
that is saved by the terminal in the database.
The MqlTick structure has two fields with time values: time (tick time in seconds) and time_ msc (tick
time in milliseconds). Both values are dated starting from 01 /01 /1 970. The filled (non-null) time_ msc
field takes precedence over time. Note that time is filled in seconds as a result of recalculation based
on the formula time_ msc / 1 000. If the time_ msc field is zero, the value from the time field is used, and
the time_ msc field in turn gets the value in milliseconds from the formula time * 1 000. If both fields are
equal to zero, the current server time (accurate to milliseconds) is put into a tick.
Of the two fields describing the volume, volume_ real has a higher priority than volume.

---

## Page 1640

Part 7. Advanced language tools
1 640
7.2 Custom symbols
Depending on what other fields are filled in a particular array element (structure MqlTick), the system
sets flags for the saved tick in the flags field:
• ticks[i].bid – TICK_FLAG_BID (the tick changed the Bid price)
• ticks[i].ask – TICK_FLAG_ASK (the tick changed the Ask price)
• ticks[i].last – TICK_FLAG_LAST (the tick changed the price of the last trade)
• ticks[i].volume or ticks[i].volume_real – TICK_FLAG_VOLUME (the tick changed volume)
If the value of some field is less than or equal to zero, the corresponding flag is not written to the flags
field.
The TICK_FLAG_BUY and TICK_FLAG_SELL flags are not added to the history of a custom symbol.
The CustomTicksReplace function completely replaces the price history of the custom symbol in the
specified time interval with the data from the passed array.
int CustomTicksReplace(const string symbol, long from_msc, long to_msc,
   const MqlTick &ticks[], uint count = WHOLE_ARRAY)
The interval is set by the parameters from_ msc and to_ msc, in milliseconds since 01 /01 /1 970. Both
values are included in the interval.
The array ticks must be ordered in chronological order of ticks' arrival, which corresponds to
increasing, or rather, non-decreasing time since ticks with the same time often occur in a row in a
stream with millisecond accuracy.
The count parameter can be used to process a part of the array.
The ticks are replaced sequentially day by day before the time specified in to_ msc, or until an error
occurs in the tick order. The first day in the specified range is processed first, then goes the next day,
and so on. As soon as a discrepancy between the tick time and the ascending (non-decreasing) order is
detected, the tick replacement process stops on the current day. In this case, the ticks for the
previous days will be successfully replaced, while the current day (at the time of the wrong tick) and all
remaining days in the specified interval will remain unchanged. The function will return -1 , with the
error code in _ LastError being 0 ("no error").
If the ticks array does not have data for some period within the general interval between from_ msc and
to_ msc (inclusive), then after executing the function, the history of the custom symbol will have a gap
corresponding to the missing data.
If there is no data in the tick database in the specified time interval, CustomTicksReplace will add ticks
to it from the array ticks.  
The CustomTicksDelete function can be used to delete all ticks in the specified time interval.
int CustomTicksDelete(const string symbol, long from_msc, long to_msc)
The name of the custom symbol being edited is set in the symbol parameter, and the interval to be
cleared is set by the parameters from_ msc and to_ msc (inclusive), in milliseconds.
The function returns the number of ticks removed or -1  in case of an error.
Attention! Deleting ticks with CustomTicksDelete leads to the automatic removal of the
corresponding bars! However, calling CustomRatesDelete, i.e., removing bars, does not remove
ticks!

---

## Page 1641

Part 7. Advanced language tools
1 641 
7.2 Custom symbols
To master the material in practice, we will solve several applied problems using the newly considered
functions.
To begin with, let's touch on such an interesting task as creating a custom symbol based on a real
symbol but with a reduced tick density. This will speed up testing and optimization, as well as reduce
resource consumption (primarily RAM) compared to the mode based on real ticks while maintaining an
acceptable, close to ideal, quality of the process.
Speeding up testing and optimization
Traders often seek ways to speed up Expert Advisor optimization and testing processes. Among the
possible solutions, there are obvious ones, for which you can simply change the settings (when it is
allowed), and there are more time-consuming ones that require the adaptation of an Expert Advisor
or a test environment. 
Among the first type of solutions are: 
· Reducing the optimization space by eliminating some parameters or reducing their step;
· Reducing the optimization period;
· Switching to the tick simulation mode of lower quality (for example, from real ones to OHLC M1 );
· Enabling profit calculation in points instead of money;
· Upgrading the computer;
· Using MQL Cloud or additional local network computers.
 Among the second type of development-related solutions are:
· Code profiling, on the basis of which you can eliminate "bottlenecks" in the code;
· If possible, use the resource-efficient calculation of indicators, that is, without the  #property
tester_ everytick_ calculate directive;
· Transferring indicator algorithms (if they are used) directly into the Expert Advisor code: indicator
calls impose certain overhead costs;
· Eliminating graphics and objects;
· Caching calculations, if possible;
· Reducing the number of simultaneously open positions and placed orders (their calculation on
each tick can become noticeable with a large number);
· Full virtualization of settlements, orders, deals, and positions: the built-in accounting mechanism,
due to its versatility, multicurrency support, and other features, has its own overheads, which can
be eliminated by performing similar actions in the MQL5 code (although this option is the most
time-consuming).
 Tick density reduction belongs to an intermediate type of solution: it requires the programmatic
creation of a custom symbol but does not affect the source code of the Expert Advisor.
A custom symbol with reduced ticks will be generated by the script CustomSymbolFilterTicks.mq5. The
initial instrument will be the working symbol of the chart on which the script is launched. In the input
parameters, you can specify the folder for the custom symbol and the start date for history processing.
By default, if no date is given, the calculation is made for the last 1 20 days.
input string CustomPath = "MQL5Book\\Part7"; // Custom Symbol Folder
input datetime _Start;                       // Start (default: 120 days back)
The name of the symbol is formed from the name of the source instrument and the ".TckFltr" suffix.
Later we will add to it the designation of the tick reducing method.

---

## Page 1642

Part 7. Advanced language tools
1 642
7.2 Custom symbols
string CustomSymbol = _Symbol + ".TckFltr";
const uint DailySeconds = 60 * 60 * 24;
datetime Start = _Start == 0 ? TimeCurrent() - DailySeconds * 120 : _Start;
For convenience, in the OnStart handler, it is possible to delete a previous copy of a symbol if it already
exists.
void OnStart()
{
   bool custom = false;
   if(PRTF(SymbolExist(CustomSymbol, custom)) && custom)
   {
      if(IDYES == MessageBox(StringFormat("Delete existing custom symbol '%s'?", CustomSymbol),
         "Please, confirm", MB_YESNO))
      {
         SymbolSelect(CustomSymbol, false);
         CustomRatesDelete(CustomSymbol, 0, LONG_MAX);
         CustomTicksDelete(CustomSymbol, 0, LONG_MAX);
         CustomSymbolDelete(CustomSymbol);
      }
      else
      {
         return;
      }
   }
Next, upon the consent of the user, a symbol is created. The history is filled with tick data in the
auxiliary function GenerateTickData. If successful, the script adds a new symbol to Market Watch and
opens the chart.
   if(IDYES == MessageBox(StringFormat("Create new custom symbol '%s'?", CustomSymbol),
      "Please, confirm", MB_YESNO))
   {
      if(PRTF(CustomSymbolCreate(CustomSymbol, CustomPath, _Symbol)))
      {
         CustomSymbolSetString(CustomSymbol, SYMBOL_DESCRIPTION, "Prunned ticks by " + EnumToString(Mode));
         if(GenerateTickData())
         {
            SymbolSelect(CustomSymbol, true);
            ChartOpen(CustomSymbol, PERIOD_H1);
         }
      }
   }
}
The GenerateTickData function processes ticks in a loop in portions, per day. Ticks per day are
requested by calling CopyTicksRange. Then they need to be reduced in one way or another, which is
implemented by the TickFilter class which we will show below. Finally, the tick array is added to the
custom symbol history using CustomTicksReplace.

---

## Page 1643

Part 7. Advanced language tools
1 643
7.2 Custom symbols
bool GenerateTickData()
{
   bool result = true;
   datetime from = Start / DailySeconds * DailySeconds; // round up to the beginning of the day
   ulong read = 0, written = 0;
   uint day = 0;
   const uint total = (uint)((TimeCurrent() - from) / DailySeconds + 1);
   MqlTick array[];
   
   while(!IsStopped() && from < TimeCurrent())
   {
      Comment(TimeToString(from, TIME_DATE), " ", day++, "/", total);
      
      const int r = CopyTicksRange(_Symbol, array, COPY_TICKS_ALL,
         from * 1000L, (from + DailySeconds) * 1000L - 1);
      if(r < 0)
      {
         Alert("Error reading ticks at ", TimeToString(from, TIME_DATE));
         result = false;
         break;
      }
      read += r;
      
      if(r > 0)
      {
         const int t = TickFilter::filter(Mode, array);
         const int w = CustomTicksReplace(CustomSymbol,
            from * 1000L, (from + DailySeconds) * 1000L - 1, array);
         if(w <= 0)
         {
            Alert("Error writing custom ticks at ", TimeToString(from, TIME_DATE));
            result = false;
            break;
         }
         written += w;
      }
      from += DailySeconds;
   }
   
   if(read > 0)
   {
      PrintFormat("Done ticks - read: %lld, written: %lld, ratio: %.1f%%",
         read, written, written * 100.0 / read);
   }
   Comment("");
   return result;
}
Error control and counting of processed ticks are implemented at all stages. It the end, we output to
the log the number of initial and remaining ticks, as well as the "compression" factor.

---

## Page 1644

Part 7. Advanced language tools
1 644
7.2 Custom symbols
Now let's turn directly to the tick reducing technique. Obviously, there can be many approaches, with
each of them being better or worse suited to a specific trading strategy. We will offer 3 basic versions
combined in the class TickFilter (TickFilter.mqh). Also, to complete the picture, the mode of copying
ticks without reduction is also supported.
Thus, the following modes are implemented in the class:
• No reduction
• Skipping sequences of ticks with a monotonous price change without a reversal (a la "zig-zag")
• Skipping price fluctuations within the spread
• Recording only ticks with a fractal configuration when the Bid or Ask price represents an extremum
between two adjacent ticks
These modes are described as elements of the FILTER_MODE enumeration.
class TickFilter
{
public:
   enum FILTER_MODE
   {
      NONE,
      SEQUENCE,
      FLUTTER,
      FRACTALS,
   };
   ...
Each of the modes is implemented by a separate static method that accepts as input an array of ticks
that needs to be thinned out. Editing an array is performed in place (without allocating a new output
array).
   static int filterBySequences(MqlTick &data[]);
   static int filterBySpreadFlutter(MqlTick &data[]);
   static int filterByFractals(MqlTick &data[]);
All methods return the number of ticks left (reduced array size).
To unify the execution of the procedure in different modes, the filter method is provided. For the mode
NONE the data array stays the same.
   static int filter(FILTER_MODE mode, MqlTick &data[])
   {
      switch(mode)
      {
      case SEQUENCE: return filterBySequences(data);
      case FLUTTER: return filterBySpreadFlutter(data);
      case FRACTALS: return filterByFractals(data);
      }
      return ArraySize(data);
   }
For example, here is how filtering by monotonous sequences of ticks is implemented in the
filterBySequences method.

---

## Page 1645

Part 7. Advanced language tools
1 645
7.2 Custom symbols
   static int filterBySequences(MqlTick &data[])
   {
      const int size = ArraySize(data);
      if(size < 3) return size;
      
      int index = 2;
      bool dirUp = data[1].bid - data[0].bid + data[1].ask - data[0].ask > 0;
      
      for(int i = 2; i < size; i++)
      {
         if(dirUp)
         {
            if(data[i].bid - data[i - 1].bid + data[i].ask - data[i - 1].ask < 0)
            {
               dirUp = false;
               data[index++] = data[i];
            }
         }
         else
         {
            if(data[i].bid - data[i - 1].bid + data[i].ask - data[i - 1].ask > 0)
            {
               dirUp = true;
               data[index++] = data[i];
            }
         }
      }
      return ArrayResize(data, index);
   }
And here is what fractal thinning looks like.

---

## Page 1646

Part 7. Advanced language tools
1 646
7.2 Custom symbols
   static int filterByFractals(MqlTick &data[])
   {
      int index = 1;
      const int size = ArraySize(data);
      if(size < 3) return size;
      
      for(int i = 1; i < size - 2; i++)
      {
         if((data[i].bid < data[i - 1].bid && data[i].bid < data[i + 1].bid)
         || (data[i].ask > data[i - 1].ask && data[i].ask > data[i + 1].ask))
         {
            data[index++] = data[i];
         }
      }
      
      return ArrayResize(data, index);
   }
Let's sequentially create a custom symbol for EURUSD in several tick density reduction modes and
compare their performance, i.e., the degree of "compression", how fast the testing will be, and how the
trading performance of the Expert Advisor will change.
For example, thinning out sequences of ticks gives the following results (for a one-and-a-half-year
history on MQ Demo).
   Create new custom symbol 'EURUSD.TckFltr-SE'?
   Fixing SYMBOL_TRADE_TICK_VALUE: 0.0 <<< 1.0
   true  SYMBOL_TRADE_TICK_VALUE 1.0 -> SUCCESS (0)
   Fixing SYMBOL_TRADE_TICK_SIZE: 0.0 <<< 1e-05
   true  SYMBOL_TRADE_TICK_SIZE 1e-05 -> SUCCESS (0)
   Number of found discrepancies: 2
   Fixed
   Done ticks - read: 31553509, written: 16927376, ratio: 53.6%
For modes of smoothing fluctuations and for fractals, the indicators are different:
   EURUSD.TckFltr-FL will be updated
   Done ticks - read: 31568782, written: 22205879, ratio: 70.3%
   ...   
   Create new custom symbol 'EURUSD.TckFltr-FR'?
   ...
   Done ticks - read: 31569519, written: 12732777, ratio: 40.3%
For practical trading experiments based on compressed ticks, we need an Expert Advisor. Let's take
the adapted version of BandOsMATicks.mq5, in which, compared to the original, trading on each tick is
enabled (in the method SimpleStrategy::trade the lineif(lastBar == iTime(_ Symbol, _ Period, 0)) return
false; is disabled), and the values of signal indicators are taken from bars 0 and 1  (previously there
were only completed bars 1  and 2).
Let's run the Expert Advisor using the dates range from the beginning of 2021  to June 1 , 2022. The
settings are attached in the file MQL5/Presets/MQL5Book/BandOsMAticks.set. The general behavior of
the balance curve in all modes is quite similar.

---

## Page 1647

Part 7. Advanced language tools
1 647
7.2 Custom symbols
Combined charts of test balances in different modes by ticks
The shift of equivalent extremums of different curves horizontally is caused by the fact that the
standard report chart uses not the time but the number of trades for the horizontal coordinate, which,
of course, differs due to the accuracy of triggering trading signals for different tick bases.
The differences in performance metrics are shown in the following table (N - number of trades, $ -
profit, PF - profit factor, RF - recovery factor, DD - drawdown):
Mode
Ticks
Time 
m m : ss. m se c
Memory
N
$
PF
RF
DD
R e a l 
3 1 0 0 2 9 1 9 
0 2 : 4 5 . 2 5 1 
8 3 5  M b 
9 6 2 
1 6 6 . 2 4 
1 . 3 2 
2 . 8 8 
5 4 . 9 9 
E m u l a t i on 
2 5 8 0 8 1 3 9 
0 1 : 5 8 . 1 3 1 
6 8 7  M b 
9 2 8 
1 7 1 . 9 4 
1 . 3 4 
3 . 4 4 
4 7 . 6 4 
O H L C M 1 
 2 0 8 4 8 2 0 
0 0 : 1 1 . 0 9 4 
2 2 4  M b 
8 5 6 
1 9 3 . 5 2 
1 . 3 9 
3 . 9 7 
4 6 . 5 5 
S e q u e n c e 
1 6 3 1 0 2 3 6 
0 1 : 2 4 . 7 8 4 
5 5 9  M b 
8 6 0 
1 6 8 . 9 5 
1 . 3 4 
2 . 9 2 
5 5 . 1 6 
F l u t t e r 
2 1 3 6 2 6 1 6 
0 1 : 5 2 . 1 7 2 
6 2 3  M b 
9 2 0 
1 7 9 . 7 5 
1 . 3 7 
3 . 6 0 
4 7 . 2 8 
F r a c t a l 
1 2 2 7 0 8 5 4 
0 1 : 0 4 . 7 5 6 
4 3 0  M b 
8 6 6 
1 4 2 . 1 9 
1 . 2 7 
2 . 4 7 
5 4 . 8 0 
We will consider the test based on real ticks to be the most reliable and evaluate the rest by how close
it is to this test. Obviously, the OHLC M1  mode showed the highest speed and lower resource costs due
to a significant loss of accuracy (the mode at opening prices was not considered). It exhibits over-
optimistic financial results.
Among the three modes with artificially compressed ticks, "Sequence" is the closest to the real one in
terms of a set of indicators. It is 2 times faster than the real one in terms of time and is 1 .5 times
more efficient in terms of memory consumption. The "Flutter" mode seems to better preserve the
original number of trades. The fastest and least memory-demanding fractal mode, of course, takes
more time and resources than OHLC M1 , but it does not overestimate trading scores.
Keep in mind that tick reduction algorithms may work differently or, conversely, give poor results with
different trading strategies, financial instruments, and even the tick history of a particular broker.
Conduct research with your Expert Advisors and in your work environment.
As part of the second example of working with custom symbols, let's consider an interesting feature
provided by tick translation using CustomTicksAdd.
Many traders use trading panels – programs with interactive controls for performing arbitrary trading
actions manually. You have to practice working with them mainly online because the tester imposes

---

## Page 1648

Part 7. Advanced language tools
1 648
7.2 Custom symbols
some restrictions. First of all, the tester does not support on-chart events and objects. This causes the
controls to stop functioning. Also, in the tester, you cannot apply arbitrary objects for graphics markup.
Let's try to solve these problems.
We can generate a custom symbol based on historical ticks in slow motion. Then the chart of such a
symbol will become an analog of a visual tester.
This approach has several advantages:
• Standard behavior of all chart events
• Interactive application and setting of indicators
• Interactive application and adjustment of objects
• Timeframe switching on the go
• Test on history up to the current time, including today (the standard tester does not allow testing
today)
Regarding the last point, we note that the developers of MetaTrader 5 deliberately prohibited checking
trading on the last (current) day, although it is sometimes needed to quickly find errors (in the code or
in the trading strategy).
It is also potentially interesting to modify prices on the go (increasing the spread, for example).
Based on the chart of such a custom symbol, later we can implement a manual trading emulator on
historical data.
The symbol generator will be the non-trading Expert Advisor CustomTester.mq5. In its input
parameters, we will provide an indication of the placement of a new custom symbol in the symbol
hierarchy, the start date in the past for tick translation (and building custom symbol quotes), as well as
a timeframe for the chart, which will be automatically opened for visual testing.
input string CustomPath = "MQL5Book\\Part7"; // Custom Symbol Folder
input datetime _Start;                       // Start (120-day indent by default)
input ENUM_TIMEFRAMES Timeframe = PERIOD_H1;
The name of the new symbol is constructed from the symbol name of the current chart and the
".Tester" suffix.
string CustomSymbol = _Symbol + ".Tester";
If the start date is not specified in the parameters, the Expert Advisor will indent back by 1 20 days
from the current date.
const uint DailySeconds = 60 * 60 * 24;
datetime Start = _Start == 0 ? TimeCurrent() - DailySeconds * 120 : _Start;
Ticks will be read from the history of real ticks of the working symbol in batches for the whole day at
once. The pointer to the day being read is stored in the Cursor variable.
bool FirstCopy = true;
// additionally 1 day ago, because otherwise, the chart will not update immediately
datetime Cursor = (Start / DailySeconds - 1) * DailySeconds; // round off at the border of the day
The ticks of one day to be reproduced will be requested in the Ticks array, from where they will be
translated in small batches of size step to the chart of a custom symbol.

---

## Page 1649

Part 7. Advanced language tools
1 649
7.2 Custom symbols
MqlTick Ticks[];       // ticks for the "current" day in the past
int Index = 0;         // position in ticks within a day
int Step = 32;         // fast forward 32 ticks at a time (default)
int StepRestore = 0;   // remember the speed for the duration of the pause
long Chart = 0;        // created custom symbol chart
bool InitDone = false; // sign of completed initialization
To play ticks at a constant rate, let's start the timer in OnInit.
void OnInit()
{
   EventSetMillisecondTimer(100);
}
   
void OnTimer()
{
   if(!GenerateData())
   {
      EventKillTimer();
   }
}
The ticks will be generated by the GenerateData function. Immediately after launching, when the
InitDone flag is reset, we will try to create a new symbol or clear the old quotes and ticks if the custom
symbol already exists.

---

## Page 1650

Part 7. Advanced language tools
1 650
7.2 Custom symbols
bool GenerateData()
{
   if(!InitDone)
   {
      bool custom = false;
      if(PRTF(SymbolExist(CustomSymbol, custom)) && custom)
      {
         if(IDYES == MessageBox(StringFormat("Clean up existing custom symbol '%s'?",
            CustomSymbol), "Please, confirm", MB_YESNO))
         {
            PRTF(CustomRatesDelete(CustomSymbol, 0, LONG_MAX));
            PRTF(CustomTicksDelete(CustomSymbol, 0, LONG_MAX));
            Sleep(1000);
            MqlRates rates[1];
            MqlTick tcks[];
            if(PRTF(CopyRates(CustomSymbol, PERIOD_M1, 0, 1, rates)) == 1
            || PRTF(CopyTicks(CustomSymbol, tcks) > 0))
            {
               Alert("Can't delete rates and Ticks, internal error");
               ExpertRemove();
            }
         }
         else
         {
            return false;
         }
      }
      else
      if(!PRTF(CustomSymbolCreate(CustomSymbol, CustomPath, _Symbol)))
      {
         return false;
      }
      ... // (A)
At this point, we'll omit something at (A) and come back to this point later.
After creating the symbol, we select it in Market Watch and open a chart for it.
 SymbolSelect(CustomSymbol, true);
      Chart = ChartOpen(CustomSymbol, Timeframe);
      ... // (B)
      ChartSetString(Chart, CHART_COMMENT, "Custom Tester");
      ChartSetInteger(Chart, CHART_SHOW_OBJECT_DESCR, true);
      ChartRedraw(Chart);
      InitDone = true;
   }
   ...
A couple of lines (B) are missing here too; they are related to future improvements, but not required
yet.
If the symbol has already been created, we start broadcasting ticks in batches of Step ticks, but no
more than 256. This limitation is related to the specifics of the CustomTicksAdd function.

---

## Page 1651

Part 7. Advanced language tools
1 651 
7.2 Custom symbols
   else
   {
      for(int i = 0; i <= (Step - 1) / 256; ++i)
      if(Step > 0 && !GenerateTicks())
      {
         return false;
      }
   }
   return true;
}
The helper function GenerateTicks broadcasts ticks in batches of Step ticks (but not more than 256),
reading them from the daily array Ticks by offset Index. When the array is empty or we have read it to
the end, we request the next day's ticks by calling FillTickBuffer.
bool GenerateTicks()
{
   if(Index >= ArraySize(Ticks)) // daily array is empty or read to the end
   {
      if(!FillTickBuffer()) return false; // fill the array with ticks per day
   }
   
   const int m = ArraySize(Ticks);
   MqlTick array[];
   const int n = ArrayCopy(array, Ticks, 0, Index, fmin(fmin(Step, 256), m));
   if(n <= 0) return false;
   
   ResetLastError();
   if(CustomTicksAdd(CustomSymbol, array) != ArraySize(array) || _LastError != 0)
   {
      Print(_LastError); // in case of ERR_CUSTOM_TICKS_WRONG_ORDER (5310)
      ExpertRemove();
   }
   Comment("Speed: ", (string)Step, " / ", STR_TIME_MSC(array[n - 1].time_msc));
   Index += Step; // move forward by 'Step' ticks
   return true;
}
The FillTickBuffer function uses CopyTicksRange for operation.

---

## Page 1652

Part 7. Advanced language tools
1 652
7.2 Custom symbols
bool FillTickBuffer()
{
   int r;
   ArrayResize(Ticks, 0);
   do
   {
      r = PRTF(CopyTicksRange(_Symbol, Ticks, COPY_TICKS_ALL, Cursor * 1000L,
         (Cursor + DailySeconds) * 1000L - 1));
      if(r > 0 && FirstCopy)
      {
         // NB: this pre-call is only needed to display the chart
         // from "Waiting for update" state
         PRTF(CustomTicksReplace(CustomSymbol, Cursor * 1000L,
            (Cursor + DailySeconds) * 1000L - 1, Ticks));
         FirstCopy = false;
         r = 0;
      }
      Cursor += DailySeconds;
   }
   while(r == 0 && Cursor < TimeCurrent()); // skip non-trading days
   Index = 0;
   return r > 0;
}
When the Expert Advisor is stopped, we will also close the dependent chart (so that it is not duplicated
at the next start).
void OnDeinit(const int)
{
   if(Chart != 0)
   {
      ChartClose(Chart);
   }
   Comment("");
}
At this point, the Expert Advisor could be considered complete, but there is a problem. The thing is
that, for one reason or another, the properties of a custom symbol are not copied "as is" from the
original working symbol, at least in the current implementation of the MQL5 API. This applies even to
very important properties, such as SYMBOL_TRADE_TICK_VALUE, SYMBOL_TRADE_TICK_SIZE. If we
print the values of these properties immediately after calling CustomSymbolCreate(CustomSymbol,
CustomPath, _ Symbol), we will see zeros there.
To organize the checking of properties, their comparison and, if necessary, correction, we have written
a special class CustomSymbolMonitor (CustomSymbolMonitor.mqh) derived from SymbolMonitor. You
can study its internal structure on your own, while here we will only present the public interface.
Constructors allow you to create a custom symbol monitor, specifying an exemplary working symbol
(by name in a string, or from The SymbolMonitor object) which serves as a source of settings.

---

## Page 1653

Part 7. Advanced language tools
1 653
7.2 Custom symbols
class CustomSymbolMonitor: public SymbolMonitor
{
public:
   CustomSymbolMonitor(); // sample - _Symbol
   CustomSymbolMonitor(const string s, const SymbolMonitor *m = NULL);
   CustomSymbolMonitor(const string s, const string other);
   
   //set/replace sample symbol   
   void inherit(const SymbolMonitor &m);
   
   // copy all properties from the sample symbol in forward or reverse order
   bool setAll(const bool reverseOrder = true, const int limit = UCHAR_MAX);
   
   // check all properties against the sample, return the number of corrections
   int verifyAll(const int limit = UCHAR_MAX);
   
   // check the specified properties with the sample, return the number of corrections
   int verify(const int &properties[]);
   
   // copy the given properties from the sample, return true if they all applied
   bool set(const int &properties[]);
   
   // copy the specific property from the sample, return true if applied
   template<typename E>
   bool set(const E e);
   
   bool set(const ENUM_SYMBOL_INFO_INTEGER property, const long value) const
   {
      return CustomSymbolSetInteger(name, property, value);
   }
   
   bool set(const ENUM_SYMBOL_INFO_DOUBLE property, const double value) const
   {
      return CustomSymbolSetDouble(name, property, value);
   }
   
   bool set(const ENUM_SYMBOL_INFO_STRING property, const string value) const
   {
      return CustomSymbolSetString(name, property, value);
   }
};
Since custom symbols, unlike standard symbols, allow you to set your own properties, a triple of set
methods has been added to the class. In particular, they are used to batch transfer the properties of a
sample and check the success of these actions in other class methods.
We can now return to the custom symbol generator and its source code snippet, as indicated earlier by
the comment (A).

---

## Page 1654

Part 7. Advanced language tools
1 654
7.2 Custom symbols
      // (A) check important properties and set them in "manual" mode
      SymbolMonitor sm; // _Symbol
      CustomSymbolMonitor csm(CustomSymbol, &sm);
      int props[] = {SYMBOL_TRADE_TICK_VALUE, SYMBOL_TRADE_TICK_SIZE};
      const int d1 = csm.verify(props); // check and try to fix
      if(d1)
      {
         Print("Number of found discrepancies: ", d1); // number of edits
         if(csm.verify(props)) // check again
         {
            Alert("Custom symbol can not be created, internal error!");
            return false; // symbol cannot be used without successful edits
         }
         Print("Fixed");
      }
Now you can run the CustomTester.mq5 Expert Advisor and observe how quotes are dynamically formed
in the automatically opened chart as well as how ticks are forwarded from history in the Market Watch
window.
However, this is done at a constant rate of 32 ticks per 0.1  second. It is desirable to change the
playback speed on the go at the request of the user, both up and down. Such control can be organized,
for example, from the keyboard.
Therefore, you need to add the OnChartEvent handler. As we know, for the CHARTEVENT_KEYDOWN
event, the program receives the code of the pressed key in the lparam parameter, and we pass it to
the CheckKeys function (see below). A fragment (C), closely related to (B), had to be postponed for the
time being and we will return to it shortly.
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   ... // (C)
   if(id == CHARTEVENT_KEYDOWN) // these events only arrive while the chart is active!
   {
      CheckKeys(lparam);
   }
}
In the CheckKeys function, we are processing the "up arrow" and "down arrow" keys to increase and
decrease the playback speed. In addition, the "pause" key allows you to completely suspend the
process of "testing" (transmission of ticks). Pressing "pause" again resumes work at the same speed.

---

## Page 1655

Part 7. Advanced language tools
1 655
7.2 Custom symbols
void CheckKeys(const long key)
{
   if(key == VK_DOWN)
   {
      Step /= 2;
      if(Step > 0)
      {
         Print("Slow down: ", Step);
         ChartSetString(Chart, CHART_COMMENT, "Speed: " + (string)Step);
      }
      else
      {
         Print("Paused");
         ChartSetString(Chart, CHART_COMMENT, "Paused");
         ChartRedraw(Chart);
      }
   }
   else if(key == VK_UP)
   {
      if(Step == 0)
      {
         Step = 1;
         Print("Resumed");
         ChartSetString(Chart, CHART_COMMENT, "Resumed");
      }
      else
      {
         Step *= 2;
         Print("Speed up: ", Step);
         ChartSetString(Chart, CHART_COMMENT, "Speed: " + (string)Step);
      }
   }
   else if(key == VK_PAUSE)
   {
      if(Step > 0)
      {
         StepRestore = Step;
         Step = 0;
         Print("Paused");
         ChartSetString(Chart, CHART_COMMENT, "Paused");
         ChartRedraw(Chart);
      }
      else
      {
         Step = StepRestore;
         Print("Resumed");
         ChartSetString(Chart, CHART_COMMENT, "Speed: " + (string)Step);
      }
   }
}

---

## Page 1656

Part 7. Advanced language tools
1 656
7.2 Custom symbols
The new code can be tested in action after first making sure that the chart on which the Expert Advisor
works is active. Recall that keyboard events only go to the active window. This is another problem of
our tester.
Since the user must perform trading actions on the custom symbol chart, the generator window will
almost always be in the background. Switching to the generator window to temporarily stop the flow of
ticks and then resume it is not practical. Therefore, it is required in some way to organize interactive
control from the keyboard directly from the custom symbol window.
For this purpose, a special indicator is suitable, which we can automatically add to the custom symbol
window that opens. The indicator will intercept keyboard events in its own window (window with a
custom symbol) and send them to the generator window.
The source code of the indicator is attached in the file KeyboardSpy.mq5. Of course, the indicator does
not have charts. A pair of input parameters is dedicated to getting the chart ID HostID, where
messages should be send and custom event code EventID, in which interactive events will be packed.
#property indicator_chart_window
#property indicator_plots 0
   
input long HostID;
input ushort EventID;
The main work is done in the OnChartEvent handler.
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN)
   {
      EventChartCustom(HostID, EventID, lparam,
         // this is always 0 when inside iCustom
         (double)(ushort)TerminalInfoInteger(TERMINAL_KEYSTATE_CONTROL),
         sparam);
   }
}
Note that all of the "hotkeys" we have chosen are simple, that is, they do not use shortcuts with
keyboard status keys, such as Ctrl or Shift. This was done by force because inside the indicators
created programmatically (in particular, through iCustom), the keyboard state is not read. In other
words, calling TerminalInfoInteger(TERMINAL_ KEYSTATE_ XYZ) always returns 0. In the handler above,
we've added it just for demonstration purposes, so that you can verify this limitation if you wish, by
displaying the incoming parameters on the "receiving side".
However, single arrow and pause clicks will be transferred to the parent chart normally, and that's
enough for us. The only thing left to do is to integrate the indicator with the Expert Advisor.
In the previously skipped fragment (B), during the initialization of the generator, we will create an
indicator and add it to the custom symbol chart.

---

## Page 1657

Part 7. Advanced language tools
1 657
7.2 Custom symbols
#define EVENT_KEY 0xDED // custom event
      ...
      // (B)
      const int handle = iCustom(CustomSymbol, Timeframe, "MQL5Book/p7/KeyboardSpy",
         ChartID(), EVENT_KEY);
      ChartIndicatorAdd(Chart, 0, handle);
Further along, in fragment (C), we will ensure the receipt of user messages from the indicator and their
transfer to the already known CheckKeys function.
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   // (C)
   if(id == CHARTEVENT_CUSTOM + EVENT_KEY) // notifications from the dependent chart when it is active
   {
      CheckKeys(lparam); // "remote" processing of key presses
   }
   else if(id == CHARTEVENT_KEYDOWN) // these events are only fired while the chart is active!
   {
      CheckKeys(lparam); // standard processing
   }
}
Thus, the playback speed can now be controlled both on the chart with the Expert Advisor and on the
chart of the custom symbol generated by it.
With the new toolkit, you can try interactive work with a chart that "lives in the past". A comment with
the current playback speed or a pause mark is displayed on the graph.
On the chart with the Expert Advisor, the time of the "current" broadcast ticks is displayed in the
comment.
An Expert Advisor that reproduces the history of ticks (and quotes) of a real symbol
There is basically nothing for the user to do in this window (if only the Expert Advisor is deleted and
custom symbol generation is stopped). The tick translation process itself is not visible here. Moreover,
since the Expert Advisor automatically opens a custom symbol chart (where historical quotes are
updated), it is this one that becomes active. To get the above screenshot, we specifically needed to
briefly switch to the original chart.

---

## Page 1658

Part 7. Advanced language tools
1 658
7.2 Custom symbols
Therefore, let's return to the chart of the custom symbol. The way it is smoothly and progressively
updated in the past is already great, but you can’t conduct trading experiments on it. For example, if
you run your usual trading panel on it, its controls, although they will formally work, will not execute
deals since the custom symbol does not exist on the server, and thus you will get errors. This feature is
observed in any programs that are not specially adapted for custom symbols. Let's show an example of
how trading with a custom symbol can be virtualized.
Instead of a trading panel (in order to simplify the example, but without loss of generality), we will take
as a basis the simplest Expert Advisor, CustomOrderSend.mq5, which can perform several trading
actions on keystrokes:
• 'B' – market buy
• 'S' – market sell
• 'U' – placing a limit buy order
• 'L' – placing a limit sell order
• 'C' – close all positions
• 'D' – delete all orders
• 'R' – output a trading report to the journal
In the Expert Advisor input parameters, we will set the volume of one trade (by default, the minimum
lot) and the distance to the stop loss and take profit levels in points.
input double Volume;           // Volume (0 = minimal lot)
input int Distance2SLTP = 0;   // Distance to SL/TP in points (0 = no)
   
const double Lot = Volume == 0 ? SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) : Volume;
If Distance2SLTP is left equal to zero, no protective levels are placed in market orders, and pending
orders are not formed. When Distance2SLTP has a non-zero value, it is used as the distance from the
current price when placing a pending order (either up or down, depending on the command).
Taking into account the previously presented classes from MqlTradeSync.mqh, the above logic is
converted to the following source code.

---

## Page 1659

Part 7. Advanced language tools
1 659
7.2 Custom symbols
#include <MQL5Book/MqlTradeSync.mqh>
   
#define KEY_B 66
#define KEY_C 67
#define KEY_D 68
#define KEY_L 76
#define KEY_R 82
#define KEY_S 83
#define KEY_U 85
   
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN)
   {
      MqlTradeRequestSync request;
      const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      const double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   
      switch((int)lparam)
      {
      case KEY_B:
         request.buy(Lot, 0,
            Distance2SLTP ? ask - point * Distance2SLTP : Distance2SLTP,
            Distance2SLTP ? ask + point * Distance2SLTP : Distance2SLTP);
         break;
      case KEY_S:
         request.sell(Lot, 0,
            Distance2SLTP ? bid + point * Distance2SLTP : Distance2SLTP,
            Distance2SLTP ? bid - point * Distance2SLTP : Distance2SLTP);
         break;
      case KEY_U:
         if(Distance2SLTP)
         {
            request.buyLimit(Lot, ask - point * Distance2SLTP);
         }
         break;
      case KEY_L:
         if(Distance2SLTP)
         {
            request.sellLimit(Lot, bid + point * Distance2SLTP);
         }
         break;
      case KEY_C:
         for(int i = PositionsTotal() - 1; i >= 0; i--)
         {
            request.close(PositionGetTicket(i));
         }
         break;
      case KEY_D:
         for(int i = OrdersTotal() - 1; i >= 0; i--)

---

## Page 1660

Part 7. Advanced language tools
1 660
7.2 Custom symbols
         {
            request.remove(OrderGetTicket(i));
         }
         break;
      case KEY_R:
 // there should be something here...
         break;
      }
   }
}
As we can see, both standard trading API functions and MqlTradeRequestSync methods are used here.
The latter, indirectly, also ends up calling a lot of built-in functions. We need to make this Expert
Advisor trade with a custom symbol.
The simplest, albeit time-consuming idea is to replace all standard functions with their own analogs that
would count orders, deals, positions, and financial statistics in some structures. Of course, this is
possible only in cases where we have the source code of the Expert Advisor, which should be adapted.
An experimental implementation of the approach is demonstrated in the attached file
CustomTrade.mqh. You can familiarize yourself with the full code on your own, since within the
framework of the book we will list only the main points.
First of all, we note that many calculations are made in a simplified form, many modes are not
supported, and a complete check of the data for correctness is not performed. Use the source code as
a starting point for your own developments.
The entire code is wrapped in the CustomTrade namespace to avoid conflicts.
The order, deal, and position entities are formalized as the corresponding classes CustomOrder,
CustomDeal, and CustomPosition. All of them are inheritors of the class MonitorInterface<I,D,S>
::TradeState. Recall that this class already automatically supports the formation of arrays of integer,
real, and string properties for each type of object and its specific triples of enumerations. For example,
CustomOrder looks like that:

---

## Page 1661

Part 7. Advanced language tools
1 661 
7.2 Custom symbols
class CustomOrder: public MonitorInterface<ENUM_ORDER_PROPERTY_INTEGER,
   ENUM_ORDER_PROPERTY_DOUBLE,ENUM_ORDER_PROPERTY_STRING>::TradeState
{
   static long ticket; // order counter and ticket provider
   static int done;    // counter of executed (historical) orders
public:
   CustomOrder(const ENUM_ORDER_TYPE type, const double volume, const string symbol)
   {
      _set(ORDER_TYPE, type);
      _set(ORDER_TICKET, ++ticket);
      _set(ORDER_TIME_SETUP, SymbolInfoInteger(symbol, SYMBOL_TIME));
      _set(ORDER_TIME_SETUP_MSC, SymbolInfoInteger(symbol, SYMBOL_TIME_MSC));
      if(type <= ORDER_TYPE_SELL)
      {
         // TODO: no deferred execution yet
         setDone(ORDER_STATE_FILLED);
      }
      else
      {
         _set(ORDER_STATE, ORDER_STATE_PLACED);
      }
      
      _set(ORDER_VOLUME_INITIAL, volume);
      _set(ORDER_VOLUME_CURRENT, volume);
      
      _set(ORDER_SYMBOL, symbol);
   }
   
   void setDone(const ENUM_ORDER_STATE state)
   {
      const string symbol = _get<string>(ORDER_SYMBOL);
      _set(ORDER_TIME_DONE, SymbolInfoInteger(symbol, SYMBOL_TIME));
      _set(ORDER_TIME_DONE_MSC, SymbolInfoInteger(symbol, SYMBOL_TIME_MSC));
      _set(ORDER_STATE, state);
      ++done;
   }
   
   bool isActive() const
   {
      return _get<long>(ORDER_TIME_DONE) == 0;
   }
   
   static int getDoneCount()
   {
      return done;
   }
};
Note that in the virtual environment of the old "current" time, you cannot use the TimeCurrent function
and the last known time of the custom symbol SymbolInfoInteger(symbol, SYMBOL_ TIME) is taken
instead.

---

## Page 1662

Part 7. Advanced language tools
1 662
7.2 Custom symbols
During virtual trading, current objects and their history are accumulated in arrays of the corresponding
classes.
AutoPtr<CustomOrder> orders[];
CustomOrder *selectedOrders[];
CustomOrder *selectedOrder = NULL;
AutoPtr<CustomDeal> deals[];
CustomDeal *selectedDeals[];
CustomDeal *selectedDeal = NULL;
AutoPtr<CustomPosition> positions[];
CustomPosition *selectedPosition = NULL;
The metaphor for selecting orders, deals, and positions was required to simulate a similar approach in
built-in functions. For them, there are duplicates in the CustomTrade namespace that replace the
originals using macro substitution directives.
#define HistorySelect CustomTrade::MT5HistorySelect
#define HistorySelectByPosition CustomTrade::MT5HistorySelectByPosition
#define PositionGetInteger CustomTrade::MT5PositionGetInteger
#define PositionGetDouble CustomTrade::MT5PositionGetDouble
#define PositionGetString CustomTrade::MT5PositionGetString
#define PositionSelect CustomTrade::MT5PositionSelect
#define PositionSelectByTicket CustomTrade::MT5PositionSelectByTicket
#define PositionsTotal CustomTrade::MT5PositionsTotal
#define OrdersTotal CustomTrade::MT5OrdersTotal
#define PositionGetSymbol CustomTrade::MT5PositionGetSymbol
#define PositionGetTicket CustomTrade::MT5PositionGetTicket
#define HistoryDealsTotal CustomTrade::MT5HistoryDealsTotal
#define HistoryOrdersTotal CustomTrade::MT5HistoryOrdersTotal
#define HistoryDealGetTicket CustomTrade::MT5HistoryDealGetTicket
#define HistoryOrderGetTicket CustomTrade::MT5HistoryOrderGetTicket
#define HistoryDealGetInteger CustomTrade::MT5HistoryDealGetInteger
#define HistoryDealGetDouble CustomTrade::MT5HistoryDealGetDouble
#define HistoryDealGetString CustomTrade::MT5HistoryDealGetString
#define HistoryOrderGetDouble CustomTrade::MT5HistoryOrderGetDouble
#define HistoryOrderGetInteger CustomTrade::MT5HistoryOrderGetInteger
#define HistoryOrderGetString CustomTrade::MT5HistoryOrderGetString
#define OrderSend CustomTrade::MT5OrderSend
#define OrderSelect CustomTrade::MT5OrderSelect
#define HistoryOrderSelect CustomTrade::MT5HistoryOrderSelect
#define HistoryDealSelect CustomTrade::MT5HistoryDealSelect
For example, this is how the MT5HistorySelectByPosition function is implemented.

---

## Page 1663

Part 7. Advanced language tools
1 663
7.2 Custom symbols
bool MT5HistorySelectByPosition(long id)
{
   ArrayResize(selectedOrders, 0);
   ArrayResize(selectedDeals, 0);
  
   for(int i = 0; i < ArraySize(orders); i++)
   {
      CustomOrder *ptr = orders[i][];
      if(!ptr.isActive())
      {
         if(ptr._get<long>(ORDER_POSITION_ID) == id)
         {
            PUSH(selectedOrders, ptr);
         }
      }
   }
   
   for(int i = 0; i < ArraySize(deals); i++)
   {
      CustomDeal *ptr = deals[i][];
      if(ptr._get<long>(DEAL_POSITION_ID) == id)
      {
         PUSH(selectedDeals, ptr);
      }
   }
   return true;
} 
As you can see, all the functions of this group have the MT5 prefix, so that their dual purpose is
immediately clear and it is easy to distinguish them from the functions of the second group.
The second group of functions in the CustomTrade namespace performs utilitarian actions: checks and
updates the states of orders, deals and positions, creates new and deletes old objects in accordance
with the situation. In particular, they include the CheckPositions and CheckOrders functions, which can
be called on a timer or in response to user actions. But you can not do this if you use a couple of other
functions designed to display the current and historical state of the virtual trading account:
• string ReportTradeState() returns a multiline text with a list of open positions and placed orders
• void PrintTradeHistory() displays the history of orders and deals in the log
These functions independently call CheckPositions and CheckOrders to provide you with up-to-date
information.
In addition, there is a function for visualizing positions and active orders on the chart in the form of
objects: DisplayTrades.
The header file CustomTrade.mqh should be included in the Expert Advisor before other headers so that
macro substitution has an effect on all subsequent lines of source codes.

---

## Page 1664

Part 7. Advanced language tools
1 664
7.2 Custom symbols
#include <MQL5Book/CustomTrade.mqh>
#include <MQL5Book/MqlTradeSync.mqh>
Now, the above algorithm CustomOrderSend.mq5 can start "trading" in the virtual environment based
on the current custom symbol (which does not require a server or a standard tester) without any extra
changes.
To quickly display the state, we will start a second timer and periodically change the comment, as well
as display graphical objects.
int OnInit()
{
   EventSetTimer(1);
   return INIT_SUCCEEDED;
}
   
void OnTimer()
{
   Comment(CustomTrade::ReportTradeState());
   CustomTrade::DisplayTrades();
}
To build a report by pressing 'R', we add the OnChartEvent handler.
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN)
   {
      switch((int)lparam)
      {
      ...
      case KEY_R:
         CustomTrade::PrintTradeHistory();
         break;
      }
   }
}
Finally, everything is ready to test the new software package in action.
Run the custom symbol generator CustomTester.mq5 on EURUSD. On the "EURUSD.Tester" chart that
opens, run CustomOrderSend.mq5 and start trading. Below is a picture of the testing process.

---

## Page 1665

Part 7. Advanced language tools
1 665
7.2 Custom symbols
Virtual trading on a custom symbol chart
Here you can see two open long positions (with protective levels) and a pending sell limit order.
After some time, one of the positions is closed (indicated below by a dotted blue line with an arrow),
and a pending sell order is triggered (red line with an arrow), resulting in the following picture.


---

## Page 1666

Part 7. Advanced language tools
1 666
7.2 Custom symbols
Virtual trading on a custom symbol chart
After closing all positions (some by take profit, and the rest by the user's command), a report was
ordered by pressing 'R'.
History Orders:
(1) #1 ORDER_TYPE_BUY 2022.02.15 01:20:50 -> 2022.02.15 01:20:50 L=0.01 @ 1.1306 
(4) #2 ORDER_TYPE_SELL_LIMIT 2022.02.15 02:34:29 -> 2022.02.15 18:10:17 L=0.01 @ 1.13626 [sell limit]
(2) #3 ORDER_TYPE_BUY 2022.02.15 10:08:20 -> 2022.02.15 10:08:20 L=0.01 @ 1.13189 
(3) #4 ORDER_TYPE_BUY 2022.02.15 15:01:26 -> 2022.02.15 15:01:26 L=0.01 @ 1.13442 
(1) #5 ORDER_TYPE_SELL 2022.02.15 15:35:43 -> 2022.02.15 15:35:43 L=0.01 @ 1.13568 
(2) #6 ORDER_TYPE_SELL 2022.02.16 09:39:17 -> 2022.02.16 09:39:17 L=0.01 @ 1.13724 
(4) #7 ORDER_TYPE_BUY 2022.02.16 23:31:15 -> 2022.02.16 23:31:15 L=0.01 @ 1.13748 
(3) #8 ORDER_TYPE_SELL 2022.02.16 23:31:15 -> 2022.02.16 23:31:15 L=0.01 @ 1.13742 
Deals:
(1) #1 [#1] DEAL_TYPE_BUY DEAL_ENTRY_IN 2022.02.15 01:20:50 L=0.01 @ 1.1306 = 0.00 
(2) #2 [#3] DEAL_TYPE_BUY DEAL_ENTRY_IN 2022.02.15 10:08:20 L=0.01 @ 1.13189 = 0.00 
(3) #3 [#4] DEAL_TYPE_BUY DEAL_ENTRY_IN 2022.02.15 15:01:26 L=0.01 @ 1.13442 = 0.00 
(1) #4 [#5] DEAL_TYPE_SELL DEAL_ENTRY_OUT 2022.02.15 15:35:43 L=0.01 @ 1.13568 = 5.08 [tp]
(4) #5 [#2] DEAL_TYPE_SELL DEAL_ENTRY_IN 2022.02.15 18:10:17 L=0.01 @ 1.13626 = 0.00 
(2) #6 [#6] DEAL_TYPE_SELL DEAL_ENTRY_OUT 2022.02.16 09:39:17 L=0.01 @ 1.13724 = 5.35 [tp]
(4) #7 [#7] DEAL_TYPE_BUY DEAL_ENTRY_OUT 2022.02.16 23:31:15 L=0.01 @ 1.13748 = -1.22 
(3) #8 [#8] DEAL_TYPE_SELL DEAL_ENTRY_OUT 2022.02.16 23:31:15 L=0.01 @ 1.13742 = 3.00 
Total: 12.21, Trades: 4
Parentheses indicate position identifiers and square brackets indicate tickets of orders for the
corresponding deals (tickets of both types are preceded by a "hash" '#').
Swaps and commissions are not taken into account here. Their calculation can be added.
We will consider another example of working with custom symbol ticks in the section on custom symbol
trading specifics. We will talk about creating equivolume charts.
7.2.7 Translation of order book changes
If necessary, an MQL program can generate an order book for a custom symbol using the
CustomBookAdd function. This, in particular, can be useful for instruments from external exchanges,
such as cryptocurrencies.
int CustomBookAdd(const string symbol, const MqlBookInfo &books[], uint count = WHOLE_ARRAY)
The function broadcasts the state of the order book to the signed MQL programs for the custom symbol
using data from the books array. The array describes the full state of the order book, that is, all buy
and sell orders. The translated state completely replaces the previous one and becomes available
through the MarketBookGet function.
Using the count parameter, you can specify the number of elements of the books array to be passed to
the function. The entire array is used by default.
The function returns an indicator of success (true) or error (false).
To obtain order books generated by the CustomBookAdd function, an MQL program that requires them
must, as usual, subscribe to the events using MarketBookAdd.

---

## Page 1667

Part 7. Advanced language tools
1 667
7.2 Custom symbols
The update of an order book does not update the Bid and Ask prices of the instrument. To update the
required prices, add ticks using CustomTicksAdd.
The transmitted data is checked for correctness: prices and volumes must be greater than zero, and
for each element, its type, price, and volume must be specified (fields volume and/or volume_ real). If
at least one element of the order book is described incorrectly, the function will return an error.
The Book Depth parameter (SYMBOL_TICKS_BOOKDEPTH) of the custom instrument is also checked.
If the number of sell or buy levels in the translated order book exceeds this value, the extra levels are
discarded.
Volume with increased accuracy volume_ real takes precedence over normal volume. If both values are
specified for the order book element, volume_ real will be used.
Attention! In the current implementation, CustomBookAdd automatically locks the custom symbol
as if it were subscribed to it made by MarketBookAdd, but at the same time, the OnBookEvent
events do not arrive (in theory, the program that generates order books can subscribe to them by
calling MarketBookAdd explicitly and controlling what other programs receive). You can remove this
lock by calling MarketBookRelease. 
This may be required due to the fact that the symbols for which there are subscriptions to the
order book cannot be hidden from Market Watch by any means (until all explicit or implicit
subscriptions are canceled from the programs, and the order book window is closed). As a
consequence, such symbols cannot be deleted.
As an example, let's create a non-trading Expert Advisor PseudoMarketBook.mq5, which will generate a
pseudo-state of the order book from the nearest tick history. This can be useful for symbols for which
the order book is not translated, in particular for Forex. If you wish, you can use such custom symbols
for formal debugging of your own trading algorithms using the order book.
Among the input parameters, we indicate the maximum depth of the order book.
input uint CustomBookDepth = 20;
The name of the custom symbol will be formed by adding the suffix ".Pseudo" to the name of the
current chart symbol.
string CustomSymbol = _Symbol + ".Pseudo";
In the OnInit handler, we create a custom symbol and set its formula to the name of the original
symbol. Thus, we will get a copy of the original symbol automatically updated by the terminal, and we
will not need to trouble ourselves with copying quotes or ticks.

---

## Page 1668

Part 7. Advanced language tools
1 668
7.2 Custom symbols
int OnInit()
{
   bool custom = false;
   if(!PRTF(SymbolExist(CustomSymbol, custom)))
   {
      if(PRTF(CustomSymbolCreate(CustomSymbol, CustomPath, _Symbol)))
      {
         CustomSymbolSetString(CustomSymbol, SYMBOL_DESCRIPTION, "Pseudo book generator");
         CustomSymbolSetString(CustomSymbol, SYMBOL_FORMULA, "\"" + _Symbol + "\"");
      }
   }
   ...
If the custom symbol already exists, the Expert Advisor can offer the user to delete it and complete the
work there (the user should first close all charts with this symbol).
   else
   {
      if(IDYES == MessageBox(StringFormat("Delete existing custom symbol '%s'?",
         CustomSymbol), "Please, confirm", MB_YESNO))
      {
         PRTF(MarketBookRelease(CustomSymbol));
         PRTF(SymbolSelect(CustomSymbol, false));
         PRTF(CustomRatesDelete(CustomSymbol, 0, LONG_MAX));
         PRTF(CustomTicksDelete(CustomSymbol, 0, LONG_MAX));
         if(!PRTF(CustomSymbolDelete(CustomSymbol)))
         {
            Alert("Can't delete ", CustomSymbol, ", please, check up and delete manually");
         }
         return INIT_PARAMETERS_INCORRECT;
      }
   }
   ...
A special feature of this symbol is setting the SYMBOL_TICKS_BOOKDEPTH property, as well as reading
the contract size SYMBOL_TRADE_CONTRACT_SIZE, which will be required when generating volumes.
   if(SymbolInfoInteger(_Symbol, SYMBOL_TICKS_BOOKDEPTH) != CustomBookDepth
   && SymbolInfoInteger(CustomSymbol, SYMBOL_TICKS_BOOKDEPTH) != CustomBookDepth)
   {
      Print("Adjusting custom market book depth");
      CustomSymbolSetInteger(CustomSymbol, SYMBOL_TICKS_BOOKDEPTH, CustomBookDepth);
   }
   
   depth = (int)PRTF(SymbolInfoInteger(CustomSymbol, SYMBOL_TICKS_BOOKDEPTH));
   contract = PRTF(SymbolInfoDouble(CustomSymbol, SYMBOL_TRADE_CONTRACT_SIZE));
   
   return INIT_SUCCEEDED;
}
The algorithm is launched in the OnTick handler. Here we call the GenerateMarketBook function which
is yet to be written. It will fill the array of structures MqlBookInfo passed by reference, and we'll send it
to a custom symbol using CustomBookAdd.

---

## Page 1669

Part 7. Advanced language tools
1 669
7.2 Custom symbols
void OnTick()
{
   MqlBookInfo book[];
   if(GenerateMarketBook(2000, book))
   {
      ResetLastError();
      if(!CustomBookAdd(CustomSymbol, book))
      {
         Print("Can't add market books, ", E2S(_LastError));
         ExpertRemove();
      }
   }
}
The GenerateMarketBook function analyzes the latest count ticks and, based on them, emulates the
possible state of the order book, guided by the following hypotheses:
• What has been bought is likely to be sold
• What has been sold is likely to be bought
The division of ticks into those that correspond to purchases and sales, in the general case (in the
absence of exchange flags) can be estimated by the movement of the price itself:
• The movement of the Ask price upwards is treated as a purchase
• The movement of the Bid price downwards is treated as a sale
As a result, we get the following algorithm.

---

## Page 1670

Part 7. Advanced language tools
1 670
7.2 Custom symbols
bool GenerateMarketBook(const int count, MqlBookInfo &book[])
{
   MqlTick tick; // order book centre
   if(!SymbolInfoTick(_Symbol, tick)) return false;
   
   double buys[];  // buy volumes by price levels
   double sells[]; // sell volumes by price levels
   
   MqlTick ticks[];
   CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, count); // request tick history
   for(int i = 1; i < ArraySize(ticks); ++i)
   {
      // we believe that ask was pushed up by buys
      int k = (int)MathRound((tick.ask - ticks[i].ask) / _Point);
      if(ticks[i].ask > ticks[i - 1].ask)
      {
         // already bought, probably will take profit by selling
         if(k <= 0)
         {
            Place(sells, -k, contract / sqrt(sqrt(ArraySize(ticks) - i)));
         }
      }
      
      // believe that the bid was pushed down by sells
      k = (int)MathRound((tick.bid - ticks[i].bid) / _Point);
      if(ticks[i].bid < ticks[i - 1].bid)
      {
         // already sold, probably will take profit by buying
         if(k >= 0)
         {
            Place(buys, k, contract / sqrt(sqrt(ArraySize(ticks) - i)));
         }
      }
   }
   ...
The helper function Place fills buys and sells arrays, accumulating volumes in them by price levels. We
will show this below. Indexes in arrays are defined as the distance in points from the current best prices
(Bid or Ask). The size of the volume is inversely proportional to the age of the tick, i.e. ticks that are
more distant in the past have less effect.
After the arrays are filled, an array of structures MqlBookInfo is formed based on them.

---

## Page 1671

Part 7. Advanced language tools
1 671 
7.2 Custom symbols
   for(int i = 0, k = 0; i < ArraySize(sells) && k < depth; ++i) // top half of the order book
   {
      if(sells[i] > 0)
      {
         MqlBookInfo info = {};
         info.type = BOOK_TYPE_SELL;
         info.price = tick.ask + i * _Point;
         info.volume = (long)sells[i];
         info.volume_real = (double)(long)sells[i];
         PUSH(book, info);
         ++k;
      }
   }
   
   for(int i = 0, k = 0; i < ArraySize(buys) && k < depth; ++i) // bottom half of the order book
   {
      if(buys[i] > 0)
      {
         MqlBookInfo info = {};
         info.type = BOOK_TYPE_BUY;
         info.price = tick.bid - i * _Point;
         info.volume = (long)buys[i];
         info.volume_real = (double)(long)buys[i];
         PUSH(book, info);
         ++k;
      }
   }
   
   return ArraySize(book) > 0;
}
The Place function is simple.
void Place(double &array[], const int index, const double value = 1)
{
   const int size = ArraySize(array);
   if(index >= size)
   {
      ArrayResize(array, index + 1);
      for(int i = size; i <= index; ++i)
      {
         array[i] = 0;
      }
   }
   array[index] += value;
}
The following screenshot shows a EURUSD chart with the PseudoMarketBook.mq5 Expert Advisor
running on it, and the resulting version of the order book.

---

## Page 1672

Part 7. Advanced language tools
1 672
7.2 Custom symbols
Synthetic order book of a custom symbol based on EURUSD
7.2.8 Custom symbol trading specifics
The custom symbol is known only to the client terminal and is not available on the trade server.
Therefore, if a custom symbol is built on the basis of some real symbol, then any Expert Advisor placed
on the chart of such a custom symbol should generate trade orders for the original symbol.
As the simplest solution to this problem, you can place an Expert Advisor on the chart of the original
symbol but receive signals (for example, from indicators) from the custom symbol. Another obvious
approach is to replace the names of the symbols when performing trading operations. To test both
approaches, we need a custom symbol and an Expert Advisor.
As an interesting practical example of custom symbols, let's take several different equivolume charts.
An equivolume (equal volume) chart is a chart of bars built on the principle of equality of the volume
contained in them. On a regular chart, each new bar is formed at a specified frequency, coinciding with
the timeframe size. On an equivolume chart, each bar is considered formed when the sum of ticks or
real volumes reaches a preset value. At this moment, the program starts calculating the amount for the
next bar. Of course, in the process of calculating volumes, price movements are controlled, and we get
the usual sets of prices on the chart: Open, High, Low, and Close.
The equal-range bars are built in a similar way: a new bar opens there when the price passes a given
number of points in any direction.
Thus, the EqualVolumeBars.mq5 Expert Advisor will support three modes, i.e., three chart types:
• EqualTickVolumes – equivolume bars by ticks
• EqualRealVolumes – equivolume bars by real volumes (if they are broadcast)
• RangeBars – equal range bars

---

## Page 1673

Part 7. Advanced language tools
1 673
7.2 Custom symbols
They are selected using the input parameter WorkMode.
The bar size and history depth for calculation are specified in the parameters TicksInBar and StartDate.
input int TicksInBar = 1000;
input datetime StartDate = 0;
Depending on the mode, the custom symbol will receive the suffix "_Eqv", "_Qrv" or "_Rng",
respectively, with the addition of the bar size.
Although the horizontal axis on an Equivolume/Equal-Range chart still represents chronology, the
timestamps of each bar are arbitrary and depend on the volatility (number or size of trades) in each
time frame. In this regard, the timeframe of the custom symbol chart should be chosen equal to the
minimum M1 .
The limitation of the platform is that all bars have the same nominal duration, but in the case of our
"artificial" charts, it should be remembered that the real duration of each bar is different and can
significantly exceed 1  minute or, on the contrary, be less. So, with a sufficiently small given volume
for one bar, a situation may arise that new bars are formed much more often than once a minute,
and then the virtual time of the custom symbol bars will run ahead of real time, into the future. To
prevent this from happening, you should increase the volume of the bar (the TicksInBar parameter)
or move old bars to the left.
Initialization and other auxiliary tasks for managing custom symbols (in particular, resetting an existing
history, and opening a chart with a new symbol) are performed in a similar way as in other examples,
and we will omit them. Let's turn to the specifics of an applied nature.
We will read the history of real ticks using built-in functions CopyTicks/CopyTicksRange: the first one is
for swapping the history in batches of 1 0,000 ticks, and the second one is for requesting new ticks
since the previous processing. All this functionality is packaged in the class TicksBuffer (full source code
attached).
class TicksBuffer
{
private:
   MqlTick array[]; // internal array of ticks
   int tick;        // incremental index of the next tick for reading
public:
   bool fill(ulong &cursor, const bool history = false);
   bool read(MqlTick &t);
};
Public method fill is designed to fill the internal array with the next portion of ticks, starting from the
cursor time (in milliseconds). At the same time, the time in cursor on each call moves forward based on
the time of the last tick read into the buffer (note that the parameter is passed by reference).
Parameter history determines whether to use CopyTicks or CopyTicksRange. As a rule, online we will
read one or more new ticks from the OnTick handler.
Method read returns one tick from the internal array and shifts the internal pointer (tick) to the next
tick. If the end of the array is reached while reading, the method will return false, which means it's
time to call the method fill.
Using these methods, the tick history bypass algorithm is implemented as follows (this code is
indirectly called from OnInit via timer).

---

## Page 1674

Part 7. Advanced language tools
1 674
7.2 Custom symbols
   ulong cursor = StartDate * 1000;
   TicksBuffer tb;
    
   while(tb.fill(cursor, true) && !IsStopped())
   {
      MqlTick t;
      while(tb.read(t))
      {
         HandleTick(t, true);
      }
   }
In the HandleTick function, it is required to take into account the properties of tick t in some global
variables that control the number of ticks, the total trading volume (real, if any), as well as the price
movement distance. Depending on the mode of operation, these variables should be analyzed differently
for the condition of the formation of a new bar. So if in the equivolume mode, the number of ticks
exceeded TicksInBar, we should start a new bar by resetting the counter to 1 . In this case, the time of
a new bar is taken as the tick time rounded to the nearest minute.
This group of global variables provides for storing the virtual time of the last ("current") bar on a
custom symbol (now_ time), its OHLC prices, and volumes.
datetime now_time;
double now_close, now_open, now_low, now_high;
long now_volume, now_real;
Variables are constantly updated both during history reading and later when the Expert Advisor starts
processing online ticks in real-time (we will return to this a bit later).
In a somewhat simplified form, the algorithm inside HandleTick looks like this:

---

## Page 1675

Part 7. Advanced language tools
1 675
7.2 Custom symbols
void HandleTick(const MqlTick &t, const bool history = false)
{
   now_volume++;               // count the number of ticks
   now_real += (long)t.volume; // sum up all real volumes
   
   if(!IsNewBar()) // continue the current bar
   {
      if(t.bid < now_low) now_low = t.bid;   // monitor price fluctuations downward
      if(t.bid > now_high) now_high = t.bid; // and upwards
      now_close = t.bid;                     // update the closing price
    
      if(!history)
      {
         // update the current bar if we are not in the history
         WriteToChart(now_time, now_open, now_low, now_high, now_close,
            now_volume - !history, now_real);
      }
   }
   else // new bar
   {
      do
      {
         // save the closed bar with all attributes
         WriteToChart(now_time, now_open, now_low, now_high, now_close,
            WorkMode == EqualTickVolumes ? TicksInBar : now_volume,
            WorkMode == EqualRealVolumes ? TicksInBar : now_real);
   
         // round up the time to the minute for the new bar
         datetime time = t.time / 60 * 60;
   
         // prevent bars with old or same time
         // if gone to the "future", we should just take the next count M1
         if(time <= now_time) time = now_time + 60;
   
         // start a new bar from the current price
         now_time = time;
         now_open = t.bid;
         now_low = t.bid;
         now_high = t.bid;
         now_close = t.bid;
         now_volume = 1;             // first tick in the new bar
         if(WorkMode == EqualRealVolumes) now_real -= TicksInBar;
         now_real += (long)t.volume; // initial real volume in the new bar
   
         // save new bar 0
         WriteToChart(now_time, now_open, now_low, now_high, now_close,
            now_volume - !history, now_real);
      }
      while(IsNewBar() && WorkMode == EqualRealVolumes);
   }
}

---

## Page 1676

Part 7. Advanced language tools
1 676
7.2 Custom symbols
Parameter history determines whether the calculation is based on history or already in real-time (on
incoming online ticks). If based on history, it is enough to form each bar once, while online, the current
bar is updated with each tick. This allows you to speed up the processing of history.
The helper function IsNewBar returns true when the condition for closing the next bar according to the
mode is met.
bool IsNewBar()
{
   if(WorkMode == EqualTickVolumes)
   {
      if(now_volume > TicksInBar) return true;
   }
   else if(WorkMode == EqualRealVolumes)
   {
      if(now_real > TicksInBar) return true;
   }
   else if(WorkMode == RangeBars)
   {
      if((now_high - now_low) / _Point > TicksInBar) return true;
   }
   
   return false;
}
The function WriteToChart creates a bar with the given characteristics by calling CustomRatesUpdate.
void WriteToChart(datetime t, double o, double l, double h, double c, long v, long m = 0)
{
   MqlRates r[1];
   
   r[0].time = t;
   r[0].open = o;
   r[0].low = l;
   r[0].high = h;
   r[0].close = c;
   r[0].tick_volume = v;
   r[0].spread = 0;
   r[0].real_volume = m;
   
   if(CustomRatesUpdate(SymbolName, r) < 1)
   {
      Print("CustomRatesUpdate failed: ", _LastError);
   }
}
The aforementioned loop of reading and processing ticks is performed during the initial access to the
history, after the creation or complete recalculation of an already existing user symbol. When it comes
to new ticks, the OnTick function uses a similar code but without the "historicity" flags.

---

## Page 1677

Part 7. Advanced language tools
1 677
7.2 Custom symbols
void OnTick()
{
   static ulong cursor = 0;
   MqlTick t;
   
   if(cursor == 0)
   {
      if(SymbolInfoTick(_Symbol, t))
      {
         HandleTick(t);
         cursor = t.time_msc + 1;
      }
   }
   else
   {
      TicksBuffer tb;
      while(tb.fill(cursor))
      {
         while(tb.read(t))
         {
            HandleTick(t);
         }
      }
   }
   
   RefreshWindow(now_time);
}
The RefreshWindow function adds a custom symbol tick in the Market Watch.
Please note that tick forwarding increases the tick counter in the bar by 1 , and therefore, when writing
the tick counter to the 0th bar, we previously subtracted one (see the expression now_ volume - !history
when calling WriteToChart).
Tick generation is important because it triggers the OnTick event on custom instrument charts, which
potentially allows Expert Advisors placed on such charts to trade. However, this technology requires
some additional tricks, which we will consider later.
void RefreshWindow(const datetime t)
{
   MqlTick ta[1];
   SymbolInfoTick(_Symbol, ta[0]);
   ta[0].time = t;
   ta[0].time_msc = t * 1000;
   if(CustomTicksAdd(SymbolName, ta) == -1)
   {
      Print("CustomTicksAdd failed:", _LastError, " ", (long) ta[0].time);
      ArrayPrint(ta);
   }
}
We emphasize that the time of the generated custom tick is always set equal to the label of the current
bar since we cannot leave the real tick time: if it has gone ahead by more than 1  minute and we will

---

## Page 1678

Part 7. Advanced language tools
1 678
7.2 Custom symbols
send such a tick to Market Watch, the terminal will create the next bar M1 , which will violate our
"equivolume" structure because our bars are formed not by time, but by volume filling (and we
ourselves control this process).
In theory, we could add one millisecond to each tick, but we have no guarantee that the bar will not
need to store more than 60,000 ticks (for example, if the user orders a chart with a certain price
range that is unpredictable in terms of how many ticks will be required for such movement).
In modes by volume, it is theoretically possible to interpolate the second and millisecond components
of the tick time using linear formulas:
• EqualTickVolumes – (now_volume - 1 ) * 60000 / TicksInBar;
• EqualRealVolumes – (now_real - 1 ) * 60000 / TicksInBar;
However, this is nothing more than a means of identifying ticks, and not an attempt to make the time
of "artificial" ticks closer to the time of real ones. This is not only about the loss of unevenness of the
real flow of ticks, which in itself will already lead to differences in price between the original symbol and
the custom symbol generated on its basis.
The main problem is the need to round off the tick time along the border of the M1  bar and "pack"
them within one minute (see the sidebar about special types of charts). For example, the next tick with
real-time 1 2:37:05'1 23 becomes the 1 001 st tick and should form a new equivolume bar. However, bar
M1  can only be timestamped to the minute, i.e. 1 2:37. As a result, the real price of the instrument at
1 2:37 will not match the price in the tick that provided the Open price for the equivolume bar 1 2:37.
Also, if the next 1 000 ticks stretch over several minutes, we will still be forced to "compress" their
time so as not to reach the 1 2:38 mark.
The problem is of a systemic nature due to time quantization when special charts are emulated by a
standard M1  timeframe chart. This problem cannot be completely solved on such charts. But when
generating custom symbols with ticks in continuous time (for example, with synthetic quotes or based
on streaming data from external services), this problem does not arise.
It is important to note that tick forwarding is done online only in this version of the generator, while
custom ticks are not generated on history! This is done in order to speed up the creation of quotes.
If you need to generate a tick history despite the slower process, the Expert Advisor
EqualVolumeBars.mq5 should be adapted: exclude the WriteToChart function and perform the entire
generation using CustomTicksReplace/CustomTicksAdd. At the same time, it should be remembered
that the original time of ticks should be replaced by another one, within a minute bar, so as not to
disturb the structure of the formed equivolume chart.
Let's see how EqualVolumeBars.mq5 works. Here is the working chart of EURUSD M1 5 with the Expert
Advisor running in it. It has the equivolume chart, in which 1 000 ticks are allotted for each bar.

---

## Page 1679

Part 7. Advanced language tools
1 679
7.2 Custom symbols
Equivolume EURUSD chart with 1000 ticks per bar generated by the EqualVolumeBars Expert Advisor
Note that the tick volumes on all bars are equal, except for the last one, which is still forming (tick
counting continues).
Statistics are displayed in the log.
Creating "EURUSD.c_Eqv1000"
Processing tick history...
End of CopyTicks at 2022.06.15 12:47:51
Bar 0: 2022.06.15 12:40:00 866 0
2119 bars written in 10 sec
Open "EURUSD.c_Eqv1000" chart to view results
Let's check another mode of operation: equal range. Below is a chart where the range of each bar is
250 points.

---

## Page 1680

Part 7. Advanced language tools
1 680
7.2 Custom symbols
EURUSD equal range chart with 250 pips bars generated by EqualVolumeBars
For exchange instruments, the Expert Advisor allows the use of the real volume mode, for example, as
follows:
Ethereum raw and equivolume chart with real volume of 10000 per bar

---

## Page 1681

Part 7. Advanced language tools
1 681 
7.2 Custom symbols
The timeframe of the working symbol when placing the Expert Advisor generator is not important, since
the tick history is always used for calculations.
At the same time, the timeframe of the custom symbol chart must be equal to M1  (the smallest
available in the terminal). Thus, the time of the bars, as a rule, corresponds as closely as possible (as
far as possible) to the moments of their formation. However, during strong movements in the market,
when the number of ticks or the size of volumes forms several bars per minute, the time of the bars will
run ahead of the real one. When the market calms down, the situation with the time marks of the equi-
volume bars will normalize. This does not affect the flow of online prices, so it is probably not
particularly critical, since the whole point of using equal-volume or equal-range bars is to decouple from
absolute time.
Unfortunately, the name of the original symbol and the custom symbol created on its basis cannot be
linked in any way by means of the platform itself. It would be convenient to have a string field
"origin" (source) among the properties of the custom symbol, in which we could write the name of the
real working tool. By default, it would be empty, but if filled in, the platform could replace the symbol in
all trade orders and history requests, and do it automatically and transparently for the user. In theory,
among the properties of user-defined symbols, there is a SYMBOL_BASIS field that is suitable in terms
of its meaning, but since we cannot guarantee that arbitrary generators of user-defined symbols (any
MQL programs) will correctly fill it in or use it exactly for this purpose, we cannot rely on its use.
Since this mechanism is not in the platform, we will need to implement it ourselves. You will have to set
the correspondence between the names of the source and user symbols using parameters.
To solve the problem, we developed the class CustomOrder (see the attached file CustomOrder.mqh). It
contains wrapper methods for all MQL API functions related to sending trading orders and requesting
history, which have a string parameter with the symbol name. In these methods, the custom symbol is
replaced with the current working one or vice versa. Other API functions do not require "hooking".
Below is a snippet.

---

## Page 1682

Part 7. Advanced language tools
1 682
7.2 Custom symbols
class CustomOrder
{
private:
   static string workSymbol;
   
   static void replaceRequest(MqlTradeRequest &request)
   {
      if(request.symbol == _Symbol && workSymbol != NULL)
      {
         request.symbol = workSymbol;
         if(MQLInfoInteger(MQL_TESTER)
            && (request.type == ORDER_TYPE_BUY
            || request.type == ORDER_TYPE_SELL))
         {
            if(TU::Equal(request.price, SymbolInfoDouble(_Symbol, SYMBOL_ASK)))
               request.price = SymbolInfoDouble(workSymbol, SYMBOL_ASK);
            if(TU::Equal(request.price, SymbolInfoDouble(_Symbol, SYMBOL_BID)))
               request.price = SymbolInfoDouble(workSymbol, SYMBOL_BID);
         }
      }
   }
   
public:
   static void setReplacementSymbol(const string replacementSymbol)
   {
      workSymbol = replacementSymbol;
   }
   
   static bool OrderSend(MqlTradeRequest &request, MqlTradeResult &result)
   {
      replaceRequest(request);
      return ::OrderSend(request, result);
   }
   ...
Please note that the main working method replaceRequest replaces not only the symbol but also the
current Ask and Bid prices. This is due to the fact that many custom tools, such as our Equivolume
plot, have a virtual time that is different from the time of the real prototype symbol. Therefore, the
prices of the custom instrument emulated by the tester are out of sync with the corresponding prices
of the real instrument.
This artifact occurs only in the tester. When trading online, the custom symbol chart will be updated
(at prices) synchronously with the real one, although the bar labels will differ (one "artificial" M1  bar
has a real duration of more or less than a minute, and its countdown time is not a multiple of a minute).
Thus, this price conversion is more of a precaution to avoid getting requotes in the tester. However, in
the tester, we usually do not need to do symbol substitution, since the tester can trade with a custom
symbol (unlike the broker's server). Further, just for the sake of interest, we will compare the results of
tests run both with and without character substitution.
To minimize edits to the client source code, global functions and macros of the following form are
provided (for all CustomOrder methods):

---

## Page 1683

Part 7. Advanced language tools
1 683
7.2 Custom symbols
  bool CustomOrderSend(const MqlTradeRequest &request, MqlTradeResult &result)
  {
    return CustomOrder::OrderSend((MqlTradeRequest)request, result);
  }
  
  #define OrderSend CustomOrderSend
They allow the automatic redirection of all standard API function calls to the CustomOrder class
methods. To do this, simply include CustomOrder.mqh into the Expert Advisor and set the working
symbol, for example, in the WorkSymbol parameter:
  #include <CustomOrder.mqh>
  #include <Expert/Expert.mqh>
  ...
  input string WorkSymbol = "";
  
  int OnInit()
  {
    if(WorkSymbol != "")
    {
      CustomOrder::setReplacementSymbol(WorkSymbol);
      
      // initiate the opening of the chart tab of the working symbol (in the visual mode of the tester)
      MqlRates rates[1];
      CopyRates(WorkSymbol, PERIOD_CURRENT, 0, 1, rates);
    }
    ...
  }
It is important that the directive #include<CustomOrder.mqh> was the very first, before the others.
Thus, it affects all source codes, including the standard libraries from the MetaTrader 5 distribution. If
no substitution symbol is specified, the connected CustomOrder.mqh has no effect on the Expert
Advisor and "transparently" transfers control to the standard API functions.
Now we have everything ready to test the idea of trading on a custom symbol, including the custom
symbol itself.
Applying the technique shown above we modify the already familiar Expert Advisor BandOsMaPro,
renaming it to BandOsMaCustom.mq5. Let's test it on the EURUSD equivolume chart with a bar size of
1 000 ticks obtained using EqualVolumeBars.mq5.
Optimization or testing mode is set to OHLC M1  prices (more accurate methods do not make sense
because we did not generate ticks and also because this version trades at the prices of formed bars).
The date range is the entire 2021  and the first half of 2022. The file with the settings
BandOsMACustom.set is attached.
In the tester settings, you should not forget to select the custom symbol EURUSD_Eqv1 000 and the
M1  timeframe, since it is on it that equi-volume bars are emulated.
When the WorkSymbol parameter is empty, the Expert Advisor trades a custom symbol. Here are the
results:

---

## Page 1684

Part 7. Advanced language tools
1 684
7.2 Custom symbols
Tester's report when trading on the EURUSD_Eqv1000 equivolume chart
If the WorkSymbol parameter equals EURUSD, the Expert Advisor trades the EURUSD pair, despite the
fact that it works on the EURUSD_Eqv1 000 chart. The results differ but not much.
Tester's report when trading EURUSD from the EURUSD_Eqv1000 equivolume chart

---

## Page 1685

Part 7. Advanced language tools
1 685
7.2 Custom symbols
However, as it was already mentioned at the beginning of the section, there is an easier way for Expert
Advisors which trade on indicator signals to support custom symbols. To do this, it is enough to create
indicators on a custom symbol and place the Expert Advisor on the chart of a working symbol.
We can easily implement this option. Let's call it BandOsMACustomSignal.mq5.
The header file CustomOrder.mqh is no longer needed. Instead of the WorkSymbol input parameter, we
add two new ones:
input string SignalSymbol = "";
input ENUM_TIMEFRAMES SignalTimeframe = PERIOD_M1;
They should be passed to the constructor of the BandOsMaSignal class which manages the indicators.
Previously, _ Symbol and _ Period were used everywhere.

---

## Page 1686

Part 7. Advanced language tools
1 686
7.2 Custom symbols
interface TradingSignal
{
   virtual int signal(void);
   virtual string symbol();
   virtual ENUM_TIMEFRAMES timeframe();
};
   
class BandOsMaSignal: public TradingSignal
{
   int hOsMA, hBands, hMA;
   int direction;
   const string _symbol;
   const ENUM_TIMEFRAMES _timeframe;
public:
   BandOsMaSignal(const string s, const ENUM_TIMEFRAMES tf,
      const int fast, const int slow, const int signal, const ENUM_APPLIED_PRICE price,
      const int bands, const int shift, const double deviation,
      const int period, const int x, ENUM_MA_METHOD method): _symbol(s), _timeframe(tf)
   {
      hOsMA = iOsMA(s, tf, fast, slow, signal, price);
      hBands = iBands(s, tf, bands, shift, deviation, hOsMA);
      hMA = iMA(s, tf, period, x, method, hOsMA);
      direction = 0;
   }
   ...
   virtual string symbol() override
   {
      return _symbol;
   }
   
   virtual ENUM_TIMEFRAMES timeframe() override
   {
      return _timeframe;
   }
}
Since the symbol and timeframe for signals can now differ from the symbol and period of the chart, we
have expanded the TradingSignal interface by adding read methods. The actual values are passed to
the constructor in OnInit.

---

## Page 1687

Part 7. Advanced language tools
1 687
7.2 Custom symbols
int OnInit()
{
   ...
   strategy = new SimpleStrategy(
      new BandOsMaSignal(SignalSymbol != "" ? SignalSymbol : _Symbol,
         SignalSymbol != "" ? SignalTimeframe : _Period,
         p.fast, p.slow, SignalOsMA, PriceOsMA,
         BandsMA, BandsShift, BandsDeviation,
         PeriodMA, ShiftMA, MethodMA),
         Magic, StopLoss, Lots);
   return INIT_SUCCEEDED;
}
In the SimpleStrategy class, the trade method now checks for the occurrence of a new bar not
according to the current chart, but according to the properties of the signal.
   virtual bool trade() override
   {
      // looking for a signal once at the opening of the bar of the desired symbol and timeframe
      if(lastBar == iTime(command[].symbol(), command[].timeframe(), 0)) return false;
      
      int s = command[].signal(); // get signal
      ...
   }
For a comparative experiment with the same settings, the Expert Advisor BandOsMACustomSignal.mq5
should be launched on EURUSD (you can use M1  or another timeframe), and EURUSD_Eqv1 000 should
be specified in the SignalSymbol parameter. SignalTimeframe should be left equal to PERIOD_M1  by
default. As a result, we will get a similar report.


---

## Page 1688

Part 7. Advanced language tools
1 688
7.2 Custom symbols
Tester's report when trading on the EURUSD chart based on signals from the EURUSD_Eqv1000 equivolume symbol
The number of bars and ticks is different here because EURUSD was chosen as the tested instrument
and not the custom EURUSD_Eqv1 000.
All three test results are slightly different. This is due to the "packing" of quotes into minute bars and a
slight desynchronization of the price movements of the original and custom instruments. Which of the
results is more accurate? This, most likely, depends on the specific trading system and the features of
its implementation. In the case of our Expert Advisor BandOsMa with control over bar opening, the
version with direct trading on EURUSD_Eqv1 000 should have the most realistic results. In theory, the
rule of thumb stating that of several alternative checks, the most reliable is the least profitable, is
almost always satisfied.
So, we have analyzed a couple of techniques for adapting Expert Advisors for trading on custom
symbols that have a prototype among the broker's working symbols. However, this situation is not
mandatory. In many cases, custom symbols are generated based on data from external systems such
as crypto exchanges. Trading on them must be done using their public API with MQL5 network
functions.
Emulating special types of charts with custom symbols
Many traders use special types of charts, in which continuous real-time is excluded from
consideration. This includes not only equivolume and equal range bars, but also Renko, Point-And-
Figure (PAF), Kagi, and others. Custom symbols allow these kinds of charts to be emulated in
MetaTrader 5 using M1  timeframe charts but should be treated with caution when it comes to
testing trading systems rather than technical analysis. 
For special types of charts, the actual bar opening time (accurate to milliseconds) almost always
does not coincide exactly with the minute with which the M1  bar will be marked. Thus, the opening
price of a custom bar differs from the opening price of the M1  bar of a standard symbol. 
Moreover, other OHLC prices will also differ because the real duration of the formation of the M1 
bar on a special chart is not equal to one minute. For example, 1 000 ticks for an equivolume chart
can accumulate for longer than 5 minutes.
The closing price of a custom bar also does not correspond to the real closing time because a
custom bar is, technically, an M1  bar, i.e. it has a nominal duration of 1  minute. 
Special care should be taken when working with such types of charts as the classic Renko or PAF.
The fact is that their reversal bars have an opening price with a gap from the closing of the previous
bar. Thus, the opening price becomes a predictor of future price movement. 
The analysis of such charts is supposed to be carried out according to the formed bars, that is,
their characteristic price is the closing price, however, when working by bar, the tester provides
only the opening price for the current (last) bar (there is no mode by closing prices). Even if we
take indicator signals from closed bars (usually from the 1 st one), deals are made at the current
price of the 0th bar anyway. And even if we turn to tick modes, the tester always generates ticks
according to the usual rules, guided by reference points based on the configuration of each bar. The
tester does not take into account the structure and behavior of special charts, which we are trying
to visually emulate with M1  bars. 
Trading in the tester using such symbols in any mode (by opening prices, M1  OHLC, or by ticks)
affects the accuracy of the results: they are too optimistic and can serve as a source of too high

---

## Page 1689

Part 7. Advanced language tools
1 689
7.2 Custom symbols
expectations. In this regard, it is essential to check the trading system not on a separate Renko or
PAF chart, but in conjunction with the execution of orders on a real symbol. 
Custom symbols can also be used for second timeframes or tick charts. In this case, virtual time is
also generated for bars and ticks, decoupled from real-time. Therefore, such charts are well suited
for operational analysis but require additional attention when developing and testing trading
strategies, especially multi-symbol ones. 
An alternative for any custom symbols is the independent calculation of arrays of bars and ticks
inside an Expert Advisor or indicator. However, debugging and visualizing such structures requires
additional effort.
7.3 Economic calendar
When developing trading strategies, it is desirable to take into account the fundamental factors that
affect the market. MetaTrader 5 has a built-in economic calendar, which is available in the program
interface as a separate tab in the toolbar, as well as labels, optionally displayed directly on the chart.
The calendar can be enabled by a separate flag on the Community tab in the terminal settings dialog
(login to the community is not necessary).
Since MetaTrader 5 supports algorithmic trading, economic calendar events can also be accessed
programmatically from the MQL5 API. In this chapter, we will introduce the functions and data
structures that enable reading, filtering, and monitoring changes in economic events.
The economic calendar contains a description, release schedule, and historical values of
macroeconomic indicators for many countries. For each event, the exact time of the planned release,
the degree of importance, the impact on specific currencies, forecast values, and other attributes are
known. Actual values of macroeconomic indicators arrive at MetaTrader 5 immediately at the time of
publication.
The availability of the calendar allows you to automatically analyze incoming events and react to them
in Expert Advisors in a variety of ways, for example, trading as part of a breakout strategy or volatility
fluctuations within the corridor. On the other hand, knowing the upcoming fluctuations in the market
allows you to find quiet hours in the schedule and temporarily turn off those robots for which strong
price movements are dangerous due to possible losses.
Values of datetime type used by all functions and structures that work with the economic calendar are
equal to the trade server time (TimeTradeServer) including its time zone and DST (Daylight Saving
Time) settings. In other words, for correct testing of news-trading Expert Advisors, their developer
must independently change the times of historical news in those periods (about half a year within each
year) when the DST mode differs from the current one.
Calendar functions cannot be used in the tester: when trying to call any of them, we get the
FUNCTION_NOT_ALLOWED (401 4) error. In this regard, testing calendar-based strategies involves first
saving calendar entries in external storages (for example, in files) when running the MQL program on
the online chart, and then loading and reading them from the MQL program running in the tester.
7.3.1  Basic concepts of the calendar
When working with the calendar, we will operate with several concepts, for the formal description of
which MQL5 defines special types of structures.

---

## Page 1690

Part 7. Advanced language tools
1 690
7.3 Economic calendar
First of all, the events are related to specific countries, and each country is described using the
MqlCalendarCountry structure.
struct MqlCalendarCountry
{ 
   ulong  id;              //country identifier according to ISO 3166-1 
   string name;            // text name of the country (in the current terminal encoding) 
   string code;            // two-letter country designation according to ISO 3166-1 alpha-2 
   string currency;        // international country currency code 
   string currency_symbol; // symbol/sign of the country's currency 
   string url_name;        // country name used in the URL on the mql5.com website 
};
How to get a list of countries available in the calendar and their attributes as an array of
MqlCalendarCountry structures, we will find out in the next section.
For now, we just pay attention to the id field. It is important because it is the key to determining
whether calendar events belong to a particular country. In each country (or a registered association of
countries, such as the European Union) there is a specific, internationally known list of types of
economic indicators and informational events that affect the market and are therefore included in the
calendar.
Each event type is defined by the MqlCalendarEvent structure, in which the field country_ id uniquely
links the event to the country. We will consider the types of enumerations used below.
struct MqlCalendarEvent
{ 
   ulong                          id;         // event ID 
   ENUM_CALENDAR_EVENT_TYPE       type;       // event type 
   ENUM_CALENDAR_EVENT_SECTOR     sector;     // sector to which the event belongs 
   ENUM_CALENDAR_EVENT_FREQUENCY  frequency;  // frequency (periodicity) of the event 
   ENUM_CALENDAR_EVENT_TIMEMODE   time_mode;  // event time mode 
   ulong                          country_id; // country identifier 
   ENUM_CALENDAR_EVENT_UNIT       unit;       // indicator unit 
   ENUM_CALENDAR_EVENT_IMPORTANCE importance; // importance of the event 
   ENUM_CALENDAR_EVENT_MULTIPLIER multiplier; // indicator multiplier 
   uint                           digits;     // number of decimal places
   string                         source_url; // URL of the event publication source 
   string                         event_code; // event code
   string                         name;       // text name of the event in the terminal language 
};
It is important to understand that the MqlCalendarEvent structure describes exactly the type of event
(for example, the publication of the Consumer Price Index, CPI) but not a specific event that may occur
once a quarter, once a month, or according to another schedule. It contains the general
characteristics of the event, including importance, frequency, relation to the sector of the economy,
units of measurement, name, and source of information. As for the actual and forecast indicators,
these will be provided in the calendar entries for each specific event of this type: these entries are
stored as MqlCalendarValue structures, which will be discussed later. Functions for querying the
supported types of events will be introduced in later sections.
The event type in the type field is specified as one of the ENUM_CALENDAR_EVENT_TYPE enumeration
values.

---

## Page 1691

Part 7. Advanced language tools
1 691 
7.3 Economic calendar
Identifier
Description
CALENDAR_TYPE_EVENT
Event (meeting, speech, etc.)
CALENDAR_TYPE_INDICATOR
Economic indicator
CALENDAR_TYPE_HOLIDAY
Holiday (weekend)
The sector of the economy to which the event belongs is selected from the
ENUM_CALENDAR_EVENT_SECTOR enumeration.
Identifier
Description
CALENDAR_SECTOR_NONE
Sector is not set
CALENDAR_SECTOR_MARKET
Market, exchange
CALENDAR_SECTOR_GDP
Gross Domestic Product (GDP)
CALENDAR_SECTOR_JOBS
Labor market
CALENDAR_SECTOR_PRICES
Prices
CALENDAR_SECTOR_MONEY
Money
CALENDAR_SECTOR_TRADE
Trade
CALENDAR_SECTOR_GOVERNMENT
Government
CALENDAR_SECTOR_BUSINESS
Business
CALENDAR_SECTOR_CONSUMER
Consumption
CALENDAR_SECTOR_HOUSING
Housing
CALENDAR_SECTOR_TAXES
Taxes
CALENDAR_SECTOR_HOLIDAYS
Holidays
The frequency of the event is indicated in the frequency field using the
ENUM_CALENDAR_EVENT_FREQUENCY enumeration.
Identifier
Description
CALENDAR_FREQUENCY_NONE
Publication frequency is not set
CALENDAR_FREQUENCY_WEEK
Weekly
CALENDAR_FREQUENCY_MONTH
Monthly
CALENDAR_FREQUENCY_QUARTER
Quarterly
CALENDAR_FREQUENCY_YEAR
Yearly
CALENDAR_FREQUENCY_DAY
Daily

---

## Page 1692

Part 7. Advanced language tools
1 692
7.3 Economic calendar
Event duration (time_ mode) can be described by one of the elements of the
ENUM_CALENDAR_EVENT_TIMEMODE enumeration.
Identifier
Description
CALENDAR_TIMEMODE_DATETIME
The exact time of the event is known
CALENDAR_TIMEMODE_DATE
The event takes all day
CALENDAR_TIMEMODE_NOTIME
Time is not published
CALENDAR_TIMEMODE_TENTATIVE
Only the day is known in advance, but not the exact
time of the event (the time is specified after the
fact)
The importance of the event is specified in the importance field using the
ENUM_CALENDAR_EVENT_IMPORTANCE enumeration.
Identifier
Description
CALENDAR_IMPORTANCE_NONE
Not set
CALENDAR_IMPORTANCE_LOW
Low
CALENDAR_IMPORTANCE_MODERATE
Moderate
CALENDAR_IMPORTANCE_HIGH
High
The units of measurement in which event values are given are defined in the unit field as a member of
the ENUM_CALENDAR_EVENT_UNIT enumeration.
Identifier
Description
CALENDAR_UNIT_NONE
Unit is not set
CALENDAR_UNIT_PERCENT
Interest (%)
CALENDAR_UNIT_CURRENCY
National currency
CALENDAR_UNIT_HOUR
Number of hours
CALENDAR_UNIT_JOB
Number of workplaces
CALENDAR_UNIT_RIG
Drilling rigs
CALENDAR_UNIT_USD
U.S. dollars
CALENDAR_UNIT_PEOPLE
Number of people
CALENDAR_UNIT_MORTGAGE
Number of mortgage loans
CALENDAR_UNIT_VOTE
Number of votes
CALENDAR_UNIT_BARREL
Amount in barrels

---

## Page 1693

Part 7. Advanced language tools
1 693
7.3 Economic calendar
Identifier
Description
CALENDAR_UNIT_CUBICFEET
Volume in cubic feet
CALENDAR_UNIT_POSITION
Net volume of speculative positions in contracts
CALENDAR_UNIT_BUILDING
Number of buildings
In some cases, the values of an economic indicator require a multiplier according to one of the
elements of the ENUM_CALENDAR_EVENT_MULTIPLIER enumeration.
Identifier
Description
CALENDAR_MULTIPLIER_NONE
Multiplier is not set
CALENDAR_MULTIPLIER_THOUSANDS
Thousands
CALENDAR_MULTIPLIER_MILLIONS
Millions
CALENDAR_MULTIPLIER_BILLIONS
Billions
CALENDAR_MULTIPLIER_TRILLIONS
Trillions
So, we have considered all the special data types used to describe the types of events in the
MqlCalendarEvent structure.
A separate calendar entry is formed as a MqlCalendarValue structure. Its detailed description is given
below, but for now, it is important to pay attention to the following nuance. MqlCalendarValue has the
event_ id field which points to the identifier of the event type, i.e., contains one of the existing id in
MqlCalendarEvent structures.
As we saw above, the MqlCalendarEvent structure in turn is related to MqlCalendarCountry via the
country_ id field. Thus, having once entered information about a specific country or type of event into
the calendar database, it is possible to register an arbitrary number of similar events for them. Of
course, the information provider is responsible for filling the database, not the developers.
Let's summarize the subtotal: the system stores three internal tables separately:
·The MqlCalendarCountry structure table to describe countries
·The MqlCalendarEvent structure table with descriptions of types of events
·The MqlCalendarValue structure table with indicators of specific events of various types
By referencing event type identifiers, duplication of information is eliminated from records of specific
events. For example, monthly publications of CPI values only refer to the same MqlCalendarEvent
structure with the general characteristics of this event type. If it were not for the different tables, it
would be necessary to repeat the same properties in each CPI calendar entry. This approach to
establishing relationships between tables with data using identifier fields is called relational, and we will
return to it in the chapter on SQLite. All this is illustrated in the following diagram.

---

## Page 1694

Part 7. Advanced language tools
1 694
7.3 Economic calendar
Diagram of links between structures by fields with identifiers
All tables are stored in the internal calendar database, which is constantly kept up to date while the
terminal is connected to the server.
Calendar entries (specific events) are MqlCalendarValue structures. They are also identified by their
own unique number in the id field (each of the three tables has its own id field).

---

## Page 1695

Part 7. Advanced language tools
1 695
7.3 Economic calendar
struct MqlCalendarValue 
{ 
   ulong      id;                 // entry ID 
   ulong      event_id;           // event type ID 
   datetime   time;               // time and date of the event 
   datetime   period;             // reporting period of the event 
   int        revision;           // revision of the published indicator in relation to the reporting period 
   long       actual_value;       // actual value in ppm or LONG_MIN 
   long       prev_value;         // previous value in ppm or LONG_MIN 
   long       revised_prev_value; // revised previous value in ppm or LONG_MIN 
   long       forecast_value;     // forecast value in ppm or LONG_MIN 
   ENUM_CALENDAR_EVENT_IMPACT impact_type;  // potential impact on the exchange rate
    
 // functions for checking values
   bool HasActualValue(void) const;     // true if the actual_value field is filled 
   bool HasPreviousValue(void) const;   // true if the prev_value field is filled 
   bool HasRevisedValue(void) const;    // true if the revised_prev_value field is filled 
   bool HasForecastValue(void) const;   // true if the forecast_value field is filled
    
   // functions for getting values 
   double GetActualValue(void) const;   // actual_value or nan if value is not set 
   double GetPreviousValue(void) const; // prev_value or nan if value is not set 
   double GetRevisedValue(void) const;  // revised_prev_value or nan if value is not set 
   double GetForecastValue(void) const; // forecast_value or nan if value is not set 
};
For each event, in addition to the time of its publication (time), the following four values are also
stored:
·Actual value (actual_ value), which becomes known immediately after the publication of the news
·Previous value (prev_ value), which became known in the last release of the same news
·Revised value of the previous indicator, revised_ prev_ value (if it has been modified since the last
publication)
·Forecast value (forecast_ value)
Obviously, not all the fields must be necessarily filled. So, the current value is absent (not yet known)
for future events, and the revision of past values also does not always occur. In addition, all four fields
make sense only for quantitative indicators, while the calendar also reflects regulators' speeches,
meetings and holidays.
An empty field (no value) is indicated by the constant LONG_MIN (-9223372036854775808). If the
value in the field is specified (not equal to LONG_MIN), then it corresponds to the real value of the
indicator increased by a million times, that is, to obtain the indicator in the usual (real) form, it is
necessary to divide the field value by 1 ,000,000.
For the convenience of the programmer, the structure defines 4 Has methods for checking the field is
filled, as well as 4 Get methods that return the value of the corresponding field already converted to a
real number, and in the case when it is not filled, the method will return NaN (Not A Number).
Sometimes, in order to obtain absolute values (if they are required for the algorithm), it is important to
additionally analyze the multiplier property in the MqlCalendarEvent structure since some values are
specified in multiple units according to the ENUM_CALENDAR_EVENT_MULTIPLIER enumeration.

---

## Page 1696

Part 7. Advanced language tools
1 696
7.3 Economic calendar
Besides, MqlCalendarEvent has the digits field, which specifies the number of significant digits in the
received values for subsequent correct formatting (for example, in a call to NormalizeDouble).
The reporting period (for which the published indicator is calculated) is set in the period field as its first
day. For example, if the indicator is calculated monthly, then the date '2022.05.01  00:00:00' means
the month of May. The duration of the period (for example, month, quarter, year) is defined in the
frequency field of the related structure MqlCalendarEvent: the type of this field is the special
ENUM_CALENDAR_EVENT_FREQUENCY enumeration described above, along with other enumerations.
Of particular interest is the impact_ type field, in which, after the release of the news, the direction of
influence of the corresponding currency on the exchange rate is automatically set by comparing the
current and forecast values. This influence can be positive (the currency is expected to appreciate) or
negative (the currency is expected to depreciate). For example, a larger drop in sales than expected
would be labeled as having a negative impact, and a larger drop in unemployment as positive. But this
characteristic is interpreted unambiguously not for all events (some economic indicators are considered
contradictory), and besides, one should pay attention to the relative numbers of changes.
The potential impact of an event on the national currency rate is indicated using the
ENUM_CALENDAR_EVENT_IMPACT enumeration.
Identifier
Description
CALENDAR_IMPACT_NA
Influence is not stated
CALENDAR_IMPACT_POSITIVE
Positive influence
CALENDAR_IMPACT_NEGATIVE
Negative influence
Another important concept of the calendar is the fact of its change. Unfortunately, there is no special
structure for change. The only property a change has is its unique ID, which is an integer assigned by
the system each time the internal calendar base is changed.
As you know, the calendar is constantly modified by information providers: new upcoming events are
added to it, and already published indicators and forecasts are corrected. Therefore, it is very
important to keep track of any edits, the occurrence of which makes it possible to detect periodically
increasing change numbers.
The edit time with a specific identifier and its essence are not available in MQL5. If necessary, MQL
programs should implement periodic calendar state queries and record analysis themselves.
A set of MQL5 functions allows getting information about countries, types of events and specific
calendar entries, as well as their changes. We will consider this in the following sections.
Attention! When accessing the calendar for the first time (if the Calendar tab in the terminal toolbar
has not been opened before), it may take several seconds to synchronize the internal calendar
database with the server.
7.3.2 Getting the list and descriptions of available countries
You can get a complete list of countries for which events are broadcast on the calendar using the
CalendarCountries function.

---

## Page 1697

Part 7. Advanced language tools
1 697
7.3 Economic calendar
int CalendarCountries(MqlCalendarCountry &countries[])
The function fills the countries array passed by reference with MqlCalendarCountry structures. The
array can be dynamic or fixed, of sufficient size.
On success, the function returns the number of country descriptions received from the server or 0 on
error. Among the possible error codes in _ LastError we may find, in particular, 5401 
(ERR_CALENDAR_TIMEOUT, request time limit exceeded) or 5400 (ERR_CALENDAR_MORE_DATA, if
the size of the fixed array is insufficient to obtain descriptions of all countries). In the latter case, the
system will copy only what fits.
Let's write a simple script CalendarCountries.mq5, which gets the full list of countries and logs it out.
void OnStart()
{
   MqlCalendarCountry countries[];
   PRTF(CalendarCountries(countries));
   ArrayPrint(countries);
}
Here is an example result.
CalendarCountries(countries)=23 / ok
     [id]           [name] [code] [currency] [currency_symbol]       [url_name] [reserved]
[ 0]  554 "New Zealand"    "NZ"   "NZD"      "$"               "new-zealand"           ...
[ 1]  999 "European Union" "EU"   "EUR"      "€"               "european-union"        ...
[ 2]  392 "Japan"          "JP"   "JPY"      "¥"               "japan"                 ...
[ 3]  124 "Canada"         "CA"   "CAD"      "$"               "canada"                ...
[ 4]   36 "Australia"      "AU"   "AUD"      "$"               "australia"             ...
[ 5]  156 "China"          "CN"   "CNY"      "¥"               "china"                 ...
[ 6]  380 "Italy"          "IT"   "EUR"      "€"               "italy"                 ...
[ 7]  702 "Singapore"      "SG"   "SGD"      "R$"              "singapore"             ...
[ 8]  276 "Germany"        "DE"   "EUR"      "€"               "germany"               ...
[ 9]  250 "France"         "FR"   "EUR"      "€"               "france"                ...
[10]   76 "Brazil"         "BR"   "BRL"      "R$"              "brazil"                ...
[11]  484 "Mexico"         "MX"   "MXN"      "Mex$"            "mexico"                ...
[12]  710 "South Africa"   "ZA"   "ZAR"      "R"               "south-africa"          ...
[13]  344 "Hong Kong"      "HK"   "HKD"      "HK$"             "hong-kong"             ...
[14]  356 "India"          "IN"   "INR"      "₹"               "india"                 ...
[15]  578 "Norway"         "NO"   "NOK"      "Kr"              "norway"                ...
[16]    0 "Worldwide"      "WW"   "ALL"      ""                "worldwide"             ...
[17]  840 "United States"  "US"   "USD"      "$"               "united-states"         ...
[18]  826 "United Kingdom" "GB"   "GBP"      "£"               "united-kingdom"        ...
[19]  756 "Switzerland"    "CH"   "CHF"      "₣"               "switzerland"           ...
[20]  410 "South Korea"    "KR"   "KRW"      "₩"               "south-korea"           ...
[21]  724 "Spain"          "ES"   "EUR"      "€"               "spain"                 ...
[22]  752 "Sweden"         "SE"   "SEK"      "Kr"              "sweden"                ...
It is important to note that the identifier 0 (code "WW" and pseudo-currency "ALL") corresponds to
global events (concerning many countries, for example, the G7, G20 meetings), and the currency
"EUR" is associated with several EU countries available in the calendar (as you can see, not the entire
Eurozone is presented). Also, the European Union itself has a generic identifier 999.

---

## Page 1698

Part 7. Advanced language tools
1 698
7.3 Economic calendar
If you are interested in a particular country, you can check its availability by a numerical code
according to the ISO 31 66-1  standard. In particular, in the log above, these codes are displayed in the
first column (field id).
To get a description of one country by its ID specified in the id parameter, you can use the
CalendarCountryById function.
bool CalendarCountryById(const long id, MqlCalendarCountry &country)
If successful, the function will return true and fill in the fields of the country structure.
If the country is not found, we get false, and in _ LastError we will get an error code 5402
(ERR_CALENDAR_NO_DATA).
For an example of using this function, see Getting event records by country or currency.
7.3.3 Querying event types by country and currency
The calendar of economic events and holidays has its own specifics in each country. An MQL program
can query the types of events within a particular country, as well as the types of events associated with
a particular currency. The latter is relevant in cases where several countries use the same currency,
as, for example, most members of the European Union.
int CalendarEventByCountry(const string country, MqlCalendarEvent &events[])
The CalendarEventByCountry function fills an array of MqlCalendarEvent structures passed by reference
with descriptions of all types of events available in the calendar for the country specified by the two-
letter country code (according to the ISO 31 66-1  alpha-2 standard). We saw examples of such codes
in the previous section, in the log: EU for the European Union, US for the USA, DE for Germany, CN for
China, and so on.
The receiving array can be dynamic or fixed of sufficient size.
The function returns the number of received descriptions and 0 in case of an error. In particular, if the
fixed array is not able to contain all events, the function will fill it with the fit part of the available data
and set the code _ LastError, equal to CALENDAR_MORE_DATA (5400). Memory allocation errors
(4004, ERR_NOT_ENOUGH_MEMORY) or calendar request timeout from the server (5401 ,
ERR_CALENDAR_TIMEOUT) are also possible.
If the country with the given code does not exist, an INTERNAL_ERROR (4001 ) will occur.
By specifying NULL or an empty string "" instead of country, you can get a complete list of events for
all countries.
Let's test the performance of the function using the simple script CalendarEventKindsByCountry.mq5. It
has a single input parameter which is the code of the country we are interested in.
input string CountryCode = "HK";
Next, a request for event types is made by calling CalendarEventByCountry, and if successful, the
resulting arrays are logged.

---

## Page 1699

Part 7. Advanced language tools
1 699
7.3 Economic calendar
void OnStart()
{
   MqlCalendarEvent events[];
   if(PRTF(CalendarEventByCountry(CountryCode, events)))
   {
      Print("Event kinds for country: ", CountryCode);
      ArrayPrint(events);
   }
}
Here is an example of the result (due to the fact that the lines are long, they are artificially divided into
2 blocks for publication in the book: the first block contains the numeric fields of the structures
MqlCalendarEvent, and the second block contains string fields).
CalendarEventByCountry(CountryCode,events)=26 / ok
Event kinds for country: HK
          [id] [type] [sector] [frequency] [time_mode] [country_id] [unit] [importance] [multiplier] [digits] »
[ 0] 344010001      1        5           2           0          344      6            1            3        1 »
[ 1] 344010002      1        5           2           0          344      1            1            0        1 »
[ 2] 344020001      1        4           2           0          344      1            1            0        1 »
[ 3] 344020002      1        2           3           0          344      1            3            0        1 »
[ 4] 344020003      1        2           3           0          344      1            2            0        1 »
[ 5] 344020004      1        6           2           0          344      1            1            0        1 »
[ 6] 344020005      1        6           2           0          344      1            1            0        1 »
[ 7] 344020006      1        6           2           0          344      2            2            3        3 »
[ 8] 344020007      1        9           2           0          344      1            1            0        1 »
[ 9] 344020008      1        3           2           0          344      1            2            0        1 »
[10] 344030001      2       12           0           1          344      0            0            0        0 »
[11] 344030002      2       12           0           1          344      0            0            0        0 »
[12] 344030003      2       12           0           1          344      0            0            0        0 »
[13] 344030004      2       12           0           1          344      0            0            0        0 »
[14] 344030005      2       12           0           1          344      0            0            0        0 »
[15] 344030006      2       12           0           1          344      0            0            0        0 »
[16] 344030007      2       12           0           1          344      0            0            0        0 »
[17] 344030008      2       12           0           1          344      0            0            0        0 »
[18] 344030009      2       12           0           1          344      0            0            0        0 »
[19] 344030010      2       12           0           1          344      0            0            0        0 »
[20] 344030011      2       12           0           1          344      0            0            0        0 »
[21] 344030012      2       12           0           1          344      0            0            0        0 »
[22] 344030013      2       12           0           1          344      0            0            0        0 »
[23] 344030014      2       12           0           1          344      0            0            0        0 »
[24] 344030015      2       12           0           1          344      0            0            0        0 »
[25] 344500001      1        8           2           0          344      0            1            0        1 »
Continuation of the log (right fragment).

---

## Page 1700

Part 7. Advanced language tools
1 700
7.3 Economic calendar
    »                      [source_url]                        [event_code]                                  [name]
[ 0]» "https://www.hkma.gov.hk/eng/"    "foreign-exchange-reserves"         "Foreign Exchange Reserves"            
[ 1]» "https://www.hkma.gov.hk/eng/"    "hkma-m3-money-supply-yy"           "HKMA M3 Money Supply y/y"             
[ 2]» "https://www.censtatd.gov.hk/en/" "cpi-yy"                            "CPI y/y"                              
[ 3]» "https://www.censtatd.gov.hk/en/" "gdp-qq"                            "GDP q/q"                              
[ 4]» "https://www.censtatd.gov.hk/en/" "gdp-yy"                            "GDP y/y"                              
[ 5]» "https://www.censtatd.gov.hk/en/" "exports-mm"                        "Exports y/y"                          
[ 6]» "https://www.censtatd.gov.hk/en/" "imports-mm"                        "Imports y/y"                          
[ 7]» "https://www.censtatd.gov.hk/en/" "trade-balance"                     "Trade Balance"                        
[ 8]» "https://www.censtatd.gov.hk/en/" "retail-sales-yy"                   "Retail Sales y/y"                     
[ 9]» "https://www.censtatd.gov.hk/en/" "unemployment-rate-3-months"        "Unemployment Rate 3-Months"           
[10]» "https://publicholidays.hk/"      "new-years-day"                     "New Year's Day"                       
[11]» "https://publicholidays.hk/"      "lunar-new-year"                    "Lunar New Year"                       
[12]» "https://publicholidays.hk/"      "ching-ming-festival"               "Ching Ming Festival"                  
[13]» "https://publicholidays.hk/"      "good-friday"                       "Good Friday"                          
[14]» "https://publicholidays.hk/"      "easter-monday"                     "Easter Monday"                        
[15]» "https://publicholidays.hk/"      "birthday-of-buddha"                "The Birthday of the Buddha"           
[16]» "https://publicholidays.hk/"      "labor-day"                         "Labor Day"                            
[17]» "https://publicholidays.hk/"      "tuen-ng-festival"                  "Tuen Ng Festival"                     
[18]» "https://publicholidays.hk/"      "hksar-establishment-day"           "HKSAR Establishment Day"              
[19]» "https://publicholidays.hk/"      "day-following-mid-autumn-festival" "The Day Following Mid-Autumn Festival"
[20]» "https://publicholidays.hk/"      "national-day"                      "National Day"                         
[21]» "https://publicholidays.hk/"      "chung-yeung-festival"              "Chung Yeung Festival"                 
[22]» "https://publicholidays.hk/"      "christmas-day"                     "Christmas Day"                        
[23]» "https://publicholidays.hk/"      "first-weekday-after-christmas-day" "The First Weekday After Christmas Day"
[24]» "https://publicholidays.hk/"      "day-following-good-friday"         "The Day Following Good Friday"        
[25]» "https://www.markiteconomics.com" "nikkei-pmi"                        "S&P Global PMI"                       
int CalendarEventByCurrency(const string currency, MqlCalendarEvent &events[])
The CalendarEventByCurrency function fills the passed events array with descriptions of all kinds of
events in the calendar that are associated with the specified currency. The three-letter designation of
currencies is known to all Forex traders.
If an invalid currency code is specified, the function will return 0 (no error) and an empty array.
Specifying NULL or an empty string "" instead of currency, you can get a complete list of calendar
events.
Let's test the function using the script CalendarEventKindsByCurrency.mq5. The input parameter
specifies the currency code.
input string Currency = "CNY";
In the handler OnStart we request events and output them to the log.

---

## Page 1701

Part 7. Advanced language tools
1 701 
7.3 Economic calendar
void OnStart()
{
   MqlCalendarEvent events[];
   if(PRTF(CalendarEventByCurrency(Currency, events)))
   {
      Print("Event kinds for currency: ", Currency);
      ArrayPrint(events);
   }
}
Here is an example of the result (given with abbreviations).
CalendarEventByCurrency(Currency,events)=40 / ok
Event kinds for currency: CNY
          [id] [type] [sector] [frequency] [time_mode] [country_id] [unit] [importance] [multiplier] [digits] »
[ 0] 156010001      1        4           2           0          156      1            2            0        1 »
[ 1] 156010002      1        4           2           0          156      1            1            0        1 »
[ 2] 156010003      1        4           2           0          156      1            1            0        1 »
[ 3] 156010004      1        2           3           0          156      1            3            0        1 »
[ 4] 156010005      1        2           3           0          156      1            2            0        1 »
[ 5] 156010006      1        9           2           0          156      1            2            0        1 »
[ 6] 156010007      1        8           2           0          156      1            2            0        1 »
[ 7] 156010008      1        8           2           0          156      0            3            0        1 »
[ 8] 156010009      1        8           2           0          156      0            3            0        1 »
[ 9] 156010010      1        8           2           0          156      1            2            0        1 »
[10] 156010011      0        5           0           0          156      0            2            0        0 »
[11] 156010012      1        3           2           0          156      1            2            0        1 »
[12] 156010013      1        8           2           0          156      1            1            0        1 »
[13] 156010014      1        8           2           0          156      1            1            0        1 »
[14] 156010015      1        8           2           0          156      0            3            0        1 »
[15] 156010016      1        8           2           0          156      1            2            0        1 »
[16] 156010017      1        9           2           0          156      1            2            0        1 »
[17] 156010018      1        2           3           0          156      1            2            0        1 »
[18] 156020001      1        6           2           3          156      6            2            3        2 »
[19] 156020002      1        6           2           3          156      1            1            0        1 »
[20] 156020003      1        6           2           3          156      1            1            0        1 »
[21] 156020004      1        6           2           3          156      2            2            3        2 »
[22] 156020005      1        6           2           3          156      1            1            0        1 »
[23] 156020006      1        6           2           3          156      1            1            0        1 »
...
Right fragment.

---

## Page 1702

Part 7. Advanced language tools
1 702
7.3 Economic calendar
    »                        [source_url]                                 [event_code]                                       [name]
[ 0]» "http://www.stats.gov.cn/english/"  "cpi-mm"                                     "CPI m/m"                                   
[ 1]» "http://www.stats.gov.cn/english/"  "cpi-yy"                                     "CPI y/y"                                   
[ 2]» "http://www.stats.gov.cn/english/"  "ppi-yy"                                     "PPI y/y"                                   
[ 3]» "http://www.stats.gov.cn/english/"  "gdp-qq"                                     "GDP q/q"                                   
[ 4]» "http://www.stats.gov.cn/english/"  "gdp-yy"                                     "GDP y/y"                                   
[ 5]» "http://www.stats.gov.cn/english/"  "retail-sales-yy"                            "Retail Sales y/y"                          
[ 6]» "http://www.stats.gov.cn/english/"  "industrial-production-yy"                   "Industrial Production y/y"                 
[ 7]» "http://www.stats.gov.cn/english/"  "manufacturing-pmi"                          "Manufacturing PMI"                         
[ 8]» "http://www.stats.gov.cn/english/"  "non-manufacturing-pmi"                      "Non-Manufacturing PMI"                     
[ 9]» "http://www.stats.gov.cn/english/"  "fixed-asset-investment-yy"                  "Fixed Asset Investment y/y"                
[10]» "http://www.stats.gov.cn/english/"  "nbs-press-conference-on-economic-situation" "NBS Press Conference on Economic Situation"
[11]» "http://www.stats.gov.cn/english/"  "unemployment-rate"                          "Unemployment Rate"                         
[12]» "http://www.stats.gov.cn/english/"  "industrial-profit-yy"                       "Industrial Profit y/y"                     
[13]» "http://www.stats.gov.cn/english/"  "industrial-profit-ytd-yy"                   "Industrial Profit YTD y/y"                 
[14]» "http://www.stats.gov.cn/english/"  "composite-pmi"                              "Composite PMI"                             
[15]» "http://www.stats.gov.cn/english/"  "industrial-production-ytd-yy"               "Industrial Production YTD y/y"             
[16]» "http://www.stats.gov.cn/english/"  "retail-sales-ytd-yy"                        "Retail Sales YTD y/y"                      
[17]» "http://www.stats.gov.cn/english/"  "gdp-ytd-yy"                                 "GDP YTD y/y"                               
[18]» "http://english.customs.gov.cn/"    "trade-balance-usd"                          "Trade Balance USD"                         
[19]» "http://english.customs.gov.cn/"    "imports-usd-yy"                             "Imports USD y/y"                           
[20]» "http://english.customs.gov.cn/"    "exports-usd-yy"                             "Exports USD y/y"                           
[21]» "http://english.customs.gov.cn/"    "trade-balance"                              "Trade Balance"                             
[22]» "http://english.customs.gov.cn/"    "imports-yy"                                 "Imports y/y"                               
[23]» "http://english.customs.gov.cn/"    "exports-yy"                                 "Exports y/y"                               
...
An attentive reader will notice that the event type identifier contains the country code, the number of
the news source and the serial number within the source (numbering starts from 1 ). So, the general
format of the event type identifier is: CCCSSNNNN, where CCC is the country code, SS is the source,
NNNN is the number. For example, 1 56020001  is the first news from the second source for China and
34403001 0 is the tenth news from the third source for Hong Kong. The only exception is global news,
for which the "country" code is not 000 but 1 000.
7.3.4 Getting event descriptions by ID
Real MQL programs, as a rule, request current or upcoming calendar events, filtering by time range,
countries, currencies, or other criteria. The API functions intended for this, which we have yet to
consider, return MqlCalendarValue structures, which store only the event identifier instead of its
description. Therefore, the CalendarEventById function can be useful if you need to extract complete
information.
bool CalendarEventById(ulong id, MqlCalendarEvent &event)
The CalendarEventById function gets the description of the event by its ID. The function returns a
success or error indication.
An example of how to use this function will be given in the next section.
7.3.5 Getting event records by country or currency
Specific events of various kinds are queried in the calendar for a given range of dates and filtered by
country or currency.

---

## Page 1703

Part 7. Advanced language tools
1 703
7.3 Economic calendar
int CalendarValueHistory(MqlCalendarValue &values[], datetime from, datetime to = 0,
   const string country = NULL, const string currency = NULL)
The CalendarValueHistory function fills the values array passed by reference with calendar entries in the
time range between from and to. Both parameters may include date and time. Value from is included in
the interval, but value to is not. In other words, the function selects calendar entries
(structuresMqlCalendarValue), in which the following compound condition is met for the time property:
from <= time < to.
The start time from must be specified, while the end time to is optional: if it is omitted or equal to 0, all
future events are copied to the array.
Time to there should be larger than from, except when it is 0. A special combination for querying all
available events (both past and future) is when from and to are both 0.
If the receiving array is dynamic, memory will be automatically allocated for it. If the array is of a fixed
size, the number of entries copied will be no more than the size of the array.
The country and currency parameters allow you to set an additional filtering of records by country or
currency. The country parameter accepts a two-letter ISO 31 66-1  alpha-2 country code (for example.
"DE", "FR", "EU"), and the currency parameter accepts a three-letter currency designation (for
example, "EUR", "CNY").
The default value NULL or an empty string "" in any of the parameters is equivalent to the absence of
the corresponding filter.
If both filters are specified, only the values of those events are selected for which both conditions –
country and currency – are satisfied simultaneously. This can come in handy if the calendar includes
countries with multiple currencies, each of which also has circulation in several countries. There are no
such events in the calendar at the moment. To get the events in the Eurozone countries, it is enough to
specify the code of a particular country or "EU", and the currency "EUR" will be assumed.
The function returns the number of elements copied and can set an error code. In particular, if the
request timeout from the server is exceeded, in _ LastError we get error 5401 
(ERR_CALENDAR_TIMEOUT). If the fixed array does not fit all the records, the code will be equal to
5400 (ERR_CALENDAR_MORE_DATA), but the array will be filled. When allocating memory for a
dynamic array, error 4004 (ERR_NOT_ENOUGH_MEMORY) is potentially possible.
Attention! The order of the elements in an array can be different from chronological. You have to
sort records by time.
Using the CalendarValueHistory function, we could query upcoming events like this:
   MqlCalendarValue values[];
   if(CalendarValueHistory(values, TimeCurrent()))
   {
      ArrayPrint(values);
   }
However, with this code, we will get a table with insufficient information, where the event names,
importance, and currency codes will be hidden behind the event ID in the MqlCalendarValue::event_ id
field and, indirectly, behind the country identifier in the MqlCalendarEvent::country_ id field. To make the
output of information more user-friendly, you should request a description of the event by the event
code, take the country code from this description, and get its attributes. Let's show it in the example
script CalendarForDates.mq5.

---

## Page 1704

Part 7. Advanced language tools
1 704
7.3 Economic calendar
In the input parameters, we will provide the ability to enter the country code and currency for filtering.
By default, events for the European Union are requested.
input string CountryCode = "EU";
input string Currency = "";
The date range of the events will automatically count for some time back and forth. This "some time"
will also be left to the user to choose from three options: a day, a week, or a month.
#define DAY_LONG   60 * 60 * 24
#define WEEK_LONG  DAY_LONG * 7
#define MONTH_LONG DAY_LONG * 30
#define YEAR_LONG  MONTH_LONG * 12
   
enum ENUM_CALENDAR_SCOPE
{
   SCOPE_DAY = DAY_LONG,
   SCOPE_WEEK = WEEK_LONG,
   SCOPE_MONTH = MONTH_LONG,
   SCOPE_YEAR = YEAR_LONG,
};
   
input ENUM_CALENDAR_SCOPE Scope = SCOPE_DAY;
Let's define our structure MqlCalendarRecord, derivative of MqlCalendarValue, and add fields to it for a
convenient presentation of attributes that will be filled in by links (identifiers) from dependent
structures.
struct MqlCalendarRecord: public MqlCalendarValue
{
   static const string importances[];
   
   string importance;
   string name;
   string currency;
   string code;
   double actual, previous, revised, forecast;
   ...
};
   
static const string MqlCalendarRecord::importances[] = {"None", "Low", "Medium", "High"};
Among the added fields there are lines with importance (one of the values of the static array
importances), the name of the event, country, and currency, as well as four values in the double
format. This actually means duplication of information for the sake of visual presentation when printing.
Later we will prepare a more advanced "wrapper" for the calendar.
To fill the object, we will need a parametric constructor that takes the original structure
MqlCalendarValue. After all the inherited fields are implicitly copied into the new object by the operator
'=', we call the specially prepared extend method.

---

## Page 1705

Part 7. Advanced language tools
1 705
7.3 Economic calendar
   MqlCalendarRecord() { }
   
   MqlCalendarRecord(const MqlCalendarValue &value)
   {
      this = value;
      extend();
   }
In the extend method, we get the description of the event by its identifier. Then, based on the country
identifier from the event description, we get a structure with country attributes. After that, we can fill
in the first half of the added fields from the received structures MqlCalendarEvent and
MqlCalendarCountry.
   void extend()
   {
      MqlCalendarEvent event;
      CalendarEventById(event_id, event);
      
      MqlCalendarCountry country;
      CalendarCountryById(event.country_id, country);
      
      importance = importances[event.importance];
      name = event.name;
      currency = country.currency;
      code = country.code;
      
      MqlCalendarValue value = this;
      
      actual = value.GetActualValue();
      previous = value.GetPreviousValue();
      revised = value.GetRevisedValue();
      forecast = value.GetForecastValue();
   }
Next, we called the built-in Get methods for filling four fields of type double with financial indicators.
Now we can use the new structure in the main OnStart handler.

---

## Page 1706

Part 7. Advanced language tools
1 706
7.3 Economic calendar
void OnStart()
{
   MqlCalendarValue values[];
   MqlCalendarRecord records[];
   datetime from = TimeCurrent() - Scope;
   datetime to = TimeCurrent() + Scope;
   if(PRTF(CalendarValueHistory(values, from, to, CountryCode, Currency)))
   {
      for(int i = 0; i < ArraySize(values); ++i)
      {
         PUSH(records, MqlCalendarRecord(values[i]));
      }
      Print("Near past and future calendar records (extended): ");
      ArrayPrint(records);
   }
}
Here the array of standard MqlCalendarValue structures is filled by calling CalendarValueHistory for the
current conditions set in the input parameters. Next, all elements are transferred to the
MqlCalendarRecord array. Moreover, while objects are being created, they are expanded with additional
information. Finally, the array of events is output to the log.
The log entries are coming quite long. First, let's show the left half, which is exactly what we would see
if we printed an array of standard MqlCalendarValue structures.
CalendarValueHistory(values,from,to,CountryCode,Currency)=6 / ok
Near past and future calendar records (extended): 
      [id] [event_id]              [time]            [period] [revision] [actual_value]         [prev_value] [revised_prev_value]     [forecast_value] [impact_type]
[0] 162723  999020003 2022.06.23 03:00:00 1970.01.01 00:00:00    0 -9223372036854775808 -9223372036854775808 -9223372036854775808 -9223372036854775808             0
[1] 162724  999020003 2022.06.24 03:00:00 1970.01.01 00:00:00    0 -9223372036854775808 -9223372036854775808 -9223372036854775808 -9223372036854775808             0
[2] 168518  999010034 2022.06.24 11:00:00 1970.01.01 00:00:00    0 -9223372036854775808 -9223372036854775808 -9223372036854775808 -9223372036854775808             0
[3] 168515  999010031 2022.06.24 13:10:00 1970.01.01 00:00:00    0 -9223372036854775808 -9223372036854775808 -9223372036854775808 -9223372036854775808             0
[4] 168509  999010014 2022.06.24 14:30:00 1970.01.01 00:00:00    0 -9223372036854775808 -9223372036854775808 -9223372036854775808 -9223372036854775808             0
[5] 161014  999520001 2022.06.24 22:30:00 2022.06.21 00:00:00    0 -9223372036854775808             -6000000 -9223372036854775808 -9223372036854775808             0
Here is the second half with the "decoding" of names, importance, and meanings.
CalendarValueHistory(values,from,to,CountryCode,Currency)=6 / ok
Near past and future calendar records (extended):
     [importance]                                                [name] [currency] [code] [actual] [previous] [revised] [forecast]
[0]  "High"       "EU Leaders Summit"                                   "EUR"      "EU"        nan        nan       nan        nan
[1]  "High"       "EU Leaders Summit"                                   "EUR"      "EU"        nan        nan       nan        nan
[2]  "Medium"     "ECB Supervisory Board Member McCaul Speech"          "EUR"      "EU"        nan        nan       nan        nan
[3]  "Medium"     "ECB Supervisory Board Member Fernandez-Bollo Speech" "EUR"      "EU"        nan        nan       nan        nan
[4]  "Medium"     "ECB Vice President de Guindos Speech"                "EUR"      "EU"        nan        nan       nan        nan
[5]  "Low"        "CFTC EUR Non-Commercial Net Positions"               "EUR"      "EU"        nan   -6.00000       nan        nan
7.3.6 Getting event records of a specific type 
If necessary, an MQL program has the ability to request events of a specific type: to do this, it is
enough to know the event identifier in advance, for example, using the CalendarEventByCountry or

---

## Page 1707

Part 7. Advanced language tools
1 707
7.3 Economic calendar
CalendarEventByCurrency functions which were presented in the section Querying event types by
country and currency.
int CalendarValueHistoryByEvent(ulong id, MqlCalendarValue &values[], datetime from, datetime to =
0)
The CalendarValueHistoryByEvent function fills the array passed by reference with records of events of
a specific type indicated by the id identifier. Parameters from and to allow you to limit the range of
dates in which events are searched.
If an optional parameter to is not specified, all calendar entries will be placed in the array, starting from
the from time and further into the future. To query all the past events, set from to 0. If both from and
to parameters are 0, all history and scheduled events will be returned. In all other cases, when to is not
equal to 0, it must be greater than from.
The values array can be dynamic (then the function will automatically expand or reduce it according to
the amount of data) or of fixed size (then only a part that fits will be copied into the array).
The function returns the number of copied elements.
As an example, consider the script CalendarStatsByEvent.mq5, which calculates the statistics
(frequency of occurrence) of events of different types for a given country or currency in a given time
range.
The analysis conditions are specified in the input variables.
input string CountryOrCurrency = "EU";
input ENUM_CALENDAR_SCOPE Scope = SCOPE_YEAR;
Depending on the length of the CountryOrCurrency string, it is interpreted as a country code (2
characters) or currency code (3 characters).
To collect statistics, we will declare a structure; its fields will store the identifier and name of the event
type, its importance, and the counter of such events.
struct CalendarEventStats
{
   static const string importances[];
   ulong id;
   string name;
   string importance;
   int count;
};
   
static const string CalendarEventStats::importances[] = {"None", "Low", "Medium", "High"};
In the OnStart function, we first request all kinds of events using the CalendarEventByCountry or
CalendarEventByCurrency function to the specified depth of history and into the future, and then, in a
loop through the event descriptions received in the events array, we call CalendarValueHistoryByEvent
for each event ID. In this application, we are not interested in the contents of the values array, as we
just need to know their count.

---

## Page 1708

Part 7. Advanced language tools
1 708
7.3 Economic calendar
void OnStart()
{
   MqlCalendarEvent events[];
   MqlCalendarValue values[];
   CalendarEventStats stats[];
   
   const datetime from = TimeCurrent() - Scope;
   const datetime to = TimeCurrent() + Scope;
   
   if(StringLen(CountryOrCurrency) == 2)
   {
      PRTF(CalendarEventByCountry(CountryOrCurrency, events));
   }
   else
   {
      PRTF(CalendarEventByCurrency(CountryOrCurrency, events));
   }
   
   for(int i = 0; i < ArraySize(events); ++i)
   {
      if(CalendarValueHistoryByEvent(events[i].id, values, from, to))
      {
         CalendarEventStats event = {events[i].id, events[i].name,
            CalendarEventStats::importances[events[i].importance], ArraySize(values)};
         PUSH(stats, event);
      }
   }
   
   SORT_STRUCT(CalendarEventStats, stats, count);
   ArrayReverse(stats);
   ArrayPrint(stats);
}
Upon successful function call, we fill the CalendarEventStats structure and add it to the array of
structures stats. Next, we sort the structure in the way we already know (the SORT_STRUCT macro is
described in the section Comparing, sorting, and searching in arrays).
Running the script with default settings generates something like this in the log (abbreviated).

---

## Page 1709

Part 7. Advanced language tools
1 709
7.3 Economic calendar
CalendarEventByCountry(CountryOrCurrency,events)=82 / ok
          [id]                                                [name] [importance] [count]
[ 0] 999520001 "CFTC EUR Non-Commercial Net Positions"               "Low"             79
[ 1] 999010029 "ECB President Lagarde Speech"                        "High"            69
[ 2] 999010035 "ECB Executive Board Member Elderson Speech"          "Medium"          37
[ 3] 999030027 "Core CPI"                                            "Low"             36
[ 4] 999030026 "CPI"                                                 "Low"             36
[ 5] 999030025 "CPI excl. Energy and Unprocessed Food y/y"           "Low"             36
[ 6] 999030024 "CPI excl. Energy and Unprocessed Food m/m"           "Low"             36
[ 7] 999030010 "Core CPI m/m"                                        "Medium"          36
[ 8] 999030013 "CPI y/y"                                             "Low"             36
[ 9] 999030012 "Core CPI y/y"                                        "Low"             36
[10] 999040006 "Consumer Confidence Index"                           "Low"             36
[11] 999030011 "CPI m/m"                                             "Medium"          36
...
[65] 999010008 "ECB Economic Bulletin"                               "Medium"           8
[66] 999030023 "Wage Costs y/y"                                      "Medium"           6
[67] 999030009 "Labour Cost Index"                                   "Low"              6
[68] 999010025 "ECB Bank Lending Survey"                             "Low"              6
[69] 999010030 "ECB Supervisory Board Member af Jochnick Speech"     "Medium"           4
[70] 999010022 "ECB Supervisory Board Member Hakkarainen Speech"     "Medium"           3
[71] 999010028 "ECB Financial Stability Review"                      "Medium"           3
[72] 999010009 "ECB Targeted LTRO"                                   "Medium"           2
[73] 999010036 "ECB Supervisory Board Member Tuominen Speech"        "Medium"           1
Please note that a total of 82 types of events were received, however, in the statistics array, we had
only 74. This is because the CalendarValueHistoryByEvent function returns false (failure) and zero error
code in _ LastError if there were no events of any kind in the specified date range. In the above test,
there are 8 such entries that theoretically exist but were never encountered within the year.
7.3.7 Reading event records by ID
Knowing the events schedule for the near future, traders can adjust their robots accordingly. There are
no functions or events in the calendar API ("events" in the sense of functions for processing new
financial information like OnCalendar, by analogy with OnTick) to automatically track news releases.
The algorithm must do this itself at any chosen frequency. In particular, you can find out the identifier
of the desired event using one of the previously discussed functions (for example,
CalendarValueHistoryByEvent, CalendarValueHistory) and then call CalendarValueById to get the current
state of the fields in the MqlCalendarValue structure.
bool CalendarValueById(ulong id, MqlCalendarValue &value)
The function fills the structure passed by reference with current information about a specific event.
The result of the function denotes a sign of success (true) or error (false).
Let's create a simple bufferless indicator CalendarRecordById.mq5, which will find in the future the
nearest event with the type of "financial indicator" (i.e., a numerical indicator) and will poll its status on
timer. When the news is published, the data will change (the "actual" value of the indicator will become
known), and the indicator will display an alert.
The frequency of polling the calendar is set in the input variable.

---

## Page 1710

Part 7. Advanced language tools
1 71 0
7.3 Economic calendar
input uint TimerSeconds = 5;
We run the timer in OnInit.
void OnInit()
{
   EventSetTimer(TimerSeconds);
}
For the convenient output to the event description log, we use the MqlCalendarRecord structure which
we already know from the example with the script CalendarForDates.mq5.
To store the initial state of news information, we describe the track structure.
MqlCalendarValue track;
When the structure is empty (and there is "0" in the field id), the program must query the upcoming
events and find among them the closest one with the CALENDAR_TYPE_INDICATOR type and for which
the current value is not yet known.
void OnTimer()
{
   if(!track.id)
   {
      MqlCalendarValue values[];
      if(PRTF(CalendarValueHistory(values, TimeCurrent(), TimeCurrent() + DAY_LONG * 3)))
      {
         for(int i = 0; i < ArraySize(values); ++i)
         {
            MqlCalendarEvent event;
            CalendarEventById(values[i].event_id, event);
            if(event.type == CALENDAR_TYPE_INDICATOR && !values[i].HasActualValue())
            {
               track = values[i];
               PrintFormat("Started monitoring %lld", track.id);
               StructPrint(MqlCalendarRecord(track), ARRAYPRINT_HEADER);
               return;
            }
         }
      }
   }
   ...
The found event is copied to track and output to the log. After that, every call to OnTimer comes down
to getting updated information about the event into the update structure, which is transferred to
CalendarValueById with the track.id identifier. Next, the original and new structures are compared using
the auxiliary function StructCompare (based on StructToCharArray and ArrayCompare, see the
complete source code). Any difference causes a new state to be printed (the forecast may have
changed), and if the current value appears, the timer stops. To start waiting for the next news, this
indicator needs to be reinitialized: this one is for demonstration, and to control the situation according
to the list of news, we will later develop a more practical filter class.

---

## Page 1711

Part 7. Advanced language tools
1 71 1 
7.3 Economic calendar
   else
   {
      MqlCalendarValue update;
      if(CalendarValueById(track.id, update))
      {
         if(fabs(StructCompare(track, update)) == 1)
         {
            Alert(StringFormat("News %lld changed", track.id));
            PrintFormat("New state of %lld", track.id);
            StructPrint(MqlCalendarRecord(update), ARRAYPRINT_HEADER);
            if(update.HasActualValue())
            {
               Print("Timer stopped");
               EventKillTimer();
            }
            else
            {
               track = update;
            }
         }
      }
      
      if(TimeCurrent() <= track.time)
      {
         Comment("Forthcoming event time: ", track.time,
            ", remaining: ", Timing::stringify((uint)(track.time - TimeCurrent())));
      }
      else
      {
         Comment("Forthcoming event time: ", track.time,
            ", late for: ", Timing::stringify((uint)(TimeCurrent() - track.time)));
      }
   }
}
While waiting for the event, the indicator displays a comment with the expected time of the news
release and how much time is left before it (or what is the delay).

---

## Page 1712

Part 7. Advanced language tools
1 71 2
7.3 Economic calendar
Comment about waiting or being late for the next news
It is important to note that the news may come out a little earlier or a little later than the scheduled
date. This creates some problems when testing news strategies on history, since the time of updating
calendar entries in the terminal and through the MQL5 API is not provided. We will try to partially solve
this problem in the next section.
Here are fragments of the log output produced by the indicator with a gap:
CalendarValueHistory(values,TimeCurrent(),TimeCurrent()+(60*60*24)*3)=186 / ok
Started monitoring 156045
  [id] [event_id]              [time]            [period] [revision] »
156045  840020013 2022.06.27 15:30:00 2022.05.01 00:00:00          0 »
»       [actual_value] [prev_value] [revised_prev_value] [forecast_value] [impact_type] »
» -9223372036854775808       400000 -9223372036854775808                0             0 »
» [importance]                     [name] [currency] [code] [actual] [previous] [revised] [forecast]
» "Medium"     "Durable Goods Orders m/m" "USD"      "US"        nan    0.40000       nan    0.00000
...
Alert: News 156045 changed
New state of 156045
  [id] [event_id]              [time]            [period] [revision] »
156045  840020013 2022.06.27 15:30:00 2022.05.01 00:00:00          0 »
» [actual_value] [prev_value] [revised_prev_value] [forecast_value] [impact_type] »
»         700000       400000 -9223372036854775808                0             1 »
» [importance]                     [name] [currency] [code] [actual] [previous] [revised] [forecast]
» "Medium"     "Durable Goods Orders m/m" "USD"      "US"    0.70000    0.40000       nan    0.00000
Timer stopped
The updated news has the actual_ value value.
In order not to wait too long during the test, it is advisable to run this indicator during the working
hours of the main markets, when the density of news releases is high.
The CalendarValueById function is not the only one, and probably not the most flexible, with which you
can monitor changes in the calendar. We will look at a couple of other approaches in the following
sections.

---

## Page 1713

Part 7. Advanced language tools
1 71 3
7.3 Economic calendar
7.3.8 Tracking event changes by country or currency
As mentioned in the section on basic concepts of the calendar, the platform registers all event changes
by some internal means. Each state is characterized by a change identifier (change_ id). Among the
MQL5 functions, there are two that allow you to find this identifier (at an arbitrary point in time) and
then request calendar entries changed later. One of these functions is CalendarValueLast, which will be
discussed in this section. The second one, CalendarValueLastByEvent, will be discussed in the next
section.
int CalendarValueLast(ulong &change_id, MqlCalendarValue &values[],
   const string country = NULL, const string currency = NULL)
The CalendarValueLast function is designed for two purposes: getting the last known calendar change
identifier change_ id and filling the values array with modified records since the previous modification
given by the passed ID in the same change_ id. In other words, the change_ id parameter works as both
input and output. That is why it is a reference and requires a variable to be specified.
If we input change_ id equal to 0 into the function, then the function will fill the variable with the current
identifier but will not fill the array.
Optionally, using parameters country and currency, you can set filtering records by country and
currency.
The function returns the number of copied calendar items. Since the array is not populated in the first
operation mode (change_ id = 0), returning 0 is not an error. We can also get 0 if the calendar has not
been modified since the specified change. Therefore, to check for an error, you should analyze
_ LastError.
So the usual way to use the function is to loop through the calendar for changes.
ulong change = 0;
MqlCalendarValue values[];
while(!IsStopped())
{
 // pass the last identifier known to us and get a new one if it appeared
   if(CalendarValueLast(change, values))
   {
 // analysis of added and changed records
      ArrayPrint(values);
      ... 
   }
   Sleep(1000);
}
This can be done in a loop, on a timer, or on other events.
Identifiers are constantly increasing, but they can go out of order, that is, jump over several values.
It is important to note that each calendar entry is always available in only one last state: the
history of changes is not provided in MQL5. As a rule, this is not a problem, since the life cycle of
each news is standard: adding to the database in advance for a sufficiently long time and
supplementing with relevant data at the time of the event. However, in practice, various deviations
can occur: editing the forecast, transferring time, or revising the values. It is impossible to find out
exactly what time and what was changed in the record through the MQL5 API from the calendar
history. Therefore, those trading systems that make decisions based on the momentary situation

---

## Page 1714

Part 7. Advanced language tools
1 71 4
7.3 Economic calendar
will require independent saving of the history of changes and its integration into an Expert Advisor
for running in the tester.
Using the CalendarValueLast function, we can create a useful service, CalendarChangeSaver.mq5, which
will check the calendar for changes at the specified intervals and, if any, save the change identifiers to
the file along with the current server time. This will allow further use of the file information for more
realistic testing of Expert Advisors on the history of the calendar. Of course, this will require organizing
the export/import of the entire calendar database, which we will deal with over time.
Let's provide input variables for specifying the file name and the period between polls (in milliseconds).
input string Filename = "calendar.chn";
input int PeriodMsc = 1000;
At the beginning of the OnStart handler, we open the binary file for writing, or rather for appending (if it
already exists). The format of an existing file is not checked here and thus you should add protection
when embedding in a real application.
void OnStart()
{
   ulong change = 0, last = 0;
   int count = 0;
   int handle = FileOpen(Filename,
      FILE_WRITE | FILE_READ | FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_BIN);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("Can't open file '%s' for writing", Filename);
      return;
   }
   
   const ulong p = FileSize(handle);
   if(p > 0)
   {
      PrintFormat("Resuming file %lld bytes", p);
      FileSeek(handle, 0, SEEK_END);
   }
   
   Print("Requesting start ID...");
   ...
Here we should make a small digression.
Each time the calendar is changed, at least a pair of integer 8-byte numbers must be written to the
file: the current time (datetime) and news ID (ulong), but there can be more than one record changed
at the same time. Therefore, in addition to the date, the number of changed records is packed into the
first number. This takes into account that dates fit in 0x7FFFFFFFF and therefore the upper 3 bytes are
left unused. It is in the two most significant bytes (at a left offset of 48 bits) that the number of
identifiers that the service will write after the corresponding timestamp is placed. The
PACK_DATETIME_COUNTER macro creates an "extended" date, and the other two, DATETIME and
COUNTER, we will need later when the archive of changes is read (by another program).

---

## Page 1715

Part 7. Advanced language tools
1 71 5
7.3 Economic calendar
#define PACK_DATETIME_COUNTER(D,C) (D | (((ulong)(C)) << 48))
#define DATETIME(A) ((datetime)((A) & 0x7FFFFFFFF))
#define COUNTER(A)  ((ushort)((A) >> 48)) 
Now let's go back to the main service code. In a loop that is activated every PeriodMsc milliseconds, we
request changes using CalendarValueLast. If there are changes, we write the current server time and
the array of received identifiers to a file.

---

## Page 1716

Part 7. Advanced language tools
1 71 6
7.3 Economic calendar
   while(!IsStopped())
   {
      if(!TerminalInfoInteger(TERMINAL_CONNECTED))
      {
         Print("Waiting for connection...");
         Sleep(PeriodMsc);
         continue;
      }
      
      MqlCalendarValue values[];
      const int n = CalendarValueLast(change, values);
      if(n > 0)
      {
         string records = "[" + Description(values[0]);
         for(int i = 1; i < n; ++i)
         {
            records += "," + Description(values[i]);
         }
         records += "]";
         Print("New change ID: ", change, " ",
            TimeToString(TimeTradeServer(), TIME_DATE | TIME_SECONDS), "\n", records);
         FileWriteLong(handle, PACK_DATETIME_COUNTER(TimeTradeServer(), n));
         for(int i = 0; i < n; ++i)
         {
            FileWriteLong(handle, values[i].id);
         }
         FileFlush(handle);
         ++count;
      }
      else if(_LastError == 0)
      {
         if(!last && change)
         {
            Print("Start change ID obtained: ", change);
         }
      }
      
      last = change;
      Sleep(PeriodMsc);
   }
   PrintFormat("%d records added", count);
   FileClose(handle);
}
For a convenient presentation of information about each news event, we have written a helper function
Description.

---

## Page 1717

Part 7. Advanced language tools
1 71 7
7.3 Economic calendar
string Description(const MqlCalendarValue &value)
{
   MqlCalendarEvent event;
   MqlCalendarCountry country;
   CalendarEventById(value.event_id, event);
   CalendarCountryById(event.country_id, country);
   return StringFormat("%lld (%s/%s @ %s)",
      value.id, country.code, event.name, TimeToString(value.time));
}
Thus, the log will display not only the identifier but also the country code, title, and scheduled time of
the news.
It is assumed that the service should work for quite a long time in order to collect information for a
period sufficient for testing (days, weeks, months). Unfortunately, just like with the order book, the
platform does not provide a ready-made history of the order book or calendar edits, so their
collection is left entirely to the developer of MQL programs.
Let's see the service in action. In the next fragment of the log (for the time period of 2022.06.28,
1 5:30 - 1 6:00), some news events relate to the distant future (they contain the values of the
prev_value field, which is also the actual_ value field of the current event of the same name). However,
something else is more important: the actual time of a news release can differ significantly, sometimes
by several minutes, from the planned one.

---

## Page 1718

Part 7. Advanced language tools
1 71 8
7.3 Economic calendar
Requesting start ID...
Start change ID obtained: 86358784
New change ID: 86359040 2022.06.28 15:30:42
[155955 (US/Wholesale Inventories m/m @ 2022.06.28 15:30)]
New change ID: 86359296 2022.06.28 15:30:45
[155956 (US/Wholesale Inventories m/m @ 2022.07.08 17:00)]
New change ID: 86359552 2022.06.28 15:30:48
[156117 (US/Goods Trade Balance @ 2022.06.28 15:30)]
New change ID: 86359808 2022.06.28 15:30:51
[156118 (US/Goods Trade Balance @ 2022.07.27 15:30)]
New change ID: 86360064 2022.06.28 15:30:54
[156231 (US/Retail Inventories m/m @ 2022.06.28 15:30)]
New change ID: 86360320 2022.06.28 15:30:57
[156232 (US/Retail Inventories m/m @ 2022.07.15 17:00)]
New change ID: 86360576 2022.06.28 15:31:00
[156255 (US/Retail Inventories excl. Autos m/m @ 2022.06.28 15:30)]
New change ID: 86360832 2022.06.28 15:31:03
[156256 (US/Retail Inventories excl. Autos m/m @ 2022.07.15 17:00)]
New change ID: 86361088 2022.06.28 15:31:07
[155956 (US/Wholesale Inventories m/m @ 2022.07.08 17:00)]
New change ID: 86361344 2022.06.28 15:31:10
[156118 (US/Goods Trade Balance @ 2022.07.27 15:30)]
New change ID: 86361600 2022.06.28 15:31:13
[156232 (US/Retail Inventories m/m @ 2022.07.15 17:00)]
New change ID: 86362368 2022.06.28 15:36:47
[158534 (US/Challenger Job Cuts y/y @ 2022.07.07 14:30)]
New change ID: 86362624 2022.06.28 15:51:23
...
New change ID: 86364160 2022.06.28 16:01:39
[154531 (US/HPI m/m @ 2022.06.28 16:00)]
New change ID: 86364416 2022.06.28 16:01:42
[154532 (US/HPI m/m @ 2022.07.26 16:00)]
New change ID: 86364672 2022.06.28 16:01:46
[154543 (US/HPI y/y @ 2022.06.28 16:00)]
New change ID: 86364928 2022.06.28 16:01:49
[154544 (US/HPI y/y @ 2022.07.26 16:00)]
New change ID: 86365184 2022.06.28 16:01:54
[154561 (US/HPI @ 2022.06.28 16:00)]
New change ID: 86365440 2022.06.28 16:01:58
[154571 (US/HPI @ 2022.07.26 16:00)]
New change ID: 86365696 2022.06.28 16:02:01
[154532 (US/HPI m/m @ 2022.07.26 16:00)]
New change ID: 86365952 2022.06.28 16:02:05
[154544 (US/HPI y/y @ 2022.07.26 16:00)]
New change ID: 86366208 2022.06.28 16:02:09
[154571 (US/HPI @ 2022.07.26 16:00)]
Of course, this is important not for all classes of trading strategies, but only for those that trade quickly
in the market. For them, the created archive of calendar edits can provide more accurate testing of
news Expert Advisors. We will discuss how you can "connect" the calendar to the tester in the future,
but for now, we will show how to read the received file.

---

## Page 1719

Part 7. Advanced language tools
1 71 9
7.3 Economic calendar
We will use the script CalendarChangeReader.mq5 to demonstrate the discussed functionality. In
practice, the given source code should be placed in the Expert Advisor.
The input variables allow you to set the name of the file to be read and the start date of the scan. If
the service continues to work (write the file), you need to copy the file under a different name or to
another folder (in the example script, the file is renamed). If the Start parameter is blank, the reading
of news changes will start from the beginning of the current day.
input string Filename = "calendar2.chn";
input datetime Start;
The ChangeState structure is described to store information about individual edits.
struct ChangeState
{
   datetime dt;
   ulong ids[];
   
   ChangeState(): dt(LONG_MAX) {}
   ChangeState(const datetime at, ulong &_ids[])
   {
      dt = at;
      ArraySwap(ids, _ids);
   }
   
   void operator=(const ChangeState &other)
   {
      dt = other.dt;
      ArrayCopy(ids, other.ids);
   }
};
It is used in the ChangeFileReader class, which does the bulk of the work of reading the file and
providing the caller with the changes that are appropriate for a particular point in time.
The file handle is passed as a parameter to the constructor, as is the start time of the test. Reading a
file and populating the ChangeState structure for one calendar edit is performed in the readState
method.

---

## Page 1720

Part 7. Advanced language tools
1 720
7.3 Economic calendar
class ChangeFileReader
{
   const int handle;
   ChangeState current;
   const ChangeState zero;
   
public:
   ChangeFileReader(const int h, const datetime start = 0): handle(h)
   {
      if(readState())
      {
         if(start)
         {
            ulong dummy[];
            check(start, dummy, true); // find the first edit after start 
         }
      }
   }
   
   bool readState()
   {
      if(FileIsEnding(handle)) return false;
      ResetLastError();
      const ulong v = FileReadLong(handle);
      current.dt = DATETIME(v);
      ArrayFree(current.ids);
      const int n = COUNTER(v);
      for(int i = 0; i < n; ++i)
      {
         PUSH(current.ids, FileReadLong(handle));
      }
      return _LastError == 0;
   }
   ...
Method check reads the file until the next edit appears in the future. In this case, all previous (by
timestamps) edits since the previous method call are placed in the output array records.

---

## Page 1721

Part 7. Advanced language tools
1 721 
7.3 Economic calendar
   bool check(datetime now, ulong &records[], const bool fastforward = false)
   {
      if(current.dt > now) return false;
      
      ArrayFree(records);
      
      if(!fastforward)
      {
         ArrayCopy(records, current.ids);
         current = zero;
      }
      
      while(readState() && current.dt <= now)
      {
         if(!fastforward) ArrayInsert(records, current.ids, ArraySize(records));
      }
      
      return true;
   }
};
Here is how the class is used in OnStart.

---

## Page 1722

Part 7. Advanced language tools
1 722
7.3 Economic calendar
void OnStart()
{
   const long day = 60 * 60 * 24;
   datetime now = Start ? Start : (datetime)(TimeCurrent() / day * day);
   
   int handle = FileOpen(Filename,
      FILE_READ | FILE_SHARE_WRITE | FILE_SHARE_READ | FILE_BIN);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("Can't open file '%s' for reading", Filename);
      return;
   }
   
   ChangeFileReader reader(handle, now);
   
   // reading step by step, time now artificially increased in this demo
   while(!FileIsEnding(handle))
   {
      // in a real application, a call to reader.check can be made on every tick
      ulong records[];
      if(reader.check(now, records))
      {
         Print(now);          // output time
         ArrayPrint(records); // array of IDs of changed news
      }
      now += 60; // add 1 minute at a time, can be per second
   }
   
   FileClose(handle);
}
Here are the results of the script for the same calendar changes that were saved by the service in the
context of the previous log fragment.
2022.06.28 15:31:00
155955 155956 156117 156118 156231 156232 156255
2022.06.28 15:32:00
156256 155956 156118 156232
2022.06.28 15:37:00
158534
...
2022.06.28 16:02:00
154531 154532 154543 154544 154561 154571
2022.06.28 16:03:00
154532 154544 154571
The same identifiers are reproduced in virtual time with the same delay as online, although here you
can see the rounding to 1  minute, which happened because we set an artificial step of this size in the
loop. In theory, for reasons of efficiency, we can postpone checks until the time stored in the
ChangeState current structure. The attached source code defines the getState method to get this
time.

---

## Page 1723

Part 7. Advanced language tools
1 723
7.3 Economic calendar
7.3.9 Tracking event changes by type
The MQL5 API allows you to request recent changes not only in general for the entire calendar or by
country or currency, but also in a narrower range, or rather, for a specific type of event.
In theory, we can say that the built-in functions provide filtering of events according to several basic
conditions: time, country, currency, or type of event. For other attributes, such as importance or
economic sector, you need to implement your own filtering, and we will deal with this later. For now,
let's introduce the CalendarValueLastByEvent function.
int CalendarValueLastByEvent(ulong id, ulong &change_id, MqlCalendarValue &values[])
The function fills the values array passed by reference with event records of a specific type with the id
identifier that have occurred since change_ id. This change_ id parameter is both input and output: the
calling code passes in it the label of the past state of the calendar, after which changes are requested,
and when control returns, the function writes the current label of the calendar database state to
change_ id. It should be used the next time the function is called.
If you pass null in change_ id, then the function does not fill the array but simply sends the current state
of the database through the parameter change_ id.
The array can be dynamic (then it will be automatically adjusted to the amount of data) or fixed size (if
its size is insufficient, only data that fit will be copied).  
The output value of the function is equal to the number of elements copied into the values array. If
there are no changes or change_ id = 0 is specified, the function will return 0.
To check for an error, analyze the built-in _ LastError variable. Some of the possible error codes are:
·4004 - ERR_NOT_ENOUGH_MEMORY (not enough memory to complete the request),
·5401  - ERR_CALENDAR_TIMEOUT (request timed out),
·5400 - ERR_CALENDAR_MORE_DATA (the size of the fixed array is not enough to get all the
values).
We will not give a separate example for CalendarValueLastByEvent. Instead, let's turn to a more
complex, but in-demand task of querying and filtering calendar entries with arbitrary conditions on news
attributes, where all the "calendar" API functions will be involved. This will be the subject of the next
section.
7.3.1 0 Filtering events by multiple conditions
As we know from the previous sections of this chapter, the MQL5 API allows you to request calendar
events based on several conditions:
• by countries (CalendarValueHistory, CalendarValueLast)
• by frequencies (CalendarValueHistory, CalendarValueLast)
• by event type IDs (CalendarValueHistoryByEvent, CalendarValueLastByEvent)
• by time range (CalendarValueHistory, CalendarValueHistoryByEvent)
• by changes since the previous calendar poll (CalendarValueLast, CalendarValueLastByEvent)
• by ID of specific news (CalendarValueById)

---

## Page 1724

Part 7. Advanced language tools
1 724
7.3 Economic calendar
This can be summarized as the following table of functions (of all CalendarValue functions, only
CalendarValueById for getting one specific value is missing here).
Conditions
Time range
Last changes
Countries
CalendarValueHistory
CalendarValueLast
Currencies
CalendarValueHistory
CalendarValueLast
Events
CalendarValueHistoryByEvent
CalendarValueLastByEvent
Such a toolkit covers main, but not all, popular calendar analysis scenarios. Therefore, in practice, it is
often necessary to implement custom filtering mechanisms in MQL5, including, in particular, event
requests by:
• several countries
• several currencies
• several types of events
• values of arbitrary properties of events (importance, sector of the economy, reporting period, type,
presence of a forecast, estimated impact on the rate, substring in the name of the event, etc.)
To solve these problems, we have created the CalendarFilter class (CalendarFilter.mqh).
Due to the specifics of the built-in API functions, some of the news attributes are given higher priority
than the rest. This includes country, currency, and date range. They can be specified in the class
constructor, and then the corresponding property cannot be dynamically changed in the filter
conditions.
This is because the filter class will subsequently be extended with the news caching capabilities to
enable reading from the tester, and the initial conditions of the constructor actually define the caching
context within which further filtering is possible. For example, if we specify the country code "EU" when
creating an object, then obviously it makes no sense to request news about the USA or Brazil through
it. It is similar to the date range: specifying it in the constructor will make it impossible to receive news
outside the range.
We can also create an object without initial conditions (because all constructor parameters are
optional), and then it will be able to cache and filter news across the entire calendar database (as of
the moment of saving).
In addition, since countries and currencies are now almost uniquely displayed (with the exception of the
European Union and EUR), they are passed to the constructor through a single parameter context: if
you specify a string with the length of 2 characters, the country code (or a combination of countries) is
implied, and if the length is 3 characters, the currency code is implied. For the codes "EU" and "EUR",
the euro area is a subset of "EU" (within countries with formal treaties). In special cases, where non-
euro area EU countries are of interest, they can also be described by "EU" context. If necessary,
narrower conditions for news on the currencies of these countries (BGN, HUF, DKK, ISK, PLN, RON,
HRK, CZK, SEK) can be added to the filter dynamically using methods that we will present later.
However, due to exotics, there are no guarantees that such news will get into the calendar.
Let's start studying the class.

---

## Page 1725

Part 7. Advanced language tools
1 725
7.3 Economic calendar
class CalendarFilter
{
protected:
   // initial (optional) conditions set in the constructor, invariants
   string context;    // country and currency
   datetime from, to; // date range
   bool fixedDates;   // if 'from'/'to' are passed in the constructor, they cannot be changed
   
   // dedicated selectors (countries/currencies/event type identifiers)
   string country[], currency[];
   ulong ids[];
   
   MqlCalendarValue values[]; // filtered results
   
   virtual void init()
   {
      fixedDates = from != 0 || to != 0;
      if(StringLen(context) == 3)
      {
         PUSH(currency, context);
      }
      else
      {
         // even if context is NULL, we take it to poll the entire calendar base
         PUSH(country, context);
      }
   }
   ...
public:
   CalendarFilter(const string _context = NULL,
      const datetime _from = 0, const datetime _to = 0):
      context(_context), from(_from), to(_to)
   {
      init();
   }
   ...
Two arrays are allocated for countries and currencies: country and currency. If they are not filled from
context during object creation, then the MQL program will be able to add conditions for several
countries or currencies in order to perform a combined news query on them.
To store conditions on all other news attributes, the selectors array is described in the CalendarFilter
object, with the second dimension equal to 3. We can say that this is a kind of table in which each row
has 3 columns.
   long selectors[][3];   // [0] - property, [1] - value, [2] - condition
At the 0th index, the news property identifiers will be located. Since the attributes are spread across
three base tables (MqlCalendarCountry, MqlCalendarEvent, MqlCalendarValue) they are described using
the elements of the generalized enumeration ENUM_CALENDAR_PROPERTY (CalendarDefines.mqh).

---

## Page 1726

Part 7. Advanced language tools
1 726
7.3 Economic calendar
enum ENUM_CALENDAR_PROPERTY
{                                      // +/- means support for field filtering
   CALENDAR_PROPERTY_COUNTRY_ID,       // -ulong
   CALENDAR_PROPERTY_COUNTRY_NAME,     // -string
   CALENDAR_PROPERTY_COUNTRY_CODE,     // +string (2 characters)
   CALENDAR_PROPERTY_COUNTRY_CURRENCY, // +string (3 characters)
   CALENDAR_PROPERTY_COUNTRY_GLYPH,    // -string (1 characters)
   CALENDAR_PROPERTY_COUNTRY_URL,      // -string
   
   CALENDAR_PROPERTY_EVENT_ID,         // +ulong (event type ID)
   CALENDAR_PROPERTY_EVENT_TYPE,       // +ENUM_CALENDAR_EVENT_TYPE
   CALENDAR_PROPERTY_EVENT_SECTOR,     // +ENUM_CALENDAR_EVENT_SECTOR
   CALENDAR_PROPERTY_EVENT_FREQUENCY,  // +ENUM_CALENDAR_EVENT_FREQUENCY
   CALENDAR_PROPERTY_EVENT_TIMEMODE,   // +ENUM_CALENDAR_EVENT_TIMEMODE
   CALENDAR_PROPERTY_EVENT_UNIT,       // +ENUM_CALENDAR_EVENT_UNIT
   CALENDAR_PROPERTY_EVENT_IMPORTANCE, // +ENUM_CALENDAR_EVENT_IMPORTANCE
   CALENDAR_PROPERTY_EVENT_MULTIPLIER, // +ENUM_CALENDAR_EVENT_MULTIPLIER
   CALENDAR_PROPERTY_EVENT_DIGITS,     // -uint
   CALENDAR_PROPERTY_EVENT_SOURCE,     // +string ("http[s]://")
   CALENDAR_PROPERTY_EVENT_CODE,       // -string
   CALENDAR_PROPERTY_EVENT_NAME,       // +string (4+ characters or wildcard '*')
   
   CALENDAR_PROPERTY_RECORD_ID,        // -ulong
   CALENDAR_PROPERTY_RECORD_TIME,      // +datetime
   CALENDAR_PROPERTY_RECORD_PERIOD,    // +datetime (like long)
   CALENDAR_PROPERTY_RECORD_REVISION,  // +int
   CALENDAR_PROPERTY_RECORD_ACTUAL,    // +long
   CALENDAR_PROPERTY_RECORD_PREVIOUS,  // +long
   CALENDAR_PROPERTY_RECORD_REVISED,   // +long
   CALENDAR_PROPERTY_RECORD_FORECAST,  // +long
   CALENDAR_PROPERTY_RECORD_IMPACT,    // +ENUM_CALENDAR_EVENT_IMPACT
   
   CALENDAR_PROPERTY_RECORD_PREVISED,  // +non-standard (previous or revised if any)
   
   CALENDAR_PROPERTY_CHANGE_ID,        // -ulong (reserved)
};
Index 1  will store values for comparison with them in the conditions for selecting news records. For
example, if you want to set a filter by sector of the economy, then we write
CALENDAR_PROPERTY_EVENT_SECTOR in selectors[i][0] and one of the values of the standard
enumeration ENUM_CALENDAR_EVENT_SECTOR in selectors[i][1 ].
Finally, the last column (under the 2nd index) is reserved for the operation of comparing the selector
value with the attribute value in the news: all supported operations are summarized in the IS
enumeration.

---

## Page 1727

Part 7. Advanced language tools
1 727
7.3 Economic calendar
enum IS
{
   EQUAL,
   NOT_EQUAL,
   GREATER,
   LESS,
   OR_EQUAL,
   ...
};
We saw a similar approach in TradeFilter.mqh. Thus, we will be able to arrange conditions not only for
equality of values but also for inequality or more/less relations. For example, it is easy to imagine a
filter on the CALENDAR_PROPERTY_EVENT_IMPORTANCE field, which should be GREATER than
CALENDAR_IMPORTANCE_LOW (this is an element of the standard
ENUM_CALENDAR_EVENT_IMPORTANCE enumeration), which means a selection of news of medium
and high importance.
The next enumeration defined specifically for the calendar is ENUM_CALENDAR_SCOPE. Since calendar
filtering is often associated with time spans, the most requested ones are listed here.
#define DAY_LONG     (60 * 60 * 24)
#define WEEK_LONG    (DAY_LONG * 7)
#define MONTH_LONG   (DAY_LONG * 30)
#define QUARTER_LONG (MONTH_LONG * 3)
#define YEAR_LONG    (MONTH_LONG * 12)
   
enum ENUM_CALENDAR_SCOPE
{
   SCOPE_DAY = DAY_LONG,         // Day
   SCOPE_WEEK = WEEK_LONG,       // Week
   SCOPE_MONTH = MONTH_LONG,     // Month
   SCOPE_QUARTER = QUARTER_LONG, // Quarter
   SCOPE_YEAR = YEAR_LONG,       // Year
};
All enumerations are placed in a separate header file CalendarDefines.mqh.
But let's go back to the class CalendarFilter. The type of the selectors array is long, which is suitable
for storing values of almost all involved types: enumerations, dates and times, identifiers, integers, and
even economic indicators values because they are stored in the calendar in the form of long numbers
(in millionths of real values). However, what to do with string properties?
This problem is solved by using the array of strings stringCache, to which all the lines mentioned in the
filter conditions will be added.
class CalendarFilter
{
protected:
   ...
   string stringCache[];  // cache of all rows in 'selectors'
   ...
Then, instead of the string value in selectors[i][1 ], we can easily save the index of an element in the
stringCache array.

---

## Page 1728

Part 7. Advanced language tools
1 728
7.3 Economic calendar
To populate the selectors array with filter conditions, there are several let methods provided, in
particular, for enumerations:
class CalendarFilter
{
...
public:
   // all fields of enum types are processed here
   template<typename E>
   CalendarFilter *let(const E e, const IS c = EQUAL)
   {
      const int n = EXPAND(selectors);
      selectors[n][0] = resolve(e); // by type E, returning the element ENUM_CALENDAR_PROPERTY
      selectors[n][1] = e;
      selectors[n][2] = c;
      return &this;
   }
   ...
For actual values of indicators:
   // the following fields are processed here:
   // CALENDAR_PROPERTY_RECORD_ACTUAL, CALENDAR_PROPERTY_RECORD_PREVIOUS,
   // CALENDAR_PROPERTY_RECORD_REVISED, CALENDAR_PROPERTY_RECORD_FORECAST,
   // and CALENDAR_PROPERTY_RECORD_PERIOD (as long)
   CalendarFilter *let(const long value, const ENUM_CALENDAR_PROPERTY property, const IS c = EQUAL)
   {
      const int n = EXPAND(selectors);
      selectors[n][0] = property;
      selectors[n][1] = value;
      selectors[n][2] = c;
      return &this;
   }
   ...
And for strings:

---

## Page 1729

Part 7. Advanced language tools
1 729
7.3 Economic calendar
   // conditions for all string properties can be found here (abbreviated)
   CalendarFilter *let(const string find, const IS c = EQUAL)
   {
      const int wildcard = (StringFind(find, "*") + 1) * 10;
      switch(StringLen(find) + wildcard)
      {
      case 2:
         // if the initial context is different from the country, we can supplement it with the country,
         // otherwise the filter is ignored
         if(StringLen(context) != 2)
         {
            if(ArraySize(country) == 1 && StringLen(country[0]) == 0)
            {
               country[0] = find; // narrow down "all countries" to one (may add more)
            }
            else
            {
               PUSH(country, find);
            }
         }
         break;
      case 3:
         // we can set a filter for a currency only if it was not in the initial context
         if(StringLen(context) != 3)
         {
            PUSH(currency, find);
         }
         break;
      default:
         {
            const int n = EXPAND(selectors);
            PUSH(stringCache, find);
            if(StringFind(find, "http://") == 0 || StringFind(find, "https://") == 0)
            {
               selectors[n][0] = CALENDAR_PROPERTY_EVENT_SOURCE;
            }
            else
            {
               selectors[n][0] = CALENDAR_PROPERTY_EVENT_NAME;
            }
            selectors[n][1] = ArraySize(stringCache) - 1;
            selectors[n][2] = c;
            break;
         }
      }
      
      return &this;
   }
In the method overload for strings, note that 2 or 3-character long strings (if they are without the
template asterisk '*', which is a replacement for an arbitrary sequence of characters) fall into the

---

## Page 1730

Part 7. Advanced language tools
1 730
7.3 Economic calendar
arrays of countries and symbols, respectively, and all other strings are treated as fragments of the
name or news source, and both of these fields involve stringCache and selectors.
In a special way, the class also supports filtering by type (identifier) of events.
protected:
   ulong ids[];           // filtered event types
   ...
public:
   CalendarFilter *let(const ulong event)
   {
      PUSH(ids, event);
      return &this;
   }
   ...
Thus, the number of priority filters (which are processed outside the selectors array) includes not only
countries, currencies, and date ranges, but also event type identifiers. Such a constructive decision is
due to the fact that these parameters can be passed to certain calendar API functions as input. We get
all other news attributes as output field values in arrays of structures (MqlCalendarValue,
MqlCalendarEvent, MqlCalendarCountry). It is by them that we will perform additional filtering,
according to the rules in the selectors array.
All let methods return a pointer to an object, which allows their calls to be chained. For example, like
this:
CalendarFilter f;
f.let(CALENDAR_IMPORTANCE_LOW, GREATER) // important and moderately important news
  .let(CALENDAR_TIMEMODE_DATETIME) // only events with exact time
  .let("DE").let("FR") // a couple of countries, or, to choose from...
  .let("USD").let("GBP") // ...a couple of currencies (but both conditions won't work at once)
  .let(TimeCurrent() - MONTH_LONG, TimeCurrent() + WEEK_LONG) // date range "around" the current time
  .let(LONG_MIN, CALENDAR_PROPERTY_RECORD_FORECAST, NOT_EQUAL) // there is a forecast
  .let("farm"); // full text search by news titles
Country and currency conditions can, in theory, be combined. However, please note that multiple
values can only be set for either countries or currencies but not both. One of these two aspects of the
context (either of the two) in the current implementation supports only one or none of the values (i.e.,
no filter on it). For example, if the currency EUR is selected, it is possible to narrow the search context
for news only in Germany and France (country codes "DE" and "FR"). As a result, ECB and Eurostat
news will be discarded, as well as, specifically, Italy and Spain news. However, the indication of EUR in
this case is redundant since there are no other currencies in Germany and France.
Since the class uses built-in functions in which the parameters country and currency are applied to the
news using the logical AND operation, check the consistency of the filter conditions.
After the calling code sets up the filtering conditions, it is necessary to select news based on them.
This is what the public method select does (given with simplifications).

---

## Page 1731

Part 7. Advanced language tools
1 731 
7.3 Economic calendar
public:
   bool select(MqlCalendarValue &result[])
   {
      int count = 0;
      ArrayFree(result);
      if(ArraySize(ids)) // identifiers of event types
      {
         for(int i = 0; i < ArraySize(ids); ++i)
         {
            MqlCalendarValue temp[];
            if(PRTF(CalendarValueHistoryByEvent(ids[i], temp, from, to)))
            {
               ArrayCopy(result, temp, ArraySize(result));
               ++count;
            }
         }
      }
      else
      {
         // several countries or currencies, choose whichever is more as a basis,
         // only the first element from the smaller array is used
         if(ArraySize(country) > ArraySize(currency))
         {
            const string c = ArraySize(currency) > 0 ? currency[0] : NULL;
            for(int i = 0; i < ArraySize(country); ++i)
            {
               MqlCalendarValue temp[];
               if(PRTF(CalendarValueHistory(temp, from, to, country[i], c)))
               {
                  ArrayCopy(result, temp, ArraySize(result));
                  ++count;
               }
            }
         }
         else
         {
            const string c = ArraySize(country) > 0 ? country[0] : NULL;
            for(int i = 0; i < ArraySize(currency); ++i)
            {
               MqlCalendarValue temp[];
               if(PRTF(CalendarValueHistory(temp, from, to, c, currency[i])))
               {
                  ArrayCopy(result, temp, ArraySize(result));
                  ++count;
               }
            }
         }
      }
      
      if(ArraySize(result) > 0)
      {

---

## Page 1732

Part 7. Advanced language tools
1 732
7.3 Economic calendar
         filter(result);
      }
      
      if(count > 1 && ArraySize(result) > 1)
      {
         SORT_STRUCT(MqlCalendarValue, result, time);
      }
      
      return ArraySize(result) > 0;
   }
Depending on which of the priority attribute arrays are filled, the method calls different API functions to
poll the calendar:
• If the ids array is filled, CalendarValueHistoryByEvent is called in a loop for all identifiers
• If the country array is filled and it's larger than the array of currencies, call CalendarValueHistory
and loop through the countries
• If the currency array is filled and it is greater than or equal to the size of the array of countries, call
CalendarValueHistory and loop through the currencies
Each function call populates a temporary array of structures MqlCalendarValue temp[], which is
sequentially accumulated in the result parameter array. After writing all relevant news into it according
to the main conditions (dates, countries, currencies, identifiers), if any, an auxiliary method filter
comes into play, which filters the array based on the conditions in selectors. At the end of the select
method, the news items are sorted in chronological order, which can be broken by combining the
results of multiple queries of "calendar" functions. Sorting is implemented using the SORT_STRUCT
macro, which was discussed in the section Comparing, sorting, and searching in arrays.
For each element of the news array, the filter method calls the worker method match, which returns a
boolean indicator of whether the news matches the filter conditions. If not, the element is removed
from the array.
protected:
   void filter(MqlCalendarValue &result[])
   {
      for(int i = ArraySize(result) - 1; i >= 0; --i)
      {
         if(!match(result[i]))
         {
            ArrayRemove(result, i, 1);
         }
      }
   }
   ...
Finally, the match method analyzes our selectors array and compares it with the fields of the passed
structure MqlCalendarValue. Here the code is provided in an abbreviated form.

---

## Page 1733

Part 7. Advanced language tools
1 733
7.3 Economic calendar
 bool match(const MqlCalendarValue &v)
   {
      MqlCalendarEvent event;
      if(!CalendarEventById(v.event_id, event)) return false;
      
      // loop through all filter conditions, except for countries, currencies, dates, IDs,
      // which have already been previously used when calling Calendar functions
      for(int j = 0; j < ArrayRange(selectors, 0); ++j)
      {
         long field = 0;
         string text = NULL;
         
         // get the field value from the news or its description
         switch((int)selectors[j][0])
         {
         case CALENDAR_PROPERTY_EVENT_TYPE:
            field = event.type;
            break;
         case CALENDAR_PROPERTY_EVENT_SECTOR:
            field = event.sector;
            break;
         case CALENDAR_PROPERTY_EVENT_TIMEMODE:
            field = event.time_mode;
            break;
         case CALENDAR_PROPERTY_EVENT_IMPORTANCE:
            field = event.importance;
            break;
         case CALENDAR_PROPERTY_EVENT_SOURCE:
            text = event.source_url;
            break;
         case CALENDAR_PROPERTY_EVENT_NAME:
            text = event.name;
            break;
         case CALENDAR_PROPERTY_RECORD_IMPACT:
            field = v.impact_type;
            break;
         case CALENDAR_PROPERTY_RECORD_ACTUAL:
            field = v.actual_value;
            break;
         case CALENDAR_PROPERTY_RECORD_PREVIOUS:
            field = v.prev_value;
            break;
         case CALENDAR_PROPERTY_RECORD_REVISED:
            field = v.revised_prev_value;
            break;
         case CALENDAR_PROPERTY_RECORD_PREVISED: // previous or revised (if any)
            field = v.revised_prev_value != LONG_MIN ? v.revised_prev_value : v.prev_value;
            break;
         case CALENDAR_PROPERTY_RECORD_FORECAST:
            field = v.forecast_value;
            break;

---

## Page 1734

Part 7. Advanced language tools
1 734
7.3 Economic calendar
         ...
         }
         
         // compare value with filter condition
         if(text == NULL) // numeric fields
         {
            switch((IS)selectors[j][2])
            {
            case EQUAL:
               if(!equal(field, selectors[j][1])) return false;
               break;
            case NOT_EQUAL:
               if(equal(field, selectors[j][1])) return false;
               break;
            case GREATER:
               if(!greater(field, selectors[j][1])) return false;
               break;
            case LESS:
               if(greater(field, selectors[j][1])) return false;
               break;
            }
         }
         else // string fields
         {
            const string find = stringCache[(int)selectors[j][1]];
            switch((IS)selectors[j][2])
            {
            case EQUAL:
               if(!equal(text, find)) return false;
               break;
            case NOT_EQUAL:
               if(equal(text, find)) return false;
               break;
            case GREATER:
               if(!greater(text, find)) return false;
               break;
            case LESS:
               if(greater(text, find)) return false;
               break;
            }
         }
      }
      
      return true;
   }
The equal and greater methods almost completely copy those used in our previous developments with
filter classes.
On this, the filtering problem is generally solved, i.e., the MQL program can use the object
CalendarFilter in the following way:

---

## Page 1735

Part 7. Advanced language tools
1 735
7.3 Economic calendar
CalendarFilter f;
f.let()... // a series of calls to the let method to set filtering conditions
MqlCalendarValue records[]; 
if(f.select(records))
{
   ArrayPrint(records);
}
In fact, the select method can do something else important that we left for an independent elective
study.
First, in the resulting list of news, it is desirable to somehow insert a separator (delimiter) between the
past and the future, so that the eye can catch on to it. In theory, this feature is extremely important
for calendars, but for some reason, it is not available in the MetaTrader 5 user interface and on the
mql5.com website. Our implementation is able to insert an empty structure between the past and the
future, which we should visually display (which we will deal with below).
Second, the size of the resulting array can be quite large (especially at the first stages of selecting
settings), and therefore the select method additionally provides the ability to limit the size of the array
(limit). This is done by removing the elements furthest from the current time.
So, the full method prototype looks like this:
bool select(MqlCalendarValue &result[],
   const bool delimiter = false, const int limit = -1);
By default, no delimiter is inserted and the array is not truncated.
A couple of paragraphs above, we mentioned an additional subtask of filtering which is the visualization
of the resulting array. The CalendarFilter class has a special method format, which turns the passed
array of structures MqlCalendarValue &data[] into an array of human-readable strings string &result[].
The code of the method can be found in the attached file CalendarFilter.mqh.
bool format(const MqlCalendarValue &data[],
   const ENUM_CALENDAR_PROPERTY &props[], string &result[],
   const bool padding = false, const bool header = false);
The fields of the MqlCalendarValue that we want to display are specified in the props array. Recall that
the ENUM_CALENDAR_PROPERTY enumeration contains fields from all three dependent calendar
structures so that an MQL program can automatically display not only economic indicators from a
specific event record but also its name, characteristics, country, or currency code. All this is
implemented by the format method.
Each row in the output result array contains a text representation of the value of one of the fields
(number, description, enumeration element). The size of the result array is equal to the product of the
number of structures at the input (in data) and the number of displayed fields (in props). The optional
parameter header allows you to add a row with the names of fields (columns) to the beginning of the
output array. The padding parameter controls the generation of additional spaces in the text so that it
is convenient to display the table in a monospaced font (for example, in a magazine).
The CalendarFilter class has another important public method: update.

---

## Page 1736

Part 7. Advanced language tools
1 736
7.3 Economic calendar
bool update(MqlCalendarValue &result[]);
Its structure almost completely repeats select. However, instead of calling the
CalendarValueHistoryByEvent and CalendarValueHistory functions, the method calls
CalendarValueLastByEvent and CalendarValueLast. The purpose of the method is obvious: it queries the
calendar for recent changes that match the filtering conditions. But for its operation, it requires an ID
of changes. Such a field is indeed defined in the class: the first time it is filled inside the select method.
class CalendarFilter
{
protected:
   ...
   ulong change;
   ...
public:
   bool select(MqlCalendarValue &result[],
      const bool delimiter = false, const int limit = -1)
   {
      ...
      change = 0;
      MqlCalendarValue dummy[];
      CalendarValueLast(change, dummy);
      ...
   }
Some nuances of the CalendarFilter class are still "behind the scenes", but we will address some of
them in the following sections.
Let's test the filter in action: first in a simple script CalendarFilterPrint.mq5 and then in a more
practical indicator CalendarMonitor.mq5.
In the input parameters of the script, you can set the context (country code or currency), time range,
and string for full-text search by event names, as well as limit the size of the resulting news table.
input string Context; // Context (country - 2 characters, currency - 3 characters, empty - no filter)
input ENUM_CALENDAR_SCOPE Scope = SCOPE_MONTH;
input string Text = "farm";
input int Limit = -1;
Given the parameters, a global filter object is created.
CalendarFilter f(Context, TimeCurrent() - Scope, TimeCurrent() + Scope);
Then, in OnStart, we configure a couple of additional constant conditions (medium and high importance
of events) and the presence of a forecast (the field is not equal to LONG_MIN), as well as pass and a
search string to the object.

---

## Page 1737

Part 7. Advanced language tools
1 737
7.3 Economic calendar
void OnStart()
{
   f.let(CALENDAR_IMPORTANCE_LOW, GREATER)
      .let(LONG_MIN, CALENDAR_PROPERTY_RECORD_FORECAST, NOT_EQUAL)
      .let(Text); // with '*' replacement support
      // NB: strings with the character length of 2 or 3 without '*' will be treated
      // as a country or currency code, respectively
Next, the select method is called and the resulting array of MqlCalendarValue structures is formatted
into a table with 9 columns using the format method.
 MqlCalendarValue records[];
   // apply the filter conditions and get the result
   if(f.select(records, true, Limit))
   {
      static const ENUM_CALENDAR_PROPERTY props[] =
      {
         CALENDAR_PROPERTY_RECORD_TIME,
         CALENDAR_PROPERTY_COUNTRY_CURRENCY,
         CALENDAR_PROPERTY_EVENT_NAME,
         CALENDAR_PROPERTY_EVENT_IMPORTANCE,
         CALENDAR_PROPERTY_RECORD_ACTUAL,
         CALENDAR_PROPERTY_RECORD_FORECAST,
         CALENDAR_PROPERTY_RECORD_PREVISED,
         CALENDAR_PROPERTY_RECORD_IMPACT,
         CALENDAR_PROPERTY_EVENT_SECTOR,
      };
      static const int p = ArraySize(props);
      
      // output the formatted result
      string result[];
      if(f.format(records, props, result, true, true))
      {
         for(int i = 0; i < ArraySize(result) / p; ++i)
         {
            Print(SubArrayCombine(result, " | ", i * p, p));
         }
      }
   }
}
The cells of the table are joined into rows and output to the log.
With the default settings (i.e., for all countries and currencies, with the "farm" part in the name of
events of medium and high importance), you can get something like this schedule.

---

## Page 1738

Part 7. Advanced language tools
1 738
7.3 Economic calendar
Selecting calendar records...
country[i]= / ok
calendarValueHistory(temp,from,to,country[i],c)=2372 / ok
Filtering 2372 records
Got 9 records
 
           TIME | CUR  |                          NAME | IMPORTAN  | ACTU  | FORE  | PREV  |   IMPACT | SECT 
2022.06.02 15:15 |  USD | ADP Nonfarm Employment Change |      HIGH |  +128 |  -225 |  +202 | POSITIVE |  JOBS
2022.06.02 15:30 |  USD |      Nonfarm Productivity q/q |  MODERATE |  -7.3 |  -7.5 |  -7.5 | POSITIVE |  JOBS
2022.06.03 15:30 |  USD |              Nonfarm Payrolls |      HIGH |  +390 |   -19 |  +436 | POSITIVE |  JOBS
2022.06.03 15:30 |  USD |      Private Nonfarm Payrolls |  MODERATE |  +333 |    +8 |  +405 | POSITIVE |  JOBS
2022.06.09 08:30 |  EUR |          Nonfarm Payrolls q/q |  MODERATE |  +0.3 |  +0.3 |  +0.3 |       NA |  JOBS
               — |    — |                             — |         — |     — |     — |     — |        — |     —
2022.07.07 15:15 |  USD | ADP Nonfarm Employment Change |      HIGH |  +nan |  -263 |  +128 |       NA |  JOBS
2022.07.08 15:30 |  USD |              Nonfarm Payrolls |      HIGH |  +nan |  -229 |  +390 |       NA |  JOBS
2022.07.08 15:30 |  USD |      Private Nonfarm Payrolls |  MODERATE |  +nan |   +51 |  +333 |       NA |  JOBS
Now let's take a look at the indicator CalendarMonitor.mq5. Its purpose is to display the current
selection of events on the chart to the user in accordance with the specified filters. To visualize the
table, we will use the already familiar scoreboard class (Tableau.mqh, see section Margin calculation for
a future order). The indicator has no buffers and charts.
The input parameters allow you to set the range of the time window (scope), as well as the global
context for the object CalendarFilter, which is either the currency or country code in Context (empty
by default, i.e. without restrictions) or using a boolean flag UseChartCurrencies. It is enabled by default,
and it is recommended to use it in order to automatically receive news of those currencies that make
up the working tool of the chart.
input string Context; // Context (country - 2 chars, currency - 3 chars, empty - all)
input ENUM_CALENDAR_SCOPE Scope = SCOPE_WEEK;
input bool UseChartCurrencies = true;
Additional filters can be applied for event type, sector, and severity.
input ENUM_CALENDAR_EVENT_TYPE_EXT Type = TYPE_ANY;
input ENUM_CALENDAR_EVENT_SECTOR_EXT Sector = SECTOR_ANY;
input ENUM_CALENDAR_EVENT_IMPORTANCE_EXT Importance = IMPORTANCE_MODERATE; // Importance (at least)
Importance sets the lower limit of the selection, not the exact match. Thus, the default value of
IMPORTANCE_MODERATE will capture not only moderate but also high importance.
An attentive reader will notice that unknown enumerations are used here:
ENUM_CALENDAR_EVENT_TYPE_EXT, ENUM_CALENDAR_EVENT_SECTOR_EXT,
ENUM_CALENDAR_EVENT_IMPORTANCE_EXT. They are in the already mentioned file
CalendarDefines.mqh, and they coincide (almost one-to-one) with similar built-in enumerations. The
only difference is that they have added an element meaning "any" value. We need to describe such
enumerations in order to simplify the input of conditions: now the filter for each field is configured using
a drop-down list where you can select either one of the values or turn off the filter. If it weren't for the
added enumeration element, we would have to enter a logical "on/off" flag into the interface for each
field.
In addition, the input parameters allow you to query events by the presence of actual, forecast, and
previous indicators in them, as well as by searching for a text string (Text).

---

## Page 1739

Part 7. Advanced language tools
1 739
7.3 Economic calendar
input string Text;
input ENUM_CALENDAR_HAS_VALUE HasActual = HAS_ANY;
input ENUM_CALENDAR_HAS_VALUE HasForecast = HAS_ANY;
input ENUM_CALENDAR_HAS_VALUE HasPrevious = HAS_ANY;
input ENUM_CALENDAR_HAS_VALUE HasRevised = HAS_ANY;
input int Limit = 30;
Objects CalendarFilter and tableau are described at the global level.
CalendarFilter f(Context);
AutoPtr<Tableau> t;
Please note that the filter is created once, while the table is represented by an autoselector and will be
recreated dynamically depending on the size of the received data.
Filter settings are made in OnInit via consecutive calls of let methods according to the input
parameters.

---

## Page 1740

Part 7. Advanced language tools
1 740
7.3 Economic calendar
int OnInit()
{
   if(!f.isLoaded()) return INIT_FAILED;
   
   if(UseChartCurrencies)
   {
      const string base = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_BASE);
      const string profit = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_PROFIT);
      f.let(base);
      if(base != profit)
      {
         f.let(profit);
      }
   }
   
   if(Type != TYPE_ANY)
   {
      f.let((ENUM_CALENDAR_EVENT_TYPE)Type);
   }
   
   if(Sector != SECTOR_ANY)
   {
      f.let((ENUM_CALENDAR_EVENT_SECTOR)Sector);
   }
   
   if(Importance != IMPORTANCE_ANY)
   {
      f.let((ENUM_CALENDAR_EVENT_IMPORTANCE)(Importance - 1), GREATER);
   }
   
   if(StringLen(Text))
   {
      f.let(Text);
   }
   
   if(HasActual != HAS_ANY)
   {
      f.let(LONG_MIN, CALENDAR_PROPERTY_RECORD_ACTUAL,
         HasActual == HAS_SET ? NOT_EQUAL : EQUAL);
   }
   ...
   
   EventSetTimer(1);
   
   return INIT_SUCCEEDED;
}
At the end, a second timer starts. All work is implemented in OnTimer.

---

## Page 1741

Part 7. Advanced language tools
1 741 
7.3 Economic calendar
void OnTimer()
{
   static const ENUM_CALENDAR_PROPERTY props[] = // table columns
   {
      CALENDAR_PROPERTY_RECORD_TIME,
      CALENDAR_PROPERTY_COUNTRY_CURRENCY,
      CALENDAR_PROPERTY_EVENT_NAME,
      CALENDAR_PROPERTY_EVENT_IMPORTANCE,
      CALENDAR_PROPERTY_RECORD_ACTUAL,
      CALENDAR_PROPERTY_RECORD_FORECAST,
      CALENDAR_PROPERTY_RECORD_PREVISED,
      CALENDAR_PROPERTY_RECORD_IMPACT,
      CALENDAR_PROPERTY_EVENT_SECTOR,
   };
   static const int p = ArraySize(props);
   
   MqlCalendarValue records[];
   
almost one to one   f.let(TimeCurrent() - Scope, TimeCurrent() + Scope); // shift the time window every time
   
   const ulong trackID = f.getChangeID();
   if(trackID) // if the state has already been removed, check for changes
   {
      if(f.update(records)) // request changes by filters
      {
         // if there are changes, notify the user
         string result[];
         f.format(records, props, result);
         for(int i = 0; i < ArraySize(result) / p; ++i)
         {
            Alert(SubArrayCombine(result, " | ", i * p, p));
         }
      // "fall through" further to update the table
      }
      else if(trackID == f.getChangeID())
      {
         return; // calendar without changes
      }
   }
   
   // request a complete set of news by filters
   f.select(records, true, Limit);
   // display the news table on the chart
   string result[];
   f.format(records, props, result, true, true);
   
   if(t[] == NULL || t[].getRows() != ArraySize(records) + 1)
   {
      t = new Tableau("CALT", ArraySize(records) + 1, p,
         TBL_CELL_HEIGHT_AUTO, TBL_CELL_WIDTH_AUTO,

---

## Page 1742

Part 7. Advanced language tools
1 742
7.3 Economic calendar
         Corner, Margins, FontSize, FontName, FontName + " Bold",
         TBL_FLAG_ROW_0_HEADER,
         BackgroundColor, BackgroundTransparency);
   }
   const string hints[] = {};
   t[].fill(result, hints);
}
If we run the indicator on the EURUSD chart with default settings, we can get the following picture.
Filtered and formatted set of news on the chart
7.3.1 1  Transferring calendar database to tester
The calendar is available for MQL programs only online, and therefore testing news trading strategies
poses some difficulties. One of the solutions is to independently create a certain image of the calendar,
that is, the cache, and then use it inside the tester. Cache storage technologies can be different, such
as files or an embedded SQLite database. In this section, we will show an implementation using a file.
In any case, when using the calendar cache, remember that it corresponds to a specific point in time
X. In all "old" events (financial reports) that happened before X, actual values are already set, and in
later ones (in "future", relative to X) there are no actual values, and will not be until a new, more
recent copy of the cache appears. In other words, it makes no sense to test indicators and Expert
Advisors to the right of X. As for those to the left of X, you should avoid looking ahead, that is, do not
read the current indicators until the time of publication of each specific news.
Attention! When requesting calendar data in the terminal, the time of all events is reported taking
into account the current time zone of the server, including a possible correction for "daylight
saving" time (as a rule, this means increasing the timestamps by 1  hour). This synchronizes news
releases with online quote times. However, past clock changes (half a year, a year ago, or more)