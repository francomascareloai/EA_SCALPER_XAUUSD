# MQL5 Book - Part 6 (Pages 1001-1200)

## Page 1001

Part 5. Creating application programs
1 001 
5.8 Graphical objects
The length of the OBJPROP_BMPFILE string must not exceed 63 characters. It can contain not only
the file name but also the path to it. If the string starts with a path separator character (forward slash
'/' or double backslash '\\'), then the file is searched relative to terminal_ data_ directory/MQL5/.
Otherwise, the file is searched relative to the folder where the MQL program is located.
For example, the string "\\Images\\euro.bmp" (or "/Images/euro.bmp") refers to a file in the directory
MQL5/Images/euro.bmp. The standard terminal delivery pack includes the Images folder in the MQL5
directory, and there are a couple of test files euro.bmp and dollar.bmp, so the path is working. If you
specify the string "Images\\euro.bmp" or ("Images/euro.bmp"), then this will imply, for example, for a
script launched from MQL5/Scripts/MQL5Book/, that the Images folder with the euro.bmp file should be
located directly there, that is, the whole path will be MQL5/Scripts/MQL5Book/Images/euro.bmp. There
is no such file in our book, and this would lead to an error loading the image. However, this
arrangement of graphic files next to the program has its advantages: it is easier to control the
assembly, and there is no confusion with mixed pictures of different programs.
The Obj ectBitmap.mq5 script creates a panel with an image on the chart and assigns two images to it:
"\\Images\\dollar.bmp" and "\\Images\\euro.bmp".
#include "ObjectPrefix.mqh"
   
void SetupBitmap(const string button, const int x, const int y,
   const string imageOn, const string imageOff = NULL)
{
   // creating a panel
   const string name = ObjNamePrefix + "Bitmap";
   ObjectCreate(0, name, OBJ_BITMAP_LABEL, 0, 0, 0);
   // set position
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   // include images
   ObjectSetString(0, name, OBJPROP_BMPFILE, 0, imageOn);
   if(imageOff != NULL) ObjectSetString(0, name, OBJPROP_BMPFILE, 1, imageOff);
}
   
void OnStart()
{
   SetupBitmap("image", 100, 100,
      "\\Images\\dollar.bmp", "\\Images\\euro.bmp");
}
As with the result of the script from the previous section, here you can also click on the picture object
and see that it switches from the dollar to the euro image and back.
5.8.23 Cropping (outputting part) of an image
For graphical objects with pictures (OBJ_BITMAP_LABEL and OBJ_BITMAP), MQL5 allows you to enable
the display of a part of the image specified by the property OBJPROP_BMPFILE. To do this, you need to
set the size of the object (OBJPROP_XSIZE and OBJPROP_YSIZE) smaller than the image size and set
the coordinates of the upper left corner of the visible rectangular fragment using the integer properties
OBJPROP_XOFFSET and OBJPROP_YOFFSET. These two properties set, respectively, the indent along
X and Y in pixels from the left and top borders of the original image.

---

## Page 1002

Part 5. Creating application programs
1 002
5.8 Graphical objects
Outputting part of an image to an object
Typically, a similar technique using part of a large image is used for toolbar icons (sets of buttons,
menus, etc.): a single file with all the icons provides more efficient resource consumption than many
small files with individual icons.
The test script Obj ectBitmapOffset.mq5 creates several panels with pictures (OBJ_BITMAP_LABEL), and
for all of them the same graphic file is specified in the OBJPROP_BMPFILE property. However, due to
the OBJPROP_XOFFSET and OBJPROP_YOFFSET properties, all objects display different parts of the
image.

---

## Page 1003

Part 5. Creating application programs
1 003
5.8 Graphical objects
void SetupBitmap(const int i, const int x, const int y, const int size,
   const string imageOn, const string imageOff = NULL)
{
   // create an object
   const string name = ObjNamePrefix + "Tool-" + (string)i;
   ObjectCreate(0, name, OBJ_BITMAP_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
   // position and size
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, size);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, size);
   // offset in the original image, according to which the i-th fragment is read
   ObjectSetInteger(0, name, OBJPROP_XOFFSET, i * size);
   ObjectSetInteger(0, name, OBJPROP_YOFFSET, 0);
   // generic image (file)
   ObjectSetString(0, name, OBJPROP_BMPFILE, imageOn);
}
   
void OnStart()
{
   const int icon = 46; // size of one icon
   for(int i = 0; i < 7; ++i) // loop through the icons in the file
   {
      SetupBitmap(i, 10, 10 + i * icon, icon,
         "\\Files\\MQL5Book\\icons-322-46.bmp");
   }
}
The original image contains several small icons 46 x 46 pixels each. The script "cuts" them out one by
one and places them vertically at the right edge of the window.
The following shows a generic file (/Files/MQL5Book/icons-322-46.bmp), and what happened on the
chart.
BMP file with icons

---

## Page 1004

Part 5. Creating application programs
1 004
5.8 Graphical objects
Button objects with icons on the chart
5.8.24 Input field properties: text alignment and read-only
For objects of type OBJ_EDIT (input field), an MQL program can set two specific properties defined
using the Obj ectSetInteger/Obj ectGetInteger functions.
Identifier
Description
Value type
OBJPROP_ALIGN
Horizontal text alignment
ENUM_ALIGN_MODE
OBJPROP_READONLY
Ability to edit text
bool
The ENUM_ALIGN_MODE enumeration contains the following members.
Identifier
Description
ALIGN_LEFT
Left alignment
ALIGN_CENTER
Center alignment
ALIGN_RIGHT
Right alignment
Note that, unlike OBJ_TEXT and OBJ_LABEL objects, the input field does not automatically resize itself
to fit the text entered, so for long strings, you may need to explicitly set the OBJPROP_XSIZE property.
In the edit mode, horizontal text scrolling works inside the input field.

---

## Page 1005

Part 5. Creating application programs
1 005
5.8 Graphical objects
The Obj ectEdit.mq5 script creates four OBJ_EDIT objects: three of them are editable with different text
alignment methods and the fourth one is in the read-only mode.
#include "ObjectPrefix.mqh"
   
void SetupEdit(const int x, const int y, const int dx, const int dy,
   const ENUM_ALIGN_MODE alignment = ALIGN_LEFT, const bool readonly = false)
{
   // create an object with a description of the properties
   const string props = EnumToString(alignment)
      + (readonly ? " read-only" : " editable");
   const string name = ObjNamePrefix + "Edit" + props;
   ObjectCreate(0, name, OBJ_EDIT, 0, 0, 0);
   // position and size
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, dx);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, dy);
   // specific properties of input fields
   ObjectSetInteger(0, name, OBJPROP_ALIGN, alignment);
   ObjectSetInteger(0, name, OBJPROP_READONLY, readonly);
   // colors (different depending on editability)
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, clrWhite);
   ObjectSetInteger(0, name, OBJPROP_COLOR, readonly ? clrRed : clrBlue);
   // content
   ObjectSetString(0, name, OBJPROP_TEXT, props);
   // tooltip for editable
   ObjectSetString(0, name, OBJPROP_TOOLTIP,
      (readonly ? "\n" : "Click me to edit"));
}
   
void OnStart()
{
   SetupEdit(100, 100, 200, 20);
   SetupEdit(100, 120, 200, 20, ALIGN_RIGHT);
   SetupEdit(100, 140, 200, 20, ALIGN_CENTER);
   SetupEdit(100, 160, 200, 20, ALIGN_CENTER, true);
}
The result of the script is shown in the image below.

---

## Page 1006

Part 5. Creating application programs
1 006
5.8 Graphical objects
Input fields in different modes
You can click on any editable field and change its content.
5.8.25 Standard deviation channel width
The standard deviation channel OBJ_STDDEVCHANNEL has a special property that defines the channel
width as a multiplier for the standard (root mean square) deviation. The property is called
OBJPROP_DEVIATION and can take positive real values (double). By default, it equals 1 .0.
We have already seen an example of its use in the Obj ectRays.mq5 script in the section on Ray
properties for objects with straight lines.
5.8.26 Setting levels in level objects
Some graphical objects are built using multiple levels (repetitive lines). These include:
• Andrews pitchfork OBJ_PITCHFORK
• Fibonacci tools:
•  OBJ_FIBO levels
•  Time zones OBJ_FIBOTIMES
•  Fan OBJ_FIBOFAN
•  Arcs OBJ_FIBOARC
•  Channel OBJ_FIBOCHANNEL
•  Extension OBJ_EXPANSION 

---

## Page 1007

Part 5. Creating application programs
1 007
5.8 Graphical objects
MQL5 allows you to set level properties for such objects. The properties include their number, colors,
values, and labels.
Identifier
Description
Type
OBJPROP_LEVELS
Number of levels
int
OBJPROP_LEVELCOLOR
Level line color
color
OBJPROP_LEVELSTYLE
Level line style
ENUM_LINE_STYLE 
OBJPROP_LEVELWIDTH
Level line width
int
OBJPROP_LEVELTEXT
Level description
string
OBJPROP_LEVELVALUE
Level value
double
When calling the Obj ectGet and Obj ectSet functions for all properties except OBJPROP_LEVELS, it is
required to provide an additional modifier parameter with the number of a specific level.
As an example, let's consider the indicator Obj ectHighLowFibo.mq5. For a given range of bars, which is
defined as the number of the last bar (baroffset) and the number of bars (BarCount) to the left of it, the
indicator finds the High and Low prices and then creates the OBJ_FIBO object for these points. As new
bars form, Fibonacci levels will shift to the right to more current prices.

---

## Page 1008

Part 5. Creating application programs
1 008
5.8 Graphical objects
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0
   
#include <MQL5Book/ColorMix.mqh>
   
input int BarOffset = 0;
input int BarCount = 24;
   
const string Prefix = "HighLowFibo-";
   
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
   static datetime now = 0;
   if(now != iTime(NULL, 0, 0))
   {
      const int hh = iHighest(NULL, 0, MODE_HIGH, BarCount, BarOffset);
      const int ll = iLowest(NULL, 0, MODE_LOW, BarCount, BarOffset);
   
      datetime t[2] = {iTime(NULL, 0, hh), iTime(NULL, 0, ll)};
      double p[2] = {iHigh(NULL, 0, hh), iLow(NULL, 0, ll)};
    
      DrawFibo(Prefix + "Fibo", t, p, clrGray);
   
      now = iTime(NULL, 0, 0);
   }
   return rates_total;
}
The direct setting of the object is done in the auxiliary function DrawFibo. In it, in particular, the levels
are painted in rainbow colors, and their style and thickness are determined based on whether the
corresponding values are "round" (without a fractional part).

---

## Page 1009

Part 5. Creating application programs
1 009
5.8 Graphical objects
bool DrawFibo(const string name, const datetime &t[], const double &p[],
   const color clr)
{
   if(ArraySize(t) != ArraySize(p)) return false;
   
   ObjectCreate(0, name, OBJ_FIBO, 0, 0, 0);
   // anchor points
   for(int i = 0; i < ArraySize(t); ++i)
   {
      ObjectSetInteger(0, name, OBJPROP_TIME, i, t[i]);
      ObjectSetDouble(0, name, OBJPROP_PRICE, i, p[i]);
   }
   // general settings
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, true);
   // level settings
   const int n = (int)ObjectGetInteger(0, name, OBJPROP_LEVELS);
   for(int i = 0; i < n; ++i)
   {
      const color gradient = ColorMix::RotateColors(ColorMix::HSVtoRGB(0),
         ColorMix::HSVtoRGB(359), n, i);
      ObjectSetInteger(0, name, OBJPROP_LEVELCOLOR, i, gradient);
      const double level = ObjectGetDouble(0, name, OBJPROP_LEVELVALUE, i);
      if(level - (int)level > DBL_EPSILON * level)
      {
         ObjectSetInteger(0, name, OBJPROP_LEVELSTYLE, i, STYLE_DOT);
         ObjectSetInteger(0, name, OBJPROP_LEVELWIDTH, i, 1);
      }
      else
      {
         ObjectSetInteger(0, name, OBJPROP_LEVELSTYLE, i, STYLE_SOLID);
         ObjectSetInteger(0, name, OBJPROP_LEVELWIDTH, i, 2);
      }
   }
   
   return true;
}
Here is a variant of what an object might look like on a chart.

---

## Page 1010

Part 5. Creating application programs
1 01 0
5.8 Graphical objects
Fibonacci object with level settings
5.8.27 Additional properties of Gann, Fibonacci, and Elliot objects
Gann, Fibonacci, and Elliot objects have specific, unique properties. Depending on the property type,
use the Obj ectGetInteger/Obj ectSetInteger or Obj ectGetDouble/Obj ectSetDouble functions.
Identifier
Description
Type
OBJPROP_DIRECTION
Gann object trend (Fan
OBJ_GANNFAN and Grid
OBJ_GANNGRID)
ENUM_GANN_DIRECTION
OBJPROP_DEGREE
Elliot wave degree levels
ENUM_ELLIOT_WAVE_DEGREE
OBJPROP_DRAWLINES
Displaying lines for the Elliot
wave levels
bool
OBJPROP_SCALE
Scale in points per bar (property
of Gann objects and the
Fibonacci Arcs object)
double
OBJPROP_ELLIPSE
Full ellipse display for the
Fibonacci Arcs object
(OBJ_FIBOARC)
bool
The ENUM_GANN_DIRECTION enumeration has the following members:

---

## Page 1011

Part 5. Creating application programs
1 01 1 
5.8 Graphical objects
Constant
Trend direction
GANN_UP_TREND
Ascending lines
GANN_DOWN_TREND
Descending lines
ENUM_ELLIOT_WAVE_DEGREE is used to set the size (labeling method) of Elliot waves.
Constant
Description
ELLIOTT_GRAND_SUPERCYCLE
Grand Supercycle
ELLIOTT_SUPERCYCLE
Supercycle
ELLIOTT_CYCLE
Cycle
ELLIOTT_PRIMARY
Primary cycle
ELLIOTT_INTERMEDIATE
Intermediate link
ELLIOTT_MINOR
Minor cycle
ELLIOTT_MINUTE
Minute
ELLIOTT_MINUETTE
Minuette
ELLIOTT_SUBMINUETTE
Subminuette
5.8.28 Chart object
The chart object OBJ_CHART allows you to create thumbnails of other charts inside the chart for other
instruments and timeframes. Chart objects are included in the general chart list, which we obtained
programmatically using the ChartFirst and ChartNext functions. As mentioned in the section on
Checking the main window status, the special chart property CHART_IS_OBJECT allows you to find out
by identifier whether it is a full-fledged window or a chart object. In the latter case, calling
ChartGetInteger(id, CHART_ IS_ OBJECT) will return true.
The chart object has a set of properties specific only to it.
Identifier
Description
Type
OBJPROP_CHART_ID
Chart ID (r/o)
long
OBJPROP_PERIOD
Chart Period
ENUM_TIMEFRAMES
OBJPROP_DATE_SCALE
Show the time scale
bool
OBJPROP_PRICE_SCALE
Show the price scale
bool
OBJPROP_CHART_SCALE
Scale (value in the range 0 - 5)
int
OBJPROP_SYMBOL
Symbol
string

---

## Page 1012

Part 5. Creating application programs
1 01 2
5.8 Graphical objects
The identifier obtained through the OBJPROP_CHART_ID property allows you to manage the object like
a regular chart using the functions from the chapter Working with charts. However, there are some
limitations:
• The object cannot be closed with ChartClose
• It is not possible to change the symbol/period in the object using the CartSetSymbolPeriod function
• Properties CHART_SCALE, CHART_BRING_TO_TOP,CHART_SHOW_DATE_SCALE and
CHART_SHOW_PRICE_SCALE are not modified in the object.
By default, all properties (except OBJPROP_CHART_ID) are equal to the corresponding properties of
the current window.
The demonstration of chart objects is implemented as a bufferless indicator Obj ectChart.mq5. It
creates a subwindow with two chart objects for the same symbol as the current chart but with adjacent
timeframes above and below the current one.
Objects snap to the upper right corner of the subwindow and have the same predefined sizes:
#define SUBCHART_HEIGHT 150
#define SUBCHART_WIDTH  200
Of course, the height of the subwindow must match the height of the objects, until we can respond
adaptively to resize events.
#property indicator_separate_window
#property indicator_height SUBCHART_HEIGHT
#property indicator_buffers 0
#property indicator_plots   0
One mini-chart is configured in the SetupSubChart function, which takes the number of the object, its
dimensions, and the required timeframe as inputs. The result of SetupSubChart is the identifier of the
chart object, which we just output into the log for reference.
void OnInit()
{
   Print(SetupSubChart(0, SUBCHART_WIDTH, SUBCHART_HEIGHT, PeriodUp(_Period)));
   Print(SetupSubChart(1, SUBCHART_WIDTH, SUBCHART_HEIGHT, PeriodDown(_Period)));
}
Macros PeriodUp and PeriodDown use the helper function PeriodRelative.

---

## Page 1013

Part 5. Creating application programs
1 01 3
5.8 Graphical objects
#define PeriodUp(P) PeriodRelative(P, +1)
#define PeriodDown(P) PeriodRelative(P, -1)
   
ENUM_TIMEFRAMES PeriodRelative(const ENUM_TIMEFRAMES tf, const int step)
{
   static const ENUM_TIMEFRAMES stdtfs[] =
   {
      PERIOD_M1,  // =1 (1)
      PERIOD_M2,  // =2 (2)
      ...
      PERIOD_W1,  // =32769 (8001)
      PERIOD_MN1, // =49153 (C001)
   };
   const int x = ArrayBsearch(stdtfs, tf == PERIOD_CURRENT ? _Period : tf);
   const int needle = x + step;
   if(needle >= 0 && needle < ArraySize(stdtfs))
   {
      return stdtfs[needle];
   }
   return tf;
}
Here is the main working function SetupSubChart.

---

## Page 1014

Part 5. Creating application programs
1 01 4
5.8 Graphical objects
long SetupSubChart(const int n, const int dx, const int dy,
   ENUM_TIMEFRAMES tf = PERIOD_CURRENT, const string symbol = NULL)
{
   // create an object
   const string name = Prefix + "Chart-"
      + (symbol == NULL ? _Symbol : symbol) + PeriodToString(tf);
   ObjectCreate(0, name, OBJ_CHART, ChartWindowFind(), 0, 0);
   
   // anchor to the top right corner of the subwindow
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_RIGHT_UPPER);
   // position and size
   ObjectSetInteger(0, name, OBJPROP_XSIZE, dx);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, dy);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, (n + 1) * dx);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, 0);
   
   // specific chart settings
   if(symbol != NULL)
   {
      ObjectSetString(0, name, OBJPROP_SYMBOL, symbol);
   }
   
   if(tf != PERIOD_CURRENT)
   {
      ObjectSetInteger(0, name, OBJPROP_PERIOD, tf);
   }
   // disable the display of lines
   ObjectSetInteger(0, name, OBJPROP_DATE_SCALE, false);
   ObjectSetInteger(0, name, OBJPROP_PRICE_SCALE, false);
   // add the MA indicator to the object by its id just for demo
   const long id = ObjectGetInteger(0, name, OBJPROP_CHART_ID);
   ChartIndicatorAdd(id, 0, iMA(NULL, tf, 10, 0, MODE_EMA, PRICE_CLOSE));
   return id;
}
For a chart object, the anchor point is always fixed in the upper left corner of the object, so when
anchoring to the right corner of the window, you need to add the width of the object (this is done by +1 
in the expression(n+1 )*dx for OBJPROP_XDISTANCE).
The following screenshot shows the result of the indicator on the XAUUSD,H1  chart.

---

## Page 1015

Part 5. Creating application programs
1 01 5
5.8 Graphical objects
Two chart objects in the indicator subwindow
As we can see, the mini-charts display the M30 and H2 timeframes.
It is important to note that you can add indicators to chart objects and apply tpl templates, including
those with Expert Advisors. However, you cannot create objects inside chart objects.
When the chart object is hidden due to disabled visualization on the current timeframe or on all
timeframes, the CHART_WINDOW_IS_VISIBLE property for the internal chart still returns true.
5.8.29 Moving objects
To move objects in time/price coordinates, you can use not only the Obj ectSet functions of property
change but also the special function Obj ectMove, which changes the coordinates of the specified anchor
point of the object.
bool ObjectMove(long chartId, const string name, int index, datetime time, double price)
The chartId parameter sets the chart ID (0 is for the current chart). The name of the object is passed
in the name parameter. The anchor point index and coordinates are specified in the index, time, and
price parameters, respectively.
The function uses an asynchronous call, that is, it sends a command to the chart's event queue and
does not wait for the movement itself.
The function returns an indication of whether the command was successfully queued (in this case, the
result is true). The actual position of the object should be learned using calls to the Obj ectGet
functions.

---

## Page 1016

Part 5. Creating application programs
1 01 6
5.8 Graphical objects
In the indicator Obj ectHighLowFibo.mq5, we modify the DrawFibo function in such a way as to enable
Obj ectMove. Instead of two calls to Obj ectSet functions in the loop through the anchor points, we now
have one Obj ectMove call:
bool DrawFibo(const string name, const datetime &t[], const double &p[],
   const color clr)
{
   ...
   for(int i = 0; i < ArraySize(t); ++i)
   {
      // was:
      // ObjectSetInteger(0, name, OBJPROP_TIME, i, t[i]);
      // ObjectSetDouble(0, name, OBJPROP_PRICE, i, p[i]);
      // became:
      ObjectMove(0, name, i, t[i], p[i]);
   }
   ...
}
It makes sense to apply the Obj ectMove function where both coordinates of the anchor point change. In
some cases, only one coordinate has an effect (for example, in the channels of standard deviation and
linear regression at the anchor points, only the start and end dates/times are important, and the
channels calculate the price value at these points automatically). In such cases, a single call of the
Obj ectSet function is more appropriate than Obj ectMove.
5.8.30 Getting time or price at the specified line points
Many graphical objects include one or more straight lines. MQL5 allows you to interpolate and
extrapolate points on these lines and get another coordinate from one coordinate, for example, price by
time or time by price.
Interpolation is always available: it works "inside" the object, i.e., between anchor points. Extrapolation
outside an object is possible only if the ray property in the corresponding direction is enabled for it (see
Ray properties for objects with straight lines).
The Obj ectGetValueByTime function returns the price value for the specified time. The
Obj ectGetTimeByValue function returns the time value for the specified price
double ObjectGetValueByTime(long chartId, const string name, datetime time, int line)
datetime ObjectGetTimeByValue(long chartId, const string name, double value, int line)
Calculations are made for an object named name on the chart with chartId. The time and value
parameters specify a known coordinate for which the unknown should be calculated. Since an object
can have several lines, several values will correspond to one coordinate, and therefore it is necessary to
specify the line number in the line parameter.
The function returns the price or time value for the projection of the point with the specified initial
coordinate relative to the line.
In case of an error, 0 will be returned, and the error code will be written to _ LastError. For example,
attempting to extrapolate a line value with the beam property disabled generates an
OBJECT_GETVALUE_FAILED (4205) error.

---

## Page 1017

Part 5. Creating application programs
1 01 7
5.8 Graphical objects
The functions are applicable to the following objects:
• Trendline (OBJ_TREND)
• Trendline by angle (OBJ_TRENDBYANGLE)
• Gann line (OBJ_GANNLINE)
• Equidistant channel (OBJ_CHANNEL), 2 lines
• Linear regression channel (OBJ_REGRESSION); 3 lines
• Standard deviation channel (OBJ_STDDEVCHANNEL); 3 lines
• Arrow line (OBJ_ARROWED_LINE)
Let's check the operation of the function using a bufferless indicator Obj ectChannels.mq5. It creates
two objects with standard deviation and linear regression channels, after which it requests and displays
in the comment the price of the upper and lower lines on future bars. For the standard deviation
channel, the OBJPROP_RAY_RIGHT property is enabled, but for the regression channel, it is not
(intentionally). In this regard, no values will be received from the second channel, and zeros are always
displayed on the screen for it.
As new bars form, the channels will automatically move to the right. The length of the channels is set in
the input parameter WorkPeriod (1 0 bars by default).
input int WorkPeriod = 10;
   
const string Prefix = "ObjChnl-";
const string ObjStdDev = Prefix + "StdDev";
const string ObjRegr = Prefix + "Regr";
   
void OnInit()
{
   CreateObjects();
   UpdateObjects();
}
The CreateObj ects function creates 2 channels and makes initial settings for them.
void CreateObjects()
{
   ObjectCreate(0, ObjStdDev, OBJ_STDDEVCHANNEL, 0, 0, 0);
   ObjectCreate(0, ObjRegr, OBJ_REGRESSION, 0, 0, 0);
   ObjectSetInteger(0, ObjStdDev, OBJPROP_COLOR, clrBlue);
   ObjectSetInteger(0, ObjStdDev, OBJPROP_RAY_RIGHT, true);
   ObjectSetInteger(0, ObjRegr, OBJPROP_COLOR, clrRed);
   // NB: ray is not enabled for the regression channel (intentionally)
}
The UpdateObj ects function moves channels to the last WorkPeriod bars.

---

## Page 1018

Part 5. Creating application programs
1 01 8
5.8 Graphical objects
void UpdateObjects()
{
   const datetime t0 = iTime(NULL, 0, WorkPeriod);
   const datetime t1 = iTime(NULL, 0, 0);
   
   // we don't use ObjectMove because channels work
   // only with time coordinate (price is calculated automatically)
   ObjectSetInteger(0, ObjStdDev, OBJPROP_TIME, 0, t0);
   ObjectSetInteger(0, ObjStdDev, OBJPROP_TIME, 1, t1);
   ObjectSetInteger(0, ObjRegr, OBJPROP_TIME, 0, t0);
   ObjectSetInteger(0, ObjRegr, OBJPROP_TIME, 1, t1);
}
In the OnCalculate handler, we update the position of the channels on new bars, and on each tick, we
call DisplayObj ectData to get price extrapolation and display it as a comment.
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
   static datetime now = 0;
   if(now != iTime(NULL, 0, 0))
   {
      UpdateObjects();
      now = iTime(NULL, 0, 0);
   }
   
   DisplayObjectData();
   
   return rates_total;
}
In the DisplayObj ectData function, we will find prices at anchor points on the middle line
(OBJPROP_PRICE). Also, using Obj ectGetValueByTime, we will request price values for the upper and
lower channel lines through WorkPeriod bars in the future.

---

## Page 1019

Part 5. Creating application programs
1 01 9
5.8 Graphical objects
void DisplayObjectData()
{
   const double p0 = ObjectGetDouble(0, ObjStdDev, OBJPROP_PRICE, 0);
   const double p1 = ObjectGetDouble(0, ObjStdDev, OBJPROP_PRICE, 1);
   
   // the following equalities are always true due to the channel calculation algorithm:
   // - the middle lines of both channels are the same,
   // - anchor points always lie on the middle line,
   // ObjectGetValueByTime(0, ObjStdDev, iTime(NULL, 0, 0), 0) == p1
   // ObjectGetValueByTime(0, ObjRegr, iTime(NULL, 0, 0), 0) == p1
   
   // trying to extrapolate future prices from the upper and lower lines
   const double d1 = ObjectGetValueByTime(0, ObjStdDev, iTime(NULL, 0, 0)
      + WorkPeriod * PeriodSeconds(), 1);
   const double d2 = ObjectGetValueByTime(0, ObjStdDev, iTime(NULL, 0, 0)
      + WorkPeriod * PeriodSeconds(), 2);
   
   const double r1 = ObjectGetValueByTime(0, ObjRegr, iTime(NULL, 0, 0)
      + WorkPeriod * PeriodSeconds(), 1);
   const double r2 = ObjectGetValueByTime(0, ObjRegr, iTime(NULL, 0, 0)
      + WorkPeriod * PeriodSeconds(), 2);
   
   // display all received prices in a comment
   Comment(StringFormat("%.*f %.*f\ndev: up=%.*f dn=%.*f\nreg: up=%.*f dn=%.*f",
      _Digits, p0, _Digits, p1,
      _Digits, d1, _Digits, d2,
      _Digits, r1, _Digits, r2));
}
It is important to note that due to the fact that the ray property is not enabled for the regression
channel, it always gives zeros in the future (although if we asked for prices within the channel's time
period, we would get the correct values).

---

## Page 1020

Part 5. Creating application programs
1 020
5.8 Graphical objects
Channels and price values at the points of their lines
Here, for channels that are 1 0 bars long, the extrapolation is also done on 1 0 bars ahead, which gives
the future values shown in the line with "dev:", approximately corresponding to the right border of the
window.
5.9 Interactive events on charts
MetaTrader 5 charts not only provide a visual representation of data and are the execution environment
for MQL programs, but also support the mechanism of interactive events, which allows programs to
respond to the actions of the user and other programs. This is done by a special event type –
OnChartEvent – which we already discussed in the Overview of event handling functions.
Any indicator or Expert Advisor can receive such events provided that the event processing function of
the same name with a predefined signature is described in the code. In some of the indicator examples
that we considered earlier, we have already had to take advantage of this opportunity. In this chapter,
we will look at the event system in detail.
The OnChartEvent event is generated by the client terminal during the following chart manipulations
performed by the user:
• Changing the chart size or settings
• Keystrokes when the chart window is in focus
• Mouse cursor movement
• Mouse clicks on the chart
• Mouse clicks on graphical objects
• Creating a graphical object

---

## Page 1021

Part 5. Creating application programs
1 021 
5.9 Interactive events on charts
• Deleting a graphical object
• Moving a graphical object with the mouse
• Finishing to edit the test in the input field of the OBJ_EDIT object
The MQL program receives the listed events only from the chart on which it is running. Like other event
types, they are added to a queue. All events are then processed one by one in the order of arrival. If
there is already an OnChartEvent event of a particular type in the MQL program queue or it is being
processed, a new event of the same type is not queued (discarded).
Some event types are always active, while others are disabled by default and must be explicitly enabled
by setting the appropriate chart properties using the ChartSetInteger call. Such disabled events
include, in particular, mouse movements and mouse wheel scrolling. All of them are characterized by
the fact that they can generate massive event streams, and in order to save resources, it is
recommended to enable them only when necessary.
In addition to standard events, there is the concept of "custom events". The meaning and content of
parameters for such events are assigned and interpreted by the MQL program itself (one or several, if
we are talking about the interaction of a complex of programs). An MQL program can send "user
events" to a chart (including another one) using the function EventChartCustom. Such events are also
handled by the OnChartEvent function.
If there are several MQL programs on the chart with the OnChartEvent handler, they will all receive the
same stream of events.
All MQL programs run in threads other than the main thread of the application. The main terminal
thread is responsible for processing all Windows system messages, and as a result of this
processing, in turn, it generates Windows messages for its own application. For example, dragging a
chart with the mouse generates several WM_MOUSE_MOVE system messages (in terms of the
Windows API) for subsequent drawing of the application window, and also sends internal messages
to Expert Advisors and indicators launched on this chart. In this case, a situation may arise that the
main thread of the application has not yet managed to process the system message about
redrawing the WM_PAINT window (and therefore has not yet changed the appearance of the chart),
and the Expert Advisor or indicator has already received an event about moving the mouse cursor.
Then the chart property CHART_FIRST_VISIBLE_BAR will be changed only after the chart is drawn.
Since of the two types of interactive MQL programs, we have studied only indicators so far, all the
examples in this chapter will be built on the basis of indicators. The second type, Expert Advisors, will
be described in the next Part of the book. However, the principles of working with events in them
completely coincide with those presented here.
5.9.1  Event handling function OnChartEvent
An indicator or Expert Advisor can receive interactive events from the terminal if the code contains the
OnChartEvent function with the following prototype.
void OnChartEvent(const int event, const long &lparam, const double &dparam, const string &sparam)
This function will be called by the terminal in response to user actions or in case of generating a "user
event" using EventChartCustom.
In the event parameter, the event identifier (its type) is passed as one of the values of the
ENUM_CHART_EVENT enumeration (see the table).

---

## Page 1022

Part 5. Creating application programs
1 022
5.9 Interactive events on charts
Identifier
Description
CHARTEVENT_KEYDOWN
Keyboard action
CHARTEVENT_MOUSE_MOVE
Moving the mouse and clicking mouse buttons (if the
CHART_EVENT_MOUSE_MOVE property is set for the chart)
CHARTEVENT_MOUSE_WHEEL
Clicking or scrolling the mouse wheel (if the
CHART_EVENT_MOUSE_WHEEL property is set for the chart)
CHARTEVENT_CLICK
Mouse-click on the chart
CHARTEVENT_OBJECT_CREATE
Creating a graphical object (if the
CHART_EVENT_OBJECT_CREATE property is set for the
chart)
CHARTEVENT_OBJECT_CHANGE
Modifying a graphical object through the properties dialog
CHARTEVENT_OBJECT_DELETE
Deleting a graphical object (if the
CHART_EVENT_OBJECT_DELETE property is set for the
chart)
CHARTEVENT_OBJECT_CLICK
Mouse-click on a graphical object
CHARTEVENT_OBJECT_DRAG
Dragging a graphical object
CHARTEVENT_OBJECT_ENDEDIT
Finishing text editing in the "input field" graphical object
CHARTEVENT_CHART_CHANGE
Changing the chart dimensions or properties (via the
properties dialog, toolbar, or context menu)
CHARTEVENT_CUSTOM
The starting number of the event from the custom event
range
CHARTEVENT_CUSTOM_LAST
The end number of the event from the custom event range
The lparam, dparam, and sparam parameters are used differently depending on the event type. In
general, we can say that they contain additional data necessary to process a particular event. The
following sections provide details for each type.
Attention! The OnChartEvent function is called only for indicators and Expert Advisors that are
directly plotted on the chart. If any indicator is created programmatically using iCustom or
IndicatorCreate, the OnChartEvent events will not be translated to it.
In addition, the OnChartEvent handler is not called in the tester, even in visual mode.
For the first demonstration of the OnChartEvent handler, let's consider a bufferless indicator
EventAll.mq5 which intercepts and logs all events.

---

## Page 1023

Part 5. Creating application programs
1 023
5.9 Interactive events on charts
void OnChartEvent(const int id,
   const long &lparam, const double &dparam, const string &sparam)
{
   ENUM_CHART_EVENT evt = (ENUM_CHART_EVENT)id;
   PrintFormat("%s %lld %f '%s'", EnumToString(evt), lparam, dparam, sparam);
}
By default, all types of events can be generated on the chart, except for four mass events, which, as
indicated in the table above, are enabled by the special properties of the chart. In the next section, we
will supplement the indicator with settings to include certain types according to preferences.
Run the indicator on a chart with existing objects or create objects while the indicator is running.
Change the size or settings of the chart, make mouse clicks, and edit the properties of objects. The
following entries will appear in the log.
CHARTEVENT_CHART_CHANGE 0 0.000000 ''
CHARTEVENT_CLICK 149 144.000000 ''
CHARTEVENT_OBJECT_CLICK 112 105.000000 'Daily Rectangle 53404'
CHARTEVENT_CLICK 112 105.000000 ''
CHARTEVENT_KEYDOWN 46 1.000000 '339'
CHARTEVENT_CLICK 13 252.000000 ''
CHARTEVENT_OBJECT_DRAG 0 0.000000 'Daily Button 61349'
CHARTEVENT_OBJECT_CLICK 145 104.000000 'Daily Button 61349'
CHARTEVENT_CLICK 145 104.000000 ''
CHARTEVENT_CHART_CHANGE 0 0.000000 ''
CHARTEVENT_OBJECT_DRAG 0 0.000000 'Daily Vertical Line 22641'
CHARTEVENT_OBJECT_DRAG 0 0.000000 'Daily Vertical Line 22641'
CHARTEVENT_OBJECT_CLICK 177 206.000000 'Daily Vertical Line 22641'
CHARTEVENT_CLICK 177 206.000000 ''
CHARTEVENT_OBJECT_CHANGE 0 0.000000 'Daily Rectangle 37930'
CHARTEVENT_CHART_CHANGE 0 0.000000 ''
CHARTEVENT_CLICK 152 118.000000 ''
Here we see events of various types, the meanings of their parameters will become clear after reading
the following sections.
5.9.2 Event-related chart properties
Four types of events are capable of generating a lot of messages and therefore are disabled by default.
To activate or disable them later, set the appropriate chart properties using the ChartSetInteger
function. All properties are of Boolean type: true means enabled, and false means disabled.

---

## Page 1024

Part 5. Creating application programs
1 024
5.9 Interactive events on charts
Identifier
Description
CHART_EVENT_MOUSE_WHEEL
Sending CHARTEVENT_MOUSE_WHEEL messages about
mouse wheel events to the chart
CHART_EVENT_MOUSE_MOVE
Sending CHARTEVENT_MOUSE_MOVE messages about
mouse movements to the chart
CHART_EVENT_OBJECT_CREATE
Sending CHARTEVENT_OBJECT_CREATE messages about the
creation of graphical objects to the chart
CHART_EVENT_OBJECT_DELETE
Sending CHARTEVENT_OBJECT_DELETE messages about the
deletion of graphical objects to the chart
If any MQL program changes one of these properties, it affects all other programs running on the same
chart and remains in effect even after the original program terminates.
By default, all properties have the false value.
Let's complement the EventAll.mq5 indicator from the previous section with four input variables that
allow you to enable any of these types of events (in addition to the rest that cannot be disabled). In
addition, we will describe four auxiliary variables in order to be able to restore the chart settings after
deleting the indicator.
input bool ShowMouseMove = false;
input bool ShowMouseWheel = false;
input bool ShowObjectCreate = false;
input bool ShowObjectDelete = false;
   
bool mouseMove, mouseWheel, objectCreate, objectDelete;
At startup, remember the current values of the properties and then apply the settings selected by the
user.
void OnInit()
{
   mouseMove = PRTF(ChartGetInteger(0, CHART_EVENT_MOUSE_MOVE));
   mouseWheel = PRTF(ChartGetInteger(0, CHART_EVENT_MOUSE_WHEEL));
   objectCreate = PRTF(ChartGetInteger(0, CHART_EVENT_OBJECT_CREATE));
   objectDelete = PRTF(ChartGetInteger(0, CHART_EVENT_OBJECT_DELETE));
   
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, ShowMouseMove);
   ChartSetInteger(0, CHART_EVENT_MOUSE_WHEEL, ShowMouseWheel);
   ChartSetInteger(0, CHART_EVENT_OBJECT_CREATE, ShowObjectCreate);
   ChartSetInteger(0, CHART_EVENT_OBJECT_DELETE, ShowObjectDelete);
}
Properties are restored in the OnDeinit handler.

---

## Page 1025

Part 5. Creating application programs
1 025
5.9 Interactive events on charts
void OnDeinit(const int)
{
   ChartSetInteger(0, CHART_EVENT_MOUSE_MOVE, mouseMove);
   ChartSetInteger(0, CHART_EVENT_MOUSE_WHEEL, mouseWheel);
   ChartSetInteger(0, CHART_EVENT_OBJECT_CREATE, objectCreate);
   ChartSetInteger(0, CHART_EVENT_OBJECT_DELETE, objectDelete);
}
Run the indicator with the new event types enabled. Be prepared for a lot of mouse movement
messages. Here is a snippet of the log:
CHARTEVENT_MOUSE_WHEEL 5308557 -120.000000 ''
CHARTEVENT_CHART_CHANGE 0 0.000000 ''
CHARTEVENT_MOUSE_WHEEL 5308557 -120.000000 ''
CHARTEVENT_CHART_CHANGE 0 0.000000 ''
CHARTEVENT_MOUSE_MOVE 141 81.000000 '2'
CHARTEVENT_MOUSE_MOVE 141 81.000000 '0'
...
CHARTEVENT_OBJECT_CREATE 0 0.000000 'Daily Rectangle 37664'
CHARTEVENT_MOUSE_MOVE 323 146.000000 '0'
CHARTEVENT_MOUSE_MOVE 322 146.000000 '0'
CHARTEVENT_MOUSE_MOVE 321 146.000000 '0'
CHARTEVENT_MOUSE_MOVE 320 146.000000 '0'
CHARTEVENT_MOUSE_MOVE 318 146.000000 '0'
CHARTEVENT_MOUSE_MOVE 316 146.000000 '0'
CHARTEVENT_MOUSE_MOVE 314 146.000000 '0'
CHARTEVENT_MOUSE_MOVE 314 145.000000 '0'
...
CHARTEVENT_OBJECT_DELETE 0 0.000000 'Daily Rectangle 37664'
CHARTEVENT_KEYDOWN 46 1.000000 '339
We will disclose the specifics of information for each type of event in the relevant sections below.
5.9.3 Chart change event
When changing the chart size, price display modes, scale, or other parameters, the terminal sends the
CHARTEVENT_CHART_CHANGE event, which has no parameters. The MQL program must find out the
changes on its own using ChartGet function calls.
We have already used this event in the ChartModeMonitor.mq5 example in the section on Chart display
modes. Now let's take another example.
As you know, MetaTrader 5 allows the saving of the screenshot of the current chart to a file of a
specified size (the Save as Picture command of the context menu). However, this method of obtaining a
screenshot is not suitable for all cases. In particular, if you need an image with a tooltip or when an
object of the input field type is active (when text is selected inside the field and the text cursor is
visible), the standard command will not help, since it re-forms the chart image without taking into
account these and some other nuances of the current state of the window.
The only alternative to get an exact copy of the window is to use means that are external to the
terminal (for example, the PrtSc key via the Windows clipboard), but this method does not guarantee
the required window size. In order not to select the size by trial and error, or some additional programs,

---

## Page 1026

Part 5. Creating application programs
1 026
5.9 Interactive events on charts
we will create an indicator EventWindowSizer.mq5, which will track the user's size setting on the go and
output the current value in a comment.
All work is done in the OnChartEvent handler, starting with checking the event ID for
CHARTEVENT_CHART_CHANGE. The dimensions of the window in pixels can be obtained using the
CHART_WIDTH_IN_PIXELS and CHART_HEIGHT_IN_PIXELS properties. However, they return
dimensions without taking into account borders, and the borders are usually wanted for a screenshot.
Therefore, we will display in the comment not only the property values (marked with the word
"Screen"), but also the corrected values (marked with the word "Picture"): 2 pixels should be added in
width, and 1  pixel in vertical (these are the features of window rendering in the terminal).
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_CHART_CHANGE)
   {
      const int w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
      const int h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
      // "Raw" sizes "as is" are displayed with the "Screen" mark,
      // correction for (-2,-1) is needed to include frames - it is displayed with the "Picture" mark,
      // correction for (-54,-22) is needed to include scales - it is displayed with "Including scales" flag.
      Comment(StringFormat("Screen: %d x %d\nPicture: %d x %d\nIncluding scales: %d x %d",
         w, h, w + 2, h + 1, w + 2 + 54, h + 1 + 22));
   }
}
Moreover, the obtained values do not take into account time and price scales. If they should also be
taken into account in the size of the screenshot, then an adjustment should be made for their size as
well. Unfortunately, the MQL5 API does not provide a way to find out these sizes, so we can only
determine them empirically: for standard Windows font settings, the price scale width is 54 pixels, and
the time scale height is 22 pixels. These constants may differ for your version of Windows, so you
should edit them, or set them using input parameters.
After running the indicator on a chart, try resizing the window and see how the numbers in the
comment will change.

---

## Page 1027

Part 5. Creating application programs
1 027
5.9 Interactive events on charts
Window s
Window screenshot with a tooltip and current sizes in the comment
5.9.4 Keyboard events
MQL programs can receive messages about keystrokes from the terminal by processing in the
OnChartEvent function the CHARTEVENT_KEYDOWN events.
It is important to note that events are generated only in the active chart, and only when it has the
input focus.
In Windows, focus is the logical and visual selection of one particular window that the user is currently
interacting with. As a rule, the focus is moved by a mouse click or special keyboard shortcuts (Tab,
Ctrl+Tab), causing the selected window to be highlighted. For example, a text cursor will appear in the
input field, the current line will be colored in the list with an alternative color, and so on.
Similar visual effects are noticeable in the terminal, in particular, when one of the Market Watch, Data
Window windows, or the Expert log receives focus. However, the situation is somewhat different with
chart windows. It is not always possible to distinguish by external signs whether the chart visible in the
foreground has the input focus or not. It is guaranteed that you can switch the focus, as already
mentioned, by clicking on the required chart (on the chart, and not on the window title or its frame) or
using hot keys:
• Alt+W brings up a window with a list of charts, where you can select one.
• Ctrl+F6 switches to the next chart (in the list of windows, where the order corresponds, as a rule,
to the order of tabs).
• Crtl+Shift+F6 switches to the previous chart.

---

## Page 1028

Part 5. Creating application programs
1 028
5.9 Interactive events on charts
The full list of MetaTrader 5 hotkeys can be found in the documentation. Please pay attention that
some combinations do not comply with the general recommendations of Microsoft (for example, F1 0
opens the quotes window, but does not activate the main menu).
The CHARTEVENT_KEYDOWN event parameters contain the following information:
• lparam – code of the pressed key
• dparam – the number of keystrokes generated during the time it was held down
• sparam – a bitmask describing the status of the keyboard keys, converted to a string
Bits
Description
0–7
Key scan code (depends on hardware, OEM)
8
Extended keyboard key attribute
9–1 2
For Windows service purposes (do not use)
1 3
Key state Alt (1  - pressed, 0 - released), not available (see below)
1 4
Previous key state (1  - pressed, 0 - released)
1 5
Changed key state (1  if released, 0 if pressed)
The state of the Alt key is actually not available, because it is intercepted by the terminal and this bit is
always 0. Bit 1 5 is also always equal to 0 due to the triggering context of this event: only key presses
are passed to the MQL program, not key releases.
The attribute of the extended keyboard (bit 8) is set, for example, for the keys of the numeric block
(on laptops it is usually activated by Fn), keys such as NumLock, ScrollLock, right Ctrl (as opposed to
the left, main Ctrl), and so on. Read more about this in the Windows documentation.
The first time any non-system key is pressed, bit 1 4 will be 0. If you keep the key pressed, subsequent
automatically generated repetitions of the event will have 1  in this bit.
The following structure will help ensure that the description of the bits is correct.

---

## Page 1029

Part 5. Creating application programs
1 029
5.9 Interactive events on charts
struct KeyState
{
   uchar scancode;
   bool extended;
   bool altPressed;
   bool previousState;
   bool transitionState;
   
   KeyState() { }
   KeyState(const ushort keymask)
   {
      this = keymask; // use operator overload=
   }
   void operator=(const ushort keymask)
   {
      scancode = (uchar)(0xFF & keymask);
      extended = 0x100 & keymask;
      altPressed = 0x2000 & keymask;
      previousState = 0x4000 & keymask;
      transitionState = 0x8000 & keymask;
   }
};
In an MQL program, it can be used like this.
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN)
   {
      PrintFormat("%lld %lld %4llX", lparam, (ulong)dparam, (ushort)sparam);
      KeyState state[1];
      state[0] =(ushort)sparam;
      ArrayPrint(state);
   }
}
For practical purposes, it is more convenient to extract bit attributes from the key mask using macros.
#define KEY_SCANCODE(SPARAM) ((uchar)(((ushort)SPARAM) & 0xFF))
#define KEY_EXTENDED(SPARAM) ((bool)(((ushort)SPARAM) & 0x100))
#define KEY_PREVIOUS(SPARAM) ((bool)(((ushort)SPARAM) & 0x4000))
You can run the EventAll.mq5 indicator from the Event-related chart properties section on the chart and
see what parameter values will be displayed in the log when certain keys are pressed.
It is important to note that the code in lparam is one of the virtual keyboard key codes. Their list can
be seen in the file MQL5/Include/VirtualKeys.mqh, which comes with MetaTrader 5. For example, here
are some of them:

---

## Page 1030

Part 5. Creating application programs
1 030
5.9 Interactive events on charts
#define VK_SPACE          0x20
#define VK_PRIOR          0x21
#define VK_NEXT           0x22
#define VK_END            0x23
#define VK_HOME           0x24
#define VK_LEFT           0x25
#define VK_UP             0x26
#define VK_RIGHT          0x27
#define VK_DOWN           0x28
...
#define VK_INSERT         0x2D
#define VK_DELETE         0x2E
...
// VK_0 - VK_9 ASCII codes of characters '0' - '9' (0x30 - 0x39)
// VK_A - VK_Z ASCII codes of characters 'A' - 'Z' (0x41 - 0x5A)
The codes are called virtual because the corresponding keys may be located differently on different
keyboards, or even implemented through the joint pressing of auxiliary keys (such as Fn on laptops). In
addition, virtuality has another side: the same key can generate different symbols or control actions.
For example, the same key can denote different letters in different language layouts. Also, each of the
letter keys can generate an uppercase or lowercase letter, depending on the mode of CapsLock and the
state of the Shift keys.
In this regard, to get a character from a virtual key code, the MQL5 API has the special function
TranslateKey.
short TranslateKey(int key)
The function returns a Unicode character based on the passed virtual key code, given the current input
language and the state of the control keys.
In case of an error, the value -1  will be returned. An error can occur if the code does not match the
correct character, for example, when trying to get a character for the Shift key.
Recall that in addition to the received code of the pressed key, an MQL program can additionally Check
keyboard status in terms of control keys and modes. By the way, constants of the form
TERMINAL_KEYSTATE_XXX, passed as a parameter to the TerminalInfoInteger function, are based on
the principle of 1 000 + virtual key code. For example, TERMINAL_KEYSTATE_UP is 1 038 because
VK_UP is 38 (0x26).
When planning algorithms that react to keystrokes, keep in mind that the terminal can intercept many
key combinations, since they are reserved for performing certain actions (the link to the
documentation was given above). In particular, pressing the spacebar opens a field for quick navigation
along the time axis. The MQL5 API allows you to partly control such built-in keyboard processing and
disable it if necessary. See the section on Mouse and keyboard control.
The simple bufferless indicator EventTranslateKey.mq5 serves as a demonstration of this function. In its
OnChartEvent handler for the CHARTEVENT_KEYDOWN events, TranslateKeyis is called to get a valid
Unicode character. If it succeeds, the symbol is added to the message string that is displayed in the
plot comment. On pressing Enter, a newline is inserted into the text, and on pressing Backspace, the
last character is erased from the end.

---

## Page 1031

Part 5. Creating application programs
1 031 
5.9 Interactive events on charts
#include <VirtualKeys.mqh>
   
string message = "";
   
void OnChartEvent(const int id,
   const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_KEYDOWN)
   {
      if(lparam == VK_RETURN)
      {
         message += "\n";
      }
      else if(lparam == VK_BACK)
      {
         StringSetLength(message, StringLen(message) - 1);
      }
      else
      {
         ResetLastError();
         const ushort c = TranslateKey((int)lparam);
         if(_LastError == 0)
         {
            message += ShortToString(c);
         }
      }
      Comment(message);
   }
}
You can try entering characters in different cases and different languages.
Be careful. The function returns the signed short value, mainly to be able to return an error code of -1 .
However, the type of a "wide" two-byte character is considered to be an unsigned integer, ushort. If
the receiving variable is declared as ushort, a check using -1  (for example,c!=-1 ) will issue a "sign
mismatch" compiler warning (explicit type casting required), while the other (c >= 0) is generally
erroneous, since it is always equal to true.
In order to be able to insert spaces between words in the message, quick navigation activated by the
spacebar is pre-disabled in the OnInit handler.
void OnInit()
{
   ChartSetInteger(0, CHART_QUICK_NAVIGATION, false);
}
As a full-fledged example of using keyboard events, consider the following application task. Terminal
users know that the scale of the main chart window can be changed interactively without opening the
settings dialog using the mouse: just press the mouse button in the price scale and, without releasing it,
move up/down. Unfortunately, this method does not work in subwindows.
Subwindows always scale automatically to fit all the content, and to change the scale you have to open
a dialog and enter values manually. Sometimes the need for this arises if the indicators in the

---

## Page 1032

Part 5. Creating application programs
1 032
5.9 Interactive events on charts
subwindow show "outliers" – too large single readings that interfere with the analysis of the rest of the
normal (medium) size data. In addition, sometimes it is desirable to simply enlarge the picture in order
to deal with finer details.
To solve this problem and allow the user to adjust the scale of the subwindow using keystrokes, we
have implemented the SubScalermq5 indicator. It has no buffers and does not display anything.
SubScaler must be the first indicator in the subwindow, or, to put it more strictly, it must be added to
the subwindow before the working indicator of interest to you is added there, the scale of which you
want to control. To make SubScaler the first indicator, it should be placed on the chart (in the main
window) and thereby create a new subwindow, where you can then add a subordinate indicator.
In the working indicator settings dialog, it is important to enable the option Inherit scale (on the tab
Scale).
When both indicators are running in a subwindow, you can use the arrow keys Up/Down to zoom in/out.
If the Shift key is pressed, the current visible range of values on the vertical axis is shifted up or down.
Zooming in means zooming in on details ("camera zoom"), so that some of the data may go outside the
window. Zooming out means that the overall picture becomes smaller ("camera zoom out").
The input parameters set are: 
• Initial maximum – the upper limit of the data during the initial placement on the chart, +1 000 by
default.
• Initial minimum – the lower data limit during the initial placement on the chart, by default -1 000.
• Scaling factor – step with which the scale will change by pressing the keys, value in the range
[0.01  ... 0.5], by default 0.1 .
We are forced to ask the user for the minimum and maximum because SubScaler cannot know in
advance the working range of values of an arbitrary third-party indicator, which will be added to the
subwindow next.
When the chart is restored after starting a new terminal session or when a tpl template is loaded,
SubScaler picks up the scale of the previous (saved) state.
Now let's look at the implementation of SubScaler.
The above settings are set in the corresponding input variables:
input double FixedMaximum = 1000;  // Initial Maximum
input double FixedMinimum = -1000; // Initial Minimum
input double _ScaleFactor = 0.1;   // Scale Factor [0.01 ... 0.5]
input bool Disabled = false;
In addition, the Disabled variable allows you to temporarily disable the keyboard response for a specific
instance of the indicator in order to set up several different scales in different subwindows (one by one).
Since the input variables are read-only in MQL5, we are forced to declare one more variable
ScaleFactor to correct the entered value within the allowed range [0.01  ... 0.5].
double ScaleFactor;
The number of the current subwindow (w) and the number of indicators in it (n) are stored in global
variables: they are all filled in the OnInit handler.

---

## Page 1033

Part 5. Creating application programs
1 033
5.9 Interactive events on charts
int w = -1, n = -1;
   
void OnInit()
{
  ScaleFactor = _ScaleFactor;
  if(ScaleFactor < 0.01 || ScaleFactor > 0.5)
  {
    PrintFormat("ScaleFactor %f is adjusted to default value 0.1,"
       " valid range is [0.01, 0.5]", ScaleFactor);
    ScaleFactor = 0.1;
  }
  w = ChartWindowFind();
  n = ChartIndicatorsTotal(0, w);
}
In the OnChartEvent function, we process two types of events: chart changes and keyboard events. The
CHARTEVENT_CHART_CHANGE event is necessary to keep track of the addition of the next indicator to
the subwindow (working indicator to be scaled). At the same time, we request the current range of
subwindow values (CHART_PRICE_MIN, CHART_PRICE_MAX) and determine whether it is degenerate,
that is, when both the maximum and minimum are equal to zero. In this case, it is necessary to apply
the initial limits specified in the input parameters (FixedMinimum,FixedMaximum).
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   switch(id)
   {
   case CHARTEVENT_CHART_CHANGE:
      if(ChartIndicatorsTotal(0, w) > n)
      {
         n = ChartIndicatorsTotal(0, w);
         const double min = ChartGetDouble(0, CHART_PRICE_MIN, w);
         const double max = ChartGetDouble(0, CHART_PRICE_MAX, w);
         PrintFormat("Change: %f %f %d", min, max, n);
         if(min == 0 && max == 0)
         {
            IndicatorSetDouble(INDICATOR_MINIMUM, FixedMinimum);
            IndicatorSetDouble(INDICATOR_MAXIMUM, FixedMaximum);
         }
      }
      break;
   ...
   }
}
When a keyboard press event is received, the main Scale function is called, which receives not only
lparam but also the state of the Shift key obtained by referring to
TerminalInfoInteger(TERMINAL_ KEYSTATE_ SHIFT).

---

## Page 1034

Part 5. Creating application programs
1 034
5.9 Interactive events on charts
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
  switch(id)
  {
    case CHARTEVENT_KEYDOWN:
      if(!Disabled)
         Scale(lparam, TerminalInfoInteger(TERMINAL_KEYSTATE_SHIFT));
      break;
    ...
  }
}
Inside the Scale function, the first thing we do is get the current range of values into the min and max
variables.
void Scale(const long cmd, const int shift)
{
   const double min = ChartGetDouble(0, CHART_PRICE_MIN, w);
   const double max = ChartGetDouble(0, CHART_PRICE_MAX, w);
   ...
Then, depending on whether the Shift key is currently pressed, either zooming or panning is performed,
i.e. shifting the visible range of values up or down. In both cases, the modification is performed with a
given step (multiplier) ScaleFactor, relative to the limits min and max, and they are assigned to the
indicator properties INDICATOR_MINIMUM and INDICATOR_MAXIMUM, respectively. Due to the fact
that the subordinate indicator has the Inherit scale setting, it becomes a working setting for it as well.

---

## Page 1035

Part 5. Creating application programs
1 035
5.9 Interactive events on charts
 if((shift &0x10000000) ==0)// Shift is not pressed - scalechange
   {
      if(cmd == VK_UP) // enlarge (zoom in)
      {
         IndicatorSetDouble(INDICATOR_MINIMUM, min / (1.0 + ScaleFactor));
         IndicatorSetDouble(INDICATOR_MAXIMUM, max / (1.0 + ScaleFactor));
         ChartRedraw();
      }
      else if(cmd == VK_DOWN) // shrink (zoom out)
      {
         IndicatorSetDouble(INDICATOR_MINIMUM, min * (1.0 + ScaleFactor));
         IndicatorSetDouble(INDICATOR_MAXIMUM, max * (1.0 + ScaleFactor));
         ChartRedraw();
      }
   }
   else // Shift pressed - pan/shift range
   {
      if(cmd == VK_UP) // shifting charts up
      {
         const double d = (max - min) * ScaleFactor;
         IndicatorSetDouble(INDICATOR_MINIMUM, min - d);
         IndicatorSetDouble(INDICATOR_MAXIMUM, max - d);
         ChartRedraw();
      }
      else if(cmd == VK_DOWN) // shifting charts down
      {
         const double d = (max - min) * ScaleFactor;
         IndicatorSetDouble(INDICATOR_MINIMUM, min + d);
         IndicatorSetDouble(INDICATOR_MAXIMUM, max + d);
         ChartRedraw();
      }
   }
}
For any change, ChartRedraw is called to update the chart.
Let's see how SubScaler works with the standard indicator of volumes (any other indicators, including
custom ones, are controlled in the same way).

---

## Page 1036

Part 5. Creating application programs
1 036
5.9 Interactive events on charts
A different scale set by SubScaler indicators in two subwindows
Here in two subwindows, two instances of SubScaler apply different vertical scales to volumes.
5.9.5 Mouse events
We already had the opportunity to make sure that we receive mouse events using the indicator
EventAll.mq5 from the section Event-related chart properties. The CHARTEVENT_CLICK event is sent to
the MQL program on each click of the mouse button in the window, and the
CHARTEVENT_MOUSE_MOVE cursor movement and CHARTEVENT_MOUSE_WHEEL wheel scroll events
require prior activation in the chart settings, for which the CHART_EVENT_MOUSE_MOVE and
CHART_EVENT_MOUSE_WHEEL properties serve, respectively (both are disabled by default).
If there is a graphic object under the mouse, when the button is pressed, not only the
CHARTEVENT_CLICK event is generated but also CHARTEVENT_OBJECT_CLICK.
For the CHARTEVENT_CLICK and CHARTEVENT_MOUSE_MOVE events, the parameters of the
OnChartEvent handler contain the following information:
·lparam - X coordinate
·dparam - Y coordinate
In addition, for the CHARTEVENT_MOUSE_MOVE event, the sparam parameter contains a string
representation of a bitmask describing the status of mouse buttons and control keys (Ctrl, Shift).
Setting a particular bit to 1  means pressing the corresponding button or key.
Bits
Description
0
Left mouse button state

---

## Page 1037

Part 5. Creating application programs
1 037
5.9 Interactive events on charts
Bits
Description
1
Right mouse button state
2
SHIFT key state
3
CTRL key state
4
Middle mouse button state
5
State of the first additional mouse button
6
State of the second additional mouse button
For example, if the 0th bit is set, it will give the number 1  (1  << 0), and if the 4th bit is set, it will give
the number 1 6 (1  << 4). Simultaneous pressing of buttons or keys is indicated by a superposition of
bits.
For the CHARTEVENT_MOUSE_WHEEL event, the X and Y coordinates, as well as the status flags of the
mouse buttons and control keys, are encoded in a special way inside the lparam parameter, and the
dparam parameter reports the direction (plus/minus) and amount of wheel scrolling (multiples of
±1 20).
8-byte integer lparam combines several of the mentioned information fields.
Bytes
Description
0
Value of short type with the X coordinate
1
2
Value of short type with the Y coordinate
3
4
Bitmask of button and key states
5
Not used
6
7
Regardless of the type of event, mouse coordinates are transmitted relative to the entire window,
including subwindows, so they should be recalculated for a specific subwindow if necessary.
For a better understanding of CHARTEVENT_MOUSE_WHEEL, use the indicator EventMouseWheel.mq5.
It receives and decodes the messages, and then outputs their description to the log.

---

## Page 1038

Part 5. Creating application programs
1 038
5.9 Interactive events on charts
#define KEY_FLAG_NUMBER 7
   
const string keyNameByBit[KEY_FLAG_NUMBER] =
{
   "[Left Mouse] ",
   "[Right Mouse] ",
   "(Shift) ",
   "(Ctrl) ",
   "[Middle Mouse] ",
   "[Ext1 Mouse] ",
   "[Ext2 Mouse] ",
};
   
void OnChartEvent(const int id,
   const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_MOUSE_WHEEL)
   {
      const int keymask = (int)(lparam >> 32);
      const short x = (short)lparam;
      const short y = (short)(lparam >> 16);
      const short delta = (short)dparam;
      string message = "";
      
      for(int i = 0; i < KEY_FLAG_NUMBER; ++i)
      {
         if(((1 << i) & keymask) != 0)
         {
            message += keyNameByBit[i];
         }
      }
      
      PrintFormat("X=%d Y=%d D=%d %s", x, y, delta, message);
   }
}
Run the indicator on the chart and scroll the mouse wheel by pressing various buttons and keys in turn.
Here is an example result:
X=186 Y=303 D=-120 
X=186 Y=312 D=120 
X=230 Y=135 D=-120 
X=230 Y=135 D=-120 (Ctrl) 
X=230 Y=135 D=-120 (Shift) (Ctrl) 
X=230 Y=135 D=-120 (Shift) 
X=230 Y=135 D=120 
X=230 Y=135 D=-120 [Middle Mouse] 
X=230 Y=135 D=120 [Middle Mouse] 
X=236 Y=210 D=-240 
X=236 Y=210 D=-360 

---

## Page 1039

Part 5. Creating application programs
1 039
5.9 Interactive events on charts
5.9.6 Graphical object events
For the graphical objects located on the chart, the terminal generates several specialized events. Most
of them apply to objects of any type. The text editing end event in the input field –
CHARTEVENT_OBJECT_ENDEDIT – is generated only for objects of the OBJ_EDIT type.
Object click (CHARTEVENT_OBJECT_CLICK), mouse drag (CHARTEVENT_OBJECT_DRAG) and object
property change (CHARTEVENT_OBJECT_CHANGE) events are always active, while
CHARTEVENT_OBJECT_CREATE object creation and CHARTEVENT_OBJECT_DELETE object creation
events require explicit enabling by setting chart the relevant properties:
CHART_EVENT_OBJECT_CREATE and CH ART_EVENT_OBJECT_DELETE.
When renaming an object manually (from the properties dialog), the terminal generates a sequence of
events CHARTEVENT_OBJECT_DELETE, CHARTEVENT_OBJECT_CREATE,
CHARTEVENT_OBJECT_CHANGE. When you programmatically rename an object, these events are not
generated.
All events in objects carry the name of the associated object in the sparam parameter of the
OnChartEvent function.
In addition, click coordinates are passed for CHARTEVENT_OBJECT_CLICK: X in the lparam parameter
and Y in the dparam parameter. Coordinates are common to the entire chart, including subwindows.
Clicking on objects works differently depending on the object type. For some, such as the ellipse, the
cursor must be over any anchor point. For others (triangle, rectangle, lines), the cursor may be over
the object's perimeter, not just over a point. In all such cases, hovering the mouse cursor over the
interactive area of the object displays a tooltip with the name of the object.
Objects linked to screen coordinates, which allow to form the graphical interface of the program, in
particular, a button, an input field, and a rectangular panel, generate events when the mouse is clicked
anywhere inside the object.
If there are multiple objects under the cursor, an event is generated for the object with the largest Z-
priority. If the priorities of the objects are equal, the event is assigned to the one that was created
later (this corresponds to their visual display, that is, the later one overlaps the earlier one).
The new version of the indicator will help you check events in objects EventAllObj ects.mq5. We will
create and configure it using the already known class Obj ect Selector of several objects, and then
intercept in the handler OnChartEvent their characteristic events.

---

## Page 1040

Part 5. Creating application programs
1 040
5.9 Interactive events on charts
#include <MQL5Book/ObjectMonitor.mqh>
   
class ObjectBuilder: public ObjectSelector
{
protected:
   const ENUM_OBJECT type;
   const int window;
public:
   ObjectBuilder(const string _id, const ENUM_OBJECT _type,
      const long _chart = 0, const int _win = 0):
      ObjectSelector(_id, _chart), type(_type), window(_win)
   {
      ObjectCreate(host, id, type, window, 0, 0);
   }
};
Initially, in OnInit we create a button object and a vertical line. For the line, we will track the event of
movement (towing), and on pressing the button, we will create an input field for which we will check the
entered text.
const string ObjNamePrefix = "EventShow-";
const string ButtonName = ObjNamePrefix + "Button";
const string EditBoxName = ObjNamePrefix + "EditBox";
const string VLineName = ObjNamePrefix + "VLine";
   
bool objectCreate, objectDelete;
   
void OnInit()
{
   // remember the original settings to restore in OnDeinit
   objectCreate = ChartGetInteger(0, CHART_EVENT_OBJECT_CREATE);
   objectDelete = ChartGetInteger(0, CHART_EVENT_OBJECT_DELETE);
   
   // set new properties
   ChartSetInteger(0, CHART_EVENT_OBJECT_CREATE, true);
   ChartSetInteger(0, CHART_EVENT_OBJECT_DELETE, true);
   
   ObjectBuilder button(ButtonName, OBJ_BUTTON);
   button.set(OBJPROP_XDISTANCE, 100).set(OBJPROP_YDISTANCE, 100)
   .set(OBJPROP_XSIZE, 200).set(OBJPROP_TEXT, "Click Me");
   
   ObjectBuilder line(VLineName, OBJ_VLINE);
   line.set(OBJPROP_TIME, iTime(NULL, 0, 0))
   .set(OBJPROP_SELECTABLE, true).set(OBJPROP_SELECTED, true)
   .set(OBJPROP_TEXT, "Drag Me").set(OBJPROP_TOOLTIP, "Drag Me");
   
   ChartRedraw();
}
Along the way, do not forget to set the chart properties CHART_EVENT_OBJECT_CREATE and
CHART_EVENT_OBJECT_DELETE to true to be notified when a set of objects changes.

---

## Page 1041

Part 5. Creating application programs
1 041 
5.9 Interactive events on charts
In the OnChartEvent function, we will provide an additional response to the required events: after the
dragging is completed, we will display the new position of the line in the log and, after editing the text in
the input field, its contents.
void OnChartEvent(const int id,
   const long &lparam, const double &dparam, const string &sparam)
{
   ENUM_CHART_EVENT evt = (ENUM_CHART_EVENT)id;
   PrintFormat("%s %lld %f '%s'", EnumToString(evt), lparam, dparam, sparam);
   if(id == CHARTEVENT_OBJECT_CLICK && sparam == ButtonName)
   {
      if(ObjectGetInteger(0, ButtonName, OBJPROP_STATE))
      {
         ObjectBuilder edit(EditBoxName, OBJ_EDIT);
         edit.set(OBJPROP_XDISTANCE, 100).set(OBJPROP_YDISTANCE, 150)
         .set(OBJPROP_BGCOLOR, clrWhite)
         .set(OBJPROP_XSIZE, 200).set(OBJPROP_TEXT, "Edit Me");
      }
      else
      {
         ObjectDelete(0, EditBoxName);
      }
      
      ChartRedraw();
   }
   else if(id == CHARTEVENT_OBJECT_ENDEDIT && sparam == EditBoxName)
   {
      Print(ObjectGetString(0, EditBoxName, OBJPROP_TEXT));
   }
   else if(id == CHARTEVENT_OBJECT_DRAG && sparam == VLineName)
   {
      Print(TimeToString((datetime)ObjectGetInteger(0, VLineName, OBJPROP_TIME)));
   }
}
Note that when the button is pressed for the first time, its state changes from released to pressed, and
in response to this, we create an input field. If you click the button again, it will change its state back,
as a result of which the input field will be removed from the chart.
Below is an image of the chart during the operation of the indicator.

---

## Page 1042

Part 5. Creating application programs
1 042
5.9 Interactive events on charts
Objects controlled by the OnChartEvent event handler
Immediately after the indicator is launched, the following lines appear in the log:
CHARTEVENT_OBJECT_CREATE 0 0.000000 'EventShow-Button'
CHARTEVENT_OBJECT_CREATE 0 0.000000 'EventShow-VLine'
CHARTEVENT_CHART_CHANGE 0 0.000000 ''
If we then drag the line with the mouse, we will see something like this:
CHARTEVENT_OBJECT_DRAG 0 0.000000 'EventShow-VLine'
2022.01.05 10:00
Next, you can click the button and edit the text in the newly created input field (when editing is
complete, click Enter or click outside the input field). This will result in the following entries in the log
(the coordinates and the text of the message may differ; the text "new message" was entered here):
CHARTEVENT_OBJECT_CLICK 181 113.000000 'EventShow-Button'
CHARTEVENT_CLICK 181 113.000000 ''
CHARTEVENT_OBJECT_CREATE 0 0.000000 'EventShow-EditBox'
CHARTEVENT_OBJECT_CLICK 152 160.000000 'EventShow-EditBox'
CHARTEVENT_CLICK 152 160.000000 ''
CHARTEVENT_OBJECT_ENDEDIT 0 0.000000 'EventShow-EditBox'
new message
If you then release the button, the input field will be deleted.

---

## Page 1043

Part 5. Creating application programs
1 043
5.9 Interactive events on charts
CHARTEVENT_OBJECT_CLICK 162 109.000000 'EventShow-Button'
CHARTEVENT_CLICK 162 109.000000 ''
CHARTEVENT_OBJECT_DELETE 0 0.000000 'EventShow-EditBox'
It is worth noting that the button works by default as a two-position switch, that is, it sticks alternately
in the pressed or released state as a result of a mouse click. For a regular button, this behavior is
redundant: in order to simply track button presses, you should return it to the released state when
processing the event by calling Obj ectSetInteger(0, ButtonName, OBJPROP_ STATE, false).
5.9.7 Generation of custom events
In addition to standard events, the terminal supports the ability to programmatically generate custom
events, the essence and content of which are determined by the MQL program. Such events are added
to the general queue of chart events and can be processed in the function OnChartEvent by all
interested programs.
A special range of 65536 integer identifiers is reserved for custom events: from
CHARTEVENT_CUSTOM to CHARTEVENT_CUSTOM_LAST inclusive. In other words, the custom event
must have the ID CHARTEVENT_CUSTOM + n, where n is between 0 and 65535.
CHARTEVENT_CUSTOM_LAST is exactly equal to CHARTEVENT_CUSTOM + 65535.
Custom events are sent to the chart using the EventChartCustom function.
bool EventChartCustom(long chartId, ushort customEventId,
   long lparam, double dparam, string sparam)
chartId is the identifier of the event recipient chart, while 0 indicates the current chart; customEventId
is the event ID (selected by the MQL program developer). This identifier is automatically added to the
CHARTEVENT_CUSTOM value and converted to an integer type. This value will be passed to the
OnChartEvent handler as the first argument. Other parameters of EventChartCustom correspond to the
standard event parameters in OnChartEvent with types long, double and string, and may contain
arbitrary information.
The function returns true in case of successful queuing of the user event or false in case of an error
(the error code will become available in _ LastError).
As we approach the most complex and important part of our book devoted directly to trading
automation, we will begin to solve applied problems that will be useful in the development of trading
robots. Now, in the context of demonstrating the capabilities of custom events, let's turn to the
multicurrency (or, more generally, multisymbol) analysis of the trading environment.
A little earlier, in the chapter on indicators, we considered multicurrency indicators but did not pay
attention to an important point: despite the fact that the indicators processed quotes of different
symbols, the calculation itself was launched in the OnCalculate handler, which was triggered by the
arrival of a new tick of only one the working symbol of the chart. It turns out that the ticks of other
instruments are essentially skipped. For example, if the indicator works on symbol A, when its tick
arrives, we simply take the last known ticks of other symbols (B, C, D), but it is likely that other ticks
managed to slip through each of them.
If you place a multicurrency indicator on the most liquid instrument (where ticks are received most
often), this is not so critical. However, different instruments can be faster than others at different times
of the day, and if an analytical or trading algorithm requires the fastest possible response to new

---

## Page 1044

Part 5. Creating application programs
1 044
5.9 Interactive events on charts
quotes of all instruments in the portfolio, we are faced with the fact that the current solution does not
suit us.
Unfortunately, the standard event of a new tick arrival works in MQL5 only for one symbol, which is the
working symbol of the current chart. In indicators, the OnCalculate handler is called at such moments,
and the OnTick handler is called in Expert Advisors.
Therefore, it is necessary to invent some mechanism so that the MQL program can receive notifications
about ticks on all instruments of interest. This is where custom events will help us. Of course, this is not
necessary for programs that analyze only one instrument.  
We will now develop an example of the EventTickSpy.mq5 indicator, which, being launched on a specific
symbol X, will be able to send tick notifications from its OnCalculate function using EventChartCustom.
As a result, in the handler OnChartEvent, which is specially prepared to receive such notifications, it will
be possible to collect notifications from different instances of the indicator from different symbols.
This example is provided for illustration purposes. Subsequently, when studying multicurrency
automated trading, we will adapt this technique for more convenient use in Expert Advisors.
First of all, let's think of a custom event number for the indicator. Since we are going to send tick
notifications for many different symbols from some given list, we can choose different tactics here. For
example, you can select one event identifier, and pass the number of the symbol in the list and/or the
name of the symbol in the lparam and sparam parameters, respectively. Or you can take some
constant (greater than and equal to CHARTEVENT_CUSTOM) and get event numbers by adding the
symbol number to this constant (then we have all parameters free, in particular, lparam and dparam,
and they can be used to transfer prices Ask, Bid or something else).
We will focus on the option when there is one event code. Let's declare it in the TICKSPY macro. This
will be the default value, which the user can change to avoid collisions (albeit unlikely) with other
programs if necessary.
#define TICKSPY 0xFEED // 65261
This value is taken on purpose as being rather far removed from the first allowed
CHARTEVENT_CUSTOM.
During the initial (interactive) launch of the indicator, the user must specify the list of instruments
whose ticks the indicator should track. For this purpose, we will describe the input string variable
SymbolList with a comma-separated list of symbols.
The identifier of the user event is set in the message parameter.
Finally, we need the identifier of the receiving chart to pass the event. We will provide the Chart
parameter for this purpose. The user should not edit it: in the first instance of the indicator launched
manually, the chart is known implicitly by attaching it to the chart. In other copies of the indicator that
our first instance will run programmatically, this parameter will fill the algorithm with a call of the
function ChartID (see below).
input string SymbolList = "EURUSD,GBPUSD,XAUUSD,USDJPY"; // List of symbols separated by commas (example)
input ushort message = TICKSPY;                          // Custom message
input longchart = 0;                                     // Receiving chart (do not edit)
In the SymbolList parameter, for example, a list with four common tools is indicated. Edit it as needed
to suit your Market Watch.

---

## Page 1045

Part 5. Creating application programs
1 045
5.9 Interactive events on charts
In the OnInit handler, we convert the list to the Symbols array of symbols, and then in a loop we run
the same indicator for all symbols from the array, except for the current one (as a rule, there is such a
match, because the current symbol is already being processed by this initial copy of the indicator).
string Symbols[];
   
void OnInit()
{
   PrintFormat("Starting for chart %lld, msg=0x%X [%s]", Chart, Message, SymbolList);
   if(Chart == 0)
   {
      if(StringLen(SymbolList) > 0)
      {
         const int n = StringSplit(SymbolList, ',', Symbols);
         for(int i = 0; i < n; ++i)
         {
            if(Symbols[i] != _Symbol)
            {
               ResetLastError();
               // run the same indicator on another symbol with different settings,
               // in particular, we pass our ChartID to receive notifications back
               iCustom(Symbols[i], PERIOD_CURRENT, MQLInfoString(MQL_PROGRAM_NAME),
                  "", Message, ChartID());
               if(_LastError != 0)
               {
                  PrintFormat("The symbol '%s' seems incorrect", Symbols[i]);
               }
            }
         }
      }
      else
      {
         Print("SymbolList is empty: tracking current symbol only!");
         Print("To monitor other symbols, fill in SymbolList, i.e."
            " 'EURUSD,GBPUSD,XAUUSD,USDJPY'");
      }
   }
}
At the beginning of OnInit, information about the launched instance of the indicator is displayed in the
log so that it is clear what is happening.
If we chose the option with separate event codes for each character, we would have to call iCustom as
follows (addi to message):
   iCustom(Symbols[i], PERIOD_CURRENT, MQLInfoString(MQL_PROGRAM_NAME), "",
      Message + i, ChartID());
Note that the non-zero value of the Chart parameter implies that this copy is launched
programmatically and that it should monitor a single symbol, that is, the working symbol of the chart.
Therefore, we don't need to pass a list of symbols when running the slave copies.

---

## Page 1046

Part 5. Creating application programs
1 046
5.9 Interactive events on charts
In the OnCalculate function, which is called when a new tick is received, we send the Message custom
event to the Chart chart by calling EventChartCustom. In this case, the lparam parameter is not used
(equal to 0). In the dparam parameter, we pass the current (last) price price[0] (this is Bid or Last,
depending on what type of price the chart is based on: it is also the price of the last tick processed by
the chart), and we pass the symbol name in the sparam parameter.
int OnCalculate(const int rates_total, const int prev_calculated,
   const int, const double &price[])
{
   if(prev_calculated)
   {
      ArraySetAsSeries(price, true);
      if(Chart > 0)
      {
         // send a tick notification to the parent chart
         EventChartCustom(Chart, Message, 0, price[0], _Symbol);
      }
      else
      {
         OnSymbolTick(_Symbol, price[0]);
      }
   }
  
   return rates_total;
}
In the original instance of the indicator, where the Chart parameter is 0, we directly call a special
function, a kind of a multiasset tick handler OnSymbolTick. In this case, there is no need to call
EventChartCustom: although such a message will still arrive on the chart and this copy of the indicator,
the transmission takes several milliseconds and loads the queue in vain.
The only purpose of OnSymbolTick in this demo is to print the name of the symbol and the new price in
the log.
void OnSymbolTick(const string &symbol, const double price)
{
   Print(symbol, " ", DoubleToString(price,
      (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)));
}
Of course, the same function is called from the OnChartEvent handler in the receiving (source) copy of
the indicator, provided that our message has been received. Recall that the terminal calls OnChartEvent
only in the interactive copy of the indicator (applied to the chart) and does not appear in those copies
that we created "invisible" using iCustom.

---

## Page 1047

Part 5. Creating application programs
1 047
5.9 Interactive events on charts
void OnChartEvent(const int id,
   const long &lparam, const double &dparam, const string &sparam)
{
   if(id >= CHARTEVENT_CUSTOM + Message)
   {
      OnSymbolTick(sparam, dparam);
      // OR (if using custom event range):
      // OnSymbolTick(Symbols[id - CHARTEVENT_CUSTOM - Message], dparam);
   }
}
We could avoid sending either the price or the name of the symbol in our event since the general list of
symbols is known in the initial indicator (which initiated the process), and therefore we could somehow
tell it the number of the symbol from the list. This could be done in the lparam parameter or, as
mentioned above, by adding a number to the base constant of the user event. Then the original
indicator, while receiving events, could take a symbol by index from the array and get all the
information about the last tick using SymbolInfoTick, including different types of prices.
Let's run the indicator on the EURUSD chart with default settings, including the
"EURUSD,GBPUSD,XAUUSD,USDJPY" test list. Here is the log:
16:45:48.745 (EURUSD,H1) Starting for chart 0, msg=0xFEED [EURUSD,GBPUSD,XAUUSD,USDJPY]
16:45:48.761 (GBPUSD,H1) Starting for chart 132358585987782873, msg=0xFEED []
16:45:48.761 (USDJPY,H1) Starting for chart 132358585987782873, msg=0xFEED []
16:45:48.761 (XAUUSD,H1) Starting for chart 132358585987782873, msg=0xFEED []
16:45:48.777 (EURUSD,H1) XAUUSD 1791.00
16:45:49.120 (EURUSD,H1) EURUSD 1.13068 *
16:45:49.135 (EURUSD,H1) USDJPY 115.797
16:45:49.167 (EURUSD,H1) XAUUSD 1790.95
16:45:49.167 (EURUSD,H1) USDJPY 115.796
16:45:49.229 (EURUSD,H1) USDJPY 115.797
16:45:49.229 (EURUSD,H1) XAUUSD 1790.74
16:45:49.369 (EURUSD,H1) XAUUSD 1790.77
16:45:49.572 (EURUSD,H1) GBPUSD 1.35332
16:45:49.572 (EURUSD,H1) XAUUSD 1790.80
16:45:49.791 (EURUSD,H1) XAUUSD 1790.80
16:45:49.791 (EURUSD,H1) USDJPY 115.796
16:45:49.931 (EURUSD,H1) EURUSD 1.13069 *
16:45:49.931 (EURUSD,H1) XAUUSD 1790.86
16:45:49.931 (EURUSD,H1) USDJPY 115.795
16:45:50.056 (EURUSD,H1) USDJPY 115.793
16:45:50.181 (EURUSD,H1) XAUUSD 1790.88
16:45:50.321 (EURUSD,H1) XAUUSD 1790.90
16:45:50.399 (EURUSD,H1) EURUSD 1.13066 *
16:45:50.727 (EURUSD,H1) EURUSD 1.13067 *
16:45:50.773 (EURUSD,H1) GBPUSD 1.35334
Please note that in the column with (symbol,timeframe) which is the source of the record, we first see
the starting indicator instances on four requested symbols.
After launch, the first tick was XAUUSD, not EURUSD. Further symbol ticks come with approximately
equal intensity, interspersed. EURUSD ticks are marked with asterisks, so you can get an idea of how
many other ticks would have been missed without notifications.

---

## Page 1048

Part 5. Creating application programs
1 048
5.9 Interactive events on charts
Timestamps have been saved in the left column for reference.
Places where two prices of two consecutive events from the same symbol coincide usually indicate that
the Ask price has changed (we simply do not display it here).
A little later, after studying the trading MQL5 API, we will apply the same principle to respond to
multicurrency ticks in Expert Advisors.

---

## Page 1049

Part 6. Trading automation
1 049
 
Part 6. Trading automation
In this part, we will study the most complex and important component of the MQL5 API which allows
the automation of trading actions.
We will start by describing the entities without which it is impossible to write a proper Expert Advisor.
These include financial symbols and trading account settings.
Then we will look at built-in trading functions and data structures, along with robot-specific events and
operating modes. In particular, the key feature of Expert Advisors is integration with the tester, which
allows users to evaluate financial performance and optimize trading strategies. We will consider the
internal optimization mechanisms and optimization management through the API.
The strategy tester is an essential tool for developing MQL programs since it provides the ability to
debug programs in various modes, including bars and ticks, based on modeled or real ticks, with or
without visualization of the price stream.
We've already tried to test indicators in visual mode. However, the set of testing parameters is limited
for indicators. When developing Expert Advisors, we will have access to the full range of tester
capabilities.
In addition, we will be introduced to a new form of market information: the Depth of Market and its
software interface.
MQL5 Programming for Traders – Source Codes from the Book. Part 6
Examples from the book are also available in the public project \MQL5\Shared Projects\MQL5Book
6.1  Financial instruments and Market Watch
MetaTrader 5 allows users to analyze and trade financial instruments (a.k.a. symbols or tickers), which
form the basis of almost all terminal subsystems. Charts, indicators, and the price history of quotes
exist in relation to trading symbols. The main functionality of the terminal is built on financial
instruments such as trading orders, deals, control of margin requirements, and trading account history.
Via the terminal, brokers deliver to traders a specified list of symbols, from which each user chooses
the preferred ones, forming the Market Watch. The Market Watch window determines the symbols for
which the terminal requests online quotes and allows you to open charts and view the history.
The MQL5 API provides similar software tools that allow you to view and analyze the characteristics of
all symbols, add them to the Market Watch, or exclude them from there.
In addition to standard symbols with information provided by brokers, MetaTrader 5 makes it possible
to create custom symbols: their properties and price history can be loaded from arbitrary data sources
and calculated using formulas or MQL programs. Custom symbols also participate in the Market Watch
and can be used for testing strategies and technical analysis, however, they also have a natural
limitation – they cannot be traded online using regular MQL5 API tools, since these symbols are not
available on the server. Custom symbols will be reviewed in a separate chapter, in the last, seventh part
of the book.

---

## Page 1050

Part 6. Trading automation
1 050
6.1  Financial instruments and Market Watch
A little while ago, in the relevant chapters, we have already touched on time series with price data of
individual symbols, including history paging using an example with indicators. All this functionality
actually assumes that the corresponding symbols are already enabled in the Market Watch. This is
especially true for multicurrency indicators and Expert Advisors that refer not only to the working
symbol of the chart but also to other symbols. In this chapter, we will learn how the Market Watch list
is managed from MQL programs.
The chapter on charts has already described some of the symbol properties made available through
basic property-getter functions of a current chart (Point, Digits) since the chart cannot work without
the symbol associated with it. Now we will study most of the properties of symbols, including their
specification. Their full set can be found in the MQL5 documentation on the website.
6.1 .1  Getting available symbols and Market Watch lists
The MQL5 API has several functions for operations with symbols. Using them, you can find the total
number of available symbols, the number of symbols selected in Market Watch, as well as their names.
As you know, the general list of symbols available in the terminal is indicated in the form of a
hierarchical structure in the dialog Symbols, which the user can open with the command View ->
Symbols, or from the Market Watch context menu. This list includes both the symbols provided by the
broker and custom symbols created locally. You can use the SymbolsTotal function to find the total
number of symbols.
int SymbolsTotal(bool selected)
The selected parameter specifies whether only symbols in Market Watch (true) or all available symbols
(false) are requested.
The SymbolName function is often used along with SymbolsTotal. It returns the name of the symbol by
its index (grouping the storage of symbols into logical folders is not taken into account here, see
property SYMBOL_PATH).
string SymbolName(int index, bool selected)
The index parameter specifies the index of the requested symbol. Index value must be between 0 and
the number of symbols, subject to the request context specified by the second parameter selected:
true limits the enumeration to the symbols chosen in Market Watch, while false matches absolutely all
symbols (by analogy with SymbolsTotal). Therefore, when calling SymbolName, set the selected
parameter to the same value as in the previous SymbolsTotal call which is used to define the index
range.
In case of an error, in particular, if the requested index is out of the list range, the function will return
an empty string, and the error code will be written to the variable _ LastError.
It is important to note that when the option selected is enabled, the pair of functions SymbolsTotal
and SymbolName returns information for the list of symbols actually updated by the terminal, that
is, symbols for which constant synchronization with the server is performed and for which the
history of quotes is available for MQL programs. This list may be larger than the list visible in Market
Watch, where elements are added explicitly: by the user or by an MQL program (to learn how to do
this, see the section Editing the list in Market Watch). Such symbols, invisible in the window, are
automatically connected by the terminal when they are needed for calculating cross-rates. Among
the symbol properties, there are two that allow you to distinguish between explicit selection
(SYMBOL_VISIBLE) and implicit selection (SYMBOL_SELECT); they will be discussed in the section
on symbol status check. Strictly speaking, for the SymbolsTotal and SymbolName functions, the

---

## Page 1051

Part 6. Trading automation
1 051 
6.1  Financial instruments and Market Watch
setting of selected to true matches the extended symbols set with SYMBOL_SELECT cocked, not
just those with SYMBOL_VISIBLE equal to true.
The order in which Market Watch symbols are returned corresponds to the terminal window (taking into
account the possible rearrangement made by the user, and not taking into account sorting by any
column, if it is enabled). Changing the order of symbols in Market Watch programmatically is not
possible.
The order in the general list of Symbols is set by the terminal itself (content and sorting of Market
Watch does not affect it).
As an example, let's look at the simple script SymbolList.mq5, which prints the available symbols to the
log. The input parameter MarketWatchOnly allows the user to limit the list to the Market Watch symbols
only (if the parameter is true) or to get the full list (false).
#property script_show_inputs
   
#include <MQL5Book/PRTF.mqh>
   
input bool MarketWatchOnly = true;
   
void OnStart()
{
   const int n = SymbolsTotal(MarketWatchOnly);
   Print("Total symbol count: ", n);
   // write a list of symbols in the Market Watch or all available
   for(int i = 0; i < n; ++i)
   {
      PrintFormat("%4d %s", i, SymbolName(i, MarketWatchOnly));
   }
   // intentionally asking for out-of-range to show an error
   PRTF(SymbolName(n, MarketWatchOnly)); // MARKET_UNKNOWN_SYMBOL(4301)
}
Below is an example log.
Total symbol count: 10
   0 EURUSD
   1 XAUUSD
   2 BTCUSD
   3 GBPUSD
   4 USDJPY
   5 USDCHF
   6 AUDUSD
   7 USDCAD
   8 NZDUSD
   9 USDRUB
SymbolName(n,MarketWatchOnly)= / MARKET_UNKNOWN_SYMBOL(4301)
6.1 .2 Editing the Market Watch list
Using the SymbolSelect function, the MQL program developer can add a specific symbol to Market
Watch or remove it from there.

---

## Page 1052

Part 6. Trading automation
1 052
6.1  Financial instruments and Market Watch
bool SymbolSelect(const string name, bool select)
The name parameter contains the name of the symbol being affected by this operation. Depending on
the value of the select parameter, a symbol is added to Market Watch (true) or removed from it.
Symbol names are case-sensitive: for example, "EURUSD.m" is not equal to "EURUSD.M".
The function returns an indication of success (true) or error (false). The error code can be found in
_ LastError.
A symbol cannot be removed if there are open charts or open positions for this symbol. In addition,
you cannot delete a symbol that is explicitly used in the formula for calculating a synthetic
(custom) instrument added to Market Watch.
It should be kept in mind that even if there are no open charts and positions for a symbol, it can be
indirectly used by MQL programs: for example, they can read its history of quotes or ticks. Removing
such a symbol may cause problems in these programs.
The following script SymbolRemoveUnused.mq5 is able to hide all symbols that are not used explicitly,
so it is recommended to check it on a demo account or save the current symbols set through the
context menu first.

---

## Page 1053

Part 6. Trading automation
1 053
6.1  Financial instruments and Market Watch
#include <MQL5Book/MqlError.mqh>
   
#define PUSH(A,V) (A[ArrayResize(A, ArraySize(A) + 1) - 1] = V)
   
void OnStart()
{
   // request user confirmation for deletion
   if(IDOK == MessageBox("This script will remove all unused symbols"
      " from the Market Watch. Proceed?", "Please, confirm", MB_OKCANCEL))
   {
      const int n = SymbolsTotal(true);
      ResetLastError();
      string removed[];
      // go through the symbols of the Market Watch in reverse order
      for(int i = n - 1; i >= 0; --i)
      {
         const string s = SymbolName(i, true);
         if(SymbolSelect(s, false))
         {
            // remember what was deleted
            PUSH(removed, s);
         }
         else
         {
            // in case of an error, display the reason
            PrintFormat("Can't remove '%s': %s (%d)", s, E2S(_LastError), _LastError);
         }
      }
      const int r = ArraySize(removed);
      PrintFormat("%d out of %d symbols removed", r, n);
      ArrayPrint(removed);
      ...
After the user confirms the analysis of the list of symbols, the program attempts to hide each symbol
sequentially by calling SymbolSelect(s, false). This only works for instruments that are not used
explicitly. The enumeration of symbols is performed in the reverse order so as not to violate the
indexing. All successfully removed symbols are collected in the removed array. The log displays
statistics and the array itself.
If Market Watch is changed, the user is then given the opportunity to restore all deleted symbols by
calling SymbolSelect(removed[i], true) in a loop.

---

## Page 1054

Part 6. Trading automation
1 054
6.1  Financial instruments and Market Watch
      if(r > 0)
      {
         // it is possible to return the deleted symbols back to the Market Watch
         // (at this point, the window displays a reduced list)
         if(IDOK == MessageBox("Do you want to restore removed symbols"
            " in the Market Watch?", "Please, confirm", MB_OKCANCEL))
         {
            int restored = 0;
            for(int i = r - 1; i >= 0; --i)
            {
               restored += SymbolSelect(removed[i], true);
            }
            PrintFormat("%d symbols restored", restored);
         }
      }
   }
}
Here's what the log output might look like.
Can't remove 'EURUSD': MARKET_SELECT_ERROR (4305)
Can't remove 'XAUUSD': MARKET_SELECT_ERROR (4305)
Can't remove 'BTCUSD': MARKET_SELECT_ERROR (4305)
Can't remove 'GBPUSD': MARKET_SELECT_ERROR (4305)
...
Can't remove 'USDRUB': MARKET_SELECT_ERROR (4305)
2 out of 10 symbols removed
"NZDUSD" "USDCAD"
2 symbols restored
Please note that although the symbols are restored in their original order, as they were in Market
Watch relative to each other, the addition occurs at the end of the list, after the remaining symbols.
Thus, all "busy" symbols will be at the beginning of the list, and all the restored will follow them. Such is
the specific operation of SymbolSelect: a symbol is always added to the end of the list, that is, it is
impossible to insert a symbol in a specific position. So, the rearrangement of the list elements is
available only for manual editing.
6.1 .3 Checking if a symbol exists
Instead of looking through the entire list of symbols, an MQL program can check for the presence of a
particular symbol by its name. For this purpose, there is the SymbolExist function.
bool SymbolExist(const string name, bool &isCustom)
In the name parameter, you should pass the name of the desired symbol. The isCustom parameter
passed by reference will be set by the function according to whether the specified symbol is standard
(false) or custom (true).
The function returns false if the symbol is not found in either the standard or custom symbols.
A partial analog of this function is the SYMBOL_EXIST property query.

---

## Page 1055

Part 6. Trading automation
1 055
6.1  Financial instruments and Market Watch
Let's analyze the simple script SymbolExists.mq5 to test this feature. In its parameter, the user can
specify the name, which is then passed to SymbolExist, and the result is logged. If an empty string is
input, the working symbol of the current chart will be checked. By default, the parameter is set to
"XYZ", which presumably does not match any of the available symbols.
#property script_show_inputs
   
input string SymbolToCheck = "XYZ";
   
void OnStart()
{
   const string _SymbolToCheck = SymbolToCheck == "" ? _Symbol : SymbolToCheck;
   bool custom = false;
   PrintFormat("Symbol '%s' is %s", _SymbolToCheck,
      (SymbolExist(_SymbolToCheck, custom) ? (custom ? "custom" : "standard") : "missing"));
}
When the script is run two times, first with the default value and then with an empty line on the
EURUSD chart, we will get the following entries in the log.
Symbol 'XYZ' is missing
Symbol 'EURUSD' is standard
If you already have custom symbols or create a new one with a simple calculation formula, you can
make sure the custom variable is populated. For example, if you open the Symbols window in the
terminal and press the Create symbol button, you can enter "SP500/FTSE1 00" (index names may
differ for your broker) in the Synthetic tool formula field and "GBPUSD.INDEX" in the field with the
Symbol name. A click on OK will create a custom instrument for which you can open a chart, and our
script should display the following on it:
Symbol 'GBPUSD.INDEX' is custom
When setting up your own symbol, do not forget to set not only the formula but also sufficiently "small"
values for the point size and the price change step (tick). Otherwise, the series of synthetic quotes may
turn out to be "stepped", or even degenerate into a straight line.
6.1 .4 Checking the symbol data relevance
Due to the distributed client-server architecture, client and server data may occasionally be different.
For example, this can happen immediately after the start of the terminal session, when the connection
is lost, or when the computer resources are heavily loaded. Also, the symbol will most likely remain out
of sync for some time immediately after it is added to the Market Watch. The MQL5 API allows you to
check the relevance of quote data for a particular symbol using the SymbolIsSynchronized function.
bool SymbolIsSynchronized(const string name)
The function returns true if the local data on the symbol named name is synchronized with the data on
the trade server.
The section Obtaining characteristics of price arrays, among other timeseries properties, introduced
the SERIES_SYNCHRONIZED property which returns an attribute of synchronization that is narrower in
its meaning: it applies to a specific combination of a symbol and a timeframe. In contrast to this
property, the SymbolIsSynchronized function returns an attribute of synchronization of the general
history for a symbol.

---

## Page 1056

Part 6. Trading automation
1 056
6.1  Financial instruments and Market Watch
The construction of all timeframes starts only after the completion of the history download. Due to the
multi-threaded architecture and parallel computing in the terminal, it might happen that
SymbolIsSynchronized will return true, and for a timeframe on the same symbol, the
SERIES_SYNCHRONIZED property will be temporarily equal to false.
Let's see how the new function works in the SymbolListSync.mq5 indicator. It is designed to periodically
check all symbols from Market Watch for synchronization. The check period is set by the user in
seconds in the SyncCheckupPeriod parameter. It causes the timer to start in OnInit.
#property indicator_chart_window
#property indicator_plots 0
   
input int SyncCheckupPeriod = 1; // SyncCheckupPeriod (seconds)
   
void OnInit()
{
   EventSetTimer(SyncCheckupPeriod);
}
In the OnTimer handler, in a loop, we call SymbolIsSynchronized and collect all unsynchronized symbols
into a common string, after which they are displayed in the comment and the log.
void OnTimer()
{
   string unsynced;
   const int n = SymbolsTotal(true);
   // check all symbols in the Market Watch
   for(int i = 0; i < n; ++i)
   {
      const string s = SymbolName(i, true);
      if(!SymbolIsSynchronized(s))
      {
         unsynced += s + "\n";
      }
   }
      
   if(StringLen(unsynced) > 0)
   {
      Comment("Unsynced symbols:\n" + unsynced);
      Print("Unsynced symbols:\n" + unsynced);
   }
   else
   {
      Comment("All Market Watch is in sync");
   }
}
For example, if we add some previously missing symbol (Brent) to the Market Watch, we get an entry
like this:

---

## Page 1057

Part 6. Trading automation
1 057
6.1  Financial instruments and Market Watch
Unsynced symbols:
Brent
Under normal conditions, most of the time (while the indicator is running) there should be no such
messages in the log. However, a flood of alerts may be generated during communication problems.
6.1 .5 Getting the last tick of a symbol
In the chapter about timeseries, in the Working with arrays of real ticks section, we introduced the
built-in structure MqlTick containing fields with price and volume values for a particular symbol, known
at the time of each change in quotes. In online mode, an MQL program can query the last received
prices and volumes using the SymbolInfoTick function that adopts the same structure.
bool SymbolInfoTick(const string symbol, MqlTick &tick)
For a symbol with a given name symbol, the function fills the tick structure passed by reference. If
successful, it returns true.
As you know, indicators and Expert Advisors are automatically called by the terminal upon the arrival of
a new tick, if they contain the description of the corresponding handlers OnCalculate and OnTick.
However, information about the meaning of price changes, the volume of the last trade, and the tick
generation time are not transferred directly to the handlers. More detailed information can be obtained
with the SymbolInfoTick function.
Tick events are generated only for a chart symbol, and therefore we have already considered the
option of obtaining our own multi-symbol event for ticks based on custom events. In this case,
SymbolInfoTick makes it possible to read information about ticks on third-party symbols about receiving
notifications.
Let's take the EventTickSpy.mq5 indicator and convert it to SymbolTickSpy.mq5, which will request the
MqlTick structure for the corresponding symbol on each "multicurrency" tick and then calculate and
display all spreads on the chart.
Let's add a new input parameter Index. It will be required for a new way of sending notifications: we will
send only the index of the changed symbol in the user event (see further along).
#define TICKSPY 0xFEED // 65261
input string SymbolList = 
   "EURUSD,GBPUSD,XAUUSD,USDJPY,USDCHF"; // List of symbols, comma separated (example)
input ushort Message = TICKSPY;          // Custom message id
input long Chart = 0;                    // Receiving chart id (do not edit)
input int Index = 0;                     // Index in symbol list (do not edit)
Also, we add the Spreads array to store spreads by symbols and the SelfIndex variable to remember
the position of the current chart's symbol in the list (if it is included in the list, which is usually so). The
latter is needed to call our new tick handling function from OnCalculate in the original copy of the
indicator. It is easier and more correct to take a ready-made index for _ Symbol explicitly and not send
it in an event back to ourselves.

---

## Page 1058

Part 6. Trading automation
1 058
6.1  Financial instruments and Market Watch
int Spreads[];
int SelfIndex = -1;
The introduced data structures are initialized in OnInit. Otherwise, OnInit remained unchanged,
including the launch of subordinate instances of the indicator on third-party symbols (these lines are
omitted here).
void OnInit()
{
   ...
         const int n = StringSplit(SymbolList, ',', Symbols);
         ArrayResize(Spreads, n);
         for(int i = 0; i < n; ++i)
         {
            if(Symbols[i] != _Symbol)
            {
               ...
            }
            else
            {
               SelfIndex = i;
            }
            Spreads[i] = 0;
         }
   ...
}
In the OnCalculate handler, we generate a custom event on each tick if the copy of the indicator works
on the other symbol (at the same time, the ID of the Chart chart to which notifications should be sent
is not equal to 0). Please note that the only parameter filled in the event is lparam which is equal to
Index (dparam is 0, and sparam is NULL). If Chart equals 0, this means we are in the main copy of the
indicator working on the chart symbol _ Symbol, and if it is found in the input symbol list, we call
directly OnSymbolTick with the corresponding SelfIndex index.

---

## Page 1059

Part 6. Trading automation
1 059
6.1  Financial instruments and Market Watch
int OnCalculate(const int rates_total, const int prev_calculated, const int, const double &price[])
{
   if(prev_calculated)
   {
      if(Chart > 0)
      {
         EventChartCustom(Chart, Message, Index, 0, NULL);
      }
      else if(SelfIndex > -1)
      {
         OnSymbolTick(SelfIndex);
      }
   }
  
   return rates_total;
}
In the receiving part of the event algorithm in OnChartEvent, we also call OnSymbolTick, but this time
we get the symbol number from the list in lparam (what was sent as the Index parameter from another
copy of the indicator).
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_CUSTOM + Message)
   {
      OnSymbolTick((int)lparam);
   }
}
The OnSymbolTick function requests full tick information using SymbolInfoTick and calculates the
spread as the difference between the Ask and Bid prices divided by the point size (the SYMBOL_POINT
property will be discussed later).

---

## Page 1060

Part 6. Trading automation
1 060
6.1  Financial instruments and Market Watch
void OnSymbolTick(const int index)
{
   const string symbol = Symbols[index];
   
   MqlTick tick;
   if(SymbolInfoTick(symbol, tick))
   {
      Spreads[index] = (int)MathRound((tick.ask - tick.bid)
         / SymbolInfoDouble(symbol, SYMBOL_POINT));
      string message = "";
      for(int i = 0; i < ArraySize(Spreads); ++i)
      {
         message += Symbols[i] + "=" + (string)Spreads[i] + "\n";
      }
      
      Comment(message);
   }
}
The new spread updates the corresponding cell in the Spreads array, after which the entire array is
displayed on the chart in the comment. Here's what it looks like.
Current spreads for the requested list of symbols
You can compare in real time the correspondence of the information in the comment and in the Market
Watch window.

---

## Page 1061

Part 6. Trading automation
1 061 
6.1  Financial instruments and Market Watch
6.1 .6 Schedules of trading and quoting sessions
A little later, in further chapters, we will discuss the MQL5 API functions that allow us to automate
trading operations. But first, we should study the technical features of the platform, which determine
the success of calling these APIs. In particular, some restrictions are imposed by the specifications of
financial instruments. In this chapter, we will gradually consider their programmatic analysis in full, and
we will start with such an item as sessions.
When trading financial instruments, it should be taken into account that many international markets,
such as stock exchanges, have predetermined opening hours, and information and trading are available
only during these hours. Despite the fact that the terminal is constantly connected to the broker's
server, an attempt to make a deal outside the working schedule will fail. In this regard, for each
symbol, the terminal stores a schedule of sessions, that is, the time periods within a day when certain
actions can be performed.
As you know, there are two main types of sessions: quoting and trading. During the quoting session, the
terminal receives (may receive) current quotes. During the trading session, it is allowed to send trade
orders and make deals. During the day, there may be several sessions of each type, with breaks (for
example, morning and evening). It is obvious that the duration of the quoting sessions is greater than or
equal to the trading ones.
In any case, session times, that is, opening and closing hours, are translated by the terminal from the
local time zone of the exchange to the broker's time zone (server time).
The MQL5 API allows you to find out the quoting and trading sessions of each instrument using the
SymbolInfoSessionQuote and SymbolInfoSessionTrade functions. In particular, this important
information allows the program to check whether the market is currently open before sending a trade
request to the server. Thus, we prevent the inevitable erroneous result and avoid unnecessary server
loads. Keep in mind that in the case of massive erroneous requests to the server due to an incorrectly
implemented MQL program, the server may begin to "ignore" your terminal, refusing to execute
subsequent commands (even correct ones) for some time.
bool SymbolInfoSessionQuote(const string symbol, ENUM_DAY_OF_WEEK dayOfWeek, uint
sessionIndex, datetime &from, datetime &to)
bool SymbolInfoSessionTrade(const string symbol, ENUM_DAY_OF_WEEK dayOfWeek, uint
sessionIndex, datetime &from, datetime &to)
The functions work in the same way. For a given symbol and day of the week dayOfWeek, they fill in the
from and to parameters passed by reference with the opening and closing times of the session with
sessionIndex. Session indexing starts from 0. The ENUM_DAY_OF_WEEK structure was described in the
section Enumerations.
There are no separate functions for querying the number of sessions: instead, we should be calling
SymbolInfoSessionQuote andSymbolInfoSessionTrade with increasing index sessionIndex, until the
function returns an error flag (false). When a session with the specified number exists, and the output
arguments from and to received correct values, the functions return a success indicator (true).
According to the MQL5 documentation, in the received values of from and to of type datetime, the date
should be ignored and only the time should be considered. This is because the information is an
intraday schedule. However, there is an important exception to this rule.
Since the market is potentially open 24 hours a day, as in the case of Forex, or an exchange on the
other side of the world, where daytime business hours coincide with the change of dates in your
broker's "timezone", the end of sessions can have a time equal to or greater than 24 hours. For

---

## Page 1062

Part 6. Trading automation
1 062
6.1  Financial instruments and Market Watch
example, if the start of Forex sessions is 00:00, then the end is 24:00. However, from the point of view
of the datetime type, 24 hours is 00 hours 00 minutes the very next day.
The situation becomes more confusing for those exchanges, where the schedule is shifted relative to
your broker's time zone by several hours in such a way that the session starts on one day and ends on
another. Because of this, the to variable registers not only time but also an extra day that cannot be
ignored, because otherwise intraday time from will be more than intraday time to (for example, a
session can last from 21 :00 today to 8:00 tomorrow, that is, 21  > 8). In this case, the check for the
occurrence of the current time inside the session ("time x is greater than the start and less than the
end") will turn out to be incorrect (for example, the condition x >= 21  && x < 8 is not fulfilled for x =
23, although the session is actually active).
Thus, we come to the conclusion that it is impossible to ignore the date in the from/to parameters, and
this point should be taken into account in the algorithms (see example).
To demonstrate the capabilities of the functions, let's return to an example of the script
EnvPermissions.mq5 which was presented in the Permissions section. One of the types of permissions
(or restrictions, if you like) refers specifically to the availability of trading. Earlier, the script took into
account the terminal settings (TERMINAL_TRADE_ALLOWED) and the settings of a specific MQL
program (MQL_TRADE_ALLOWED). Now we can add to it session checks to determine the trading
permissions that are valid at a given moment for a particular symbol.
The new version of the script is called SymbolPermissions.mq5. It is also not final: in one of the following
chapters, we will study the limitations imposed by the trading account settings.
Recall that the script implements the class Permissions, which provides a centralized description of all
types of permissions/restrictions applicable to MQL programs. Among other things, the class has
methods for checking the availability of trading: isTradeEnabled and isTradeOnSymbolEnabled. The first
of these relates to global permissions and will remain almost unchanged:
class Permissions
{
public:
   static bool isTradeEnabled(const string symbol = NULL, const datetime now = 0)
   {
      return TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)
          && MQLInfoInteger(MQL_TRADE_ALLOWED)
          && isTradeOnSymbolEnabled(symbol == NULL ? _Symbol : symbol, now);
   }
   ...
After checking the properties of the terminal and the MQL program, the script proceeds to
isTradeOnSymbolEnabled where the symbol specification is analyzed. Previously, this method was
practically empty.
In addition to the working symbol passed in the symbol parameter, the isTradeOnSymbolEnabled
function receives the current time (now) and the required trading mode (mode). We will discuss the
latter in more detail in the following sections (see Trading permissions). For now, let's just note that the
default value of SYMBOL_TRADE_MODE_FULL gives maximum freedom (all trading operations are
allowed).

---

## Page 1063

Part 6. Trading automation
1 063
6.1  Financial instruments and Market Watch
   static bool isTradeOnSymbolEnabled(string symbol, const datetime now = 0,
      const ENUM_SYMBOL_TRADE_MODE mode = SYMBOL_TRADE_MODE_FULL)
   {
      // checking sessions
      bool found = now == 0;
      if(!found)
      {
         const static ulong day = 60 * 60 * 24;
         const ulong time = (ulong)now % day;
         datetime from, to;
         int i = 0;
         
         ENUM_DAY_OF_WEEK d = TimeDayOfWeek(now);
         
         while(!found && SymbolInfoSessionTrade(symbol, d, i++, from, to))
         {
            found = time >= (ulong)from && time < (ulong)to;
         }
      }
      // checking the trading mode for the symbol
      return found && (SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) == mode);
   }
If the now time is not specified (it is equal to 0 by default), we consider that we are not interested in
sessions. This means that the found variable with an indication that a suitable session has been found
(that is, a session containing the given time) is immediately set to true. But if the now parameter is
specified, the function gets into the trading session analysis block.
To extract time without taking into account the date from the values of the datetime type, we describe
the day constant equal to the number of seconds in a day. An expression like now % day will return the
remainder of dividing the full date and time by the duration of one day, which will give only the time
(the most significant digits in datetime will be null).
The TimeDayOfWeek function returns the day of the week for the given datetime value. It is located in
the MQL5Book/DateTime.mqh header file which we have already used before (see Date and time).
Further in the while loop, we call the SymbolInfoSessionTrade function while constantly incrementing
the session index i until a suitable session is found or the function returns false (no more sessions).
Thus, the program can get a complete list of sessions by day of the week, similar to what is displayed
in the terminal in the symbol Specifications window.
Obviously, a suitable session is the one that contains the specified time value between the session
beginning from and end to times. It is here that we take into account the problem associated with the
possible round-the-clock trading: from and to are compared against time "as is", without discarding the
day (from % day or to % day).
Once found becomes equal to true, we exit the loop. Otherwise, the loop will end when the allowed
number of sessions is exceeded (function SymbolInfoSessionTrade will return false) and a suitable
session will never be found.
If, according to the session schedule, trading is now allowed, we additionally check the trading mode
for the symbol (SYMBOL_TRADE_MODE). For example, symbol trading can be completely prohibited
("indicative") or be in the "only closing positions" mode.

---

## Page 1064

Part 6. Trading automation
1 064
6.1  Financial instruments and Market Watch
The above code has some simplifications compared to the final version in the file
SymbolPermissions.mq5. It additionally implements a mechanism for marking the source of the
restriction that caused the trade to be disabled. All such sources are summarized in the
TRADE_RESTRICTIONS enumeration.
   enum TRADE_RESTRICTIONS
   {
      TERMINAL_RESTRICTION = 1,
      PROGRAM_RESTRICTION = 2,
      SYMBOL_RESTRICTION = 4,
      SESSION_RESTRICTION = 8,
   };
At the moment, the restriction can come from 4 instances: the terminal, the program, the symbol, and
the session schedule. We will add more options later.
To register the fact that a constraint was found in the Permissions class, we have the
lastFailReasonBitMask variable which allows the collection of a bit mask from the elements of the
enumeration using an auxiliary method pass (the bit is set up when the checked condition value is false,
and the bit equals false).
   static uint lastFailReasonBitMask;
   static bool pass(const bool value, const uint bitflag) 
   {
      if(!value) lastFailReasonBitMask |= bitflag;
      return value;
   }
Calling the pass method with a specific flag is done at the appropriate validation steps. For example, the
isTradeEnabled method in full looks like this:
   static bool isTradeEnabled(const string symbol = NULL, const datetime now = 0)
   {
      lastFailReasonBitMask = 0;
      return pass(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED), TERMINAL_RESTRICTION)
          && pass(MQLInfoInteger(MQL_TRADE_ALLOWED), PROGRAM_RESTRICTION)
          && isTradeOnSymbolEnabled(symbol == NULL ? _Symbol : symbol, now);
   }
Due to this, with a negative result of the call TerminalInfoInteger(TERMINAL_ TRADE_ ALLOWED) or
MQLInfoInteger(MQL_ TRADE_ ALLOWED), either the TERMINAL_RESTRICTION or the
PROGRAM_RESTRICTION flag will be set, respectively.
The isTradeOnSymbolEnabled method also sets its own flags when problems are detected, including
session flags.

---

## Page 1065

Part 6. Trading automation
1 065
6.1  Financial instruments and Market Watch
   static bool isTradeOnSymbolEnabled(string symbol, const datetime now = 0,
      const ENUM_SYMBOL_TRADE_MODE mode = SYMBOL_TRADE_MODE_FULL)
   {
      ...
      return pass(found, SESSION_RESTRICTION)
         && pass(SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) == mode, SYMBOL_RESTRICTION);
   }
As a result, the MQL program that is using the query Permissions::isTradeEnabled, after receiving a
restriction, may clarify its meaning using the getFailReasonBitMask and explainBitMask methods: the
first one returns the mask of set prohibition flags "as is", and the second one forms a user-friendly text
description of the restrictions.
   static uint getFailReasonBitMask()
   {
      return lastFailReasonBitMask;
   }
   
   static string explainBitMask()
   {
      string result = "";
      for(int i = 0; i < 4; ++i)
      {
         if(((1 << i) & lastFailReasonBitMask) != 0)
         {
            result += EnumToString((TRADE_RESTRICTIONS)(1 << i));
         }
      }
      return result;
   }
With the above Permissions class in the OnStart handler, a check is made for the availability of trading
for all symbols from the Market Watch (currently,TimeCurrent).

---

## Page 1066

Part 6. Trading automation
1 066
6.1  Financial instruments and Market Watch
void OnStart()
{
   string disabled = "";
   
   const int n = SymbolsTotal(true);
   for(int i = 0; i < n; ++i)
   {
      const string s = SymbolName(i, true);
      if(!Permissions::isTradeEnabled(s, TimeCurrent()))
      {
         disabled += s + "=" + Permissions::explainBitMask() +"\n";
      }
   }
   if(disabled != "")
   {
      Print("Trade is disabled for the following symbols and origins:");
      Print(disabled);
   }
}
If trading is prohibited for a certain symbol, we will see an explanation in the log.
   Trade is disabled for following symbols and origins:
   USDRUB=SESSION_RESTRICTION
   SP500m=SYMBOL_RESTRICTION
In this case, the market is closed for "USDRUB" and trading is disabled for the "SP500m" symbol
(more strictly, it does not correspond to the SYMBOL_TRADE_MODE_FULL mode).
It is assumed that when running the script, algorithmic trading was enabled globally in the terminal.
Otherwise, we will additionally see TERMINAL_RESTRICTION and PROGRAM_RESTRICTION prohibitions
in the log.
6.1 .7 Symbol margin rates
Among the characteristics of the symbol specification available in the MQL5 API, which we will discuss
in detail in further sections, there are several characteristics related to margin requirements, which
apply when opening and maintaining trading positions. Due to the fact that the terminal provides trading
on different markets and different types of instruments, these requirements may vary significantly. In a
generalized form, this is expressed in the application of margin correction rates that are set individually
for symbols and different types of trading operations. For the user, the rates are displayed in the
terminal in the Specifications window.
As we will see below, the multiplier (if applied) is multiplied by the margin value from the symbol
properties. The margin ratio can be obtained programmatically using the SymbolInfoMarginRate
function.
bool SymbolInfoMarginRate(const string symbol, ENUM_ORDER_TYPE orderType, double &initial,
double &maintenance)
For the specified symbol and order type (ENUM_ORDER_TYPE), the function fills in passed by reference
initial and maintenance parameters with initial and maintaining margin ratios, respectively. The
resulting rates should be multiplied by the margin value of the corresponding type (how to request it is

---

## Page 1067

Part 6. Trading automation
1 067
6.1  Financial instruments and Market Watch
described in the section on margin requirements) to get the amount that will be reserved in the
account when placing an order such as orderType.
The function returns true in case of successful execution.
Let's use as an example a simple script SymbolMarginRate.mq5, which outputs margin ratios for Market
Watch or all available symbols, depending on the MarketWatchOnly parameter. The operation type can
be specified in the OrderType parameter.
#include <MQL5Book/MqlError.mqh>
   
input bool MarketWatchOnly = true;
input ENUM_ORDER_TYPE OrderType = ORDER_TYPE_BUY;
   
void OnStart()
{
   const int n = SymbolsTotal(MarketWatchOnly);
   PrintFormat("Margin rates per symbol for %s:", EnumToString(OrderType));
   for(int i = 0; i < n; ++i)
   {
      const string s = SymbolName(i, MarketWatchOnly);
      double initial = 1.0, maintenance = 1.0;
      if(!SymbolInfoMarginRate(s, OrderType, initial, maintenance))
      {
         PrintFormat("Error: %s(%d)", E2S(_LastError), _LastError);
      }
      PrintFormat("%4d %s = %f %f", i, s, initial, maintenance);
   }
}
Below is the log.
Margin rates per symbol for ORDER_TYPE_BUY:
   0 EURUSD = 1.000000 0.000000
   1 XAUUSD = 1.000000 0.000000
   2 BTCUSD = 0.330000 0.330000
   3 USDCHF = 1.000000 0.000000
   4 USDJPY = 1.000000 0.000000
   5 AUDUSD = 1.000000 0.000000
   6 USDRUB = 1.000000 1.000000
You can compare the received values with the symbol specifications in the terminal.
6.1 .8 Overview of functions for getting symbol properties
The complete specification of each symbol can be obtained by querying its properties: for this purpose,
the MQL5 API provides three functions, namely SymbolInfoInteger, SymbolInfoDouble, and
SymbolInfoString, each of which is responsible for the properties of a particular type. The properties
are described as members of three enumerations: ENUM_SYMBOL_INFO_INTEGER,
ENUM_SYMBOL_INFO_DOUBLE, and ENUM_SYMBOL_INFO_STRING, respectively. A similar technique
is used in the chart and object APIs we already know.
The name of the symbol and the identifier of the requested property are passed to any of the functions.

---

## Page 1068

Part 6. Trading automation
1 068
6.1  Financial instruments and Market Watch
Each of the functions is presented in two forms: abbreviated and full. The abbreviated version directly
returns the requested property, while the full one writes it to the out parameter passed by reference.
For example, for properties that are compatible with an integer type, the functions have prototypes like
this:
long SymbolInfoInteger(const string symbol, ENUM_SYMBOL_INFO_INTEGER property)
bool SymbolInfoInteger(const string symbol, ENUM_SYMBOL_INFO_INTEGER property, long &value)
The second form returns a boolean indicator of success (true) or error (false). The most possible
reasons why a function might return false include an invalid symbol name
(MARKET_UNKNOWN_SYMBOL, 4301 ) or an invalid identifier for the requested property
(MARKET_WRONG_PROPERTY, 4303). The details are provided in _ LastError.
As before, the properties in the ENUM_SYMBOL_INFO_INTEGER enumeration are of various integer-
compatible types: bool, int, long, color, datetime, and special enumerations (all of which will be
discussed in separate sections).
For properties with a real number type, the following two forms of the SymbolInfoDouble function are
defined.
double SymbolInfoDouble(const string symbol, ENUM_SYMBOL_INFO_DOUBLE property)
bool SymbolInfoDouble(const string symbol, ENUM_SYMBOL_INFO_DOUBLE property, double &value)
Finally, for string properties, similar functions look like this:
string SymbolInfoString(const string symbol, ENUM_SYMBOL_INFO_STRING property)
bool SymbolInfoString(const string symbol, ENUM_SYMBOL_INFO_STRING property, string &value)
The properties of various types which will be often used later when developing Expert Advisors are
logically grouped in the descriptions of the following sections of this chapter.
Based on the above functions, we will create a universal class SymbolMonitor (fileSymbolMonitor.mqh)
to get any symbol properties. It will be based on a set of overloaded get methods for three
enumerations.

---

## Page 1069

Part 6. Trading automation
1 069
6.1  Financial instruments and Market Watch
class SymbolMonitor
{
public:
   const string name;
   SymbolMonitor(): name(_Symbol) { }
   SymbolMonitor(const string s): name(s) { }
   
   long get(const ENUM_SYMBOL_INFO_INTEGER property) const
   {
      return SymbolInfoInteger(name, property);
   }
   
   double get(const ENUM_SYMBOL_INFO_DOUBLE property) const
   {
      return SymbolInfoDouble(name, property);
   }
   
   string get(const ENUM_SYMBOL_INFO_STRING property) const
   {
      return SymbolInfoString(name, property);
   }
   ...
The other three similar methods make it possible to eliminate the enumeration type in the first
parameter and select the necessary overload by the compiler due to the second dummy parameter (its
type here always matches the result type). We will use this in future template classes.
   long get(const int property, const long) const
   {
      return SymbolInfoInteger(name, (ENUM_SYMBOL_INFO_INTEGER)property);
   }
   double get(const int property, const double) const
   {
      return SymbolInfoDouble(name, (ENUM_SYMBOL_INFO_DOUBLE)property);
   }
   string get(const int property, const string) const
   {
      return SymbolInfoString(name, (ENUM_SYMBOL_INFO_STRING)property);
   }
   ...
Thus, by creating an object with the desired symbol name, you can uniformly query its properties of
any type. To query and log all properties of the same type, we could implement something like this.

---

## Page 1070

Part 6. Trading automation
1 070
6.1  Financial instruments and Market Watch
   // project (draft)
   template<typename E,typename R>
   void list2log()
   {
      E e = (E)0;
      int array[];
      const int n = EnumToArray(e, array, 0, USHORT_MAX);
      for(int i = 0; i < n; ++i)
      {
         e = (E)array[i];
         R r = get(e);
         PrintFormat("% 3d %s=%s", i, EnumToString(e), (string)r);
      }
   }
However, due to the fact that in properties of the type long values of other types are actually "hidden",
which should be displayed in a specific way (for example, by calling EnumToString for enumerations,
imeToString for date and time, etc.), it makes sense to define another three overloaded methods that
would return a string representation of the property. Let's call them stringify. Then in the above list2log
draft, it is possible to use stringify instead of casting values to (string), and the method itself will
eliminate one template parameter.
   template<typename E>
   void list2log()
   {
      E e = (E)0;
      int array[];
      const int n = EnumToArray(e, array, 0, USHORT_MAX);
      for(int i = 0; i < n; ++i)
      {
         e = (E)array[i];
         PrintFormat("% 3d %s=%s", i, EnumToString(e), stringify(e));
      }
   }
For real and string types, the implementation of stringify looks pretty straightforward.
   string stringify(const ENUM_SYMBOL_INFO_DOUBLE property, const string format = NULL) const
   {
      if(format == NULL) return (string)SymbolInfoDouble(name, property);
      return StringFormat(format, SymbolInfoDouble(name, property));
   }
   
   string stringify(const ENUM_SYMBOL_INFO_STRING property) const
   {
      return SymbolInfoString(name, property);
   }
But for ENUM_SYMBOL_INFO_INTEGER, everything is a little more complicated. Of course, when the
property is of type long or int, it is enough to cast it to (string). All other cases need to be individually
analyzed and converted within the switch operator.

---

## Page 1071

Part 6. Trading automation
1 071 
6.1  Financial instruments and Market Watch
   string stringify(const ENUM_SYMBOL_INFO_INTEGER property) const
   {
      const long v = SymbolInfoInteger(name, property);
      switch(property)
      {
         ...
      }
      
      return (string)v;
   }
For example, if a property has a boolean type, it is convenient to represent it with the string "true" or
"false" (thus it will be visually different from the simple numbers 1  and 0). Looking ahead, for the sake
of giving an example, let's say that among the properties there is SYMBOL_EXIST, which is equivalent
to the SymbolExist function, that is, returning a boolean indication of whether the specified character
exists. For its processing and other logical properties, it makes sense to implement an auxiliary method
boolean.
   static string boolean(const long v)
   {
      return v ? "true" : "false";
   }
   
   string stringify(const ENUM_SYMBOL_INFO_INTEGER property) const
   {
      const long v = SymbolInfoInteger(name, property);
      switch(property)
      {
         case SYMBOL_EXIST:
            return boolean(v);
         ...
      }
      
      return (string)v;
   }
For properties that are enumerations, the most appropriate solution would be a template method using
the EnumToString function.
   template<typename E>
   static string enumstr(const long v)
   {
      return EnumToString((E)v);
   }
For example, the SYMBOL_SWAP_ROLLOVER3DAYS property determines on which day of the week a
triple swap is charged on open positions for a symbol, and this property has the type
ENUM_DAY_OF_WEEK. So, to process it, we can write the following inside switch:
         case SYMBOL_SWAP_ROLLOVER3DAYS:
            return enumstr<ENUM_DAY_OF_WEEK>(v);
A special case is presented by properties whose values are combinations of bit flags. In particular, for
each symbol, the broker sets permissions for orders of specific types, such as market, limit, stop loss,

---

## Page 1072

Part 6. Trading automation
1 072
6.1  Financial instruments and Market Watch
take profit, and others (we will consider these permissions separately). Each type of order is denoted
by a constant with one bit enabled, so their superposition (combined by the bitwise OR operator '| ') is
stored in the SYMBOL_ORDER_MODE property, and in the absence of restrictions, all bits are enabled at
the same time. For such properties, we will define our own enumerations in our header file, for example:
enum SYMBOL_ORDER
{
   _SYMBOL_ORDER_MARKET = 1,
   _SYMBOL_ORDER_LIMIT = 2,
   _SYMBOL_ORDER_STOP = 4,
   _SYMBOL_ORDER_STOP_LIMIT = 8,
   _SYMBOL_ORDER_SL = 16,
   _SYMBOL_ORDER_TP = 32,
   _SYMBOL_ORDER_CLOSEBY = 64,
};
Here, for each built-in constant, such as SYMBOL_ORDER_MARKET, a corresponding element is
declared, whose identifier is the same as the constant but is preceded by an underscore to avoid
naming conflicts.
To represent combinations of flags from such enumerations in the form of a string, we implement
another template method, maskstr.
   template<typename E>
   static string maskstr(const long v)
   {
      string text = "";
      for(int i = 0; ; ++i)
      {
         ResetLastError();
         const string s = EnumToString((E)(1 << i));
         if(_LastError != 0)
         {
            break;
         }
         if((v & (1 << i)) != 0)
         {
            text += s + " ";
         }
      }
      return text;
   }
Its meaning is like enumstr, but the function EnumToString is called for each enabled bit in the property
value, after which the resulting strings are "glued".
Now processing SYMBOL_ORDER_MODE in the statement switch is possible in a similar way:
         case SYMBOL_ORDER_MODE:
            return maskstr<SYMBOL_ORDER>(v);
Here is the full code of the stringify method for ENUM_SYMBOL_INFO_INTEGER. With all the properties
and enumerations, we will gradually get acquainted in the following sections.

---

## Page 1073

Part 6. Trading automation
1 073
6.1  Financial instruments and Market Watch
   string stringify(const ENUM_SYMBOL_INFO_INTEGER property) const
   {
      const long v = SymbolInfoInteger(name, property);
      switch(property)
      {
         case SYMBOL_SELECT:
         case SYMBOL_SPREAD_FLOAT:
         case SYMBOL_VISIBLE:
         case SYMBOL_CUSTOM:
         case SYMBOL_MARGIN_HEDGED_USE_LEG:
         case SYMBOL_EXIST:
            return boolean(v);
         case SYMBOL_TIME:
            return TimeToString(v, TIME_DATE|TIME_SECONDS);
         case SYMBOL_TRADE_CALC_MODE:   
            return enumstr<ENUM_SYMBOL_CALC_MODE>(v);
         case SYMBOL_TRADE_MODE:
            return enumstr<ENUM_SYMBOL_TRADE_MODE>(v);
         case SYMBOL_TRADE_EXEMODE:
            return enumstr<ENUM_SYMBOL_TRADE_EXECUTION>(v);
         case SYMBOL_SWAP_MODE:
            return enumstr<ENUM_SYMBOL_SWAP_MODE>(v);
         case SYMBOL_SWAP_ROLLOVER3DAYS:
            return enumstr<ENUM_DAY_OF_WEEK>(v);
         case SYMBOL_EXPIRATION_MODE:
            return maskstr<SYMBOL_EXPIRATION>(v);
         case SYMBOL_FILLING_MODE:
            return maskstr<SYMBOL_FILLING>(v);
         case SYMBOL_START_TIME:
         case SYMBOL_EXPIRATION_TIME:
            return TimeToString(v);
         case SYMBOL_ORDER_MODE:
            return maskstr<SYMBOL_ORDER>(v);
         case SYMBOL_OPTION_RIGHT:
            return enumstr<ENUM_SYMBOL_OPTION_RIGHT>(v);
         case SYMBOL_OPTION_MODE:
            return enumstr<ENUM_SYMBOL_OPTION_MODE>(v);
         case SYMBOL_CHART_MODE:
            return enumstr<ENUM_SYMBOL_CHART_MODE>(v);
         case SYMBOL_ORDER_GTC_MODE:
            return enumstr<ENUM_SYMBOL_ORDER_GTC_MODE>(v);
         case SYMBOL_SECTOR:
            return enumstr<ENUM_SYMBOL_SECTOR>(v);
         case SYMBOL_INDUSTRY:
            return enumstr<ENUM_SYMBOL_INDUSTRY>(v);
         case SYMBOL_BACKGROUND_COLOR: // Bytes: Transparency Blue Green Red
            return StringFormat("TBGR(0x%08X)", v);
      }
      
      return (string)v;
   }

---

## Page 1074

Part 6. Trading automation
1 074
6.1  Financial instruments and Market Watch
To test the SymbolMonitor class, we have created a simple script SymbolMonitor.mq5. It logs all the
properties of the working chart symbol.
#include <MQL5Book/SymbolMonitor.mqh>
   
void OnStart()
{
   SymbolMonitor m;
   m.list2log<ENUM_SYMBOL_INFO_INTEGER>();
   m.list2log<ENUM_SYMBOL_INFO_DOUBLE>();
   m.list2log<ENUM_SYMBOL_INFO_STRING>();
}
For example, if we run the script on the EURUSD chart, we can get the following records (given in a
shortened form).

---

## Page 1075

Part 6. Trading automation
1 075
6.1  Financial instruments and Market Watch
ENUM_SYMBOL_INFO_INTEGER Count=36
  0 SYMBOL_SELECT=true
  ...
  4 SYMBOL_TIME=2022.01.12 10:52:22
  5 SYMBOL_DIGITS=5
  6 SYMBOL_SPREAD=0
  7 SYMBOL_TICKS_BOOKDEPTH=10
  8 SYMBOL_TRADE_CALC_MODE=SYMBOL_CALC_MODE_FOREX
  9 SYMBOL_TRADE_MODE=SYMBOL_TRADE_MODE_FULL
 10 SYMBOL_TRADE_STOPS_LEVEL=0
 11 SYMBOL_TRADE_FREEZE_LEVEL=0
 12 SYMBOL_TRADE_EXEMODE=SYMBOL_TRADE_EXECUTION_INSTANT
 13 SYMBOL_SWAP_MODE=SYMBOL_SWAP_MODE_POINTS
 14 SYMBOL_SWAP_ROLLOVER3DAYS=WEDNESDAY
 15 SYMBOL_SPREAD_FLOAT=true
 16 SYMBOL_EXPIRATION_MODE=_SYMBOL_EXPIRATION_GTC _SYMBOL_EXPIRATION_DAY »
    _SYMBOL_EXPIRATION_SPECIFIED _SYMBOL_EXPIRATION_SPECIFIED_DAY
 17 SYMBOL_FILLING_MODE=_SYMBOL_FILLING_FOK
 ... 
 23 SYMBOL_ORDER_MODE=_SYMBOL_ORDER_MARKET _SYMBOL_ORDER_LIMIT _SYMBOL_ORDER_STOP »
    _SYMBOL_ORDER_STOP_LIMIT _SYMBOL_ORDER_SL _SYMBOL_ORDER_TP _SYMBOL_ORDER_CLOSEBY
 ... 
 26 SYMBOL_VISIBLE=true
 27 SYMBOL_CUSTOM=false
 28 SYMBOL_BACKGROUND_COLOR=TBGR(0xFF000000)
 29 SYMBOL_CHART_MODE=SYMBOL_CHART_MODE_BID
 30 SYMBOL_ORDER_GTC_MODE=SYMBOL_ORDERS_GTC
 31 SYMBOL_MARGIN_HEDGED_USE_LEG=false
 32 SYMBOL_EXIST=true
 33 SYMBOL_TIME_MSC=1641984742149
 34 SYMBOL_SECTOR=SECTOR_CURRENCY
 35 SYMBOL_INDUSTRY=INDUSTRY_UNDEFINED
ENUM_SYMBOL_INFO_DOUBLE Count=57
  0 SYMBOL_BID=1.13681
  1 SYMBOL_BIDHIGH=1.13781
  2 SYMBOL_BIDLOW=1.13552
  3 SYMBOL_ASK=1.13681
  4 SYMBOL_ASKHIGH=1.13781
  5 SYMBOL_ASKLOW=1.13552
 ...
 12 SYMBOL_POINT=1e-05
 13 SYMBOL_TRADE_TICK_VALUE=1.0
 14 SYMBOL_TRADE_TICK_SIZE=1e-05
 15 SYMBOL_TRADE_CONTRACT_SIZE=100000.0
 16 SYMBOL_VOLUME_MIN=0.01
 17 SYMBOL_VOLUME_MAX=500.0
 18 SYMBOL_VOLUME_STEP=0.01
 19 SYMBOL_SWAP_LONG=-0.7
 20 SYMBOL_SWAP_SHORT=-1.0
 21 SYMBOL_MARGIN_INITIAL=0.0
 22 SYMBOL_MARGIN_MAINTENANCE=0.0

---

## Page 1076

Part 6. Trading automation
1 076
6.1  Financial instruments and Market Watch
 ...
 28 SYMBOL_TRADE_TICK_VALUE_PROFIT=1.0
 29 SYMBOL_TRADE_TICK_VALUE_LOSS=1.0
 ...
 43 SYMBOL_MARGIN_HEDGED=100000.0
 ...
 47 SYMBOL_PRICE_CHANGE=0.0132
ENUM_SYMBOL_INFO_STRING Count=15
  0 SYMBOL_BANK=
  1 SYMBOL_DESCRIPTION=Euro vs US Dollar
  2 SYMBOL_PATH=Forex\EURUSD
  3 SYMBOL_CURRENCY_BASE=EUR
  4 SYMBOL_CURRENCY_PROFIT=USD
  5 SYMBOL_CURRENCY_MARGIN=EUR
 ...
 13 SYMBOL_SECTOR_NAME=Currency
In particular, you can see that the symbol prices are broadcast with 5 digits (SYMBOL_DIGITS), the
symbol does exist (SYMBOL_EXIST), the contract size is 1 00000.0
(SYMBOL_TRADE_CONTRACT_SIZE), etc. All information corresponds to the specification.
6.1 .9 Checking symbol status
Earlier we looked at several functions related to the status of a symbol. Recall that SymbolExist is used
to check for the existence of a symbol, and SymbolSelect is used to check for inclusion or exclusion
from the Market Watch list. Among the properties of the symbol, there are several flags similar in
purpose, the use of which has both pluses and minuses compared to the above functions.
In particular, the SYMBOL_SELECT property allows you to find out if the specified symbol is selected in
Market Watch, while the SymbolSelect function changes this property.
The SymbolExist function, unlike the similar SYMBOL_EXIST property, additionally populates the output
variable with an indication that the symbol is a user-defined one. When querying properties, it would be
necessary to analyze these two attributes separately, since the attribute of the custom symbol is
stored in another property, SYMBOL_CUSTOM. However, in some cases, the program may need only
one property, and then the possibility of a separate query becomes a plus.
All flags are boolean values obtained through the SymbolInfoInteger function.
Identifier
Description
SYMBOL_EXIST
Indicates that a symbol with the given name exists
SYMBOL_SELECT
Indicates that the symbol is selected in Market W atch
SYMBOL_VISIBLE
Indicates that the specified symbol is displayed in Market W atch
Of particular interest is SYMBOL_VISIBLE. The fact is that some symbols (as a rule, these are cross
rates that are necessary for calculating margin requirements and profit in the deposit currency) are
selected in Market Watch automatically and are not displayed in the list visible to the user. Such
symbols must be explicitly chosen (by the user or programmatically) to be displayed. Thus, it is the
SYMBOL_VISIBLE property that allows you to determine whether a symbol is visible in the window: it

---

## Page 1077

Part 6. Trading automation
1 077
6.1  Financial instruments and Market Watch
can be equal to false for some elements of the list, obtained using a pair of functions SymbolsTotal and
SymbolName with the selected parameter equal to true.
Consider a simple script (SymbolInvisible.mq5), which searches the terminal for implicitly selected
symbols, that is, those that are not displayed in the Market Watch (SYMBOL_VISIBLE is reset) while
SYMBOL_SELECT for them is equal to true.
#define PUSH(A,V) (A[ArrayResize(A, ArraySize(A) + 1) - 1] = V)
   
void OnStart()
{
   const int n = SymbolsTotal(false);
   int selected = 0;
   string invisible[];
   // loop through all available symbols 
   for(int i = 0; i < n; ++i)
   {
      const string s = SymbolName(i, false);
      if(SymbolInfoInteger(s, SYMBOL_SELECT))
      {
         selected++;
         if(!SymbolInfoInteger(s, SYMBOL_VISIBLE))
         {
            // collect selected but invisible symbols into an array 
            PUSH(invisible, s);
         }
      }
   }
   PrintFormat("Symbols: total=%d, selected=%d, implicit=%d",
      n, selected, ArraySize(invisible));
   if(ArraySize(invisible))
   {
      ArrayPrint(invisible);
   }
}
Try compiling and running the script on different accounts. The situation when a symbol is implicitly
selected is not always encountered. For example, if in Market Watch tickers of Russian blue chips that
are quoted in rubles are selected, and the trading account is in a different currency (for example,
dollars or euros, but not rubles), then the USDRUB symbol will be automatically selected. Of course,
this assumes that it has not been previously added to the Market Watch explicitly. Then we get the
following result in the log:
Symbols: total=50681, selected=49, implicit=1
"USDRUB"
6.1 .1 0 Price type for building symbol charts
Bars on MetaTrader 5 price charts can be plotted based on Bid or Last prices, and the plotting type is
indicated in the specification of each instrument. An MQL program can find this characteristic by calling
the SymbolInfoInteger function for the SYMBOL_CHART_MODE property. The return value is a member
of the ENUM_SYMBOL_CHART_MODE enumeration.

---

## Page 1078

Part 6. Trading automation
1 078
6.1  Financial instruments and Market Watch
Identifier
Description
SYMBOL_CHART_MODE_BID
Bars are built at Bid prices
SYMBOL_CHART_MODE_LAST
Bars are built at Last prices
The mode with Last prices is used for symbols traded on exchanges (as opposed to the decentralized
Forex market), and the Depth of Market is available for such symbols. The depth of the market can be
found based on the SYMBOL_TICKS_BOOKDEPTH property.
The SYMBOL_CHART_MODE property is useful for adjusting the signals of indicators or strategies that
are built, for example, at the chart's Last prices, while orders will be executed "at the market price",
that is, at Ask or Bid prices depending on direction.
Also, the price type is required when calculating bars of the custom instrument: if it depends on
standard symbols, it may make sense to consider their settings by price type. When the user enters
the formula of the synthetic instrument in the Custom Symbol window (opened by selecting Create
Symbol in the Symbols dialogue), it is possible to select price types according to the specifications of
the respective standard symbols used. However, when the calculation algorithm is formed in an MQL
program, precisely it is responsible for the correct choice of the price type.
First, let's collect statistics on the use of Bid and Last prices to build charts on a specific account. This
is what the script SymbolStatsByPriceType.mq5 will do.
const bool MarketWatchOnly = false;
   
void OnStart()
{
   const int n = SymbolsTotal(MarketWatchOnly);
   int k = 0;
   // loop through all available characters
   for(int i = 0; i < n; ++i)
   {
      if(SymbolInfoInteger(SymbolName(i, MarketWatchOnly), SYMBOL_CHART_MODE)
          == SYMBOL_CHART_MODE_LAST)
      {
         k++;
      }
   }
   PrintFormat("Symbols in total: %d", n);
   PrintFormat("Symbols using price types: Bid=%d, Last=%d", n - k, k);
}
Try it on different accounts (some may not have stock symbols). Here's what the result might look like:
   Symbols in total: 52304
   Symbols using price types: Bid=229, Last=52075
A more practical example is the indicator SymbolBidAskChart.mq5, designed to draw a diagram in the
form of bars formed based on prices of the specified type. This will allow you to compare candlesticks
of a chart that uses prices from the SYMBOL_CHART_MODE property for its construction with bars on
an alternative price type. For example, you can see bars at the Bid price on the instrument chart at the
price Last or get bars for the Ask price, which the standard terminal charts do not support.

---

## Page 1079

Part 6. Trading automation
1 079
6.1  Financial instruments and Market Watch
As a basis for a new indicator, we will take a ready-made indicator IndDeltaVolume.mq5 presented in
the section Waiting for data and managing visibility. In that indicator, we downloaded a tick history for
a certain number of bars BarCount and calculated the delta of volumes, that is, separately buy and sell
volumes. In the new indicator, we only need to replace the calculation algorithm with the search for
Open, High, Low, and Close prices based on ticks inside each bar.
Indicator settings include four buffers and one bar chart (DRAW_BARS) displayed in the main window.
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   1
   
#property indicator_type1   DRAW_BARS
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2
#property indicator_label1  "Open;High;Low;Close;"
The display as bars is chosen to make them easier to read when run over the main chart candlesticks
so that both versions of each bar are visible.
The new ChartMode input parameter allows the user to select one of three price types (note that Ask is
our addition compared to the standard set of elements in ENUM_SYMBOL_CHART_MODE).
enum ENUM_SYMBOL_CHART_MODE_EXTENDED
{
   _SYMBOL_CHART_MODE_BID,  // SYMBOL_CHART_MODE_BID
   _SYMBOL_CHART_MODE_LAST, // SYMBOL_CHART_MODE_LAST
   _SYMBOL_CHART_MODE_ASK,  // SYMBOL_CHART_MODE_ASK*
};
   
input int BarCount = 100;
input COPY_TICKS TickType = INFO_TICKS;
input ENUM_SYMBOL_CHART_MODE_EXTENDED ChartMode = _SYMBOL_CHART_MODE_BID;
The former CalcDeltaVolume class changed its name to CalcCustomBars but remained almost
unchanged. The differences include a new set of four buffers and the chartMode field which is initialized
in the constructor from the input variable ChartMode.

---

## Page 1080

Part 6. Trading automation
1 080
6.1  Financial instruments and Market Watch
class CalcCustomBars
{
   const int limit;
   const COPY_TICKS tickType;
   const ENUM_SYMBOL_CHART_MODE_EXTENDED chartMode;
   
   double open[];
   double high[];
   double low[];
   double close[];
   ...
public:
   CalcCustomBars(
      const int bars,
      const COPY_TICKS type,
      const ENUM_SYMBOL_CHART_MODE_EXTENDED mode)
      : limit(bars), tickType(type), chartMode(mode) ...
   {
      // register arrays as indicator buffers
      SetIndexBuffer(0, open);
      SetIndexBuffer(1, high);
      SetIndexBuffer(2, low);
      SetIndexBuffer(3, close);
      const static string defTitle[] = {"Open;High;Low;Close;"};
      const static string types[] = {"Bid", "Last", "Ask"};
      string name = defTitle[0];
      StringReplace(name, ";", types[chartMode] + ";");
      PlotIndexSetString(0, PLOT_LABEL, name);
      IndicatorSetInteger(INDICATOR_DIGITS, _Digits);
   }
   ...
Depending on the mode of chartMode, the auxiliary method price returns a specific price type from
each tick.

---

## Page 1081

Part 6. Trading automation
1 081 
6.1  Financial instruments and Market Watch
protected:
   double price(const MqlTick &t) const
   {
      switch(chartMode)
      {
      case _SYMBOL_CHART_MODE_BID:
         return t.bid;
      case _SYMBOL_CHART_MODE_LAST:
         return t.last;
      case _SYMBOL_CHART_MODE_ASK:
         return t.ask;
      }
      return 0; // error
   }
   ...
Using the price method, we can easily implement the modification of the main calculation method calc
which fills the buffers for the bar numbered i based on an array of ticks for this bar.
   void calc(const int i, const MqlTick &ticks[], const int skip = 0)
   {
      const int n = ArraySize(ticks);
      for(int j = skip; j < n; ++j)
      {
         const double p = price(ticks[j]);
         if(open[i] == EMPTY_VALUE)
         {
            open[i] = p;
         }
         
         if(p > high[i] || high[i] == EMPTY_VALUE)
         {
            high[i] = p;
         }
         
         if(p < low[i])
         {
            low[i] = p;
         }
         
         close[i] = p;
      }
   }
The remaining fragments of the source code and the principles of their work correspond to the
description of IndDeltaVolume.mq5.
In the OnInit handler, we additionally display the current price type of the chart and return a warning if
the user decides to build an indicator based on the Last price type for the instrument where the Last is
absent.

---

## Page 1082

Part 6. Trading automation
1 082
6.1  Financial instruments and Market Watch
int OnInit()
{
   ...
   ENUM_SYMBOL_CHART_MODE mode =
      (ENUM_SYMBOL_CHART_MODE)SymbolInfoInteger(_Symbol, SYMBOL_CHART_MODE);
   Print("Chart mode: ", EnumToString(mode));
   
   if(mode == SYMBOL_CHART_MODE_BID
      && ChartMode == _SYMBOL_CHART_MODE_LAST)
   {
      Alert("Last price is not available for ", _Symbol);
   }
   
   return INIT_SUCCEEDED;
}
Below is a screenshot of an instrument with the chart plotting mode based on the Last price; an
indicator with the price type Bid is laid over the chart.
Indicator with bars at Bid prices on the chart at Last prices
It is also interesting to look at the bars for the Ask price running over a regular Bid price chart.

---

## Page 1083

Part 6. Trading automation
1 083
6.1  Financial instruments and Market Watch
Indicator with bars at Ask prices on the chart at Bid prices
During hours of low liquidity, when the spread widens, you can see a significant difference between Bid
and Ask charts.
6.1 .1 1  Base, quote, and margin currencies of the instrument
One of the most important properties of each financial instrument is its working currencies:
• The base currency in which the purchased or sold asset is expressed (for Forex instruments)
• The profit calculation (quotation) currency
• The margin calculation currency
An MQL program can get the names of these currencies using the SymbolInfoString function and three
properties from the following table.
Identifier
Description
SYMBOL_CURRENCY_BASE
Base currency
SYMBOL_CURRENCY_PROFIT
Profit currency
SYMBOL_CURRENCY_MARGIN
Margin currency
These properties help to analyze Forex instruments, in the names of which many brokers add various
prefixes and suffixes, as well as exchange instruments. In particular, the algorithm will be able to find a
symbol to obtain a cross rate of two given currencies or select a portfolio of indexes with a given
common quote currency.

---

## Page 1084

Part 6. Trading automation
1 084
6.1  Financial instruments and Market Watch
Since searching for tools according to certain requirements is a very common task, let's create a class
SymbolFilter (SymbolFilter.mqh) to build a list of suitable symbols and their selected properties. In the
future, we will use this class not only to analyze currencies but also other characteristics.
First, we will consider a simplified version and then supplement it with convenient functionality.
In development, we will use ready-made auxiliary tools: an associative map array (MapArray.mqh) to
store key-value pairs of selected types and a symbol property monitor (SymbolMonitor.mqh).
#include <MQL5Book/MapArray.mqh>
#include <MQL5Book/SymbolMonitor.mqh>
To simplify the statements for accumulating the results of work in arrays, we use an improved version
of the PUSH macro, which we have already seen in previous examples, as well as its EXPAND version for
multidimensional arrays (simple assignment is impossible in this case).
#define PUSH(A,V) (A[ArrayResize(A, ArraySize(A) + 1, ArraySize(A) * 2) - 1] = V)
#define EXPAND(A) (ArrayResize(A, ArrayRange(A, 0) + 1, ArrayRange(A, 0) * 2) - 1)
An object of the SymbolFilter class must have a storage for the property values which will be used to
filter symbols. Therefore, we will describe three MapArray arrays in the class for integer, real, and
string properties.
class SymbolFilter
{
   MapArray<ENUM_SYMBOL_INFO_INTEGER,long> longs;
   MapArray<ENUM_SYMBOL_INFO_DOUBLE,double> doubles;
   MapArray<ENUM_SYMBOL_INFO_STRING,string> strings;
   ...
Setting the required filter properties is done using overloaded the let methods.
public:
   SymbolFilter *let(const ENUM_SYMBOL_INFO_INTEGER property, const long value)
   {
      longs.put(property, value);
      return &this;
   }
   
   SymbolFilter *let(const ENUM_SYMBOL_INFO_DOUBLE property, const double value)
   {
      doubles.put(property, value);
      return &this;
   }
   
   SymbolFilter *let(const ENUM_SYMBOL_INFO_STRING property, const string value)
   {
      strings.put(property, value);
      return &this;
   }
   ...
Please note that the methods return a pointer to the filter, which allows you to write conditions as a
chain: for example, if earlier in the code an object f of type SymbolFilter was described, then you can
impose two conditions on the price type and the name of the profit currency as follows:

---

## Page 1085

Part 6. Trading automation
1 085
6.1  Financial instruments and Market Watch
f.let(SYMBOL_CHART_MODE, SYMBOL_CHART_MODE_LAST).let(SYMBOL_CURRENCY_PROFIT, "USD");
The formation of an array of symbols that satisfy the conditions is performed by the filter object in
several variants of the select method, the simplest of which is presented below (other options will be
discussed later).
The watch parameter defines the search context for symbols: among those selected in Market Watch
(true) or all available (false). The output array symbols will be filled with the names of matching
symbols. We already know the code structure inside the method: it has a loop through the symbols for
each of which a monitor object m is created.
   void select(const bool watch, string &symbols[]) const
   {
      const int n = SymbolsTotal(watch);
      for(int i = 0; i < n; ++i)
      {
         const string s = SymbolName(i, watch);
         SymbolMonitor m(s);
         if(match<ENUM_SYMBOL_INFO_INTEGER,long>(m, longs)
         && match<ENUM_SYMBOL_INFO_DOUBLE,double>(m, doubles)
         && match<ENUM_SYMBOL_INFO_STRING,string>(m, strings))
         {
            PUSH(symbols, s);
         }
      }
   }
It is with the help of the monitor that we can get the value of any property in a unified way. Checking if
the properties of the current symbol match the stored set of conditions in longs, doubles, and strings
arrays is implemented by a helper method match. Only of all requested properties match, the symbol
name will be saved in the symbols output array.
In the simplest case, the implementation of the match method is as follows (subsequently it will be
changed).
protected:
   template<typename K,typename V>
   bool match(const SymbolMonitor &m, const MapArray<K,V> &data) const
   {
      for(int i = 0; i < data.getSize(); ++i)
      {
         const K key = data.getKey(i);
         if(!equal(m.get(key), data.getValue(i)))
         {
            return false;
         }
      }
      return true;
   }
If at least one of the values in the data array does not match the corresponding character property,
the method returns false. If all properties match (or there are no conditions for properties of this type),
the method returns true.

---

## Page 1086

Part 6. Trading automation
1 086
6.1  Financial instruments and Market Watch
The comparison of two values is performed using the equal. Given the fact that among the properties
there may be properties of type double, the implementation is not as simple as one might think.
   template<typename V>
   static bool equal(const V v1, const V v2)
   {
      return v1 == v2 || eps(v1, v2);
   }
For type double, the expression v1  == v2 may not work for close numbers, and therefore the precision
of the real DBL_EPSILON type should be taken into account. This is done in a separate method eps,
overloaded separately for type double and all other types due to the template.
   static bool eps(const double v1, const double v2)
   {
      return fabs(v1 - v2) < DBL_EPSILON * fmax(v1, v2);
   }
   
   template<typename V>
   static bool eps(const V v1, const V v2)
   {
      return false;
   }
When values of any type except double are equal, the template method eps just won't be called, and in
all other cases (including when values differ), it returns false as required (thus, only the condition v1  ==
v2).
The filter option described above only allows you to check properties for equality. However, in practice,
it is often required to analyze conditions for inequality, as well as for greater/less. For this reason, the
SymbolFilter class has the IS enumeration with basic comparison operations (if desired, it can be
supplemented).
class SymbolFilter
{
   ...
   enum IS
   {
      EQUAL,
      GREATER,
      NOT_EQUAL,
      LESS
   };
   ...
For each property from the ENUM_SYMBOL_INFO_INTEGER, ENUM_SYMBOL_INFO_DOUBLE, and
ENUM_SYMBOL_INFO_STRING enumerations, it is required to save not only the desired property value
(recall about associative arrays longs, doubles, strings), but also the comparing method from the new
IS enumeration.
Since the elements of standard enumerations have non-overlapping values (there is one exception
related to volumes but it is not critical), it makes sense to reserve one common map array conditions
for the comparison method. This raises the question of which type to choose for the map key in order
to technically "combine" different enumerations. To do this, we had to describe the dummy

---

## Page 1087

Part 6. Trading automation
1 087
6.1  Financial instruments and Market Watch
enumeration ENUM_ANY which only denotes a certain type of generic enumeration. Recall that all
enumerations have an internal representation equivalent to an integer int, and therefore can be
reduced to one another.
   enum ENUM_ANY
   {
   };
   
   MapArray<ENUM_ANY,IS> conditions;
   MapArray<ENUM_ANY,long> longs;
   MapArray<ENUM_ANY,double> doubles;
   MapArray<ENUM_ANY,string> strings;
   ...
Now we can complete all let methods which set the desired value of the property by adding the cmp
input parameter that specifies the comparison method. By default, it sets the check for equality
(EQUAL).
   SymbolFilter *let(const ENUM_SYMBOL_INFO_INTEGER property, const long value,
      const IS cmp = EQUAL)
   {
      longs.put((ENUM_ANY)property, value);
      conditions.put((ENUM_ANY)property, cmp);
      return &this;
   }
Here's a variant for integer properties. The other two overloads change in the same way.
Taking into account new information about different ways of comparing and simultaneously eliminating
different types of keys in map arrays, we modify the match method. In it, for each specified property,
we retrieve a condition from the conditions array based on the key in the data map array, and
appropriate checks are performed using the switch operator.

---

## Page 1088

Part 6. Trading automation
1 088
6.1  Financial instruments and Market Watch
   template<typename V>
   bool match(const SymbolMonitor &m, const MapArray<ENUM_ANY,V> &data) const
   {
      // dummy variable to select m.get method overload below
      static const V type = (V)NULL;
      // cycle by conditions imposed on the properties of the symbol
      for(int i = 0; i < data.getSize(); ++i)
      {
         const ENUM_ANY key = data.getKey(i);
         // choice of comparison method in the condition
         switch(conditions[key])
         {
         case EQUAL:
            if(!equal(m.get(key, type), data.getValue(i))) return false;
            break;
         case NOT_EQUAL:
            if(equal(m.get(key, type), data.getValue(i))) return false;
            break;
         case GREATER:
            if(!greater(m.get(key, type), data.getValue(i))) return false;
            break;
         case LESS:
            if(greater(m.get(key, type), data.getValue(i))) return false;
            break;
         }
      }
      return true;
   }
The new template greater method is implemented simplistically.
   template<typename V>
   static bool greater(const V v1, const V v2)
   {
      return v1 > v2;
   }
Now the match method call can be written in a shorter form since the only remaining type of the
template V is automatically determined by the passed data argument (and this is one of the arrays
longs, doubles, or strings).

---

## Page 1089

Part 6. Trading automation
1 089
6.1  Financial instruments and Market Watch
   void select(const bool watch, string &symbols[]) const
   {
      const int n = SymbolsTotal(watch);
      for(int i = 0; i < n; ++i)
      {
         const string s = SymbolName(i, watch);
         SymbolMonitor m(s);
         if(match(m, longs)
            && match(m, doubles)
            && match(m, strings))
         {
            PUSH(symbols, s);
         }
      }
   }
This is not the final version of the SymbolFilter class yet, but we can already test it in action.
Let's create a script SymbolFilterCurrency.mq5 that can filter symbols based on the properties of the
base currency and profit currency; in this case, it is USD. The MarketWatchOnly parameter only
searches in the Market Watch by default.
#include <MQL5Book/SymbolFilter.mqh>
   
input bool MarketWatchOnly = true;
   
void OnStart()
{
   SymbolFilter f;   // filter object
   string symbols[]; // array for results
   ...
Let's say that we want to find Forex instruments that have direct quotes, that is, "USD" appears in
their names at the beginning. In order not to depend on the specifics of the formation of names for a
particular broker, we will use the SYMBOL_CURRENCY_BASE property, which contains the first
currency.
Let's write down the condition that the base currency of the symbol is equal to USD and apply the
filter.
   f.let(SYMBOL_CURRENCY_BASE, "USD")
   .select(MarketWatchOnly, symbols);
   Print("===== Base is USD =====");
   ArrayPrint(symbols);
   ...
The resulting array is output to the log.
===== Base is USD =====
"USDCHF" "USDJPY" "USDCNH" "USDRUB" "USDCAD" "USDSEK" "SP500m" "Brent" 
As you can see, the array includes not only Forex symbols with USD at the beginning of the ticker but
also the S&P500 index and the commodity (oil). The last two symbols are quoted in dollars, but they
also have the same base currency. At the same time, the quote currency of Forex symbols (it is also

---

## Page 1090

Part 6. Trading automation
1 090
6.1  Financial instruments and Market Watch
the profit currency) is second and differs from USD. This allows you to supplement the filter in such a
way that non-Forex symbols no longer match it.
Let's clear the array, add a condition that the profit currency is not equal to "USD", and again request
suitable symbols (the previous condition was saved in the f object).
   ...
   ArrayResize(symbols, 0);
   
   f.let(SYMBOL_CURRENCY_PROFIT, "USD", SymbolFilter::IS::NOT_EQUAL)
   .select(MarketWatchOnly, symbols);
   Print("===== Base is USD and Profit is not USD =====");
   ArrayPrint(symbols);
}
This time, only the symbols you are looking for are actually displayed in the log.
===== Base is USD and Profit is not USD =====
"USDCHF" "USDJPY" "USDCNH" "USDRUB" "USDCAD" "USDSEK"
6.1 .1 2 Price representation accuracy and change steps
Earlier, we have already met two interrelated properties of the working symbol of the chart: the
minimum price change step (Point) and the price presentation accuracy which is expressed in the
number of decimal places (Digits). They are also available in Predefined variables. To get similar
properties of an arbitrary symbol, you should query the SYMBOL_POINT and SYMBOL_DIGITS
properties, respectively. The SYMBOL_POINT property is closely related to the minimum price change
(known to an MQL program as the SYMBOL_TRADE_TICK_SIZE property) and its value
(SYMBOL_TRADE_TICK_VALUE), usually in the currency of the trading account (but some symbols can
be configured to use the base currency; you may contact your broker for details if necessary). The
table below shows the entire group of these properties.
Identifier
Description
S YM B O L _D IG ITS 
The number of decimal places
S YM B O L _P O IN T
The value of one point in the quote currency
S YM B O L _TR AD E _TICK _VAL U E 
SYMBOL_TRADE_TICK_VALUE_PROFIT value
S YM B O L _TR AD E _TICK _VAL U E _P R O F IT
Current tick value for a profitable position
S YM B O L _TR AD E _TICK _VAL U E _L O S S 
Current tick value for a losing position
S YM B O L _TR AD E _TICK _S IZE 
Minimum price change in the quote currency
All properties except SYMBOL_DIGITS are real numbers and are requested using the SymbolInfoDouble
function. The SYMBOL_DIGITS property is available via SymbolInfoInteger. To test the work with these
properties, we will use ready-made classes SymbolFilter and SymbolMonitor, which will automatically
call the desired function for any property.

---

## Page 1091

Part 6. Trading automation
1 091 
6.1  Financial instruments and Market Watch
We will also improve the SymbolFilter class by adding a new overload of the select method, which will be
able to fill not only an array with the names of suitable symbols but also another array with the values
of their specific property.
In a more general case, we may be interested in several properties for each symbol at once, so it is
advisable to use not one of the built-in data types for the output array but a special composite type
with different fields.
In programming, such types are called tuples and are somewhat equivalent to MQL5 structures.
template<typename T1,typename T2,typename T3> // we can describe up to 64 fields
struct Tuple3                                 // MQL5 allows 64 template parameters
{
   T1 _1;
   T2 _2;
   T3 _3;
};
However, structures require a preliminary description with all fields, while we do not know in advance
the number and list of requested symbol properties. Therefore, in order to simplify the code, we will
represent our tuple as a vector in the second dimension of a dynamic array that receives the results of
the query.
T array[][S];
As a data type T we can use any of the built-in types and enumerations used for properties. Size S
must match the number of properties requested.
To tell the truth, such a simplification limits us in one query to values of the same types, that is, only
integers, only reals, or only strings. However, filter conditions can include any properties. We will
implement the approach with tuples a little later, using the example of filters of other trading entities:
orders, deals, and positions.
So the new version of the SymbolFilter::select method takes as an input a reference to the property
array with property identifiers to read from the filtered symbols. The names of the symbols themselves
and the values of these properties will be written to the symbols and data output arrays.

---

## Page 1092

Part 6. Trading automation
1 092
6.1  Financial instruments and Market Watch
   template<typename E,typename V>
   bool select(const bool watch, const E &property[], string &symbols[],
      V &data[][], const bool sort = false) const
   {
      // the size of the array of requested properties must match the output tuple
      const int q = ArrayRange(data, 1);
      if(ArraySize(property) != q) return false;
      
      const int n = SymbolsTotal(watch);
      // iterate over characters
      for(int i = 0; i < n; ++i)
      {
         const string s = SymbolName(i, watch);
         // access to the symbol properties is provided by the monitor
         SymbolMonitor m(s);
         // check all filter conditions
         if(match(m, longs)
         && match(m, doubles)
         && match(m, strings))
         {
            // properties of a suitable symbol are written to arrays
            const int k = EXPAND(data);
            for(int j = 0; j < q; ++j)
            {
               data[k][j] = m.get(property[j]);
            }
            PUSH(symbols, s);
         }
      }
      if(sort)
      {
         ...
      }
      return true;
   }
Additionally, the new method can sort the output array by the first dimension (the first requested
property): this functionality is left for independent study using source codes. To enable sorting, set the
sort parameter to true. Arrays with symbol names and data are sorted consistently.
To avoid tuples in the calling code when only one property needs to be requested from the filtered
characters, the following select option is implemented in SymbolFilter: inside it, we define intermediate
arrays of properties (properties) and values (tuples) with size 1  in the second dimension, which are
used to call the above full version of select.

---

## Page 1093

Part 6. Trading automation
1 093
6.1  Financial instruments and Market Watch
   template<typename E,typename V>
   bool select(const bool watch, const E property, string &symbols[], V &data[],
      const bool sort = false) const
   {
      E properties[1] = {property};
      V tuples[][1];
      
      const bool result = select(watch, properties, symbols, tuples, sort);
      ArrayCopy(data, tuples);
      return result;
   }
Using the advanced filter, let's try to build a list of symbols sorted by tick value
SYMBOL_TRADE_TICK_VALUE (see file SymbolFilterTickValue.mq5). Assuming that the deposit
currency is USD, we should obtain a value equal to 1 .0 for Forex instruments quoted in USD (of the
type XXXUSD). For other assets, we will see non-trivial values.
#include <MQL5Book/SymbolFilter.mqh>
   
input bool MarketWatchOnly = true;
   
void OnStart()
{
   SymbolFilter f;      // filter object
   string symbols[];    // array with symbol names
   double tickValues[]; // array for results
   
   // apply the filter without conditions, fill and sort the array
   f.select(MarketWatchOnly, SYMBOL_TRADE_TICK_VALUE, symbols, tickValues, true);
   
   PrintFormat("===== Tick values of the symbols (%d) =====",
      ArraySize(tickValues));
   ArrayPrint(symbols);
   ArrayPrint(tickValues, 5);
}
Here is the result of running the script.
===== Tick values of the symbols (13) =====
"BTCUSD" "USDRUB" "XAUUSD" "USDSEK" "USDCNH" "USDCAD" "USDJPY" "NZDUSD" "AUDUSD" "EURUSD" "GBPUSD" "USDCHF" "SP500m"
 0.00100  0.01309  0.10000  0.10955  0.15744  0.80163  0.87319  1.00000  1.00000  1.00000  1.00000  1.09212 10.00000
6.1 .1 3 Permitted volumes of trading operations
In later chapters, where we will learn how to program Expert Advisors, we will need to control many of
the symbol characteristics that determine the success of sending trading orders. In particular, this
applies to the part of the symbol specification that specifies the allowed scope of operations. The
corresponding properties are also available in MQL5. All of them are of type double and are requested
by the SymbolInfoDouble function.

---

## Page 1094

Part 6. Trading automation
1 094
6.1  Financial instruments and Market Watch
Identifier
Description
SYMBOL_VOLUME_MIN
Minimum deal volume in lots
SYMBOL_VOLUME_MAX
Maximum deal volume in lots
SYMBOL_VOLUME_STEP
Minimum step for changing the deal volume in lots
SYMBOL_VOLUME_LIMIT
Maximum allowable total volume of an open position and
pending orders in one direction (buy or sell)
SYMBOL_TRADE_CONTRACT_SIZE
Trading contract size = 1  lot
Attempts to buy or sell a financial instrument with a volume less than the minimum, more than the
maximum, or not a multiple of a step will result in an error. In the chapter related to trading APIs, we
will implement a code to unify the necessary checks and normalize volumes before calling the MQL5
API trading functions.
Among other things, the MQL program should also check SYMBOL_VOLUME_LIMIT. For example, with a
limit of 5 lots, you can have an open buy position with a volume of 5 lots and place a pending order Sell
Limit with a volume of 5 lots. However, you cannot place a pending Buy Limit order (because the
cumulative volume in one direction will exceed the limit) or set Sell Limit of more than 5 lots.
As an introductory example, consider the script SymbolFilterVolumes.mq5 which logs the values of the
above properties for the selected symbols. Let's add the MinimalContractSize variable to the input
parameters to be able to filter symbols by the SYMBOL_TRADE_CONTRACT_SIZE property: we display
only those which contract size is greater than the specified one (by default, 0, that is, all symbols
satisfy the condition).
#include <MQL5Book/SymbolFilter.mqh>
   
input bool MarketWatchOnly = true;
input double MinimalContractSize = 0;
At the beginning of OnStart, let's define a filter object and output arrays to get lists of property names
and values as vectors double for four fields. The list of the four required properties is indicated in the
volumeIds array.

---

## Page 1095

Part 6. Trading automation
1 095
6.1  Financial instruments and Market Watch
void OnStart()
{
   SymbolFilter f;                      // filter object
   string symbols[];                    // receiving array with names
   double volumeLimits[][4];            // receiving array with data vectors
   
   // requested symbol properties
   ENUM_SYMBOL_INFO_DOUBLE volumeIds[] =
   {
      SYMBOL_VOLUME_MIN,
      SYMBOL_VOLUME_STEP,
      SYMBOL_VOLUME_MAX,
      SYMBOL_VOLUME_LIMIT
   };
   ...
Next, we apply a filter by contract size (should be greater than the specified one) and get the
specification fields associated with volumes for matching symbols.
   f.let(SYMBOL_TRADE_CONTRACT_SIZE, MinimalContractSize, SymbolFilter::IS::GREATER)
   .select(MarketWatchOnly, volumeIds, symbols, volumeLimits);
   
   const int n = ArraySize(volumeLimits);
   PrintFormat("===== Volume limits of the symbols (%d) =====", n);
   string title = "";
   for(int i = 0; i < ArraySize(volumeIds); ++i)
   {
      title += "\t" + EnumToString(volumeIds[i]);
   }
   Print(title);
   for(int i = 0; i < n; ++i)
   {
      Print(symbols[i]);
      ArrayPrint(volumeLimits, 3, NULL, i, 1, 0);
   }
}
For default settings, the script might show results like the following (with abbreviations).

---

## Page 1096

Part 6. Trading automation
1 096
6.1  Financial instruments and Market Watch
===== Volume limits of the symbols (13) =====
SYMBOL_VOLUME_MIN SYMBOL_VOLUME_STEP SYMBOL_VOLUME_MAX SYMBOL_VOLUME_LIMIT
EURUSD
  0.010   0.010 500.000   0.000
GBPUSD
  0.010   0.010 500.000   0.000
USDCHF
  0.010   0.010 500.000   0.000
USDJPY
  0.010   0.010 500.000   0.000
USDCNH
   0.010    0.010 1000.000    0.000
USDRUB
   0.010    0.010 1000.000    0.000
...
XAUUSD
  0.010   0.010 100.000   0.000
BTCUSD
   0.010    0.010 1000.000    0.000
SP500m
 0.100  0.100  5.000 15.000
Some symbols may not be limited by SYMBOL_VOLUME_LIMIT (value is 0). You can compare the
results against the symbol specifications: they must match.
6.1 .1 4 Trading permission
As a continuation of the subject related to the correct preparation of trading orders which we started in
the previous section, let's turn to the following pair of properties that play a very important role in the
development of Expert Advisors.
Identifier
Description
SYMBOL_TRADE_MODE
Permissions for different trading modes for the symbol (see
ENUM_SYMBOL_TRADE_MODE)
SYMBOL_ORDER_MODE
Flags of allowed order types, bit mask (see further)
Both properties are of integer type and are available through the SymbolInfoInteger function.
We have already used the SYMBOL_TRADE_MODE property in the script SymbolPermissions.mq5. Its
value is one of the elements of the ENUM_SYMBOL_TRADE_MODE enumeration.

---

## Page 1097

Part 6. Trading automation
1 097
6.1  Financial instruments and Market Watch
Identifier
Value
Description
SYMBOL_TRADE_MODE_DISABLED
0
Trading is disabled for the
symbol
SYMBOL_TRADE_MODE_LONGONLY
1
Only buy trades are allowed
SYMBOL_TRADE_MODE_SHORTONLY
2
Only sell trades are allowed
SYMBOL_TRADE_MODE_CLOSEONLY
3
Only closing operations are
allowed
SYMBOL_TRADE_MODE_FULL
4
No restrictions on trading
operations
Recall the Permissions class contains the isTradeOnSymbolEnabled method which checks several
aspects that affect the availability of symbol trading, and one of them is the SYMBOL_TRADE_MODE
property. By default, we consider that we are interested in full access to trading, that is, selling and
buying: SYMBOL_TRADE_MODE_FULL. Depending on the trading strategy, the MQL program may
consider sufficient, for example, permissions only to buy, only to sell, or only to close operations.
   static bool isTradeOnSymbolEnabled(string symbol, const datetime now = 0,
      const ENUM_SYMBOL_TRADE_MODE mode = SYMBOL_TRADE_MODE_FULL)
   {
      // checking sessions
      bool found = now == 0;
      ...
      // checking the trading mode for the symbol
      return found && (SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) == mode);
   }
In addition to the trading mode, we will need to analyze the permissions for orders of different types in
the future: they are indicated by separate bits in the SYMBOL_ORDER_MODE property and can be
arbitrarily combined with a logical OR ('| '). For example, the value 1 27 (0x7F) corresponds to all bits
set, that is, the availability of all types of orders.

---

## Page 1098

Part 6. Trading automation
1 098
6.1  Financial instruments and Market Watch
Identifier
Value
Description
S YM B O L _O R D E R _M AR K E T
1
Market orders are allowed (Buy and Sell)
S YM B O L _O R D E R _L IM IT
2
Limit orders are allowed (Buy Limit and Sell
Limit)
S YM B O L _O R D E R _S TO P 
4
Stop orders are allowed (Buy Stop and Sell
Stop)
S YM B O L _O R D E R _S TO P _L IM IT
8
Stop limit orders are allowed (Buy Stop Limit
and Sell Stop Limit)
S YM B O L _O R D E R _S L 
1 6
Setting Stop Loss levels is allowed
S YM B O L _O R D E R _TP 
32
Setting Take Profit levels is allowed
S YM B O L _O R D E R _CL O S E B Y
64
Permission to close a position by an
opposite one for the same symbol, Close By
operation
The SYMBOL_ORDER_CLOSEBY property is only set for accounts with hedging accounting
(ACCOUNT_MARGIN_MODE_RETAIL_HEDGING, see Account type).
In the test script SymbolFilterTradeMode.mq5, we will request a couple of described properties for
symbols visible in Market Watch. The output of bits and their combinations as numbers is not very
informative, so we will utilize the fact that in the SymbolMonitor class, we have a convenient method
stringify to print enumeration members and bit masks of all properties.

---

## Page 1099

Part 6. Trading automation
1 099
6.1  Financial instruments and Market Watch
void OnStart()
{
   SymbolFilter f;                      // filter object
   string symbols[];                    // array for names 
   long permissions[][2];               // array for data (property values)
   
   // list of requested symbol properties
   ENUM_SYMBOL_INFO_INTEGER modes[] =
   {
      SYMBOL_TRADE_MODE,
      SYMBOL_ORDER_MODE
   };
   
   // apply the filter, get arrays with results
   f.let(SYMBOL_VISIBLE, true).select(true, modes, symbols, permissions);
   
   const int n = ArraySize(symbols);
   PrintFormat("===== Trade permissions for the symbols (%d) =====", n);
   for(int i = 0; i < n; ++i)
   {
      Print(symbols[i] + ":");
      for(int j = 0; j < ArraySize(modes); ++j)
      {
         // display bit and number descriptions "as is"
         PrintFormat("  %s (%d)",
            SymbolMonitor::stringify(permissions[i][j], modes[j]),
            permissions[i][j]);
      }
   }
}
Below is part of the log resulting from running the script.

---

## Page 1100

Part 6. Trading automation
1 1 00
6.1  Financial instruments and Market Watch
===== Trade permissions for the symbols (13) =====
EURUSD:
  SYMBOL_TRADE_MODE_FULL (4)
  [ _SYMBOL_ORDER_MARKET _SYMBOL_ORDER_LIMIT _SYMBOL_ORDER_STOP
  _SYMBOL_ORDER_STOP_LIMIT _SYMBOL_ORDER_SL _SYMBOL_ORDER_TP
  _SYMBOL_ORDER_CLOSEBY ] (127)
GBPUSD:
  SYMBOL_TRADE_MODE_FULL (4)
  [ _SYMBOL_ORDER_MARKET _SYMBOL_ORDER_LIMIT _SYMBOL_ORDER_STOP
  _SYMBOL_ORDER_STOP_LIMIT _SYMBOL_ORDER_SL _SYMBOL_ORDER_TP
  _SYMBOL_ORDER_CLOSEBY ] (127)
... 
SP500m:
  SYMBOL_TRADE_MODE_DISABLED (0)
  [ _SYMBOL_ORDER_MARKET _SYMBOL_ORDER_LIMIT _SYMBOL_ORDER_STOP
  _SYMBOL_ORDER_STOP_LIMIT _SYMBOL_ORDER_SL _SYMBOL_ORDER_TP ] (63)
Please note that trading for the last symbol SP500m is completely disabled (its quotes are provided
only as "indicative"). At the same time, its set of flags by order types is not 0 but does not make any
difference.
Depending on the events in the market, the broker can change the properties of the symbol at their
own discretion, for example, leaving only the opportunity to close positions for some time, so a correct
trading robot must control these properties before each operation.
6.1 .1 5 Symbol trading conditions and order execution modes
In this section, we will dive deeper into aspects of trading automation that depend on the settings of
financial instruments. For now, we will study only the properties, while their practical application will be
presented in later chapters. It is assumed that the reader is already familiar with the basic terminology
such as market and pending order, trade, and position.
When sending a trade request for execution, it should be taken into account that in the financial
markets there is no guarantee that at a particular moment, the entire requested volume is available for
this financial instrument at the desired price. Therefore, real-time trading is regulated by price and
volume execution modes. Modes, or in other words, execution policies, define the rules for cases when
the price has changed or the requested volume cannot be fully executed at the current moment.
In the MQL5 API, these modes are available for each symbol as the following properties which can be
obtained through the function SymbolInfoInteger.
Identifier
Description
SYMBOL_TRADE_EXEMODE
Trade execution modes related to the price
SYMBOL_FILLING_MODE
Flags of allowed order filling modes related to the
volume (bitmask, see further)
The value of the SYMBOL_TRADE_EXEMODE property is a member of the
ENUM_SYMBOL_TRADE_EXECUTION enumeration.

---

## Page 1101

Part 6. Trading automation
1 1 01 
6.1  Financial instruments and Market Watch
Identifier
Description
S YM B O L _TR AD E _E XE CU TIO N _R E Q U E S T
Trade at the requested price
S YM B O L _TR AD E _E XE CU TIO N _IN S TAN T
Instant execution (trading at streamed prices)
S YM B O L _TR AD E _E XE CU TIO N _M AR K E T
Market execution
S YM B O L _TR AD E _E XE CU TIO N _E XCH AN G E 
Exchange execution
All or most of these modes should be known to terminal users from the drop-down list Type in the New
order dialogue (F9). Let's briefly recall what they mean. For further details, please refer to the terminal
documentation.
• Execution on request (SYMBOL_TRADE_EXECUTION_REQUEST) – execution of a market order at a
price previously received from the broker. Before sending a market order, the trader requests the
current price from the broker. Further execution of the order at this price can either be confirmed
or rejected.
• Instant execution (SYMBOL_TRADE_EXECUTION_INSTANT) – execution of a market order at the
current price. When sending a trade request for execution, the terminal automatically inserts the
current prices into the order. If the broker accepts the price, the order is executed. If the broker
does not accept the requested price, the broker returns the prices at which this order can be
executed, which is called a requote.
• Market execution (SYMBOL_TRADE_EXECUTION_MARKET) – the broker inserts the execution price
into the order without additional confirmation from the trader. Sending a market order in this mode
implies an early agreement with the price at which it will be executed.
• Exchange execution (SYMBOL_TRADE_EXECUTION_EXCHANGE) – trading operations are performed
at the prices of current market offers.
As for the bits in SYMBOL_FILLING_MODE that can be combined with the logical operator OR ('| '), their
presence or absence indicates the following actions.
Identifier
Value
Fill policy
SYMBOL_FILLING_FOK
1
Fill Or Kill (FOK); the order must be
executed exclusively in the specified
volume or canceled
SYMBOL_FILLING_IOC
2
Immediate or Cancel (IOC); trade the
maximum volume available on the
market within the limits specified in the
order or cancel
(Identifier missing)
(any, including
0)
Return; in case of partial execution, the
market or limit order with the
remaining volume is not canceled but
stays valid
The possibility of using FOK and IOC modes is determined by the trade server.
If the SYMBOL_FILLING_FOK mode is enabled, then, while sending an order with the OrderSend
function, the MQL program will be able to use the relevant order fill type in the MqlTradeRequest

---

## Page 1102

Part 6. Trading automation
1 1 02
6.1  Financial instruments and Market Watch
structure: ORDER_FILLING_FOK. If at the same time, there is not enough volume of the financial
instrument on the market, the order will not be executed. It should be taken into account that the
required volume can be made up of several offers currently available on the market, resulting in several
transactions.
If the SYMBOL_FILLING_IOC mode is enabled, the MQL program will have access to the
ORDER_FILLING_IOC order filling method of the same name (it is also specified in the special "filling"
field (type_ filling) in the MqlTradeRequest structure before sending the order to the OrderSend
function). When using this mode, in case of impossibility of full execution, the order will be executed on
the available volume, and the remaining volume of the order will be canceled.
The last policy without an identifier is the default mode and is available regardless of other modes
(which is why it matches zero or any other value). In other words, even if we get the value 1 
(SYMBOL_FILLING_FOK), 2 (SYMBOL_FILLING_IOC), or 3 (SYMBOL_FILLING_FOK | 
SYMBOL_FILLING_IOC) for the SYMBOL_FILLING_MODE property, the return mode will be implied. To
use this policy, when forming an order (filling in the MqlTradeRequest structure) we should specify the
fill type ORDER_FILLING_RETURN.
Among all SYMBOL_TRADE_EXEMODE modes, there is one specificity regarding market execution
(SYMBOL_TRADE_EXECUTION_MARKET): Return orders are always prohibited in market execution
mode.
Since ORDER_FILLING_FOK corresponds to the constant 0, the absence of an explicit indication of the
filling type in a trade request will imply this particular mode.
We will consider all these nuances in practice when developing Expert Advisors but for now, let's check
the reading of properties in a simple script SymbolFilterExecMode.mq5.

---

## Page 1103

Part 6. Trading automation
1 1 03
6.1  Financial instruments and Market Watch
#include <MQL5Book/SymbolFilter.mqh>
   
void OnStart()
{
   SymbolFilter f;                      // filter object
   string symbols[];                    // array of symbol names
   long permissions[][2];               // array with property value vectors
   
   // properties to read
   ENUM_SYMBOL_INFO_INTEGER modes[] =
   {
      SYMBOL_TRADE_EXEMODE,
      SYMBOL_FILLING_MODE
   };
   // apply filter - fill arrays
   f.select(true, modes, symbols, permissions);
   
   const int n = ArraySize(symbols);
   PrintFormat("===== Trade execution and filling modes for the symbols (%d) =====", n);
   for(int i = 0; i < n; ++i)
   {
      Print(symbols[i] + ":");
      for(int j = 0; j < ArraySize(modes); ++j)
      {
         // output properties as descriptions and numbers
         PrintFormat("  %s (%d)",
            SymbolMonitor::stringify(permissions[i][j], modes[j]),
            permissions[i][j]);
      }
   }
}
Below is a fragment of the log with the results of the script. Almost all symbols here have an immediate
execution mode at prices (SYMBOL_TRADE_EXECUTION_INSTANT) except for the last SP500m
(SYMBOL_TRADE_EXECUTION_MARKET). Here we can find various volume filling modes, both separate
SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC, and their combination. Only BTCUSD has
SYMBOL_FILLING_RETURN specified, i.e. a value of 0 was received (no FOK and IOC bits).

---

## Page 1104

Part 6. Trading automation
1 1 04
6.1  Financial instruments and Market Watch
===== Trade execution and filling modes for the symbols (13) =====
EURUSD:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [ _SYMBOL_FILLING_FOK ] (1)
GBPUSD:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [ _SYMBOL_FILLING_FOK ] (1)
...
USDCNH:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [ _SYMBOL_FILLING_FOK _SYMBOL_FILLING_IOC ] (3)
USDRUB:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [ _SYMBOL_FILLING_IOC ] (2)
AUDUSD:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [ _SYMBOL_FILLING_FOK ] (1)
NZDUSD:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [ _SYMBOL_FILLING_FOK _SYMBOL_FILLING_IOC ] (3)
...
XAUUSD:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [ _SYMBOL_FILLING_FOK _SYMBOL_FILLING_IOC ] (3)
BTCUSD:
  SYMBOL_TRADE_EXECUTION_INSTANT (1)
  [(_SYMBOL_FILLING_RETURN)] (0)
SP500m:
  SYMBOL_TRADE_EXECUTION_MARKET (2)
  [ _SYMBOL_FILLING_FOK ] (1)
Recall that the underscores in the fill mode identifiers appear due to the fact that we had to define our
own enumeration SYMBOL_FILLING (SymbolMonitor.mqh) with elements with constant values. This was
done because MQL5 does not have such a built-in enumeration, but at the same time, we cannot name
the elements of our enumeration exactly as built-in constants as this would cause a name conflict.
6.1 .1 6 Margin requirements
Among the most important information about a financial instrument for a trader is the amount of funds
required to open a position. Without knowing how much money is needed to buy or sell a given number
of lots, it is impossible to implement a money management system within an Expert Advisor and control
the account balance.
Since MetaTrader 5 is used to trade various instruments (currencies, commodities, stocks, bonds,
options, and futures), the margin calculation principles differ significantly. The documentation provides
details, in particular for Forex and futures, as well as exchanges.
Several properties of the MQL5 API allow you to define the type of market and the method of
calculating the margin for a specific instrument.
Looking ahead, let's say that for a given combination of parameters such as the trading operation type,
instrument, volume, and price, MQL5 allows you to calculate the margin using the OrderCalcMargin

---

## Page 1105

Part 6. Trading automation
1 1 05
6.1  Financial instruments and Market Watch
function. This is the simplest method, but it has a significant limitation: the function does not take into
account current open positions and pending orders. This, in particular, ignores possible adjustments for
overlapping volumes when opposite positions are allowed on the account.
Thus, in order to obtain a breakdown of the account funds currently used as a margin for open positions
and orders, an MQL program may need to analyze the following properties and calculations using
formulas. Furthermore, the OrderCalcMargin function is prohibited for use in indicators. You can
estimate the free margin in advance after the proposed transaction is completed using OrderCheck.
Identifier
Description
S YM B O L _TR AD E _CAL C_M O D E 
The method for calculating margin and profit (see
ENUM_SYMBOL_CALC_MODE)
S YM B O L _M AR G IN _H E D G E D _U S E _L E G 
Boolean flag to enable (true) or disable (false) the hedged
margin calculation mode for the largest of the overlapped
positions (buy and sell)
S YM B O L _M AR G IN _IN ITIAL 
Initial margin for an exchange instrument
S YM B O L _M AR G IN _M AIN TE N AN CE 
Maintenance margin for an exchange instrument
S YM B O L _M AR G IN _H E D G E D 
Contract size or margin for one lot of covered positions
(opposite positions for one symbol)
The first two properties are included in the ENUM_SYMBOL_INFO_INTEGER enumeration, and the last
three are in ENUM_SYMBOL_INFO_DOUBLE, and they can be read, respectively, by functions
SymbolInfoInteger and SymbolInfoDouble.
Specific margin calculation formulas depend on the SYMBOL_TRADE_CALC_MODE property and are
shown in the table below. More complete information can be found in MQL5 documentation.
Please note that initial and maintenance margins are not used for Forex instruments, and these
properties are always 0 for them.
The initial margin indicates the amount of required security deposit in margin currency to open a
position with a volume of one lot. It is used when checking the sufficiency of the client's funds before
entering the market. To get the final amount of margin charged depending on the type and direction of
the order, check the margin ratios using the SymbolInfoMarginRate function. Thus, the broker can set
an individual leverage or discount for each instrument.
The maintenance margin indicates the minimum value of funds in the instrument's margin currency to
maintain an open position of one lot. It is used when checking the sufficiency of the client's funds when
the account status (trading conditions) changes. If the level of funds falls below the amount of the
maintenance margin of all positions, the broker will start to close them forcibly.
If the maintenance margin property is 0, then the initial margin is used. As in the case of the initial
margin, to obtain the final amount of margin charged depending on the type and direction, you should
check the margin ratios using the SymbolInfoMarginRate function. 
Hedged positions, that is, multidirectional positions for the same symbol, can only exist on hedging
trading accounts. Obviously, the calculation of the hedged margin together with the properties
SYMBOL_MARGIN_HEDGED_USE_LEG, SYMBOL_MARGIN_HEDGED make sense only on such accounts.
The hedged margin is applied for the covered volume.

---

## Page 1106

Part 6. Trading automation
1 1 06
6.1  Financial instruments and Market Watch
The broker can choose for each instrument one of the two existing methods for calculating the margin
for covered positions:
·The base calculation is applied when the longest side calculation mode is disabled, i.e. the
SYMBOL_MARGIN_HEDGED_USE_LEG property is equal to false. In this case, the margin consists
of three components: the margin for the uncovered volume of the existing position, the margin for
the covered volume (if there are opposite positions and the SYMBOL_MARGIN_HEDGED property is
non-zero), the margin for pending orders. If the initial margin is set for the instrument (the
SYMBOL_MARGIN_INITIAL property is non-zero), then the hedged margin is specified as an
absolute value (in money). If the initial margin is not set (equal to 0), then
SYMBOL_MARGIN_HEDGED specifies the contract size that will be used when calculating the
margin according to the formula corresponding to the type of trading instrument
(SYMBOL_TRADE_CALC_MODE).
·The highest position calculation is applied when the SYMBOL_MARGIN_HEDGED_USE_LEG property
is equal to true. The value of SYMBOL_MARGIN_HEDGED is ignored in this case. Instead, the
volume of all short and long positions on the instrument is calculated, and the weighted average
opening price is calculated for each side. Further, using the formulas corresponding to the
instrument type (SYMBOL_TRADE_CALC_MODE), the margin for the short side and the long side is
calculated. The largest value is used as the final value.
The following table lists the ENUM_SYMBOL_CALC_MODE elements and their respective margin
calculation methods. The same property (SYMBOL_TRADE_CALC_MODE) is also responsible for
calculating the profit/loss of a position, but we will consider this aspect later, in the chapter on MQL5
trading functions.
Identifier
Formula
S YM B O L _CAL C_M O D E _F O R E X
F ore x
Lots * ContractSize * MarginRate / Leverage
S YM B O L _CAL C_M O D E _F O R E X_N O _L E VE R AG E 
F ore x w it hout  le v e rage 
Lots * ContractSize * MarginRate
S YM B O L _CAL C_M O D E _CF D 
CF D 
L ot s * Con t r a c t S i z e  * M a r k e t P r i c e  * M a r g i n R a t e 
S YM B O L _CAL C_M O D E _CF D L E VE R AG E 
CF D  w it h le v e rage 
L ot s * Con t r a c t S i z e  * M a r k e t P r i c e  * M a r g i n R a t e  / L e ve r a g e 
S YM B O L _CAL C_M O D E _CF D IN D E X
CF D s on indice s
L ot s * Con t r a c t S i z e  * M a r k e t P r i c e  * Ti c k P r i c e  / Ti c k S i z e  * M a r g i n R a t e 
S YM B O L _CAL C_M O D E _E XCH _S TO CK S 
Se curit ie s on t he  st ock e xchange 
Lots * ContractSize * LastPrice * MarginRate
S YM B O L _CAL C_M O D E _E XCH _S TO CK S _M O E X
Se curit ie s on M O EX
Lots * ContractSize * LastPrice * MarginRate
S YM B O L _CAL C_M O D E _F U TU R E S 
F ut ure s
Lots * InitialMargin * MarginRate
S YM B O L _CAL C_M O D E _E XCH _F U TU R E S 
F ut ure s on t he  st ock e xchange 
Lots * InitialMargin * MarginRate                 or 
Lots * MaintenanceMargin * MarginRate

---

## Page 1107

Part 6. Trading automation
1 1 07
6.1  Financial instruments and Market Watch
Identifier
Formula
S YM B O L _CAL C_M O D E _E XCH _F U TU R E S _F O R TS 
F ut ure s on F O RT S
Lots * InitialMargin * MarginRate                 or 
Lots * MaintenanceMargin * MarginRate
S YM B O L _CAL C_M O D E _E XCH _B O N D S 
B onds on t he  st ock e xchange 
Lots * ContractSize * FaceValue * OpenPrice / 1 00
S YM B O L _CAL C_M O D E _E XCH _B O N D S _M O E X
B onds on M O EX
Lots * ContractSize * FaceValue * OpenPrice / 1 00
S YM B O L _CAL C_M O D E _S E R V_CO L L ATE R AL 
Non-tradable asset (margin not applicable)
The following notation is used in the formulas:
·Lots – position or order volume in lots (shares of the contract)
·ContractSize – contract size (one lot, SYMBOL_TRADE_CONTRACT_SIZE)
·Leverage – trading account leverage (ACCOUNT_LEVERAGE)
·InitialMargin – initial margin (SYMBOL_MARGIN_INITIAL)
·MaintenanceMargin – maintenance margin (SYMBOL_MARGIN_MAINTENANCE)
·TickPrice – tick price (SYMBOL_TRADE_TICK_VALUE)
·TickSize –  tick size (SYMBOL_TRADE_TICK_SIZE)
·MarketPrice – last known Bid/Ask price depending on the type of transaction
·LastPrice – last known Last price
·OpenPrice – weighted average price of a position or order opening
·FaceValue – face value of the bond
·MarginRate – margin ratio according to the SymbolInfoMarginRate function, can also have 2
different values: for initial and maintenance margin
An alternative implementation of formula calculations for most types of symbols is given in the file
MarginProfitMeter.mqh (see section Estimating the profit of a trading operation). It can also be used in
indicators.
Let's make a couple of comments on some modes.
In the table above, only three of the futures formulas use the initial margin
(SYMBOL_MARGIN_INITIAL). However, if this property has a non-zero value in the specification of any
other symbol, then it determines the margin.
Some exchanges may impose their own specifics on margin adjustment, such as the discount system
for FORTS (SYMBOL_CALC_MODE_EXCH_FUTURES_FORTS). See the MQL5 documentation and your
broker for details.
In the SYMBOL_CALC_MODE_SERV_COLLATERAL mode, the value of an instrument is taken into
account in Assets, which are added to Equity. Thus, open positions on such an instrument increase the
amount of Free Margin and serve as an additional collateral for open positions on traded instruments.
The market value of an open position is calculated based on the volume, contract size, current market
price, and liquidity ratio: Lots * ContractSize * MarketPrice * LiquidityRate (the latter value can be
obtained as the SYMBOL_TRADE_LIQUIDITY_RATE property).

---

## Page 1108

Part 6. Trading automation
1 1 08
6.1  Financial instruments and Market Watch
As an example of working with margin-related properties, consider the script
SymbolFilterMarginStats.mq5. Its purpose will be to calculate statistics on margin calculation methods
in the list of selected symbols, as well as optionally log these properties for each symbol. We will select
symbols for analysis using the already known filter class SymbolFilter and conditions for it supplied from
the input variables.
#include <MQL5Book/SymbolFilter.mqh>
   
input bool UseMarketWatch = false;
input bool ShowPerSymbolDetails = false;
input bool ExcludeZeroInitMargin = false;
input bool ExcludeZeroMainMargin = false;
input bool ExcludeZeroHedgeMargin = false;
By default, information is requested for all available symbols. To limit the context to the market
overview only, we should set UseMarketWatch to true.
Parameter ShowPerSymbolDetails allows you to enable the output of detailed information about each
symbol (by default, the parameter is false, and only statistics are displayed).
The last three parameters are intended for filtering symbols according to the conditions of zero margin
values (initial, maintenance, and hedging, respectively).
To collect and conveniently display in the log a complete set of properties for each symbol (when the
ShowPerSymbolDetails is on), the structure MarginSettings is defined in the code.
struct MarginSettings
{
   string name;
   ENUM_SYMBOL_CALC_MODE calcMode;
   bool hedgeLeg;
   double initial;
   double maintenance;
   double hedged;
};
Since some of the properties are integer (SYMBOL_TRADE_CALC_MODE,
SYMBOL_MARGIN_HEDGED_USE_LEG), and some are real (SYMBOL_MARGIN_INITIAL,
SYMBOL_MARGIN_MAINTENANCE, SYMBOL_MARGIN_HEDGED), they will have to be requested by the
filter object separately.
Now let's go directly to the working code in OnStart. Here, as usual, we define the filter object (f),
output arrays for character names (symbols), and values of requested properties (flags,values). In
addition to them, we add an array of structures MarginSettings.

---

## Page 1109

Part 6. Trading automation
1 1 09
6.1  Financial instruments and Market Watch
void OnStart()
{
   SymbolFilter f;                // filter object
   string symbols[];              // array for names
   long flags[][2];               // array for integer vectors
   double values[][3];            // array for real vectors
   MarginSettings margins[];      // composite output array
   ...
The stats array map has been introduced to calculate statistics with a key like
ENUM_SYMBOL_CALC_MODE and the int integer value for the number of times each method was
encountered. Also, all cases of zero margin and the enabled calculation mode on the longer leg should
be recorded in the corresponding counter variables.
   MapArray<ENUM_SYMBOL_CALC_MODE,int> stats; // counters for each method/mode
   int hedgeLeg = 0;                          // and other counters
   int zeroInit = 0;                          // ...
   int zeroMaintenance = 0;
   int zeroHedged = 0;
   ...
Next, we specify the properties of interest to us which are related to the margin, which will be read
from the symbol settings. First, integers in the ints array and then the real ones in the doubles array.
   ENUM_SYMBOL_INFO_INTEGER ints[] =
   {
      SYMBOL_TRADE_CALC_MODE,
      SYMBOL_MARGIN_HEDGED_USE_LEG
   };
   
   ENUM_SYMBOL_INFO_DOUBLE doubles[] =
   {
      SYMBOL_MARGIN_INITIAL,
      SYMBOL_MARGIN_MAINTENANCE,
      SYMBOL_MARGIN_HEDGED
   };
   ...
Depending on the input parameters, we will set the filtering conditions.
   if(ExcludeZeroInitMargin) f.let(SYMBOL_MARGIN_INITIAL, 0, SymbolFilter::IS::GREATER);
   if(ExcludeZeroMainMargin) f.let(SYMBOL_MARGIN_MAINTENANCE, 0, SymbolFilter::IS::GREATER);
   if(ExcludeZeroHedgeMargin) f.let(SYMBOL_MARGIN_HEDGED, 0, SymbolFilter::IS::GREATER);
   ...
Now everything is ready for selecting symbols by conditions and getting their properties into arrays. We
do this twice, separately for integer and real properties.

---

## Page 1110

Part 6. Trading automation
1 1 1 0
6.1  Financial instruments and Market Watch
   f.select(UseMarketWatch, ints, symbols, flags);
   const int n = ArraySize(symbols);
   ArrayResize(symbols, 0, n);
   f.select(UseMarketWatch, doubles, symbols, values);
   ...
An array with symbols has to be zeroed out after the first application of the filter so that the names do
not double up. Despite two separate queries, the order of elements in all output arrays (ints and
doubles) is the same, since the filtering conditions do not change.
If a detailed log is enabled by the user, we allocate memory for the margins array of structures.
   if(ShowPerSymbolDetails) ArrayResize(margins, n);
Finally, we calculate the statistics by iterating over all the elements of the resulting arrays and
optionally populate the array of structures.
   for(int i = 0; i < n; ++i)
   {
      stats.inc((ENUM_SYMBOL_CALC_MODE)flags[i].value[0]);
      hedgeLeg += (int)flags[i].value[1];
      if(values[i].value[0] == 0) zeroInit++;
      if(values[i].value[1] == 0) zeroMaintenance++;
      if(values[i].value[2] == 0) zeroHedged++;
      
      if(ShowPerSymbolDetails)
      {
         margins[i].name = symbols[i];
         margins[i].calcMode = (ENUM_SYMBOL_CALC_MODE)flags[i][0];
         margins[i].hedgeLeg = (bool)flags[i][1];
         margins[i].initial = values[i][0];
         margins[i].maintenance = values[i][1];
         margins[i].hedged = values[i][2];
      }
   }
   ...
Now we display the statistics in the log.
   PrintFormat("===== Margin calculation modes for %s symbols %s=====",
      (UseMarketWatch ? "Market Watch" : "all available"),
      (ExcludeZeroInitMargin || ExcludeZeroMainMargin || ExcludeZeroHedgeMargin
         ? "(with conditions) " : ""));
   PrintFormat("Total symbols: %d", n);
   PrintFormat("Hedge leg used in: %d", hedgeLeg);
   PrintFormat("Zero margin counts: initial=%d, maintenance=%d, hedged=%d",
      zeroInit, zeroMaintenance, zeroHedged);
   
   Print("Stats per calculation mode:");
   stats.print();
   ...
Since the members of the ENUM_SYMBOL_CALC_MODE enumeration are displayed as integers (which is
not very informative), we also display a text where each value has a name (from EnumToString).

---

## Page 1111

Part 6. Trading automation
1 1 1 1 
6.1  Financial instruments and Market Watch
   Print("Legend: key=calculation mode, value=count");
   for(int i = 0; i < stats.getSize(); ++i)
   {
      PrintFormat("%d -> %s", stats.getKey(i), EnumToString(stats.getKey(i)));
   }
   ...
If detailed information on the selected characters is required, we output the margins array of
structures.
   if(ShowPerSymbolDetails)
   {
      Print("Settings per symbol:");
      ArrayPrint(margins);
   }
}
Let's run the script a couple of times with different settings. Let's start with the default settings.
===== Margin calculation modes for all available symbols =====
Total symbols: 131
Hedge leg used in: 14
Zero margin counts: initial=123, maintenance=130, hedged=32
Stats per calculation mode:
    [key] [value]
[0]     0     101
[1]     4      16
[2]     1       1
[3]     2      11
[4]     5       2
Legend: key=calculation mode, value=count
0 -> SYMBOL_CALC_MODE_FOREX
4 -> SYMBOL_CALC_MODE_CFDLEVERAGE
1 -> SYMBOL_CALC_MODE_FUTURES
2 -> SYMBOL_CALC_MODE_CFD
5 -> SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE
For the second run, let's set ShowPerSymbolDetails and ExcludeZeroInitMargin to true. This requests
detailed information about all symbols that have a non-zero value of the initial margin.

---

## Page 1112

Part 6. Trading automation
1 1 1 2
6.1  Financial instruments and Market Watch
===== Margin calculation modes for all available symbols (with conditions) =====
Total symbols: 8
Hedge leg used in: 0
Zero margin counts: initial=0, maintenance=7, hedged=0
Stats per calculation mode:
    [key] [value]
[0]     0       5
[1]     1       1
[2]     5       2
Legend: key=calculation mode, value=count
0 -> SYMBOL_CALC_MODE_FOREX
1 -> SYMBOL_CALC_MODE_FUTURES
5 -> SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE
Settings per symbol:
      [name] [calcMode] [hedgeLeg]    [initial] [maintenance]    [hedged]
[0] "XAUEUR"          0      false    100.00000       0.00000    50.00000
[1] "XAUAUD"          0      false    100.00000       0.00000   100.00000
[2] "XAGEUR"          0      false   1000.00000       0.00000  1000.00000
[3] "USDGEL"          0      false 100000.00000  100000.00000 50000.00000
[4] "SP500m"          1      false   6600.00000       0.00000  6600.00000
[5] "XBRUSD"          5      false    100.00000       0.00000    50.00000
[6] "XNGUSD"          0      false  10000.00000       0.00000 10000.00000
[7] "XTIUSD"          5      false    100.00000       0.00000    50.00000
6.1 .1 7 Pending order expiration rules
When working with pending orders (including Stop Loss and Take Profit levels), an MQL program should
check a couple of properties that define the rules for their expiration. Both properties are available as
members of the ENUM_SYMBOL_INFO_INTEGER enumeration for the call of the SymbolInfoInteger
function.
Identifier
Description
SYMBOL_EXPIRATION_MODE
Flags of allowed order expiration modes (bit mask)
SYMBOL_ORDER_GTC_MODE
The validity period is defined by one of the elements of the
ENUM_SYMBOL_ORDER_GTC_MODE enumeration 
The SYMBOL_ORDER_GTC_MODE property is taken into account only if SYMBOL_EXPIRATION_MODE
contains SYMBOL_EXPIRATION_GTC (see further). GTC is an acronym for Good Till Canceled.
For each financial instrument, the SYMBOL_EXPIRATION_MODE property can specify several modes of
validity (expiration) of pending orders. Each mode has a flag (bit) associated with it.

---

## Page 1113

Part 6. Trading automation
1 1 1 3
6.1  Financial instruments and Market Watch
Identifier (Value)
Description
SYMBOL_EXPIRATION_GTC (1 )
Order is valid according to the
ENUM_SYMBOL_ORDER_GTC_MODE property
SYMBOL_EXPIRATION_DAY (2)
Order is valid until the end of the current day
SYMBOL_EXPIRATION_SPECIFIED (4)
The expiration date and time are specified in the order
SYMBOL_EXPIRATION_SPECIFIED_DAY (8)
The expiration date is specified in the order
The flags can be combined with a logical OR ('| ') operation, for example, SYMBOL_EXPIRATION_GTC | 
SYMBOL_EXPIRATION_SPECIFIED, equivalent to 1  |  4, which is the number 5. To check whether a
particular mode is enabled for a tool, perform a logical AND ('&') operation on the function result and
the desired mode bit: a non-zero value means the mode is available.
In the case of SYMBOL_EXPIRATION_SPECIFIED_DAY, the order is valid until 23:59:59 of the specified
day. If this time does not fall on the trading session, the expiration will occur at the nearest next
trading time.
The ENUM_SYMBOL_ORDER_GTC_MODE enumeration contains the following members.
Identifier
Description
SYMBOL_ORDERS_GTC
Pending orders and Stop Loss/Take Profit levels are
valid indefinitely until explicitly canceled
SYMBOL_ORDERS_DAILY
Orders are valid only within one trading day: upon its
completion, all pending orders are deleted, as well as
Stop Loss and Take Profit levels
S YM B O L _O R D E R S _D AIL Y_E XCL U D IN G _S TO P S 
When changing the trading day, only pending orders
are deleted, but Stop Loss and Take Profit levels are
saved
Depending on the set bits in the SYMBOL_EXPIRATION_MODE property, when preparing an order for
sending, an MQL program can select one of the modes corresponding to these bits. Technically, this is
done by filling in the type_time field in a special structure MqlTradeRequest before calling the
OrderSend function. The field value must be an element of the ENUM_ORDER_TYPE_TIME enumeration
(see Pending order expiration dates): as we will see later, it has something in common with the above
set of flags, that is, each flag sets the corresponding mode in the order: ORDER_TIME_GTC,
ORDER_TIME_DAY, ORDER_TIME_SPECIFIED, ORDER_TIME_SPECIFIED_DAY. The expiration time or
day itself must be specified in another field of the same structure.
The script SymbolFilterExpiration.mq5 allows you to find out the statistics of the use of each of the flags
in the available symbols (in the market overview or in general, depending on the input parameter
UseMarketWatch). The second parameter in ShowPerSymbolDetails, being set to true, will cause all
flags for each character to be logged, so be careful: if at the same time, the mode UseMarketWatch
equals false, a very large number of log entries will be generated.

---

## Page 1114

Part 6. Trading automation
1 1 1 4
6.1  Financial instruments and Market Watch
#property script_show_inputs
   
#include <MQL5Book/SymbolFilter.mqh>
   
input bool UseMarketWatch = false;
input bool ShowPerSymbolDetails = false;
In the OnStart function, in addition to the filter object and receiving arrays for symbol names and
property values, we describe MapArray to calculate statistics separately for each of the
SYMBOL_EXPIRATION_MODE and SYMBOL_ORDER_GTC_MODE properties.
void OnStart()
{
   SymbolFilter f;                // filter object
   string symbols[];              // receiving array for symbol names
   long flags[][2];               // receiving array for property values
   
   MapArray<SYMBOL_EXPIRATION,int> stats;        // mode counters
   MapArray<ENUM_SYMBOL_ORDER_GTC_MODE,int> gtc; // GTC counters
   
   ENUM_SYMBOL_INFO_INTEGER ints[] =
   {
      SYMBOL_EXPIRATION_MODE,
      SYMBOL_ORDER_GTC_MODE
   };
   ...
Next, apply the filter and calculate the statistics.

---

## Page 1115

Part 6. Trading automation
1 1 1 5
6.1  Financial instruments and Market Watch
   f.select(UseMarketWatch, ints, symbols, flags);
   const int n = ArraySize(symbols);
   
   for(int i = 0; i < n; ++i)
   {
      if(ShowPerSymbolDetails)
      {
         Print(symbols[i] + ":");
         for(int j = 0; j < ArraySize(ints); ++j)
         {
            // properties in the form of descriptions and numbers
            PrintFormat("  %s (%d)",
               SymbolMonitor::stringify(flags[i][j], ints[j]),
               flags[i][j]);
         }
      }
      
      const SYMBOL_EXPIRATION mode = (SYMBOL_EXPIRATION)flags[i][0];
      for(int j = 0; j < 4; ++j)
      {
         const SYMBOL_EXPIRATION bit = (SYMBOL_EXPIRATION)(1 << j);
         if((mode & bit) != 0)
         {
            stats.inc(bit);
         }
         
         if(bit == SYMBOL_EXPIRATION_GTC)
         {
            gtc.inc((ENUM_SYMBOL_ORDER_GTC_MODE)flags[i][1]);
         }
      }
   }
   ...
Finally, we output the received numbers to the log.

---

## Page 1116

Part 6. Trading automation
1 1 1 6
6.1  Financial instruments and Market Watch
   PrintFormat("===== Expiration modes for %s symbols =====",
      (UseMarketWatch ? "Market Watch" : "all available"));
   PrintFormat("Total symbols: %d", n);
   
   Print("Stats per expiration mode:");
   stats.print();
   Print("Legend: key=expiration mode, value=count");
   for(int i = 0; i < stats.getSize(); ++i)
   {
      PrintFormat("%d -> %s", stats.getKey(i), EnumToString(stats.getKey(i)));
   }
   Print("Stats per GTC mode:");
   gtc.print();
   Print("Legend: key=GTC mode, value=count");
   for(int i = 0; i < gtc.getSize(); ++i)
   {
      PrintFormat("%d -> %s", gtc.getKey(i), EnumToString(gtc.getKey(i)));
   }
}
Let's run the script two times. The first time, with the default settings, we can get something like the
following picture.
===== Expiration modes for all available symbols =====
Total symbols: 52357
Stats per expiration mode:
    [key] [value]
[0]     1   52357
[1]     2   52357
[2]     4   52357
[3]     8   52303
Legend: key=expiration mode, value=count
1 -> _SYMBOL_EXPIRATION_GTC
2 -> _SYMBOL_EXPIRATION_DAY
4 -> _SYMBOL_EXPIRATION_SPECIFIED
8 -> _SYMBOL_EXPIRATION_SPECIFIED_DAY
Stats per GTC mode:
    [key] [value]
[0]     0   52357
Legend: key=GTC mode, value=count
0 -> SYMBOL_ORDERS_GTC
Here you can see that almost all flags are allowed for most symbols, and for the
SYMBOL_EXPIRATION_GTC mode, the only variant SYMBOL_ORDERS_GTC is used.
Run the script a second time by setting UseMarketWatch and ShowPerSymbolDetails to true (it is
assumed that a limited number of symbols is selected in Market Watch).

---

## Page 1117

Part 6. Trading automation
1 1 1 7
6.1  Financial instruments and Market Watch
GBPUSD:
  [ _SYMBOL_EXPIRATION_GTC _SYMBOL_EXPIRATION_DAY _SYMBOL_EXPIRATION_SPECIFIED ] (7)
  SYMBOL_ORDERS_GTC (0)
USDCHF:
  [ _SYMBOL_EXPIRATION_GTC _SYMBOL_EXPIRATION_DAY _SYMBOL_EXPIRATION_SPECIFIED ] (7)
  SYMBOL_ORDERS_GTC (0)
USDJPY:
  [ _SYMBOL_EXPIRATION_GTC _SYMBOL_EXPIRATION_DAY _SYMBOL_EXPIRATION_SPECIFIED ] (7)
  SYMBOL_ORDERS_GTC (0)
...
XAUUSD:
  [ _SYMBOL_EXPIRATION_GTC _SYMBOL_EXPIRATION_DAY _SYMBOL_EXPIRATION_SPECIFIED
  _SYMBOL_EXPIRATION_SPECIFIED_DAY ] (15)
  SYMBOL_ORDERS_GTC (0)
SP500m:
  [ _SYMBOL_EXPIRATION_GTC _SYMBOL_EXPIRATION_DAY _SYMBOL_EXPIRATION_SPECIFIED
  _SYMBOL_EXPIRATION_SPECIFIED_DAY ] (15)
  SYMBOL_ORDERS_GTC (0)
UK100:
  [ _SYMBOL_EXPIRATION_GTC _SYMBOL_EXPIRATION_DAY _SYMBOL_EXPIRATION_SPECIFIED
  _SYMBOL_EXPIRATION_SPECIFIED_DAY ] (15)
  SYMBOL_ORDERS_GTC (0)
===== Expiration modes for Market Watch symbols =====
Total symbols: 15
Stats per expiration mode:
    [key] [value]
[0]     1      15
[1]     2      15
[2]     4      15
[3]     8       6
Legend: key=expiration mode, value=count
1 -> _SYMBOL_EXPIRATION_GTC
2 -> _SYMBOL_EXPIRATION_DAY
4 -> _SYMBOL_EXPIRATION_SPECIFIED
8 -> _SYMBOL_EXPIRATION_SPECIFIED_DAY
Stats per GTC mode:
    [key] [value]
[0]     0      15
Legend: key=GTC mode, value=count
0 -> SYMBOL_ORDERS_GTC
Of the 1 5 selected symbols, only 6 have the SYMBOL_EXPIRATION_SPECIFIED_DAY flag set. Details
about the flags for each symbol can be found above.
6.1 .1 8 Spreads and order distance from the current price
For many trading strategies, especially those based on short-term trades, information about the spread
and the distance from the current price, allowing the installation or modification of orders, is important.
All of these properties are part of the ENUM_SYMBOL_INFO_INTEGER enumeration and are available
through the SymbolInfoInteger function.

---

## Page 1118

Part 6. Trading automation
1 1 1 8
6.1  Financial instruments and Market Watch
Identifier
Description
SYMBOL_SPREAD
Spread size (in points)
SYMBOL_SPREAD_FLOAT
Boolean sign of a floating spread
SYMBOL_TRADE_STOPS_LEVEL
Minimum allowed distance from the current price (in points)
for setting Stop Loss, Take Profit, and pending orders
SYMBOL_TRADE_FREEZE_LEVEL
Distance from the current price (in points) to freeze orders
and positions
In the table above, the current price refers to the Ask or Bid price, depending on the nature of the
operation being performed.
Protective Stop Loss and Take Profit levels indicate that a position should be closed. This is performed
by an operation opposite to the opening. Therefore, for buy orders opened at the Ask price, protective
levels indicate Bid, and for sell orders opened at Bid, protective levels indicate Ask. When placing
pending orders, the open price type is selected in the standard way: buy orders (Buy Stop, Buy Limit,
Buy Stop Limit) are based on Ask and sell orders (Sell Stop, Sell Limit, Sell Stop Limit) are based on Bid.
Taking into account such types of prices in the context of the mentioned trading operations, the
distance in points is calculated for the SYMBOL_TRADE_STOPS_LEVEL and
SYMBOL_TRADE_FREEZE_LEVEL properties.
The SYMBOL_TRADE_STOPS_LEVEL property, if it is non-zero, disables modification of Stop Loss and
Take Profit levels for an open position if the new level would be closer to the current price than the
specified distance. Similarly, it is impossible to move the opening price of a pending order closer than
SYMBOL_TRADE_STOPS_LEVEL points from the current price.
The SYMBOL_TRADE_FREEZE_LEVEL property, if it is non-zero, restricts any trading operations for a
pending order or an open position within the specified distance from the current price. For a pending
order, freezing occurs when the specified open price is at a distance less than
SYMBOL_TRADE_FREEZE_LEVEL points from the current price (again, the type of the current price is
Ask or Bid, depending on whether it is buying or selling). For a position, freezing occurs for Stop Loss
and Take Profit levels, which happened to be near the current price, and therefore the measurement for
them is performed for "opposite" price types.
If the SYMBOL_SPREAD_FLOAT property is true, the SYMBOL_SPREAD property is not part of the
symbol specification but contains the actual spread, dynamically changing with each call according to
market conditions. It can also be found as the difference between Ask and Bid prices in the MqlTick
structure by calling SymbolInfoTick.
The script SymbolFilterSpread.mq5 will allow you to analyze the specified properties. It defines a
custom enumeration ENUM_SYMBOL_INFO_INTEGER_PART, which includes only the properties of
interest to us in this context from ENUM_SYMBOL_INFO_INTEGER.

---

## Page 1119

Part 6. Trading automation
1 1 1 9
6.1  Financial instruments and Market Watch
enum ENUM_SYMBOL_INFO_INTEGER_PART
{
   SPREAD_FIXED = SYMBOL_SPREAD,
   SPREAD_FLOAT = SYMBOL_SPREAD_FLOAT,
   STOPS_LEVEL = SYMBOL_TRADE_STOPS_LEVEL,
   FREEZE_LEVEL = SYMBOL_TRADE_FREEZE_LEVEL
};
The new enumeration defines the Property input parameter, which specifies which of the four properties
will be analyzed. Parameters UseMarketWatch and ShowPerSymbolDetails control the process in the
already known way, as in the previous test scripts.
input bool UseMarketWatch = true;
input ENUM_SYMBOL_INFO_INTEGER_PART Property = SPREAD_FIXED;
input bool ShowPerSymbolDetails = true;
For the convenient display of information for each symbol (property name and value in each line) using
the ArrayPrint function, an auxiliary structure SymbolDistance is defined (used only when
ShowPerSymbolDetails equals true).
struct SymbolDistance
{
   string name;
   int value;
};
In the OnStart handler, we describe the necessary objects and arrays.
void OnStart()
{
   SymbolFilter f;                // filter object
   string symbols[];              // receiving array for names
   long values[];                 // receiving array for values
   SymbolDistance distances[];    // array to print
   MapArray<long,int> stats;      // counters of specific values of the selected property
   ...
Then we apply the filter and fill the receiving arrays with the values of the specified Property while also
applying sorting.
   f.select(UseMarketWatch, (ENUM_SYMBOL_INFO_INTEGER)Property, symbols, values, true);
   const int n = ArraySize(symbols);
   if(ShowPerSymbolDetails) ArrayResize(distances, n);
   ...
In a loop, we count the statistics and fill in the SymbolDistance structures, if it is needed.

---

## Page 1120

Part 6. Trading automation
1 1 20
6.1  Financial instruments and Market Watch
   for(int i = 0; i < n; ++i)
   {
      stats.inc(values[i]);
      if(ShowPerSymbolDetails)
      {
         distances[i].name = symbols[i];
         distances[i].value = (int)values[i];
      }
   }
   ...
Finally, we output the results to the log.
   PrintFormat("===== Distances for %s symbols =====",
      (UseMarketWatch ? "Market Watch" : "all available"));
   PrintFormat("Total symbols: %d", n);
   
   PrintFormat("Stats per %s:", EnumToString((ENUM_SYMBOL_INFO_INTEGER)Property));
   stats.print();
   
   if(ShowPerSymbolDetails)
   {
      Print("Details per symbol:");
      ArrayPrint(distances);
   }
}
Here's what you get when you run the script with default settings, which is consistent with spread
analysis.

---

## Page 1121

Part 6. Trading automation
1 1 21 
6.1  Financial instruments and Market Watch
===== Distances for Market Watch symbols =====
Total symbols: 13
Stats per SYMBOL_SPREAD:
    [key] [value]
[0]     0       2
[1]     2       3
[2]     3       1
[3]     6       1
[4]     7       1
[5]     9       1
[6]   151       1
[7]   319       1
[8]  3356       1
[9]  3400       1
Details per symbol:
       [name] [value]
[ 0] "USDJPY"       0
[ 1] "EURUSD"       0
[ 2] "USDCHF"       2
[ 3] "USDCAD"       2
[ 4] "GBPUSD"       2
[ 5] "AUDUSD"       3
[ 6] "XAUUSD"       6
[ 7] "SP500m"       7
[ 8] "NZDUSD"       9
[ 9] "USDCNH"     151
[10] "USDSEK"     319
[11] "BTCUSD"    3356
[12] "USDRUB"    3400
To understand whether the spreads are floating (changing dynamically) or fixed, let's run the script
with different settings: Property = SPREAD_FLOAT, ShowPerSymbolDetails = false.
===== Distances for Market Watch symbols =====
Total symbols: 13
Stats per SYMBOL_SPREAD_FLOAT:
    [key] [value]
[0]     1      13
According to this data, all symbols in the market watch have a floating spread (value 1  in the key  key
is true in SYMBOL_SPREAD_FLOAT). Therefore, if we run the script again and again with the default
settings, we will receive new values (with an open market).
6.1 .1 9 Getting swap sizes
For the implementation of medium-term and long-term strategies, the swap sizes become important,
since they can have a significant, usually negative, impact on the financial result. However, some
readers are probably fans of the "Carry Trade" strategy, which was originally built on profiting from
positive swaps. MQL5 has several symbol properties that provide access to specification strings that
are associated with swaps.

---

## Page 1122

Part 6. Trading automation
1 1 22
6.1  Financial instruments and Market Watch
Identifier
Description
SYMBOL_SWAP_MODE
Swap calculation model ENUM_SYMBOL_SWAP_MODE
SYMBOL_SWAP_ROLLOVER3DAYS
Day of the week for triple swap credit ENUM_DAY_OF_WEEK
SYMBOL_SWAP_LONG
Swap size for a long position
SYMBOL_SWAP_SHORT
Swap size for a short position
The ENUM_SYMBOL_SWAP_MODE enumeration contains elements that specify options for units of
measure and principles for calculating swaps. As well as SYMBOL_SWAP_ROLLOVER3DAYS, they refer
to the integer properties of ENUM_SYMBOL_INFO_INTEGER.
The swap sizes are directly specified in the SYMBOL_SWAP_LONG and SYMBOL_SWAP_SHORT
properties as part of ENUM_SYMBOL_INFO_DOUBLE, that is, of type double.
Following are the elements of ENUM_SYMBOL_SWAP_MODE.
Identifier
Description
SYMBOL_SWAP_MODE_DISABLED
no swaps
SYMBOL_SWAP_MODE_POINTS
points
SYMBOL_SWAP_MODE_CURRENCY_SYMBOL
the base currency of the symbol
SYMBOL_SWAP_MODE_CURRENCY_MARGIN
symbol margin currency
SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT
deposit currency
SYMBOL_SWAP_MODE_INTEREST_CURRENT
annual percentage of the price of the instrument
at the time of swap calculation
SYMBOL_SWAP_MODE_INTEREST_OPEN
annual percentage of the symbol position opening
price
SYMBOL_SWAP_MODE_REOPEN_CURRENT
points (with re-opening of the position at the
closing price)
SYMBOL_SWAP_MODE_REOPEN_BID
points (with re-opening of the position at the Bid
price of the new day). (in SYMBOL_SWAP_LONG
and SYMBOL_SWAP_SHORT parameters)
For the SYMBOL_SWAP_MODE_INTEREST_CURRENT and SYMBOL_SWAP_MODE_INTEREST_OPEN
options, there is assumed to be 360 banking days in a year.
For the SYMBOL_SWAP_MODE_REOPEN_CURRENT and SYMBOL_SWAP_MODE_REOPEN_BID options,
the position is forcibly closed at the end of the trading day, and then their behavior is different.
With SYMBOL_SWAP_MODE_REOPEN_CURRENT, the position is reopened the next day at yesterday's
closing price +/- the specified number of points. With SYMBOL_SWAP_MODE_REOPEN_BID, the position
is reopened the next day at the current Bid price +/- the specified number of points. In both cases, the
number of points is in the SYMBOL_SWAP_LONG and SYMBOL_SWAP_SHORT parameters.

---

## Page 1123

Part 6. Trading automation
1 1 23
6.1  Financial instruments and Market Watch
Let's check the operation of the properties using the script SymbolFilterSwap.mq5. In the input
parameters, we provide the choice of the analysis context: Market Watch or all symbols depending on
UseMarketWatch. When the ShowPerSymbolDetails parameter is false, we will calculate statistics, how
many times one or another mode from ENUM_SYMBOL_SWAP_MODE is used in symbols. When the
ShowPerSymbolDetails parameter is true, we will output an array of all symbols with the mode specified
in mode, and sort the array in descending order of values in the fields SYMBOL_SWAP_LONG and
SYMBOL_SWAP_SHORT.
input bool UseMarketWatch = true;
input bool ShowPerSymbolDetails = false;
input ENUM_SYMBOL_SWAP_MODE Mode = SYMBOL_SWAP_MODE_POINTS;
For the elements of the combined array of swaps, we describe the SymbolSwap structure with the
symbol name and swap value. The direction of the swap will be denoted by a prefix in the name field:
"+" for swaps of long positions, "-" for swaps of short positions.
struct SymbolSwap
{
   string name;
   double value;
};
By tradition, we describe the filter object at the beginning of OnStart. However, the following code
differs significantly depending on the value of the ShowPerSymbolDetails variable.
void OnStart()
{
   SymbolFilter f;                // filter object
   PrintFormat("===== Swap modes for %s symbols =====",
      (UseMarketWatch ? "Market Watch" : "all available"));
   
   if(ShowPerSymbolDetails)
   {
      // summary table of swaps of the selected Mode
      ...
   }
   else
   {
      // calculation of mode statistics
      ...
   }
}
Let's introduce the second branch first. Here we fill arrays with symbol names using the filter (symbols)
and swap modes (values) that are taken from the SYMBOL_SWAP_MODE property. The resulting values
are accumulated in an array map MapArray<ENUM_ SYMBOL_ SWAP_ MODE,int> stats.

---

## Page 1124

Part 6. Trading automation
1 1 24
6.1  Financial instruments and Market Watch
      // calculation of mode statistics
      string symbols[];
      long values[];
      MapArray<ENUM_SYMBOL_SWAP_MODE,int> stats; // counters for each mode
      // apply filter and collect mode values
      f.select(UseMarketWatch, SYMBOL_SWAP_MODE, symbols, values);
      const int n = ArraySize(symbols);
      for(int i = 0; i < n; ++i)
      {
         stats.inc((ENUM_SYMBOL_SWAP_MODE)values[i]);
      }
      ...
Next, we display the collected statistics.
      PrintFormat("Total symbols: %d", n);
      Print("Stats per swap mode:");
      stats.print();
      Print("Legend: key=swap mode, value=count");
      for(int i = 0; i < stats.getSize(); ++i)
      {
         PrintFormat("%d -> %s", stats.getKey(i), EnumToString(stats.getKey(i)));
      }
For the case of constructing a table with swap values, the algorithm is as follows. Swaps for long and
short positions are requested separately, so we define paired arrays for names and values. Together
they will be brought together in the swaps array of structures.
      // summary table of swaps of the selected Mode
      string buyers[], sellers[];    // arrays for names
      double longs[], shorts[];      // arrays for swap values
      SymbolSwap swaps[];            // total array to print
Set the condition for the selected swap mode in the filter. This is necessary to be able to compare and
sort array elements.
      f.let(SYMBOL_SWAP_MODE, Mode);
Then we apply the filter twice for different properties (SYMBOL_SWAP_LONG, SYMBOL_SWAP_SHORT)
and fill different arrays with their values (longs, shorts). Within each call, the arrays are sorted in
ascending order.
      f.select(UseMarketWatch, SYMBOL_SWAP_LONG, buyers, longs, true);
      f.select(UseMarketWatch, SYMBOL_SWAP_SHORT, sellers, shorts, true);
In theory, the sizes of the arrays should be the same, since the filter condition is the same, but for
clarity, let's allocate a variable for each size. Since each symbol will appear in the resulting table twice,
for the long and short sides, we provide a double size for the swaps array.

---

## Page 1125

Part 6. Trading automation
1 1 25
6.1  Financial instruments and Market Watch
      const int l = ArraySize(longs);
      const int s = ArraySize(shorts);
      const int n = ArrayResize(swaps, l + s); // should be l == s
      PrintFormat("Total symbols with %s: %d", EnumToString(Mode), l);
Next, we join the two arrays longs and shorts, processing them in reverse order, since we need to sort
from positive to negative values.
      if(n > 0)
      {
         int i = l - 1, j = s - 1, k = 0;
         while(k < n)
         {
            const double swapLong = i >= 0 ? longs[i] : -DBL_MAX;
            const double swapShort = j >= 0 ? shorts[j] : -DBL_MAX;
            
            if(swapLong >= swapShort)
            {
               swaps[k].name = "+" + buyers[i];
               swaps[k].value = longs[i];
               --i;
               ++k;
            }
            else
            {
               swaps[k].name = "-" + sellers[j];
               swaps[k].value = shorts[j];
               --j;
               ++k;
            }
         }
         Print("Swaps per symbols (ordered):");
         ArrayPrint(swaps);
      }
It is interesting to run the script several times with different settings. For example, by default, we can
get the following results.
===== Swap modes for Market Watch symbols =====
Total symbols: 13
Stats per swap mode:
    [key] [value]
[0]     1      10
[1]     0       2
[2]     2       1
Legend: key=swap mode, value=count
1 -> SYMBOL_SWAP_MODE_POINTS
0 -> SYMBOL_SWAP_MODE_DISABLED
2 -> SYMBOL_SWAP_MODE_CURRENCY_SYMBOL
These statistics show that 1 0 symbols have the swap mode SYMBOL_SWAP_MODE_POINTS, for two
the swaps are disabled, SYMBOL_SWAP_MODE_DISABLED, and for one it is in the base currency
SYMBOL_SWAP_MODE_CURRENCY_SYMBOL.

---

## Page 1126

Part 6. Trading automation
1 1 26
6.1  Financial instruments and Market Watch
Let's find out what kind of symbols have SYMBOL_SWAP_MODE_POINTS and find out their swaps. For
this, we will set ShowPerSymbolDetails to true (parameter mode already set to
SYMBOL_SWAP_MODE_POINTS).
===== Swap modes for Market Watch symbols =====
Total symbols with SYMBOL_SWAP_MODE_POINTS: 10
Swaps per symbols (ordered):
        [name]   [value]
[ 0] "+AUDUSD"   6.30000
[ 1] "+NZDUSD"   2.80000
[ 2] "+USDCHF"   0.10000
[ 3] "+USDRUB"   0.00000
[ 4] "-USDRUB"   0.00000
[ 5] "+USDJPY"  -0.10000
[ 6] "+GBPUSD"  -0.20000
[ 7] "-USDCAD"  -0.40000
[ 8] "-USDJPY"  -0.60000
[ 9] "+EURUSD"  -0.70000
[10] "+USDCAD"  -0.80000
[11] "-EURUSD"  -1.00000
[12] "-USDCHF"  -1.00000
[13] "-GBPUSD"  -2.20000
[14] "+USDSEK"  -4.50000
[15] "-XAUUSD"  -4.60000
[16] "-USDSEK"  -4.90000
[17] "-NZDUSD"  -6.70000
[18] "+XAUUSD" -12.60000
[19] "-AUDUSD" -14.80000
You can compare the values with symbol specifications.
Finally, we change the Mode to SYMBOL_SWAP_MODE_CURRENCY_SYMBOL. In our case, we should get
one symbol, but spaced into two lines: with a plus and a minus in the name.
===== Swap modes for Market Watch symbols =====
Total symbols with SYMBOL_SWAP_MODE_CURRENCY_SYMBOL: 1
Swaps per symbols (ordered):
       [name]   [value]
[0] "-SP500m" -35.00000
[1] "+SP500m" -41.41000
From the table, both swaps are negative.
6.1 .20 Current market information (tick)
In the section Getting the last tick of a symbol, we have already seen the SymbolInfoTick function,
which provides complete information about the last tick (price change event) in the form of the MqlTick
structure. If necessary, the MQL program can request the values of prices and volumes corresponding
to the fields of this structure separately. All of them are denoted by properties of different types that
are part of the ENUM_SYMBOL_INFO_INTEGER and ENUM_SYMBOL_INFO_DOUBLE enumerations.

---

## Page 1127

Part 6. Trading automation
1 1 27
6.1  Financial instruments and Market Watch
Identifier
Description
Property type
SYMBOL_TIME
Last quote time
datetime
SYMBOL_BID
Bid price; the best sell offer
double
SYMBOL_ASK
Ask price; the best buy offer
double
SYMBOL_LAST
Last; the price of the last deal
double
SYMBOL_VOLUME
The volume of the last deal
long
SYMBOL_TIME_MSC
The time of the last quote in
milliseconds since 1 970.01 .01 
long
SYMBOL_VOLUME_REAL
The volume of the last deal with
increased accuracy
double
Note that the code for the two volume-related properties, SYMBOL_VOLUME and
SYMBOL_VOLUME_REAL, is the same in both enumerations. This is the only case where the element
IDs of different enumerations overlap. The thing is that they return essentially the same tick property,
but with different representation accuracy.
Unlike a structure, properties do not provide an analog to the uint flags field, which tells what kind of
changes in the market caused the tick generation. This field is only meaningful within a structure.
Let's try to request tick properties separately and compare them with the result of the SymbolInfoTick
call. In a fast market, there is a possibility that the results will differ. A new tick (or even several ticks)
may come between function calls.
void OnStart()
{
   PRTF(TimeToString(SymbolInfoInteger(_Symbol, SYMBOL_TIME), TIME_DATE | TIME_SECONDS));
   PRTF(SymbolInfoDouble(_Symbol, SYMBOL_BID));
   PRTF(SymbolInfoDouble(_Symbol, SYMBOL_ASK));
   PRTF(SymbolInfoDouble(_Symbol, SYMBOL_LAST));
   PRTF(SymbolInfoInteger(_Symbol, SYMBOL_VOLUME));
   PRTF(SymbolInfoInteger(_Symbol, SYMBOL_TIME_MSC));
   PRTF(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_REAL));
   
   MqlTick tick[1];
   SymbolInfoTick(_Symbol, tick[0]);
   ArrayPrint(tick);
}
It is easy to verify that in a particular case, the information coincided.

---

## Page 1128

Part 6. Trading automation
1 1 28
6.1  Financial instruments and Market Watch
TimeToString(SymbolInfoInteger(_Symbol,SYMBOL_TIME),TIME_DATE|TIME_SECONDS)
   =2022.01.25 13:52:51 / ok
SymbolInfoDouble(_Symbol,SYMBOL_BID)=1838.44 / ok
SymbolInfoDouble(_Symbol,SYMBOL_ASK)=1838.49 / ok
SymbolInfoDouble(_Symbol,SYMBOL_LAST)=0.0 / ok
SymbolInfoInteger(_Symbol,SYMBOL_VOLUME)=0 / ok
SymbolInfoInteger(_Symbol,SYMBOL_TIME_MSC)=1643118771166 / ok
SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_REAL)=0.0 / ok
                 [time]   [bid]   [ask] [last] [volume]    [time_msc] [flags] [volume_real]
[0] 2022.01.25 13:52:51 1838.44 1838.49   0.00        0 1643118771166       6          0.00
6.1 .21  Descriptive symbol properties
The platform provides a group of text properties for MQL programs that describe important qualitative
characteristics. For example, when developing indicators or trading strategies based on a basket of
financial instruments, it may be necessary to select symbols by country of origin, economic sector, or
name of the underlying asset (if the instrument is a derivative).
Identifier
Description
SYMBOL_BASIS
The name of the underlying asset for the derivative
SYMBOL_CATEGORY
The name of the category to which the financial instrument
belongs
SYMBOL_COUNTRY
The country to which the financial instrument is assigned
SYMBOL_SECTOR_NAME
The sector of the economy to which the financial instrument
belongs
SYMBOL_INDUSTRY_NAME
The branch of the economy or type of industry to which the
financial instrument belongs
SYMBOL_BANK
Current quote source
SYMBOL_DESCRIPTION
String description of the symbol
SYMBOL_EXCHANGE
The name of the exchange or marketplace where the symbol
is traded
SYMBOL_ISIN
A unique 1 2-digit alphanumeric code in the system of
international securities identification codes - ISIN
(International Securities Identification Number)
SYMBOL_PAGE
Internet page address with information on the symbol
SYMBOL_PATH
Path in symbol tree
Another case where the program can apply the analysis of these properties occurs when looking for a
conversion rate from one currency to another. We already know how to find a symbol with the right
combination of base and quote currency, but the difficulty is that there may be several such symbols.
Reading properties like SYMBOL_SECTOR_NAME (you need to look for "Currency" or a synonym; check
with your broker's specification) or SYMBOL_PATH can help in such cases.

---

## Page 1129

Part 6. Trading automation
1 1 29
6.1  Financial instruments and Market Watch
The SYMBOL_PATH contains the entire hierarchy of folders in the symbols directory that contain the
specific symbol: folder names are separated by backslashes ('\\') in the same way as the file system.
The last element of the path is the name of the symbol itself.
Some string properties have integer counterparts. In particular, instead of SYMBOL_SECTOR_NAME,
you can use the SYMBOL_SECTOR property, which returns an enumeration member
ENUM_SYMBOL_SECTOR with all supported sectors. By analogy, for SYMBOL_INDUSTRY_NAME there
is a similar property SYMBOL_INDUSTRY with the ENUM_SYMBOL_INDUSTRY enumeration type.
If necessary, an MQL program can even find the background color used when displaying a symbol in the
Market Watch by simply reading the SYMBOL_BACKGROUND_COLOR property. This will allow those
programs that create their own interface on the chart using graphic objects (dialog boxes, lists, etc.),
to make it unified with the native terminal controls.
Consider the example script SymbolFilterDescription.mq5, which outputs four predefined text properties
for Market Watch symbols. The first of them is SYMBOL_DESCRIPTION (not to be confused with the
name of the symbol itself), and it is by it that the resulting list will be sorted. The other three are purely
for reference: SYMBOL_SECTOR_NAME, SYMBOL_COUNTRY, SYMBOL_PATH. All values are filled in in a
specific way for each broker (there may be discrepancies for the same ticker).
We haven't mentioned it, but our SymbolFilter class implements a special overload of the equal method
to compare strings. It supports searching for the occurrence of a substring with a pattern in which the
wildcard character '*' stands for 0 or more arbitrary characters. For example, "*ian*" will find all
characters that contain the substring "ian" (anywhere), and "*Index" will only find strings ending in
"Index".
This feature resembles a substring search in the Symbols dialog available to users. However, there is no
need to specify a wildcard character, because a substring is always searched for. In the algorithm
which can be found in the source codes (SymbolFilter.mqh), we left the possibility to search for either a
full match (there are no '*' characters) or a substring (there is at least one asterisk).
The comparison is case-sensitive. If necessary, it is easy to adapt the code for comparison without
distinguishing between lowercase and uppercase letters.
Given the new feature, let's define an input variable for the search string in the description of the
symbols. If the variable is empty, all symbols from the Market Watch window will be displayed.
input string SearchPattern = "";
Further, everything is as usual.

---

## Page 1130

Part 6. Trading automation
1 1 30
6.1  Financial instruments and Market Watch
void OnStart()
{
   SymbolFilter f;                      // filter object
   string symbols[];                    // array of names
   string text[][4];                    // array of vectors with data
   
   // properties to read
   ENUM_SYMBOL_INFO_STRING fields[] =
   {
      SYMBOL_DESCRIPTION,
      SYMBOL_SECTOR_NAME,
      SYMBOL_COUNTRY,
      SYMBOL_PATH
   };
   
   if(SearchPattern != "")
   {
      f.let(SYMBOL_DESCRIPTION, SearchPattern);
   }
   
   // apply the filter and get arrays sorted by description
   f.select(true, fields, symbols, text, true);
   
   const int n = ArraySize(symbols);
   PrintFormat("===== Text fields for symbols (%d) =====", n);
   for(int i = 0; i < n; ++i)
   {
      Print(symbols[i] + ":");
      ArrayPrint(text, 0, NULL, i, 1, 0);
   }
}
Here is a possible version of the list (with abbreviations).

---

## Page 1131

Part 6. Trading automation
1 1 31 
6.1  Financial instruments and Market Watch
===== Text fields for symbols (16) =====
AUDUSD:
"Australian Dollar vs US Dollar" "Currency"   ""   "Forex\AUDUSD"
EURUSD:
"Euro vs US Dollar" "Currency"   ""   "Forex\EURUSD"     
UK100:
"FTSE 100 Index" "Undefined"   ""   "Indexes\UK100" 
XAUUSD:
"Gold vs US Dollar" "Commodities"   ""   "Metals\XAUUSD"    
JAGG:
    "JPMorgan U.S. Aggregate Bond ETF"  "Financial"
    "USA"   "ETF\United States\NYSE\JPMorgan\JAGG"
NZDUSD:
"New Zealand Dollar vs US Dollar" "Currency"   ""   "Forex\NZDUSD"
GBPUSD:
"Pound Sterling vs US Dollar" "Currency"   ""   "Forex\GBPUSD"
SP500m:
"Standard & Poor's 500" "Undefined"   ""   "Indexes\SP500m"
FIHD:
    "UBS AG FI Enhanced Global High Yield ETN" "Financial"
    "USA"   "ETF\United States\NYSE\UBS\FIHD"         
...
If we enter the search string "*ian*" into the input variable SearchPattern, we get the following result.
===== Text fields for symbols (3) =====
AUDUSD:
"Australian Dollar vs US Dollar" "Currency"   ""   "Forex\AUDUSD"
USDCAD:
"US Dollar vs Canadian Dollar" "Currency"   ""   "Forex\USDCAD"
USDRUB:
"US Dollar vs Russian Ruble" "Currency"   ""   "Forex\USDRUB"
6.1 .22 Depth of Market
When it comes to exchange instruments, MetaTrader 5 allows you to get not only price and volume
information packed in ticks, but also the Depth of Market (order book, level II prices), that is, the
distribution of volumes in placed buy and sell orders at several nearest levels around the current price.
One of the integer properties of the symbol SYMBOL_TICKS_BOOKDEPTH contains the maximum
number of levels shown in the Depth of Market. This amount is allowed for each of the parties, that is,
the total size of the order book can be two times larger (and this does not take into account price
levels with zero volumes that are not broadcast).
Depending on the market situation, the actual size of the transmitted order book may become smaller
than indicated in this property. For non-exchange instruments, this property is usually equal to 0,
although some brokers can broadcast the order book for Forex symbols, limited only by the orders of
their clients.
The order book itself and notifications about its update must be requested by the interested MQL
program using a special API, which we will discuss in the next chapter.
It should be noted that due to the architectural features of the platform, this property is not
directly related to the translation of the order book, that is, it is just a specification field filled in by

---

## Page 1132

Part 6. Trading automation
1 1 32
6.1  Financial instruments and Market Watch
the broker. In other words, a non-zero value of the property does not mean that the order book will
necessarily arrive at the terminal in an open market. This depends on other server settings and
whether it has an active connection to the data provider.
Let's try to get statistics on the depth of the market for all or selected symbols using the script
SymbolFilterBookDepth.mq5.
input bool UseMarketWatch = false;
input int ShowSymbolsWithDepth = -1;
Parameter ShowSymbolsWithDepth, which is equal to -1  by default, instructs to collect statistics on
different Depth of Market settings among all symbols. If you set the parameter to a different value, the
program will try to find all symbols with the specified order book depth.
void OnStart()
{
   SymbolFilter f;                // filter object
   string symbols[];              // array for symbol names
   long depths[];                 // array of property values
   MapArray<long,int> stats;      // counters of occurrences of each depth
   
   if(ShowSymbolsWithDepth > -1)
   {
      f.let(SYMBOL_TICKS_BOOKDEPTH, ShowSymbolsWithDepth);
   }
   
   // apply filter and fill arrays
   f.select(UseMarketWatch, SYMBOL_TICKS_BOOKDEPTH, symbols, depths, true);
   const int n = ArraySize(symbols);
   
   PrintFormat("===== Book depths for %s symbols %s=====",
      (UseMarketWatch ? "Market Watch" : "all available"),
      (ShowSymbolsWithDepth > -1 ? "(filtered by depth="
      + (string)ShowSymbolsWithDepth + ") " : ""));
   PrintFormat("Total symbols: %d", n);
   ...
If a specific depth is given, we simply output an array of symbols (they all satisfy the filter condition),
and exit.
   if(ShowSymbolsWithDepth > -1)
   {
      ArrayPrint(symbols);
      return;
   }
   ...
Otherwise, we calculate the statistics and display them.

---

## Page 1133

Part 6. Trading automation
1 1 33
6.1  Financial instruments and Market Watch
   for(int i = 0; i < n; ++i)
   {
      stats.inc(depths[i]);
   }
   
   Print("Stats per depth:");
   stats.print();
   Print("Legend: key=depth, value=count");
}
With the default settings, we can get the following picture.
===== Book depths for all available symbols =====
Total symbols: 52357
Stats per depth:
    [key] [value]
[0]     0   52244
[1]     5       3
[2]    10      67
[3]    16       5
[4]    20      13
[5]    32      25
Legend: key=depth, value=count
If you set ShowSymbolsWithDepth to one of the detected values, for example, 32, we get a list of
symbols with this order book depth.
===== Book depths for all available symbols (filtered by depth=32) =====
Total symbols: 25
[ 0] "USDCNH" "USDZAR" "USDHUF" "USDPLN" "EURHUF" "EURNOK" "EURPLN" "EURSEK" "EURZAR" "GBPNOK" "GBPPLN" "GBPSEK" "GBPZAR"
[13] "NZDCAD" "NZDCHF" "USDMXN" "EURMXN" "GBPMXN" "CADMXN" "CHFMXN" "MXNJPY" "NZDMXN" "USDCOP" "USDARS" "USDCLP"
6.1 .23 Custom symbol properties
In the introduction to this chapter, we mentioned custom symbols. These are the symbols with the
quotes created directly in the terminal at the user's command or programmatically.
Custom symbols can be used, for example, to create a synthetic instrument based on a formula that
includes other Market Watch symbols. This is available to the user directly in the terminal interface.
An MQL program can implement more complex scenarios in MQL5, such as merging different
instruments for different periods, generating series according to a given random distribution, or
receiving data (quotes, bars, or ticks) from external sources.
In order to be able to distinguish a standard symbol from a custom symbol in algorithms, MQL5
provides the SYMBOL_CUSTOM property, which is a logical sign that a symbol is custom.
If the symbol has a formula, it is available through the SYMBOL_FORMULA string property. In formulas,
as you know, you can use the names of other symbols, as well as mathematical functions and
operators. Here are some examples:
• Synthetic symbol: "@ESU1 9"/EURCAD
• Calendar spread: "Si-9.1 3"-"Si-6.1 3"

---

## Page 1134

Part 6. Trading automation
1 1 34
6.1  Financial instruments and Market Watch
• Euro index: 34.38805726 * pow(EURUSD,0.31 55) * pow(EURGBP,0.3056) * pow(EURJPY,0.1 891 )
* pow(EURCHF,0.1 1 1 3) * pow(EURSEK,0.0785)
Specifying a formula is convenient for the user, but usually not used from MQL programs since they can
calculate formulas directly in the code, with non-standard functions and with more control, in
particular, on each tick and not on a timer 1  time per 1 00ms.
Let's check the work with properties in the script SymbolFilterCustom.mq5: it logs all custom symbols
and their formulas (if any).
input bool UseMarketWatch = false;
   
void OnStart()
{
   SymbolFilter f;                // filter object
   string symbols[];              // array for symbol names
   string formulae[];             // array for formulas
   
 // apply filter and fill arrays
   f.let(SYMBOL_CUSTOM, true)
   .select(UseMarketWatch, SYMBOL_FORMULA, symbols, formulae);
   const int n = ArraySize(symbols);
   
   PrintFormat("===== %s custom symbols =====",
      (UseMarketWatch ? "Market Watch" : "All available"));
   PrintFormat("Total symbols: %d", n);
   
   for(int i = 0; i < n; ++i)
   {
      Print(symbols[i], " ", formulae[i]);
   }
}
Below is the result with the only custom character found.
===== All available custom symbols =====
Total symbols: 1
synthEURUSD SP500m/UK100
6.1 .24 Specific properties (stock exchange, derivatives, bonds)
In this final section of the chapter, we will briefly review other symbol properties that are outside the
scope of the book but may be useful for implementing advanced trading strategies. Detailed information
about these properties can be found in the MQL5 documentation.
As you know, MetaTrader 5 allows you to trade derivatives market instruments, including options,
futures, and bonds. This is reflected in the software interface as well. The MQL5 API provides a lot of
specific symbol properties related to the mentioned instrument categories.
In particular, for options, this is the circulation period (the start date SYMBOL_START_TIME and the
end date SYMBOL_EXPIRATION_TIME of trading), the strike price (SYMBOL_OPTION_STRIKE), the
right to buy or sell (SYMBOL_OPTION_RIGHT, Call/Put), European or American type
(SYMBOL_OPTION_MODE) depending on the possibility of early exercising, day-to-day change in closing

---

## Page 1135

Part 6. Trading automation
1 1 35
6.1  Financial instruments and Market Watch
prices (SYMBOL_PRICE_CHANGE) and volatility (SYMBOL_PRICE_VOLATILITY), as well as estimated
coefficients (the Greeks) characterizing the dynamics of price behavior.
For bonds, the accumulated coupon income (SYMBOL_TRADE_ACCRUED_INTEREST), face value
(SYMBOL_TRADE_FACE_VALUE), liquidity ratio (SYMBOL_TRADE_LIQUIDITY_RATE) are of particular
interest.
For futures – open interest (SYMBOL_SESSION_INTEREST) and total order volumes by buy
(SYMBOL_SESSION_BUY_ORDERS_VOLUME) and sell (SYMBOL_SESSION_SELL_ORDERS_VOLUME),
clearing price at the close of the trading session (SYMBOL_SESSION_PRICE_SETTLEMENT).
Apart from the current market data that make up a tick, MQL5 allows you to know their daily range:
the maximum and minimum values for each of the tick fields. For example, SYMBOL_BIDHIGH is the
maximum Bid per day, and SYMBOL_BIDLOW is the minimum. Note that the properties
SYMBOL_VOLUMEHIGH, SYMBOL_VOLUMELOW (of type long) actually duplicate, but only with less
precision, the volumes in SYMBOL_VOLUMEHIGH_REAL and SYMBOL_VOLUMELOW_REAL (double).
Information about the Last prices and volumes is available, as a rule, only for exchange symbols.
Bear in mind that filling in the properties depends on the settings of the server implemented by the
broker.
6.2 Depth of market
In addition to several types of up-to-date market price data (Ask/Bid/Last) and the last traded volumes
are received in the terminal in the form of ticks, MetaTrader 5 supports the Depth of Market (order
book), which is an array of records about the volumes of placed buy and sell orders around the current
market price. Volumes are aggregated at several levels above and below the current price, with the
smallest increment of price movement according to the symbol specification. As we have seen, the
maximum order book size (number of price levels) is set in the SYMBOL_TICKS_BOOKDEPTH symbol
property.
Terminal users know the Depth of Market feature in the interface and its operating principles. If you
need further details, please see the documentation.
The order book contains extended market information which is commonly referred to as "market
depth". Knowing it allows you to create more sophisticated trading systems.
Indeed, information about a tick is only a small slice of the order book. In a somewhat simplified sense,
a tick is a 2-level order book with one nearest Ask price (available offer) and one nearest Bid price
(available demand). Furthermore, ticks do not provide order volumes at these prices.
Depth of Market changes can occur much more frequently than ticks, since they affect not only the
reaction to concluded deals but also changes in the volume of pending limit orders in the Depth of
Market.
Usually, data providers for the order book and quotes (ticks, deals) are different instances, and tick
events (OnTick in Expert Advisors or OnCalculate in indicators) do not match the Depth of Market
events. Both threads arrive asynchronously and in parallel but eventually end up in the event queue of
an MQL program.
It is important to note that an order book is available, as a rule, for exchange instruments, but there
are exceptions both in one direction and in the other:

---

## Page 1136

Part 6. Trading automation
1 1 36
6.2 Depth of Market
• Depth of Market may be missing for one reason or another for an exchange instrument
• Depth of Market can be provided by a broker for an OTC instrument based on the information they
have collected about their clients' orders
In MQL5, Depth of Market data is available for Expert Advisors and indicators. By using special functions
(MarketBookAdd, MarketBookRelease), programs can enable or disable their subscription to receive
notifications about Depth of Market changes in the platform. To receive the notifications, the program
must define the OnBookEvent event handler function in its code. After receiving a notification, the order
book data can be read using the MarketBookGet function.
The terminal maintains the history of quotes and ticks, but not of the Depth of Market data. In
particular, the user or an MQL program can download the history at the required retrospective (if
the broker has it) and test Expert Advisors and indicators on it. 
In contrast, the Depth of Market is only broadcast online and is not available in the tester. A broker
does not have an archive of Depth of Market data on the server. To emulate the behavior of the
order book in the tester, you should collect the Depth of Market history online and then read it from
the MQL program running in the tester. You can find ready-made products in the MQL5 Market.
6.2.1  Managing subscriptions to Depth of Market events
The terminal receives the Depth of Market information on a subscription basis: an MQL program must
express its intent to receive Depth of Market (order book) events or, conversely, terminate its
subscription by calling the appropriate functions, MarketBookAdd and MarketBookRelease.
The MarketBookAdd function subscribes to receive notifications about changes in the order book for
the specified instrument. Thus, you can subscribe to order books for many instruments, and not just
the working instrument of the current chart.
bool MarketBookAdd(const string symbol)
Usually, this function is called from OnInit or in the class constructor of a long-lived object.
Notifications about the order book change are sent to the program in the form of OnBookEvent events,
therefore, to process them, the program must have a handler function of the same name.
If the specified symbol was not selected in the Market Watch before calling the function, it will be
added to the window automatically.
The MarketBookRelease function unsubscribes from notifications about changes in the specified order
book.
bool MarketBookRelease(const string symbol)
As a rule, this function should be called from OnDeinit or from the class destructor of a long-lived
object.
Both functions return a value true if successful and false otherwise.
For all applications running on the same chart, separate subscription counters are maintained by
symbols. In other words, there can be several subscriptions to different symbols on the chart, and each
of them has its own counter.
Subscription or unsubscription by a single call of any of the functions changes the subscription counter
only for a specific symbol, on a specific chart where the program is running. This means that two

---

## Page 1137

Part 6. Trading automation
1 1 37
6.2 Depth of Market
charts can have subscriptions to OnBookEvent events of the same symbol, but with different values of
subscription counters.
The initial value of the subscription counter is zero. On every call of MarketBookAdd, the subscription
counter for the specified symbol on the given chart is incremented by 1  (the chart symbol and the
symbol in MarketBookAdd do not have to match). When calling MarketBookRelease, the counter of
subscriptions to the specified symbol within the chart decreases by 1 .
OnBookEvent events for any symbol within the chart are generated as long as the subscription counter
for this symbol is greater than zero. Therefore, it is important that every MQL program that contains
MarketBookAdd calls, upon completion of its work, correctly unsubscribes from receiving events for
each symbol using MarketBookRelease. For this, you should make sure that the number of
MarketBookAdd calls and MarketBookRelease calls match. MQL5 does not allow you to find out the
value of the counter.
The first example is a simple bufferless indicator MarketBookAddRelease.mq5, which enables a
subscription to the order book at the time of launch and disables it when it is unloaded. In the
WorkSymbol input parameter, you can specify a symbol to subscribe. If it is left empty (default value),
the subscription will be initiated for the working symbol of the current chart.
input string WorkSymbol = ""; // WorkSymbol (empty means current chart symbol)
   
const string _WorkSymbol = StringLen(WorkSymbol) == 0 ? _Symbol : WorkSymbol;
string symbols[];
   
void OnInit()
{
   const int n = StringSplit(_WorkSymbol, ',', symbols);
   for(int i = 0; i < n; ++i)
   {
      if(!PRTF(MarketBookAdd(symbols[i])))
         PrintFormat("MarketBookAdd(%s) failed", symbols[i]);
   }
}
   
int OnCalculate(const int rates_total, const int prev_calculated, const int, const double &price[])
{
   return rates_total;
}
   
void OnDeinit(const int)
{
   for(int i = 0; i < ArraySize(symbols); ++i)
   {
      if(!PRTF(MarketBookRelease(symbols[i])))
         PrintFormat("MarketBookRelease(%s) failed", symbols[i]);
   }
}
As an additional feature, it is allowed to specify several instruments separated by commas. In this case,
a subscription to all will be requested.

---

## Page 1138

Part 6. Trading automation
1 1 38
6.2 Depth of Market
When the indicator is launched, a sign of subscription success or an error code is displayed in the log.
The indicator then tries to unsubscribe from the events in the OnDeinit handler.
With the default settings, on the chart with the symbol for which the order book is available, we will get
the following entries in the log.
MarketBookAdd(symbols[i])=true / ok
MarketBookRelease(symbols[i])=true / ok
If you put the indicator on a chart with a symbol without the order book, we will see error codes.
MarketBookAdd(symbols[i])=false / BOOKS_CANNOT_ADD(4901)
MarketBookAdd(XPDUSD) failed
MarketBookRelease(symbols[i])=false / BOOKS_CANNOT_DELETE(4902)
MarketBookRelease(XPDUSD) failed
You can experiment by specifying in the input parameter WorkSymbol existing or missing characters.
We will consider the case of subscribing to order books of several symbols in the next section.
6.2. Receiving events about changes in the Depth of Market
The OnBookEvent event is generated by the terminal when the order book status changes. The event is
processed by the OnBookEvent function defined in the source code. In order for the terminal to start
sending OnBookEvent notifications to the MQL program for a specific symbol, you must first subscribe
to receive them using the MarketBookAdd function.
To unsubscribe from receiving the OnBookEvent event for a symbol, call the MarketBookRelease
function.
The OnBookEvent event is broadcast, which means that it is enough for one MQL program on the chart
to subscribe to OnBookEvent events, and all other programs on the same chart will also start receiving
the events provided they have the OnBookEvent handler in the code. Therefore, it is necessary to
analyze the name of the symbol, which is passed to the handler as a parameter.
The OnBookEvent handler prototype is as follows.
void OnBookEvent(const string &symbol)
OnBookEvent events are queued even if the processing of the previous OnBookEvent event has not yet
been completed.
It is important that the events OnBookEvent are only notifications and do not provide the state of the
order book. To get the Depth of Market data, call the MarketBookGet function.
It should be noted, however, that the MarketBookGet call, even if it is made directly from the
OnBookEvent handler, will receive the current state of the order book at the time when MarketBookGet
is called, which does not necessarily match the order book state that triggered sending of the event
OnBookEvent. This can happen when a sequence of very fast order book changes arrives at the
terminal.
In this regard, in order to obtain the most complete chronology of Depth of Market changes, we need
to write an implementation of OnBookEvent and prioritize the optimization by the execution speed.
At the same time, there is no guaranteed way to get all unique Depth of Market states in MQL5.

---

## Page 1139

Part 6. Trading automation
1 1 39
6.2 Depth of Market
If your program started receiving notifications successfully, and then they disappeared when the
market was open (and ticks continue to come), this may indicate problems in the subscription. In
particular, another MQL program which is poorly designed could unsubscribe more times than
required. In such cases, it is recommended to resubscribe with a new MarketBookAdd call after a
predefined timeout (for example, several tens of seconds or a minute).
An example of bufferless indicator MarketBookEvent.mq5 allows you to track the arrival of OnBookEvent
events and prints the symbol name and the current time (millisecond system counter) in a comment.
For clarity, we use the multi-line comment function from the Comments.mqh file, section Displaying
messages in the chart window.
Interestingly, if you leave the input parameter WorkSymbol empty (default value), the indicator itself
will not initiate a subscription to the order book but will be able to intercept messages requested by
other MQL programs on the same chart. Let's check it.
#include <MQL5Book/Comments.mqh>
   
input string WorkSymbol = ""; // WorkSymbol (if empty, intercept events initiated by others)
   
void OnInit()
{
   if(StringLen(WorkSymbol))
   {
      PRTF(MarketBookAdd(WorkSymbol));
   }
   else
   {
      Print("Start listening to OnBookEvent initiated by other programs");
   }
}
   
void OnBookEvent(const string &symbol)
{
   ChronoComment(symbol + " " + (string)GetTickCount());
}
void OnDeinit(const int)
{
   Comment("");
   if(StringLen(WorkSymbol))
   {
      PRTF(MarketBookRelease(WorkSymbol));
   }
}
Let's run MarketBookEvent with default settings (no own subscription) and then add the
MarketBookAddRelease indicator from the previous section, and specify for it a list of several symbols
with available order books (in the example below, this is "XAUUSD,BTCUSD,USDCNH"). It doesn't
matter which chart to run the indicators on: it can be a completely different symbol, like EURUSD.
Immediately after launching MarketBookEvent, the chart will be empty (no comments) because there
are no subscriptions yet. Once MarketBookAddRelease starts (three lines should appear in the log with
the status of a successful subscription equal to true), the names of symbols will begin to appear in the

---

## Page 1140

Part 6. Trading automation
1 1 40
6.2 Depth of Market
comments alternately as their order books are updated (we have not yet learned how to read the order
book; this will be discussed in the next section).
Here's how it looks on the screen.
N otifications about changes in the order books of "third-party" symbols
If we now remove the MarketBookAddRelease indicator, it will cancel its subscriptions, and the
comment will stop updating. Subsequent removal of MarketBookEvent will clear the comment.
Please note that some time (a second or two) passes between the request to unsubscribe and the
moment when Depth of Market events actually stop updating the comment.
You can run the MarketBookEvent indicator alone on the chart, specifying some symbol in its
WorkSymbol parameter to make sure notifications work within the same app. MarketBookAddRelease
was previously used only to demonstrate the broadcast nature of notifications. In other words, enabling
a subscription to order book changes in one program does affect the receipt of notifications in another.
6.2.3 Reading the current Depth of Market data
After successful execution of the MarketBookAdd function, an MQL program can query the order book
states using the MarketBookGet function upon the arrival of OnBookEvent events. The MarketBookGet
function populates the MqlBookInfo array of structures passed by reference with the Depth of Market
values of the specified symbol.
bool MarketBookGet(string symbol, MqlBookInfo &book[])
For the receiving array, you can pre-allocate memory for a sufficient number of records. If the dynamic
array has zero or insufficient size, the terminal itself will allocate memory for it.
The function returns an indication of success (true) or error (false).

---

## Page 1141

Part 6. Trading automation
1 1 41 
6.2 Depth of Market
MarketBookGet is usually utilized directly in the OnBookEvent handler code or in functions called from
it.
A separate record about the Depth of Market price level is stored in the MqlBookInfo structure.
struct MqlBookInfo 
{ 
   ENUM_BOOK_TYPE type;            // request type 
   double         price;           // price 
   long           volume;          // volume 
   double         volume_real;     // volume with increased accuracy 
};
The enumeration ENUM_BOOK_TYPE contains the following members.
Identifier
Description
BOOK_TYPE_SELL
Request to sell
BOOK_TYPE_BUY
Request to buy
BOOK_TYPE_SELL_MARKET
Request to sell at market price
BOOK_TYPE_BUY_MARKET
Request to buy at market price
In the order book, sell orders are located in its upper half and buy orders are placed in the lower half.
As a rule, this leads to a sequence of elements from high prices to low prices. In other words, below the
0th index, there is the highest price, and the last entry is the lowest one, while between them prices
decrease gradually. In this case, the minimum price step between the levels is
SYMBOL_TRADE_TICK_SIZE, however, levels with zero volumes are not translated, that is, adjacent
elements can be separated by a large amount.
In the terminal user interface, the order book window provides an option to enable/disable Advanced
Mode, in which levels with zero volumes are also displayed. But by default, in the standard mode, such
levels are hidden (skipped in the table).
In practice, the order book content can sometimes contradict the announced rules. In particular, some
buy or sell requests may fall into the opposite half of the order book (probably, someone placed a buy
at an unfavorably high price or a sell at an unfavorably low price, but the provider can also have data
aggregation errors). As a result, due to the observance of the priority "all sell orders from above, all
buy orders from below", the sequence of prices in the order book will be violated (see example below).
In addition, repeated values of prices (levels) can be found both in one half of the order book and in the
opposite ones.
In theory, the coincidence of buy and sell prices in the middle of the order book is correct. It means
zero spread. However, unfortunately, duplicate levels also happen at a greater depth of the order book.
When we say "half" of the order book, it should not be taken literally. Depending on liquidity, the
number of supply and demand levels may not match. In general, the book is not symmetrical.
The MQL program must check the correctness of the order book (in particular, the price sorting order)
and be ready to handle potential deviations.

---

## Page 1142

Part 6. Trading automation
1 1 42
6.2 Depth of Market
Less serious abnormal situations (which, nevertheless, should be taken into account in the algorithm)
include:
·Consecutive identical order books, without changes
·Empty order book
·Order book with one level
Below is a fragment of a real Depth of Market received from a broker. The letters 'S' and 'B' mark,
respectively, the prices of sell and buy requests.
Note that the buy and sell levels actually overlap: visually, this is not very noticeable, because all the
'S' records in the order book are specially placed up (the beginning of the receiving array), and the 'B'
records are down (the end of the array). However, take a closer look: the buy prices in elements 20
and 21  are 1 43.23 and 1 38.86, respectively, and this is more than all sell offers. And, at the same
time, the selling prices in elements 1 8 and 1 9 are 1 34.62 and 1 33.55, which is lower than all buy
offers.
...
10 S 138.48 652
11 S 138.47 754
12 S 138.45 2256
13 S 138.43 300
14 S 138.42 14
15 S 138.40 1761
16 S 138.39 670    // Duplicate
17 S 138.11 200
18 S 134.62 420    // Low
19 S 133.55 10627  // Low
20 B 143.23 9564   // High
21 B 138.86 533    // High
22 B 138.39 739    // Duplicate
23 B 138.38 106
24 B 138.31 100
25 B 138.25 29
26 B 138.24 6072
27 B 138.23 571
28 B 138.21 17
29 B 138.20 201
30 B 138.19 1
...
In addition, the price of 1 38.39 is found both in the upper half at number 1 6 and in the lower half at
number 22.
Errors in the order book are most likely to be present in extreme conditions: with strong volatility or
lack of liquidity.
Let's check the receipt of the order book using the indicator MarketBookDisplay.mq5. It will subscribe
to Depth of Market events for the specified symbol in the parameter WorkSymbol (if you leave an
empty line there, the working symbol of the current chart is assumed).

---

## Page 1143

Part 6. Trading automation
1 1 43
6.2 Depth of Market
input string WorkSymbol = ""; // WorkSymbol (if empty, use current chart symbol)
   
const string _WorkSymbol = StringLen(WorkSymbol) == 0 ? _Symbol : WorkSymbol;
int digits;
   
void OnInit()
{
   PRTF(MarketBookAdd(_WorkSymbol));
   digits = (int)SymbolInfoInteger(_WorkSymbol, SYMBOL_DIGITS);
   ...
}
   
void OnDeinit(const int)
{
   Comment("");
   PRTF(MarketBookRelease(_WorkSymbol));
}
The OnBookEvent handler is defined in the code for handling events, in which MarketBookGet is called,
and all elements of the resulting MqlBookInfo array output as a multiline comment.

---

## Page 1144

Part 6. Trading automation
1 1 44
6.2 Depth of Market
void OnBookEvent(const string &symbol)
{
   if(symbol == _WorkSymbol) // take only order books of the requested symbol
   {
      MqlBookInfo mbi[];
      if(MarketBookGet(symbol, mbi)) // getting the current order book
      {
         ...
         int half = ArraySize(mbi) / 2; // estimate of the middle of the order book
         bool correct = true;
         // collect information about levels and volumes in one line (with hyphens)
         string s = "";
         for(int i = 0; i < ArraySize(mbi); ++i)
         {
            s += StringFormat("%02d %s %s %d %g\n", i,
               (mbi[i].type == BOOK_TYPE_BUY ? "B" : 
               (mbi[i].type == BOOK_TYPE_SELL ? "S" : "?")),
               DoubleToString(mbi[i].price, digits),
               mbi[i].volume, mbi[i].volume_real);
               
            if(i > 0) // look for the middle of the order book as a change in request type
            {
               if(mbi[i - 1].type == BOOK_TYPE_SELL
                  && mbi[i].type == BOOK_TYPE_BUY)
               {
                  half = i; // this is the middle, because there has been a type change
               }
               
               if(mbi[i - 1].price <= mbi[i].price)
               {
                  correct = false; // reverse order = data problem
               }
            }
         }
         Comment(s + (!correct ? "\nINCORRECT BOOK" : ""));
         ...
      }
   }
}
Since the order book changes rather quickly, it is not very convenient to follow the comment.
Therefore, we will add a couple of buffers to the indicator, in which we will display the contents of two
halves of the order book as histograms: sell and buy separately. The zero bar will correspond to the
central levels that form the spread. With an increase in bar numbers, there is an increase in the "depth
of the market", that is, more and more distant price levels are displayed there: in the upper histogram,
this means lower prices with buy orders, and in the lower one there are higher prices with sell orders.

---

## Page 1145

Part 6. Trading automation
1 1 45
6.2 Depth of Market
#property indicator_separate_window
#property indicator_plots 2
#property indicator_buffers 2
   
#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2
#property indicator_label1  "Buys"
   
#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  clrOrangeRed
#property indicator_width2  2
#property indicator_label2  "Sells"
   
double buys[], sells[];
Let's provide an opportunity to visualize the order book in standard and extended modes (that is, skip
or show levels with zero volumes), as well as display the volumes themselves in fractions of lots or units.
Both options have analogs in the built-in Depth of Market window.
input bool AdvancedMode = false;
input bool ShowVolumeInLots = false;
Let's set buffers and obtaining of some symbol properties (which we will need later) in OnInit.
int depth, digits;
double tick, contract;
   
void OnInit()
{
   ...
   // setting indicator buffers
   SetIndexBuffer(0, buys);
   SetIndexBuffer(1, sells);
   ArraySetAsSeries(buys, true);
   ArraySetAsSeries(sells, true);
   // getting the necessary symbol properties
   depth = (int)PRTF(SymbolInfoInteger(_WorkSymbol, SYMBOL_TICKS_BOOKDEPTH));
   tick = SymbolInfoDouble(_WorkSymbol, SYMBOL_TRADE_TICK_SIZE);
   contract = SymbolInfoDouble(_WorkSymbol, SYMBOL_TRADE_CONTRACT_SIZE);
}
Let's add buffer filling to the handler OnBookEvent .

---

## Page 1146

Part 6. Trading automation
1 1 46
6.2 Depth of Market
#define VOL(V) (ShowVolumeInLots ? V / contract : V)
   
void OnBookEvent(const string &symbol)
{
   if(symbol == _WorkSymbol) // take only order books of the requested symbol
   {
      MqlBookInfo mbi[];
      if(MarketBookGet(symbol, mbi)) // getting the current order book
      {
         // clear the buffers to the depth with 10 times the margin of the maximum depth,
         // because extended mode can have a lot of empty elements
         for(int i = 0; i <= depth * 10; ++i)
         {
            buys[i] = EMPTY_VALUE;
            sells[i] = EMPTY_VALUE;
         }
         ...// further along we form and display the comment as before
         if(!correct) return;
         
         // filling buffers with data
         if(AdvancedMode) // show skips enabled
         {
            for(int i = 0; i < ArraySize(mbi); ++i)
            {
               if(i < half)
               {
                  int x = (int)MathRound((mbi[i].price - mbi[half - 1].price) / tick);
                  sells[x] = -VOL(mbi[i].volume_real);
               }
               else
               {
                  int x = (int)MathRound((mbi[half].price - mbi[i].price) / tick);
                  buys[x] = VOL(mbi[i].volume_real);
               }
            }
         }
         else // standard mode: show only significant elements
         {
            for(int i = 0; i < ArraySize(mbi); ++i)
            {
               if(i < half)
               {
                  sells[half - i - 1] = -VOL(mbi[i].volume_real);
               }
               else
               {
                  buys[i - half] = VOL(mbi[i].volume_real);
               }
            }
         }
      }

---

## Page 1147

Part 6. Trading automation
1 1 47
6.2 Depth of Market
   }
}
The following image demonstrates how the indicator works with settings AdvancedMode=true,
ShowVolumeInLots=true.
The contents of the order book in the MarketBookDisplay.mq5 indicator on the USDCN H chart
Buys are displayed as positive values (blue bar at the top), and sells are negative (red at the bottom).
For clarity, there is a standard Depth of Market window on the right with the same settings (in
advanced mode, volumes in lots), so you can make sure that the values match.
It should be noted that the indicator may not have time to redraw quickly enough to keep
synchronization with the built-in order book. This does not mean that the MQL program did not receive
the event in time, but only a side effect of asynchronous chart rendering. Working algorithms usually
have analytical processing and order placing with the order book, rather than visualization.
In this case, updating the chart is implicitly requested at the time of calling the Comment function.
6.2.4 Using Depth of Market data in applied algorithms
The Depth of Market is considered to be a very useful technology for developing advanced trading
systems. In particular, analysis of the distribution of Depth of Market volumes at levels close to the
market allows you to find out in advance the average order execution price of a specific volume: simply
sum up the volumes of the levels (in the opposite direction) that will ensure its filling. In a thin market,
with insufficient volumes, the algorithm may refrain from opening a trade in order to avoid significant
price slippage.
Based on the Depth of Market data, other strategies can also be constructed. For example, it can be
important to know the price levels at which large volumes are located.

---

## Page 1148

Part 6. Trading automation
1 1 48
6.2 Depth of Market
MarketBookVolumeAlert.mq5
In the next test indicator MarketBookVolumeAlert.mq5, we implement a simple algorithm for tracking
volumes or their changes that exceed a given value.
#property indicator_chart_window
#property indicator_plots 0
   
input string WorkSymbol = ""; // WorkSymbol (if empty, use current chart symbol)
input bool CountVolumeInLots = false;
input double VolumeLimit = 0;
   
const string _WorkSymbol = StringLen(WorkSymbol) == 0 ? _Symbol : WorkSymbol;
There are no graphics in the indicator. The controlled symbol is entered in the WorkSymbol parameter
(if left blank, the chart's working symbol is implied). The minimum threshold of tracked objects, that is,
the sensitivity of the algorithm, is specified in the VolumeLimit parameter. Depending on the
CountVolumeInLots parameter, the volumes are analyzed and displayed to the user in lots (true) or
units (false). This also affects how the VolumeLimit value should be entered. The conversion from units
to fractions of lots is provided by the VOL macro: the contract size used in it contract is initialized in
OnInit (see below).
#define VOL(V) (CountVolumeInLots ? V / contract : V)
If large volumes are found above the threshold, the program will display a message about the
corresponding level in the comment. To save the nearest history of warnings, we use the class of multi-
line comments already known to us (Comments.mqh).
#define N_LINES 25                // number of lines in the comment buffer
#include <MQL5Book/Comments.mqh>
In the handler OnInit let's prepare the necessary settings and subscribe to the DOM events.
double contract;
int digits;
   
void OnInit()
{
   MarketBookAdd(_WorkSymbol);
   contract = SymbolInfoDouble(_WorkSymbol, SYMBOL_TRADE_CONTRACT_SIZE);
   digits = (int)MathRound(MathLog10(contract));
   Print(SymbolInfoDouble(_WorkSymbol, SYMBOL_SESSION_BUY_ORDERS_VOLUME));
   Print(SymbolInfoDouble(_WorkSymbol, SYMBOL_SESSION_SELL_ORDERS_VOLUME));
}
The SYMBOL_SESSION_BUY_ORDERS_VOLUME and SYMBOL_SESSION_SELL_ORDERS_VOLUME
properties, if they are filled by your broker for the selected symbol, will help you figure out which
threshold it makes sense to choose. By default, VolumeLimit is 0, which is why absolutely all changes in
the order book will generate warnings. To filter out insignificant fluctuations, it is recommended to set
VolumeLimit to a value that exceeds the average size of volumes at all levels (look in advance in the
built-in order book or in the MarketBookDisplay.mq5 indicator).
In the usual way, we implement the finalization.

---

## Page 1149

Part 6. Trading automation
1 1 49
6.2 Depth of Market
void OnDeinit(const int)
{
   MarketBookRelease(_WorkSymbol);
   Comment("");
}
The main work is done by the OnBookEvent processor. It describes a static array MqlBookInfo mbp to
store the previous version of the order book (since the last function call).
void OnBookEvent(const string &symbol)
{
   if(symbol != _WorkSymbol) return; // process only the requested symbol 
   
   static MqlBookInfo mbp[];      // previous table/book
   MqlBookInfo mbi[];
   if(MarketBookGet(symbol, mbi)) // read the current book
   {
      if(ArraySize(mbp) == 0) // first time we just save, because nothing to compare
      {
         ArrayCopy(mbp, mbi);
         return;
      }
      ...
If there is an old and a new order book, we compare the volumes at their levels with each other in
nested loops by i and j. Recall that an increase in the index means a decrease in price.

---

## Page 1150

Part 6. Trading automation
1 1 50
6.2 Depth of Market
      int j = 0;
      for(int i = 0; i < ArraySize(mbi); ++i)
      {
         bool found = false;
         for( ; j < ArraySize(mbp); ++j)
         {
            if(MathAbs(mbp[j].price - mbi[i].price) < DBL_EPSILON * mbi[i].price)
            {       // mbp[j].price == mbi[i].price
               if(VOL(mbi[i].volume_real - mbp[j].volume_real) >= VolumeLimit)
               {
                  NotifyVolumeChange("Enlarged", mbp[j].price,
                     VOL(mbp[j].volume_real), VOL(mbi[i].volume_real));
               }
               else
               if(VOL(mbp[j].volume_real - mbi[i].volume_real) >= VolumeLimit)
               {
                  NotifyVolumeChange("Reduced", mbp[j].price,
                     VOL(mbp[j].volume_real), VOL(mbi[i].volume_real));
               }
               found = true;
               ++j;
               break;
            }
            else if(mbp[j].price > mbi[i].price)
            {
               if(VOL(mbp[j].volume_real) >= VolumeLimit)
               {
                  NotifyVolumeChange("Removed", mbp[j].price,
                     VOL(mbp[j].volume_real), 0.0);
               }
               // continue the loop increasing ++j to lower prices
            }
            else // mbp[j].price < mbi[i].price
            {
               break;
            }
         }
         if(!found) // unique (new) price
         {
            if(VOL(mbi[i].volume_real) >= VolumeLimit)
            {
               NotifyVolumeChange("Added", mbi[i].price, 0.0, VOL(mbi[i].volume_real));
            }
         }
      }
      ...
Here, the emphasis is not on the type of level, but on the volume value only. However, if you wish, you
can easily add the designation of buys or sells to notifications, depending on the type field of that level
where the important change took place.
Finally, we save a new copy of mbi in a static array mbp to compare against it on the next function call.

---

## Page 1151

Part 6. Trading automation
1 1 51 
6.2 Depth of Market
      if(ArrayCopy(mbp, mbi) <= 0)
      {
         Print("ArrayCopy failed:", _LastError);
      }
      if(ArrayResize(mbp, ArraySize(mbi)) <= 0) // shrink if needed
      {
         Print("ArrayResize failed:", _LastError);
      }
   }
}
ArrayCopy does not automatically shrink a dynamic destination array if it happens to be larger than the
source array, so we explicitly set its exact size with ArrayResize.
An auxiliary function NotifyVolumeChange simply adds information about the found change to the
comment.
void NotifyVolumeChange(const string action, const double price,
   const double previous, const double volume)
{
   const string message = StringFormat("%s: %s %s -> %s",
      action,
      DoubleToString(price, (int)SymbolInfoInteger(_WorkSymbol, SYMBOL_DIGITS)),
      DoubleToString(previous, digits),
      DoubleToString(volume, digits));
   ChronoComment(message);
}
The following image shows the result of the indicator for settings CountVolumeInLots=false,
VolumeLimit=20.

---

## Page 1152

Part 6. Trading automation
1 1 52
6.2 Depth of Market
N otifications about volume changes in the order book
MarketBookQuasiTicks.mq5
As a second example of the possible use of the order book, let's turn to the problem of obtaining
multicurrency ticks. We have already touched on it in the section Generation of custom events, where
we have seen one of the possible solutions and the indicator EventTickSpy.mq5. Now, after getting
acquainted with the Depth of Market API, we can implement an alternative.
Let's create an indicator MarketBookQuasiTicks.mq5, which will subscribe to order books of a given list
of instruments and find the prices of the best offer and demand in them, that is, pairs of prices around
the spread, which are just prices Ask and Bid.
Of course, this information is not a complete equivalent of standard ticks (recall that trade/tick and
order book flows can come from completely different providers), but it provides an adequate and timely
view of the market.
New values of prices by symbols will be displayed in a multi-line comment.
The list of working symbols is specified in the SymbolList input parameter as a comma-separated list.
Enabling and disabling subscriptions to Depth of Market events is done in the OnInit and OnDeinit
handlers.

---

## Page 1153

Part 6. Trading automation
1 1 53
6.2 Depth of Market
#define N_LINES 25                // number of lines in the comment buffer
#include <MQL5Book/Comments.mqh>
   
input string SymbolList = "EURUSD,GBPUSD,XAUUSD,USDJPY"; // SymbolList (comma,separated,list)
   
const string WorkSymbols = StringLen(SymbolList) == 0 ? _Symbol : SymbolList;
string symbols[];
   
void OnInit()
{
   const int n = StringSplit(WorkSymbols, ',', symbols);
   for(int i = 0; i < n; ++i)
   {
      if(!MarketBookAdd(symbols[i]))
      {
         PrintFormat("MarketBookAdd(%s) failed with code %d", symbols[i], _LastError);
      }
   }
}
   
void OnDeinit(const int)
{
   for(int i = 0; i < ArraySize(symbols); ++i)
   {
      if(!MarketBookRelease(symbols[i]))
      {
         PrintFormat("MarketBookRelease(%s) failed with code %d", symbols[i], _LastError);
      }
   }
   Comment("");
}
The analysis of each new order book is carried out in OnBookEvent.

---

## Page 1154

Part 6. Trading automation
1 1 54
6.2 Depth of Market
void OnBookEvent(const string &symbol)
{
   MqlBookInfo mbi[];
   if(MarketBookGet(symbol, mbi)) // getting the current order book
   {
      int half = ArraySize(mbi) / 2; // estimate the middle of the order book
      bool correct = true;
      for(int i = 0; i < ArraySize(mbi); ++i)
      {
         if(i > 0)
         {
            if(mbi[i - 1].type == BOOK_TYPE_SELL
               && mbi[i].type == BOOK_TYPE_BUY)
            {
               half = i; // specify the middle of the order book
            }
            
            if(mbi[i - 1].price <= mbi[i].price)
            {
               correct = false;
            }
         }
      }
      
      if(correct) // retrieve the best Bid/Ask prices from the correct order book 
      {
         // mbi[half - 1].price // Ask
         // mbi[half].price     // Bid
         OnSymbolTick(symbol, mbi[half].price);
      }
   }
}
Found market Ask/Bid prices are passed to helper function OnSymbolTick to be displayed in a
comment.
void OnSymbolTick(const string &symbol, const double price)
{
   const string message = StringFormat("%s %s",
      symbol, DoubleToString(price, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)));
   ChronoComment(message);
}
If you wish, you can make sure that our synthesized ticks do not differ much from the standard ticks.
This is how information about incoming quasi-ticks looks on the chart.

---

## Page 1155

Part 6. Trading automation
1 1 55
6.2 Depth of Market
Multi-symbol quasi-ticks based on order book events
At the same time, it should be noted once again that order book events are available on the platform
online only, but not in the tester. If the trading system is built exclusively on quasi-ticks from the order
book, its testing will require the use of third-party solutions that ensure the collection and playback of
order books in the tester.
6.3 Trading account information
In this chapter, we will study the last important aspect of the trading environment of MQL programs
and, specifically, Expert Advisors, which we will develop in detail in the next few chapters. Let's talk
about a trading account.
Having a valid account and an active connection to it are a necessary condition for the functioning of
most MQL programs. Until now, we have not focused on this, but getting quotes, ticks, and, in general,
the ability to open a workable chart implies a successful connection to a trading account.
In the context of Expert Advisors, an account additionally reflects the financial condition of the client,
accumulates the trading history and determines the specific modes allowed for trading.
The MQL5 API allows you to get the properties of an account, starting with its number and ending with
the current profit. All of them are read-only in the terminal and are installed by the broker on the
server.
The terminal can only be connected to one account at a time. All MQL programs work with this
account. As we have already noted in the section Features of starting and stopping programs of various
types, switching an account initiates a reload of the indicators and Expert Advisors attached to the
charts. However, in the OnDeinit handler, the program can find the reason for deinitialization, which,
when switching the account, will be equal to REASON_ACCOUNT.

---

## Page 1156

Part 6. Trading automation
1 1 56
6.3 Trading account information
6.3.1  Overview of functions for getting account properties
The full set of account properties is logically divided into three groups depending on their type. String
properties are summarized in the ENUM_ACCOUNT_INFO_STRING enumeration and are queried by the
AccountInfoString function. Real-type properties are combined in the ENUM_ACCOUNT_INFO_DOUBLE
enumeration, and the function that works for them is AccountInfoDouble. The
ENUM_ACCOUNT_INFO_INTEGER enumeration used in the AccountInfoInteger function contains
identifiers of integer and boolean properties (flags), as well as several applied ENUM_ACCOUNT_INFO
enumerations.
double AccountInfoDouble(ENUM_ACCOUNT_INFO_DOUBLE property)
long AccountInfoInteger(ENUM_ACCOUNT_INFO_INTEGER property)
string AccountInfoString(ENUM_ACCOUNT_INFO_STRING property)
We have created the AccountMonitor class (AccountMonitor.mqh) to simplify the reading of properties.
By overloading get methods, the class provides the automatic call of the required API function
depending on the element of a specific enumeration passed in the parameter.

---

## Page 1157

Part 6. Trading automation
1 1 57
6.3 Trading account information
class AccountMonitor
{
public:
   long get(const ENUM_ACCOUNT_INFO_INTEGER property) const
   {
      return AccountInfoInteger(property);
   }
   
   double get(const ENUM_ACCOUNT_INFO_DOUBLE property) const
   {
      return AccountInfoDouble(property);
   }
   
   string get(const ENUM_ACCOUNT_INFO_STRING property) const
   {
      return AccountInfoString(property);
   }
   
   long get(const int property, const long) const
   {
      return AccountInfoInteger((ENUM_ACCOUNT_INFO_INTEGER)property);
   }
   
   double get(const int property, const double) const
   {
      return AccountInfoDouble((ENUM_ACCOUNT_INFO_DOUBLE)property);
   }
   
   string get(const int property, const string) const
   {
      return AccountInfoString((ENUM_ACCOUNT_INFO_STRING)property);
   }
   ...
In addition, it has several overloads of the stringify method, which form a user-friendly string
representation of property values (in particular, it is useful for applied enumerations, which would
otherwise be displayed as uninformative numbers). The features of each property will be discussed in
the following sections.

---

## Page 1158

Part 6. Trading automation
1 1 58
6.3 Trading account information
   static string boolean(const long v)
   {
      return v ? "true" : "false";
   }
   
   template<typename E>
   static string enumstr(const long v)
   {
      return EnumToString((E)v);
   }
   
   // "decode" properties according to subtype inside integer values
   static string stringify(const long v, const ENUM_ACCOUNT_INFO_INTEGER property)
   {
      switch(property)
      {
         case ACCOUNT_TRADE_ALLOWED:
         case ACCOUNT_TRADE_EXPERT:
         case ACCOUNT_FIFO_CLOSE:
            return boolean(v);
         case ACCOUNT_TRADE_MODE:
            return enumstr<ENUM_ACCOUNT_TRADE_MODE>(v);
         case ACCOUNT_MARGIN_MODE:
            return enumstr<ENUM_ACCOUNT_MARGIN_MODE>(v);
         case ACCOUNT_MARGIN_SO_MODE:
            return enumstr<ENUM_ACCOUNT_STOPOUT_MODE>(v);
      }
      
      return (string)v;
   }
   
   string stringify(const ENUM_ACCOUNT_INFO_INTEGER property) const
   {
      return stringify(AccountInfoInteger(property), property);
   }
   
   string stringify(const ENUM_ACCOUNT_INFO_DOUBLE property, const string format = NULL) const
   {
      if(format == NULL) return DoubleToString(AccountInfoDouble(property),
         (int)get(ACCOUNT_CURRENCY_DIGITS));
      return StringFormat(format, AccountInfoDouble(property));
   }
   
   string stringify(const ENUM_ACCOUNT_INFO_STRING property) const
   {
      return AccountInfoString(property);
   }
   ...
Finally, there is a template method list2log that allows getting comprehensive information about the
account.

---

## Page 1159

Part 6. Trading automation
1 1 59
6.3 Trading account information
   // list of names and values of all properties of enum type E
   template<typename E>
   void list2log()
   {
      E e = (E)0; // suppress warning 'possible use of uninitialized variable'
      int array[];
      const int n = EnumToArray(e, array, 0, USHORT_MAX);
      Print(typename(E), " Count=", n);
      for(int i = 0; i < n; ++i)
      {
         e = (E)array[i];
         PrintFormat("% 3d %s=%s", i, EnumToString(e), stringify(e));
      }
   }
};
We'll test the new class in action in the next section.
6.3.2 Identifying the account, client, server, and broker
Perhaps the most important properties of an account are its number and identification data: the name
of the server and the broker's company, as well as the name of the client. All of these properties,
except for the number, are string properties.
Identifier
Description
ACCOUNT_LOGIN
Account number (long)
ACCOUNT_NAME
Client name
ACCOUNT_SERVER
Trade server name
ACCOUNT_COMPANY
Name of the company servicing the account 
Let's use the AccountMonitor class from the previous section to log these and many other properties
that will be discussed in a moment. Let's create the corresponding object and call its properties in the
script AccountInfo.mq5.
#include <MQL5Book/AccountMonitor.mqh>
   
void OnStart()
{
   AccountMonitor m;
   m.list2log<ENUM_ACCOUNT_INFO_INTEGER>();
   m.list2log<ENUM_ACCOUNT_INFO_DOUBLE>();
   m.list2log<ENUM_ACCOUNT_INFO_STRING>();
}
Here is an example of a possible result of the script.

---

## Page 1160

Part 6. Trading automation
1 1 60
6.3 Trading account information
ENUM_ACCOUNT_INFO_INTEGER Count=10
  0 ACCOUNT_LOGIN=30000003
  1 ACCOUNT_TRADE_MODE=ACCOUNT_TRADE_MODE_DEMO
  2 ACCOUNT_TRADE_ALLOWED=true
  3 ACCOUNT_TRADE_EXPERT=true
  4 ACCOUNT_LEVERAGE=100
  5 ACCOUNT_MARGIN_SO_MODE=ACCOUNT_STOPOUT_MODE_PERCENT
  6 ACCOUNT_LIMIT_ORDERS=200
  7 ACCOUNT_MARGIN_MODE=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
  8 ACCOUNT_CURRENCY_DIGITS=2
  9 ACCOUNT_FIFO_CLOSE=false
ENUM_ACCOUNT_INFO_DOUBLE Count=14
  0 ACCOUNT_BALANCE=10000.00
  1 ACCOUNT_CREDIT=0.00
  2 ACCOUNT_PROFIT=-78.76
  3 ACCOUNT_EQUITY=9921.24
  4 ACCOUNT_MARGIN=1000.00
  5 ACCOUNT_MARGIN_FREE=8921.24
  6 ACCOUNT_MARGIN_LEVEL=992.12
  7 ACCOUNT_MARGIN_SO_CALL=50.00
  8 ACCOUNT_MARGIN_SO_SO=30.00
  9 ACCOUNT_MARGIN_INITIAL=0.00
 10 ACCOUNT_MARGIN_MAINTENANCE=0.00
 11 ACCOUNT_ASSETS=0.00
 12 ACCOUNT_LIABILITIES=0.00
 13 ACCOUNT_COMMISSION_BLOCKED=0.00
ENUM_ACCOUNT_INFO_STRING Count=4
  0 ACCOUNT_NAME=Vincent Silver
  1 ACCOUNT_COMPANY=MetaQuotes Software Corp.
  2 ACCOUNT_SERVER=MetaQuotes-Demo
  3 ACCOUNT_CURRENCY=USD
Pay attention to the properties of this section (ACCOUNT_LOGIN, ACCOUNT_NAME,
ACCOUNT_COMPANY, ACCOUNT_SERVER). In this case, the script was executed on the account of the
demo server "MetaQuotes-Demo". Obviously, this should be a demo account, and this is indicated not
only by the name of the server but also by another property, ACCOUNT_TRADE_MODE, which will be
discussed in the next section.
Account identifiers are usually used to link MQL programs to a specific trading environment. An
example of such an algorithm was presented in the Services section.
6.3.3 Account type: real, demo or contest
MetaTrader 5 supports several types of accounts that can be opened for a client. The
ACCOUNT_TRADE_MODE property, which is part of ENUM_ACCOUNT_INFO_INTEGER, allows you to
find out the current account type. Possible values for this property are described in the
ENUM_ACCOUNT_TRADE_MODE enumeration.

---

## Page 1161

Part 6. Trading automation
1 1 61 
6.3 Trading account information
Identifier
Description
ACCOUNT_TRADE_MODE_DEMO
Demo trading account
ACCOUNT_TRADE_MODE_CONTEST
Contest trading account
ACCOUNT_TRADE_MODE_REAL
Real trading account
This property is convenient for building demo (free) versions of MQL programs. A full-featured, paid
version may require linking to an account number, and the account must be real.
As we saw in the example of running the script AccountInfo.mq5 in the previous section, the account on
the "MetaQuotes-Demo" server is of the ACCOUNT_TRADE_MODE_DEMO type.
6.3.4 Account currency
Balance, profit, margin, commissions, and other financial indicators are always converted to the
account currency in the end, even if the specifications for some trades require settlement in other
currencies, for example, in the margin currency of a Forex pair.
The MQL5 API provides two properties that describe the account currency: its name and the accuracy
of the representation, that is, the size of the minimum unit of measurement (such as cents).
Identifier
Description
ACCOUNT_CURRENCY
Deposit currency (string)
ACCOUNT_CURRENCY_DIGITS
Number of decimal places for account currency required for
the accurate display of trading results (integer)
For example, for the demo account used to test the AccountInfo script in the section on Account
identification, the ACCOUNT_CURRENCY property was "USD", and the accuracy of
ACCOUNT_CURRENCY_DIGITS was 2 decimal places. We have used the ACCOUNT_CURRENCY_DIGITS
property in the AccountMonitor class in the stringify method for values of type double (in the
characteristics of the account, they are all associated with money).
6.3.5 Account type: netting or hedging
MetaTrader 5 supports several types of accounts, in particular, netting and hedging. For netting, it is
allowed to have only one position for each symbol. For hedging, you can open several positions for a
symbol, including multidirectional ones. Orders, trades, and positions will be discussed in detail in the
following chapters.
An MQL program determines the account type by querying the ACCOUNT_MARGIN_MODE property
using the AccountInfoInteger function. As you can understand from the name of the property, it
describes not only the account type but also the margin calculation mode. Its possible values are
specified in the ENUM_ACCOUNT_MARGIN_MODE enumeration.

---

## Page 1162

Part 6. Trading automation
1 1 62
6.3 Trading account information
Identifier
Description
ACCO U N T_M AR G IN _M O D E _R E TAIL _N E TTIN G 
OTC market, considering positions in the netting mode.
Margin calculation is based on the
SYMBOL_TRADE_CALC_MODE property.
ACCO U N T_M AR G IN _M O D E _E XCH AN G E 
Exchange market, considering positions in the netting
mode. The margin is calculated based on the rules of
the exchange with the possibility of discounts specified
by the broker in the instrument settings.
ACCO U N T_M AR G IN _M O D E _R E TAIL _H E D G IN G 
OTC market with independent consideration of
positions in the hedging mode. Margin calculation is
based on the SYMBOL_TRADE_CALC_MODE symbol
property while considering the size of the hedged
margin SYMBOL_MARGIN_HEDGED.
For example, running the AccountInfo script in the section Account identification showed that the
account is of type ACCOUNT_MARGIN_MODE_RETAIL_HEDGING.
6.3.6 Restrictions and permissions for account operations
Among the properties of the account, there are restrictions on trading operations, including completely
disabled trading. All of these properties belong to the ENUM_ACCOUNT_INFO_INTEGER enumeration
and are boolean flags, except ACCOUNT_LIMIT_ORDERS.
Identifier
Description
ACCOUNT_TRADE_ALLOWED
Permission to trade on a current account
ACCOUNT_TRADE_EXPERT
Permission for algorithmic trading using Expert Advisors and
scripts
ACCOUNT_LIMIT_ORDERS
Maximum allowed number of valid pending orders
ACCOUNT_FIFO_CLOSE
Requirement to close positions only according to the FIFO
rule
Since our book is about MQL5 programming, which includes algorithmic trading, it should be noted that
the disabled ACCOUNT_TRADE_EXPERT permission is just as critical as the general prohibition to trade
when ACCOUNT_TRADE_ALLOWED is equal to false. The broker has the ability to prohibit trading using
Expert Advisors and scripts while allowing manual trading.
The ACCOUNT_TRADE_ALLOWED property is usually equal to false if the connection to the account was
made using the investment password. 
If the value of the ACCOUNT_FIFO_CLOSE property is true, positions for each symbol can only be
closed in the same order in which they were opened, that is, first you close the oldest order, then the
newer one, and so on until the last one. If you try to close positions in a different order, you will receive
an error. For accounts without position hedging, that is, if the ACCOUNT_MARGIN_MODE property is
not equal to ACCOUNT_MARGIN_MODE_RETAIL_HEDGING, the ACCOUNT_FIFO_CLOSE property is
always false.

---

## Page 1163

Part 6. Trading automation
1 1 63
6.3 Trading account information
In the Permissions and Schedules of trading and quoting sessions sections, we have already started
developing a class for detecting trade operations available to the MQL program. Now we can
supplement it with account permission checks and bring it to the final version (Permissions.mqh).
Restriction levels are provided in the TRADE_RESTRICTIONS enumeration, which, after adding two new
elements related to account properties, takes the following form.
class Permissions
{
   enum TRADE_RESTRICTIONS
   {
      NO_RESTRICTIONS = 0,
      TERMINAL_RESTRICTION = 1, // user's restriction for all programs
      PROGRAM_RESTRICTION = 2,  // user's restriction for a specific program
      SYMBOL_RESTRICTION = 4,   // the symbol is not traded according to the specification
      SESSION_RESTRICTION = 8,  // the market is closed according to the session schedule
      ACCOUNT_RESTRICTION = 16, // investor password or broker restriction
      EXPERTS_RESTRICTION = 32, // broker restricted algorithmic trading
   };
   ...
During the check, the MQL program may detect several restrictions for various reasons, and therefore
the elements are encoded by separate bits. The final result can represent their superposition.
The last two restrictions just correspond to the new properties and are set in the
getTradeRestrictionsOnAccount method. The general bitmask of detected restrictions (if any) is formed
in the lastRestrictionBitMask variable.
private:
   static uint lastRestrictionBitMask;
   static bool pass(const uint bitflag) 
   {
      lastRestrictionBitMask |= bitflag;
      return lastRestrictionBitMask == 0;
   }
   
public:
   static uint getTradeRestrictionsOnAccount()
   {
      return (AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) ? 0 : ACCOUNT_RESTRICTION)
         | (AccountInfoInteger(ACCOUNT_TRADE_EXPERT) ? 0 : EXPERTS_RESTRICTION);
   }
   
   static bool isTradeOnAccountEnabled()
   {
      lastRestrictionBitMask = 0;
      return pass(getTradeRestrictionsOnAccount());
   }
   ... 
If the calling code is not interested in the reason for restriction but only needs to determine the
possibility of performing trading operations, it is more convenient to use the isTradeOnAccountEnabled
method which returns a boolean sign (true/false).

---

## Page 1164

Part 6. Trading automation
1 1 64
6.3 Trading account information
Checks of symbol and terminal properties have been reorganized according to a similar principle. For
example, the getTradeRestrictionsOnSymbol method contains the source code already familiar from the
previous version of the class (checking the symbol's trading sessions and trading modes) but returns a
flags mask. If at least one bit is set, it describes the source of the restriction.
   static uint getTradeRestrictionsOnSymbol(const string symbol, datetime now = 0,
      const ENUM_SYMBOL_TRADE_MODE mode = SYMBOL_TRADE_MODE_FULL)
   {
      if(now == 0) now = TimeTradeServer();
      bool found = false;
      // checking the symbol trading sessions and setting 'found' to 'true',
      // if the 'now' time is inside one of the sessions
      ...
      
      // in addition to sessions, check the trading mode
      const ENUM_SYMBOL_TRADE_MODE m = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
      return (found ? 0 : SESSION_RESTRICTION)
         | (((m & mode) != 0) || (m == SYMBOL_TRADE_MODE_FULL) ? 0 : SYMBOL_RESTRICTION);
   }
   
   static bool isTradeOnSymbolEnabled(const string symbol, const datetime now = 0,
      const ENUM_SYMBOL_TRADE_MODE mode = SYMBOL_TRADE_MODE_FULL)
   {
      lastRestrictionBitMask = 0;
      return pass(getTradeRestrictionsOnSymbol(symbol, now, mode));
   }
   ...
Finally, a general check of all potential "instances", including (in addition to the previous levels) the
settings of the terminal and the program, is performed in the getTradeRestrictions and isTradeEnabled
methods.
   static uint getTradeRestrictions(const string symbol = NULL, const datetime now = 0,
      const ENUM_SYMBOL_TRADE_MODE mode = SYMBOL_TRADE_MODE_FULL)
   {
      return (TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) ? 0 : TERMINAL_RESTRICTION)
          | (MQLInfoInteger(MQL_TRADE_ALLOWED) ? 0 : PROGRAM_RESTRICTION)
          | getTradeRestrictionsOnSymbol(symbol == NULL ? _Symbol : symbol, now, mode)
          | getTradeRestrictionsOnAccount();
   }
   
   static bool isTradeEnabled(const string symbol = NULL, const datetime now = 0,
      const ENUM_SYMBOL_TRADE_MODE mode = SYMBOL_TRADE_MODE_FULL)
   {
      lastRestrictionBitMask = 0;
      return pass(getTradeRestrictions(symbol, now, mode));
   }
A comprehensive check of trade permissions with a new class is demonstrated by the script
AccountPermissions.mq5.

---

## Page 1165

Part 6. Trading automation
1 1 65
6.3 Trading account information
#include <MQL5Book/Permissions.mqh>
   
void OnStart()
{
   PrintFormat("Run on %s", _Symbol);
   if(!Permissions::isTradeEnabled()) // checking for current character, default
   {
      Print("Trade is disabled for the following reasons:");
      Print(Permissions::explainLastRestrictionBitMask());
   }
   else
   {
      Print("Trade is enabled");
   }
}
If restrictions are found, their bit mask can be displayed in a clear string representation using the
explainLastRestrictionBitMask method.
Here are some script results. In the first two cases, trading was disabled in the global settings of the
terminal (properties TERMINAL_TRADE_ALLOWED and MQL_TRADE_ALLOWED were equal to false,
which corresponds to the TERMINAL_RESTRICTION and PROGRAM_RESTRICTION bits).
When run on USDRUB during the hours when the market is closed, we will additionally receive
SESSION_RESTRICTION:
Trade is disabled for USDRUB following reasons:
TERMINAL_RESTRICTION PROGRAM_RESTRICTION SESSION_RESTRICTION 
For the symbol SP500m, for which trading is totally disabled, the SYMBOL_RESTRICTION flag appears.
Trade is disabled for SP500m following reasons:
TERMINAL_RESTRICTION PROGRAM_RESTRICTION SYMBOL_RESTRICTION SESSION_RESTRICTION 
Finally, having allowed trading in the terminal but having logged into the account under the investor's
password, we will see ACCOUNT_RESTRICTION on any symbol.
Run on XAUUSD
Trade is disabled for following reasons:
ACCOUNT_RESTRICTION 
Early check of permissions in the MQL program helps avoid serial unsuccessful attempts to send trading
orders.
6.3.7 Account margin settings
For trading robots, it is important to control the amount of margin blocked and the amount available to
secure new trades. In particular, if there are not enough free funds, the program will not be able to
execute a trade. When maintaining open unprofitable positions, first a Margin Call is received, and if it is
not fulfilled, the positions are forcedly closed by the broker (Stop Out). All associated account
properties are included in the ENUM_ACCOUNT_INFO_DOUBLE enumeration.

---

## Page 1166

Part 6. Trading automation
1 1 66
6.3 Trading account information
Identifier
Description
ACCOUNT_MARGIN
Current reserved margin on the account in the deposit
currency
ACCOUNT_MARGIN_FREE
Current free margin on the account in the deposit currency,
available for opening a position
ACCOUNT_MARGIN_LEVEL
Margin level on the account in percent (equity/margin*1 00)
ACCOUNT_MARGIN_SO_CALL
The minimum margin level at which account replenishment
will be required (Margin Call)
ACCOUNT_MARGIN_SO_SO
The minimum margin level at which the most unprofitable
position will be forced to close (Stop Out)
ACCOUNT_MARGIN_INITIAL
Funds reserved on the account to provide margin for all
pending orders
ACCOUNT_MARGIN_MAINTENANCE
Funds reserved on the account to provide the minimum
required margin for all open positions
ACCOUNT_MARGIN_SO_CALL and ACCOUNT_MARGIN_SO_SO are expressed as a percentage or
deposit currency depending on the set ACCOUNT_MARGIN_SO_MODE (see further). This property, with
the possibility to measure margin thresholds for the Margin Call or Stop Out, is included in the
ENUM_ACCOUNT_INFO_INTEGER enumeration. In addition, the total leverage (used to calculate the
margin for certain types of instruments) is also indicated there.
Identifier
Description
ACCOUNT_LEVERAGE
The leverage amount
ACCOUNT_MARGIN_SO_MODE
The mode for setting the minimum allowable margin level
from the ENUM_ACCOUNT_STOPOUT_MODE enumeration 
And here are the elements of the ENUM_ACCOUNT_STOPOUT_MODE enumeration.
Identifier
Description
ACCO U N T_S TO P O U T_M O D E _P E R CE N T
The level is set as a percentage
ACCO U N T_S TO P O U T_M O D E _M O N E Y
The level is set in the account currency
For example, for the ACCOUNT_STOPOUT_MODE_PERCENT option, the specified percentage (Margin
Call or Stop Out) should be checked against the ratio of equity to the value of the ACCOUNT_MARGIN
property:
AccountInfoDouble(ACCOUNT_EQUITY) / AccountInfoDouble(ACCOUNT_MARGIN) * 100
   > AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL)
In the next section, you will find more details about the ACCOUNT_EQUITY property and other financial
indicators of the account.

---

## Page 1167

Part 6. Trading automation
1 1 67
6.3 Trading account information
However, the current margin level in percent is already provided in the ACCOUNT_MARGIN_LEVEL
property. This is easy to check using the AccountInfo.mq5 script which logs all account properties,
including those listed above.
We have already run this script in the section Account identification. At that moment, one position was
opened (1  lot USDRUB, equal to 1 00,000 USD), and the financials were as follows:
  0 ACCOUNT_BALANCE=10000.00
  1 ACCOUNT_CREDIT=0.00
  2 ACCOUNT_PROFIT=-78.76
  3 ACCOUNT_EQUITY=9921.24
  4 ACCOUNT_MARGIN=1000.00
  5 ACCOUNT_MARGIN_FREE=8921.24
  6 ACCOUNT_MARGIN_LEVEL=992.12
  7 ACCOUNT_MARGIN_SO_CALL=50.00
  8 ACCOUNT_MARGIN_SO_SO=30.00
With a margin of 1 000.00 USD, it is easy to check that the leverage of the account,
ACCOUNT_LEVERAGE, is indeed 1 00 (according to the formula for calculating margin for Forex and
margin ratio which is equal to 1 .0). The margin amount does not need to be converted at the current
rate into the account currency, since it is the same as the base currency of the instrument.
To get 992.1 2 in ACCOUNT_MARGIN_LEVEL, just divide 9921 .24 by 1 000.00 and multiply by 1 00%.
Then another 1  lot position was opened, and the quotes went in an unfavorable direction, as a result of
which the situation changed:
  0 ACCOUNT_BALANCE=10000.00
  1 ACCOUNT_CREDIT=0.00
  2 ACCOUNT_PROFIT=-1486.07
  3 ACCOUNT_EQUITY=8513.93
  4 ACCOUNT_MARGIN=2000.00
  5 ACCOUNT_MARGIN_FREE=6513.93
  6 ACCOUNT_MARGIN_LEVEL=425.70
We can see a loss in the ACCOUNT_PROFIT column and a corresponding decrease in equity
ACCOUNT_EQUITY. The margin ACCOUNT_MARGIN increased proportionally from 1 000 to 2000, free
margin and margin level decreased (but still far from the 50% and 30% limits). Again, the level 425.70
is obtained as the result of calculating the expression 851 3.93 / 2000.00 * 1 00.
It is more practical to use this formula to calculate the future margin level before opening a new
position. In this case, it is necessary to increase the amount of the existing margin by the additional
margin of X. In addition, if a market entry deal involves an instant commission deduction C, then,
strictly speaking, it should also be taken into account (although usually it has a size significantly less
than the margin and it can be neglected, plus the API does not provide a way to find out the
commission in advance, before performing a trade: it can only be estimated by the commissions of
already completed trades in trading history).
(AccountInfoDouble(ACCOUNT_EQUITY) - C) / (AccountInfoDouble(ACCOUNT_MARGIN) + X) * 100
   > AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL)
Later we will learn how to obtain the X value using the OrderCalcMargin function, but in addition to it,
adjustments may be required according to the rules announced in the Margin Requirements section, in
particular, taking into account the possible position hedging, discounts, and margin adjustments.

---

## Page 1168

Part 6. Trading automation
1 1 68
6.3 Trading account information
For the option of setting the margin limit in money (ACCOUNT_STOPOUT_MODE_MONEY), the check
for sufficient funds must be different.
AccountInfoDouble(ACCOUNT_EQUITY) > AccountInfoDouble(ACCOUNT_MARGIN_SO_CALL)
Here the commission is omitted. Please note that the margin X for a new position being prepared for
opening 'now' does not affect the assessment of the 'future' margin in any way.
However, in any case, it is desirable not to load the deposit so much that the inequalities are barely
fulfilled. The values of ACCOUNT_MARGIN_SO_CALL and ACCOUNT_MARGIN_SO_SO are quite close,
and although the margin at the ACCOUNT_MARGIN_SO_CALL level is just a warning to the trader, it is
easy to get a forced closing. That is why the formulas use the ACCOUNT_MARGIN_SO_CALL property.
6.3.8 Current financial performance of the account
The MQL5 API allows control over several account properties via its main financial indicators. They are
all included in the ENUM_ACCOUNT_INFO_DOUBLE enumeration.
Identifier
Description
ACCOUNT_BALANCE
Account balance in the deposit currency
ACCOUNT_PROFIT
The amount of current profit on the account in the deposit
currency
ACCOUNT_EQUITY
Account equity in the deposit currency
ACCOUNT_CREDIT
The amount of the credit provided by the broker in the
deposit currency
ACCOUNT_ASSETS
Current amount of assets on the account
ACCOUNT_LIABILITIES
Current amount of liabilities on the account
ACCOUNT_COMMISSION_BLOCKED
The current amount of blocked commissions on the account
In the previous sections, we saw examples of the values of these properties when running the
AccountInfo.mq5 script under different conditions. Try to compare these properties for your different
accounts.
In the trading process, we will be primarily interested in the first three properties: balance, profit (or
loss if the value is negative), and equity, which together cover the account balance, credit, profit, and
overhead costs (swap and commission).
Commissions can be considered in different ways, depending on the broker's settings. If commissions
are immediately deducted from the account balance at the time of trades and are reflected in the deal
properties, the account property ACCOUNT_COMMISSION_BLOCKED will be equal to 0. However, if the
commission calculation is postponed until the end of the period (for example, a day or a month), the
amount blocked for the commission will appear in this property. Then, when the final commission
amount is determined and deducted from the balance at the end of the period, the property will be
reset.
The properties ACCOUNT_ASSETS and ACCOUNT_LIABILITIES are filled, as a rule, only for exchange
trading. They reflect the current value of long and short positions in securities.

---

## Page 1169

Part 6. Trading automation
1 1 69
6.4 Creating Expert Advisors
6.4 Creating Expert Advisors
In this chapter, we begin to study the MQL5 trading API used to implement Expert Advisors. This type
of program is perhaps the most complex and demanding in terms of error-free coding and the number
and variety of technologies involved. In particular, we will need to utilize many of the skills acquired
from the previous chapters, ranging from OOP to the applied aspects of working with graphical objects,
indicators, symbols, and software environment settings.
Depending on the chosen trading strategy, the Expert Advisor developer may need to pay special
attention to the following:
·Decision-making and order-sending speed (for HFT, High-Frequency Trading)
·Selecting the optimal portfolio of instruments based on their correlations and volatility (for cluster
trading)
·Dynamically calculating lots and distance between orders (for martingale and grid strategies)
·Analysis of news or external data sources (this will be discussed in the 7th part of the book)
All such features should be optimally applied by the developer to the described trading mechanisms
provided by the MQL5 API.
Next, we will consider in detail built-in functions for managing trading activity, the Expert Advisor event
model, and specific data structures, and recall the basic principles of interaction between the terminal
and the server, as well as the basic concepts for algorithmic trading in MetaTrader 5: order, deal, and
position.
At the same time, due to the versatility of the material, many important nuances of Expert Advisor
development, such as testing and optimization, are highlighted in the next chapter.
We have previously considered the Design of MQL programs of various types, including Expert Advisors,
as well as started Features of starting and stopping programs. Despite the fact that an Expert Advisor is
launched on a specific chart, for which a working symbol is defined, there are no obstacles to centrally
manage trading of an arbitrary set of financial instruments. Such Expert Advisors are traditionally
referred to as multicurrency, although in fact, their portfolio may include CFDs, stocks, commodities,
and tickers of other markets.
In Expert Advisors, as well as in indicators, there are Key events OnInit and OnDeinit. They are not
mandatory, but, as a rule, they are present in the code for the preparation and regular completion of
the program: we used them and will continue using them in the examples. In a separate section, we
provided an Overview of all event handling functions: we have already studied some of them in detail by
now (for example, OnCalculate indicator events and the OnTimer timer).  Expert Advisor-specific events
(OnTick, ontrade, OnTradeTransaction) will be described in this chapter.
Expert Advisors can use the widest range of source data as trading signals: quotes, tics, depth of
market, trading account history, or indicator readings. In the latter case, the principles of creating
indicator instances and reading values from their buffers are no different from those discussed in the
chapter Using ready-made indicators from MQL programs. In the Expert Advisor examples in the
following sections, we will demonstrate most of these tricks.
It should be noted that trading functions can be used not only in Expert Advisors but also in scripts. We
will see examples for both options.