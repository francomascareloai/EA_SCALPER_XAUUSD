# MQL5 Book - Part 5 (Pages 801-1000)

## Page 801

Part 5. Creating application programs
801 
5.5 Using ready-made indicators from MQL programs
UseUnityPercentPro multi-symbol indicator with major Forex pairs
Shown here is a basket of 8 major Forex currencies (default setting) averaged over 1 1  bars and
calculated based on the typical price. Two thick lines correspond to the relative value of the currencies
of the current chart: EUR is marked in blue and USD is green.
5.5.8 Advanced way to create indicators: IndicatorCreate
Creating an indicator using the iCustom function or one of those functions that make up a set of built-in
indicators requires knowledge of the list of parameters at the coding stage. However, in practice, it
often becomes necessary to write programs that are flexible enough to replace one indicator with
another.
For example, when optimizing an Expert Advisor in the tester, it makes sense to select not only the
period of the moving average, but also the algorithm for its calculation. Of course, if we build the
algorithm on a single indicator iMA, you can provide the possibility to specify ENUM_MA_METHOD in its
method settings. But someone would probably like to expand the choice by switching between double
exponential, triple exponential and fractal moving average. At first glance, this could be done using
switch with a call of DEMA, iTEMA, and iFrAMA, respectively. However, what about including custom
indicators in this list?
Although the name of the indicator can be easily replaced in the iCustom call, the list of parameters
may differ significantly. In the general case, an Expert Advisor may need to generate signals based on a
combination of any indicators that are not known in advance, and not just moving averages.
For such cases, MQL5 has a universal method for creating an arbitrary technical indicator using the
IndicatorCreate function.

---

## Page 802

Part 5. Creating application programs
802
5.5 Using ready-made indicators from MQL programs
int IndicatorCreate(const string symbol, ENUM_TIMEFRAMES timeframe, ENUM_INDICATOR indicator,
int count = 0, const MqlParam &parameters[] = NULL)
The function creates an indicator instance for the specified symbol and timeframe. The indicator type
is set using the indicator parameter. Its type is the ENUM_INDICATOR enumeration (see further along)
containing identifiers for all built-in indicators, as well as an option for iCustom. The number of indicator
parameters and their descriptions are passed, respectively, in the count argument and in the MqlParam
array of structures (see below).
Each element of this array describes the corresponding input parameter of the indicator being created,
so the content and order of the elements must correspond to the prototype of the built-in indicator
function or, in the case of a custom indicator, to the descriptions of the input variables in its source
code.
Violation of this rule may result in an error at the program execution stage (see example below) and in
the inability to create a handle. In the worst case, the passed parameters will be interpreted
incorrectly and the indicator will not behave as expected, but due to the lack of errors, this is not easy
to notice. The exception is passing an empty array or not passing it at all (because the arguments
count and parameters are optional): in this case, the indicator will be created with default settings.
Also, for custom indicators, you can omit an arbitrary number of parameters from the end of the list.
The MqlParam structure is specially designed to pass input parameters when creating an indicator using
IndicatorCreate or to obtain information about the parameters of a third-party indicator (performed on
the chart) using IndicatorParameters.
struct MqlParam 
{ 
   ENUM_DATATYPE type;          // input parameter type
   long          integer_value; // field for storing an integer value
   double        double_value;  // field for storing double or float values
   string        string_value;  // field for storing a value of string type
};
The actual value of the parameter must be set in one of the fields integer_ value, double_ value,
string_ value, according to the value of the first type field. In turn, the type field is described using the
ENUM_DATATYPE enumeration containing identifiers for all built-in MQL5 types.

---

## Page 803

Part 5. Creating application programs
803
5.5 Using ready-made indicators from MQL programs
Identifier
Data type
TYPE_BOOL
bool
TYPE_CHAR
char
TYPE_UCHAR
uchar
TYPE_SHORT
short
TYPE_USHORT
ushort
TYPE_COLOR
color
TYPE_INT
int
TYPE_UINT
uint
TYPE_DATETIME
datetime
TYPE_LONG
long
TYPE_ULONG
ulong
TYPE_FLOAT
float
TYPE_DOUBLE
double
TYPE_STRING
string
If any indicator parameter has an enumeration type, you should use the TYPE_INT value in the type
field to describe it.
The ENUM_INDICATOR enumeration used in the third parameter IndicatorCreate to indicate the
indicator type contains the following constants.
Identifier
Indicator
IND_AC
Accelerator Oscillator
IND_AD
Accumulation/Distribution
IND_ADX
Average Directional Index
IND_ADXW
ADX by Welles Wilder
IND_ALLIGATOR
Alligator
IND_AMA
Adaptive Moving Average
IND_AO
Awesome Oscillator
IND_ATR
Average True Range
IND_BANDS
Bollinger Bands®
IND_BEARS
Bears Power

---

## Page 804

Part 5. Creating application programs
804
5.5 Using ready-made indicators from MQL programs
Identifier
Indicator
IND_BULLS
Bulls Power
IND_BWMFI
Market Facilitation Index
IND_CCI
Commodity Channel Index
IND_CHAIKIN
Chaikin Oscillator
IND_CUSTOM
Custom indicator
IND_DEMA
Double Exponential Moving Average
IND_DEMARKER
DeMarker
IND_ENVELOPES
Envelopes
IND_FORCE
Force Index
IND_FRACTALS
Fractals
IND_FRAMA
Fractal Adaptive Moving Average
IND_GATOR
Gator Oscillator
IND_ICHIMOKU
Ichimoku Kinko Hyo
IND_MA
Moving Average
IND_MACD
MACD
IND_MFI
Money Flow Index
IND_MOMENTUM
Momentum
IND_OBV
On Balance Volume
IND_OSMA
OsMA
IND_RSI
Relative Strength Index
IND_RVI
Relative Vigor Index
IND_SAR
Parabolic SAR
IND_STDDEV
Standard Deviation
IND_STOCHASTIC
Stochastic Oscillator
IND_TEMA
Triple Exponential Moving Average
IND_TRIX
Triple Exponential Moving Averages Oscillator
IND_VIDYA
Variable Index Dynamic Average
IND_VOLUMES
Volumes
IND_WPR
Williams Percent Range

---

## Page 805

Part 5. Creating application programs
805
5.5 Using ready-made indicators from MQL programs
It is important to note that if the IND_CUSTOM value is passed as the indicator type, then the first
element of the parameters array must have the type field with the value TYPE_STRING, and the
string_ value field must contain the name (path) of the custom indicator.
If successful, the IndicatorCreate function returns a handle of the created indicator, and in case of
failure it returns INVALID_HANDLE. The error code will be provided in _ LastError.
Recall that in order to test MQL programs that create custom indicators whose names are not known
at the compilation stage (which is also the case when using IndicatorCreate), you must explicitly bind
them using the directive:
#property tester_indicator "indicator_name.ex5"
This allows the tester to send the required auxiliary indicators to the testing agents but limits the
process to only indicators known in advance.
Let's look at a few examples. Let's start with a simple application IndicatorCreate as an alternative to
already known functions, and then, to demonstrate the flexibility of the new approach, we will create a
universal wrapper indicator for visualizing arbitrary built-in or custom indicators.
The first example of UseEnvelopesParams1 .mq5 creates an embedded copy of the Envelopes indicator.
To do this, we describe two buffers, two plots, arrays for them, and input parameters that repeat the
iEnvelopes parameters.
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2
   
// drawing settings
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrBlue
#property indicator_width1  1
#property indicator_label1  "Upper"
#property indicator_style1  STYLE_DOT
   
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrRed
#property indicator_width2  1
#property indicator_label2  "Lower"
#property indicator_style2  STYLE_DOT
   
input int WorkPeriod = 14;
input int Shift = 0;
input ENUM_MA_METHOD Method = MODE_EMA;
input ENUM_APPLIED_PRICE Price = PRICE_TYPICAL;
input double Deviation = 0.1; // Deviation, %
   
double UpBuffer[];
double DownBuffer[];
   
int Handle; // handle of the subordinate indicator
The handler OnInit could look like this if you use the function iEnvelopes.

---

## Page 806

Part 5. Creating application programs
806
5.5 Using ready-made indicators from MQL programs
int OnInit()
{
   SetIndexBuffer(0, UpBuffer);
   SetIndexBuffer(1, DownBuffer);
   
   Handle = iEnvelopes(WorkPeriod, Shift, Method, Price, Deviation);
   return Handle == INVALID_HANDLE ? INIT_FAILED : INIT_SUCCEEDED;
}
The buffer bindings will remain the same, but to create a handle, we will now go the other way. Let's
describe the MqlParam array, fill it in and call the IndicatorCreate function.
int OnInit()
{
   ...
   MqlParam params[5] = {};
   params[0].type = TYPE_INT;
   params[0].integer_value = WorkPeriod;
   params[1].type = TYPE_INT;
   params[1].integer_value = Shift;
   params[2].type = TYPE_INT;
   params[2].integer_value = Method;
   params[3].type = TYPE_INT;
   params[3].integer_value = Price;
   params[4].type = TYPE_DOUBLE;
   params[4].double_value = Deviation;
   Handle = IndicatorCreate(_Symbol, _Period, IND_ENVELOPES,
      ArraySize(params), params);
   return Handle == INVALID_HANDLE ? INIT_FAILED : INIT_SUCCEEDED;
}
Having received the handle, we use it in OnCalculate to fill two of its buffers.
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &data[])
{
   if(BarsCalculated(Handle) != rates_total)
   {
      return prev_calculated;
   }
   
   const int n = CopyBuffer(Handle, 0, 0, rates_total - prev_calculated + 1, UpBuffer);
   const int m = CopyBuffer(Handle, 1, 0, rates_total - prev_calculated + 1, DownBuffer);
      
   return n > -1 && m > -1 ? rates_total : 0;
}
Let's check how the created indicator UseEnvelopesParams1  looks on the chart.

---

## Page 807

Part 5. Creating application programs
807
5.5 Using ready-made indicators from MQL programs
UseEnvelopesParams1 indicator
Above was a standard but not very elegant way to populate properties. Since the IndicatorCreate call
may be required in many projects, it makes sense to simplify the procedure for the calling code. For
this purpose, we will develop a class entitled MqlParamBuilder (see file MqlParamBuilder.mqh). Its task
will be to accept parameter values using some methods, determine their type, and add appropriate
elements (correctly filled structures) to the array.
MQL5 does not fully support the concept of the Run-Time Type Information (RTTI). With it, programs
can ask the runtime for descriptive meta-data about their constituent parts, including variables,
structures, classes, functions, etc. The few built-in features of MQL5 that can be classified as RTTI are
operators typename and offsetof. Because typename returns the name of the type as a string, let's
build our type autodetector on strings (see file RTTI.mqh).

---

## Page 808

Part 5. Creating application programs
808
5.5 Using ready-made indicators from MQL programs
template<typename T>
ENUM_DATATYPE rtti(T v = (T)NULL)
{
   static string types[] =
   {
      "null",     //               (0)
      "bool",     // 0 TYPE_BOOL=1 (1)
      "char",     // 1 TYPE_CHAR=2 (2)
      "uchar",    // 2 TYPE_UCHAR=3 (3)
      "short",    // 3 TYPE_SHORT=4 (4)
      "ushort",   // 4 TYPE_USHORT=5 (5)
      "color",    // 5 TYPE_COLOR=6 (6)
      "int",      // 6 TYPE_INT=7 (7)
      "uint",     // 7 TYPE_UINT=8 (8)
      "datetime", // 8 TYPE_DATETIME=9 (9)
      "long",     // 9 TYPE_LONG=10 (A)
      "ulong",    // 10 TYPE_ULONG=11 (B)
      "float",    // 11 TYPE_FLOAT=12 (C)
      "double",   // 12 TYPE_DOUBLE=13 (D)
      "string",   // 13 TYPE_STRING=14 (E)
   };
   const string t = typename(T);
   for(int i = 0; i < ArraySize(types); ++i)
   {
      if(types[i] == t)
      {
         return (ENUM_DATATYPE)i;
      }
   }
   return (ENUM_DATATYPE)0;
}
The template function rtti uses typename to receive a string with the name of the template type
parameter and compares it with the elements of an array containing all built-in types from the
ENUM_DATATYPE enumeration. The order of enumeration of names in the array corresponds to the
value of the enumeration element, so when a matching string is found, it is enough to cast the index to
type (ENUM_DATATYPE) and return it to the calling code. For example, call to rtti(1 .0) or rtti<double>
() will give the value TYPE_DOUBLE.
With this tool, we can return to working on MqlParamBuilder. In the class, we describe the MqlParam
array of structures and the n variable which will contain the index of the last element to be filled.
class MqlParamBuilder
{
protected:
   MqlParam array[];
   int n;
   ...
Let's make the public method for adding the next value to the list of parameters a template one.
Moreover, we implement it as an overload of the operator '<<' , which returns a pointer to the "builder"
object itself. This will allow to write multiple values to the array in one line, for example, like this:
builder << WorkPeriod << PriceType << SmoothingMode.

---

## Page 809

Part 5. Creating application programs
809
5.5 Using ready-made indicators from MQL programs
It is in this method that we increase the size of the array, get the working index n to fill, and
immediately reset this n-th structure.
...
public:
   template<typename T>
   MqlParamBuilder *operator<<(T v)
   {
 // expand the array
      n = ArraySize(array);
      ArrayResize(array, n + 1);
      ZeroMemory(array[n]);
      ...
      return &this;
   }
Where there is an ellipsis, the main working part will follow, that is, filling in the fields of the structure.
It could be assumed that we will directly determine the type of the parameter using a self-made rtti.
But you should pay attention to one nuance. If we write instructions array[n].type = rtti(v), it will not
work correctly for enumerations. Each enumeration is an independent type with its own name, despite
the fact that it is stored in the same way as integers. For enumerations, the function rtti will return 0,
and therefore, you need to explicitly replace it with TYPE_INT.
      ...
      // define value type
      array[n].type = rtti(v);
      if(array[n].type == 0) array[n].type = TYPE_INT; // imply enum
      ...
Now we only need to put the v value to one of the three fields of the structure: integer_ value of type
long (note, long is a long integer, hence the name of the field), double_ value of type double or
string_ value of type string. Meanwhile, the number of built-in types is much larger, so it is assumed that
all integral types (including int, short, char, color, datetime, and enumerations) must fall into the field
integer_ value, float values must fall in field double_ value, and only for the string_ value field has is an
unambiguous interpretation: it is always string.
To accomplish this task, we implement several overloaded assign methods: three with specific types of
float, double, and string, and one template for everything else.

---

## Page 810

Part 5. Creating application programs
81 0
5.5 Using ready-made indicators from MQL programs
class MqlParamBuilder
{
protected:
   ...
   void assign(const float v)
   {
      array[n].double_value = v;
   }
   
   void assign(const double v)
   {
      array[n].double_value = v;
   }
   
   void assign(const string v)
   {
      array[n].string_value = v;
   }
   
   // here we process int, enum, color, datetime, etc. compatible with long
   template<typename T>
   void assign(const T v)
   {
      array[n].integer_value = v;
   }
   ...
This completes the process of filling structures, and the question remains of passing the generated
array to the calling code. This action is assigned to a public method with an overload of the operator
'>>', which has a single argument: a reference to the receiving array MqlParam.
   // export the inner array to the outside
   void operator>>(MqlParam &params[])
   {
      ArraySwap(array, params);
   }
Now that everything is ready, we can work with the source code of the modified indicator
UseEnvelopesParams2.mq5. Changes compared to the first version concern only filling of the MqlParam
array in the OnInit handler. In it, we describe the "builder" object, send all parameters to it via '<<'
and return the finished array via '>>'. All is done in one line.

---

## Page 811

Part 5. Creating application programs
81 1 
5.5 Using ready-made indicators from MQL programs
int OnInit()
{
   ...
   MqlParam params[];
   MqlParamBuilder builder;
   builder << WorkPeriod << Shift << Method << Price << Deviation >> params;
   ArrayPrint(params);
   /*
       [type] [integer_value] [double_value] [string_value]
   [0]      7              14        0.00000 null            <- "INT" period
   [1]      7               0        0.00000 null            <- "INT" shift
   [2]      7               1        0.00000 null            <- "INT" EMA
   [3]      7               6        0.00000 null            <- "INT" TYPICAL
   [4]     13               0        0.10000 null            <- "DOUBLE" deviation
   */
For control, we output the array to the log (the result for the default values is shown above).
If the array is not completely filled, IndicatorCreate call will end with an error. For example, if you pass
only 3 parameters out of 5 required for Envelopes, you will get error 4002 and an invalid handle.
   Handle = PRTF(IndicatorCreate(_Symbol, _Period, IND_ENVELOPES, 3, params));
   // Error example:
   // indicator Envelopes cannot load [4002]   
   // IndicatorCreate(_Symbol,_Period,IND_ENVELOPES,3,params)=
      -1 / WRONG_INTERNAL_PARAMETER(4002)
However, a longer array than in the indicator specification is not considered an error: extra values are
simply not taken into account.
Note that when the value types differ from the expected parameter types, the system performs an
implicit cast, and this does not raise obvious errors, although the generated indicator may not work as
expected. For example, if instead of Deviation we send a string to the indicator, it will be interpreted as
the number 0, as a result of which the "envelope" will collapse: both lines will be aligned on the middle
line, relative to which the indent is made by the size of Deviation (in percentages). Similarly, passing a
real number with a fractional part in a parameter where an integer is expected will cause it to be
rounded.
But we, of course, leave the correct version of the IndicatorCreate call and get a working indicator, just
like in the first version.
   ...
   Handle = PRTF(IndicatorCreate(_Symbol, _Period, IND_ENVELOPES,
      ArraySize(params), params));
   // success:
   // IndicatorCreate(_Symbol,_Period,IND_ENVELOPES,ArraySize(params),params)=10 / ok
   return Handle == INVALID_HANDLE ? INIT_FAILED : INIT_SUCCEEDED;
}
By the look of it, the new indicator is no different from the previous one.

---

## Page 812

Part 5. Creating application programs
81 2
5.5 Using ready-made indicators from MQL programs
5.5.9 Flexible creation of indicators with IndicatorCreate
After getting acquainted with a new way of creating indicators, let's turn to a task that is closer to
reality. IndicatorCreate is usually used in cases where the called indicator is not known in advance.
Such a need, for example, arises when writing universal Expert Advisors capable of trading on arbitrary
signals configured by the user. And even the names of the indicators can be set by the user.
We are not yet ready to develop Expert Advisors, and therefore we will study this technology using the
example of a wrapper indicator UseDemoAll.mq5, capable of displaying the data of any other indicator.
The process should look like this. When we run UseDemoAll on the chart, a list appears in the
properties dialog where we should select one of the built-in indicators or a custom one, and in the latter
case, we will additionally need to specify its name in the input field. In another string parameter, we
can enter a list of parameters separated by commas. Parameter types will be determined automatically
based on their spelling. For example, a number with a decimal point (1 0.0) will be treated as a double,
a number without a dot (1 5) as an integer, and something enclosed in quotes ("text") as a string.
These are just basics settings of UseDemoAll, but not all possible. We will consider other settings later.
Let's take the ENUM_INDICATOR enumeration as the basis for the solution: it already has elements for
all types of indicators, including custom ones (IND_CUSTOM). To tell the truth, in its pure form, it does
not fit for several reasons. First, it is impossible to get metadata about a specific indicator from it, such
as the number and types of arguments, the number of buffers, and in which window the indicator is
displayed (main or subwindow). This information is important for the correct creation and visualization
of the indicator. Second, if we define an input variable of type ENUM_INDICATOR so that the user can
select the desired indicator, in the properties dialog this will be represented by a drop-down list, where
the options contain only the name of the element. Actually, it would be desirable to provide hints for
the user in this list (at least about parameters). Therefore, we will describe our own enumeration
IndicatorType. Recall that MQL5 allows for each element to specify a comment on the right, which is
shown in the interface.
In each element of the IndicatorType enumeration, we will encode not only the corresponding identifier
(ID) from ENUM_INDICATOR, but also the number of parameters (P), the number of buffers (B) and the
number of the working window (W). The following macros have been developed for this purpose.
#define MAKE_IND(P,B,W,ID) (int)((W << 24) | ((B & 0xFF) << 16) | ((P & 0xFF) << 8) | (ID & 0xFF))
#define IND_PARAMS(X)   ((X >> 8) & 0xFF)
#define IND_BUFFERS(X)  ((X >> 16) & 0xFF)
#define IND_WINDOW(X)   ((uchar)(X >> 24))
#define IND_ID(X)       ((ENUM_INDICATOR)(X & 0xFF))
The MAKE_IND macro takes all of the above characteristics as parameters and packs them into
different bytes of a single 4-byte integer, thus forming a unique code for the element of the new
enumeration. The remaining 4 macros allow you to perform the reverse operation, that is, to calculate
all the characteristics of the indicator using the code.
We will not provide the whole IndicatorType enumeration here, but only a part of it. The full source code
can be found in the file AutoIndicator.mqh.

---

## Page 813

Part 5. Creating application programs
81 3
5.5 Using ready-made indicators from MQL programs
enum IndicatorType
{
   iCustom_ = MAKE_IND(0, 0, 0, IND_CUSTOM), // {iCustom}(...)[?]
   
   iAC_ = MAKE_IND(0, 1, 1, IND_AC), // iAC( )[1]*
   iAD_volume = MAKE_IND(1, 1, 1, IND_AD), // iAD(volume)[1]*
   iADX_period = MAKE_IND(1, 3, 1, IND_ADX), // iADX(period)[3]*
   iADXWilder_period = MAKE_IND(1, 3, 1, IND_ADXW), // iADXWilder(period)[3]*
   ...
   iMomentum_period_price = MAKE_IND(2, 1, 1, IND_MOMENTUM), // iMomentum(period,price)[1]*
   iMFI_period_volume = MAKE_IND(2, 1, 1, IND_MFI), // iMFI(period,volume)[1]*
   iMA_period_shift_method_price = MAKE_IND(4, 1, 0, IND_MA), // iMA(period,shift,method,price)[1]
   iMACD_fast_slow_signal_price = MAKE_IND(4, 2, 1, IND_MACD), // iMACD(fast,slow,signal,price)[2]*
   ...
   iTEMA_period_shift_price = MAKE_IND(3, 1, 0, IND_TEMA), // iTEMA(period,shift,price)[1]
   iVolumes_volume = MAKE_IND(1, 1, 1, IND_VOLUMES), // iVolumes(volume)[1]*
   iWPR_period = MAKE_IND(1, 1, 1, IND_WPR) // iWPR(period)[1]*
};
The comments, which will become elements of the drop-down list visible to the user, indicate
prototypes with named parameters, the number of buffers in square brackets, and star marks of those
indicators that are displayed in their own window. The identifiers themselves are also made informative,
because they are the ones that are converted to text by the function EnumToString that is used to
output messages to the log.
The parameter list is particularly important, as the user will need to enter the appropriate comma-
separated values into the input variable reserved for this purpose. We could also show the types of the
parameters, but for simplicity, it was decided to leave only the names with a meaning, from which the
type can also be concluded. For example, period, fast, slow are integers with a period (number of bars),
method is the averaging method ENUM_MA_METHOD, price is the price type ENUM_APPLIED_PRICE,
volume is the volume type ENUM_APPLIED_VOLUME.
For the convenience of the user (so as not to remember the values of the enumeration elements), the
program will support the names of all enumerations. In particular, the sma identifier denotes
MODE_SMA, ema denotes MODE_EMA, and so on. Price close will turn into PRICE_CLOSE, open will turn
into PRICE_OPEN, and other types of prices will behave alike, by the last word (after underlining) in the
enumeration element identifier. For example, for the list of iMA indicator parameters
(iMA_period_shift_method_price), you can write the following line: 1 1 ,0,sma,close. Identifiers do not
need to be quoted. However, if necessary, you can pass a string with the same text, for example, a list
1 .5,"close" contains the real number 1 .5 and the string "close".
The indicator type, as well as strings with a list of parameters and, optionally, a name (if the indicator
is custom) are the main data for the AutoIndicator class constructor.

---

## Page 814

Part 5. Creating application programs
81 4
5.5 Using ready-made indicators from MQL programs
class AutoIndicator
{
protected:
   IndicatorTypetype;       // selected indicator type
   string symbols;          // working symbol (optional)
   ENUM_TIMEFRAMES tf;      // working timeframe (optional)
   MqlParamBuilder builder; // "builder" of the parameter array
   int handle;              // indicator handle
   string name;             // custom indicator name
   ...
public:
   AutoIndicator(const IndicatorType t, const string custom, const string parameters,
      const string s = NULL, const ENUM_TIMEFRAMES p = 0):
      type(t), name(custom), symbol(s), tf(p), handle(INVALID_HANDLE)
   {
      PrintFormat("Initializing %s(%s) %s, %s",
         (type == iCustom_ ? name : EnumToString(type)), parameters,
         (symbol == NULL ? _Symbol : symbol), EnumToString(tf == 0 ? _Period : tf));
      // split the string into an array of parameters (formed inside the builder)
      parseParameters(parameters);
      // create and store the handle
      handle = create();
   }
   
   int getHandle() const
   {
      return handle;
   }
};
Here and below, some fragments related to checking the input data for correctness are omitted. The
full source code is included with the book.
The process of analyzing a string with parameters is entrusted to the method parseParameters. It
implements the scheme described above with recognition of value types and their transfer to an the
MqlParamBuilder object, which we met in the previous example.

---

## Page 815

Part 5. Creating application programs
81 5
5.5 Using ready-made indicators from MQL programs
   int parseParameters(const string &list)
   {
      string sparams[];
      const int n = StringSplit(list, ',', sparams);
      
      for(int i = 0; i < n; i++)
      {
         // normalization of the string (remove spaces, convert to lower case)
         StringTrimLeft(sparams[i]);
         StringTrimRight(sparams[i]);
         StringToLower(sparams[i]);
   
         if(StringGetCharacter(sparams[i], 0) == '"'
         && StringGetCharacter(sparams[i], StringLen(sparams[i]) - 1) == '"')
         {
            // everything inside quotes is taken as a string
            builder << StringSubstr(sparams[i], 1, StringLen(sparams[i]) - 2);
         }
         else
         {
            string part[];
            int p = StringSplit(sparams[i], '.', part);
            if(p == 2) // double/float
            {
               builder << StringToDouble(sparams[i]);
            }
            else if(p == 3) // datetime
            {
               builder << StringToTime(sparams[i]);
            }
            else if(sparams[i] == "true")
            {
               builder << true;
            }
            else if(sparams[i] == "false")
            {
               builder << false;
            }
            else // int
            {
               int x = lookUpLiterals(sparams[i]);
               if(x == -1)
               {
                  x = (int)StringToInteger(sparams[i]);
               }
               builder << x;
            }
         }
      }
      
      return n;

---

## Page 816

Part 5. Creating application programs
81 6
5.5 Using ready-made indicators from MQL programs
   }
The helper function lookUpLiterals provides conversion of identifiers to standard enumeration constants.
   int lookUpLiterals(const string &s)
   {
      if(s == "sma") return MODE_SMA;
      else if(s == "ema") return MODE_EMA;
      else if(s == "smma") return MODE_SMMA;
      else if(s == "lwma") return MODE_LWMA;
      
      else if(s == "close") return PRICE_CLOSE;
      else if(s == "open") return PRICE_OPEN;
      else if(s == "high") return PRICE_HIGH;
      else if(s == "low") return PRICE_LOW;
      else if(s == "median") return PRICE_MEDIAN;
      else if(s == "typical") return PRICE_TYPICAL;
      else if(s == "weighted") return PRICE_WEIGHTED;
   
      else if(s == "lowhigh") return STO_LOWHIGH;
      else if(s == "closeclose") return STO_CLOSECLOSE;
   
      else if(s == "tick") return VOLUME_TICK;
      else if(s == "real") return VOLUME_REAL;
      
      return -1;
   }
After the parameters are recognized and saved in the object's internal array MqlParamBuilder, the
create method is called. Its purpose is to copy the parameters to the local array, supplement it with
the name of the custom indicator (if any), and call the IndicatorCreate function.
   int create()
   {
      MqlParam p[];
      // fill 'p' array with parameters collected by 'builder' object
      builder >> p;
      
      if(type == iCustom_)
      {
         // insert the name of the custom indicator at the very beginning
         ArraySetAsSeries(p, true);
         const int n = ArraySize(p);
         ArrayResize(p, n + 1);
         p[n].type = TYPE_STRING;
         p[n].string_value = name;
         ArraySetAsSeries(p, false);
      }
      
      return IndicatorCreate(symbol, tf, IND_ID(type), ArraySize(p), p);
   }
The method returns the received handle.

---

## Page 817

Part 5. Creating application programs
81 7
5.5 Using ready-made indicators from MQL programs
Of particular interest is how an additional string parameter with the name of the custom indicator is
inserted at the very beginning of the array. First, the array is assigned an indexing order "as in
timeseries" (see ArraySetAsSeries), as a result of which the index of the last (physically, by location in
memory) element becomes equal to 0, and the elements are counted from right to left. Then the array
is increased in size and the indicator name is written to the added element. Due to reverse indexing,
this addition does not occur to the right of existing elements, but to the left. Finally, we return the
array to its usual indexing order, and at index 0 is the new element with the string that was just the
last.
Optionally, the AutoIndicator class can form an abbreviated name of the built-in indicator from the
name of an enumeration element.
   ...
   string getName() const
   {
      if(type != iCustom_)
      {
         const string s = EnumToString(type);
         const int p = StringFind(s, "_");
         if(p > 0) return StringSubstr(s, 0, p);
         return s;
      }
      return name;
   }
};
Now everything is ready to go directly to the source code UseDemoAll.mq5. But let's start with a
slightly simplified version UseDemoAllSimple.mq5.
First of all, let's define the number of indicator buffers. Since the maximum number of buffers among
the built-in indicators is five (for Ichimoku), we take it as a limiter. We will assign the registration of this
number of arrays as buffers to the class already known to us, BufferArray (see the section
Multicurrency and multitimeframe indicators, example IndUnityPercent).
#define BUF_NUM 5
   
#property indicator_chart_window
#property indicator_buffers BUF_NUM
#property indicator_plots   BUF_NUM
   
#include <MQL5Book/IndBufArray.mqh>
BufferArray buffers(5);
It is important to remember that an indicator can be designed either to be displayed in the main
window or in a separate window. MQL5 does not allow combining two modes. However, we do not know
in advance which indicator the user will choose, and therefore we need to invent some kind of
"workaround". For now, let's place our indicator in the main window, and we'll deal with the problem of
a separate window later.
Purely technically, there are no obstacles to copying data from indicator buffers with the property
indicator_ separate_ window into their buffers displayed in the main window. However, it should be kept
in mind that the range of values of such indicators often does not coincide with the scale of prices, and

---

## Page 818

Part 5. Creating application programs
81 8
5.5 Using ready-made indicators from MQL programs
therefore it is unlikely that you will be able to see them on the chart (the lines will be somewhere far
beyond the visible area, at the top or bottom), although the values are still will be output to Data
window.
With the help of input variables, we will select the indicator type, the name of the custom indicator, and
the list of parameters. We will also add variables for the rendering type and line width. Since buffers will
be connected to work dynamically, depending on the number of buffers of the source indicator, we do
not describe buffer styles statically using directives and will do this in OnInit via calls of built-in Plot
functions.
input IndicatorType IndicatorSelector = iMA_period_shift_method_price; // Built-in Indicator Selector
input string IndicatorCustom = ""; // Custom Indicator Name
input string IndicatorParameters = "11,0,sma,close"; // Indicator Parameters (comma,separated,list)
input ENUM_DRAW_TYPE DrawType = DRAW_LINE; // Drawing Type
input int DrawLineWidth = 1; // Drawing Line Width
Let's define a global variable to store the indicator descriptor.
int Handle;
In the OnInit handler, we use the AutoIndicator class presented earlier, for parsing an input data,
preparing the MqlParam array and obtaining a handle based on it.
#include <MQL5Book/AutoIndicator.mqh>
   
int OnInit()
{
   AutoIndicator indicator(IndicatorSelector, IndicatorCustom, IndicatorParameters);
   Handle = indicator.getHandle();
   if(Handle == INVALID_HANDLE)
   {
      Alert(StringFormat("Can't create indicator: %s",
         _LastError ? E2S(_LastError) : "The name or number of parameters is incorrect"));
      return INIT_FAILED;
   }
   ...
To customize the plots, we describe a set of colors and get the short name of the indicator from the
AutoIndicator object. We also calculate the number of used n buffers of the built-in indicator using the
IND_BUFFERS macro, and for any custom indicator (which is not known in advance), for lack of a
better solution, we will include all buffers. Further, in the process of copying data, unnecessary
CopyBuffer calls will simply return an error, and such arrays can be filled with empty values.
   ...
   static color defColors[BUF_NUM] = {clrBlue, clrGreen, clrRed, clrCyan, clrMagenta};
   const string s = indicator.getName();
   const int n = (IndicatorSelector != iCustom_) ? IND_BUFFERS(IndicatorSelector) : BUF_NUM;
   ...
In the loop, we will set the properties of the charts, taking into account the limiter n: the buffers above
it are hidden.

---

## Page 819

Part 5. Creating application programs
81 9
5.5 Using ready-made indicators from MQL programs
   for(int i = 0; i < BUF_NUM; ++i)
   {
      PlotIndexSetString(i, PLOT_LABEL, s + "[" + (string)i + "]");
      PlotIndexSetInteger(i, PLOT_DRAW_TYPE, i < n ? DrawType : DRAW_NONE);
      PlotIndexSetInteger(i, PLOT_LINE_WIDTH, DrawLineWidth);
      PlotIndexSetInteger(i, PLOT_LINE_COLOR, defColors[i]);
      PlotIndexSetInteger(i, PLOT_SHOW_DATA, i < n);
   }
   
   Comment("DemoAll: ", (IndicatorSelector == iCustom_ ? IndicatorCustom : s),
      "(", IndicatorParameters, ")");
   
   return INIT_SUCCEEDED;
}
In the upper left corner of the chart, the comment will display the name of the indicator with
parameters.
In the OnCalculate handler, when the handle data is ready, we read them into our arrays.
int OnCalculate(ON_CALCULATE_STD_SHORT_PARAM_LIST)
{
   if(BarsCalculated(Handle) != rates_total)
   {
      return prev_calculated;
   }
   
   const int m = (IndicatorSelector != iCustom_) ? IND_BUFFERS(IndicatorSelector) : BUF_NUM;
   for(int k = 0; k < m; ++k)
   {
      // fill our buffers with data form the indicator with the 'Handle' handle
      const int n = buffers[k].copy(Handle,
         k, 0, rates_total - prev_calculated + 1);
         
      // in case of error clean the buffer
      if(n < 0)
      {
         buffers[k].empty(EMPTY_VALUE, prev_calculated, rates_total - prev_calculated);
      }
   }
   
   return rates_total;
}
The above implementation is simplified and matches the original file UseDemoAllSimple.mq5. We will
deal with its extension further, but for now we will check the behavior of the current version. The
following image shows 2 instance of the indicator: blue line with default settings
(iMA_ period_ shift_ method_ price, options "1 1 ,0,sma,close"), and the red iRSI_ period_ price with
parameters "1 1  close".

---

## Page 820

Part 5. Creating application programs
820
5.5 Using ready-made indicators from MQL programs
Two instances of the UseDemoAllSimple indicator with iMA and iRSI readings
The USDRUB chart was intentionally chosen for demonstration, because the values of the quotes here
more or less coincide with the range of the RSI indicator (which should have been displayed in a
separate window). On most charts of other symbols, we would not notice the RSI. If you only care
about programmatic access to values, then this is not a big deal, but if you have visualization
requirements, this is a problem that should be solved.
So, you should somehow provide a separate display of the indicators intended for the subwindow.
Basically, there is a popular request from the MQL developers community to enable the display of
graphics both in the main window and in a subwindow at the same time. We will present one of the
solutions, but for this you need to first get acquainted with some of the new features.
5.5.1 0 Overview of functions managing indicators on the chart
As we have already figured out, indicators are the type of MQL programs that combine the calculation
part and visualization. The calculations are performed internally, imperceptibly to the user, but the
visualization requires linking to the chart. That is why indicators are closely related to charts, and the
MQL5 API even contains a group of functions that manage indicators on charts. We will discuss these
functions in more detail in the chapter on charts, and here we just give a list of them.

---

## Page 821

Part 5. Creating application programs
821 
5.5 Using ready-made indicators from MQL programs
Function
Purpose
ChartWindowFind
Returns the number of the subwindow containing the current
indicator or an indicator with the given name
ChartIndicatorAdd
Adds an indicator with the specified handle to the specified chart
window
ChartIndicatorDelete
Removes an indicator with the specified name from the specified
chart window
ChartIndicatorGet
Returns the indicator handle with the specified short name on the
specified chart window
ChartIndicatorName
Returns the short name of the indicator by number in the list of
indicators on the specified chart window
ChartIndicatorsTotal
Returns the number of all indicators attached to the specified chart
window
In the next section about Combining information output in the main and auxiliary window, we will see an
example UseDemoAll.mq5, which uses some of these functions.
5.5.1 1  Combining output to main and auxiliary windows 
Let's return to the problem of displaying graphics from one indicator in the main window and in a
subwindow, since we encountered it when developing the example UseDemoAllSimple.mq5. I intended
for a separate window are not suitable for visualization on the main chart, and indicators for the main
window do not have additional windows. There are several alternative approaches:
·Implement a parent indicator for a separate window and display charts there and use it in the main
window to display data of type graphic objects. This is bad, because data from objects cannot be
read the same way as from a timeseries, and many objects consume extra resources.
·Develop your own virtual panel (class) for the main window and, with the correct scale, represent
there timeseries which should be displayed in the subwindow.
·Use several indicators, at least one for the main window and one for the subwindow, and exchange
data between them via shared memory (DLL required), resources or database.
·Duplicate calculations (use common source code) in indicators for the main window and subwindow.
We will present one of the solutions which goes beyond a single MQL program: we need an additional
indicator with the indicator_ separate_ window property. We actually already have it since we create its
calculated part by requesting a handle. We only need to somehow display it in a separate subwindow.
In the new (full) version of UseDemoAll.mq5, we will analyze the metadata of the indicator requested to
be created in the corresponding IndicatorType enumeration element. Recall that, among other things,
the working window of each type of built-in indicator is encoded there. When an indicator requires a
separate window, we will create one using special MQL5 functions, which we have yet to learn.
There is no way to get information about the working window for custom indicators. So, let's add the
IndicatorCustomSubwindow input variable, in which the user can specify that a subwindow is required.

---

## Page 822

Part 5. Creating application programs
822
5.5 Using ready-made indicators from MQL programs
input bool IndicatorCustomSubwindow = false; // Custom Indicator Subwindow
In OnInit, we hide the buffers intended for the subwindow.
int OnInit()
{
   ...
   const bool subwindow = (IND_WINDOW(IndicatorSelector) > 0)
      || (IndicatorSelector == iCustom_ && IndicatorCustomSubwindow);
   for(int i = 0; i < BUF_NUM; ++i)
   {
      ...
      PlotIndexSetInteger(i, PLOT_DRAW_TYPE,
         i < n && !subwindow ? DrawType : DRAW_NONE);
   }
   ...
After this setup, we will have to use a couple of functions that apply not only to working with indicators,
but also with charts. We will study them in detail in the corresponding chapter, while an introductory
overview is presented in the previous section.
One of the functions ChartIndicatorAdd allows you to add the indicator specified by the handle to the
window, and not only to the main part, but also to the subwindow. We will talk about chart identifiers
and window numbering in the chapter on charts, and for now it is enough to know that the next
ChartIndicatorAdd function call adds an indicator with the handle to the current chart, to a new
subwindow.
 inthandle = ...// get indicator handle, iCustom or IndicatorCreate
                    // set the current chart (0)
                    // |
                    // |     set the window number to the current number of windows
                    // |                          |
                    // |                          | passing the descriptor
                    // |                          |                       |
                    // v                          v                       v
   ChartIndicatorAdd(  0, (int)ChartGetInteger(0, CHART_WINDOWS_TOTAL), handle); 
Knowing about this possibility, we can think about calling ChartIndicatorAdd and pass to it the handle of
a ready-made subordinate indicator.
The second function we need is ChartIndicatorName. It returns the short name of the indicator by its
handle. This name corresponds to the INDICATOR_SHORTNAME property set in the indicator code and
may differ from the file name. The name will be required to clean up after itself, that is, to remove the
auxiliary indicator and its subwindow, after deleting or reconfiguring the parent indicator.

---

## Page 823

Part 5. Creating application programs
823
5.5 Using ready-made indicators from MQL programs
string subTitle = "";
   
int OnInit()
{
   ...
   if(subwindow)
   {
      // show a new indicator in the subwindow
      const int w = (int)ChartGetInteger(0, CHART_WINDOWS_TOTAL);
      ChartIndicatorAdd(0, w, Handle);
      // save the name to remove the indicator in OnDeinit
      subTitle = ChartIndicatorName(0, w, 0);
   }
   ...
}
In the OnDeinit handler, we use the saved subTitle to call another function which we will study later –
ChartIndicatorDelete. It removes the indicator with the name specified in the last argument from the
chart.
void OnDeinit(const int)
{
   Print(__FUNCSIG__, (StringLen(subTitle) > 0 ? " deleting " + subTitle : ""));
   if(StringLen(subTitle) > 0)
   {
      ChartIndicatorDelete(0, (int)ChartGetInteger(0, CHART_WINDOWS_TOTAL) - 1,
         subTitle);
   }
}
It is assumed here that only our indicator works on the chart, and only in a single instance. In a more
general case, all subwindows should be analyzed for correct deletion, but this would require a few more
functions from those that will be presented in the chapter on charts, so we restrict ourselves to a
simple version for the time being.
If now we run UseDemoAll and select an indicator marked with an asterisk (that is, the one that
requires a subwindow) from the list, for example, RSI, we will see the expected result: RSI in a separate
window.

---

## Page 824

Part 5. Creating application programs
824
5.5 Using ready-made indicators from MQL programs
RSI in the subwindow created by the UseDemoAll indicator
5.5.1 2 Reading data from charts that have a shift
Our new indicator UseDemoAll is almost ready. We only need to consider one more point.
In the subordinate indicator, some charts can have an offset set by the PLOT_SHIFT property. For
example, with a positive shift, the timeseries elements are shifted into the future and displayed to the
right of the bar with index 0. Their indexes, oddly enough, are negative. As you move to the right, the
numbers decrease more and more: -1 , -2, -3, etc. This addressing also affects the CopyBuffer function.
When we use the first form of CopyBuffer, the offset parameter set to 0 refers to the element with the
current time in the timeseries. But if the timeseries itself is shifted to the right, we will get data starting
from the element numbered N, where N is the shift value in the source indicator. At the same time, the
elements located in our buffer to the right of index N will not be filled with data, and "garbage" will
remain in them.
To demonstrate the problem, let's start with an indicator without a shift: Awesome Oscillator fits
perfectly to this requirement. Recall that UseDemoAll copies all values to its arrays, and although they
are not visible on the chart due to different price scales and indicator readings, we can check against
Data Window. Wherever we move the mouse cursor on the chart, the indicator values in the subwindow
in the Data Window and in UseDemoAll buffers will match. For example, in the image below, you can
clearly see that on the hourly bar at 1 6:00 both values are equal to 0.001 797.

---

## Page 825

Part 5. Creating application programs
825
5.5 Using ready-made indicators from MQL programs
AO indicator data in UseDemoAll buffers
Now, in UseDemoAll settings, we select the iGator (Gator Oscillator) indicator. For simplicity, clear the
field with Gator parameters, so that it will be built with its default parameters. In this case, the
histogram shift is 5 bars (forward), which is clearly seen on the chart.
Gator indicator data in UseDemoAll buffers without correction for future shift

---

## Page 826

Part 5. Creating application programs
826
5.5 Using ready-made indicators from MQL programs
The black vertical line marks the 1 6:00 hour bar. However, the Gator indicator values in the Data
Window and in our arrays read from the same indicator are different. Yellow color UseDemoAll highlights
buffers containing garbage.
If we examine the data moving 5 bars into  the past, at 1 1 :00 (orange vertical line), we will find there
the values that Gator outputs at 1 6:00. The pairwise correct values of the upper and lower histograms
are highlighted in green and pink, respectively.
To solve this problem, we have to add to UseDemoAll an input variable for the user to specify a chart
shift, and then make a correction for it when calling CopyBuffer.
input int IndicatorShift = 0; // Plot Shift
...
int OnCalculate(ON_CALCULATE_STD_SHORT_PARAM_LIST)
{
   ...
   for(int k = 0; k < m; ++k)
   {
      const int n = buffers[k].copy(Handle, k,
         -IndicatorShift, rates_total - prev_calculated + 1);
      ...
   }
}
Unfortunately, it is impossible to find the PLOT_SHIFT property for a third-party indicator from MQL5.
Let's check how introducing a shift of 5 fixes the situation with the Gator indicator (with default
settings).
Gator indicator data in UseDemoAll buffers after adjusting for future shift

---

## Page 827

Part 5. Creating application programs
827
5.5 Using ready-made indicators from MQL programs
Now the readings of UseDemoAll at the 1 6:00 bar correspond to the actual data from Gator from the
virtual future 5 bars ahead (lilac vertical line at 21 :00).
You may wonder why only 2 buffers are displayed in the Gator window while our one has 4. The point is
that the color histogram of Gator uses one additional buffer for color encoding. But there are only two
colors, red and green, and we see them in our arrays as 0 or 1 .
5.5.1 3 Deleting indicator instances: IndicatorRelease
As mentioned in the introductory part of this chapter, the terminal maintains a reference counter for
each created indicator and leaves it in operation for as long as at least one MQL program or chart uses
it. In an MQL program, a sign of the need for an indicator is a valid handle. Usually, we ask for a handle
during initialization and use it in algorithms until the end of the program.
At the moment the program is unloaded, all created unique handles are automatically released, that is,
their counters are decremented by 1  (and if they reach zero, those indicators are also unloaded from
memory). Therefore, there is no need to explicitly release the handle.
However, there are situations when a sub-indicator becomes unnecessary during program operation.
Then the useless indicator continues to consume resources. Therefore, you must explicitly release the
handle with IndicatorRelease.
bool IndicatorRelease(int handle)
The function deletes the specified indicator handle and unloads the indicator itself if no one else uses it.
Unloading occurs with a slight delay.
The function returns an indicator of success (true) or errors (false).
After the call of IndicatorRelease, the handle passed to it becomes invalid, even though the variable
itself retains its previous value. An attempt to use such a handle in other indicator functions like
CopyBuffer will fail with error 4807 (ERR_INDICATOR_WRONG_HANDLE). To avoid misunderstandings,
it is desirable to assign the value INVALID_HANDLE to the corresponding variable immediately after the
handle is freed.
However, if the program then requests a handle for a new indicator, that handle will most likely have
the same value as the previously released one but will now be associated with the new indicator's data.
When working in the strategy tester, the IndicatorRelease function is not performed.
To demonstrate the application of IndicatorRelease, let's prepare a special version of
UseDemoAllLoop.mq5, which will periodically recreate an auxiliary indicator in a cycle from the list,
which will include only indicators for the main window (for clarity).

---

## Page 828

Part 5. Creating application programs
828
5.5 Using ready-made indicators from MQL programs
IndicatorType MainLoop[] =
{
   iCustom_,
   iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price,
   iAMA_period_fast_slow_shift_price,
   iBands_period_shift_deviation_price,
   iDEMA_period_shift_price,
   iEnvelopes_period_shift_method_price_deviation,
   iFractals_,
   iFrAMA_period_shift_price,
   iIchimoku_tenkan_kijun_senkou,
   iMA_period_shift_method_price,
   iSAR_step_maximum,
   iTEMA_period_shift_price,
   iVIDyA_momentum_smooth_shift_price,
};
   
const int N = ArraySize(MainLoop);
int Cursor = 0; // current position inside the MainLoop array
      
const string IndicatorCustom = "LifeCycle";
The first element of the array contains one custom indicator as an exception, LifeCycle from the section
Features of starting and stopping programs of different types. Although this indicator does not display
any lines, it is appropriate here because it displays messages in the log when its OnInit/OnDeinit
handlers are called, which will allow you to track its life cycle. Life cycles of other indicators are similar.
 
In the input variables, we will leave only the rendering settings. The default output of DRAW_ARROW
labels is optimal for displaying different types of indicators.
input ENUM_DRAW_TYPE DrawType = DRAW_ARROW; // Drawing Type
input int DrawLineWidth = 1; // Drawing Line Width
To recreate indicators "on the go", let's run 5 second timer in OnInit, and the entire previous
initialization (with some modifications described below) will be moved to the OnTimer handler.

---

## Page 829

Part 5. Creating application programs
829
5.5 Using ready-made indicators from MQL programs
int OnInit()
{
   Comment("Wait 5 seconds to start looping through indicator set");
   EventSetTimer(5);
   return INIT_SUCCEEDED;
}
   
IndicatorType IndicatorSelector; // currently selected indicator type
   
void OnTimer()
{
   if(Handle != INVALID_HANDLE && ClearHandles)
   {
      IndicatorRelease(Handle);
      /*
      // descriptor is still 10, but is no longer valid
      // if we uncomment the fragment, we get the following error
      double data[1];
      const int n = CopyBuffer(Handle, 0, 0, 1, data);
      Print("Handle=", Handle, " CopyBuffer=", n, " Error=", _LastError);
      // Handle=10 CopyBuffer=-1 Error=4807 (ERR_INDICATOR_WRONG_HANDLE)
      */
   }
   IndicatorSelector = MainLoop[Cursor];
   Cursor = ++Cursor % N;
   
   // create a handle with default parameters
   // (because we pass an empty string in the third argument of the constructor)
   AutoIndicator indicator(IndicatorSelector,
      (IndicatorSelector == iCustom_ ? IndicatorCustom : ""), "");
   Handle = indicator.getHandle();
   if(Handle == INVALID_HANDLE)
   {
      Print(StringFormat("Can't create indicator: %s",
         _LastError ? E2S(_LastError) : "The name or number of parameters is incorrect"));
   }
   else
   {
      Print("Handle=", Handle);
   }
   
   buffers.empty(); // clear buffers because a new indicator will be displayed
   ChartSetSymbolPeriod(0,NULL,0); // request a full redraw
   ...
   // further setup of diagrams - similar to the previous one
   ...
   Comment("DemoAll: ", (IndicatorSelector == iCustom_ ? IndicatorCustom : s),
      "(default-params)");
}
The main difference is that the type of the currently created indicator IndicatorSelector now it is not
set by the user but is sequentially selected from the MainLoop array at the Cursor index. Each time the

---

## Page 830

Part 5. Creating application programs
830
5.5 Using ready-made indicators from MQL programs
timer is called, this index increases cyclically, that is, when the end of the array is reached, we jump to
its beginning.
For all indicators, the line with parameters is empty. This is done to unify their initialization. As a result,
each indicator will be created with its own defaults.
At the beginning of the OnTimer handler, we call IndicatorRelease for the previous handle. However, we
have provided an input variable ClearHandles to disable the given if operator branch and see what
happens if you do not clean the handles.
input bool ClearHandles = true;
By default, ClearHandles is equal to true, that is, the indicators will be deleted as expected.
Finally, another additional setting is the lines with clearing buffers and requesting a complete redrawing
of the chart. Both are needed, because we have replaced the slave indicator that supplies the displayed
data.
The OnCalculate handler has not changed.
Let's run UseDemoAllLoop with default settings. The following entries will appear in the log (only the
beginning is shown):
UseDemoAllLoop (EURUSD,H1) Initializing LifeCycle() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) Handle=10
LifeCycle      (EURUSD,H1) Loader::Loader()
LifeCycle      (EURUSD,H1) void OnInit() 0 DEINIT_REASON_PROGRAM
UseDemoAllLoop (EURUSD,H1) Initializing iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price requires 8 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=10
LifeCycle      (EURUSD,H1) void OnDeinit(const int) DEINIT_REASON_REMOVE
LifeCycle      (EURUSD,H1) Loader::~Loader()
UseDemoAllLoop (EURUSD,H1) Initializing iAMA_period_fast_slow_shift_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iAMA_period_fast_slow_shift_price requires 5 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=10
UseDemoAllLoop (EURUSD,H1) Initializing iBands_period_shift_deviation_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iBands_period_shift_deviation_price requires 4 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=10
...
Note that we get the same handle "number" (1 0) every time because we free it before creating a new
handle.
It is also important that the LifeCycle indicator unloaded shortly after we freed it (assuming it was not
added to the same chart by itself, because then its reference count would not be reset to zero).
The image below shows the moment when our indicator renders Alligator data.

---

## Page 831

Part 5. Creating application programs
831 
5.5 Using ready-made indicators from MQL programs
UseDemoAllLoop in the Alligator demo step
If you change the ClearHandles value to false, we will see a completely different picture in the log.
Handle numbers will now constantly increase, indicating that the indicators remain in the terminal and
continue to work, consuming resources in vain. In particular, no deinitialization message is received
from the LifeCycle indicator.

---

## Page 832

Part 5. Creating application programs
832
5.5 Using ready-made indicators from MQL programs
UseDemoAllLoop (EURUSD,H1) Initializing LifeCycle() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) Handle=10
LifeCycle      (EURUSD,H1) Loader::Loader()
LifeCycle      (EURUSD,H1) void OnInit() 0 DEINIT_REASON_PROGRAM
UseDemoAllLoop (EURUSD,H1) Initializing iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price requires 8 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=11
UseDemoAllLoop (EURUSD,H1) Initializing iAMA_period_fast_slow_shift_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iAMA_period_fast_slow_shift_price requires 5 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=12
UseDemoAllLoop (EURUSD,H1) Initializing iBands_period_shift_deviation_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iBands_period_shift_deviation_price requires 4 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=13
UseDemoAllLoop (EURUSD,H1) Initializing iDEMA_period_shift_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iDEMA_period_shift_price requires 3 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=14
UseDemoAllLoop (EURUSD,H1) Initializing iEnvelopes_period_shift_method_price_deviation() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iEnvelopes_period_shift_method_price_deviation requires 5 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=15
...
UseDemoAllLoop (EURUSD,H1) Initializing iVIDyA_momentum_smooth_shift_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iVIDyA_momentum_smooth_shift_price requires 4 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=22
UseDemoAllLoop (EURUSD,H1) Initializing LifeCycle() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) Handle=10
UseDemoAllLoop (EURUSD,H1) Initializing iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price requires 8 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=11
UseDemoAllLoop (EURUSD,H1) Initializing iAMA_period_fast_slow_shift_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iAMA_period_fast_slow_shift_price requires 5 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=12
UseDemoAllLoop (EURUSD,H1) Initializing iBands_period_shift_deviation_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iBands_period_shift_deviation_price requires 4 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=13
UseDemoAllLoop (EURUSD,H1) Initializing iDEMA_period_shift_price() EURUSD, PERIOD_H1
UseDemoAllLoop (EURUSD,H1) iDEMA_period_shift_price requires 3 parameters, 0 given
UseDemoAllLoop (EURUSD,H1) Handle=14
UseDemoAllLoop (EURUSD,H1) void OnDeinit(const int)
...
When the index in the loop over the array of indicator types reaches the last element and circles from
the beginning, the terminal will start returning handles of already existing indicators to our code (the
same values: handle 22 is followed by 1 0 again).
5.5.1 4 Getting indicator settings by its handle
Sometimes an MQL program needs to know the parameters of a running indicator instance. These can
be third-party indicators on the chart, or a handle passed from the main program to library or header
file. For this purpose, MQL5 provides the IndicatorParameters function.
int IndicatorParameters(int handle, ENUM_INDICATOR &type, MqlParam &params[])
By the specified handle, the function returns the number of indicator input parameters, as well as their
types and values.

---

## Page 833

Part 5. Creating application programs
833
5.5 Using ready-made indicators from MQL programs
On success, the function populates the params array passed to it, and the indicator type is saved in the
type parameter.
In case of an error, the function returns -1 .
As an example of working with this function, let's improve the indicator UseDemoAllLoop.mq5 presented
in the section on Deleting indicator instances. Let's call the new version UseDemoAllParams.mq5.
As you remember, we sequentially created some built-in indicators in the loop in the list and left the list
of parameters empty, which leads to the fact that the indicators use some unknown default values. In
this regard, we displayed a generalized prototype in a comment on the chart: with a name, but without
specific values.
// UseDemoAllLoop.mq5
void OnTimer()
{
   ...
   Comment("DemoAll: ", (IndicatorSelector == iCustom_ ? IndicatorCustom : s),
      "(default-params)");
   ...
}
Now we have the opportunity to find out its parameters based on the indicator handle and display them
to the user.
// UseDemoAllParams.mq5
void OnTimer()
{
   ...   
   // read the parameters applied by the indicator by default
   ENUM_INDICATOR itype;
   MqlParam defParams[];
   const int p = IndicatorParameters(Handle, itype, defParams);
   ArrayPrint(defParams);
   Comment("DemoAll: ", (IndicatorSelector == iCustom_ ? IndicatorCustom : s),
      "(" + MqlParamStringer::stringify(defParams) + ")");
   ...
}
Conversion of the MqlParam array into a string is implemented in the special class MqlParamStringer
(see file MqlParamStringer.mqh).

---

## Page 834

Part 5. Creating application programs
834
5.5 Using ready-made indicators from MQL programs
class MqlParamStringer
{
public:
   static string stringify(const MqlParam &param)
   {
      switch(param.type)
      {
      case TYPE_BOOL:
      case TYPE_CHAR:
      case TYPE_UCHAR:
      case TYPE_SHORT:
      case TYPE_USHORT:
      case TYPE_DATETIME:
      case TYPE_COLOR:
      case TYPE_INT:
      case TYPE_UINT:
      case TYPE_LONG:
      case TYPE_ULONG:
         return IntegerToString(param.integer_value);
      case TYPE_FLOAT:
      case TYPE_DOUBLE:
         return (string)(float)param.double_value;
      case TYPE_STRING:
         return param.string_value;
      }
      return NULL;
   }
   
   static string stringify(const MqlParam &params[])
   {
      string result = "";
      const int p = ArraySize(params);
      for(int i = 0; i < p; ++i)
      {
         result += stringify(params[i]) + (i < p - 1 ? "," : "");
      }
      return result;
   }
};
After compiling and running the new indicator, you can make sure that the specific list of parameters of
the indicator being rendered is now displayed in the upper left corner of the chart.
For a single custom indicator from the list (LifeCycle), the first parameter will contain the path and file
name of the indicator. The second parameter is described in the source code as an integer. But the
third parameter is interesting because it implicitly describes the 'Apply to' property, which is inherent
in all indicators with a short form of the OnCalculate handler. In this case, by default, the indicator is
applied to PRICE_CLOSE (value 1 ).

---

## Page 835

Part 5. Creating application programs
835
5.5 Using ready-made indicators from MQL programs
Initializing LifeCycle() EURUSD, PERIOD_H1
Handle=10
    [type] [integer_value] [double_value] [string_value]
[0]     14               0        0.00000 "Indicators\MQL5Book\p5\LifeCycle.ex5"
[1]      7               0        0.00000 null
[2]      7               1        0.00000 null
Initializing iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price() EURUSD, PERIOD_H1
iAlligator_jawP_jawS_teethP_teethS_lipsP_lipsS_method_price requires 8 parameters, 0 given
Handle=10
    [type] [integer_value] [double_value] [string_value]
[0]      7              13        0.00000 null          
[1]      7               8        0.00000 null          
[2]      7               8        0.00000 null          
[3]      7               5        0.00000 null          
[4]      7               5        0.00000 null          
[5]      7               3        0.00000 null          
[6]      7               2        0.00000 null          
[7]      7               5        0.00000 null          
Initializing iAMA_period_fast_slow_shift_price() EURUSD, PERIOD_H1
iAMA_period_fast_slow_shift_price requires 5 parameters, 0 given
Handle=10
    [type] [integer_value] [double_value] [string_value]
[0]      7               9        0.00000 null          
[1]      7               2        0.00000 null          
[2]      7              30        0.00000 null          
[3]      7               0        0.00000 null          
[4]      7               1        0.00000 null          
According to the log, the settings of the built-in indicators also correspond to the defaults.
5.5.1 5 Defining data source for an indicator
Among the MQL program built-in variables, there is one that can only be used in indicators. This is the
_ AppliedTo variable of type int, which allows you to read the Apply to property from the indicator
settings dialog. In addition, if the indicator is created by calling the iCustom function, to which the
handle of the third-party indicator was passed, then the _ AppliedTo variable will contain this handle.
The following table describes the possible values for the _ AppliedTo variable.

---

## Page 836

Part 5. Creating application programs
836
5.5 Using ready-made indicators from MQL programs
Value
Description of data for calculation
0
The indicator uses the full form of OnCalculate, and the data for the calculation is
not set by one data array
1
Close Price
2
Open price
3
High Price
4
Low price
5
Average Price = (High+Low)/2
6
Typical Price = (High+Low+Close)/3
7
Weighted Price = (Open+High+Low+Close)/4
8
Data of the indicator that was launched on the chart before this indicator
9
Data of the indicator that was launched on the chart the very first
1 0+
Data of indicator with the handle contained in _AppliedTo; this handle was passed as
a last parameter to the iCustom function when creating the indicator 
For the convenience of analyzing the values, attached to this book is a header file AppliedTo.mqh with
the enumeration.
5.6 Working with timer
For many applied tasks, it is important to be able to perform actions on a schedule, with some specified
interval. In MQL5, this functionality is provided by the timer, a system time counter that can be
configured to send regular notifications to an MQL program.
There are several functions for setting or canceling timer notifications in the MQL5 API: EventSetTimer,
EventSetMillisecondTimer, EventKillTimer. The notifications themselves enter the program as events of
a special type: the OnTimer handler is reserved for them in the source code. This group of functions will
be discussed in this chapter.
Recall that in MQL5 events can only be received by interactive programs running on charts, that is,
indicators and Expert Advisors. Scripts and Services do not support any events, including those from
the timer.
However, in the chapter Functions for working with time, we have already touched on related topics:
·Getting the timestamps of the current local or server clock (TimeLocal / TimeCurrent)
·Pausing the execution of the program for a specified period using Sleep
·Getting the state of the computer's system time counter, counted from the start of the operating
system (GetTickCount) or since the launch of MQL-program (GetMicrosecondCount)
These options are open to absolutely all types of MQL programs.

---

## Page 837

Part 5. Creating application programs
837
5.6 Working with timer
In the previous chapters, we have already used the timer functions many times, although their formal
description will be given only now. Due to the fact that timer events are available only in indicators or
Expert Advisors, it would be difficult to study it before the programs themselves. After we have
mastered the creation of indicators, the topic of timers will become a logical continuation.
Basically, we used timers to wait for the timeseries to be built. Such examples can be found in the
sections Waiting for data, Multicurrency and multitimeframe indicators, Support for multiple symbols
and timeframes, Using built-in indicators.
In addition, we timed (every 5 seconds) the type of the subordinate indicator in the indicator
"animation" demo in the section Deleting indicator instances.
5.6.1  Turning timer on and off: EventSetTimer/EventKillTimer
MQL5 allows you to enable or disable the standard timer to perform any scheduled actions. There are
two functions for this: EventSetTimer and EventKillTimer.
bool EventSetTimer(int seconds)
The function indicates to the client terminal that for this Expert Advisor or indicator it is necessary to
generate events from the timer with the specified frequency, which is set in seconds (parameter
seconds).
The function returns a sign of success (true) or error (false). The error code can be obtained from
_ LastError.
In order to process timer events, an Expert Advisor or an indicator must have the OnTimer function in
its code. The first timer event will not occur immediately after the call of EventSetTimer, but after
seconds seconds.
For each Expert Advisor or indicator that calls the EventSetTimer function, it creates its own, dedicated
timer. The program will receive events only from it. Timers in different programs work independently.
Each interactive MQL program placed on a chart has a separate event queue where the events received
for it are added. If there is already an event in the queue OnTimer or it is in processing state, then the
new OnTimer event is not queued.
If the timer is no longer needed, it should be disabled with the EventKillTimer function.
void EventKillTimer(void)
The function stops the timer that was enabled before by the EventSetTimer function (or by
EventSetMillisecondTimer, which we will discuss next). The function can also be called from the OnTimer
handler. Thus, in particular, it is possible to perform a delayed single action.
The call of EventKillTimer in indicators does not clear the queue, so after it you can get the last
residual OnTimer event.
When the MQL program terminates, the timer is forcibly destroyed if it was created but not disabled by
the EventKillTimer function.
Each program can only set one timer. Therefore, if you want to call different parts of the algorithm at
different intervals, you should enable a timer with a period that is the least common divisor of the
required periods (in the limiting case, with a minimum period of 1  second), and in the OnTimer handler
independently track larger periods. We'll look at an example of this approach in the next section.

---

## Page 838

Part 5. Creating application programs
838
5.6 Working with timer
MQL5 also allows to create timers with a period of less than 1  second: there is a function for this,
EventSetMillisecondTimer.
5.6.2 Timer event: OnTimer
The OnTimer event is one of the standard events supported by MQL5 programs (see section Overview of
event handling functions). To receive timer events in the program code, you should describe a function
with the following prototype.
void OnTimer(void)
The OnTimer event is periodically generated by the client terminal for an Expert Advisor or an indicator
that has activated the timer using the EventSetTimer or EventSetMillisecondTimer functions (see the
next section).
Attention! In dependent indicators created by calling iCustom or IndicatorCreate from other
programs, the timer does not work, and the OnTimer event is not generated. This is an architectural
limitation of MetaTrader 5.
It should be understood that the presence of an enabled timer and OnTimer handler does not make the
MQL program multi-threaded. No more than one thread is allocated per MQL program (an indicator can
even share a thread with other indicators on the same symbol), so the call of OnTimer and other
handlers always happen sequentially, in agreement with the event queue. If one of the handlers,
including  OnTimer, will start lengthy calculations, this will suspend the execution of all other events and
sections of the program code.
If you need to organize parallel data processing, you should run several MQL programs simultaneously
(perhaps, instances of the same program on different charts or chart objects) and exchange commands
and data between them using their own protocol, for example, using custom events.
As an example, let's create classes that can organize several logical timers in one program. The periods
of all logical timers will be set as a multiplier of the base period, that is, the period of a single hardware
timer supplying events to the standard handler OnTimer. In this handler, we must call a certain method
of our new MultiTimer class which will manage all logical timers.
void OnTimer()
{
   // call the MultiTimer method to check and call dependent timers when needed
   MultiTimer::onTimer();
}
Class MultiTimer and related classes of individual timers will be combined in one file, MultiTimer.mqh.
The base class for working timers will be TimerNotification. Strictly speaking, this could be an interface,
but it is convenient to output some details of the general implementation into it: in particular, store the
reading of the counter chronometer, using which we will ensure that the timer fires with a certain
multiplier of the relative period of the main timer, as well as a method for checking the moment when
the timer should fire isTimeCome. That's why TimerNotification is an abstract class. It lacks
implementations of two virtual methods: notify - for actions when the timer fires - and getInterval to
obtain a multiplier that determines the period of a particular timer relative to the period of the main
timer.

---

## Page 839

Part 5. Creating application programs
839
5.6 Working with timer
class TimerNotification
{
protected:
   int chronometer; // counter of timer checks (isTimeCome calls)
public:
   TimerNotification(): chronometer(0)
   {
   }
   
   // timer work event
   // pure virtual method, it is required to be described in the heirs
   virtual void notify() = 0;
   // returns the period of the timer (it can be changed on the go)
   // pure virtual method, it is required to be described in the heirs
   virtual int getInterval() = 0;
   // check if it's time for the timer to fire, and if so, call notify
   virtual bool isTimeCome()
   {
      if(chronometer >= getInterval() - 1)
      {
         chronometer = 0; // reset the counter
         notify();        // notify application code
         return true;
      }
      
      ++chronometer;
      return false;
   }
};
All logic is provided in the isTimeCome method. Each time it is called, the chronometer counter is
incremented, and if it reaches the last iteration according to the getInterval method, the notify method
is called to notify the application code.
For example, if the main timer is started with a period of 1  second (EventSetTimer(1 )), then the child
object TimerNotification, which will return 5 from getInterval, will receive calls to its notify method
every 5 seconds.
As we have already said, such timer objects will be managed by the MultiTimer manager object. We
need only one such object. Therefore, its constructor is declared protected, and a single instance is
created statically within the class.

---

## Page 840

Part 5. Creating application programs
840
5.6 Working with timer
class MultiTimer
{
protected:
   static MultiTimer _mainTimer;
   
   MultiTimer()
   {
   }
   ...
Inside this class, we organize the storage of the TimerNotification array of objects (we will see how it is
filled in a few paragraphs). Once we have the array, we can easily write the checkTimers method which
loops through all logical timers. For external access, this method is duplicated by the public static
method onTimer, which we have already seen in the global OnTimer handler. Since the only manager
instance is created statically, we can access it from a static method.
   ...
   TimerNotification *subscribers[];
   
   void checkTimers()
   {
      int n = ArraySize(subscribers);
      for(int i = 0; i < n; ++i)
      {
         if(CheckPointer(subscribers[i]) != POINTER_INVALID)
         {
            subscribers[i].isTimeCome();
         }
      }
   }
   
public:
   static void onTimer()
   {
      _mainTimer.checkTimers();
   }
   ...
The TimerNotification object is added into the subscribers array using the bind method.

---

## Page 841

Part 5. Creating application programs
841 
5.6 Working with timer
   void bind(TimerNotification &tn)
   {
      int i, n = ArraySize(subscribers);
      for(i = 0; i < n; ++i)
      {
         if(subscribers[i] == &tn) return; // there is already such an object
         if(subscribers[i] == NULL) break; // found an empty slot
      }
      if(i == n)
      {
         ArrayResize(subscribers, n + 1);
      }
      else
      {
         n = i;
      }
      subscribers[n] = &tn;
   }
The method is protected from repeated addition of the object, and, if possible, the pointer is placed in
an empty element of the array, if there is one, which eliminates the need to expand the array. Empty
elements in an array may appear if any of the TimerNotification objects was removed using the unbind
method (timers can be used occasionally).
   void unbind(TimerNotification &tn)
   {
      const int n = ArraySize(subscribers);
      for(int i = 0; i < n; ++i)
      {
         if(subscribers[i] == &tn)
         {
            subscribers[i] = NULL;
            return;
         }
      }
   }
Note that the manager does not take ownership of the timer object and does not attempt to call
delete. If you are going to register dynamically allocated timer objects in the manager, you can add the
following code inside if before zeroing:
            if(CheckPointer(subscribers[i]) == POINTER_DYNAMIC) delete subscribers[i];
Now it remains to understand how we can conveniently organize bind/unbind calls, so as not to load the
application code with these utilitarian operations. If you do it "manually", then it's easy to forget to
create or, on the contrary, delete the timer somewhere.
Let's develop the SingleTimer class derived from TimerNotification, in which we implement bind and
unbind calls from the constructor and destructor, respectively. In addition, we describe in it the
multiplier variable to store the timer period.

---

## Page 842

Part 5. Creating application programs
842
5.6 Working with timer
   class SingleTimer: public TimerNotification
   {
   protected:
      int multiplier;
      MultiTimer *owner;
   
   public:
      // creating a timer with the specified base period multiplier, optionally paused
      // automatically register the object in the manager
      SingleTimer(const int m, const bool paused = false): multiplier(m)
      {
         owner = &MultiTimer::_mainTimer;
         if(!paused) owner.bind(this);
      }
   
      // automatically disconnect the object from the manager
      ~SingleTimer()
      {
         owner.unbind(this);
      }
   
      // return timer period
      virtual int getInterval() override 
      {
         return multiplier;
      }
   
      // pause this timer
      virtual void stop()
      {
         owner.unbind(this);
      }
   
      // resume this timer
      virtual void start()
      {
         owner.bind(this);
      }
   };
The second parameter of the constructor (paused) allows you to create an object, but not start the
timer immediately. Such a delayed timer can then be activated using the start method.
The scheme of subscribing some objects to events in others is one of the popular design patterns in
OOP and is called "publisher/subscriber".
It is important to note that this class is also abstract because it does not implement the notify method.
Based on SingleTimer, let's describe the classes of timers with additional functionality.
Let's start with the class CountableTimer. It allows you to specify how many times it should trigger,
after which it will be automatically stopped. With it, in particular, it is easy to organize a single delayed
action. The CountableTimer constructor has parameters for setting the timer period, the pause flag,

---

## Page 843

Part 5. Creating application programs
843
5.6 Working with timer
and the number of retries. By default, the number of repetitions is not limited, so this class will become
the basis for most application timers.
class CountableTimer: public MultiTimer::SingleTimer
{
protected:
   const uint repeat;
   uint count;
   
public:
   CountableTimer(const int m, const uint r = UINT_MAX, const bool paused = false):
      SingleTimer(m, paused), repeat(r), count(0) { }
   
   virtual bool isTimeCome() override
   {
      if(count >= repeat && repeat != UINT_MAX)
      {
         stop();
         return false;
      }
      // delegate the time check to the parent class,
      // increment our counter only if the timer fired (returned true)
      return SingleTimer::isTimeCome() && (bool)++count;
   }
   // reset our counter on stop
   virtual void stop() override
   {
      SingleTimer::stop();
      count = 0;
   }
   uint getCount() const
   {
      return count;
   }
   
   uint getRepeat() const
   {
      return repeat;
   }
};
In order to use CountableTimer, we have to describe the derived class in our program as follows.

---

## Page 844

Part 5. Creating application programs
844
5.6 Working with timer
// MultipleTimers.mq5 
class MyCountableTimer: public CountableTimer
{
public:
   MyCountableTimer(const int s, const uint r = UINT_MAX):
      CountableTimer(s, r) { }
   
   virtual void notify() override
   {
      Print(__FUNCSIG__, multiplier, " ", count);
   }
};
In this implementation of the notify method, we just log the timer period and the number of times it
triggered. By the way, this is a fragment of the MultipleTimers.mq5 indicator, which we will use as a
working example.
Let's call the second class derived from SingleTimer FunctionalTimer. Its purpose is to provide a simple
timer implementation for those who like the functional style of programming and don't feel like writing
derived classes. The constructor of the FunctionalTimer class will take, in addition to the period, a
pointer to a function of a special type, TimerHandler.
// MultiTimer.mqh
typedef bool (*TimerHandler)(void);
   
class FunctionalTimer: public MultiTimer::SingleTimer
{
   TimerHandler func;
public:
   FunctionalTimer(const int m, TimerHandler f):
      SingleTimer(m), func(f) { }
      
   virtual void notify() override
   {
      if(func != NULL)
      {
         if(!func())
         {
            stop();
         }
      }
   }
};
In this implementation of the notify method, the object calls the function by the pointer. With such a
class, we can define a macro that, when placed before a block of statements in curly brackets, will
"make" it the body of the timer function.

---

## Page 845

Part 5. Creating application programs
845
5.6 Working with timer
// MultiTimer.mqh
#define OnTimerCustom(P) OnTimer##P(); \
FunctionalTimer ft##P(P, OnTimer##P); \
bool OnTimer##P()
Then in the application code you can write like this:
// MultipleTimers.mq5
bool OnTimerCustom(3)
{
   Print(__FUNCSIG__);
   return true;        // continue the timer
}
This construct declares a timer with a period of 3 and a set of instructions inside parentheses (here,
just printing to a log). If this function returns false, this timer will be stopped.
Let's consider the indicator MultipleTimers.mq5 more. Since it does not provide visualization, we will
specify the number of diagrams equal to zero.
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0
To use the classes of logical timers, we include the header file MultiTimer.mqh and add an input variable
for the base (global) timer period.
#include <MQL5Book/MultiTimer.mqh>
   
input int BaseTimerPeriod = 1;
The base timer is started in OnInit.
void OnInit()
{
   Print(__FUNCSIG__, " ", BaseTimerPeriod, " Seconds");
   EventSetTimer(BaseTimerPeriod);
}
Recall that the operation of all logical timers is ensured by the interception of the global OnTimer event.
void OnTimer()
{
   MultiTimer::onTimer();
}
In addition to the timer application class MyCountableTimer above, let's describe another class of the
suspended timer MySuspendedTimer.

---

## Page 846

Part 5. Creating application programs
846
5.6 Working with timer
class MySuspendedTimer: public CountableTimer
{
public:
   MySuspendedTimer(const int s, const uint r = UINT_MAX):
      CountableTimer(s, r, true) { }
   virtual void notify() override
   {
      Print(__FUNCSIG__, multiplier, " ", count);
      if(count == repeat - 1) // execute last time
      {
         Print("Forcing all timers to stop");
         EventKillTimer();
      }
   }
};
A little lower we will see how it starts. It is also important to note here that after reaching the specified
number of operations, this timer will turn off all timers by calling EventKillTimer.
Now let's show how (in the global context) the objects of different timers of these two classes are
described.
MySuspendedTimer st(1, 5);
MyCountableTimer t1(2);
MyCountableTimer t2(4);
The st timer of the MySuspendedTimer class has period 1  (1 *BaseTimerPeriod) and should stop after 5
operations.
The t1  and t2 timers of the MyCountableTimer class have periods 2 (2 * BaseTimerPeriod) and 4 (4 *
BaseTimerPeriod), respectively. With default value BaseTimerPeriod = 1  all periods represent seconds.
These two timers are started immediately after the start of the program.
We will also create two timers in a functional style.
bool OnTimerCustom(5)
{
   Print(__FUNCSIG__);
   st.start();         // start delayed timer
   return false;       // and stop this timer object
}
   
bool OnTimerCustom(3)
{
   Print(__FUNCSIG__);
   return true;        // this timer keeps running
}
Please note that OnTimerCustom5 has only one task: 5 periods after the start of the program, it needs
to start a delayed timer st and terminate its own execution. Considering that the delayed timer should
deactivate all timers after 5 periods, we get 1 0 seconds of program activity at default settings.
The OnTimerCustom3 timer should trigger three times during this period.

---

## Page 847

Part 5. Creating application programs
847
5.6 Working with timer
So, we have 5 timers with different periods: 1 , 2, 3, 4, 5 seconds.
Let's analyze an example of what is being output to the log (time stamps are schematically shown on
the right).
                                                // time
17:08:45.174  void OnInit() 1 Seconds             |
17:08:47.202  void MyCountableTimer::notify()2 0    |
17:08:48.216  bool OnTimer3()                        |
17:08:49.230  void MyCountableTimer::notify()2 1      |
17:08:49.230  void MyCountableTimer::notify()4 0      |
17:08:50.244  bool OnTimer5()                          |
17:08:51.258  void MyCountableTimer::notify()2 2        |
17:08:51.258  bool OnTimer3()                           |
17:08:51.258  void MySuspendedTimer::notify()1 0        |
17:08:52.272  void MySuspendedTimer::notify()1 1         |
17:08:53.286  void MyCountableTimer::notify()2 3          |
17:08:53.286  void MyCountableTimer::notify()4 1          |
17:08:53.286  void MySuspendedTimer::notify()1 2          |
17:08:54.300  bool OnTimer3()                              |
17:08:54.300  void MySuspendedTimer::notify()1 3           |
17:08:55.314  void MyCountableTimer::notify()2 4            |
17:08:55.314  void MySuspendedTimer::notify()1 4            |
17:08:55.314  Forcing all timers to stop                    |
The first message from the two-second timer arrives, as expected, about 2 seconds after the start (we
are saying "about" because the hardware timer has a limitation in accuracy and, in addition, other
computer load affects the execution). One second later, the three-second timer triggers for the first
time. The second hit of the two-second timer coincides with the first output from the four-second
timer. After a single execution of the five-second timer, messages from the one-second timer begin to
appear in the log regularly (its counter increases from 0 to 4). On its last iteration, it stops all timers.
5.6.3 High-precision timer: EventSetMillisecondTimer
If your program requires the timer to trigger more frequently than 1  second, instead of EventSetTimer
use the EventSetMillisecondTimer function.
Timers with different units cannot be started at the same time: either one function or the other must
be used. The type of timer actually running is determined by which function was called later. All
features inherent to standard timer remain valid for the high-precision timer.
bool EventSetMillisecondTimer(int milliseconds)
The function indicates to the client terminal that it is necessary to generate timer events for this
Expert Advisor or indicator with a frequency of less than one second. The periodicity is set in
milliseconds (parameter milliseconds).
The function returns a sign of success (true) or error (false).
When working in the strategy tester, keep in mind that the shorter the timer period, the longer the
testing will take, as the number of calls to the timer event handler increases.
During normal operation, timer events are generated no more than once every 1 0-1 6 milliseconds,
which is due to hardware limitations.

---

## Page 848

Part 5. Creating application programs
848
5.6 Working with timer
To demonstrate how to work with the millisecond timer, let's expand the indicator example
MultipleTimers.mq5. Since the activation of the global timer is left to the application program, we can
easily change the type of the timer, leaving the logical timer classes unchanged. The only difference
will be that their multipliers will be applied to the base period in milliseconds that we will specify in the
EventSetMillisecondTimer function.
To select the timer type, we will describe the enumeration and add a new input variable.
enum TIMER_TYPE
{
   Seconds,
   Milliseconds
};
   
input TIMER_TYPE TimerType = Seconds;
By default, we use a second timer. In OnInit, start the timer of the required type.
void OnInit()
{
   Print(__FUNCSIG__, " ", BaseTimerPeriod, " ", EnumToString(TimerType));
   if(TimerType == Seconds)
   {
      EventSetTimer(BaseTimerPeriod);
   }
   else
   {
      EventSetMillisecondTimer(BaseTimerPeriod);
   }
}
Let's see what will be displayed in the log when choosing a millisecond timer.

---

## Page 849

Part 5. Creating application programs
849
5.6 Working with timer
                                               // time ms
17:27:54.483  void OnInit() 1 Milliseconds        |             
17:27:54.514  void MyCountableTimer::notify()2 0    |           +31
17:27:54.545  bool OnTimer3()                        |          +31
17:27:54.561  void MyCountableTimer::notify()2 1      |         +16
17:27:54.561  void MyCountableTimer::notify()4 0      |
17:27:54.577  bool OnTimer5()                          |        +16
17:27:54.608  void MyCountableTimer::notify()2 2        |       +31
17:27:54.608  bool OnTimer3()                           |
17:27:54.608  void MySuspendedTimer::notify()1 0        |
17:27:54.623  void MySuspendedTimer::notify()1 1         |      +15
17:27:54.655  void MyCountableTimer::notify()2 3          |     +32
17:27:54.655  void MyCountableTimer::notify()4 1          |
17:27:54.655  void MySuspendedTimer::notify()1 2          |
17:27:54.670  bool OnTimer3()                              |    +15
17:27:54.670  void MySuspendedTimer::notify()1 3           |
17:27:54.686  void MyCountableTimer::notify()2 4            |   +16
17:27:54.686  void MySuspendedTimer::notify()1 4            |
17:27:54.686  Forcing all timers to stop                    |
The sequence of event generation is exactly the same as what we saw for the second timer, but
everything happens much faster, almost instantly.
Due to the fact that the accuracy of the system timer is limited to a couple of tens of milliseconds, the
real interval between events significantly exceeds the unattainable small 1  millisecond. In addition,
there is a spread of the size of one "step". Thus, even when using a millisecond timer, it is desirable not
to stick to periods less than a few tens of milliseconds.
5.7 Working with charts
Most MQL programs, such as scripts, indicators, and Expert Advisors, are executed on charts. Only
services run in the background, without being tied to a schedule. A rich set of functions is provided for
obtaining and changing the properties of graphs, analyzing their list, and searching for other running
programs.
Since charts are the natural environment for indicators, we have already had a chance to get
acquainted with some of these features in the previous indicator chapters. In this chapter, we will study
all these functions in a targeted manner.
When working with charts, we will use the concept of a window. A window is a dedicated area that
displays price charts and/or indicator charts. The top and, as a rule, the largest window contains price
charts, has the number 0, and always exists. All additional windows added to the lower part when
placing indicators are numbered from 1  and higher (numbering from top to bottom). Each subwindow
exists only as long as it has at least one indicator.
Since the user can delete all indicators in an arbitrary subwindow, including the one that is not the last
(the lowest), the indexes of the remaining subwindows can decrease.
The event model of charts related to receiving and processing notifications about events on charts and
generating custom events will be discussed in a separate chapter.

---

## Page 850

Part 5. Creating application programs
850
5.7 Working with charts
In addition to the "charts in windows" discussed here, MetaTrader 5 also allows you to create "charts
in objects". We will deal with graphical objects in the next chapter.
5.7.1  Functions for getting the basic properties of the current chart
In many examples in the book, we have already had to use Predefined Variables, containing the main
properties of the chart and its working symbol. MQL programs also have access to functions that return
the values of some of these variables. It does not matter what is used, a variable or a function, and
thus you can use your preferred source code styles.
Each chart is characterized by a working symbol and timeframe. They can be found using the Symbol
and Period functions, respectively. In addition, MQL5 provides simplified access to the two most
commonly used symbol properties: price point size (Point) and the associated number of significant
digits (Digits) after the decimal point in the price.
string Symbol()
The Symbol function returns the symbol name of the current chart, i.e. the value of the system variable
_ Symbol. To get the symbol of an arbitrary chart, there is the ChartSymbol function which operates
based on the chart identifier. We will discuss the methods for obtaining chart identifiers a little later.
ENUM_TIMEFRAMES Period()
The Period function returns the timeframe value (ENUM_TIMEFRAMES) of the current chart, which
corresponds to the _ Period variable. To get the timeframe of an arbitrary chart, use the function
ChartPeriod, and it also needs an identifier as a parameter.
double Point()
The Point function returns the point size of the current instrument in the quote currency, which is the
same as the value of the _ Point variable.
int Digits()
The function returns the number of decimal places after the decimal point, which determines the
accuracy of measuring the price of the symbol of the current chart, which is equivalent to the variable
_ Digits.
Other properties of the current tool allow you to get SymbolInfo-functions, which in a more general
case provide an analysis of all instruments.
The following simple example of the script ChartMainProperties.mq5 logs the properties described in this
section.

---

## Page 851

Part 5. Creating application programs
851 
5.7 Working with charts
void OnStart()
{
   PRTF(_Symbol);
   PRTF(Symbol());
   PRTF(_Period);
   PRTF(Period());
   PRTF(_Point);
   PRTF(Point());
   PRTF(_Digits);
   PRTF(Digits());
   PRTF(DoubleToString(_Point, _Digits));
   PRTF(EnumToString(_Period));
}
For the EURUSD,H1  chart, we will get the following log entries.
_Symbol=EURUSD / ok
Symbol()=EURUSD / ok
_Period=16385 / ok
Period()=16385 / ok
_Point=1e-05 / ok
Point()=1e-05 / ok
_Digits=5 / ok
Digits()=5 / ok
DoubleToString(_Point,_Digits)=0.00001 / ok
EnumToString(_Period)=PERIOD_H1 / ok
5.7.2 Chart identification
Each chart in MetaTrader 5 operates in a separate window and has a unique identifier. For
programmers familiar with the principles of Windows operation, we would like to clarify that this
identifier is not a system window handle (although the MQL5 API allows you to get the latter through
the property CHART_WINDOW_HANDLE). As we know, in addition to the main working area of the chart
with quotes, additional areas (subwindows) with indicators that have the property
indicator_ separate_ window. All subwindows are part of the chart and belong to the same Windows
window.
long ChartID()
The function returns a unique identifier for the current chart.
Many of the functions that we'll look at require a chart ID as a parameter, but you can specify 0 for the
current chart instead of calling ChartID. It makes sense to use ChartID in cases where the identifier is
sent between MQL programs, for example, when exchanging messages (custom events) on the same
chart, or on different ones. Specifying an invalid ID will result in the ERR_CHART_WRONG_ID (41 01 )
error.
The chart ID generally stays the same from session to session.
We will demonstrate the function ChartID and what the identifiers look like in the example script
ChartList1 .mq5 after studying the method for obtaining a chart list.

---

## Page 852

Part 5. Creating application programs
852
5.7 Working with charts
5.7.3 Getting the list of charts
An MQL program can get a list of charts opened in the terminal (both windows and graph objects) using
the functions ChartFirst and ChartNext.
long ChartFirst()
long ChartNext(long chartId)
The ChartFirst function returns the identifier of the first chart in the client terminal. MetaTrader 5
maintains an internal list of all charts, the order in which may differ from what we see on the screen,
for example, in window tabs when they are maximized. In particular, the order in the list can change as
a result of dragging tabs, undocking, and docking windows. After loading the terminal, the visible order
of the bookmarks is the same as the internal list view.
The ChartNext function returns the ID of the chart following the chart with the specified chartId.
Unlike other functions for working with graphs, the value 0 in the ChartId parameter means not the
current chart, but the beginning of the list. In other words, ChartNext(0) call is equivalent to ChartFirst.
If the end of the list is reached, the function returns -1 .
The script ChartList1 .mq5 outputs the list of charts into the log. The main work is performed by the
ChartList function which is called from OnStart. At the very beginning of the function, we get the
identifier of the current chart using ChartID and then we mark it with an asterisk in the list. At the end,
the total number of charts is output.
void OnStart()
{
   ChartList();
}
   
void ChartList()
{
   const long me = ChartID();
   long id = ChartFirst();
   // long id = ChartNext(0); - analogue of calling ChartFirst()
   int count = 0, used = 0;
   Print("Chart List\nN, ID, *active");
   // keep iterating over charts until there are none left
   while(id != -1)
   {
      const string header = StringFormat("%d %lld %s",
         count, id, (id == me ? " *" : ""));
    
      // fields: N, id, label of the current chart
      Print(header);
      count++;
      id = ChartNext(id);
   }
   Print("Total chart number: ", count);
}
An example result is shown below.

---

## Page 853

Part 5. Creating application programs
853
5.7 Working with charts
Chart List
N, ID, *active
0 132358585987782873 
1 132360375330772909  *
2 132544239145024745 
3 132544239145024732 
4 132544239145024744 
Total chart number: 5
5.7.4 Getting the symbol and timeframe of an arbitrary chart
Two fundamental properties of any chart are its working symbol and timeframe. As we saw earlier,
these properties for the current chart are available as built-in variables _ Symbol and _ Period, as well as
through the relevant functions Symbol and Period. The following functions can be used to determine the
same properties for other charts: ChartSymbol and ChartPeriod.
string ChartSymbol(long chartId = 0)
The function returns the name of the symbol of the chart with the specified identifier. If the parameter
is 0, the current chart is assumed.
If the chart does not exist, an empty string ("") is returned and _ LastError sets error code
ERR_CHART_WRONG_ID (41 01 ).
ENUM_TIMEFRAMES ChartPeriod(long chartId = 0)
The function returns the period value for the chart with the specified identifier.
If the chart does not exist, 0 is returned.
The script ChartList2.mq5, similar to ChartList1 .mq5, generates a list of charts indicating the symbol
and timeframe.

---

## Page 854

Part 5. Creating application programs
854
5.7 Working with charts
#include <MQL5Book/Periods.mqh>
   
void OnStart()
{
   ChartList();
}
   
void ChartList()
{
   const long me = ChartID();
   long id = ChartFirst();
   int count = 0;
   
   Print("Chart List\nN, ID, Symbol, TF, *active");
   // keep iterating over charts until there are none left
   while(id != -1)
   {
      const string header = StringFormat("%d %lld %s %s %s",
         count, id, ChartSymbol(id), PeriodToString(ChartPeriod(id)),
         (id == me ? " *" : ""));
    
      // fields: N, id, symbol, timeframe, label of the current chart
      Print(header);
      count++;
      id = ChartNext(id);
   }
   Print("Total chart number: ", count);
}
Here is an example of the log content after running the script on the EURUSD, H1  chart (on the second
line).
Chart List
N, ID, Symbol, TF, *active
0 132358585987782873 EURUSD M15 
1 132360375330772909 EURUSD H1  *
2 132544239145024745 XAUUSD H1 
3 132544239145024732 USDRUB D1 
4 132544239145024744 EURUSD H1 
Total chart number: 5
MQL5 allows not only to identify but also to switch the symbol and timeframe of any chart.
5.7.5 Overview of functions for working with the complete set of chart properties
Chart properties are readable and editable via groups ChartSet- and ChartGet-functions, each of which
contains properties of a certain type: real numbers (double), whole numbers (long, int, datetime, color,
bool, enums), and strings.
All functions receive the chart ID as the first parameter. The value 0 means the current chart, that is,
it is equivalent to passing the result of the call ChartID(). However, this does not mean that the ID of
the current chart is 0.

---

## Page 855

Part 5. Creating application programs
855
5.7 Working with charts
The constants describing all properties form three enumerations ENUM_CHART_PROPERTY_INTEGER,
ENUM_CHART_PROPERTY_DOUBLE, ENUM_CHART_PROPERTY_STRING, which are used as function
parameters for the corresponding type. A summary table of all properties can be found in the MQL5
documentation, on the page about chart properties. In the following sections of this chapter, we will
gradually cover virtually all the properties, grouping them according to their purpose. The only
exception is the properties of managing events on the chart - we will describe them in the relevant
section of the chapter on events.
The elements of all three enumerations are assigned such values that they form a single list without
intersections (repetitions). This allows you to determine the type of enumeration by a specific value.
For example, given a constant, we can consistently try to convert it to a string with the name of one of
the enums until we succeed.
int value = ...;
   
ResetLastError(); // clear the error code if there was one
EnumToString((ENUM_CHART_PROPERTY_INTEGER)value); // resulting string is not important
if(_LastError == 0) // analyze if there is a new error
{
   // success is an element of ENUM_CHART_PROPERTY_INTEGER
   return ChartGetInteger(0, (ENUM_CHART_PROPERTY_INTEGER)value);
}
   
ResetLastError();
EnumToString((ENUM_CHART_PROPERTY_DOUBLE)value);
if(_LastError == 0)
{
   // success is an ENUM_CHART_PROPERTY_DOUBLE element
   return ChartGetDouble(0, (ENUM_CHART_PROPERTY_DOUBLE)value);
}
   
... // continue a similar check for ENUM_CHART_PROPERTY_STRING
Later we will use this approach in test scripts.
Some properties (for example, the number of visible bars) are read-only and cannot be changed. They
will be further marked "r/o" (read-only).
Property read functions have a short form and a long form: the short form directly returns the
requested value, and the long form returns a boolean attribute of success (true) or error (false), while
the value itself is placed in the last parameter passed by reference. When using the short form, it is
especially important to check the error code in the _ LastError variable, because the value 0 (NULL)
returned in case of problems may be generally correct.
When accessing some properties, you must specify an additional parameter window, which is used to
indicate the chart window/subwindow. 0 means the main window. Subwindows are numbered starting
from 1 . Some properties apply to the chart as a whole and thus they have function variants without the
window parameter.
Following are the function prototypes for reading and writing integer properties. Please note that the
type of values in them is long.

---

## Page 856

Part 5. Creating application programs
856
5.7 Working with charts
b o o l  C h a r tSetIn teg er ( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_IN TE GE R  p r o p er ty, l o n g  v a l u e)
b o o l  C h a r tSetIn teg er ( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_IN TE GE R  p r o p er ty, i n t w i n d o w , l o n g  v a l u e)
l o n g  C h a r tGetIn teg er ( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_IN TE GE R  p r o p er ty, i n t w i n d o w  =  0 )
b o o l  C h a r tGetIn teg er ( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_IN TE GE R  p r o p er ty, i n t w i n d o w , l o n g  & v a l u e)
Functions for real properties are described similarly. There are no writable real properties for
subwindows, so there is only one form of ChartSetDouble, which is without the window parameter.
b o o l  C h a r tSetD o u b l e( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_D OU B L E  p r o p er ty, d o u b l e v a l u e)
d o u b l e C h a r tGetD o u b l e( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_D OU B L E  p r o p er ty, i n t w i n d o w  =  0 )
b o o l  C h a r tGetD o u b l e( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_D OU B L E  p r o p er ty, i n t w i n d o w , d o u b l e & v a l u e)
The same applies to string properties, but one more nuance should be taken into account: the length of
the string cannot exceed 2045 characters (extra characters will be cut off).
b o o l  C h a r tSetStr i n g ( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_STR IN G p r o p er ty, s tr i n g  v a l u e)
s tr i n g  C h a r tGetStr i n g ( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_STR IN G p r o p er ty) 
b o o l  C h a r tGetStr i n g ( l o n g  ch a r tId , E N U M _C H AR T_PR OPE R TY_STR IN G p r o p er ty, s tr i n g  & v a l u e) 
When reading properties using the short form of ChartGetInteger/ChartGetDouble, the window
parameter is optional and defaults to the main window (window=0).
Functions for setting chart properties (ChartSetInteger, ChartSetDouble, ChartSetString) are
asynchronous and serve to send change commands to the chart. If these functions are successfully
executed, the command is added to the common queue of chart events, and true is returned. When an
error occurs, the function returns false. In this case, you should check the error code in the _ LastError
variable.
Chart properties are changed later, during the processing of the event queue of this chart, and, as a
rule, with some delay, so you should not expect an immediate update of the chart after applying new
settings. To force the update of the appearance and properties of the chart, use the function
ChartRedraw. If you want to change several chart properties at once, then you need to call the
corresponding functions in one code block and then once in ChartRedraw.
In general, the chart is updated automatically by the terminal in response to events such as the arrival
of a new quote, changes in the chart window size, scaling, scrolling, adding an indicator, etc.
Functions for getting chart properties (ChartGetInteger, ChartGetDouble, ChartGetString) are
synchronous, that is, the calling code waits for the result of their execution.
5.7.6 Descriptive chart properties
The ChartSetString/ChartGetString functions enable the reading and setting of the following string
properties of the charts.
Identifier
Description
CHART_COMMENT
Chart comment text
CHART_EXPERT_NAME
Name of the Expert Advisor running on the chart (r/o)
CHART_SCRIPT_NAME
Name of the script running on the chart (r/o)

---

## Page 857

Part 5. Creating application programs
857
5.7 Working with charts
In chapter Displaying messages in the chart window, we learned about the Comment function which
displays a text message in the upper left corner of the chart. The CHART_COMMENT property allows
you to read the current chart comment: ChartGetString(0, CHART_ COMMENT). It is also possible to
access comments on other charts by passing their identifiers to the function. By using ChartSetString,
you can change comments on the current and other charts, if you know their ID: ChartSetString(ID,
CHART_ COMMENT, "text").
If an Expert Advisor or/and a script is running in any chart, we can find out their names using these
calls: ChartGetString(ID, CHART_ EXPERT_ NAME) and ChartGetString(ID, CHART_ SCRIPT_ NAME).
The script ChartList3.mq5, similar to ChartList2.mq5, supplements the list of charts with information
about Expert Advisors and scripts. Later we will add to it information about indicators.
void ChartList()
{
   const long me = ChartID();
   long id = ChartFirst();
   int count = 0, used = 0, temp, experts = 0, scripts = 0;
   Print("Chart List\nN, ID, Symbol, TF, *active");
   // keep iterating over charts until there are none left
   while(id != -1)
   {
      temp =0;// sign of MQL programs on this chart
      const string header = StringFormat("%d %lld %s %s %s",
         count, id, ChartSymbol(id), PeriodToString(ChartPeriod(id)),
         (id == me ? " *" : ""));
      // fields: N, id, symbol, timeframe, label of the current chart
      Print(header);
      string expert = ChartGetString(id, CHART_EXPERT_NAME);
      string script = ChartGetString(id, CHART_SCRIPT_NAME);
      if(StringLen(expert) > 0) expert = "[E] " + expert;
      if(StringLen(script) > 0) script = "[S] " + script;
      if(expert != NULL || script != NULL)
      {
         Print(expert, " ", script);
         if(expert != NULL) experts++;
         if(script != NULL) scripts++;
         temp++;
      }
      count++;
      if(temp > 0)
      {
         used++;
      }
      id = ChartNext(id);
   }
   Print("Total chart number: ", count, ", with MQL-programs: ", used);
   Print("Experts: ", experts, ", Scripts: ", scripts);
}
This is an example of the output of this script.

---

## Page 858

Part 5. Creating application programs
858
5.7 Working with charts
Chart List
N, ID, Symbol, TF, *active
0 132358585987782873 EURUSD M15 
1 132360375330772909 EURUSD H1  *
 [S] ChartList3
2 132544239145024745 XAUUSD H1 
3 132544239145024732 USDRUB D1 
4 132544239145024744 EURUSD H1 
Total chart number: 5, with MQL-programs: 1
Experts: 0, Scripts: 1
Here you can see that only one script is being executed.
5.7.7 Checking the status of the main window
The pair of functions ChartSetInteger/ChartGetInteger allows you to find out some of the chart state
characteristics, as well as change some of them.
Identifier
Description
Value type
CHART_BRING_TO_TOP
Chart activity (input focus) on top of all others
bool
CHART_IS_MAXIMIZED
Chart maximized
bool
CHART_IS_MINIMIZED
Chart minimized
bool
CH AR T_W IN D O W _H AN D L E 
Windows-handle of the chart  window (r/o)
int
CHART_IS_OBJECT
A flag that a chart is a Chart object (OBJ_CHART);
true is for a graphic object and false is for a
normal chart (r/o)
bool
As expected, the Window handle and the attribute of the chart object are read-only. Other properties
are editable: for example, by calling ChartSetInteger(ID, CHART_ BRING_ TO_ TOP, true), you activate the
chart with the specified ID.
An example of applying properties is given in the ChartList4.mq5 script in the next section.
5.7.8 Getting the number and visibility of windows/subwindows
Using the ChartGetInteger function, an MQL program can find the number of windows on a chart
(including subwindows), as well as their visibility.

---

## Page 859

Part 5. Creating application programs
859
5.7 Working with charts
Identifier
Description
Value type
CHART_WINDOWS_TOTAL
Total number of chart windows,
including indicator subwindows
(r/o)
int
CHART_WINDOW_IS_VISIBLE
Subwindow visibility, the
'window' parameter is the
subwindow number (r/o)
bool
Some subwindows can be hidden if the indicators placed in them are disabled on the current timeframe
in the Properties dialog, on the Visualization tab. It is impossible to reset all flags: due to the nature of
storage of tpl templates, such a state is interpreted as the enabling of all timeframes. Therefore, if the
user wants to hide the subwindow for some time, it is necessary to leave at least one enabled flag on
the most rarely used timeframe.
Setting indicator visibility on different timeframes
It should be noted that there are no standard tools in MQL5 for programmatic determination of the
state and switching of specific flags. The easiest way to simulate such control is to save the tpl
template and analyze it, with possible subsequent editing and loading (see section Working with tpl
templates).
In the new version of the script ChartList4.mq5, we output the number of subwindows (one window,
which is the main one, is always present), a sign of chart activity, a sign of a chart object, and a
Windows handle.

---

## Page 860

Part 5. Creating application programs
860
5.7 Working with charts
      const int win = (int)ChartGetInteger(id, CHART_WINDOWS_TOTAL);
      const string header = StringFormat("%d %lld %s %s %s %s %s %s %lld",
         count, id, ChartSymbol(id), PeriodToString(ChartPeriod(id)),
         (win > 1 ? "#" + (string)(win - 1) : ""), (id == me ? " *" : ""),
         (ChartGetInteger(id, CHART_BRING_TO_TOP, 0) ? "active" : ""),
         (ChartGetInteger(id, CHART_IS_OBJECT) ? "object" : ""),
         ChartGetInteger(id, CHART_WINDOW_HANDLE));
      ...
      for(int i = 0; i < win; i++)
      {
         const bool visible = ChartGetInteger(id, CHART_WINDOW_IS_VISIBLE, i);
         if(!visible)
         {
            Print("  ", i, "/Hidden");
         }
      }
Here's what the result might be.
Chart List
N, ID, Symbol, TF, #subwindows, *active, Windows handle
0 132358585987782873 EURUSD M15 #1    68030
1 132360375330772909 EURUSD H1  * active  68048
 [S] ChartList4
2 132544239145024745 XAUUSD H1     395756
3 132544239145024732 USDRUB D1     395768
4 132544239145024744 EURUSD H1 #2    461286
  2/Hidden
Total chart number: 5, with MQL-programs: 1
Experts: 0, Scripts: 1
On the first chart (index 0) there is one subwindow (#1 ). There are two subwindows (#2) on the last
chart, and the second one is currently hidden. Later, in the section Managing indicators on the chart,
we will present the full version of ChartList.mq5, where we include in the report information about the
indicators located in the subwindows and the main window.
Attention! A chart inside a chart object always has the CHART_WINDOW_IS_VISIBLE property
equal to true, even if object visualization is disabled on the current timeframe or on all timeframes.
5.7.9 Chart display modes
Four properties from the ENUM_CHART_PROPERTY_INTEGER enumeration describe chart display
modes. All these properties are available for reading through ChartGetInteger, and for recording
through ChartSetInteger, which allows you to change the appearance of the chart.

---

## Page 861

Part 5. Creating application programs
861 
5.7 Working with charts
Identifier
Description
Value type
CH AR T_M O D E 
Chart type (candles, bars, or line)
E N U M _CH AR T_M O D E 
CH AR T_F O R E G R O U N D 
Price chart in the foreground
bool
CH AR T_S H IF T
Price chart indent mode from the right edge
bool
CH AR T_AU TO S CR O L L 
Automatic scrolling to the right edge of the chart
bool
There is a special enumeration ENUM_CHART_MODE for the CHART_MODE mode in MQL5. Its elements
are shown in the following table.
Identifier
Description
Value
CHART_BARS
Display as bars
0
CHART_CANDLES
Display as Japanese candlesticks
1
CHART_LINE
Display as a line drawn at Close prices
2
Let's implement the script ChartMode.mq5, which will monitor the state of the modes and print
messages to the log when changes are detected. Since the property processing algorithms are of a
general nature, we will put them in a separate header file ChartModeMonitor.mqh, which we will then
connect to different tests.
Let's lay the foundation in an abstract class ChartModeMonitorInterface: it provides overloaded get-
and set- methods for all types. Derived classes will have to directly check the properties to the required
extent by overriding the virtual method snapshot.

---

## Page 862

Part 5. Creating application programs
862
5.7 Working with charts
class ChartModeMonitorInterface
{
public:
   long get(const ENUM_CHART_PROPERTY_INTEGER property, const int window = 0)
   {
      return ChartGetInteger(0, property, window);
   }
   double get(const ENUM_CHART_PROPERTY_DOUBLE property, const int window = 0)
   {
      return ChartGetDouble(0, property, window);
   }
   string get(const ENUM_CHART_PROPERTY_STRING property)
   {
      return ChartGetString(0, property);
   }
   bool set(const ENUM_CHART_PROPERTY_INTEGER property, const long value, const int window = 0)
   {
      return ChartSetInteger(0, property, window, value);
   }
   bool set(const ENUM_CHART_PROPERTY_DOUBLE property, const double value)
   {
      return ChartSetDouble(0, property, value);
   }
   bool set(const ENUM_CHART_PROPERTY_STRING property, const string value)
   {
      return ChartSetString(0, property, value);
   }
   
   virtual void snapshot() = 0;
   virtual void print() { };
   virtual void backup() { }
   virtual void restore() { }
};
The class also has reserved methods: print, for example, to output to a log, backup to save the current
state, and restore to recover it. They are declared not abstract, but with an empty implementation,
since they are optional.
It makes sense to define certain classes for properties of different types as a single template inherited
from ChartModeMonitorInterface and accepting parametric value (T) and enumeration (E) types. For
example, for integer properties, you would need to set T=long and
E=ENUM_ CHART_ PROPERTY_ INTEGER.
The object contains the data array to store [key,value] pairs with all requested properties. It has a
generic type MapArray<K,V>, which we introduced earlier for the indicator IndUnityPercent in the
chapter Multicurrency and multitimeframe indicators. Its peculiarity lies in the fact that in addition to
the usual access to array elements by numbers, addressing by key can be used.
To fill the array, an array of integers is passed to the constructor, while the integers are first checked
for compliance with the identifiers of the given enumeration E using the detect method. All correct
properties are immediately read through the get call, and the resulting values are stored in the map
along with their identifiers.

---

## Page 863

Part 5. Creating application programs
863
5.7 Working with charts
#include <MQL5Book/MapArray.mqh>
   
template<typename T,typename E>
class ChartModeMonitorBase: public ChartModeMonitorInterface
{
protected:
   MapArray<E,T> data; // array-map of pairs [property, value]
   
   // the method checks if the passed constant is an enumeration element,
   // and if it is, then add it to the map array
   bool detect(const int v)
   {
      ResetLastError();
      EnumToString((E)v); // resulting string is not used
      if(_LastError == 0) // it only matters if there is an error or not
      {
         data.put((E)v, get((E)v));
         return true;
      }
      return false;
   }
public:
   ChartModeMonitorBase(int &flags[])
   {
      for(int i = 0; i < ArraySize(flags); ++i)
      {
         detect(flags[i]);
      }
   }
   
   virtual void snapshot() override
   {
      MapArray<E,T> temp;
      // collect the current state of all properties
      for(int i = 0; i < data.getSize(); ++i)
      {
         temp.put(data.getKey(i), get(data.getKey(i)));
      }
      
      // compare with previous state, display differences
      for(int i = 0; i < data.getSize(); ++i)
      {
         if(data[i] != temp[i])
         {
            Print(EnumToString(data.getKey(i)), " ", data[i], " -> ", temp[i]);
         }
      }
      
      // save for next comparison
      data = temp;

---

## Page 864

Part 5. Creating application programs
864
5.7 Working with charts
   }
   ...
};
The snapshot method iterates through all the elements of the array and requests the value for each
property. Since we want to detect changes, the new data is first stored in a temporary map array
temp. Then arrays data and temp are compared element by element, and for each difference, a
message is displayed with the name of the property, its old and new value. This simplified example uses
only the journal. However, if necessary, the program can call some application functions that adapt the
behavior to the environment.
Methods print, backup, and restore are implemented as simply as possible.
template<typename T,typename E>
class ChartModeMonitorBase: public ChartModeMonitorInterface
{
protected:
   ...
   MapArray<E,T> store; // backup
public:
   ...
   virtual void print() override
   {
      data.print();
   }
   virtual void backup() override
   {
      store = data;
   }
   
   virtual void restore() override
   {
      data = store;
      // restore chart properties
      for(int i = 0; i < data.getSize(); ++i)
      {
         set(data.getKey(i), data[i]);
      }
   }
A combination of methods backup/restore allows you to save the state of the chart before starting
experiments with it, and after the completion of the test script, restore everything as it was.
Finally, the last class in the file ChartModeMonitor.mqh is ChartModeMonitor. It combines three
instances of ChartModeMonitorBase, created for the available combinations of property types. They
have an array of m pointers to the base interface ChartModeMonitorInterface. The class itself is also
derived from it.

---

## Page 865

Part 5. Creating application programs
865
5.7 Working with charts
#include <MQL5Book/AutoPtr.mqh>
   
#define CALL_ALL(A,M) for(int i = 0, size = ArraySize(A); i < size; ++i) A[i][].M
   
class ChartModeMonitor: public ChartModeMonitorInterface
{
   AutoPtr<ChartModeMonitorInterface> m[3];
   
public:
   ChartModeMonitor(int &flags[])
   {
      m[0] = new ChartModeMonitorBase<long,ENUM_CHART_PROPERTY_INTEGER>(flags);
      m[1] = new ChartModeMonitorBase<double,ENUM_CHART_PROPERTY_DOUBLE>(flags);
      m[2] = new ChartModeMonitorBase<string,ENUM_CHART_PROPERTY_STRING>(flags);
   }
   
   virtual void snapshot() override
   {
      CALL_ALL(m, snapshot());
   }
   
   virtual void print() override
   {
      CALL_ALL(m, print());
   }
   
   virtual void backup() override
   {
      CALL_ALL(m, backup());
   }
   
   virtual void restore() override
   {
      CALL_ALL(m, restore());
   }
};
To simplify the code, the CALL_ALL macro is used here, which calls the specified method for all objects
from the array, and does this taking into account the overloaded operator [] in the class AutoPtr (it is
used to dereference a smart pointer and get a direct pointer to the "protected" object).
The destructor is usually responsible for freeing objects, but in this case, it was decided to use the
AutoPtr array (this class was discussed in the section Object type templates). This guarantees the
automatic deletion of dynamic objects when the m array is freed normally.
A more complete version of the monitor with support for subwindow numbers is provided in the file
ChartModeMonitorFull.mqh.
Based on the ChartModeMonitor class, you can easily implement the intended script ChartMode.mq5.
Its task is to check the state of a given set of properties every half a second. Now we are using an
infinite loop and Sleep here, but soon we will learn how to react to events on the charts in a different
way: due to notifications from the terminal.

---

## Page 866

Part 5. Creating application programs
866
5.7 Working with charts
#include <MQL5Book/ChartModeMonitor.mqh>
   
void OnStart()
{
   int flags[] =
   {
      CHART_MODE, CHART_FOREGROUND, CHART_SHIFT, CHART_AUTOSCROLL
   };
   ChartModeMonitor m(flags);
   Print("Initial state:");
   m.print();
   m.backup();
   
   while(!IsStopped())
   {
      m.snapshot();
      Sleep(500);
   }
   m.restore();
}
Run the script on any chart and try to change modes using the tool buttons. This way you can access
all elements except for CHART_FOREGROUND, which can be switched from the properties dialog (the
Common tab, flag Chart on top).
Toolbar buttons for switching chart modes
For example, the following log was created by switching the display from candles to bars, from bars to
lines, and back to candles, and then enabling indentation and auto-scrolling to the beginning.

---

## Page 867

Part 5. Creating application programs
867
5.7 Working with charts
Initial state:
    [key] [value]
[0]     0       1
[1]     1       0
[2]     2       0
[3]     4       0
CHART_MODE 1 -> 0
CHART_MODE 0 -> 2
CHART_MODE 2 -> 1
CHART_SHIFT 0 -> 1
CHART_AUTOSCROLL 0 -> 1
A more practical example of using the CHART_MODE property is an improved version of the indicator
IndSubChart.mq5 (we discussed its simplified version IndSubChartSimple.mq5 in the section
Multicurrency and multitimeframe indicators). The indicator is designed to display quotes of a third-
party symbol in a subwindow, and earlier we had to request a display method (candles, bars, or lines)
from the user through an input parameter. Now the parameter is no longer needed because we can
automatically switch the indicator to the mode that is used in the main window.
The current mode is stored in the global variable mode and is assigned first during initialization.
ENUM_CHART_MODE mode = 0;
   
int OnInit()
{
   ...
   mode = (ENUM_CHART_MODE)ChartGetInteger(0, CHART_MODE);
   ...
}
Detection of a new mode is best done in a specially designed event handler OnChartEvent, which we will
study in a separate chapter. At this stage, it is important to know that with any change in the chart,
the MQL program can receive notifications from the terminal if the code describes a function with this
predefined prototype (name and list of parameters). In particular, its first parameter contains an event
identifier that describes its meaning. We are still interested in the chart itself, and so we check if
eventId is equal to CHARTEVENT_CHART_CHANGE. This is necessary because the handler is also
capable of tracking graphical objects, keyboard, mouse, and arbitrary user messages.

---

## Page 868

Part 5. Creating application programs
868
5.7 Working with charts
void OnChartEvent(const int eventId,
                 // parameters not used here
                  const long &, const double &, const string &)
{
   if(eventId == CHARTEVENT_CHART_CHANGE)
   {
      const ENUM_CHART_MODE newmode = (ENUM_CHART_MODE)ChartGetInteger(0, CHART_MODE);
      if(mode != newmode)
      {
         const ENUM_CHART_MODE oldmode = mode;
         mode = newmode;
         // change buffer bindings and rendering type on the go
         InitPlot(0, InitBuffers(mode), Mode2Style(mode));
         // TODO: we will auto-adjust colors later
         // SetPlotColors(0, mode);
         if(oldmode == CHART_LINE || newmode == CHART_LINE)
         {
            // switching to or from CHART_LINE mode requires updating the entire chart,
            // because the number of buffers changes
            Print("Refresh");
            ChartSetSymbolPeriod(0, _Symbol, _Period);
         }
         else
         {
           // when switching between candles and bars, it is enough
           // just redraw the chart in a new manner,
           // because data doesn't change (previous 4 buffers with values)
            Print("Redraw");
            ChartRedraw();
         }
      }
   }
}
You can test the new indicator yourself by running it on the chart and switching the drawing methods.
These are not all the improvements made in IndSubChart.mq5. A little later, in the section on chart
colors, we will show the automatic adjustment of graphics to the chart color scheme.
5.7.1 0 Managing the visibility of chart elements
A large set of properties in ENUM_CHART_PROPERTY_INTEGER controls the visibility of chart
elements. Almost all of them are of boolean type: true corresponds to showing the element, and false
corresponds to hiding it. The exception is CHART_SHOW_VOLUMES, which uses the
ENUM_CHART_VOLUME_MODE enumeration (see below).
Identifier
Description
Value type
CH AR T_S H O W 
General price chart display. If set to false,
then rendering of any price chart
bool

---

## Page 869

Part 5. Creating application programs
869
5.7 Working with charts
Identifier
Description
Value type
attributes is disabled and all padding
along the chart edges is eliminated: time
and price scales, quick navigation bar,
calendar event markers, trade icons,
indicator and bar tooltips, indicator
subwindows, volume histograms, etc.
CH AR T_S H O W _TICK E R 
Show symbol ticker in the upper left
corner. Disabling the ticker automatically
disables OHLC (CHART_SHOW_OHLC)
bool
CH AR T_S H O W _O H L C
Show the OHLC values in the upper left
corner. Enabling OHLC automatically
enables the ticker
(CHART_SHOW_TICKER)
bool
CH AR T_S H O W _B ID _L IN E 
Show the Bid value as a horizontal line
bool
CH AR T_S H O W _AS K _L IN E 
Show the Ask value as a horizontal line
bool
CH AR T_S H O W _L AS T_L IN E 
Show the Last value as a horizontal line
bool
CH AR T_S H O W _P E R IO D _S E P 
Show vertical separators between
adjacent periods
bool
CH AR T_S H O W _G R ID 
Show grid on the chart
bool
CH AR T_S H O W _VO L U M E S 
Show volumes on a chart
E N U M _CH AR T_VO L U M E _
M O D E 
CH AR T_S H O W _O B JE CT_D E S CR 
Show text descriptions of objects
(descriptions are not shown for all types
of objects)
bool
CH AR T_S H O W _TR AD E _L E VE L S 
Show trading levels on the chart (levels
of open positions, Stop Loss, Take Profit
and pending orders)
bool
CH AR T_S H O W _D ATE _S CAL E 
Show the date scale on the chart
bool
CH AR T_S H O W _P R ICE _S CAL E 
Show the price scale on the chart
bool
CH AR T_S H O W _O N E _CL ICK 
Show the quick trading panel on the
chart ("One click trading" option)
bool

---

## Page 870

Part 5. Creating application programs
870
5.7 Working with charts
Flags in the settings dialog for some EN UM_CHART_PROPERTY_IN TEGER properties
Some of these properties are available for the user from the chart context menu, while some are only
available from the settings dialog. There are also settings that can only be changed from MQL5, in
particular, the display of the vertical (CHART_SHOW_DATE_SCALE) and horizontal
(CHART_SHOW_DATE_SCALE) scales, as well as the visibility of the entire chart (CHART_SHOW). The
last case should be especially noted, because turning off rendering is the ideal solution for creating your
own program interface using graphical resources and graphical objects, which are always rendered,
regardless of the value of CHART_SHOW.
The book comes with the script ChartBlackout.mq5, which toggles CHART_SHOW mode from current to
reverse on every run.
void OnStart()
{
   ChartSetInteger(0, CHART_SHOW, !ChartGetInteger(0, CHART_SHOW));
}
Thus, you can apply it on a normal chart to completely clear the window, and then apply it again to
restore the previous appearance.
The aforementioned ENUM_CHART_VOLUME_MODE enumeration contains the following members.

---

## Page 871

Part 5. Creating application programs
871 
5.7 Working with charts
Identifier
Description
Value
CHART_VOLUME_HIDE
Volumes are hidden
0
CHART_VOLUME_TICK
Tick volumes
1
CHART_VOLUME_REAL
Trading volumes (if any)
2
Similar to the script ChartMode.mq5, we implement a visibility monitor for chart elements in the script
ChartElements.mq5. The main difference lies in the different sets of controlled flags.
void OnStart()
{
   int flags[] =
   {
      CHART_SHOW,
      CHART_SHOW_TICKER, CHART_SHOW_OHLC,
      CHART_SHOW_BID_LINE, CHART_SHOW_ASK_LINE, CHART_SHOW_LAST_LINE,
      CHART_SHOW_PERIOD_SEP, CHART_SHOW_GRID,
      CHART_SHOW_VOLUMES,
      CHART_SHOW_OBJECT_DESCR,
      CHART_SHOW_TRADE_LEVELS,
      CHART_SHOW_DATE_SCALE, CHART_SHOW_PRICE_SCALE,
      CHART_SHOW_ONE_CLICK
   };
   ...
In addition, after creating a backup of the settings, we intentionally disable the time scales and prices
programmatically (when the script ends, it will restore them from the backup).
   ...
   m.backup();
   
   ChartSetInteger(0, CHART_SHOW_DATE_SCALE, false); 
   ChartSetInteger(0, CHART_SHOW_PRICE_SCALE, false);
   ... 
}
The following is a fragment of the log with comments about the actions taken. The first two entries
appeared exactly because the scales were disabled in the MQL code after the initial backup was
created.
CHART_SHOW_DATE_SCALE 1 -> 0 // disabled the time scale in the MQL5 code
CHART_SHOW_PRICE_SCALE 1 -> 0 // disabled the price scale in the MQL5 code
CHART_SHOW_ONE_CLICK 0 -> 1 // disabled "One click trading"
CHART_SHOW_GRID 1 -> 0 // disable "Grid"
CHART_SHOW_VOLUMES 0 -> 2 // showed real "Volumes"
CHART_SHOW_VOLUMES 2 -> 1 // showed "Tick volumes"
CHART_SHOW_TRADE_LEVELS 1 -> 0 // disabled "Trade levels"

---

## Page 872

Part 5. Creating application programs
872
5.7 Working with charts
5.7.1 1  Horizontal shifts
Another nuance of displaying charts is the horizontal indents from the left and right edges. They work
slightly differently but are described in the same enumeration ENUM_CHART_PROPERTY_DOUBLE and
use the type double.
Identifier
Description
CHART_SHIFT_SIZE
The indent of the zero bar from the right edge in percentages (from 1 0
to 50). Active only when the CHART_SHIFT mode is on. The shift is
indicated on the chart by a small inverted gray triangle on the top frame,
on the right side of the window.
CHART_FIXED_POSITION
The location of the fixed position of the chart from the left edge in
percent (from 0 to 1 00). A fixed chart position is indicated by a small
gray triangle on the horizontal time axis and is shown only if automatic
scrolling to the right when a new tick arrives is disabled
(CHART_AUTOCROLL). A bar that is in a fixed position stays in the same
place when you zoom in and out. By default, the triangle is in the very
corner of the chart (bottom left).
Visual representation of horizontal padding properties
We have the ChartShifts.mq5 script to check access to these properties, which works similarly to
ChartMode.mq5 and differs only in the set of controlled properties.

---

## Page 873

Part 5. Creating application programs
873
5.7 Working with charts
void OnStart()
{
   int flags[] =
   {
      CHART_SHIFT_SIZE, CHART_FIXED_POSITION
   };
   ChartModeMonitor m(flags);
   ...
}
Dragging a fixed position label (lower left) with the mouse results in this logging output.
Initial state:
    [key]  [value]
[0]     3 21.78771
[1]    41 17.87709
CHART_FIXED_POSITION 17.87709497206704 -> 26.53631284916201
CHART_FIXED_POSITION 26.53631284916201 -> 27.93296089385475
CHART_FIXED_POSITION 27.93296089385475 -> 28.77094972067039
CHART_FIXED_POSITION 28.77094972067039 -> 50.0
5.7.1 2 Horizontal scale (by time)
To determine the scale and number of bars along the horizontal axis, use the group of integer
properties from ENUM_CHART_PROPERTY_INTEGER. Among them, only CHART_SCALE is editable.
Identifier
Description
CH AR T_S CAL E 
Scale (0 to 5)
CH AR T_VIS IB L E _B AR S 
Number of bars currently visible on the chart (can be less than
CHART_WIDTH_IN_BARS due to CHART_SHIFT_SIZE indent) (r/o)
CH AR T_F IR S T_VIS IB L E _B AR 
Number of the first visible bar on the chart. The numbering goes from
right to left, as in a timeseries. (r/o)
CH AR T_W ID TH _IN _B AR S 
Chart width in bars (potential capacity, extreme bars on the left and right
may be partially visible) (r/o)
CH AR T_W ID TH _IN _P IXE L S 
Chart width in pixels (r/o)

---

## Page 874

Part 5. Creating application programs
874
5.7 Working with charts
EN UM_CHART_PROPERTY_IN TEGER properties on a chart
We are all ready to implement the next test script ChartScaleTime.mq5, which allows you to analyze
changes in these properties.
void OnStart()
{
   int flags[] =
   {
      CHART_SCALE,
      CHART_VISIBLE_BARS,
      CHART_FIRST_VISIBLE_BAR,
      CHART_WIDTH_IN_BARS,
      CHART_WIDTH_IN_PIXELS
   };
   ChartModeMonitor m(flags);
   ...
}
Below is a part of the log with comments about the actions taken.

---

## Page 875

Part 5. Creating application programs
875
5.7 Working with charts
Initial state:
    [key] [value]
[0]     5       4
[1]   100      35
[2]   104      34
[3]   105      45
[4]   106     715
                                 // 1) changed the scale to a smaller one:
CHART_SCALE 4 -> 3              // - the value of the "scale" property has changed
CHART_VISIBLE_BARS 35 -> 69        // - increased the number of visible bars
CHART_FIRST_VISIBLE_BAR 34 -> 68 // - the number of the first visible bar has increased
CHART_WIDTH_IN_BARS 45 -> 90 // - increased the potential number of bars
                                 // 2) disabled padding at the right edge
CHART_VISIBLE_BARS 69 -> 89 // - the number of visible bars has increased
CHART_FIRST_VISIBLE_BAR 68 -> 88 // - the number of the first visible bar has increased
                                 // 3) reduced the window size
CHART_VISIBLE_BARS 89 -> 86 // - number of visible bars decreased
CHART_WIDTH_IN_BARS 90 -> 86 // - the potential number of bars has decreased
CHART_WIDTH_IN_PIXELS 715 -> 680 // - decreased width in pixels
                                 // 4) clicked the "End" button to move to the current time
CHART_VISIBLE_BARS 86 -> 85 // - number of visible bars decreased
CHART_FIRST_VISIBLE_BAR 88 -> 84 // - the number of the first visible bar has decreased
5.7.1 3 Vertical scale (by price and indicator readings)
Properties related to the vertical scale are set and parsed using the elements of two enumerations:
ENUM_CHART_PROPERTY_INTEGER and ENUM_CHART_PROPERTY_DOUBLE. In the following table,
the properties are listed along with their value type.
Some properties allow you to access not only the main window but also a subwindow, for which
ChartSet and ChartGet functions should use the parameter window (0 means the main window and is
the default value for the short form of ChartGet).

---

## Page 876

Part 5. Creating application programs
876
5.7 Working with charts
Identifier
Description
Value type
CH AR T_S CAL E F IX
Fixed scale mode
bool
CH AR T_F IXE D _M AX
Fixed maximum of the window subwindow or
the initial maximum of the main window 
double
CH AR T_F IXE D _M IN 
Fixed minimum of the window subwindow or
the initial minimum of the main window
double
CH AR T_S CAL E F IX_1 1 
Scale mode 1 :1 
bool
CH AR T_S CAL E _P T_P E R _B AR 
Scale indication mode in points per bar
bool
CH AR T_P O IN TS _P E R _B AR 
Scale value in points per bar
double
CH AR T_P R ICE _M IN 
Minimum values in the window window or
subwindow (r/o)
double
CH AR T_P R ICE _M AX
Maximum values in the window window or
subwindow (r/o)
double
CH AR T_H E IG H T_IN _P IXE L S 
Fixed height of window or subwindow in
pixels, window parameter required
int
CH AR T_W IN D O W _YD IS TAN CE 
Distance in pixels along the vertical Y-axis
between the top frame of the window
subwindow and the upper frame of the main
chart window. (r/o)
int
By default, charts support adaptive scale so that quotes or indicator lines fit completely vertically on a
visible time period. For some applications, it is desirable to fix the scale, for which the terminal offers
several modes. In them, the chart can be scrolled with the mouse or with the keys (Shift + arrow) not
only left/right, but also up/down, and a slider bar appears at the right scale, using which you can
quickly scroll the chart with the mouse.
The fixed mode is set by turning on the CHART_SCALEFIX flag and specifying the required maximum
and minimum in the CHART_FIXED_MAX and CHART_FIXED_MIN fields (in the main window, the user
will be able to move the chart up or down, due to which the CHART_FIXED_MAX and
CHART_FIXED_MIN values will change synchronously, but the vertical scale will remain the same). The
user will also be able to change the vertical scale by pressing the mouse button on the price scale and,
without releasing it, moving it up or down. Subwindows do not provide interactive editing of the vertical
scale. In this regard, we will later present an indicator SubScaler.mq5 (see keyboard events section),
which will allow the user to control the range of values in the subwindow using the keyboard, rather
than from the settings dialog, using the fields on the Scale tab.
The CHART_SCALEFIX_1 1  mode provides an approximate visual equality of the sides of the square on
the screen: X bars in pixels (horizontally) will be equal to X points in pixels (vertically). The equality is
approximate, because the size of the pixels, as a rule, is not the same vertically and horizontally.
Finally, there is a mode for fixing the ratio of the number of points per bar, which is enabled by the
CHART_SCALE_PT_PER_BAR option, and the required ratio itself is set using the
CHART_POINTS_PER_BAR property. Unlike the CHART_SCALEFIX mode, the user will not be able to
interactively change the scale with the mouse on the chart. In this mode, a horizontal distance of one

---

## Page 877

Part 5. Creating application programs
877
5.7 Working with charts
bar will be displayed on the screen in the same ratio to the specified number of vertical points as the
aspect ratio of the chart (in pixels). If the timeframes and sizes of the two charts are equal, one will
look compressed in price compared to the other according to the ratio of their
CHART_POINTS_PER_BAR values. Obviously, the smaller the timeframe, the smaller the range of bars,
and therefore, with the same scale, small timeframes look more "flattened".
Programmatically setting the CHART_HEIGHT_IN_PIXELS property makes it impossible for the user to
edit the window/subwindow size. This is often used for windows that host trading panels with a
predefined set of controls (buttons, input fields, etc.). In order to remove the fixation of the size, set
the value of the property to -1 .
The CHART_WINDOW_YDISTANCE value is required to convert the absolute coordinates of the main
chart into local coordinates of the subwindow for correct work with graphical objects. The point is that
when mouse events occur, cursor coordinates are transferred relative to the main chart window, while
the coordinates of graphical objects in the indicator subwindow are set relative to the upper left corner
of the subwindow.
Let's prepare the ChartScalePrice.mq5 script for analyzing changes in vertical scales and sizes.
void OnStart()
{
   int flags[] =
   {
      CHART_SCALEFIX, CHART_SCALEFIX_11,
      CHART_SCALE_PT_PER_BAR, CHART_POINTS_PER_BAR,
      CHART_FIXED_MAX, CHART_FIXED_MIN,
      CHART_PRICE_MIN, CHART_PRICE_MAX,
      CHART_HEIGHT_IN_PIXELS, CHART_WINDOW_YDISTANCE
   };
   ChartModeMonitor m(flags);
   ...
}
It reacts to chart manipulation in the following way:

---

## Page 878

Part 5. Creating application programs
878
5.7 Working with charts
Initial state:
    [key] [value]   // ENUM_CHART_PROPERTY_INTEGER
[0]     6       0
[1]     7       0
[2]    10       0
[3]   107     357
[4]   110       0
    [key]  [value]  // ENUM_CHART_PROPERTY_DOUBLE
[0]    11 10.00000
[1]     8  1.13880
[2]     9  1.12330
[3]   108  1.12330
[4]   109  1.13880
// reduced the vertical size of the window
CHART_HEIGHT_IN_PIXELS 357 -> 370
CHART_HEIGHT_IN_PIXELS 370 -> 408
CHART_FIXED_MAX 1.1389 -> 1.1388
CHART_FIXED_MIN 1.1232 -> 1.1233
CHART_PRICE_MIN 1.1232 -> 1.1233
CHART_PRICE_MAX 1.1389 -> 1.1388
// reduced the horizontal scale, which increased the price range
CHART_FIXED_MAX 1.1388 -> 1.139
CHART_FIXED_MIN 1.1233 -> 1.1183
CHART_PRICE_MIN 1.1233 -> 1.1183
CHART_PRICE_MAX 1.1388 -> 1.139
CHART_FIXED_MAX 1.139 -> 1.1406
CHART_FIXED_MIN 1.1183 -> 1.1167
CHART_PRICE_MIN 1.1183 -> 1.1167
CHART_PRICE_MAX 1.139 -> 1.1406
// expand the price range using the mouse (quotes "shrink" vertically)
CHART_FIXED_MAX 1.1406 -> 1.1454
CHART_FIXED_MIN 1.1167 -> 1.1119
CHART_PRICE_MIN 1.1167 -> 1.1119
CHART_PRICE_MAX 1.1406 -> 1.1454
5.7.1 4 Colors
An MQL program can recognize and change colors to display all chart elements. The corresponding
properties are part of the ENUM_CHART_PROPERTY_INTEGER enumeration.
Identifier
Description
CH AR T_CO L O R _B ACK G R O U N D 
Chart background color
CH AR T_CO L O R _F O R E G R O U N D 
Color of axes, scales, and OHLC lines
CH AR T_CO L O R _G R ID 
Grid color
CH AR T_CO L O R _VO L U M E 
Color of volumes and position opening levels

---

## Page 879

Part 5. Creating application programs
879
5.7 Working with charts
Identifier
Description
CH AR T_CO L O R _CH AR T_U P 
The color of the up bar, the shadow, and the edging of the
body of a bullish candle
CH AR T_CO L O R _CH AR T_D O W N 
The color of the down bar, the shadow, and the edging of the
body of a bearish candle
CH AR T_CO L O R _CH AR T_L IN E 
The color of the chart line and of the contours of Japanese
candlesticks
CH AR T_CO L O R _CAN D L E _B U L L 
Bullish candlestick body color
CH AR T_CO L O R _CAN D L E _B E AR 
Bearish candlestick body color
CH AR T_CO L O R _B ID 
Bid price line color
CH AR T_CO L O R _AS K 
Ask price line color
CH AR T_CO L O R _L AS T
Color of the last traded price line (Last)
CH AR T_CO L O R _S TO P _L E VE L 
Color of stop order levels (Stop Loss and Take Profit)
As an example of working with these properties, let's create a script – ChartColorInverse.mq5. It will
change all the colors of the graph to inverse, that is, for the bit representation of the color in the
format RGB XOR ('^',XOR). Thus, after restarting the script on the same chart, its settings will be
restored.
#define RGB_INVERSE(C) ((color)C ^ 0xFFFFFF)
   
void OnStart()
{
   ENUM_CHART_PROPERTY_INTEGER colors[] =
   {
      CHART_COLOR_BACKGROUND,
      CHART_COLOR_FOREGROUND,
      CHART_COLOR_GRID,
      CHART_COLOR_VOLUME,
      CHART_COLOR_CHART_UP,
      CHART_COLOR_CHART_DOWN,
      CHART_COLOR_CHART_LINE,
      CHART_COLOR_CANDLE_BULL,
      CHART_COLOR_CANDLE_BEAR,
      CHART_COLOR_BID,
      CHART_COLOR_ASK,
      CHART_COLOR_LAST,
      CHART_COLOR_STOP_LEVEL
   };
   
   for(int i = 0; i < ArraySize(colors); ++i)
   {
      ChartSetInteger(0, colors[i], RGB_INVERSE(ChartGetInteger(0, colors[i])));
   }

---

## Page 880

Part 5. Creating application programs
880
5.7 Working with charts
}
The following image combines the images of the chart before and after applying the script.
Inverting chart colors from an MQL program
Now let's finish editing IndSubChart.mq5. We need to read the colors of the main chart and apply them
to our indicator chart. There is a function for these purposes: SetPlotColors, whose call was
commented out in the OnChartEvent handler (see the last example in the section Chart Display Modes).
void SetPlotColors(const int index, const ENUM_CHART_MODE m)
{
   if(m == CHART_CANDLES)
   {
      PlotIndexSetInteger(index, PLOT_COLOR_INDEXES, 3);
      PlotIndexSetInteger(index, PLOT_LINE_COLOR, 0, (int)ChartGetInteger(0, CHART_COLOR_CHART_LINE));  // rectangle
      PlotIndexSetInteger(index, PLOT_LINE_COLOR, 1, (int)ChartGetInteger(0, CHART_COLOR_CANDLE_BULL)); // up
      PlotIndexSetInteger(index, PLOT_LINE_COLOR, 2, (int)ChartGetInteger(0, CHART_COLOR_CANDLE_BEAR)); // down
   }
   else
   {
      PlotIndexSetInteger(index, PLOT_COLOR_INDEXES, 1);
      PlotIndexSetInteger(index, PLOT_LINE_COLOR, (int)ChartGetInteger(0, CHART_COLOR_CHART_LINE));
   }
}
In this new function, we get, depending on the chart drawing mode, either the color of the contours and
bodies of bullish and bearish candlesticks, or the color of the lines, and apply the colors to the charts.
Of course, do not forget to call this function during initialization.

---

## Page 881

Part 5. Creating application programs
881 
5.7 Working with charts
int OnInit()
{
   ...
   mode = (ENUM_CHART_MODE)ChartGetInteger(0, CHART_MODE);
   InitPlot(0, InitBuffers(mode), Mode2Style(mode));
   SetPlotColors(0, mode);
   ...
}
The indicator is ready. Try running it in the window and changing the colors in the chart properties
dialog. The chart should automatically adapt to the new settings.
5.7.1 5 Mouse and keyboard control
In this section, we will get acquainted with a group of properties that affect how the chart will capture
certain mouse and keyboard manipulations, which are regarded by default as control actions. In
particular, MetaTrader 5 users are well aware that the chart can be scrolled with the mouse, and the
context menu can be called to execute the most requested commands. MQL5 allows you to disable this
behavior of the chart completely or partially. It is important to note that this can only be done
programmatically: there are no similar options in the terminal user interface.
The only exception is the CHART_DRAG_TRADE_LEVELS option (see in the table below): the terminal
settings provide the Charts tab with a drop-down list that controls the permission to drag trading levels
with the mouse.
All properties of this group have a boolean type (true for allowed and false for disabled) and they are in
the ENUM_CHART_PROPERTY_INTEGER enumeration.
Identifier
Description
CHART_CONTEXT_MENU
Enabling/disabling access to the context menu by pressing the
right mouse button. The false value disables only the context menu
of the chart, while the context menu for objects on the chart
remains available. The default value is true.
CHART_CROSSHAIR_TOOL
Enable/disable access to the Crosshair tool by pressing the middle
mouse button. The default value is true.
CHART_MOUSE_SCROLL
Scrolling the chart with the left mouse button or wheel. When
scrolling is enabled, this applies not only to scrolling horizontally,
but also vertically, but the latter is only available when a fixed scale
is set: one of the CHART_SCALEFIX, CHART_SCALEFIX_1 1 , or
CHART_SCALE_PT_PER_BAR properties. The default value is true.
CHART_KEYBOARD_CONTROL
Ability to manage the chart from the keyboard (buttons Home, End,
PageUp/PageDown, +/-, up/down arrows, etc.). Setting to false
allows you to disable scrolling and scaling of the chart, but at the
same time, it is possible to receive keystroke events for these keys
in OnChartEvent. The default value is true.
CHART_QUICK_NAVIGATION
Enabling the quick navigation bar in the chart, which automatically
appears in the left corner of the timeline when you double-click the
mouse or press the Space or Input keys. Using the bar, you can

---

## Page 882

Part 5. Creating application programs
882
5.7 Working with charts
Identifier
Description
quickly change the symbol, timeframe, or date of the first visible
bar. By default, the property is set to true and quick navigation is
enabled.
CHART_DRAG_TRADE_LEVELS
Permission to drag trading levels on the chart with the mouse.
Drag mode is enabled by default (true).
In the test script ChartInputControl.mq5, we will set the monitor to all of the above properties, and in
addition, we will provide input variables for arbitrary setting of values by the user. Our script saves a
backup copy of the settings at startup, so all changed properties will be restored when the script ends.

---

## Page 883

Part 5. Creating application programs
883
5.7 Working with charts
#property script_show_inputs
   
#include <MQL5Book/ChartModeMonitor.mqh>
   
input bool ContextMenu = true; // CHART_CONTEXT_MENU
input bool CrossHairTool = true; // CHART_CROSSHAIR_TOOL
input bool MouseScroll = true; // CHART_MOUSE_SCROLL
input bool KeyboardControl = true; // CHART_KEYBOARD_CONTROL
input bool QuickNavigation = true; // CHART_QUICK_NAVIGATION
input bool DragTradeLevels = true; // CHART_DRAG_TRADE_LEVELS
   
void OnStart()
{
   const bool Inputs[] =
   {
      ContextMenu, CrossHairTool, MouseScroll,
      KeyboardControl, QuickNavigation, DragTradeLevels
   };
   const int flags[] =
   {
      CHART_CONTEXT_MENU, CHART_CROSSHAIR_TOOL, CHART_MOUSE_SCROLL,
      CHART_KEYBOARD_CONTROL, CHART_QUICK_NAVIGATION, CHART_DRAG_TRADE_LEVELS
   };
   ChartModeMonitor m(flags);
   Print("Initial state:");
   m.print();
   m.backup();
   
   for(int i = 0; i < ArraySize(flags); ++i)
   {
      ChartSetInteger(0, (ENUM_CHART_PROPERTY_INTEGER)flags[i], Inputs[i]);
   }
   
   while(!IsStopped())
   {
      m.snapshot();
      Sleep(500);
   }
   m.restore();
}
For example, when we run the script, we can reset the permissions for the context menu, crosshair
tool, mouse, and keyboard controls to false. The result is in the following log.

---

## Page 884

Part 5. Creating application programs
884
5.7 Working with charts
Initial state:
    [key] [value]
[0]    50       1
[1]    49       1
[2]    42       1
[3]    47       1
[4]    45       1
[5]    43       1
CHART_CONTEXT_MENU 1 -> 0
CHART_CROSSHAIR_TOOL 1 -> 0
CHART_MOUSE_SCROLL 1 -> 0
CHART_KEYBOARD_CONTROL 1 -> 0
In this case, you will not be able to move the chart with either the mouse or the keyboard, and even
call the context menu. Therefore, in order to restore its performance, you will have to drop the same or
another script on the chart (recall that there can be only one script on the chart, and when a new one
is applied, the previous one is unloaded;). It is enough to drop a new instance of the script, but not to
run it (press Cancel in the dialog for entering input variables).
5.7.1 6 Undocking chart window
Chart windows in the terminal can be undocked from the main window, after which they can be moved
to any place on the desktop, including other monitors. MQL5 allows you to find out and change this
setting: the corresponding properties are included in the ENUM_CHART_PROPERTY_INTEGER
enumeration.
Identifier
Description
Value type
CHART_IS_DOCKED
The chart window is docked (true by
default). If set to false, then the chart can be
dragged outside the terminal
bool
CHART_FLOAT_LEFT
The left coordinate of the undocked chart
relative to the virtual screen
int
CHART_FLOAT_TOP
The top coordinate of the undocked chart
relative to the virtual screen
int
CHART_FLOAT_RIGHT
The right coordinate of the undocked chart
relative to the virtual screen
int
CHART_FLOAT_BOTTOM
The bottom coordinate of the undocked
chart relative to the virtual screen
int
Let's set the tracking of these properties in the ChartDock.mq5 script.

---

## Page 885

Part 5. Creating application programs
885
5.7 Working with charts
void OnStart()
{
   const int flags[] =
   {
      CHART_IS_DOCKED,
      CHART_FLOAT_LEFT, CHART_FLOAT_TOP, CHART_FLOAT_RIGHT, CHART_FLOAT_BOTTOM
   };
   ChartModeMonitor m(flags);
   ...
}
If you now run the script, then undock the chart using the context menu (unpress the Docked switch
command) and move or resize the chart, the corresponding logs will be added to the journal.
Initial state:
    [key] [value]
[0]    51       1
[1]    52       0
[2]    53       0
[3]    54       0
[4]    55       0
                              // undocked
CHART_IS_DOCKED 1 -> 0
CHART_FLOAT_LEFT 0 -> 299
CHART_FLOAT_TOP 0 -> 75
CHART_FLOAT_RIGHT 0 -> 1263
CHART_FLOAT_BOTTOM 0 -> 472
                              // changed the vertical size
CHART_FLOAT_BOTTOM 472 -> 500
CHART_FLOAT_BOTTOM 500 -> 539
                              // changed the horizontal size
CHART_FLOAT_RIGHT 1263 -> 1024
CHART_FLOAT_RIGHT 1024 -> 1023
                              // docked back
CHART_IS_DOCKED 0 -> 1
This section completes the description of properties managed through the ChartGet and ChartSet
functions, so let's summarize the material using a common script ChartFullSet.mq5. It keeps track of
the state of all properties of all types. The initialization of the flags array is done by simply filling in
successive indexes in a loop. The maximum value is taken with a margin in case of new properties, and
extra non-existent numbers will be automatically discarded by the check built into the
ChartModeMonitorBase class (remember the detect method).
After activating the script, try changing any settings while watching the program messages in the log.
5.7.1 7 Getting MQL program drop coordinates on a chart
Users often drag MQL programs onto a chart using a mouse. In addition to being convenient, this allows
you to set some context for the algorithm. For example, an indicator can be applied in different
subwindows, or a script can place a pending order at the price where the user placed it on the chart.
The next group of functions is designed to get the coordinates of the point to which the program was
dragged and dropped.

---

## Page 886

Part 5. Creating application programs
886
5.7 Working with charts
int ChartWindowOnDropped()
This function returns the number of the chart subwindow on which the current Expert Advisor, script, or
indicator is dropped by the mouse. The main window, as we know, is numbered 0, and the subwindows
are numbered starting from 1 . The number of a subwindow does not depend on whether there are
hidden subwindows above it, as their indices remain assigned to them. In other words, the visible
subwindow number may differ from its real index if there are hidden subwindows.
double ChartPriceOnDropped()
datetime ChartTimeOnDropped()
This pair of functions returns the program drop point coordinates in units of price and time. Please note
that arbitrary data can be displayed in subwindows, and not just prices, although the function name
ChartPriceOnDropped includes 'Price'.
Attention! The target point time is not rounded by the size of the chart timeframe, so even on the
H1  and D1  charts, you can get a value with minutes and even seconds.
int ChartXOnDropped()
int ChartYOnDropped()
These two functions return the X and Y screen coordinates of a point in pixels. The origin of the
coordinates is located in the upper left corner of the main chart window. We talked about the direction
of the axes in the Screen specifications section.
The Y coordinate is always counted from the upper left corner of the main chart, even if the drop point
belongs to a subwindow. To translate this value into a coordinate y relative to a subwindow, use the
property CHART_WINDOW_YDISTANCE (see example).
Let's output the values of all mentioned functions to the log in the script ChartDrop.mq5.
void OnStart()
{
   const int w = PRTF(ChartWindowOnDropped());
   PRTF(ChartTimeOnDropped());
   PRTF(ChartPriceOnDropped());
   PRTF(ChartXOnDropped());
   PRTF(ChartYOnDropped());
   
   // for the subwindow, recalculate the y coordinate to the local one
   if(w > 0)
   {
      const int y = (int)PRTF(ChartGetInteger(0, CHART_WINDOW_YDISTANCE, w));
      PRTF(ChartYOnDropped() - y);
   }
}
For example, if we drop this script into the first subwindow where the WPR indicator is running, we can
get the following results.

---

## Page 887

Part 5. Creating application programs
887
5.7 Working with charts
ChartWindowOnDropped()=1 / ok
ChartTimeOnDropped()=2021.11.30 03:52:30 / ok
ChartPriceOnDropped()=-50.0 / ok
ChartXOnDropped()=217 / ok
ChartYOnDropped()=312 / ok
ChartGetInteger(0,CHART_WINDOW_YDISTANCE,w)=282 / ok
ChartYOnDropped()-y=30 / ok
Despite the fact that the script is dropped on the EURUSD, H1  chart, we got a timestamp with minutes
and seconds.
Note that the "price" value is -50 because the range of WPR values is [0,-1 00].
In addition, the vertical coordinate of point 31 2 (relative to the entire chart window) was converted to
the local coordinate of the subwindow: since the vertical distance from the beginning of the main chart
to the subwindow was 282, the value y inside the subwindow turned out to be 30.
5.7.1 8 Translation of screen coordinates to time/price and vice versa
The presence of different principles for measuring the working space of the chart leads to the need to
recalculate the units of measurement among themselves. There are two functions for this.
bool ChartTimePriceToXY(long chartId, int window, datetime time, double price, int &x, int &y)
bool ChartXYToTimePrice(long chartId, int x, int y, int &window, datetime &time, double &price)
The ChartTimePriceToXY function converts chart coordinates from time/price representation
(time/price) to X and Y coordinates in pixels (x/y). The ChartXYToTimePrice function performs the
reverse operation: it converts the X and Y coordinates into time and price values.
Both functions require the chart ID to be specified in the first parameter chartId. In addition to this, the
number of the window subwindow is passed in ChartTimePriceToXY (it should be within the number of
windows). If there are several subwindows, each of them has its own timeseries and a scale along the
vertical axis (conditionally called "price" with the price parameter).
The window parameter is the output in the ChartXYToTimePrice function. The function fills this
parameter along with time and price. This is because pixel coordinates are common to the entire
screen, and the origin x/y can fall into any subwindow.

---

## Page 888

Part 5. Creating application programs
888
5.7 Working with charts
Time, price, and screen coordinates
Functions return true upon successful completion.
Please note that the visible rectangular area that corresponds to quotes or screen coordinates is
limited in both coordinate systems. Therefore, situations are possible when, with specific initial data,
the received time, prices, or pixels will be out of the visibility area. In particular, negative values can
also be obtained. We will look at an interactive recalculation example in the chapter on events on
charts.
In the previous section, we saw how you can find out where an MQL program was launched. Although
physically there is only one end drop point, its representation in quotation and screen coordinates, as a
rule, contains a calculation error. Two new functions for converting pixels into price/time and vice versa
will help us to make sure of this.
The modified script is called ChartXY.mq5. It can be roughly divided into 3 stages. In the first stage, we
derive the coordinates of the drop point, as before.
void OnStart()
{
   const int w1 = PRTF(ChartWindowOnDropped());
   const datetime t1 = PRTF(ChartTimeOnDropped());
   const double p1 = PRTF(ChartPriceOnDropped());
   const int x1 = PRTF(ChartXOnDropped());
   const int y1 = PRTF(ChartYOnDropped());
   ...
In the second stage, we try to transform the screen coordinates x1  and y1  during (t2) and price (p2),
and compare them with those obtained from OnDropped functions above.

---

## Page 889

Part 5. Creating application programs
889
5.7 Working with charts
   int w2;
   datetime t2;
   double p2;
   PRTF(ChartXYToTimePrice(0, x1, y1, w2, t2, p2));
   Print(w2, " ", p2, " ", t2);
   PRTF(w1 == w2 && t1 == t2 && p1 == p2);
   ...
Then we perform the inverse transformation: we use the obtained quotation coordinates t1  and p1  to
calculate screen coordinates -x2 and y2 and also compare with the original values x1  and y1 .
   int x2, y2;
   PRTF(ChartTimePriceToXY(0, w1, t1, p1, x2, y2));
   Print(x2, " ", y2);
   PRTF(x1 == x2 && y1 == y2);
   ...
As we will see later in the example log, all of the above checks will fail (there will be slight discrepancies
in the values). So we need a third step.
Let's recalculate the screen and quote coordinates with the suffix 2 in the variable names and save
them in the variables with the new suffix 3. Then we compare all the values from the first and third
stages with each other.
   int w3;
   datetime t3;
   double p3;
   PRTF(ChartXYToTimePrice(0, x2, y2, w3, t3, p3));
   Print(w3, " ", p3, " ", t3);
   PRTF(w1 == w3 && t1 == t3 && p1 == p3);
   
   int x3, y3;
   PRTF(ChartTimePriceToXY(0, w2, t2, p2, x3, y3));
   Print(x3, " ", y3);
   PRTF(x1 == x3 && y1 == y3);
}
Let's run the script on the XAUUSD, H1  chart. Here is the original point data.
ChartWindowOnDropped()=0 / ok
ChartTimeOnDropped()=2021.11.22 18:00:00 / ok
ChartPriceOnDropped()=1797.7 / ok
ChartXOnDropped()=234 / ok
ChartYOnDropped()=280 / ok
Converting pixels to quotes gives the following results.
ChartXYToTimePrice(0,x1,y1,w2,t2,p2)=true / ok
0 1797.16 2021.11.22 18:30:00
w1==w2&&t1==t2&&p1==p2=false / ok
There are differences in both time and price. Backcounting is also not perfect in terms of accuracy.

---

## Page 890

Part 5. Creating application programs
890
5.7 Working with charts
ChartTimePriceToXY(0,w1,t1,p1,x2,y2)=true / ok
232 278
x1==x2&&y1==y2=false / ok
The loss of precision occurs due to the quantization of the values on the axes according to the units of
measurement, in particular pixels and points.
Finally, the last step proves that the errors obtained above are not function artifacts, because round-
robin recalculation leads to the original result.
ChartXYToTimePrice(0,x2,y2,w3,t3,p3)=true / ok
0 1797.7 2021.11.22 18:00:00
w1==w3&&t1==t3&&p1==p3=true / ok
ChartTimePriceToXY(0,w2,t2,p2,x3,y3)=true / ok
234 280
x1==x3&&y1==y3=true / ok
In pseudocode, this can be expressed by the following equalities:
ChartTimePriceToXY(ChartXYToTimePrice(XY)) = XY
ChartXYToTimePrice(ChartTimePriceToXY(TP)) = TP
Applying the ChartTimePriceToXY function to ChartXYToTimePrice work results will give the original
coordinates. The same is true for transformations in the other direction: applying ChartXYToTimePrice
to the results of ChartTimePriceToXY will give a match.
Thus, one should carefully consider the implementation of algorithms that use recalculation functions if
they are subject to increased accuracy requirements.
Another example of using ChartWindowOnDropped will be given in the script ChartIndicatorMove.mq5 in
the section Managing indicators on the chart.
5.7.1 9 Scrolling charts along the time axis
MetaTrader 5 users are familiar with the quick chart navigation panel, which opens by double-clicking in
the left corner of the timeline or by pressing the Space or Input keys. A similar possibility is also
available programmatically by using the ChartNavigate function.
bool ChartNavigate(long chartId, ENUM_CHART_POSITION position, int shift = 0)
The function shifts the chartId chart by the specified number of bars relative to the predefined chart
position specified by the position parameter. It is of ENUM_CHART_POSITION enumeration type with
the following elements.
Identifier
Description
CHART_BEGIN
Chart beginning (oldest prices)
CHART_CURRENT_POS
Current position
CHART_END
Chart end (latest prices)

---

## Page 891

Part 5. Creating application programs
891 
5.7 Working with charts
The shift parameter sets the number of bars by which the chart should be shifted. A positive value
shifts the chart to the right (towards the end), and a negative value shifts the chart to the left (towards
the beginning).
The function returns true if successful or false as a result of an error.
To test the function, let's create a simple script ChartNavigate.mq5. With the help of input variables,
the user can choose a starting point and a shift in bars.
#property script_show_inputs
input ENUM_CHART_POSITION Position = CHART_CURRENT_POS;
input int Shift = 0;
   
void OnStart()
{
   ChartSetInteger(0, CHART_AUTOSCROLL, false);
   const int start = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR);
   ChartNavigate(0, Position, Shift);
   const int stop = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR);
   Print("Moved by: ", stop - start, ", from ", start, " to ", stop);
}
The log displays the number of the first visible bar before and after the move.
A more practical example would be the script ChartSynchro.mq5, which allows you to synchronously
scroll through all charts it is running on, in response to the user manually scrolling through one of the
charts. Thus, you can synchronize windows of different timeframes of the same instrument or analyze
parallel price movements on different instruments.

---

## Page 892

Part 5. Creating application programs
892
5.7 Working with charts
void OnStart()
{
   datetime bar = 0; // current position (time of the first visible bar)
  
   conststring namePosition =__FILE__;// global variable name
  
   ChartSetInteger(0,CHART_AUTOSCroll,false); // disable autoscroll
  
   while(!IsStopped())
   {
      const bool active = ChartGetInteger(0, CHART_BRING_TO_TOP);
      const int move = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR);
   
      // the active chart is the leader, and the rest are slaves
      if(active)
      {
         const datetime first = iTime(_Symbol, _Period, move);
         if(first != bar)
         {
            // if the position has changed, save it in a global variable
            bar = first;
            GlobalVariableSet(namePosition, bar);
            Comment("Chart ", ChartID(), " scrolled to ", bar);
         }
      }
      else
      {
         const datetime b = (datetime)GlobalVariableGet(namePosition);
      
         if(b != bar)
         {
            // if the value of the global variable has changed, adjust the position
            bar = b;
            const int difference = move - iBarShift(_Symbol, _Period, bar);
            ChartNavigate(0, CHART_CURRENT_POS, difference);
            Comment("Chart ", ChartID(), " forced to ", bar);
         }
      }
    
      Sleep(250);
   }
   Comment("");
}
Alignment is performed by the date and time of the first visible bar (CHART_FIRST_VISIBLE_BAR). The
script in a loop checks this value and, if it works on an active chart, writes it to a global variable.
Scripts on other charts read this variable and adjust their position accordingly with ChartNavigate. The
parameters specify the relative movement of the chart (CHART_CURRENT_POS), and the number of
bars to move is defined as the difference between the current number of the first visible bar and the
one read from the global variable.
The following image shows the result of synchronizing the H1  and M1 5 charts for EURUSD.

---

## Page 893

Part 5. Creating application programs
893
5.7 Working with charts
An example of the script for synchronizing chart positions
After we get familiar with the system events on charts, we will convert this script into an indicator and
get rid of the infinite loop.
5.7.20 Chart redraw request
In most cases, charts automatically respond to changes in data and terminal settings, refreshing the
window image accordingly (price charts, indicator charts, etc.). However, MQL programs are too
versatile and can perform arbitrary actions, for which it is not so easy to determine whether redrawing
is required or not. In addition, analyzing any action of each MQL program on this account can be
resource-intensive and cause a drop in the overall performance of the terminal. Therefore, the MQL5
API provides the ChartRedraw function, with the help of which the MQL program itself can, if necessary,
request a redrawing of the chart.
void ChartRedraw(long chartId = 0)
The function causes a forced redrawing of the chart with the specified identifier (default value 0 means
the current chart). Usually, it is applied after the program changes the properties of the chart or
objects placed on it.
We have seen an example of using ChartRedraw in the indicator IndSubChart.mq5 in the section Chart
display modes. Another example will be given in the section Opening and closing charts.
This function affects exactly the redrawing of the chart, without causing the recalculation of timeseries
with quotes and indicators. The last option for updating (in fact, rebuilding) the chart is more "hard"
and is performed by the ChartSetSymbolPeriod function (see the next section).

---

## Page 894

Part 5. Creating application programs
894
5.7 Working with charts
5.7.21  Switching symbol and timeframe
Sometimes an MQL program needs to switch the current symbol or timeframe of a chart. In particular,
this is a familiar functionality for many multicurrency, multitimeframe trading panels or trading history
analyzing utilities. For this purpose, the MQL5 API provides the ChartSetSymbolPeriod function.
You can also use this function to initiate the recalculation of the entire chart, including the indicators
located on it. You can simply specify the current symbol and timeframe as parameters. This technique
can be useful for indicators that could not be fully calculated on the first call of OnCalculate, and wait
for the loading of third-party data (other symbols, ticks, or indicators). Also, changing the
symbol/timeframe leads to the reinitialization of the Expert Advisors attached to the chart. The script
(if it is executed periodically in a cycle) will completely disappear from the chart during this procedure
(it will be unloaded from the old symbol/timeframe combination but will not be loaded automatically for
the new combination).
bool ChartSetSymbolPeriod(long chartId, string symbol, ENUM_TIMEFRAMES timeframe)
The function changes the symbol and timeframe of the specified chart with the chartId identifier to the
values of the corresponding parameters: symbol and timeframe. 0 in the chartId parameter means the
current chart, NULL in the symbol parameter is the current character, and 0 in the timeframe
parameter is the current timeframe.
Changes take effect asynchronously, that is, the function only sends a command to the terminal and
does not wait for its execution. The command is added to the chart's message queue and is executed
only after all previous commands have been processed.
The function returns true in case of successful placement of the command in the chart queue or false
in case of problems. Information about the error can be found in _ LastError.
We have seen examples of using the function to update several indicators, in particular:
• IndDeltaVolume.mq5 (see Waiting for data and managing visibility)
• IndUnityPercent.mq5 (see Multicurrency and multitimeframe indicators)
• UseWPRMTF.mq5 (see Support for multiple symbols and timeframes)
• UseM1 MA.mq5 (see Using built-in indicators)
• UseDemoAllLoop.mq5 (see Deleting indicator instances)
• IndSubChart.mq5 (see Chart display modes)
5.7.22 Managing indicators on the chart
As we have already found out, charts are the execution and visualization environment for indicators.
Their close connection finds additional confirmation in the form of a whole group of built-in functions
that provide control over indicators on charts. In one of the previous chapters, we already completed
an overview of these features. Now we are ready to consider them in detail after getting acquainted
with the charts.
All functions are united by the fact that the first two parameters are unified: this is the chart identifier
(chartId) and window number (window). Zero values of the parameters denote the current chart and
the main window, respectively.

---

## Page 895

Part 5. Creating application programs
895
5.7 Working with charts
int ChartIndicatorsTotal(long chartId, int window)
The function returns the number of all indicators attached to the specified chart window. It can be used
to enumerate all the indicators attached to a given chart. The number of all chart windows can be
obtained from the property CHART_WINDOWS_TOTAL using the function ChartGetInteger.
string ChartIndicatorName(long chartId, int window, int index)
The function returns the indicator's short name by the index in the list of indicators located in the
specified chart window. The short name is the name specified in the property INDICATOR_SHORTNAME
by the function IndicatorSetString (if it is not set, then by default it is equal to the name of the
indicator file).
int ChartIndicatorGet(long chartId, int window, const string shortname)
It returns the handle of the indicator with the specified short name in the specific chart window. We
can say that the identification of the indicator in the function ChartIndicatorGet is made exactly by the
short name, and therefore it is recommended to compose it in such a way that it contains the values of
all input parameters. If this is not possible for one reason or another, there is another way to identify
an indicator instance through the list of its parameters, which can be obtained by a given descriptor
using the IndicatorParameters function.
Getting a handle from a function ChartIndicatorGet increases the internal counter for using this
indicator. The terminal execution system keeps loaded all indicators whose counter is greater than
zero. Therefore, an indicator that is no longer needed must be explicitly freed by calling
IndicatorRelease. Otherwise, the indicator will remain idle and consume resources.
bool ChartIndicatorAdd(long chartId, int window, int handle)
The function adds an indicator with the descriptor passed in the last parameter to the specified chart
window. The indicator and chart must have the same combination of symbol and timeframe. Otherwise,
the error ERR_CHART_INDICATOR_CANNOT_ADD (41 1 4) will occur.
To add an indicator to a new window, the window parameter must be by one greater than the index of
the last existing window, that is, equal to the CHART_WINDOWS_TOTAL property received via the
ChartGetInteger call. If the parameter value exceeds the value of
ChartGetInteger(ID,CHART_ WINDOWS_ TOTAL), a new window and indicator will not be created.
If an indicator is added to the main chart window, which should be drawn in a separate subwindow
(for example, a built-in iMACD or a custom indicator with the specified property #property
indicator_ separate_ window), then such an indicator may seem invisible, although it will be present
in the list of indicators. This usually means that the values of this indicator do not fall within the
displayed range of the price chart. The values of such an "invisible" indicator can be observed in the
Data window and read using functions from other MQL programs.
Adding an indicator to a chart increases the internal counter of its use due to its binding to the chart. If
the MQL program keeps its descriptor and it is no longer needed, then it is worth deleting it by calling
IndicatorRelease. This will actually decrease the counter, but the indicator will remain on the chart.
bool ChartIndicatorDelete(long chartId, int window, const string shortname)
The function removes the indicator with the specified short name from the window with the window
number on the chart with chartId. If there are several indicators with the same short name in the
specified chart subwindow, the first one in order will be deleted.

---

## Page 896

Part 5. Creating application programs
896
5.7 Working with charts
If other indicators are calculated using the values of the removed indicator on the same chart, they will
also be removed.
Deleting an indicator from a chart does not mean that its calculated part will also be deleted from the
terminal memory if the descriptor remains in the MQL program. To free the indicator handle, use the
IndicatorRelease function.
The ChartWindowFind function returns the number of the subwindow where the indicator is located.
There are 2 forms designed to search for the current indicator on its chart or an indicator with a given
short name on an arbitrary chart with the chartId identifier.
int ChartWindowFind()
int ChartWindowFind(long chartId, string shortname)
The second form can be used in scripts and Experts Advisors.
As a first example demonstrating these functions, let's consider the full version of the script
ChartList.mq5. We created and gradually refined it in the previous sections, up to the section Getting
the number and visibility of windows/subwindows. Compared to ChartList4.mq5 presented there, we will
add input variables to be able to list only charts with MQL programs and suppress the display of hidden
windows.
input bool IncludeEmptyCharts = true;
input bool IncludeHiddenWindows = true;
With the default value (true) the IncludeEmptyCharts parameter instructs to include all charts into the
list, including empty ones. The IncludeHiddenWindows parameter sets the display of hidden windows by
default. These settings correspond to the previous scripting logic ChartListN.
To calculate the total number of indicators and indicators in subwindows, we define the indicators and
subs variables.
void ChartList()
{
   ...
   int indicators = 0, subs = 0;
   ...
The working loop over the windows of the current chart has undergone major changes.

---

## Page 897

Part 5. Creating application programs
897
5.7 Working with charts
void ChartList()
{
      ...
      for(int i = 0; i < win; i++)
      {
         const bool visible = ChartGetInteger(id, CHART_WINDOW_IS_VISIBLE, i);
         if(!visible && !IncludeHiddenWindows) continue;
         if(!visible)
         {
            Print("  ", i, "/Hidden");
         }
         const int n = ChartIndicatorsTotal(id, i);
         for(int k = 0; k < n; k++)
         {
            if(temp == 0)
            {
               Print(header);
            }
            Print("  ", i, "/", k, " [I] ", ChartIndicatorName(id, i, k));
            indicators++;
            if(i > 0) subs++;
            temp++;
         }
      }
      ...
Here we have added ChartIndicatorsTotal and ChartIndicatorName calls. Now the list will mention MQL
programs of all types: [E] – Expert Advisors, [S] – scripts, [I] – indicators.
Here is an example of the log entries generated by the script for the default settings.
Chart List
N, ID, Symbol, TF, #subwindows, *active, Windows handle
0 132358585987782873 EURUSD M15 #1    133538
  1/0 [I] ATR(11)
1 132360375330772909 EURUSD D1     133514
2 132544239145024745 EURUSD M15   *   395646
 [S] ChartList
3 132544239145024732 USDRUB D1     395688
4 132544239145024744 EURUSD H1 #2  active  2361730
  1/0 [I] %R(14)
  2/Hidden
  2/0 [I] Momentum(15)
5 132544239145024746 EURUSD H1     133584
Total chart number: 6, with MQL-programs: 3
Experts: 0, Scripts: 1, Indicators: 3 (main: 0 / sub: 3)
If set both input parameters to false, we get a reduced list.

---

## Page 898

Part 5. Creating application programs
898
5.7 Working with charts
Chart List
N, ID, Symbol, TF, #subwindows, *active, Windows handle
0 132358585987782873 EURUSD M15 #1    133538
  1/0 [I] ATR(11)
2 132544239145024745 EURUSD M15   * active  395646
 [S] ChartList
4 132544239145024744 EURUSD H1 #2    2361730
  1/0 [I] %R(14)
Total chart number: 6, with MQL-programs: 3
Experts: 0, Scripts: 1, Indicators: 2 (main: 0 / sub: 2)
As a second example, let's consider an interesting script ChartIndicatorMove.mq5.
When running several indicators on a chart, we often may need to change the order of the indicators.
MetaTrader 5 does not have built-in tools for this, which forces you to delete some indicators and add
them again, while it is important to save and restore the settings. The ChartIndicatorMove.mq5 script
provides an option to automate this procedure. It is important to note that the script transfers only
indicators: if you need to change the order of subwindows along with graphical objects (if they are
inside), then you should use tpl templates.
The basis of operation of ChartIndicatorMove.mq5 is as follows. When the script is applied to a chart, it
determines to which window/subwindow it was added, and starts listing the indicators found there to
the user with a request to confirm the transfer. The user can agree, or continue the listing.
The direction of movement, up or down, is set in the MoveDirection input variable. The DIRECTION
enumeration will describe it.
#property script_show_inputs
   
enum DIRECTION
{
   Up = -1,
   Down = +1,
};
   
input DIRECTION MoveDirection = Up;
To transfer the indicator not to the neighboring subwindow but to the next one, that is, to actually swap
the subwindows with indicators in places (which is usually required), we introduce the j umpover input
variable.
input bool JumpOver = true;
The loop through the indicators of the target window obtained from ChartWindowOnDropped starts in
OnStart.

---

## Page 899

Part 5. Creating application programs
899
5.7 Working with charts
void OnStart()
{
   const int w = ChartWindowOnDropped();
   if(w == 0 && MoveDirection == Up)
   {
      Alert("Can't move up from window at index 0");
      return;
   }
   const int n = ChartIndicatorsTotal(0, w);
   for(int i = 0; i < n; ++i)
   {
      ...
   }
}
Inside the loop, we define the name of the next indicator, display a message to the user, and move the
indicator from one window to another using a sequence of the following manipulations:
• Get the handle by calling ChartIndicatorGet.
• Add it to the window above or below the current one via ChartIndicatorAdd, in accordance with the
selected direction, and when moving down, a new subwindow can be automatically created.
• Remove the indicator from the previous window with ChartIndicatorDelete.
• Release the descriptor, because we no longer need it in the program.
      ...
      const string name = ChartIndicatorName(0, w, i);
      const string caption = EnumToString(MoveDirection);
      const int button = MessageBox("Move '" + name + "' " + caption + "?",
         caption, MB_YESNOCANCEL);
      if(button == IDCANCEL) break;
      if(button == IDYES)
      {
         const int h = ChartIndicatorGet(0, w, name);
         ChartIndicatorAdd(0, w + MoveDirection, h);
         ChartIndicatorDelete(0, w, name);
         IndicatorRelease(h);
         break;
      }
      ...
The following image shows the result of swapping subwindows with indicators WPR and Momentum. The
script was launched by dropping it on the top sub-window with the WPR indicator, the direction of
movement was chosen downward (Down), jump (JumpOver) was enabled by default.

---

## Page 900

Part 5. Creating application programs
900
5.7 Working with charts
Swapping Indicators in subwindows
Please note that if you move the indicator from the subwindow to the main window, its charts will most
likely not be visible due to the values going beyond the displayed price range. If this happened by
mistake, you can use the script to transfer the indicator back to the subwindow.
5.7.23 Opening and closing charts
An MQL program can not only analyze the list of charts but also modify it: open new ones or close
existing ones. Two functions are allocated for these purposes: ChartOpen and ChartClose.
long ChartOpen(const string symbol, ENUM_TIMEFRAMES timeframe)
The function opens a new chart with the specified symbol and timeframe and returns the ID of the new
chart. If an error occurs during execution, the result is 0, and the error code can be read in the built-in
variable _ LastError.
If the symbol parameter is NULL, it means the symbol of the current chart (on which the MQL program
is being executed). The 0 value in the timeframe parameter corresponds to PERIOD_CURRENT.
The maximum possible number of simultaneously open charts in the terminal cannot exceed
CHARTS_MAX (1 00).
We will see an example of using the ChartOpen function in the next section, after studying the functions
for working with tpl templates.
Please note that the terminal allows you to create not only full-fledged windows with charts but also
chart objects. They are placed inside normal charts in the same way as other graphical objects
such as trend lines, channels, price labels, etc. Chart objects allow you to display within one
standard chart several small fragments of price series for alternative symbols and timeframes.

---

## Page 901

Part 5. Creating application programs
901 
5.7 Working with charts
bool ChartClose(long chartId = 0)
The function closes the chart with the specified ID (the default value of 0 means the current chart).
The function returns a success indicator.
As an example, let's implement the script ChartCloseIdle.mq5, which will close duplicate charts with
repeated symbol and timeframe combinations if they do not contain MQL programs and graphical
objects.
First, we need to make a list that counts the charts for a particular symbol/timeframe pair. This task is
implemented by the ChartIdleList function, which is very similar to what we saw in the script
ChartList.mq5. The list itself is formed in the map array MapArray<string,int> chartCounts.
#include <MQL5Book/Periods.mqh>
#include <MQL5Book/MapArray.mqh>
   
#define PUSH(A,V) (A[ArrayResize(A, ArraySize(A) + 1) - 1] = V)
   
void OnStart()
{
   MapArray<string,int> chartCounts;
   ulong duplicateChartIDs[];
   // collect duplicate empty charts
   if(ChartIdleList(chartCounts, duplicateChartIDs))
   {
      ...
   }
   else
   {
      Print("No idle charts.");
   }
}
Meanwhile, the ChartIdleList function fills the duplicateChartIDs array with identifiers of free charts that
match the closing conditions.

---

## Page 902

Part 5. Creating application programs
902
5.7 Working with charts
int ChartIdleList(MapArray<string,int> &map, ulong &duplicateChartIDs[])
{
   // list charts until their list ends
   for(long id = ChartFirst(); id != -1; id = ChartNext(id))
   {
      // skip objects
      if(ChartGetInteger(id, CHART_IS_OBJECT)) continue;
   
      // getting the main properties of the chart
      const int win = (int)ChartGetInteger(id, CHART_WINDOWS_TOTAL);
      const string expert = ChartGetString(id, CHART_EXPERT_NAME);
      const string script = ChartGetString(id, CHART_SCRIPT_NAME);
      const int objectCount = ObjectsTotal(id);
   
      // count the number of indicators
      int indicators = 0;
      for(int i = 0; i < win; ++i)      
      {
         indicators += ChartIndicatorsTotal(id, i);
      }
      
      const string key = ChartSymbol(id) + "/" + PeriodToString(ChartPeriod(id));
      
      if(map[key] == 0     // the first time we always read a new symbol/TF combination
                           // otherwise, only empty charts are counted:
         || (indicators == 0           // no indicators
            && StringLen(expert) == 0  // no Expert Advisor
            && StringLen(script) == 0  // no script
            && objectCount == 0))      // no objects
      {
         const int i = map.inc(key);
         if(map[i] > 1)                // duplicate
         {
            PUSH(duplicateChartIDs, id);
         }
      }
   }
   return map.getSize();
}
After the list for deletion is formed, in OnStart we call the ChartClose function in a loop over the list.

---

## Page 903

Part 5. Creating application programs
903
5.7 Working with charts
void OnStart()
{
   ...
   if(ChartIdleList(chartCounts, duplicateChartIDs))
   {
      for(int i = 0; i < ArraySize(duplicateChartIDs); ++i)
      {
         const ulong id = duplicateChartIDs[i];
         // request to bring the chart to the front
         ChartSetInteger(id, CHART_BRING_TO_TOP, true);
         // update the state of the windows, pumping the queue with the request
         ChartRedraw(id);
         // ask user for confirmation
         const int button = MessageBox(
            "Remove idle chart: "
            + ChartSymbol(id) + "/" + PeriodToString(ChartPeriod(id)) + "?",
            __FILE__, MB_YESNOCANCEL);
         if(button == IDCANCEL) break;   
         if(button == IDYES)
         {
            ChartClose(id);
         }
      }
      ...
For each chart, first the function ChartSetInteger(id, CHART_ BRING_ TO_ TOP, true) is called to show the
user which window is supposed to be closed. Since this function is asynchronous (only puts the
command to activate the window in the event queue), you need to additionally call ChartRedraw, which
processes all accumulated messages. The user is then prompted to confirm the action. The chart only
closes on clicking Yes. Selecting No skips the current chart (leaves it open), and the loop continues. By
pressing Cancel, you can interrupt the loop ahead of time.
5.7.24 Working with tpl chart templates
MQL5 API provides two functions for working with templates. Templates are files with the extension tpl,
which save the contents of the charts, that is, all their settings, along with plotted objects, indicators,
and an EA (if any).
bool ChartSaveTemplate(long chartId, const string filename)
The function saves the current chart settings to a tpl template with the specified name.
The chart is set by the chartId, 0 means the current graph.
The name of the file to save the template (filename) can be specified without the ".tpl" extension: it will
be added automatically. The default template is saved to the terminal_ dir/Profiles/Templates/ folder
and can then be used for manual application in the terminal. However, it is possible to specify not just a
name, but also a path relative to the MQL5 directory, in particular, starting with "/Files/". Thus, it will
be possible to open the saved template using files operation functions, analyze, and, if necessary, edit
them (see the example ChartTemplate.mq5 further along).
If a file with the same name already exists at the specified path, its contents will be overwritten.

---

## Page 904

Part 5. Creating application programs
904
5.7 Working with charts
We will look at a combined example for saving and applying a template a little later.
bool ChartApplyTemplate(long chartId, const string filename)
The function applies a template from the specified file to the chartId chart.
The template file is searched according to the following rules:
·If filename contains a path (starts with a backslash "\\" or a forward slash "/"), then the pattern is
matched relative to the path terminal_ data_ directory/MQL5.
·If there is no path in the name, the template is searched for in the same place where the
executable of the EX5 file is located, in which the function is called.
·If the template is not found in the first two places, it is searched for in the standard template folder
terminal_ dir/Profiles/Templates/.
Note that terminal_ data_ directory refers to the folder where modified files are stored, and its location
may vary depending on the type of the operating system, username, and computer security settings.
Normally it differs from the terminal_ dir folder although in some cases (for example, when working
under an account from the Administrators group), they may be the same. The location of folders
terminal_ data_ directory and terminal_ directory can be found using the TerminalInfoString function (see
constants TERMINAL_DATA_PATH and TERMINAL_PATH, respectively).
ChartApplyTemplate call creates a command that is added to the chart's message queue and is only
executed after all previous commands have been processed.
Loading a template stops all MQL programs running on the chart, including the one that initiated the
loading. If the template contains indicators and an Expert Advisor, its new instances will be launched.
For security purposes, when applying a template with an Expert Advisor to a chart, trading
permissions can be limited. If the MQL program that calls the ChartApplyTemplate function has no
permission to trade, then the Expert Advisor loaded using the template will have no permission to
trade, regardless of the template settings. If the MQL program that calls ChartApplyTemplate is
allowed to trade but trading is not allowed in the template settings, then the Expert Advisor loaded
using the template will not be allowed to trade.
An example of script ChartDuplicate.mq5 allows you to create a copy of the current chart.

---

## Page 905

Part 5. Creating application programs
905
5.7 Working with charts
void OnStart()
{
   const string temp = "/Files/ChartTemp";
   if(ChartSaveTemplate(0, temp))
   {
      const long id = ChartOpen(NULL, 0);
      if(!ChartApplyTemplate(id, temp))
      {
         Print("Apply Error: ", _LastError);
      }
   }
   else
   {
      Print("Save Error: ", _LastError);
   }
}
First, a temporary tpl file is created using ChartSaveTemplate, then a new chart is opened (ChartOpen
call), and finally, the ChartApplyTemplate function applies this template to the new chart.
However, in many cases, the programmer faces a more difficult task: not just apply the template but
pre-edit it.
Using templates, you can change many chart properties that are not available using other MQL5 API
functions, for example, the visibility of indicators in the context of timeframes, the order of indicator
subwindows along with the objects applied to them, etc.
The tpl file format is identical to the chr files used by the terminal for storing charts between sessions
(in the folder terminal_ directory/Profiles/Charts/profile_ name).
A tpl file is a text file with a special syntax. The properties in it can be a key=value pair written on a
single line or some kind of group containing several key=value properties. Such groups will be called
containers below because, in addition to individual properties, they can also contain other, nested
containers.
The container starts with a line that looks like "<tag>", where tag is one of the predefined container
types (see below), and ends with a pair of lines like "</tag>" (tag names must match). In other words,
the format is similar in some sense to XML (without a header), in which all lexical units must be written
on separate lines and tag properties are not indicated by their attributes (as in XML inside the opening
part "<tag attribute1 =value1 ...> "), but in the inner text of the tag.
The list of supported tags:
·chart – a root container with main chart properties and all subordinate containers;
·expert – a container with general properties of an Expert Advisor, for example, permission to trade
(inside a chart);
·window – a container with window/subwindow properties and its subordinate containers (inside
chart);
·object – a container with graphical object properties (inside window);
·indicator – a container with indicator properties (inside window);
·graph – a container with indicator chart properties (inside indicator);
·level – a container with indicator level properties (inside indicator);

---

## Page 906

Part 5. Creating application programs
906
5.7 Working with charts
·period – a container with visibility properties of an object or indicator on a specific timeframe
(inside an object or indicator);
·inputs – a container with settings (input variables) of custom indicators and Expert Advisors.
The possible list of properties in key=value pairs is quite extensive and has no official documentation. If
necessary, you can deal with these features of the platform yourself.
Here are fragments from one tpl file (the indents in the formatting are made to visualize the nesting of
containers).

---

## Page 907

Part 5. Creating application programs
907
5.7 Working with charts
<chart>
id=0
symbol=EURUSD
description=Euro vs US Dollar
period_type=1
period_size=1
digits=5
...
<window>
  height=117.133747
  objects=0
  <indicator>
    name=Main
    path=
    apply=1
    show_data=1
    ...
    fixed_height=-1
  </indicator>
</window>
<window>
  <indicator>
    name=Momentum
    path=
    apply=6
    show_data=1
    ...
    fixed_height=-1
    period=14
    <graph>
      name=
      draw=1
      style=0
      width=1
      color=16748574
    </graph>
  </indicator>
  ...
</window>
</chart>
We have the TplFile.mqh header file for working with tpl files, with which you can analyze and modify
templates. It has two classes:
·Container – to read and store file elements, taking into account the hierarchy (nesting), as well as
to write to a file after possible modification;
·Selector – to sequentially traverse the elements of the hierarchy (Container objects) in search of a
match with a certain query that writes as a string similar to an xpath selector
("/path/element[attribute=value]").
Objects of the Container class are created using a constructor that takes the tpl file descriptor to read
as the first parameter and the tag name as the second parameter. By default, the tag name is NULL,

---

## Page 908

Part 5. Creating application programs
908
5.7 Working with charts
which means the root container (the entire file). Thus, the container itself fills itself with content in the
process of reading the file (see the read method).
The properties of the current element, that is, the "key=value" pairs located directly inside this
container, are supposed to be added to the map MapArray<string,string> properties. Nested containers
are added to the array Container *children[].

---

## Page 909

Part 5. Creating application programs
909
5.7 Working with charts
#include <MQL5Book/MapArray.mqh>
   
#define PUSH(A,V) (A[ArrayResize(A, ArraySize(A) + 1) - 1] = V)
   
class Container
{
   MapArray<string,string> properties;
   Container *children[];
   const string tag;
   const int handle;
public:
   Container(const int h, const string t = NULL): handle(h), tag(t) { }
   ~Container()
   {
      for(int i = 0; i < ArraySize(children); ++i)
      {
         if(CheckPointer(children[i]) == POINTER_DYNAMIC) delete children[i];
      }
   }
      
   bool read(const bool verbose = false)
   {
      while(!FileIsEnding(handle))
      {
         string text = FileReadString(handle);
         const int len = StringLen(text);
         if(len > 0)
         {
            if(text[0] == '<' && text[len - 1] == '>')
            {
               const string subtag = StringSubstr(text, 1, len - 2);
               if(subtag[0] == '/' && StringFind(subtag, tag) == 1)
               {
                  if(verbose)
                  {
                     print();
                  }
                  return true;       // the element is ready
               }
               
               PUSH(children, new Container(handle, subtag)).read(verbose);
            }
            else
            {
               string pair[];
               if(StringSplit(text, '=', pair) == 2)
               {
                  properties.put(pair[0], pair[1]);
               }
            }
         }

---

## Page 910

Part 5. Creating application programs
91 0
5.7 Working with charts
      }
      return false;
   }
   ...
};
In the read method, we read and parse the file line by line. If the opening tag of the form "<tag>", we
create a new container object and continue reading in it. If a closing tag of the form "</tag>" with the
same name, we return a flag of success (true) which means that the container has been generated. In
the remaining lines, we read the "key=value" pairs and add them to the properties array.
We have prepared the Selector to search for elements in a template. A string with the hierarchy of the
searched tags is passed to its constructor. For example, the string "/chart/window/indicator"
corresponds to a chart that has a window/subwindow, which, in turn, contains any indicator. The
search result will be the first match. This query, as a rule, will find the quotes chart, because it is
stored in the template as an indicator named "Main" and goes at the beginning of the file, before other
subwindows.
More practical queries specifying the name and value of a particular attribute. In particular, the
modified string "/chart/window/indicator[name=Momentum]" will only look for the Momentum
indicator. This search is different from calling ChartWindowFind, because here the name is specified
without parameters, while ChartWindowFind uses a short name of the indicator, which usually includes
parameter values, but they can vary.
For built-in indicators, the name property contains the name itself, and for custom ones, it will say
"Custom Indicator". The link to the custom indicator is given in the path property as a path to the
executable file, for example, "Indicators\MQL5Book\IndTripleEMA.ex5".
Let's see the internal structure of the Selector class.
class Selector
{
   const string selector;
   string path[];
   int cursor;
public:
   Selector(const string s): selector(s), cursor(0)
   {
      StringSplit(selector, '/', path);
   }
   ...
In the constructor, we decompose the selector query into separate components and save them in the
path array. The current path component that is being matched in the pattern is given by the cursor
variable. At the beginning of the search, we are in the root container (we are considering the entire tpl
file), and the cursor is 0. As matches are found, cursor should increase (see the accept method below).
The operator [], with the help of which you can get the i-th fragment of the path, is overloaded in the
class. It also takes into account that in the fragment, in square brackets, the pair "[key=value]" can
be specified.

---

## Page 911

Part 5. Creating application programs
91 1 
5.7 Working with charts
   string operator[](int i) const
   {
      if(i < 0 || i >= ArraySize(path)) return NULL;
      const int param = StringFind(path[i], "[");
      if(param > 0)
      {
         return StringSubstr(path[i], 0, param);
      }
      return path[i];
   }
   ...
The accept method checks if the element name (tag) and its properties (properties) match with the
data specified in the selector path for the current cursor position. The this[cursor] record uses the
above overload of the operator [] .
   bool accept(const string tag, MapArray<string,string> &properties)
   {
      const string name = this[cursor];
      if(!(name == "" && tag == NULL) && (name != tag))
      {
         return false;
      }
      
      // if the request has a parameter, check it among the properties
      // NB! so far only one attribute is supported, but many "tag[a1=v1][a2=v2]..." are needed
      const int start = StringLen(path[cursor]) > 0 ? StringFind(path[cursor], "[") : 0;
      if(start > 0)
      {
         const int stop = StringFind(path[cursor], "]");
         const string prop = StringSubstr(path[cursor], start + 1, stop - start - 1);
         
         // NB! only '=' is supported, but it should be '>', '<', etc.
         string kv[];   // key and value
         if(StringSplit(prop, '=', kv) == 2)
         {
            const string value = properties[kv[0]];
            if(kv[1] != value)
            {
               return false;
            }
         }
      }
      
      cursor++;
      return true;
   }
   ...
The method will return false if the tag name does not match the current fragment of the path, and also
if the fragment contained the value of some parameter and it is not equal or is not in the array

---

## Page 912

Part 5. Creating application programs
91 2
5.7 Working with charts
properties. In other cases, we will get a match of the conditions, as a result of which the cursor will
move forward (cursor++) and the method will return true.
The search process will be completed successfully when the cursor reaches the last fragment in the
request, so we need a method to determine this moment, which is isComplete.
   bool isComplete() const
   {
      return cursor == ArraySize(path);
   }
   
   int level() const
   {
      return cursor;
   }
Also, during the template analysis, there may be situations when we went through the container
hierarchy part of the path (that is, found several matches), after which the next request fragment did
not match. In this case, you need to "return" to the previous levels of the request, for which the
method unwind is implemented.
   bool unwind()
   {
      if(cursor > 0)
      {
         cursor--;
         return true;
      }
      return false;
   }
};
Now everything is ready to organize the search in the hierarchy of containers (which we get after
reading the tpl file) using the Selector object. All necessary actions will be performed by the find
method in the Container class. It takes the Selector object as an input parameter and recursively calls
itself while there are matches according to the method Selector::accept. Reaching the end of the
request means success, and the find method will return the current container to the calling code.

---

## Page 913

Part 5. Creating application programs
91 3
5.7 Working with charts
   Container *find(Selector *selector)
   {
      const string element = StringFormat("%*s", 2 * selector.level(), " ")
         + "<" + tag + "> " + (string)ArraySize(children);
      if(selector.accept(tag, properties))
      {
         Print(element + " accepted");
         
         if(selector.isComplete())
         {
            return &this;
         }
         
         for(int i = 0; i < ArraySize(children); ++i)
         {
            Container *c = children[i].find(selector);
            if(c) return c;
         }
         selector.unwind();
      }
      else
      {
         Print(element);
      }
      
      return NULL;
   }
   ...
Note that as we move along the object tree, the find method logs the tag name of the current object
and the number of nested objects, and does so with an indent proportional to the nesting level of the
objects. If the item matches the request, the log entry is appended with the word "accepted".
It is also important to note that this implementation returns the first matching element and does not
continue searching for other candidates, and in theory, this can be useful for templates because they
often have several tags of the same type inside the same container. For example, a window may
contain many objects, and an MQL program may be interested in parsing the entire list of objects. This
aspect is proposed to be studied optionally.
To simplify the search call, a method of the same name has been added that takes a string parameter
and creates the Selector object locally.
   Container *find(const string selector)
   {
      Selector s(selector);
      return find(&s);
   }
Since we are going to edit the template, we should provide methods for modifying the container, in
particular, to add a key=value pair and a new nested container with a given tag.

---

## Page 914

Part 5. Creating application programs
91 4
5.7 Working with charts
   void assign(const string key, const string value)
   {
      properties.put(key, value);
   }
   
   Container *add(const string subtag)
   {
      return PUSH(children, new Container(handle, subtag));
   }
   
   void remove(const string key)
   {
      properties.remove(key);
   }
After editing, you will need to write the contents of the containers back to a file (same or different). A
helper method save saves the object in the tpl format described above: starts with the opening tag
"<tag>", continues by unloading all key=value properties, and calls save for nested objects, after which
it ends with the closing tag "</tag>". The file descriptor is passed as a parameter for saving.
   bool save(const int h)
   {
      if(tag != NULL)
      {
         if(FileWriteString(h, "<" + tag + ">\n") <= 0)
            return false;
      }
      for(int i = 0; i < properties.getSize(); ++i)
      {
         if(FileWriteString(h, properties.getKey(i) + "=" + properties[i] + "\n") <= 0)
            return false;
      }
      for(int i = 0; i < ArraySize(children); ++i)
      {
         children[i].save(h);
      }
      if(tag != NULL)
      {
         if(FileWriteString(h, "</" + tag + ">\n") <= 0)
            return false;
      }
      return true;
   }
The high-level method of writing an entire template to a file is called write. Its input parameter (file
descriptor) can be equal to 0, which means writing to the same file from which it was read. However,
the file must be opened with permission to write.
It is important to note that when overwriting a Unicode text file, MQL5 does not write the initial UTF
mark (the so-called BOM, Byte Order Mark), and therefore we have to do it ourselves. Otherwise,
without the mark, the terminal will not read and apply our template.

---

## Page 915

Part 5. Creating application programs
91 5
5.7 Working with charts
If the calling code passes in the h parameter another file opened exclusively for writing in Unicode
format, MQL5 will write the BOM automatically.
   bool write(int h = 0)
   {
      bool rewriting = false;
      if(h == 0)
      {
         h = handle;
         rewriting = true;
      }
      if(!FileGetInteger(h, FILE_IS_WRITABLE))
      {
         Print("File is not writable");
         return false;
      }
      
      if(rewriting)
      {
         // NB! We write the BOM manually because MQL5 does not do this when overwritten
         ushort u[1] = {0xFEFF};
         FileSeek(h, SEEK_SET, 0);
         FileWriteString(h, ShortArrayToString(u));
      }
      
      bool result = save(h);
      
      if(rewriting)
      {
         // NB! MQL5 does not allow to reduce file size,
         // so we fill in the extra ending with spaces
         while(FileTell(h) < FileSize(h) && !IsStopped())
         {
            FileWriteString(h, " ");
         }
      }
      return result;
   }
To demonstrate the capabilities of the new classes, consider the problem of hiding the window of a
specific indicator. As you know, the user can achieve this by resetting the visibility flags for timeframes
in the indicator properties dialog (tab Display). Programmatically, this cannot be done directly. This is
where the ability to edit the template comes to the rescue.
In the template, indicator visibility for timeframes is specified in the container <indicator>, inside which
a separate container is written for each visible timeframe <period>. For example, visibility on the M1 5
timeframe looks like this:

---

## Page 916

Part 5. Creating application programs
91 6
5.7 Working with charts
<period>
period_type=0
period_size=15
</period>
Inside the container <period> properties period_ type and period_ size are used. period_ type is a unit of
measurement, one of the following:
·0 for minutes
·1  for hours
·2 for weeks
·3 for months
period_ size is the number of measurement units in the timeframe. It should be noted that the daily
timeframe is designated as 24 hours.
When there is no nested container <period> in the container <indicator>, the indicator is displayed on
all timeframes.
The book comes with the script ChartTemplate.mq5, which adds the Momentum indicator to the chart
(if it is not already present) and makes it visible on a single monthly timeframe.
void OnStart()
{
   // if Momentum(14) is not on the chart yet, add it
   const int w = ChartWindowFind(0, "Momentum(14)");
   if(w == -1)
   {
      const int momentum = iMomentum(NULL, 0, 14, PRICE_TYPICAL);
      ChartIndicatorAdd(0, (int)ChartGetInteger(0, CHART_WINDOWS_TOTAL), momentum);
      // not necessarily here because the script will exit soon,
      // however explicitly declares that the handle will no longer be needed in the code
      IndicatorRelease(momentum);
   }
   ...
Next, we save the current chart template to a file, which we then open for writing and reading. It would
be possible to allocate a separate file for writing.
   const string filename = _Symbol + "-" + PeriodToString(_Period) + "-momentum-rw";
   if(PRTF(ChartSaveTemplate(0, "/Files/" + filename)))
   {
      int handle = PRTF(FileOpen(filename + ".tpl",
         FILE_READ | FILE_WRITE | FILE_TXT | FILE_SHARE_READ | FILE_SHARE_WRITE));
      // alternative - another file open for writing only
      // int writer = PRTF(FileOpen(filename + "w.tpl",
      //    FILE_WRITE | FILE_TXT | FILE_SHARE_READ | FILE_SHARE_WRITE));
Having received a file descriptor, we create a root container main and read the entire file into it (nested
containers and all their properties will be read automatically).

---

## Page 917

Part 5. Creating application programs
91 7
5.7 Working with charts
      Container main(handle);
      main.read();
Then we define a selector to search for the Momentum indicator. In theory, a more rigorous approach
would also require checking the specified period (1 4), but our classes do not support querying multiple
properties at the same time (this possibility is left for independent study).
Using the selector, we search, print the found object (just for reference) and add its nested container
<period> with settings for displaying the monthly timeframe.
      Container *found = main.find("/chart/window/indicator[name=Momentum]");
      if(found)
      {
         found.print();
         Container *period = found.add("period");
         period.assign("period_type", "3");
         period.assign("period_size", "1");
      }
Finally, we write the modified template to the same file, close it and apply it on the chart.
      main.write(); // or main.write(writer);
      FileClose(handle);
      
      PRTF(ChartApplyTemplate(0, "/Files/" + filename));
   }
}
When running the script on a clean chart, we will see such entries in the log.

---

## Page 918

Part 5. Creating application programs
91 8
5.7 Working with charts
ChartSaveTemplate(0,/Files/+filename)=true / ok
FileOpen(filename+.tpl,FILE_READ|FILE_WRITE|FILE_TXT| »
» FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_UNICODE)=1 / ok
 <> 1 accepted
  <chart> 2 accepted
    <window> 1 accepted
      <indicator> 0
    <window> 1 accepted
      <indicator> 1 accepted
Tag: indicator
                    [key]    [value]
[ 0] "name"               "Momentum"
[ 1] "path"               ""        
[ 2] "apply"              "6"       
[ 3] "show_data"          "1"       
[ 4] "scale_inherit"      "0"       
[ 5] "scale_line"         "0"       
[ 6] "scale_line_percent" "50"      
[ 7] "scale_line_value"   "0.000000"
[ 8] "scale_fix_min"      "0"       
[ 9] "scale_fix_min_val"  "0.000000"
[10] "scale_fix_max"      "0"       
[11] "scale_fix_max_val"  "0.000000"
[12] "expertmode"         "0"       
[13] "fixed_height"       "-1"      
[14] "period"             "14"      
ChartApplyTemplate(0,/Files/+filename)=true / ok
It can be seen here that before finding the required indicator (marked "accepted"), the algorithm found
the indicator in the previous, main window, but it did not fit, because its name is not equal to the
desired "Momentum".
Now, if you open the list of indicators on the chart, there will be momentum, and in its properties
dialog, on the Display tab the only enabled timeframe is Month.
The book is accompanied by an extended version of the file TplFileFull.mqh, which supports different
comparison operations in the conditions for selecting tags and their multiple selection into arrays. An
example of using it can be found in the script ChartUnfix.mq5, which unfixes the sizes of all chart
subwindows.
5.7.25 Saving a chart image
In MQL programs, it often becomes necessary to document the current state of the program itself and
the trading environment. As a rule, for this purpose, the output of various analytical or financial
indicators to the journal is used, but some things are more clearly represented by the image of the
graph, for example, at the time of the transaction. The MQL5 API includes a function that allows you to
save a chart image to a file.

---

## Page 919

Part 5. Creating application programs
91 9
5.7 Working with charts
bool ChartScreenShot(long chartId, string filename, int width, int height,
   ENUM_ALIGN_MODE alignment = ALIGN_RIGHT)
The function takes a snapshot of the specified chart in GIF, PNG, or BMP format depending on the
extension in the line with the name of the file filename (maximum 63 characters). The screenshot is
placed in the directory MQL5/Files.
Parameters width and height set the width and height of the image in pixels.
Parameter alignment affects what part of the graph will be included in the file. The value ALIGN_RIGHT
(default) means that the snapshot is taken for the most recent prices (this can be thought of as the
terminal silently making a transition on pressing End before the snapshot). The ALIGN_LEFT value
ensures that bars are hit on the image, starting from the first bar visible on the left at the moment.
Thus, if you need to take a screenshot of a chart from a certain position, you must first position the
chart manually or using the ChartNavigate function.
The ChartScreenShot function returns true in case of success.
Let's test the function in the script ChartPanorama.mq5. Its task is to save a copy of the chart from
the current left visible bar up to the current time. By first shifting the beginning of the graph back to
the desired depth of history, you can get a fairly extended panorama. In this case, you do not need to
think about what width of the image to choose. However, keep in mind that a story that is too long will
require a huge image, potentially exceeding the capabilities of the graphics format or software.
The height of the image will automatically be determined equal to the current height of the chart.

---

## Page 920

Part 5. Creating application programs
920
5.7 Working with charts
void OnStart()
{
   // the exact width of the price scale is not known, we take it empirically
   const int scale = 60;
   
   // calculating the total height, including gaps between windows
   const int w = (int)ChartGetInteger(0, CHART_WINDOWS_TOTAL);
   int height = 0;
   int gutter = 0;
   for(int i = 0; i < w; ++i)
   {
      if(i == 1)
      {
         gutter = (int)ChartGetInteger(0, CHART_WINDOW_YDISTANCE, i) - height;
      }
      height += (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, i);
   }
   
   Print("Gutter=", gutter, ", total=", gutter * (w - 1));
   height += gutter * (w - 1);
   Print("Height=", height);
   
   // calculate the total width based on the number of pixels in one bar,
   // and also including chart offset from the right edge and scale width
   const int shift = (int)(ChartGetInteger(0, CHART_SHIFT) ?
      ChartGetDouble(0, CHART_SHIFT_SIZE) * ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) / 100 : 0);
   Print("Shift=", shift);
   const int pixelPerBar = (int)MathRound(1.0 * ChartGetInteger(0, CHART_WIDTH_IN_PIXELS)
      / ChartGetInteger(0, CHART_WIDTH_IN_BARS));
   const int width = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR) * pixelPerBar + scale + shift;
   Print("Width=", width);
   
   // write a file with a picture in PNG format
   const string filename = _Symbol + "-" + PeriodToString() + "-panorama.png";
   if(ChartScreenShot(0, filename, width, height, ALIGN_LEFT))
   {
      Print("File saved: ", filename);
   }
}
We could also use the ALIGN_RIGHT mode, but then we would have to force the offset from the right
edge to be disabled, because it is recalculated for the image, depending on its size, and the result will
look completely different from what it looks like on the screen (the indent on the right will become too
large, since it is specified as a percentage of the width).
Below is an example of the log after running the script on the chart XAUUSD,H1 .

---

## Page 921

Part 5. Creating application programs
921 
5.7 Working with charts
Gutter=2, total=2
Height=440
Shift=74
Width=2086
File saved: XAUUSD-H1-panorama.png
Taking into account navigation to a not very distant history, the following screenshot was obtained
(represented as a 4-fold reduced copy).
Chart Panorama
5.8 Graphical objects
MetaTrader 5 users are well aware of the concept of graphical objects: trend lines, price labels,
channels, Fibonacci levels, geometric shapes, and many other visual elements that are used for the
analytical chart markup. The MQL5 language allows you to create, edit, and delete graphical objects
programmatically. This can be useful, for example, when it is desirable to display certain data
simultaneously in a subwindow and the main window of an indicator. Since the platform only supports
the output of indicator buffers in one window, we can generate objects in the other window. With the
markup created from graphical objects, it is easy to organize semi-automated trading using Expert
Advisors. Additionally, objects are often used to build custom graphical interfaces for MQL programs,
such as buttons, input fields, and flags. These programs can be controlled without opening the
properties dialog, and the panels created in MQL can offer much greater flexibility than standard input
variables.  
Each object exists in the context of a particular chart. That's why the functions we will discuss in this
chapter share a common characteristic: the first parameter specifies the chart ID. In addition, each
graphical object is characterized by a name that is unique within one chart, including all subwindows.
Changing the name of a graphical object involves deleting the object with the old name and creating the
same object with a new name. You cannot create two objects with the same name.
The functions that define the properties of graphical objects, as well as the operations of creating
(Obj ectCreate) and moving (Obj ectMove) objects on the chart, essentially serve to send asynchronous
commands to the chart. If these functions are successfully executed, the command enters the shared
event queue of the chart. The visual modification of the properties of graphical objects occurs during
the processing of the event queue for that particular chart. Therefore, the external representation of
the chart may reflect the changed state of objects with some delay after the function calls.
In general, the update of graphical objects on the chart is done automatically by the terminal in
response to chart-related events such as receiving a new quote, resizing the window, and so on. To
force the update of graphical objects, you can use the function for requesting chart redraw
(ChartRedraw). This is particularly important after mass creation or modification of objects.
Objects serve as a source of programmatic events, such as creation, deletion, modification of their
properties, and mouse clicks. All aspects of event occurrence and handling are discussed in a separate
chapter, along with events in the general window context.

---

## Page 922

Part 5. Creating application programs
922
5.8 Graphical objects
We will begin with the theoretical foundations and gradually move on to practical aspects.
5.8.1  Object types and features of specifying their coordinates
As we know from the chapter on charts, there are two coordinate systems in the window: screen (pixel)
coordinates and quote (time and price) coordinates. In this regard, the total set of supported types of
objects is divided into two large groups: those objects that are linked to the screen, and those that are
linked to the price chart. The first ones always remain in place relative to one of the corners of the
window (which corner is the reference one is determined by the user or programmer in the object
properties). The latter are scrolled along with the working area of the window.
The following image shows two objects with text labels for comparison: one attached to the screen
(OBJ_LABEL), and the other to the price chart (OBJ_TEXT). Their types, given in brackets, as well as
the properties by which coordinates are set, we will study in the relevant sections of this chapter. It is
important to note that when scrolling the price chart, the text OBJ_TEXT moves synchronously with it,
while the inscription OBJ_LABEL remains in the same place.
Two different coordinate systems for objects
Also, the objects differ in the number of anchor points. For example, a single price label ("arrow")
requires one time/price point, and a trend line requires two such points. There are object types with
more anchor points, such as Equidistant Channels, Triangles, or Elliott Waves.
When an object is selected (for example, in the Obj ect List dialog, by double-clicking or single-clicking
on the chart, depending on the Charts tab / Select obj ects with a single mouse click option), its anchor
points are indicated by small squares in a contrasting color. It is the anchor points that are used to
drag the object and to change its size and orientation.

---

## Page 923

Part 5. Creating application programs
923
5.8 Graphical objects
All supported object types are described in the ENUM_OBJECT enumeration. You can read it in its
entirety in the MQL5 documentation. We will consider its elements gradually, in parts.
5.8.2 Time and price bound objects
The following table provides objects with their time and price coordinates, their identifiers in the
ENUM_OBJECT enumeration, and the number of anchor points. 
Identifier
Name
A n ch or  p oi n ts
Single straight lines
OBJ_VLINE
Vertical (time coordinate only)
1
OBJ_HLINE
Horizontal (price coordinate only)
1
OBJ_TREND
Trend
2
OBJ_ARROWED_LINE
With an arrow at the end
2
Periodically repeating vertical lines
OBJ_CYCLES
Cyclic
2
Channels
OBJ_CHANNEL
Equidistant
3
OBJ_STDDEVCHANNEL
Standard deviation
2
OBJ_REGRESSION
Linear regression
2
OBJ_PITCHFORK
Andrews' pitchfork
3
Fibonacci Tools
OBJ_FIBO
Levels
2
OBJ_FIBOTIMES
Time zones
2
OBJ_FIBOFAN
Fan
2
OBJ_FIBOARC
Arcs
2
OBJ_FIBOCHANNEL
Channel
3
OBJ_EXPANSION
Expansion
3
Gann Tools
OBJ_GANNLINE
Line
2
OBJ_GANNFAN
Fan
2
OBJ_GANNGRID
Net
2
Elliot waves

---

## Page 924

Part 5. Creating application programs
924
5.8 Graphical objects
Identifier
Name
A n ch or  p oi n ts
OBJ_ELLIOTWAVE5
Impulse
5
OBJ_ELLIOTWAVE3
Corrective
3
Shapes
OBJ_RECTANGLE
Rectangle
2
OBJ_TRIANGLE
Triangle
3
OBJ_ELLIPSE
Ellipse
3
Single marks and labels
O B J_AR R O W _TH U M B _U P 
Thumbs up
1 *
O B J_AR R O W _TH U M B _D O W N 
Thumbs down
1 *
O B J_AR R O W _U P 
Up arrow
1 *
O B J_AR R O W _D O W N 
Down arrow
1 *
O B J_AR R O W _S TO P 
Stop mark
1 *
O B J_AR R O W _CH E CK 
Check mark
1 *
O B J_AR R O W _L E F T_P R ICE 
Left price label
1
O B J_AR R O W _R IG H T_P R ICE 
Right price label
1
O B J_AR R O W _B U Y
Buy sign (blue up arrow)
1
O B J_AR R O W _S E L L 
Sell sign (red down arrow)
1
O B J_AR R O W 
Arbitrary Wingdings character
1 *
Text and graphics
OBJ_TEXT
Text
1 *
OBJ_BITMAP
Picture
1 *
Events
OBJ_EVENT
Timestamp at the bottom of the main window
(time coordinate only)
1
An asterisk marks those objects for which it is allowed to select an anchor point on the object (for
example, in one of the corners of the object or in the middle of one of the sides). The selection methods
can vary for different object types, and the details will be outlined in the section on Defining the object
anchor point. Anchor points are required because objects have a certain size, and there would be
positional ambiguity without them.

---

## Page 925

Part 5. Creating application programs
925
5.8 Graphical objects
5.8.3 Objects bound to screen coordinates
The following table lists the names and ENUM_OBJECT identifiers of objects positioned based on the
screen coordinates. Almost all of them, except for the chart object, are designed to create a user
interface for programs. In particular, there are such basic controls as a button and an input field, as
well as labels and panels for visual grouping of objects. Based on them, you can create more complex
controls (for example, drop-down lists or checkboxes). Together with the terminal, a class library with
ready-made controls is supplied as a set of header files (see the MQL5/Include/Controls directory).
Identifier
Name
Setting 
anchor point
OBJ_LABEL
Text label
Yes
OBJ_RECTANGLE_LABEL
Rectangular panel
OBJ_BITMAP_LABEL
Panel with an image
Yes
OBJ_BUTTON
Button
OBJ_EDIT
Input field
OBJ_CHART
Chart object
All these objects require the determining of the anchor corner in the chart window. By default, their
coordinates are relative to the upper left corner of the window.
The types in this list also use an anchor point on the object, and only one. It is editable in some objects
and is hard-coded in others. For example, a rectangular panel, a button, an input field, and a chart
object are always anchored at their top left corner. And for a label or a panel with a picture, many
options are available. The choice is made from the ENUM_ANCHOR_POINT enumeration described in
the section on Defining the object anchor point.
The text label (OBJ_LABEL) provides text output without the possibility of editing it. For editing, use the
input field (OBJ_EDIT).
5.8.4 Creating objects
To create an object, a certain minimum set of attributes is required that is common to all types.
Additional properties specific to each type can be set or changed later on an already existing object.
The required attributes include the identifier of the chart where the object should be created, the name
of the object, the number of the window/subwindow, and two coordinates for the first anchor point:
time and price.
Even though there is a group of objects positioned in screen coordinates, creating them still requires
you to pass two values, usually zero because they aren't used.
In general, a prototype of the Obj ectCreate function looks as follows:

---

## Page 926

Part 5. Creating application programs
926
5.8 Graphical objects
bool ObjectCreate(long chartId, const string name, ENUM_OBJECT type, int window,
   datetime time1 , double price1 , datetime time2 = 0, double price2 = 0, ...)
A value of 0 for chartId implies the current chart. The name parameter parameter must be unique
within the entire chart, including subwindows, and should not exceed 63 characters.
We have given in the previous sections object types for the type parameter: these are the elements of
the ENUM_OBJECT enumeration.
As we know, the numbering of windows/subwindows for the window parameter starts from 0, which
means the main chart window. If a larger index is specified for a subwindow, it must exist, as otherwise,
the function will terminate with an error and return false.
Just to remind you, the returned success flag (true) only indicates that the command to create the
object has been successfully placed in the queue. The result of its execution is not immediately known.
This is the flip side of the asynchronous call, which is employed to enhance performance.
To check the execution result, you can use the Obj ectFind function or any ObjectGet functions, which
query the properties of an object. But you should keep in mind that such functions wait for the
execution of the entire queue of chart commands and only then return the actual result (the state of
the object). This process may take some time, during which the MQL program code will be suspended.
In other words, the functions for checking the state of objects are synchronous, unlike the functions for
creating and modifying objects.
Additional anchor points, starting with the second one, are optional. The allowed number of anchor
points, up to 30, is provided for future use, and no more than 5 are used in current object types.
It is important to note that the call to the Obj ectCreate function with the name of an already existing
object simply changes the anchor point(s) (if the coordinates have been changed since the previous
call). This is convenient to use for writing unified code without branching into conditions based on the
presence or absence of an object. In other words, an unconditional Obj ectCreate call guarantees the
existence of the object, if we do not care whether it existed before or not. However, there is a nuance.
If, when calling Obj ectCreate, the object type or the subwindow index is different from an already
existing object, the relevant data remains the same, while no errors occur.
When calling Obj ectCreate, you can leave all anchor points with default values (null), provided that
ObjectSet functions with the appropriate OBJPROP_TIME and OBJPROP_PRICE properties are called
after this instruction.
The order in which anchor points are specified can be important for some object types. For
channels such as OBJ_REGRESSION (Linear Regression Channel) and OBJ_STDDEVCHANNEL
(Standard Deviation Channel), it is mandatory for the conditions time1 <time2 to be met. Otherwise,
the channel will not be built normally, although the object will be created without errors.
As an example of the function, let's take the Obj ectSimpleShowcase.mq5 script which creates several
objects of different types on the last bars of the chart, requiring a single anchor point.
All examples of working with objects will use the Obj ectPrefix.mqh header file, which contains a string
definition with a common prefix for object names. Thus, it will be more convenient for us, if necessary,
to clear the charts from "its own" objects.
const string ObjNamePrefix = "ObjShow-";
In the OnStart function, an array is defined containing object types.

---

## Page 927

Part 5. Creating application programs
927
5.8 Graphical objects
void OnStart()
{
   ENUM_OBJECT types[] =
   {
      // straight lines
      OBJ_VLINE, OBJ_HLINE,
      // labels (arrows and other signs)
      OBJ_ARROW_THUMB_UP, OBJ_ARROW_THUMB_DOWN,
      OBJ_ARROW_UP, OBJ_ARROW_DOWN,
      OBJ_ARROW_STOP, OBJ_ARROW_CHECK,
      OBJ_ARROW_LEFT_PRICE, OBJ_ARROW_RIGHT_PRICE,
      OBJ_ARROW_BUY, OBJ_ARROW_SELL,
      // OBJ_ARROW, // see the ObjectWingdings.mq5 example
      
      // text
      OBJ_TEXT,
      // event flag (like in a calendar) at the bottom of the window
      OBJ_EVENT,
   };
Next, in the loop through its elements, we create objects in the main window, passing the time and
closing price of the i-th bar.
   const int n = ArraySize(types);
   for(int i = 0; i < n; ++i)
   {
      ObjectCreate(0, ObjNamePrefix + (string)iTime(_Symbol, _Period, i), types[i],
         0, iTime(_Symbol, _Period, i), iClose(_Symbol, _Period, i));
   }
   
   PrintFormat("%d objects of various types created", n);
}
Here's the possible result from running the script.

---

## Page 928

Part 5. Creating application programs
928
5.8 Graphical objects
Objects of simple types at the closing points of the last bars
The drawing of lines by the Close price and the grid display are enabled in this example. We will learn
how to adjust the size, color, and other attributes of objects later. In particular, the anchor points of
most icons are located by default in the middle of the top side, so they are visually offset under the
line. However, the sell icon is above the line because the anchor point is always in the middle of the
bottom side.
Please note that objects created programmatically are not displayed by default in the list of objects
in the dialog of the same name. To see them there, click the All button.
5.8.5 Deleting objects
The MQL5 API offers two functions for deleting objects. For bulk deletion of objects that meet the
conditions for a prefix in the name, type, or subwindow number, use Obj ectsDeleteAll. If you need to
select the objects to be deleted by some other criteria (for example, by an outdated date and time
coordinate), or if this is a single object, use the Obj ectDelete function.
The Obj ectsDeleteAll function has two forms: with a parameter for the name prefix and without it.
int ObjectsDeleteAll(long chartId, int window = -1 , int type = -1 )
int ObjectsDeleteAll(long chartId, const string prefix, int window = -1 , int type = -1 )
The function deletes all objects on the chart with the specified chartId, taking into account the
subwindow, type, and initial substring in the name.
Value 0 in the chartId parameter represents the current chart, as usual.
Default values (-1 ) in the window and type parameters define all subwindows and all types of objects,
respectively.

---

## Page 929

Part 5. Creating application programs
929
5.8 Graphical objects
If the prefix is empty, objects with any name will be deleted.
The function is executed synchronously, that is, it blocks the calling MQL program until its completion
and returns the number of deleted objects. Since the function waits for the execution of all commands
that were in the chart queue before calling it, the action may take some time.
bool ObjectDelete(long chartId, const string name)
The function deletes an object with the specified name on the chart with chartId.
Unlike Obj ectsDeleteAll, Obj ectDelete is executed asynchronously, that is, it sends a command to the
graphics to delete the object and immediately returns control to the MQL program. The result of true
indicates the successful placement of the command in the queue. To check the result of execution, you
can use the Obj ectFind function or any Obj ectGet functions, which query the properties of an object.
As an example, consider the Obj ectCleanup1 .mq5 script. Its task is to remove objects with "our" prefix,
which are generated by the Obj ectSimpleShowcase.mq5 script from the previous section.
In the simplest case, we could write this:
#include "ObjectPrefix.mqh"
   
void OnStart()
{
   const int n = ObjectsDeleteAll(0, ObjNamePrefix);
   PrintFormat("%d objects deleted", n);
}
But to add variety, we can also provide the option to delete objects using the Obj ectDelete function
through multiple calls. Of course, this approach does not make sense when Obj ectsDeleteAll meets all
requirements. However, this is not always the case: when objects need to be selected according to
special conditions, that is, not only by prefix and type, Obj ectsDeleteAll won't help anymore.
Later, when we get acquainted with the functions of reading the properties of objects, we will complete
the example. In the meantime, we will introduce only an input variable for switching to the "advanced"
delete mode (UseCustomDeleteAll).
#property script_show_inputs
input bool UseCustomDeleteAll = false;
In the OnStart function depending on the selected mode, we will call the standard Obj ectsDeleteAll, or
our own implementation CustomDeleteAllObj ects.
void OnStart()
{
   const int n = UseCustomDeleteAll ?
      CustomDeleteAllObjects(0, ObjNamePrefix) :
      ObjectsDeleteAll(0, ObjNamePrefix);
      
   PrintFormat("%d objects deleted", n);
}
Let's sketch this function first, and then refine it.

---

## Page 930

Part 5. Creating application programs
930
5.8 Graphical objects
int CustomDeleteAllObjects(const long chart, const string prefix,
   const int window = -1, const int type = -1)
{
   int count = 0;
   const int n = ObjectsTotal(chart, window, type);
   
   // NB: cycle through objects in reverse order of the internal chart list
   // to keep numbering as we move away from the tail
   for(int i = n - 1; i >= 0; --i)
   {
      const string name = ObjectName(chart, i, window, type);
      if(StringLen(prefix) == 0 || StringFind(name, prefix) == 0)
      // additional checks that ObjectsDeleteAll does not provide,
      // for example, by coordinates, color or anchor point
      ...
      {
         // send a command to delete a specific object
         count += ObjectDelete(chart, name);
      }
   }
   return count;
}
Here we see several new features that will be described in the next section (Obj ectsTotal, Obj ectName).
Their point should be clear in general: the first function returns the index of objects on the chart, and
the second one returns the name of the object under the specified index.
It is also worth noting that the loop through objects goes in index descending order. If we did it in the
usual way, then deleting objects at the beginning of the list would lead to a violation of the numbering.
Strictly speaking, even the current loop does not guarantee complete deletion, assuming that another
MQL program starts adding objects in parallel with our deletion. Indeed, a new "foreign" object can be
added to the beginning of the list (it is formed in alphabetical order of object names) and increase the
remaining indexes, pushing "our" next object to be deleted beyond the current index i. The more new
objects are added to the beginning, the more likely it is to miss deleting your own.
Therefore, to improve reliability, it would be possible after the loop to check that the number of
remaining objects is equal to the difference between the initial number and the number of objects
removed. Although this does not give a 1 00% guarantee, since other programs could delete objects in
parallel. We will leave these specifics for independent study.
In the current implementation, our script should delete all objects with "our" prefix, regardless of
switching the UseCustomDeleteAll mode. The log should show something like this:
ObjectSimpleShowcase (XAUUSD,H1) 14 objects of various types created 
ObjectCleanup1 (XAUUSD,H1) 14 objects deleted
Let's get to know the Obj ectsTotal and Obj ectName functions, which we just used, and then return to
the Obj ectCleanup2.mq5 version of the script.
5.8.6 Finding objects
There are three functions to search for objects on the chart. The first two, the Obj ectsTotal and
Obj ectName, allow you to sort through objects by name, and then, if necessary, use the name of each

---

## Page 931

Part 5. Creating application programs
931 
5.8 Graphical objects
object to analyze its other properties (we will describe how this is done in the next section). The third
function, Obj ectFind, allows you to check the existence of an object by a known name. The same could
be done by simply requesting some property via the Obj ectGet function: if there is no object with the
passed name, we will get an error in _ LastError, but this is less convenient than calling Obj ectFind.
Besides, the function immediately returns the number of the window in which the object is located.
int ObjectsTotal(long chartId, int window = -1 , int type = -1 )
The function returns the number of objects on the chart with the chartId identifier (0 means current
chart). Only objects in the subwindow with the specified window number are considered in the
calculation (0 represents the main window, -1  represents the main window and all subwindows). Note
that only objects of the specific type specified in the type parameter are taken into account (-1 
indicates all types by default). The value of type can be an element from the ENUM_OBJECT
enumeration.
The function is executed synchronously, that is, it blocks the execution of the calling MQL program until
the result is received.
string ObjectName(long chartId, int index, int window = -1 , int type = -1 )
The function returns the name of the object under the index number on the chart with the chartId
identifier. When compiling the internal list, within which the object is searched, the specified subwindow
number (window) and object type (type) are taken into account. The list is sorted by object names in
lexicographic order, that is, in particular, alphabetically, case sensitive.
Like Obj ectsTotal, during its execution, Obj ectName waits for the entire queue of chart commands to be
fetched, and then returns the name of the object from the updated list of objects.
In case of an error, an empty string will be obtained, and the OBJECT_NOT_FOUND (4202) error code
will be stored in _ LastError.
To test the functionality of these two functions, let's create a script called Obj ectFinder.mq5 that logs
all objects on all charts. It uses chart iteration functions (ChartFirst and ChartNext), as well as
functions for getting chart properties (ChartSymbol, ChartPeriod, and ChartGetInteger).

---

## Page 932

Part 5. Creating application programs
932
5.8 Graphical objects
#include <MQL5Book/Periods.mqh>
   
void OnStart()
{
   int count = 0;
   long id = ChartFirst();
   // loop through charts
   while(id != -1)
   {
      PrintFormat("%s %s (%lld)", ChartSymbol(id), PeriodToString(ChartPeriod(id)), id);
      const int win = (int)ChartGetInteger(id, CHART_WINDOWS_TOTAL);
      // loop through windows
      for(int k = 0; k < win; ++k)
      {
         PrintFormat("  Window %d", k);
         const int n = ObjectsTotal(id, k);
         // loop through objects
         for(int i = 0; i < n; ++i)
         {
            const string name = ObjectName(id, i, k);
            const ENUM_OBJECT type = (ENUM_OBJECT)ObjectGetInteger(id, name, OBJPROP_TYPE);
            PrintFormat("    %s %s", EnumToString(type), name);
            ++count;
         }
      }
      id = ChartNext(id);
   }
   
   PrintFormat("%d objects found", count);
}
For each chart, we determine the number of subwindows (ChartGetInteger(id,
CHART_ WINDOWS_ TOTAL)), call Obj ectsTotal for each subwindow, and call Obj ectName in the inner
loop. Next, by name, we find the type of object and display them together in the log.
Below is a version of the possible result of the script (with abbreviations).

---

## Page 933

Part 5. Creating application programs
933
5.8 Graphical objects
EURUSD H1 (132358585987782873)
  Window 0
    OBJ_FIBO H1 Fibo 58513
    OBJ_TEXT H1 Text 40688
    OBJ_TREND H1 Trendline 3291
    OBJ_VLINE H1 Vertical Line 28732
    OBJ_VLINE H1 Vertical Line 33752
    OBJ_VLINE H1 Vertical Line 35549
  Window 1
  Window 2
EURUSD D1 (132360375330772909)
  Window 0
EURUSD M15 (132544239145024745)
  Window 0
    OBJ_VLINE H1 Vertical Line 27032
...
XAUUSD D1 (132544239145024746)
  Window 0
    OBJ_EVENT ObjShow-2021.11.25 00:00:00
    OBJ_TEXT ObjShow-2021.11.26 00:00:00
    OBJ_ARROW_SELL ObjShow-2021.11.29 00:00:00
    OBJ_ARROW_BUY ObjShow-2021.11.30 00:00:00
    OBJ_ARROW_RIGHT_PRICE ObjShow-2021.12.01 00:00:00
    OBJ_ARROW_LEFT_PRICE ObjShow-2021.12.02 00:00:00
    OBJ_ARROW_CHECK ObjShow-2021.12.03 00:00:00
    OBJ_ARROW_STOP ObjShow-2021.12.06 00:00:00
    OBJ_ARROW_DOWN ObjShow-2021.12.07 00:00:00
    OBJ_ARROW_UP ObjShow-2021.12.08 00:00:00
    OBJ_ARROW_THUMB_DOWN ObjShow-2021.12.09 00:00:00
    OBJ_ARROW_THUMB_UP ObjShow-2021.12.10 00:00:00
    OBJ_HLINE ObjShow-2021.12.13 00:00:00
    OBJ_VLINE ObjShow-2021.12.14 00:00:00
...
35 objects found
Here, in particular, you can see that on the XAUUSD, D1  chart there are objects generated by the
Obj ectSimpleShowcase.mq5 script. There are no objects in some charts and in some subwindows.
int ObjectFind(long chartId, const string name)
The function searches for an object by name on the chart specified by the identifier and, if successful,
returns the number of the window where it was found.
If the object is not found, the function returns a negative number. Like the previous functions in this
section, the Obj ectFind function uses a synchronous call.
We will see an example of using this function in the Obj ectCopy.mq5 script in the next section.

---

## Page 934

Part 5. Creating application programs
934
5.8 Graphical objects
5.8.7 Overview of object property access functions
Objects have various types of properties that can be read and set using Obj ectGet and Obj ectSet
functions. As we know, this principle has already been applied to the chart  (see the Overview of
functions for working with the full set of chart properties section).
All such functions take as their first three parameters a chart identifier, an object name, and a property
identifier, which must be a member of one of the ENUM_OBJECT_PROPERTY_INTEGER,
ENUM_OBJECT_PROPERTY_DOUBLE, or ENUM_OBJECT_PROPERTY_STRING enumerations. We will
study specific properties gradually in the following sections. Their complete pivot tables can be found in
the MQL5 documentation, on the page with Object Properties.
It should be noted that property identifiers in all three enumerations do not intersect, which makes it
possible to combine their joint processing into a single unified code. We will use this in the examples.
Some properties are read-only and will be marked "r/o" (read-only).
As in the case of the plotting API, the property read functions have a short form and a long form: the
short form directly returns the requested value, and the long form returns a boolean success (true) or
errors (false), and the value itself is placed in the last parameter passed by reference. The absence of
an error when calling the short form should be checked using the built-in _ LastError variable.
When accessing some properties, you must specify an additional parameter (modifier), which is used to
indicate the value number or level if the property is multivalued. For example, if an object has several
anchor points, then the modifier allows you to select a specific one.
Following are the function prototypes for reading and writing integer properties. Note that the type of
values in them is long, which allows you to store properties not only of the int or long types, but also
bool, color, datetime, and various enumerations (see below).
bool ObjectSetInteger(long chartId, const string name, ENUM_OBJECT_PROPERTY_INTEGER
property, long value)
bool ObjectSetInteger(long chartId, const string name, ENUM_OBJECT_PROPERTY_INTEGER
property, int modifier, long value)
long ObjectGetInteger(long chartId, const string name, ENUM_OBJECT_PROPERTY_INTEGER
property, int modifier = 0)
bool ObjectGetInteger(long chartId, const string name, ENUM_OBJECT_PROPERTY_INTEGER
property, int modifier, long &value)
Functions for real properties are described similarly.
bool ObjectSetDouble(long chartId, const string name, ENUM_OBJECT_PROPERTY_DOUBLE property,
double value)
bool ObjectSetDouble(long chartId, const string name, ENUM_OBJECT_PROPERTY_DOUBLE property,
int modifier, double value)
double ObjectGetDouble(long chartId, const string name, ENUM_OBJECT_PROPERTY_DOUBLE
property, int modifier = 0)
bool ObjectGetDouble(long chartId, const string name, ENUM_OBJECT_PROPERTY_DOUBLE property,
int modifier, double &value)
Finally, four of the same functions exist for strings.

---

## Page 935

Part 5. Creating application programs
935
5.8 Graphical objects
bool ObjectSetString(long chartId, const string name, ENUM_OBJECT_PROPERTY_STRING property,
const string value)
bool ObjectSetString(long chartId, const string name, ENUM_OBJECT_PROPERTY_STRING property,
int modifier, const string value)
string ObjectGetString(long chartId, const string name, ENUM_OBJECT_PROPERTY_STRING property,
int modifier = 0)
bool ObjectGetString(long chartId, const string name, ENUM_OBJECT_PROPERTY_STRING property,
int modifier, string &value)
To enhance performance, all functions for setting object properties (Obj ectSetInteger, Obj ectSetDouble,
and Obj ectSetString) are asynchronous and essentially send commands to the chart to modify the
object. Upon successful execution of these functions, the commands are placed in the shared event
queue of the chart, indicated by the returned result of true. When an error occurs, the functions will
return false, and the error code must be checked in the _ LastError variable.
Object properties are changed with some delay, during the processing of the chart event queue. To
force the update of the appearance and properties of objects on the chart, especially after changing
many objects at once, use the ChartRedraw function.
The functions for getting chart properties (Obj ectGetInteger, Obj ectGetDouble, and Obj ectGetString)
are synchronous, that is, the calling code waits for the result of their execution. In this case, all
commands in the chart queue are executed to get the actual value of the properties.
Let's go back to the example of the script for deleting objects, more precisely, to its new version,
Obj ectCleanup2.mq5. Recall that in the CustomDeleteAllObj ects function, we wanted to implement the
ability to select objects based on their properties. Let's say that these properties should be the color
and anchor point. To get them, use the Obj ectGetInteger function and a pair of
ENUM_OBJECT_PROPERTY_INTEGER enumeration elements: OBJPROP_COLOR and
OBJPROP_ANCHOR. We will look at them in detail later.
Given this information, the code would be supplemented with the following checks (here, for simplicity,
the color and anchor point are given by the clrRed and ANCHOR_TOP constants. In fact, we will provide
input variables for them).

---

## Page 936

Part 5. Creating application programs
936
5.8 Graphical objects
int CustomDeleteAllObjects(const long chart, const string prefix,
   const int window = -1, const int type = -1)
{
   int count = 0;
   
   for(int i = ObjectsTotal(chart, window, type) - 1; i >= 0; --i)
   {
      const string name = ObjectName(chart, i, window, type);
      // condition on the name and additional properties, such as color and anchor point
      if((StringLen(prefix) == 0 || StringFind(name, prefix) == 0)
         && ObjectGetInteger(0, name, OBJPROP_COLOR) == clrRed
         && ObjectGetInteger(0, name, OBJPROP_ANCHOR) == ANCHOR_TOP)
      {
         count += ObjectDelete(chart, name);
      }
   }
   return count;
}
Pay attention to the lines with Obj ectGetInteger.
Their entry is long and contains some tautology because specific properties are tied to Obj ectGet
functions of known types. Also, as the number of conditions increases, it may seem redundant to
repeat the chart ID and object name.
To simplify the record, let's turn to the technology that we tested in the ChartModeMonitor.mqh file in
the section on Chart Display Modes. Its meaning is to describe an intermediary class with method
overloads for reading and writing properties of all types. Let's name the new Obj ectMonitor.mqh header
file.
The Obj ectProxy class closely replicates the structure of the ChartModeMonitorInterface class for
charts. The main difference is the presence of virtual methods for setting and getting the chart ID and
object name.

---

## Page 937

Part 5. Creating application programs
937
5.8 Graphical objects
class ObjectProxy
{
public:
   long get(const ENUM_OBJECT_PROPERTY_INTEGER property, const int modifier = 0)
   {
      return ObjectGetInteger(chart(), name(), property, modifier);
   }
   double get(const ENUM_OBJECT_PROPERTY_DOUBLE property, const int modifier = 0)
   {
      return ObjectGetDouble(chart(), name(), property, modifier);
   }
   string get(const ENUM_OBJECT_PROPERTY_STRING property, const int modifier = 0)
   {
      return ObjectGetString(chart(), name(), property, modifier);
   }
   bool set(const ENUM_OBJECT_PROPERTY_INTEGER property, const long value,
      const int modifier = 0)
   {
      return ObjectSetInteger(chart(), name(), property, modifier, value);
   }
   bool set(const ENUM_OBJECT_PROPERTY_DOUBLE property, const double value,
      const int modifier = 0)
   {
      return ObjectSetDouble(chart(), name(), property, modifier, value);
   }
   bool set(const ENUM_OBJECT_PROPERTY_STRING property, const string value,
      const int modifier = 0)
   {
      return ObjectSetString(chart(), name(), property, modifier, value);
   }
   
   virtual string name() = 0;
   virtual void name(const string) { }
   virtual long chart() { return 0; }
   virtual void chart(const long) { }
};
Let's implement these methods in the descendant class (later we will supplement the class hierarchy
with the object property monitor, similar to the chart property monitor).

---

## Page 938

Part 5. Creating application programs
938
5.8 Graphical objects
class ObjectSelector: public ObjectProxy
{
protected:
   long host; // chart ID
   string id; // chart ID
public:
   ObjectSelector(const string _id, const long _chart = 0): id(_id), host(_chart) { }
   
   virtual string name()
   {
      return id;
   }
   virtual void name(const string _id)
   {
      id = _id;
   }
   virtual void chart(const long _chart) override
   {
      host = _chart;
   }
};
We have separated the abstract interface Obj ectProxy and its minimal implementation in Obj ectSelector
because later we may need to implement an array of proxies for multiple objects of the same type, for
example. Then it is enough to store an array of names or their common prefix in the new
"multiselector" class and ensure that one of them is returned from the name method by calling the
overloaded operator []:multiSelector[i].get(OBJPROP_ XYZ).
Now let's go back to the Obj ectCleanup2.mq5 script and describe two input variables for specifying a
color and an anchor point as additional conditions for selecting objects to be deleted.
// ObjectCleanup2.mq5
...
input color CustomColor = clrRed;
input ENUM_ARROW_ANCHOR CustomAnchor = ANCHOR_TOP;
Let's pass these values to the CustomDeleteAllObj ects function, and the new condition checks in the
loop over objects can be formulated more compactly thanks to the mediator class.

---

## Page 939

Part 5. Creating application programs
939
5.8 Graphical objects
#include <MQL5Book/ObjectMonitor.mqh>
   
void OnStart()
{
   const int n = UseCustomDeleteAll ?
      CustomDeleteAllObjects(0, ObjNamePrefix, CustomColor, CustomAnchor) :
      ObjectsDeleteAll(0, ObjNamePrefix);
   PrintFormat("%d objects deleted", n);
}
   
int CustomDeleteAllObjects(const long chart, const string prefix,
   color clr, ENUM_ARROW_ANCHOR anchor,
   const int window = -1, const int type = -1)
{
   int count = 0;
   for(int i = ObjectsTotal(chart, window, type) - 1; i >= 0; --i)
   {
      const string name = ObjectName(chart, i, window, type);
      
      ObjectSelector s(name);
      ResetLastError();
      if((StringLen(prefix) == 0 || StringFind(s.get(OBJPROP_NAME), prefix) == 0)
      && s.get(OBJPROP_COLOR) == CustomColor
      && s.get(OBJPROP_ANCHOR) == CustomAnchor
      && _LastError != 4203) // OBJECT_WRONG_PROPERTY
      {
         count += ObjectDelete(chart, name);
      }
   }
   return count;
}
It is important to note that we specify the name of the object (and the implicit identifier of the current
chart 0) only once when creating the Obj ectSelector object. Further, all properties are requested by the
get method with a single parameter describing the desired property, and the appropriate Obj ectGet
function will be chosen by the compiler automatically.
The additional check for error code 4203 (OBJECT_WRONG_PROPERTY) allows filtering out objects
that do not have the requested property, such as OBJPROP_ANCHOR. In this way, in particular, it is
possible to make a selection in which all types of arrows will fall (without the need to separately request
different types of OBJ_ARROW_XYZ), but lines and "events" will be excluded from processing.
This is easy to check by first running the Obj ectSimpleShowcase.mq5 script on the chart (it will create
1 4 objects of different types) and then Obj ectCleanup2.mq5. If you turn on the UseCustomDeleteAll
mode, there will be 5 non-deleted objects on the chart: OBJ_VLINE, OBJ_HLINE, OBJ_ARROW_BUY,
OBJ_ARROW_SELL, and OBJ_EVENT. The first two and the last do not have the OBJPROP_ANCHOR
property, and the buy and sell arrows do not pass by color (it is assumed that the color of all other
created objects is red by default).
However, Obj ectSelector is provided not only for the sake of the above simple application. It is the basis
for creating a property monitor for a single object, similar to what was implemented for charts. So the
Obj ectMonitor.mqh header file contains something more interesting.  

---

## Page 940

Part 5. Creating application programs
940
5.8 Graphical objects
class ObjectMonitorInterface: public ObjectSelector
{
public:
   ObjectMonitorInterface(const string _id, const long _chart = 0):
      ObjectSelector(_id, _chart) { }
   virtual int snapshot() = 0;
   virtual void print() { };
   virtual int backup() { return 0; }
   virtual void restore() { }
   virtual void applyChanges(ObjectMonitorInterface *reference) { }
};
This set of methods should remind you ChartModeMonitorInterface from ChartModeMonitor.mqh. The
only innovation is the applyChanges method, which copies the properties of one object to another.
Based on Obj ectMonitorInterface, here is the description of the basic implementation of a property
monitor for a pair of template types: a property value type (one of long, double, or string) and the
enumeration type (one of ENUM_OBJECT_PROPERTY_-ish).
template<typename T,typename E>
class ObjectMonitorBase: public ObjectMonitorInterface
{
protected:
   MapArray<E,T> data;  // array of pairs [property, value], current state
   MapArray<E,T> store; // backup (filled on demand)
   MapArray<E,T> change;// committed changes between two states
   ...
The Obj ectMonitorBase constructor has two parameters: the name of the object and an array of flags
with identifiers of the properties to be observed in the specified object. A significant portion of this code
is almost identical to ChartModeMonitor. In particular, as before, an array of flags is passed to the
helper method detect, the main purpose of which is to identify those integer constants that are
elements of the E enumeration, and weed out all the rest. The only addition that needs to be clarified is
getting a property with the number of levels in an object via Obj ectGetInteger(0, id, OBJPROP_ LEVELS).
This is necessary to support iteration of properties with multiple values due to the presence of levels
(for example, Fibonacci). For objects without levels, we will get the quantity 0, and such a property will
be the usual, scalar one.
public:
   ObjectMonitorBase(const string _id, const int &flags[]): ObjectMonitorInterface(_id)
   {
      const int levels = (int)ObjectGetInteger(0, id, OBJPROP_LEVELS);
      for(int i = 0; i < ArraySize(flags); ++i)
      {
         detect(flags[i], levels);
      }
   }
   ...
Of course, the detect method is somewhat different from what we saw in ChartModeMonitor. Recall that
to begin with, it contains a fragment with a check if the v constant belongs to the E enumeration, using
a call to the EnumToString function: if there is no such element in the enumeration, an error code will
be raised. If the element exists, we add the value of the corresponding property to the data array.

---

## Page 941

Part 5. Creating application programs
941 
5.8 Graphical objects
   // ChartModeMonitor.mqh
   bool detect(const int v)
   {
      ResetLastError();
      conststrings = EnumToString((E)v); // resulting string is not important
      if(_LastError == 0)                // analyze the error code
      {
         data.put((E)v, get((E)v));
         return true;
      }
      return false;
   }
In the object monitor, we are forced to complicate this scheme, since some properties are multi-valued
due to the modifier parameter in the Obj ectGet and Obj ectSet functions.
So we introduce a static array modifiables with a list of those properties that modifiers support (each
property will be discussed in detail later). The bottom line is that for such multi-valued properties, you
need to read them and store them in the data array not once, but several times.
// ObjectMonitor.mqh
   bool detect(const int v, const int levels)
   {
      // the following properties support multiple values
      static const int modifiables[] =
      {
         OBJPROP_TIME,        // anchor point by time
         OBJPROP_PRICE,       // anchor point by price
         OBJPROP_LEVELVALUE,  // level value
         OBJPROP_LEVELTEXT,   // inscription on the level line
         // NB: the following properties do not generate errors when exceeded
         // actual number of levels or files
         OBJPROP_LEVELCOLOR,  // level line color
         OBJPROP_LEVELSTYLE,  // level line style
         OBJPROP_LEVELWIDTH,  // width of the level line
         OBJPROP_BMPFILE,     // image files
      };
      ...
Here, we also use the trick with EnumToString to check the existence of a property with the v
identifier. If successful, we check if it is in the list of modifiables and set the corresponding flag
modifiable to true or false.

---

## Page 942

Part 5. Creating application programs
942
5.8 Graphical objects
      bool result = false;
      ResetLastError();
 conststrings =EnumToString((E)v); // resulting string is not important
 if(_LastError ==0)// analyze the error code
      {
         bool modifiable = false;
         for(int i = 0; i < ArraySize(modifiables); ++i)
         {
            if(v == modifiables[i])
            {
               modifiable = true;
               break;
            }
         }
         ...
By default, any property is considered unambiguous and therefore the required number of readings
through the Obj ectGet function or entries via the Obj ectSet function is equal to 1  (the k variable below).
         int k = 1;
         // for properties with modifiers, set the correct amount
         if(modifiable)
         {
            if(levels > 0) k = levels;
            else if(v == OBJPROP_TIME || v == OBJPROP_PRICE) k = MOD_MAX;
            else if(v == OBJPROP_BMPFILE) k = 2;
         }
If an object supports levels, we limit the potential number of reads/writes with the levels parameter (as
we recall, it is obtained in the calling code from the OBJPROP_LEVELS property).
For the OBJPROP_BMPFILE property, as we will soon learn, only two states are allowed: on (button
pressed, flag set) or off (button released, flag cleared), so k = 2.
Finally, object coordinates - OBJPROP_TIME and OBJPROP_PRICE - are convenient because they
generate an error when trying to read/write a non-existent anchor point. Therefore we assign to k
some obviously large value of MOD_MAX, and then we can interrupt the cycle of reading points at a
non-zero value _ LastError.

---

## Page 943

Part 5. Creating application programs
943
5.8 Graphical objects
         // read property value - one or many
         for(int i = 0; i < k; ++i)
         {
            ResetLastError();
            T temp = get((E)v, i);
            // if there is no i-th modifier, we will get an error and break the loop
            if(_LastError != 0) break;
            data.put((E)MOD_COMBINE(v, i), temp);
            result = true;
         }
      }
      return result;
   }
Since one property can have several values, which are read in a loop up to k, we can no longer simply
write data.put((E)v, get((E)v)). We need to somehow combine the property identifier v and its
modification number i. Fortunately, the number of properties is also limited in an integer constant
(typeint) no more than two lower bytes are occupied. So we can use bitwise operators to put i to the
top byte. The MOD_COMBINE macro has been developed for this purpose.
#define MOD_COMBINE(V,I) (V | (I << 24))
Of course, reverse macros are provided to retrieve the property ID and revision number.
#define MOD_GET_NAME(V)  (V & 0xFFFFFF)
#define MOD_GET_INDEX(V) (V >> 24)
For example, here we can see how they are used in the snapshot method.

---

## Page 944

Part 5. Creating application programs
944
5.8 Graphical objects
   virtual int snapshot() override
   {
      MapArray<E,T> temp;
      change.reset();
      
      // collect all required properties in temp
      for(int i = 0; i < data.getSize(); ++i)
      {
         const E e = (E)MOD_GET_NAME(data.getKey(i));
         const int m = MOD_GET_INDEX(data.getKey(i));
         temp.put((E)data.getKey(i), get(e, m));
      }
      
      int changes = 0;
      // compare previous and new state
      for(int i = 0; i < data.getSize(); ++i)
      {
         if(data[i] != temp[i])
         {
            // save the differences in the change array
            if(changes == 0) Print(id);
            const E e = (E)MOD_GET_NAME(data.getKey(i));
            const int m = MOD_GET_INDEX(data.getKey(i));
            Print(EnumToString(e), (m > 0 ? (string)m : ""), " ", data[i], " -> ", temp[i]);
            change.put(data.getKey(i), temp[i]);
            changes++;
         }
      }
      
      // save the new state as current
      data = temp;
      return changes;
   }
This method repeats all the logic of the method of the same name in ChartModeMonitor.mqh, however,
to read properties everywhere, you must first extract the property name from the stored key using
MOD_GET_NAME and the number using MOD_GET_INDEX.
A similar complication has to be done in the restore method.
   virtual void restore() override
   {
      data = store;
      for(int i = 0; i < data.getSize(); ++i)
      {
         const E e = (E)MOD_GET_NAME(data.getKey(i));
         const int m = MOD_GET_INDEX(data.getKey(i));
         set(e, data[i], m);
      }
   }
The most interesting innovation of Obj ectMonitorBase is how it works with changes.

---

## Page 945

Part 5. Creating application programs
945
5.8 Graphical objects
   MapArray<E,T> * const getChanges()
   {
      return &change;
   }
   
   virtual void applyChanges(ObjectMonitorInterface *intf) override
   {
      ObjectMonitorBase *reference = dynamic_cast<ObjectMonitorBase<T,E> *>(intf);
      if(reference)
      {
         MapArray<E,T> *event = reference.getChanges();
         if(event.getSize() > 0)
         {
            Print("Modifing ", id, " by ", event.getSize(), " changes");
            for(int i = 0; i < event.getSize(); ++i)
            {
               data.put(event.getKey(i), event[i]);
               const E e = (E)MOD_GET_NAME(event.getKey(i));
               const int m = MOD_GET_INDEX(event.getKey(i));
               Print(EnumToString(e), " ", m, " ", event[i]);
               set(e, event[i], m);
            }
         }
      }
   }
Passing to the applyChanges method states of the monitor of another object, we can adopt all the
latest changes from it.
To support properties of all three basic types (long,double,string), we need to implement the
Obj ectMonitor class (analog of ChartModeMonitor from ChartModeMonitor.mqh).
class ObjectMonitor: public ObjectMonitorInterface
{
protected:
   AutoPtr<ObjectMonitorInterface> m[3];
   
   ObjectMonitorInterface *getBase(const int i)
   {
      return m[i][];
   }
   
public:
   ObjectMonitor(const string objid, const int &flags[]): ObjectMonitorInterface(objid)
   {
      m[0] = new ObjectMonitorBase<long,ENUM_OBJECT_PROPERTY_INTEGER>(objid, flags);
      m[1] = new ObjectMonitorBase<double,ENUM_OBJECT_PROPERTY_DOUBLE>(objid, flags);
      m[2] = new ObjectMonitorBase<string,ENUM_OBJECT_PROPERTY_STRING>(objid, flags);
   }
   ...
The previous code structure is also preserved here, and only methods have been added to support
changes and names (charts, as we remember, do not have names).

---

## Page 946

Part 5. Creating application programs
946
5.8 Graphical objects
   ...
   virtual string name() override
   {
      return m[0][].name();
   }
   
   virtual void name(const string objid) override
   {
      m[0][].name(objid);
      m[1][].name(objid);
      m[2][].name(objid);
   }
   
   virtual void applyChanges(ObjectMonitorInterface *intf) override
   {
      ObjectMonitor *monitor = dynamic_cast<ObjectMonitor *>(intf);
      if(monitor)
      {
         m[0][].applyChanges(monitor.getBase(0));
         m[1][].applyChanges(monitor.getBase(1));
         m[2][].applyChanges(monitor.getBase(2));
      }
   }
Based on the created object monitor, it is easy to implement several tricks that are not supported in
the terminal. In particular, this is the creation of copies of objects and group editing of objects.
Script ObjectCopy
The Obj ectCopy.mq5 script demonstrates how to copy selected objects. At the beginning of its OnStart
function, we fill the flags array with consecutive integers that are candidates for elements of
ENUM_OBJECT_PROPERTY_ enumerations of different types. The numbering of the enumeration
elements has a pronounced grouping by purpose, and there are large gaps between the groups
(apparently, a margin for future elements), so the formed array is quite large: 2048 elements.
#include <MQL5Book/ObjectMonitor.mqh>
   
#define PUSH(A,V) (A[ArrayResize(A, ArraySize(A) + 1) - 1] = V)
   
void OnStart()
{
   int flags[2048];
   // filling the array with consecutive integers, which will be
   // checked against the elements of enumerations of object properties,
   // invalid values will be discarded in the monitor's detect method
   for(int i = 0; i < ArraySize(flags); ++i)
   {
      flags[i] = i;
   }
   ...
Next, we collect into an array the names of objects that are currently selected on the chart. For this,
we use the OBJPROP_SELECTED property.

---

## Page 947

Part 5. Creating application programs
947
5.8 Graphical objects
   string selected[];
   const int n = ObjectsTotal(0);
   for(int i = 0; i < n; ++i)
   {
      const string name = ObjectName(0, i);
      if(ObjectGetInteger(0, name, OBJPROP_SELECTED))
      {
         PUSH(selected, name);
      }
   }
   ...
Finally, in the main loop over the selected elements, we read the properties of each object, form the
name of its copy, and create an object under it with the same set of attributes.
   for(int i = 0; i < ArraySize(selected); ++i)
   {
      const string name = selected[i];
      
     // make a backup of the properties of the current object using the monitor
      ObjectMonitor object(name, flags);
      object.print();
      object.backup();
      // form a correct, appropriate name for the copy
      const string copy = GetFreeName(name);
      
      if(StringLen(copy) > 0)
      {
         Print("Copy name: ", copy);
         // create an object of the same type OBJPROP_TYPE
         ObjectCreate(0, copy,
            (ENUM_OBJECT)ObjectGetInteger(0, name, OBJPROP_TYPE),
            ObjectFind(0, name), 0, 0);
         // change the name of the object in the monitor to a new one
         object.name(copy);
         // restore all properties from the backup to a new object
         object.restore();
      }
      else
      {
         Print("Can't create copy name for: ", name);
      }
   }
}
It is important to note here that the OBJPROP_TYPE property is one of the few read-only properties,
and therefore it is vital to create an object of the required type to begin with.
The helper function GetFreeName tries to append the string "/Copy #x" to the object name, where x is
the copy number. Thus, by running the script several times, you can create the 2nd, 3rd, and so on
copies.

---

## Page 948

Part 5. Creating application programs
948
5.8 Graphical objects
string GetFreeName(const string name)
{
   const string suffix = "/Copy №";
   // check if there is a copy in the suffix name
   const int pos = StringFind(name, suffix);
   string prefix;
   int n;
   
   if(pos <= 0)
   {
      // if suffix is not found, assume copy number 1
      const string candidate = name + suffix + "1";
      // checking if the copy name is free, and if so, return it
      if(ObjectFind(0, candidate) < 0)
      {
         return candidate;
      }
      // otherwise, prepare for a loop with iteration of copy numbers
      prefix = name;
      n = 0;
   }
   else
   {
      // if the suffix is found, select the name without it
      prefix = StringSubstr(name, 0, pos);
      // and find the copy number in the string
      n = (int)StringToInteger(StringSubstr(name, pos + StringLen(suffix)));
   }
   
   Print("Found: ", prefix, " ", n);
   // loop trying to find a free copy number above n, but no more than 1000
   for(int i = n + 1; i < 1000; ++i)
   {
      const string candidate = prefix + suffix + (string)i;
      // check for the existence of an object with a name ending "Copy #i"
      if(ObjectFind(0, candidate) < 0)
      {
         return candidate; // return vacant copy name
      }
   }
   return NULL; // too many copies
}
The terminal remembers the last settings of a particular type of object, and if they are created one
after the other, this is equivalent to copying. However, the settings usually change in the process of
working with different charts, and if after a while there is a need to duplicate some "old" object, then
the settings for it, as a rule, have to be done completely. This is especially expensive for object types
with a large number of properties, for example, Fibonacci tools. In such cases, this script will come in
handy.
Some of the pictures from this chapter, which contain objects of the same type, were created using
this script.

---

## Page 949

Part 5. Creating application programs
949
5.8 Graphical objects
ObjectGroupEdit indicator
The second example of using Obj ectMonitor is the Obj ectGroupEdit.mq5 indicator, which allows you to
edit the properties of a group of selected objects at once.
Imagine that we have selected several objects on the chart (not necessarily of the same type), for
which it is necessary to uniformly change one or another property. Next, we open the properties dialog
of any of these objects, configure it, and by clicking OK these changes are applied to all selected
objects. This is how our next MQL program works.
We needed an indicator as a type of program because it involves chart events. For this aspect of MQL5
programming, there will be a whole dedicated chapter, but we will get to know some of the basics right
now.
Since the indicator does not have charts, the #property directives contain zeros and the OnCalculate
function is virtually empty. 
#property indicator_chart_window
#property indicator_buffers 0
#property indicator_plots   0
   
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
   return rates_total;
}
To automatically generate a complete set of all properties for an object, we will again use an array of
2048 elements with consecutive integer values. We will also provide an array for the names of the
selected elements and an array of monitor objects of the Obj ectMonitor class.
int consts[2048];
string selected[];
ObjectMonitor *objects[];
In the OnInit handler, we initialize the array of numbers and start the timer.
void OnInit()
{
   for(int i = 0; i < ArraySize(consts); ++i)
   {
      consts[i] = i;
   }
   
   EventSetTimer(1);
}
In the timer handler, we save the names of the selected objects in an array. If the selection list has
changed, you need to reconfigure the monitor objects, for which the auxiliary TrackSelectedObj ects
function is called.

---

## Page 950

Part 5. Creating application programs
950
5.8 Graphical objects
void OnTimer()
{
   string updates[];
   const int n = ObjectsTotal(0);
   for(int i = 0; i < n; ++i)
   {
      const string name = ObjectName(0, i);
      if(ObjectGetInteger(0, name, OBJPROP_SELECTED))
      {
         PUSH(updates, name);
      }
   }
   
   if(ArraySize(selected) != ArraySize(updates))
   {
      ArraySwap(selected, updates);
      Comment("Selected objects: ", ArraySize(selected));
      TrackSelectedObjects();
   }
}
The TrackSelectedObj ects function itself is quite simple: delete the old monitors and create new ones. If
you wish, you can make it more intelligent by maintaining the unchanged part of the selection.
void TrackSelectedObjects()
{
   for(int j = 0; j < ArraySize(objects); ++j)
   {
      delete objects[j];
   }
   
   ArrayResize(objects, 0);
   
   for(int i = 0; i < ArraySize(selected); ++i)
   {
      const string name = selected[i];
      PUSH(objects, new ObjectMonitor(name, consts));
   }
}
Recall that when creating a monitor object, it immediately takes a "cast" of all the properties of the
corresponding graphical object.
Now we finally get to the part where events come into play. As was already mentioned in the overview
of event functions, the handler is responsible for the OnChartEvent events on the chart. In this example,
we are interested in a specific CHARTEVENT_OBJECT_CHANGE event: it occurs when the user changes
any attributes in the object's properties dialog. The name of the modified object is passed in the
sparam parameter.
If this name matches one of the monitored objects, we ask the monitor to make a new snapshot of its
properties, that is, we call obj ects[i].snapshot().

---

## Page 951

Part 5. Creating application programs
951 
5.8 Graphical objects
void OnChartEvent(const int id,
   const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CHANGE)
   {
      Print("Object changed: ", sparam);
      for(int i = 0; i < ArraySize(selected); ++i)
      {
         if(sparam == selected[i])
         {
            const int changes = objects[i].snapshot();
            if(changes > 0)
            {
               for(int j = 0; j < ArraySize(objects); ++j)
               {
                  if(j != i)
                  {
                     objects[j].applyChanges(objects[i]);
                  }
               }
            }
            ChartRedraw();
            break;
         }
      }
   }
}
If the changes are confirmed (and it is unlikely otherwise), their number in the changes variable will be
greater than 0. Then a loop is started over all the selected objects, and the detected changes are
applied to each of them, except for the original one.
Since we can potentially change many objects, we call the chart redraw request with ChartRedraw.
In the OnDeinit handler, we remove all monitors.
void OnDeinit(const int)
{
   for(int j = 0; j < ArraySize(objects); ++j)
   {
      delete objects[j];
   }
   Comment("");
}
That's all: the new tool is ready.
This indicator allowed you to customize the general appearance of several groups of label objects in the
section on Defining the object anchor point.
By the way, according to a similar principle with the help of Obj ectMonitor you can make another
popular tool that is not available in the terminal: to undo edits to object properties, as the restore
method is ready now.

---

## Page 952

Part 5. Creating application programs
952
5.8 Graphical objects
5.8.8 Main object properties
All objects have some universal attributes. The main ones are listed in the following table. We will see
other general special-purpose properties later (see the Object state management, Z-order, and
Visibility of objects in timeframe context sections).
Identifier
Description
Type
OBJPROP_NAME
Object name
string
OBJPROP_TYPE
Object type (r/o)
ENUM_OBJECT
OBJPROP_CREATETIME
Object creation time (r/o)
datetime
OBJPROP_TEXT
Description of the object (text
contained in the object)
string
OBJPROP_TOOLTIP
Mouseover tooltip text
string
The OBJPROP_NAME property is an object identifier. Editing it is equivalent to deleting the old object
and creating a new one.
For some types of objects capable of displaying text (such as labels or buttons), the OBJPROP_TEXT
property is always displayed directly on the chart, inside the object. For other objects (for example,
lines), this property contains a description that is displayed on the chart next to the object and only if
the "Show object descriptions option" is enabled in the chart settings. In either case, OBJPROP_TEXT
is displayed in the tooltip.
The OBJPROP_CREATETIME property exists only until the end of the current session and is not written
to chr files.
You can change the name of an object programmatically or manually (in the object's properties dialog),
while its creation time will remain the same. Looking ahead, we note that programmatic renaming does
not cause any events about objects on the chart. As we are about to learn in the next chapter, manual
renaming triggers three events:
• deleting an object under the old name (CHARTEVENT_OBJECT_DELETE),
• creating an object under a new name (CHARTEVENT_OBJECT_CREATE) and
• modification of a new object (CHARTEVENT_OBJECT_CHANGE).
If the OBJPROP_TOOLTIP property is not set, a tooltip is displayed for the object, automatically
generated by the terminal. To disable the tooltip, set its value to "\n" (line feed).
Let's adapt the Obj ectFinder.mq5 script from the Finding objects section to log all the above properties
of objects on the current chart. Let's name the new script as Obj ectListing.mq5.
At the very beginning of OnStart, we will create or modify a vertical straight line located on the last bar
(at the moment the script is launched). If there is an option to show object descriptions in the chart
settings, then we will see the "Latest Bar At The Moment" text along the right vertical line.

---

## Page 953

Part 5. Creating application programs
953
5.8 Graphical objects
void OnStart()
{
   const string vline = ObjNamePrefix + "current";
   ObjectCreate(0, vline, OBJ_VLINE, 0, iTime(NULL, 0, 0), 0);
   ObjectSetString(0, vline, OBJPROP_TEXT, "Latest Bar At The Moment");
   ...
Next, in a loop through the subwindows, we will query all objects up to Obj ectsTotal and their main
properties.
   int count = 0;
   const long id = ChartID();
   const int win = (int)ChartGetInteger(id, CHART_WINDOWS_TOTAL);
   // loop through subwindows
   for(int k = 0; k < win; ++k)
   {
      PrintFormat("  Window %d", k);
      const int n = ObjectsTotal(id, k);
      //loop through objects
      for(int i = 0; i < n; ++i)
      {
         const string name = ObjectName(id, i, k);
         const ENUM_OBJECT type =
            (ENUM_OBJECT)ObjectGetInteger(id, name, OBJPROP_TYPE);
         const datetime created =
            (datetime)ObjectGetInteger(id, name, OBJPROP_CREATETIME);
         const string description = ObjectGetString(id, name, OBJPROP_TEXT);
         const string hint = ObjectGetString(id, name, OBJPROP_TOOLTIP);
         PrintFormat("    %s %s %s %s %s", EnumToString(type), name,
            TimeToString(created), description, hint);
         ++count;
      }
   }
   
   PrintFormat("%d objects found", count);
}
We get the following entries in the log.
  Window 0
    OBJ_VLINE ObjShow-current 2021.12.21 20:20 Latest Bar At The Moment 
    OBJ_VLINE abc 2021.12.21 19:25  
    OBJ_VLINE xyz 1970.01.01 00:00  
3 objects found
A zero OBJPROP_CREATETIME value (1 970.01 .01  00:00) means that the object was not created
during the current session, but earlier.
5.8.9 Price and time coordinates
For objects of the types that exist in the quotes coordinate system, the MQL5 API supports a couple of
properties for specifying time and price bindings. In the event that an object has several anchor points,

---

## Page 954

Part 5. Creating application programs
954
5.8 Graphical objects
properties require the specification of a modifier parameter containing the index of the anchor point
when calling the Obj ectSet and Obj ectGet functions.
Identifier
Description
Value type
OBJPROP_TIME
Time coordinate
datetime
OBJPROP_PRICE
Price coordinate
double
These properties are available for absolutely all objects, but it makes no sense to set or read them for
objects with screen coordinates.
To demonstrate how to work with coordinates, let's analyze the bufferless indicator
Obj ectHighLowChannel.mq5. For a given segment of bars, it draws two trend lines. Their start and end
points on the time axis coincide with the first and last bar of the segment, and along the price axis, the
values are calculated differently for each of the lines: the highest and lowest High prices are used for
the upper line and the highest and lowest Low prices are used for the lower line. As the chart updates,
our impromptu channel should move with prices.
The range of bars is set using two input variables: the number of the initial bar BarOffset and the
number of bars BarCount. By default, the lines are drawn at the most recent prices, because bar offset
= 0.
input int BarOffset = 0;
input int BarCount = 10;
   
const string Prefix = "HighLowChannel-";
Objects have a common name prefix "HighLowChannel-".
In the OnCalculate handler, we monitor the emergence of new bars over the iTime time of the 0-th bar.
As soon as the bar is formed, the prices are analyzed on the specified segment, the maximum and
minimum values of the prices of each of the two types (MODE_HIGH, MODE_LOW) are taken and the
auxiliary function DrawFigure is called for them, and this is where the work with objects takes place: the
creation and modification of coordinates.

---

## Page 955

Part 5. Creating application programs
955
5.8 Graphical objects
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const int begin,
                const double &price[])
{
   static datetime now = 0;
   if(now != iTime(NULL, 0, 0))
   {
      const int hh = iHighest(NULL, 0, MODE_HIGH, BarCount, BarOffset);
      const int lh = iLowest(NULL, 0, MODE_HIGH, BarCount, BarOffset);
      const int ll = iLowest(NULL, 0, MODE_LOW, BarCount, BarOffset);
      const int hl = iHighest(NULL, 0, MODE_LOW, BarCount, BarOffset);
   
      datetime t[2] = {iTime(NULL, 0, BarOffset + BarCount), iTime(NULL, 0, BarOffset)};
      double ph[2] = {iHigh(NULL, 0, fmax(hh, lh)), iHigh(NULL, 0, fmin(hh, lh))};
      double pl[2] = {iLow(NULL, 0, fmax(ll, hl)), iLow(NULL, 0, fmin(ll, hl))};
    
      DrawFigure(Prefix + "Highs", t, ph, clrBlue);
      DrawFigure(Prefix + "Lows", t, pl, clrRed);
   
      now = iTime(NULL, 0, 0);
   }
   return rates_total;
}
And here is the DrawFigure function itself.
bool DrawFigure(const string name, const datetime &t[], const double &p[],
   const color clr)
{
   if(ArraySize(t) != ArraySize(p)) return false;
   
   ObjectCreate(0, name, OBJ_TREND, 0, 0, 0);
   
   for(int i = 0; i < ArraySize(t); ++i)
   {
      ObjectSetInteger(0, name, OBJPROP_TIME, i, t[i]);
      ObjectSetDouble(0, name, OBJPROP_PRICE, i, p[i]);
   }
   
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   return true;
}
After the Obj ectCreate call that guarantees the existence of an object, the appropriate ObjectSet
functions for OBJPROP_TIME and OBJPROP_PRICE are called at all anchor points (two in this case).
The image below shows the result of the indicator.

---

## Page 956

Part 5. Creating application programs
956
5.8 Graphical objects
Channel on two trend lines at High and Low prices
You can run the indicator in the visual tester to see how the line coordinates change on the go.
5.8.1 0 Anchor window corner and screen coordinates
For objects that use a coordinate system in the form of points (pixels) on the chart, you must select
one of the four corners of the window, relative to which the values along the horizontal X-axis and
vertical Y-axis will be counted to the anchor point on the object. These aspects are controlled by the
properties in the following table.
Identifier
Description
Type
OBJPROP_CORNER
Chart corner for anchoring the
graphic object
ENUM_BASE_CORNER
OBJPROP_XDISTANCE
Distance in pixels along the X-axis
from the anchor corner
int
OBJPROP_YDISTANCE
Distance in pixels along the Y-axis
from the anchor corner
int
Valid options for OBJPROP_CORNER are summarized in the ENUM_BASE_CORNER enumeration.

---

## Page 957

Part 5. Creating application programs
957
5.8 Graphical objects
Identifier
Coordinate center location
CORNER_LEFT_UPPER
Upper left corner of the window
CORNER_LEFT_LOWER
Lower left corner of the window
CORNER_RIGHT_LOWER
Lower right corner of the window
CORNER_RIGHT_UPPER
Upper right corner of the window
The default is the top left corner.
The following figure shows four Button objects with the same size and distance from the anchor corner
in the window. Each of these objects differs only in the binding angle itself. Recall that buttons have one
anchor point which is always located in the upper left corner of the button.
Arrangement of objects bound to different corners of the main window
All four objects are currently selected on the chart, so their anchor points are highlighted in a
contrasting color.
When we talk about window corners, we mean the specific window or subwindow in which the object is
located and not the entire chart. In other words, in objects in subwindows, the Y coordinate is
measured from the top or bottom border of this subwindow.
The following illustration shows similar objects in a subwindow, snapped to the corners of the
subwindow.

---

## Page 958

Part 5. Creating application programs
958
5.8 Graphical objects
Location of objects with binding to different corners of the subwindow
Using the Obj ectCornerLabel.mq5 script the user can test the movement of a text inscription, for which
the anchor angle in the window is specified in the input parameter Corner.
#property script_show_inputs
   
input ENUM_BASE_CORNER Corner = CORNER_LEFT_UPPER;
The coordinates change periodically and are displayed in the text of the inscription itself. Thus, the
inscription moves in the window and, when it reaches the border, bounces off it. The object is created
in the window or subwindow where the script was dropped by the mouse.
void OnStart()
{
   const int t = ChartWindowOnDropped();
   const string legend = EnumToString(Corner);
   
   const string name = "ObjCornerLabel-" + legend;
   int h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, t);
   int w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int x = w / 2;
   int y = h / 2;
   ...
For correct positioning, we find the dimensions of the window (and then check if they have changed)
and find the middle for the initial placement of the object: the variables with the coordinates x and y.
Next, we create and set up an inscription, without coordinates yet. It is important to note that we
enable the ability to select an object (OBJPROP_SELECTABLE) and select it (OBJPROP_SELECTED), as
this allows us to see the anchor point on the object itself, to which the distance from the window corner

---

## Page 959

Part 5. Creating application programs
959
5.8 Graphical objects
(coordinate center) is measured. These two properties are described in more detail in the section on
Object state management.
   ObjectCreate(0, name, OBJ_LABEL, t, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTED, true);
   ObjectSetInteger(0, name, OBJPROP_CORNER, Corner);
   ...
In the variables px and py, we will record the increments of coordinates for motion emulation. The
coordinate modification itself will be performed in an infinite loop until it is interrupted by the user. The
iteration counter will allow periodically, at every 50 iterations, to change the direction of movement at
random.
   int px = 0, py = 0;
   int pass = 0;
   
   for( ;!IsStopped(); ++pass)
   {
      if(pass % 50 == 0)
      {
         h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, t);
         w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
         px = rand() * (w / 20) / 32768 - (w / 40);
         py = rand() * (h / 20) / 32768 - (h / 40);
      }
   
      // bounce off window borders so object doesn't hide
      if(x + px > w || x + px < 0) px = -px;
      if(y + py > h || y + py < 0) py = -py;
      // recalculate label positions
      x += px;
      y += py;
      
      // update the coordinates of the object and add them to the text
      ObjectSetString(0, name, OBJPROP_TEXT, legend
         + "[" + (string)x + "," + (string)y + "]");
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   
      ChartRedraw();
      Sleep(100);
   }
   
   ObjectDelete(0, name);
}
Try running the script multiple times, specifying different anchor corners.
In the next section, we'll augment this script by also controlling the anchor point on the object.

---

## Page 960

Part 5. Creating application programs
960
5.8 Graphical objects
5.8.1 1  Defining anchor point on an object
Some types of objects allow you to select an anchor point. The types that fall into this category include
text label (OBJ_TEXT) and bitmap image (OBJ_BITMAP) linked to quotes, as well as caption
(OBJ_LABEL) and panel with image (OBJ_BITMAP_LABEL), positioned in screen coordinates.
To read and set the anchor point, use the functions Obj ectGetInteger and Obj ectSetInteger with the
OBJPROP_ANCHOR property.
All point selection options are collected in the ENUM_ANCHOR_POINT enumeration.
Identifier
Anchor point location
ANCHOR_LEFT_UPPER
In the upper left corner
ANCHOR_LEFT
Left center
ANCHOR_LEFT_LOWER
In the lower left corner
ANCHOR_LOWER
Bottom center
ANCHOR_RIGHT_LOWER
In the lower right corner
ANCHOR_RIGHT
Right center
ANCHOR_RIGHT_UPPER
In the upper right corner
ANCHOR_UPPER
Top center
ANCHOR_CENTER
Strictly in the center of the object
The points are clearly shown in the image below, where several label objects are applied to the chart.

---

## Page 961

Part 5. Creating application programs
961 
5.8 Graphical objects
OBJ_LABEL text objects with different anchor points
The upper group of four labels has the same pair of coordinates (X,Y), however, due to anchoring to
different corners of the object, they are located on different sides of the point. A similar situation is in
the second group of four text labels, however, there the anchoring is made to the midpoints of different
sides of the objects. Finally, the caption is shown separately at the bottom, anchored in its center, so
that the point is inside the object.
The button (OBJ_BUTTON), rectangular panel (OBJ_RECTANGLE_LABEL), input field (OBJ_EDIT), and
chart object (OBJ_CHART) have a fixed anchor point in the upper left corner (ANCHOR_LEFT_UPPER).
Some graphical objects of the group of single price marks (OBJ_ARROW, OBJ_ARROW_THUMB_UP,
OBJ_ARROW_THUMB_DOWN, OBJ_ARROW_UP, OBJ_ARROW_DOWN, OBJ_ARROW_STOP,
OBJ_ARROW_CHECK) have two ways of anchoring their coordinates, specified by identifiers of another
enumeration ENUM_ARROW_ANCH OR.
Identifier
Anchor point location
ANCHOR_TOP
Top center
ANCHOR_BOTTOM
Bottom center
The rest of the objects in this group have predefined anchor points: the buy (OBJ_ARROW_BUY) and
sell (OBJ_ARROW_SELL) arrows are respectively in the middle of the upper and lower sides, and the
price labels (OBJ_ARROW_RIGHT_PRICE, OBJ_ARROW_LEFT_PRICE) are on the left and right.
Similar to the script Obj ectCornerLabel.mq5 from the previous section, let's create the script
Obj ectAnchorLabel.mq5. In the new version, in addition to moving the inscription, we will randomly
change the anchor point on it.

---

## Page 962

Part 5. Creating application programs
962
5.8 Graphical objects
The corner of the window for anchoring will be selected, as before, by the user when the script is
launched.
input ENUM_BASE_CORNER Corner = CORNER_LEFT_UPPER;
We will display the name of the angle on the chart as a comment.
void OnStart()
{
   Comment(EnumToString(Corner));
   ...
In an infinite loop, one of 9 possible anchor point values is generated at selected times.
   ENUM_ANCHOR_POINT anchor = 0;
   for( ;!IsStopped(); ++pass)
   {
      if(pass % 50 == 0)
      {
        ...
         anchor = (ENUM_ANCHOR_POINT)(rand() * 9 / 32768);
         ObjectSetInteger(0, name, OBJPROP_ANCHOR, anchor);
      }
      ...
The name of the anchor point becomes the text content of the label, along with the current
coordinates.
      ObjectSetString(0, name, OBJPROP_TEXT, EnumToString(anchor)
         + "[" + (string)x + "," + (string)y + "]");
The rest of the code snippets remained largely unchanged.
After compiling and running the script, notice how the inscription changes its position relative to the
current coordinates (x, y) depending on the selected anchor point.
For now, we control and prevent the anchor point itself from going outside the window. However, the
object has some dimensions, and therefore it may turn out that most of the inscription is cut off. In the
future, after studying the relevant properties, we will deal with this problem (see the
Obj ectSizeLabel.mq5 example in the section on Determining object width and height).
5.8.1 2 Managing the object state
Among the general properties of objects, there are several ones that control the state of objects. All
such properties have a Boolean type, meaning they can be turned on (true) or off (false), and therefore
require the use of the functions Obj ectGetInteger and Obj ectSetInteger.

---

## Page 963

Part 5. Creating application programs
963
5.8 Graphical objects
Identifier
Description
OBJPROP_HIDDEN
Disable displaying the name of a graphical object in the list of objects
in the relevant dialog (called from the context menu of the chart or by
pressing Ctrl+B).
OBJPROP_SELECTED
Object selection
OBJPROP_SELECTABLE
Availability of an object for selection
A value of true for OBJPROP_HIDDEN allows you to hide an unnecessary object from the user's list. By
default, true is set for objects that display calendar events, the trading history, as well as for objects
created from MQL programs. To see such graphical objects and access their properties, press the All
button in the Obj ect List dialog.
An object hidden in the list remains visible on the chart. To hide an object on the chart without deleting
it, you can use the Visibility of objects in the context of timeframes setting.
The user cannot select and change the properties of objects for which OBJPROP_SELECTABLE is equal
to false. Objects created programmatically are not allowed to be selected by default. As we saw in the
Obj ectCornerLabel.mq5 and Obj ectAnchorLabel.mq5 scripts in the previous sections, it was necessary to
explicitly set OBJPROP_SELECTABLE to true to unlock the ability to include OBJPROP_SELECTED as
well. This is how we highlighted the anchor points on the object.
Usually, MQL programs allow the selection of their objects only if these objects serve as controls. For
example, a trend line with a predefined name, which the user moves at will, can mean a condition for
sending a trade order when the price crosses it.
5.8.1 3 Priority of objects (Z-Order)
Objects on the chart provide not only the presentation of information but also interaction with the user
and MQL programs through events, which will be discussed in detail in the next chapter. One of the
event sources is the mouse pointer. The chart is able, in particular, to track the movement of the
mouse and pressing its buttons.
If an object is under the mouse, specific event handling can be performed for it. However, objects can
overlap each other (when their coordinates overlap, taking into account sizes). In this case, the
OBJPROP_ZORDER integer property comes into play. It sets the priority of the graphical object to
receive mouse events. When objects overlap, only one object, whose priority is higher than the rest will
receive the event.
By default, when an object is created, its Z-order is zero, but you can increase it if necessary.
It's important to note that Z-order only affects the handling of mouse events, not the drawing of
objects. Objects are always drawn in the order they were added to the chart. This can be a source
of misunderstanding. For example, a tooltip may not be displayed for an object that is visually on
top of another because the overlapped object has a higher Z-priority (see example).
In the Obj ectZorder.mq5 script we will create 1 2 objects of type OBJ_RECTANGLE_LABEL, placing them
in a circle, like on a clock face. The order of adding objects corresponds to hours: from 1  to 1 2. For
clarity, all rectangles will get a random color (for the OBJPROP_BGCOLOR property, see the next
section), as well as random priority. By moving the mouse over objects, the user will be able to
determine which object it belongs to by means of a tooltip.

---

## Page 964

Part 5. Creating application programs
964
5.8 Graphical objects
For the convenience of setting the properties of objects, we define the special class Obj ectBuilder,
derived from Obj ect Selector.
#include "ObjectPrefix.mqh"
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
   
   // changing the name and chart is prohibited
   virtual void name(const string _id) override = delete;
   virtual void chart(const long _chart) override = delete;
};
Fields with identifiers of the object (id) and chart (host) are already in the Obj ectSelector class. In the
derivative, we add an object type (ENUM_ OBJECT type) and a window number (int window). The
constructor calls Obj ectCreate.
Setting and reading properties is fully inherited as a group of get and set methods from Obj ectSelector.
As in the previous test scripts, we determine the window where the script is dropped, the dimensions of
the window, and the coordinates of the middle.
void OnStart()
{
   const int t = ChartWindowOnDropped();
   int h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, t);
   int w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   int x = w / 2;
   int y = h / 2;
   ...
Since the object type OBJ_RECTANGLE_LABEL supports explicit pixel dimensions, we calculate the
width of dx and height of dy of each rectangle as a quarter window. We use them to set the
OBJPROP_XSIZE and OBJPROP_YSIZE properties discussed in the section on Determining object width
and height.
   const int dx = w / 4;
   const int dy = h / 4;
   ...
Next, in the loop, we create 1 2 objects. Variables px and py contain the offset of the next "mark" on
the "dial" relative to the center (x, y). The priority of z is chosen randomly. The name of the object and

---

## Page 965

Part 5. Creating application programs
965
5.8 Graphical objects
its tooltip (OBJPROP_TOOLTIP) include a string like "XX - YYY", XX is the number of the "hour" (the
position on the dial is from 1  to 1 2), YYY is the priority.
   for(int i = 0; i < 12; ++i)
   {
      const int px = (int)(MathSin((i + 1) * 30 * M_PI / 180) * dx) - dx / 2;
      const int py = -(int)(MathCos((i + 1) * 30 * M_PI / 180) * dy) - dy / 2;
      
      const int z = rand();
      const string text = StringFormat("%02d - %d", i + 1, z);
   
      ObjectBuilder *builder =
         new ObjectBuilder(ObjNamePrefix + text, OBJ_RECTANGLE_LABEL);
      builder.set(OBJPROP_XDISTANCE, x + px).set(OBJPROP_YDISTANCE, y + py)
      .set(OBJPROP_XSIZE, dx).set(OBJPROP_YSIZE, dy)
      .set(OBJPROP_TOOLTIP, text)
      .set(OBJPROP_ZORDER, z)
      .set(OBJPROP_BGCOLOR, (rand() << 8) | rand());
      delete builder;
   }
After the Obj ectBuilder constructor is called, for the new builder object the calls to the overloaded set
method for different properties are chained (the set method returns a pointer to the object itself).
Since the MQL object is no longer needed after the creation and configuration of the graphical object,
we immediately delete builder.
As a result of the script execution, approximately the following objects will appear on the chart.
Object overlay and Z-order priority tooltips

---

## Page 966

Part 5. Creating application programs
966
5.8 Graphical objects
The colors and priorities will be different each time you run it, but the visual overlay of the rectangles
will always be the same, in the order of creation from 1  at the bottom to 1 2 at the top (here we mean
the overlay of objects, not the fact that 1 2 is located at the top of the watch face ).
In the image, the mouse cursor is positioned in a place where two objects exist, that is, 01  (fluorescent
lime green) and 1 2 (sandy). In this case, the tooltip for object 01  is visible, although visually object 1 2
is displayed on top of object 01 . This is because 01  was randomly generated with a higher priority than
1 2.
Only one tooltip is displayed at a time, so you can check the priority relationship by moving the mouse
cursor to other areas where there is no object overlap and the information in the tooltip belongs to the
single object under the cursor.
When we learn about mouse event handling in the next chapter, we can improve on this example and
test the effect of Z-order on mouse clicks on objects.
To delete the created objects, you can use the Obj ectCleanup1 .mq5 script.
5.8.1 4 Object display settings: color, style, and frame
The appearance of objects can be changed using a variety of properties, which we'll explore in this
section, starting with color, style, line width, and borders. Other formatting aspects such as font, skew,
and text alignment will be covered in the following sections.
All properties from the table below have types that are compatible with integers and therefore are
managed by the functions Obj ectGetInteger and Obj ectSetInteger.

---

## Page 967

Part 5. Creating application programs
967
5.8 Graphical objects
Identifier
Description
Property type
OBJPROP_COLOR
The color of the line and the
main element of the object (for
example, font or fill)
color
OBJPROP_STYLE
Line style
ENUM_LINE_STYLE
OBJPROP_WIDTH
Line thickness in pixels
int
OBJPROP_FILL
Filling an object with color (for
OBJ_RECTANGLE,
OBJ_TRIANGLE, OBJ_ELLIPSE,
OBJ_CHANNEL,
OBJ_STDDEVCHANNEL,
OBJ_REGRESSION)
bool
OBJPROP_BACK
Object in the background
bool
OBJPROP_BGCOLOR
Background color for
OBJ_EDIT, OBJ_BUTTON,
OBJ_RECTANGLE_LABEL
color
OBJPROP_BORDER_TYPE
Frame type for rectangular
panel OBJ_RECTANGLE_LABEL
ENUM_BORDER_TYPE
OBJPROP_BORDER_COLOR
Frame color for input field
OBJ_EDIT and button
OBJ_BUTTON
color
Unlike most objects with lines (separate vertical and horizontal, trend, cyclic, channels, etc.), where
the OBJPROP_COLOR property defines the color of the line, for the OBJ_BITMAP_LABEL and
OBJ_BITMAP images it defines the frame color, and OBJPROP_STYLE defines the frame drawing type.
We have already met the ENUM_LINE_STYLE enumeration, used for OBJPROP_STYLE, in the chapter
on indicators, in the section on Plot settings.
It is necessary to distinguish the fill performed by the foreground color OBJPROP_COLOR from the
background color OBJPROP_BGCOLOR. Both are supported by different groups of object types, which
are listed in the table.
The OBJPROP_BACK property requires a separate explanation. The fact is that objects and indicators
are displayed on top of the price chart by default. The user can change this behavior for the entire
chart by going to the Setting dialog of the chart, and further to the Shared bookmark, the Chart on top
option. This flag also has a software counterpart, the CHART_FOREGROUND property (see Chart
display modes). However, sometimes it is desirable to remove not all objects, but only selected ones,
into the background. Then for them, you can set OBJPROP_BACK to true. In this case, the object will
be overlapped even by the grid and period separators, if they are enabled on the chart.
When the OBJPROP_FILL fill mode is enabled, the color of the bars falling inside the shape depends on
the OBJPROP_BACK property. By default, with OBJPROP_BACK equal to false, bars overlapping the
object are drawn in inverted color with respect to OBJPROP_COLOR (the inverted color is obtained by
switching all bits in the color value to the opposite ones, for example, 0x00FF7F is obtained for

---

## Page 968

Part 5. Creating application programs
968
5.8 Graphical objects
0xFF0080). With OBJPROP_BACK equal to true, bars are drawn in the usual way, since the object is
displayed in the background, "under" the chart (see an example below).
The ENUM_BORDER_TYPE enumeration contains the following elements:
Identifier
Appearance
BORDER_FLAT
Flat
BORDER_RAISED
Convex
BORDER_SUNKEN
Concave
When the border is flat (BORDER_FLAT), it is drawn as a line with color, style, and width according to
the properties OBJPROP_COLOR, OBJPROP_STYLE, OBJPROP_WIDTH. The convex and concave
versions imitate volume chamfers around the perimeter in shades of OBJPROP_BGCOLOR.
When the border color OBJPROP_BORDER_COLOR is not set (default, which corresponds to clrNone),
the input field is framed by a line of the main color OBJPROP_COLOR, and a three-dimensional frame
with chamfers in shades of OBJPROP_BGCOLOR is drawn around the button.
To test the new properties, consider the Obj ectStyle.mq5 script. In it, we will create 5 rectangles of the
OBJ_RECTANGLE type, i.e., with reference to time and prices. They will be evenly spaced across the
entire width of the window, highlighting the range between the maximum price High and minimum price
Low in each of the five time periods. For all objects, we will adjust and periodically change the line color,
style, and thickness, as well as the filling and display option behind the chart.
Let's use again the helper class Obj ectBuilder, derived from the Obj ect Selector. In contrast to the
previous section, we add to Obj ectBuilder a destructor in which we will call Obj ectDelete.
#include <MQL5Book/ObjectMonitor.mqh>
#include <MQL5Book/AutoPtr.mqh>
   
class ObjectBuilder: public ObjectSelector
{
...
public:
   ~ObjectBuilder()
   {
      ObjectDelete(host, id);
   }
   ...
};
This will make it possible to assign to this class not only the configuration of objects but also their
automatic removal upon completion of the script.
In the OnStart function, we find out the number of visible bars and the index of the first bar, and also
calculate the width of one rectangle in bars.

---

## Page 969

Part 5. Creating application programs
969
5.8 Graphical objects
#define OBJECT_NUMBER 5
   
void OnStart()
{
   const string name = "ObjStyle-";
   const int bars = (int)ChartGetInteger(0, CHART_VISIBLE_BARS);
   const int first = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR);
   const int rectsize = bars / OBJECT_NUMBER;
   ...
Let's reserve an array of smart pointers for objects to ensure the call of Obj ectBuilder destructors.
   AutoPtr<ObjectBuilder> objects[OBJECT_NUMBER];
Define a color palette and create 5 rectangle objects.
   color colors[OBJECT_NUMBER] = {clrRed, clrGreen, clrBlue, clrMagenta, clrOrange};
   
   for(int i = 0; i < OBJECT_NUMBER; ++i)
   {
      // find the indexes of the bars that determine the range of prices in the i-th time subrange
      const int h = iHighest(NULL, 0, MODE_HIGH, rectsize, i * rectsize);
      const int l = iLowest(NULL, 0, MODE_LOW, rectsize, i * rectsize);
      // create and set up an object in the i-th subrange
      ObjectBuilder *object = new ObjectBuilder(name + (string)(i + 1), OBJ_RECTANGLE);
      object.set(OBJPROP_TIME, iTime(NULL, 0, i * rectsize), 0);
      object.set(OBJPROP_TIME, iTime(NULL, 0, (i + 1) * rectsize), 1);
      object.set(OBJPROP_PRICE, iHigh(NULL, 0, h), 0);
      object.set(OBJPROP_PRICE, iLow(NULL, 0, l), 1);
      object.set(OBJPROP_COLOR, colors[i]);
      object.set(OBJPROP_WIDTH, i + 1);
      object.set(OBJPROP_STYLE, (ENUM_LINE_STYLE)i);
      // save to array
      objects[i] = object;
   }
   ...
Here, for each object, the coordinates of two anchor points are calculated; the initial color, style, and
line width are set.
Next, in an infinite loop, we change the properties of objects. When ScrollLock is on, the animation can
be paused.

---

## Page 970

Part 5. Creating application programs
970
5.8 Graphical objects
   const int key = TerminalInfoInteger(TERMINAL_KEYSTATE_SCRLOCK);
   int pass = 0;
   int offset = 0;
   
   for( ;!IsStopped(); ++pass)
   {
      Sleep(200);
      if(TerminalInfoInteger(TERMINAL_KEYSTATE_SCRLOCK) != key) continue;
      // change color/style/width/fill/background from time to time
      if(pass % 5 == 0)
      {
         ++offset;
         for(int i = 0; i < OBJECT_NUMBER; ++i)
         {
            objects[i][].set(OBJPROP_COLOR, colors[(i + offset) % OBJECT_NUMBER]);
            objects[i][].set(OBJPROP_WIDTH, (i + offset) % OBJECT_NUMBER + 1);
            objects[i][].set(OBJPROP_FILL, rand() > 32768 / 2);
            objects[i][].set(OBJPROP_BACK, rand() > 32768 / 2);
         }
      }
      ChartRedraw();
   }
Here's what it looks like on a chart.
OBJ_RECTAN GLE rectangles with different display settings
The left-most red rectangle has its fill mode turned on and is in the foreground. So, the bars inside it
are displayed in contrasting bright blue (clrAqua, also commonly known as cyan, which is the inverted

---

## Page 971

Part 5. Creating application programs
971 
5.8 Graphical objects
clrRed). The purple rectangle also has a fill, but with a background option, so the bars in it are
displayed in a standard way.
Please note that the orange rectangle completely covers the bars at the beginning and end of its sub-
range due to the large width of the lines and display on top of the chart.
When the fill is on, the line width is not taken into account. When the border width is greater than 1 ,
some broken line styles are not applied.
ObjectShapesDraw
For the second example of this section, remember the hypothetical shape-drawing program we
sketched out in Part 3 when we learned OOP. Our progress stopped at the fact that in the virtual
drawing method (and it was called draw) we could only print a message to the log that we were drawing
a specific shape. Now, after getting acquainted with graphic objects, we have the opportunity to
implement drawing.
Let's take the Shapes5stats.mq5 script as a starting point. The updated version will be called
Obj ectShapesDraw.mq5.
Recall that in addition to the base class Shape we have described several classes of shapes: Rectangle,
Ellipse, Triangle, Square, Circle. All of them successfully overlay graphic objects of types
OBJ_RECTANGLE, OBJ_ELLIPSE, OBJ_TRIANGLE. But there are some nuances.
All specified objects are bound to time and price coordinates, while our drawing program assumes
unified X and Y axes with point positioning. In this regard, we will need to set up a graph for drawing in a
special way and use the ChartXYToTimePrice function to recalculate screen points in time and price.
In addition, OBJ_ELLIPSE and OBJ_TRIANGLE objects allow arbitrary rotation (in particular, the small
and large radius of an ellipse can be rotated), while OBJ_RECTANGLE always has its sides oriented
horizontally and vertically. To simplify the example, we restrict ourselves to the standard position of all
shapes.
In theory, the new implementation should be viewed as a demonstration of graphical objects, and not a
drawing program. A more correct approach for full-fledged drawing, devoid of the restrictions that
graphic objects impose (since they are intended for other purposes in general - like chart marking), is
using graphic resources. Therefore, we will return to rethinking the drawing program in the chapter on
resources.
In the new Shape class, let's get rid of the nested structure Pair with object coordinates: this structure
served as a means to demonstrate several principles of OOP, but now it is easier to return the original
description of the fields int x, y directly to the class Shape. We will also add a field with the name of the
object.

---

## Page 972

Part 5. Creating application programs
972
5.8 Graphical objects
class Shape
{
   ...
protected:
   int x, y;
   color backgroundColor;
   const string type;
   string name;
   
   Shape(int px, int py, color back, string t) :
      x(px), y(py),
      backgroundColor(back),
      type(t)
   {
   }
   
public:
   ~Shape()
   {
      ObjectDelete(0, name);
   }
   ...
The name field will be required to set the properties of a graphical object, as well as to remove it from
the chart, which is logical to do in the destructor.
Since different types of shapes require a different number of points or characteristic sizes, we will add
the setup method, in addition to the draw virtual method, into the Shape interface:
virtual void setup(const int &parameters[]) = 0;
Recall that in the script we have implemented a nested class Shape::Registrator, which was engaged in
counting the number of shapes by type. The time has come to entrust it with something more
responsible to work as a factory of shapes. The "factory" classes or methods are good because they
allow you to create objects of different classes in a unified way.
To do this, we add to Registrator a method for creating a shape with the parameters that include the
mandatory coordinates of the first point, a color, and an array of additional parameters (each shape will
be able to interpret it according to its own rules, and in the future, read from or write to a file).
virtual Shape *create(const int px, const int py, const color back,
         const int &parameters[]) = 0;
The method is abstract virtual because certain types of shapes can only be created by derived registrar
classes described in descendant classes of Shape. To simplify the writing of derived logger classes, we
introduce a template class MyRegistrator with an implementation of the create method suitable for all
cases.

---

## Page 973

Part 5. Creating application programs
973
5.8 Graphical objects
template<typename T>
class MyRegistrator : public Shape::Registrator
{
public:
   MyRegistrator() : Registrator(typename(T))
   {
   }
   
   virtual Shape *create(const int px, const int py, const color back,
      const int &parameters[]) override
   {
      T *temp = new T(px, py, back);
      temp.setup(parameters);
      return temp;
   }
};
Here we call the constructor of some previously unknown shape T, adjust it by calling setup and return
an instance to the calling code.
Here's how it's used in the Rectangle class, which has two additional parameters for width and height.

---

## Page 974

Part 5. Creating application programs
974
5.8 Graphical objects
class Rectangle : public Shape
{
   static MyRegistrator<Rectangle> r;
   
protected:
   int dx, dy; // dimensions (width, height)
   
   Rectangle(int px, int py, color back, string t) :
      Shape(px, py, back, t), dx(1), dy(1)
   {
   }
   
public:
   Rectangle(int px, int py, color back) :
      Shape(px, py, back, typename(this)), dx(1), dy(1)
   {
      name = typename(this) + (string)r.increment();
   }
   
   virtual void setup(const int &parameters[]) override
   {
      if(ArraySize(parameters) < 2)
      {
         Print("Insufficient parameters for Rectangle");
         return;
      }
      dx = parameters[0];
      dy = parameters[1];
   }
   ...
};
   
static MyRegistrator<Rectangle> Rectangle::r;
When creating a shape, its name will contain not only the class name (typename), but also the ordinal
number of the instance, calculated in the r.increment() call.
Other classes of shapes are described similarly.
Now it's time to look into the draw method for Rectangle. In it, we translate a pair of points (x,y) and (x
+ dx, y + dy) into time/price coordinates using ChartXYToTimePrice and create an OBJ_RECTANGLE
object.

---

## Page 975

Part 5. Creating application programs
975
5.8 Graphical objects
   void draw() override
   {
      // Print("Drawing rectangle");
      int subw;
      datetime t;
      double p;
      ChartXYToTimePrice(0, x, y, subw, t, p);
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, t, p);
      ChartXYToTimePrice(0, x + dx, y + dy, subw, t, p);
      ObjectSetInteger(0, name, OBJPROP_TIME, 1, t);
      ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p);
   
      ObjectSetInteger(0, name, OBJPROP_COLOR, backgroundColor);
      ObjectSetInteger(0, name, OBJPROP_FILL, true);
   }
Of course, don't forget to set the color to OBJPROP_COLOR and the fill to OBJPROP_FILL.
For the Square class, nothing needs to be changed as such: it is enough just to set dx and dy equal to
each other.
For the Ellipse class, two additional options, dx and dy, determine the small and large radii plotted
relative to the center (x,y). Accordingly, in the method draw we calculate 3 anchor points and create
an OBJ_ELLIPSE object.

---

## Page 976

Part 5. Creating application programs
976
5.8 Graphical objects
class Ellipse : public Shape
{
   static MyRegistrator<Ellipse> r;
protected:
   int dx, dy; // large and small radii 
   ...
public:
   void draw() override
   {
      // Print("Drawing ellipse");
      int subw;
      datetime t;
      double p;
      
      // (x, y) center
      // p0: x + dx, y
      // p1: x - dx, y
      // p2: x, y + dy
      
      ChartXYToTimePrice(0, x + dx, y, subw, t, p);
      ObjectCreate(0, name, OBJ_ELLIPSE, 0, t, p);
      ChartXYToTimePrice(0, x - dx, y, subw, t, p);
      ObjectSetInteger(0, name, OBJPROP_TIME, 1, t);
      ObjectSetDouble(0, name, OBJPROP_PRICE, 1, p);
      ChartXYToTimePrice(0, x, y + dy, subw, t, p);
      ObjectSetInteger(0, name, OBJPROP_TIME, 2, t);
      ObjectSetDouble(0, name, OBJPROP_PRICE, 2, p);
      
      ObjectSetInteger(0, name, OBJPROP_COLOR, backgroundColor);
      ObjectSetInteger(0, name, OBJPROP_FILL, true);
   }
};
   
static MyRegistrator<Ellipse> Ellipse::r;
Circle is a special case of an ellipse with equal radii.
Finally, only equilateral triangles are supported at this stage: the size of the side is contained in an
additional field dx. You are invited to learn their methoddraw in the source code independently.
The new script will, as before, generate a given number of random shapes. They are created by the
function addRandomShape.

---

## Page 977

Part 5. Creating application programs
977
5.8 Graphical objects
Shape *addRandomShape()
{
   const int w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);
   const int h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);
   
   const int n = random(Shape::Registrator::getTypeCount());
   
   int cx = 1 + w / 4 + random(w / 2), cy = 1 + h / 4 + random(h / 2);
   int clr = ((random(256) << 16) | (random(256) << 8) | random(256));
   int custom[] = {1 + random(w / 4), 1 + random(h / 4)};
   return Shape::Registrator::get(n).create(cx, cy, clr, custom);
}
This is where we see the use of the factory method create, called on a randomly selected registrar
object with the number n. If we decide to add other shape classes later, we won't need to change
anything in the generation logic.
All shapes are placed in the central part of the window and have dimensions no larger than a quarter of
the window.
It remains to consider directly the calls to the addRandomShape function, and the special schedule
setting we have already mentioned.
To provide a "square" representation of points on the screen, set the CHART_SCALEFIX_1 1  mode. In
addition, we will choose the densest (compressed) scale along the time axis CHART_SCALE (0),
because in it one bar occupies 1  horizontal pixel (maximum accuracy). Finally, disable the display of the
chart itself by setting CHART_SHOW to false.
void OnStart()
{
   const int scale = (int)ChartGetInteger(0, CHART_SCALE);
   ChartSetInteger(0, CHART_SCALEFIX_11, true);
   ChartSetInteger(0, CHART_SCALE, 0);
   ChartSetInteger(0, CHART_SHOW, false);
   ChartRedraw();
   ...
To store the shapes, let's reserve an array of smart pointers and fill it with random shapes.

---

## Page 978

Part 5. Creating application programs
978
5.8 Graphical objects
#define FIGURES 21
...
void OnStart()
{
   ...
   AutoPtr<Shape> shapes[FIGURES];
   
   for(int i = 0; i < FIGURES; ++i)
   {
      Shape *shape = shapes[i] = addRandomShape();
      shape.draw();
   }
   
   ChartRedraw();
   ...
Then we run an infinite loop until the user stops the script, in which we slightly move the shapes using
the move method.
   while(!IsStopped())
   {
      Sleep(250);
      for(int i = 0; i < FIGURES; ++i)
      {
         shapes[i][].move(random(20) - 10, random(20) - 10);
         shapes[i][].draw();
      }
      ChartRedraw();
   }
   ...
In the end, we restore the chart settings.
   // it's not enough to disable CHART_SCALEFIX_11, you need CHART_SCALEFIX
   ChartSetInteger(0, CHART_SCALEFIX, false);
   ChartSetInteger(0, CHART_SCALE, scale);
   ChartSetInteger(0, CHART_SHOW, true);
}
The following screenshot shows what a graph with the shapes drawn might look like.

---

## Page 979

Part 5. Creating application programs
979
5.8 Graphical objects
Chart shape objects
The specifics of drawing objects is the "multiplication" of colors in those places where they overlap.
Because the Y-axis goes up and down, all the triangles are upside down, but that's not critical, because
we're going to redo the resource-based paint program anyway.
5.8.1 5 Font settings
All types of objects enable the setting of certain texts for them (OBJPROP_TEXT). Many of them
display the specified text directly on the chart, for the rest it becomes an informative part of the
tooltip.
When text is displayed inside an object (for types OBJ_TEXT, OBJ_LABEL, OBJ_BUTTON, and
OBJ_EDIT), you can choose a font name and size. For objects of other types, the font settings are not
applied: their descriptions are always displayed in the chart's standard font.
Identifier
Description
Type
OBJPROP_FONTSIZE
Font size in pixels
int
OBJPROP_FONT
Font
string
You cannot set the font size in printing points here.
The test script Obj ectFont.mq5 creates objects with text and changes the name and font size. Let's use
the Obj ectBuilder class from the previous script.

---

## Page 980

Part 5. Creating application programs
980
5.8 Graphical objects
At the beginning of OnStart, the script calculates the middle of the window both in screen coordinates
and in the time/price axes. This is required because objects of different types participating in the test
use different coordinate systems.
void OnStart()
{
   const string name = "ObjFont-";
   
   const int bars = (int)ChartGetInteger(0, CHART_WIDTH_IN_BARS);
   const int first = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR);
   
   const datetime centerTime = iTime(NULL, 0, first - bars / 2);
   const double centerPrice =
      (ChartGetDouble(0, CHART_PRICE_MIN)
      + ChartGetDouble(0, CHART_PRICE_MAX)) / 2;
   
   const int centerX = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) / 2;
   const int centerY = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS) / 2;
   ...
The list of tested object types is specified in the types array. For some of them, in particular
OBJ_HLINE and OBJ_VLINE, the font settings will have no effect, although the text of the descriptions
will appear on the screen (to ensure this, we turn on the CHART_SHOW_OBJECT_DESCR mode).
   ChartSetInteger(0, CHART_SHOW_OBJECT_DESCR, true);
   
   ENUM_OBJECT types[] =
   {
      OBJ_HLINE,
      OBJ_VLINE,
      OBJ_TEXT,
      OBJ_LABEL,
      OBJ_BUTTON,
      OBJ_EDIT,
   };
   int t = 0; // cursor
   ...
The t variable will be used to sequentially switch from one type to another.
The fonts array contains the most popular standard Windows fonts.

---

## Page 981

Part 5. Creating application programs
981 
5.8 Graphical objects
   string fonts[] =
   {
      "Comic Sans MS",
      "Consolas",
      "Courier New",
      "Lucida Console",
      "Microsoft Sans Serif",
      "Segoe UI",
      "Tahoma",
      "Times New Roman",
      "Trebuchet MS",
      "Verdana"
   };
   
   int f = 0; // cursor
   ...
We will iterate over them using the f variable.
Inside the demo loop, we instruct Obj ectBuilder to create an object of the current type types[t] in the
middle of the window (for unification, the coordinates are specified in both coordinate systems, so as
not to make differences in the code depending on the type: coordinates not supported by the object
simply will not have an effect).
   while(!IsStopped())
   {
      
      const string str = EnumToString(types[t]);
      ObjectBuilder *object = new ObjectBuilder(name + str, types[t]);
      object.set(OBJPROP_TIME, centerTime);
      object.set(OBJPROP_PRICE, centerPrice);
      object.set(OBJPROP_XDISTANCE, centerX);
      object.set(OBJPROP_YDISTANCE, centerY);
      object.set(OBJPROP_XSIZE, centerX / 3 * 2);
      object.set(OBJPROP_YSIZE, centerY / 3 * 2);
      ...
Next, we set up the text and font (the size is chosen randomly).
      const int size = rand() * 15 / 32767 + 8;
      Comment(str + " " + fonts[f] + " " + (string)size);
      object.set(OBJPROP_TEXT, fonts[f] + " " + (string)size);
      object.set(OBJPROP_FONT, fonts[f]);
      object.set(OBJPROP_FONTSIZE, size);
      ...
For the next pass, we move the cursors in the arrays of object types and font names.
      t = ++t % ArraySize(types);
      f = ++f % ArraySize(fonts);
      ...
Finally, we update the chart, wait 1  second, and delete the object to create another one.

---

## Page 982

Part 5. Creating application programs
982
5.8 Graphical objects
      ChartRedraw();
      Sleep(1000);
      delete object;
   }
}
The image below shows the moment the script is running.
Button with custom font settings
5.8.1 6 Rotating text at an arbitrary angle
Objects of text types – label OBJ_TEXT (in quotation coordinates) and panel OBJ_LABEL (in screen
coordinates) – allow you to rotate the text label at an arbitrary angle. For this purpose, there is the
OBJPROP_ANGLE property of type double. It contains the angle in degrees relative to the object's
normal position. Positive values rotate the object counterclockwise, and negative values rotate it
clockwise.
However, it should be borne in mind that angles with a difference that is a multiple of 360 degrees are
identical, that is, for example, +31 5 and -45 are the same. Rotation is performed around the anchor
point on the object (by default, top left).

---

## Page 983

Part 5. Creating application programs
983
5.8 Graphical objects
Rotate OBJ_LABEL and OBJ_TEXT objects by angles that are multiples of 45 degrees
You can check the effect of the OBJPROP_ANGLE property on an object using the Obj ectAngle.mq5
script. It creates a text label OBJ_LABEL in the center of the window, after which it begins to
periodically rotate 45 degrees until the user stops the process.
void OnStart()
{
   const string name = "ObjAngle";
   ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   const int centerX = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) / 2;
   const int centerY = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS) / 2;
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, centerX);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, centerY);
   ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_CENTER);
   
   int angle = 0;
   while(!IsStopped())
   {
      ObjectSetString(0, name, OBJPROP_TEXT, StringFormat("Angle: %d°", angle));
      ObjectSetDouble(0, name, OBJPROP_ANGLE, angle);
      angle += 45;
     
      ChartRedraw();
      Sleep(1000);
   }
   ObjectDelete(0, name);
}
The text displays the current value of the angle.

---

## Page 984

Part 5. Creating application programs
984
5.8 Graphical objects
5.8.1 7 Determining object width and height
Some types of objects allow you to set their dimensions in pixels. These include OBJ_BUTTON,
OBJ_CHART, OBJ_BITMAP, OBJ_BITMAP_LABEL, OBJ_EDIT, and OBJ_RECTANGLE_LABEL. In addition,
OBJ_LABEL objects support reading (but not setting) sizes because labels automatically expand or
contract to fit the text they contain. Attempting to access properties on other types of objects will
result in an OBJECT_WRONG_PROPERTY (4203) error.
Identifier
Description
OBJPROP_XSIZE
Width of the object along the X-axis in pixels
OBJPROP_YSIZE
Height of the object along the Y-axis in pixels
Both sizes are integers and are therefore handled by the Obj ectGetInteger/Obj ectSetInteger functions.
Special dimension handling is performed for OBJ_BITMAP and OBJ_BITMAP_LABEL objects.
Without assigning an image, these objects allow you to set an arbitrary size. At the same time, they are
drawn transparent (only the frame is visible if it is not "hidden" also by setting the color clrNone), but
they receive all events, in particular, about mouse movements (with a text description, if any, in a
tooltip) and clicks of its buttons on the object.
When an image is assigned, it defaults to the object's height and width. However, an MQL program can
set smaller sizes and select a fragment of an image to display; more on this in the section on framing.
If you try to set the height or width larger than the image size, it stops being displayed, and the
object's dimensions do not change.
As an example, let's develop an improved version of the script Obj ectAnchorLabel.mq5 from the section
titled Defining anchor point on an object. In that section, we moved the text label around the window
and reversed it when it reached any of the window borders, but we did this while only taking into
account the anchor point. Because of this, depending on the location of the anchor point on the object,
there could be a situation where the label is almost completely traveled beyond the window. For
example, if the anchor point was on the right side of the object, moving to the left would cause almost
all of the text to go beyond the left border of the window before the anchor point touched the edge.
In the new script Obj ectSizeLabel.mq5, we will take into account the size of the object and change the
direction of movement as soon as it touches the edge of the window with any of its sides.
For the correct implementation of this mode, it should be taken into account that each window corner
used as the center of reference of coordinates to the anchor point on the object determines the
characteristic direction of both the X and Y axes. For example, if the user selects the upper left corner
in the ENUM_BASE_CORNER Corner input variable, then X increases from left to right and Y increases
from top to bottom. If the center is considered to be the lower right corner, then X increases from right
to left of it, and Y increases from bottom to top.
A different mutual combination of the anchor corner in the window and the anchor point on the object
requires different adjustments of the distances between the object edges and the window borders. In
particular, when one of the right corners and one of the anchor points on the right side of the object is
selected, then the correction at the right border of the window is not required, and at the opposite side,
the left, we must take into account the width of the object (so that its dimensions do not go out of the
window to the left).

---

## Page 985

Part 5. Creating application programs
985
5.8 Graphical objects
This rule about correcting for the size of an object can be generalized:
·On the border of the window adjacent to the anchor corner, the correction is needed when the
anchor point is on the far side of the object relative to this corner;
·On the border of the window opposite the anchor corner, the correction is needed when the anchor
point is on the near side of the object relative to this corner.
In other words, if the name of the corner (in the ENUM_BASE_CORNER element) and the anchor point
(in the ENUM_ANCHOR_POINT element) contain a common word (for example, RIGHT), the correction
is needed on the far side of the window (that is, far from the selected corner). If opposite directions are
found in the combination of ENUM_BASE_CORNER and ENUM_ANCHOR_POINT sides (for example,
LEFT and RIGHT), the correction is needed at the nearest side of the window. These rules work the
same for the horizontal and vertical axes.
Additionally, it should be taken into account that the anchor point can be in the middle of any side of
the object. Then in the perpendicular direction, an indent from the window borders is required, equal to
half the size of the object.
A special case is the anchor point at the center of the object. For it, you should always have a margin
of distance in any direction, equal to half the size of the object.
The logic described is implemented in a special function called GetMargins. It takes as inputs the
selected corner and anchor point, as well as the dimensions of the object (dx and dy). The function
returns a structure with 4 fields containing the sizes of additional indents that should be set aside from
the anchor point in the direction of the near and far borders of the window so that the object does not
go out of view. Indents reserve the distance according to the dimensions and relative position of the
object itself.
struct Margins
{
   int nearX; // X increment between the object point and the window border adjacent to the corner
   int nearY; // Y increment between the object point and the window border adjacent to the corner
   int farX;  // X increment between the object's point and the opposite corner of the window border
   int farY;  // Y increment between the object's point and the opposite corner of the window border
};
   
Margins GetMargins(const ENUM_BASE_CORNER corner, const ENUM_ANCHOR_POINT anchor,
   int dx, int dy)
{
   Margins margins = {}; // zero corrections by default
   ...
   return margins;
}
To unify the algorithm, the following macro definitions of directions (sides) are introduced:
   #define LEFT 0x1
   #define LOWER 0x2
   #define RIGHT 0x4
   #define UPPER 0x8
   #define CENTER 0x16
With their help, bit masks (combinations) are defined that describe the elements of the
ENUM_BASE_CORNER and ENUM_ANCHOR_POINT enumerations.

---

## Page 986

Part 5. Creating application programs
986
5.8 Graphical objects
   const int corner_flags[] = // flags for ENUM_BASE_CORNER elements
   {
      LEFT | UPPER,
      LEFT | LOWER,
      RIGHT | LOWER,
      RIGHT | UPPER
   };
   
   const int anchor_flags[] = // flags for ENUM_ANCHOR_POINT elements
   {
      LEFT | UPPER,
      LEFT,
      LEFT | LOWER,
      LOWER,
      RIGHT | LOWER,
      RIGHT,
      RIGHT | UPPER,
      UPPER,
      CENTER
   };
Each of the arrays, corner_ flags and anchor_ flags, contains exactly as many elements as there are in
the corresponding enumeration.
Next comes the main function code. First of all, let's deal with the simplest option: the central anchor
point.
   if(anchor == ANCHOR_CENTER)
   {
      margins.nearX = margins.farX = dx / 2;
      margins.nearY = margins.farY = dy / 2;
   }
   else
   {
      ...
   }
To analyze the rest of the situations, we will use the bit masks from the above arrays by directly
addressing them by the received values corner and anchor.
      const int mask = corner_flags[corner] & anchor_flags[anchor];
      ...
If the corner and the anchor point are on the same horizontal side, the following condition will work and
the object width at the far edge of the window will be adjusted.
      if((mask & (LEFT | RIGHT)) != 0)
      {
         margins.farX = dx;
      }
      ...
If they are not on the same side, then they may be on opposite sides, or it may be the case that the
anchor point is in the middle of the horizontal side (top or bottom). Checking for an anchor point in the

---

## Page 987

Part 5. Creating application programs
987
5.8 Graphical objects
middle is done using the expression (anchor_ flags[anchor] & (LEFT |  RIGHT)) == 0 - then the correction
is equal to half the width of the object.
      else
      {
         if((anchor_flags[anchor] & (LEFT | RIGHT)) == 0)
         {
            margins.nearX = dx / 2;
            margins.farX = dx / 2;
         }
         else
         {
            margins.nearX = dx;
         }
      }
      ...
Otherwise, with the opposite orientation of the corner and the anchor point, we make an adjustment to
the width of the object at the near border of the window.
Similar checks are made for the Y-axis.
      if((mask & (UPPER | LOWER)) != 0)
      {
         margins.farY = dy;
      }
      else
      {
         if((anchor_flags[anchor] & (UPPER | LOWER)) == 0)
         {
            margins.farY = dy / 2;
            margins.nearY = dy / 2;
         }
         else
         {
            margins.nearY = dy;
         }
      }
Now the GetMargins function is ready, and we can proceed to the main code of the script in the
OnStart function. As before, we determine the size of the window, calculate the initial coordinates in
the center, create an OBJ_LABEL object, and select it.

---

## Page 988

Part 5. Creating application programs
988
5.8 Graphical objects
void OnStart()
{
   const int t = ChartWindowOnDropped();
   Comment(EnumToString(Corner));
   
   const string name = "ObjSizeLabel";
   int h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, t) - 1;
   int w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) - 1;
   int x = w / 2;
   int y = h / 2;
      
   ObjectCreate(0, name, OBJ_LABEL, t, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTED, true);
   ObjectSetInteger(0, name, OBJPROP_CORNER, Corner);
   ...
For animation, an infinite loop provides variables pass (iteration counter) and anchor (the anchor point,
which will be periodically chosen randomly).
   int pass = 0;
   ENUM_ANCHOR_POINT anchor = 0;
   ...
But there are some changes compared to Obj ectAnchorLabel.mq5.
We will not generate random movements of the object. Instead, let's set a constant speed of 5 pixels
diagonally.
   int px = 5, py = 5;
To store the size of the text label, we will reserve two new variables.
   int dx = 0, dy = 0;
The result of counting additional indents will be stored in a variable m of type Margins.
   Margins m = {};
This is followed directly by the loop of moving and modifying the object. In it, at every 75th iteration
(one iteration of 1 00 ms, see further), we randomly select a new anchor point, form a new text (the
contents of the object) from it, and wait for the changes to be applied to the object (calling
ChartRedraw). The latter is necessary because the size of the inscription is automatically adjusted to
the content, and the new size is important for us in order to correctly calculate the indents in the
GetMargins call.
We get the dimensions using calls Obj ectGetInteger with properties OBJPROP_XSIZE and
OBJPROP_YSIZE.

---

## Page 989

Part 5. Creating application programs
989
5.8 Graphical objects
   for( ;!IsStopped(); ++pass)
   {
      if(pass % 75 == 0)
      {
         // ENUM_ANCHOR_POINT consists of 9 elements: randomly choose one
         const int r = rand() * 8 / 32768 + 1;
         anchor = (ENUM_ANCHOR_POINT)((anchor + r) % 9);
         ObjectSetInteger(0, name, OBJPROP_ANCHOR, anchor);
         ObjectSetString(0, name, OBJPROP_TEXT, " " + EnumToString(anchor)
            + StringFormat("[%3d,%3d] ", x, y));
         ChartRedraw();
         Sleep(1);
   
         dx = (int)ObjectGetInteger(0, name, OBJPROP_XSIZE);
         dy = (int)ObjectGetInteger(0, name, OBJPROP_YSIZE);
         
         m = GetMargins(Corner, anchor, dx, dy);
      }
      ...
Once we know the anchor point and all distances, we move the object. If it "bumps" against the wall,
we change the direction of movement to the opposite (px to -px or py to -py, depending on the side).

---

## Page 990

Part 5. Creating application programs
990
5.8 Graphical objects
      // bounce off window borders, object fully visible
      if(x + px >= w - m.farX)
      {
         x = w - m.farX + px - 1;
         px = -px;
      }
      else if(x + px < m.nearX)
      {
         x = m.nearX + px;
         px = -px;
      }
      
      if(y + py >= h - m.farY)
      {
         y = h - m.farY + py - 1;
         py = -py;
      }
      else if(y + py < m.nearY)
      {
         y = m.nearY + py;
         py = -py;
      }
      
      // calculate the new label position
      x += px;
      y += py;
      ...
It remains to update the state of the object itself: display the current coordinates in the text label and
assign them to the OBJPROP_XDISTANCE and OBJPROP_YDISTANCE properties.
      ObjectSetString(0, name, OBJPROP_TEXT, " " + EnumToString(anchor)
         + StringFormat("[%3d,%3d] ", x, y));
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
      ...
After changing the object, we call ChartRedraw and wait 1 00ms to ensure a reasonably smooth
animation.
      ChartRedraw();
      Sleep(100);
      ...
At the end of the loop, we check the window size again, since the user can change it while the script is
running, and we also repeat the size request.
      h = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS, t) - 1;
      w = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) - 1;
      
      dx = (int)ObjectGetInteger(0, name, OBJPROP_XSIZE);
      dy = (int)ObjectGetInteger(0, name, OBJPROP_YSIZE);
      m = GetMargins(Corner, anchor, dx, dy);

---

## Page 991

Part 5. Creating application programs
991 
5.8 Graphical objects
   }
We omitted some other innovations of the Obj ectSizeLabel.mq5 script in order to keep the explanation
concise. Those who wish can refer to the code. In particular, distinctive colors were used for the
inscription: each specific color corresponds to its own anchor point, which makes the switching points
more noticeable. Also, you can click Delete while the script is running: this will remove the selected
object from the chart and the script will automatically end.
5.8.1 8 Visibility of objects in the context of timeframes
MetaTrader 5 users know about the Display tab in the object properties dialog where you can use the
switches to select on which timeframes the object will be displayed and on which it will be hidden. In
particular, you can temporarily hide the object completely on all timeframes.
MQL5 has a similar program property, OBJPROP_TIMEFRAMES, which controls the object visibility on a
timeframe. The value of this property can be any combination of special integer flags: each flag
(constant) contains a bit corresponding to one timeframe (see the table). To set/get the
OBJPROP_TIMEFRAMES property, use the Obj ectSetInteger/Obj ectGetInteger functions.
Constant
Value
Visibility in timeframes
OBJ_NO_PERIODS
0
The object is hidden in all timeframes
OBJ_PERIOD_M1 
0x00000001 
M1 
OBJ_PERIOD_M2
0x00000002
M2
OBJ_PERIOD_M3
0x00000004
M3
OBJ_PERIOD_M4
0x00000008
M4
OBJ_PERIOD_M5
0x0000001 0
M5
OBJ_PERIOD_M6
0x00000020
M6
OBJ_PERIOD_M1 0
0x00000040
M1 0
OBJ_PERIOD_M1 2
0x00000080
M1 2
OBJ_PERIOD_M1 5
0x000001 00
M1 5
OBJ_PERIOD_M20
0x00000200
M20
OBJ_PERIOD_M30
0x00000400
M30
OBJ_PERIOD_H1 
0x00000800
H1 
OBJ_PERIOD_H2
0x00001 000
H2
OBJ_PERIOD_H3
0x00002000
H3
OBJ_PERIOD_H4
0x00004000
H4
OBJ_PERIOD_H6
0x00008000
H6
OBJ_PERIOD_H8
0x0001 0000
H8

---

## Page 992

Part 5. Creating application programs
992
5.8 Graphical objects
Constant
Value
Visibility in timeframes
OBJ_PERIOD_H1 2
0x00020000
H1 2
OBJ_PERIOD_D1 
0x00040000
D1 
OBJ_PERIOD_W1 
0x00080000
W1 
OBJ_PERIOD_MN1 
0x001 00000
MN1 
OBJ_ALL_PERIODS
0x001 fffff
All timeframes
The flags can be combined using the bitwise OR operator ("| "), for example, the superposition of flags
OBJ_PERIOD_M1 5 |  OBJ_PERIOD_H4 means that the object will be visible on the 1 5-minute and 4-hour
timeframes.
Note that each flag can be obtained by shifting by 1  to the left by the number of bits equal to the
number of the constant in the table. This makes it easier to generate flags dynamically when the
algorithm operates in multiple timeframes rather than one particular one.
We will use this feature in the test script Obj ectTimeframes.mq5. Its task is to create a lot of large text
labels on the chart with the names of the timeframes, and each title should be displayed only in the
corresponding timeframe. For example, a large label "D1 " will be visible only on the daily chart, and
when switching to H4, we will see "H4".
To get the short name of the timeframe, without the "PERIOD_" prefix, a simple auxiliary function is
implemented.
string GetPeriodName(const int tf)
{
   const static int PERIOD_ = StringLen("PERIOD_");
   return StringSubstr(EnumToString((ENUM_TIMEFRAMES)tf), PERIOD_);
}
To get the list of all timeframes from the ENUM_TIMEFRAMES enumeration, we will use the
EnumToArray function which was presented in the section on conversion of Enumerations.
#include "ObjectPrefix.mqh"
#include <MQL5Book/EnumToArray.mqh>
void OnStart()
{
   ENUM_TIMEFRAMES tf = 0;
   int values[];
   const int n = EnumToArray(tf, values, 0, USHORT_MAX);
   ...
All labels will be displayed in the center of the window at the moment the script is launched. Resizing
the window after the script ends will cause the created captions to no longer be centered. This is a
consequence of the fact that MQL5 supports anchoring only to the corners of the window, but not to
the center. If you want to automatically maintain the position of objects, you should implement a
similar algorithm in the indicator and respond to window resize events. Alternatively, we could display
labels in a corner, for example, the lower right.

---

## Page 993

Part 5. Creating application programs
993
5.8 Graphical objects
   const int centerX = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS) / 2;
   const int centerY = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS) / 2;
   ...
In the cycle through timeframes, we create an OBJ_LABEL object for each of them, and place it in the
middle of the window anchored in the center of the object.
   for(int i = 1; i < n; ++i)
   {
      // create and setup text label for each timeframe
      const string name = ObjNamePrefix + (string)i;
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, centerX);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, centerY);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_CENTER);
      ...
Next, we set the text (name of the timeframe), large font size, gray color and display property in the
background.
      ObjectSetString(0, name, OBJPROP_TEXT, GetPeriodName(values[i]));
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fmin(centerY, centerX));
      ObjectSetInteger(0, name, OBJPROP_COLOR, clrLightGray);
      ObjectSetInteger(0, name, OBJPROP_BACK, true);
      ...
Finally, we generate the correct visibility flag for the i-th timeframe and write it to the
OBJPROP_TIMEFRAMES property.
      const int flag = 1 << (i - 1);
      ObjectSetInteger(0, name, OBJPROP_TIMEFRAMES, flag);
   }
See what happened on the chart when switching timeframes.

---

## Page 994

Part 5. Creating application programs
994
5.8 Graphical objects
Labels with timeframe names
If you open the Obj ect List dialog and enable All objects in the list, it is easy to make sure that there are
generated labels for all timeframes and check their visibility flags.
To remove objects, you can run the Obj ectCleanup1 .mq5 script.
5.8.1 9 Assigning a character code to a label
As mentioned in the review of Objects linked to time and price, the OBJ_ARROW label allows you to
display an arbitrary Wingdings font symbol on the chart (the full list of available symbols is provided in
the MQL5 documentation). The character code for the object itself is determined by the integer
property OBJPROP_ARROWCODE.
Script allows to demonstrate all characters of the Obj ectWingdings.mq5 font. In it, we create labels with
different characters in a loop, placing them one by one on the bar.

---

## Page 995

Part 5. Creating application programs
995
5.8 Graphical objects
#include "ObjectPrefix.mqh"
   
void OnStart()
{
   for(int i = 33; i < 256; ++i) // character codes
   {
      const int b = i - 33; // bar number
      const string name = ObjNamePrefix + "Wingdings-"
         + (string)iTime(_Symbol, _Period, b);
      ObjectCreate(0, name, OBJ_ARROW,
         0, iTime(_Symbol, _Period, b), iOpen(_Symbol, _Period, b));
      ObjectSetInteger(0, name, OBJPROP_ARROWCODE, i);
   }
   
   PrintFormat("%d objects with arrows created", 256 - 33);
}
How it looks on the chart is shown in the following screenshot.
Wingdings characters in OBJ_ARROW labels
5.8.20 Ray properties for objects with straight lines
Among graphical objects, there are several types in which the lines between anchor points can be
displayed either as segments (i.e., strictly between a pair of points) or as endless straight lines
continuing in one or another direction across the entire window visibility area. Such objects are:
• Trend line
• Trendline by angle

---

## Page 996

Part 5. Creating application programs
996
5.8 Graphical objects
• All types of channels (equidistant, standard deviations, regression, Andrews pitchfork)
• Gann line
• Fibonacci lines
• Fibonacci channel
• Fibonacci expansion
For them, you can separately enable line continuation to the left or right using the OBJPROP_RAY_LEFT
and OBJPROP_RAY_RIGHT properties, respectively. In addition, for a vertical line, you can specify
whether it should be drawn in all chart subwindows or only in the current one (where the anchor point is
located): the OBJPROP_RAY property is responsible for this. All properties are boolean, meaning they
can be enabled (true) or disabled (false).
Identifier
Description
OBJPROP_RAY_LEFT
Ray continues to the left
OBJPROP_RAY_RIGHT
Ray continues to the right
OBJPROP_RAY
Vertical line extends to all chart windows
You can check the operation of the rays using the Obj ectRays.mq5 script. It creates 3 standard
deviation channels with different ray settings.
One specific object is created and configured by the helper function SetupChannel. Through its
parameters, the channel length in bars and the channel width (deviation) are set, as well as options for
displaying rays to the left and right, and color.

---

## Page 997

Part 5. Creating application programs
997
5.8 Graphical objects
#include "ObjectPrefix.mqh"
   
void SetupChannel(const int length, const double deviation = 1.0,
   const bool right = false, const bool left = false,
   const color clr = clrRed)
{
   const string name = ObjNamePrefix + "Channel"
      + (right ? "R" : "") + (left ? "L" : "");
   // NB: Anchor point 0 must have an earlier time than anchor point 1,
   // otherwise the channel will degenerate
   ObjectCreate(0, name, OBJ_STDDEVCHANNEL, 0, iTime(NULL, 0, length), 0);
   ObjectSetInteger(0, name, OBJPROP_TIME, 1, iTime(NULL, 0, 0));
   // deviation
   ObjectSetDouble(0, name, OBJPROP_DEVIATION, deviation);
   // color and description
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetString(0, name, OBJPROP_TEXT, StringFormat("%2.1", deviation)
      + ((!right && !left) ? " NO RAYS" : "")
      + (right ? " RIGHT RAY" : "") + (left ? " LEFT RAY" : ""));
   // properties of rays
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, right);
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT, left);
   // lighting up objects by highlighting
   // (besides, it's easier for the user to remove them)
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, true);
   ObjectSetInteger(0, name, OBJPROP_SELECTED, true);
}
In the OnStart function, we call SetupChannel for 3 different channels.
void OnStart()
{
   SetupChannel(24, 1.0, true);
   SetupChannel(48, 2.0, false, true, clrBlue);
   SetupChannel(36, 3.0, false, false, clrGreen);
}
As a result, we get a chart of the following form.

---

## Page 998

Part 5. Creating application programs
998
5.8 Graphical objects
Channels with different OBJPROP_RAY_LEFT and OBJPROP_RAY_RIGHT property settings
When rays are enabled, it becomes possible to request the object to extrapolate time and price values
using the functions that we will describe in the Getting time or price at given points on lines section.
5.8.21  Managing object pressed state
For objects like buttons (OBJ_BUTTON) and panels with an image (OBJ_BITMAP_LABEL), the terminal
supports a special property that visually switches the object from the normal (released) state to the
pressed state and vice versa. The OBJPROP_STATE constant is reserved for this. The property is of a
Boolean type: when the value is true, the object is considered to be pressed, and when it is false, it is
considered to be released (by default).
For OBJ_BUTTON, the effect of a three-dimensional frame is drawn by the terminal itself, while for
OBJ_BITMAP_LABEL the programmer must specify two images (as files or resources) that will provide a
suitable external representation. Because this property is technically just a toggle, it's easy to use it for
other purposes, and not just for "press" and "release" effects. For example, with the help of appropriate
images, you can implement a flag (option).
The use of images in objects will be discussed in the next section.
The object state usually changes in interactive MQL programs that respond to user actions, in
particular mouse clicks. We will discuss this possibility in the chapter on events.
Now let's test the property on simple buttons, in static mode. The Obj ectButtons.mq5 script creates
two buttons on the chart: one in the pressed state, and the other in the released state. 
The setting of a single button is given to the SetupButton function with parameters that specify the
name and text of the button, as well as its coordinates, size, and state.

---

## Page 999

Part 5. Creating application programs
999
5.8 Graphical objects
#include "ObjectPrefix.mqh"
   
void SetupButton(const string button,
   const int x, const int y,
   const int dx, const int dy,
   const bool state = false)
{
   const string name = ObjNamePrefix + button;
   ObjectCreate(0, name, OBJ_BUTTON, 0, 0, 0);
   // position and size
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, dx);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, dy);
   // label on the button
   ObjectSetString(0, name, OBJPROP_TEXT, button);
   
   // pressed (true) / released (false)
   ObjectSetInteger(0, name, OBJPROP_STATE, state);
}
Then in OnStart we call this function twice.
void OnStart()
{
   SetupButton("Pressed", 100, 100, 100, 20, true);
   SetupButton("Normal", 100, 150, 100, 20);
}
The resulting buttons might look like this.

---

## Page 1000

Part 5. Creating application programs
1 000
5.8 Graphical objects
Pressed and released OBJ_BUTTON  buttons
Interestingly, you can click on any of the buttons with the mouse, and the button will change its state.
However, we have not yet discussed how to intercept a notification about this.
It is important to note that this automatic state switching is performed only if the option Disable
selection is checked in the object properties, but this condition is the default for all objects created
programmatically. Recall that, if necessary, this selection can be enabled: for this, you must explicitly
set the OBJPROP_SELECTABLE property to true. We used it in some previous examples.
To remove buttons that have become unnecessary, use the Obj ectCleanup1 .mq5 script.
5.8.22 Adjusting images in bitmap objects
Objects of OBJ_BITMAP_LABEL type (a panel with a picture positioned in screen coordinates) allow
displaying bitmap images. Bitmap images mean the BMP graphic format: although in principle there are
many other raster formats (for example, PNG or GIF), they are currently not supported in MQL5, just
like vector ones.
The string property OBJPROP_BMPFILE allows you to specify an image for an object. It must contain
the name of the BMP file or resource.
Since this object supports the possibility of two-position state switching (see OBJPROP_STATE), a
modifier parameter should be used for it: a picture for the "on"/"pressed" state is set under index 0,
and the "off"/"released" state is set under index 1 . If you specify only one picture (no modifier, which
is equivalent to 0), it will be used for both states. The default state of an object is "off"/"released".
The size of the object becomes equal to the size of the image, but it can be changed by specifying
smaller values in the OBJPROP_XSIZE and OBJPROP_YSIZE properties: in this case, only a part of the
image is displayed (for details, see the next section on framing).

---

