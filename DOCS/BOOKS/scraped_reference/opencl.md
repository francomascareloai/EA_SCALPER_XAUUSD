---
title: "Working with OpenCL"
url: "https://www.mql5.com/en/docs/opencl"
hierarchy: []
scraped_at: "2025-11-28 09:31:06"
---

# Working with OpenCL

[MQL5 Reference](/en/docs "MQL5 Reference")Working with OpenCL

* [CLHandleType](/en/docs/opencl/clhandletype "CLHandleType")
* [CLGetInfoInteger](/en/docs/opencl/clgetinfointeger "CLGetInfoInteger")
* [CLGetInfoString](/en/docs/opencl/clgetinfostring "CLGetInfoString")
* [CLContextCreate](/en/docs/opencl/clcontextcreate "CLContextCreate")
* [CLContextFree](/en/docs/opencl/clcontextfree "CLContextFree")
* [CLGetDeviceInfo](/en/docs/opencl/clgetdeviceinfo "CLGetDeviceInfo")
* [CLProgramCreate](/en/docs/opencl/clprogramcreate "CLProgramCreate")
* [CLProgramFree](/en/docs/opencl/clprogramfree "CLProgramFree")
* [CLKernelCreate](/en/docs/opencl/clkernelcreate "CLKernelCreate")
* [CLKernelFree](/en/docs/opencl/clkernelfree "CLKernelFree")
* [CLSetKernelArg](/en/docs/opencl/clsetkernelarg "CLSetKernelArg")
* [CLSetKernelArgMem](/en/docs/opencl/clsetkernelargmem "CLSetKernelArgMem")
* [CLSetKernelArgMemLocal](/en/docs/opencl/clsetkernelargmemlocal "CLSetKernelArgMemLocal")
* [CLBufferCreate](/en/docs/opencl/clbuffercreate "CLBufferCreate")
* [CLBufferFree](/en/docs/opencl/clbufferfree "CLBufferFree")
* [CLBufferWrite](/en/docs/opencl/clbufferwrite "CLBufferWrite")
* [CLBufferRead](/en/docs/opencl/clbufferread "CLBufferRead")
* [CLExecute](/en/docs/opencl/clexecute "CLExecute")
* [CLExecutionStatus](/en/docs/opencl/clexecutionstatus "CLExecutionStatus")

# Working with OpenCL

[OpenCL](https://www.khronos.org/opencl/ "The official website of the OpenCL standard") programs are used for performing computations on video cards that support OpenCL 1.1 or higher. Modern video cards contain hundreds of small specialized processors that can simultaneously perform simple mathematical operations with incoming data streams. The OpenCL language organizes parallel computing and provides greater speed for a certain class of tasks.

In some graphic cards working with the [double](/en/docs/basis/types/double) type numbers is disabled by default. This can lead to compilation error 5105. To enable support for the double type numbers, please add the following directive to your OpenCL program: [#pragma OPENCL EXTENSION cl\_khr\_fp64 : enable](https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/cl_khr_fp64.html). However if a graphic card doesn't support double, enabling this directive won't be of help.

It is recommended to write the source code for OpenCL in separate CL files, which can later be included in the MQL5 program using the [resource variables](/en/docs/runtime/resources#resourcevariables).

### Handling errors in OpenCL programs

To get information about the last error in an OpenCL program, use the [CLGetInfoInteger](/en/docs/opencl/clgetinfointeger) and [CLGetInfoString](/en/docs/opencl/clgetinfostring)functions that allow getting the error code and text description.

OpenCL last error code: To get the latest OpenCL error, call [CLGetInfoInteger](/en/docs/opencl/clgetinfointeger), while thehandleparameter is ignored (can be set to zero). Description of errors: <https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#CL_SUCCESS>.

For an unknown error code, the"unknown OpenCL error N" string is returned where N is an error code. Example:

| |
| --- |
| //--- the first 'handle' parameter is ignored when getting the last error code intcode= (int)CLGetInfoInteger(0,CL\_LAST\_ERROR); |

Text description of the OpenCL error: To get the latest OpenCL error, call [CLGetInfoString](/en/docs/opencl/clgetinfostring). The error code should be passed as thehandleparameter.

Description of errors: <https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#CL_SUCCESS>. If CL\_LAST\_ERROR is passed instead of the error code, then the function returns the description of the last error. For example:

| |
| --- |
| //--- get the code of the last OpenCL error int   code= (int)CLGetInfoInteger(0,CL\_LAST\_ERROR); stringdesc;// to get an error text description   //--- use the error code to get an error text description if(!CLGetInfoString(code,CL\_ERROR\_DESCRIPTION,desc))  desc="cannot get OpenCL error description,"+ (string)GetLastError(); Print(desc);   //--- in order to get the description of the last OpenCL error without getting the code first, pass CL\_LAST\_ERROR   if(!CLGetInfoString(CL\_LAST\_ERROR,CL\_ERROR\_DESCRIPTION,desc))  desc="cannot get OpenCL error description,"+ (string)GetLastError(); Print(desc);; |

So far, the name of the internal enumeration is given as an error description. You can find its decoding here: <https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#CL_SUCCESS>. For example, the CL\_INVALID\_KERNEL\_ARGS value means "Returned when enqueuing a kernel when some kernel arguments have not been set or are invalid."

Functions for running programs in OpenCL:

| Function | Action |
| --- | --- |
| [CLHandleType](/en/docs/opencl/clhandletype) | Returns the type of an OpenCL handle as a value of the ENUM\_OPENCL\_HANDLE\_TYPE enumeration |
| [CLGetInfoInteger](/en/docs/opencl/clgetinfointeger) | Returns the value of an integer property for an OpenCL object or device |
| [CLContextCreate](/en/docs/opencl/clcontextcreate) | Creates an OpenCL context |
| [CLContextFree](/en/docs/opencl/clcontextfree) | Removes an OpenCL context |
| [CLGetDeviceInfo](/en/docs/opencl/clgetdeviceinfo) | Receives device property from OpenCL driver |
| [CLProgramCreate](/en/docs/opencl/clprogramcreate) | Creates an OpenCL program from a source code |
| [CLProgramFree](/en/docs/opencl/clprogramfree) | Removes an OpenCL program |
| [CLKernelCreate](/en/docs/opencl/clkernelcreate) | Creates an OpenCL start function |
| [CLKernelFree](/en/docs/opencl/clkernelfree) | Removes an OpenCL start function |
| [CLSetKernelArg](/en/docs/opencl/clsetkernelarg) | Sets a parameter for the OpenCL function |
| [CLSetKernelArgMem](/en/docs/opencl/clsetkernelargmem) | Sets an OpenCL buffer as a parameter of the OpenCL function |
| [CLSetKernelArgMemLocal](/en/docs/opencl/clsetkernelargmemlocal) | Sets the local buffer as an argument of the kernel function |
| [CLBufferCreate](/en/docs/opencl/clbuffercreate) | Creates an OpenCL buffer |
| [CLBufferFree](/en/docs/opencl/clbufferfree) | Deletes an OpenCL buffer |
| [CLBufferWrite](/en/docs/opencl/clbufferwrite) | Writes an array into an OpenCL buffer |
| [CLBufferRead](/en/docs/opencl/clbufferread) | Reads an OpenCL buffer into an array |
| [CLExecute](/en/docs/opencl/clexecute) | Runs an OpenCL program |
| [CLExecutionStatus](/en/docs/opencl/clexecutionstatus) | Returns the OpenCL program execution status |

See also

[OpenCL](/en/docs/standardlibrary/copencl), [Resources](/en/docs/runtime/resources#resourcevariables)

[EventChartCustom](/en/docs/eventfunctions/eventchartcustom "EventChartCustom")

[CLHandleType](/en/docs/opencl/clhandletype "CLHandleType")