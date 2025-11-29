---
title: "Working with Optimization Results"
url: "https://www.mql5.com/en/docs/optimization_frames"
hierarchy: []
scraped_at: "2025-11-28 09:31:30"
---

# Working with Optimization Results

[MQL5 Reference](/en/docs "MQL5 Reference")Working with Optimization Results

* [FrameFirst](/en/docs/optimization_frames/framefirst "FrameFirst")
* [FrameFilter](/en/docs/optimization_frames/framefilter "FrameFilter")
* [FrameNext](/en/docs/optimization_frames/framenext "FrameNext")
* [FrameInputs](/en/docs/optimization_frames/frameinputs "FrameInputs")
* [FrameAdd](/en/docs/optimization_frames/frameadd "FrameAdd")
* [ParameterGetRange](/en/docs/optimization_frames/parametergetrange "ParameterGetRange")
* [ParameterSetRange](/en/docs/optimization_frames/parametersetrange "ParameterSetRange")

# Working with Optimization Results

Functions for organizing custom processing of the optimization results in the strategy tester. They can be called during optimization in testing agents, as well as locally in Expert Advisors and scripts.

When you run an Expert Advisor in the strategy tester, you can create your own data array based on the simple types or [simple structures](/en/docs/basis/types/classes#simple_structure) (they do not contain strings, class objects or objects of dynamic arrays). This data set can be saved using the [FrameAdd()](/en/docs/optimization_frames/frameadd) function in a special structure called a frame. During the optimization of an Expert Advisor, each agent can send a series of frames to the terminal. All the received frames are written in the \*.MQD file named as the Expert Advisor in the terminal\_directory\MQL5\Files\Tester folder. They are written in the order they are received from the agents. Receipt of a frame in the client terminal from a testing agent generates the [TesterPass](/en/docs/runtime/event_fire#testerpass) event.

Frames can be stored in the computer memory and in a file with the specified name. The MQL5 language sets no limitations on the number of frames.

### Memory and disk space limits in MQL5 Cloud Network

The following limitation applies to optimizations run in the [MQL5 Cloud Network](https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization#cloud_start): the Expert Advisor must not write to disk more than 4GB of information or use more than 4GB of RAM. If the limit is exceeded, the network agent will not be able to complete the calculation correctly, and you will not receive the result. However, you will be charged for all the time spent on the calculations.

If you need to get information from each optimization pass, [send frames](/en/docs/optimization_frames) without writing to disk. To avoid using [file operations](/en/docs/files) in Expert Advisors during calculations in the MQL5 Cloud Network, you can use the following check:

| |
| --- |
| int handle=INVALID\_HANDLE;    bool file\_operations\_allowed=true;    if(MQLInfoInteger(MQL\_OPTIMIZATION) || MQLInfoInteger(MQL\_FORWARD))       file\_operations\_allowed=false;      if(file\_operations\_allowed)      {       ...       handle=FileOpen(...);       ...      } |

 

| Function | Action |
| --- | --- |
| [FrameFirst](/en/docs/optimization_frames/framefirst) | Moves a pointer of frame reading to the beginning and resets the previously set filter |
| [FrameFilter](/en/docs/optimization_frames/framefilter) | Sets the frame reading filter and moves the pointer to the beginning |
| [FrameNext](/en/docs/optimization_frames/framenext) | Reads a frame and moves the pointer to the next one |
| [FrameInputs](/en/docs/optimization_frames/frameinputs) | Receives [input parameters](/en/docs/basis/variables/inputvariables), on which the frame is formed |
| [FrameAdd](/en/docs/optimization_frames/frameadd) | Adds a frame with data |
| [ParameterGetRange](/en/docs/optimization_frames/parametergetrange) | Receives data on the values range and the change step for an [input variable](/en/docs/basis/variables/inputvariables) when optimizing an Expert Advisor in the Strategy Tester |
| [ParameterSetRange](/en/docs/optimization_frames/parametersetrange) | Specifies the use of [input variable](/en/docs/basis/variables/inputvariables) when optimizing an Expert Advisor in the Strategy Tester: value, change step, initial and final values |

See also

[Testing Statistics](/en/docs/constants/environment_state/statistics), [Properties of a Running MQL5 Program](/en/docs/constants/environment_state/mql5_programm_info)

[iVolumes](/en/docs/indicators/ivolumes "iVolumes")

[FrameFirst](/en/docs/optimization_frames/framefirst "FrameFirst")