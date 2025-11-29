---
title: "ONNX Models in Machine Learning"
url: "https://www.mql5.com/en/docs/onnx"
hierarchy: []
scraped_at: "2025-11-28 09:31:04"
---

# ONNX Models in Machine Learning

[MQL5 Reference](/en/docs "MQL5 Reference")ONNX models

* [ONNX Support](/en/docs/onnx/onnx_intro "ONNX Support")
* [Format Conversion](/en/docs/onnx/onnx_conversion "Format Conversion")
* [Automatic data type conversion](/en/docs/onnx/onnx_types_autoconversion "Automatic data type conversion")
* [Creating a Model](/en/docs/onnx/onnx_prepare "Creating a Model")
* [Running a model](/en/docs/onnx/onnx_mql5 "Running a model")
* [Validation in the Strategy Tester](/en/docs/onnx/onnx_test "Validation in the Strategy Tester")
* [OnnxCreate](/en/docs/onnx/onnxcreate "OnnxCreate")
* [OnnxCreateFromBuffer](/en/docs/onnx/onnxcreatefrombuffer "OnnxCreateFromBuffer")
* [OnnxRelease](/en/docs/onnx/onnxrelease "OnnxRelease")
* [OnnxRun](/en/docs/onnx/onnxrun "OnnxRun")
* [OnnxGetInputCount](/en/docs/onnx/onnxgetinputcount "OnnxGetInputCount")
* [OnnxGetOutputCount](/en/docs/onnx/onnxgetoutputcount "OnnxGetOutputCount")
* [OnnxGetInputName](/en/docs/onnx/onnxgetinputname "OnnxGetInputName")
* [OnnxGetOutputName](/en/docs/onnx/onnxgetoutputname "OnnxGetOutputName")
* [OnnxGetInputTypeInfo](/en/docs/onnx/onnxgetinputtypeinfo "OnnxGetInputTypeInfo")
* [OnnxGetOutputTypeInfo](/en/docs/onnx/onnxgetoutputtypeinfo "OnnxGetOutputTypeInfo")
* [OnnxSetInputShape](/en/docs/onnx/onnxsetinputshape "OnnxSetInputShape")
* [OnnxSetOutputShape](/en/docs/onnx/onnxsetoutputshape "OnnxSetOutputShape")
* [Data structures](/en/docs/onnx/onnx_structures "Data structures")

# ONNX Models in Machine Learning

[ONNX](https://onnx.ai/) (Open Neural Network Exchange) is an open-source format for machine learning models. This project has several major advantages:

* [ONNX](/en/docs/onnx/onnx_intro) is supported by large companies such as Microsoft, Facebook, Amazon and other partners.
* Its open format enables [format conversions](/en/docs/onnx/onnx_conversion) between different machine learning toolkits, while Microsoft's [ONNXMLTools](https://learn.microsoft.com/ru-ru/windows/ai/windows-ml/onnxmltools) allows converting models to the ONNX format.
* MQL5 provides [automatic data type conversion](/en/docs/onnx/onnx_types_autoconversion) for model inputs and outputs if the passed parameter type does not match the model.
* [ONNX models](/en/docs/onnx/onnx_prepare) can be created using various machine learning tools. They are currently supported in Caffe2, Microsoft Cognitive Toolkit, MXNet, PyTorch and OpenCV. Interfaces for other popular frameworks and libraries are also available.
* With the MQL5 language, you can implement an [ONNX model in a trading strategy](/en/docs/onnx/onnx_mql5) and use it along with all the advantages of the MetaTrader 5 platform for efficient operations in the financial markets.
* Before tunning a model for live trading, you can [test the model behavior on historical data](/en/docs/onnx/onnx_test) in the Strategy Tester, without using third-party tools.

MQL5 provides the following functions for working with ONNX:

| Function | Action |
| --- | --- |
| [OnnxCreate](/en/docs/onnx/onnxcreate) | Create an ONNX session, loading a model from an \*.onnx file |
| [OnnxCreateFromBuffer](/en/docs/onnx/onnxcreatefrombuffer) | Create an ONNX session, loading a model from a data array |
| [OnnxRelease](/en/docs/onnx/onnxrelease) | Close an ONNX session |
| [OnnxRun](/en/docs/onnx/onnxrun) | Run an ONNX model |
| [OnnxGetInputCount](/en/docs/onnx/onnxgetinputcount) | Get the number of inputs in an ONNX model |
| [OnnxGetOutputCount](/en/docs/onnx/onnxgetoutputcount) | Get the number of outputs in an ONNX model |
| [OnnxGetInputName](/en/docs/onnx/onnxgetinputname) | Get the name of a model's input by index |
| [OnnxGetOutputName](/en/docs/onnx/onnxgetoutputname) | Get the name of a model's output by index |
| [OnnxGetInputTypeInfo](/en/docs/onnx/onnxgetinputtypeinfo) | Get the description of the input type from the model |
| [OnnxGetOutputTypeInfo](/en/docs/onnx/onnxgetoutputtypeinfo) | Get the description of the output type from the model |
| [OnnxSetInputShape](/en/docs/onnx/onnxsetinputshape) | Set the shape of a model's input data by index |
| [OnnxSetOutputShape](/en/docs/onnx/onnxsetoutputshape) | Set the shape of a model's output data by index |

[history\_deals\_get](/en/docs/python_metatrader5/mt5historydealsget_py "history_deals_get")

[ONNX Support](/en/docs/onnx/onnx_intro "ONNX Support")