---
title: "Working with DirectX"
url: "https://www.mql5.com/en/docs/directx"
hierarchy: []
scraped_at: "2025-11-28 09:31:02"
---

# Working with DirectX

[MQL5 Reference](/en/docs "MQL5 Reference")Working with DirectX

* [DXContextCreate](/en/docs/directx/dxcontextcreate "DXContextCreate")
* [DXContextSetSize](/en/docs/directx/dxcontextsetsize "DXContextSetSize")
* [DXContextGetSize](/en/docs/directx/dxcontextgetsize "DXContextGetSize")
* [DXContextClearColors](/en/docs/directx/dxcontextclearcolors "DXContextClearColors")
* [DXContextClearDepth](/en/docs/directx/dxcontextcleardepth "DXContextClearDepth")
* [DXContextGetColors](/en/docs/directx/dxcontextgetcolors "DXContextGetColors")
* [DXContextGetDepth](/en/docs/directx/dxcontextgetdepth "DXContextGetDepth")
* [DXBufferCreate](/en/docs/directx/dxbuffercreate "DXBufferCreate")
* [DXTextureCreate](/en/docs/directx/dxtexturecreate "DXTextureCreate")
* [DXInputCreate](/en/docs/directx/dxinputcreate "DXInputCreate")
* [DXInputSet](/en/docs/directx/dxinputset "DXInputSet")
* [DXShaderCreate](/en/docs/directx/dxshadercreate "DXShaderCreate")
* [DXShaderSetLayout](/en/docs/directx/dxshadersetlayout "DXShaderSetLayout")
* [DXShaderInputsSet](/en/docs/directx/dxshaderinputsset "DXShaderInputsSet")
* [DXShaderTexturesSet](/en/docs/directx/dxshadertexturesset "DXShaderTexturesSet")
* [DXDraw](/en/docs/directx/dxdraw "DXDraw")
* [DXDrawIndexed](/en/docs/directx/dxdrawindexed "DXDrawIndexed")
* [DXPrimiveTopologySet](/en/docs/directx/dxprimivetopologyset "DXPrimiveTopologySet")
* [DXBufferSet](/en/docs/directx/dxbufferset "DXBufferSet")
* [DXShaderSet](/en/docs/directx/dxshaderset "DXShaderSet")
* [DXHandleType](/en/docs/directx/dxhandletype "DXHandleType")
* [DXRelease](/en/docs/directx/dxrelease "DXRelease")

# Working with DirectX

DirectX 11 functions and shaders are designed for 3D visualization directly on a price chart.

Creating 3D graphics requires a graphic context ([DXContextCreate](/en/docs/directx/dxcontextcreate)) with the necessary image size. Besides, it is necessary to prepare vertex and index buffers ([DXBufferCreate](/en/docs/directx/dxbuffercreate)), as well as create vertex and pixel shaders ([DXShaderCreate](/en/docs/directx/dxshadercreate)). This is enough to display graphics in color.

The next level of graphics requires the inputs ([DXInputSet](/en/docs/directx/dxinputset)) for passing additional rendering parameters to shaders. This allows setting the camera and 3D object positions, describe light sources and implement mouse and keyboard control.

Thus, the built-in MQL5 functions enable you to create animated 3D charts directly in MetaTrader 5 with no need for third-party tools. A video card should support DX 11 and Shader Model 5.0 for the functions to work.

To start working with the library, simply read the article [How to create 3D graphics using DirectX in MetaTrader 5](https://www.mql5.com/en/articles/7708).

| Function | Action |
| --- | --- |
| [DXContextCreate](/en/docs/directx/dxcontextcreate) | Creates a graphic context for rendering frames of a specified size |
| [DXContextSetSize](/en/docs/directx/dxcontextsetsize) | Changes a frame size of a graphic context created in DXContextCreate() |
| [DXContextSetSize](/en/docs/directx/dxcontextsetsize) | Gets a frame size of a graphic context created in DXContextCreate() |
| [DXContextClearColors](/en/docs/directx/dxcontextclearcolors) | Sets a specified color to all pixels for the rendering buffer |
| [DXContextClearDepth](/en/docs/directx/dxcontextcleardepth) | Clears the depth buffer |
| [DXContextGetColors](/en/docs/directx/dxcontextgetcolors) | Gets an image of a specified size and offset from a graphic context |
| [DXContextGetDepth](/en/docs/directx/dxcontextgetdepth) | Gets the depth buffer of a rendered frame |
| [DXBufferCreate](/en/docs/directx/dxbuffercreate) | Creates a buffer of a specified type based on a data array |
| [DXTextureCreate](/en/docs/directx/dxtexturecreate) | Creates a 2D texture out of a rectangle of a specified size cut from a passed image |
| [DXInputCreate](/en/docs/directx/dxinputcreate) | Creates shader inputs |
| [DXInputSet](/en/docs/directx/dxinputset) | Sets shader inputs |
| [DXShaderCreate](/en/docs/directx/dxshadercreate) | Creates a shader of a specified type |
| [DXShaderSetLayout](/en/docs/directx/dxshadersetlayout) | Sets vertex layout for the vertex shader |
| [DXShaderInputsSet](/en/docs/directx/dxshaderinputsset) | Sets shader inputs |
| [DXShaderTexturesSet](/en/docs/directx/dxshadertexturesset) | Sets shader textures |
| [DXDraw](/en/docs/directx/dxdraw) | Renders the vertices of the vertex buffer set in DXBufferSet() |
| [DXDrawIndexed](/en/docs/directx/dxdrawindexed) | Renders graphic primitives described by the index buffer from DXBufferSet() |
| [DXPrimiveTopologySet](/en/docs/directx/dxprimivetopologyset) | Sets the type of primitives for rendering using DXDrawIndexed() |
| [DXBufferSet](/en/docs/directx/dxbufferset) | Sets a buffer for the current rendering |
| [DXShaderSet](/en/docs/directx/dxshaderset) | Sets a shader for rendering |
| [DXHandleType](/en/docs/directx/dxhandletype) | Returns a handle type |
| [DXRelease](/en/docs/directx/dxrelease) | Releases a handle |

[DatabaseColumnBlob](/en/docs/database/databasecolumnblob "DatabaseColumnBlob")

[DXContextCreate](/en/docs/directx/dxcontextcreate "DXContextCreate")