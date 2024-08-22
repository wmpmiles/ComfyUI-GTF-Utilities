# Generalised Tensor Format (GTF) Utilities

'Unified Tensor Format' makes more sense but UTF is overloaded.

## What is this?

This is a set of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom 
nodes that provide implementations or building blocks for implementations of a
variety of image processing algorithms and methods. 

These nodes operate on a generalised tensor format that can be converted to and 
from images, latents, and masks and allow for combining raw tensors from these 
sources. They are also implemented purely with PyTorch and have no external 
dependencies.

The general philosophy of this node set is to provide a selection of fundamental
operations that can be usefully composed together to create a wide variety of 
more complex operations.

## Node Categories

- __Interface__ - conversion between the `GTF` type and standard ComfyUI 
    `IMAGE`, `MASK`, and `LATENT` types
- __Filter__ - image filtering algorithms
- __Resample__ - image resampling (resizing) algorithms
- __Transform__ - procedures that change tensor dimensions, such as cropping and batching
- __Colorspace__ - colorspace conversion
- __Tonemap__ - tonemapping algorithms
- __Convert__ - TODO
- __Source__ - new GTFs from wholecloth
- __Math__ - elementwise mathematical operations
- __BBOX__ - bounding box creation and manipulation
- __Dimensions__ - dimension creation and manipulation
- __Primitive__ - primitive value providers

## TODO
 - Write documentation for all of the nodes and node categories
 - Reorganise the Transform, Convert, and Filter categories
 - Create a selection of reference group nodes and workflows
 - Tests

 ## Changelog

 ### v0.0.2 
 - Updated README with a reasonable placeholder.

 ### v0.0.1
 - Initial release.
