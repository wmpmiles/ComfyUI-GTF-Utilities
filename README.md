# ComfyUI Resampling

## What is this?

This is a set of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom 
nodes that provide image and latent resampling using a number of traditional 
resampling algorithms, implemented purely with PyTorch.

Supported algorithms are:
- Nearest Neighbor
- Triangle (Bilinear)
- Mitchell-Netravali (Bicubic)
- Lanczos
- Area

Each algorithm provides full access to their filter parameters, and for those
algorithms for which it makes sense both seperable and non-seperable 
implementations are available.

Most of the implementation was derived from information available at:
- https://entropymine.com/imageworsener/
- https://mazzo.li/posts/lanczos.html
- https://en.wikipedia.org/wiki/Mitchell%E2%80%93Netravali_filters
- https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3

## Warning

These nodes only implement the direct resampling functions and in no way handle
colorspace transformation or premultiplication. To do correct image resampling 
the to-be-resampled images must be in a linear colorspace, and if you're
resampling an image with an alpha channel then the image must be premultiplied.

If you plan to resample images that are in a non-linear colorspace (most images 
on the web and the outputs from all t2i models that I am aware of) then please 
find other nodes to do colorspace transformations into a linear colorspace
before resampling and back to the original colorspace after resampling.

See http://www.ericbrasseur.org/gamma.html for further details.

## Node Parameters

### In general

All nodes take:
- `images` or `latents` - a standard ComfyUI image/latent batch
- `width` - the width of the resampled images
- `height` - the height of the resampled images

### Triangle

The triangle filter takes a `radius` parameter which determines the 'radius` of 
the triangle used. A radius of 1 applied seperably will provide standard bilinear 
filtering, larger radii tend to be much blurier.

### Lanczos

Lanczos filtering takes a `radius` like the triangle filter, however this does
not change the shape of the existing filter but rather adds further 'support'
as you can think of the Lanczos filter as tapering off a sync function at a
distance specified by the radius.

### Mitchell-Netravali

Mitchell-Netravali filters are parameterised by two values: `b` and `c`, but 
have a fixed radius. The effects of changing these paramaters can be seen in 
the diagram below, and the recommended values are 0.33 and 0.33, which lies
along the satisfactory line.

![Mitchell-Netravali paramaters.](https://upload.wikimedia.org/wikipedia/commons/6/6f/Mitchell-Netravali_artifacts.svg)

## Node List

- Resample Image - Nearest Neighbor
- Resample Image - Area
- Resample Image - Triangle
- Resample Image - Mitchell-Netravali
- Resample Image - Lanczos
- Resample Latent - Nearest Neighbor
- Resample Latent - Area
- Resample Latent - Triangle
- Resample Latent - Mitchell-Netravali
- Resample Latent - Lanczos
