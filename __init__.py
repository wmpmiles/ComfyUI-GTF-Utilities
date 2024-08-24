"""
@author: wmpmiles
@title: GTF Utilities
@nickname: GTFU
@description: TODO
"""

import sys
from pathlib import Path
package_path = Path(__file__).resolve().parent
sys.path += [str(package_path)]
from gtf_nodes import interface, bbox, resample, dimensions, colorspace, \
    filter, math, source, transform, tonemap, convert, primitive  # noqa: E402
sys.path = sys.path[:-1]


NODE_CLASS_MAPPINGS = {
    # Interface
    "GTF | From Images":    interface.FromImages,
    "GTF | From Masks":     interface.FromMasks,
    "GTF | From Latents":   interface.FromLatents,
    "GTF | To Images":      interface.ToImages,
    "GTF | To Masks":       interface.ToMasks,
    "GTF | Update Latents": interface.UpdateLatents,
    # Resample
    "GTF | Resample - Nearest Neighbor":   resample.NearestNeighbor,
    "GTF | Resample - Area":               resample.Area,
    "GTF | Resample - Triangle":           resample.Triangle,
    "GTF | Resample - Mitchell-Netravali": resample.MitchellNetravali,
    "GTF | Resample - Lanczos":            resample.Lanczos,
    # Filter
    "GTF | Filter - Convolve": filter.Convolve,
    "GTF | Filter - Morphological": filter.MorphologicalFilter,
    "GTF | Kernel - Gaussian": filter.KernelGaussian,
    # Colorspace
    "GTF | Colorspace - SRGB Linear to Gamma": colorspace.SRGBLinearToGamma,
    "GTF | Colorspace - SRGB Gamma to Linear": colorspace.SRGBGammaToLinear,
    "GTF | Colorspace - Linear to Log":        colorspace.LinearToLog,
    "GTF | Colorspace - Log to Linear":        colorspace.LogToLinear,
    "GTF | Colorspace - Standard Linear to Gamma":
    colorspace.StandardLinearToGamma,
    "GTF | Colorspace - Standard Gamma to Linear":
    colorspace.StandardGammaToLinear,
    # Transform
    "GTF | Transform - Crop/Uncrop with Anchor": transform.CropUncropRelative,
    "GTF | Transform - Batch":                   transform.Batch,
    "GTF | Transform - Crop to BBOX":            transform.CropToBBOX,
    "GTF | Transform - Uncrop from BBOX":        transform.UncropFromBBOX,
    "GTF | Transform - Connected Components":    transform.ConnectedComponents,
    "GTF | Transform - 1 Channel to 3":          transform.Channels1To3Repeat,
    "GTF | Transform - 1 Channel to 4":          transform.Channels1To4Repeat,
    # Math
    "GTF | Math - Add":        math.Add,
    "GTF | Math - Subtract":   math.Subtract,
    "GTF | Math - Multiply":   math.Multiply,
    "GTF | Math - Divide":     math.Divide,
    "GTF | Math - Reciprocal": math.Reciprocal,
    "GTF | Math - Negative":   math.Negate,
    "GTF | Math - Lerp":       math.Lerp,
    "GTF | Math - Pow":        math.Pow,
    "GTF | Math - Equal":      math.Equal,
    "GTF | Math - Less Than":  math.LessThan,
    # Source
    "GTF | Source - Zero":  source.Zero,
    "GTF | Source - One":   source.One,
    "GTF | Source - Value": source.Value,
    "GTF | Source - RGB":   source.RGB,
    # Tonemap
    "GTF | Tonemap - Reinhard":                tonemap.Reinhard,
    "GTF | Tonemap - Reinhard Extended":       tonemap.ReinhardExtended,
    "GTF | Tonemap - Reinhard over Luminance": tonemap.ReinhardLuminance,
    "GTF | Tonemap - Reinhard Extended over Luminance":
    tonemap.ReinhardExtendedLuminance,
    "GTF | Tonemap - Reinhard-Jodie":          tonemap.ReinhardJodie,
    "GTF | Tonemap - Reinhard-Jodie Extended": tonemap.ReinhardJodieExtended,
    "GTF | Tonemap - Uncharted 2":             tonemap.Uncharted2,
    "GTF | Tonemap - ACES":                    tonemap.ACES,
    # Convert
    "GTF | Convert - Luminance":           convert.Luminance,
    "GTF | Convert - Min":                 convert.Min,
    "GTF | Convert - Max":                 convert.Max,
    "GTF | Convert - Batch Min":           convert.BatchMin,
    "GTF | Convert - Batch Max":           convert.BatchMax,
    "GTF | Convert - Channel Min":         convert.ChannelMin,
    "GTF | Convert - Channel Max":         convert.ChannelMax,
    "GTF | Convert - Binary Threshold":    convert.BinaryThreshold,
    "GTF | Convert - Quantize Normalized": convert.QuantizeNormalized,
    # BBOX
    "BBOX | From Mask":  bbox.FromMask,
    "BBOX | Change":     bbox.Change,
    "BBOX | Scale Area": bbox.AreaScale,
    # Dimensions
    "Dimensions | Scale":               dimensions.Scale,
    "Dimensions | Change":              dimensions.Change,
    "Dimensions | Scale to Megapixels": dimensions.ScaleToMegapixels,
    "Dimensions | Align To":            dimensions.AlignTo,
    "Dimensions | From Raw":            dimensions.FromRaw,
    "Dimensions | To Raw":              dimensions.ToRaw,
    "Dimensions | From GTF":            dimensions.FromGTF,
    # Primitive
    "Primitive | Integer": primitive.Integer,
    "Primitive | Float":   primitive.Float,
    "Primitive | Boolean": primitive.Boolean,
    "Primitive | String":  primitive.String,
    "Primitive | Text":    primitive.Text,
}


__all__ = ["NODE_CLASS_MAPPINGS"]
