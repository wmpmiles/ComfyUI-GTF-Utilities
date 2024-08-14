"""
@author: wmpmiles
@title: GTF Utilities
@nickname: GTFU
@description: TODO
"""

from .nodes.interface import *
from .nodes.bbox import *
from .nodes.resample import *
from .nodes.dimensions import *
from .nodes.colorspace import *
from .nodes.filter import *
from .nodes.arithmetic import *
from .nodes.transform import *
from .nodes.tonemap import *
from .nodes.grading import *
from .nodes.convert import *

NODE_CLASS_MAPPINGS = {
    # Interface
    "GTF | From Images": ImagesToGTF,
    "GTF | From Masks": MasksToGTF,
    "GTF | From Latents": LatentsToGTF,
    "GTF | To Images": GTFToImages,
    "GTF | To Masks": GTFToMasks,
    "GTF | To New Latents": GTFToNewLatents,
    "GTF | Update Latents": GTFUpdateLatents,
    # Resample
    "GTF | Resample - Nearest Neighbor": ResampleNearestNeighbor,
    "GTF | Resample - Area": ResampleArea,
    "GTF | Resample - Triangle": ResampleTriangle,
    "GTF | Resample - Mitchell-Netravali": ResampleMitchellNetravali,
    "GTF | Resample - Lanczos": ResampleLanczos,
    # Filter
    "GTF | Filter - Gaussian Blur": BlurGaussian,
    "GTF | Filter - Morphological": MorphologicalFilter,
    # Colorspace
    "GTF | Colorspace - SRGB Linear to Gamma": ColorspaceSRGBLinearToGamma,
    "GTF | Colorspace - SRGB Gamma to Linear": ColorspaceSRGBGammaToLinear,
    # Transform
    "GTF | Transform - Crop/Uncrop with Anchor": CropUncropRelative,
    "GTF | Transform - Batch": BatchGTF,
    "GTF | Transform - Crop to BBOX": CropToBoundingBox,
    "GTF | Transform - Uncrop from BBOX": UncropFromBoundingBox,
    # Arithmetic
    "GTF | Arithmetic - Zero": Zero,
    "GTF | Arithmetic - One": One,
    "GTF | Arithmetic - Float": Float,
    "GTF | Arithmetic - Add": Add,
    "GTF | Arithmetic - Multiply": Multiply,
    "GTF | Arithmetic - Reciprocal": Reciprocal,
    "GTF | Arithmetic - Negative": Negate,
    # Tonemap
    "GTF | Tonemap - Reinhard": TonemapReinhard,
    "GTF | Tonemap - Reinhard Extended": TonemapReinhardExtended,
    "GTF | Tonemap - Reinhard over Luminance": TonemapReinhardLuminance,
    "GTF | Tonemap - Reinhard Extended over Luminance": TonemapReinhardExtendedLuminance,
    "GTF | Tonemap - Reinhard-Jodie": TonemapReinhardJodie,
    "GTF | Tonemap - Reinhard-Jodie Extended": TonemapReinhardJodieExtended,
    # Convert
    "GTF | Convert - Luminance": Luminance,
    "GTF | Convert - Min": Min,
    "GTF | Convert - Max": Max,
    "GTF | Convert - Batch Min": BatchMin,
    "GTF | Convert - Batch Max": BatchMax,
    "GTF | Convert - Channel Min": ChannelMin,
    "GTF | Convert - Channel Max": ChannelMax,
    # BBOX
    "BBOX | From Mask": MaskToBoundingBox,
    "BBOX | Change": ChangeBoundingBox,
    "BBOX | Scale Area": BoundingBoxAreaScale,
    # Dimensions
    "Dimensions | Scale": ScaleDimensions,
    "Dimensions | Change": ChangeDimensions,
    "Dimensions | Scale to Megapixels": ScaleDimensionsToMegapixels,
    "Dimensions | Align To": DimensionsAlignTo,
    "Dimensions | From Raw": DimensionsFromRaw,
    "Dimensions | To Raw": DimensionsToRaw,
    "Dimensions | From GTF": GTFDimensions,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
