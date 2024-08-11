from .nodes_conversion import *
from .nodes_bbox import *
from .nodes_resample import *
from .nodes_dimensions import *
from .nodes_colorspace import *
from .nodes_filter import *
from .nodes_arithmetic import *
from .nodes_transform import *

NODE_CLASS_MAPPINGS = {
    "GTF | From Images": ImagesToGTF,
    "GTF | From Masks": MasksToGTF,
    "GTF | From Latents": LatentsToGTF,
    "GTF | To Images": GTFToImages,
    "GTF | To Masks": GTFToMasks,
    "GTF | To Latents": GTFToLatents,
    "GTF | Resample - Nearest Neighbor": ResampleNearestNeighbor,
    "GTF | Resample - Area": ResampleArea,
    "GTF | Resample - Triangle": ResampleTriangle,
    "GTF | Resample - Mitchell-Netravali": ResampleMitchellNetravali,
    "GTF | Resample - Lanczos": ResampleLanczos,
    "GTF | Filter - Gaussian Blur": BlurGaussian,
    "GTF | Filter - Morphological": MorphologicalFilter,
    "GTF | Colorspace - SRGB Linear to Gamma": ColorspaceSRGBLinearToGamma,
    "GTF | Colorspace - SRGB Gamma to Linear": ColorspaceSRGBGammaToLinear,
    "GTF | Crop/Uncrop with Anchor": CropUncropRelative,
    "GTF | Zero": Zero,
    "GTF | One": One,
    "GTF | Float": Float,
    "GTF | Add": Add,
    "GTF | Multiply": Multiply,
    "GTF | Reciprocal": Reciprocal,
    "GTF | Negative": Negate,
    "BBOX | From Mask": MaskToBoundingBox,
    "BBOX | Change": ChangeBoundingBox,
    "BBOX | Scale Area": BoundingBoxAreaScale,
    "GTF | Crop to BBOX": CropToBoundingBox,
    "GTF | Uncrop from BBOX": UncropFromBoundingBox,
    "Dimensions | Scale": ScaleDimensions,
    "Dimensions | Change": ChangeDimensions,
    "Dimensions | Scale to Megapixels": ScaleDimensionsToMegapixels,
    "Dimensions | Align To": DimensionsAlignTo,
    "Dimensions | From Raw": DimensionsFromRaw,
    "Dimensions | To Raw": DimensionsToRaw,
    "Dimensions | From GTF": GTFDimensions,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
