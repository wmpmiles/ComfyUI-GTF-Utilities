from .nodes_conversion import *
from .nodes_bbox import *
from .nodes_resample import *
from .nodes_utils import *
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
    "GTF | Dimensions": Dimensions,
    "GTF | Crop with Anchor": CropRelative,
    "GTF | Pad to Multiple of 8": PadToMultOf8,
    "GTF | Zero": Zero,
    "GTF | One": One,
    "GTF | Float": Float,
    "GTF | Add": Add,
    "GTF | Multiply": Multiply,
    "GTF | Reciprocal": Reciprocal,
    "GTF | Negative": Negate,
    "BBOX | From Mask": MaskToBoundingBox,
    "BBOX | Pad": PadBoundingBox,
    "BBOX | Scale Area": BoundingBoxAreaScale,
    "BBOX | Align to 8": BoundingBoxToNearest8,
    "GTF | Crop to BBOX": CropToBoundingBox,
    "GTF | Uncrop from BBOX": UncropFromBoundingBox,
    "Scale Dimensions": ScaleDimensions,
    "Scale Dimensions to Megapixels": ScaleDimensionsToMegapixels,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
