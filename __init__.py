"""
@author: wmpmiles
@title: GTF Utilities
@nickname: GTFU
@description: TODO
"""

from .gtf_nodes import *


NODE_CLASS_MAPPINGS = {
    **bbox.NODE_CLASS_MAPPINGS,
    **colorspace.NODE_CLASS_MAPPINGS,
    **convert.NODE_CLASS_MAPPINGS,
    **dimensions.NODE_CLASS_MAPPINGS,
    **filters.NODE_CLASS_MAPPINGS,
    **grading.NODE_CLASS_MAPPINGS,
    **interface.NODE_CLASS_MAPPINGS,
    **logic.NODE_CLASS_MAPPINGS,
    **math.NODE_CLASS_MAPPINGS,
    **primitive.NODE_CLASS_MAPPINGS,
    **resample.NODE_CLASS_MAPPINGS,
    **source.NODE_CLASS_MAPPINGS,
    **special.NODE_CLASS_MAPPINGS,
    **tonemap.NODE_CLASS_MAPPINGS,
    **transform.NODE_CLASS_MAPPINGS,
    **trigonometry.NODE_CLASS_MAPPINGS,
    **window.NODE_CLASS_MAPPINGS,
}


__all__ = ["NODE_CLASS_MAPPINGS"]
