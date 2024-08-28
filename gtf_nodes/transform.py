import torch
from .. import types as T
from ..gtf_impl import transform as TF


# BASE CLASSES

class TransformBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", {})}}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"


# NODES

class BatchConcatenate(TransformBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_1": ("GTF", {}),
            "gtf_2": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf_1: torch.Tensor, gtf_2: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf_1.shape[1:] != gtf_2.shape[1:]:
            raise ValueError("GTFs must have the same dimensions in all but batch count to be concatenated.")
        concatenated = torch.cat((gtf_1, gtf_2))
        return (concatenated, )


class ChannelConcatenate(TransformBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_1": ("GTF", {}),
            "gtf_2": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf_1: torch.Tensor, gtf_2: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf_1.shape[0] != gtf_2.shape[0] or gtf_1.shape[2:] != gtf_2.shape[2:]:
            raise ValueError("GTFs must have the same dimensions in all but channel count to be concatenated.")
        concatenated = torch.cat((gtf_1, gtf_2), 1)
        return (concatenated, )


class Transpose(TransformBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        transposed = gtf.transpose(2, 3)
        return (transposed, )


class FlipHorizontal(TransformBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        flipped = gtf.flip((3, ))
        return (flipped, )


class FlipVertical(TransformBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        flipped = gtf.flip((2, ))
        return (flipped, )


class RotateCW(TransformBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        rotated = gtf.rot90(k=1, dims=(2, 3))
        return (rotated, )


class RotateCCW(TransformBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        rotated = gtf.rot90(k=-1, dims=(2, 3))
        return (rotated, )


class Rotate180(TransformBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        rotated = gtf.rot90(k=2, dims=(2, 3))
        return (rotated, )


class CropUncropRelative(TransformBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gtf": ("GTF", {}),
            "dimensions": ("DIM", {}),
            "anchor": (cls.ANCHORS, {}),
            "mode": (cls.MODES, {}),
        }}

    ANCHORS = ("top-left", "top", "top-right", "left", "middle", "right", "bottom-left", "bottom", "bottom-right")
    SINGLE_ANCHORS = ("left", "middle", "right")
    MODES = ("zero", "reflect")

    @classmethod
    def f(cls, gtf: torch.Tensor, dimensions: tuple[int, int], anchor: str, mode: str) -> tuple[torch.Tensor]:
        width, height = dimensions
        index = cls.ANCHORS.index(anchor)
        width_anchor = cls.SINGLE_ANCHORS[index % 3]
        height_anchor = cls.SINGLE_ANCHORS[index // 3]
        cuc_width = TF.crop_uncrop(gtf, 3, width, width_anchor, mode)
        cuc_height = TF.crop_uncrop(cuc_width, 2, height, height_anchor, mode)
        return (cuc_height, )


class CropToBBOX(TransformBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    OUTPUT_IS_LIST = (True, )

    @staticmethod
    def f(gtf: torch.Tensor, bbox: T.BoundingBox) -> tuple[list[torch.Tensor]]:
        wh, lrud = bbox
        w, h = (int(x) for x in wh)
        if gtf.shape[2] != h or gtf.shape[3] != w:
            raise ValueError("GTF dimensions do not match those expected by the bounding box.")
        if gtf.shape[0] != lrud.shape[0]:
            raise ValueError("bbox and tensor batch size must match")
        cropped = []
        unbatched = torch.split(gtf, 1)
        lruds = (x.squeeze() for x in lrud.split(1))
        for single_lrud, single_tensor in zip(lruds, unbatched):
            l, r, u, d = (int(x) for x in single_lrud)
            cropped += [single_tensor[:, :, u:d, l:r]]
        return (cropped, )


class UncropFromBBOX(TransformBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )

    @staticmethod
    def f(gtf: list[torch.Tensor], bbox: list[T.BoundingBox] ) -> tuple[list[torch.Tensor]]:
        wh, lrud = bbox[0]
        if lrud.shape[0] != len(gtf):
            raise ValueError("GTF and bbox batch size must match.")
        unbatched = (x.squeeze() for x in lrud.split(1))
        uncropped_list = []
        for single_gtf, single_lrud in zip(gtf, unbatched):
            uncropped = T.uncrop_bbox(single_gtf, single_lrud, wh)
            uncropped_list += [uncropped]
        return (uncropped_list, )


class ConnectedComponents(TransformBase):
    OUTPUT_IS_LIST = (True, )
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[list[torch.Tensor]]:
        (coloring, max_unique) = T.component_coloring(gtf)
        if max_unique == 0:
            return ([coloring], )
        colorings = []
        for i in range(1, max_unique + 1):
            colorings += [(coloring == i).to(gtf.dtype)]
        return (colorings, )


NODE_CLASS_MAPPINGS = {
    "GTF | Transform - Batch Concatenate":       BatchConcatenate,
    "GTF | Transform - Channel Concatenate":     ChannelConcatenate,
    "GTF | Transform - Transpose":               Transpose,
    "GTF | Transform - Flip Vertical":           FlipVertical,
    "GTF | Transform - Flip Horizontal":         FlipHorizontal,
    "GTF | Transform - Rotate CW":               RotateCW,
    "GTF | Transform - Rotate CCW":              RotateCCW,
    "GTF | Transform - Rotate 180":              Rotate180,
    "GTF | Transform - Crop/Uncrop with Anchor": CropUncropRelative,
    "GTF | Transform - Crop to BBOX":            CropToBBOX,
    "GTF | Transform - Uncrop from BBOX":        UncropFromBBOX,
    "GTF | Transform - Connected Components":    ConnectedComponents,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
