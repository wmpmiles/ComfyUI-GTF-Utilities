import torch
from gtf_impl import transform as T


class CropUncropRelative:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
                "anchor": (cls.ANCHORS, {}),
                "mode": (cls.MODES, {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    ANCHORS = ("top-left", "top", "top-right", "left", "middle", "right",
               "bottom-left", "bottom", "bottom-right")
    SINGLE_ANCHORS = ("left", "middle", "right")
    MODES = ("zero", "reflect")

    @classmethod
    def f(
        cls,
        gtf: torch.Tensor,
        dimensions: tuple[int, int],
        anchor: str,
        mode: str,
    ) -> tuple[torch.Tensor]:
        width, height = dimensions
        index = cls.ANCHORS.index(anchor)
        width_anchor = cls.SINGLE_ANCHORS[index % 3]
        height_anchor = cls.SINGLE_ANCHORS[index // 3]
        cuc_width = T.crop_uncrop(gtf, 3, width, width_anchor, mode)
        cuc_height = T.crop_uncrop(cuc_width, 2, height, height_anchor, mode)
        return (cuc_height, )


class CropToBBOX:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor,
        bbox: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[list[torch.Tensor]]:
        wh, lrud = bbox
        w, h = (int(x) for x in wh)
        if gtf.shape[2] != h or gtf.shape[3] != w:
            raise ValueError("GTF dimensions do not match those expected by \
                the bounding box.")
        if gtf.shape[0] != lrud.shape[0]:
            raise ValueError("bbox and tensor batch size must match")
        cropped = []
        unbatched = torch.split(gtf, 1)
        lruds = (x.squeeze() for x in lrud.split(1))
        for single_lrud, single_tensor in zip(lruds, unbatched):
            l, r, u, d = (int(x) for x in single_lrud)
            cropped += [single_tensor[:, :, u:d, l:r]]
        return (cropped, )


class UncropFromBBOX:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: list[torch.Tensor],
        bbox: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[list[torch.Tensor]]:
        wh, lrud = bbox[0]
        if lrud.shape[0] != len(gtf):
            raise ValueError("GTF and bbox batch size must match.")
        unbatched = (x.squeeze() for x in lrud.split(1))
        uncropped_list = []
        for single_gtf, single_lrud in zip(gtf, unbatched):
            uncropped = T.uncrop_bbox(single_gtf, single_lrud, wh)
            uncropped_list += [uncropped]
        return (uncropped_list, )


class Batch:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_1": ("GTF", {}),
                "gtf_2": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_1: torch.Tensor, gtf_2: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf_1.shape[1:] != gtf_2.shape[1:]:
            raise ValueError("GTFs must have the same dimensions in all but \
                batch count to be batched together.")
        batched = torch.cat((gtf_1, gtf_2))
        return (batched, )


class ConnectedComponents:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[list[torch.Tensor]]:
        (coloring, max_unique) = T.component_coloring(gtf)
        if max_unique == 0:
            return ([coloring], )
        colorings = []
        for i in range(1, max_unique + 1):
            colorings += [(coloring == i).to(gtf.dtype)]
        return (colorings, )


class Channels1To3Repeat:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf.shape[1] != 1:
            raise ValueError("Can only convert single channel GTFs.")
        tensor = gtf.repeat(1, 3, 1, 1)
        return (tensor, )


class Channels1To4Repeat:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf.shape[1] != 1:
            raise ValueError("Can only convert single channel GTFs.")
        tensor = gtf.repeat(1, 4, 1, 1)
        return (tensor, )


class Transpose:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        transposed = gtf.permute(0, 1, 3, 2)
        return (transposed, )


class FlipHorizontal:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        flipped = gtf.flip((3, ))
        return (flipped, )


class FlipVertical:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        flipped = gtf.flip((2, ))
        return (flipped, )


class RotateCW:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        rotated = gtf.rot90(k=1, dims=(2, 3))
        return (rotated, )


class RotateCCW:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        rotated = gtf.rot90(k=-1, dims=(2, 3))
        return (rotated, )


class Rotate180:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        rotated = gtf.rot90(k=2, dims=(2, 3))
        return (rotated, )
