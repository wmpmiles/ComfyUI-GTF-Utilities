import torch


class Zero:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {}
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/source"
    FUNCTION = "f"

    @staticmethod
    def f() -> tuple[torch.Tensor]:
        zero = torch.zeros(1, 1, 1, 1)
        return (zero, )


class One:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {}
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/source"
    FUNCTION = "f"

    @staticmethod
    def f() -> tuple[torch.Tensor]:
        one = torch.ones(1, 1, 1, 1)
        return (one, )


class Value:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.0001,
                    "min": -1_000_000,
                    "max": 1_000_000,
                })
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/source"
    FUNCTION = "f"

    @staticmethod
    def f(value: float) -> tuple[torch.Tensor]:
        value_tensor = torch.tensor(value).reshape(1, 1, 1, 1)
        return (value_tensor, )


class RGB:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "b": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/source"
    FUNCTION = "f"

    @staticmethod
    def f(r: int, g: int, b: int) -> tuple[torch.Tensor]:
        rgb = torch.tensor((r, g, b)).reshape(1, 3, 1, 1).to(torch.float) / 255
        return (rgb, )
