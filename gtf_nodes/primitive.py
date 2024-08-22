class Integer:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": -1_000_000_000,
                    "max": 1_000_000_000,
                })
            }
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("int", )
    CATEGORY = "gtf/primitive"
    FUNCTION = "f"

    @staticmethod
    def f(value: int) -> tuple[int]:
        if not isinstance(value, int):
            raise ValueError("Value must be an integer.")
        return (value, )


class Float:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "step": 0.0001,
                    "min": -1_000_000_000,
                    "max": 1_000_000_000,
                })
            }
        }

    RETURN_TYPES = ("FLOAT", )
    RETURN_NAMES = ("float", )
    CATEGORY = "gtf/primitive"
    FUNCTION = "f"

    @staticmethod
    def f(value: float) -> tuple[float]:
        if not isinstance(value, float):
            raise ValueError("Value must be a float.")
        return (value, )


class Boolean:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "value": ("BOOLEAN", {})
            }
        }

    RETURN_TYPES = ("BOOLEAN", )
    RETURN_NAMES = ("boolean", )
    CATEGORY = "gtf/primitive"
    FUNCTION = "f"

    @staticmethod
    def f(value: bool) -> tuple[bool]:
        if not isinstance(value, bool):
            raise ValueError("Value must be a boolean.")
        return (value, )


class String:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "value": ("STRING", {})
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("string", )
    CATEGORY = "gtf/primitive"
    FUNCTION = "f"

    @staticmethod
    def f(value: str) -> tuple[str]:
        if not isinstance(value, str):
            raise ValueError("Value must be a string.")
        return (value, )


class Text:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "value": ("STRING", {"multiline": True})
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("string", )
    CATEGORY = "gtf/primitive"
    FUNCTION = "f"

    @staticmethod
    def f(value: str) -> tuple[str]:
        if not isinstance(value, str):
            raise ValueError("Value must be a string.")
        return (value, )
