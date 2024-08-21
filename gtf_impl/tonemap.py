import torch


# All based on https://64.github.io/tonemapping/


def reinhard(tensor: torch.Tensor) -> torch.Tensor:
    tonemapped = tensor / (1 + tensor)
    return tonemapped


def reinhard_luminance(
    tensor: torch.Tensor,
    luminance: torch.Tensor
) -> torch.Tensor:
    tonemapped = tensor / (1 + luminance)
    return tonemapped


def reinhard_extended(
    tensor: torch.Tensor,
    whitepoint: torch.Tensor
) -> torch.Tensor:
    tonemapped = (tensor * (1 + (tensor / (whitepoint ** 2)))) / (1 + tensor)
    return tonemapped


def reinhard_extended_luminance(
    tensor: torch.Tensor,
    luminance: torch.Tensor,
    whitepoint: torch.Tensor
) -> torch.Tensor:
    tonemapped = \
        (tensor * (1 + (luminance / (whitepoint ** 2)))) / (1 + luminance)
    return tonemapped


def reinhard_jodie(
    tensor: torch.Tensor,
    luminance: torch.Tensor
) -> torch.Tensor:
    t_reinhard = reinhard(tensor)
    t_reinhard_luminance = reinhard_luminance(tensor, luminance)
    lerped = torch.lerp(t_reinhard_luminance, t_reinhard, t_reinhard)
    return lerped


def reinhard_jodie_extended(
    tensor: torch.Tensor,
    luminance: torch.Tensor,
    whitepoint: torch.Tensor
) -> torch.Tensor:
    t_reinhard = reinhard_extended(tensor, whitepoint)
    t_reinhard_luminance = \
        reinhard_extended_luminance(tensor, luminance, whitepoint)
    lerped = torch.lerp(t_reinhard_luminance, t_reinhard, t_reinhard)
    return lerped


def uncharted_2(tensor: torch.Tensor) -> torch.Tensor:
    def tonemap_partial(t: torch.Tensor | float) -> torch.Tensor | float:
        A, B, C, D, E, F = 0.15, 0.5, 0.1, 0.2, 0.02, 0.3
        partial = ((t*(A*t+C*B)+D*E)/(t*(A*t+B)+D*F))-E/F
        return partial
    EXPOSURE_BIAS = 2.0
    W = 11.2
    partial = tonemap_partial(tensor * EXPOSURE_BIAS)
    white_scale = 1 / tonemap_partial(W)
    tonemapped = partial * white_scale
    return tonemapped


def aces(tensor: torch.Tensor) -> torch.Tensor:
    ACES_INPUT = torch.tensor((
        (0.59719, 0.35458, 0.04823),
        (0.07600, 0.90834, 0.01566),
        (0.02840, 0.13383, 0.83777),
    )).reshape(1, 3, 3, 1, 1)
    ACES_OUTPUT = torch.tensor((
        (+1.60475, -0.53108, -0.07367),
        (-0.10208, +1.10813, -0.00605),
        (-0.00327, -0.07276, +1.07602),
    )).reshape(1, 3, 3, 1, 1)
    input = (tensor.unsqueeze(2) * ACES_INPUT).sum(2)
    rtt_odt_fit = (input * (input + 0.0245786) - 0.000090537) \
        / (input * (0.983729 * input + 0.4329510) + 0.238081)
    output = (rtt_odt_fit.unsqueeze(2) * ACES_OUTPUT).sum(2)
    return output
