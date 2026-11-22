"""Shared constants, PBR thresholds, false-color palettes, and pure math helpers."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

# Internal naming prefixes and datablock identifiers
PREFIX = "__FLIGHTCHECK_"
CTRL_GROUP_NAME = "__FLIGHTCHECK_CTRL__"
OVERLAY_GROUP_NAME = "__FLIGHTCHECK_OVERLAY__"
CLAMP_GROUP_NAME = "__FLIGHTCHECK_CLAMP__"
CTRL_NODE_NAME = "__FLIGHTCHECK_CTRL_NODE__"
OVERLAY_NODE_NAME = "__FLIGHTCHECK_OVERLAY_NODE__"
CLAMP_NODE_NAME = "__FLIGHTCHECK_CLAMP_NODE__"
TEXEL_ATTR_NAME = "__flightcheck_td"

# Diagnostic visualization modes
MODE_OFF = 0
MODE_ALBEDO = 1
MODE_ROUGHNESS = 2
MODE_TEXEL_DENSITY = 3
MODE_COMBINED = 4

MODE_NAME_TO_INT = {
    "OFF": MODE_OFF,
    "ALBEDO": MODE_ALBEDO,
    "ROUGHNESS": MODE_ROUGHNESS,
    "TEXEL_DENSITY": MODE_TEXEL_DENSITY,
    "COMBINED": MODE_COMBINED,
}

MODE_INT_TO_NAME = {value: key for key, value in MODE_NAME_TO_INT.items()}

MODE_ENUM_ITEMS = [
    ("OFF", "Off", "Disable real-time shader diagnostic overlay"),
    ("ALBEDO", "Albedo", "Inspect dielectric/conductor luminance and semi-metal violations"),
    ("ROUGHNESS", "Roughness / Specular", "Inspect roughness extremes, specular IOR level, and IOR outliers"),
    ("TEXEL_DENSITY", "Texel Density", "Visualize face-corner texel density relative to target px/m"),
    ("COMBINED", "Combined", "Composite all active PBR and texel diagnostic overlays"),
]

# Linear Rec.709 luminance weights: Y = 0.2126*R + 0.7152*G + 0.0722*B
LUMA_WEIGHTS: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722)

# PBR validation thresholds
METALLIC_CONDUCTOR_THRESHOLD = 0.5
DIELECTRIC_MIN_LUMA = 0.02
DIELECTRIC_WARN_LUMA = 0.70
DIELECTRIC_CRIT_LUMA = 0.81
CONDUCTOR_MIN_LUMA = 0.45
SEMI_METAL_LOW = 0.05
SEMI_METAL_HIGH = 0.95

ROUGHNESS_MIN = 0.02
ROUGHNESS_MAX = 0.98
SPECULAR_MIN = 0.25
SPECULAR_MAX = 0.75
IOR_MIN = 1.30
IOR_MAX = 1.80
TRANSMISSION_THRESHOLD = 0.5

# Texel density thresholds and attribute encoding
TEXEL_RATIO_MIN = 0.5
TEXEL_RATIO_MAX = 2.0
TEXEL_DEFAULT_TARGET = 1024.0
TEXEL_DEFAULT_RES = 1024
TEXEL_ATTR_MIN_VALID = 0.05
TEXEL_ATTR_PRESENCE_EPS = 0.005

# Object scale and world dimension thresholds
SCALE_TOLERANCE = 0.001
DIM_MIN_METERS = 0.001
DIM_MAX_METERS = 500.0

# Soft-knee remediation parameters (C^1-smooth hyperbolic tangent compression)
CLAMP_DIELECTRIC_KNEE = 0.58
CLAMP_DIELECTRIC_MAX = 0.695
CLAMP_DIELECTRIC_FLOOR = 0.022
CLAMP_DIELECTRIC_FLOOR_KNEE = 0.045
CLAMP_CONDUCTOR_FLOOR = 0.46
CLAMP_CONDUCTOR_FLOOR_KNEE = 0.55

CLAMP_ROUGHNESS_FLOOR = 0.022
CLAMP_ROUGHNESS_FLOOR_KNEE = 0.05
CLAMP_ROUGHNESS_CEIL = 0.978
CLAMP_ROUGHNESS_CEIL_KNEE = 0.95

CLAMP_SPECULAR_MIN = 0.255
CLAMP_SPECULAR_MIN_KNEE = 0.30
CLAMP_SPECULAR_MAX = 0.745
CLAMP_SPECULAR_MAX_KNEE = 0.70

CLAMP_IOR_MIN = 1.305
CLAMP_IOR_MIN_KNEE = 1.35
CLAMP_IOR_MAX = 1.795
CLAMP_IOR_MAX_KNEE = 1.75

# Socket classification for color space analysis
DATA_MAP_SOCKETS = frozenset({
    "Metallic",
    "Roughness",
    "IOR",
    "Alpha",
    "Normal",
    "Bump",
    "Displacement",
    "Height",
    "Specular IOR Level",
    "Anisotropic",
    "Anisotropic Rotation",
    "Transmission Weight",
    "Coat Weight",
    "Coat Roughness",
    "Coat IOR",
    "Coat Normal",
    "Sheen Weight",
    "Sheen Roughness",
    "Subsurface Weight",
    "Subsurface Scale",
    "Subsurface IOR",
    "Subsurface Anisotropy",
    "Thin Film Thickness",
    "Thin Film IOR",
})

COLOR_MAP_SOCKETS = frozenset({
    "Base Color",
    "Emission Color",
    "Specular Tint",
    "Coat Tint",
    "Sheen Tint",
    "Subsurface Radius",
})

DATA_ONLY_NODE_TYPES = frozenset({
    "NORMAL_MAP",
    "BUMP",
    "DISPLACEMENT",
    "VECTOR_DISPLACEMENT",
})

# False-color diagnostic RGBA palettes (linear values for shader nodes / GPU)
COLOR_CRUSHED_BLACK = (0.05, 0.25, 1.0, 1.0)
COLOR_BLOWN_WARN = (1.0, 0.5, 0.0, 1.0)
COLOR_BLOWN_CRIT = (1.0, 0.1, 0.1, 1.0)
COLOR_DARK_METAL = (0.0, 0.85, 1.0, 1.0)
COLOR_SEMI_METAL = (1.0, 0.0, 0.85, 1.0)
COLOR_MIRROR_ROUGH = (0.65, 0.1, 0.95, 1.0)
COLOR_DEAD_ROUGH = (1.0, 0.45, 0.05, 1.0)
COLOR_HATCH_YELLOW = (1.0, 0.9, 0.0, 1.0)
COLOR_HATCH_RED = (1.0, 0.1, 0.1, 1.0)

COLOR_TEXEL_LOW = (0.05, 0.30, 1.0, 1.0)
COLOR_TEXEL_OK = (0.10, 0.85, 0.25, 1.0)
COLOR_TEXEL_HIGH = (1.0, 0.12, 0.10, 1.0)

# Aviation HUD colors (2D overlay)
HUD_COLOR_GO = (0.16, 0.86, 0.44, 1.0)
HUD_COLOR_CAUTION = (1.0, 0.65, 0.12, 1.0)
HUD_COLOR_NO_GO = (1.0, 0.22, 0.22, 1.0)
HUD_BG = (0.05, 0.07, 0.10, 0.84)
HUD_PANEL_BG = (0.09, 0.12, 0.16, 0.92)
HUD_BORDER = (0.28, 0.36, 0.46, 0.90)
HUD_TEXT_PRIMARY = (0.93, 0.96, 1.0, 1.0)
HUD_TEXT_MUTED = (0.62, 0.70, 0.80, 1.0)
HUD_BANNER_BG = (0.55, 0.12, 0.08, 0.94)

# 3D wireframe bounding cage colors
CAGE_COLOR_CRIT = (1.0, 0.15, 0.15, 0.95)
CAGE_COLOR_WARN = (1.0, 0.62, 0.10, 0.90)
CAGE_COLOR_DIM = (1.0, 0.92, 0.12, 0.90)


def linear_luminance(rgb: Iterable[float]) -> float:
    """Compute Linear Rec.709 luminance from an RGB(A) sequence."""
    it = iter(rgb)
    r = float(next(it, 0.0))
    g = float(next(it, 0.0))
    b = float(next(it, 0.0))
    return LUMA_WEIGHTS[0] * r + LUMA_WEIGHTS[1] * g + LUMA_WEIGHTS[2] * b


def srgb_to_linear(c: float) -> float:
    """Convert a single sRGB channel value to linear space."""
    if c <= 0.04045:
        return max(0.0, c) / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def soft_knee_upper(y: float, knee: float, ceiling: float) -> float:
    """Apply C^1-smooth hyperbolic tangent compression above ``knee`` toward ``ceiling``."""
    if y <= knee:
        return y
    span = max(ceiling - knee, 1e-6)
    return knee + span * math.tanh((y - knee) / span)


def soft_knee_lower(y: float, floor: float, knee: float) -> float:
    """Apply C^1-smooth hyperbolic tangent lift below ``knee`` toward ``floor``."""
    if y >= knee:
        return y
    span = max(knee - floor, 1e-6)
    return knee - span * math.tanh((knee - y) / span)


def soft_knee_clamp_metallic(m: float, is_linked: bool = False) -> float:
    """Snap unlinked semi-metals cleanly to 0.0 or 1.0; clamp linked values to [0, 1]."""
    clamped = min(max(float(m), 0.0), 1.0)
    if not is_linked and SEMI_METAL_LOW < clamped < SEMI_METAL_HIGH:
        return 1.0 if clamped >= METALLIC_CONDUCTOR_THRESHOLD else 0.0
    return clamped


def soft_knee_clamp_color(
    rgb: Iterable[float],
    metallic: float,
    dielectric_knee: float = CLAMP_DIELECTRIC_KNEE,
    dielectric_max: float = CLAMP_DIELECTRIC_MAX,
) -> Tuple[float, float, float]:
    """Soft-knee compress or lift linear RGB while preserving hue and saturation."""
    vals = list(rgb)
    r = max(float(vals[0]) if len(vals) > 0 else 0.0, 0.0)
    g = max(float(vals[1]) if len(vals) > 1 else 0.0, 0.0)
    b = max(float(vals[2]) if len(vals) > 2 else 0.0, 0.0)

    y = linear_luminance((r, g, b))
    if metallic >= METALLIC_CONDUCTOR_THRESHOLD:
        y_target = soft_knee_lower(y, CLAMP_CONDUCTOR_FLOOR, CLAMP_CONDUCTOR_FLOOR_KNEE)
    else:
        y_up = soft_knee_upper(y, dielectric_knee, dielectric_max)
        y_target = soft_knee_lower(y_up, CLAMP_DIELECTRIC_FLOOR, CLAMP_DIELECTRIC_FLOOR_KNEE)

    if y >= 1e-3:
        scale = y_target / y
        return (r * scale, g * scale, b * scale)

    t = min(max(y / 1e-3, 0.0), 1.0)
    scale = y_target / max(y, 1e-6)
    return (
        y_target * (1.0 - t) + (r * scale) * t,
        y_target * (1.0 - t) + (g * scale) * t,
        y_target * (1.0 - t) + (b * scale) * t,
    )


def soft_knee_clamp_roughness(r: float, transmission: float = 0.0) -> float:
    """Soft-knee clamp non-transmissive roughness away from 0.0/1.0 extremes."""
    val = min(max(float(r), 0.0), 1.0)
    if transmission >= TRANSMISSION_THRESHOLD:
        return val
    low_lifted = soft_knee_lower(val, CLAMP_ROUGHNESS_FLOOR, CLAMP_ROUGHNESS_FLOOR_KNEE)
    return soft_knee_upper(low_lifted, CLAMP_ROUGHNESS_CEIL_KNEE, CLAMP_ROUGHNESS_CEIL)


def soft_knee_clamp_specular(s: float) -> float:
    """Soft-knee clamp Specular IOR Level into [0.255, 0.745]."""
    val = float(s)
    low_lifted = soft_knee_lower(val, CLAMP_SPECULAR_MIN, CLAMP_SPECULAR_MIN_KNEE)
    return soft_knee_upper(low_lifted, CLAMP_SPECULAR_MAX_KNEE, CLAMP_SPECULAR_MAX)


def soft_knee_clamp_ior(ior: float) -> float:
    """Soft-knee clamp dielectric IOR into [1.305, 1.795]."""
    val = float(ior)
    low_lifted = soft_knee_lower(val, CLAMP_IOR_MIN, CLAMP_IOR_MIN_KNEE)
    return soft_knee_upper(low_lifted, CLAMP_IOR_MAX_KNEE, CLAMP_IOR_MAX)


def normalize_texel_ratio(ratio: float) -> float:
    """Map linear density/target ratio to [0.05, 1.0], keeping 0.0 reserved for missing data.

    Mapping: ratio=0.25 -> 0.05, ratio=0.5 -> 0.25, ratio=1.0 -> 0.50, ratio=2.0 -> 0.75, ratio=4.0 -> 1.0.
    """
    if ratio <= 0.0:
        return TEXEL_ATTR_MIN_VALID
    encoded = (math.log2(max(ratio, 1e-6)) + 2.0) / 4.0
    return min(max(encoded, TEXEL_ATTR_MIN_VALID), 1.0)
