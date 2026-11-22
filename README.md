# FlightCheck

**Real-Time PBR, Texel Density, and Scale Pre-Flight Suite for Blender 5.0+**

FlightCheck is a Blender extension that provides real-time false-color shader overlays, an aviation-style 2D viewport Flight HUD, 3D bounding-cage alerts, and non-destructive automated remediation tools to catch physically invalid PBR shaders, sRGB data-map bugs, texel density mismatches, and transform/scale anomalies before lookdev, export, or final rendering.

---

## Features

- **Real-Time Shader Diagnostic Overlays (`O(1)` Mode Switching):** Splices a global controller (`__FLIGHTCHECK_CTRL__`) and overlay shader group (`__FLIGHTCHECK_OVERLAY__`) onto active materials to visualize **Albedo**, **Roughness / Specular / IOR**, **Texel Density**, and **Combined** diagnostics directly in Material Preview or Rendered viewport shading.
- **2D Aviation Flight HUD (`COMPACT`, `FULL`, `MINIMAL`):** Renders a viewport telemetry strip/card (`GO` / `CAUTION` / `NO-GO`), per-subsystem compliance gauges (`ALB`, `RGH`, `TXL`, `SCL`), stale-cache indicators, Solid-shading hints, data-map color-space warnings, and active mode legends in any viewport corner with full UI scale awareness.
- **3D Wireframe Bounding Cages:** Draws world-space wireframe warning cages for inverted (`SCALE_INVERTED`), non-uniform (`SCALE_NON_UNIFORM`), unapplied (`SCALE_UNAPPLIED`), and out-of-range dimension (`< 1 mm` / `> 500 m`) mesh objects.
- **Non-Destructive Soft-Knee PBR Clamps:** Inserts `__FLIGHTCHECK_CLAMP__` shader groups using $C^1$-smooth hyperbolic tangent (`TANH`) compression/lifting on out-of-spec BSDF inputs while preserving original upstream node links, hue/saturation ratios, and compliant materials.
- **Data-Map Color-Space Fixer:** Automatically sets data-only textures (`Roughness`, `Metallic`, `Normal`, `Bump`, `Displacement`, etc.) to `Non-Color` (with dynamic colorspace fallback) while protecting textures shared with color sockets such as `Base Color`.
- **Headless-Safe Transform Fixer:** Bakes unapplied `scale` and `delta_scale` directly into mesh data (including shape keys and negative-determinant face winding), splits multi-user meshes with conflicting scales, preserves child world transforms (`matrix_parent_inverse`), and skips keyframed/driven scales.
- **Relative & Absolute Texel Equalizer:**
  - `RELATIVE` (default): Equalizes UV island density to the object's area-weighted average and packs islands cleanly inside `[0, 1]`.
  - `ABSOLUTE`: Scales UV islands directly to the configured target texel density (`px/m`) with preset buttons (`256`, `512`, `1024`, `2048`, `4096`).

---

## Requirements

- **Blender 5.0.0 or newer** (tested and verified on **Blender 5.3.0**)
- Uses Blender 5.0+ Extension packaging (`blender_manifest.toml`) and modern `gpu` / `gpu_extras` / `blf` APIs (zero `bgl`, zero `bl_info`, zero legacy `.use_nodes`)

---

## Installation

Download `flightcheck-<version>.zip` from the [GitHub Releases](https://github.com/Savvyboi/FlightCheck/releases) page, then install using any of the following methods:

1. **Drag and Drop (Recommended):**
   Drag `flightcheck-<version>.zip` directly into the open Blender 5.0+ window and confirm installation.
2. **Preferences UI:**
   Open **Edit → Preferences → Get Extensions**, open the top-right **⌄** menu, choose **Install from Disk...**, and select `flightcheck-<version>.zip`.
3. **Blender CLI:**
   ```bash
   blender --command extension install-file -r user_default -e flightcheck-1.0.0.zip
   ```

Enable **FlightCheck** in **Edit → Preferences → Add-ons** (or Extensions) if not automatically enabled.

---

## Quick Start

- **N-Panel:** Open the 3D Viewport sidebar (`N`) → **FlightCheck** tab to run **Pre-Flight Scan**, toggle real-time overlays, configure targets/thresholds, run fixers, or inspect and jump to individual issues in the checklist.
- **3D Viewport Header Popover:** Click the **`FC:`** status badge (`FC: READY`, `FC: GO`, `FC: CAUTION`, or `FC: NO-GO`) in the 3D Viewport header for quick mode switching, HUD style/corner selection, safe fixes, and cleanup.
- **Pie Menu (`Alt+Shift+F`):** Press `Alt+Shift+F` in the 3D Viewport for rapid switching between **Off**, **Albedo**, **Roughness**, **Texel Density**, **Combined**, **Scan**, **2D HUD Toggle**, and **Fix Safe Issues**.

> **Note:** Real-time false-color shader overlays require **Material Preview** or **Rendered** viewport shading. When viewport shading is in `SOLID` or `WIREFRAME` with an overlay mode active, the HUD and panel display a shading hint.

---

## Diagnostic Rules

All luminance evaluations operate in **Linear Rec.709** ($Y = 0.2126 R + 0.7152 G + 0.0722 B$):

| Code | Category | Severity | Trigger Condition | Overlay / Visual |
| :--- | :--- | :--- | :--- | :--- |
| `DIELECTRIC_CRUSHED_BLACK` | Albedo | Critical | Dielectric (`Metallic < 0.5`), $Y < 0.02$ | Deep Blue |
| `BLOWN_ALBEDO_WARN` | Albedo | Warning | Dielectric (`Metallic < 0.5`), $0.70 < Y \le 0.81$ | Amber |
| `BLOWN_ALBEDO_CRIT` | Albedo | Critical | Dielectric (`Metallic < 0.5`), $Y > 0.81$ | Red |
| `DARK_METAL` | Albedo | Critical | Conductor (`Metallic >= 0.5`), $Y < 0.45$ | Cyan |
| `SEMI_METAL` | Albedo | Critical | Unlinked `Metallic` in $(0.05, 0.95)$ | Bright Magenta |
| `MIRROR_ROUGHNESS` | Roughness | Warning | Non-transmissive (`Transmission < 0.5`), `Roughness < 0.02` | Purple |
| `DEAD_ROUGHNESS` | Roughness | Warning | Non-transmissive (`Transmission < 0.5`), `Roughness > 0.98` | Orange |
| `SPECULAR_OUTLIER` | Roughness | Warning | `Specular IOR Level` outside $[0.25, 0.75]$ | Yellow-Red Hatching |
| `IOR_OUTLIER` | Roughness | Warning | Non-transmissive dielectric `IOR` outside $[1.30, 1.80]$ | Yellow-Red Hatching |
| `DATA_MAP_SRGB` | Color Space | Critical | Non-color data socket/node image using non-data color space (Alpha-only excluded) | HUD Banner |
| `UNEVALUABLE_INPUT` | Albedo / Roughness | Info | Dynamic/procedural node input that cannot be statically bounded | Checklist Info |
| `TEXEL_NO_UV` | Texel | Warning | Mesh faces lack an active UV map | Pass-Through Gray |
| `TEXEL_LOW` / `TEXEL_HIGH` | Texel | Warning | Texel density ratio outside $[0.5, 2.0]$ vs. target $\text{px/m}$ | Blue → Green → Red |
| `TEXEL_SHARED_MESH_SCALE` | Texel | Info | Multiple objects share a mesh with differing world scales | Checklist Info |
| `SCALE_INVERTED` / `SCALE_NON_UNIFORM` | Scale | Critical | Negative determinant/inverted scale or non-uniform scale | Red 3D Cage |
| `SCALE_UNAPPLIED` | Scale | Warning | Uniform positive scale or `delta_scale` $\ne 1.0$ | Amber 3D Cage |
| `DIMENSION_TOO_SMALL` / `DIMENSION_TOO_LARGE` | Scale | Warning | Mesh bounding dimensions $< 1\text{ mm}$ or $> 500\text{ m}$ | Yellow 3D Cage |

Textured `Base Color` and `Roughness` channels use 1st/99th percentile (`p01`/`p99`) evaluation across opaque pixels so isolated single-pixel noise does not cause false positives.

---

## Non-Destructive, Render & Save Safety

- **Render Safety (`render_pre` / `render_complete`):** Temporarily forces `__FLIGHTCHECK_CTRL__` to `0` (`OFF`) before rendering begins and restores your active diagnostic mode after rendering completes or cancels—so false-color overlays never contaminate final renders.
- **Save Safety (`save_pre` / `save_post`):** Temporarily unsplices diagnostic overlay nodes, removes `__flightcheck_td` mesh attributes, and deletes internal helper groups (`__FLIGHTCHECK_OVERLAY__`, `__FLIGHTCHECK_CTRL__`) prior to saving `.blend` files, then restores them afterward. Soft-knee `__FLIGHTCHECK_CLAMP__` nodes remain intact unless explicitly removed.
- **Clean Uninstall / Reset:** Disabling the extension (`unregister()`) resets control mode to `0` and unsplices all viewport overlays. Running **Clean Up All** (`flightcheck.cleanup`) removes all overlay splices, clamp nodes (optional), `__flightcheck_td` attributes, and cached state, restoring original material links and values losslessly.

---

## Development & Headless Testing

Clone the repository and run the automated 8-suite verification harness headlessly with Blender 5.0+:

```bash
git clone https://github.com/Savvyboi/FlightCheck.git
cd FlightCheck

# Run all 8 automated test suites (API compliance, lifecycle, scanners, splicing, texels, HUD, fixers, P0-P3 regressions)
blender --factory-startup --background --python tests/run_tests.py

# Optionally generate a deterministic violation test .blend scene
blender --factory-startup --background --python tests/generate_test_scene.py -- ./test_out.blend
```

---

## License

FlightCheck is licensed under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`). See [LICENSE](LICENSE) for full terms.
