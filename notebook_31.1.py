# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 31.1: Enhanced Integrated Fire Management (MIF) Simulation with PyTorch
# DERIVATIVE OF: notebook_31.py (Enhanced with real-time fire spread dynamics)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Enhanced Integrated Fire Management (MIF) Simulation with PyTorch emphasizing apex predator presence.
- Indicator species: Cagaita (Eugenia dysenterica).
- Pollination lens: flowering phenology under drought stress.
- Human impact lens: tourism peak season stress.

Scientific Relevance (PIGT RESEX Recanto das Araras -- 2024):
- Integrates the socio-environmental complexity of 
  de Cima, Goiás, Brazil.
- Models landscape connectivity, karst vulnerability (Bacia do Rio Lapa),
  biological corridors, and seed-dispersal networks.
- Demonstrates parameters for ecological succession, biodiversity indices,
  integrated fire management (MIF), and ornithochory dynamics.
- Outputs are published via Google Sites: 
- SVG artefacts archived at: https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing
"""


from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ===================================================================================================
# 1. SCIENTIFIC CONTEXT & PARAMETERS
# ===================================================================================================

@dataclass
class FireConfig:
    """Class `FireConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 300
    fps: int = 12
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_resolution: int = 64
    fire_spread_rate: float = 0.15
    wind_speed: float = 2.5
    wind_direction: float = 45.0
    vegetation_fuel: float = 0.8
    suppression_efficiency: float = 0.6

CONFIG = FireConfig()

# ===================================================================================================
# 2. FIRE SPREAD MODEL (TENSORIZED)
# ===================================================================================================

class FireSpreadModel:
    """Class `FireSpreadModel` -- simulation component."""

    def __init__(self, cfg: FireConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Fire grid (0=unburned, 0.5=burning, 1=burned)
        self.fire_grid = torch.zeros((cfg.grid_resolution, cfg.grid_resolution), device=self.dev)
        self.fire_intensity = torch.zeros((cfg.grid_resolution, cfg.grid_resolution), device=self.dev)

        # Wind vector
        wind_rad = math.radians(cfg.wind_direction)
        self.wind_vec = torch.tensor([math.cos(wind_rad), math.sin(wind_rad)], device=self.dev) * cfg.wind_speed

        # Temperature and humidity grids
        self.temperature = torch.rand((cfg.grid_resolution, cfg.grid_resolution), device=self.dev) * 30 + 15
        self.humidity = torch.rand((cfg.grid_resolution, cfg.grid_resolution), device=self.dev) * 0.4 + 0.3

        # Vegetation fuel map (karst areas have less fuel)
        x = torch.linspace(0, cfg.width, cfg.grid_resolution, device=self.dev)
        y = torch.linspace(0, cfg.height, cfg.grid_resolution, device=self.dev)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        karst_influence = torch.sin(xx / 100) * torch.cos(yy / 80) * 0.3 + 0.7
        self.fuel_map = karst_influence * cfg.vegetation_fuel

        # Initialize fire ignition points (controlled burns + lightning)
        self.ignition_points = [
            (cfg.grid_resolution // 4, cfg.grid_resolution // 3),
            (3 * cfg.grid_resolution // 4, 2 * cfg.grid_resolution // 3),
        ]
        for ix, iy in self.ignition_points:
            self.fire_grid[ix, iy] = 1.0
            self.fire_intensity[ix, iy] = 1.0

        self.history = [self.fire_grid.cpu().numpy().copy()]

    def step(self):
        """Function `step` -- simulation component."""
        new_fire = self.fire_grid.clone()
        new_intensity = self.fire_intensity.clone()

        # Update already burning
        burning = self.fire_grid > 0
        new_fire[burning] = torch.clamp(self.fire_grid[burning] + 0.1, max=1.0)
        
        intensifying = burning & (self.fire_grid < 0.8)
        dying = burning & (self.fire_grid >= 0.8)
        new_intensity[intensifying] = torch.clamp(self.fire_intensity[intensifying] + 0.05, max=1.0)
        new_intensity[dying] = torch.clamp(self.fire_intensity[dying] - 0.02, min=0.0)

        # Spread to unburned
        unburned = self.fire_grid == 0
        if unburned.any() and burning.any():
            spread_prob = torch.zeros_like(self.fire_grid)
            wind_effect = self.wind_vec / (self.cfg.grid_resolution / 10.0)
            
            # 8 neighbors
            shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            
            temp_factor = (self.temperature - 15) / 30.0
            humidity_factor = 1.0 - self.humidity
            base_prob = self.cfg.fire_spread_rate * self.fuel_map * (1.0 + temp_factor * 0.5) * humidity_factor
            
            for di, dj in shifts:
                src_i_start = max(0, -di); src_i_end = self.cfg.grid_resolution - max(0, di)
                src_j_start = max(0, -dj); src_j_end = self.cfg.grid_resolution - max(0, dj)
                
                dst_i_start = max(0, di);   dst_i_end = self.cfg.grid_resolution - max(0, -di)
                dst_j_start = max(0, dj);   dst_j_end = self.cfg.grid_resolution - max(0, -dj)
                
                wind_alignment = ((-di) * wind_effect[0] + (-dj) * wind_effect[1])
                neighbor_burning = self.fire_grid[src_i_start:src_i_end, src_j_start:src_j_end] > 0
                
                prob = base_prob[dst_i_start:dst_i_end, dst_j_start:dst_j_end] * (1.0 + wind_alignment * 0.3)
                
                spread_prob[dst_i_start:dst_i_end, dst_j_start:dst_j_end] = torch.max(
                    spread_prob[dst_i_start:dst_i_end, dst_j_start:dst_j_end],
                    torch.where(neighbor_burning, prob, torch.zeros_like(prob))
                )
            
            ignite = unburned & (torch.rand_like(self.fire_grid) < spread_prob)
            new_fire[ignite] = 0.3
            new_intensity[ignite] = 0.5

        # Apply suppression (managed burn lines)
        suppression_zone = int(self.cfg.suppression_efficiency * 10)
        for ix, iy in self.ignition_points:
            i_start = max(0, ix - suppression_zone)
            i_end = min(self.cfg.grid_resolution, ix + suppression_zone + 1)
            j_start = max(0, iy - suppression_zone)
            j_end = min(self.cfg.grid_resolution, iy + suppression_zone + 1)
            
            mask = new_fire[i_start:i_end, j_start:j_end] > 0
            suppress = mask & (torch.rand_like(new_fire[i_start:i_end, j_start:j_end]) < 0.1 * self.cfg.suppression_efficiency)
            
            # Need to carefully apply suppression via view to modify new_fire
            zone = new_fire[i_start:i_end, j_start:j_end]
            zone[suppress] = torch.clamp(zone[suppress] - 0.3, min=0.0)

        self.fire_grid = new_fire
        self.fire_intensity = new_intensity
        self.history.append(self.fire_grid.cpu().numpy().copy())

# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class FireRenderer:
    """Class `FireRenderer` -- simulation component."""

    def __init__(self, cfg: FireConfig, model: FireSpreadModel):
        self.cfg = cfg
        self.model = model

    def generate_svg(self) -> str:
        """Animated SVG of fire spread across all recorded history keyframes."""

        w, h = self.cfg.width, self.cfg.height
        G    = self.cfg.grid_resolution
        sx   = w / G
        sy   = h / G

        history = self.model.history          # list[ndarray(G,G)], len ≈ frames+1
        stride  = max(1, len(history) // 40)  # ~40 keyframes
        kframes = history[::stride]
        n       = len(kframes)
        dur     = 15.0                         # 15 s animation loop

        def _fill(v: float) -> str:
            if v >= 0.7:  return "#ffaa00"
            if v >= 0.3:  return "#ff4400"
            if v >  0.0:  return "#882200"
            return "#121212"

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background:#121212;font-family:system-ui,sans-serif;">',

            # gradient for glow effect on heavy fire
            '<defs>'
            '<radialGradient id="fglow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#ffaa00" stop-opacity="0.9"/>'
            '<stop offset="100%" stop-color="#ff4400" stop-opacity="0"/>'
            '</radialGradient>'
            '</defs>',

            f'<rect width="{w}" height="{h}" fill="#121212"/>',
        ]

        # ── Fuel map - subtle green tint shows vegetation density ──────────
        fuel_np = self.model.fuel_map.cpu().numpy()
        for i in range(G):
            for j in range(G):
                fv = float(fuel_np[i, j])
                gv = int(30 + fv * 35)
                svg.append(
                    f'<rect x="{i*sx:.1f}" y="{j*sy:.1f}" '
                    f'width="{sx:.1f}" height="{sy:.1f}" '
                    f'fill="rgb(0,{gv},0)" opacity="0.22"/>'
                )

        # ── Animated fire cells ─────────────────────────────────────────────
        for i in range(G):
            for j in range(G):
                vals = [float(kframes[f][i, j]) for f in range(n)]
                if max(vals) == 0.0:
                    continue

                x, y     = i * sx, j * sy
                op_str   = ";".join(f"{v:.2f}" for v in vals)
                fill_str = ";".join(_fill(v) for v in vals)

                svg.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" '
                    f'width="{sx:.1f}" height="{sy:.1f}" '
                    f'fill="#882200" opacity="0">'
                    f'<animate attributeName="fill" values="{fill_str}" '
                    f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'<animate attributeName="opacity" values="{op_str}" '
                    f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'</rect>'
                )

        # ── Wind compass ────────────────────────────────────────────────────
        wx, wy = w - 150, 70
        angle  = self.cfg.wind_direction
        ax = wx + 38 * math.cos(math.radians(angle))
        ay = wy + 38 * math.sin(math.radians(angle))
        svg.append(
            f'<circle cx="{wx}" cy="{wy}" r="34" fill="#1a1a1a" stroke="#444" stroke-width="1"/>'
            f'<text font-weight="bold" x="{wx}" y="{wy-42}" fill="#888" font-size="15" text-anchor="middle">N</text>'
            f'<line x1="{wx}" y1="{wy}" x2="{ax:.1f}" y2="{ay:.1f}" '
            f'stroke="#ffaa00" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{wx}" cy="{wy}" r="4" fill="#ffaa00"/>'
            f'<text font-weight="bold" x="{wx}" y="{wy+52}" fill="#cccccc" font-size="15" text-anchor="middle">'
            f'Wind {self.cfg.wind_speed} m/s</text>'
        )

        # ── Legend ──────────────────────────────────────────────────────────
        lx, ly = w - 200, h - 120
        svg.append(
            f'<rect x="{lx}" y="{ly}" width="190" height="110" '
            f'fill="#1a1a2e" opacity="0.85" rx="6"/>'
        )
        legend = [
            ("#ffaa00", "High intensity"),
            ("#ff4400", "Spreading fire"),
            ("#882200", "Smoldering"),
            ("#333333", "Burned area"),
            ("#005500", "Vegetation/fuel"),
        ]
        for k, (col, label) in enumerate(legend):
            ey = ly + 18 + k * 19
            svg.append(f'<rect x="{lx+10}" y="{ey-11}" width="13" height="13" fill="{col}"/>')
            svg.append(f'<text font-weight="bold" x="{lx+29}" y="{ey}" fill="#cccccc" font-size="15">{label}</text>')

        # ── Title bar ───────────────────────────────────────────────────────
        svg.append(
            f'<rect x="0" y="0" width="{w}" height="66" fill="#000" opacity="0.6"/>'
            f'<text x="20" y="30" fill="#ffffff" font-size="15" font-weight="bold">'
            f'ECO-SIM: Enhanced MIF - Fire Spread Dynamics</text>'
            f'<text font-weight="bold" x="20" y="52" fill="#aaaaaa" font-size="15">'
            f'wind {self.cfg.wind_direction}° · spread rate {self.cfg.fire_spread_rate} · '
            f'suppression {self.cfg.suppression_efficiency} · '
            f'{dur:.0f} s loop</text>'
        )

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

# -- SVG Export to Google Drive --------------------------------------------
def save_svg_to_drive(svg_content: str, notebook_id: str):
    """Persist SVG artefact to Google Drive for publication on Google Sites."""
    import os
    drive_folder = "/content/drive/MyDrive/ReservaAraras_SVGs"
    # Colab: mount drive first; local: use fallback folder
    if os.path.isdir('/content/drive'):
        save_dir = drive_folder
    else:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        save_dir = os.path.join(base_dir, 'svg_output')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'{notebook_id}.svg')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"SVG saved -> {filepath}")
    return filepath

def main():
    """Function `main` -- simulation component."""

    print(f"Initializing enhanced MIF fire model on {CONFIG.device}...")
    model = FireSpreadModel(CONFIG)

    for _ in range(CONFIG.frames):
        model.step()

    print("Simulation complete. Rendering...")
    renderer = FireRenderer(CONFIG, model)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    
    

    save_svg(svg_content, 'notebook_31.1')
    return svg_content

if __name__ == "__main__":
    main()