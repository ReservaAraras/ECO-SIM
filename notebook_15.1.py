# -*- coding: utf-8 -*-
# pyre-ignore-all-errors
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 15.1: Enhanced Invasive Species Competition with Spatial Dynamics
# DERIVATIVE OF: notebook_15.py (Enhanced with competitive exclusion and management zones)
# ===================================================================================================

"""
Notebook Differentiation:
- Differentiation Focus: Enhanced Invasive Species Competition with Spatial Dynamics emphasizing fruiting synchrony windows.
- Indicator species: Jacare-do-pantanal (Caiman yacare).
- Pollination lens: flowering phenology under drought stress.
- Human impact lens: tourism peak season stress.
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
class InvasiveCompetitionConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 250
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_resolution: int = 32
    initial_native_ratio: float = 0.7

CONFIG = InvasiveCompetitionConfig()

# ===================================================================================================
# 2. COMPETITIVE DYNAMICS MODEL
# ===================================================================================================

class PlantCell:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y
        self.native_cover: float = random.uniform(0.3, 0.9)  # Native species coverage
        self.invasive_cover: float = 0.0  # Invasive species coverage
        self.soil_quality: float = random.uniform(0.4, 1.0)
        self.management_zone: bool = False  # Areas under active management
        self.succession_stage: float = random.uniform(0, 1)  # 0 = early, 1 = mature

class InvasiveCompetitionModel:
    def __init__(self, cfg: InvasiveCompetitionConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Initialize grid
        self.grid_resolution = cfg.grid_resolution
        self.cell_width = cfg.width / cfg.grid_resolution
        self.cell_height = cfg.height / cfg.grid_resolution

        self.cells: List[List[PlantCell]] = []
        for i in range(cfg.grid_resolution):
            row = []
            for j in range(cfg.grid_resolution):
                row.append(PlantCell(i, j))
            self.cells.append(row)

        # Set up initial invasive introduction points
        self.invasive_centers: List[Tuple[int, int]] = [
            (cfg.grid_resolution // 4, cfg.grid_resolution // 4),
            (3 * cfg.grid_resolution // 4, 3 * cfg.grid_resolution // 4),
            (cfg.grid_resolution // 2, cfg.grid_resolution // 3),
        ]

        # Introduce invasive species at centers
        for cx, cy in self.invasive_centers:
            self.cells[cx][cy].invasive_cover = 0.3

        # Set up management zones (controlled areas)
        management_centers = [
            (cfg.grid_resolution // 3, 2 * cfg.grid_resolution // 3),
            (2 * cfg.grid_resolution // 3, cfg.grid_resolution // 3),
        ]
        for mx, my in management_centers:
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    if 0 <= mx + di < cfg.grid_resolution and 0 <= my + dj < cfg.grid_resolution:
                        self.cells[mx + di][my + dj].management_zone = True

        # Track metrics over time
        self.native_coverage_history: List[float] = []
        self.invasive_coverage_history: List[float] = []
        self.grid_snapshots: List[List[List[Tuple[float, float]]]] = []
        self._snap_counter: int = 0

    @staticmethod
    def _enforce_competition_capacity(cell: PlantCell) -> None:
        """Keep total vegetative cover ecologically coherent: native + invasive <= 1."""
        total_cover = cell.native_cover + cell.invasive_cover
        if total_cover > 1.0:
            scale = 1.0 / total_cover
            cell.native_cover *= scale
            cell.invasive_cover *= scale

    def step(self) -> None:
        new_grid: List[List[PlantCell]] = [
            [self.cells[i][j] for j in range(self.grid_resolution)]
            for i in range(self.grid_resolution)
        ]

        total_native = 0.0
        total_invasive = 0.0

        for i, (cells_row, new_row) in enumerate(zip(self.cells, new_grid)):
            for j, cell in enumerate(cells_row):

                growth_rate = 0.05
                if cell.management_zone:
                    invasive_removal = 0.08
                    native_boost = 0.03
                    cell.invasive_cover = max(0.0, cell.invasive_cover - invasive_removal)
                    cell.native_cover = min(1.0, cell.native_cover + native_boost)
                else:
                    invasive_growth = growth_rate * cell.soil_quality * (1 - cell.invasive_cover)
                    invasive_growth *= 1.2
                    native_growth = growth_rate * cell.soil_quality * (1 - cell.native_cover) * 0.7
                    native_growth *= (0.5 + cell.succession_stage * 0.5)

                    cell.invasive_cover = min(1.0, cell.invasive_cover + invasive_growth)
                    cell.native_cover = min(1.0, cell.native_cover + native_growth)

                # Dispersal to neighbors
                neighbors: List[PlantCell] = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_resolution and 0 <= nj < self.grid_resolution:
                            _ni_row = self.cells[ni]
                            neighbors.append(_ni_row[nj])

                if cell.invasive_cover > 0.3:
                    for neighbor in neighbors:
                        spread = cell.invasive_cover * 0.02
                        neighbor.invasive_cover = min(1.0, neighbor.invasive_cover + spread)
                        self._enforce_competition_capacity(neighbor)

                if cell.native_cover > 0.3:
                    for neighbor in neighbors:
                        spread = cell.native_cover * 0.01
                        neighbor.native_cover = min(1.0, neighbor.native_cover + spread)
                        self._enforce_competition_capacity(neighbor)

                self._enforce_competition_capacity(cell)

                total_native += cell.native_cover
                total_invasive += cell.invasive_cover

                new_row[j] = cell

        self.cells = new_grid

        total_cells = float(self.grid_resolution * self.grid_resolution)
        self.native_coverage_history.append(float(total_native) / total_cells)
        self.invasive_coverage_history.append(float(total_invasive) / total_cells)
        self._snap_counter += 1
        if self._snap_counter % 5 == 0:
            snapshot: List[List[Tuple[float, float]]] = []
            for cells_row in self.cells:
                snapshot.append([
                    (cell.native_cover, cell.invasive_cover)
                    for cell in cells_row
                ])
            self.grid_snapshots.append(snapshot)

# ===================================================================================================
# 3. VISUALIZATION - vivid, ludic animated SVG
# ===================================================================================================

class InvasiveCompetitionRenderer:
    def __init__(self, cfg: InvasiveCompetitionConfig, model: InvasiveCompetitionModel):
        self.cfg = cfg
        self.model = model

    def generate_svg(self) -> str:
        """Redesigned SVG: 16×16 coarse grid with habitat zone overlay, clear legend,
        time-series charts, and Cerrado spatial context. File size << 200 kB."""

        w, h  = self.cfg.width, self.cfg.height
        # ── Use a coarser display grid (16×16) derived from the 32×32 model ──
        DG = 16
        G  = self.cfg.grid_resolution  # 32
        scale = G // DG                # 2 — average 2×2 blocks
        dcw = w / DG
        dch = (h - 70) / DG           # leave header + footer strip
        dy0 = 70                       # content starts below header

        snaps = self.model.grid_snapshots
        # Sample up to 8 evenly-spaced snapshots for animation keyframes
        if len(snaps) > 8:
            idxs = [int(i * (len(snaps) - 1) / 7) for i in range(8)]
            snaps = [snaps[i] for i in idxs]
        n   = max(1, len(snaps))
        dur = 16.0

        def _avg_block(snap, bi, bj):
            nat_sum = inv_sum = 0.0
            for di in range(scale):
                for dj in range(scale):
                    ri, rj = bi * scale + di, bj * scale + dj
                    if ri < G and rj < G:
                        c = snap[ri][rj]
                        nat_sum += c[0]; inv_sum += c[1]
            n2 = scale * scale
            return nat_sum / n2, inv_sum / n2

        def _fill(nat: float, inv: float) -> str:
            if inv > nat:
                t = min(1.0, inv)
                r = int(60 + t * 35); g = int(55 - t * 20); b = int(22 + t * 12)
                return f'rgb({r},{g},{b})'
            t = min(1.0, nat)
            r = int(16 + (1 - t) * 22); g = int(72 + t * 58); b = int(24 + t * 24)
            return f'rgb({r},{g},{b})'

        # ── Cerrado habitat zones (static backdrop bands) ────────────────────
        # Three horizontal biome strips: Cerrado (top), Vereda/Gallery (mid), Mata Seca (bottom)
        ZONES_DEF = [
            (0,          dy0,              h * 0.32, '#1b3a14', 'Cerrado stricto sensu'),
            (0,          dy0 + h * 0.32,   h * 0.20, '#0d2b38', 'Vereda / Gallery Forest'),
            (0,          dy0 + h * 0.52,   h * 0.48, '#2b1f0d', 'Mata Seca'),
        ]

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background:#000000;font-family:\'Trebuchet MS\',system-ui,sans-serif;">',
            '<defs>'
            '<radialGradient id="mgtGlow15"><stop offset="0%" stop-color="#00e676" stop-opacity="0.55"/>'
            '<stop offset="100%" stop-color="#004d2a" stop-opacity="0.0"/></radialGradient>'
            '<filter id="cellBlur15"><feGaussianBlur stdDeviation="1.2"/></filter>'
            f'<pattern id="mossStripes" width="56" height="56" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="56" height="56" fill="#000000"/>'
            f'<rect x="0" y="0" width="8" height="56" fill="#3f5f2f" opacity="0.22"/>'
            f'</pattern>'
            '</defs>',
            f'<rect width="{w}" height="{h}" fill="#000000"/>',
            f'<rect width="{w}" height="{h}" fill="url(#mossStripes)"/>',
        ]

        # Habitat zone tinted strips
        for zx, zy, zh, zc, _ in ZONES_DEF:
            svg.append(f'<rect x="{zx}" y="{zy:.0f}" width="{w}" height="{zh:.0f}" fill="{zc}" opacity="0.20"/>')

        # Zone labels (left edge)
        for zx, zy, zh, zc, zlabel in ZONES_DEF:
            lbl_y = zy + zh / 2 + 5
            svg.append(
                f'<text x="8" y="{lbl_y:.0f}" fill="#a0b8a0" font-size="15" '
                f'font-weight="bold" opacity="0.7">{zlabel}</text>'
            )

        # ── Animated coarse grid cells ───────────────────────────────────────
        for bi in range(DG):
            for bj in range(DG):
                x  = bi * dcw
                y  = dy0 + bj * dch

                if snaps:
                    fi_list, op_list = [], []
                    for idx, s in enumerate(snaps):
                        nat, inv = _avg_block(s, bi, bj)
                        fi_list.append(_fill(nat, inv))
                        op_list.append(f'{0.26 + max(nat, inv) * 0.30:.2f}')
                    fi_init = fi_list[0]; op_init = op_list[0]
                    fi_vals = ';'.join(fi_list); op_vals = ';'.join(op_list)
                    svg.append(
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{dcw:.1f}" height="{dch:.1f}" rx="2"'
                        f' fill="{fi_init}" opacity="{op_init}" filter="url(#cellBlur15)">'
                        f'<animate attributeName="fill" values="{fi_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                        f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                        f'</rect>'
                    )
                else:
                    nat, inv = _avg_block(self.model.cells, bi, bj) if snaps else (0.7, 0.0)
                    svg.append(
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{dcw:.1f}" height="{dch:.1f}" rx="2"'
                        f' fill="{_fill(nat, inv)}" opacity="0.45"/>'
                    )

        # Grid lines (subtle)
        for bi in range(DG + 1):
            lx = bi * dcw
            svg.append(f'<line x1="{lx:.0f}" y1="{dy0}" x2="{lx:.0f}" y2="{h - 55}" stroke="#ffffff" stroke-width="0.3" opacity="0.12"/>')
        for bj in range(DG + 1):
            ly = dy0 + bj * dch
            svg.append(f'<line x1="0" y1="{ly:.0f}" x2="{w}" y2="{ly:.0f}" stroke="#ffffff" stroke-width="0.3" opacity="0.12"/>')

        # ── Management zone halos (centroid only, no per-cell overlays) ──────
        mgt_cells = [(i, j) for i in range(G) for j in range(G)
                     if self.model.cells[i][j].management_zone]
        if mgt_cells:
            mi = sum(c[0] for c in mgt_cells) / len(mgt_cells)
            mj = sum(c[1] for c in mgt_cells) / len(mgt_cells)
            mx_pix = (mi / G) * w
            my_pix = dy0 + (mj / G) * (h - 70 - 55)
            r_m = 55
            svg.append(
                f'<circle cx="{mx_pix:.0f}" cy="{my_pix:.0f}" r="{r_m}" fill="url(#mgtGlow15)" opacity="0.5">'
                f'<animate attributeName="opacity" values="0.3;0.65;0.3" dur="3.5s" repeatCount="indefinite"/>'
                f'<animate attributeName="r" values="{r_m-6};{r_m+10};{r_m-6}" dur="4s" repeatCount="indefinite"/>'
                f'</circle>'
                f'<text x="{mx_pix:.0f}" y="{my_pix + r_m + 16:.0f}" text-anchor="middle" '
                f'fill="#80cbc4" font-size="15" font-weight="bold">Management Zone</text>'
            )

        # ── Invasion-source markers ──────────────────────────────────────────
        for cx, cy in self.model.invasive_centers:
            px = (cx / G) * w + dcw / 2
            py = dy0 + (cy / G) * (h - 70 - 55) + dch / 2
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" r="10" fill="none" '
                f'stroke="#ff5722" stroke-width="2" opacity="0">'
                f'<animate attributeName="r" values="6;22;6" dur="2.8s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.7;0;0.7" dur="2.8s" repeatCount="indefinite"/>'
                f'</circle>'
                f'<circle cx="{px:.0f}" cy="{py:.0f}" r="5" fill="#ff5722">'
                f'<animate attributeName="r" values="3;6;3" dur="1.5s" repeatCount="indefinite"/>'
                f'</circle>'
                f'<text x="{px:.0f}" y="{py - 14:.0f}" text-anchor="middle" '
                f'fill="#ffab91" font-size="15" font-weight="bold">Capim-gordura source</text>'
            )

        # ── Right panel: coverage chart + legend ─────────────────────────────
        px0 = w - 295
        # Chart background
        ch_h = 130
        ch_y = 80
        ch_w = 270
        svg.append(
            f'<rect x="{px0}" y="{ch_y}" width="{ch_w}" height="{ch_h}" fill="#0c1218" rx="8"'
            f' stroke="#1e3a5f" stroke-width="1" opacity="0.95"/>'
            f'<text x="{px0 + 12}" y="{ch_y + 20}" fill="#90b8d9" font-size="15" font-weight="bold">'
            f'Relative Coverage (%)</text>'
        )
        history_nat = self.model.native_coverage_history
        history_inv = self.model.invasive_coverage_history
        tot = max(1, len(history_nat))
        cx0 = px0 + 12; cy1 = ch_y + ch_h - 14; cw1 = ch_w - 24; cih = ch_h - 42

        def _polyline(hist, color, dash=""):
            pts = [f"{cx0 + (fi / (tot - 1)) * cw1:.1f},{cy1 - hist[fi] * cih:.1f}"
                   for fi in range(tot)]
            st = f'stroke-dasharray="{dash}"' if dash else ''
            return (f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}"'
                    f' stroke-width="2.5" stroke-linecap="round" {st} opacity="0.9"/>')

        if len(history_nat) > 1:
            svg.append(_polyline(history_nat, '#4cff88'))
            svg.append(_polyline(history_inv, '#ffab40', '4,3'))

        # Y-axis tick labels
        for pct, label in [(0, '0%'), (0.5, '50%'), (1.0, '100%')]:
            ty2 = cy1 - pct * cih
            svg.append(
                f'<line x1="{cx0 - 4}" y1="{ty2:.0f}" x2="{cx0}" y2="{ty2:.0f}"'
                f' stroke="#4a6a8a" stroke-width="1"/>'
                f'<text font-weight="bold" x="{cx0 - 6}" y="{ty2 + 4:.0f}" text-anchor="end" fill="#5a7898" font-size="15">{label}</text>'
            )

        final_nat = history_nat[-1] if history_nat else 0.0
        final_inv = history_inv[-1] if history_inv else 0.0

        # Legend panel
        lx, ly, lh = px0, ch_y + ch_h + 10, 120
        svg.append(
            f'<rect x="{lx}" y="{ly}" width="{ch_w}" height="{lh}" fill="#0c1218" rx="8"'
            f' stroke="#1e3a5f" stroke-width="1" opacity="0.95"/>'
            f'<text x="{lx + 12}" y="{ly + 20}" fill="#90b8d9" font-size="15" font-weight="bold">Legend</text>'
        )
        legend_items = [
            ('#4cff88', f'Native Cerrado Flora  {final_nat:.0%}', False),
            ('#ffab40', f'Invasive Capim-gordura  {final_inv:.0%}', True),
            ('#00e676', 'Active Management Zone', False),
            ('#ff5722', 'Invasion Introduction Site', False),
        ]
        for k, (lcolor, ltext, dashed) in enumerate(legend_items):
            iy = ly + 36 + k * 20
            if dashed:
                svg.append(f'<line x1="{lx + 14}" y1="{iy}" x2="{lx + 28}" y2="{iy}"'
                           f' stroke="{lcolor}" stroke-width="3" stroke-dasharray="4,2"/>')
            else:
                svg.append(f'<rect x="{lx + 14}" y="{iy - 7}" width="14" height="10" rx="2" fill="{lcolor}" opacity="0.85"/>')
            svg.append(f'<text font-weight="bold" x="{lx + 34}" y="{iy + 4}" fill="#c0d0e0" font-size="15">{ltext}</text>')

        # ── Bottom stats bar ─────────────────────────────────────────────────
        bar_y = h - 52
        svg.append(
            f'<rect x="0" y="{bar_y}" width="{w}" height="52" fill="#06080e" opacity="0.85"/>'
            f'<text x="20" y="{bar_y + 20}" fill="#90b8d9" font-size="15" font-weight="bold">'
            f'Final State:</text>'
            f'<rect x="130" y="{bar_y + 8}" width="16" height="16" rx="3" fill="#4cff88"/>'
            f'<text font-weight="bold" x="152" y="{bar_y + 21}" fill="#c0d8c0" font-size="15">Native: {final_nat:.1%}</text>'
            f'<rect x="310" y="{bar_y + 8}" width="16" height="16" rx="3" fill="#ffab40"/>'
            f'<text font-weight="bold" x="332" y="{bar_y + 21}" fill="#d8c0a0" font-size="15">Invasive: {final_inv:.1%}</text>'
            f'<text font-weight="bold" x="530" y="{bar_y + 21}" fill="#607d8b" font-size="15">{dur:.0f}s animated loop</text>'
            f'<text font-weight="bold" x="680" y="{bar_y + 21}" fill="#607d8b" font-size="15">'
            f'Grid: {DG}×{DG} blocks · {len(snaps)} keyframes</text>'
            # Second row: Cerrado habitat breakdown
            f'<text font-weight="bold" x="20" y="{bar_y + 42}" fill="#607d8b" font-size="15">'
            f'Habitats: Cerrado stricto sensu · Vereda/Gallery Forest · Mata Seca  '
            f'— RESEX Recanto das Araras, Terra Ronca-GO</text>'
        )

        # ── Header / title bar ───────────────────────────────────────────────
        svg.append(
            f'<rect x="0" y="0" width="{w}" height="{dy0}" fill="#06080e" opacity="0.82"/>'
            f'<text x="20" y="32" fill="#f0f4e8" font-size="15" font-weight="bold" letter-spacing="0.5">'
            f'ECO-SIM: Invasive Species Competition \u2014 Capim-gordura Spatial Dynamics</text>'
            f'<text font-weight="bold" x="20" y="54" fill="#8bb4d9" font-size="15">'
            f'Competitive exclusion of native Cerrado flora \u00b7 management zones \u00b7 {dur:.0f}s animated loop</text>'
            f'<line x1="20" y1="64" x2="640" y2="64" stroke="#4fc3f7" stroke-width="2"'
            f' stroke-linecap="round" opacity="0.6">'
            f'<animate attributeName="x2" values="200;640;200" dur="6s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

def main():
    print(f"Initializing invasive competition model on {CONFIG.device}...")
    model = InvasiveCompetitionModel(CONFIG)

    for _ in range(CONFIG.frames):
        model.step()

    final_native = model.native_coverage_history[-1] if model.native_coverage_history else 0
    final_invasive = model.invasive_coverage_history[-1] if model.invasive_coverage_history else 0
    print(f"Final coverage - Native: {final_native:.3f}, Invasive: {final_invasive:.3f}")

    print("Simulation complete. Rendering...")
    renderer = InvasiveCompetitionRenderer(CONFIG, model)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_15.1')
    return svg_content

if __name__ == "__main__":
    main()