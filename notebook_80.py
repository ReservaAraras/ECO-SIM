# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 80: Sauva (Atta spp.) <-> Leucoagaricus (Fungus Garden)
#            Seasonal Wet/Dry Subterranean Symbiosis Phenological Clock
# INTERVENTION: Seasonal / Migratory Cerrado Dynamics -- Passage of Months Series
# ===================================================================================================
"""
notebook_80.py -- Sauva x Leucoagaricus: Seasonal Subterranean Symbiosis Clock
Notebook Differentiation:
- Differentiation Focus: Seasonal/Migratory Cerrado Dynamics (Part 2) emphasizing dry-season stress refuges.
- Indicator species: Queixada (Tayassu pecari).
- Pollination lens: ground-nesting bee vulnerability.
- Human impact lens: traditional harvesting pressure.

Models the subterranean fungus-garden mutualism between Atta leaf-cutter ants
and their cultivated Leucoagaricus basidiomycete. The Cerrado's seasonal extremes
drive dramatic behavioral plasticity:

  WET SEASON (Oct-Mar): Intensive surface leaf-foraging. Nuptial flights (alates)
  at first heavy rains (Sep-Oct). Maximum fungus garden productivity.

  DRY SEASON (Apr-Sep): Surface foraging contracts. Colony retreats to deeper,
  moister chambers to protect the fungus from desiccation. Microclimate regulation
  and deeper soil gardening dominate.

Seasonal phenology mapped on the annual clock:
  * Monthly surface foraging intensity (wet: maximum; dry: contracted)
  * Nuptial flight / alate emergence pulse (Sep-Oct first rains)
  * Fungus garden productivity (tied to leaf input rate)
  * Colony depth regulation (shallow wet, deeper dry)
  * Rainfall backbone (shared with nb71-79 series)

Cross-references within the Passage of Months series:
  * nb78: Cupim x Termitomyces -- parallel animal x fungus subterranean mutualism.
  * nb62: Tamandua x Termite mounds -- top-down colony regulation clock.
  * nb69: Tamandua x Cupinzeiros -- predator seasonal interaction clock.
  * nb79: Suiriri x Murici -- complementary wet-season surface dynamics.

Scientific References:
  - Holldobler & Wilson (1990). The Ants. Harvard University Press.
  - Branstetter et al. (2017). Dry habitats were ancestral to Atta leaf-cutter ants.
  - Wirth et al. (2003). Leaf-cutting ants in Cerrado ecosystem processes. Biotropica.
  - Pinheiro et al. (2002). Termite swarming activity and rainfall (Cerrado).
  - PIGT RESEX Recanto das Araras observations, Goias (2022-2024).

Scientific Relevance (PIGT  -- 2024):
- Explores complex Animal x Fungus interactions critical to Cerrado soil health.
- Demonstrates behavioral plasticity of social insects in response to seasonal
  aridity and temperature fluctuations.
- Emphasizes seasonal aspects as per the latest RESEX parameters.
- SVG artefacts archived at: https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from typing import List
from eco_base import save_svg, draw_phenology_chart , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES  (month 0 = January)
# ===================================================================================================

# Surface foraging intensity (proportion of colony active above-ground)
FORAGING_SURFACE = [0.85, 0.80, 0.75, 0.55, 0.35, 0.20, 0.15, 0.25, 0.45, 0.80, 0.90, 0.90]

# Nuptial flight / alate emergence (Sep-Oct first rains trigger)
NUPTIAL_FLIGHT = [0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.20, 0.60, 0.90, 0.50]

# Fungus garden productivity (tied to fresh leaf input x moisture)
FUNGUS_PRODUCTIVITY = [0.80, 0.75, 0.70, 0.55, 0.40, 0.30, 0.28, 0.35, 0.55, 0.85, 0.95, 0.90]

# Colony retreat depth (1.0 = deepest dry-season; 0.0 = shallow wet-season surface)
COLONY_DEPTH = [0.10, 0.15, 0.25, 0.55, 0.80, 0.95, 1.00, 0.85, 0.55, 0.20, 0.10, 0.10]

# Rainfall -- shared wet-season backbone (Oct-Mar)
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]


@dataclass
class SauvaConfig:
    width:  int   = 1280
    height: int = CANVAS_HEIGHT
    frames: int   = 360
    fps:    int   = 10
    device: str   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry -- consistent with nb71-78 series
    clock_cx:     float = 390.0
    clock_cy:     float = 295.0
    clock_radius: float = 230.0

    num_ants:  int   = 120
    nest_cx:   float = 390.0
    nest_cy:   float = 360.0   # slightly below clock center
    surface_y: float = 200.0   # y-coordinate of surface line in clock arena


CONFIG = SauvaConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class SauvaSim:

    def __init__(self, cfg: SauvaConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        nest = torch.tensor([cfg.nest_cx, cfg.nest_cy], device=self.dev, dtype=torch.float32)
        self.nest = nest

        torch.manual_seed(11)
        self.ant_pos = (
            nest.unsqueeze(0).repeat(cfg.num_ants, 1)
            + torch.randn((cfg.num_ants, 2), device=self.dev) * 25.0
        )
        self.ant_vel = torch.randn((cfg.num_ants, 2), device=self.dev)

        # 0: in nest  1: surface foraging  2: returning  3: deep culture  4: alate nuptial
        self.ant_state = torch.zeros(cfg.num_ants, dtype=torch.long, device=self.dev)

        self.hist_ant:    List = []
        self.hist_fungus: List = []
        self.hist_month:  List = []
        self.hist_depth:  List = []

        self.total_leaf_trips      = 0
        self.total_nuptial_events  = 0
        self.peak_foragers         = 0

    def _interp(self, curve: list, month_frac: float) -> float:
        m  = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t  = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, fi: int):
        cfg = self.cfg
        dev = self.dev
        month_frac = (fi / cfg.frames) * 12.0
        self.hist_month.append(month_frac)

        forage_lvl = self._interp(FORAGING_SURFACE,    month_frac)
        nuptial    = self._interp(NUPTIAL_FLIGHT,      month_frac)
        fungus_p   = self._interp(FUNGUS_PRODUCTIVITY, month_frac)
        depth      = self._interp(COLONY_DEPTH,        month_frac)

        self.hist_fungus.append(fungus_p)
        self.hist_depth.append(depth)

        # Allocate ants to behavioural states by seasonal level
        num_foragers  = max(0, int(cfg.num_ants * forage_lvl  * 0.65))
        num_nuptial   = max(0, int(cfg.num_ants * nuptial     * 0.15))
        num_returning = max(0, int(cfg.num_ants * forage_lvl  * 0.15))
        num_deep      = max(0, cfg.num_ants - num_foragers - num_nuptial - num_returning)

        idx = 0
        self.ant_state[idx:idx+num_foragers]  = 1; idx += num_foragers
        self.ant_state[idx:idx+num_returning] = 2; idx += num_returning
        self.ant_state[idx:idx+num_nuptial]   = 4; idx += num_nuptial
        self.ant_state[idx:]                  = 3

        if num_foragers > self.peak_foragers:
            self.peak_foragers = num_foragers

        cx, cy = cfg.clock_cx, cfg.clock_cy
        R      = cfg.clock_radius
        force  = torch.zeros_like(self.ant_vel)

        for state_val in (1, 2, 3, 4):
            mask = self.ant_state == state_val
            if not mask.any():
                continue
            idxs = mask.nonzero(as_tuple=True)[0]

            if state_val == 1:   # surface foraging
                sy = cfg.surface_y
                pull_y = (sy - 20) - self.ant_pos[idxs, 1]
                force[idxs, 1] += pull_y * 0.04
                force[idxs]    += torch.randn(idxs.numel(), 2, device=dev) * 1.8
                self.total_leaf_trips += int(
                    (self.ant_pos[idxs, 1] < sy).sum().item()
                ) // 12

            elif state_val == 2:  # returning to nest
                pull = self.nest - self.ant_pos[idxs]
                dist = torch.norm(pull, dim=1, keepdim=True).clamp(min=1e-5)
                force[idxs] += (pull / dist) * 3.5
                arrived = dist.squeeze(1) < 28
                if arrived.any():
                    self.ant_state[idxs[arrived]] = 0
                    self.total_leaf_trips += int(arrived.sum().item())

            elif state_val == 3:  # deep culture cluster
                deep_nest = self.nest.clone()
                deep_nest[1] = deep_nest[1] + depth * 55.0
                pull = deep_nest - self.ant_pos[idxs]
                dist = torch.norm(pull, dim=1, keepdim=True).clamp(min=1e-5)
                force[idxs] += (pull / dist) * 2.0
                force[idxs] += torch.randn(idxs.numel(), 2, device=dev) * 0.5

            elif state_val == 4:  # alate nuptial spiral
                ang = torch.atan2(
                    self.ant_pos[idxs, 1] - cy,
                    self.ant_pos[idxs, 0] - cx,
                )
                force[idxs, 0] += -torch.sin(ang) * 2.2 + (self.ant_pos[idxs, 0] - cx) * 0.018
                force[idxs, 1] +=  torch.cos(ang) * 2.2 + (self.ant_pos[idxs, 1] - cy) * 0.018
                force[idxs]    += torch.randn(idxs.numel(), 2, device=dev) * 0.9
                if nuptial > 0.5:
                    self.total_nuptial_events += max(0, idxs.numel() // 15)

        self.ant_vel = self.ant_vel * 0.80 + force * 0.20
        v_norm = torch.norm(self.ant_vel, dim=1, keepdim=True).clamp(min=1e-5)
        self.ant_vel = (self.ant_vel / v_norm) * 2.5
        alate = self.ant_state == 4
        if alate.any():
            self.ant_vel[alate] *= 1.8
        self.ant_pos += self.ant_vel

        # Clamp all ants to clock arena (alates wrap back if too far)
        dx = self.ant_pos[:, 0] - cx
        dy = self.ant_pos[:, 1] - cy
        dr = torch.sqrt(dx**2 + dy**2).clamp(min=1e-5)
        out = dr > (R - 8)
        non_alate_out = out & (~alate)
        if non_alate_out.any():
            sc = (R - 10) / dr[non_alate_out]
            self.ant_pos[non_alate_out, 0] = cx + dx[non_alate_out] * sc
            self.ant_pos[non_alate_out, 1] = cy + dy[non_alate_out] * sc
        alate_out = out & alate
        if alate_out.any():
            reset = alate_out.nonzero(as_tuple=True)[0]
            self.ant_pos[reset, 0] = cfg.nest_cx + (torch.rand(reset.numel(), device=dev) - 0.5) * 18
            self.ant_pos[reset, 1] = cfg.nest_cy + (torch.rand(reset.numel(), device=dev) - 0.5) * 18

        self.hist_ant.append(self.ant_pos.cpu().numpy().copy())


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class SauvaRenderer:

    def __init__(self, cfg: SauvaConfig, sim: SauvaSim):
        self.cfg = cfg
        self.sim = sim

    def _arc(self, cx, cy, r, start_m, end_m, sweep=1):
        """SVG arc between two month positions (sweep=1 clockwise in SVG)."""
        a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
        a2 = (end_m   / 12) * 2 * math.pi - math.pi / 2
        x1 = cx + math.cos(a1) * r;  y1 = cy + math.sin(a1) * r
        x2 = cx + math.cos(a2) * r;  y2 = cy + math.sin(a2) * r
        span  = abs(end_m - start_m) % 12
        large = 1 if span > 6 else 0
        return f"M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} {sweep} {x2:.1f} {y2:.1f}"

    def generate_svg(self) -> str:
        cfg  = self.cfg
        sim  = self.sim
        w, h = cfg.width, cfg.height
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R    = cfg.clock_radius
        F    = cfg.frames
        dur  = F / cfg.fps

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:#0d0a05; font-family:system-ui,-apple-system,sans-serif;">'
        ]

        # -- Defs ---------------------------------------------------------------
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="soilBg80">'
            '<stop offset="0%"   stop-color="#2e1b0f" stop-opacity="0.95"/>'
            '<stop offset="65%"  stop-color="#1a0e07" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#0d0a05" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="fungusGlow80">'
            '<stop offset="0%"   stop-color="#f8bbd0" stop-opacity="0.90"/>'
            '<stop offset="55%"  stop-color="#f48fb1" stop-opacity="0.40"/>'
            '<stop offset="100%" stop-color="#e91e63" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="alateGlow80">'
            '<stop offset="0%"   stop-color="#ffb74d" stop-opacity="0.80"/>'
            '<stop offset="100%" stop-color="#ff6d00" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # -- Background ---------------------------------------------------------
        svg.append(f'<rect width="{w}" height="{h}" fill="#0d0a05"/>')
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R + 55}" fill="url(#soilBg80)"/>'
        )

        # Surface / subterranean split line inside clock arena
        sy = cfg.surface_y
        half_x = math.sqrt(max(0.0, R**2 - (sy - cy)**2))
        svg.append(
            f'<line x1="{cx - half_x:.0f}" y1="{sy:.0f}" '
            f'x2="{cx + half_x:.0f}" y2="{sy:.0f}" '
            f'stroke="#5d4037" stroke-width="2" stroke-dasharray="8,4" opacity="0.55"/>'
        )
        svg.append(
            f'<text font-weight="bold" x="{cx - half_x + 6:.0f}" y="{sy - 5:.0f}" '
            f'font-size="15" fill="#8d6e63">Superficie</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="{cx - half_x + 6:.0f}" y="{sy + 15:.0f}" '
            f'font-size="15" fill="#a1887f">Subterraneo</text>'
        )

        # Wet / foraging season amber glow
        wet_fills = ";".join(
            f"rgba(255,183,77,{sim._interp(FORAGING_SURFACE, (f/F)*12) * 0.16:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{wet_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )
        # Nuptial flight orange burst
        nupt_fills = ";".join(
            f"rgba(255,109,0,{sim._interp(NUPTIAL_FLIGHT, (f/F)*12) * 0.24:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{nupt_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # -- Title --------------------------------------------------------------
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ffb74d" font-weight="bold">'
            f'ECO-SIM: Sau\u00edva \u00d7 Leucoagaricus \u2014 Seasonal Subterranean Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#ffa726">'
            f'Leaf-cutter ant fungus-garden seasonal contraction | </text>'
        )

        # -- Nest chamber -------------------------------------------------------
        nx, ny = cfg.nest_cx, cfg.nest_cy
        svg.append(
            f'<ellipse cx="{nx:.0f}" cy="{ny:.0f}" rx="82" ry="56" fill="#1c0a03" opacity="0.82"/>'
        )

        # Fungus garden: pulsing pink blob that migrates down in dry season
        nest_y_vals = ";".join(
            f"{ny + sim.hist_depth[fi] * 38:.0f}" for fi in range(F)
        )
        fungus_r = ";".join(
            f"{28 + sim.hist_fungus[fi] * 30:.1f}" for fi in range(F)
        )
        fungus_op = ";".join(
            f"{0.55 + sim.hist_fungus[fi] * 0.40:.2f}" for fi in range(F)
        )
        svg.append(
            f'<circle fill="url(#fungusGlow80)" cx="{nx:.0f}">'
            f'<animate attributeName="cy" values="{nest_y_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="r"  values="{fungus_r}"  dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="{fungus_op}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</circle>'
        )

        # -- Ants ---------------------------------------------------------------
        for i in range(cfg.num_ants):
            px_v = ";".join(f"{p[i, 0]:.1f}" for p in sim.hist_ant)
            py_v = ";".join(f"{p[i, 1]:.1f}" for p in sim.hist_ant)
            is_alate = i >= cfg.num_ants - int(cfg.num_ants * 0.15)
            col  = "#ffb74d" if is_alate else "#ff5722"
            r_pt = 3.0 if is_alate else 2.5
            svg.append(
                f'<circle r="{r_pt}" fill="{col}" opacity="0.85">'
                f'<animate attributeName="cx" values="{px_v}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{py_v}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # -- Clock Face: month ring ---------------------------------------------
        months = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
        month_cols = {
            0: "#ffb74d", 1: "#ffb74d",    # Jan-Feb: wet, high foraging
            2: "#ffa726", 3: "#a1887f",    # Mar-Apr: wet ending, drying
            4: "#8d6e63", 5: "#6d4c41",    # May-Jun: dry
            6: "#5d4037", 7: "#795548",    # Jul-Aug: dry nadir
            8: "#ff7043", 9: "#ff6d00",    # Sep-Oct: nuptial flights!
            10: "#ffb74d", 11: "#ffcc80",  # Nov-Dec: wet return
        }
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="{month_cols[i]}" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 6)
            ly2 = cy + math.sin(angle) * (R - 6)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#2e1b0f" stroke-width="2"/>'
            )

        # -- Season Arcs --------------------------------------------------------
        # Wet season / max foraging (Oct-Apr, months 9-14.8, clockwise through top via JAN)
        d_wet = self._arc(cx, cy, R + 10, 9, 14.8)
        svg.append(
            f'<path d="{d_wet}" fill="none" stroke="#ffb74d" stroke-width="9" '
            f'stroke-linecap="round" opacity="0.52"/>'
        )
        mid_wet = ((9 + 14.8) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(
            f'<text font-weight="bold" x="{cx + math.cos(mid_wet) * (R + 22):.0f}" '
            f'y="{cy + math.sin(mid_wet) * (R + 22):.0f}" font-size="15" '
            f'fill="#ffb74d" text-anchor="middle">\U0001f33f Forrageamento</text>'
        )

        # Dry season / deep-culture retreat (Apr-Sep, months 3-9, through bottom via JUL)
        d_dry = self._arc(cx, cy, R + 10, 3, 9)
        svg.append(
            f'<path d="{d_dry}" fill="none" stroke="#5d4037" stroke-width="7" '
            f'stroke-linecap="round" opacity="0.50" stroke-dasharray="5,4"/>'
        )
        mid_dry = ((3 + 9) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(
            f'<text font-weight="bold" x="{cx + math.cos(mid_dry) * (R + 22):.0f}" '
            f'y="{cy + math.sin(mid_dry) * (R + 22):.0f}" font-size="15" '
            f'fill="#8d6e63" text-anchor="middle">\u2600 Seca / Profundo</text>'
        )

        # Nuptial flight pulse (Aug-Nov, months 7.5-11)
        d_nup = self._arc(cx, cy, R + 26, 7.5, 11)
        svg.append(
            f'<path d="{d_nup}" fill="none" stroke="#ff6d00" stroke-width="8" '
            f'stroke-linecap="round" opacity="0.68"/>'
        )
        mid_np = ((7.5 + 11) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(
            f'<text font-weight="bold" x="{cx + math.cos(mid_np) * (R + 36):.0f}" '
            f'y="{cy + math.sin(mid_np) * (R + 36):.0f}" font-size="15" '
            f'fill="#ff6d00" text-anchor="middle">\U0001f41e Revoada Alada</text>'
        )

        # Clock ring base
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R}" fill="none" '
            f'stroke="#2e1b0f" stroke-width="1.5" opacity="0.8"/>'
        )

        # -- Clock Hand ---------------------------------------------------------
        hand_x = ";".join(
            f"{cx + math.cos((m / 12) * 2 * math.pi - math.pi / 2) * (R - 10):.1f}"
            for m in sim.hist_month
        )
        hand_y = ";".join(
            f"{cy + math.sin((m / 12) * 2 * math.pi - math.pi / 2) * (R - 10):.1f}"
            for m in sim.hist_month
        )
        svg.append(
            f'<line x1="{cx:.0f}" y1="{cy:.0f}" x2="{cx:.0f}" y2="{cy - R + 10:.0f}" '
            f'stroke="#ffb74d" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="14" fill="#0d0a05" '
            f'stroke="#ffb74d" stroke-width="2"/>'
        )
        svg.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="4" fill="#ff8f00"/>')

        # ===========================================================================
        # RIGHT INFO PANELS
        # ===========================================================================
        panel_x = w - 420
        panel_w = 400

        # -- Panel 1: Ecological Logic -------------------------------------------
        py1, ph1 = 20, 262
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(
            f'<rect width="{panel_w}" height="{ph1}" fill="#150b04" rx="8" '
            f'stroke="#5d4037" stroke-width="1" opacity="0.95"/>'
        )
        svg.append(
            f'<text x="12" y="22" fill="#ffb74d" font-size="15" font-weight="bold">'
            f'\U0001f41c Subterranean Symbiosis Phenological Logic</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
            f'Sauva (Atta spp.) colonies farm Leucoagaricus fungus on</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
            f'freshly cut leaves. The garden requires precise humidity --</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
            f'colonies retreat DEEPER in dry season (Apr-Sep) to protect it.</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
            f'Wet season: maximum foraging, rapid fungus growth, CO2 cycling.</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="98" fill="#ff7043" font-size="15">'
            f'Nuptial flights (revoada alada) triggered by first rains (Sep-Oct):</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="112" fill="#ff7043" font-size="15">'
            f'virgin queens carry fungal mycelia to found new nests.</text>'
        )
        svg.append('</g>')

        # -- Panel 2: Simulation Metrics ----------------------------------------
        py2 = py1 + ph1 + 10
        ph2 = 130
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(
            f'<rect width="{panel_w}" height="{ph2}" fill="#150b04" rx="8" '
            f'stroke="#37474f" stroke-width="1" opacity="0.95"/>'
        )
        svg.append(
            f'<text x="12" y="22" fill="#90a4ae" font-size="15" font-weight="bold">'
            f'\U0001f4ca Colony Activity Metrics</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="44" fill="#ffb74d" font-size="15">'
            f'Leaf Transport Trips (sim.): {sim.total_leaf_trips:,}</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="62" fill="#ff7043" font-size="15">'
            f'Nuptial Alate Flight Events: {sim.total_nuptial_events:,}</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="80" fill="#90a4ae" font-size="15">'
            f'Peak surface foragers: {sim.peak_foragers} / {cfg.num_ants}</text>'
        )
        svg.append('</g>')

        # -- Panel 3: Phenology Chart (eco_base helper) -------------------------
        py3 = py2 + ph2 + 10
        ph3 = 165
        curves_data = [
            (FORAGING_SURFACE,    "#ff5722", "Surface Foraging"),
            (NUPTIAL_FLIGHT,      "#ff6d00", "Nuptial Flight"),
            (FUNGUS_PRODUCTIVITY, "#f48fb1", "Fungus Productivity"),
            (COLONY_DEPTH,        "#8d6e63", "Colony Depth (retreat)"),
            (RAINFALL_CURVE,      "#29b6f6", "Rainfall"),
        ]
        chart_snip = draw_phenology_chart(
            curves_data,
            chart_w=360, chart_h=78, panel_h=ph3,
            title="\U0001f4c8 Phenological Curves \u2014 nb80 Sau\u00edva \u00d7 Leucoagaricus",
            title_color="#ffb74d",
            bg_color="#150b04",
            border_color="#5d4037",
        )
        svg.append(f'<g transform="translate({panel_x}, {py3})">{chart_snip}</g>')

        # -- Current Month Sidebar (animated) -----------------------------------
        px5 = 20;  py5 = h - 232;  pw5 = 255; ph5 = 222
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(
            f'<rect width="{pw5}" height="{ph5}" fill="#150b04" rx="8" '
            f'stroke="#5d4037" stroke-width="1.5" opacity="0.97"/>'
        )
        svg.append(
            f'<text x="12" y="22" font-size="15" fill="#ffb74d" font-weight="bold">'
            f'Status M\u00eas Atual:</text>'
        )

        month_names = [
            "January", "February", "Mar\u00e7o", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12
            vals[m_idx] = "1"
            op_str = ";".join(vals + ["0"])

            svg.append(
                f'<text x="12" y="50" font-size="15" fill="#ffb74d" font-weight="bold">'
            )
            svg.append(m_name)
            svg.append(
                f'<animate attributeName="opacity" values="{op_str}" '
                f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>'
                f'</text>'
            )

            fo = sim._interp(FORAGING_SURFACE,    float(m_idx))
            nu = sim._interp(NUPTIAL_FLIGHT,      float(m_idx))
            fg = sim._interp(FUNGUS_PRODUCTIVITY, float(m_idx))
            dp = sim._interp(COLONY_DEPTH,        float(m_idx))
            rn = sim._interp(RAINFALL_CURVE,      float(m_idx))

            if nu > 0.5:
                st1, c1 = "\U0001f41e REVOADA ALADA -- Nuptial!",    "#ff6d00"
                st2, c2 = f"Alate emergence: {nu*100:.0f}%",         "#ffb74d"
                st3, c3 = "Queens carry fungus mycelia",              "#f48fb1"
            elif fo > 0.7:
                st1, c1 = "\U0001f33f PEAK SURFACE FORAGING",        "#ffb74d"
                st2, c2 = f"Foraging: {fo*100:.0f}% | Fungus: {fg*100:.0f}%", "#ff5722"
                st3, c3 = f"Rainfall: {rn*100:.0f}%",                "#29b6f6"
            elif dp > 0.75:
                st1, c1 = "\u2193 DEEP RETREAT -- Dry Season",       "#8d6e63"
                st2, c2 = f"Colony depth: {dp*100:.0f}%",            "#a1887f"
                st3, c3 = f"Fungus protected: {fg*100:.0f}%",        "#f48fb1"
            elif fo < 0.35:
                st1, c1 = "\u2600 SECA -- Foraging Contracted",      "#795548"
                st2, c2 = f"Surface: {fo*100:.0f}% active",          "#a1887f"
                st3, c3 = f"Fungus management: {fg*100:.0f}%",       "#f8bbd0"
            else:
                st1, c1 = "\U0001f6a7 SEASONAL TRANSITION",          "#ffa726"
                st2, c2 = f"Foraging: {fo*100:.0f}%",                "#ffb74d"
                st3, c3 = f"Depth shift: {dp*100:.0f}%",             "#8d6e63"

            for y_off, txt, col in ((76, st1, c1), (96, st2, c2), (114, st3, c3)):
                svg.append(
                    f'<text x="12" y="{y_off}" font-size="15" fill="{col}" font-weight="bold">'
                )
                svg.append(txt)
                svg.append(
                    f'<animate attributeName="opacity" values="{op_str}" '
                    f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>'
                    f'</text>'
                )

        svg.append(
            '<text x="12" y="138" fill="#546e7a" font-size="15" font-weight="bold">Legenda:</text>'
        )
        legend = [
            (22, 154, "#ff5722", "Sauva (Atta spp.) -- forager"),
            (22, 170, "#ffb74d", "Alate (nuptial flight queen)"),
            (22, 186, "#f8bbd0", "Leucoagaricus (Fungus Garden)"),
            (22, 202, "#29b6f6", "Chuvas / wet season"),
            (22, 218, "#8d6e63", "Dry season / deep retreat"),
        ]
        for (ex, ey, ec, elabel) in legend:
            svg.append(
                f'<ellipse cx="{ex}" cy="{ey}" rx="6" ry="2.5" fill="{ec}" opacity="0.9"/>'
            )
            svg.append(f'<text font-weight="bold" x="36" y="{ey+4}" fill="{ec}" font-size="15">{elabel}</text>')
        svg.append('</g>')

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(
        f" \u2014 Sau\u00edva \u00d7 Leucoagaricus Seasonal Symbiosis Clock on "
        f"{CONFIG.device}..."
    )

    sim = SauvaSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(
        f"Done: {sim.total_leaf_trips:,} leaf trips, "
        f"{sim.total_nuptial_events:,} nuptial events, "
        f"peak {sim.peak_foragers} surface foragers."
    )
    print("Generating SVG...")
    renderer = SauvaRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg(svg_content, "notebook_80")
    return svg_content


if __name__ == "__main__":
    main()
