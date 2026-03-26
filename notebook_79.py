# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 79: Suiriri (Tyrannus melancholicus) <-> Murici (Byrsonima crassifolia)
#            Wet-Season Frugivory & Seed Dispersal Phenological Clock
# INTERVENTION: Seasonal / Migratory Cerrado Dynamics -- Passage of Months Series
# ===================================================================================================
"""
notebook_79.py -- Suiriri x Murici: Wet-Season Frugivory Clock
Notebook Differentiation:
- Differentiation Focus: Seasonal/Migratory Cerrado Dynamics emphasizing wet-season resource pulses.
- Indicator species: Caititu (Pecari tajacu).
- Pollination lens: floral resource concentration in veredas.
- Human impact lens: carbon stock monitoring incentives.

The Tropical Kingbird / Suiriri (Tyrannus melancholicus) is an aerial insectivore
and facultative frugivore whose activity peaks during the Cerrado wet season
(October -- March), when Murici (Byrsonima crassifolia) trees reach maximum fruit
production. The phenological synchrony between bird presence and Murici fruiting
is a key ornithochory seed-dispersal service across open cerrado and ecotones.

Phenological dynamics mapped on the annual clock:
  * Monthly Murici fruiting intensity (peak: Aug-Nov, dry-to-wet transition)
  * Suiriri presence/foraging activity in RESEX (peak: Oct-Feb)
  * Rainfall backbone (Oct-Mar wet season)
  * Seed dispersal probability (= fruiting x bird activity)
  * Sapling recruitment pulse (Jan-Mar -- peak soil moisture)

Cross-references within the Passage of Months series:
  * nb74: Andorinha migration clock -- shares wet-season insect bloom backbone.
  * nb57: Lobo-guara x Lobeira endozoochory -- parallel fruit-disperser synchrony.
  * nb60: Arara-caninde x Buriti -- vereda fruiting calendar comparison.
  * nb80: Sauva x Leucoagaricus -- complementary dry-season colony stress story.

Scientific References:
  - Melo, F.P.L. et al. (2009). Frugivory and seed dispersal in Cerrado. Biotropica.
  - Pougy et al. (2015). Murici (Byrsonima crassifolia) phenology across biomes. Flora.
  - Pizo, M.A. & Oliveira, P.S. (2001). The use of fruits by birds in cerrado. Biotropica.
  - Jordano, P. (2000). Fruits and frugivory. Seeds: The Ecology of Regeneration.
  - PIGT RESEX Recanto das Araras observations, Goias (2022-2024).

Scientific Relevance (PIGT  -- 2024):
  - Integrates the socio-environmental complexity of 
    de Cima, Goias, Brazil.
  - Models landscape connectivity, karst vulnerability (Bacia do Rio Lapa),
    biological corridors, and seed-dispersal networks.
  - Demonstrates wet-season frugivory-dispersal synchrony critical for Cerrado
    ecosystem regeneration under climate stress.
  - SVG artefacts archived at:
    https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing
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

# Suiriri (Tyrannus melancholicus) foraging activity index in RESEX
SUIRIRI_ACTIVITY = [0.85, 0.90, 0.80, 0.55, 0.30, 0.15, 0.10, 0.20, 0.45, 0.80, 0.95, 0.90]

# Murici (Byrsonima crassifolia) fruiting intensity -- peak Aug-Nov (months 7-10)
MURICI_FRUITING = [0.25, 0.15, 0.10, 0.08, 0.12, 0.18, 0.38, 0.72, 0.95, 1.00, 0.78, 0.42]

# Seed dispersal effectiveness (fruiting x bird activity)
DISPERSAL_PROB = [0.30, 0.20, 0.12, 0.07, 0.05, 0.05, 0.10, 0.25, 0.58, 0.92, 0.85, 0.52]

# Sapling recruitment (moisture x dispersal lag ~2 months)
SAPLING_RECRUIT = [0.72, 0.82, 0.65, 0.28, 0.08, 0.02, 0.00, 0.04, 0.18, 0.32, 0.42, 0.55]

# Rainfall -- shared wet-season backbone (Oct-Mar)
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]


@dataclass
class SuiririConfig:
    width:  int   = 1280
    height: int = CANVAS_HEIGHT
    frames: int   = 360
    fps:    int   = 10
    device: str   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry -- consistent with nb71-74 series
    clock_cx:     float = 390.0
    clock_cy:     float = 295.0
    clock_radius: float = 230.0

    num_birds:    int   = 35
    num_trees:    int   = 6
    bird_speed:   float = 5.0


CONFIG = SuiririConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class SuiririSim:

    def __init__(self, cfg: SuiririConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # Trees scattered inside the clock arena
        self.tree_pos = torch.zeros((cfg.num_trees, 2), device=self.dev)
        random.seed(42)
        for i in range(cfg.num_trees):
            angle = (i / cfg.num_trees) * 2 * math.pi + random.uniform(-0.35, 0.35)
            r = random.uniform(R * 0.28, R * 0.70)
            self.tree_pos[i, 0] = cx + math.cos(angle) * r
            self.tree_pos[i, 1] = cy + math.sin(angle) * r

        # Birds start distributed inside the clock
        torch.manual_seed(7)
        angles_b = torch.rand(cfg.num_birds, device=self.dev) * 2 * math.pi
        radii_b  = torch.rand(cfg.num_birds, device=self.dev) * (R - 25)
        self.bird_pos = torch.stack([
            cx + torch.cos(angles_b) * radii_b,
            cy + torch.sin(angles_b) * radii_b,
        ], dim=1)
        self.bird_vel = torch.randn((cfg.num_birds, 2), device=self.dev)

        # 0: absent/low-activity  1: fruiting-tree foraging  2: aerial insect hunting
        self.bird_state = torch.zeros(cfg.num_birds, dtype=torch.long, device=self.dev)

        self.hist_bird:       List = []
        self.hist_tree_fruit: List = []
        self.hist_month:      List = []

        self.total_dispersals  = 0
        self.total_saplings    = 0
        self.peak_active_birds = 0

    def _interp(self, curve: list, month_frac: float) -> float:
        m  = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t  = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, fi: int):
        cfg = self.cfg
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        month_frac = (fi / cfg.frames) * 12.0
        self.hist_month.append(month_frac)

        bird_act  = self._interp(SUIRIRI_ACTIVITY, month_frac)
        fruit_lvl = self._interp(MURICI_FRUITING,  month_frac)
        rain      = self._interp(RAINFALL_CURVE,   month_frac)

        # Active bird count scales with seasonal activity
        num_active = max(1, int(cfg.num_birds * bird_act))
        active_mask = torch.zeros(cfg.num_birds, dtype=torch.bool, device=self.dev)
        active_mask[:num_active] = True
        if num_active > self.peak_active_birds:
            self.peak_active_birds = num_active

        # Assign foraging mode by season
        self.bird_state[:] = 0
        if fruit_lvl > 0.35:
            self.bird_state[active_mask] = 1   # fruit foraging
        else:
            self.bird_state[active_mask] = 2   # aerial insect hunting

        # Move absent birds off-screen
        self.bird_pos[~active_mask, 0] = cx - R - 80

        force = torch.zeros_like(self.bird_vel)
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        if active_idx.numel() > 0:
            ap      = self.bird_pos[active_idx]
            state_a = self.bird_state[active_idx]

            # State 1: seek nearest fruiting tree
            fr_mask = state_a == 1
            if fr_mask.any():
                fb_pos   = ap[fr_mask]
                dists    = torch.cdist(fb_pos, self.tree_pos)
                min_d, nearest = torch.min(dists, dim=1)
                tgt  = self.tree_pos[nearest]
                pull = tgt - fb_pos
                plen = torch.norm(pull, dim=1, keepdim=True).clamp(min=1e-5)
                force[active_idx[fr_mask]] += (pull / plen) * 3.5
                close = min_d < 22
                if close.any():
                    n_disp = int(close.sum().item())
                    self.total_dispersals += n_disp
                    if rain > 0.25:
                        self.total_saplings += max(0, int(n_disp * rain * 0.25))

            # State 2: aerial patrol with center pull
            fl_mask = state_a == 2
            if fl_mask.any():
                center = torch.tensor([cx, cy], device=self.dev, dtype=torch.float32)
                fb2    = ap[fl_mask]
                pull2  = center - fb2
                plen2  = torch.norm(pull2, dim=1, keepdim=True).clamp(min=1e-5)
                force[active_idx[fl_mask]] += (pull2 / plen2) * 0.7

            force[active_idx] += torch.randn(active_idx.numel(), 2, device=self.dev) * 1.6

        # Velocity integration
        self.bird_vel = self.bird_vel * 0.74 + force * 0.26
        v_norm = torch.norm(self.bird_vel, dim=1, keepdim=True).clamp(min=1e-5)
        speed  = torch.full((cfg.num_birds,), cfg.bird_speed, device=self.dev)
        speed[~active_mask] = 0.0
        speed[self.bird_state == 2] *= 1.4
        self.bird_vel = (self.bird_vel / v_norm) * speed.unsqueeze(1)
        self.bird_pos += self.bird_vel

        # Clamp active birds to clock arena
        dx = self.bird_pos[:, 0] - cx
        dy = self.bird_pos[:, 1] - cy
        dr = torch.sqrt(dx**2 + dy**2).clamp(min=1e-5)
        out = (dr > R - 10) & active_mask
        if out.any():
            sc = (R - 12) / dr[out]
            self.bird_pos[out, 0] = cx + dx[out] * sc
            self.bird_pos[out, 1] = cy + dy[out] * sc

        self.hist_tree_fruit.append([4.0 + fruit_lvl * 14.0] * cfg.num_trees)
        self.hist_bird.append(self.bird_pos.cpu().numpy().copy())


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class SuiririRenderer:

    def __init__(self, cfg: SuiririConfig, sim: SuiririSim):
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
            f'style="background-color:#030f07; font-family:system-ui,-apple-system,sans-serif;">'
        ]

        # -- Defs ---------------------------------------------------------------
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="cerradoBg79">'
            '<stop offset="0%"   stop-color="#0d2914" stop-opacity="0.95"/>'
            '<stop offset="70%"  stop-color="#081a09" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#030f07" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="fruitGlow79">'
            '<stop offset="0%"   stop-color="#ffd54f" stop-opacity="0.85"/>'
            '<stop offset="60%"  stop-color="#ffb300" stop-opacity="0.30"/>'
            '<stop offset="100%" stop-color="#ff8f00" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # -- Background ----------------------------------------------------------
        svg.append(f'<rect width="{w}" height="{h}" fill="#030f07"/>')
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R + 55}" fill="url(#cerradoBg79)"/>'
        )

        # Fruiting-season amber glow driven by Murici
        fruit_fills = ";".join(
            f"rgba(255,213,79,{sim._interp(MURICI_FRUITING, (f/F)*12) * 0.22:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{fruit_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )
        # Rainfall blue glow
        rain_fills = ";".join(
            f"rgba(41,182,246,{sim._interp(RAINFALL_CURVE, (f/F)*12) * 0.13:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{rain_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # -- Title ---------------------------------------------------------------
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#a5d6a7" font-weight="bold">'
            f'ECO-SIM: Suiriri \u00d7 Murici \u2014 Wet-Season Frugivory Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#81c784">'
            f'Seed dispersal synchrony locked to Cerrado rainfall | </text>'
        )

        # -- Trees inside clock arena -------------------------------------------
        tree_np = sim.tree_pos.cpu().numpy()
        for i in range(cfg.num_trees):
            tx, ty = float(tree_np[i, 0]), float(tree_np[i, 1])
            svg.append(
                f'<line x1="{tx:.0f}" y1="{ty+4:.0f}" x2="{tx:.0f}" y2="{ty+28:.0f}" '
                f'stroke="#5d4037" stroke-width="5" stroke-linecap="round"/>'
            )
            svg.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="26" fill="#2e7d32" opacity="0.60"/>'
            )
            svg.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="16" fill="#388e3c" opacity="0.45"/>'
            )
            r_v = ";".join(f"{fs[i]:.1f}" for fs in sim.hist_tree_fruit)
            for ox, oy in ((10, -10), (-11, 5), (5, 14), (-7, -18)):
                svg.append(
                    f'<circle cx="{tx+ox:.0f}" cy="{ty+oy:.0f}" fill="#ffd54f">'
                    f'<animate attributeName="r" values="{r_v}" dur="{dur}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )

        # -- Birds ---------------------------------------------------------------
        for i in range(cfg.num_birds):
            px_v = ";".join(f"{p[i, 0]:.1f}" for p in sim.hist_bird)
            py_v = ";".join(f"{p[i, 1]:.1f}" for p in sim.hist_bird)
            op_v = ";".join(
                "1.0" if p[i, 0] > (cx - R - 40) else "0.0"
                for p in sim.hist_bird
            )
            svg.append(
                f'<ellipse rx="5" ry="2.5" fill="#ffee58" stroke="#fff" stroke-width="0.7">'
                f'<animate attributeName="cx" values="{px_v}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{py_v}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{op_v}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            cap_px = ";".join(f"{p[i, 0]+3:.1f}" for p in sim.hist_bird)
            svg.append(
                f'<ellipse rx="2.5" ry="2" fill="#212121">'
                f'<animate attributeName="cx" values="{cap_px}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{py_v}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{op_v}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )

        # -- Clock Face: month ring ---------------------------------------------
        months = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
        month_cols = {
            0: "#a5d6a7", 1: "#a5d6a7",   # Jan-Feb: wet + sapling flush
            2: "#81c784", 3: "#66bb6a",   # Mar-Apr: wet ending
            4: "#78909c", 5: "#546e7a",   # May-Jun: dry onset
            6: "#546e7a", 7: "#ffd54f",   # Jul-Aug: dry; Murici starts
            8: "#ffb300", 9: "#ff8f00",   # Sep-Oct: peak fruiting + dispersal
            10: "#a5d6a7", 11: "#66bb6a", # Nov-Dec: bird arrival
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
                f'stroke="#1b3a20" stroke-width="2"/>'
            )

        # -- Season Arcs --------------------------------------------------------
        # Murici fruiting peak (Aug-Nov, months 7-11), clockwise lower-left arc
        d_murici = self._arc(cx, cy, R + 10, 7, 11)
        svg.append(
            f'<path d="{d_murici}" fill="none" stroke="#ffd54f" stroke-width="9" '
            f'stroke-linecap="round" opacity="0.60"/>'
        )
        mid_fr = ((7 + 11) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(
            f'<text font-weight="bold" x="{cx + math.cos(mid_fr) * (R + 22):.0f}" '
            f'y="{cy + math.sin(mid_fr) * (R + 22):.0f}" font-size="15" '
            f'fill="#ffd54f" text-anchor="middle">Murici Fruiting</text>'
        )

        # Wet season (Oct-Apr, months 9-14.8: clockwise through top via JAN)
        d_wet = self._arc(cx, cy, R + 10, 9, 14.8)
        svg.append(
            f'<path d="{d_wet}" fill="none" stroke="#29b6f6" stroke-width="7" '
            f'stroke-linecap="round" opacity="0.42" stroke-dasharray="6,4"/>'
        )
        mid_wet = ((9 + 14.8) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(
            f'<text font-weight="bold" x="{cx + math.cos(mid_wet) * (R + 22):.0f}" '
            f'y="{cy + math.sin(mid_wet) * (R + 22):.0f}" font-size="15" '
            f'fill="#29b6f6" text-anchor="middle">\u2602 Chuvas</text>'
        )

        # Dry season (Apr-Sep, months 3-9, clockwise through bottom via JUL)
        d_dry = self._arc(cx, cy, R + 26, 3, 9)
        svg.append(
            f'<path d="{d_dry}" fill="none" stroke="#78909c" stroke-width="5" '
            f'stroke-linecap="round" opacity="0.48" stroke-dasharray="4,4"/>'
        )
        mid_dry = ((3 + 9) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(
            f'<text font-weight="bold" x="{cx + math.cos(mid_dry) * (R + 36):.0f}" '
            f'y="{cy + math.sin(mid_dry) * (R + 36):.0f}" font-size="15" '
            f'fill="#78909c" text-anchor="middle">\u2600 Seca</text>'
        )

        # Sapling recruitment burst (Jan-Mar, months 0-3)
        d_sap = self._arc(cx, cy, R + 26, 0, 3)
        svg.append(
            f'<path d="{d_sap}" fill="none" stroke="#a5d6a7" stroke-width="6" '
            f'stroke-linecap="round" opacity="0.62"/>'
        )
        mid_sap = (1.5 / 12) * 2 * math.pi - math.pi / 2
        svg.append(
            f'<text font-weight="bold" x="{cx + math.cos(mid_sap) * (R + 36):.0f}" '
            f'y="{cy + math.sin(mid_sap) * (R + 36):.0f}" font-size="15" '
            f'fill="#a5d6a7" text-anchor="middle">\U0001f331 Plantulas</text>'
        )

        # Clock ring base
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{R}" fill="none" '
            f'stroke="#1b3a20" stroke-width="1.5" opacity="0.8"/>'
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
            f'stroke="#a5d6a7" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="14" fill="#030f07" '
            f'stroke="#a5d6a7" stroke-width="2"/>'
        )
        svg.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="4" fill="#66bb6a"/>')

        # ===========================================================================
        # RIGHT INFO PANELS
        # ===========================================================================
        panel_x = w - 420
        panel_w = 400

        # -- Panel 1: Ecological Logic -------------------------------------------
        py1, ph1 = 20, 222
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(
            f'<rect width="{panel_w}" height="{ph1}" fill="#041208" rx="8" '
            f'stroke="#2e7d32" stroke-width="1" opacity="0.95"/>'
        )
        svg.append(
            f'<text x="12" y="22" fill="#a5d6a7" font-size="15" font-weight="bold">'
            f'\U0001f426 Frugivory-Dispersal Phenological Logic</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
            f'Suiriri (Tyrannus melancholicus) peaks</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
            f'Oct-Feb, when insects and Murici</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
            f'fruits are most abundant in the</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
            f'cerrado matrix. Murici (Byrsonima</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
            f'crassifolia) fruits Aug-Nov,</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="112" fill="#ccc" font-size="15">'
            f'bridging the dry-to-wet transition</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="126" fill="#ccc" font-size="15">'
            f'and attracting a diverse frugivore</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="140" fill="#ffd54f" font-size="15">'
            f'guild. Dispersal peak: Sep-Oct --</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="154" fill="#ffd54f" font-size="15">'
            f'bird activity x fruit availability.</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="168" fill="#a5d6a7" font-size="15">'
            f'Sapling recruitment peak: Jan-Mar</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="182" fill="#a5d6a7" font-size="15">'
            f'(maximum soil moisture).</text>'
        )
        svg.append('</g>')

        # -- Panel 2: Simulation Metrics ----------------------------------------
        py2 = py1 + ph1 + 10
        ph2 = 95
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(
            f'<rect width="{panel_w}" height="{ph2}" fill="#041208" rx="8" '
            f'stroke="#37474f" stroke-width="1" opacity="0.95"/>'
        )
        svg.append(
            f'<text x="12" y="22" fill="#90a4ae" font-size="15" font-weight="bold">'
            f'\U0001f4ca Dispersal & Recruitment Metrics</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="44" fill="#a5d6a7" font-size="15">'
            f'Dispersal Events (sim.): {sim.total_dispersals:,}</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="62" fill="#81c784" font-size="15">'
            f'Sapling Recruitments: {sim.total_saplings:,}</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="12" y="80" fill="#90a4ae" font-size="15">'
            f'Peak birds active: {sim.peak_active_birds} / {cfg.num_birds}</text>'
        )
        svg.append('</g>')

        # -- Panel 3: Phenology Chart (eco_base helper) -------------------------
        py3 = py2 + ph2 + 10
        ph3 = 130
        curves_data = [
            (SUIRIRI_ACTIVITY, "#ffee58", "Suiriri Activity"),
            (MURICI_FRUITING,  "#ffd54f", "Murici Fruiting"),
            (DISPERSAL_PROB,   "#a5d6a7", "Dispersal Effectiveness"),
            (SAPLING_RECRUIT,  "#66bb6a", "Sapling Recruitment"),
            (RAINFALL_CURVE,   "#29b6f6", "Rainfall"),
        ]
        chart_snip = draw_phenology_chart(
            curves_data,
            chart_w=360, chart_h=58, panel_h=ph3,
            title="\U0001f4c8 Phenological Curves \u2014 nb79 Suiriri \u00d7 Murici",
            title_color="#a5d6a7",
            bg_color="#041208",
            border_color="#2e7d32",
        )
        svg.append(f'<g transform="translate({panel_x}, {py3})">{chart_snip}</g>')

        # -- Current Month Sidebar (animated) -----------------------------------
        px5 = 20;  py5 = h - 232;  pw5 = 255; ph5 = 222
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(
            f'<rect width="{pw5}" height="{ph5}" fill="#041208" rx="8" '
            f'stroke="#2e7d32" stroke-width="1.5" opacity="0.97"/>'
        )
        svg.append(
            f'<text x="12" y="22" font-size="15" fill="#a5d6a7" font-weight="bold">'
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
                f'<text x="12" y="50" font-size="15" fill="#a5d6a7" font-weight="bold">'
            )
            svg.append(m_name)
            svg.append(
                f'<animate attributeName="opacity" values="{op_str}" '
                f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>'
                f'</text>'
            )

            ba  = sim._interp(SUIRIRI_ACTIVITY, float(m_idx))
            fr  = sim._interp(MURICI_FRUITING,  float(m_idx))
            dp  = sim._interp(DISPERSAL_PROB,   float(m_idx))
            sp  = sim._interp(SAPLING_RECRUIT,  float(m_idx))
            rn  = sim._interp(RAINFALL_CURVE,   float(m_idx))

            if fr > 0.7 and ba > 0.6:
                st1, c1 = "\U0001f350 PEAK FRUGIVORY",                       "#ffd54f"
                st2, c2 = f"Murici: {fr*100:.0f}% | Bird: {ba*100:.0f}%",   "#ffee58"
                st3, c3 = f"Dispersal eff.: {dp*100:.0f}%",                  "#a5d6a7"
            elif sp > 0.5:
                st1, c1 = "\U0001f331 SAPLING FLUSH",                        "#a5d6a7"
                st2, c2 = f"Recruitment: {sp*100:.0f}%",                     "#66bb6a"
                st3, c3 = f"Rainfall: {rn*100:.0f}%",                        "#29b6f6"
            elif fr > 0.4:
                st1, c1 = "\U0001f7e1 FRUITING ONSET",                       "#ffd54f"
                st2, c2 = f"Murici: {fr*100:.0f}%",                          "#ffb300"
                st3, c3 = f"Bird activity: {ba*100:.0f}%",                   "#ffee58"
            elif ba < 0.25:
                st1, c1 = "\u2600 DRY SEASON NADIR",                         "#78909c"
                st2, c2 = "Low bird activity",                                "#546e7a"
                st3, c3 = f"Rainfall: {rn*100:.0f}%",                        "#546e7a"
            else:
                st1, c1 = "\U0001f9a2 FORAGING ACTIVE",                      "#81c784"
                st2, c2 = f"Bird: {ba*100:.0f}%",                            "#a5d6a7"
                st3, c3 = f"Murici: {fr*100:.0f}%",                          "#ffd54f"

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
            (22, 154, "#ffee58", "Suiriri (Tyrannus melancholicus)"),
            (22, 170, "#ffd54f", "Murici fruits (Byrsonima crassifolia)"),
            (22, 186, "#a5d6a7", "Sapling / new recruitment"),
            (22, 202, "#29b6f6", "Chuvas / wet season"),
            (22, 218, "#78909c", "Dry season low-activity"),
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
    print(f" \u2014 Suiriri \u00d7 Murici Phenological Clock on {CONFIG.device}...")

    sim = SuiririSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(
        f"Done: {sim.total_dispersals:,} dispersals, "
        f"{sim.total_saplings:,} saplings, "
        f"peak {sim.peak_active_birds} birds active."
    )
    print("Generating SVG...")
    renderer = SuiririRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg(svg_content, "notebook_79")
    return svg_content


if __name__ == "__main__":
    main()
