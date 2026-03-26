# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 75: Lobo-guará (Chrysocyon brachyurus) ↔ Lobeira (Solanum lycocarpum)
#            Seasonal Frugivory · Seed Dispersal · Annual Range-Shift Clock
# INTERVENTION 1/4: Cerrado Trophic Cascade & Seasonal Connectivity Series
# ===================================================================================================
"""
notebook_75.py — Maned Wolf ↔ Wolf-Apple (Animal ↔ Plant):
Notebook Differentiation:
- Differentiation Focus: Lobo-guará (Chrysocyon brachyurus) ↔ Lobeira (Solanum lycocarpum) emphasizing gallery forest refugia.
- Indicator species: Cervo-do-pantanal (Blastocerus dichotomus).
- Pollination lens: orchid mimicry with specialist bees.
- Human impact lens: invasive grass spread.

              Obligate Frugivory, Seasonal Range Shift & Long-Distance Seed Dispersal

The Maned Wolf (Chrysocyon brachyurus) is the largest South American canid and one
of the Cerrado's most iconic keystone species.  It has a surprising nutritional
strategy: despite being one of the largest neotropical predators, ~50 % of its
annual diet consists of the fruit of the Lobeira (Solanum lycocarpum, "wolf-apple"),
a large multi-seeded berry that is almost exclusively adapted for wolf dispersal
(López-Bao & González-Varo 2011).

SEASONAL DYNAMICS at RESEX Recanto das Araras (Goiás):

  FRUITING PEAK — WET SEASON (Nov–Mar):
    Lobeira produces fruit massively in the wet season.  Each plant produces
    15–40 fruits/season.  Wolves consume 40–70 % of all fruit produced within
    their home range and disperse viable seeds up to 8.5 km (Barnett 2000).
    Gut passage time ~7 hours + travel at night = effective dispersal kernel.

  DRY-SEASON RANGE SHIFT (Apr–Oct):
    As Lobeira fruit dwindles, wolves shift their home ranges toward gallery forests,
    veredas (palm swamps), and moist valley bottoms where moisture-retaining species
    fruit later.  This creates a documented seasonal overlap with the RESEX's wetland
    corridor, turning the reserve into a refuge during the lean dry period.

  SEED SHADOW:
    The combination of (a) nocturnal movements of 20–35 km/night, (b) high fruit
    intake (~5 fruits/night at peak), and (c) seed viability after gut passage
    creates a dispersal shadow that effectively plants Lobeira in open grasslands
    far from the parent plant, driving landscape-level succession in fallow areas.

  MIGRATORY ASPECT:
    Wolves at the RESEX periphery make seasonal excursions of 12–30 km to track
    the fruiting phenology across the broader landscape (Rodden et al. 2004).
    This means inter-patch connectivity in a fragmented Cerrado landscape is
    mediated largely by this single species.

Connecting threads to nb76–78:
  • The Lobeira patches maintained by wolf dispersal are foraging habitat for
    Capuchin monkeys (nb76) and roosting microhabitat for bats (nb77).
  • Termite mounds (nb78) are the principal denning structures for wolf pups
    (abandoned mound cavities); wolf territorial scent-marking anchors around
    mound clusters — nutrient feedback loop.

Scientific references:
  • López-Bao & González-Varo (2011): Frugivory and seed dispersal by the maned wolf.
  • Barnett (2000): Long-distance seed dispersal by Chrysocyon brachyurus.
  • Rodden et al. (2004): Maned wolf ecology and home range dynamics.
  • Juriniak & Queirolo (2015): Habitat use and diet in the RESEX zone, Goiás.
  • Fonseca et al. (2019): Solanum lycocarpum fruit phenology and frugivore guild.
  • PIGT  field observations, Goiás (2022–2024).
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from eco_base import save_svg, sanitize_svg_text, draw_phenology_chart, draw_migration_map , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES  (month 0 = January)
# ===================================================================================================

# Lobeira fruit availability (peak wet season, Nov–Mar)
LOBEIRA_FRUIT_CURVE = [
    0.75, 0.85, 0.60, 0.35, 0.15, 0.04, 0.02, 0.03, 0.08, 0.25, 0.60, 0.85
]

# Lobo-guará dietary reliance on Lobeira (high when fruit available)
WOLF_LOBEIRA_DIET_CURVE = [
    0.70, 0.80, 0.60, 0.40, 0.25, 0.10, 0.08, 0.10, 0.15, 0.30, 0.55, 0.75
]

# Wolf nightly travel distance index (longer in dry season — range expansion)
WOLF_TRAVEL_CURVE = [
    0.45, 0.40, 0.55, 0.70, 0.85, 1.00, 1.00, 0.95, 0.80, 0.65, 0.50, 0.40
]

# Seed dispersal intensity (fruit eaten ↔ travel = effective dispersal)
SEED_DISPERSAL_CURVE = [
    0.55, 0.65, 0.58, 0.42, 0.25, 0.08, 0.06, 0.08, 0.14, 0.28, 0.55, 0.72
]

# Rainfall (common backbone across all 4 notebooks)
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]

# Wolf range overlap with RESEX territory (seasonal shift into refuge)
RESEX_OVERLAP_CURVE = [
    0.40, 0.35, 0.45, 0.60, 0.75, 0.90, 0.95, 0.85, 0.70, 0.55, 0.40, 0.35
]

# Lobeira plant flowering (precedes fruiting by ~6-8 weeks)
LOBEIRA_FLOWER_CURVE = [
    0.30, 0.20, 0.15, 0.10, 0.05, 0.00, 0.00, 0.05, 0.20, 0.55, 0.90, 0.70
]


@dataclass
class LoboCfg:
    width:  int   = 1280
    height: int = CANVAS_HEIGHT
    frames: int   = 360
    fps:    int   = 10
    device: str   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry (consistent with nb71–74)
    clock_cx:     float = 420.0
    clock_cy:     float = 310.0
    clock_radius: float = 240.0

    num_wolves:    int   = 3    # individual wolf territories tracked
    num_lobeiras:  int   = 30   # Lobeira plant agents in landscape
    wolf_speed:    float = 4.2  # m/s equivalent in arena
    seed_range:    float = 140.0  # arena units (~8 km seed dispersal kernel)


CONFIG = LoboCfg()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class LoboLobeiraSim:
    """Simulates seasonal frugivory, seed dispersal shadowing,
    and the maned wolf's range shift into RESEX territory."""

    def __init__(self, cfg: LoboCfg):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy, R = cfg.clock_cx, cfg.clock_cy, cfg.clock_radius

        # ── Lobeira plants (static, scattered in clock arena) ─────────────────
        self.lobeiras: List[Dict] = []
        for k in range(cfg.num_lobeiras):
            angle = random.uniform(0, 2 * math.pi)
            r     = random.uniform(20, R - 30)
            plant_cx = cx + math.cos(angle) * r
            plant_cy = cy + math.sin(angle) * r
            self.lobeiras.append({
                "pos":       [plant_cx, plant_cy],
                "radius":    random.uniform(8, 16),   # canopy size proxy
                "fruit_load": 0.0,                     # current fruit load
                "seeds_deposited": 0,
            })

        # ── Seed cloud buffer (dispersal shadow tracking) ─────────────────────
        # Stores (x, y, frame_born) for each deposited seed
        self.seeds: List[Dict] = []

        # ── Wolves ─────────────────────────────────────────────────────────────
        self.wolves: List[Dict] = []
        for k in range(cfg.num_wolves):
            angle = (k / cfg.num_wolves) * 2 * math.pi
            r     = random.uniform(R * 0.3, R * 0.7)
            self.wolves.append({
                "pos":       torch.tensor(
                                 [cx + math.cos(angle) * r,
                                  cy + math.sin(angle) * r],
                                 device=self.dev, dtype=torch.float32),
                "vel":       torch.zeros(2, device=self.dev, dtype=torch.float32),
                "territory_angle": angle,
                "energy":    80.0,
                "state":     "foraging",
                "target_plant": -1,
                "fruits_eaten": 0,
                "seeds_dispersed": 0,
                "nightly_km": 0.0,
            })

        # History buffers
        self.hist_month:       List[float]                            = []
        self.hist_wolf_xy:     List[List[Tuple[float,float,float]]] = [
            [] for _ in range(cfg.num_wolves)
        ]
        self.hist_fruit_total: List[float]                            = []
        self.hist_seed_events: List[Tuple[float,float,int]]          = []  # (x, y, frame)

        # Aggregate metrics
        self.total_fruits_eaten  = 0
        self.total_seeds_dispersed = 0
        self.peak_fruit_load     = 0.0
        self.resex_overlap_frames = 0

    def _interp(self, curve: list, month_frac: float) -> float:
        m  = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t  = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy, R = cfg.clock_cx, cfg.clock_cy, cfg.clock_radius
        month_frac = (frame / cfg.frames) * 12.0
        self.hist_month.append(month_frac)

        fruit_avail  = self._interp(LOBEIRA_FRUIT_CURVE,    month_frac)
        wolf_diet    = self._interp(WOLF_LOBEIRA_DIET_CURVE, month_frac)
        travel_idx   = self._interp(WOLF_TRAVEL_CURVE,      month_frac)
        dispersal    = self._interp(SEED_DISPERSAL_CURVE,   month_frac)
        resex_ov     = self._interp(RESEX_OVERLAP_CURVE,    month_frac)

        # ── 1. Update Lobeira fruit loads ─────────────────────────────────────
        total_fruit = 0.0
        for plant in self.lobeiras:
            plant["fruit_load"] = fruit_avail * (0.7 + random.uniform(0, 0.3))
            total_fruit += plant["fruit_load"]
        self.hist_fruit_total.append(total_fruit / len(self.lobeiras))
        self.peak_fruit_load = max(self.peak_fruit_load, total_fruit)

        if resex_ov > 0.7:
            self.resex_overlap_frames += 1

        # ── 2. Wolf behaviour ─────────────────────────────────────────────────
        fruiting_plants = [p for p in self.lobeiras if p["fruit_load"] > 0.3]

        for ki, wolf in enumerate(self.wolves):
            pos = wolf["pos"]

            # Speed modulated by travel index (faster in dry season)
            speed = cfg.wolf_speed * (0.6 + travel_idx * 0.6)

            # Find closest fruiting plant to target
            wolf["target_plant"] = -1
            if fruiting_plants and wolf_diet > 0.15:
                best_pi = -1; best_d = 1e9
                for pi, plant in enumerate(fruiting_plants):
                    dx_ = pos[0].item() - plant["pos"][0]
                    dy_ = pos[1].item() - plant["pos"][1]
                    d_  = math.sqrt(dx_*dx_ + dy_*dy_)
                    if d_ < best_d:
                        best_d = d_; best_pi = pi
                wolf["target_plant"] = best_pi

            if wolf["target_plant"] != -1:
                # Move toward target plant
                plant = fruiting_plants[wolf["target_plant"]]
                tgt   = torch.tensor(plant["pos"], device=self.dev, dtype=torch.float32)
                vec   = tgt - pos
                dist  = torch.norm(vec).item()
                if dist > 12.0:
                    jitter = torch.randn(2, device=self.dev) * (speed * 0.25)
                    pos += (vec / max(dist, 1e-5)) * speed + jitter
                else:
                    # Arrived — eat fruit and drop seed nearby
                    if plant["fruit_load"] > 0.1 and random.random() < wolf_diet * 0.35:
                        plant["fruit_load"]     = max(0, plant["fruit_load"] - 0.08)
                        wolf["fruits_eaten"]    += 1
                        self.total_fruits_eaten += 1
                        # Seed deposited after ~7hr gut passage = ~15-20 km travel
                        # In arena coords: up to cfg.seed_range away
                        if random.random() < dispersal * 0.6:
                            seed_angle = random.uniform(0, 2 * math.pi)
                            seed_dist  = random.uniform(30, cfg.seed_range)
                            sx = pos[0].item() + math.cos(seed_angle) * seed_dist
                            sy = pos[1].item() + math.sin(seed_angle) * seed_dist
                            # Clamp to clock arena bounds
                            sx = max(20, min(sx, cfg.width - 400))
                            sy = max(60, min(sy, cfg.height - 20))
                            self.seeds.append({"pos": [sx, sy], "frame": frame,
                                               "opacity": 0.9})
                            plant["seeds_deposited"] += 1
                            wolf["seeds_dispersed"]  += 1
                            self.total_seeds_dispersed += 1
                            self.hist_seed_events.append((sx, sy, frame))
                    wolf["target_plant"] = -1
            else:
                # Patrol territory: slow spiral weighted toward RESEX centre
                # (simulates dry-season range shift inward)
                angle_now = wolf["territory_angle"] + frame * 0.008
                r_now     = R * (0.6 - resex_ov * 0.25)  # contract range in dry season
                patrol_x  = cx + math.cos(angle_now) * r_now
                patrol_y  = cy + math.sin(angle_now) * r_now
                tgt2 = torch.tensor([patrol_x, patrol_y], device=self.dev, dtype=torch.float32)
                vec2 = tgt2 - pos
                dist2 = torch.norm(vec2).item()
                if dist2 > 5.0:
                    jitter = torch.randn(2, device=self.dev) * (speed * 0.4)
                    pos += (vec2 / max(dist2, 1e-5)) * speed * 0.5 + jitter

            # Clamp wolf to arena bounds
            dx_ = pos[0].item() - cx; dy_ = pos[1].item() - cy
            dr_ = math.sqrt(dx_*dx_ + dy_*dy_)
            if dr_ > R - 15:
                pos[0] = cx + (dx_/dr_) * (R - 15)
                pos[1] = cy + (dy_/dr_) * (R - 15)
            pos[0] = pos[0].clamp(20, cfg.width - 400)
            pos[1] = pos[1].clamp(55, cfg.height - 20)

            self.hist_wolf_xy[ki].append((pos[0].item(), pos[1].item(), 1.0))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class LoboRenderer:

    def __init__(self, cfg: LoboCfg, sim: LoboLobeiraSim):
        self.cfg = cfg
        self.sim = sim

    def _arc_path(self, cx, cy, radius, start_m, end_m):
        a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
        a2 = (end_m   / 12) * 2 * math.pi - math.pi / 2
        x1 = cx + math.cos(a1) * radius; y1 = cy + math.sin(a1) * radius
        x2 = cx + math.cos(a2) * radius; y2 = cy + math.sin(a2) * radius
        span  = (end_m - start_m) % 12
        large = 1 if span > 6 else 0
        return f"M {x1:.0f} {y1:.0f} A {radius} {radius} 0 {large} 1 {x2:.0f} {y2:.0f}"

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
            f'style="background-color:#0a100a; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # ── Defs ──────────────────────────────────────────────────────────────
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="cerradoBg">'
            '<stop offset="0%"   stop-color="#1a2e12" stop-opacity="0.95"/>'
            '<stop offset="65%"  stop-color="#0e1e0a" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#0a100a" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="lobGlow">'
            '<stop offset="0%"   stop-color="#7b1fa2" stop-opacity="0.75"/>'
            '<stop offset="60%"  stop-color="#ab47bc" stop-opacity="0.25"/>'
            '<stop offset="100%" stop-color="#7b1fa2" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="wolfGlow">'
            '<stop offset="0%"   stop-color="#ff8f00" stop-opacity="0.65"/>'
            '<stop offset="100%" stop-color="#ff8f00" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="seedGlow">'
            '<stop offset="0%"   stop-color="#aed581" stop-opacity="0.80"/>'
            '<stop offset="100%" stop-color="#aed581" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<filter id="softGlow">'
            '<feGaussianBlur stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
        )
        svg.append('</defs>')

        # ── Background ────────────────────────────────────────────────────────
        svg.append(f'<rect width="{w}" height="{h}" fill="#0a100a"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#cerradoBg)"/>')

        # Wet-season wash (green tint follows rainfall)
        rain_fills = ";".join(
            f"rgba(56,142,60,{sim._interp(RAINFALL_CURVE, (f/F)*12) * 0.16:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{rain_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # Dry-season orange dust
        dry_fills = ";".join(
            f"rgba(255,143,0,{max(0, sim._interp(WOLF_TRAVEL_CURVE, (f/F)*12) - 0.5) * 0.10:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{dry_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # ── Title ─────────────────────────────────────────────────────────────
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ff8f00" font-weight="bold">'
            f'ECO-SIM: Maned Wolf × Wolf Apple    - Frugivory & Seasonal Range-Shift Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#ce93d8">'
            f'Obligate frugivory · Long-distance seed dispersal · Wet/dry seasonal range shift | </text>'
        )

        # ── Clock Face ────────────────────────────────────────────────────────
        months     = ["JAN","FEB","MAR","APR","MAY","JUN",
                      "JUL","AUG","SEP","OCT","NOV","DEC"]
        # Colour key: purple = peak fruit / wet; amber = dry/range expansion
        month_cols = {
            0:"#ce93d8", 1:"#ba68c8", 2:"#ab47bc",     # Jan–Mar: fruit peak
            3:"#ff8f00", 4:"#ef6c00", 5:"#e65100",     # Apr–Jun: drying out
            6:"#e65100", 7:"#ef6c00", 8:"#ff8f00",     # Jul–Sep: dry max
            9:"#ab47bc", 10:"#ba68c8", 11:"#ce93d8",   # Oct–Dec: rains return
        }
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="{month_cols[i]}" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R;     ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R-6); ly2 = cy + math.sin(angle) * (R-6)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#1a2e12" stroke-width="2"/>'
            )

        # ── Season Arcs ───────────────────────────────────────────────────────
        # Wet season / Lobeira fruiting window (Nov–Mar)
        d1 = self._arc_path(cx, cy, R + 11, 10, 3)
        svg.append(f'<path d="{d1}" fill="none" stroke="#ab47bc" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.50"/>')
        mid_wet = (((10 + 3 + 12) / 2 % 12) / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid_wet)*(R+22):.0f}" '
                   f'y="{cy+math.sin(mid_wet)*(R+22)+26:.0f}" font-size="15" '
                   f'fill="#ce93d8" text-anchor="middle">Lobeira Fruiting</text>')

        # Dry season / range-expansion window (May–Sep)
        d2 = self._arc_path(cx, cy, R + 11, 4, 9)
        svg.append(f'<path d="{d2}" fill="none" stroke="#ef6c00" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.45"/>')
        mid_dry = ((4 + 9) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid_dry)*(R+22):.0f}" '
                   f'y="{cy+math.sin(mid_dry)*(R+22):.0f}" font-size="15" '
                   f'fill="#ff8f00" text-anchor="middle">Range Expansion</text>')

        # Transition / dispersal maximum arcs
        d3 = self._arc_path(cx, cy, R + 25, 10.5, 3)
        svg.append(f'<path d="{d3}" fill="none" stroke="#aed581" stroke-width="5" '
                   f'stroke-linecap="round" opacity="0.55" stroke-dasharray="4,3"/>')
        svg.append(
            f'<text font-weight="bold" x="{cx+math.cos(mid_wet)*(R+36):.0f}" '
            f'y="{cy+math.sin(mid_wet)*(R+36):.0f}" font-size="15" '
            f'fill="#aed581" text-anchor="middle">Seed Dispersal Peak</text>'
        )

        # Clock ring
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
                   f'stroke="#1a2e12" stroke-width="1.5" opacity="0.9"/>')

        # ── Lobeira plants (glowing purple bubbles) ───────────────────────────
        for li, plant in enumerate(sim.lobeiras):
            px, py = plant["pos"]
            pr     = plant["radius"]
            # Build per-frame fruit-state opacity driven by LOBEIRA_FRUIT_CURVE
            fruit_ops_list = []
            for fi in range(F):
                fv = sim._interp(LOBEIRA_FRUIT_CURVE, (fi / F) * 12)
                # Each plant offset slightly for visual variety
                local_fv = min(1.0, fv * (0.6 + (li % 5) * 0.08))
                fruit_ops_list.append(f"{local_fv * 0.75:.2f}")
            fruit_ops = ";".join(fruit_ops_list)

            # Outer glow ring (fruiting indicator)
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" r="{pr + 8}" fill="url(#lobGlow)" opacity="0">'
                f'<animate attributeName="opacity" values="{fruit_ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            # Plant base (always visible, dim when not fruiting)
            plant_base_ops = ";".join(
                f"{sim._interp(LOBEIRA_FRUIT_CURVE, (fi / F) * 12) * 0.55 + 0.15:.2f}"
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" r="{pr}" fill="#4a0072" opacity="0.55">'
                f'<animate attributeName="opacity" values="{plant_base_ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            # Fruit dots (small circles when fruiting)
            fruit_dot_ops = ";".join([
                f"{sim._interp(LOBEIRA_FRUIT_CURVE,(fi/F)*12)*0.90:.2f}" for fi in range(F)
            ])
            svg.append(
                f'<circle cx="{px + pr*0.45:.0f}" cy="{py - pr*0.45:.0f}" r="3.5" fill="#e040fb" opacity="0">'
                f'<animate attributeName="opacity" values="{fruit_dot_ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # ── Dispersed seeds (bright green stars that persist) ─────────────────
        # Render up to 80 seed events as small animated markers
        MAX_SEEDS = min(80, len(sim.hist_seed_events))
        for si, (sx, sy, sf) in enumerate(sim.hist_seed_events[:MAX_SEEDS]):
            # Appear at frame sf, persist to end
            ops_list = ["0.0"] * F
            for fi in range(sf, F):
                fade = min(1.0, (fi - sf) / 5)
                ops_list[fi] = f"{fade * 0.75:.2f}"
            seed_op = ";".join(ops_list)
            svg.append(
                f'<circle cx="{sx:.0f}" cy="{sy:.0f}" r="3.5" fill="#aed581" opacity="0">'
                f'<animate attributeName="opacity" values="{seed_op}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # ── Wolves ────────────────────────────────────────────────────────────
        wolf_colors = ["#ff8f00", "#ef6c00", "#ffa000"]
        for ki in range(cfg.num_wolves):
            hist = sim.hist_wolf_xy[ki]
            if not hist:
                continue
            wxs  = ";".join(str(round(h[0], 1)) for h in hist)
            wys  = ";".join(str(round(h[1], 1)) for h in hist)
            col  = wolf_colors[ki % len(wolf_colors)]

            # Territory trail (every 4th frame)
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::4]]
            if len(trail_pts) > 3:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="{col}" stroke-width="1.0" stroke-dasharray="3,5" opacity="0.28"/>'
                )

            # Wolf body (stylised: elongated ellipse + leg stubs)
            svg.append(
                f'<ellipse rx="9" ry="5" fill="{col}">'
                f'<animate attributeName="cx" values="{wxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{wys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</ellipse>'
            )
            # Black leg bar (long legs — distinctive Lobo-guará trait)
            svg.append(
                f'<rect width="4" height="10" fill="#1a1a1a">'
                f'<animate attributeName="x" values="'
                + ";".join(str(round(h[0] - 2, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="y" values="'
                + ";".join(str(round(h[1] + 4, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'</rect>'
            )
            # Amber mane tuft (small ellipse above body; wolf has reddish dorsal mane)
            svg.append(
                f'<ellipse rx="4" ry="3" fill="#b71c1c">'
                f'<animate attributeName="cx" values="{wxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="'
                + ";".join(str(round(h[1] - 6, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'</ellipse>'
            )

        # ── Clock Hand ────────────────────────────────────────────────────────
        hand_x = ";".join(
            str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R - 10), 1))
            for m in sim.hist_month
        )
        hand_y = ";".join(
            str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R - 10), 1))
            for m in sim.hist_month
        )
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#ff8f00" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#0a100a" stroke="#ff8f00" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ef6c00"/>')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 395
        panel_w = 375

        # ── Panel 1: Ecological Logic ──────────────────────────────────────────
        py1, ph1 = 20, 266
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#050a05" rx="8" '
                   f'stroke="#ff8f00" stroke-width="1" opacity="0.95"/>')
        svg.append(f'<text x="12" y="22" fill="#ff8f00" font-size="15" font-weight="bold">'
                   f'Obligate Frugivory & Dispersal Strategy</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Chrysocyon brachyurus, Cerrado\'s top predator,</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'≈50% of diet: Solanum lycocarpum (Lobeira).</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'Each wolf disperses seeds up to 8.5 km/year.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ce93d8" font-size="15">'
                   f'Wet (Nov–Mar): frugivory; range contracts.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ff8f00" font-size="15">'
                   f'Dry (May–Sep): fruit depleted; range +30–50%</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#ff8f00" font-size="15">'
                   f'toward veredas and moist valleys (RESEX area).</text>')
        svg.append(f'<text font-weight="bold" x="12" y="126" fill="#aed581" font-size="15">'
                   f'This seed shadow drives Cerrado succession.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="140" fill="#81c784" font-size="15">'
                   f'</text>')
        svg.append('</g>')

        # ── Panel 2: Simulation Metrics ────────────────────────────────────────
        py2 = py1 + ph1 + 10
        ph2 = 110
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#050a05" rx="8" '
                   f'stroke="#37474f" stroke-width="1" opacity="0.95"/>')
        svg.append(f'<text x="12" y="22" fill="#90a4ae" font-size="15" font-weight="bold">'
                   f'Dispersal & Frugivory Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#ff8f00" font-size="15">'
                   f'Total Fruits Consumed: {sim.total_fruits_eaten:,}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#aed581" font-size="15">'
                   f'Seeds Dispersed (sim): {sim.total_seeds_dispersed:,} events</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#ce93d8" font-size="15">'
                   f'Peak Lobeira Fruit Load: {sim.peak_fruit_load:.1f} au</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'RESEX overlap: {sim.resex_overlap_frames} | '
                   f'Wolves: {cfg.num_wolves} | Plants: {cfg.num_lobeiras}</text>')
        svg.append('</g>')

        # ── Panel 3: Phenology Chart (eco_base helper) ────────────────────────
        py3 = py2 + ph2 + 10
        ph3 = 135
        curves_data = [
            (LOBEIRA_FRUIT_CURVE,      "#ab47bc", "Lobeira Fruit Availability"),
            (WOLF_LOBEIRA_DIET_CURVE,  "#ff8f00", "Wolf Lobeira Diet Fraction"),
            (SEED_DISPERSAL_CURVE,     "#aed581", "Seed Dispersal Intensity"),
            (WOLF_TRAVEL_CURVE,        "#ef6c00", "Nightly Travel Range Index"),
            (RESEX_OVERLAP_CURVE,      "#4dd0e1", "RESEX Range Overlap"),
        ]
        chart_snippet = draw_phenology_chart(
            curves_data,
            chart_w=330, chart_h=58, panel_h=ph3,
            title="Frugivory &amp; Seasonal Phenology",
            title_color="#ff8f00",
            bg_color="#050a05",
            border_color="#ff8f00",
        )
        svg.append(f'<g transform="translate({panel_x}, {py3})">{chart_snippet}</g>')

        # ── Animated Bottom Status Card ───────────────────────────────────────
        px5 = 20; py5 = h - 235; pw5 = 255; ph5 = 225
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(f'<rect width="{pw5}" height="{ph5}" fill="#050a05" rx="8" '
                   f'stroke="#ff8f00" stroke-width="1.5" opacity="0.97"/>')
        svg.append(f'<text x="12" y="22" font-size="15" fill="#ff8f00" font-weight="bold">'
                   f'Active Season Status:</text>')

        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12; vals[m_idx] = "1"
            op_str = ";".join(vals + ["0"])

            svg.append(f'<text x="12" y="52" font-size="15" fill="#ff8f00" font-weight="bold">')
            svg.append(m_name)
            svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                       f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')

            fruit = sim._interp(LOBEIRA_FRUIT_CURVE,     m_idx)
            travel= sim._interp(WOLF_TRAVEL_CURVE,       m_idx)
            disp  = sim._interp(SEED_DISPERSAL_CURVE,    m_idx)
            resex = sim._interp(RESEX_OVERLAP_CURVE,     m_idx)

            if fruit > 0.55:
                st1, c1 = "LOBEIRA FRUITING PEAK",     "#ce93d8"
                st2, c2 = f"Fruit availability: {fruit*100:.0f}%","#ab47bc"
                st3, c3 = f"Dispersal: {disp*100:.0f}%", "#aed581"
            elif travel > 0.75:
                st1, c1 = "DRY-SEASON RANGE SHIFT",   "#ff8f00"
                st2, c2 = f"Travel index: {travel*100:.0f}%","#ef6c00"
                st3, c3 = f"RESEX overlap: {resex*100:.0f}%","#4dd0e1"
            elif resex > 0.6:
                st1, c1 = "WOLF IN RESEX CORRIDOR",   "#4dd0e1"
                st2, c2 = "Refuge use — veredas & moist valley","#81c784"
                st3, c3 = f"Fruit: {fruit*100:.0f}%", "#ce93d8"
            else:
                st1, c1 = "Seasonal transition",      "#90a4ae"
                st2, c2 = "Foraging mixed strategy",     "#78909c"
                st3, c3 = f"Travel: {travel*100:.0f}%", "#ff8f00"

            for yoff, txt, col in [(80, st1, c1), (100, st2, c2), (118, st3, c3)]:
                svg.append(f'<text x="12" y="{yoff}" font-size="15" fill="{col}" font-weight="bold">')
                svg.append(sanitize_svg_text(txt))
                svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                           f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append('</text>')

        # Static legend
        svg.append('<text x="12" y="142" fill="#546e7a" font-size="15" font-weight="bold">Legend:</text>')
        entries = [
            (22, 158, "#ff8f00",  "Lobo-guará (Chrysocyon brachyurus)"),
            (22, 174, "#ab47bc",  "Lobeira plant (Solanum lycocarpum)"),
            (22, 190, "#aed581",  "Dispersed seed"),
            (22, 206, "#e040fb",  "Lobeira fruit (ripe)"),
            (22, 221, "#4dd0e1",  "RESEX dry-season refuge corridor"),
        ]
        for (ex, ey, ec, elabel) in entries:
            svg.append(f'<ellipse cx="{ex}" cy="{ey}" rx="6" ry="4" fill="{ec}" opacity="0.85"/>')
            svg.append(f'<text font-weight="bold" x="36" y="{ey+4}" fill="{ec}" font-size="15">{elabel}</text>')
        svg.append('</g>')

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Lobo-guará ↔ Lobeira (Frugivory & Range-Shift Clock) on {CONFIG.device}...")

    sim = LoboLobeiraSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_fruits_eaten:,} fruits eaten, "
          f"{sim.total_seeds_dispersed:,} seed dispersal events recorded, "
          f"peak fruit load {sim.peak_fruit_load:.1f}, "
          f"RESEX overlap {sim.resex_overlap_frames}.")

    print("Generating SVG...")
    renderer = LoboRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_75')
    return svg_content


if __name__ == "__main__":
    main()
