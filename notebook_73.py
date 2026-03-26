# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 73: Buriti (Mauritia flexuosa) ↔ Fungo-Ectomicorrízico (Pisolithus/Scleroderma)
#            Flood-Pulse Hyphal Network Expansion Clock
# INTERVENTION 3/4: Cerrado Ecological Web — Seasons & Migration Series
# ===================================================================================================
"""
notebook_73.py — Buriti Palm ↔ Ectomycorrhizal Fungi (Plant ↔ Fungus):
Notebook Differentiation:
- Differentiation Focus: Buriti (Mauritia flexuosa) ↔ Fungo-Ectomicorrízico (Pisolithus/Scleroderma) emphasizing vereda hydrology pulses.
- Indicator species: Pato-mergulhao (Mergus octosetaceus).
- Pollination lens: wind pollination during grass senescence.
- Human impact lens: grazing pressure on understory.

                 Flood-Pulse Hyphal Network Expansion & Drought Collapse

The Buriti (Mauritia flexuosa) is the emblematic palm of the Cerrado Veredas —
linear gallery-forest wetlands that cut through the arid savanna matrix, fed by
springs and seeping groundwater from the karst limestone.  In the RESEX Recanto
das Araras the Rio Lapa watershed sustains a chain of Veredas collectively known
as "Bacia do Rio Lapa".

The Buriti forms a *obligate* ectomycorrhizal association with root-colonising
basidiomycete fungi (Pisolithus tinctorius, Scleroderma spp.) that dramatically
extends the palm's effective root zone deep into the waterlogged, anaerobic soils.
This network exhibits a pronounced FLOOD-PULSE cycle:

  WET SEASON (Nov–Mar): Heavy rains raise the water table and flood the vereda
  floor.  Fungal hyphae expand explosively through the damp soil, colonising new
  Buriti root tips within days.  The network spans up to 40 m between palms,
  exchanging phosphorus, manganese, and iron — micronutrients that become
  bio-available only under partial anaerobic digestion of organic matter.

  DRY-SEASON CONTRACTION (Jun–Sep): The water table drops >1 m, soil desiccates.
  Peripheral hyphal strands lyse and die.  The network shrinks toward the
  permanent spring seeps at the vereda core.  Fungi form drought-resistant
  spore carpophores (truffles / puffballs) that serve as food for mammals and
  armadillos, redistributing spores.

  FIRE DISTURBANCE (Aug–Sep): Vereda margins can burn during exceptional droughts.
  Post-fire, the residual mycelial network facilitates ultra-rapid Buriti
  regeneration — juvenile palms can access the pre-existing hyphal highway.

Migratory / seasonal interface:
  The flood pulse synchronises with the Cerrado wet season that also triggers the
  Seriema (nb71) dry-period retreat and the Ipê bloom end (nb72), and that will
  bring the migratory Andorinha (nb74) as a seasonal cue.  All four notebooks
  share 12 phenological curves anchored to the same annual calendar.

Spatial pattern:
  A vereda cross-section is modelled: Buriti palms in the central waterlogged zone;
  ectomycorrhizal network as animated hyphal threads connecting root zones;
  water table shown as an animated blue flood plane rising/falling.
  The phenological clock occupies the left panel; a vereda profile view occupies
  the central/right zone inside the clock.

Scientific references:
  • Gottsberger & Silberbauer-Gottsberger (2006): Life in the Cerrado — Veredas.
  • Leake et al. (2004): Networks of power and influence: ectomycorrhizal webs.
  • Neves et al. (2010): Ectomycorrhizal fungi under flooding in tropical wetlands.
  • Haridasan (2008): Nutritional adaptations of native plants in the Cerrado.
  • PIGT RESEX Recanto das Araras / Rio Lapa watershed surveys (2022–2024).
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from eco_base import save_svg, sanitize_svg_text, draw_phenology_chart , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES  (month 0 = January)
# ===================================================================================================

# Cerrado rainfall / soil moisture (wet Nov–Mar, dry Jun–Sep)
SOIL_MOISTURE_CURVE = [0.80, 0.70, 0.85, 0.60, 0.35, 0.15, 0.05, 0.05, 0.25, 0.50, 0.80, 0.90]
WATER_TABLE_CURVE = [
    0.80, 0.90, 0.85, 0.55, 0.25, 0.08, 0.02, 0.02, 0.12, 0.38, 0.72, 0.88
]

# Fungal hyphal network density (expands with moisture, collapses in drought)
FUNGAL_NETWORK_CURVE = [
    0.85, 0.95, 0.90, 0.65, 0.35, 0.12, 0.04, 0.03, 0.15, 0.42, 0.75, 0.90
]

# Buriti palm nutrient uptake efficiency (mediated by fungal network)
BURITI_UPTAKE_CURVE = [
    0.80, 0.90, 0.85, 0.60, 0.35, 0.15, 0.05, 0.05, 0.20, 0.45, 0.75, 0.85
]

# Fungal fruiting body (spore dispersal via carpophores — peaks in dry then post-wet)
SPOROCARP_CURVE = [
    0.10, 0.05, 0.05, 0.15, 0.35, 0.70, 0.90, 0.80, 0.40, 0.15, 0.05, 0.05
]

# Fire disturbance risk to vereda margin (Aug–Sep peak)
FIRE_MARGIN_CURVE = [
    0.00, 0.00, 0.00, 0.00, 0.05, 0.25, 0.55, 0.85, 0.70, 0.20, 0.02, 0.00
]


@dataclass
class BuritiFungusConfig:
    width: int  = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int    = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry (consistent with nb69–72)
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0

    num_palms: int      = 8    # Buriti palms in vereda core
    num_fungi_hubs: int = 60   # mycorrhizal hyphal junction nodes
    num_spores: int     = 25   # animated spore/carpophore dispersal dots
    network_max_dist: float = 90.0  # max inter-hub hyphal connection distance


CONFIG = BuritiFungusConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class BuritiFungusSim:

    def __init__(self, cfg: BuritiFungusConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R      = cfg.clock_radius

        # --- Buriti Palms: placed in the "vereda core" — a narrow band
        #     through the clock horizontally ~60 % down (representing wetland strip)
        self.palms: List[Dict] = []
        for k in range(cfg.num_palms):
            # Distribute along the vereda axis (left to right inside the clock)
            frac = (k + 0.5) / cfg.num_palms
            angle_spread = 0.55 * math.pi   # span roughly 100° of the clock
            angle = -angle_spread / 2 + frac * angle_spread + random.uniform(-0.08, 0.08)
            r = random.uniform(R * 0.20, R * 0.68)
            self.palms.append({
                "pos": (cx + math.cos(angle) * r, cy + math.sin(angle) * r * 0.7),
                "uptake": 0.0,
                "height": random.uniform(0.7, 1.0),   # visual scale
                "total_nutrients": 0.0,
            })

        # --- Fungal Hyphal Hub Nodes (scattered around vereda moisture zone)
        self.hubs: List[Dict] = []
        for _ in range(cfg.num_fungi_hubs):
            angle = random.uniform(-math.pi * 0.65, math.pi * 0.65)
            r = random.uniform(15, R - 30)
            self.hubs.append({
                "pos": (cx + math.cos(angle) * r, cy + math.sin(angle) * r * 0.72),
                "active": False,
                "density": 0.0,
            })

        # Build static edge list: connect hubs within max_dist
        self.hyphal_edges: List[Tuple[int, int]] = []
        for i in range(cfg.num_fungi_hubs):
            for j in range(i + 1, cfg.num_fungi_hubs):
                hx1, hy1 = self.hubs[i]["pos"]
                hx2, hy2 = self.hubs[j]["pos"]
                d = math.sqrt((hx1 - hx2)**2 + (hy1 - hy2)**2)
                if d < cfg.network_max_dist:
                    self.hyphal_edges.append((i, j))

        # Connect each palm to 3 nearest hubs
        self.palm_hub_connections: List[Tuple[int, int]] = []
        for pi, p in enumerate(self.palms):
            dists = [(math.sqrt((p["pos"][0]-h["pos"][0])**2 +
                                (p["pos"][1]-h["pos"][1])**2), hi)
                     for hi, h in enumerate(self.hubs)]
            dists.sort()
            for _, hi in dists[:3]:
                self.palm_hub_connections.append((pi, hi))

        # --- Spore dispersal dots (visible in dry season)
        self.spores: List[Dict] = []
        for _ in range(cfg.num_spores):
            angle = random.uniform(-math.pi * 0.65, math.pi * 0.65)
            r = random.uniform(30, R - 40)
            self.spores.append({
                "pos": [cx + math.cos(angle) * r, cy + math.sin(angle) * r * 0.7],
                "vel": [random.uniform(-1.5, 1.5), random.uniform(-0.8, 0.8)],
                "active": False,
            })

        self.hist_month: List[float] = []
        self.hist_hub_density: List[List[float]] = []   # per-frame hub densities
        self.hist_water_level: List[float] = []          # water table level
        self.hist_spore_pos: List[List[Tuple[float, float, float]]] = []  # (x, y, opacity)

        self.total_nutrients_exchanged = 0.0
        self.peak_network_density      = 0.0
        self.total_spore_dispersal_frames = 0

    def _interp(self, curve: list, month_frac: float) -> float:
        m = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t  = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy = cfg.clock_cx, cfg.clock_cy
        month_frac   = (frame / cfg.frames) * 12.0
        self.hist_month.append(month_frac)

        moisture     = self._interp(SOIL_MOISTURE_CURVE,    month_frac)
        water_level  = self._interp(WATER_TABLE_CURVE,      month_frac)
        net_density  = self._interp(FUNGAL_NETWORK_CURVE,   month_frac)
        uptake_eff   = self._interp(BURITI_UPTAKE_CURVE,    month_frac)
        spore_prob   = self._interp(SPOROCARP_CURVE,        month_frac)

        self.hist_water_level.append(water_level)

        # --- Hub density modulated by soil moisture with stochastic noise
        frame_hubs = []
        for hub in self.hubs:
            target = net_density * random.uniform(0.8, 1.2)
            hub["density"] = max(0.0, min(1.0,
                hub["density"] * 0.8 + target * 0.2
            ))
            hub["active"] = hub["density"] > 0.05
            frame_hubs.append(hub["density"])
        self.hist_hub_density.append(frame_hubs)
        self.peak_network_density = max(self.peak_network_density, net_density)

        # --- Palm nutrient uptake
        for p in self.palms:
            gain = uptake_eff * random.uniform(0.5, 1.5)
            p["uptake"] = max(0.0, min(1.0, gain))
            p["total_nutrients"] += gain
            self.total_nutrients_exchanged += gain

        # --- Spore dispersal (active in dry season / post-fire)
        frame_spores: List[Tuple[float, float, float]] = []
        for sp in self.spores:
            sp["active"] = random.random() < spore_prob * 0.7
            if sp["active"]:
                # Drift gently
                sp["pos"][0] += sp["vel"][0] + random.uniform(-0.5, 0.5)
                sp["pos"][1] += sp["vel"][1] + random.uniform(-0.3, 0.3)
                # Wrap within clock arena
                dx = sp["pos"][0] - cx; dy = sp["pos"][1] - cy
                dr = math.sqrt(dx*dx + dy*dy)
                if dr > cfg.clock_radius - 20:
                    angle = math.atan2(dy, dx) + random.uniform(-0.3, 0.3)
                    r_new = random.uniform(20, cfg.clock_radius - 30)
                    sp["pos"] = [cx + math.cos(angle) * r_new,
                                 cy + math.sin(angle) * r_new * 0.72]
                self.total_spore_dispersal_frames += 1
                frame_spores.append((sp["pos"][0], sp["pos"][1], 0.85))
            else:
                frame_spores.append((sp["pos"][0], sp["pos"][1], 0.0))
        self.hist_spore_pos.append(frame_spores)


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class BuritiFungusRenderer:

    def __init__(self, cfg: BuritiFungusConfig, sim: BuritiFungusSim):
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
            f'style="background-color:#080e12; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # ── Defs ──────────────────────────────────────────────────────────────
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="veredaBg">'
            '<stop offset="0%"   stop-color="#0d2030" stop-opacity="0.96"/>'
            '<stop offset="60%"  stop-color="#091520" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#080e12" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            # Flood water: animated teal-blue fill rising
            '<linearGradient id="waterGrad" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%"   stop-color="#0288d1" stop-opacity="0.55"/>'
            '<stop offset="100%" stop-color="#01579b" stop-opacity="0.30"/>'
            '</linearGradient>'
        )
        svg.append(
            '<radialGradient id="mycelGlow">'
            '<stop offset="0%"   stop-color="#ce93d8" stop-opacity="0.9"/>'
            '<stop offset="100%" stop-color="#ce93d8" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="palmGlow">'
            '<stop offset="0%"   stop-color="#00897b" stop-opacity="0.7"/>'
            '<stop offset="100%" stop-color="#00897b" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<filter id="mycelFilter" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur stdDeviation="4" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
        )
        svg.append('</defs>')

        # ── Background ────────────────────────────────────────────────────────
        svg.append(f'<rect width="{w}" height="{h}" fill="#080e12"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#veredaBg)"/>')

        # Animated flood-plane inside clock —water table rises/falls each frame
        # We approximate by animating a rect's y and height clipped to the clock circle
        # Using an animated clip on an ellipse for the water plane
        svg.append(f'<clipPath id="clockClip">'
                   f'<circle cx="{cx}" cy="{cy}" r="{R - 8}"/>'
                   f'</clipPath>')

        # Water table level: animates the y of a fill rect (bottom-anchored)
        water_ys = ";".join(
            str(round(cy + R - (sim._interp(WATER_TABLE_CURVE, (f/F)*12) * R * 0.7) - 5, 1))
            for f in range(F)
        )
        water_hs = ";".join(
            str(round(sim._interp(WATER_TABLE_CURVE, (f/F)*12) * R * 0.7 + 10, 1))
            for f in range(F)
        )
        svg.append(
            f'<g clip-path="url(#clockClip)">'
            f'<rect x="{cx - R}" width="{R * 2}" fill="url(#waterGrad)" opacity="0.65">'
            f'<animate attributeName="y" values="{water_ys}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<animate attributeName="height" values="{water_hs}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</rect>'
            f'</g>'
        )

        # Fire margin risk wash (orange edges when fire risk high)
        fire_fills = ";".join(
            f"rgba(255,87,34,{sim._interp(FIRE_MARGIN_CURVE, (f/F)*12) * 0.28:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="none" '
            f'stroke="transparent" clip-path="url(#clockClip)">'
            f'</circle>'
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="transparent" clip-path="url(#clockClip)">'
            f'<animate attributeName="fill" values="{fire_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # ── Title ─────────────────────────────────────────────────────────────
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#80cbc4" font-weight="bold">'
            f'ECO-SIM: Buriti Palm Palm Palm × Ectomycorrhizal Fungi    - Flood-Pulse Hyphal Network</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#80deea">'
            f'Vereda water-table cycle &amp; mycorrhizal expansion/collapse </text>'
        )

        # ── Clock Face ────────────────────────────────────────────────────────
        months = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            # Blue for wet months, orange for dry/fire, teal for transition
            if i in [10, 11, 0, 1, 2]:
                m_col = "#29b6f6"
            elif i in [6, 7, 8]:
                m_col = "#ff7043"
            else:
                m_col = "#4db6ac"
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="{m_col}" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 6)
            ly2 = cy + math.sin(angle) * (R - 6)
            svg.append(f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                       f'stroke="#1a3040" stroke-width="2"/>')

        # ── Season Arcs ───────────────────────────────────────────────────────
        # Flood/expansion arc (Nov–Mar)
        d = self._arc_path(cx, cy, R + 10, 10, 15)
        svg.append(f'<path d="{d}" fill="none" stroke="#29b6f6" stroke-width="10" '
                   f'stroke-linecap="round" opacity="0.50"/>')
        mid_wet = ((10 + 15) / 2 % 12 / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text x="{cx + math.cos(mid_wet)*(R+20):.0f}" '
                   f'y="{cy + math.sin(mid_wet)*(R+20):.0f}" font-size="15" '
                   f'fill="#29b6f6" text-anchor="middle" font-weight="bold">Flood Expansion</text>')

        # Dry contraction arc (Jun–Sep)
        d2 = self._arc_path(cx, cy, R + 10, 5, 9.5)
        svg.append(f'<path d="{d2}" fill="none" stroke="#ff7043" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.45"/>')
        mid_dry = ((5 + 9.5) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text x="{cx + math.cos(mid_dry)*(R+20):.0f}" '
                   f'y="{cy + math.sin(mid_dry)*(R+20):.0f}" font-size="15" '
                   f'fill="#ff7043" text-anchor="middle" font-weight="bold">Network Collapse</text>')

        # Sporocarp/fruiting arc (Jun–Sep)
        d3 = self._arc_path(cx, cy, R + 24, 5.5, 9)
        svg.append(f'<path d="{d3}" fill="none" stroke="#ce93d8" stroke-width="6" '
                   f'stroke-linecap="round" opacity="0.60" stroke-dasharray="5,4"/>')
        svg.append(f'<text x="{cx + math.cos(((5.5+9)/2/12)*2*math.pi-math.pi/2)*(R+34):.0f}" '
                   f'y="{cy + math.sin(((5.5+9)/2/12)*2*math.pi-math.pi/2)*(R+34):.0f}" '
                   f'font-size="15" fill="#ce93d8" text-anchor="middle" font-weight="bold">Sporocarps</text>')

        # Post-fire regeneration arc (Sep–Oct)
        d4 = self._arc_path(cx, cy, R + 24, 9, 10.5)
        svg.append(f'<path d="{d4}" fill="none" stroke="#ff8a65" stroke-width="5" '
                   f'stroke-linecap="round" opacity="0.55"/>')

        # Clock outer ring with luminous glow pulse
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 4}" fill="none" '
                   f'stroke="#0288d1" stroke-width="3" opacity="0.5">'
                   f'<animate attributeName="stroke-opacity" values="0.5;0.9;0.5" dur="4.0s" repeatCount="indefinite"/>'
                   f'<animate attributeName="stroke-width" values="2;5;2" dur="4.0s" repeatCount="indefinite"/>'
                   f'</circle>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
                   f'stroke="#1a3040" stroke-width="1.5" opacity="0.8"/>')

        # ── Fungal Hyphal Network ─────────────────────────────────────────────
        # Draw each edge with animated opacity driven by average hub density at endpoints
        for (i, j) in sim.hyphal_edges:
            hx1, hy1 = sim.hubs[i]["pos"]
            hx2, hy2 = sim.hubs[j]["pos"]
            # Opacity follows average density of both endpoint hubs across frames
            edge_ops = ";".join(
                f"{(sim.hist_hub_density[fi][i] + sim.hist_hub_density[fi][j]) / 2 * 0.65:.2f}"
                for fi in range(F)
            )
            svg.append(
                f'<line x1="{hx1:.1f}" y1="{hy1:.1f}" x2="{hx2:.1f}" y2="{hy2:.1f}" '
                f'stroke="#ce93d8" stroke-width="2.0" opacity="0">'
                f'<animate attributeName="opacity" values="{edge_ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="stroke-width" values="1.5;3.0;1.5" dur="3.5s" repeatCount="indefinite"/>'
                f'</line>'
            )

        # Hub glow nodes
        for hi, hub in enumerate(sim.hubs):
            hx, hy = hub["pos"]
            hub_ops = ";".join(
                f"{sim.hist_hub_density[fi][hi] * 0.8:.2f}" for fi in range(F)
            )
            hub_r = ";".join(
                f"{max(2.5, sim.hist_hub_density[fi][hi] * 8):.1f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{hx:.1f}" cy="{hy:.1f}" fill="#ce93d8" opacity="0" filter="url(#mycelFilter)">'
                f'<animate attributeName="r" values="{hub_r}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{hub_ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # Palm–hub connection threads (animated, coloured by uptake direction)
        for (pi, hi) in sim.palm_hub_connections:
            px0, py0 = sim.palms[pi]["pos"]
            hx0, hy0 = sim.hubs[hi]["pos"]
            conn_ops = ";".join(
                f"{sim.hist_hub_density[fi][hi] * 0.55:.2f}" for fi in range(F)
            )
            svg.append(
                f'<line x1="{px0:.1f}" y1="{py0:.1f}" x2="{hx0:.1f}" y2="{hy0:.1f}" '
                f'stroke="#00bcd4" stroke-width="1.5" opacity="0">'
                f'<animate attributeName="opacity" values="{conn_ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</line>'
            )
            # Animated nutrient flow dots (travel from hub to palm, wet season)
            for offset_frac in [0.0, 0.35, 0.65]:
                move_dur = random.uniform(0.8, 1.8)
                svg.append(
                    f'<circle r="2.5" fill="#00bcd4" opacity="0">'
                    f'<animate attributeName="cx" '
                    f'values="{hx0:.1f};{px0:.1f}" dur="{move_dur:.1f}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="cy" '
                    f'values="{hy0:.1f};{py0:.1f}" dur="{move_dur:.1f}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="opacity" values="{conn_ops}" '
                    f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'</circle>'
                )

        # ── Spore Dispersal Dots (visible in dry season) ─────────────────────
        for si in range(cfg.num_spores):
            sp_xs = ";".join(str(round(sim.hist_spore_pos[fi][si][0], 1)) for fi in range(F))
            sp_ys = ";".join(str(round(sim.hist_spore_pos[fi][si][1], 1)) for fi in range(F))
            sp_os = ";".join(str(round(sim.hist_spore_pos[fi][si][2], 2)) for fi in range(F))
            svg.append(
                f'<circle r="4" fill="#e1bee7" opacity="0">'
                f'<animate attributeName="cx" values="{sp_xs}" '
                f'dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{sp_ys}" '
                f'dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{sp_os}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="r" values="3;6;3" dur="{0.8 + (si % 7)*0.15:.2f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # ── Buriti Palms ──────────────────────────────────────────────────────
        for pi, palm in enumerate(sim.palms):
            px, py = palm["pos"]

            # Glow disk (scales with uptake / network density)
            glow_r = ";".join(
                f"{max(5, sim._interp(FUNGAL_NETWORK_CURVE, (fi/F)*12) * 28):.1f}"
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" fill="url(#palmGlow)" opacity="0.45">'
                f'<animate attributeName="r" values="{glow_r}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

            # Trunk (tall thin rect)
            trunk_h = palm["height"] * 55
            svg.append(
                f'<rect x="{px - 3:.0f}" y="{py - trunk_h:.0f}" width="6" height="{trunk_h:.0f}" '
                f'fill="#4e342e" rx="3"/>'
            )

            # Crown (palm fronds — radiating lines from top of trunk)
            crown_y = py - trunk_h
            for k in range(8):
                angle = (k / 8) * 2 * math.pi
                fx = px + math.cos(angle) * 18 * palm["height"]
                fy = crown_y + math.sin(angle) * 10 * palm["height"]
                svg.append(
                    f'<line x1="{px:.0f}" y1="{crown_y:.0f}" '
                    f'x2="{fx:.0f}" y2="{fy:.0f}" '
                    f'stroke="#2e7d32" stroke-width="2.5" stroke-linecap="round" opacity="0.85"/>'
                )

            # Fruit clusters (animated: appear in wet season)
            fruit_op = ";".join(
                f"{max(0, sim._interp(WATER_TABLE_CURVE, (fi/F)*12) * 0.9):.2f}"
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{crown_y + 8:.0f}" r="5" fill="#e64a19" opacity="0">'
                f'<animate attributeName="opacity" values="{fruit_op}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{px + 6:.0f}" cy="{crown_y + 5:.0f}" r="4" fill="#ff7043" opacity="0">'
                f'<animate attributeName="opacity" values="{fruit_op}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
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
            f'stroke="#80cbc4" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#080e12" stroke="#80cbc4" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#0288d1"/>')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 355
        panel_w = 335

        # ── Panel 1: Ecological Logic ─────────────────────────────────────────
        py1, ph1 = 20, 232
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#060c10" rx="8" '
                   f'stroke="#0288d1" stroke-width="1" opacity="0.93"/>')
        svg.append(f'<text x="12" y="24" fill="#29b6f6" font-size="15" font-weight="bold">'
                   f'Flood-Pulse Mycorrhizal Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="48" fill="#ccc" font-size="15">'
                   f'When the vereda floods (Nov–Mar), soil oxygen</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'drops and mineral solubility changes.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="92" fill="#ccc" font-size="15">'
                   f'Ectomycorrhizal hyphae (Pisolithus) expand</text>')
        svg.append(f'<text font-weight="bold" x="12" y="114" fill="#ccc" font-size="15">'
                   f'explosively, colonising new Buriti root tips</text>')
        svg.append(f'<text font-weight="bold" x="12" y="136" fill="#ccc" font-size="15">'
                   f'within days, bridging palms up to 40 m apart.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="158" fill="#80cbc4" font-size="15">'
                   f'In the dry season, hyphae lyse near the surface.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="180" fill="#80cbc4" font-size="15">'
                   f'Fungi form sporocarps eaten by armadillos</text>')
        svg.append(f'<text font-weight="bold" x="12" y="202" fill="#80cbc4" font-size="15">'
                   f'that redistribute spores across the Cerrado.</text>')
        svg.append('</g>')

        # ── Panel 2: Metrics ──────────────────────────────────────────────────
        py2 = py1 + ph1 + 10
        ph2 = 158
        avg_density = sim.peak_network_density
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#060c10" rx="8" '
                   f'stroke="#9c27b0" stroke-width="1" opacity="0.93"/>')
        svg.append(f'<text x="12" y="24" fill="#ce93d8" font-size="15" font-weight="bold">'
                   f'Network &amp; Exchange Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#29b6f6" font-size="15">'
                   f'Nutrients Exchanged: {sim.total_nutrients_exchanged:,.0f} units</text>')
        svg.append(f'<text font-weight="bold" x="12" y="88" fill="#ce93d8" font-size="15">'
                   f'Peak Network Density: {avg_density:.2f} (normalised)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="120" fill="#ff8a65" font-size="15">'
                   f'Spore Events: {sim.total_spore_dispersal_frames:,}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="148" fill="#90a4ae" font-size="15">'
                   f'Palms: {cfg.num_palms} | Hubs: {cfg.num_fungi_hubs} | '
                   f'Edges: {len(sim.hyphal_edges)}</text>')
        svg.append('</g>')

        # ── Panel 3: Phenology Chart ──────────────────────────────────────────
        py3 = py2 + ph2 + 10
        ph3 = int(h - 10 - py3)   # fill remaining canvas height
        curves_data = [
            (WATER_TABLE_CURVE,     "#29b6f6", "Water Table Level"),
            (FUNGAL_NETWORK_CURVE,  "#ce93d8", "Hyphal Network Density"),
            (BURITI_UPTAKE_CURVE,   "#00897b", "Buriti Nutrient Uptake"),
            (SPOROCARP_CURVE,       "#e1bee7", "Sporocarp Production"),
            (FIRE_MARGIN_CURVE,     "#ff7043", "Fire Risk (margin)"),
        ]
        chart_snippet = draw_phenology_chart(
            curves_data,
            chart_w=305, chart_h=max(60, ph3 - 80), panel_h=ph3,
            title="Vereda Flood-Pulse &amp; Fungal Phenology",
            title_color="#29b6f6",
            bg_color="#060c10",
            border_color="#0288d1",
        )
        chart_snippet = chart_snippet.replace('font-size="15"', 'font-size="15"')
        chart_snippet = chart_snippet.replace('font-size="15"', 'font-size="15"')
        svg.append(f'<g transform="translate({panel_x}, {py3})">{chart_snippet}</g>')

        # ── Current Month Sidebar (animated status card) ───────────────────
        px5   = int(cfg.clock_cx + cfg.clock_radius + 40)
        py5   = h - 270
        pw5   = 210
        ph5   = 260
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(f'<rect width="{pw5}" height="{ph5}" fill="#060c10" rx="8" '
                   f'stroke="#0288d1" stroke-width="1.5" opacity="0.96"/>')
        svg.append(f'<text x="12" y="22" font-size="15" fill="#80cbc4" font-weight="bold">'
                   f'Active Season Status:</text>')

        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12; vals[m_idx] = "1"
            op_str = ";".join(vals + ["0"])

            svg.append(f'<text x="12" y="50" font-size="15" fill="#80cbc4" font-weight="bold">')
            svg.append(m_name)
            svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                       f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')

            water  = sim._interp(WATER_TABLE_CURVE,    m_idx)
            net    = sim._interp(FUNGAL_NETWORK_CURVE, m_idx)
            spore  = sim._interp(SPOROCARP_CURVE,      m_idx)
            fire   = sim._interp(FIRE_MARGIN_CURVE,    m_idx)

            if water > 0.70:
                st1, c1 = "FLOOD PEAK — Hyphae EXPLODE",  "#29b6f6"
                st2, c2 = "Network spans 40 m between palms","#80cbc4"
                st3, c3 = "Buriti uptake at maximum",        "#00897b"
            elif water > 0.35:
                st1, c1 = "Rising / Receding Flood",      "#4fc3f7"
                st2, c2 = f"Network: {net*100:.0f}% density","#ce93d8"
                st3, c3 = "Transitional hyphal activity",    "#80deea"
            elif fire > 0.5:
                st1, c1 = "Drought + Fire Risk",           "#ff7043"
                st2, c2 = "Hyphae retreating to spring core","#ff8a65"
                st3, c3 = f"Sporocarps: {spore*100:.0f}% max","#e1bee7"
            else:
                st1, c1 = "Dry — Network Contracting",    "#ff8a65"
                st2, c2 = f"Sporocarp dispersal active",      "#ce93d8"
                st3, c3 = "Armadillos spread spores",         "#a5d6a7"

            for yoff, txt, col in [(80, st1, c1), (104, st2, c2), (128, st3, c3)]:
                svg.append(f'<text x="12" y="{yoff}" font-size="15" fill="{col}" font-weight="bold">')
                svg.append(txt)
                svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                           f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append('</text>')

        # Static legend
        svg.append('<text x="12" y="156" fill="#546e7a" font-size="15" font-weight="bold">Legend:</text>')
        entries = [
            (22, 176, 6, "#2e7d32",  "Buriti palm"),
            (22, 198, 4, "#ce93d8",  "Mycorrhizal hub"),
            (22, 218, 4, "#0288d1",  "Flood water table"),
            (22, 238, 4, "#e1bee7",  "Sporocarp dot"),
        ]
        for ex, ey, er, ec, elabel in entries:
            svg.append(f'<circle cx="{ex}" cy="{ey}" r="{er}" fill="{ec}" opacity="0.9"/>')
            svg.append(f'<text font-weight="bold" x="36" y="{ey + 5}" fill="{ec}" font-size="15">{elabel}</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Buriti ↔ Fungo-Ectomicorrízico (Flood-Pulse Clock) on {CONFIG.device}...")

    sim = BuritiFungusSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_nutrients_exchanged:,.0f} nutrient units exchanged, "
          f"peak network density {sim.peak_network_density:.2f}, "
          f"{sim.total_spore_dispersal_frames:,} spore dispersal events.")

    print("Generating SVG...")
    renderer = BuritiFungusRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_73')
    return svg_content


if __name__ == "__main__":
    main()
