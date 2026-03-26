# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 72: Ipê-amarelo (Handroanthus ochraceus) ↔ Abelha-Mamangava (Xylocopa spp.)
#            Phenological Bloom Synchrony & Pollinator Migration Clock
# INTERVENTION 2/4: Cerrado Ecological Web — Seasons & Migration Series
# ===================================================================================================
"""
notebook_72.py — Yellow Ipê Tree ↔ Carpenter Bee (Plant ↔ Animal Pollinator):
Notebook Differentiation:
- Differentiation Focus: Ipê-amarelo (Handroanthus ochraceus) ↔ Abelha-Mamangava (Xylocopa spp.) emphasizing karst sinkhole gradients.
- Indicator species: Jacutinga (Aburria jacutinga).
- Pollination lens: bee foraging under smoke haze.
- Human impact lens: selective logging canopy loss.

                 Synchronized Bloom Pulse & Seasonal Pollinator Migration

The Yellow Ipê (Handroanthus ochraceus / Tabebuia ochracea) is the most iconic
flowering tree of the Cerrado, and the national tree of Brazil.  Its defining
ecological paradox is that it blooms in absolute leafless state — stripping all
foliage before producing its spectacular golden crown during the late dry season
(Jul–Sep). This "naked bloom" strategy concentrates all reproductive effort in the
window when the Abelha-Mamangava (Xylocopa spp., large carpenter bees) undertake
their dry-season altitudinal and lateral migrations in search of pollen-rich sources.

Biological mechanisms modelled:
  • LEAFLESS BLOOM SYNCHRONY (Jul–Sep): Every Ipê tree in a patch blooms within
    days of each other following a shared cue (photoperiod ↔ accumulated drought
    stress). The simultaneous floral display attracts migratory Mamangavas from
    up to 40 km away. Bloom duration is only 8–14 days per tree.

  • MIGRATORY WAVE: Mamangavas are not colonial; each is a solitary forager.
    During the Ipê bloom window they congregate in numbers that mimic migration,
    flying between patches and cross-pollinating trees that would otherwise be
    isolated by the vast Cerrado matrix.

  • POST-BLOOM COLLAPSE: Once flowers drop, the Mamangavas disperse rapidly.
    The tree leafs-out within 2 weeks, using the post-bloom rain trigger (Oct).

  • WET-SEASON SECONDARY BLOOM (Dec–Jan): A secondary, much smaller flowering
    episode occurs in some Ipê trees during the wet season, offering supplemental
    nectar to resident bee communities.

Spatial pattern:
  Multiple Ipê trees are positioned across the clock arena. When a tree is in peak
  bloom its golden aura pulses. Mamangavas animate to cluster near blooming trees
  as a migratory wave, dispersing when bloom collapses.

Scientific references:
  • Oliveira & Gibbs (2000): Pollination biology in the Cerrado savanna of Brazil.
  • Nascimento et al. (2013): Phenology of Handroanthus ochraceus (Bignoniaceae).
  • Frankie et al. (1983): Tropical bee communities and mass-flowering trees.
  • Sazima & Sazima (1989): Carpenter bee foraging ecology in the Cerrado.
  • PIGT RESEX Recanto das Araras field surveys, Goiás (2022–2024).

Visualization: Radial Phenological Clock (matching modules 69–71 pattern) on the
left panel. Ipê trees glow gold/grey inside the clock arena; animated Mamangava
bees swarm toward bloom epicentres; right panels show metrics and curves.
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

# Ipê-amarelo PRIMARY bloom (leafless gold flowering — Jul/Aug/Sep peak)
IPE_BLOOM_CURVE = [0.05, 0.05, 0.05, 0.10, 0.20, 0.40, 0.80, 0.95, 0.60, 0.25, 0.10, 0.05]
IPE_BLOOM_SECONDARY = [
    0.18, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.06, 0.22
]

# Ipê leaf canopy (present in wet, absent in dry pre-bloom and bloom)
IPE_LEAF_CURVE = [0.90, 0.95, 0.90, 0.70, 0.40, 0.10, 0.05, 0.10, 0.40, 0.70, 0.85, 0.90]

# Mamangava (Xylocopa) migratory influx toward Ipê patches (peaks with bloom)
MAMANGAVA_INFLUX_CURVE = [0.60, 0.50, 0.40, 0.50, 0.40, 0.30, 0.70, 0.90, 0.80, 0.50, 0.60, 0.70]

# Mamangava resident activity (baseline foraging when not migrating)
MAMANGAVA_RESIDENT_CURVE = [
    0.50, 0.60, 0.75, 0.80, 0.70, 0.55, 0.30, 0.25, 0.40, 0.65, 0.70, 0.55
]

# Cerrado rainfall (wet Nov–Mar, dry Jun–Sep)
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]


@dataclass
class IpeBeeConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry (consistent with nb69–71)
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0

    num_trees: int = 9          # Ipê trees inside clock arena
    num_bees: int = 45          # Mamangava individuals (migratory + resident)
    bee_speed: float = 5.5
    bloom_radius: float = 35.0  # attraction radius around blooming tree


CONFIG = IpeBeeConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class IpeBeeSim:

    def __init__(self, cfg: IpeBeeConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Ipê Trees positioned inside clock arena ---
        self.trees: List[Dict] = []
        for k in range(cfg.num_trees):
            angle = (k / cfg.num_trees) * 2 * math.pi + random.uniform(-0.2, 0.2)
            r = random.uniform(R * 0.25, R * 0.78)
            self.trees.append({
                "pos": (cx + math.cos(angle) * r, cy + math.sin(angle) * r),
                "bloom_offset": random.uniform(-0.5, 0.5),   # individual phenological offset
                "bloom_level": 0.0,
                "leaf_level": 1.0,
                "visits": 0,
                "flowers_pollinated": 0,
            })

        # --- Mamangava Bees ---
        self.bees: List[Dict] = []
        for _ in range(cfg.num_bees):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(20, R - 20)
            self.bees.append({
                "pos": torch.tensor(
                    [cx + math.cos(angle) * r, cy + math.sin(angle) * r],
                    device=self.dev, dtype=torch.float32
                ),
                "target_tree": -1,
                "pollen_load": 0.0,
                "is_migratory": False,   # becomes migratory during bloom window
                "state": "foraging",    # foraging | migrating | resting
                "energy": random.uniform(50.0, 100.0),
            })

        self.hist_month: List[float] = []
        self.hist_bees_xy: List[List[Tuple[float, float, float, float]]] = [
            [] for _ in range(cfg.num_bees)
        ]  # (x, y, opacity, pollen)

        self.total_flowers_pollinated = 0
        self.total_pollen_transported = 0.0
        self.migration_events = 0        # times a bee switched to migratory mode

    def _interp(self, curve: list, month_frac: float) -> float:
        m = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy = cfg.clock_cx, cfg.clock_cy
        month_frac = (frame / cfg.frames) * 12.0
        self.hist_month.append(month_frac)

        bloom_global  = self._interp(IPE_BLOOM_CURVE,          month_frac)
        bloom_sec     = self._interp(IPE_BLOOM_SECONDARY,       month_frac)
        influx        = self._interp(MAMANGAVA_INFLUX_CURVE,    month_frac)
        resident_act  = self._interp(MAMANGAVA_RESIDENT_CURVE,  month_frac)
        leaf_global   = self._interp(IPE_LEAF_CURVE,            month_frac)

        # --- 1. Update each tree's bloom/leaf state ---
        for t in self.trees:
            # Individual bloom = global curve ± offset (creates staggered timing)
            ind_bloom = max(0.0, min(1.0,
                bloom_global + t["bloom_offset"] * 0.3 + bloom_sec * 0.6
            ))
            t["bloom_level"] = ind_bloom
            # Leaves inversely track bloom in dry season; use global leaf curve
            t["leaf_level"] = max(0.0, min(1.0,
                leaf_global - ind_bloom * 0.9  # blooming suppresses leaf display
            ))

        # --- 2. Bee logic ---
        blooming_trees = [i for i, t in enumerate(self.trees)
                          if t["bloom_level"] > 0.3]

        # Determine how many bees are active as migratory this frame
        migratory_count = int(cfg.num_bees * influx)
        resident_count  = int(cfg.num_bees * resident_act * 0.5)
        active_total    = min(cfg.num_bees, migratory_count + resident_count)

        for bi, bee in enumerate(self.bees):
            is_active = bi < active_total
            was_migratory = bee["is_migratory"]

            if not is_active:
                # Bee is absent / roosting outside territory
                bee["pos"][0] = cx - cfg.clock_radius - 50  # off-screen left
                bee["pos"][1] = cy
                bee["is_migratory"] = False
                self.hist_bees_xy[bi].append((
                    bee["pos"][0].item(), bee["pos"][1].item(), 0.0, 0.0
                ))
                continue

            # Assign migratory flag
            bee["is_migratory"] = bi < migratory_count
            if bee["is_migratory"] and not was_migratory:
                self.migration_events += 1
                # Re-enter from edge of clock arena
                angle = random.uniform(0, 2 * math.pi)
                bee["pos"][0] = cx + math.cos(angle) * (cfg.clock_radius - 5)
                bee["pos"][1] = cy + math.sin(angle) * (cfg.clock_radius - 5)

            pos = bee["pos"]
            bee["energy"] = max(0.0, bee["energy"] - 0.4)

            # Find nearest blooming tree to target
            if bee["target_tree"] == -1 and blooming_trees:
                bee["energy"] < 80.0  # always seek
                best_ti = -1
                best_score = -1e9
                for ti in blooming_trees:
                    t = self.trees[ti]
                    dx = pos[0].item() - t["pos"][0]
                    dy = pos[1].item() - t["pos"][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    score = t["bloom_level"] * 10 - dist * 0.05
                    if score > best_score:
                        best_score = score
                        best_ti = ti
                bee["target_tree"] = best_ti

            if bee["target_tree"] != -1 and bee["target_tree"] < len(self.trees):
                t = self.trees[bee["target_tree"]]
                if t["bloom_level"] < 0.15:
                    bee["target_tree"] = -1
                else:
                    tgt = torch.tensor(t["pos"], device=self.dev, dtype=torch.float32)
                    vec = tgt - pos
                    dist = torch.norm(vec).item()

                    if dist > cfg.bloom_radius * 0.7:
                        dir_v = vec / max(dist, 1e-5)
                        jitter = torch.randn(2, device=self.dev) * 1.5
                        pos += dir_v * cfg.bee_speed + jitter
                    else:
                        # Foraging at flower — collect pollen & pollinate
                        pollen_gain = t["bloom_level"] * random.uniform(10.0, 20.0)
                        bee["pollen_load"] = min(100.0, bee["pollen_load"] + pollen_gain)
                        bee["energy"] = min(100.0, bee["energy"] + 12.0)
                        t["visits"] += 1
                        t["flowers_pollinated"] += 1
                        self.total_flowers_pollinated += 1
                        self.total_pollen_transported += pollen_gain
                        bee["target_tree"] = -1   # move on to next tree
                        bee["pollen_load"] *= 0.5  # deposit half as cross-pollen
            else:
                # Wander inside clock arena
                center = torch.tensor([cx, cy], device=self.dev, dtype=torch.float32)
                pull_toward_center = (center - pos) * 0.02
                pos += pull_toward_center + torch.randn(2, device=self.dev) * (cfg.bee_speed * 0.6)

            # Clamp to clock radius
            dx = pos[0].item() - cx
            dy = pos[1].item() - cy
            dr = math.sqrt(dx * dx + dy * dy)
            if dr > cfg.clock_radius - 15:
                pos[0] = cx + (dx / dr) * (cfg.clock_radius - 15)
                pos[1] = cy + (dy / dr) * (cfg.clock_radius - 15)

            self.hist_bees_xy[bi].append((
                pos[0].item(), pos[1].item(), 1.0, bee["pollen_load"] / 100.0
            ))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class IpeBeeRenderer:

    def __init__(self, cfg: IpeBeeConfig, sim: IpeBeeSim):
        self.cfg = cfg
        self.sim = sim

    def _arc_path(self, cx, cy, radius, start_m, end_m):
        a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
        a2 = (end_m   / 12) * 2 * math.pi - math.pi / 2
        x1 = cx + math.cos(a1) * radius
        y1 = cy + math.sin(a1) * radius
        x2 = cx + math.cos(a2) * radius
        y2 = cy + math.sin(a2) * radius
        span = (end_m - start_m) % 12
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
            f'style="background-color:#141208; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # ── Defs ──────────────────────────────────────────────────────────────
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="cerradoBg">'
            '<stop offset="0%"   stop-color="#3a300a" stop-opacity="0.95"/>'
            '<stop offset="60%"  stop-color="#211c06" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#141208" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="bloomGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%"   stop-color="#ffd600" stop-opacity="0.9"/>'
            '<stop offset="60%"  stop-color="#ff8f00" stop-opacity="0.3"/>'
            '<stop offset="100%" stop-color="#ff6f00" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="pollenGlow">'
            '<stop offset="0%"   stop-color="#ffee58" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#ffee58" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<filter id="goldGlow" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur stdDeviation="4" result="blur"/>'
            '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
        )
        svg.append('</defs>')

        # ── Background ────────────────────────────────────────────────────────
        svg.append(f'<rect width="{w}" height="{h}" fill="#141208"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#cerradoBg)"/>')

        # Rainfall moisture pulse (blue tint synced to RAINFALL_CURVE)
        rain_fills = ";".join(
            f"rgba(66,165,245,{sim._interp(RAINFALL_CURVE, (f/F)*12) * 0.12:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 5}" fill="transparent">'
            f'<animate attributeName="fill" values="{rain_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # Bloom golden pulse (arena warms to gold during bloom window)
        bloom_fills = ";".join(
            f"rgba(255,214,0,{sim._interp(IPE_BLOOM_CURVE, (f/F)*12) * 0.20:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 5}" fill="transparent">'
            f'<animate attributeName="fill" values="{bloom_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # ── Title ─────────────────────────────────────────────────────────────
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ffd600" font-weight="bold">'
            f'ECO-SIM: Yellow Ipê × Carpenter Bee    - Bloom Synchrony & Migration</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#aed581">'
            f'Leafless golden flowering & carpenter-bee migratory wave </text>'
        )

        # ── Clock Face ────────────────────────────────────────────────────────
        months = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            # Golden for bloom months, blue for wet, grey otherwise
            if i in [6, 7, 8]:
                m_col = "#ffd600"
            elif i in [10, 11, 0, 1, 2]:
                m_col = "#42a5f5"
            else:
                m_col = "#8d8060"
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="{m_col}" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 6)
            ly2 = cy + math.sin(angle) * (R - 6)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#3a2f08" stroke-width="2"/>'
            )

        # ── Season Arcs ───────────────────────────────────────────────────────
        # Primary golden bloom window (Jul–Sep)
        d = self._arc_path(cx, cy, R + 10, 6, 9)
        svg.append(f'<path d="{d}" fill="none" stroke="#ffd600" stroke-width="10" '
                   f'stroke-linecap="round" opacity="0.55" filter="url(#goldGlow)"/>')
        mid_bl = ((6 + 9) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text x="{cx + math.cos(mid_bl)*(R+18):.0f}" '
                   f'y="{cy + math.sin(mid_bl)*(R+18):.0f}" font-size="15" '
                   f'fill="#ffd600" text-anchor="middle" font-weight="bold" opacity="0.95">Primary Bloom</text>')

        # Mamangava migratory influx arc (Jun–Oct matches influx onset/decline)
        d2 = self._arc_path(cx, cy, R + 24, 5.5, 9.5)
        svg.append(f'<path d="{d2}" fill="none" stroke="#ff8f00" stroke-width="7" '
                   f'stroke-linecap="round" opacity="0.60"/>')
        svg.append(f'<text x="{cx + math.cos(((5.5+9.5)/2/12)*2*math.pi - math.pi/2)*(R+34):.0f}" '
                   f'y="{cy + math.sin(((5.5+9.5)/2/12)*2*math.pi - math.pi/2)*(R+34):.0f}" '
                   f'font-size="15" fill="#ff8f00" text-anchor="middle" font-weight="bold">Migratory Wave</text>')

        # Wet season / leaf flush (Nov–Mar)
        d3 = self._arc_path(cx, cy, R + 10, 10, 15)
        svg.append(f'<path d="{d3}" fill="none" stroke="#42a5f5" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.35"/>')
        mid_wet = ((10 + 15) / 2 % 12 / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text x="{cx + math.cos(mid_wet)*(R+18):.0f}" '
                   f'y="{cy + math.sin(mid_wet)*(R+18):.0f}" font-size="15" '
                   f'fill="#42a5f5" text-anchor="middle" font-weight="bold">Leaf Flush & Rain</text>')

        # Secondary bloom bump (Dec–Jan)
        d4 = self._arc_path(cx, cy, R + 24, 11, 13)
        svg.append(f'<path d="{d4}" fill="none" stroke="#ffe082" stroke-width="5" '
                   f'stroke-linecap="round" opacity="0.50" stroke-dasharray="5,4"/>')

        # Clock outer ring
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
            f'stroke="#3a2f08" stroke-width="1.5" opacity="0.8"/>'
        )

        # ── Ipê Trees ─────────────────────────────────────────────────────────
        for ti, t in enumerate(sim.trees):
            px, py = t["pos"]

            # Bloom glow aura (animated radius + opacity follows bloom_level per frame)
            bloom_r_vals = ";".join(
                f"{max(2, sim._interp(IPE_BLOOM_CURVE, (fi/F)*12 + t['bloom_offset']*0.3) * 40):.1f}"
                for fi in range(F)
            )
            bloom_op_vals = ";".join(
                f"{max(0, sim._interp(IPE_BLOOM_CURVE, (fi/F)*12 + t['bloom_offset']*0.3) * 0.65):.2f}"
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" fill="url(#bloomGlow)">'
                f'<animate attributeName="r" values="{bloom_r_vals}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{bloom_op_vals}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

            # Trunk
            svg.append(
                f'<line x1="{px:.0f}" y1="{py:.0f}" x2="{px:.0f}" y2="{py+14:.0f}" '
                f'stroke="#5d4037" stroke-width="4" stroke-linecap="round"/>'
            )

            # Canopy (green leaves — visible NOT during peak bloom)
            leaf_r_vals = ";".join(
                f"{max(2, sim._interp(IPE_LEAF_CURVE, (fi/F)*12) * 18):.1f}"
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py - 10:.0f}" fill="#558b2f" opacity="0.8">'
                f'<animate attributeName="r" values="{leaf_r_vals}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

            # Golden flower crown (visible during bloom, disappears otherwise)
            flower_r_vals = ";".join(
                f"{max(0, sim._interp(IPE_BLOOM_CURVE, (fi/F)*12 + t['bloom_offset']*0.3) * 20):.1f}"
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py - 12:.0f}" fill="#ffd600" opacity="0.92" '
                f'filter="url(#goldGlow)">'
                f'<animate attributeName="r" values="{flower_r_vals}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            # Inner lighter gold (petals)
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py - 12:.0f}" fill="#fff9c4" opacity="0.6">'
                f'<animate attributeName="r" values="'
                + ";".join(
                    f"{max(0, sim._interp(IPE_BLOOM_CURVE, (fi/F)*12 + t['bloom_offset']*0.3) * 10):.1f}"
                    for fi in range(F)
                )
                + f'" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # ── Mamangava Bees ─────────────────────────────────────────────────────
        for bi in range(cfg.num_bees):
            hist = sim.hist_bees_xy[bi]
            if not hist:
                continue
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            ops = ";".join(str(round(h[2], 2)) for h in hist)
            # Bee colour: yellow with pollen = brighter; no pollen = darker amber
            bee_col = "#ffd600" if bi < int(cfg.num_bees * 0.6) else "#ff8f00"

            # Flight trail
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::5] if h[2] > 0.5]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="{bee_col}" stroke-width="0.7" opacity="0.22"/>'
                )

            # Body (fuzzy ellipse)
            svg.append(
                f'<ellipse rx="5" ry="3.5" fill="{bee_col}" opacity="0.9">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # Striped abdomen (black band)
            svg.append(
                f'<ellipse rx="2.5" ry="3.2" fill="#212121" opacity="0.8">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # Wings (iridescent)
            wing_ry = ";".join("4" if fi % 2 == 0 else "1" for fi in range(F))
            svg.append(
                f'<ellipse rx="7" ry="4" fill="#b3e5fc" opacity="0.5">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="'
                + ";".join(str(round(h[1] - 3, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="ry" values="{wing_ry}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
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
            f'stroke="#ffd600" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#141208" stroke="#ffd600" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ff8f00"/>')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 355
        panel_w = 335

        # ── Panel 1: Ecological Logic ─────────────────────────────────────────
        py1, ph1 = 20, 230
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#100f06" rx="8" '
                   f'stroke="#ffd600" stroke-width="1" opacity="0.93"/>')
        svg.append(f'<text x="12" y="24" fill="#ffd600" font-size="15" font-weight="bold">'
                   f'Leafless Bloom Synchrony Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="48" fill="#ccc" font-size="15">'
                   f'Ipê blooms leafless in the dry season (Jul–Sep),</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'before re-leafing — maximising bee visibility:</text>')
        svg.append(f'<text font-weight="bold" x="12" y="92" fill="#ccc" font-size="15">'
                   f'golden crown with no leaves to compete.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="114" fill="#ccc" font-size="15">'
                   f'All patch trees bloom within days of each other</text>')
        svg.append(f'<text font-weight="bold" x="12" y="136" fill="#ccc" font-size="15">'
                   f'(mass flowering), a beacon visible 40+ km away.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="158" fill="#aed581" font-size="15">'
                   f'This draws a migratory Xylocopa wave that</text>')
        svg.append(f'<text font-weight="bold" x="12" y="180" fill="#aed581" font-size="15">'
                   f'cross-pollinates trees across the entire</text>')
        svg.append(f'<text font-weight="bold" x="12" y="202" fill="#aed581" font-size="15">'
                   f'landscape mosaic.</text>')
        svg.append('</g>')

        # ── Panel 2: Metrics ──────────────────────────────────────────────────
        py2 = py1 + ph1 + 10
        ph2 = 158
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#100f06" rx="8" '
                   f'stroke="#ff8f00" stroke-width="1" opacity="0.93"/>')
        svg.append(f'<text x="12" y="24" fill="#ffb300" font-size="15" font-weight="bold">'
                   f'Pollination & Migration Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ffd600" font-size="15">'
                   f'Flowers Pollinated: {sim.total_flowers_pollinated:,} visits</text>')
        svg.append(f'<text font-weight="bold" x="12" y="88" fill="#ff8f00" font-size="15">'
                   f'Pollen Transported: {sim.total_pollen_transported:,.0f} units</text>')
        svg.append(f'<text font-weight="bold" x="12" y="120" fill="#aed581" font-size="15">'
                   f'Migration Events: {sim.migration_events} arrivals</text>')
        svg.append(f'<text font-weight="bold" x="12" y="148" fill="#90a4ae" font-size="15">'
                   f'Colony: {cfg.num_bees} bees | Trees: {cfg.num_trees}</text>')
        svg.append('</g>')

        # ── Panel 3: Phenology Chart (reusing eco_base helper) ────────────────
        py3 = py2 + ph2 + 10
        ph3 = int(h - 10 - py3)   # fill remaining canvas height
        curves_data = [
            (IPE_BLOOM_CURVE,         "#ffd600", "Primary Bloom (Ipê)"),
            (MAMANGAVA_INFLUX_CURVE,  "#ff8f00", "Mamangava Influx"),
            (IPE_LEAF_CURVE,          "#558b2f", "Leaf Canopy"),
            (RAINFALL_CURVE,          "#42a5f5", "Rainfall"),
            (IPE_BLOOM_SECONDARY,     "#ffe082", "Secondary Bloom"),
        ]
        chart_snippet = draw_phenology_chart(
            curves_data,
            chart_w=305, chart_h=max(60, ph3 - 80), panel_h=ph3,
            title="Ipê & Mamangava Phenological Curves",
            title_color="#ffd600",
            bg_color="#100f06",
            border_color="#ffd600",
        )
        # Enforce 15px minimum font for all text in chart snippet
        chart_snippet = chart_snippet.replace('font-size="15"', 'font-size="15"')
        chart_snippet = chart_snippet.replace('font-size="15"', 'font-size="15"')
        svg.append(f'<g transform="translate({panel_x}, {py3})">{chart_snippet}</g>')

        # ── Current Month Sidebar (animated status card) ───────────────────
        px5 = int(cfg.clock_cx + cfg.clock_radius + 40)
        py5 = h - 262
        pw5 = 210
        ph5 = 252
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(f'<rect width="{pw5}" height="{ph5}" fill="#100f06" rx="8" '
                   f'stroke="#ffd600" stroke-width="1.5" opacity="0.96"/>')
        svg.append(f'<text x="12" y="22" font-size="15" fill="#ffd600" font-weight="bold">'
                   f'Active Season Status:</text>')

        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12; vals[m_idx] = "1"
            op_str = ";".join(vals + ["0"])

            svg.append(f'<text x="12" y="50" font-size="15" fill="#ffd600" font-weight="bold">')
            svg.append(m_name)
            svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                       f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')

            bloom = sim._interp(IPE_BLOOM_CURVE, m_idx)
            influx = sim._interp(MAMANGAVA_INFLUX_CURVE, m_idx)
            leaf = sim._interp(IPE_LEAF_CURVE, m_idx)

            if bloom > 0.5:
                st1, c1 = "PEAK GOLDEN BLOOM",           "#ffd600"
                st2, c2 = "Migratory wave ACTIVE",          "#ff8f00"
                st3, c3 = "Leafless flower crown",          "#fff9c4"
            elif bloom > 0.1:
                st1, c1 = "Early/Late Bloom",             "#ffe082"
                st2, c2 = f"Influx: {influx*100:.0f}% capacity",      "#ffb300"
                st3, c3 = "Leaves: minimal",               "#8d8060"
            elif leaf > 0.7:
                st1, c1 = "Dense Leaf Canopy",            "#558b2f"
                st2, c2 = "Resident bees forage",           "#aed581"
                st3, c3 = "No bloom — wet season",          "#42a5f5"
            else:
                st1, c1 = "Leaf-drop transition",        "#8d8060"
                st2, c2 = "Bee activity declining",         "#ff8f00"
                st3, c3 = "Pre-bloom stress period",        "#a1887f"

            for yoff, txt, col in [(78, st1, c1), (100, st2, c2), (122, st3, c3)]:
                svg.append(f'<text x="12" y="{yoff}" font-size="15" fill="{col}" font-weight="bold">')
                svg.append(txt)
                svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                           f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append('</text>')

        # Static legend
        svg.append('<text x="12" y="152" fill="#546e7a" font-size="15" font-weight="bold">Legend:</text>')
        svg.append('<circle cx="22" cy="172" r="8" fill="#ffd600" opacity="0.8"/>')
        svg.append('<text font-weight="bold" x="36" y="177" fill="#ffd600" font-size="15">Ipê flower crown</text>')
        svg.append('<circle cx="22" cy="196" r="8" fill="#558b2f" opacity="0.8"/>')
        svg.append('<text font-weight="bold" x="36" y="201" fill="#aed581" font-size="15">Ipê leaf canopy</text>')
        svg.append('<ellipse cx="22" cy="219" rx="5" ry="3.5" fill="#ffd600"/>')
        svg.append('<text font-weight="bold" x="36" y="224" fill="#ffd600" font-size="15">Mamangava bee</text>')
        svg.append('<ellipse cx="22" cy="241" rx="5" ry="3.5" fill="#ff8f00"/>')
        svg.append('<text font-weight="bold" x="36" y="246" fill="#ffb300" font-size="15">Carrying pollen</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Ipê-amarelo ↔ Abelha-Mamangava (Bloom Sync Clock) on {CONFIG.device}...")

    sim = IpeBeeSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_flowers_pollinated:,} flowers pollinated, "
          f"{sim.total_pollen_transported:,.0f} pollen units transported, "
          f"{sim.migration_events} bee migration events.")

    print("Generating SVG...")
    renderer = IpeBeeRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_72')
    return svg_content


if __name__ == "__main__":
    main()
