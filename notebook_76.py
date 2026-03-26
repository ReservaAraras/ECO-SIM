# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 76: Gavião-real (Harpia harpyja) ↔ Macaco-prego (Sapajus libidinosus)
#            Apex Predation · Seasonal Patch Use · Multi-species Alarm Cascade
# INTERVENTION 2/4: Cerrado Trophic Cascade & Seasonal Connectivity Series
# ===================================================================================================
"""
notebook_76.py — Harpy Eagle ↔ Black-striped Capuchin (Animal ↔ Animal):
Notebook Differentiation:
- Differentiation Focus: Gavião-real (Harpia harpyja) ↔ Macaco-prego (Sapajus libidinosus) emphasizing campo rupestre mosaics.
- Indicator species: Jaguatirica (Leopardus pardalis).
- Pollination lens: ant-guarded nectar dynamics.
- Human impact lens: restoration weeding benefits.

              Apex Predation Pressure, Seasonal Monkey Patch Shifts
              & Multi-level Alarm Response in Cerrado Gallery Forests

The Harpy Eagle (Harpia harpyja) is the largest and most powerful raptor in the
Americas, and one of the top apex predators of Neotropical forests.  Within the
RESEX Recanto das Araras landscape (Goiás), gallery forests along the Lapa River
and its tributaries provide residual nesting and hunting habitat.

The Black-striped Capuchin (Sapajus libidinosus) is its primary documented prey
in the Cerrado-transition zone — but the relationship is far richer than simple
predator/prey:

SEASONAL DYNAMICS at  (Goiás):

  WET SEASON (Nov–Mar) — FRAGMENT ISOLATION:
    Floodwaters isolate forest patches.  Capuchin troops are confined to elevated
    islands of gallery forest.  Harpy Eagle hunting success peaks because monkey
    escape routes are blocked by flooded corridors.  Troop cohesion increases
    (anti-predator flocking), vigilance time rises to 35–45 % of daily activity,
    and foraging efficiency drops sharply.

  DRY SEASON (Apr–Oct) — LANDSCAPE CONNECTIVITY:
    As water recedes, capuchins expand into open Cerrado, cerradão, and rocky
    outcrops — a 3↔ expansion of home-range area has been recorded (Fragaszy 2004).
    Harpy typically follows troop movements but hunts with lower success in open
    terrain.  Capuchin groups adopt spread-out foraging, reducing collision risk.

  ALARM CASCADE (Multi-species phenomenon):
    When a Harpy Eagle is detected, capuchin alarm calls trigger a cascade:
      – Other capuchin groups within 300 m enter high-vigilance immediately.
      – Seriemas (nb71) on the ground interpret the aerial alarm and freeze/flush.
      – The wolf (nb75) resting near gallery forest margins also responds by moving
        into denser cover — linking all four notebooks via a shared alarm signal.
    This cross-species alarm propagation is one of the most important functional
    links in the Cerrado multi-trophic web.

  MIGRATORY / RANGING ASPECT:
    Harpy Eagles at the RESEX are NOT migratory but they are NOMADIC hunters —
    a breeding pair patrols a 30–100 km² territory in 2–5 day circuits.
    Their seasonal presence over any given forest patch follows a detectability
    curve that peaks in the dry season (higher visibility through bare canopy,
    more open-ground hunting opportunities near capuchin foraging areas).

Scientific references:
  • Miranda et al. (2019): Harpy eagle diet and prey biomass in Cerrado edges.
  • Fragaszy et al. (2004): The Complete Capuchin — habitat use and ranging.
  • Boinski & Campbell (1995): Use of tonal vocalisations for alarm and vigilance.
  • Peres (1993): Anti-predatory responses of capuchin monkeys to aerial predators.
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

# Harpy Eagle hunting success rate in RESEX territory (peak dry-season / open terrain)
HARPY_HUNT_CURVE = [
    0.35, 0.30, 0.30, 0.45, 0.60, 0.75, 0.85, 0.90, 0.80, 0.65, 0.40, 0.35
]

# Harpy detectability / presence over gallery forest patch (nomadic circuit)
HARPY_PRESENCE_CURVE = [
    0.50, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90, 0.90, 0.80, 0.70, 0.55, 0.50
]

# Capuchin patch isolation index (1 = confined to gallery, 0 = open landscape)
MONKEY_ISOLATION_CURVE = [
    0.80, 0.90, 0.75, 0.55, 0.30, 0.10, 0.05, 0.05, 0.15, 0.35, 0.65, 0.75
]

# Capuchin troop vigilance fraction (% of day in anti-predator scanning)
MONKEY_VIGILANCE_CURVE = [
    0.42, 0.45, 0.38, 0.30, 0.22, 0.18, 0.15, 0.15, 0.20, 0.28, 0.38, 0.42
]

# Alarm cascade intensity (cross-species propagation following eagle detection)
ALARM_CASCADE_CURVE = [
    0.40, 0.35, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.85, 0.70, 0.45, 0.38
]

# Capuchin home-range expansion index (1 = max open-cerrado expansion in dry)
MONKEY_RANGE_CURVE = [
    0.30, 0.25, 0.35, 0.55, 0.75, 0.95, 1.00, 1.00, 0.85, 0.65, 0.40, 0.30
]

# Shared rainfall backbone (links to nb75/77/78)
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]

# Gallery forest food availability for capuchins (fruit + invertebrates)
MONKEY_FOOD_CURVE = [
    0.70, 0.75, 0.65, 0.50, 0.40, 0.30, 0.25, 0.28, 0.40, 0.55, 0.70, 0.75
]


@dataclass
class HarpyCfg:
    width:  int   = 1280
    height: int = CANVAS_HEIGHT
    frames: int   = 360
    fps:    int   = 10
    device: str   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry (consistent with nb71–75)
    clock_cx:     float = 420.0
    clock_cy:     float = 310.0
    clock_radius: float = 240.0

    num_harpy:       int   = 2    # harpy eagle pair (nomadic circuit)
    num_troops:      int   = 4    # capuchin troops
    monkeys_per_troop: int = 8    # individuals per troop
    harpy_speed:     float = 7.5  # soaring raptor
    monkey_speed:    float = 4.8
    alarm_radius:    float = 160.0  # cross-troop alarm propagation range
    strike_radius:   float = 20.0


CONFIG = HarpyCfg()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class HarpyCapuchinSim:
    """Simulates apex-predation pressure, seasonal patch use,
    vigilance responses, and the multi-species alarm cascade."""

    def __init__(self, cfg: HarpyCfg):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy, R = cfg.clock_cx, cfg.clock_cy, cfg.clock_radius

        # ── Capuchin troops ───────────────────────────────────────────────────
        # Each troop has a home territory angle and a centroid position
        self.troops: List[Dict] = []
        for t in range(cfg.num_troops):
            angle    = (t / cfg.num_troops) * 2 * math.pi
            base_r   = random.uniform(R * 0.35, R * 0.65)
            troop_cx = cx + math.cos(angle) * base_r
            troop_cy = cy + math.sin(angle) * base_r
            members  = []
            for _ in range(cfg.monkeys_per_troop):
                mx = troop_cx + random.uniform(-25, 25)
                my = troop_cy + random.uniform(-20, 20)
                members.append({
                    "pos": torch.tensor([mx, my], device=self.dev, dtype=torch.float32),
                    "alarm_level": 0.0,
                })
            self.troops.append({
                "centroid":       torch.tensor([troop_cx, troop_cy],
                                               device=self.dev, dtype=torch.float32),
                "home_angle":     angle,
                "base_r":         base_r,
                "members":        members,
                "alarm_level":    0.0,         # troop alarm (0–1)
                "vigilance":      0.0,
                "prey_count":     cfg.monkeys_per_troop,
                "state":          "foraging",
            })

        # ── Harpy Eagles (nomadic pair) ───────────────────────────────────────
        self.eagles: List[Dict] = []
        for k in range(cfg.num_harpy):
            angle = k * math.pi
            r     = random.uniform(R * 0.4, R * 0.8)
            self.eagles.append({
                "pos":       torch.tensor(
                                 [cx + math.cos(angle) * r,
                                  cy + math.sin(angle) * r],
                                 device=self.dev, dtype=torch.float32),
                "vel":       torch.zeros(2, device=self.dev, dtype=torch.float32),
                "target_troop": -1,
                "state":     "soaring",
                "kills":     0,
                "circuit_angle": angle,
                "energy":    90.0,
            })

        # History buffers
        self.hist_month:          List[float] = []
        self.hist_eagle_xy:       List[List[Tuple[float,float,float]]] = [
            [] for _ in range(cfg.num_harpy)
        ]
        self.hist_troop_centroid: List[List[Tuple[float,float,float]]] = [
            [] for _ in range(cfg.num_troops)
        ]
        self.hist_alarm_events:   List[Tuple[float,float,int]] = []  # (x, y, frame)
        self.hist_kill_events:    List[Tuple[float,float,int]] = []

        # Aggregate metrics
        self.total_kills         = 0
        self.total_alarm_events  = 0
        self.peak_vigilance      = 0.0
        self.cascade_events      = 0  # frames with alarm spread across ≥2 troops

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

        hunt_success  = self._interp(HARPY_HUNT_CURVE,         month_frac)
        presence      = self._interp(HARPY_PRESENCE_CURVE,     month_frac)
        isolation     = self._interp(MONKEY_ISOLATION_CURVE,   month_frac)
        vigilance_base= self._interp(MONKEY_VIGILANCE_CURVE,   month_frac)
        alarm_str     = self._interp(ALARM_CASCADE_CURVE,      month_frac)
        range_exp     = self._interp(MONKEY_RANGE_CURVE,       month_frac)

        # ── 1. Update capuchin positions (range-expansion in dry season) ──────
        for ti, troop in enumerate(self.troops):
            ha = troop["home_angle"]
            # Effective patrol radius expands in dry season (range_exp)
            eff_r = troop["base_r"] * (0.55 + range_exp * 0.55)
            # Wet-season isolation: troops compress toward gallery patches
            if isolation > 0.6:
                eff_r *= (1.0 - (isolation - 0.6) * 0.5)
            # Slowly drift around territory
            drift_angle = ha + math.sin(frame * 0.015 + ti) * 0.6
            target_x = cx + math.cos(drift_angle) * eff_r
            target_y = cy + math.sin(drift_angle) * eff_r
            tgt = torch.tensor([target_x, target_y], device=self.dev, dtype=torch.float32)
            vec = tgt - troop["centroid"]
            dist = torch.norm(vec).item()
            if dist > 5.0:
                jitter = torch.randn(2, device=self.dev) * 2.0
                troop["centroid"] += (vec / max(dist, 1e-5)) * cfg.monkey_speed * 0.3 + jitter
            # Clamp to arena
            dx_ = troop["centroid"][0].item() - cx
            dy_ = troop["centroid"][1].item() - cy
            dr_ = math.sqrt(dx_*dx_ + dy_*dy_)
            if dr_ > R - 15:
                troop["centroid"][0] = cx + (dx_/dr_) * (R - 15)
                troop["centroid"][1] = cy + (dy_/dr_) * (R - 15)

            # Update vigilance
            troop["vigilance"] = vigilance_base
            troop["alarm_level"] = max(0.0, troop["alarm_level"] - 0.04)
            self.peak_vigilance = max(self.peak_vigilance, troop["vigilance"])

            # Record centroid
            cx_t = troop["centroid"][0].item()
            cy_t = troop["centroid"][1].item()
            opacity = 1.0 if troop["prey_count"] > 0 else 0.2
            self.hist_troop_centroid[ti].append((cx_t, cy_t, opacity))

        # ── 2. Harpy Eagle behaviour ──────────────────────────────────────────
        active_troops = [t for t in self.troops if t["prey_count"] > 0]

        for ki, eagle in enumerate(self.eagles):
            pos = eagle["pos"]

            # Harpy only active when present in territory
            if random.random() > presence:
                # Off circuit — park off-screen but still need to log position
                eagle["state"] = "absent"
                self.hist_eagle_xy[ki].append((pos[0].item(), pos[1].item(), 0.0))
                continue

            eagle["state"] = "soaring"
            speed = cfg.harpy_speed * (0.8 + hunt_success * 0.35)

            # Select target troop (prefer least vigilant, closest)
            eagle["target_troop"] = -1
            if active_troops and random.random() < hunt_success * 0.8:
                best_score = -1e9; best_ti = -1
                for ti, troop in enumerate(active_troops):
                    dx_ = pos[0].item() - troop["centroid"][0].item()
                    dy_ = pos[1].item() - troop["centroid"][1].item()
                    dist_ = math.sqrt(dx_*dx_ + dy_*dy_)
                    # Score: closer + less vigilant = better hunting opportunity
                    score = (1.0 - troop["vigilance"]) * 0.6 - dist_ * 0.003
                    if score > best_score:
                        best_score = score; best_ti = ti
                eagle["target_troop"] = best_ti

            if eagle["target_troop"] != -1:
                troop = active_troops[eagle["target_troop"]]
                tgt   = troop["centroid"].clone()
                vec   = tgt - pos
                dist  = torch.norm(vec).item()

                if dist > cfg.strike_radius:
                    jitter = torch.randn(2, device=self.dev) * (speed * 0.2)
                    pos += (vec / max(dist, 1e-5)) * speed + jitter

                    # Within alarm radius: trigger troop alarm + cascade
                    if dist < cfg.alarm_radius:
                        troop["alarm_level"]  = min(1.0, troop["alarm_level"] + 0.35)
                        troop["vigilance"]    = min(1.0, troop["vigilance"] + 0.25)
                        troop["state"] = "alarm"
                        self.hist_alarm_events.append(
                            (troop["centroid"][0].item(),
                             troop["centroid"][1].item(), frame)
                        )
                        self.total_alarm_events += 1

                        # Cascade to nearby troops
                        affected = 0
                        for other_t in self.troops:
                            if other_t is troop:
                                continue
                            d_troop = torch.norm(
                                other_t["centroid"] - troop["centroid"]
                            ).item()
                            if d_troop < cfg.alarm_radius * 1.8:
                                other_t["alarm_level"] = min(
                                    1.0,
                                    other_t["alarm_level"] + alarm_str * 0.25
                                )
                                other_t["state"] = "fleeing"
                                affected += 1
                        if affected >= 2:
                            self.cascade_events += 1

                else:
                    # Strike!
                    if troop["prey_count"] > 0 and random.random() < hunt_success * 0.6:
                        troop["prey_count"] -= 1
                        eagle["kills"]       += 1
                        self.total_kills     += 1
                        self.hist_kill_events.append(
                            (pos[0].item(), pos[1].item(), frame)
                        )
                        eagle["energy"] = min(100.0, eagle["energy"] + 35.0)
                    eagle["target_troop"] = -1
                    eagle["state"] = "soaring"
            else:
                # Nomadic circuit — slowly sweep the arena
                eagle["circuit_angle"] += 0.018
                r_circ = R * 0.65
                circ_x = cx + math.cos(eagle["circuit_angle"]) * r_circ
                circ_y = cy + math.sin(eagle["circuit_angle"]) * r_circ
                circ_tgt = torch.tensor([circ_x, circ_y], device=self.dev, dtype=torch.float32)
                vec2 = circ_tgt - pos
                dist2 = torch.norm(vec2).item()
                if dist2 > 5.0:
                    jitter = torch.randn(2, device=self.dev) * (speed * 0.15)
                    pos += (vec2 / max(dist2, 1e-5)) * speed * 0.7 + jitter

            # Clamp harpy to arena
            dx_ = pos[0].item() - cx; dy_ = pos[1].item() - cy
            dr_ = math.sqrt(dx_*dx_ + dy_*dy_)
            if dr_ > R - 12:
                pos[0] = cx + (dx_/dr_) * (R - 12)
                pos[1] = cy + (dy_/dr_) * (R - 12)
            pos[0] = pos[0].clamp(20, cfg.width - 400)
            pos[1] = pos[1].clamp(55, cfg.height - 20)

            self.hist_eagle_xy[ki].append((pos[0].item(), pos[1].item(), 1.0))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class HarpyRenderer:

    def __init__(self, cfg: HarpyCfg, sim: HarpyCapuchinSim):
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
            f'style="background-color:#080c10; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # ── Defs ──────────────────────────────────────────────────────────────
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="forestBg">'
            '<stop offset="0%"   stop-color="#1a2635" stop-opacity="0.95"/>'
            '<stop offset="65%"  stop-color="#0d1720" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#080c10" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="harpyGlow">'
            '<stop offset="0%"   stop-color="#b0bec5" stop-opacity="0.80"/>'
            '<stop offset="100%" stop-color="#b0bec5" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="alarmGlow">'
            '<stop offset="0%"   stop-color="#ff1744" stop-opacity="0.85"/>'
            '<stop offset="60%"  stop-color="#ff1744" stop-opacity="0.30"/>'
            '<stop offset="100%" stop-color="#ff1744" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="monkeyGlow">'
            '<stop offset="0%"   stop-color="#8d6e63" stop-opacity="0.75"/>'
            '<stop offset="100%" stop-color="#8d6e63" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<filter id="eagleBlur">'
            '<feGaussianBlur stdDeviation="2.5" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
        )
        svg.append('</defs>')

        # ── Background ────────────────────────────────────────────────────────
        svg.append(f'<rect width="{w}" height="{h}" fill="#080c10"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#forestBg)"/>')

        # Wet-season flood tint (blue wash)
        rain_fills = ";".join(
            f"rgba(13,71,161,{sim._interp(RAINFALL_CURVE, (f/F)*12) * 0.18:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{rain_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # Alarm cascade pulse (red wash when alarm active)
        alarm_fills = ";".join(
            f"rgba(255,23,68,{sim._interp(ALARM_CASCADE_CURVE, (f/F)*12) * 0.10:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{alarm_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # ── Title ─────────────────────────────────────────────────────────────
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#b0bec5" font-weight="bold">'
            f'ECO-SIM: Harpy Eagle × Capuchin Monkey    - Apex Predation & Alarm Cascade Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#ef9a9a">'
            f'Seasonal patch isolation · Hunting pressure · Multi-species alarm propagation</text>'
        )

        # ── Clock Face ────────────────────────────────────────────────────────
        months    = ["JAN","FEB","MAR","APR","MAY","JUN",
                     "JUL","AUG","SEP","OCT","NOV","DEC"]
        # Colour key: blue/steel = wet confinement; red = alarm/predation peak
        month_cols = {
            0:"#4fc3f7", 1:"#29b6f6", 2:"#4fc3f7",   # Jan–Mar: wet/isolated
            3:"#ef9a9a", 4:"#ef5350", 5:"#e53935",   # Apr–Jun: alarm raising
            6:"#e53935", 7:"#ef5350", 8:"#ef9a9a",   # Jul–Sep: peak predation
            9:"#b0bec5", 10:"#90a4ae", 11:"#4fc3f7", # Oct–Dec: transition
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
                f'stroke="#1a2635" stroke-width="2"/>'
            )

        # ── Season Arcs ───────────────────────────────────────────────────────
        # Wet season / patch isolation (Nov–Mar)
        d1 = self._arc_path(cx, cy, R + 11, 10, 3)
        svg.append(f'<path d="{d1}" fill="none" stroke="#1565c0" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.50"/>')
        mid_wet = (((10+3+12)/2 % 12) / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid_wet)*(R+22):.0f}" '
                   f'y="{cy+math.sin(mid_wet)*(R+22):.0f}" font-size="15" '
                   f'fill="#4fc3f7" text-anchor="middle">Patch Isolation</text>')

        # Dry season / peak hunting (May–Sep)
        d2 = self._arc_path(cx, cy, R + 11, 4, 9)
        svg.append(f'<path d="{d2}" fill="none" stroke="#c62828" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.50"/>')
        mid_dry = ((4+9)/2/12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid_dry)*(R+22):.0f}" '
                   f'y="{cy+math.sin(mid_dry)*(R+22):.0f}" font-size="15" '
                   f'fill="#ef5350" text-anchor="middle">Peak Hunting</text>')

        # Alarm cascade season arc (Apr–Oct)
        d3 = self._arc_path(cx, cy, R + 25, 3, 10)
        svg.append(f'<path d="{d3}" fill="none" stroke="#ff1744" stroke-width="5" '
                   f'stroke-linecap="round" opacity="0.45" stroke-dasharray="5,4"/>')
        svg.append(
            f'<text font-weight="bold" x="{cx+math.cos(mid_dry)*(R+38):.0f}" '
            f'y="{cy+math.sin(mid_dry)*(R+38):.0f}" font-size="15" '
            f'fill="#ff1744" text-anchor="middle">Alarm Cascade</text>'
        )

        # Range expansion arc (Jun–Sep = widest foraging)
        d4 = self._arc_path(cx, cy, R + 25, 5.5, 9)
        svg.append(f'<path d="{d4}" fill="none" stroke="#8d6e63" stroke-width="4" '
                   f'stroke-linecap="round" opacity="0.45" stroke-dasharray="4,3"/>')

        # Clock ring
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
                   f'stroke="#1a2635" stroke-width="1.5" opacity="0.9"/>')

        # ── Capuchin troops ────────────────────────────────────────────────────
        troop_cols = ["#a1887f", "#8d6e63", "#795548", "#6d4c41"]
        for ti in range(cfg.num_troops):
            hist = sim.hist_troop_centroid[ti]
            if not hist:
                continue
            col = troop_cols[ti % len(troop_cols)]
            txs = ";".join(str(round(h[0],1)) for h in hist)
            tys = ";".join(str(round(h[1],1)) for h in hist)
            tops= ";".join(str(round(h[2],2)) for h in hist)

            # Patrol trail
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::4]]
            if len(trail_pts) > 3:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="{col}" stroke-width="0.8" stroke-dasharray="2,5" opacity="0.22"/>'
                )

            # Troop glow halo (radius expands when alarm is active)
            alarm_r_vals = ";".join(
                str(round(20 + sim._interp(ALARM_CASCADE_CURVE, (fi/F)*12) * 20, 1))
                for fi in range(F)
            )
            alarm_op_vals = ";".join(
                f"{sim._interp(ALARM_CASCADE_CURVE, (fi/F)*12) * 0.35:.2f}"
                for fi in range(F)
            )
            svg.append(
                f'<circle fill="url(#alarmGlow)" opacity="0">'
                f'<animate attributeName="cx" values="{txs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{tys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="r" values="{alarm_r_vals}" dur="{dur}s" '
                f'repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{alarm_op_vals}" dur="{dur}s" '
                f'repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

            # Troop dot cluster (8 monkeys shown as offset dots)
            for mi in range(cfg.monkeys_per_troop):
                off_angle = (mi / cfg.monkeys_per_troop) * 2 * math.pi
                off_r = 12
                ox = math.cos(off_angle) * off_r
                oy = math.sin(off_angle) * off_r
                m_txs = ";".join(str(round(h[0] + ox, 1)) for h in hist)
                m_tys = ";".join(str(round(h[1] + oy, 1)) for h in hist)
                svg.append(
                    f'<circle r="4" fill="{col}">'
                    f'<animate attributeName="cx" values="{m_txs}" dur="{dur}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="cy" values="{m_tys}" dur="{dur}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="opacity" values="{tops}" dur="{dur}s" '
                    f'repeatCount="indefinite" calcMode="discrete"/>'
                    f'</circle>'
                )

        # ── Alarm pulse rings (at kill events) ────────────────────────────────
        MAX_ALARM_RINGS = min(50, len(sim.hist_alarm_events))
        for ax, ay, af in sim.hist_alarm_events[:MAX_ALARM_RINGS]:
            pulse_ops = []
            for fi in range(F):
                elapsed = (fi - af) % F
                pulse = max(0.0, 0.75 - elapsed * 0.04)
                pulse_ops.append(f"{pulse:.2f}")
            ring_rs = []
            for fi in range(F):
                elapsed = (fi - af) % F
                ring_rs.append(str(round(10 + elapsed * 4, 1)))
            svg.append(
                f'<circle cx="{ax:.0f}" cy="{ay:.0f}" fill="none" '
                f'stroke="#ff1744" stroke-width="1.5" opacity="0">'
                f'<animate attributeName="r" values="{";".join(ring_rs)}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{";".join(pulse_ops)}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # ── Kill markers ──────────────────────────────────────────────────────
        MAX_KILLS = min(30, len(sim.hist_kill_events))
        for kx, ky, kf in sim.hist_kill_events[:MAX_KILLS]:
            k_ops = ["0.0"] * F
            for fi in range(kf, min(kf + 40, F)):
                fade = max(0.0, 1.0 - (fi - kf) / 40)
                k_ops[fi] = f"{fade * 0.90:.2f}"
            svg.append(
                f'<circle cx="{kx:.0f}" cy="{ky:.0f}" r="5" fill="#ff1744" opacity="0">'
                f'<animate attributeName="opacity" values="{";".join(k_ops)}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # ── Harpy Eagles ──────────────────────────────────────────────────────
        eagle_cols = ["#eceff1", "#cfd8dc"]
        for ki in range(cfg.num_harpy):
            hist = sim.hist_eagle_xy[ki]
            if not hist:
                continue
            col = eagle_cols[ki % len(eagle_cols)]
            exs = ";".join(str(round(h[0],1)) for h in hist)
            eys = ";".join(str(round(h[1],1)) for h in hist)
            eops= ";".join(str(round(h[2],2)) for h in hist)

            # Soaring trail
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::4] if h[2] > 0.5]
            if len(trail_pts) > 3:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="{col}" stroke-width="1.0" stroke-dasharray="4,6" opacity="0.32"/>'
                )

            # Eagle body — wide-wingspan silhouette (horizontal ellipse)
            svg.append(
                f'<ellipse rx="14" ry="5" fill="{col}">'
                f'<animate attributeName="cx" values="{exs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{eys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{eops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # Dark-masked head (Harpy has distinctive black facial ruff)
            svg.append(
                f'<circle r="5" fill="#37474f">'
                f'<animate attributeName="cx" values="'
                + ";".join(str(round(h[0]+10,1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{eys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{eops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            # White belly patch
            svg.append(
                f'<ellipse rx="8" ry="3" fill="#fafafa" opacity="0.7">'
                f'<animate attributeName="cx" values="{exs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{eys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{eops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
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
            f'stroke="#b0bec5" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#080c10" stroke="#b0bec5" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ef5350"/>')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 395
        panel_w = 375

        # ── Panel 1: Ecological Logic ──────────────────────────────────────────
        py1, ph1 = 20, 183
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#050810" rx="8" '
                   f'stroke="#b0bec5" stroke-width="1" opacity="0.95"/>')
        svg.append(f'<text x="12" y="22" fill="#b0bec5" font-size="15" font-weight="bold">'
                   f'Apex Predation & Alarm Cascade Strategy</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Harpia harpyja holds a 30–100 km² territory;</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'primary prey: Sapajus libidinosus troops.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#4fc3f7" font-size="15">'
                   f'Wet: flooding isolates patches → confinement</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#4fc3f7" font-size="15">'
                   f'and higher eagle success at patch boundaries.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ef5350" font-size="15">'
                   f'Dry season: capuchins range expands; visible.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#ff1744" font-size="15">'
                   f'Eagle detected: alarm cascade reaches Seriema,</text>')
        svg.append(f'<text font-weight="bold" x="12" y="126" fill="#ff1744" font-size="15">'
                   f'Lobo-guará, and other species.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="140" fill="#90a4ae" font-size="15">'
                   f'</text>')
        svg.append('</g>')

        # ── Panel 3: Phenology Chart ───────────────────────────────────────────
        py3 = py1 + ph1 + 10
        ph3 = 184
        curves_data = [
            (HARPY_HUNT_CURVE,         "#b0bec5", "Harpy Hunting Success"),
            (HARPY_PRESENCE_CURVE,     "#eceff1", "Harpy Presence in Territory"),
            (MONKEY_ISOLATION_CURVE,   "#4fc3f7", "Capuchin Patch Isolation"),
            (MONKEY_VIGILANCE_CURVE,   "#8d6e63", "Capuchin Vigilance Fraction"),
            (ALARM_CASCADE_CURVE,      "#ff1744", "Alarm Cascade Intensity"),
        ]
        chart_snippet = draw_phenology_chart(
            curves_data,
            chart_w=330, chart_h=58, panel_h=ph3,
            title="Predation &amp; Vigilance Phenology",
            title_color="#b0bec5",
            bg_color="#050810",
            border_color="#b0bec5",
            legend_letter_spacing=1.5,
            legend_cols=1,
            legend_row_h=16,
        )
        svg.append(f'<g transform="translate({panel_x}, {py3})">{chart_snippet}</g>')

        # ── Panel 4: Seasonal Range Map ───────────────────────────────────────
        py4 = py3 + ph3 + 10
        ph4 = 185
        range_patches = [
            {"x":  60, "y":  19, "rx": 30, "ry": 18, "color": "#1565c0",
             "label": "Wet-season gallery",   "opacity": 0.45},
            {"x": 180, "y":  34, "rx": 28, "ry": 16, "color": "#1565c0",
             "label": "Flood-isolated patch", "opacity": 0.38},
            {"x": 290, "y":  19, "rx": 25, "ry": 16, "color": "#1565c0",
             "label": "N gallery forest",     "opacity": 0.35},
            {"x": 120, "y": 105, "rx": 45, "ry": 25, "color": "#4e342e",
             "label": "Dry-season cerradão range", "opacity": 0.38},
            {"x": 270, "y": 105, "rx": 35, "ry": 22, "color": "#4e342e",
             "label": "Expanded dry range",   "opacity": 0.32},
        ]
        corridors = [
            {"x1":  60, "y1":  50, "x2": 120, "y2": 105, "color": "#ef5350",
             "label": "Eagle patrol N", "label_y_offset": 24},
            {"x1": 180, "y1":  50, "x2": 270, "y2": 105, "color": "#ff1744",
             "label": "Alarm cascade", "label_y_offset": 24},
            {"x1": 290, "y1":  50, "x2": 270, "y2": 105, "color": "#8d6e63",
             "label": "Monkey expansion"},
        ]
        map_snippet = draw_migration_map(
            range_patches, corridors,
            map_w=340, map_h=130, panel_h=ph4,
            title="Patch Use &amp; Alarm Cascade Map",
            title_color="#b0bec5",
            bg_color="#050810",
            border_color="#b0bec5",
            label_letter_spacing=1.5,
        )
        svg.append(f'<g transform="translate({panel_x}, {py4})">{map_snippet}</g>')

        # ── Animated Bottom Status Card ───────────────────────────────────────
        px5 = 20; py5 = h - 238; pw5 = 258; ph5 = 228
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(f'<rect width="{pw5}" height="{ph5}" fill="#050810" rx="8" '
                   f'stroke="#b0bec5" stroke-width="1.5" opacity="0.97"/>')
        svg.append(f'<text x="12" y="22" font-size="15" fill="#b0bec5" font-weight="bold">'
                   f'Active Season Status:</text>')

        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12; vals[m_idx] = "1"
            op_str = ";".join(vals + ["0"])

            svg.append(f'<text x="12" y="52" font-size="15" fill="#b0bec5" font-weight="bold">')
            svg.append(m_name)
            svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                       f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')

            hunt   = sim._interp(HARPY_HUNT_CURVE,       m_idx)
            iso    = sim._interp(MONKEY_ISOLATION_CURVE, m_idx)
            vigil  = sim._interp(MONKEY_VIGILANCE_CURVE, m_idx)
            alarm  = sim._interp(ALARM_CASCADE_CURVE,    m_idx)
            rng    = sim._interp(MONKEY_RANGE_CURVE,     m_idx)

            if iso > 0.65:
                st1, c1 = "PATCH ISOLATION",              "#4fc3f7"
                st2, c2 = f"Flood confinement: {iso*100:.0f}%","#29b6f6"
                st3, c3 = f"Eagle success: {hunt*100:.0f}%",  "#ef5350"
            elif alarm > 0.7:
                st1, c1 = "ALARM CASCADE ACTIVE",         "#ff1744"
                st2, c2 = f"Cascade intensity: {alarm*100:.0f}%","#ff5252"
                st3, c3 = f"Vigilance: {vigil*100:.0f}% of day","#8d6e63"
            elif hunt > 0.7:
                st1, c1 = "PEAK HUNTING WINDOW",          "#b0bec5"
                st2, c2 = f"Eagle success: {hunt*100:.0f}%", "#eceff1"
                st3, c3 = f"Range expansion: {rng*100:.0f}%","#a1887f"
            else:
                st1, c1 = "Seasonal transition",          "#78909c"
                st2, c2 = "Moderate predation pressure",     "#90a4ae"
                st3, c3 = f"Vigilance: {vigil*100:.0f}%",   "#8d6e63"

            for yoff, txt, col in [(80, st1, c1), (100, st2, c2), (118, st3, c3)]:
                svg.append(f'<text x="12" y="{yoff}" font-size="15" fill="{col}" font-weight="bold">')
                svg.append(sanitize_svg_text(txt))
                svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                           f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append('</text>')

        # Legend
        svg.append('<text x="12" y="142" fill="#546e7a" font-size="15" font-weight="bold">Legend:</text>')
        entries = [
            (22, 158, "#eceff1", "Harpy Eagle (Harpia harpyja) — soaring"),
            (22, 174, "#8d6e63", "Capuchin troop (Sapajus libidinosus)"),
            (22, 190, "#ff1744", "Alarm event / Cascade ring"),
            (22, 206, "#4fc3f7", "Wet-season patch isolation"),
            (22, 222, "#b0bec5", "Eagle patrol circuit"),
        ]
        for (ex, ey, ec, elabel) in entries:
            svg.append(f'<ellipse cx="{ex}" cy="{ey}" rx="8" ry="4" fill="{ec}" opacity="0.85"/>')
            svg.append(f'<text font-weight="bold" x="38" y="{ey+4}" fill="{ec}" font-size="15">{elabel}</text>')
        svg.append('</g>')

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Gavião-real ↔ Macaco-prego (Apex Predation & Alarm Cascade Clock) on {CONFIG.device}...")

    sim = HarpyCapuchinSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_kills} kills, "
          f"{sim.total_alarm_events:,} alarm events, "
          f"{sim.cascade_events} multi-troop cascade events, "
          f"peak vigilance {sim.peak_vigilance*100:.0f}%.")

    print("Generating SVG...")
    renderer = HarpyRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_76')
    return svg_content


if __name__ == "__main__":
    main()
