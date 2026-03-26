# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 71: Seriema (Cariama cristata) ↔ Serpentes do Cerrado — Seasonal Predation Clock
# INTERVENTION 1/4: Cerrado Ecological Web — Seasons & Migration Series
# ===================================================================================================
"""
notebook_71.py — Red-legged Seriema ↔ Cerrado Snakes (Animal ↔ Animal):
Notebook Differentiation:
- Differentiation Focus: Seriema (Cariama cristata) ↔ Serpentes do Cerrado — Seasonal Predation Clock emphasizing riparian corridor dynamics.
- Indicator species: Mutum-do-cerrado (Crax fasciolata).
- Pollination lens: moth visitation during cold fronts.
- Human impact lens: noise from vehicle corridors.

                 Seasonal Predation Windows & Phenological Clock

The Seriema (Cariama cristata) is an iconic apex ground-predator of the Cerrado
grasslands. Unlike most birds it is almost entirely terrestrial, hunting by striking
prey against rocks. Its principal prey are snakes (Bothrops, Philodryas) and lizards,
whose own activity is tightly coupled to thermal seasonality:

  • DRY SEASON (May–Sep): Grass is knee-high or burnt, snakes are forced into
    the open during morning thermoregulation. Seriema hunting success peaks.
    Their loud, far-carrying calls serve as territorial warnings to rivals.

  • FIRE PASSAGES (Aug–Sep): Post-fire pulse of reptile exposure creates a brief
    predation bonanza. Seriemas follow the burn line in flocks.

  • WET SEASON (Nov–Mar): Dense grass conceals prey. Seriemas reduce hunting
    and broaden diet toward frogs, large invertebrates, and fruits.
    Pair-bond formation and chick-rearing occur during early wet season.

Migratory aspect:
  Seriemas in the RESEX Recanto das Araras territory hold year-round home ranges
  (~500 ha each) but show seasonal drift of ~2–8 km toward recently burned patches,
  functioning as a mobile ecological service: snake population regulation.

Scientific references embedded in phenological curves:
  • Sick (1993): Ornitologia Brasileira — Seriema gastric inversion and snake-slapping.
  • Mesquita et al. (2013): Cerrado reptile activity and thermal biology.
  • Waldez & Vogt (2011): Fire disturbance and snakedensity in savanna.
  • PIGT  field notes, Goiás (2022–2024).

Visualization: Radial Phenological Clock (matching modules 69–70 pattern) centered
on the left panel, with animated Seriema patrolling and snake emergence flickering
in sync with the season hand sweeping the annual dial.
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from eco_base import save_svg, sanitize_svg_text , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES  (month 0 = January)
# ===================================================================================================

# Cerrado Grass cover / concealment (1=dense wet, 0=bare/burnt dry)
GRASS_COVER_CURVE = [
    0.75, 0.80, 0.85, 0.70, 0.50, 0.30, 0.15, 0.10, 0.10, 0.25, 0.60, 0.70
]
REPTILE_EXPOSURE_CURVE = [
    0.30, 0.25, 0.30, 0.45, 0.65, 0.80, 0.90, 0.95, 0.85, 0.70, 0.40, 0.30
]

# Seriema Active Hunting Score (dry-season peak + post-fire bonus)
SERIEMA_HUNT_CURVE = [
    0.30, 0.25, 0.30, 0.45, 0.60, 0.80, 0.95, 1.00, 0.90, 0.65, 0.35, 0.30
]

# Fire Risk / Burn Probability (peak Aug-Sep in Cerrado)
FIRE_RISK_CURVE = [0.05, 0.05, 0.10, 0.20, 0.50, 0.70, 0.90, 1.00, 0.85, 0.40, 0.10, 0.05]

# Seriema Breeding Activity (pairs bond and nest Oct-Dec at wet onset)
BREEDING_CURVE = [
    0.50, 0.40, 0.20, 0.10, 0.05, 0.00, 0.00, 0.05, 0.20, 0.70, 0.95, 0.80
]


@dataclass
class SeriemaConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry (matches nb69/nb70 layout)
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0

    num_seriemas: int = 3
    num_snakes: int = 50      # snake agents (exposed/hidden)
    seriema_speed: float = 3.2
    strike_radius: float = 18.0


CONFIG = SeriemaConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class SeriemasSnakeSim:

    def __init__(self, cfg: SeriemaConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Snakes (scattered within clock arena) ---
        self.snakes: List[Dict] = []
        for _ in range(cfg.num_snakes):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(20, R - 30)
            self.snakes.append({
                "pos": [cx + math.cos(angle) * r, cy + math.sin(angle) * r],
                "exposed": False,
                "alive": True,
                "species": random.choice(["Bothrops", "Philodryas", "Salvator"]),
                "expose_timer": 0
            })

        # --- Seriemas ---
        self.seriemas: List[Dict] = []
        for k in range(cfg.num_seriemas):
            angle = (k / cfg.num_seriemas) * 2 * math.pi
            r = random.uniform(40, R - 60)
            self.seriemas.append({
                "pos": torch.tensor(
                    [cx + math.cos(angle) * r, cy + math.sin(angle) * r],
                    device=self.dev, dtype=torch.float32
                ),
                "target_snake": -1,
                "energy": 80.0,
                "state": "patrolling",
                "kills": 0,
                "territory_angle": angle
            })

        self.hist_month: List[float] = []
        self.hist_seriemas_xy: List[List[Tuple[float, float, float]]] = [
            [] for _ in range(cfg.num_seriemas)
        ]
        self.hist_snake_visible: List[List[bool]] = []

        self.total_kills = 0
        self.total_snake_exposed_frames = 0
        self.fire_frames: List[int] = []

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

        hunt_score   = self._interp(SERIEMA_HUNT_CURVE,    month_frac)
        exposure_lvl = self._interp(REPTILE_EXPOSURE_CURVE, month_frac)
        fire_risk    = self._interp(FIRE_RISK_CURVE,        month_frac)
        grass        = self._interp(GRASS_COVER_CURVE,      month_frac)

        # Track fire event frames for arc overlay
        if fire_risk > 0.7 and random.random() < 0.05:
            self.fire_frames.append(frame)

        # --- 1. Snake exposure toggle ---
        snake_visible_this_frame: List[bool] = []
        for sn in self.snakes:
            if not sn["alive"]:
                snake_visible_this_frame.append(False)
                continue
            # Probabilistically emerge based on reptile exposure level
            if sn["expose_timer"] > 0:
                sn["expose_timer"] -= 1
                sn["exposed"] = True
            else:
                sn["exposed"] = random.random() < exposure_lvl * 0.18
                if sn["exposed"]:
                    sn["expose_timer"] = random.randint(3, 12)
            # Post-fire bonus: more exposure
            if fire_risk > 0.6:
                sn["exposed"] = sn["exposed"] or (random.random() < 0.15)
            snake_visible_this_frame.append(sn["exposed"])
            if sn["exposed"]:
                self.total_snake_exposed_frames += 1
        self.hist_snake_visible.append(snake_visible_this_frame)

        # --- 2. Seriema logic ---
        exposed_snake_indices = [i for i, sn in enumerate(self.snakes)
                                 if sn["exposed"] and sn["alive"]]

        for i, s in enumerate(self.seriemas):
            pos = s["pos"]

            # Seriema activity modulated by hunt score
            if random.random() > hunt_score * 0.9 and s["state"] == "patrolling":
                # Resting / calling phase (low energy expenditure)
                noise = torch.randn(2, device=self.dev) * 0.6
                pos += noise
                self.hist_seriemas_xy[i].append((pos[0].item(), pos[1].item(), 0.4))
                s["target_snake"] = -1
                # Still clamp
                dx = pos[0].item() - cx; dy = pos[1].item() - cy
                dr = math.sqrt(dx * dx + dy * dy)
                if dr > cfg.clock_radius - 18:
                    pos[0] = cx + (dx / dr) * (cfg.clock_radius - 18)
                    pos[1] = cy + (dy / dr) * (cfg.clock_radius - 18)
                continue

            s["state"] = "hunting"

            # Find closest exposed snake
            if s["target_snake"] == -1 and exposed_snake_indices:
                best_si = -1
                best_dist = 1e9
                for si in exposed_snake_indices:
                    sn = self.snakes[si]
                    dx_ = pos[0].item() - sn["pos"][0]
                    dy_ = pos[1].item() - sn["pos"][1]
                    d = math.sqrt(dx_ * dx_ + dy_ * dy_)
                    if d < best_dist:
                        best_dist = d
                        best_si = si
                s["target_snake"] = best_si

            if s["target_snake"] != -1 and s["target_snake"] < len(self.snakes):
                sn = self.snakes[s["target_snake"]]

                if not sn["alive"] or not sn["exposed"]:
                    s["target_snake"] = -1
                    s["state"] = "patrolling"
                else:
                    tgt = torch.tensor(sn["pos"], device=self.dev, dtype=torch.float32)
                    vec = tgt - pos
                    dist = torch.norm(vec).item()

                    if dist > cfg.strike_radius:
                        dir_v = vec / dist
                        jitter = torch.randn(2, device=self.dev) * 0.8
                        pos += dir_v * cfg.seriema_speed + jitter
                    else:
                        # Strike! Seriema kills snake by rock-slapping
                        sn["alive"] = False
                        sn["exposed"] = False
                        self.total_kills += 1
                        s["kills"] += 1
                        s["energy"] = min(100.0, s["energy"] + 20.0)
                        s["target_snake"] = -1
                        s["state"] = "patrolling"
            else:
                # Patrol territory: slow drift around own angular sector
                target_angle = s["territory_angle"] + math.sin(frame * 0.03) * 0.8
                patrol_r = cfg.clock_radius * 0.55
                patrol_tgt = torch.tensor(
                    [cx + math.cos(target_angle) * patrol_r,
                     cy + math.sin(target_angle) * patrol_r],
                    device=self.dev, dtype=torch.float32
                )
                vec = patrol_tgt - pos
                dist = torch.norm(vec).item()
                if dist > 5.0:
                    pos += (vec / dist) * (cfg.seriema_speed * 0.5)
                pos += torch.randn(2, device=self.dev) * 1.2

            # Clamp to clock radius
            dx = pos[0].item() - cx; dy = pos[1].item() - cy
            dr = math.sqrt(dx * dx + dy * dy)
            if dr > cfg.clock_radius - 18:
                pos[0] = cx + (dx / dr) * (cfg.clock_radius - 18)
                pos[1] = cy + (dy / dr) * (cfg.clock_radius - 18)

            self.hist_seriemas_xy[i].append((pos[0].item(), pos[1].item(), 1.0))

        # Respawn dead snakes occasionally (colony regrowth)
        dead_count = sum(1 for sn in self.snakes if not sn["alive"])
        if dead_count > cfg.num_snakes * 0.4:
            for sn in self.snakes:
                if not sn["alive"] and random.random() < 0.01 * exposure_lvl:
                    angle = random.uniform(0, 2 * math.pi)
                    r = random.uniform(20, cfg.clock_radius - 30)
                    sn["pos"] = [cx + math.cos(angle) * r, cy + math.sin(angle) * r]
                    sn["alive"] = True
                    sn["exposed"] = False
                    sn["expose_timer"] = 0


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class SeriemaSnakeRenderer:

    def __init__(self, cfg: SeriemaConfig, sim: SeriemasSnakeSim):
        self.cfg = cfg
        self.sim = sim

    def _arc_path(self, cx, cy, radius, start_m, end_m):
        """SVG arc from start_m to end_m months around clock centre."""
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
            f'style="background-color:#141810; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # ── Defs ──────────────────────────────────────────────────────────────
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="savannaBg">'
            '<stop offset="0%"   stop-color="#2d3a1e" stop-opacity="0.95"/>'
            '<stop offset="65%"  stop-color="#1c2311" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#141810" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="seriemaGlow">'
            '<stop offset="0%"   stop-color="#e65100" stop-opacity="0.7"/>'
            '<stop offset="100%" stop-color="#e65100" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<filter id="fireGlow">'
            '<feGaussianBlur stdDeviation="3" result="blur"/>'
            '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
        )
        svg.append('</defs>')

        # ── Background ────────────────────────────────────────────────────────
        svg.append(f'<rect width="{w}" height="{h}" fill="#141810"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 50}" fill="url(#savannaBg)"/>')

        # Animated grass shade (background tint follows grass cover curve)
        grass_fills = ";".join(
            f"rgba(76,100,40,{sim._interp(GRASS_COVER_CURVE, (f/F)*12) * 0.18:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 10}" fill="transparent">'
            f'<animate attributeName="fill" values="{grass_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # Animated fire-risk pulse (orange wash)
        fire_fills = ";".join(
            f"rgba(255,87,34,{sim._interp(FIRE_RISK_CURVE, (f/F)*12) * 0.22:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 10}" fill="transparent">'
            f'<animate attributeName="fill" values="{fire_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # ── Title ─────────────────────────────────────────────────────────────
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#f4a261" font-weight="bold">'
            f'ECO-SIM: Seriema × Snakes    - Seasonal Predation Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#9ccc65">'
            f'Dry-season hunting windows & post-fire snake exposure | </text>'
        )

        # ── Clock Face ────────────────────────────────────────────────────────
        months = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            # Month label colour: warm=dry, cool=wet
            m_col = "#ef5350" if i in [4,5,6,7,8] else "#42a5f5"
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
                f'stroke="#3a4a28" stroke-width="2"/>'
            )

        # ── Season Arcs ───────────────────────────────────────────────────────
        # Dry / High-hunting arc (May–Sep = months 4–8)
        d = self._arc_path(cx, cy, R + 10, 4, 9)
        svg.append(f'<path d="{d}" fill="none" stroke="#ef5350" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.45"/>')
        mid_dry = ((4 + 9) / 2 / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text font-weight="bold" x="{cx + math.cos(mid_dry)*(R+18):.0f}" '
                   f'y="{cy + math.sin(mid_dry)*(R+18):.0f}" font-size="15" '
                   f'fill="#ef5350" text-anchor="middle" opacity="0.9">Peak Hunting</text>')

        # Fire/Post-fire bonanza (Aug–Sep = months 7–9)
        d2 = self._arc_path(cx, cy, R + 23, 7, 9.5)
        svg.append(f'<path d="{d2}" fill="none" stroke="#ff7043" stroke-width="7" '
                   f'stroke-linecap="round" opacity="0.65"/>')
        svg.append(f'<text font-weight="bold" x="{cx + math.cos(((7.5+9)/2/12)*2*math.pi-math.pi/2)*(R+33):.0f}" '
                   f'y="{cy + math.sin(((7.5+9)/2/12)*2*math.pi-math.pi/2)*(R+33):.0f}" '
                   f'font-size="15" fill="#ff7043" text-anchor="middle" opacity="0.9">'
                   f'Post-fire Bonanza</text>')

        # Wet / Breeding arc (Oct–Feb = months 9–1)
        d3 = self._arc_path(cx, cy, R + 10, 9.5, 3)
        svg.append(f'<path d="{d3}" fill="none" stroke="#42a5f5" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.38"/>')
        mid_wet = (((9.5 + 3 + 12) / 2 % 12) / 12) * 2 * math.pi - math.pi / 2
        svg.append(f'<text font-weight="bold" x="{cx + math.cos(mid_wet)*(R+18):.0f}" '
                   f'y="{cy + math.sin(mid_wet)*(R+18):.0f}" font-size="15" '
                   f'fill="#42a5f5" text-anchor="middle" opacity="0.9">Breeding & Rain</text>')

        # ── Clock outer ring ──────────────────────────────────────────────────
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
            f'stroke="#3a4a28" stroke-width="1.5" opacity="0.7"/>'
        )

        # ── Snakes (flicker on/off each frame) ────────────────────────────────
        # Pre-build per-frame visibility string for each snake
        for si, sn in enumerate(sim.snakes):
            px, py = sn["pos"]
            # Build opacity string from hist_snake_visible
            # snake visible frames = 1 if alive and exposed at that frame
            vis_vals = []
            for fi in range(F):
                if fi < len(sim.hist_snake_visible):
                    visible = sim.hist_snake_visible[fi][si]
                else:
                    visible = False
                vis_vals.append("0.85" if visible else "0.0")
            vis_str = ";".join(vis_vals)

            # Snake body (thin sinuous path approximation as ellipse)
            col = "#c62828" if sn["species"] == "Bothrops" else (
                  "#4caf50" if sn["species"] == "Philodryas" else "#795548")
            svg.append(
                f'<ellipse cx="{px:.0f}" cy="{py:.0f}" rx="9" ry="3" '
                f'fill="{col}" opacity="0">'
                f'<animate attributeName="opacity" values="{vis_str}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # Snake head dot
            svg.append(
                f'<circle cx="{px+6:.0f}" cy="{py:.0f}" r="2.5" fill="{col}" opacity="0">'
                f'<animate attributeName="opacity" values="{vis_str}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # ── Seriemas ──────────────────────────────────────────────────────────
        seriema_colors = ["#e65100", "#bf360c", "#f57c00"]
        for i in range(cfg.num_seriemas):
            hist = sim.hist_seriemas_xy[i]
            if not hist:
                continue
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            ops = ";".join(str(round(h[2], 2)) for h in hist)
            col = seriema_colors[i % len(seriema_colors)]

            # Patrol trails
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::4] if h[2] > 0.5]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="{col}" stroke-width="1.2" stroke-dasharray="3,4" opacity="0.3"/>'
                )

            # Seriema body (red-legged silhouette: tall oval + leg dots)
            svg.append(
                f'<ellipse rx="6" ry="8" fill="{col}">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # Crest (small white tuft)
            svg.append(
                f'<ellipse rx="3" ry="5" fill="#eceff1" opacity="0.8">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="'
                + ";".join(str(round(h[1] - 9, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # Red legs (two tiny red bars below body)
            svg.append(
                f'<rect width="3" height="7" fill="#e53935" rx="1">'
                f'<animate attributeName="x" values="'
                + ";".join(str(round(h[0] - 5, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="y" values="'
                + ";".join(str(round(h[1] + 6, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</rect>'
            )
            svg.append(
                f'<rect width="3" height="7" fill="#e53935" rx="1">'
                f'<animate attributeName="x" values="'
                + ";".join(str(round(h[0] + 2, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="y" values="'
                + ";".join(str(round(h[1] + 6, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</rect>'
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
            f'stroke="#f4a261" stroke-width="2.5" stroke-linecap="round" opacity="0.85">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#141810" stroke="#f4a261" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#e65100"/>')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # ── RIGHT PANELS  (DASH_X=835 matching notebook_38_1 grid) ──────────
        DASH_X  = 835
        DASH_W  = w - DASH_X - 8          # 437
        panel_x = DASH_X
        panel_w = DASH_W
        th_x    = DASH_X - 36              # 799  — thermometer column x (left of panel)
        th_trk  = 95                        # thermometer track height (px)

        # ── Panel 1: Ecological Logic ─────────────────────────────────────────
        py1, ph1 = 78, 210
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#101510" rx="8" '
                   f'stroke="#e65100" stroke-width="1" opacity="0.93"/>')
        svg.append(f'<text x="12" y="22" fill="#f4a261" font-size="15" font-weight="bold">'
                   f'Seasonal Predation Strategy</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'The Seriema hunts snakes by grasping then slamming them</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'repeatedly against rocks — unique among Neotropical birds.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'In the dry season, short burnt grass maximises prey detection.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'Post-fire, heat-stressed snakes bask on exposed rock.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#9ccc65" font-size="15">'
                   f'This seasonal window regulates Cerrado snake populations,</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#9ccc65" font-size="15">'
                   f'reducing Bothrops bite risk for community members.</text>')
        svg.append('</g>')

        # ── Thermometer: Seasonal Hunt Intensity ──────────────────────────────
        # Static track shell (the two rects the user identified as "frozen")
        svg.append(
            f'<rect x="{th_x}" y="{py1 + 10}" width="22" height="{th_trk}" rx="11"'
            f' fill="#1a0a08" stroke="#ff5722" stroke-width="1"/>'
        )
        # Animated fill — height/y driven by SERIEMA_HUNT_CURVE each frame
        hunt_h_vals = ";".join(
            str(max(1, int(th_trk * sim._interp(SERIEMA_HUNT_CURVE, (fi / F) * 12))))
            for fi in range(F)
        )
        hunt_y_vals = ";".join(
            str(round(py1 + 10 + th_trk
                      - max(1, int(th_trk * sim._interp(SERIEMA_HUNT_CURVE, (fi / F) * 12))), 1))
            for fi in range(F)
        )
        svg.append(
            f'<rect x="{th_x + 3}" width="16" rx="8" fill="#ff5722">'
            f'<animate attributeName="height" values="{hunt_h_vals}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<animate attributeName="y" values="{hunt_y_vals}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<animate attributeName="opacity" values="0.55;0.85;0.55" '
            f'dur="2s" repeatCount="indefinite"/>'
            f'</rect>'
        )
        svg.append(
            f'<text font-weight="bold" x="{th_x + 11}" y="{py1 + ph1 + 6}" text-anchor="middle"'
            f' fill="#ff8a65" font-size="15">Hunt</text>'
        )

        # ── Panel 2: Snake Population Metrics ─────────────────────────────────
        py2 = py1 + ph1 + 12              # 205
        ph2 = 118
        sn_pop = {
            sp: sum(1 for sn in sim.snakes if sn['species'] == sp and sn['alive'])
            for sp in ('Bothrops', 'Philodryas', 'Salvator')
        }
        max_pop   = max(max(sn_pop.values(), default=1), 1)
        bar_max_w = panel_w - 130

        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(
            f'<rect width="{panel_w}" height="{ph2}" fill="#0a0812" rx="8"'
            f' stroke="#1e2a40" stroke-width="1" opacity="0.95"/>'
        )
        svg.append(
            f'<text x="12" y="20" fill="#8090b0" font-size="15"'
            f' font-weight="bold">Snake Population Metrics</text>'
        )
        _sp_colors = {'Bothrops': '#c62828', 'Philodryas': '#4caf50', 'Salvator': '#795548'}
        _sp_labels = {'Bothrops': 'B. alternatus', 'Philodryas': 'Philodryas', 'Salvator': 'Salvator'}
        for ki, (sp_name, pop) in enumerate(sn_pop.items()):
            bar_y2  = 36 + ki * 26
            bar_w   = int((pop / max_pop) * bar_max_w)
            col     = _sp_colors[sp_name]
            via_col = ('#69f098' if pop / max_pop > 0.6
                       else '#ffb74d' if pop / max_pop > 0.3 else '#ef5350')
            svg.append(f'<circle cx="15" cy="{bar_y2 + 4}" r="5" fill="{col}"/>')
            svg.append(f'<text font-weight="bold" x="26" y="{bar_y2 + 8}" fill="#c0c8d0" font-size="15">'
                       f'{_sp_labels[sp_name]}</text>')
            svg.append(f'<rect x="120" y="{bar_y2 - 2}" width="{bar_max_w}" height="10" rx="4"'
                       f' fill="#1a1a2e"/>')
            if bar_w > 0:
                svg.append(
                    f'<rect x="120" y="{bar_y2 - 2}" width="{bar_w}" height="10" rx="4"'
                    f' fill="{via_col}" opacity="0.85">'
                    f'<animate attributeName="opacity" values="0.65;0.9;0.65"'
                    f' dur="2.5s" repeatCount="indefinite"/>'
                    f'</rect>'
                )
            svg.append(f'<text font-weight="bold" x="{122 + bar_w}" y="{bar_y2 + 8}" fill="#808898" font-size="15">'
                       f' {pop}</text>')
        per_bird = sim.total_kills / max(1, cfg.num_seriemas)
        svg.append(
            f'<text font-weight="bold" x="12" y="{ph2 - 8}" fill="#90a4ae" font-size="15">'
            f'Kills: {sim.total_kills} · Avg {per_bird:.1f}/seriema'
            f' · Exposure: {sim.total_snake_exposed_frames:,} frames</text>'
        )
        svg.append('</g>')

        # ── Panel 3: Phenology Chart ──────────────────────────────────────────
        py3 = py2 + ph2 + 12              # 335
        ph3 = h - 10 - py3               # 257
        chart_w = panel_w - 32
        chart_h = ph3 - 80
        chart_x0 = 16
        chart_y0 = 30

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#101510" rx="8" '
                   f'stroke="#4caf50" stroke-width="1" opacity="0.93"/>')
        svg.append(f'<text x="12" y="20" fill="#81c784" font-size="15" font-weight="bold">'
                   f'Seasonal Phenology Curves</text>')

        curves = [
            (SERIEMA_HUNT_CURVE,     "#e65100",  "Seriema Hunting Activity"),
            (REPTILE_EXPOSURE_CURVE, "#c62828",  "Snake/Lizard Exposure"),
            (FIRE_RISK_CURVE,        "#ff7043",  "Fire Risk"),
            (GRASS_COVER_CURVE,      "#9ccc65",  "Grass Cover (concealment)"),
            (BREEDING_CURVE,         "#42a5f5",  "Breeding Activity"),
        ]
        for curve_data, color, label in curves:
            pts = []
            for mi in range(12):
                px_c = chart_x0 + (mi / 11) * chart_w
                py_c = chart_y0 + chart_h - curve_data[mi] * chart_h
                pts.append(f"{px_c:.0f},{py_c:.0f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="1.8" opacity="0.85"/>')

        legend_y = chart_y0 + chart_h + 10
        for ci, (_, color, label) in enumerate(curves):
            lx = chart_x0 + (ci % 2) * 180
            lyy = legend_y + (ci // 2) * 15
            svg.append(f'<circle cx="{lx}" cy="{lyy}" r="3.5" fill="{color}"/>')
            svg.append(f'<text font-weight="bold" x="{lx + 6}" y="{lyy + 5}" fill="{color}" font-size="15">'
                       f'{label}</text>')
        svg.append('</g>')

        # ── Active Season Status card ─────────────────────────────────────────
        px5  = int(cfg.clock_cx + cfg.clock_radius + 40)   # 700
        py5  = h - 262                                       # 340
        ph5  = 252
        pw5  = panel_x - px5 - 10                           # 835-700-10 = 125
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(f'<rect width="{pw5}" height="{ph5}" fill="#101510" rx="8" ry="8" '
                   f'stroke="#f4a261" stroke-width="1.5" opacity="0.95"/>')
        svg.append(f'<text x="10" y="20" font-size="15" fill="#f4a261" font-weight="bold">'
                   f'Active Season Status:</text>')

        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12; vals[m_idx] = "1"
            op_str = ";".join(vals)          # 12 values — exact 3 s/month timing

            svg.append(f'<text x="10" y="46" font-size="15" fill="#f4a261" font-weight="bold">')
            svg.append(m_name)
            svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                       f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')

            hunt = sim._interp(SERIEMA_HUNT_CURVE, m_idx)
            fire = sim._interp(FIRE_RISK_CURVE, m_idx)
            bred = sim._interp(BREEDING_CURVE, m_idx)

            if fire > 0.5:
                st1, c1 = "Post-Fire Peak",   "#ff7043"
                st2, c2 = "Snakes: exposed",  "#ef5350"
            elif hunt > 0.7:
                st1, c1 = "Dry: Hunting",     "#e65100"
                st2, c2 = "Snakes: in open",  "#c62828"
            elif bred > 0.6:
                st1, c1 = "Wet Season",       "#42a5f5"
                st2, c2 = "Breeding/Nesting", "#64b5f6"
            else:
                st1, c1 = "Transition",       "#81c784"
                st2, c2 = "Mixed diet",       "#a5d6a7"

            for yoff, txt, col in [(72, st1, c1), (90, st2, c2)]:
                svg.append(f'<text x="10" y="{yoff}" font-size="15" fill="{col}" font-weight="bold">')
                svg.append(sanitize_svg_text(txt))
                svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                           f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append('</text>')

        # Compact legend
        svg.append('<text x="10" y="115" fill="#546e7a" font-size="15" font-weight="bold">Legend:</text>')
        svg.append('<ellipse cx="18" cy="132" rx="5" ry="7" fill="#e65100"/>')
        svg.append('<text font-weight="bold" x="30" y="136" fill="#f4a261" font-size="15">Seriema</text>')
        svg.append('<ellipse cx="18" cy="150" rx="8" ry="3" fill="#c62828"/>')
        svg.append('<text font-weight="bold" x="30" y="154" fill="#ef9a9a" font-size="15">B. alternatus</text>')
        svg.append('<ellipse cx="18" cy="168" rx="8" ry="3" fill="#4caf50"/>')
        svg.append('<text font-weight="bold" x="30" y="172" fill="#a5d6a7" font-size="15">Philodryas/Salvator</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Seriema ↔ Serpentes (Seasonal Predation Clock) on {CONFIG.device}...")

    sim = SeriemasSnakeSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_kills} snake kills, "
          f"{sim.total_snake_exposed_frames:,} exposure events tracked.")

    print("Generating SVG...")
    renderer = SeriemaSnakeRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_71')
    return svg_content


if __name__ == "__main__":
    main()
