# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 69: Tamanduá-bandeira ↔ Cupinzeiros — Macro-predator Regulation Clock
# INTERVENTION 3/4: Seasonal & Migratory Interventions Series
# ===================================================================================================
"""
notebook_69.py — Giant Anteater ↔ Termite Mounds:
Notebook Differentiation:
- Differentiation Focus: Tamanduá-bandeira ↔ Cupinzeiros — Macro-predator Regulation Clock emphasizing canopy stratification.
- Indicator species: Sauva (Atta sexdens).
- Pollination lens: masting-to-bloom handoff in late dry season.
- Human impact lens: trail disturbance and repeated flushing.

                 Invertebrate Regulation & Seasonal Swarm Clock

Models the profound regulatory relationship between the Tamanduá-bandeira 
(Myrmecophaga tridactyla) and the ubiquitous Cerrado termite mounds (Cupinzeiros).

Termite surface activity and reproductive swarm flights (revoada) are tightly 
coupled to the onset of the first heavy rains (Oct/Nov/Dec). The Giant Anteater 
paces its foraging, briefly feeding at hundreds of mounds daily but never destroying 
the colony, ensuring a sustainable caloric yield season after season.

The radial phenological clock maps:
  • Wet/Dry Season transitions (Precipitation).
  • Termite Mound Surface Activity (deep in dry, surface in wet).
  • Alate Swarm Flights (Revoada) triggered by first rains.
  • Anteater Sustainable Foraging (Rotational predation).

Scientific Relevance:
  - Demonstrates macro-predator regulation of landscape-level invertebrate populations.
  - Highlights sustainable predatory pacing (rotational harvesting) allowing colony recovery.
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
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES
# ===================================================================================================

# Cerrado Precipitation (Wet Nov-Mar, Dry Jun-Sep)
PRECIPITATION_CURVE = [
    0.80, 0.70, 0.85, 0.50, 0.20, 0.05, 0.00, 0.05, 0.20, 0.50, 0.90, 1.00
]
TERMITE_SURFACE_CURVE = [
    0.80, 0.70, 0.60, 0.50, 0.30, 0.10, 0.05, 0.05, 0.20, 0.70, 1.00, 0.90
]

# Reproductive Swarm Flights ("Revoada" - Alates emerge to mate)
TERMITE_SWARM_CURVE = [0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.20, 0.60, 0.90, 0.50]

# Anteater Foraging Activity (Relies on mounds year-round, but easier in wet)
ANTEATER_ACTIVITY_CURVE = [0.70, 0.65, 0.70, 0.80, 0.90, 0.95, 1.00, 0.95, 0.85, 0.75, 0.70, 0.75]


@dataclass
class AnteaterConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_mounds: int = 40
    num_anteaters: int = 2
    anteater_speed: float = 2.5


CONFIG = AnteaterConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class AnteaterSim:

    def __init__(self, cfg: AnteaterConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Termite Mounds ---
        self.mounds: List[Dict] = []
        for _ in range(cfg.num_mounds):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(30, R - 40)
            self.mounds.append({
                "pos": (cx + math.cos(angle)*r, cy + math.sin(angle)*r),
                "population": 100.0,
                "recovering_frames": 0,
                "swarming": False
            })

        # --- Giant Anteaters ---
        self.anteaters: List[Dict] = []
        for _ in range(cfg.num_anteaters):
            self.anteaters.append({
                "pos": torch.tensor([
                    cx + random.uniform(-R/2, R/2), 
                    cy + random.uniform(-R/2, R/2)
                ], device=self.dev, dtype=torch.float32),
                "target_mound": -1,
                "stomach_filled": 0.0,
                "state": "foraging"
            })

        self.hist_month: List[float] = []
        self.hist_anteaters_xy: List[List[Tuple[float, float, float]]] = [[] for _ in range(cfg.num_anteaters)]
        
        self.total_termites_eaten = 0.0
        self.total_swarms = 0
        self.mounds_visited = 0

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
        
        surf_act = self._interp(TERMITE_SURFACE_CURVE, month_frac)
        swarm_prob = self._interp(TERMITE_SWARM_CURVE, month_frac)
        a_act = self._interp(ANTEATER_ACTIVITY_CURVE, month_frac)

        # 1. Mound logic
        for m in self.mounds:
            if m["recovering_frames"] > 0:
                m["recovering_frames"] -= 1
                m["population"] += 0.5 * surf_act # Recover faster in wet season
            else:
                m["population"] = min(100.0, m["population"] + 1.0 * surf_act)
            
            # Swarming
            m["swarming"] = False
            if m["population"] > 80.0 and swarm_prob > 0.5:
                if random.random() < swarm_prob * 0.1:
                    m["swarming"] = True
                    self.total_swarms += 1

        # 2. Anteater logic
        for i, a in enumerate(self.anteaters):
            pos = a["pos"]
            
            # Metabolism & digestion
            a["stomach_filled"] = max(0.0, a["stomach_filled"] - 0.5)
            
            if a["stomach_filled"] > 80.0:
                a["state"] = "digesting"
                a["target_mound"] = -1
            else:
                a["state"] = "foraging"
            
            if a["state"] == "foraging":
                if a["target_mound"] == -1:
                    # Find a healthy mound
                    best_m = -1
                    best_score = -999.0
                    for mi, m in enumerate(self.mounds):
                        if m["recovering_frames"] <= 0 and m["population"] > 50.0:
                            tgt = torch.tensor(m["pos"], device=self.dev, dtype=torch.float32)
                            dist = torch.norm(tgt - pos).item()
                            score = m["population"] - dist * 0.5
                            if score > best_score:
                                best_score = score
                                best_m = mi
                    
                    if best_m != -1:
                        a["target_mound"] = best_m
                
                # Move to mound
                if a["target_mound"] != -1:
                    m = self.mounds[a["target_mound"]]
                    tgt = torch.tensor(m["pos"], device=self.dev, dtype=torch.float32)
                    vec = tgt - pos
                    dist = torch.norm(vec).item()
                    
                    if dist > 3.0:
                        dir_v = vec / dist
                        pos += dir_v * cfg.anteater_speed
                    else:
                        # Feed rotationally: Take a portion, don't destroy
                        bite = min(m["population"], 30.0) # Anteater eats quickly and moves on
                        m["population"] -= bite
                        m["recovering_frames"] = 40 # Mound repairs
                        a["stomach_filled"] += bite
                        self.total_termites_eaten += bite
                        self.mounds_visited += 1
                        a["target_mound"] = -1
            else:
                # Digesting/wandering slowly
                pos += torch.randn(2, device=self.dev) * (cfg.anteater_speed * 0.3)

            # Clamp to clock radius
            dx = pos[0].item() - cx
            dy = pos[1].item() - cy
            dr = math.sqrt(dx*dx + dy*dy)
            if dr > cfg.clock_radius - 20:
                pos[0] = cx + (dx/dr)*(cfg.clock_radius - 20)
                pos[1] = cy + (dy/dr)*(cfg.clock_radius - 20)
        
            self.hist_anteaters_xy[i].append((pos[0].item(), pos[1].item(), 1.0))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class AnteaterRenderer:

    def __init__(self, cfg: AnteaterConfig, sim: AnteaterSim):
        self.cfg = cfg
        self.sim = sim

    def generate_svg(self) -> str:
        cfg = self.cfg
        sim = self.sim
        w, h = cfg.width, cfg.height
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius
        F = cfg.frames
        dur = F / cfg.fps

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:#1e1a17; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        svg.append(
            '<radialGradient id="soilBg">'
            '<stop offset="0%" stop-color="#3e2723" stop-opacity="0.9"/>'
            '<stop offset="60%" stop-color="#2d1d16" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#1e1a17" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # Background Ground Space
        svg.append(f'<rect width="{w}" height="{h}" fill="#1e1a17"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="url(#soilBg)"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#d7ccc8" font-weight="bold">'
            f'ECO-SIM: Giant Anteater × Termite Mound    - Macro-predator Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#a1887f">'
            f'Sustainable foraging & invertebrate regulation | RESEX Recanto das Araras</text>'
        )

        # Clock Face
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#6d4c41" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 5)
            ly2 = cy + math.sin(angle) * (R - 5)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#4e342e" stroke-width="2"/>'
            )

        # Season Arcs
        def draw_arc(start_m, end_m, radius, color, label, opacity=0.3):
            a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
            a2 = (end_m / 12) * 2 * math.pi - math.pi / 2
            x1 = cx + math.cos(a1) * radius
            y1 = cy + math.sin(a1) * radius
            x2 = cx + math.cos(a2) * radius
            y2 = cy + math.sin(a2) * radius
            span = (end_m - start_m) % 12
            large = 1 if span > 6 else 0
            d = f"M {x1:.0f} {y1:.0f} A {radius} {radius} 0 {large} 1 {x2:.0f} {y2:.0f}"
            svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="8" '
                       f'stroke-linecap="round" opacity="{opacity}"/>')
            mid = ((start_m + end_m) / 2 / 12) * 2 * math.pi - math.pi / 2
            lx = cx + math.cos(mid) * (radius + 15)
            ly = cy + math.sin(mid) * (radius + 15)
            svg.append(f'<text font-weight="bold" x="{lx:.0f}" y="{ly:.0f}" font-size="15" fill="{color}" '
                       f'text-anchor="middle" opacity="0.9">{label}</text>')

        draw_arc(5.5, 9.5, R + 10, "#d84315", "Deep Dry Retreat", 0.4)
        draw_arc(9.5, 11.5, R + 22, "#ffca28", "Alate Swarm Flights (Revoada)", 0.6)
        draw_arc(10.5, 3.5, R + 10, "#4caf50", "High Surface Activity", 0.4)

        # --- Precipitation Background Pulse ---
        rain_colors = ";".join(
            f"rgba(76, 175, 80, {sim._interp(PRECIPITATION_CURVE, (f/F)*12) * 0.1:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R-10}" fill="transparent">'
            f'<animate attributeName="fill" values="{rain_colors}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # --- Termite Mounds (Cupinzeiros) ---
        for mi, m in enumerate(sim.mounds):
            px, py = m["pos"]
            
            # Mound Base
            svg.append(
                f'<path d="M{px-6},{py+4} Q{px},{py-12} {px+6},{py+4} Z" fill="#795548" stroke="#3e2723" stroke-width="1.5"/>'
            )
            
            # Swarming Alates Visual (if swarming at all during sim)
            # Find frames where swarming occurred
            # For simplicity, we make the swarm follow the precipitation/swarm curve
            s_vals = ";".join(
                f"{max(0, sim._interp(TERMITE_SWARM_CURVE, (fi/F)*12) * random.uniform(0, 1.0)):.1f}"
                if fi > F*0.75 else "0.0"
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py-10:.0f}" r="8" fill="#ffca28" opacity="0.0">'
                f'<animate attributeName="opacity" values="{s_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # --- Giant Anteaters (Tamanduá-bandeira) ---
        for i in range(cfg.num_anteaters):
            hist = sim.hist_anteaters_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            
            # Foraging trails
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::3] if h[2] > 0.5]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#5d4037" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.4"/>'
                )
            
            # Anteater Body (Brown ellipse with grey stripe)
            svg.append(
                f'<ellipse rx="8" ry="4" fill="#4e342e">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</ellipse>'
            )
            svg.append(
                f'<line x1="-6" y1="0" x2="6" y2="0" stroke="#9e9e9e" stroke-width="1.5">'
                f'<animate attributeName="x1" values="{cxs}" dur="{dur}s" repeatCount="indefinite" additive="sum"/>'
                f'<animate attributeName="x2" values="{cxs}" dur="{dur}s" repeatCount="indefinite" additive="sum"/>'
                f'<animate attributeName="y1" values="{cys}" dur="{dur}s" repeatCount="indefinite" additive="sum"/>'
                f'<animate attributeName="y2" values="{cys}" dur="{dur}s" repeatCount="indefinite" additive="sum"/>'
                f'</line>'
            )

        # Clock hand
        hand_x = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#ffca28" stroke-width="2.5" stroke-linecap="round" opacity="0.8">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#1e1a17" stroke="#ffca28" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#d84315"/>')


        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 156
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#161311" rx="8" '
                   f'stroke="#d84315" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#d84315" font-size="15" font-weight="bold">'
                   f'Sustainable Rotational Foraging Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Anteater consumes thousands of termites daily</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'but rarely destroys a mound; it feeds briefly</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'and moves on, letting defensive soldiers mount</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'a defense, preventing colony collapse. Creates</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#4caf50" font-size="15">'
                   f'a sustainable rotation across the Cerrado.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 145
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#161311" rx="8" '
                   f'stroke="#5d4037" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#a1887f" font-size="15" font-weight="bold">'
                   f'Predation & Colony Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#d84315" font-size="15">'
                   f'Total Termites Eaten: {sim.total_termites_eaten:,.0f}k units</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#4caf50" font-size="15">'
                   f'Rotational Visits: {sim.mounds_visited} mound raids</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#ffca28" font-size="15">'
                   f'Mass Swarm Events (Alates): {sim.total_swarms}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'Mound Survival: 100% (Sustainable Predation)</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 130
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#161311" rx="8" '
                   f'stroke="#6d4c41" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#8d6e63" font-size="15" font-weight="bold">'
                   f'Invertebrate Phenology Curves</text>')

        curves = [
            (PRECIPITATION_CURVE, "#4fc3f7", "Precipitation"),
            (TERMITE_SURFACE_CURVE, "#4caf50", "Mound Surface Activity"),
            (TERMITE_SWARM_CURVE, "#ffca28", "Revoada (Swarms)"),
            (ANTEATER_ACTIVITY_CURVE, "#d84315", "Anteater Foraging"),
        ]
        for curve_data, color, label in curves:
            pts = []
            for mi in range(12):
                px_c = chart_x0 + (mi / 11) * chart_w
                py_c = chart_y0 + chart_h - curve_data[mi] * chart_h
                pts.append(f"{px_c:.0f},{py_c:.0f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="1.8" opacity="0.85"/>')

        legend_y = chart_y0 + chart_h + 12
        for ci, (_, color, label) in enumerate(curves):
            lx = chart_x0 + (ci % 2) * 160
            lyy = legend_y if ci < 2 else legend_y + 16
            svg.append(f'<circle cx="{lx}" cy="{lyy}" r="3.5" fill="{color}"/>')
            svg.append(f'<text font-weight="bold" x="{lx + 6}" y="{lyy + 5}" fill="{color}" font-size="15">'
                       f'{label}</text>')
        svg.append('</g>')

        # --- Panel 4: Context ---
        py4 = py3 + ph3 + 26
        ph4 = 118
        svg.append(f'<g transform="translate({panel_x}, {py4})">')
        svg.append(f'<rect width="{panel_w}" height="{ph4}" fill="#161311" rx="8" '
                   f'stroke="#4e342e" stroke-width="1" opacity="0.88"/>')
        svg.append(f'<text x="12" y="18" fill="#795548" font-size="15" font-weight="bold">'
                   f'Keystone Regulator Context</text>')
        svg.append(f'<text font-weight="bold" x="12" y="34" fill="#6d4c41" font-size="15">'
                   f'Without Tamanduá, termite populations could</text>')
        svg.append(f'<text font-weight="bold" x="12" y="48" fill="#6d4c41" font-size="15">'
                   f'overconsume, breaking the nutrient cycle.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#6d4c41" font-size="15">'
                   f'It dictates invertebrate health of the RESEX.</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Tamanduá ↔ Cupinzeiros Clock on {CONFIG.device}...")

    sim = AnteaterSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_termites_eaten:.0f}k termites eaten over {sim.mounds_visited} mound raids.")

    print("Generating SVG...")
    renderer = AnteaterRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_69')
    return svg_content


if __name__ == "__main__":
    main()

