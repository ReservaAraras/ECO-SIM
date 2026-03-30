# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 62: Tamanduá-bandeira ↔ Termite Mounds — Myrmecophagy Phenological Clock
# INTERVENTION 8/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_62.py — Giant Anteater (Tamanduá-bandeira) ↔ Termite Mounds:
Notebook Differentiation:
- Differentiation Focus: Tamanduá-bandeira ↔ Termite Mounds — Myrmecophagy Phenological Clock emphasizing policy compliance patterns.
- Indicator species: Piranha (Serrasalmus sp.).
- Pollination lens: stingless bee corridor dependency.
- Human impact lens: fence permeability conflicts.

                 Predator-Prey & Soil Engineering Phenological Clock

Models the intensive myrmecophagous diet of the Giant Anteater
(Myrmecophaga tridactyla) and its interaction with Cerrado termite 
mounds (Cornitermes cumulans).

The radial phenological clock maps:
  • Termite mound density and seasonal termite activity (surface vs deep).
  • Anteater foraging routines (brief visits to many mounds to avoid 
    depleting the colony and minimizing soldier defense).
  • Seasonal thermal stress: Anteaters are prone to overheating. In the dry
    season, they shift from diurnal to nocturnal foraging and seek dense 
    gallery forest patches for thermal refuge.
  • Fire impact and mound resilience.

Scientific References:
  - Redford, K.H. (1985). "Feeding and food preference in captive and wild 
    Giant Anteaters." Journal of Zoology.
  - Mourão, G. & Medri, Í.M. (2007). "Activity of a specialized insectivorous
    mammal (Myrmecophaga tridactyla) in the Pantanal." Journal of Zoology.
  - Cunha, H.F. et al. (2006). "Termite mound distribution in the Cerrado."
    Sociobiology.

Scientific Relevance (PIGT RESEX Recanto das Araras — 2024):
  - Integrates the socio-environmental complexity of 
    de Cima, Goiás, Brazil.
  - Models landscape connectivity, karst vulnerability (Bacia do Rio Lapa),
    biological corridors, and seed-dispersal networks.
  - Outputs are published via Google Sites: 
  - SVG artefacts archived at: https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing
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

# Thermal Stress Index (Temperature + Dryness)
# In Cerrado, Aug-Sep are the hottest and driest months.
THERMAL_STRESS_CURVE = [
    0.30,   # JAN
    0.35,   # FEV
    0.30,   # MAR
    0.40,   # ABR
    0.50,   # MAI
    0.60,   # JUN
    0.70,   # JUL
    0.90,   # AGO — Peak dry/heat
    1.00,   # SET — Peak dry/heat/fire
    0.80,   # OUT
    0.50,   # NOV
    0.40,   # DEZ
]
ANTEATER_ACTIVITY_CURVE = [0.70, 0.65, 0.70, 0.80, 0.90, 0.95, 1.00, 0.95, 0.85, 0.75, 0.70, 0.75]

# Termite Surface Activity (Foraging outside the mound/in outer galleries)
# Heavy rains (Jan/Feb) restrict them; intense dry restricts them. Peaking in transitions.
TERMITE_ACTIVITY_CURVE = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 0.80]

# Anteater Foraging Shift: 0 = purely Diurnal, 1 = purely Nocturnal
# Shifts to nocturnal to escape dry season heat (Mourão & Medri 2007)
NOCTURNAL_SHIFT_CURVE = [
    0.20,   # JAN — mostly diurnal/crepuscular
    0.20,   # FEV
    0.30,   # MAR
    0.40,   # ABR
    0.60,   # MAI
    0.70,   # JUN
    0.85,   # JUL
    1.00,   # AGO — fully nocturnal
    1.00,   # SET
    0.85,   # OUT
    0.50,   # NOV
    0.30,   # DEZ
]


@dataclass
class MyrmecophagyConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_anteaters: int = 3
    anteater_speed: float = 2.5
    
    num_mounds: int = 80
    max_visit_frames: int = 5  # Very brief visits (1-2 mins in real life)


CONFIG = MyrmecophagyConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class MyrmecophagySim:

    def __init__(self, cfg: MyrmecophagyConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Termite Mounds (Scattered widely in Cerrado) ---
        self.mounds: List[Dict] = []
        for _ in range(cfg.num_mounds):
            r = random.uniform(30, R - 20)
            angle = random.uniform(0, 2 * math.pi)
            mx = cx + math.cos(angle) * r
            my = cy + math.sin(angle) * r
            self.mounds.append({
                "pos": (mx, my),
                "angle": angle,
                "health": 1.0,         # 1.0 = fully populated
                "recovering": 0        # frames until fully recovered
            })

        # --- Giant Anteaters ---
        self.anteaters: List[Dict] = []
        for _ in range(cfg.num_anteaters):
            angle = random.uniform(0, 2 * math.pi)
            tx = cx + math.cos(angle) * 150
            ty = cy + math.sin(angle) * 150
            self.anteaters.append({
                "pos": torch.tensor([tx, ty], device=self.dev, dtype=torch.float32),
                "state": "foraging",  # foraging, resting(thermal refuge)
                "target_mound": -1,
                "visit_timer": 0,
                "energy": 100.0,
                "home_refuge": torch.tensor([cx + random.uniform(-40, 40), cy + random.uniform(-40, 40)], device=self.dev, dtype=torch.float32) # dense brush/gallery
            })

        self.hist_xy: List[List[Tuple[float, float]]] = [[] for _ in range(cfg.num_anteaters)]
        self.hist_month: List[float] = []
        self.visit_events: List[Dict] = [] # For rendering feeding sparks
        
        self.total_mounds_visited = 0
        self.total_termites_eaten = 0.0

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
        
        thermal = self._interp(THERMAL_STRESS_CURVE, month_frac)
        nocturnal_shift = self._interp(NOCTURNAL_SHIFT_CURVE, month_frac)
        activity_lvl = self._interp(ANTEATER_ACTIVITY_CURVE, month_frac)
        termite_act = self._interp(TERMITE_ACTIVITY_CURVE, month_frac)

        self.hist_month.append(month_frac)

        # Fast daily cycle (8 frames = 1 day)
        time_of_day = (frame % 8) / 8.0 
        is_night = time_of_day < 0.25 or time_of_day > 0.75
        
        # Determine if current time allows foraging based on shift
        # If low shift (<0.3), prefers day. If high shift (>0.7), prefers night.
        if nocturnal_shift < 0.5:
            can_forage = not is_night
        else:
            can_forage = is_night
            
        # Severe thermal stress forces resting regardless
        if thermal > 0.8 and not is_night:
            can_forage = False

        # Mound recovery
        for m in self.mounds:
            if m["recovering"] > 0:
                m["recovering"] -= 1
                if m["recovering"] == 0:
                    m["health"] = 1.0

        for i, ant in enumerate(self.anteaters):
            pos = ant["pos"]
            
            if not can_forage or random.random() > activity_lvl:
                ant["state"] = "resting"
                ant["target_mound"] = -1
                target = ant["home_refuge"]
            else:
                ant["state"] = "foraging"
                
                # Pick a mound if needed
                if ant["target_mound"] == -1:
                    best_mound = -1
                    best_d = float('inf')
                    for mi, m in enumerate(self.mounds):
                        if m["health"] > 0.3: # won't visit depleted mounds
                            d = math.sqrt((pos[0].item() - m["pos"][0])**2 + (pos[1].item() - m["pos"][1])**2)
                            if d < best_d and random.random() < 0.6:
                                best_d = d
                                best_mound = mi
                    
                    ant["target_mound"] = best_mound
                
                target = None
                if ant["target_mound"] != -1:
                    m = self.mounds[ant["target_mound"]]
                    target = torch.tensor(m["pos"], device=self.dev, dtype=torch.float32)

            # Move or Eat
            if target is not None:
                to_target = target - pos
                dist = torch.norm(to_target).item()
                
                if ant["state"] == "foraging" and dist < 8.0:
                    # Arrived at mound, start eating
                    ant["visit_timer"] += 1
                    # Termites eaten depends on termite surface activity
                    eaten = 50 * termite_act
                    self.total_termites_eaten += eaten
                    
                    if len(self.visit_events) < 200 and random.random() < 0.5:
                        self.visit_events.append({
                            "pos": m["pos"],
                            "frame": frame
                        })
                    
                    # Leaving mound after brief visit to avoid soldier defense
                    if ant["visit_timer"] >= cfg.max_visit_frames or random.random() < 0.2:
                        self.mounds[ant["target_mound"]]["health"] -= 0.3
                        self.mounds[ant["target_mound"]]["recovering"] = 40 # frames to recover
                        ant["target_mound"] = -1
                        ant["visit_timer"] = 0
                        self.total_mounds_visited += 1
                elif dist > 1.0:
                    dir_vec = to_target / dist
                    speed = cfg.anteater_speed * (0.5 if ant["state"] == "resting" else 1.0)
                    ant["pos"] += dir_vec * speed

            # Clamp
            ant["pos"][0] = torch.clamp(ant["pos"][0], cx - self.cfg.clock_radius + 10, cx + self.cfg.clock_radius - 10)
            ant["pos"][1] = torch.clamp(ant["pos"][1], cy - self.cfg.clock_radius + 10, cy + self.cfg.clock_radius - 10)

            self.hist_xy[i].append((ant["pos"][0].item(), ant["pos"][1].item()))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class MyrmecophagyRenderer:

    def __init__(self, cfg: MyrmecophagyConfig, sim: MyrmecophagySim):
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
            f'style="background-color:#1c1e19; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        svg.append(
            '<radialGradient id="soilBg62">'
            '<stop offset="0%" stop-color="#332d20" stop-opacity="0.8"/>'
            '<stop offset="70%" stop-color="#222019" stop-opacity="0.5"/>'
            '<stop offset="100%" stop-color="#1c1e19" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        # Termite mound gradient
        svg.append(
            '<radialGradient id="moundGrad">'
            '<stop offset="0%" stop-color="#d4a373"/>'
            '<stop offset="80%" stop-color="#8d6e63"/>'
            '<stop offset="100%" stop-color="#5d4037"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # Background Landscape
        svg.append(f'<rect width="{w}" height="{h}" fill="#1c1e19"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="url(#soilBg62)"/>')
        
        # Central Thermal Refuges (Gallery forest patches)
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="50" fill="#2e3b32" opacity="0.4"/>')
        for ant in sim.anteaters:
            rx, ry = ant["home_refuge"][0].item(), ant["home_refuge"][1].item()
            svg.append(f'<circle cx="{rx:.0f}" cy="{ry:.0f}" r="15" fill="#388e3c" opacity="0.15"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#d4a373" font-weight="bold">'
            f'ECO-SIM: Giant Anteater × Termites    - Myrmecophagy Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#a1887f">'
            f'Foraging traits, brief mound visits, and thermal stress regulation </text>'
        )

        # Clock Face
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#8d6e63" '
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

        draw_arc(7, 10, R + 10, "#e53935", "Peak Thermal Stress", 0.4)
        draw_arc(7, 10, R + 38, "#3949ab", "Nocturnal Shift", 0.4)
        draw_arc(11, 5, R + 10, "#fb8c00", "High Diurnal Activity", 0.3)

        # Termite Mounds
        for m in sim.mounds:
            mx, my = m["pos"]
            size = 4
            svg.append(f'<circle cx="{mx:.0f}" cy="{my:.0f}" r="{size}" fill="url(#moundGrad)" stroke="#3e2723" stroke-width="1"/>')
            # Base shadow
            svg.append(f'<ellipse cx="{mx:.0f}" cy="{my+2:.0f}" rx="{size}" ry="{size/2}" fill="#1a1c18" opacity="0.6"/>')

        # Visit Events (Termite Sparks)
        for ve in sim.visit_events:
            vx, vy = ve["pos"]
            vf = ve["frame"]
            ops = ";".join("0.0" if fi < vf else f"{max(0, 0.9 - (fi-vf)*0.1):.2f}" for fi in range(F))
            szs = ";".join("0.0" if fi < vf else f"{min(9, (fi-vf)*1.2):.1f}" for fi in range(F))
            
            svg.append(
                f'<circle cx="{vx:.0f}" cy="{vy:.0f}" fill="none" stroke="#ffcc80" stroke-width="1.5" stroke-dasharray="2,2" opacity="0.0">'
                f'<animate attributeName="r" values="{szs}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # Anteater Trajectories
        for i in range(cfg.num_anteaters):
            hist = sim.hist_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            
            # Trail path (faint)
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::3]]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#795548" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.3"/>'
                )
            
            # Animated Anteater body (Grey-brown)
            svg.append(
                f'<circle r="6" fill="#4e342e" stroke="#000" stroke-width="1">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Long snout indication
            offs_x = ";".join(str(round(h[0]+4, 1)) for h in hist)
            offs_y = ";".join(str(round(h[1]-2, 1)) for h in hist)
            svg.append(
                f'<circle r="2.5" fill="#212121">'
                f'<animate attributeName="cx" values="{offs_x}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{offs_y}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Flag tail indication
            offs_x_t = ";".join(str(round(h[0]-4, 1)) for h in hist)
            offs_y_t = ";".join(str(round(h[1]+2, 1)) for h in hist)
            svg.append(
                f'<circle r="4" fill="#3e2723" opacity="0.8">'
                f'<animate attributeName="cx" values="{offs_x_t}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{offs_y_t}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Clock hand
        hand_x_vals = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y_vals = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#d4a373" stroke-width="2.5" stroke-linecap="round" opacity="0.7">'
            f'<animate attributeName="x2" values="{hand_x_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#1c1e19" stroke="#d4a373" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#d4a373"/>')


        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 215
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#161814" rx="8" '
                   f'stroke="#d4a373" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#d4a373" font-size="15" font-weight="bold">'
                   f'Myrmecophagy Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Anteaters traverse vast Cerrado to locate</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'termite mounds; visits are brief (1-2 mins)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'to avoid soldier defenses and preserve the</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'nest for reuse. Prone to thermal overheating,</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#e53935" font-size="15">'
                   f'they go nocturnal in the intense dry season,</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#81c784" font-size="15">'
                   f'retreating to gallery forest (center) to cool.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 170
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#161814" rx="8" '
                   f'stroke="#ffb74d" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ffb74d" font-size="15" font-weight="bold">'
                   f'Yearly Feeding Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#e0e0e0" font-size="15">'
                   f'Total Termites Consumed: ~{sim.total_termites_eaten:,.0f}k (Est.)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#d4a373" font-size="15">'
                   f'Mound Visits: {sim.total_mounds_visited:,.0f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#90a4ae" font-size="15">'
                   f'Termite Mounds mapped: {cfg.num_mounds} (Cornitermes cumulans)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#78909c" font-size="15">'
                   f'Anteater Population: {cfg.num_anteaters} | Vast territory ranges</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 171
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#161814" rx="8" '
                   f'stroke="#5c6bc0" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#5c6bc0" font-size="15" font-weight="bold">'
                   f'Phenological Curves</text>')

        curves = [
            (THERMAL_STRESS_CURVE, "#e53935", "Thermal Stress"),
            (NOCTURNAL_SHIFT_CURVE, "#3949ab", "Nocturnal Shift"),
            (ANTEATER_ACTIVITY_CURVE, "#fb8c00", "Overall Activity"),
            (TERMITE_ACTIVITY_CURVE, "#8d6e63", "Termite Foraging"),
        ]
        for curve_data, color, label in curves:
            pts = []
            for mi in range(12):
                px_c = chart_x0 + (mi / 11) * chart_w
                py_c = chart_y0 + chart_h - curve_data[mi] * chart_h
                pts.append(f"{px_c:.0f},{py_c:.0f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="1.8" opacity="0.85"/>')

        legend_y = chart_y0 + chart_h + 14
        col_w = chart_w // 2
        for ci, (_, color, label) in enumerate(curves):
            lx = chart_x0 + (ci % 2) * col_w
            ly = legend_y + (ci // 2) * 16
            svg.append(f'<circle cx="{lx}" cy="{ly}" r="3.5" fill="{color}"/>')
            svg.append(f'<text font-weight="bold" x="{lx + 6}" y="{ly + 4}" fill="{color}" font-size="15">'
                       f'{label}</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Tamanduá-bandeira ↔ Termites Clock on {CONFIG.device}...")

    sim = MyrmecophagySim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_mounds_visited} mound visits, ~{sim.total_termites_eaten:,.0f}k termites eaten.")

    print("Generating SVG...")
    renderer = MyrmecophagyRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_62')
    return svg_content


if __name__ == "__main__":
    main()
