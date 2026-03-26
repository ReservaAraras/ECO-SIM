# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 66: Onça-pintada ↔ Queixadas — Apex Predator-Prey & Karst Phenological Clock
# INTERVENTION 12/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_66.py — Jaguar (Onça-pintada) ↔ White-lipped Peccaries (Queixadas):
Notebook Differentiation:
- Differentiation Focus: Onça-pintada ↔ Queixadas — Apex Predator-Prey & Karst Phenological Clock emphasizing flood pulse timing.
- Indicator species: Mico-estrela (Callithrix penicillata).
- Pollination lens: secondary pollination by beetles.
- Human impact lens: light pollution near settlements.

                 Apex Predator-Prey & Karst Refuge Phenological Clock

Models the apex of the Cerrado trophic web. The Jaguar (Panthera onca) is the
master of the gallery forests and karst formations, while wide-ranging herds of
White-lipped Peccaries (Tayassu pecari) act as both ecosystem engineers and
primary prey.

In parallel with the service-based framing used elsewhere in the notebooks and
the broader conservation logic reflected in base.txt, predation is treated here
as a regulating process: hydrology and karst refuge geometry compress prey
movement, encounter rates rise, and jaguars reshape herd space use at the dry-
season bottleneck.

The radial phenological clock maps:
  • Massive Peccary Herds: Highly cohesive groups that roam the Cerrado following
    fruiting seasons (linking back to Pequi [nb55], Jatobá/Macaúba [nb61]).
  • Water-driven concentration: In the harsh dry season (Aug-Sep), herds are
    forced to concentrate near permanent rivers and gallery forests.
  • Jaguar Hunting Success: Ambush predators that rely on cover. Hunting
    success peaks during the dry season when prey is predictably concentrated
    near water and physically weaker.
  • Karst Cave Denning: Jaguars utilize the limestone caves (common in Bacia 
    do Rio Lapa) as secure thermal and reproductive refuges to raise cubs.

Scientific References:
  - Silveira, L. et al. (2010). "Ecology of the Jaguar in the Cerrado."
    Journal of Zoology.
  - Keuroghlian, A. et al. (2004). "Habitat use by white-lipped peccaries."
    Journal of Tropical Ecology.
  - Reyna-Hurtado, R. et al. (2009). "Historical and current distribution of
    white-lipped peccaries."

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

# Surface Water Dispersion (Open Cerrado vs Gallery/Karst Rivers)
# High in wet season (prey scatters everywhere), zero in dry season (prey concentrates).
WATER_DISPERSION_CURVE = [
    1.00,   # JAN
    1.00,   # FEV
    0.90,   # MAR
    0.70,   # ABR
    0.40,   # MAI
    0.20,   # JUN
    0.10,   # JUL
    0.00,   # AGO — Peak dry (forced to river/karst core)
    0.05,   # SET
    0.30,   # OUT
    0.70,   # NOV
    0.90,   # DEZ
]
# High after massive wet-season fruiting; plummets during late dry season.
PREY_CONDITION_CURVE = [
    0.70,   # JAN
    0.80,   # FEV
    0.90,   # MAR — Max fat
    1.00,   # ABR
    0.85,   # MAI
    0.70,   # JUN
    0.50,   # JUL
    0.30,   # AGO — Weakening
    0.20,   # SET — Starving
    0.40,   # OUT
    0.50,   # NOV
    0.60,   # DEZ
]

# Jaguar Ambush Success Probability
# Inversely correlated to prey condition and positively to restricted prey movement.
JAGUAR_HUNT_SUCCESS = [
    0.20,   # JAN
    0.20,   # FEV
    0.15,   # MAR — Hard to catch strong, scattered prey
    0.15,   # ABR
    0.25,   # MAI
    0.40,   # JUN
    0.60,   # JUL
    0.85,   # AGO — Peak ambush success (prey at water, weak)
    0.90,   # SET
    0.70,   # OUT
    0.40,   # NOV
    0.30,   # DEZ
]


@dataclass
class ApexConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_peccaries: int = 35 # Forming a large cohesive herd
    num_jaguars: int = 2
    
    # Speeds
    peccary_speed: float = 2.0
    jaguar_speed: float = 2.8
    dash_speed: float = 7.0
    
    karst_cave_radius: float = 40.0


CONFIG = ApexConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class ApexSim:

    def __init__(self, cfg: ApexConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Karst Caves / Gallery Forest Center ---
        self.cave_pos = torch.tensor([cx, cy], device=self.dev, dtype=torch.float32)

        # --- Queixadas (White-lipped Peccary Herd) ---
        self.peccaries: List[Dict] = []
        # Start herd somewhat together
        herd_cx, herd_cy = cx + 100, cy + 100
        for _ in range(cfg.num_peccaries):
            tx = herd_cx + random.uniform(-30, 30)
            ty = herd_cy + random.uniform(-30, 30)
            self.peccaries.append({
                "pos": torch.tensor([tx, ty], device=self.dev, dtype=torch.float32),
                "alive": True
            })

        # --- Onça-pintada (Jaguars) ---
        self.jaguars: List[Dict] = []
        for i in range(cfg.num_jaguars):
            angle = (i/cfg.num_jaguars) * 2 * math.pi
            self.jaguars.append({
                "pos": torch.tensor([cx + math.cos(angle)*50, cy + math.sin(angle)*50], device=self.dev, dtype=torch.float32),
                "state": "stalking", # denning, stalking, dashing, feeding
                "target_peccary": -1,
                "hunger": 100.0,
                "timer": 0
            })

        # History tracking
        self.hist_peccary_xy: List[List[Tuple[float, float, bool]]] = [[] for _ in range(cfg.num_peccaries)]
        self.hist_jaguar_xy: List[List[Tuple[float, float, str]]] = [[] for _ in range(cfg.num_jaguars)]
        self.hist_month: List[float] = []
        
        self.herd_centroid = self.cave_pos.clone()
        
        # Kill events for visual blood/spark effects
        self.kill_events: List[Dict] = []
        self.total_kills = 0

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
        
        water_disp = self._interp(WATER_DISPERSION_CURVE, month_frac)
        prey_cond = self._interp(PREY_CONDITION_CURVE, month_frac)
        hunt_supp = self._interp(JAGUAR_HUNT_SUCCESS, month_frac)

        self.hist_month.append(month_frac)

        # 1. Update Peccary Herd (Cohesive flocking behavior)
        alive_peccaries = [p for p in self.peccaries if p["alive"]]
        if alive_peccaries:
            # Calculate centroid
            cx_sum = sum(p["pos"][0].item() for p in alive_peccaries)
            cy_sum = sum(p["pos"][1].item() for p in alive_peccaries)
            self.herd_centroid[0] = cx_sum / len(alive_peccaries)
            self.herd_centroid[1] = cy_sum / len(alive_peccaries)
            
            # Target for the herd as a whole based on season
            # Wet season: scatter around open Cerrado. Dry season: forced to center (karst/river)
            herd_radius = 40 + water_disp * (cfg.clock_radius - 80)
            target_angle = (frame * 0.05) % (2*math.pi) # Slowly orbit
            herd_global_target = torch.tensor([
                cx + math.cos(target_angle) * herd_radius,
                cy + math.sin(target_angle) * herd_radius
            ], device=self.dev, dtype=torch.float32)

            for p in alive_peccaries:
                # Flocking forces
                cohesion = (self.herd_centroid - p["pos"]) * 0.02
                attraction = (herd_global_target - p["pos"]) * 0.03
                
                # Separation
                separation = torch.zeros(2, device=self.dev)
                for other in alive_peccaries:
                    if other is not p:
                        diff = p["pos"] - other["pos"]
                        dist = torch.norm(diff).item()
                        if dist > 0 and dist < 15:
                            separation += (diff / dist) * (15 - dist) * 0.1
                            
                move_vec = cohesion + attraction + separation
                
                # Evade nearby dashing jaguars
                flee = torch.zeros(2, device=self.dev)
                for j in self.jaguars:
                    if j["state"] in ["stalking", "dashing"]:
                        diff = p["pos"] - j["pos"]
                        dist = torch.norm(diff).item()
                        if dist < 60:
                            flee += (diff / dist) * (60 - dist) * 0.2
                
                move_vec += flee
                
                # Apply speed limit (slower if body condition is low)
                speed_cap = cfg.peccary_speed * (0.5 + prey_cond * 0.5)
                norm = torch.norm(move_vec).item()
                if norm > speed_cap:
                    move_vec = (move_vec / norm) * speed_cap
                    
                p["pos"] += move_vec
                
                # Clamp to clock
                dist_from_center = torch.norm(p["pos"] - self.cave_pos).item()
                if dist_from_center > cfg.clock_radius - 10:
                    push_in = (self.cave_pos - p["pos"]) / dist_from_center
                    p["pos"] += push_in * 5.0

        # Record peccary histories
        for i, p in enumerate(self.peccaries):
            self.hist_peccary_xy[i].append((p["pos"][0].item(), p["pos"][1].item(), p["alive"]))

        # 2. Update Jaguars
        for i, jag in enumerate(self.jaguars):
            jag["hunger"] += 0.5 # Gets hungry over time
            
            if jag["state"] == "feeding":
                jag["timer"] -= 1
                if jag["timer"] <= 0:
                    jag["state"] = "denning"
                    jag["hunger"] = 0.0
                    jag["timer"] = 30 # Rest in cave
            
            elif jag["state"] == "denning":
                jag["timer"] -= 1
                target = self.cave_pos
                to_target = target - jag["pos"]
                dist = torch.norm(to_target).item()
                if dist > 5:
                    jag["pos"] += (to_target/dist) * cfg.jaguar_speed
                if jag["timer"] <= 0 and jag["hunger"] > 60:
                    jag["state"] = "stalking"
            
            elif jag["state"] == "stalking":
                # Approach herd centroid slowly from behind
                if alive_peccaries:
                    target = self.herd_centroid
                    to_target = target - jag["pos"]
                    dist = torch.norm(to_target).item()
                    
                    if dist > 80:
                        jag["pos"] += (to_target/dist) * cfg.jaguar_speed * 0.8
                    elif dist < 50:
                        # Close enough, try to dash!
                        jag["state"] = "dashing"
                        # Pick weakest/closest target
                        best_p = -1
                        best_d = 999
                        for pi, p in enumerate(self.peccaries):
                            if p["alive"]:
                                d = torch.norm(p["pos"] - jag["pos"]).item()
                                if d < best_d:
                                    best_d = d
                                    best_p = pi
                        jag["target_peccary"] = best_p
                        jag["timer"] = 15 # Max dash frames
                else:
                    jag["state"] = "denning" # no prey left
                    
            elif jag["state"] == "dashing":
                if jag["target_peccary"] != -1 and self.peccaries[jag["target_peccary"]]["alive"]:
                    p_target = self.peccaries[jag["target_peccary"]]["pos"]
                    to_target = p_target - jag["pos"]
                    dist = torch.norm(to_target).item()
                    
                    if dist < 12:
                        # Attack!
                        if random.random() < hunt_supp:
                            # SUCCESSFUL KILL
                            self.peccaries[jag["target_peccary"]]["alive"] = False
                            jag["state"] = "feeding"
                            jag["timer"] = 25
                            self.kill_events.append({
                                "pos": (p_target[0].item(), p_target[1].item()),
                                "frame": frame
                            })
                            self.total_kills += 1
                        else:
                            # MISSED
                            jag["state"] = "stalking"
                            jag["target_peccary"] = -1
                    else:
                        jag["pos"] += (to_target/dist) * cfg.dash_speed
                        jag["timer"] -= 1
                        
                if jag["timer"] <= 0 and jag["state"] == "dashing":
                    # Ran out of breath, failed hunt
                    jag["state"] = "stalking"
                    jag["target_peccary"] = -1

            # Clamp
            jag["pos"][0] = torch.clamp(jag["pos"][0], cx - cfg.clock_radius + 10, cx + cfg.clock_radius - 10)
            jag["pos"][1] = torch.clamp(jag["pos"][1], cy - cfg.clock_radius + 10, cy + cfg.clock_radius - 10)

            self.hist_jaguar_xy[i].append((jag["pos"][0].item(), jag["pos"][1].item(), jag["state"]))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class ApexRenderer:

    def __init__(self, cfg: ApexConfig, sim: ApexSim):
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
            f'style="background-color:#141712; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        
        # Karst/Gallery Central Refuge Gradient
        svg.append(
            '<radialGradient id="karstBg">'
            '<stop offset="0%" stop-color="#1b3021" stop-opacity="0.9"/>'
            '<stop offset="30%" stop-color="#132417" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#141712" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        # Jaguar Rosette pattern for the background or panels (conceptual)
        svg.append(
            '<pattern id="rosette" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="20" cy="20" r="10" fill="none" stroke="#2a231b" stroke-width="3" stroke-dasharray="8 4"/>'
            '<circle cx="20" cy="20" r="3" fill="#2a231b"/>'
            '</pattern>'
        )
        svg.append('</defs>')

        # Background Landscape
        svg.append(f'<rect width="{w}" height="{h}" fill="#141712"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#rosette)" opacity="0.03"/>')
        
        # Open Cerrado Zone constraints
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="#191c15" opacity="0.8"/>')
        
        # Central Karst Cave & Gallery Forest (Jaguar Refuge)
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{cfg.karst_cave_radius * 2}" fill="url(#karstBg)"/>')
        
        # Cave mouth
        svg.append(f'<path d="M{cx-20} {cy+10} Q{cx} {cy-15} {cx+20} {cy+10} L{cx} {cy+20} Z" fill="#0c100e"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ffb300" font-weight="bold">'
            f'ECO-SIM: Jaguar × White-lipped Peccaries    - APEX CLOCK</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#ffecb3">'
            f'Predator-Prey seasonal dynamics, herd cohesion, and karst refuges</text>'
        )

        # Clock Face
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#795548" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 5)
            ly2 = cy + math.sin(angle) * (R - 5)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#3e2723" stroke-width="2"/>'
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

        draw_arc(11, 4.5, R + 10, "#4caf50", "Prey Dispersal (Wet)", 0.4)
        draw_arc(7, 10, R + 38, "#ff3d00", "Peak Hunting Success", 0.5)
        draw_arc(7.5, 9.5, R + 10, "#00acc1", "Water Concentration", 0.4)

        # Kill Events (Blood splat / impact)
        for ke in sim.kill_events:
            kx, ky = ke["pos"]
            kf = ke["frame"]
            
            ops = ";".join("0.0" if fi < kf else f"{max(0, 0.9 - (fi-kf)*0.02):.2f}" for fi in range(F))
            szs = ";".join("0.0" if fi < kf else f"{min(20, (fi-kf)*2):.1f}" for fi in range(F))
            
            svg.append(
                f'<circle cx="{kx:.0f}" cy="{ky:.0f}" fill="none" stroke="#d50000" stroke-width="2" opacity="0.0">'
                f'<animate attributeName="r" values="{szs}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # Peccaries (Queixadas) - Dark grey dots moving as a flock
        for i in range(cfg.num_peccaries):
            hist = sim.hist_peccary_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            ops = ";".join("0.9" if h[2] else "0.0" for h in hist) # Disappear if killed
            
            svg.append(
                f'<circle r="3" fill="#546e7a" stroke="#263238" stroke-width="0.5">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Jaguars (Onças) - Golden/Orange large dots with intense action
        for i in range(cfg.num_jaguars):
            hist = sim.hist_jaguar_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            
            # Trail path (faint, highlights stalking vs dashing)
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::3] if h[2] != "denning"]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#ffb300" stroke-width="1.2" stroke-dasharray="2,3" opacity="0.3"/>'
                )
            
            # Change outline color based on state (Red when dashing/feeding)
            stroke_cols = ";".join("#d50000" if h[2] in ["dashing", "feeding"] else "#ffb300" for h in hist)
            
            # Animated Jaguar body
            svg.append(
                f'<circle r="5.5" fill="#f57f17" stroke-width="2">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="stroke" values="{stroke_cols}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            # Center rosette dot
            svg.append(
                f'<circle r="1.5" fill="#212121">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Clock hand
        hand_x = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#ffb300" stroke-width="2.5" stroke-linecap="round" opacity="0.6">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#141712" stroke="#ffb300" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#f57f17"/>')


        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 215
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1c1813" rx="8" '
                   f'stroke="#ffb300" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ffb300" font-size="15" font-weight="bold">'
                   f'Apex Predator-Prey Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Dry season compresses peccaries into karst.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'Bottleneck raises ambush rates for predators.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'Hunting success rises as prey condition falls,</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'routes narrow near gallery forests and rivers.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'Predation feeds back on vigilance and habitat.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#d50000" font-size="15">'
                   f'Red rings = ambush outcomes; caves as refuges.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 174
        
        deaths = sum(1 for p in sim.peccaries if not p["alive"])
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1c1813" rx="8" '
                   f'stroke="#d50000" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#d50000" font-size="15" font-weight="bold">'
                   f'Trophic Regulation Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#ff5722" font-size="15">'
                   f'Recorded ambush events: {sim.total_kills:,.0f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#ffb300" font-size="15">'
                   f'Peccary Herd Survival: {cfg.num_peccaries - deaths} / {cfg.num_peccaries}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#00acc1" font-size="15">'
                   f'Prey Concentrated by Water: {"Yes (Dry Season Peak)" if max(sim.hist_month) > 7 and max(sim.hist_month) < 10 else "Dispersed"}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'Regulation grows as water scarcity increases.</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 173
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1c1813" rx="8" '
                   f'stroke="#5c6bc0" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#5c6bc0" font-size="15" font-weight="bold">'
                   f'Phenological Curves</text>')

        curves = [
            (WATER_DISPERSION_CURVE, "#00acc1", "Water Scarcity"),
            (PREY_CONDITION_CURVE, "#8bc34a", "Prey Health"),
            (JAGUAR_HUNT_SUCCESS, "#d50000", "Ambush Success Rate"),
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
    print(f" — Onça-pintada ↔ Queixadas Clock on {CONFIG.device}...")

    sim = ApexSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    survivors = sum(1 for p in sim.peccaries if p["alive"])
    print(f"Done: {sim.total_kills} kills. Herd survival: {survivors}/{CONFIG.num_peccaries}")

    print("Generating SVG...")
    renderer = ApexRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_66')
    return svg_content


if __name__ == "__main__":
    main()
