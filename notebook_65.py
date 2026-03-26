# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 65: Beija-flor ↔ Canela-de-ema — Pyrophytic Pollination Clock
# INTERVENTION 11/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_65.py — Hummingbirds (Beija-flor) ↔ Canela-de-ema (Vellozia squamata):
Notebook Differentiation:
- Differentiation Focus: Beija-flor ↔ Canela-de-ema — Pyrophytic Pollination Clock emphasizing riparian buffer integrity.
- Indicator species: Tamandua-mirim (Tamandua tetradactyla).
- Pollination lens: flowering phenology under drought stress.
- Human impact lens: tourism peak season stress.

                 Pyrophytic Pollination & Fire Phenological Clock

Models the profound adaptation of Cerrado flora to fire (Pyrophytism), focusing
on the iconic 'Canela-de-ema' bush. Fire is a natural driver of the Cerrado
ecosystem, particularly in the late dry season (Aug-Sep).

The radial phenological clock maps:
  • Dry season progression and accumulation of dry biomass (fuel).
  • Wildfire events (natural and anthropogenic) sweeping the landscape in Aug/Sep.
  • Fire-stimulated blooming (Pyrophytic response): Vellozia drops its scorched
    outer leaves and produces massive, synchronized purple blooms within days 
    after a fire, utilizing heavily protected underground reserves extending 
    from the CMN (Notebook 59).
  • Hummingbird (Beija-flor) territorial foraging: Highly aggressive birds
    rely on this intense post-fire resource precisely when the rest of the 
    landscape is completely devoid of nectar.

Scientific References:
  - Simon, M.F. et al. (2009). "Recent assembly of the Cerrado, a neotropical
    plant diversity hotspot, by in situ evolution of adaptations to fire." PNAS.
  - Jacomassa, F.A.F. & Schiavini, I. (2012). "Seed dispersal of Vellozia squamata..."
  - Fiedler, N.C. et al. (2004). "Fire behavior and tree mortality in Cerrado."

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

# Dry Biomass Accumulation (Fuel for fire)
# High during the dry season, consumed by fire, slow regrowth in wet season.
BIOMASS_FUEL_CURVE = [0.20, 0.15, 0.20, 0.40, 0.60, 0.80, 0.95, 1.00, 0.80, 0.45, 0.25, 0.20]
# Clustered in late dry season (Aug-Sep). Often triggered by dry lightning
# or human activities.
FIRE_RISK_CURVE = [0.05, 0.05, 0.10, 0.20, 0.50, 0.70, 0.90, 1.00, 0.85, 0.40, 0.10, 0.05]

# Fire-Stimulated Blooming (Canela-de-ema)
# Baseline is low; spikes dramatically 1-3 weeks AFTER a fire event (Sep/Oct).
# In this simulation, tightly coupled to the drop in Biomass (the fire).
VELLOZIA_BLOOM_CURVE = [0.05, 0.05, 0.05, 0.10, 0.20, 0.40, 0.60, 0.95, 0.80, 0.50, 0.20, 0.05]

# Hummingbird Activity (Territoriality & Nectar demand)
# Driven by nectar need; they concentrate heavily on Vellozia when it blooms.
HUMMINGBIRD_ACTIVITY = [
    0.30, 0.20, 0.10, 0.10, 0.10, 0.15, 0.40, 0.70, 1.00, 0.90, 0.60, 0.40
]


@dataclass
class PyrophyticConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_plants: int = 25
    flowers_per_plant: int = 4
    
    num_birds: int = 8
    bird_speed: float = 6.0
    
    fire_spread_speed: float = 30.0


CONFIG = PyrophyticConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class PyrophyticSim:

    def __init__(self, cfg: PyrophyticConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Canela-de-ema Plants ---
        self.plants: List[Dict] = []
        for _ in range(cfg.num_plants):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(40, R - 30)
            self.plants.append({
                "pos": (cx + math.cos(angle)*r, cy + math.sin(angle)*r),
                "burnt": False,
                "blooming": 0.0,
                "nectar": 0.0
            })

        # --- Hummingbirds ---
        self.birds: List[Dict] = []
        # Birds have fixed territories they return to
        for i in range(cfg.num_birds):
            angle = (i / cfg.num_birds) * 2 * math.pi
            r = R - 60
            tx, ty = cx + math.cos(angle)*r, cy + math.sin(angle)*r
            self.birds.append({
                "pos": torch.tensor([tx, ty], device=self.dev, dtype=torch.float32),
                "territory": torch.tensor([tx, ty], device=self.dev, dtype=torch.float32),
                "state": "perching", # perching, foraging, chasing
                "target_plant": -1,
                "energy": 100.0
            })

        self.hist_xy: List[List[Tuple[float, float, float]]] = [[] for _ in range(cfg.num_birds)]
        self.hist_month: List[float] = []
        
        # Fire Event tracking
        self.fire_active = False
        self.fire_radius = 0.0
        self.fire_origin = (cx, cy)
        self.fire_frame_start = -1
        self.fire_drawn_events: List[Dict] = [] # to render the flame waves
        
        self.total_nectar_drunk = 0.0
        self.total_flowers_pollinated = 0

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
        
        fuel = self._interp(BIOMASS_FUEL_CURVE, month_frac)
        fire_risk = self._interp(FIRE_RISK_CURVE, month_frac)
        bloom_potential = self._interp(VELLOZIA_BLOOM_CURVE, month_frac)
        bird_act = self._interp(HUMMINGBIRD_ACTIVITY, month_frac)

        self.hist_month.append(month_frac)

        # 1. Fire Dynamics
        if not self.fire_active:
            # Trigger fire in peak dry season (guaranteed near mid August)
            if self.fire_frame_start == -1 and month_frac > 7.5 and fire_risk > 0.8:
                self.fire_active = True
                self.fire_radius = 0.0
                # Start fire somewhere in the Cerrado
                fa = random.uniform(0, 2*math.pi)
                fr = random.uniform(0, cfg.clock_radius - 80)
                self.fire_origin = (cx + math.cos(fa)*fr, cy + math.sin(fa)*fr)
                self.fire_frame_start = frame
                self.fire_drawn_events.append({
                    "origin": self.fire_origin,
                    "frame": frame
                })
        else:
            # Fire spreading!
            self.fire_radius += cfg.fire_spread_speed
            
            # Burn plants in radius
            for p in self.plants:
                if not p["burnt"]:
                    dist = math.sqrt((p["pos"][0] - self.fire_origin[0])**2 + (p["pos"][1] - self.fire_origin[1])**2)
                    if dist <= self.fire_radius:
                        p["burnt"] = True
                        p["nectar"] = 0.0
            
            # Fire dies out if it reaches edge
            if self.fire_radius > cfg.clock_radius * 2:
                self.fire_active = False

        # Reset plants when rains return
        if month_frac > 10.5 or (month_frac > 0 and month_frac < 2): # Nov-Feb
            for p in self.plants:
                p["burnt"] = False

        # 2. Plant Blooming (Pyrophytic response & Nectar)
        for p in self.plants:
            if p["burnt"] and bloom_potential > 0.3:
                # Fire stimulates massive bloom
                p["blooming"] = bloom_potential * (0.8 + 0.2*random.random())
                # Generate nectar
                p["nectar"] = min(1.0, p["nectar"] + 0.05 * bloom_potential)
            else:
                p["blooming"] *= 0.8 # fade out
                p["nectar"] *= 0.9

        # 3. Hummingbird Foraging
        for i, bird in enumerate(self.birds):
            pos = bird["pos"]
            is_active = (i / cfg.num_birds) < bird_act
            
            if not is_active:
                bird["state"] = "perching"
                bird["energy"] = 100.0
                target = bird["territory"]
                to_target = target - pos
                dist = torch.norm(to_target).item()
                if dist > 3.0:
                    dir_vec = to_target / dist
                    bird["pos"] += dir_vec * cfg.bird_speed
                else:
                    bird["pos"] = target
            else:
                # Fast metabolism: energy drops constantly
                bird["energy"] -= 2.0
                
                # State transitions
                if bird["energy"] < 50 and bird["state"] == "perching":
                    bird["state"] = "foraging"
                    bird["target_plant"] = -1
                elif bird["energy"] > 90 and bird["state"] == "foraging":
                    bird["state"] = "perching"
                    bird["target_plant"] = -1

                if bird["state"] == "foraging":
                    if bird["target_plant"] == -1:
                        # Find highest nectar plant, preferring within territory
                        best_p = -1
                        best_score = -9999
                        for pi, p in enumerate(self.plants):
                            if p["nectar"] > 0.2:
                                # distance from bird to plant
                                dt = math.sqrt((pos[0].item() - p["pos"][0])**2 + (pos[1].item() - p["pos"][1])**2)
                                # distance from plant to territory center
                                d_terr = math.sqrt((bird["territory"][0].item() - p["pos"][0])**2 + (bird["territory"][1].item() - p["pos"][1])**2)
                                
                                score = p["nectar"] * 100 - dt * 0.5 - d_terr * 1.5
                                if score > best_score:
                                    best_score = score
                                    best_p = pi
                                    
                        if best_p != -1:
                            bird["target_plant"] = best_p
                
                # Chasing intruders (Hummingbird aggressive territoriality)
                if bird["state"] == "perching":
                    for j, other in enumerate(self.birds):
                        if i != j and other["state"] == "foraging":
                            d_intruder = torch.norm(other["pos"] - bird["territory"]).item()
                            if d_intruder < 40: # IN MY TERRITORY!
                                bird["state"] = "chasing"
                                bird["target_plant"] = j # Target is the other bird
                                break

                # Move
                target = None
                if bird["state"] == "perching":
                    target = bird["territory"]
                elif bird["state"] == "foraging" and bird["target_plant"] != -1:
                    p = self.plants[bird["target_plant"]]
                    target = torch.tensor(p["pos"], device=self.dev, dtype=torch.float32)
                elif bird["state"] == "chasing":
                    target = self.birds[bird["target_plant"]]["pos"]
                    # Give up chase if far
                    if torch.norm(bird["pos"] - bird["territory"]).item() > 80:
                        bird["state"] = "perching"

                if target is not None:
                    to_target = target - pos
                    dist = torch.norm(to_target).item()
                    if dist > 3.0:
                        speed_mult = 1.8 if bird["state"] == "chasing" else 1.0
                        dir_vec = to_target / dist
                        # Hummingbird erratic flight (Ziz-zag)
                        jitter = torch.tensor([random.uniform(-3,3), random.uniform(-3,3)], device=self.dev)
                        bird["pos"] += dir_vec * cfg.bird_speed * speed_mult + jitter
                    else:
                        if bird["state"] == "foraging" and bird["target_plant"] != -1:
                            # Drink nectar
                            p = self.plants[bird["target_plant"]]
                            drunk = min(p["nectar"], 50.0)
                            p["nectar"] -= drunk
                            bird["energy"] += drunk * 2.5
                            self.total_nectar_drunk += drunk
                            self.total_flowers_pollinated += 1
                            bird["target_plant"] = -1 # look for next
                        elif bird["state"] == "chasing":
                            bird["state"] = "perching"

            # Clamp
            bird["pos"][0] = torch.clamp(bird["pos"][0], cx - cfg.clock_radius + 10, cx + cfg.clock_radius - 10)
            bird["pos"][1] = torch.clamp(bird["pos"][1], cy - cfg.clock_radius + 10, cy + cfg.clock_radius - 10)

            opacity = 1.0 if is_active else 0.0
            self.hist_xy[i].append((bird["pos"][0].item(), bird["pos"][1].item(), opacity))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class PyrophyticRenderer:

    def __init__(self, cfg: PyrophyticConfig, sim: PyrophyticSim):
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
            f'style="background-color:#141110; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        svg.append(
            '<radialGradient id="ashBg">'
            '<stop offset="0%" stop-color="#2c2826" stop-opacity="0.9"/>'
            '<stop offset="70%" stop-color="#211d1c" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#141110" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="fireGrad">'
            '<stop offset="0%" stop-color="#ff9800" stop-opacity="0.1"/>'
            '<stop offset="80%" stop-color="#ff5722" stop-opacity="0.6"/>'
            '<stop offset="95%" stop-color="#ff9800" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#d50000" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # Background Landscape
        svg.append(f'<rect width="{w}" height="{h}" fill="#141110"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="url(#ashBg)"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ff9800" font-weight="bold">'
            f'ECO-SIM: Hummingbird × Vellozia    - Pyrophytic Pollination Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#ffcc80">'
            f'Late dry-season wildfires & fire-stimulated blooming dynamics</text>'
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

        draw_arc(4.5, 7.5, R + 10, "#fb8c00", "Fuel Accumulation", 0.4)
        draw_arc(7.5, 9.5, R + 22, "#ff3d00", "Wildfire Season", 0.5)
        draw_arc(8.5, 10.5, R + 10, "#e040fb", "Pyrophytic Blooming", 0.4)

        # Fire Wave Render (Expanding Ring)
        for fv in sim.fire_drawn_events:
            fx, fy = fv["origin"]
            start_f = fv["frame"]
            # Fire lasts about 20 frames visually covering terrain
            
            r_vals = ";".join("0" if fi < start_f else f"{(fi-start_f)*cfg.fire_spread_speed:.0f}" for fi in range(F))
            op_vals = ";".join("0.0" if (fi < start_f or fi > start_f+25) else "0.7" for fi in range(F))
            
            svg.append(
                f'<circle cx="{fx:.0f}" cy="{fy:.0f}" fill="url(#fireGrad)" stroke="#ffab00" stroke-width="2" opacity="0.0">'
                f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # Canela-de-ema Plants
        for pi, p in enumerate(sim.plants):
            px, py = p["pos"]
            
            # Stem (Thick black/grey trunk typical of Vellozia)
            svg.append(f'<line x1="{px:.0f}" y1="{py+8:.0f}" x2="{px:.0f}" y2="{py:.0f}" stroke="#212121" stroke-width="4" stroke-linecap="round"/>')
            
            # Scorched / Green leaves (depends on fire)
            leaf_cols = ";".join("#4caf50" if sim._interp(BIOMASS_FUEL_CURVE, (fi/F)*12) > 0.4 and fi < sim.fire_frame_start else "#212121" for fi in range(F))
            svg.append(
                f'<path d="M{px:.0f},{py:.0f} L{px-6:.0f},{py-10:.0f} M{px:.0f},{py:.0f} L{px+6:.0f},{py-10:.0f}" stroke="#4caf50" stroke-width="1.5">'
                f'<animate attributeName="stroke" values="{leaf_cols}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</path>'
            )

            # Pyrophytic Blooms (Vibrant purple)
            b_vals = ";".join(
                f"{max(0, sim._interp(VELLOZIA_BLOOM_CURVE, (fi/F)*12) * 6.0 if fi > sim.fire_frame_start else 0.0):.1f}" 
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py-5:.0f}" fill="#e040fb" opacity="0.85">'
                f'<animate attributeName="r" values="{b_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Hummingbirds
        for i in range(cfg.num_birds):
            hist = sim.hist_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            ops = ";".join(str(round(h[2], 2)) for h in hist)
            
            # Flight path
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::3] if h[2] > 0.5]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#00e5ff" stroke-width="0.8" opacity="0.15"/>'
                )
            
            # Animated Bird Core (Emerald Green)
            svg.append(
                f'<circle r="3" fill="#00e676">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            # Wings (Blue/Cyan rapid flapping!)
            wing_r = ";".join("4" if fi % 2 == 0 else "1" for fi in range(F))
            svg.append(
                f'<ellipse rx="4" ry="1.5" fill="none" stroke="#00b0ff" stroke-width="1.5">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="ry" values="{wing_r}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )

        # Clock hand
        hand_x = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#ff9800" stroke-width="2.5" stroke-linecap="round" opacity="0.8">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#141110" stroke="#ff9800" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#e040fb"/>')


        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 272
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1e1816" rx="8" '
                   f'stroke="#ff5722" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ff5722" font-size="15" font-weight="bold">'
                   f'Pyrophytism & Fire Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Fire is a natural Cerrado driver. At peak</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'dry season (Aug-Sep), biomass fuels wildfires.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'Canela-de-ema (Vellozia squamata) is adapted:</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'its thick trunk survives flames; intense heat</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#e040fb" font-size="15">'
                   f'triggers a synchronized purple bloom later.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#00e676" font-size="15">'
                   f'Hummingbirds defend rare post-fire resources.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 142
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1e1816" rx="8" '
                   f'stroke="#e040fb" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#e040fb" font-size="15" font-weight="bold">'
                   f'Ecological Resilience Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#00e676" font-size="15">'
                   f'Flowers Pollinated: {sim.total_flowers_pollinated:,.0f} visits</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#ff9800" font-size="15">'
                   f'Total Nectar Drunk: {sim.total_nectar_drunk:,.1f} mg</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#8d6e63" font-size="15">'
                   f'Vellozia bushes triggered: {sum(1 for p in sim.plants if p["burnt"])} / {cfg.num_plants}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'Hummingbird population: {cfg.num_birds} (highly territorial)</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 142
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1e1816" rx="8" '
                   f'stroke="#5c6bc0" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#5c6bc0" font-size="15" font-weight="bold">'
                   f'Phenological Curves</text>')

        curves = [
            (BIOMASS_FUEL_CURVE, "#4caf50", "Fuel Biomass"),
            (FIRE_RISK_CURVE, "#ff3d00", "Fire Risk"),
            (VELLOZIA_BLOOM_CURVE, "#e040fb", "Canela-de-ema"),
            (HUMMINGBIRD_ACTIVITY, "#00e676", "Foraging Need"),
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
    print(f" — Beija-flor ↔ Canela-de-ema Clock on {CONFIG.device}...")

    sim = PyrophyticSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_flowers_pollinated} flowers pollinated, {sim.total_nectar_drunk:.1f} nectar drunk.")

    print("Generating SVG...")
    renderer = PyrophyticRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_65')
    return svg_content


if __name__ == "__main__":
    main()
