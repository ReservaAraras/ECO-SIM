# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 63: Formiga-Cerrado ↔ Extrafloral Nectaries (EFN) — Mutualistic Defense Clock
# INTERVENTION 9/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_63.py — Cerrado Ants ↔ Extrafloral Nectaries (EFNs):
Notebook Differentiation:
- Differentiation Focus: Formiga-Cerrado ↔ Extrafloral Nectaries (EFN) — Mutualistic Defense Clock emphasizing illegal extraction gaps.
- Indicator species: Cascavel (Crotalus durissus).
- Pollination lens: nectar robbing effects.
- Human impact lens: agroforestry edge subsidies.

                 Mutualistic Defense Phenological Clock

Models the highly specialized mutualism between predatory Cerrado ants
(e.g., Camponotus spp., Ectatomma spp.) and plants bearing extrafloral
nectaries (e.g., Qualea grandiflora "Pau-terra", Caryocar brasiliense).
EFNs exude nectar on leaves/stems to attract ants, which in turn patrol
the plant and aggressively defend it from insect herbivores (caterpillars,
leaf beetles) during the vulnerable young-leaf stage.

The notebook therefore treats EFN defense as a conditional contract rather than
as a permanent harmony: plants invest in nectar when tissues are vulnerable,
ants recruit when rewards overlap with herbivore pressure, and the ecological
benefit is judged by tissue saved during the attack window.

The radial phenological clock maps:
  • Plant leaf flush and EFN nectar production (pulsing in the early wet season).
  • Herbivore pressure (outbreaks synchronized with the wet season leaf flush).
  • Ant patrolling activity and recruitment dynamics on host plants.
  • Defensive events where ants repel or prey upon herbivores.
  • Dry season dormancy: leaves harden or drop, EFN activity ceases,
    and ants retreat to soil nests to avoid heat/fire.

Scientific References:
  - Oliveira, P.S. (1997). "The ecological function of extrafloral nectaries:
    herbivore deterrence by visiting ants and reproductive output in Caryocar."
    Journal of Ecology.
  - Del-Claro, K. et al. (1996). "Herbivore dispersal by ants..."
    Journal of Tropical Ecology.
  - Rico-Gray, V. & Oliveira, P.S. (2007). "The Ecology and Evolution of
    Ant-Plant Interactions."

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

# Leaf Flush & EFN Secretion
# High during the transition to the wet season (Sep-Nov) when plants produce young, soft leaves.
EFN_PRODUCTION_CURVE = [0.60, 0.70, 0.80, 0.90, 0.90, 0.80, 0.60, 0.45, 0.30, 0.40, 0.50, 0.55]
# Herbivores emerge to exploit the soft, nutrient-rich young leaves.
HERBIVORE_PRESSURE_CURVE = [
    0.50,   # JAN
    0.30,   # FEV
    0.20,   # MAR
    0.10,   # ABR
    0.05,   # MAI
    0.00,   # JUN
    0.00,   # JUL
    0.10,   # AGO
    0.60,   # SET — outbreaks begin
    0.90,   # OUT — peak herbivory threat
    1.00,   # NOV
    0.80,   # DEZ
]

# Ant Patrolling Activity
# Driven by EFN availability and favourable foraging temperatures.
# Drops in the cold/dry peak (Jun-Jul).
ANT_PATROL_CURVE = [0.60, 0.70, 0.80, 0.90, 0.90, 0.80, 0.60, 0.45, 0.30, 0.40, 0.50, 0.55]


@dataclass
class EFNMutualismConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_plants: int = 8
    leaves_per_plant: int = 6
    num_ants: int = 120
    
    ant_speed: float = 3.0
    herbivore_spawn_prob: float = 0.08
    defense_radius: float = 12.0


CONFIG = EFNMutualismConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class EFNMutualismSim:

    def __init__(self, cfg: EFNMutualismConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- EFN Plants (e.g., Qualea, Caryocar) ---
        self.plants: List[Dict] = []
        for i in range(cfg.num_plants):
            angle = (i / cfg.num_plants) * 2 * math.pi + random.uniform(-0.2, 0.2)
            r = random.uniform(60, R - 60)
            px = cx + math.cos(angle) * r
            py = cy + math.sin(angle) * r
            
            # Plant has several primary leaves/branches with EFNs
            leaves = []
            for _ in range(cfg.leaves_per_plant):
                la = random.uniform(0, 2*math.pi)
                lr = random.uniform(15, 35)
                leaves.append({
                    "rel_x": math.cos(la) * lr,
                    "rel_y": math.sin(la) * lr,
                    "herbivore_timer": 0,
                    "defended_events": 0,
                    "damage": 0.0
                })

            self.plants.append({
                "pos": (px, py),
                "leaves": leaves,
                "efn_active": False,
                "ant_count": 0
            })

        # --- Ants ---
        self.ants: List[Dict] = []
        for _ in range(cfg.num_ants):
            self.ants.append({
                "pos": torch.tensor([cx, cy], device=self.dev, dtype=torch.float32),
                "state": "nest", # nest, foraging, defending
                "target_plant": -1,
                "target_leaf": -1,
                "patrol_timer": 0
            })

        self.hist_xy: List[List[Tuple[float, float]]] = [[] for _ in range(cfg.num_ants)]
        self.hist_month: List[float] = []
        
        # Herbivores (Caterpillars/Beetles trying to eat leaves)
        # Stored as active threats per plant-leaf
        self.herbivores: List[Dict] = []
        
        self.defenses_won = 0
        self.leaves_lost = 0

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
        
        efn_level = self._interp(EFN_PRODUCTION_CURVE, month_frac)
        herb_pressure = self._interp(HERBIVORE_PRESSURE_CURVE, month_frac)
        ant_act = self._interp(ANT_PATROL_CURVE, month_frac)

        self.hist_month.append(month_frac)

        # 1. Herbivore Spawning (driven by leaf flush / herb_pressure)
        # Herbivores attack leaves independent of ants
        if herb_pressure > 0.1 and random.random() < cfg.herbivore_spawn_prob * herb_pressure:
            pi = random.randint(0, cfg.num_plants - 1)
            li = random.randint(0, cfg.leaves_per_plant - 1)
            leaf = self.plants[pi]["leaves"][li]
            if leaf["damage"] < 1.0 and leaf["herbivore_timer"] == 0:
                leaf["herbivore_timer"] = 1 # Active threat
                self.herbivores.append({"pi": pi, "li": li, "hp": 1.0})

        # Process active herbivores (damaging leaves)
        surviving_herbs = []
        for h in self.herbivores:
            pi, li = h["pi"], h["li"]
            leaf = self.plants[pi]["leaves"][li]
            if h["hp"] > 0:
                leaf["damage"] = min(1.0, leaf["damage"] + 0.02) # Herbivore eats leaf
                if leaf["damage"] >= 1.0:
                    self.leaves_lost += 1
                    leaf["herbivore_timer"] = 0
                else:
                    surviving_herbs.append(h)
            else:
                leaf["herbivore_timer"] = 0
        self.herbivores = surviving_herbs

        # 2. Ant Foraging & Patrol
        for i, ant in enumerate(self.ants):
            pos = ant["pos"]
            
            # Nest state check
            if ant["state"] == "nest":
                if random.random() < ant_act * 0.2:
                    ant["state"] = "foraging"
                    ant["target_plant"] = -1
            else:
                # Decide target
                if ant["target_plant"] == -1:
                    # Look for plants with EFN
                    if efn_level > 0.1:
                        # Find nearest non-crowded plant
                        best_p = -1
                        best_d = float('inf')
                        for pi, p in enumerate(self.plants):
                            d = math.sqrt((pos[0].item() - p["pos"][0])**2 + (pos[1].item() - p["pos"][1])**2)
                            if d < best_d and random.random() < 0.5:
                                best_d = d
                                best_p = pi
                        if best_p != -1:
                            ant["target_plant"] = best_p
                            ant["target_leaf"] = random.randint(0, cfg.leaves_per_plant - 1)
                            ant["patrol_timer"] = 0
                    else:
                        # No EFNs, go back to nest
                        ant["state"] = "nest"
                        
            # Move
            if ant["state"] == "nest":
                target = torch.tensor([cx, cy], device=self.dev, dtype=torch.float32)
                # If close to nest, disappear/stay
                dist = torch.norm(target - pos).item()
                if dist > 5.0:
                    dir_vec = (target - pos) / dist
                    ant["pos"] += dir_vec * cfg.ant_speed
                else:
                    ant["pos"] = target
            elif ant["state"] in ["foraging", "defending"]:
                p = self.plants[ant["target_plant"]]
                leaf = p["leaves"][ant["target_leaf"]]
                tx = p["pos"][0] + leaf["rel_x"]
                ty = p["pos"][1] + leaf["rel_y"]
                target = torch.tensor([tx, ty], device=self.dev, dtype=torch.float32)
                
                dist = torch.norm(target - pos).item()
                if dist > 3.0:
                    dir_vec = (target - pos) / dist
                    ant["pos"] += dir_vec * cfg.ant_speed * (1.5 if ant["state"] == "defending" else 1.0)
                else:
                    # At leaf
                    ant["patrol_timer"] += 1
                    
                    # Defend leaf!
                    if leaf["herbivore_timer"] > 0:
                        ant["state"] = "defending"
                        # Ant attacks herbivore
                        for h in self.herbivores:
                            if h["pi"] == ant["target_plant"] and h["li"] == ant["target_leaf"]:
                                h["hp"] -= 0.5 # Ant bites herbivore
                                if h["hp"] <= 0:
                                    self.defenses_won += 1
                                    leaf["defended_events"] += 1
                                    ant["state"] = "foraging"
                    else:
                        ant["state"] = "foraging"
                        
                    # Switch leaf or drop off plant
                    if ant["patrol_timer"] > 15:
                        if random.random() < 0.2:
                            ant["target_plant"] = -1 # leave plant
                        else:
                            ant["target_leaf"] = random.randint(0, cfg.leaves_per_plant - 1)
                            ant["patrol_timer"] = 0

            self.hist_xy[i].append((ant["pos"][0].item(), ant["pos"][1].item()))

        # Dry season leaf reset
        if efn_level < 0.05 and month_frac > 4 and month_frac < 6:
            # Leaves senesce and prep for next flush
            for p in self.plants:
                for l in p["leaves"]:
                    l["damage"] = 0.0
                    l["defended_events"] = 0


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class EFNMutualismRenderer:

    def __init__(self, cfg: EFNMutualismConfig, sim: EFNMutualismSim):
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
            f'style="background-color:#141a16; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        svg.append(
            '<radialGradient id="plantBg">'
            '<stop offset="0%" stop-color="#1b261e" stop-opacity="0.9"/>'
            '<stop offset="70%" stop-color="#172e21" stop-opacity="0.4"/>'
            '<stop offset="100%" stop-color="#141a16" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # Background Landscape
        svg.append(f'<rect width="{w}" height="{h}" fill="#141a16"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="url(#plantBg)"/>')
        
        # Central Ant Nest
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="20" fill="#2e1b12" stroke="#5d4037" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx-5}" cy="{cy+5}" r="3" fill="#140c08"/>')
        svg.append(f'<circle cx="{cx+8}" cy="{cy-4}" r="4" fill="#140c08"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#81c784" font-weight="bold">'
            f'ECO-SIM: Ant × EFN    - Mutualistic Defense Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#a5d6a7">'
            f'Predatory ants patrolling Extrafloral Nectaries during leaf flush | </text>'
        )

        # Clock Face
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#81c784" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 5)
            ly2 = cy + math.sin(angle) * (R - 5)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#2e4635" stroke-width="2"/>'
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
                       f'text-anchor="middle" opacity="0.9">{sanitize_svg_text(label)}</text>')

        draw_arc(8, 0.5, R + 10, "#4caf50", "Leaf Flush & EFN Active", 0.4)
        draw_arc(8, 11, R + 22, "#e53935", "High Herbivory Threat", 0.3)
        draw_arc(5, 7.5, R + 10, "#795548", "Dry Season Dormancy", 0.3)

        # Draw Plants and Leaves
        for pi, p in enumerate(sim.plants):
            px, py = p["pos"]
            
            # Plant stem
            svg.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="4" fill="#5d4037"/>')
            
            for li, leaf in enumerate(p["leaves"]):
                lx = px + leaf["rel_x"]
                ly = py + leaf["rel_y"]
                
                # Branch
                svg.append(f'<line x1="{px:.0f}" y1="{py:.0f}" x2="{lx:.0f}" y2="{ly:.0f}" stroke="#4caf50" stroke-width="2" opacity="0.6"/>')
                
                # Leaf shape (varies by damage)
                dmg = leaf["damage"]
                leaf_col = "#cddc39" if dmg < 0.3 else ("#8bc34a" if dmg < 0.7 else "#9e9d24")
                if dmg >= 1.0: leaf_col = "#5d4037" # dead/eaten
                
                angle = math.atan2(leaf["rel_y"], leaf["rel_x"])
                # Rotate a small ellipse for the leaf
                svg.append(
                    f'<g transform="translate({lx:.0f}, {ly:.0f}) rotate({math.degrees(angle):.0f})">'
                    f'<ellipse cx="6" cy="0" rx="{8*(1-dmg*0.5):.1f}" ry="{4*(1-dmg*0.5):.1f}" fill="{leaf_col}"/>'
                    f'</g>'
                )
                
                # EFN Nectar Drop (if active)
                # Pulsates dynamically based on global EFN curve
                efn_vals = ";".join(f"{sim._interp(EFN_PRODUCTION_CURVE, (fi/F)*12)*3:.1f}" for fi in range(F))
                svg.append(
                    f'<circle cx="{lx:.0f}" cy="{ly:.0f}" fill="#ffeb3b" opacity="0.8">'
                    f'<animate attributeName="r" values="{efn_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )

                # Defense Sparks (if defended)
                if leaf["defended_events"] > 0:
                    svg.append(f'<circle cx="{lx:.0f}" cy="{ly:.0f}" r="8" fill="none" stroke="#00e5ff" stroke-width="1.5" stroke-dasharray="2,2" opacity="0.4"/>')

        # Ants (Orange/Brown dots patrolling fast)
        for i in range(cfg.num_ants):
            hist = sim.hist_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            
            # Trail path (faint)
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::4]]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#e65100" stroke-width="0.8" opacity="0.2"/>'
                )
            
            # Animated Ant body
            svg.append(
                f'<circle r="2" fill="#ff6d00">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Clock hand
        hand_x = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#81c784" stroke-width="2" stroke-linecap="round" opacity="0.8">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<text font-weight="bold" x="{cx}" y="{cy + 25}" font-size="15" fill="#ffb74d" text-anchor="middle">FORMIGUEIRO</text>')


        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 218
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#16201a" rx="8" '
                   f'stroke="#4caf50" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#4caf50" font-size="15" font-weight="bold">'
                   f'Mutualistic Defense Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'EFN secretion peaks at leaf flush, not yearly.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'Sugary rewards recruit ants facing herbivory.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'Defense is conditional: EFN output and patrol</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'is whether outbreaks are repelled or escape.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'Plant benefit = tissue saved, not ants alone.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#00e5ff" font-size="15">'
                   f'Cyan halos = defense events at the leaf stage.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 169
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#16201a" rx="8" '
                   f'stroke="#ffb74d" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ffb74d" font-size="15" font-weight="bold">'
                   f'Ecological Outcomes</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#00e5ff" font-size="15">'
                   f'Repelled herbivore attacks: {sim.defenses_won:,.0f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#e53935" font-size="15">'
                   f'Leaf tissue lost when defense fails: {sim.leaves_lost:,.0f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#cddc39" font-size="15">'
                   f'EFN plants: {cfg.num_plants} | Vulnerable leaves: {cfg.num_plants * cfg.leaves_per_plant}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'Ant foragers: {cfg.num_ants} tracking EFN pulses</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 169
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#16201a" rx="8" '
                   f'stroke="#cddc39" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#cddc39" font-size="15" font-weight="bold">'
                   f'Phenological Curves</text>')

        curves = [
            (EFN_PRODUCTION_CURVE, "#ffeb3b", "EFN Production"),
            (HERBIVORE_PRESSURE_CURVE, "#e53935", "Herbivores"),
            (ANT_PATROL_CURVE, "#ff6d00", "Ant Patrols"),
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
    print(f" — Formiga ↔ EFN Clock on {CONFIG.device}...")

    sim = EFNMutualismSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.defenses_won} attack repelled, {sim.leaves_lost} leaves lost to herbivory.")

    print("Generating SVG...")
    renderer = EFNMutualismRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_63')
    return svg_content


if __name__ == "__main__":
    main()
