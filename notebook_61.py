# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 61: Anta ↔ Macaúba/Jatobá — Megaherbivore Endozoochory Clock
# INTERVENTION 7/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_61.py — Lowland Tapir (Anta) ↔ Large Cerrado Fruits:
Notebook Differentiation:
- Differentiation Focus: Anta ↔ Macaúba/Jatobá — Megaherbivore Endozoochory Clock emphasizing cultural keystone sites.
- Indicator species: Traira (Hoplias malabaricus).
- Pollination lens: pollinator dilution across open fields.
- Human impact lens: apiary placement competition.

                 The Gardener of the Forest Phenological Clock

Models the critical ecological role of the Anta (Tapirus terrestris) as a
megaherbivore seed disperser. Tapirs are the only animals in the Neotropics
large enough to swallow and transport massive seeds (like those of Jatobá and 
Macaúba) intact over long distances. 

The notebook therefore makes the same kind of causal ecological argument used
in the project's stronger examples: hydrology constrains tapir movement,
movement determines where large seeds leave the parent population, and those
deposition patterns reshape gallery-forest expansion and connectivity.

The radial phenological clock maps:
  • Tapir commuting: Day resting in Gallery Forests/Karst Rivers vs. 
    Night foraging in Open Cerrado.
  • Seasonal range expansion: Restricted to permanent water in the dry
    season, expanding deep into the Cerrado during the wet season.
  • Fruiting phenology of target large-seeded species (Macaúba, Jatobá).
  • Seed dispersal via latrines often placed near water or forest edges,
    facilitating gallery forest expansion.

Scientific References:
  - Fragoso, J.M.V. (1997). "Tapir-generated seed shadows: scale-dependent
    patchiness in the Amazon rain forest." Journal of Ecology.
  - Talamoni, S.A. et al. (2020). "Diet of the lowland tapir in a Brazilian
    Cerrado." Mammalia.
  - Giombini, M.I. et al. (2009). "Seed dispersal of the palm Syagrus 
    romanzoffiana by tapirs in the Atlantic Forest." Biotropica.

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

# Precipitation / Surface Water Availability
# High surface water in wet season allows tapirs to forage further from rivers
WATER_AVAILABILITY = [
    1.00,   # JAN
    1.00,   # FEV
    0.90,   # MAR
    0.70,   # ABR
    0.40,   # MAI
    0.20,   # JUN
    0.10,   # JUL
    0.05,   # AGO — lowest water, tapirs restricted to permanent rivers
    0.15,   # SET
    0.40,   # OUT
    0.70,   # NOV
    0.90,   # DEZ
]
MACAOUBA_FRUITING = [
    0.20, 0.10, 0.05, 0.05, 0.05, 0.10, 0.30, 0.60, 0.90, 1.00, 0.80, 0.50
    # Peak: Sep-Nov
]

# Fruiting: Jatobá (Hymenaea courbaril) — hard pods broken by tapir teeth
JATOBA_FRUITING = [
    0.10, 0.05, 0.05, 0.10, 0.30, 0.60, 0.90, 1.00, 0.80, 0.40, 0.20, 0.10
    # Peak: Jul-Sep (Dry season dry pods)
]

# Tapir Diet Shift: Graminoids/Leaves vs Fruits
# Eat more fruit when available (dry/early wet), browse leaves in wet season
FRUGIVORY_CURVE = [
    0.40,   # JAN
    0.30,   # FEV
    0.30,   # MAR
    0.40,   # ABR
    0.50,   # MAI
    0.60,   # JUN
    0.80,   # JUL — Jatobá peak
    0.90,   # AGO
    0.90,   # SET — Macaúba peak
    0.85,   # OUT
    0.70,   # NOV
    0.50,   # DEZ
]


@dataclass
class TapirConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_tapirs: int = 5
    tapir_speed: float = 1.8
    
    num_macaubas: int = 15
    num_jatobas: int = 12
    
    gut_passage_frames: int = 25  # long digestion time roughly ~2 days
    drop_probability: float = 0.6


CONFIG = TapirConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class TapirSim:

    def __init__(self, cfg: TapirConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # The Gallery Forest / River runs vertically near the center (x = cx, +-40)
        # Open Cerrado is towards the left and right edges of the clock
        
        # --- Fruit Trees ---
        self.trees: List[Dict] = []
        
        # Macaúbas (Open Cerrado & Edges)
        for _ in range(cfg.num_macaubas):
            r = random.uniform(80, R - 20)
            angle = random.uniform(0, 2 * math.pi)
            if math.cos(angle) > -0.2 and math.cos(angle) < 0.2:
                # push out of pure center
                angle += 0.5
            tx = cx + math.cos(angle) * r
            ty = cy + math.sin(angle) * r
            self.trees.append({
                "pos": (tx, ty),
                "type": "Macauba",
                "fruit_load": 0.0
            })
            
        # Jatobás (Mostly Cerradão / Edges)
        for _ in range(cfg.num_jatobas):
            r = random.uniform(60, R - 50)
            angle = random.uniform(0, 2 * math.pi)
            tx = cx + math.cos(angle) * r
            ty = cy + math.sin(angle) * r
            self.trees.append({
                "pos": (tx, ty),
                "type": "Jatoba",
                "fruit_load": 0.0
            })

        # --- Tapirs ---
        self.tapirs: List[Dict] = []
        for i in range(cfg.num_tapirs):
            # Start in gallery forest near y axis
            ty = cy + random.uniform(-100, 100)
            tx = cx + random.uniform(-20, 20)
            self.tapirs.append({
                "pos": torch.tensor([tx, ty], device=self.dev, dtype=torch.float32),
                "stomach": [],
                "state": "resting" # resting (day), foraging (night)
            })

        self.hist_xy: List[List[Tuple[float, float]]] = [[] for _ in range(cfg.num_tapirs)]
        self.hist_month: List[float] = []
        
        self.seed_drops: List[Dict] = []
        self.total_fruits_eaten = 0
        self.total_seeds_dispersed = 0
        self.seeds_per_month = [0] * 12

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
        m_idx = int(month_frac) % 12
        
        water = self._interp(WATER_AVAILABILITY, month_frac)
        frugivory = self._interp(FRUGIVORY_CURVE, month_frac)
        
        self.hist_month.append(month_frac)

        # Update Tree Fruits
        mac_f = self._interp(MACAOUBA_FRUITING, month_frac)
        jat_f = self._interp(JATOBA_FRUITING, month_frac)
        for t in self.trees:
            if t["type"] == "Macauba": t["fruit_load"] = mac_f
            else: t["fruit_load"] = jat_f

        # Day/Night cycle (fast pulsing, 6 frames = 1 day)
        time_of_day = (frame % 6) / 6.0 
        is_day = 0.2 < time_of_day < 0.8
        
        # Tapir Max Range depends on Water availability (Dry season = restricted near center river)
        # Wet season = can forage far away
        max_forage_dist = 60 + water * 160

        for i, tapir in enumerate(self.tapirs):
            pos = tapir["pos"]
            
            # Digestive tract progression
            new_stomach = []
            for seed in tapir["stomach"]:
                seed["timer"] += 1
                if seed["timer"] >= cfg.gut_passage_frames:
                    # Defecate
                    if random.random() < cfg.drop_probability:
                        at_water = abs(pos[0].item() - cx) < 30 # Defecating near/in water is common for tapirs
                        self.seed_drops.append({
                            "pos": (pos[0].item() + random.uniform(-5, 5), pos[1].item() + random.uniform(-5, 5)),
                            "type": seed["type"],
                            "frame": frame,
                            "at_water": at_water
                        })
                        self.total_seeds_dispersed += 1
                        self.seeds_per_month[m_idx] += 1
                else:
                    new_stomach.append(seed)
            tapir["stomach"] = new_stomach

            # Movement Target
            target = None
            if is_day: # Day time: Rest near the river / gallery forest
                tx = cx + random.uniform(-25, 25)
                # Keep current Y roughly but drift slightly
                ty = pos[1].item() + random.uniform(-10, 10)
                # Clamp Y to clock radius
                ty = min(max(ty, cy - cfg.clock_radius + 30), cy + cfg.clock_radius - 30)
                target = torch.tensor([tx, ty], device=self.dev, dtype=torch.float32)
            else: # Night time: Forage!
                # Find near fruiting tree within max range
                best_tree = None
                best_dist = float('inf')
                for t in self.trees:
                    if t["fruit_load"] > 0.2:
                        dx = t["pos"][0] - pos[0].item()
                        dy = t["pos"][1] - pos[1].item()
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Check distance from river (cx)
                        dist_from_river = abs(t["pos"][0] - cx)
                        
                        if dist_from_river <= max_forage_dist and dist < best_dist and random.random() < frugivory:
                            best_tree = t
                            best_dist = dist
                
                if best_tree and best_dist < 100:
                    target = torch.tensor(best_tree["pos"], device=self.dev, dtype=torch.float32)
                    # Eat fruit if close enough
                    if best_dist < 15 and random.random() < 0.3 * frugivory:
                        tapir["stomach"].append({"type": best_tree["type"], "timer": 0})
                        self.total_fruits_eaten += 1
                else:
                    # Random browse within max_forage_dist
                    sign = 1 if (pos[0].item() > cx) else -1
                    tx = cx + sign * random.uniform(10, max_forage_dist - 10)
                    ty = pos[1].item() + random.uniform(-30, 30)
                    target = torch.tensor([tx, ty], device=self.dev, dtype=torch.float32)

            # Move
            if target is not None:
                to_target = target - tapir["pos"]
                dist = torch.norm(to_target).item()
                if dist > 2.0:
                    dir_vec = to_target / dist
                    # Move slower during the day, faster at night
                    speed = cfg.tapir_speed * (0.3 if is_day else 1.0)
                    tapir["pos"] += dir_vec * speed
            
            # Boundary Clamp
            tapir["pos"][0] = torch.clamp(tapir["pos"][0], cx - cfg.clock_radius + 10, cx + cfg.clock_radius - 10)
            tapir["pos"][1] = torch.clamp(tapir["pos"][1], cy - cfg.clock_radius + 10, cy + cfg.clock_radius - 10)

            self.hist_xy[i].append((tapir["pos"][0].item(), tapir["pos"][1].item()))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class TapirRenderer:

    def __init__(self, cfg: TapirConfig, sim: TapirSim):
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
            f'style="background-color:#18201a; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        # River Gradient
        svg.append(
            '<linearGradient id="riverGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
            '<stop offset="0%" stop-color="#18201a" stop-opacity="0.0"/>'
            '<stop offset="40%" stop-color="#00695c" stop-opacity="0.6"/>'
            '<stop offset="50%" stop-color="#00838f" stop-opacity="0.8"/>'
            '<stop offset="60%" stop-color="#00695c" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#18201a" stop-opacity="0.0"/>'
            '</linearGradient>'
        )
        svg.append('</defs>')

        # Background Landscape
        svg.append(f'<rect width="{w}" height="{h}" fill="#18201a"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="#1c251d" stroke="#2e3b32" stroke-width="1"/>')
        
        # Central Gallery Forest / River
        svg.append(f'<rect x="{cx-80}" y="{cy-R-30}" width="160" height="{2*R+60}" fill="url(#riverGrad)"/>')

        # Limit lines for Wet Season Max Forage Range
        svg.append(f'<line x1="{cx-220}" y1="{cy-R}" x2="{cx-220}" y2="{cy+R}" stroke="#33691e" stroke-dasharray="4,4" opacity="0.4"/>')
        svg.append(f'<line x1="{cx+220}" y1="{cy-R}" x2="{cx+220}" y2="{cy+R}" stroke="#33691e" stroke-dasharray="4,4" opacity="0.4"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#aed581" font-weight="bold">'
            f'ECO-SIM: Tapir × Cerrado Fruits    - Megaherbivore Endozoochory</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#78909c">'
            f'Tapirus terrestris moving between Gallery Forests and Open Cerrado</text>'
        )

        # Clock Face
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#90a4ae" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
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

        draw_arc(4.5, 9.5, R + 10, "#fb8c00", "Jatobá Peak", 0.4)
        draw_arc(7.5, 11.5, R + 22, "#ffb300", "Macaúba Peak", 0.4)
        draw_arc(0, 4, R + 15, "#00bcd4", "Wet Season / Expanded Range", 0.3)

        # Trees
        for t in sim.trees:
            tx, ty = t["pos"]
            tt = t["type"]
            base_col = "#afb42b" if tt == "Macauba" else "#8d6e63"
            fruit_col = "#fbc02d" if tt == "Macauba" else "#795548"
            curve = MACAOUBA_FRUITING if tt == "Macauba" else JATOBA_FRUITING
            
            # Static tree canopy
            svg.append(f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="5" fill="none" stroke="{base_col}" stroke-width="1.5" opacity="0.6"/>')
            
            # Animated Fruiting
            f_vals = ";".join(f"{sim._interp(curve, (fi/F)*12)*4:.1f}" for fi in range(F))
            svg.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" fill="{fruit_col}">'
                f'<animate attributeName="r" values="{f_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Seed drops (Defecation at latrines/water)
        for d in sim.seed_drops:
            dx, dy = d["pos"]
            df = d["frame"]
            dt = d["type"]
            col = "#fbc02d" if dt == "Macauba" else "#795548"
            sz = 2.5
            
            ops = ";".join("0.0" if fi < df else f"{max(0.1, 0.9 - (fi-df)*0.01):.2f}" for fi in range(F))
            
            svg.append(
                f'<rect x="{dx-sz:.1f}" y="{dy-sz:.1f}" width="{sz*2}" height="{sz*2}" fill="{col}" stroke="#3e2723" stroke-width="0.5" rx="1" opacity="0.0">'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</rect>'
            )

        # Tapirs (Brown circles dragging a faint tail)
        for i in range(cfg.num_tapirs):
            hist = sim.hist_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            
            # Trail path (faint)
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::3]]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#5d4037" stroke-width="1.5" opacity="0.25"/>'
                )
            
            # Animated Tapir
            svg.append(
                f'<circle r="5" fill="#5d4037" stroke="#3e2723" stroke-width="2">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Snout / Head indicator
            offs = ";".join(str(round(h[0]+2, 1)) for h in hist)
            svg.append(
                f'<circle r="2" fill="#3e2723">'
                f'<animate attributeName="cx" values="{offs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Clock hand
        hand_x = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#aed581" stroke-width="2" stroke-linecap="round" opacity="0.8">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#18201a" stroke="#aed581" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="#aed581"/>')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 210
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1b211d" rx="8" '
                   f'stroke="#aed581" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#aed581" font-size="15" font-weight="bold">'
                   f'Anta Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Tapirs link refuges to Cerrado by commuting.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'Dry-season scarcity narrows routes to rivers.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'Large fruits move intact across habitat edges.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'Latrines focus seed rain near moist corridors.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'Tapirs are key long-distance plant engineers.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#fbc02d" font-size="15">'
                   f'Yellow drops = seeds via long tapir transport.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 182
        
        water_drops = sum(1 for d in sim.seed_drops if d["at_water"])
        water_pct = (water_drops / max(1, len(sim.seed_drops))) * 100
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1b211d" rx="8" '
                   f'stroke="#ffe082" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ffe082" font-size="15" font-weight="bold">'
                   f'Yearly Dispersal Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#e0e0e0" font-size="15">'
                   f'Large Fruits Swallowed: {sim.total_fruits_eaten:,.0f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#aed581" font-size="15">'
                   f'Viable Seeds Dispersed: {sim.total_seeds_dispersed:,.0f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#4fc3f7" font-size="15">'
                   f'Seeds deposited near water/forest: {water_pct:.0f}%</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'Population: {cfg.num_tapirs} | Foraging Range limits</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 160
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1b211d" rx="8" '
                   f'stroke="#80cbc4" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#80cbc4" font-size="15" font-weight="bold">'
                   f'Phenological Curves</text>')

        curves = [
            (WATER_AVAILABILITY, "#0288d1", "Surface Water"),
            (MACAOUBA_FRUITING, "#ffb300", "Macaúba"),
            (JATOBA_FRUITING, "#fb8c00", "Jatobá"),
            (FRUGIVORY_CURVE, "#8d6e63", "Tapir Frugivory"),
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
    print(f" — Anta ↔ Cerrado Fruits Clock on {CONFIG.device}...")

    sim = TapirSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    water_drops = sum(1 for d in sim.seed_drops if d["at_water"])
    water_pct = (water_drops / max(1, len(sim.seed_drops))) * 100
    print(f"Done: {sim.total_fruits_eaten} fruits, {sim.total_seeds_dispersed} seeds dispersed "
          f"({water_pct:.1f}% near water/gallery forest).")

    print("Generating SVG...")
    renderer = TapirRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_61')
    return svg_content


if __name__ == "__main__":
    main()
