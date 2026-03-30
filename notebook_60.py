# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 60: Arara-canindé ↔ Buriti Palm — Vereda Phenological Clock
# INTERVENTION 6/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_60.py — Blue-and-yellow Macaw ↔ Buriti Palm:
Notebook Differentiation:
- Differentiation Focus: Arara-canindé ↔ Buriti Palm — Vereda Phenological Clock emphasizing traditional knowledge patches.
- Indicator species: Dourado-do-cerrado (Salminus brasiliensis).
- Pollination lens: ground-nesting bee vulnerability.
- Human impact lens: traditional harvesting pressure.

                 Nesting & Foraging Vereda Phenological Clock

Models the profound ecological connection between the Arara-canindé
(Ara ararauna) and the Buriti palm (Mauritia flexuosa) within the
Cerrado's characteristic 'Veredas' (palm swamps). In the RESEX
Recanto das Araras (named for these macaws), this relationship is
emblematic of conservation.

The radial phenological clock maps:
  • Buriti fruiting phenology (highly dependent on Vereda water levels)
  • Macaw nesting season (Aug-Dec), utilizing dead Buriti snags
  • Daily commute/foraging flights between roosting sites and feeding grounds
  • Seed predation vs. incidental dispersal by macaws
  • Vereda hydrological cycle (wet season flood, dry season retreat)

Scientific References:
  - Antas, P.T.Z. et al. (2010). "Nesting of the Blue-and-yellow Macaw
    in the Cerrado." Ornitologia Neotropical.
  - Isasi-Catalá, E. (2011). "The role of Mauritia flexuosa palm swamps
    in neotropical landscapes."
  - Tubelis, D.P. (2009). "Buriti palm stands as key habitats for birds
    in the Cerrado."

Scientific Relevance (PIGT RESEX Recanto das Araras — 2024):
  - Integrates the socio-environmental complexity of 
    de Cima, Goiás, Brazil (The very namesake of the reserve).
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

# Vereda Water Level (Hydrological Cycle)
# High during late wet season, retreats during dry season
VEREDA_WATER_LEVEL = [
    0.80,   # JAN
    0.90,   # FEV
    1.00,   # MAR — Peak flood
    0.95,   # ABR
    0.80,   # MAI
    0.60,   # JUN
    0.40,   # JUL
    0.20,   # AGO — lowest water, isolated pools
    0.15,   # SET — transition
    0.30,   # OUT — first rains
    0.50,   # NOV
    0.70,   # DEZ
]
# Peak fruit maturation occurs December to May (wet season)
BURITI_FRUITING_CURVE = [
    0.70,   # JAN
    0.90,   # FEV — Peak fruit
    1.00,   # MAR
    0.85,   # ABR
    0.60,   # MAI
    0.30,   # JUN
    0.10,   # JUL
    0.05,   # AGO
    0.10,   # SET
    0.20,   # OUT
    0.40,   # NOV
    0.60,   # DEZ
]

# Macaw Nesting Activity (Ara ararauna)
# Courtship starts Jul/Aug, eggs in Sep/Oct, fledging Nov/Dec
MACAW_NESTING_CURVE = [
    0.20,   # JAN — some late fledglings
    0.10,   # FEV
    0.05,   # MAR
    0.05,   # ABR
    0.05,   # MAI
    0.15,   # JUN — exploring nest sites
    0.40,   # JUL
    0.70,   # AGO — courtship, nest preparation in dead Buriti
    0.90,   # SET — eggs and incubation
    1.00,   # OUT — chicks hatching
    0.85,   # NOV — rearing chicks
    0.50,   # DEZ — fledging
]


@dataclass
class VeredaConfig:
    """Configuration for Arara-canindé ↔ Buriti clock."""
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_macaws: int = 16  # Represents pairs/flock
    macaw_speed: float = 4.5
    
    num_buritis: int = 25
    num_snags: int = 6    # Dead buritis for nesting
    
    forage_radius_min: float = 120.0
    forage_radius_max: float = 230.0


CONFIG = VeredaConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class VeredaSim:
    """Phenological clock model for Macaw ↔ Buriti Vereda system."""

    def __init__(self, cfg: VeredaConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Buriti Palms (Live) ---
        self.buritis: List[Dict] = []
        for _ in range(cfg.num_buritis):
            # Buritis cluster near the center (Vereda core) to Mid radius
            r = random.uniform(40, R - 80)
            angle = random.uniform(0, 2 * math.pi)
            bx = cx + math.cos(angle) * r
            by = cy + math.sin(angle) * r
            self.buritis.append({
                "pos": (bx, by),
                "angle": angle,
                "fruit_load": 0.0
            })

        # --- Dead Buriti Snags (for Nesting) ---
        self.snags: List[Dict] = []
        for _ in range(cfg.num_snags):
            r = random.uniform(30, 100)
            angle = random.uniform(0, 2 * math.pi)
            sx = cx + math.cos(angle) * r
            sy = cy + math.sin(angle) * r
            self.snags.append({
                "pos": (sx, sy),
                "occupied": False,
                "occupants": []
            })

        # --- Arara-canindé (Macaws, generally paired) ---
        self.macaws: List[Dict] = []
        for i in range(cfg.num_macaws):
            # Macaws start at roosts on the periphery
            r = R - 20
            angle = (i / cfg.num_macaws) * 2 * math.pi
            mx = cx + math.cos(angle) * r
            my = cy + math.sin(angle) * r
            
            # Pair logic (every two macaws share a nest)
            pair_id = i // 2
            nest_id = pair_id % cfg.num_snags
            
            self.macaws.append({
                "pos": torch.tensor([mx, my], device=self.dev, dtype=torch.float32),
                "angle": angle,
                "pair_id": pair_id,
                "nest_id": nest_id,
                "state": "roosting",  # roosting, foraging, nesting
                "target": None
            })

        self.hist_xy: List[List[Tuple[float, float]]] = [[] for _ in range(cfg.num_macaws)]
        self.hist_month: List[float] = []
        
        # Tracking curves
        self.hist_water: List[float] = []
        self.hist_fruit: List[float] = []
        self.hist_nesting: List[float] = []
        
        self.fruit_consumed: int = 0
        self.chicks_fledged: int = 0
        
        # We record foraging events to show "sparks"
        self.forage_events: List[Dict] = []

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
        
        water_lvl = self._interp(VEREDA_WATER_LEVEL, month_frac)
        fruiting = self._interp(BURITI_FRUITING_CURVE, month_frac)
        nesting = self._interp(MACAW_NESTING_CURVE, month_frac)
        
        self.hist_month.append(month_frac)
        self.hist_water.append(water_lvl)
        self.hist_fruit.append(fruiting)
        self.hist_nesting.append(nesting)

        # Update Buriti fruit loads
        for b in self.buritis:
            b["fruit_load"] = fruiting * (0.8 + 0.2 * random.random())

        # Sub-daily cycle approximation (fast pulsing)
        # Using a sine wave based on frame to simulate daily commute
        time_of_day = (frame % 8) / 8.0  # 8 frames per "day"
        
        for i, macaw in enumerate(self.macaws):
            # State transitions based on season and time of day
            is_nesting_season = random.random() < nesting
            
            target_pos = None
            
            if is_nesting_season:
                # Nesting behavior: tied to snag
                snag = self.snags[macaw["nest_id"]]
                if time_of_day < 0.3 or time_of_day > 0.7:
                    # At nest
                    target_pos = torch.tensor(snag["pos"], device=self.dev, dtype=torch.float32)
                    macaw["state"] = "nesting"
                else:
                    # Foraging for chicks
                    macaw["state"] = "foraging"
            else:
                # Roosting vs Foraging
                if time_of_day < 0.2 or time_of_day > 0.8:
                    macaw["state"] = "roosting"
                    # Roost on periphery
                    periph_a = macaw["pair_id"] * 0.5
                    target_pos = torch.tensor([cx + math.cos(periph_a)*220, cy + math.sin(periph_a)*220], device=self.dev, dtype=torch.float32)
                else:
                    macaw["state"] = "foraging"

            # Determine foraging target
            if macaw["state"] == "foraging":
                if macaw["target"] is None or random.random() < 0.1:
                    # Pick a fruiting buriti
                    if fruiting > 0.1:
                        best_b = random.choice(self.buritis)
                        macaw["target"] = torch.tensor(best_b["pos"], device=self.dev, dtype=torch.float32)
                    else:
                        # Forage farther out in Cerrado
                        fa = random.uniform(0, 2*math.pi)
                        fr = random.uniform(150, 240)
                        macaw["target"] = torch.tensor([cx + math.cos(fa)*fr, cy + math.sin(fa)*fr], device=self.dev, dtype=torch.float32)
                target_pos = macaw["target"]

            # Move towards target
            if target_pos is not None:
                to_target = target_pos - macaw["pos"]
                dist = torch.norm(to_target).item()
                if dist > 5.0:
                    dir_vec = to_target / dist
                    macaw["pos"] += dir_vec * cfg.macaw_speed * random.uniform(0.8, 1.2)
                else:
                    if macaw["state"] == "foraging" and fruiting > 0.1 and dist < 10.0:
                        # Eat fruit
                        if random.random() < 0.2:
                            self.fruit_consumed += 1
                            if len(self.forage_events) < 150:
                                self.forage_events.append({
                                    "pos": (macaw["pos"][0].item(), macaw["pos"][1].item()),
                                    "frame": frame
                                })
            
            # Fledgling simulation: If late nesting season, add to count
            if month_frac > 10.5 and nesting > 0.5 and i % 2 == 0 and random.random() < 0.005:
                self.chicks_fledged += random.randint(1, 2)

            self.hist_xy[i].append((macaw["pos"][0].item(), macaw["pos"][1].item()))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class VeredaRenderer:
    """Renders the Vereda phenological clock as animated SVG."""

    def __init__(self, cfg: VeredaConfig, sim: VeredaSim):
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
            f'style="background-color:#0f1a15; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # --- Defs ---
        svg.append('<defs>')
        
        # Vereda Water gradient (changes size via animation later)
        svg.append(
            '<radialGradient id="waterBg">'
            '<stop offset="0%" stop-color="#0277bd" stop-opacity="0.8"/>'
            '<stop offset="70%" stop-color="#01579b" stop-opacity="0.4"/>'
            '<stop offset="100%" stop-color="#0f1a15" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        
        # Macaw colors
        svg.append(
            '<linearGradient id="macawGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
            '<stop offset="0%" stop-color="#ffd600"/>'  # Yellow breast
            '<stop offset="100%" stop-color="#0288d1"/>' # Blue back
            '</linearGradient>'
        )
        svg.append('</defs>')

        # --- Base Clock Background ---
        svg.append(f'<rect width="{w}" height="{h}" fill="#0f1a15"/>')
        
        # --- Animated Vereda Water Core ---
        # The central water pool expands and contracts with the seasons
        water_r_vals = ";".join(str(int(30 + sim.hist_water[fi] * 90)) for fi in range(F))
        water_op_vals = ";".join(f"{0.3 + sim.hist_water[fi]*0.5:.2f}" for fi in range(F))
        
        svg.append(f'<circle cx="{cx}" cy="{cy}" fill="url(#waterBg)">'
                   f'<animate attributeName="r" values="{water_r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="{water_op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'</circle>')

        # --- Clock face & Labels ---
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 32)
            ty = cy + math.sin(angle) * (R + 32)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#78909c" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 5)
            ly2 = cy + math.sin(angle) * (R - 5)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#263238" stroke-width="2"/>'
            )

        # --- Season Arcs ---
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
            svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="6" '
                       f'stroke-linecap="round" opacity="{opacity}"/>')
            mid = ((start_m + end_m) / 2 / 12) * 2 * math.pi - math.pi / 2
            lx = cx + math.cos(mid) * (radius + 12)
            ly = cy + math.sin(mid) * (radius + 12)
            svg.append(f'<text font-weight="bold" x="{lx:.0f}" y="{ly:.0f}" font-size="15" fill="{color}" '
                       f'text-anchor="middle" opacity="0.9">{label}</text>')

        draw_arc(11, 4.5, R + 10, "#fb8c00", "Buriti Fruiting", 0.4)
        draw_arc(7, 11.5, R + 22, "#0288d1", "Macaw Nesting", 0.4)

        # --- Dead Snags (Nests) ---
        for snag in sim.snags:
            sx, sy = snag["pos"]
            svg.append(
                f'<rect x="{sx-3:.0f}" y="{sy-8:.0f}" width="6" height="16" fill="#5d4037" '
                f'rx="2" stroke="#3e2723" stroke-width="1"/>'
            )
            # Nest cavity
            svg.append(f'<circle cx="{sx:.0f}" cy="{sy-4:.0f}" r="2" fill="#1a100c"/>')

        # --- Live Buriti Palms ---
        for b in sim.buritis:
            bx, by = b["pos"]
            
            # Palm leaves (static)
            for _ in range(5):
                la = random.uniform(0, 2*math.pi)
                lr = random.uniform(8, 14)
                lx = bx + math.cos(la)*lr
                ly = by + math.sin(la)*lr
                svg.append(
                    f'<line x1="{bx:.0f}" y1="{by:.0f}" x2="{lx:.0f}" y2="{ly:.0f}" '
                    f'stroke="#388e3c" stroke-width="1.5" opacity="0.8"/>'
                )
            # Trunk top
            svg.append(f'<circle cx="{bx:.0f}" cy="{by:.0f}" r="3" fill="#4e342e"/>')
            
            # Animated Fruiting clusters (orange-red)
            # Scales with BURITI_FRUITING_CURVE
            fruit_sizes = ";".join(
                f"{sim._interp(BURITI_FRUITING_CURVE, (fi/F)*12) * 5.0:.1f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{bx+2:.0f}" cy="{by+2:.0f}" fill="#bf360c">'
                f'<animate attributeName="r" values="{fruit_sizes}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Macaw Foraging Events ("Sparkles") ---
        for fe in sim.forage_events:
            fx, fy = fe["pos"]
            ff = fe["frame"]
            
            ops = ";".join("0.0" if fi < ff else f"{max(0, 0.8 - (fi-ff)*0.05):.2f}" for fi in range(F))
            szs = ";".join("0.0" if fi < ff else f"{min(6, (fi-ff)*0.5):.1f}" for fi in range(F))
            
            svg.append(
                f'<circle cx="{fx:.0f}" cy="{fy:.0f}" fill="none" stroke="#ffd600" stroke-width="1" opacity="0.0">'
                f'<animate attributeName="r" values="{szs}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # --- Macaw Particles (Pairs flying) ---
        for i in range(cfg.num_macaws):
            hist = sim.hist_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            
            svg.append(
                f'<circle r="3.5" fill="url(#macawGrad)">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Clock hand ---
        hand_x_vals = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-15), 1)) for m in sim.hist_month)
        hand_y_vals = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-15), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 15}" '
            f'stroke="#0288d1" stroke-width="2.5" stroke-linecap="round" opacity="0.8">'
            f'<animate attributeName="x2" values="{hand_x_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        # --- Centre hub ---
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="16" fill="#01579b" stroke="#ffd600" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#ffd600"/>')

        # --- Title Info ---
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ffd600" font-weight="bold">'
            f'ECO-SIM: Blue-and-yellow Macaw × Buriti Palm Palm Palm    - Vereda Phenology Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#81d4fa">'
            f'Nesting dynamics & foraging within the Mauritia flexuosa palm swamps</text>'
        )

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 218
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#101a24" rx="8" '
                   f'stroke="#0288d1" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ffd600" font-size="15" font-weight="bold">'
                   f'Vereda Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Cerrado Veredas (palm swamps): dynamic water</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'levels. Buriti palms fruit in the wet season.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'season. Macaws commute from roosts to forage.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'During breeding (Aug-Dec), pairs nest in</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'dead Buriti snags, altering flight paths.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#ffd600" font-size="15">'
                   f'Yellow ripples = Foraging on fruit clusters</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 169
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#101a24" rx="8" '
                   f'stroke="#fb8c00" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#fb8c00" font-size="15" font-weight="bold">'
                   f'Yearly Simulation Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#e0e0e0" font-size="15">'
                   f'Buriti Fruits Consumed: {sim.fruit_consumed:,.0f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#0288d1" font-size="15">'
                   f'Chicks Fledged successfully: {sim.chicks_fledged}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#ffd600" font-size="15">'
                   f'Macaw Population: {cfg.num_macaws} ( {cfg.num_macaws//2} breeding pairs )</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#81d4fa" font-size="15">'
                   f'Live Buritis: {cfg.num_buritis}  |  Dead Snags (Nests): {cfg.num_snags}</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 169
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#101a24" rx="8" '
                   f'stroke="#26a69a" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#26a69a" font-size="15" font-weight="bold">'
                   f'Phenological Curves</text>')

        curves = [
            (VEREDA_WATER_LEVEL, "#0288d1", "Vereda Water Lvl"),
            (BURITI_FRUITING_CURVE, "#fb8c00", "Buriti Fruit"),
            (MACAW_NESTING_CURVE, "#ffd600", "Macaw Nesting"),
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
            svg.append(f'<text font-weight="bold" x="{lx + 7}" y="{ly + 4}" fill="{color}" font-size="15">'
                       f'{label}</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    """Run Macaw ↔ Buriti clock simulation."""
    print(f" — Arara-canindé ↔ Buriti Clock on {CONFIG.device}...")

    sim = VeredaSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.fruit_consumed} fruits consumed, {sim.chicks_fledged} chicks fledged.")

    print("Generating SVG...")
    renderer = VeredaRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_60')
    return svg_content


if __name__ == "__main__":
    main()
