# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 78: Cupim (Nasutitermes sp.) ↔ Fungo (Termitomyces sp.)
#            Fungiculture · Seasonal Swarming · Rainfall-Triggered Fruiting
# INTERVENTION 4/4: Cerrado Trophic Cascade & Seasonal Connectivity Series
# ===================================================================================================
"""
notebook_78.py - Termite × Ectosymbiotic Fungus (Animal × Fungus):
Notebook Differentiation:
- Differentiation Focus: Cupim (Nasutitermes sp.) ↔ Fungo (Termitomyces sp.) emphasizing rocky outcrop shelters.
- Indicator species: Raposa-do-campo (Lycalopex vetulus).
- Pollination lens: temporal mismatch with migratory pollinators.
- Human impact lens: extreme drought thresholds.

              Nutrient Cycling Backbone, Synchronised Swarming & Fungal Bloom

The mound-building termites of the Cerrado (e.g., Nasutitermes and Cornitermes spp.)
are the primary decomposers of the biome, responsible for cycling massive amounts of
carbon and nitrogen. Many advanced termite species maintain an obligate mutualism
with Termitomyces fungi.

SEASONAL DYNAMICS at RESEX Recanto das Araras (Goiás):

  FUNGICULTURE (Year-round, peaking in dry season):
    Termites forage at night or in covered galleries, bringing dead plant material
    (grasses, wood) into the mound. They deposit this as "fungus comb." The fungus
    breaks down complex lignin and cellulose, producing protein-rich nodules that
    the termites consume.

  THE RAINFALL TRIGGER (Oct–Nov):
    The arrival of the first heavy rains of the wet season creates a dramatic 
    ecological pulse:
    1. Alate Swarming (Revoada): Thousands of winged reproductive termites (alates)
       emerge synchronously to mate and establish new colonies. This is a massive
       food pulse for predators across the ecosystem.
    2. Fungal Fruiting: The Termitomyces fungus pushes massive structural mushrooms
       out through the walls of the termite mound to release its own spores into 
       the humid air.

  ECOSYSTEM ENGINEERING (Connecting to nb75-77):
    • Abandoned or hollowed termite mounds become denning sites for the Lobo-guará
      (nb75) to raise its pups.
    • The synchronized swarming provides an emergency calorie pulse for migrating
      birds and bats (nb77) arriving at the start of the wet season.
    • Capuchin monkeys (nb76) actively break into mounds to eat the protein-rich
      fungus combs and termite larvae.

Scientific references:
  • Aanen et al. (2002): The evolution of fungus-growing termites.
  • Constantino (2005): Termite diversity in the Cerrado.
  • PIGT  field observations, Goiás (2022–2024).
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # pyre-ignore[21]
import numpy as np  # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from eco_base import save_svg, sanitize_svg_text, draw_phenology_chart, draw_migration_map  , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML  # pyre-ignore[21]
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES  (month 0 = January)
# ===================================================================================================

# Shared rainfall backbone (links to nb75/76/77 - Oct/Nov onset)
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]

# Termite foraging intensity (higher in early dry season, drops before rains)
P_FORAGING_CURVE = [
    0.50, 0.60, 0.70, 0.85, 0.95, 0.90, 0.70, 0.50, 0.30, 0.40, 0.45, 0.50
]

# Fungal comb internal metabolism (peaks in dry season when mound is sealed)
FUNGAL_METABOLISM_CURVE = [
    0.60, 0.50, 0.55, 0.70, 0.80, 0.95, 1.00, 0.90, 0.75, 0.60, 0.55, 0.65
]

# Alate Swarming & Fungal Fruiting pulse (strictly tied to first rains: Oct-Nov)
SWARM_FRUIT_PULSE = [0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.20, 0.60, 0.90, 0.50]

@dataclass
class MoundCfg:
    width:  int   = 1280
    height: int = CANVAS_HEIGHT
    frames: int   = 360
    fps:    int   = 10
    device: str   = 'cuda' if torch.cuda.is_available() else 'cpu'

    clock_cx:     float = 420.0
    clock_cy:     float = 310.0
    clock_radius: float = 240.0

    num_mounds:   int   = 6     # Central macro-mounds
    num_foragers: int   = 60    # Abstracted foraging parties
    mound_radius: float = 18.0
    swarm_count:  int   = 150   # Max flying alates/spores active during pulse


CONFIG = MoundCfg()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class TermitomycesSim:
    def __init__(self, cfg: MoundCfg):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy, R = cfg.clock_cx, cfg.clock_cy, cfg.clock_radius

        # Mounds
        self.mounds: List[Dict] = []
        for k in range(cfg.num_mounds):
            angle = (k / cfg.num_mounds) * 2 * math.pi + random.uniform(-0.3, 0.3)
            r = random.uniform(R * 0.2, R * 0.7)
            self.mounds.append({
                "pos": [cx + math.cos(angle) * r, cy + math.sin(angle) * r],
                "biomass": 10.0,
                "mushroom_level": 0.0,
                "swarming": False,
            })

        # Foragers (termites moving between mounds and the surrounding area)
        self.foragers: List[Dict] = []
        for _ in range(cfg.num_foragers):
            mound_idx = random.randint(0, cfg.num_mounds - 1)
            mp = self.mounds[mound_idx]["pos"]
            self.foragers.append({
                "pos": torch.tensor([mp[0], mp[1]], device=self.dev, dtype=torch.float32),
                "home_mound": mound_idx,
                "target": None,
                "carrying": False,
                "angle": random.uniform(0, 2*math.pi)
            })

        # Swarmers (Alates and Spores)
        self.swarmers: List[Dict] = []
        
        # Histories
        self.hist_month: List[float] = []
        self.hist_forager_xy: List[List[Tuple[float,float,float]]] = [[] for _ in range(cfg.num_foragers)]
        self.hist_mound_state: List[List[Tuple[float,float]]] = [[] for _ in range(cfg.num_mounds)] # (biomass, mushroom_level)
        self.hist_swarmer_xy: List[List[Tuple[float,float,float,float]]] = []  # per-frame lists of (x, y, size, opacity)

        self.total_forage_trips = 0
        self.total_swarm_events = 0

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

        rain       = self._interp(RAINFALL_CURVE, month_frac)
        forage_idx = self._interp(P_FORAGING_CURVE, month_frac)
        metab      = self._interp(FUNGAL_METABOLISM_CURVE, month_frac)
        pulse      = self._interp(SWARM_FRUIT_PULSE, month_frac)

        # 1. Update Mounds
        for mi, m in enumerate(self.mounds):
            # Fungus consumes biomass, converts to energy inside mound
            consume_rate = metab * 0.05
            if m["biomass"] > consume_rate:
                m["biomass"] -= consume_rate
            
            # Mushroom fruiting triggered strongly by pulse
            target_mushroom = pulse * 15.0
            m["mushroom_level"] += (target_mushroom - m["mushroom_level"]) * 0.1
            
            # Swarm state
            m["swarming"] = pulse > 0.4
            
            self.hist_mound_state[mi].append((m["biomass"], m["mushroom_level"]))

        # 2. Update Foragers (bringing biomass)
        for fi, f in enumerate(self.foragers):
            pos = f["pos"]
            home_pos = self.mounds[f["home_mound"]]["pos"]
            speed = 2.5 * forage_idx
            
            opacity = 1.0 if forage_idx > 0.2 else 0.0
            
            if not f["carrying"]:
                # Go out to forage
                if f["target"] is None:
                    dist = random.uniform(30, 90)
                    a = f["angle"] + random.uniform(-0.5, 0.5)
                    f["angle"] = a
                    f["target"] = torch.tensor([home_pos[0] + math.cos(a)*dist, home_pos[1] + math.sin(a)*dist], device=self.dev, dtype=torch.float32)
                
                vec = f["target"] - pos
                d = torch.norm(vec).item()
                if d > 2.0:
                    pos += (vec / max(d, 1e-5)) * speed + torch.randn(2, device=self.dev)*0.5
                else:
                    # Pick up biomass
                    f["carrying"] = True
                    f["target"] = torch.tensor(home_pos, device=self.dev, dtype=torch.float32)
            else:
                # Return home
                vec = f["target"] - pos
                d = torch.norm(vec).item()
                if d > 3.0:
                    pos += (vec / max(d, 1e-5)) * speed + torch.randn(2, device=self.dev)*0.5
                else:
                    # Deposit biomass
                    self.mounds[f["home_mound"]]["biomass"] += 0.5
                    self.total_forage_trips += 1
                    f["carrying"] = False
                    f["target"] = None
                    
            # Clamp inside clock
            dx_ = pos[0].item() - cx
            dy_ = pos[1].item() - cy
            dr_ = math.sqrt(dx_*dx_ + dy_*dy_)
            if dr_ > R - 5:
                pos[0] = cx + (dx_/dr_) * (R - 5)
                pos[1] = cy + (dy_/dr_) * (R - 5)
            
            self.hist_forager_xy[fi].append((pos[0].item(), pos[1].item(), opacity))

        # 3. Dynamic Swarmers (Alates / Spores)
        active_swarmers = []
        target_swarm_count = int(pulse * cfg.swarm_count)
        
        # Manage swarmer pool
        while len(self.swarmers) < target_swarm_count:
            m_idx = random.randint(0, len(self.mounds)-1)
            mp = self.mounds[m_idx]["pos"]
            self.swarmers.append({
                "pos": [mp[0], mp[1]],
                "vx": random.uniform(-4, 4),
                "vy": random.uniform(-4, 4),
                "type": "alate" if random.random() > 0.3 else "spore",
                "life": 1.0
            })
            self.total_swarm_events += 1
            
        for s in list(self.swarmers):
            s["pos"][0] += s["vx"]
            s["pos"][1] += s["vy"]
            # Wind drift
            s["vx"] += 0.1
            s["life"] -= 0.05
            
            if s["life"] <= 0 or pulse < 0.1:
                self.swarmers.remove(s)
            else:
                sz = 3.0 if s["type"] == "alate" else 1.5
                active_swarmers.append((s["pos"][0], s["pos"][1], sz, s["life"]))
                
        self.hist_swarmer_xy.append(active_swarmers)


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class MoundRenderer:
    def __init__(self, cfg: MoundCfg, sim: TermitomycesSim):
        self.cfg = cfg
        self.sim = sim

    def _arc_path(self, cx, cy, radius, start_m, end_m):
        a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
        a2 = (end_m   / 12) * 2 * math.pi - math.pi / 2
        span  = (end_m - start_m) % 12
        large = 1 if span > 6 else 0
        return f"M {cx+math.cos(a1)*radius:.0f} {cy+math.sin(a1)*radius:.0f} A {radius} {radius} 0 {large} 1 {cx+math.cos(a2)*radius:.0f} {cy+math.sin(a2)*radius:.0f}"

    def generate_svg(self) -> str:
        cfg = self.cfg
        sim = self.sim
        w, h = cfg.width, cfg.height
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius
        F = cfg.frames
        dur = F / cfg.fps

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
               f'style="background-color:#100d08; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append('<defs>')
        svg.append('<radialGradient id="soilBg">'
                   '<stop offset="0%" stop-color="#2a2015" stop-opacity="0.95"/>'
                   '<stop offset="65%" stop-color="#1c140d" stop-opacity="0.85"/>'
                   '<stop offset="100%" stop-color="#100d08" stop-opacity="0.0"/>'
                   '</radialGradient>')
        svg.append('</defs>')

        # Background
        svg.append(f'<rect width="{w}" height="{h}" fill="#100d08"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R+55}" fill="url(#soilBg)"/>')

        # Ambient Swarm/Wet Wash
        # A faint yellow/white wash during the pulse
        pulse_fills = ";".join(
            f"rgba(255,230,150,{sim._interp(SWARM_FRUIT_PULSE, (f/F)*12)*0.15:.2f})"
            for f in range(F)
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R-8}" fill="transparent">'
                   f'<animate attributeName="fill" values="{pulse_fills}" '
                   f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Title
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffb300" font-weight="bold">'
                   f'ECO-SIM: Cupim × Fungo-termitomyces - Nutrient and Swarming Clock</text>')
        svg.append(f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#dce775">'
                   f'Fungiculture · Seasonal Swarming · Rainfall-Triggered Fungal Bloom </text>')

        # Clock Face
        months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        month_cols = {0:"#7cb342", 1:"#7cb342", 2:"#7cb342", 3:"#ffa000", 4:"#ffa000", 5:"#ff8f00",
                      6:"#ff8f00", 7:"#ff6f00", 8:"#e65100", 9:"#ffee58", 10:"#fff59d", 11:"#8bc34a"}
        for i, m in enumerate(months):
            angle = (i/12)*2*math.pi - math.pi/2
            tx = cx + math.cos(angle)*(R+35); ty = cy + math.sin(angle)*(R+35)
            svg.append(f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="{month_cols[i]}" '
                       f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>')
            svg.append(f'<line x1="{cx+math.cos(angle)*R:.0f}" y1="{cy+math.sin(angle)*R:.0f}" '
                       f'x2="{cx+math.cos(angle)*(R-6):.0f}" y2="{cy+math.sin(angle)*(R-6):.0f}" '
                       f'stroke="#2a2015" stroke-width="2"/>')

        # Arcs
        # Dry season foraging
        d1 = self._arc_path(cx, cy, R+11, 4, 9)
        svg.append(f'<path d="{d1}" fill="none" stroke="#ffa000" stroke-width="8" opacity="0.6"/>')
        mid1 = ((4+9)/2/12)*2*math.pi - math.pi/2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid1)*(R+24):.0f}" y="{cy+math.sin(mid1)*(R+24):.0f}" '
                   f'font-size="15" fill="#ffa000" text-anchor="middle">Termite Foraging</text>')

        # Swarm Pulse
        d2 = self._arc_path(cx, cy, R+22, 9.5, 11)
        svg.append(f'<path d="{d2}" fill="none" stroke="#ffee58" stroke-width="6" stroke-dasharray="4,4" opacity="0.8"/>')
        mid2 = ((9.5+11)/2/12)*2*math.pi - math.pi/2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid2)*(R+34):.0f}" y="{cy+math.sin(mid2)*(R+34):.0f}" '
                   f'font-size="15" fill="#ffee58" text-anchor="middle">Alate Swarm and Fungi</text>')

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="#3e2723" stroke-width="1.5"/>')

        # Forager trails
        for fi in range(cfg.num_foragers):
            hist = sim.hist_forager_xy[fi]
            if not hist: continue
            fxs = ";".join(f"{h[0]:.1f}" for h in hist)
            fys = ";".join(f"{h[1]:.1f}" for h in hist)
            fops = ";".join(f"{h[2] * 0.4:.2f}" for h in hist)
            
            svg.append(f'<circle r="1.5" fill="#ffb300" opacity="0">'
                       f'<animate attributeName="cx" values="{fxs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{fys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{fops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # Mounds
        for mi, m in enumerate(sim.mounds):
            px, py = m["pos"]
            # Base mound
            svg.append(f'<ellipse cx="{px:.0f}" cy="{py:.0f}" rx="{cfg.mound_radius}" ry="{cfg.mound_radius*0.8}" fill="#5d4037" stroke="#3e2723" stroke-width="1"/>')
            
            # Biomass indicator (internal glow)
            b_ops = ";".join(f"{sim.hist_mound_state[mi][fi][0]/20.0:.2f}" for fi in range(F))
            svg.append(f'<ellipse cx="{px:.0f}" cy="{py:.0f}" rx="{cfg.mound_radius*0.6}" ry="{cfg.mound_radius*0.4}" fill="#e65100" opacity="0">'
                       f'<animate attributeName="opacity" values="{b_ops}" '
                       f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></ellipse>')

            # Fungal Mushrooms
            f_ops = ";".join(f"{sim.hist_mound_state[mi][fi][1]/15.0:.2f}" for fi in range(F))
            svg.append(f'<circle cx="{px+8:.0f}" cy="{py-10:.0f}" r="5" fill="#e0e0e0" opacity="0">'
                       f'<animate attributeName="opacity" values="{f_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            svg.append(f'<circle cx="{px-6:.0f}" cy="{py-8:.0f}" r="4" fill="#e0e0e0" opacity="0">'
                       f'<animate attributeName="opacity" values="{f_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Swarmers (Animated via pre-rendered DOM updates due to dynamic creation)
        # To represent dynamic swarmers efficiently in SVG without bloating, we'll draw fixed nodes that fade in/out and teleport.
        # We will render ~80 dummy swarmers and animate their properties based on a subsample of real swarm events.
        MAX_DUMMIES = 80
        dummy_x = [["0"]*F for _ in range(MAX_DUMMIES)]
        dummy_y = [["0"]*F for _ in range(MAX_DUMMIES)]
        dummy_op = [["0.0"]*F for _ in range(MAX_DUMMIES)]
        dummy_r = [["0"]*F for _ in range(MAX_DUMMIES)]
        
        for fi, event_list in enumerate(sim.hist_swarmer_xy):
            for i, swarmer in enumerate(event_list[:MAX_DUMMIES]):
                sx, sy, sz, slife = swarmer
                dummy_x[i][fi] = f"{sx:.1f}"
                dummy_y[i][fi] = f"{sy:.1f}"
                dummy_op[i][fi] = f"{slife * 0.9:.2f}"
                dummy_r[i][fi] = f"{sz:.1f}"

        for i in range(MAX_DUMMIES):
            # Only include if it ever appears
            if any(float(op) > 0 for op in dummy_op[i]):
                dxs = ";".join(dummy_x[i])
                dys = ";".join(dummy_y[i])
                dops = ";".join(dummy_op[i])
                drs = ";".join(dummy_r[i])
                
                # Check dominant type for color
                col = "#ffee58" if "3" in drs else "#cfd8dc"
                
                svg.append(f'<circle fill="{col}" opacity="0">'
                           f'<animate attributeName="cx" values="{dxs}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="cy" values="{dys}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="r" values="{drs}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="opacity" values="{dops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'</circle>')


        # Hand
        hx = ";".join(f"{cx+math.cos((m/12)*2*math.pi-math.pi/2)*(R-10):.1f}" for m in sim.hist_month)
        hy = ";".join(f"{cy+math.sin((m/12)*2*math.pi-math.pi/2)*(R-10):.1f}" for m in sim.hist_month)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
                   f'stroke="#ffb300" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
                   f'<animate attributeName="x2" values="{hx}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="y2" values="{hy}" dur="{dur}s" repeatCount="indefinite"/></line>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#100d08" stroke="#ffb300" stroke-width="2"/>')

        # PANELS
        px5 = w - 395; pw = 375
        
        # P1
        svg.append(f'<g transform="translate({px5}, 20)"><rect width="{pw}" height="178" fill="#0f0c08" rx="8" stroke="#ffb300" stroke-width="1" opacity="0.95"/>')
        svg.append(f'<text x="12" y="22" fill="#ffb300" font-size="15" font-weight="bold">Subterranean Fungiculture</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#bcaaa4" font-size="15">Nasutitermes are the Cerrado\'s chief decomposers.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#bcaaa4" font-size="15">Obligate mutualism with Termitomyces fungi.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ffa000" font-size="15">Dry Season: Peak foraging; termites build combs</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ffa000" font-size="15">inside the mound; fungus yields protein nodules.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="100" fill="#ffee58" font-size="15">Wet Season (Oct-Nov): First heavy rains provoke</text>')
        svg.append(f'<text font-weight="bold" x="12" y="114" fill="#ffee58" font-size="15">synchronized Revoada of winged alate termites.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="128" fill="#e0e0e0" font-size="15">Simultaneously, the fungus fruits large mushrooms.</text>')
        svg.append('</g>')

        # P2 Metrics
        svg.append(f'<g transform="translate({px5}, 198)"><rect width="{pw}" height="172" fill="#0f0c08" rx="8" stroke="#4e342e"/>')
        svg.append(f'<text x="12" y="22" fill="#bcaaa4" font-size="15" font-weight="bold">Mound Metadata</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#ffa000" font-size="15">Biomass Foraging Trips: {sim.total_forage_trips}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#ffee58" font-size="15">Simulated Swarm Particles: {sim.total_swarm_events}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#6d4c41" font-size="15">Macro-mounds: {cfg.num_mounds} | Foraging Parties: {cfg.num_foragers}</text></g>')

        # P3 Chart
        cvs = [(P_FORAGING_CURVE, "#ffa000", "Termite Foraging Rate"),
               (FUNGAL_METABOLISM_CURVE, "#e65100", "Internal Fungal Metabolism"),
               (SWARM_FRUIT_PULSE, "#ffee58", "Alate Swarm and Fungal Bloom"),
               (RAINFALL_CURVE, "#4fc3f7", "Rainfall (Trigger)")]
        chart = draw_phenology_chart(cvs, 345, 98, 212, "Decomposer Phenology", "#ffb300", "#0f0c08", "#ffb300", legend_row_h=28)
        svg.append(f'<g transform="translate({px5}, 380)">{chart}</g>')

        # Status Line
        svg.append(f'<g transform="translate(20, {h-240})"><rect width="260" height="230" fill="#0f0c08" rx="8" stroke="#ffb300" opacity="0.95"/>')
        svg.append(f'<text x="12" y="22" fill="#ffb300" font-weight="bold">Active Season Status:</text>')
        mnames = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        for mi, m in enumerate(mnames):
            vs = ["0"]*12; vs[mi] = "1"
            op = ";".join(vs+["0"])
            svg.append(f'<text x="12" y="52" font-size="15" fill="#ffffff" font-weight="bold">{m}'
                       f'<animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></text>')
            
            plse = sim._interp(SWARM_FRUIT_PULSE, mi)
            frge = sim._interp(P_FORAGING_CURVE, mi)
            if plse > 0.4:
                st1, c1 = "RAINFALL TRIGGER ACTIVATED", "#4fc3f7"
                st2, c2 = "Huge Alate Swarming (Revoada)", "#ffee58"
                st3, c3 = "Termitomyces mushrooms fruit", "#e0e0e0"
            elif frge > 0.7:
                st1, c1 = "PEAK DRY-SEASON FORAGING", "#ffa000"
                st2, c2 = "Building interior fungus combs", "#e65100"
                st3, c3 = "Mound sealed; metabolism high", "#ff8f00"
            else:
                st1, c1 = "WET SEASON REBUILDING", "#8bc34a"
                st2, c2 = "Colony expansion underway", "#9ccc65"
                st3, c3 = "Low foraging exteriorly", "#7cb342"
            
            for yo, txt, co in [(80, st1, c1), (100, st2, c2), (120, st3, c3)]:
                svg.append(f'<text x="12" y="{yo}" font-size="15" fill="{co}" font-weight="bold">{txt}'
                           f'<animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></text>')
        
        svg.append(f'<text x="12" y="145" fill="#6d4c41" font-size="15" font-weight="bold">Legend:</text>')
        leg = [(160,"#5d4037","Termite Macro-Mound"),(175,"#e65100","Internal Fungal Comb (Metabolism)"),
               (190,"#ffb300","Foraging Party"),(205,"#ffee58","Alate Swarm / Fungal Spores"),
               (220,"#e0e0e0","Termitomyces Mushroom Fruiting")]
        for ey, ec, el in leg:
            svg.append(f'<circle cx="22" cy="{ey}" r="5" fill="{ec}"/><text font-weight="bold" x="36" y="{ey+4}" fill="{ec}" font-size="15">{el}</text>')
        svg.append('</g>')

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    sim = TermitomycesSim(CONFIG)
    for frame in range(CONFIG.frames): sim.step(frame)
    print(f"Done: {sim.total_forage_trips} trips, {sim.total_swarm_events} swarmers.")
    svg_content = MoundRenderer(CONFIG, sim).generate_svg()
    save_svg(svg_content, 'notebook_78')

if __name__ == "__main__": main()
