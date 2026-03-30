# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 64: Sapo-cururu & Pererecas ↔ Ephemeral Ponds — Explosive Breeding Clock
# INTERVENTION 10/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_64.py — Cerrado Amphibians ↔ Temporary Ponds:
Notebook Differentiation:
- Differentiation Focus: Sapo-cururu & Pererecas ↔ Ephemeral Ponds — Explosive Breeding Clock emphasizing edge colonization dynamics.
- Indicator species: Teiu (Salvator merianae).
- Pollination lens: hummingbird territorial patches.
- Human impact lens: fire management tradeoffs.

                 Explosive Breeding Phenological Clock

Models the extreme dependence of Cerrado amphibians (e.g., Rhinella diptycha
'Sapo-cururu', Physalaemus spp., Boana spp.) on the unpredictable and highly
seasonal rainfall of the biome, which forms temporary ponds.

The radial phenological clock maps:
  • Rainfall pulses and the rapid formation of temporary ponds.
  • "Explosive breeding" events: amphibians emerge from dry-season
    aestivation (often underground or in bromeliads) immediately after the
    first heavy rains (Oct-Nov) to mate and lay eggs en masse.
  • Tadpole development races against pond desiccation. If the pond dries
    before metamorphosis, the cohort perishes.
  • Acoustic choruses calling mates to the ephemeral water bodies.
  • Retreat into dormancy as the long dry season sets in.

Scientific References:
  - Prado, C.P.A. et al. (2005). "Breeding activity patterns, reproductive
    modes, and habitat use by anurans (Amphibia) in a seasonal environment
    in the Pantanal/Cerrado." Amphibia-Reptilia.
  - Kopp, K. et al. (2006). "Amphibian species richness and temporary ponds
    in the Cerrado."
  - Vasconcelos, T.S. et al. (2014). "Biogeographic patterns of Cerrado
    anurans."

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

# Rainfall Intensity (Sharp pulses in early wet season)
# Rain fills the ephemeral ponds. Highly clustered in Nov-Jan.
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]
# Lags slightly behind rainfall, drops rapidly due to evaporation/infiltration.
POND_WATER_LEVEL = [
    0.85,   # JAN
    0.80,   # FEV
    0.70,   # MAR
    0.50,   # ABR
    0.20,   # MAI
    0.05,   # JUN -> Ponds drying up
    0.00,   # JUL
    0.00,   # AGO -> Completely dry
    0.00,   # SET
    0.30,   # OUT
    0.80,   # NOV -> Rapid fill
    0.95,   # DEZ
]

# Aestivation (Dormancy Underground/Refuges)
# 1.0 = All frogs dormant. Drops to 0 during rainy season.
AESTIVATION_CURVE = [
    0.0,    # JAN — Active
    0.0,    # FEV
    0.1,    # MAR
    0.4,    # ABR -> starting to seek shelter
    0.8,    # MAI
    1.0,    # JUN -> fully dormant
    1.0,    # JUL
    1.0,    # AGO
    0.8,    # SET
    0.3,    # OUT -> emergence!
    0.0,    # NOV -> explosive breeding
    0.0,    # DEZ
]


@dataclass
class AmphibianConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_frogs: int = 40
    frog_speed: float = 1.2
    
    num_ponds: int = 4
    
    metamorphosis_frames: int = 35  # Time for tadpoles to become froglets
    breeding_threshold: float = 0.6 # Pond water level needed to breed


CONFIG = AmphibianConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class AmphibianSim:

    def __init__(self, cfg: AmphibianConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Ephemeral Ponds ---
        self.ponds: List[Dict] = []
        for i in range(cfg.num_ponds):
            angle = (i / cfg.num_ponds) * 2 * math.pi + random.uniform(-0.5, 0.5)
            r = random.uniform(50, R - 80)
            px = cx + math.cos(angle) * r
            py = cy + math.sin(angle) * r
            self.ponds.append({
                "pos": (px, py),
                "water": 0.0,
                "tadpoles": []  # List of timers
            })

        # --- Frogs/Toads ---
        self.frogs: List[Dict] = []
        for _ in range(cfg.num_frogs):
            angle = random.uniform(0, 2 * math.pi)
            tx = cx + math.cos(angle) * (R - 20)
            ty = cy + math.sin(angle) * (R - 20)
            self.frogs.append({
                "pos": torch.tensor([tx, ty], device=self.dev, dtype=torch.float32),
                "state": "aestivating", # aestivating, commuting, calling, breeding
                "target_pond": -1,
                "refuge_pos": torch.tensor([tx, ty], device=self.dev, dtype=torch.float32),
                "bred_this_season": False
            })

        self.hist_xy: List[List[Tuple[float, float]]] = [[] for _ in range(cfg.num_frogs)]
        self.hist_month: List[float] = []
        
        self.chorus_events: List[Dict] = [] # Visualizing frog calls
        self.metamorph_events: int = 0      # Surviving froglets
        self.desiccated_events: int = 0     # Tadpoles that dried up

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
        
        water_lvl = self._interp(POND_WATER_LEVEL, month_frac)
        aestivation = self._interp(AESTIVATION_CURVE, month_frac)
        rain = self._interp(RAINFALL_CURVE, month_frac)

        self.hist_month.append(month_frac)

        # Reset breeding status early in the dry season
        if month_frac > 6 and month_frac < 8:
            for f in self.frogs:
                f["bred_this_season"] = False

        # 1. Update Ponds & Tadpoles
        for pi, p in enumerate(self.ponds):
            p["water"] = water_lvl * (0.8 + random.random()*0.4)
            p["water"] = min(1.0, max(0.0, p["water"]))
            
            surviving_tadpoles = []
            for t_timer in p["tadpoles"]:
                if p["water"] < 0.1:
                    # Pond dried up!
                    self.desiccated_events += 1
                else:
                    t_timer += 1
                    if t_timer >= cfg.metamorphosis_frames:
                        # Success! Metamorphosis
                        self.metamorph_events += 1
                    else:
                        surviving_tadpoles.append(t_timer)
            p["tadpoles"] = surviving_tadpoles

        # 2. Update Frogs
        for i, frog in enumerate(self.frogs):
            pos = frog["pos"]
            
            # State Management
            if random.random() < aestivation:
                frog["state"] = "aestivating"
                frog["target_pond"] = -1
                target = frog["refuge_pos"]
            else:
                if frog["state"] == "aestivating":
                    frog["state"] = "commuting"
                
                # If active, look for a suitable pond to breed
                if frog["state"] != "calling" and frog["state"] != "breeding" and not frog["bred_this_season"]:
                    best_pond = -1
                    best_w = 0.0
                    for pi, p in enumerate(self.ponds):
                        if p["water"] > cfg.breeding_threshold and p["water"] > best_w:
                            best_w = p["water"]
                            best_pond = pi
                            
                    if best_pond != -1:
                        frog["target_pond"] = best_pond
                        target = torch.tensor(self.ponds[best_pond]["pos"], device=self.dev, dtype=torch.float32)
                    else:
                        # No good pond, wander near refuge
                        target = frog["refuge_pos"] + torch.tensor([random.uniform(-10,10), random.uniform(-10,10)], device=self.dev)
                elif frog["bred_this_season"]:
                    # Done breeding, forage near refuge
                    target = frog["refuge_pos"] + torch.tensor([random.uniform(-15,15), random.uniform(-15,15)], device=self.dev)
                    frog["state"] = "commuting"

            # Movement & Action
            if 'target' in locals() and target is not None:
                to_target = target - pos
                dist = torch.norm(to_target).item()
                
                if frog["state"] in ["commuting", "aestivating"]:
                    if dist > 2.0:
                        dir_vec = to_target / dist
                        frog["pos"] += dir_vec * cfg.frog_speed * random.uniform(0.5, 1.5)
                        # Hopping jitter
                        if frame % 2 == 0:
                            frog["pos"] += torch.tensor([random.uniform(-2,2), random.uniform(-2,2)], device=self.dev)
                    else:
                        if frog["state"] == "commuting" and frog["target_pond"] != -1:
                            frog["state"] = "calling" # Reached pond
                            
                elif frog["state"] == "calling":
                    # At the pond, calling for mates
                    if len(self.chorus_events) < 150 and random.random() < 0.1:
                        self.chorus_events.append({
                            "pos": (pos[0].item(), pos[1].item()),
                            "frame": frame
                        })
                    
                    # Chance to breed (explosive!)
                    p = self.ponds[frog["target_pond"]]
                    if p["water"] > cfg.breeding_threshold and random.random() < 0.05:
                        frog["state"] = "breeding"
                        frog["bred_this_season"] = True
                        p["tadpoles"].append(0) # Lay eggs!
                    elif p["water"] < cfg.breeding_threshold - 0.1:
                        # Pond shrinking, stop calling
                        frog["state"] = "commuting"
                        frog["target_pond"] = -1
                        
                elif frog["state"] == "breeding":
                    # Brief breeding state, then done
                    frog["state"] = "commuting"
                    frog["target_pond"] = -1

            # Clamp
            frog["pos"][0] = torch.clamp(frog["pos"][0], cx - self.cfg.clock_radius + 10, cx + self.cfg.clock_radius - 10)
            frog["pos"][1] = torch.clamp(frog["pos"][1], cy - self.cfg.clock_radius + 10, cy + self.cfg.clock_radius - 10)

            self.hist_xy[i].append((frog["pos"][0].item(), frog["pos"][1].item()))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class AmphibianRenderer:

    def __init__(self, cfg: AmphibianConfig, sim: AmphibianSim):
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
            f'style="background-color:#161d27; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        svg.append(
            '<radialGradient id="mudBg">'
            '<stop offset="0%" stop-color="#2a231b" stop-opacity="0.9"/>'
            '<stop offset="70%" stop-color="#1e1b19" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#161d27" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="pondGrad">'
            '<stop offset="0%" stop-color="#00bcd4" stop-opacity="0.8"/>'
            '<stop offset="50%" stop-color="#0097a7" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#3e2723" stop-opacity="0.2"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # Background Landscape
        svg.append(f'<rect width="{w}" height="{h}" fill="#161d27"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="url(#mudBg)"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#00bcd4" font-weight="bold">'
            f'ECO-SIM: Amphibians × Ephemeral Ponds    - Explosive Breeding Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#a1c4ca">'
            f'Dry-season aestivation & race against pond desiccation</text>'
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

        draw_arc(4.5, 9.5, R + 10, "#8d6e63", "Underground Aestivation", 0.4)
        draw_arc(10, 1.5, R + 10, "#00bcd4", "Explosive Breeding", 0.4)
        draw_arc(10.5, 3.5, R + 22, "#4caf50", "Tadpole Metamorphosis", 0.4)

        # Ephemeral Ponds
        for p in sim.ponds:
            px, py = p["pos"]
            # Dry mud basin
            svg.append(f'<ellipse cx="{px:.0f}" cy="{py:.0f}" rx="25" ry="18" fill="#3e2723" opacity="0.5"/>')
            
            # Animated Water filling/drying
            r_vals_x = ";".join(f"{sim._interp(POND_WATER_LEVEL, (fi/F)*12)*22:.1f}" for fi in range(F))
            r_vals_y = ";".join(f"{sim._interp(POND_WATER_LEVEL, (fi/F)*12)*15:.1f}" for fi in range(F))
            op_vals = ";".join(f"{sim._interp(POND_WATER_LEVEL, (fi/F)*12):.2f}" for fi in range(F))
            
            svg.append(
                f'<ellipse cx="{px:.0f}" cy="{py:.0f}" fill="url(#pondGrad)">'
                f'<animate attributeName="rx" values="{r_vals_x}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="ry" values="{r_vals_y}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</ellipse>'
            )

        # Chorus Events (Sound waves from calling frogs)
        for ce in sim.chorus_events:
            cx_e, cy_e = ce["pos"]
            cf = ce["frame"]
            
            ops = ";".join("0.0" if fi < cf else f"{max(0, 0.8 - (fi-cf)*0.08):.2f}" for fi in range(F))
            szs = ";".join("0.0" if fi < cf else f"{min(12, (fi-cf)*1.5):.1f}" for fi in range(F))
            
            svg.append(
                f'<circle cx="{cx_e:.0f}" cy="{cy_e:.0f}" fill="none" stroke="#4dd0e1" stroke-width="1.5" opacity="0.0">'
                f'<animate attributeName="r" values="{szs}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # Frogs (Hopping blobs)
        for i in range(cfg.num_frogs):
            hist = sim.hist_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            
            # Opacity depends on aestivation (hidden underground during dry season)
            op_vals = ";".join(f"{1.0 - sim._interp(AESTIVATION_CURVE, (fi/F)*12)*0.8:.2f}" for fi in range(F))
            col_vals = ";".join("#5d4037" if sim._interp(AESTIVATION_CURVE, (fi/F)*12) > 0.5 else "#8bc34a" for fi in range(F))
            
            # Animated Frog body
            svg.append(
                f'<ellipse rx="3" ry="2.5" stroke="#33691e" stroke-width="0.5">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="fill" values="{col_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</ellipse>'
            )

        # Clock hand
        hand_x = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#00bcd4" stroke-width="2.5" stroke-linecap="round" opacity="0.7">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#161d27" stroke="#00bcd4" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#8bc34a"/>')


        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 218
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1b2430" rx="8" '
                   f'stroke="#00bcd4" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#00bcd4" font-size="15" font-weight="bold">'
                   f'Explosive Breeding Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Cerrado amphibians survive the dry season by</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'aestivating in bromeliads or soil (brown).</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'Oct-Nov rains trigger frog emergence (green).</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'They rush to temporary ponds to call (cyan</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'rings) and breed en masse. Tadpoles race to</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#e57373" font-size="15">'
                   f'metamorphose before the pond dries completely.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 169
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1b2430" rx="8" '
                   f'stroke="#4caf50" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#4caf50" font-size="15" font-weight="bold">'
                   f'Cohort Survival Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#8bc34a" font-size="15">'
                   f'Successful Metamorphosis: {sim.metamorph_events:,.0f} froglets</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#e57373" font-size="15">'
                   f'Desiccated (Dried up): {sim.desiccated_events:,.0f} cohorts perished</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#00e5ff" font-size="15">'
                   f'Temporary Ponds Mapped: {cfg.num_ponds}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'Adult Population: {cfg.num_frogs} | Acoustic chorus events: {len(sim.chorus_events)}</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 169
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1b2430" rx="8" '
                   f'stroke="#5c6bc0" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#5c6bc0" font-size="15" font-weight="bold">'
                   f'Phenological Curves</text>')

        curves = [
            (RAINFALL_CURVE, "#4dd0e1", "Rainfall"),
            (POND_WATER_LEVEL, "#0288d1", "Pond Water"),
            (AESTIVATION_CURVE, "#8d6e63", "Dormancy/Aestivation"),
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
    print(f" — anfíbios ↔ Ephemeral Ponds Clock on {CONFIG.device}...")

    sim = AmphibianSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.metamorph_events} metamorphosed, {sim.desiccated_events} dried up.")

    print("Generating SVG...")
    renderer = AmphibianRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_64')
    return svg_content


if __name__ == "__main__":
    main()
