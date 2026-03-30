# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 59: AMF Fungi ↔ Cerrado Root Network - Mycorrhizal Seasonality Clock
# INTERVENTION 5/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_59.py - Arbuscular Mycorrhizal Fungi (AMF) × Cerrado Roots:
Notebook Differentiation:
- Differentiation Focus: AMF Fungi ↔ Cerrado Root Network - Mycorrhizal Seasonality Clock emphasizing recreational disturbance.
- Indicator species: Lambari (Astyanax sp.).
- Pollination lens: floral resource concentration in veredas.
- Human impact lens: carbon stock monitoring incentives.

                 Common Mycorrhizal Network (CMN) Phenological Clock

Models the hidden "upside-down forest" of the Cerrado, focusing on the
crucial obligate mutualism between plant root systems and AMF networks.
Since Cerrado soils (Oxisols/Ultisols) are naturally acidic and poor in
available Phosphorus (P) and Nitrogen (N), AMF hyphae act as extensions
of the root system, exchanging scavenged soil nutrients and water for
plant-synthesized Carbon (C).

This keeps the notebook aligned with the project's evidence-based ecological
argumentation: below-ground interaction is presented as a seasonal exchange
economy, where carbon supply, phosphorus demand, hydraulic redistribution, and
post-disturbance recovery co-vary instead of appearing as isolated processes.

The radial phenological clock maps:
  • Wet/Dry season shifts in resource exchange (Carbon vs. Nutrients/Water)
  • Phosphorus (P) transfer peaks (early wet season root flush)
  • Hydraulic lift: deep-rooted trees pulling water for the shallow CMN
    during the dry season stress (Jul-Sep)
  • Fire survival: post-fire rapid resprouting fueled by underground
    fungal-supported reserves
  • Sporulation: AMF spore production at the end of the wet season

Scientific References:
  - Miranda, J.C.C. et al. (2005). "Arbuscular mycorrhizal fungi in
    Cerrado soils." Braz. J. Microbiol.
  - Oliveira, R.S. et al. (2005). "Hydraulic redistribution in three
    Amazonian trees." Oecologia. (Applied to deep Cerrado roots).
  - Zangaro, W. et al. (2003). "Mycorrhizal response and successional
    status in tropical woody species." Mycorrhiza.

Scientific Relevance (PIGT RESEX Recanto das Araras - 2024):
  - Integrates the socio-environmental complexity of 
    de Cima, Goiás, Brazil.
  - Models landscape connectivity, karst vulnerability (Bacia do Rio Lapa),
    biological corridors, and seed-dispersal networks.
  - Outputs are published via Google Sites: 
  - SVG artefacts archived at: https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # pyre-ignore[21]
import numpy as np  # pyre-ignore[21]
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from eco_base import save_svg, sanitize_svg_text  , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML  # pyre-ignore[21]
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES
# ===================================================================================================

# Soil Moisture (Oxisols) - drives overall microbial and root activity
SOIL_MOISTURE_CURVE = [0.80, 0.70, 0.85, 0.60, 0.35, 0.15, 0.05, 0.05, 0.25, 0.50, 0.80, 0.90]

# Plant Photosynthesis / Carbon Allocation to roots
# High during wet season leaf flush, plummets during dry season leaf drop
CARBON_ALLOCATION_CURVE = [
    0.85,   # JAN
    0.90,   # FEV - Max canopy
    0.80,   # MAR
    0.60,   # ABR
    0.40,   # MAI
    0.20,   # JUN - Senescence
    0.15,   # JUL
    0.10,   # AGO - Minimum
    0.30,   # SET - Flush begins
    0.50,   # OUT
    0.70,   # NOV
    0.80,   # DEZ
]

# AMF Nutrient Gathering (Phosphorus/Nitrogen)
# Peaks early wet season to support new leaf flush, uses stored carbon
AMF_NUTRIENT_CURVE = [
    0.60,   # JAN
    0.50,   # FEV
    0.40,   # MAR
    0.30,   # ABR
    0.20,   # MAI
    0.10,   # JUN
    0.10,   # JUL
    0.15,   # AGO
    0.40,   # SET - Pre-rain activation
    0.85,   # OUT - Peak P demand for flush
    0.80,   # NOV
    0.70,   # DEZ
]

# Hydraulic Redistribution (Deep Roots -> CMN -> Shallow Roots)
# Occurs intensely during the dry season
HYDRAULIC_LIFT_CURVE = [
    0.05,   # JAN
    0.05,   # FEV
    0.10,   # MAR
    0.20,   # ABR
    0.40,   # MAI
    0.70,   # JUN
    0.90,   # JUL - Peak reliance on deep water
    0.95,   # AGO
    0.85,   # SET
    0.40,   # OUT
    0.15,   # NOV
    0.05,   # DEZ
]

# Underground Fire Survival / Resprout Potential
# Late dry season fires top-kill plants, but fungi fuel rapid resprout
RESPROUT_POTENTIAL_CURVE = [
    0.20, 0.20, 0.20, 0.20, 0.30, 0.40, 0.60, 0.90, 0.95, 0.80, 0.40, 0.20
]


@dataclass
class MycorrhizaConfig:
    """Configuration for the AMF × Root Network clock."""
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 230.0
    
    num_plants: int = 12
    num_fungal_hubs: int = 18
    num_particles: int = 150  # Resources flowing through hyphae
    
    particle_speed: float = 0.05
    max_spores: int = 60


CONFIG = MycorrhizaConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class MycorrhizaSim:
    """Phenological clock for Common Mycorrhizal Network (CMN)."""

    def __init__(self, cfg: MycorrhizaConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Plant Nodes (Peripheral) ---
        self.plants: List[Dict] = []
        for i in range(cfg.num_plants):
            angle = (i / cfg.num_plants) * 2 * math.pi - math.pi/2
            # Plant positions roughly define the clock perimeter
            px = cx + math.cos(angle) * (R - 20)
            py = cy + math.sin(angle) * (R - 20)
            
            # Deep rooted vs shallow rooted
            is_deep = (i % 3 == 0) 
            
            self.plants.append({
                "pos": (px, py),
                "angle": angle,
                "is_deep": is_deep,
                "c_reserve": 0.5,
                "month_idx": int((i / cfg.num_plants) * 12)
            })

        # --- Fungal Hubs (Inner area) ---
        self.fungal_hubs: List[Dict] = []
        for i in range(cfg.num_fungal_hubs):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(30, R - 60)
            fx = cx + math.cos(angle) * r
            fy = cy + math.sin(angle) * r
            self.fungal_hubs.append({
                "pos": (fx, fy),
                "angle": angle,
                "r": r,
                "connected_plants": []
            })

        # Connect hubs to nearby plants (Hyphal network)
        self.hyphae: List[Tuple[int, int]] = [] # (hub_idx, plant_idx)
        for hi, hub in enumerate(self.fungal_hubs):
            hx, hy = hub["pos"]
            for pi, p in enumerate(self.plants):
                px, py = p["pos"]
                dist = math.sqrt((hx-px)**2 + (hy-py)**2)
                if dist < 120:  # Connection radius
                    hub["connected_plants"].append(pi)
                    self.hyphae.append((hi, pi))

        # --- Resource Particles (Carbon, Phosphorus, Water) ---
        # Particles travel along hyphae
        self.particles = []
        for i in range(cfg.num_particles):
            if not self.hyphae: break
            hi, pi = random.choice(self.hyphae)
            # 0=Carbon (Plant->Fungus), 1=Phosphorus (Fungus->Plant), 2=Water (DeepPlant->Fungus->ShallowPlant)
            ptype = random.choice([0, 1, 2])
            pos_t = random.random() # 0.0 to 1.0 along the hypha
            self.particles.append({
                "hi": hi,
                "pi": pi,
                "type": ptype,
                "t": pos_t,
                "speed": random.uniform(0.02, 0.06),
                "active": False
            })

        self.hist_month: List[float] = []
        self.hist_c: List[float] = []
        self.hist_p: List[float] = []
        self.hist_w: List[float] = []
        
        # History of particle positions for rendering
        self.particle_hist_xy: List[List[Tuple[float, float, int]]] = [[] for _ in range(cfg.num_particles)]
        
        self.spore_events: List[Dict] = []
        self.total_c_transferred = 0.0
        self.total_p_transferred = 0.0
        self.total_w_transferred = 0.0

    def _interp(self, curve: list, month_frac: float) -> float:
        m = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        month_frac = (frame / cfg.frames) * 12.0
        m_idx = int(month_frac) % 12

        # Seasonal curves
        moisture = self._interp(SOIL_MOISTURE_CURVE, month_frac)
        c_flow = self._interp(CARBON_ALLOCATION_CURVE, month_frac)
        p_flow = self._interp(AMF_NUTRIENT_CURVE, month_frac)
        w_flow = self._interp(HYDRAULIC_LIFT_CURVE, month_frac)
        resprout = self._interp(RESPROUT_POTENTIAL_CURVE, month_frac)

        self.hist_month.append(month_frac)
        self.hist_c.append(c_flow)
        self.hist_p.append(p_flow)
        self.hist_w.append(w_flow)

        # Update Particles
        clock_angle = (month_frac / 12) * 2 * math.pi - math.pi/2
        
        for i, p in enumerate(self.particles):
            hi, pi_idx = p["hi"], p["pi"]
            hx, hy = self.fungal_hubs[hi]["pos"]
            px, py = self.plants[pi_idx]["pos"]
            p_angle = self.plants[pi_idx]["angle"]
            
            # sweeping activation based on clock hand proximity
            angle_diff = abs((p_angle - clock_angle + math.pi) % (2*math.pi) - math.pi)
            if angle_diff < 1.0: # active sector
                p["active"] = True
            else:
                p["active"] = False

            if p["active"]:
                # Flow direction and speed depends on resource type
                if p["type"] == 0: # Carbon: Plant -> Fungus
                    flow_rate = c_flow * p["speed"]
                    p["t"] -= flow_rate 
                    if p["t"] <= 0:
                        p["t"] = 1.0  # Reset to plant
                        self.total_c_transferred += 1.0
                elif p["type"] == 1: # Phosphorus: Fungus -> Plant
                    flow_rate = p_flow * p["speed"] * 1.5
                    p["t"] += flow_rate
                    if p["t"] >= 1.0:
                        p["t"] = 0.0  # Reset to hub
                        self.total_p_transferred += 1.0
                elif p["type"] == 2: # Water (Hydraulic lift): Hub -> Plant (driven by dry season)
                    if not self.plants[pi_idx]["is_deep"]:
                        flow_rate = w_flow * p["speed"] * 2.0
                        p["t"] += flow_rate
                        if p["t"] >= 1.0:
                            p["t"] = 0.0
                            self.total_w_transferred += 1.0
                    else:
                        # Deep roots pull water TO the hub
                        flow_rate = w_flow * p["speed"] * 2.0
                        p["t"] -= flow_rate
                        if p["t"] <= 0.0:
                            p["t"] = 1.0
            
            # Interpolate position
            curr_x = hx + (px - hx) * p["t"]
            curr_y = hy + (py - hy) * p["t"]
            self.particle_hist_xy[i].append((curr_x, curr_y, p["type"] if p["active"] else -1))

        # Sporulation event: Late wet season (Apr-May)
        if 3.0 < month_frac < 5.0 and len(self.spore_events) < cfg.max_spores:
            if random.random() < 0.05:
                # Spawn near a fungal hub
                hub = random.choice(self.fungal_hubs)
                hx, hy = hub["pos"]
                sx = hx + random.uniform(-15, 15)
                sy = hy + random.uniform(-15, 15)
                self.spore_events.append({
                    "pos": (sx, sy),
                    "frame": frame
                })


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class MycorrhizaRenderer:
    """Renders the AMF × Root Network phenological clock as animated SVG."""

    def __init__(self, cfg: MycorrhizaConfig, sim: MycorrhizaSim):
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
            f'style="background-color:#14110f; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # --- Defs ---
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="soilBg">'
            '<stop offset="0%" stop-color="#3e2723" stop-opacity="0.9"/>'
            '<stop offset="60%" stop-color="#261a14" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#14110f" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<pattern id="soilTexture" width="60" height="60" patternUnits="userSpaceOnUse">'
            '<path d="M10 10 L15 12 M40 30 L38 35 M20 50 L25 48" stroke="#3e2723" stroke-width="1" opacity="0.3"/>'
            '<circle cx="30" cy="20" r="1" fill="#4e342e" opacity="0.4"/>'
            '<circle cx="50" cy="50" r="1.5" fill="#4e342e" opacity="0.2"/>'
            '</pattern>'
        )
        svg.append(
            '<filter id="glowFungus" x="-30%" y="-30%" width="160%" height="160%">'
            '<feGaussianBlur stdDeviation="2.5" result="blur" />'
            '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
            '</filter>'
        )
        svg.append('</defs>')

        # --- Background (Underground Soil) ---
        svg.append(f'<rect width="{w}" height="{h}" fill="#14110f"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#soilTexture)"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 40}" fill="url(#soilBg)"/>')

        # --- Title ---
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#dce775" font-weight="bold">'
            f'ECO-SIM: AMF Fungi × Cerrado Roots - Mycorrhizal Seasonality</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#a1887f">'
            f'The Upside-Down Forest: Carbon, Phosphorus, and Hydraulic Lift via Common Mycorrhizal Networks</text>'
        )

        # --- Clock face ---
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 38)
            ty = cy + math.sin(angle) * (R + 38)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#8d6e63" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 8)
            ly2 = cy + math.sin(angle) * (R - 8)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#5d4037" stroke-width="2"/>'
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
            svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="8" '
                       f'stroke-linecap="round" opacity="{opacity}"/>')
            mid = ((start_m + end_m) / 2 / 12) * 2 * math.pi - math.pi / 2
            lx = cx + math.cos(mid) * (radius + 12)
            ly = cy + math.sin(mid) * (radius + 12)
            svg.append(f'<text font-weight="bold" x="{lx:.0f}" y="{ly:.0f}" font-size="15" fill="{color}" '
                       f'text-anchor="middle" opacity="0.85">{label}</text>')

        draw_arc(9.5, 3.5, R + 15, "#4caf50", "Carbon Export (Wet Season)", 0.4)
        draw_arc(4.5, 8.5, R + 33, "#29b6f6", "Hydraulic Lift (Dry Season)", 0.4)
        draw_arc(8.5, 10.5, R + 25, "#ff5722", "High Fire Risk", 0.3)

        # --- Hyphal Network (Lines) ---
        for hi, pi in sim.hyphae:
            hx, hy = sim.fungal_hubs[hi]["pos"]
            px, py = sim.plants[pi]["pos"]
            # Wavy faint line for hyphae
            svg.append(
                f'<line x1="{hx:.0f}" y1="{hy:.0f}" x2="{px:.0f}" y2="{py:.0f}" '
                f'stroke="#dce775" stroke-width="0.8" opacity="0.15" stroke-dasharray="2,3"/>'
            )

        # --- Fungal Hubs ---
        for hub in sim.fungal_hubs:
            hx, hy = hub["pos"]
            svg.append(
                f'<circle cx="{hx:.0f}" cy="{hy:.0f}" r="4" fill="#aed581" filter="url(#glowFungus)" opacity="0.6"/>'
            )

        # --- Plant Roots ---
        for p in sim.plants:
            px, py = p["pos"]
            col = "#00bcd4" if p["is_deep"] else "#8bc34a"  # Cyan for deep roots, Green for shallow
            size = 7 if p["is_deep"] else 5
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" r="{size}" fill="{col}" stroke="#33691e" stroke-width="1.5" opacity="0.85"/>'
            )
            # small root hairs
            for _ in range(3):
                ha = random.uniform(0, 2*math.pi)
                hr = random.uniform(6, 12)
                hx = px + math.cos(ha)*hr
                hy = py + math.sin(ha)*hr
                svg.append(f'<line x1="{px:.0f}" y1="{py:.0f}" x2="{hx:.0f}" y2="{hy:.0f}" stroke="{col}" stroke-width="1" opacity="0.6"/>')

        # --- Resource Particles Flowing ---
        # 0=Carbon (yellow), 1=Phosphorus (magenta), 2=Water (blue)
        pt_colors = ["#ffee58", "#e040fb", "#29b6f6"]
        
        for i, p_hist in enumerate(sim.particle_hist_xy):
            if not p_hist: continue
            
            # Since some frames the particle is inactive (-1), we use opacity
            cxs = ";".join(str(round(h[0], 1)) for h in p_hist)
            cys = ";".join(str(round(h[1], 1)) for h in p_hist)
            ops = ";".join("0.8" if h[2] != -1 else "0.0" for h in p_hist)
            
            ptype = int(sim.particles[i]["type"])
            col = pt_colors[ptype]
            
            svg.append(
                f'<circle r="2" fill="{col}">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # --- Spore Events ---
        for se in sim.spore_events:
            sx, sy = se["pos"]
            sf = se["frame"]
            
            ops = ";".join("0.0" if fi < sf else "0.7" for fi in range(F))
            szs = ";".join("0" if fi < sf else f"{min(3.0, (fi-sf)*0.2):.1f}" for fi in range(F))
            
            svg.append(
                f'<circle cx="{sx:.0f}" cy="{sy:.0f}" fill="#cddc39" stroke="#9e9d24" stroke-width="0.5" opacity="0.0">'
                f'<animate attributeName="r" values="{szs}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # --- Clock hand ---
        hand_x_vals = ";".join(f"{cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-15):.1f}" for m in sim.hist_month)
        hand_y_vals = ";".join(f"{cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-15):.1f}" for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 15}" '
            f'stroke="#dce775" stroke-width="2.5" stroke-linecap="round" opacity="0.8">'
            f'<animate attributeName="x2" values="{hand_x_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        # --- Centre hub ---
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="18" fill="#261a14" stroke="#aed581" stroke-width="3"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#aed581"/>')
        svg.append(f'<text font-weight="bold" x="{cx}" y="{cy + 4}" font-size="15" fill="#fff" text-anchor="middle">SOIL</text>')

        # ═══════════════════════════════════════════════════════════════════
        # RIGHT PANELS
        # ═══════════════════════════════════════════════════════════════════

        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 210
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1a1a2e" rx="8" '
                   f'stroke="#dce775" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#dce775" font-size="15" font-weight="bold">'
                   f'CMN Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'CMN: seasonal exchange network below ground.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ffee58" font-size="15">'
                   f'• Yellow: wet-season carbon to fungal partners</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#e040fb" font-size="15">'
                   f'• Magenta: phosphorus returned for root flush</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#29b6f6" font-size="15">'
                   f'• Cyan: hydraulic lift buffers shallow roots</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'keeping the shallow CMN active under drought stress.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#cddc39" font-size="15">'
                   f'• Green dots: sporulation builds resilience after rains.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 185
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1a1a2e" rx="8" '
                   f'stroke="#4fc3f7" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#4fc3f7" font-size="15" font-weight="bold">'
                   f'Exchange Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#ffee58" font-size="15">'
                   f'Total Carbon Assimilated: {sim.total_c_transferred:,.0f} units</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#e040fb" font-size="15">'
                   f'Phosphorus Extracted: {sim.total_p_transferred:,.0f} units</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#29b6f6" font-size="15">'
                   f'Dry-season Water Lifted: {sim.total_w_transferred:,.0f} units</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#cddc39" font-size="15">'
                   f'Spore Banks Created: {len(sim.spore_events)}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="116" fill="#8bc34a" font-size="15">'
                   f'Network: {cfg.num_plants} Plants, {cfg.num_fungal_hubs} Fungal Hubs, {len(sim.hyphae)} Links</text>')
        svg.append('</g>')

        # --- Panel 3: Annual curves ---
        py3 = py2 + ph2 + 10
        ph3 = 167
        chart_w = panel_w - 30
        chart_h = 52
        chart_x0 = 15
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1a1a2e" rx="8" '
                   f'stroke="#7e57c2" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#7e57c2" font-size="15" font-weight="bold">'
                   f'Annual Curves</text>')

        curves = [
            (CARBON_ALLOCATION_CURVE, "#ffee58", "Carbon Flow"),
            (AMF_NUTRIENT_CURVE, "#e040fb", "P Extraction"),
            (HYDRAULIC_LIFT_CURVE, "#29b6f6", "Hydraulic Lift"),
            (SOIL_MOISTURE_CURVE, "#8d6e63", "Soil Moisture"),
        ]
        for curve_data, color, label in curves:
            pts = []
            for mi in range(12):
                px_c = chart_x0 + (mi / 11) * chart_w
                py_c = chart_y0 + chart_h - curve_data[mi] * chart_h
                pts.append(f"{px_c:.0f},{py_c:.0f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="1.8" opacity="0.8"/>')

        legend_y = chart_y0 + chart_h + 14
        col_w = (chart_w) // 2
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
    """Run AMF × Cerrado Roots clock simulation."""
    print(f" - AMF Fungi × Cerrado Roots Clock on {CONFIG.device}...")

    sim = MycorrhizaSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_c_transferred:,.0f}C, {sim.total_p_transferred:,.0f}P, "
          f"{sim.total_w_transferred:,.0f}W units transferred.")

    print("Generating SVG...")
    renderer = MycorrhizaRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_59')
    return svg_content


if __name__ == "__main__":
    main()
