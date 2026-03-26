# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 70: Jatobá (Hymenaea courbaril) ↔ Cutia — Seed Scatter-Hoarding Clock
# INTERVENTION 4/4: Seasonal & Migratory Interventions Series
# ===================================================================================================
"""
notebook_70.py — Jatobá Tree ↔ Agouti (Cutia):
Notebook Differentiation:
- Differentiation Focus: Jatobá (Hymenaea courbaril) ↔ Cutia — Seed Scatter-Hoarding Clock emphasizing edge microclimate.
- Indicator species: Arara-azul (Anodorhynchus hyacinthinus).
- Pollination lens: nocturnal bat pollination window.
- Human impact lens: road mortality and barrier effects.

                 Scatter-Hoarding Dispersion & Timed Germination Clock

Models the obligate scatter-hoarding mutualism between the Jatobá and the Cutia.
The Jatobá drops heavy, extremely hard-shelled fruit pods late in the dry season.
The Cutia (Dasyprocta), driven by the scarcity of food, breaks the outer shell,
eats some seeds, and buries (caches) the rest across its territory for later use.

Many buried seeds are forgotten or abandoned. When the defining heavy summer rains
arrive (Nov/Dec), the buried, slightly gnawed seeds are perfectly positioned
and scarified to absorb water, triggering massive synchronized germination far 
from the parent tree.

The radial phenological clock maps:
  • Jatobá Pod Drop (Late Dry Season).
  • Agouti (Cutia) Foraging & Scatter-hoarding caching.
  • Wet Season Onset (Triggering germination).
  • Forgotten Cache Germination (Seedlings).

Scientific Relevance:
  - Demonstrates animal-mediated zoochory essential for heavy-seeded flora.
  - Highlights behavioral caching aligned with precise seasonal weather cues.
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

# Seasonality (Wet Nov-Mar, Dry Jun-Sep)
RAIN_GERMINATION_CUE = [
    0.80, 0.70, 0.50, 0.20, 0.05, 0.00, 0.00, 0.00, 0.10, 0.40, 0.85, 0.95
]
JATOBA_FRUIT_DROP = [
    0.05, 0.00, 0.00, 0.00, 0.00, 0.10, 0.30, 0.80, 1.00, 0.70, 0.20, 0.10
]

# Cutia Hoarding Activity (Peaks when food is abundant on ground)
CUTIA_HOARDING_ACTIVITY = [
    0.20, 0.10, 0.10, 0.20, 0.30, 0.40, 0.60, 0.90, 1.00, 0.80, 0.50, 0.30
]

# Germination Success (Requires buried seed + high rain)
GERMINATION_SUCCESS = [0.80, 0.75, 0.60, 0.40, 0.20, 0.10, 0.05, 0.05, 0.10, 0.40, 0.80, 0.90]


@dataclass
class ScatterHoardingConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    
    num_trees: int = 6
    num_cutias: int = 12
    cutia_speed: float = 3.5


CONFIG = ScatterHoardingConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class ScatterHoardingSim:

    def __init__(self, cfg: ScatterHoardingConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Jatobá Trees ---
        self.trees: List[Dict] = []
        for _ in range(cfg.num_trees):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(20, R - 60)
            self.trees.append({
                "pos": (cx + math.cos(angle)*r, cy + math.sin(angle)*r),
                "pods_on_ground": 0.0
            })

        # --- Cutias (Agoutis) ---
        self.cutias: List[Dict] = []
        for _ in range(cfg.num_cutias):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(20, R - 20)
            self.cutias.append({
                "territory": torch.tensor([cx + math.cos(angle)*r, cy + math.sin(angle)*r], device=self.dev, dtype=torch.float32),
                "pos": torch.tensor([cx + math.cos(angle)*r, cy + math.sin(angle)*r], device=self.dev, dtype=torch.float32),
                "state": "searching", # searching, carrying, burying
                "carrying_pod": False,
                "target_xy": None
            })

        # --- Seed Caches ---
        self.caches: List[Dict] = []

        self.hist_month: List[float] = []
        self.hist_cutias_xy: List[List[Tuple[float, float, float, int]]] = [[] for _ in range(cfg.num_cutias)]
        
        self.total_pods_hoarded = 0
        self.total_germinations = 0

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
        
        fruit_drop = self._interp(JATOBA_FRUIT_DROP, month_frac)
        hoard_act = self._interp(CUTIA_HOARDING_ACTIVITY, month_frac)
        rain_cue = self._interp(RAIN_GERMINATION_CUE, month_frac)

        # 1. Trees drop pods
        for t in self.trees:
            if fruit_drop > 0.1:
                t["pods_on_ground"] += fruit_drop * 2.0
            else:
                t["pods_on_ground"] *= 0.98  # Slowly rot if not taken

        # 2. Caches Germinate
        for c in self.caches:
            if c["state"] == "buried" and rain_cue > 0.7:
                # Need strong rains to germinate
                if random.random() < 0.02:
                    c["state"] = "germinated"
                    self.total_germinations += 1
            if c["state"] == "germinated":
                c["growth"] = min(1.0, c["growth"] + 0.05)
                # Seedlings die if dry season returns fully
                if rain_cue < 0.2 and c["growth"] < 1.0:
                    if random.random() < 0.05:
                        c["state"] = "dead"

        # 3. Cutias Forage
        # Cutias are hyperactive in late dry season
        for i, cutia in enumerate(self.cutias):
            pos = cutia["pos"]
            
            is_active = random.random() < hoard_act
            if not is_active and cutia["state"] == "searching":
                # Rest in territory
                tgt = cutia["territory"]
                vec = tgt - pos
                if torch.norm(vec).item() > 2.0:
                    pos += (vec / torch.norm(vec)) * (cfg.cutia_speed * 0.5)
                self.hist_cutias_xy[i].append((pos[0].item(), pos[1].item(), 0.3, 0)) # 0 = not carrying
                continue
            
            if cutia["state"] == "searching":
                # Look for tree with dropped pods
                if cutia["target_xy"] is None:
                    best_t = -1
                    best_pods = 0
                    for ti, t in enumerate(self.trees):
                        if t["pods_on_ground"] > 1.0:
                            dist = math.sqrt((pos[0].item()-t["pos"][0])**2 + (pos[1].item()-t["pos"][1])**2)
                            score = t["pods_on_ground"] - dist * 0.1
                            if score > best_pods:
                                best_pods = score
                                best_t = ti
                    
                    if best_t != -1:
                        cutia["target_xy"] = torch.tensor(self.trees[best_t]["pos"], device=self.dev, dtype=torch.float32)
                        cutia["target_tree_idx"] = best_t
                    else:
                        # Wander randomly
                        cutia["target_xy"] = pos + torch.randn(2, device=self.dev) * 20.0
                
                # Move to target
                if cutia["target_xy"] is not None:
                    vec = cutia["target_xy"] - pos
                    dist = torch.norm(vec).item()
                    if dist > 3.0:
                        pos += (vec / dist) * cfg.cutia_speed
                    else:
                        if "target_tree_idx" in cutia and cutia["target_tree_idx"] != -1:
                            t = self.trees[cutia["target_tree_idx"]]
                            if t["pods_on_ground"] >= 1.0:
                                t["pods_on_ground"] -= 1.0
                                cutia["state"] = "carrying"
                                cutia["carrying_pod"] = True
                                # Pick a cache spot far away within territory
                                angle = random.uniform(0, 2*math.pi)
                                r = random.uniform(30, 80)
                                cutia["target_xy"] = cutia["territory"] + torch.tensor([math.cos(angle)*r, math.sin(angle)*r], device=self.dev)
                        cutia["target_tree_idx"] = -1
                        if cutia["state"] != "carrying":
                            cutia["target_xy"] = None # Wander again

            elif cutia["state"] == "carrying":
                vec = cutia["target_xy"] - pos
                dist = torch.norm(vec).item()
                if dist > 3.0:
                    pos += (vec / dist) * (cfg.cutia_speed * 0.8) # Slower carrying
                else:
                    # Bury it!
                    cutia["state"] = "burying"
                    cutia["bury_timer"] = 5
            
            elif cutia["state"] == "burying":
                cutia["bury_timer"] -= 1
                if cutia["bury_timer"] <= 0:
                    self.caches.append({
                        "pos": (pos[0].item() + random.uniform(-2,2), pos[1].item() + random.uniform(-2,2)),
                        "state": "buried", # buried, germinated, dead
                        "growth": 0.0,
                        "frame_buried": frame
                    })
                    self.total_pods_hoarded += 1
                    cutia["state"] = "searching"
                    cutia["carrying_pod"] = False
                    cutia["target_xy"] = None

            # Clamp
            dx = pos[0].item() - cx
            dy = pos[1].item() - cy
            dr = math.sqrt(dx*dx + dy*dy)
            if dr > cfg.clock_radius - 15:
                pos[0] = cx + (dx/dr)*(cfg.clock_radius - 15)
                pos[1] = cy + (dy/dr)*(cfg.clock_radius - 15)
                
            carry_int = 1 if cutia["carrying_pod"] else 0
            self.hist_cutias_xy[i].append((pos[0].item(), pos[1].item(), 1.0, carry_int))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class ScatterHoardingRenderer:

    def __init__(self, cfg: ScatterHoardingConfig, sim: ScatterHoardingSim):
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
            f'style="background-color:#181c14; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        svg.append(
            '<radialGradient id="forestFloor">'
            '<stop offset="0%" stop-color="#2d3326" stop-opacity="0.9"/>'
            '<stop offset="70%" stop-color="#1b1f17" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#181c14" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # Background Ground
        svg.append(f'<rect width="{w}" height="{h}" fill="#181c14"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 45}" fill="url(#forestFloor)"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#cddc39" font-weight="bold">'
            f'ECO-SIM: Jatobá × Agouti    - Zoochory & Scatter-Hoarding Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#9ccc65">'
            f'Late dry-season caching & heavy rain germination sync | RESEX Recanto das Araras</text>'
        )

        # Clock Face
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#8bc34a" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 5)
            ly2 = cy + math.sin(angle) * (R - 5)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#558b2f" stroke-width="2"/>'
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

        draw_arc(6.5, 9.5, R + 10, "#ffa000", "Jatobá Pod Fall", 0.4)
        draw_arc(7.5, 10.5, R + 22, "#ff5722", "Mass Scatter Hoarding (Cutia)", 0.6)
        draw_arc(10.5, 2.5, R + 10, "#00bcd4", "Heavy Rains = Germination", 0.4)

        # Soil Moisture pulse (Syncs with germination)
        bg_colors = ";".join(
            f"rgba(0, 188, 212, {sim._interp(RAIN_GERMINATION_CUE, (f/F)*12) * 0.15:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R-5}" fill="transparent">'
            f'<animate attributeName="fill" values="{bg_colors}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # --- Caches (Buried Seeds & Seedlings) ---
        for c in sim.caches:
            px, py = c["pos"]
            start_f = c["frame_buried"]
            
            # Draw Seed (Brown) appearing when buried
            op_seed = ";".join(
                "0.0" if fi < start_f else ("0.8" if c["state"] == "buried" or fi < start_f+20 else "0.3") 
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" r="2" fill="#5d4037">'
                f'<animate attributeName="opacity" values="{op_seed}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            
            # Draw Seedling (Green) appearing when germinated
            if c["state"] == "germinated" or c["state"] == "dead":
                # For simplicity, if it germinated at some point, we find the frame
                # Approximate germination frame by assuming it happened during peak rain
                # We'll just animate a tiny plant emerging
                g_vals = ";".join(
                    f"{max(0, sim._interp(RAIN_GERMINATION_CUE, (fi/F)*12) * 6.0):.1f}" 
                    if fi > start_f+10 else "0.0" 
                    for fi in range(F)
                )
                col = "#7cb342" if c["state"] == "germinated" else "#795548"
                svg.append(
                    f'<path d="M{px},{py} Q{px-4},{py-4} {px},{py-10} Q{px+4},{py-4} {px},{py}" fill="{col}" opacity="0.85">'
                    f'<animateTransform attributeName="transform" type="scale" values="{g_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'</path>'
                )

        # --- Jatobá Trees ---
        for ti, t in enumerate(sim.trees):
            px, py = t["pos"]
            # Mother Tree Trunk (Thick)
            svg.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="8" fill="#4e342e" stroke="#3e2723" stroke-width="2"/>')
            
            # Pod drop visual (Brown dots clustered around base)
            # Size pulses with JATOBA_FRUIT_DROP
            p_vals = ";".join(
                f"{sim._interp(JATOBA_FRUIT_DROP, (fi/F)*12) * 5 + 0.1:.1f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px+4:.0f}" cy="{py+4:.0f}" fill="#bf360c" opacity="0.9">'
                f'<animate attributeName="r" values="{p_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{px-5:.0f}" cy="{py+2:.0f}" fill="#d84315" opacity="0.9">'
                f'<animate attributeName="r" values="{p_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # --- Cutias (Agoutis) ---
        for i in range(cfg.num_cutias):
            hist = sim.hist_cutias_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            ops = ";".join(str(round(h[2], 2)) for h in hist)
            carrys = ";".join("2.5" if h[3] == 1 else "0" for h in hist) # Dot appears if carrying
            
            # Foraging path (Reddish-brown)
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::3] if h[2] > 0.5]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#ff5722" stroke-width="0.8" stroke-dasharray="2,2" opacity="0.25"/>'
                )
            
            # Cutia Body (Orange-brown ellipse)
            svg.append(
                f'<ellipse rx="4" ry="2.5" fill="#e64a19">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # Pod being carried (in mouth)
            svg.append(
                f'<circle fill="#ffb300" opacity="0.9">'
                f'<animate attributeName="r" values="{carrys}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Clock hand
        hand_x = ";".join(str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        hand_y = ";".join(str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R-10), 1)) for m in sim.hist_month)
        
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#cddc39" stroke-width="2.5" stroke-linecap="round" opacity="0.7">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#181c14" stroke="#cddc39" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#ff5722"/>')


        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 168
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#11140e" rx="8" '
                   f'stroke="#ff5722" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ff5722" font-size="15" font-weight="bold">'
                   f'Scatter-Hoarding Zoochory Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Jatobá pods are large, cement-hard. Few can</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'crack them; the Cutia (Agouti) has adapted</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'teeth. Dropped in dry season, the Cutia eats</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'some and buries the rest as caches across</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#cddc39" font-size="15">'
                   f'the territory. Forgotten seeds, buried and</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#cddc39" font-size="15">'
                   f'scarified, germinate when summer rains arrive.</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 117
        
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#11140e" rx="8" '
                   f'stroke="#ff9800" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ffb300" font-size="15" font-weight="bold">'
                   f'Cache & Germination Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#ff5722" font-size="15">'
                   f'Total Jatobá Seeds Hoarded: {sim.total_pods_hoarded} pods cached</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#8bc34a" font-size="15">'
                   f'Successful Sync Germinations: {sim.total_germinations} new trees</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#00bcd4" font-size="15">'
                   f'Seed-Rain Cue Efficacy: {(sim.total_germinations/max(1,sim.total_pods_hoarded))*100:.1f}% success rate</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'Cutia Population: {cfg.num_cutias} active</text>')
        svg.append('</g>')

        # --- Panel 3: Annual Curves ---
        py3 = py2 + ph2 + 10
        ph3 = 117
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#11140e" rx="8" '
                   f'stroke="#8bc34a" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#9ccc65" font-size="15" font-weight="bold">'
                   f'Phenology & Climate Triggers</text>')

        curves = [
            (JATOBA_FRUIT_DROP, "#ff9800", "Jatobá Pod Fall"),
            (CUTIA_HOARDING_ACTIVITY, "#ff5722", "Cutia Foraging"),
            (RAIN_GERMINATION_CUE, "#00bcd4", "Summer Rains (Onset)"),
            (GERMINATION_SUCCESS, "#8bc34a", "Germination Spike"),
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
            lx = chart_x0 + (ci % 2) * 155
            lyy = legend_y if ci < 2 else legend_y + 16
            svg.append(f'<circle cx="{lx}" cy="{lyy}" r="3.5" fill="{color}"/>')
            svg.append(f'<text font-weight="bold" x="{lx + 6}" y="{lyy + 5}" fill="{color}" font-size="15">'
                       f'{label}</text>')
        svg.append('</g>')

        # --- Panel 4: Context ---
        py4 = py3 + ph3 + 26
        ph4 = 124
        svg.append(f'<g transform="translate({panel_x}, {py4})">')
        svg.append(f'<rect width="{panel_w}" height="{ph4}" fill="#11140e" rx="8" '
                   f'stroke="#558b2f" stroke-width="1" opacity="0.88"/>')
        svg.append(f'<text x="12" y="18" fill="#8bc34a" font-size="15" font-weight="bold">'
                   f'The "Ecological Forgetfulness"</text>')
        svg.append(f'<text font-weight="bold" x="12" y="34" fill="#aed581" font-size="15">'
                   f'Without Agouti, seeds simply drop and rot.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="48" fill="#aed581" font-size="15">'
                   f'This mutualism expands the forest, leveraging</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#aed581" font-size="15">'
                   f'memory lapses to ensure plant reproduction.</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Jatobá ↔ Cutia Clock on {CONFIG.device}...")

    sim = ScatterHoardingSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_pods_hoarded} seeds hoarded, resulting in {sim.total_germinations} germinations.")

    print("Generating SVG...")
    renderer = ScatterHoardingRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_70')
    return svg_content


if __name__ == "__main__":
    main()

