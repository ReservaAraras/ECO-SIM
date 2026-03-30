# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 57: Lobo-guará ↔ Lobeira — Endozoochory Phenological Clock
# INTERVENTION 3/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_57.py — Maned Wolf ↔ Lobeira: Seasonal Endozoochory Clock
Notebook Differentiation:
- Differentiation Focus: Lobo-guará ↔ Lobeira — Endozoochory Phenological Clock emphasizing pesticide drift effects.
- Indicator species: Capim-flecha (Tristachya leiostachya).
- Pollination lens: pollen limitation in fragmented edges.
- Human impact lens: climate warming on water balance.

Models the obligate mutualistic relationship between the Maned Wolf
(Chrysocyon brachyurus) and the Wolf Apple (Solanum lycocarpum, "lobeira").
The wolf is the primary seed disperser via endozoochory: fruits are ingested
whole, seeds pass through the gut (~24h), and are deposited in latrines
(scent-marked defecation sites) across its territory.

As in the project's other evidence-based notebooks, the argument is framed as
an ecological service chain rather than a simple co-occurrence: fruiting pulses
change wolf diet, territorial patrols redistribute seeds, repeated latrine use
creates concentrated seed-rain nodes, and recruitment only occurs where post-
fire and moisture conditions allow establishment.

The radial phenological clock maps:
  • Lobeira fruiting seasonality (bimodal: Feb-Apr and Aug-Oct)
  • Wolf territorial patrol routes & latrine site placement
  • Gut-passage transit time and seed viability
  • Germination success tied to soil conditions and fire regime
  • Wolf diet composition shift (frugivory vs. small prey) by season
  • Territory overlap & genetic dispersal corridors

Scientific References:
  - Dietz, J.M. (1984). "Ecology and social organization of the maned
    wolf." Smithsonian Contributions to Zoology 392.
  - Motta-Junior, J.C. et al. (1996). "Fruits in the diet of the maned
    wolf in southeastern Brazil." Mammalia 60(4).
  - Aragona, M. & Setz, E.Z.F. (2001). "Diet of the maned wolf during
    wet and dry seasons." J. Zoology 254(1).
  - Courtenay, O. et al. (2006). "Maned wolf ecology: patterns of
    territory and ranging behaviour." J. Mammalogy.

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
    from IPython.display import display, HTML  # pyre-ignore[21]
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES
# ===================================================================================================

# Lobeira fruiting intensity — bimodal pattern
# Motta-Junior et al. (1996): two peaks in southeastern Cerrado
LOBEIRA_FRUITING_CURVE = [
    0.40,   # JAN — moderate
    0.70,   # FEV — first peak onset
    0.85,   # MAR — first peak
    0.60,   # ABR — declining
    0.25,   # MAI — transition
    0.10,   # JUN — low
    0.15,   # JUL — rising
    0.55,   # AGO — second peak onset
    0.80,   # SET — second peak
    0.65,   # OUT — declining
    0.30,   # NOV — moderate
    0.25,   # DEZ — low-moderate
]

# Diet composition: % frugivory (remainder = small prey, insects)
# Aragona & Setz (2001): maned wolf is ~50% frugivore in wet season
WOLF_FRUGIVORY_CURVE = [
    0.45,   # JAN
    0.60,   # FEV — high fruit availability
    0.65,   # MAR — peak frugivory
    0.50,   # ABR
    0.30,   # MAI — more hunting
    0.20,   # JUN — mostly prey
    0.25,   # JUL
    0.50,   # AGO — lobeira 2nd peak
    0.60,   # SET
    0.55,   # OUT
    0.35,   # NOV
    0.35,   # DEZ
]

# Wolf territorial activity (crepuscular/nocturnal, most active at dusk)
# Dietz (1984): activity peaks around sunset, reduced midday
WOLF_ACTIVITY_CURVE = [
    0.70,   # JAN — wet season, moderate
    0.65,   # FEV
    0.70,   # MAR
    0.75,   # ABR — longer nights
    0.80,   # MAI — peak territorial patrols
    0.85,   # JUN — long nights, very active
    0.85,   # JUL — peak
    0.80,   # AGO
    0.75,   # SET
    0.70,   # OUT
    0.65,   # NOV
    0.68,   # DEZ
]

# Germination probability (requires fire scarification + moisture)
# Seeds benefit from fire: natural fire regime breaks seed dormancy
GERMINATION_CURVE = [
    0.55,   # JAN — wet + post-fire
    0.50,   # FEV
    0.35,   # MAR
    0.15,   # ABR
    0.05,   # MAI — dry, no germination
    0.02,   # JUN
    0.02,   # JUL
    0.10,   # AGO — fire season starts, scarification
    0.30,   # SET — fire + early rains
    0.50,   # OUT — optimal: post-fire + wet
    0.55,   # NOV — peak germination
    0.55,   # DEZ
]

# Soil moisture (same as notebook_55 for consistency)
SOIL_MOISTURE_CURVE = [0.80, 0.70, 0.85, 0.60, 0.35, 0.15, 0.05, 0.05, 0.25, 0.50, 0.80, 0.90]


@dataclass
class EndozoochoryConfig:
    """Configuration for the Lobo-guará ↔ Lobeira endozoochory clock."""
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360          # 1 full annual cycle
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Clock geometry
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    # Wolf parameters
    num_wolves: int = 4         # low density, solitary
    wolf_speed: float = 3.5
    territory_radius: float = 200.0
    patrol_jitter: float = 0.12
    # Lobeira trees
    num_lobeira: int = 14
    # Endozoochory
    fruit_pick_prob: float = 0.06
    gut_passage_frames: int = 30   # ~24h equivalent
    latrine_deposit_prob: float = 0.7     # chance of defecating at latrine vs random
    max_latrines: int = 25
    max_seeds: int = 180
    germination_delay: int = 45
    # Latrine establishment
    latrine_remark_interval: int = 50     # frames between re-visiting


CONFIG = EndozoochoryConfig()


def scale_right_panels(
    panel_x: float,
    panel_w: float,
    canvas_w: float,
    scale: float = 1.10,
) -> tuple:
    """Enlarge a right-anchored panel by *scale* factor, expanding leftward.

    The right edge (panel_x + panel_w) is kept fixed so the panel grows
    toward the left — harmonious with right-aligned card layouts.

    Args:
        panel_x:   Current x-coordinate of the panel's left edge.
        panel_w:   Current panel width.
        canvas_w:  Total SVG canvas width (used to preserve right margin).
        scale:     Enlargement factor (default 1.10 → +10%).

    Returns:
        (new_panel_x, new_panel_w) as a tuple of ints.
    """
    right_edge = panel_x + panel_w          # fixed anchor (right margin preserved)
    new_panel_w = round(panel_w * scale)
    new_panel_x = right_edge - new_panel_w
    return int(new_panel_x), new_panel_w


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class WolfLobeiraSim:
    """Phenological clock model for Maned Wolf ↔ Lobeira endozoochory."""

    def __init__(self, cfg: EndozoochoryConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # --- Lobeira trees scattered across the clock area ---
        self.lobeira_positions: List[Tuple[float, float]] = []
        self.lobeira_angles: List[float] = []
        for i in range(cfg.num_lobeira):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(60, R - 20)
            lx = cx + math.cos(angle) * r
            ly = cy + math.sin(angle) * r
            self.lobeira_positions.append((lx, ly))
            self.lobeira_angles.append(angle)

        # --- Wolves: each has a territory sector ---
        self.wolves: List[Dict] = []
        for wi in range(cfg.num_wolves):
            sector_angle = (wi / cfg.num_wolves) * 2 * math.pi
            home_x = cx + math.cos(sector_angle) * 110
            home_y = cy + math.sin(sector_angle) * 110
            self.wolves.append({
                "pos": torch.tensor([home_x, home_y], device=self.dev, dtype=torch.float32),
                "home": torch.tensor([home_x, home_y], device=self.dev, dtype=torch.float32),
                "angle": sector_angle + random.uniform(-0.3, 0.3),
                "has_fruit": False,
                "fruit_timer": 0,
                "fruits_eaten": 0,
                "seeds_deposited": 0,
                "sector": sector_angle,
                "last_latrine_visit": -100,
            })

        # --- Latrines (scent-marked defecation sites) ---
        self.latrines: List[Dict] = []
        # Pre-establish a few ancestral latrines
        for wi in range(cfg.num_wolves):
            sa = self.wolves[wi]["sector"]
            for k in range(2):
                lr = random.uniform(80, 160)
                la = sa + random.uniform(-0.4, 0.4)
                lx = cx + math.cos(la) * lr
                ly = cy + math.sin(la) * lr
                self.latrines.append({
                    "pos": (lx, ly),
                    "owner": wi,
                    "visits": 0,
                    "seeds_deposited": 0,
                    "established_frame": 0,
                })

        # History
        self.wolf_histories: List[List[Dict]] = [[] for _ in range(cfg.num_wolves)]
        self.hist_month: List[float] = []

        # Seed events
        self.seed_events: List[Dict] = []
        self.germinated_count = 0
        self.total_seeds = 0
        self.seeds_per_month = [0] * 12

        # Tracking curves
        self.hist_fruiting: List[float] = []
        self.hist_frugivory: List[float] = []
        self.hist_activity: List[float] = []

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

        # Environmental state
        fruiting = self._interp(LOBEIRA_FRUITING_CURVE, month_frac)
        frugivory = self._interp(WOLF_FRUGIVORY_CURVE, month_frac)
        activity = self._interp(WOLF_ACTIVITY_CURVE, month_frac)
        moisture = self._interp(SOIL_MOISTURE_CURVE, month_frac)
        germ_prob = self._interp(GERMINATION_CURVE, month_frac)

        self.hist_month.append(month_frac)
        self.hist_fruiting.append(fruiting)
        self.hist_frugivory.append(frugivory)
        self.hist_activity.append(activity)

        # --- Wolf movement: territorial patrol ---
        for wi, wolf in enumerate(self.wolves):
            # Crepuscular random walk biased toward territory boundaries
            wolf["angle"] += random.uniform(-cfg.patrol_jitter, cfg.patrol_jitter)

            # Speed modulation: more active during high-activity months
            spd = cfg.wolf_speed * (0.5 + activity * 0.8)
            dx = math.cos(wolf["angle"]) * spd
            dy = math.sin(wolf["angle"]) * spd
            wolf["pos"][0] += dx
            wolf["pos"][1] += dy

            # Territory anchoring: pull back toward home sector
            to_home = wolf["home"] - wolf["pos"]
            dist_home = torch.norm(to_home).item()
            if dist_home > cfg.territory_radius:
                wolf["pos"] += to_home * 0.08
                wolf["angle"] = math.atan2(to_home[1].item(), to_home[0].item()) + random.uniform(-0.3, 0.3)

            # Boundary clamping
            wolf["pos"][0] = torch.clamp(wolf["pos"][0], cx - cfg.clock_radius + 10, cx + cfg.clock_radius - 10)
            wolf["pos"][1] = torch.clamp(wolf["pos"][1], cy - cfg.clock_radius + 10, cy + cfg.clock_radius - 10)

            wx, wy = wolf["pos"][0].item(), wolf["pos"][1].item()
            self.wolf_histories[wi].append({"x": wx, "y": wy})

            # --- Fruit pickup (near lobeira during fruiting) ---
            if not wolf["has_fruit"] and fruiting > 0.15:
                for lx, ly in self.lobeira_positions:
                    d = math.sqrt((wx - lx) ** 2 + (wy - ly) ** 2)
                    if d < 30 and random.random() < cfg.fruit_pick_prob * fruiting * frugivory * 2:
                        wolf["has_fruit"] = True
                        wolf["fruit_timer"] = 0
                        wolf["fruits_eaten"] += 1
                        break

            # --- Gut passage & seed deposition ---
            if wolf["has_fruit"]:
                wolf["fruit_timer"] += 1
                if wolf["fruit_timer"] >= cfg.gut_passage_frames:
                    # Deposit seeds: prefer latrines
                    deposited = False
                    if random.random() < cfg.latrine_deposit_prob:
                        # Find nearest owned latrine
                        best_lat = None
                        best_d = float('inf')
                        for li, lat in enumerate(self.latrines):
                            if lat["owner"] == wi:
                                ld = math.sqrt((wx - lat["pos"][0]) ** 2 + (wy - lat["pos"][1]) ** 2)
                                if ld < best_d:
                                    best_d = ld
                                    best_lat = li
                        if best_lat is not None and best_d < 80:
                            sx, sy = self.latrines[best_lat]["pos"]
                            self.latrines[best_lat]["visits"] += 1
                            self.latrines[best_lat]["seeds_deposited"] += 1
                            deposited = True
                        else:
                            # Create a new latrine at current position
                            if len(self.latrines) < cfg.max_latrines:
                                self.latrines.append({
                                    "pos": (wx, wy),
                                    "owner": wi,
                                    "visits": 1,
                                    "seeds_deposited": 1,
                                    "established_frame": frame,
                                })
                            sx, sy = wx, wy
                            deposited = True

                    if not deposited:
                        sx, sy = wx + random.uniform(-10, 10), wy + random.uniform(-10, 10)

                    # Record seed event
                    if len(self.seed_events) < cfg.max_seeds:
                        will_germ = random.random() < germ_prob
                        self.seed_events.append({
                            "pos": (sx, sy),
                            "frame": frame,
                            "germinated": will_germ,
                            "sprout_frame": frame + cfg.germination_delay if will_germ else -1,
                            "wolf": wi,
                            "at_latrine": deposited,
                        })
                        self.total_seeds += 1
                        self.seeds_per_month[m_idx] += 1
                        if will_germ:
                            self.germinated_count += 1
                        wolf["seeds_deposited"] += 1

                    wolf["has_fruit"] = False
                    wolf["fruit_timer"] = 0


# ===================================================================================================
# 3. VISUALIZATION — ENDOZOOCHORY PHENOLOGICAL CLOCK
# ===================================================================================================

class EndozoochoryRenderer:
    """Renders the maned wolf ↔ lobeira endozoochory clock as animated SVG."""

    def __init__(self, cfg: EndozoochoryConfig, sim: WolfLobeiraSim):
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
            f'style="background-color:#0b0d12; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # --- Defs ---
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="clockBg57">'
            '<stop offset="0%" stop-color="#1c1510" stop-opacity="0.85"/>'
            '<stop offset="75%" stop-color="#120e0a" stop-opacity="0.35"/>'
            '<stop offset="100%" stop-color="#0b0d12" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="latrineGlow">'
            '<stop offset="0%" stop-color="#d84315" stop-opacity="0.5"/>'
            '<stop offset="100%" stop-color="#bf360c" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="lobeiraGlow">'
            '<stop offset="0%" stop-color="#7cb342" stop-opacity="0.5"/>'
            '<stop offset="100%" stop-color="#33691e" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<pattern id="dotGrid57" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="1.2" fill="#3e2723" opacity="0.18"/>'
            '</pattern>'
        )
        svg.append('</defs>')

        # --- Background ---
        svg.append(f'<rect width="{w}" height="{h}" fill="#0b0d12"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid57)"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#clockBg57)"/>')

        # --- Title ---
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#d84315" font-weight="bold">'
            f'ECO-SIM: Maned Wolf × Wolf Apple    - Endozoochory Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#b0bec5">'
            f'Seed dispersal via gut passage: Chrysocyon brachyurus ↔ Solanum lycocarpum'
            f'</text>'
        )

        # --- Clock face: months ---
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 48)
            ty = cy + math.sin(angle) * (R + 48)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#bcaaa4" '
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

        # --- Concentric guide rings ---
        for r_ring in [70, 120, 170]:
            svg.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_ring}" fill="none" '
                f'stroke="#3e2723" stroke-width="0.5" stroke-dasharray="3,6"/>'
            )

        # --- Season arcs ---
        def draw_arc(start_m, end_m, radius, color, label, opacity=0.28):
            a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
            a2 = (end_m / 12) * 2 * math.pi - math.pi / 2
            x1 = cx + math.cos(a1) * radius
            y1 = cy + math.sin(a1) * radius
            x2 = cx + math.cos(a2) * radius
            y2 = cy + math.sin(a2) * radius
            span = (end_m - start_m) % 12
            large = 1 if span > 6 else 0
            d = f"M {x1:.0f} {y1:.0f} A {radius} {radius} 0 {large} 1 {x2:.0f} {y2:.0f}"
            svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="16" '
                       f'stroke-linecap="round" opacity="{opacity}"/>')
            mid = ((start_m + end_m) / 2 / 12) * 2 * math.pi - math.pi / 2
            lx = cx + math.cos(mid) * (radius + 20)
            ly = cy + math.sin(mid) * (radius + 20)
            svg.append(f'<text font-weight="bold" x="{lx:.0f}" y="{ly:.0f}" font-size="15" fill="{color}" '
                       f'text-anchor="middle" opacity="0.85">{label}</text>')

        draw_arc(1, 4.5, 200, "#7cb342", "Lobeira Peak 1", 0.30)
        draw_arc(7, 10, 200, "#8bc34a", "Lobeira Peak 2", 0.30)
        draw_arc(7.5, 10.5, 160, "#ff8a65", "Fire season", 0.20)
        draw_arc(9.5, 14, 130, "#42a5f5", "Germination window", 0.22)

        # --- Territory sectors (faint wedges) ---
        wolf_colors = ["#d84315", "#e65100", "#bf360c", "#a1887f"]
        for wi in range(cfg.num_wolves):
            sa = self.sim.wolves[wi]["sector"]
            span = 2 * math.pi / cfg.num_wolves
            a1 = sa - span / 2
            a2 = sa + span / 2
            x1 = cx + math.cos(a1) * R
            y1 = cy + math.sin(a1) * R
            svg.append(
                f'<line x1="{cx}" y1="{cy}" x2="{x1:.0f}" y2="{y1:.0f}" '
                f'stroke="{wolf_colors[wi]}" stroke-width="0.8" stroke-dasharray="5,8" opacity="0.25"/>'
            )

        # --- Lobeira trees ---
        for lx, ly in sim.lobeira_positions:
            r_vals = ";".join(
                f"{6 + sim._interp(LOBEIRA_FRUITING_CURVE, (fi / F) * 12) * 10:.1f}" for fi in range(F)
            )
            op_vals = ";".join(
                f"{0.3 + sim._interp(LOBEIRA_FRUITING_CURVE, (fi / F) * 12) * 0.6:.2f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{lx:.0f}" cy="{ly:.0f}" fill="#7cb342" stroke="#33691e" stroke-width="1.5">'
                f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Latrines (scent-marked sites) ---
        for lat in sim.latrines:
            lx, ly = lat["pos"]
            svg.append(
                f'<circle cx="{lx:.0f}" cy="{ly:.0f}" r="4" fill="#d84315" opacity="0.6">'
                f'<animate attributeName="r" values="3;5;3" dur="3s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{lx:.0f}" cy="{ly:.0f}" r="12" fill="url(#latrineGlow)" opacity="0.4"/>'
            )

        # --- Seed dispersal events (sprouting animations) ---
        for se in sim.seed_events:
            sx, sy = se["pos"]
            sf = se["frame"]
            if se["germinated"]:
                spf = se["sprout_frame"]
                ops_s = ";".join(
                    "0.0" if fi < sf else ("0.3" if fi < spf else "0.9") for fi in range(F)
                )
                rv = ";".join(
                    "0" if fi < spf else f"{min(7.0, (fi - spf) / 15.0 * 7.0):.1f}" for fi in range(F)
                )
                svg.append(
                    f'<circle cx="{sx:.0f}" cy="{sy:.0f}" fill="#aed581" opacity="0.0">'
                    f'<animate attributeName="r" values="{rv}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'<animate attributeName="opacity" values="{ops_s}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'</circle>'
                )
            else:
                ops_u = ";".join(
                    "0.0" if fi < sf else (f"{max(0.0, 0.4 - (fi-sf)/40.0):.2f}") for fi in range(F)
                )
                svg.append(
                    f'<circle cx="{sx:.0f}" cy="{sy:.0f}" r="1.8" fill="#5d4037" opacity="0.0">'
                    f'<animate attributeName="opacity" values="{ops_u}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'</circle>'
                )

        # --- Sweeping clock hand showing current month ---
        hand_x_vals = ";".join(
            f"{cx + math.cos((m / 12) * 2 * math.pi - math.pi / 2) * (R - 20):.1f}"
            for m in sim.hist_month
        )
        hand_y_vals = ";".join(
            f"{cy + math.sin((m / 12) * 2 * math.pi - math.pi / 2) * (R - 20):.1f}"
            for m in sim.hist_month
        )
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 20}" '
            f'stroke="#d84315" stroke-width="2.5" stroke-linecap="round" opacity="0.8">'
            f'<animate attributeName="x2" values="{hand_x_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        # --- Wolf particles ---
        for wi in range(cfg.num_wolves):
            wx_vals = ";".join(f"{h['x']:.1f}" for h in sim.wolf_histories[wi])
            wy_vals = ";".join(f"{h['y']:.1f}" for h in sim.wolf_histories[wi])
            svg.append(
                f'<circle r="5" fill="{wolf_colors[wi]}" opacity="0.9">'
                f'<animate attributeName="cx" values="{wx_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{wy_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle r="10" fill="{wolf_colors[wi]}" opacity="0.15">'
                f'<animate attributeName="cx" values="{wx_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{wy_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Centre hub ---
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="15" fill="#1c1510" stroke="#d84315" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#d84315"/>')

        # --- Fruiting intensity radial bars ---
        for i in range(12):
            a = (i / 12) * 2 * math.pi - math.pi / 2
            val = LOBEIRA_FRUITING_CURVE[i]
            bar_len = val * 55
            bx1 = cx + math.cos(a) * 25
            by1 = cy + math.sin(a) * 25
            bx2 = cx + math.cos(a) * (25 + bar_len)
            by2 = cy + math.sin(a) * (25 + bar_len)
            svg.append(
                f'<line x1="{bx1:.0f}" y1="{by1:.0f}" x2="{bx2:.0f}" y2="{by2:.0f}" '
                f'stroke="#7cb342" stroke-width="5" stroke-linecap="round" opacity="{0.2 + val*0.8:.2f}"/>'
            )


        panel_x = w - 390
        panel_w = 370
        # Enlarge right-side cards by 10 %, expanding leftward.
        panel_x, panel_w = scale_right_panels(panel_x, panel_w, w)

        # --- Panel 1: Simulation Logic ---
        py1 = 20
        ph1 = 216
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1a1a2e" rx="8" '
                   f'stroke="#ff8a65" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ff8a65" font-size="15" font-weight="bold">'
                   f'Endozoochory Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Fruit peaks raise wolf frugivory and route fidelity.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'Reused latrines turn patrol paths into seed-rain nodes.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="72" fill="#ccc" font-size="15">'
                   f'Gut passage moves seeds away from parent crowns.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="86" fill="#ccc" font-size="15">'
                   f'Recruitment succeeds once post-fire soils get moisture.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="102" fill="#7cb342" font-size="15">'
                   f'Green circles = established recruits after dispersal</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 160
        germ_pct = (sim.germinated_count / max(1, sim.total_seeds)) * 100
        total_fruits = sum(w["fruits_eaten"] for w in sim.wolves)
        lat_with_seeds = sum(1 for lat in sim.latrines if lat["seeds_deposited"] > 0)

        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1a1a2e" rx="8" '
                   f'stroke="#4fc3f7" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#4fc3f7" font-size="15" font-weight="bold">'
                   f'Dispersal Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#e0e0e0" font-size="15">'
                   f'Fruits consumed: {total_fruits} ({cfg.num_wolves} wolves)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#ff8a65" font-size="15">'
                   f'Seeds deposited: {sim.total_seeds}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#aed581" font-size="15">'
                   f'Germinated: {sim.germinated_count} ({germ_pct:.0f}%)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#bcaaa4" font-size="15">'
                   f'Active latrines: {lat_with_seeds} / {len(sim.latrines)}</text>')
        best_m = sim.seeds_per_month.index(max(sim.seeds_per_month))
        svg.append(f'<text font-weight="bold" x="12" y="116" fill="#90caf9" font-size="15">'
                   f'Peak dispersal month: {months[best_m]}</text>')
        co2 = sim.germinated_count * 15.0  # ~15 kg CO₂/yr per lobeira
        svg.append(f'<text font-weight="bold" x="12" y="134" fill="#a5d6a7" font-size="15">'
                   f'CO₂ potential: {co2:,.0f} kg/yr (~15 kg/tree)</text>')
        svg.append('</g>')

        # --- Panel 3: Annual curves ---
        py3 = py2 + ph2 + 10
        ph3 = 186
        chart_w = panel_w - 32
        chart_h = 52
        chart_x0 = 16
        chart_y0 = 28

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1a1a2e" rx="8" '
                   f'stroke="#7e57c2" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#7e57c2" font-size="15" font-weight="bold">'
                   f'Annual Curves</text>')

        curves = [
            (LOBEIRA_FRUITING_CURVE, "#7cb342", "Lobeira fruit"),
            (WOLF_FRUGIVORY_CURVE, "#d84315", "Wolf frugivory"),
            (WOLF_ACTIVITY_CURVE, "#ffab00", "Wolf activity"),
            (GERMINATION_CURVE, "#42a5f5", "Germination"),
        ]
        for curve_data, color, label in curves:
            pts = []
            for mi in range(12):
                px_c = chart_x0 + (mi / 11) * chart_w
                py_c = chart_y0 + chart_h - curve_data[mi] * chart_h
                pts.append(f"{px_c:.0f},{py_c:.0f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="1.8" opacity="0.75"/>')

        legend_y = chart_y0 + chart_h + 12
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
    """Run maned wolf ↔ lobeira endozoochory clock simulation."""
    print(f" — Lobo-guará ↔ Lobeira Endozoochory Clock on {CONFIG.device}...")

    sim = WolfLobeiraSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    germ_pct = (sim.germinated_count / max(1, sim.total_seeds)) * 100
    total_fruits = sum(w["fruits_eaten"] for w in sim.wolves)
    print(f"Done: {total_fruits} fruits eaten, {sim.total_seeds} seeds deposited, "
          f"{sim.germinated_count} germinated ({germ_pct:.0f}%)")
    print(f"Latrines: {len(sim.latrines)}, seeds by month: {sim.seeds_per_month}")

    print("Generating SVG...")
    renderer = EndozoochoryRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_57')
    return svg_content


if __name__ == "__main__":
    main()
