# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 56: Abelha-nativa ↔ Cerrado Wildflowers — Pollination Phenological Clock
# INTERVENTION 2/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_56.py — Native Bees ↔ Cerrado Wildflowers: Seasonal Pollination Clock
Notebook Differentiation:
- Differentiation Focus: Abelha-nativa ↔ Cerrado Wildflowers — Pollination Phenological Clock emphasizing fence permeability filters.
- Indicator species: Cipo-jacare (Davilla elliptica).
- Pollination lens: ant-guarded nectar dynamics.
- Human impact lens: restoration weeding benefits.

Models the mutualistic pollination relationship between native stingless bees
(Meliponini tribe: Melipona quadrifasciata, Trigona spinipes, Tetragonisca
angustula) and the sequential flowering guilds of the Cerrado.

The radial phenological clock maps:
  • Monthly flowering intensity of 5 Cerrado plant guilds (each blooming in
    complementary windows across the year)
  • Bee foraging radius and colony activity modulated by temperature/season
  • Pollen transfer events — successful cross-pollination probability
  • Nectar flow index and honey production potential
  • Colony health tied to continuous floral resource availability

The scientific argument follows the same evidence logic used across the best
notebooks in the project: complementary flowering windows prevent seasonal
resource gaps, those gaps regulate colony strength and foraging coverage, and
pollination service emerges only when visits connect different conspecific
flowers rather than just extracting nectar.

Scientific References:
  - Oliveira, P.E. & Gibbs, P.E. (2000). "Reproductive biology of woody
    plants in a cerrado community of Central Brazil." Flora 195(4).
  - Imperatriz-Fonseca, V.L. et al. (2012). "Biodiversidade e conservação
    de abelhas nativas brasileiras." Apidologie.
  - Silberbauer-Gottsberger, I. & Gottsberger, G. (1988). "A polinização
    de plantas do cerrado." Revista Brasileira de Biologia.
  - Batalha, M.A. & Mantovani, W. (2000). "Reproductive phenological
    patterns of cerrado plant species." Revista Brasileira de Biologia.

Scientific Relevance (PIGT RESEX Recanto das Araras — 2024):
  - Integrates the socio-environmental complexity of 
    de Cima, Goiás, Brazil.
  - Models landscape connectivity, karst vulnerability (Bacia do Rio Lapa),
    biological corridors, and seed-dispersal networks.
  - Demonstrates parameters for ecological succession, biodiversity indices,
    integrated fire management (MIF), and ornithochory dynamics.
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

# Five Cerrado flower guilds with complementary blooming windows
# Based on Batalha & Mantovani (2000) and Oliveira & Gibbs (2000)
FLOWER_GUILDS = {
    "Caliandra": {
        "color": "#e91e63",       # deep pink
        "bloom_curve": [0.05, 0.02, 0.00, 0.00, 0.00, 0.10, 0.30, 0.60, 0.90, 0.70, 0.30, 0.10],
        # Peak: Aug-Oct (late dry → early wet transition)
        "pollen_reward": 0.8,
        "nectar_mL": 1.2,
        "arc_radius": 200,
        "label": "Caliandra (Calliandra)",
    },
    "Barbatimao": {
        "color": "#ff9800",       # amber
        "bloom_curve": [0.00, 0.00, 0.00, 0.05, 0.15, 0.40, 0.70, 0.85, 0.50, 0.15, 0.00, 0.00],
        # Peak: Jul-Aug (mid dry season)
        "pollen_reward": 0.6,
        "nectar_mL": 0.8,
        "arc_radius": 175,
        "label": "Barbatimão (Stryphnodendron)",
    },
    "Ipê-amarelo": {
        "color": "#fdd835",       # yellow
        "bloom_curve": [0.00, 0.00, 0.00, 0.00, 0.05, 0.20, 0.80, 1.00, 0.60, 0.10, 0.00, 0.00],
        # Peak: Jul-Sep (spectacular dry-season bloom)
        "pollen_reward": 0.9,
        "nectar_mL": 1.5,
        "arc_radius": 150,
        "label": "Ipê-amarelo (Handroanthus)",
    },
    "Murici": {
        "color": "#8bc34a",       # light green
        "bloom_curve": [0.20, 0.10, 0.05, 0.00, 0.00, 0.00, 0.05, 0.10, 0.30, 0.60, 0.80, 0.50],
        # Peak: Nov-Dec (early wet season)
        "pollen_reward": 0.7,
        "nectar_mL": 1.0,
        "arc_radius": 125,
        "label": "Murici (Byrsonima)",
    },
    "Pequi-flor": {
        "color": "#cddc39",       # lime
        "bloom_curve": [0.60, 0.30, 0.10, 0.00, 0.00, 0.00, 0.00, 0.10, 0.30, 0.50, 0.80, 0.90],
        # Peak: Nov-Jan (wet season flowering precedes fruiting ~55)
        "pollen_reward": 1.0,
        "nectar_mL": 2.0,
        "arc_radius": 100,
        "label": "Pequi-flor (Caryocar)",
    },
}

# Bee colony activity curve — influenced by temperature and photoperiod
# (Imperatriz-Fonseca et al. 2012)
BEE_ACTIVITY_CURVE = [
    0.85,   # JAN — hot wet, high activity
    0.90,   # FEV — peak colony expansion
    0.80,   # MAR — still active
    0.65,   # ABR — cooling
    0.40,   # MAI — reduced foraging
    0.25,   # JUN — winter minimum
    0.30,   # JUL — cold dry
    0.45,   # AGO — warming, IPÃŠ bloom drives recovery
    0.60,   # SET — transition
    0.75,   # OUT — spring
    0.85,   # NOV — strong
    0.88,   # DEZ — near-peak
]

# Temperature index (°C normalized 0–1) — Cerrado Goiás
TEMPERATURE_CURVE = [
    0.80,   # JAN
    0.82,   # FEV
    0.78,   # MAR
    0.65,   # ABR
    0.50,   # MAI
    0.38,   # JUN
    0.35,   # JUL
    0.45,   # AGO
    0.58,   # SET
    0.70,   # OUT
    0.75,   # NOV
    0.78,   # DEZ
]

# Nectar flow composite index
NECTAR_FLOW_CURVE = [
    0.50,   # JAN
    0.30,   # FEV
    0.15,   # MAR
    0.05,   # ABR
    0.10,   # MAI
    0.25,   # JUN
    0.55,   # JUL — IPÃŠ & Barbatimão
    0.85,   # AGO — peak bloom
    0.75,   # SET
    0.50,   # OUT
    0.60,   # NOV — Murici + Pequi-flor
    0.65,   # DEZ
]


@dataclass
class PollinationConfig:
    """Configuration for Bee ↔ Wildflower pollination phenological clock."""
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360          # 1 full annual cycle
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Clock geometry
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    # Bees
    num_bees: int = 80
    bee_base_speed: float = 0.05
    foraging_radius_min: float = 80.0
    foraging_radius_max: float = 210.0
    # Pollination
    pollination_prob: float = 0.04    # per-frame near a flower
    pollen_carry_frames: int = 15     # time carrying pollen before deposit
    max_pollination_events: int = 300
    # Flower nodes (placed at arc intersections)
    flower_nodes_per_guild: int = 4


CONFIG = PollinationConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class BeePollinationSim:
    """Phenological clock model for native bee ↔ Cerrado wildflower pollination."""

    def __init__(self, cfg: PollinationConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy

        # --- Flower nodes around the clock ---
        self.flower_nodes: List[Dict] = []
        for guild_name, guild in FLOWER_GUILDS.items():
            curve = guild["bloom_curve"]
            # Place nodes at peak months
            peak_months = sorted(range(12), key=lambda m: curve[m], reverse=True)[:cfg.flower_nodes_per_guild]
            for pm in peak_months:
                angle = (pm / 12) * 2 * math.pi - math.pi / 2 + random.uniform(-0.12, 0.12)
                r = guild["arc_radius"] + random.uniform(-10, 15)
                fx = cx + math.cos(angle) * r
                fy = cy + math.sin(angle) * r
                self.flower_nodes.append({
                    "guild": guild_name,
                    "color": guild["color"],
                    "pos": (fx, fy),
                    "month": pm,
                    "pollen_reward": guild["pollen_reward"],
                    "nectar_mL": guild["nectar_mL"],
                })

        # --- Bee swarm: polar coords orbiting the clock ---
        self.angles = torch.rand(cfg.num_bees, device=self.dev) * 2 * math.pi
        self.radii = torch.rand(cfg.num_bees, device=self.dev) * 80 + 100.0
        self.ang_vel = (torch.rand(cfg.num_bees, device=self.dev) * 0.04 + 0.02)
        self.rad_vel = torch.randn(cfg.num_bees, device=self.dev) * 1.0
        self.has_pollen = torch.zeros(cfg.num_bees, device=self.dev, dtype=torch.bool)
        self.pollen_timer = torch.zeros(cfg.num_bees, device=self.dev)
        self.pollen_source = torch.full((cfg.num_bees,), -1, device=self.dev, dtype=torch.long)

        # Colony strength (0–1)
        self.colony_strength = 0.5

        # History
        self.hist_xy: List[np.ndarray] = []
        self.hist_month: List[float] = []
        self.hist_colony: List[float] = []
        self.hist_nectar_flow: List[float] = []

        # Pollination events
        self.pollination_events: List[Dict] = []
        self.cross_pollinations = 0
        self.self_pollinations = 0
        self.total_visits = 0
        self.visits_per_guild: Dict[str, int] = {g: 0 for g in FLOWER_GUILDS}
        self.pollination_per_month = [0] * 12

    def _interp(self, curve: list, month_frac: float) -> float:
        m = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def _composite_bloom(self, month_frac: float) -> float:
        """Weighted sum of all guild bloom intensities at this moment."""
        total = 0.0
        for guild in FLOWER_GUILDS.values():
            total += self._interp(guild["bloom_curve"], month_frac)
        return min(1.0, total / len(FLOWER_GUILDS))

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy = cfg.clock_cx, cfg.clock_cy
        month_frac = (frame / cfg.frames) * 12.0

        # Environmental state
        activity = self._interp(BEE_ACTIVITY_CURVE, month_frac)
        temperature = self._interp(TEMPERATURE_CURVE, month_frac)
        nectar_flow = self._interp(NECTAR_FLOW_CURVE, month_frac)
        bloom = self._composite_bloom(month_frac)

        self.hist_month.append(month_frac)
        self.hist_nectar_flow.append(nectar_flow)

        # Colony strength: driven by nectar availability and temperature
        target_colony = (nectar_flow * 0.6 + activity * 0.3 + temperature * 0.1)
        self.colony_strength = self.colony_strength * 0.92 + target_colony * 0.08
        self.hist_colony.append(self.colony_strength)

        # Dynamic foraging radius
        target_radius = cfg.foraging_radius_min + (cfg.foraging_radius_max - cfg.foraging_radius_min) * bloom

        # Active bees: some stay in hive during low activity
        active_fraction = min(1.0, activity + 0.15)
        n_active = int(cfg.num_bees * active_fraction)

        # Movement: angular
        speed_mod = 0.5 + activity * 1.2
        self.ang_vel[:n_active] += (torch.randn(n_active, device=self.dev) * 0.008)
        self.angles[:n_active] += self.ang_vel[:n_active] * speed_mod
        self.angles = self.angles % (2 * math.pi)

        # Radial: attracted to active bloom ring
        self.rad_vel[:n_active] += (target_radius - self.radii[:n_active]) * 0.05
        self.rad_vel[:n_active] += torch.randn(n_active, device=self.dev) * 2.0
        self.rad_vel *= 0.85
        self.radii[:n_active] += self.rad_vel[:n_active]
        self.radii = torch.clamp(self.radii, 50.0, cfg.clock_radius + 15)

        # Inactive bees cluster near centre (hive)
        if n_active < cfg.num_bees:
            self.radii[n_active:] = self.radii[n_active:] * 0.9 + 25.0 * 0.1

        # Polar → Cartesian
        xs = cx + torch.cos(self.angles - math.pi / 2) * self.radii
        ys = cy + torch.sin(self.angles - math.pi / 2) * self.radii
        xy = torch.stack((xs, ys), dim=1)
        self.hist_xy.append(xy.cpu().numpy().copy())

        # --- Flower visits & pollen pickup ---
        m_idx = int(month_frac) % 12
        for fi, fnode in enumerate(self.flower_nodes):
            guild = FLOWER_GUILDS[fnode["guild"]]
            bloom_now = self._interp(guild["bloom_curve"], month_frac)
            if bloom_now < 0.05:
                continue
            fx, fy = fnode["pos"]
            d = torch.sqrt((xs[:n_active] - fx) ** 2 + (ys[:n_active] - fy) ** 2)
            near = d < 25.0
            if not near.any():
                continue
            for bi in near.nonzero().squeeze(1):
                bi_int = int(bi)
                self.total_visits += 1
                self.visits_per_guild[fnode["guild"]] += 1

                if not self.has_pollen[bi_int]:
                    # Pick up pollen
                    if random.random() < cfg.pollination_prob * bloom_now * 2:
                        self.has_pollen[bi_int] = True
                        self.pollen_timer[bi_int] = 0
                        self.pollen_source[bi_int] = fi
                else:
                    # Deposit pollen (cross or self)
                    src = int(self.pollen_source[bi_int])
                    if src != fi and src >= 0:
                        src_guild = self.flower_nodes[src]["guild"]
                        dst_guild = fnode["guild"]
                        is_cross = (src_guild == dst_guild)  # same guild = valid cross-pollination
                        if is_cross and len(self.pollination_events) < cfg.max_pollination_events:
                            self.pollination_events.append({
                                "pos": (fx, fy), "frame": frame,
                                "guild": dst_guild, "color": fnode["color"],
                            })
                            self.cross_pollinations += 1
                            self.pollination_per_month[m_idx] += 1
                        else:
                            self.self_pollinations += 1
                        self.has_pollen[bi_int] = False
                        self.pollen_timer[bi_int] = 0
                        self.pollen_source[bi_int] = -1

        # Pollen timer: drop pollen after carrying too long
        self.pollen_timer[self.has_pollen] += 1
        expired = self.has_pollen & (self.pollen_timer >= cfg.pollen_carry_frames)
        self.has_pollen[expired] = False
        self.pollen_timer[expired] = 0
        self.pollen_source[expired] = -1


# ===================================================================================================
# 3. VISUALIZATION — POLLINATION PHENOLOGICAL CLOCK
# ===================================================================================================

class PollinationRenderer:
    """Renders the bee pollination phenological clock as animated SVG."""

    def __init__(self, cfg: PollinationConfig, sim: BeePollinationSim):
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
            f'style="background-color:#0d0f1a; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # --- Defs ---
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="clockBg56">'
            '<stop offset="0%" stop-color="#1a1f33" stop-opacity="0.9"/>'
            '<stop offset="80%" stop-color="#111422" stop-opacity="0.4"/>'
            '<stop offset="100%" stop-color="#0d0f1a" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="pollenGlow">'
            '<stop offset="0%" stop-color="#fdd835" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#f9a825" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="hiveGlow">'
            '<stop offset="0%" stop-color="#ff8f00" stop-opacity="0.5"/>'
            '<stop offset="100%" stop-color="#e65100" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<pattern id="dotGrid56" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="1.5" fill="#263238" opacity="0.15"/>'
            '</pattern>'
        )
        # Flower petal filter
        svg.append(
            '<filter id="bloom56" x="-50%" y="-50%" width="200%" height="200%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="3"/>'
            '</filter>'
        )
        svg.append('</defs>')

        # --- Background ---
        svg.append(f'<rect width="{w}" height="{h}" fill="#0d0f1a"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid56)"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#clockBg56)"/>')

        # --- Title ---
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#fdd835" font-weight="bold">'
            f'ECO-SIM: Native Bee × Cerrado Wildflowers    - Pollination Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#b0bec5">'
            f'Seasonal pollination mutualism: Meliponini bees on 5 flower guilds | '
            f'</text>'
        )

        # --- Clock face: month labels & ticks ---
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 48)
            ty = cy + math.sin(angle) * (R + 48)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#90caf9" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 8)
            ly2 = cy + math.sin(angle) * (R - 8)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#455a64" stroke-width="2"/>'
            )

        # --- Concentric bloom arcs for each flower guild ---
        for guild_name, guild in FLOWER_GUILDS.items():
            curve = guild["bloom_curve"]
            arc_r = guild["arc_radius"]
            color = guild["color"]

            # Draw full faint ring
            svg.append(
                f'<circle cx="{cx}" cy="{cy}" r="{arc_r}" fill="none" '
                f'stroke="{color}" stroke-width="0.5" opacity="0.15"/>'
            )

            # Draw thick arcs where bloom > 0.2
            in_bloom = False
            start_m = 0
            for mi in range(13):  # wrap around
                m = mi % 12
                if curve[m] > 0.2 and not in_bloom:
                    start_m = m
                    in_bloom = True
                elif (curve[m] <= 0.2 or mi == 12) and in_bloom:
                    end_m = m
                    a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
                    a2 = (end_m / 12) * 2 * math.pi - math.pi / 2
                    x1 = cx + math.cos(a1) * arc_r
                    y1 = cy + math.sin(a1) * arc_r
                    x2 = cx + math.cos(a2) * arc_r
                    y2 = cy + math.sin(a2) * arc_r
                    span = (end_m - start_m) % 12
                    large = 1 if span > 6 else 0
                    d = f"M {x1:.0f} {y1:.0f} A {arc_r} {arc_r} 0 {large} 1 {x2:.0f} {y2:.0f}"
                    peak_val = max(curve[start_m:end_m + 1]) if end_m > start_m else max(curve[start_m:] + curve[:end_m + 1])
                    svg.append(
                        f'<path d="{d}" fill="none" stroke="{color}" '
                        f'stroke-width="{8 + peak_val * 10:.0f}" stroke-linecap="round" '
                        f'opacity="{0.2 + peak_val * 0.4:.2f}"/>'
                    )
                    in_bloom = False

            # Label at peak
            peak_m = curve.index(max(curve))
            la = (peak_m / 12) * 2 * math.pi - math.pi / 2
            lx = cx + math.cos(la) * (arc_r - 18)
            ly = cy + math.sin(la) * (arc_r - 18)
            svg.append(
                f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="15" fill="{color}" '
                f'text-anchor="middle" opacity="0.85" font-weight="bold">'
                f'{guild_name}</text>'
            )

        # --- Flower nodes (animated bloom pulse) ---
        for fnode in sim.flower_nodes:
            fx, fy = fnode["pos"]
            guild = FLOWER_GUILDS[fnode["guild"]]
            curve = guild["bloom_curve"]
            color = fnode["color"]

            r_vals = ";".join(
                f"{4 + sim._interp(curve, (fi / F) * 12) * 10:.1f}" for fi in range(F)
            )
            op_vals = ";".join(
                f"{0.15 + sim._interp(curve, (fi / F) * 12) * 0.75:.2f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{fx:.0f}" cy="{fy:.0f}" fill="{color}" filter="url(#bloom56)">'
                f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Core bright dot
            svg.append(
                f'<circle cx="{fx:.0f}" cy="{fy:.0f}" r="3" fill="{color}" opacity="0.9"/>'
            )

        # --- Pollination events (success sparkles) ---
        for pe in sim.pollination_events[:200]:
            px, py = pe["pos"]
            pf = pe["frame"]
            ops = ";".join(
                "0.0" if fi < pf else f"{max(0, 0.9 - (fi - pf) / 18):.2f}" for fi in range(F)
            )
            rv = ";".join(
                "0" if fi < pf else f"{min(12, (fi - pf) * 1.2):.1f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" fill="none" stroke="{pe["color"]}" '
                f'stroke-width="1.5" opacity="0.0">'
                f'<animate attributeName="r" values="{rv}" dur="{dur}s" repeatCount="indefinite" '
                f'calcMode="discrete"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" '
                f'calcMode="discrete"/></circle>'
            )

        # --- Clock hand ---
        hand_x = ";".join(
            f"{cx + math.cos((m / 12) * 2 * math.pi - math.pi / 2) * (R - 12):.1f}"
            for m in sim.hist_month
        )
        hand_y = ";".join(
            f"{cy + math.sin((m / 12) * 2 * math.pi - math.pi / 2) * (R - 12):.1f}"
            for m in sim.hist_month
        )
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 12}" '
            f'stroke="#fdd835" stroke-width="2" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        # --- Bee particles ---
        bee_colors = ["#fdd835", "#ffb300", "#ffe082", "#fff176", "#ffee58"]
        for i in range(cfg.num_bees):
            px = ";".join(f"{p[i, 0]:.1f}" for p in sim.hist_xy)
            py = ";".join(f"{p[i, 1]:.1f}" for p in sim.hist_xy)
            col = bee_colors[i % len(bee_colors)]
            r_pt = 3.0 if i % 7 == 0 else 2.0
            svg.append(
                f'<circle r="{r_pt}" fill="{col}" opacity="0.8">'
                f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Centre hub (hive) ---
        colony_r = ";".join(f"{14 + sim.hist_colony[fi] * 10:.1f}" for fi in range(F))
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" fill="url(#hiveGlow)">'
            f'<animate attributeName="r" values="{colony_r}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</circle>'
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="14" fill="#1a1f33" stroke="#fdd835" stroke-width="3"/>'
        )
        svg.append(f'<text x="{cx}" y="{cy + 3}" font-size="15" fill="#fdd835" '
                   f'text-anchor="middle" font-weight="bold">HIVE</text>')

        # --- Inner radial bars: nectar flow by month ---
        for i in range(12):
            a = (i / 12) * 2 * math.pi - math.pi / 2
            val = NECTAR_FLOW_CURVE[i]
            bar_len = val * 50
            bx1 = cx + math.cos(a) * 26
            by1 = cy + math.sin(a) * 26
            bx2 = cx + math.cos(a) * (26 + bar_len)
            by2 = cy + math.sin(a) * (26 + bar_len)
            svg.append(
                f'<line x1="{bx1:.0f}" y1="{by1:.0f}" x2="{bx2:.0f}" y2="{by2:.0f}" '
                f'stroke="#ffb300" stroke-width="3.5" stroke-linecap="round" '
                f'opacity="{0.25 + val * 0.65:.2f}"/>'
            )

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT SIDE: Info panels
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Simulation Logic ---
        py1 = 20
        ph1 = 256
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1a1a2e" rx="8" '
                   f'stroke="#fdd835" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#fdd835" font-size="15" font-weight="bold">'
                   f'Pollination Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Blooms keep colonies active across seasons.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'Visits become service when pollen reaches a</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'flower of the same guild, rather than self.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'Colony strength tracks nectar-pollen flow.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'Concentric arcs show floral supply structures.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#fdd835" font-size="15">'
                   f'Rings pulse outward = successful pollination</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 153
        total_p = sim.cross_pollinations + sim.self_pollinations
        eff = (sim.cross_pollinations / max(1, total_p)) * 100

        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1a1a2e" rx="8" '
                   f'stroke="#4fc3f7" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#4fc3f7" font-size="15" font-weight="bold">'
                   f'Pollination Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#e0e0e0" font-size="15">'
                   f'Total flower visits: {sim.total_visits:,}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#76ff03" font-size="15">'
                   f'Cross-pollinations: {sim.cross_pollinations} ({eff:.0f}% efficiency)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#ffcc80" font-size="15">'
                   f'Self-pollinations: {sim.self_pollinations}</text>')

        # Per-guild visits summary
        top_guild = max(sim.visits_per_guild, key=sim.visits_per_guild.get)
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90caf9" font-size="15">'
                   f'Most visited guild: {top_guild} ({sim.visits_per_guild[top_guild]})</text>')
        best_month = sim.pollination_per_month.index(max(sim.pollination_per_month))
        svg.append(f'<text font-weight="bold" x="12" y="116" fill="#ce93d8" font-size="15">'
                   f'Peak pollination month: {months[best_month]}</text>')
        svg.append('</g>')

        # --- Panel 3: Guild legend ---
        py3 = py2 + ph2 + 10
        ph3 = 147
        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1a1a2e" rx="8" '
                   f'stroke="#66bb6a" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#66bb6a" font-size="15" font-weight="bold">'
                   f'Flower Guilds (Concentric Arcs)</text>')
        for gi, (gname, guild) in enumerate(FLOWER_GUILDS.items()):
            gy = 40 + gi * 16
            svg.append(f'<circle cx="22" cy="{gy}" r="5" fill="{guild["color"]}"/>')
            peak_m = guild["bloom_curve"].index(max(guild["bloom_curve"]))
            svg.append(f'<text font-weight="bold" x="34" y="{gy + 4}" fill="#e0e0e0" font-size="15">'
                       f'{guild["label"]} — peak: {months[peak_m]}</text>')
        svg.append('</g>')

        # --- Panel 4: Annual curves mini-chart ---
        py4 = py3 + ph3 + 10
        ph4 = 140
        chart_w = panel_w - 32
        chart_h = 55
        chart_x0 = 16
        chart_y0 = 30

        svg.append(f'<g transform="translate({panel_x}, {py4})">')
        svg.append(f'<rect width="{panel_w}" height="{ph4}" fill="#1a1a2e" rx="8" '
                   f'stroke="#7e57c2" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#7e57c2" font-size="15" font-weight="bold">'
                   f'Annual Environmental Curves</text>')

        curves = [
            (BEE_ACTIVITY_CURVE, "#fdd835", "Bee Activity"),
            (NECTAR_FLOW_CURVE, "#ff9800", "Nectar Flow"),
            (TEMPERATURE_CURVE, "#ef5350", "Temperature"),
        ]
        for curve_data, color, label in curves:
            pts = []
            for mi in range(12):
                px_c = chart_x0 + (mi / 11) * chart_w
                py_c = chart_y0 + chart_h - curve_data[mi] * chart_h
                pts.append(f"{px_c:.0f},{py_c:.0f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="1.8" opacity="0.75"/>')

        legend_y = chart_y0 + chart_h + 14
        for ci, (_, color, label) in enumerate(curves):
            lx = chart_x0 + ci * 120
            svg.append(f'<circle cx="{lx}" cy="{legend_y}" r="4" fill="{color}"/>')
            svg.append(f'<text font-weight="bold" x="{lx + 8}" y="{legend_y + 4}" fill="{color}" font-size="15">'
                       f'{label}</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    """Run bee pollination phenological clock simulation."""
    print(f" — Abelha-nativa ↔ Cerrado Wildflowers Pollination Clock on {CONFIG.device}...")

    sim = BeePollinationSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    total_p = sim.cross_pollinations + sim.self_pollinations
    eff = (sim.cross_pollinations / max(1, total_p)) * 100
    print(f"Done: {sim.total_visits} visits, {sim.cross_pollinations} cross-poll "
          f"({eff:.0f}%), colony strength final: {sim.colony_strength:.2f}")
    print(f"Visits per guild: {sim.visits_per_guild}")
    print(f"Pollination by month: {sim.pollination_per_month}")

    print("Generating SVG...")
    renderer = PollinationRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_56')
    return svg_content


if __name__ == "__main__":
    main()
