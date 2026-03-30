# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 58: Morcego-nectarívoro ↔ Babassu Palm — Chiropterophily Clock
# INTERVENTION 4/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_58.py — Nectar Bats ↔ Babassu/Cerrado Night-Blooming Flora:
Notebook Differentiation:
- Differentiation Focus: Morcego-nectarívoro ↔ Babassu Palm — Chiropterophily Clock emphasizing mining runoff risks.
- Indicator species: Taboa (Typha domingensis).
- Pollination lens: temporal mismatch with migratory pollinators.
- Human impact lens: extreme drought thresholds.

                 Chiropterophily & Chiropterochory Phenological Clock

Models the nocturnal pollination and seed-dispersal mutualism between
nectar-feeding bats (Glossophaga soricina, Anoura geoffroyi) and night-
blooming Cerrado flora, particularly:
  • Babassu palm (Attalea speciosa) — bat-pollinated, seasonal flowering
  • Pseudobombax (Shaving-brush tree) — iconic chiropterophilous flowers
  • Caryocar brasiliense (Pequi) — partially bat-pollinated at night
  • Hymenaea stigonocarpa (Jatobá-do-cerrado) — bat/moth pollinated

The radial phenological clock maps:
  • Night-bloom phenology across 4 plant species (complementary windows)
  • Bat foraging routes: trapline pollination strategy
  • Nectar production peaks (correlated with temperature + humidity)
  • Pollen-load transfer events and pollination efficiency
  • Bat reproductive cycle and roost colony dynamics
  • Day-length influence on nocturnal foraging window

Following the evidence logic established elsewhere in the suite and echoed in
base.txt, bats are treated here as mobile ecosystem-service providers: intact
roosts, dark flight corridors, and complementary flowering windows determine
whether visits become effective cross-pollination rather than simple nectar
removal.

Scientific References:
  - Fleming, T.H. et al. (2009). "The evolution of bat pollination."
    Annals of Botany 104(6).
  - Bobrowiec, P.E.D. & Oliveira, P.E. (2012). "Removal effects on
    nectar production in bat-pollinated flowers." Biotropica 44(1).
  - Gribel, R. & Hay, J.D. (1993). "Pollination ecology of Caryocar
    brasiliense in Central Brazil." J. Trop. Ecology 9(2).
  - Teixeira, R.C. et al. (2020). "Bat assemblages in Cerrado fragments."
    Acta Chiropterologica.

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

# Night-blooming flora guilds — chiropterophilous species
# Based on Fleming et al. (2009), Gribel & Hay (1993)
NIGHT_FLORA = {
    "Babassu": {
        "color": "#ab47bc",       # purple
        "bloom_curve": [0.30, 0.20, 0.10, 0.05, 0.00, 0.00, 0.05, 0.25, 0.60, 0.85, 0.70, 0.45],
        # Peak: Oct-Nov (late dry → wet transition)
        "nectar_mL": 2.5,        # copious nectar producer
        "scent_radius": 45,
        "arc_radius": 210,
        "label": "Babassu (Attalea speciosa)",
    },
    "Pseudobombax": {
        "color": "#ec407a",       # magenta-pink
        "bloom_curve": [0.00, 0.00, 0.00, 0.00, 0.10, 0.40, 0.80, 1.00, 0.60, 0.15, 0.00, 0.00],
        # Peak: Jul-Aug (dry season, leafless canopy display)
        "nectar_mL": 3.0,        # very high nectar
        "scent_radius": 55,
        "arc_radius": 180,
        "label": "Pseudobombax (Shaving-brush)",
    },
    "Pequi-noturno": {
        "color": "#ffb74d",       # soft orange
        "bloom_curve": [0.50, 0.25, 0.10, 0.00, 0.00, 0.00, 0.05, 0.15, 0.35, 0.55, 0.80, 0.70],
        # Peak: Nov-Dec (overlap with bee-pollinated day phase — nb56)
        "nectar_mL": 1.8,
        "scent_radius": 35,
        "arc_radius": 150,
        "label": "Pequi-noturno (Caryocar)",
    },
    "Jatoba": {
        "color": "#78909c",       # blue-grey
        "bloom_curve": [0.10, 0.05, 0.00, 0.00, 0.00, 0.15, 0.35, 0.55, 0.70, 0.50, 0.20, 0.10],
        # Peak: Aug-Sep (dry season)
        "nectar_mL": 1.5,
        "scent_radius": 30,
        "arc_radius": 120,
        "label": "Jatobá (Hymenaea stigonocarpa)",
    },
}

# Bat colony activity — nocturnal, influenced by night length and temperature
# Teixeira et al. (2020): bats more active in warmer months with longer nights
BAT_ACTIVITY_CURVE = [
    0.80,   # JAN — warm, moderate nights
    0.75,   # FEV
    0.70,   # MAR
    0.60,   # ABR — cooling
    0.50,   # MAI — shorter warm hours
    0.40,   # JUN — coldest, least active
    0.45,   # JUL — cold but dry (some bloom)
    0.60,   # AGO — warming, Pseudobombax peak
    0.75,   # SET — recovering
    0.85,   # OUT — peak activity, many blooms
    0.90,   # NOV — peak
    0.85,   # DEZ — high
]

# Night length index (hours of darkness, normalized 0–1)
# Goiás ~15°S: nights longer May-Jul, shorter Nov-Jan
NIGHT_LENGTH_CURVE = [
    0.45,   # JAN — short nights (summer)
    0.48,   # FEV
    0.52,   # MAR — equinox
    0.58,   # ABR
    0.65,   # MAI
    0.70,   # JUN — winter solstice, longest night
    0.68,   # JUL
    0.62,   # AGO
    0.55,   # SET — equinox
    0.50,   # OUT
    0.45,   # NOV
    0.43,   # DEZ — shortest night
]

# Humidity index (higher humidity = more nectar production)
HUMIDITY_CURVE = [
    0.85,   # JAN — peak wet
    0.88,   # FEV
    0.75,   # MAR
    0.55,   # ABR
    0.30,   # MAI
    0.18,   # JUN — driest
    0.15,   # JUL
    0.20,   # AGO
    0.35,   # SET
    0.55,   # OUT
    0.70,   # NOV
    0.80,   # DEZ
]

# Bat reproductive cycle (pregnancy peaks → lactation → weaning)
BAT_REPRODUCTION_CURVE = [
    0.60,   # JAN — lactating females
    0.40,   # FEV — weaning
    0.20,   # MAR
    0.10,   # ABR — non-reproductive
    0.10,   # MAI
    0.15,   # JUN
    0.30,   # JUL — mating season starts
    0.50,   # AGO — peak mating
    0.70,   # SET — pregnancy
    0.80,   # OUT — late pregnancy
    0.85,   # NOV — births begin
    0.75,   # DEZ — lactation
]


@dataclass
class ChiropterophilyConfig:
    """Configuration for the Bat ↔ Night Flora pollination clock."""
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Clock geometry
    clock_cx: float = 420.0
    clock_cy: float = 310.0
    clock_radius: float = 240.0
    # Bats
    num_bats: int = 50
    bat_speed_base: float = 0.06
    foraging_radius_min: float = 90.0
    foraging_radius_max: float = 200.0
    # Roosts
    num_roosts: int = 3
    roost_radius: float = 20.0
    # Pollination
    pollen_pickup_prob: float = 0.05
    pollen_carry_frames: int = 12
    max_pollination_events: int = 250
    # Flora nodes
    flora_nodes_per_guild: int = 3


CONFIG = ChiropterophilyConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class BatPollinationSim:
    """Phenological clock for bat ↔ night-blooming flora chiropterophily."""

    def __init__(self, cfg: ChiropterophilyConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy

        # --- Night-blooming flora nodes ---
        self.flora_nodes: List[Dict] = []
        for guild_name, guild in NIGHT_FLORA.items():
            curve = guild["bloom_curve"]
            peaks = sorted(range(12), key=lambda m: curve[m], reverse=True)[:cfg.flora_nodes_per_guild]
            for pm in peaks:
                angle = (pm / 12) * 2 * math.pi - math.pi / 2 + random.uniform(-0.15, 0.15)
                r = guild["arc_radius"] + random.uniform(-15, 15)
                fx = cx + math.cos(angle) * r
                fy = cy + math.sin(angle) * r
                self.flora_nodes.append({
                    "guild": guild_name,
                    "color": guild["color"],
                    "pos": (fx, fy),
                    "month": pm,
                    "nectar_mL": guild["nectar_mL"],
                    "scent_radius": guild["scent_radius"],
                })

        # --- Bat roosts (caves/karst formations — RESEX-specific) ---
        self.roosts: List[Dict] = []
        roost_angles = [0.5, 2.5, 4.8]  # spread around clock
        for i in range(cfg.num_roosts):
            ra = roost_angles[i]
            rx = cx + math.cos(ra) * 50
            ry = cy + math.sin(ra) * 50
            self.roosts.append({"pos": (rx, ry), "population": cfg.num_bats // cfg.num_roosts})

        # --- Bats: polar coordinates ---
        self.angles = torch.rand(cfg.num_bats, device=self.dev) * 2 * math.pi
        self.radii = torch.rand(cfg.num_bats, device=self.dev) * 30 + 30.0  # start near roosts
        self.ang_vel = (torch.rand(cfg.num_bats, device=self.dev) * 0.04 + 0.02)
        self.rad_vel = torch.randn(cfg.num_bats, device=self.dev) * 1.2
        self.has_pollen = torch.zeros(cfg.num_bats, device=self.dev, dtype=torch.bool)
        self.pollen_timer = torch.zeros(cfg.num_bats, device=self.dev)
        self.pollen_source = torch.full((cfg.num_bats,), -1, device=self.dev, dtype=torch.long)
        self.is_foraging = torch.ones(cfg.num_bats, device=self.dev, dtype=torch.bool)

        # History
        self.hist_xy: List[np.ndarray] = []
        self.hist_month: List[float] = []
        self.hist_activity: List[float] = []
        self.hist_night_len: List[float] = []

        # Pollination events
        self.pollination_events: List[Dict] = []
        self.cross_pollinations = 0
        self.total_visits = 0
        self.visits_per_guild: Dict[str, int] = {g: 0 for g in NIGHT_FLORA}
        self.pollination_per_month = [0] * 12
        self.nectar_harvested = 0.0

    def _interp(self, curve: list, month_frac: float) -> float:
        m = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def _composite_bloom(self, month_frac: float) -> float:
        total = sum(self._interp(g["bloom_curve"], month_frac) for g in NIGHT_FLORA.values())
        return min(1.0, total / len(NIGHT_FLORA))

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy = cfg.clock_cx, cfg.clock_cy
        month_frac = (frame / cfg.frames) * 12.0
        m_idx = int(month_frac) % 12

        activity = self._interp(BAT_ACTIVITY_CURVE, month_frac)
        night_len = self._interp(NIGHT_LENGTH_CURVE, month_frac)
        humidity = self._interp(HUMIDITY_CURVE, month_frac)
        bloom = self._composite_bloom(month_frac)

        self.hist_month.append(month_frac)
        self.hist_activity.append(activity)
        self.hist_night_len.append(night_len)

        # Foraging window: bats forage proportionally to night length ↔ activity
        forage_fraction = min(1.0, activity * night_len * 1.5 + 0.1)
        n_foraging = int(cfg.num_bats * forage_fraction)

        # Target radius expands toward active blooms
        target_r = cfg.foraging_radius_min + (cfg.foraging_radius_max - cfg.foraging_radius_min) * bloom

        # Bat movement — trapline strategy: angular progression with radial pull
        speed_mod = 0.4 + activity * 1.0
        self.ang_vel[:n_foraging] += torch.randn(n_foraging, device=self.dev) * 0.006
        self.angles[:n_foraging] += self.ang_vel[:n_foraging] * speed_mod
        self.angles = self.angles % (2 * math.pi)

        # Radial dynamics
        self.rad_vel[:n_foraging] += (target_r - self.radii[:n_foraging]) * 0.04
        self.rad_vel[:n_foraging] += torch.randn(n_foraging, device=self.dev) * 2.5
        self.rad_vel *= 0.82
        self.radii[:n_foraging] += self.rad_vel[:n_foraging]
        self.radii = torch.clamp(self.radii, 25.0, cfg.clock_radius + 10)

        # Roosting bats cluster at centre
        if n_foraging < cfg.num_bats:
            self.radii[n_foraging:] = self.radii[n_foraging:] * 0.88 + 20.0 * 0.12

        # Polar → Cartesian
        xs = cx + torch.cos(self.angles - math.pi / 2) * self.radii
        ys = cy + torch.sin(self.angles - math.pi / 2) * self.radii
        xy = torch.stack((xs, ys), dim=1)
        self.hist_xy.append(xy.cpu().numpy().copy())

        # --- Flora visits & pollen transfer ---
        for fi_node, fnode in enumerate(self.flora_nodes):
            guild = NIGHT_FLORA[fnode["guild"]]
            bloom_now = self._interp(guild["bloom_curve"], month_frac)
            if bloom_now < 0.05:
                continue

            # Nectar production scales with humidity
            nectar_now = fnode["nectar_mL"] * bloom_now * (0.5 + humidity * 0.5)
            fx, fy = fnode["pos"]
            scent_r = fnode["scent_radius"] * (0.7 + bloom_now * 0.5)

            d = torch.sqrt((xs[:n_foraging] - fx) ** 2 + (ys[:n_foraging] - fy) ** 2)
            near = d < scent_r
            if not near.any():
                continue

            for bi in near.nonzero().squeeze(1):
                bi_int = int(bi)
                self.total_visits += 1
                self.visits_per_guild[fnode["guild"]] += 1
                self.nectar_harvested += nectar_now * 0.1

                if not self.has_pollen[bi_int]:
                    if random.random() < cfg.pollen_pickup_prob * bloom_now * 2.5:
                        self.has_pollen[bi_int] = True
                        self.pollen_timer[bi_int] = 0
                        self.pollen_source[bi_int] = fi_node
                else:
                    src = int(self.pollen_source[bi_int])
                    if src != fi_node and src >= 0:
                        src_guild = self.flora_nodes[src]["guild"]
                        dst_guild = fnode["guild"]
                        if src_guild == dst_guild:
                            if len(self.pollination_events) < cfg.max_pollination_events:
                                self.pollination_events.append({
                                    "pos": (fx, fy), "frame": frame,
                                    "guild": dst_guild, "color": fnode["color"],
                                })
                                self.cross_pollinations += 1
                                self.pollination_per_month[m_idx] += 1
                        self.has_pollen[bi_int] = False
                        self.pollen_timer[bi_int] = 0
                        self.pollen_source[bi_int] = -1

        # Pollen expiry
        self.pollen_timer[self.has_pollen] += 1
        expired = self.has_pollen & (self.pollen_timer >= cfg.pollen_carry_frames)
        self.has_pollen[expired] = False
        self.pollen_timer[expired] = 0
        self.pollen_source[expired] = -1


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class ChiropterophilyRenderer:
    """Renders the bat ↔ night-flora chiropterophily clock as animated SVG."""

    def __init__(self, cfg: ChiropterophilyConfig, sim: BatPollinationSim):
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
            f'style="background-color:#06060e; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # --- Defs ---
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="nightBg58">'
            '<stop offset="0%" stop-color="#1a1040" stop-opacity="0.9"/>'
            '<stop offset="60%" stop-color="#0d0820" stop-opacity="0.5"/>'
            '<stop offset="100%" stop-color="#06060e" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="moonGlow">'
            '<stop offset="0%" stop-color="#e1bee7" stop-opacity="0.4"/>'
            '<stop offset="100%" stop-color="#4a148c" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="nectarGlow">'
            '<stop offset="0%" stop-color="#ec407a" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#880e4f" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<pattern id="starField" width="80" height="80" patternUnits="userSpaceOnUse">'
            '<circle cx="10" cy="15" r="0.8" fill="#e8eaf6" opacity="0.3"/>'
            '<circle cx="45" cy="8" r="0.5" fill="#c5cae9" opacity="0.25"/>'
            '<circle cx="70" cy="40" r="0.7" fill="#e8eaf6" opacity="0.2"/>'
            '<circle cx="25" cy="55" r="0.6" fill="#c5cae9" opacity="0.3"/>'
            '<circle cx="60" cy="70" r="0.5" fill="#e8eaf6" opacity="0.2"/>'
            '</pattern>'
        )
        svg.append('</defs>')

        # --- Background (night sky) ---
        svg.append(f'<rect width="{w}" height="{h}" fill="#06060e"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#starField)"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#nightBg58)"/>')

        # Moon accent
        svg.append(f'<circle cx="{w - 80}" cy="60" r="25" fill="#e1bee7" opacity="0.12"/>')
        svg.append(f'<circle cx="{w - 75}" cy="55" r="22" fill="#06060e"/>')

        # --- Title ---
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ce93d8" font-weight="bold">'
            f'ECO-SIM: Bat × Night Flora    - Chiropterophily Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#b0bec5">'
            f'Nocturnal pollination: Glossophaga/Anoura bats on 4 chiropterophilous guilds'
            f'</text>'
        )

        # --- Clock face ---
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 48)
            ty = cy + math.sin(angle) * (R + 48)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#b39ddb" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 8)
            ly2 = cy + math.sin(angle) * (R - 8)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#4a148c" stroke-width="2"/>'
            )

        # --- Concentric bloom arcs ---
        for guild_name, guild in NIGHT_FLORA.items():
            curve = guild["bloom_curve"]
            arc_r = guild["arc_radius"]
            color = guild["color"]

            svg.append(
                f'<circle cx="{cx}" cy="{cy}" r="{arc_r}" fill="none" '
                f'stroke="{color}" stroke-width="0.5" opacity="0.12"/>'
            )

            # Draw arcs where bloom > 0.2
            in_bloom = False
            start_m = 0
            for mi in range(13):
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
                    svg.append(
                        f'<path d="{d}" fill="none" stroke="{color}" '
                        f'stroke-width="14" stroke-linecap="round" opacity="0.28"/>'
                    )
                    in_bloom = False

            peak_m = curve.index(max(curve))
            la = (peak_m / 12) * 2 * math.pi - math.pi / 2
            lx = cx + math.cos(la) * (arc_r - 16)
            ly = cy + math.sin(la) * (arc_r - 16)
            svg.append(
                f'<text x="{lx:.0f}" y="{ly:.0f}" font-size="15" fill="{color}" '
                f'text-anchor="middle" opacity="0.85" font-weight="bold">{guild_name}</text>'
            )

        # --- Flora nodes (pulsating night-blooms) ---
        for fnode in sim.flora_nodes:
            fx, fy = fnode["pos"]
            guild = NIGHT_FLORA[fnode["guild"]]
            curve = guild["bloom_curve"]
            color = fnode["color"]

            r_vals = ";".join(
                f"{3 + sim._interp(curve, (fi / F) * 12) * 12:.1f}" for fi in range(F)
            )
            op_vals = ";".join(
                f"{0.1 + sim._interp(curve, (fi / F) * 12) * 0.7:.2f}" for fi in range(F)
            )
            # Scent halo
            scent_r_vals = ";".join(
                str(int(fnode["scent_radius"] * sim._interp(curve, (fi / F) * 12)))
                for fi in range(F)
            )
            svg.append(
                f'<circle cx="{fx:.0f}" cy="{fy:.0f}" fill="{color}" opacity="0.08">'
                f'<animate attributeName="r" values="{scent_r_vals}" '
                f'dur="{dur}s" repeatCount="indefinite"/></circle>'
            )
            # Bloom core
            svg.append(
                f'<circle cx="{fx:.0f}" cy="{fy:.0f}" fill="{color}">'
                f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Pollination sparkles ---
        for pe in sim.pollination_events[:180]:
            px, py = pe["pos"]
            pf = pe["frame"]
            ops = ";".join(
                "0.0" if fi < pf else f"{max(0, 0.85 - (fi - pf) / 15):.2f}" for fi in range(F)
            )
            rv = ";".join(
                "0" if fi < pf else f"{min(15, (fi - pf) * 1.5):.1f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py:.0f}" fill="none" stroke="{pe["color"]}" '
                f'stroke-width="1" opacity="0.0">'
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
            f'stroke="#ce93d8" stroke-width="2" stroke-linecap="round" opacity="0.85">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        # --- Bat particles ---
        bat_colors = ["#ce93d8", "#ba68c8", "#ab47bc", "#9c27b0", "#7b1fa2"]
        for i in range(cfg.num_bats):
            px = ";".join(f"{p[i, 0]:.1f}" for p in sim.hist_xy)
            py = ";".join(f"{p[i, 1]:.1f}" for p in sim.hist_xy)
            col = bat_colors[i % len(bat_colors)]
            r_pt = 3.0 if i % 8 == 0 else 2.0
            svg.append(
                f'<circle r="{r_pt}" fill="{col}" opacity="0.75">'
                f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Roost markers ---
        for roost in sim.roosts:
            rx, ry = roost["pos"]
            svg.append(
                f'<circle cx="{rx:.0f}" cy="{ry:.0f}" r="{cfg.roost_radius}" '
                f'fill="url(#moonGlow)" stroke="#7b1fa2" stroke-width="1.5" '
                f'stroke-dasharray="3,3" opacity="0.5"/>'
            )
            svg.append(
                f'<text font-weight="bold" x="{rx:.0f}" y="{ry + 3:.0f}" font-size="15" fill="#e1bee7" '
                f'text-anchor="middle" opacity="0.6">Roost</text>'
            )

        # --- Centre hub ---
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="16" fill="#1a1040" stroke="#ce93d8" stroke-width="3"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#ce93d8"/>')
        svg.append(f'<text font-weight="bold" x="{cx}" y="{cy + 3}" font-size="15" fill="#fff" text-anchor="middle">NOITE</text>')

        # --- Inner radial bars: nectar flow by month ---
        for i in range(12):
            a = (i / 12) * 2 * math.pi - math.pi / 2
            composite = sum(NIGHT_FLORA[g]["bloom_curve"][i] * NIGHT_FLORA[g]["nectar_mL"]
                           for g in NIGHT_FLORA) / sum(NIGHT_FLORA[g]["nectar_mL"] for g in NIGHT_FLORA)
            bar_len = composite * 50
            bx1 = cx + math.cos(a) * 24
            by1 = cy + math.sin(a) * 24
            bx2 = cx + math.cos(a) * (24 + bar_len)
            by2 = cy + math.sin(a) * (24 + bar_len)
            svg.append(
                f'<line x1="{bx1:.0f}" y1="{by1:.0f}" x2="{bx2:.0f}" y2="{by2:.0f}" '
                f'stroke="#ce93d8" stroke-width="3" stroke-linecap="round" '
                f'opacity="{0.25 + composite * 0.6:.2f}"/>'
            )

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Logic ---
        py1 = 20
        ph1 = 259
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1a1a2e" rx="8" '
                   f'stroke="#ce93d8" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ce93d8" font-size="15" font-weight="bold">'
                   f'Chiropterophily Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Bats leave karst roosts at dusk on traplines.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'Blooms keep nectar available across seasons.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'Pollination needs roost access and floral</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'overlap; drought or lighting can cut visits.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ccc" font-size="15">'
                   f'Snout and chest pollen moves between crowns.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#ce93d8" font-size="15">'
                   f'Rings = successful cross-pollination events</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 154
        total_p = sim.cross_pollinations
        best_month = sim.pollination_per_month.index(max(sim.pollination_per_month))
        top_guild = max(sim.visits_per_guild, key=sim.visits_per_guild.get)

        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1a1a2e" rx="8" '
                   f'stroke="#4fc3f7" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#4fc3f7" font-size="15" font-weight="bold">'
                   f'Pollination Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#e0e0e0" font-size="15">'
                   f'Total nightly visits: {sim.total_visits:,}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#ce93d8" font-size="15">'
                   f'Cross-pollinations: {sim.cross_pollinations}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#ffb74d" font-size="15">'
                   f'Nectar harvested: {sim.nectar_harvested:.0f} mL equivalent</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90caf9" font-size="15">'
                   f'Peak month: {months[best_month]} | Top guild: {top_guild}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="116" fill="#78909c" font-size="15">'
                   f'Roosts: {cfg.num_roosts} karst caves | Bats: {cfg.num_bats}</text>')
        svg.append('</g>')

        # --- Panel 3: Guild legend ---
        py3 = py2 + ph2 + 10
        ph3 = 117
        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1a1a2e" rx="8" '
                   f'stroke="#66bb6a" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="18" fill="#66bb6a" font-size="15" font-weight="bold">'
                   f'Night-Blooming Guilds</text>')
        for gi, (gname, guild) in enumerate(NIGHT_FLORA.items()):
            gy = 36 + gi * 16
            peak_m = guild["bloom_curve"].index(max(guild["bloom_curve"]))
            svg.append(f'<circle cx="22" cy="{gy}" r="5" fill="{guild["color"]}"/>')
            svg.append(f'<text font-weight="bold" x="34" y="{gy + 4}" fill="#e0e0e0" font-size="15">'
                       f'{guild["label"]} — peak: {months[peak_m]}</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    """Run bat ↔ night flora chiropterophily clock simulation."""
    print(f" — Morcego ↔ Night Flora Chiropterophily Clock on {CONFIG.device}...")

    sim = BatPollinationSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_visits} visits, {sim.cross_pollinations} cross-pollinations, "
          f"nectar harvested: {sim.nectar_harvested:.0f} mL")
    print(f"Visits per guild: {sim.visits_per_guild}")
    print(f"Pollination by month: {sim.pollination_per_month}")

    print("Generating SVG...")
    renderer = ChiropterophilyRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_58')
    return svg_content


if __name__ == "__main__":
    main()
