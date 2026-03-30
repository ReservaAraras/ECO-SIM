# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 55: Tucano-toco ↔ Pequi Tree — Ornithochory Phenological Clock
# INTERVENTION 1/12: Ecological Relationship Clocks Series
# ===================================================================================================
"""
notebook_55.py — Tucano-toco ↔ Pequi: Seasonal Ornithochory Clock
Notebook Differentiation:
- Differentiation Focus: Tucano-toco ↔ Pequi Tree — Ornithochory Phenological Clock emphasizing road mortality hotspots.
- Indicator species: Fungo-termitomyces (Termitomyces sp.).
- Pollination lens: orchid mimicry with specialist bees.
- Human impact lens: invasive grass spread.

Models the mutualistic frugivory-dispersal relationship between the Toucan
(Ramphastos toco) and the Pequi tree (Caryocar brasiliense).  The radial
phenological clock maps:
  • Monthly Pequi fruiting intensity (peak: Nov-Feb wet season)
  • Tucano foraging radius expansion during fruiting
  • Seed gut-passage and regurgitation dispersal events
  • Germination probability tied to soil moisture (wet season)
  • Seasonal body-condition index for the bird population

As in the stronger evidence-based notebooks of the suite, the argument here is
service-based rather than descriptive: fruit pulses alter toucan movement,
gut passage relocates seeds beyond parent crowns, and recruitment only occurs
when dispersal coincides with wet-season soil moisture and suitable microsites.

Scientific References:
  - Melo, C. et al. (2009). "Frugivory and seed dispersal by birds in
    Cerrado remnants." Revista Brasileira de Botânica 32(4).
  - Oliveira, P.E. & Moreira, A.G. (1992). "Anemochory, zoochory and
    phenology in cerrado." Biotropica.
  - Christianini, A.V. & Oliveira, P.S. (2010). "Birds and ants provide
    complementary seed dispersal in cerrado."

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
# 1. SCIENTIFIC PARAMETERS & CONFIGURATION
# ===================================================================================================

# Monthly Pequi fruiting intensity curve (0–1): based on Oliveira & Moreira (1992)
# Peak fruiting: November–February (wet season); near-zero: May–August (dry season)
PEQUI_FRUITING_CURVE = [
    0.40,   # JAN — full wet, heavy fruiting
    0.50,   # FEV — peak fruiting
    0.30,   # MAR — late wet
    0.10,   # ABR — transition
    0.02,   # MAI — dry season onset
    0.00,   # JUN — dry
    0.00,   # JUL — dry
    0.01,   # AGO — late dry
    0.05,   # SET — transition
    0.15,   # OUT — early wet, buds
    0.25,   # NOV — flowering + early fruits
    0.45,   # DEZ — heavy fruiting
]

# Soil moisture curve for germination probability (linked to precipitation)
SOIL_MOISTURE_CURVE = [0.80, 0.70, 0.85, 0.60, 0.35, 0.15, 0.05, 0.05, 0.25, 0.50, 0.80, 0.90]

# Tucano body condition index — influenced by diet availability
TUCANO_CONDITION_CURVE = [
    0.75,   # JAN — good; abundant fruit
    0.80,   # FEV — peak condition
    0.70,   # MAR — declining fruit
    0.55,   # ABR — transition
    0.40,   # MAI — lean season begins
    0.30,   # JUN — lowest
    0.28,   # JUL — nadir
    0.32,   # AGO — slow recovery
    0.45,   # SET — insects supplement
    0.55,   # OUT — early fruit return
    0.65,   # NOV — building
    0.72,   # DEZ — near-peak
]


@dataclass
class OrnithochoryConfig:
    """Configuration for the Tucano ↔ Pequi ornithochory phenological clock."""
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360          # 1 full annual cycle
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Clock geometry
    clock_cx: float = 420.0    # clock centre X
    clock_cy: float = 310.0    # clock centre Y
    clock_radius: float = 240.0
    # Birds
    num_tucanos: int = 60
    tucano_base_speed: float = 0.04      # angular velocity rad/frame
    tucano_radial_jitter: float = 2.5
    foraging_radius_min: float = 120.0   # dry season
    foraging_radius_max: float = 220.0   # fruiting season
    # Seed dispersal
    seed_drop_prob: float = 0.03         # per-frame per-bird during fruiting
    seed_carry_frames: int = 25          # gut passage time
    germination_delay: int = 40          # frames until sprouting if wet enough
    max_seeds: int = 200
    # Pequi trees
    num_pequi_trees: int = 8


CONFIG = OrnithochoryConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class TucanoPequiSim:
    """Phenological clock model for Tucano ↔ Pequi ornithochory."""

    def __init__(self, cfg: OrnithochoryConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy

        # --- Pequi trees placed around the clock perimeter ---
        self.pequi_positions: List[Tuple[float, float]] = []
        self.pequi_months: List[int] = []  # which month arc each tree aligns with
        for i in range(cfg.num_pequi_trees):
            month = random.choice([10, 11, 0, 1, 2])  # wet-season months
            angle = (month / 12) * 2 * math.pi - math.pi / 2
            r = cfg.clock_radius + random.uniform(15, 40)
            tx = cx + math.cos(angle + random.uniform(-0.15, 0.15)) * r
            ty = cy + math.sin(angle + random.uniform(-0.15, 0.15)) * r
            self.pequi_positions.append((tx, ty))
            self.pequi_months.append(month)

        # --- Tucano birds: polar coordinates orbiting clock ---
        self.angles = torch.rand(cfg.num_tucanos, device=self.dev) * 2 * math.pi
        self.radii = torch.rand(cfg.num_tucanos, device=self.dev) * 60 + 140.0
        self.ang_vel = (torch.rand(cfg.num_tucanos, device=self.dev) * 0.03 + 0.02)
        self.rad_vel = torch.randn(cfg.num_tucanos, device=self.dev) * 1.5
        self.has_seed = torch.zeros(cfg.num_tucanos, device=self.dev, dtype=torch.bool)
        self.seed_timer = torch.zeros(cfg.num_tucanos, device=self.dev)
        self.condition = torch.ones(cfg.num_tucanos, device=self.dev) * 0.5

        # History for animation
        self.hist_xy: List[np.ndarray] = []
        self.hist_month: List[float] = []

        # Seed dispersal events: {"pos":(x,y), "frame":int, "germinated":bool, "sprout_frame":int}
        self.seed_events: List[Dict] = []
        self.germinated_count = 0
        self.seeds_dropped = 0

        # Per-month summary stats
        self.month_fruiting: List[float] = []
        self.month_moisture: List[float] = []
        self.month_condition: List[float] = []
        self.dispersal_per_month = [0] * 12

    def _interp_curve(self, curve: list, month_frac: float) -> float:
        """Linearly interpolate a 12-element monthly curve."""
        m = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy = cfg.clock_cx, cfg.clock_cy
        month_frac = (frame / cfg.frames) * 12.0

        # Current environmental state
        fruiting = self._interp_curve(PEQUI_FRUITING_CURVE, month_frac)
        moisture = self._interp_curve(SOIL_MOISTURE_CURVE, month_frac)
        condition = self._interp_curve(TUCANO_CONDITION_CURVE, month_frac)

        self.month_fruiting.append(fruiting)
        self.month_moisture.append(moisture)
        self.month_condition.append(condition)
        self.hist_month.append(month_frac)

        # Dynamic foraging radius — expands during fruiting
        target_radius = cfg.foraging_radius_min + (cfg.foraging_radius_max - cfg.foraging_radius_min) * fruiting

        # Update bird dynamics
        # Angular motion: faster foraging during fruiting
        speed_mod = 1.0 + fruiting * 0.8
        self.angles += self.ang_vel * speed_mod
        self.angles = self.angles % (2 * math.pi)

        # Radial: pulled toward "active foraging ring" which varies seasonally
        self.rad_vel += (target_radius - self.radii) * 0.06
        self.rad_vel += torch.randn(cfg.num_tucanos, device=self.dev) * cfg.tucano_radial_jitter
        self.rad_vel *= 0.88  # damping
        self.radii += self.rad_vel
        self.radii = torch.clamp(self.radii, 80.0, cfg.clock_radius + 20)

        # Body condition tracking
        self.condition = self.condition * 0.95 + condition * 0.05

        # Polar → Cartesian
        xs = cx + torch.cos(self.angles - math.pi / 2) * self.radii
        ys = cy + torch.sin(self.angles - math.pi / 2) * self.radii
        xy = torch.stack((xs, ys), dim=1)
        self.hist_xy.append(xy.cpu().numpy().copy())

        # --- Seed pickup (near fruiting trees during peak) ---
        if fruiting > 0.1:
            eligible = (~self.has_seed)
            for tx, ty in self.pequi_positions:
                d = torch.sqrt((xs - tx) ** 2 + (ys - ty) ** 2)
                near = eligible & (d < 35.0)
                pickup = near & (torch.rand(cfg.num_tucanos, device=self.dev) < cfg.seed_drop_prob * fruiting * 3)
                self.has_seed[pickup] = True
                self.seed_timer[pickup] = 0

        # --- Seed timer & drop ---
        self.seed_timer[self.has_seed] += 1
        drop = self.has_seed & (self.seed_timer >= cfg.seed_carry_frames)
        if drop.any() and len(self.seed_events) < cfg.max_seeds:
            for idx in drop.nonzero().squeeze(1):
                sx = xs[idx].item()
                sy = ys[idx].item()
                will_germinate = random.random() < moisture * 0.6
                se = {
                    "pos": (sx, sy),
                    "frame": frame,
                    "germinated": will_germinate,
                    "sprout_frame": frame + cfg.germination_delay if will_germinate else -1,
                }
                self.seed_events.append(se)
                self.seeds_dropped += 1
                m_idx = int(month_frac) % 12
                self.dispersal_per_month[m_idx] += 1
                if will_germinate:
                    self.germinated_count += 1
            self.has_seed[drop] = False
            self.seed_timer[drop] = 0


# ===================================================================================================
# 3. VISUALIZATION — RADIAL PHENOLOGICAL CLOCK
# ===================================================================================================

class OrnithochoryRenderer:
    """Renders the radial phenological clock as animated SVG."""

    def __init__(self, cfg: OrnithochoryConfig, sim: TucanoPequiSim):
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
            f'style="background-color:#0a0e1a; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # --- Defs: gradients & patterns ---
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="clockBg">'
            '<stop offset="0%" stop-color="#1a2744" stop-opacity="0.9"/>'
            '<stop offset="85%" stop-color="#0d1520" stop-opacity="0.3"/>'
            '<stop offset="100%" stop-color="#0a0e1a" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="fruitGlow">'
            '<stop offset="0%" stop-color="#ffab00" stop-opacity="0.7"/>'
            '<stop offset="100%" stop-color="#ff6d00" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="seedGlow">'
            '<stop offset="0%" stop-color="#76ff03" stop-opacity="0.6"/>'
            '<stop offset="100%" stop-color="#33691e" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<pattern id="dotGrid55" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="1.5" fill="#37474f" opacity="0.2"/>'
            '</pattern>'
        )
        svg.append('</defs>')

        # --- Background ---
        svg.append(f'<rect width="{w}" height="{h}" fill="#0a0e1a"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid55)"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 60}" fill="url(#clockBg)"/>')

        # --- Title ---
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#ffab00" font-weight="bold">'
            f'ECO-SIM: Toco Toucan × Pequi    - Ornithochory Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#b0bec5">'
            f'Seasonal seed dispersal by Ramphastos toco on Caryocar brasiliense'
            f'</text>'
        )

        # --- Clock face: month labels & ticks ---
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            # Outer label
            tx = cx + math.cos(angle) * (R + 50)
            ty = cy + math.sin(angle) * (R + 50)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="#90caf9" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            # Tick
            lx1 = cx + math.cos(angle) * R
            ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R - 10)
            ly2 = cy + math.sin(angle) * (R - 10)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#546e7a" stroke-width="2"/>'
            )
            # Minor ticks
            for sub in range(1, 4):
                sa = angle + (sub / 4) * (2 * math.pi / 12)
                sx1 = cx + math.cos(sa) * R
                sy1 = cy + math.sin(sa) * R
                sx2 = cx + math.cos(sa) * (R - 5)
                sy2 = cy + math.sin(sa) * (R - 5)
                svg.append(
                    f'<line x1="{sx1:.0f}" y1="{sy1:.0f}" x2="{sx2:.0f}" y2="{sy2:.0f}" '
                    f'stroke="#37474f" stroke-width="1"/>'
                )

        # --- Concentric guide rings ---
        for r_ring in [80, 120, 160, 200]:
            svg.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r_ring}" fill="none" '
                f'stroke="#263238" stroke-width="0.5" stroke-dasharray="3,5"/>'
            )

        # --- Fruiting arc (Pequi season: Oct–Feb) ---
        def draw_season_arc(start_m, end_m, radius, color, label, opacity=0.35):
            a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
            a2 = (end_m / 12) * 2 * math.pi - math.pi / 2
            x1 = cx + math.cos(a1) * radius
            y1 = cy + math.sin(a1) * radius
            x2 = cx + math.cos(a2) * radius
            y2 = cy + math.sin(a2) * radius
            large = 1 if (end_m - start_m) % 12 > 6 else 0
            d = f"M {x1:.0f} {y1:.0f} A {radius} {radius} 0 {large} 1 {x2:.0f} {y2:.0f}"
            svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="18" '
                       f'stroke-linecap="round" opacity="{opacity}"/>')
            mid_a = ((start_m + end_m) / 2 / 12) * 2 * math.pi - math.pi / 2
            lx = cx + math.cos(mid_a) * (radius + 22)
            ly = cy + math.sin(mid_a) * (radius + 22)
            svg.append(f'<text font-weight="bold" x="{lx:.0f}" y="{ly:.0f}" font-size="15" fill="{color}" '
                       f'text-anchor="middle" opacity="0.9">{label}</text>')

        draw_season_arc(9.5, 14.5, 200, "#ff9800", "Pequi fruiting", 0.30)
        draw_season_arc(4, 8, 160, "#795548", "Dry season stress", 0.18)
        draw_season_arc(0, 4, 180, "#4caf50", "Germination window", 0.22)

        # --- Pequi trees ---
        for (tx, ty), pm in zip(sim.pequi_positions, sim.pequi_months):
            # Pulse opacity with fruiting
            fruit_ops = []
            for fi in range(F):
                mf = (fi / F) * 12.0
                intensity = sim._interp_curve(PEQUI_FRUITING_CURVE, mf)
                fruit_ops.append(f"{0.3 + intensity * 0.7:.2f}")
            r_vals = ";".join(f"{8 + sim._interp_curve(PEQUI_FRUITING_CURVE, (fi/F)*12)*10:.1f}" for fi in range(F))
            ops = ";".join(fruit_ops)
            svg.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" fill="#ff9800" stroke="#e65100" stroke-width="1.5">'
                f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<text font-weight="bold" x="{tx:.0f}" y="{ty - 16:.0f}" font-size="15" fill="#ffe082" '
                f'text-anchor="middle" opacity="0.7">Pequi</text>'
            )

        # --- Seed dispersal events (sprouting animations) ---
        for se in sim.seed_events:
            sx, sy = se["pos"]
            sf = se["frame"]
            if se["germinated"]:
                spf = se["sprout_frame"]
                ops_s = ";".join(
                    "0.0" if fi < sf else ("0.3" if fi < spf else "0.8") for fi in range(F)
                )
                rv = ";".join(
                    "0" if fi < spf else f"{min(6.0, (fi - spf) / 20.0 * 6.0):.1f}" for fi in range(F)
                )
                svg.append(
                    f'<circle cx="{sx:.0f}" cy="{sy:.0f}" fill="#76ff03" opacity="0.0">'
                    f'<animate attributeName="r" values="{rv}" dur="{dur}s" repeatCount="indefinite" '
                    f'calcMode="discrete"/>'
                    f'<animate attributeName="opacity" values="{ops_s}" dur="{dur}s" repeatCount="indefinite" '
                    f'calcMode="discrete"/></circle>'
                )
            else:
                # Ungerminated seed: brief dot that fades
                ops_u = ";".join(
                    "0.0" if fi < sf else (f"{max(0, 0.5 - (fi-sf)/30):.2f}") for fi in range(F)
                )
                svg.append(
                    f'<circle cx="{sx:.0f}" cy="{sy:.0f}" r="2" fill="#ffcc80" opacity="0.0">'
                    f'<animate attributeName="opacity" values="{ops_u}" dur="{dur}s" repeatCount="indefinite" '
                    f'calcMode="discrete"/></circle>'
                )

        # --- Sweeping clock hand showing current month ---
        hand_x_vals = ";".join(
            f"{cx + math.cos((m / 12) * 2 * math.pi - math.pi / 2) * (R - 15):.1f}"
            for m in sim.hist_month
        )
        hand_y_vals = ";".join(
            f"{cy + math.sin((m / 12) * 2 * math.pi - math.pi / 2) * (R - 15):.1f}"
            for m in sim.hist_month
        )
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 15}" '
            f'stroke="#ffab00" stroke-width="2.5" stroke-linecap="round">'
            f'<animate attributeName="x2" values="{hand_x_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )

        # --- Tucano particles ---
        # Color encodes condition: low = dull, high = vibrant
        for i in range(cfg.num_tucanos):
            px = ";".join(f"{p[i, 0]:.1f}" for p in sim.hist_xy)
            py = ";".join(f"{p[i, 1]:.1f}" for p in sim.hist_xy)
            # Alternate colors between beak (orange) and body (black)
            col = "#fe4db7" if i % 3 == 0 else ("#ffab00" if i % 3 == 1 else "#00e5ff")
            r_pt = 3.5 if i % 5 == 0 else 2.5
            svg.append(
                f'<circle r="{r_pt}" fill="{col}" opacity="0.85">'
                f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # --- Centre hub ---
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="18" fill="#1a2744" stroke="#ffab00" stroke-width="3"/>'
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="5" fill="#ffab00"/>'
        )
        svg.append(
            f'<text font-weight="bold" x="{cx}" y="{cy + 4}" font-size="15" fill="#fff" text-anchor="middle">ANO</text>'
        )

        # --- Fruiting intensity radial bar chart (inner ring) ---
        for i in range(12):
            a = (i / 12) * 2 * math.pi - math.pi / 2
            val = PEQUI_FRUITING_CURVE[i]
            bar_len = val * 60
            bx1 = cx + math.cos(a) * 30
            by1 = cy + math.sin(a) * 30
            bx2 = cx + math.cos(a) * (30 + bar_len)
            by2 = cy + math.sin(a) * (30 + bar_len)
            op = 0.3 + val * 0.7
            svg.append(
                f'<line x1="{bx1:.0f}" y1="{by1:.0f}" x2="{bx2:.0f}" y2="{by2:.0f}" '
                f'stroke="#ff9800" stroke-width="4" stroke-linecap="round" opacity="{op:.2f}"/>'
            )

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT SIDE: Info panels
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

        panel_x = w - 390
        panel_w = 370

        # --- Panel 1: Simulation Logic ---
        py1 = 20
        ph1 = 241
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#1a1a2e" rx="8" ry="8" '
                   f'stroke="#ffab00" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#ffab00" font-size="15" font-weight="bold">'
                   f'Ornithochory Clock Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'Pequi pulses redirect toucan foraging paths.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'Gut passage disperses seeds from parent crown.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="72" fill="#ccc" font-size="15">'
                   f'Dispersal recruits where moisture returns.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="86" fill="#ccc" font-size="15">'
                   f'This links bird diet to Cerrado regeneration.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="102" fill="#76ff03" font-size="15">'
                   f'Green dots = established dispersal events</text>')
        svg.append('</g>')

        # --- Panel 2: Metrics ---
        py2 = py1 + ph1 + 10
        ph2 = 181
        germ_pct = (sim.germinated_count / max(1, sim.seeds_dropped)) * 100
        peak_month_idx = PEQUI_FRUITING_CURVE.index(max(PEQUI_FRUITING_CURVE))
        peak_month_name = months[peak_month_idx]
        best_disp_month = sim.dispersal_per_month.index(max(sim.dispersal_per_month))
        co2_est = sim.germinated_count * 20.0  # ~20 kg CO₂/yr per mature Pequi tree

        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#1a1a2e" rx="8" ry="8" '
                   f'stroke="#4fc3f7" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="22" fill="#4fc3f7" font-size="15" font-weight="bold">'
                   f'Dispersal Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#e0e0e0" font-size="15">'
                   f'Seeds dropped: {sim.seeds_dropped}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#76ff03" font-size="15">'
                   f'Germinated: {sim.germinated_count} ({germ_pct:.0f}%)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#ffcc80" font-size="15">'
                   f'Peak fruiting month: {peak_month_name}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90caf9" font-size="15">'
                   f'Best dispersal month: {months[best_disp_month]}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="116" fill="#a5d6a7" font-size="15">'
                   f'CO₂ sequestration potential: {co2_est:,.0f} kg/yr</text>')
        svg.append(f'<text font-weight="bold" x="12" y="134" fill="#78909c" font-size="15">'
                   f'(each mature Pequi ≈ 20 kg CO₂/yr)</text>')
        svg.append('</g>')

        # --- Panel 3: Seasonal curves mini-chart ---
        py3 = py2 + ph2 + 10
        ph3 = 156
        chart_w = panel_w - 32
        chart_h = 70
        chart_x0 = 16
        chart_y0 = 30

        svg.append(f'<g transform="translate({panel_x}, {py3})">')
        svg.append(f'<rect width="{panel_w}" height="{ph3}" fill="#1a1a2e" rx="8" ry="8" '
                   f'stroke="#7e57c2" stroke-width="1" opacity="0.92"/>')
        svg.append(f'<text x="12" y="20" fill="#7e57c2" font-size="15" font-weight="bold">'
                   f'Annual Curves</text>')

        # Draw 3 curves: fruiting (orange), moisture (blue), condition (pink)
        curves = [
            (PEQUI_FRUITING_CURVE, "#ff9800", "Fruiting"),
            (SOIL_MOISTURE_CURVE, "#42a5f5", "Moisture"),
            (TUCANO_CONDITION_CURVE, "#fe4db7", "Condition"),
        ]
        for curve_data, color, label in curves:
            pts = []
            for mi in range(12):
                px_c = chart_x0 + (mi / 11) * chart_w
                py_c = chart_y0 + chart_h - curve_data[mi] * chart_h
                pts.append(f"{px_c:.0f},{py_c:.0f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
                       f'stroke="{color}" stroke-width="2" opacity="0.8"/>')

        # Legend
        legend_y = chart_y0 + chart_h + 14
        for ci, (_, color, label) in enumerate(curves):
            lx = chart_x0 + ci * 110
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
    """Run ornithochory phenological clock simulation."""
    print(f" — Tucano-toco ↔ Pequi Ornithochory Clock on {CONFIG.device}...")

    sim = TucanoPequiSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    germ_pct = (sim.germinated_count / max(1, sim.seeds_dropped)) * 100
    print(f"Done: {sim.seeds_dropped} seeds dropped, {sim.germinated_count} germinated ({germ_pct:.0f}%)")
    print(f"Dispersal by month: {sim.dispersal_per_month}")

    print("Generating SVG...")
    renderer = OrnithochoryRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_55')
    return svg_content


if __name__ == "__main__":
    main()
