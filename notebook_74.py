# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 74: Andorinha-rabo-branco (Tachycineta leucorrhoa) ↔ Surto de Invertebrados do Cerrado
#            Trans-continental Migration ↔ Wet-Season Insect Bloom Clock
# INTERVENTION 4/4: Cerrado Ecological Web — Seasons & Migration Series
# ===================================================================================================
"""
notebook_74.py — White-rumped Swallow ↔ Cerrado Insect Bloom (Animal ↔ Animal):
Notebook Differentiation:
- Differentiation Focus: Andorinha-rabo-branco (Tachycineta leucorrhoa) ↔ Surto de Invertebrados do Cerrado emphasizing savanna grassland matrix.
- Indicator species: Tracaja (Podocnemis unifilis).
- Pollination lens: riparian flowering pulse after rains.
- Human impact lens: poaching risk hot spots.

                 Trans-continental Aerial Migration & Seasonal Invertebrate Pulse

The White-rumped Swallow (Tachycineta leucorrhoa) is a long-distance Neotropical
migrant that breeds in the temperate grasslands of southern South America
(Argentina, Uruguay, southern Brazil) during Oct–Mar and migrates northward,
reaching the Cerrado of central Goiás — including RESEX Recanto das Araras —
during the southern hemisphere autumn departure (Mar–May) and again on the return
passage (Aug–Oct).

What draws them to the RESEX during these two windows?  The Cerrado's wet-season
ending generates a MASSIVE invertebrate emergence pulse:

  WET-SEASON TERMITE SWARMS (Revoada — Oct–Nov): Heavy rains trigger simultaneous
  swarming flights of reproductive alate termites from thousands of mounds
  simultaneously. Tens of millions of winged termites fill the air in a cloud
  that lasts only 30–90 minutes but provides an enormous aerial food source for
  aerial insectivores.

  DIPTERA & EPHEMEROPTERA SURGE (Jan–Mar): Rising floodwaters in gallery forests
  and veredas trigger mass-emergence of aquatic flies, mayflies, and midges.

  FLYING ANT EMERGENCE (Sep–Oct): The first rains detonate nuptial flights of
  leaf-cutter ants (Atta spp.) and fire ants — a calorie-dense food source that
  coincides precisely with the southward Andorinha return passage.

Migratory circuit modelled:
  • NORTHWARD PASSAGE (Mar–May): Swallows stream through heading NW/N toward
    Mato Grosso and Amazon wintering grounds. They stop over the RESEX 2–3 weeks.
  • RESIDENCY GAP (Jun–Aug): Absent from RESEX — in northern wintering grounds.
  • SOUTHWARD RETURN (Aug–Oct): Return passage, this time flying SE. They
    linger longer (4–6 weeks) to exploit the termite revoada and flying-ant flush.
  • ABSENT (Nov–Feb): On breeding grounds in the southern Cone.

Connecting threads to the series (nb71–73):
  • The same termite swarms that draw Andinhas are controlled year-round by the
    Seriema (nb71) rotational predation — swarm density is partly regulated by
    this. The Seriema and the Andorinha are temporal PARTNERS in termite biocontrol.
  • The wet-season insect bloom is triggered by the same rainfall that floods the
    Buriti vereda (nb73) and signals the end of the Ipê bloom (nb72).
  • Thus all four notebooks share a single annual precipitation spine.

Spatial layout:
  A dual-lobe migration MAP rather than a pure clock. The clock still controls
  timing, but the MAIN ARENA shows:
    – Two migration corridors (NW and SE) as animated swallow streams through the
      RESEX territory midpoint.
    – Insect cloud pulses at specific seasonal peaks.
    – Territorial foraging inside the RESEX during stop-over windows.

Scientific references:
  • Marini & Garcia (2005): Bird extinction and fragmentation in the Cerrado.
  • Jahn et al. (2020): Migratory connectivity of austral migrants in South America.
  • Hölldobler & Wilson (1990): The Ants — nuptial flight synchrony.
  • Pinheiro et al. (2002): Termite swarming activity and rainfall (Cerrado).
  • Neves et al. (2010): Aerial insects in Cerrado gallery forests.
  • PIGT  observations, Goiás (2022–2024).
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from eco_base import save_svg, sanitize_svg_text, draw_phenology_chart , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES  (month 0 = January)
# ===================================================================================================

# Andorinha presence in RESEX territory (0 = absent, 1 = peak stop-over)
# Northward passage: Mar–May; Return passage: Aug–Oct
ANDORINHA_PRESENCE = [0.80, 0.70, 0.40, 0.15, 0.05, 0.00, 0.00, 0.05, 0.30, 0.70, 0.90, 0.95]
NORTH_PASSAGE_CURVE = [0.70, 0.75, 0.40, 0.15, 0.05, 0.00, 0.00, 0.05, 0.30, 0.70, 0.80, 0.85]

# Southward return corridor activity (Aug–Oct)
SOUTH_PASSAGE_CURVE = [0.10, 0.05, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 0.90, 0.70, 0.40, 0.20]

# Termite alate swarm (Revoada) — Oct–Nov first-rain trigger
TERMITE_SWARM_CURVE = [0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00, 0.05, 0.20, 0.60, 0.90, 0.50]

# Flying ant nuptial flights (Atta spp.) — Aug–Oct
FLYING_ANT_CURVE = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.60, 0.95, 0.80, 0.30, 0.00
]

# Diptera / aquatic insect emergence — Jan–Mar
DIPTERA_SURGE_CURVE = [
    0.85, 0.95, 0.80, 0.40, 0.15, 0.05, 0.00, 0.00, 0.05, 0.15, 0.40, 0.70
]

# Total aerial insect biomass (combined food availability)
INSECT_BIOMASS_CURVE = [
    0.55, 0.65, 0.55, 0.30, 0.10, 0.04, 0.04, 0.35, 0.65, 0.95, 0.90, 0.40
]

# Rainfall (common seasonal backbone linking all 4 notebooks)
RAINFALL_CURVE = [0.80, 0.70, 0.60, 0.30, 0.10, 0.05, 0.00, 0.00, 0.10, 0.50, 0.90, 0.95]


@dataclass
class AndinhaConfig:
    width:  int   = 1280
    height: int = CANVAS_HEIGHT
    frames: int   = 360
    fps:    int   = 10
    device: str   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clock geometry (consistent with nb71–73)
    clock_cx:     float = 420.0
    clock_cy:     float = 310.0
    clock_radius: float = 240.0

    # Migration corridor geometry
    # North corridor: enters bottom-left, exits top-right (of left panel)
    north_entry:  Tuple[float,float] = (90.0,  560.0)
    north_exit:   Tuple[float,float] = (760.0,  40.0)
    # South corridor: enters top-left, exits bottom-right
    south_entry:  Tuple[float,float] = (90.0,  40.0)
    south_exit:   Tuple[float,float] = (760.0, 560.0)

    num_swallows:    int   = 60   # total simulated swallow individuals
    num_insects:     int   = 120  # aerial insect agents (food pulse)
    swallow_speed:   float = 9.5  # fast aerial insectivore
    insect_speed:    float = 2.5


CONFIG = AndinhaConfig()


# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class AndinhaInsectSim:

    def __init__(self, cfg: AndinhaConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius

        # ── Swallows ──────────────────────────────────────────────────────────
        self.swallows: List[Dict] = []
        for k in range(cfg.num_swallows):
            # Divide into northward/southward/resident cohorts
            cohort = k % 3   # 0=north passage, 1=south passage, 2=local forager
            angle  = random.uniform(0, 2 * math.pi)
            r      = random.uniform(20, R - 20)
            start  = (cx + math.cos(angle) * r, cy + math.sin(angle) * r)
            self.swallows.append({
                "pos":    torch.tensor(start, device=self.dev, dtype=torch.float32),
                "vel":    torch.zeros(2, device=self.dev, dtype=torch.float32),
                "cohort": cohort,   # 0=north, 1=south, 2=local
                "state":  "absent",
                "energy": 80.0,
                "target_insect": -1,
                "catches": 0,
            })

        # ── Insects (aerial food cloud) ────────────────────────────────────────
        self.insects: List[Dict] = []
        for _ in range(cfg.num_insects):
            angle = random.uniform(0, 2 * math.pi)
            r     = random.uniform(15, R - 25)
            self.insects.append({
                "pos":   [cx + math.cos(angle) * r, cy + math.sin(angle) * r],
                "vel":   [random.uniform(-2, 2), random.uniform(-1.5, 1.5)],
                "alive": True,
                "type":  random.choice(["termite_alate", "flying_ant", "diptera"]),
            })

        self.hist_month:         List[float] = []
        self.hist_swallows_xy:   List[List[Tuple[float,float,float,int]]] = [
            [] for _ in range(cfg.num_swallows)
        ]  # (x, y, opacity, cohort)
        self.hist_insect_cloud:  List[Tuple[float,float,float]] = []  # aggregated cloud center + density

        self.total_insects_caught = 0
        self.total_stopover_frames  = 0   # swallow-frames inside RESEX territory
        self.peak_swallow_count     = 0
        self.north_passages         = 0
        self.south_passages         = 0

    def _interp(self, curve: list, month_frac: float) -> float:
        m  = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t  = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy = cfg.clock_cx, cfg.clock_cy
        month_frac = (frame / cfg.frames) * 12.0
        self.hist_month.append(month_frac)

        presence    = self._interp(ANDORINHA_PRESENCE,   month_frac)
        north_act   = self._interp(NORTH_PASSAGE_CURVE,  month_frac)
        south_act   = self._interp(SOUTH_PASSAGE_CURVE,  month_frac)
        insect_bm   = self._interp(INSECT_BIOMASS_CURVE, month_frac)
        swarm_lvl   = self._interp(TERMITE_SWARM_CURVE,  month_frac)
        ant_lvl     = self._interp(FLYING_ANT_CURVE,     month_frac)

        # Total aerial insects: spawn/despawn dynamically
        alive_insects = [ins for ins in self.insects if ins["alive"]]

        # ── Insect dynamics ────────────────────────────────────────────────────
        target_alive = int(cfg.num_insects * insect_bm)
        # Respawn dead insects proportional to biomass
        dead = [ins for ins in self.insects if not ins["alive"]]
        if len(alive_insects) < target_alive and dead:
            to_spawn = min(target_alive - len(alive_insects), len(dead), 8)
            for ins in dead[:to_spawn]:
                angle   = random.uniform(0, 2 * math.pi)
                r       = random.uniform(15, cfg.clock_radius - 25)
                ins["pos"]   = [cx + math.cos(angle) * r, cy + math.sin(angle) * r]
                ins["alive"] = True
                # Weight type by season
                if swarm_lvl > 0.4:
                    ins["type"] = "termite_alate"
                elif ant_lvl > 0.4:
                    ins["type"] = "flying_ant"
                else:
                    ins["type"] = "diptera"

        # Move each insect
        for ins in self.insects:
            if not ins["alive"]:
                continue
            ins["pos"][0] += ins["vel"][0] + random.uniform(-0.5, 0.5)
            ins["pos"][1] += ins["vel"][1] + random.uniform(-0.4, 0.4)
            # Clamp to clock arena
            dx = ins["pos"][0] - cx; dy = ins["pos"][1] - cy
            dr = math.sqrt(dx*dx + dy*dy)
            if dr > cfg.clock_radius - 15:
                angle_b = math.atan2(dy, dx) + math.pi + random.uniform(-0.4, 0.4)
                ins["vel"][0] = math.cos(angle_b) * cfg.insect_speed
                ins["vel"][1] = math.sin(angle_b) * cfg.insect_speed

        # Compute insect cloud centroid for overlay glyph
        alive_now = [ins for ins in self.insects if ins["alive"]]
        if alive_now:
            cloud_cx = sum(i["pos"][0] for i in alive_now) / len(alive_now)
            cloud_cy = sum(i["pos"][1] for i in alive_now) / len(alive_now)
            cloud_dens = len(alive_now) / cfg.num_insects
        else:
            cloud_cx, cloud_cy, cloud_dens = cx, cy, 0.0
        self.hist_insect_cloud.append((cloud_cx, cloud_cy, cloud_dens))

        # ── Swallow dynamics ───────────────────────────────────────────────────
        # Assign active counts per cohort
        north_active  = int(cfg.num_swallows / 3 * north_act)
        south_active  = int(cfg.num_swallows / 3 * south_act)
        local_active  = int(cfg.num_swallows / 3 * presence)

        active_count_this_frame = 0
        for ki, sw in enumerate(self.swallows):
            cohort = sw["cohort"]
            pos    = sw["pos"]

            if cohort == 0:   # northward passage cohort
                is_active = ki < north_active
            elif cohort == 1: # southward passage cohort
                is_active = (ki - cfg.num_swallows // 3) < south_active
            else:             # local forager
                is_active = (ki - 2 * cfg.num_swallows // 3) < local_active

            if not is_active:
                # Send off-screen
                pos[0] = cx - cfg.clock_radius - 80
                pos[1] = cy
                sw["state"] = "absent"
                self.hist_swallows_xy[ki].append((pos[0].item(), pos[1].item(), 0.0, cohort))
                continue

            active_count_this_frame += 1
            self.total_stopover_frames += 1
            sw["state"] = "active"
            sw["energy"] = max(0.0, sw["energy"] - 0.3)

            # --- Migration corridor logic (cohort 0 and 1)
            if cohort == 0:
                # Stream from south-left toward north-right (NW direction across arena)
                target = torch.tensor(cfg.north_exit, device=self.dev, dtype=torch.float32)
                dx_ = pos[0].item() - cx; dy_ = pos[1].item() - cy
                if pos[0].item() > cfg.north_exit[0] - 20:
                    # Wrap: re-enter from entry side
                    pos[0] = cfg.north_entry[0] + random.uniform(-30, 30)
                    pos[1] = cfg.north_entry[1] + random.uniform(-30, 30)
                    self.north_passages += 1
            elif cohort == 1:
                # Stream from north-left toward south-right (SE direction)
                target = torch.tensor(cfg.south_exit, device=self.dev, dtype=torch.float32)
                if pos[0].item() > cfg.south_exit[0] - 20:
                    pos[0] = cfg.south_entry[0] + random.uniform(-30, 30)
                    pos[1] = cfg.south_entry[1] + random.uniform(-30, 30)
                    self.south_passages += 1
            else:
                # Local forager: hunt insects inside clock
                target = None

            # --- Food-seeking override: if insect nearby, intercept
            sw["target_insect"] = -1
            if alive_now:
                best_di = 1e9; best_ii = -1
                for ii, ins in enumerate(alive_now):
                    dx_ = pos[0].item() - ins["pos"][0]
                    dy_ = pos[1].item() - ins["pos"][1]
                    d_  = math.sqrt(dx_*dx_ + dy_*dy_)
                    if d_ < best_di:
                        best_di = d_; best_ii = ii
                if best_di < 80:
                    sw["target_insect"] = best_ii

            if sw["target_insect"] != -1 and sw["target_insect"] < len(alive_now):
                ins_target = alive_now[sw["target_insect"]]
                tgt = torch.tensor(ins_target["pos"], device=self.dev, dtype=torch.float32)
                vec = tgt - pos; dist = torch.norm(vec).item()
                if dist > 5.0:
                    pos += (vec / max(dist, 1e-5)) * cfg.swallow_speed
                else:
                    # Catch!
                    ins_target["alive"] = False
                    sw["energy"] = min(100.0, sw["energy"] + 15.0)
                    sw["catches"] += 1
                    self.total_insects_caught += 1
            elif target is not None:
                # Follow migration corridor with jitter
                vec = target - pos; dist = torch.norm(vec).item()
                jitter = torch.randn(2, device=self.dev) * 8.0
                pos += (vec / max(dist, 1e-5)) * cfg.swallow_speed + jitter
            else:
                # Local patrol: swoop around clock arena
                center_pull = torch.tensor([cx, cy], device=self.dev, dtype=torch.float32)
                pull = (center_pull - pos) * 0.01
                pos += pull + torch.randn(2, device=self.dev) * (cfg.swallow_speed * 0.8)

            # Clamp to arena (migration cohorts allowed to exit right side — they wrap above)
            if cohort == 2:
                dx_ = pos[0].item() - cx; dy_ = pos[1].item() - cy
                dr_ = math.sqrt(dx_*dx_ + dy_*dy_)
                if dr_ > cfg.clock_radius - 12:
                    pos[0] = cx + (dx_/dr_) * (cfg.clock_radius - 12)
                    pos[1] = cy + (dy_/dr_) * (cfg.clock_radius - 12)
            else:
                # Allow migration birds to move across full left panel
                pos[0] = pos[0].clamp(20, 800)
                pos[1] = pos[1].clamp(20, cfg.height - 20)

            self.hist_swallows_xy[ki].append((pos[0].item(), pos[1].item(), 1.0, cohort))

        self.peak_swallow_count = max(self.peak_swallow_count, active_count_this_frame)


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class AndinhaRenderer:

    def __init__(self, cfg: AndinhaConfig, sim: AndinhaInsectSim):
        self.cfg = cfg
        self.sim = sim

    def _arc_path(self, cx, cy, radius, start_m, end_m):
        a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
        a2 = (end_m   / 12) * 2 * math.pi - math.pi / 2
        x1 = cx + math.cos(a1) * radius; y1 = cy + math.sin(a1) * radius
        x2 = cx + math.cos(a2) * radius; y2 = cy + math.sin(a2) * radius
        span  = (end_m - start_m) % 12
        large = 1 if span > 6 else 0
        return f"M {x1:.0f} {y1:.0f} A {radius} {radius} 0 {large} 1 {x2:.0f} {y2:.0f}"

    def generate_svg(self) -> str:
        cfg  = self.cfg
        sim  = self.sim
        w, h = cfg.width, cfg.height
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R    = cfg.clock_radius
        F    = cfg.frames
        dur  = F / cfg.fps

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:#05080f; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        # ── Defs ──────────────────────────────────────────────────────────────
        svg.append('<defs>')
        svg.append(
            '<radialGradient id="skyBg">'
            '<stop offset="0%"   stop-color="#0d1b35" stop-opacity="0.95"/>'
            '<stop offset="65%"  stop-color="#080f22" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#05080f" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="swarmGlow">'
            '<stop offset="0%"   stop-color="#ffca28" stop-opacity="0.85"/>'
            '<stop offset="50%"  stop-color="#ffb300" stop-opacity="0.35"/>'
            '<stop offset="100%" stop-color="#ff8f00" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append(
            '<radialGradient id="antGlow">'
            '<stop offset="0%"   stop-color="#e53935" stop-opacity="0.80"/>'
            '<stop offset="100%" stop-color="#e53935" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        # Migration corridor gradients
        svg.append(
            '<linearGradient id="northCorr" x1="0" y1="1" x2="1" y2="0">'
            '<stop offset="0%"   stop-color="#80deea" stop-opacity="0.0"/>'
            '<stop offset="40%"  stop-color="#80deea" stop-opacity="0.25"/>'
            '<stop offset="100%" stop-color="#80deea" stop-opacity="0.0"/>'
            '</linearGradient>'
        )
        svg.append(
            '<linearGradient id="southCorr" x1="0" y1="0" x2="1" y2="1">'
            '<stop offset="0%"   stop-color="#ef9a9a" stop-opacity="0.0"/>'
            '<stop offset="40%"  stop-color="#ef9a9a" stop-opacity="0.25"/>'
            '<stop offset="100%" stop-color="#ef9a9a" stop-opacity="0.0"/>'
            '</linearGradient>'
        )
        svg.append(
            '<filter id="glowFilter">'
            '<feGaussianBlur stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
        )
        svg.append('</defs>')

        # ── Background ────────────────────────────────────────────────────────
        svg.append(f'<rect width="{w}" height="{h}" fill="#05080f"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R + 55}" fill="url(#skyBg)"/>')

        # Insect swarm density glow on the arena background
        swarm_fills = ";".join(
            f"rgba(255,202,40,{sim._interp(TERMITE_SWARM_CURVE, (f/F)*12) * 0.18:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{swarm_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # Flying ant surge (red glow)
        ant_fills = ";".join(
            f"rgba(239,83,80,{sim._interp(FLYING_ANT_CURVE, (f/F)*12) * 0.14:.2f})"
            for f in range(F)
        )
        svg.append(
            f'<circle cx="{cx}" cy="{cy}" r="{R - 8}" fill="transparent">'
            f'<animate attributeName="fill" values="{ant_fills}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # ── Migration corridors (background ribbons) ──────────────────────────
        ne = cfg.north_entry; nx_ = cfg.north_exit
        # North corridor band (south-left → north-right)
        north_op_vals = ";".join(
            f"{sim._interp(NORTH_PASSAGE_CURVE, (f/F)*12) * 0.6:.2f}" for f in range(F)
        )
        svg.append(
            f'<polygon points="{ne[0]-25},{ne[1]+25} {ne[0]+25},{ne[1]-25} '
            f'{nx_[0]+25},{nx_[1]+25} {nx_[0]-25},{nx_[1]-25}" '
            f'fill="url(#northCorr)" opacity="0">'
            f'<animate attributeName="opacity" values="{north_op_vals}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</polygon>'
        )
        svg.append(
            f'<line x1="{ne[0]}" y1="{ne[1]}" x2="{nx_[0]}" y2="{nx_[1]}" '
            f'stroke="#80deea" stroke-width="1.5" stroke-dasharray="8,6" opacity="0.3"/>'
        )
        svg.append(
            f'<text font-weight="bold" x="{(ne[0]+nx_[0])//2 - 60}" y="{(ne[1]+nx_[1])//2}" '
            f'font-size="15" fill="#80deea" opacity="0.7" '
            f'transform="rotate(-35,{(ne[0]+nx_[0])//2},{(ne[1]+nx_[1])//2})">'
            f'↑ Mar–May northward</text>'
        )

        se = cfg.south_entry; sx_ = cfg.south_exit
        south_op_vals = ";".join(
            f"{sim._interp(SOUTH_PASSAGE_CURVE, (f/F)*12) * 0.6:.2f}" for f in range(F)
        )
        svg.append(
            f'<polygon points="{se[0]-25},{se[1]-25} {se[0]+25},{se[1]+25} '
            f'{sx_[0]+25},{sx_[1]-25} {sx_[0]-25},{sx_[1]+25}" '
            f'fill="url(#southCorr)" opacity="0">'
            f'<animate attributeName="opacity" values="{south_op_vals}" '
            f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</polygon>'
        )
        svg.append(
            f'<line x1="{se[0]}" y1="{se[1]}" x2="{sx_[0]}" y2="{sx_[1]}" '
            f'stroke="#ef9a9a" stroke-width="1.5" stroke-dasharray="8,6" opacity="0.3"/>'
        )
        svg.append(
            f'<text font-weight="bold" x="{(se[0]+sx_[0])//2 - 60}" y="{(se[1]+sx_[1])//2}" '
            f'font-size="15" fill="#ef9a9a" opacity="0.7" '
            f'transform="rotate(35,{(se[0]+sx_[0])//2},{(se[1]+sx_[1])//2})">'
            f'↘ Aug–Oct southward</text>'
        )

        # ── Title ─────────────────────────────────────────────────────────────
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#80deea" font-weight="bold">'
            f'ECO-SIM: Swallow × Invertebrates    - Trans-continental Migration Clock</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#b2ebf2">'
            f'Migratory stopover windows locked to wet-season insect bloom | </text>'
        )

        # ── Clock Face ────────────────────────────────────────────────────────
        months = ["JAN","FEB","MAR","APR","MAY","JUN",
                  "JUL","AUG","SEP","OCT","NOV","DEC"]
        # Month colours: cyan=northward, salmon=southward, amber=swarm, grey=absent
        month_cols = {
            0:"#80deea", 1:"#80deea",          # Jan–Feb: Diptera peak
            2:"#80cbc4", 3:"#4dd0e1", 4:"#80cbc4",   # Mar–May: northward
            5:"#546e7a", 6:"#546e7a", 7:"#ef9a9a",   # Jun–Jul absent, Aug return starts
            8:"#ef9a9a", 9:"#ffca28", 10:"#ff8f00",  # Sep–Nov: return + swarm
            11:"#80deea"                               # Dec
        }
        for i, m in enumerate(months):
            angle = (i / 12) * 2 * math.pi - math.pi / 2
            tx = cx + math.cos(angle) * (R + 35)
            ty = cy + math.sin(angle) * (R + 35)
            svg.append(
                f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="{month_cols[i]}" '
                f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>'
            )
            lx1 = cx + math.cos(angle) * R;     ly1 = cy + math.sin(angle) * R
            lx2 = cx + math.cos(angle) * (R-6); ly2 = cy + math.sin(angle) * (R-6)
            svg.append(
                f'<line x1="{lx1:.0f}" y1="{ly1:.0f}" x2="{lx2:.0f}" y2="{ly2:.0f}" '
                f'stroke="#0d1b35" stroke-width="2"/>'
            )

        # ── Season Arcs ───────────────────────────────────────────────────────
        # Northward passage window (Mar–May)
        d1 = self._arc_path(cx, cy, R + 10, 2, 5)
        svg.append(f'<path d="{d1}" fill="none" stroke="#80deea" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.55"/>')
        mid_n = ((2+5)/2/12)*2*math.pi - math.pi/2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid_n)*(R+20):.0f}" '
                   f'y="{cy+math.sin(mid_n)*(R+20):.0f}" font-size="15" '
                   f'fill="#80deea" text-anchor="middle">↑ N. Passage</text>')

        # Absence gap (Jun–Aug)
        d2 = self._arc_path(cx, cy, R + 10, 5, 7.5)
        svg.append(f'<path d="{d2}" fill="none" stroke="#546e7a" stroke-width="7" '
                   f'stroke-linecap="round" opacity="0.40" stroke-dasharray="5,4"/>')
        mid_ab = ((5+7.5)/2/12)*2*math.pi - math.pi/2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid_ab)*(R+20):.0f}" '
                   f'y="{cy+math.sin(mid_ab)*(R+20):.0f}" font-size="15" '
                   f'fill="#546e7a" text-anchor="middle">⬆ Absent S→N</text>')

        # Southward return passage (Aug–Oct)
        d3 = self._arc_path(cx, cy, R + 10, 7.5, 11)
        svg.append(f'<path d="{d3}" fill="none" stroke="#ef9a9a" stroke-width="9" '
                   f'stroke-linecap="round" opacity="0.55"/>')
        mid_s = ((7.5+11)/2/12)*2*math.pi - math.pi/2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid_s)*(R+20):.0f}" '
                   f'y="{cy+math.sin(mid_s)*(R+20):.0f}" font-size="15" '
                   f'fill="#ef9a9a" text-anchor="middle">↘ S. Passage</text>')

        # Termite Revoada peak (Oct–Nov)
        d4 = self._arc_path(cx, cy, R + 24, 9.5, 11.5)
        svg.append(f'<path d="{d4}" fill="none" stroke="#ffca28" stroke-width="7" '
                   f'stroke-linecap="round" opacity="0.70"/>')
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(((9.5+11.5)/2/12)*2*math.pi-math.pi/2)*(R+34):.0f}" '
                   f'y="{cy+math.sin(((9.5+11.5)/2/12)*2*math.pi-math.pi/2)*(R+34):.0f}" '
                   f'font-size="15" fill="#ffca28" text-anchor="middle">Revoada</text>')

        # Flying ant nuptial (Aug–Oct)
        d5 = self._arc_path(cx, cy, R + 24, 7, 10)
        svg.append(f'<path d="{d5}" fill="none" stroke="#ef5350" stroke-width="5" '
                   f'stroke-linecap="round" opacity="0.55" stroke-dasharray="4,3"/>')

        # Clock ring
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" '
                   f'stroke="#0d1b35" stroke-width="1.5" opacity="0.8"/>')

        # ── Insects (animated dots, type-coloured) ────────────────────────────
        # We use the insect cloud density to drive a single pulsing glow + individual dots
        cloud_rs   = ";".join(str(round(sim.hist_insect_cloud[fi][2] * 70, 1)) for fi in range(F))
        cloud_ops  = ";".join(f"{sim.hist_insect_cloud[fi][2] * 0.35:.2f}"   for fi in range(F))
        cloud_cxs  = ";".join(str(round(sim.hist_insect_cloud[fi][0], 1))    for fi in range(F))
        cloud_cys  = ";".join(str(round(sim.hist_insect_cloud[fi][1], 1))    for fi in range(F))
        svg.append(
            f'<circle fill="url(#swarmGlow)" opacity="0">'
            f'<animate attributeName="cx" values="{cloud_cxs}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{cloud_cys}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="r"  values="{cloud_rs}"  dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<animate attributeName="opacity" values="{cloud_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )

        # Sample individual insects for animated dots (limit for SVG size)
        max_insect_dots = 40
        for ii, ins in enumerate(sim.insects[:max_insect_dots]):
            # Build per-frame alive mask from biomass curve sampling
            ins_op_vals_list = []
            for fi in range(F):
                bm = sim._interp(INSECT_BIOMASS_CURVE, (fi/F)*12)
                # Probabilistic per-insect: show ~bm fraction
                show = 1.0 if (ii / max_insect_dots) < bm else 0.0
                # Add slight phase offset per insect
                show *= 0.85 if random.random() > 0.2 else 0.0
                ins_op_vals_list.append(f"{show:.2f}")
            ins_ops = ";".join(ins_op_vals_list)

            col = "#ffca28" if ins["type"] == "termite_alate" else (
                  "#ef5350" if ins["type"] == "flying_ant" else "#b2ebf2")

            # Use starting position and random drift key-frames
            px0, py0 = ins["pos"]
            # Build a small animation path (4 keyframes for performance)
            xs = f"{px0:.0f};{px0+random.uniform(-20,20):.0f};{px0+random.uniform(-30,30):.0f};{px0:.0f}"
            ys = f"{py0:.0f};{py0+random.uniform(-15,15):.0f};{py0+random.uniform(-20,20):.0f};{py0:.0f}"
            svg.append(
                f'<circle r="2.5" fill="{col}" opacity="0">'
                f'<animate attributeName="cx" values="{xs}" dur="{random.uniform(1.5,4):.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{ys}" dur="{random.uniform(1.5,4):.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ins_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # ── Swallows ──────────────────────────────────────────────────────────
        cohort_cols = ["#80deea", "#ef9a9a", "#e0f2f1"]  # north=cyan, south=salmon, local=white
        for ki in range(cfg.num_swallows):
            hist = sim.hist_swallows_xy[ki]
            if not hist:
                continue
            cxs  = ";".join(str(round(h[0], 1)) for h in hist)
            cys  = ";".join(str(round(h[1], 1)) for h in hist)
            ops  = ";".join(str(round(h[2], 2)) for h in hist)
            cohort_id = hist[0][3]
            col  = cohort_cols[cohort_id % 3]

            # Flight trail (every 6th frame for perf)
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for h in hist[::6] if h[2] > 0.5]
            if len(trail_pts) > 3:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="{col}" stroke-width="0.6" opacity="0.20"/>'
                )

            # Swallow body (small streamlined ellipse — long axis oriented to velocity)
            svg.append(
                f'<ellipse rx="6" ry="2.5" fill="{col}" opacity="0.9">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )
            # White rump patch (distinctive field mark)
            svg.append(
                f'<ellipse rx="2" ry="2" fill="#eceff1" opacity="0.85">'
                f'<animate attributeName="cx" values="'
                + ";".join(str(round(h[0] - 4, 1)) for h in hist)
                + f'" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" '
                f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )

        # ── Clock Hand ────────────────────────────────────────────────────────
        hand_x = ";".join(
            str(round(cx + math.cos((m/12)*2*math.pi - math.pi/2)*(R - 10), 1))
            for m in sim.hist_month
        )
        hand_y = ";".join(
            str(round(cy + math.sin((m/12)*2*math.pi - math.pi/2)*(R - 10), 1))
            for m in sim.hist_month
        )
        svg.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy - R + 10}" '
            f'stroke="#80deea" stroke-width="2.5" stroke-linecap="round" opacity="0.9">'
            f'<animate attributeName="x2" values="{hand_x}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="y2" values="{hand_y}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</line>'
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="14" fill="#05080f" stroke="#80deea" stroke-width="2"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#29b6f6"/>')

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # RIGHT PANELS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        panel_x = w - 390
        panel_w = 370

        # ── Panel 1: Ecological Logic ──────────────────────────────────────────
        py1, ph1 = 20, 297
        svg.append(f'<g transform="translate({panel_x}, {py1})">')
        svg.append(f'<rect width="{panel_w}" height="{ph1}" fill="#030811" rx="8" '
                   f'stroke="#0288d1" stroke-width="1" opacity="0.94"/>')
        svg.append(f'<text x="12" y="22" fill="#80deea" font-size="15" font-weight="bold">'
                   f'Migratory Stop-over Logic</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#ccc" font-size="15">'
                   f'T. leucorrhoa breeds in S. Cone (Oct–Mar)</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#ccc" font-size="15">'
                   f'and migrates N/NW to Cerrado wintering areas.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#ccc" font-size="15">'
                   f'over  TWICE per year, each time</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#ccc" font-size="15">'
                   f'exploiting a different insect emergence event.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#ffca28" font-size="15">'
                   f'Termite Revoada (Oct–Nov): alates fill the air</text>')
        svg.append(f'<text font-weight="bold" x="12" y="112" fill="#ef9a9a" font-size="15">'
                   f'Atta spp. nuptial flights (Aug–Sep): fat-rich</text>')
        svg.append(f'<text font-weight="bold" x="12" y="126" fill="#b2ebf2" font-size="15">'
                   f'Diptera at vereda flooding (Jan–Mar): midges.</text>')
        svg.append('</g>')

        # ── Panel 2: Metrics ──────────────────────────────────────────────────
        py2 = py1 + ph1 + 10
        ph2 = 156
        svg.append(f'<g transform="translate({panel_x}, {py2})">')
        svg.append(f'<rect width="{panel_w}" height="{ph2}" fill="#030811" rx="8" '
                   f'stroke="#37474f" stroke-width="1" opacity="0.94"/>')
        svg.append(f'<text x="12" y="22" fill="#90a4ae" font-size="15" font-weight="bold">'
                   f'Migration & Predation Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#80deea" font-size="15">'
                   f'Total Insects Caught: {sim.total_insects_caught:,}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#b2ebf2" font-size="15">'
                   f'Stopovers in RESEX: {sim.total_stopover_frames:,}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#80deea" font-size="15">'
                   f'Peak swallows present: {sim.peak_swallow_count} individuals</text>')
        svg.append(f'<text font-weight="bold" x="12" y="98" fill="#90a4ae" font-size="15">'
                   f'N-passages: {sim.north_passages} | S-passages: {sim.south_passages} | '
                   f'Birds: {cfg.num_swallows}</text>')
        svg.append('</g>')

        # ── Panel 3: Phenology Chart (eco_base helper) ────────────────────────
        py3 = py2 + ph2 + 10
        ph3 = 183
        curves_data = [
            (ANDORINHA_PRESENCE,   "#80deea", "Andorinha in RESEX"),
            (INSECT_BIOMASS_CURVE, "#ffca28", "Aerial Insect Biomass"),
            (TERMITE_SWARM_CURVE,  "#ff8f00", "Termite Revoada"),
            (FLYING_ANT_CURVE,     "#ef5350", "Flying Ants (Atta)"),
            (DIPTERA_SURGE_CURVE,  "#b2ebf2", "Diptera / Aquatic Insects"),
        ]
        chart_snippet = draw_phenology_chart(
            curves_data,
            chart_w=330, chart_h=58, panel_h=ph3,
            title="Migratory &amp; Insect Bloom Phenology",
            title_color="#80deea",
            bg_color="#030811",
            border_color="#0288d1",
        )
        svg.append(f'<g transform="translate({panel_x}, {py3})">{chart_snippet}</g>')

        # ── Current Month Sidebar (animated status) ───────────────────────────
        px5 = 20; py5 = h - 238; pw5 = 252; ph5 = 228
        svg.append(f'<g transform="translate({px5}, {py5})">')
        svg.append(f'<rect width="{pw5}" height="{ph5}" fill="#030811" rx="8" '
                   f'stroke="#0288d1" stroke-width="1.5" opacity="0.97"/>')
        svg.append(f'<text x="12" y="22" font-size="15" fill="#80deea" font-weight="bold">'
                   f'Active Season Status:</text>')

        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12; vals[m_idx] = "1"
            op_str = ";".join(vals + ["0"])

            svg.append(f'<text x="12" y="50" font-size="15" fill="#80deea" font-weight="bold">')
            svg.append(m_name)
            svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                       f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')

            presence = sim._interp(ANDORINHA_PRESENCE,   m_idx)
            north    = sim._interp(NORTH_PASSAGE_CURVE,  m_idx)
            south    = sim._interp(SOUTH_PASSAGE_CURVE,  m_idx)
            swarm    = sim._interp(TERMITE_SWARM_CURVE,  m_idx)
            ant      = sim._interp(FLYING_ANT_CURVE,     m_idx)
            diptera  = sim._interp(DIPTERA_SURGE_CURVE,  m_idx)

            if north > 0.4:
                st1, c1 = "NORTHWARD PASSAGE",          "#80deea"
                st2, c2 = f"Heading to Amazon wintering",  "#b2ebf2"
                st3, c3 = f"Diptera food: {diptera*100:.0f}%", "#b2ebf2"
            elif south > 0.4:
                st1, c1 = "SOUTHWARD RETURN",           "#ef9a9a"
                st2, c2 = f"Termite revoada: {swarm*100:.0f}%","#ffca28"
                st3, c3 = f"Flying ants: {ant*100:.0f}%", "#ef5350"
            elif swarm  > 0.6:
                st1, c1 = "REVOADA PEAK — Feast!",     "#ffca28"
                st2, c2 = "Millions of alates airborne",   "#ff8f00"
                st3, c3 = f"Birds present: {presence*100:.0f}%","#80deea"
            elif presence < 0.1:
                st1, c1 = "⬆ Birds absent — wintering",   "#546e7a"
                st2, c2 = "Breeding grounds in S. Cone",  "#78909c"
                st3, c3 = "Insect populations recovering","#80cbc4"
            else:
                st1, c1 = "Local foraging stop-over",  "#4dd0e1"
                st2, c2 = f"Presence: {presence*100:.0f}%","#80deea"
                st3, c3 = f"Biomass: {(swarm+ant+diptera)/3*100:.0f}%","#ffca28"

            for yoff, txt, col in [(76, st1, c1), (96, st2, c2), (114, st3, c3)]:
                svg.append(f'<text x="12" y="{yoff}" font-size="15" fill="{col}" font-weight="bold">')
                svg.append(txt)
                svg.append(f'<animate attributeName="opacity" values="{op_str}" '
                           f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append('</text>')

        # Static legend
        svg.append('<text x="12" y="138" fill="#546e7a" font-size="15" font-weight="bold">Legend:</text>')
        entries = [
            (22, 154, "#80deea",  "Andorinha northward passage cohort"),
            (22, 170, "#ef9a9a",  "Andorinha southward return cohort"),
            (22, 186, "#e0f2f1",  "Local forager (resident stop-over)"),
            (22, 202, "#ffca28",  "Termite alate (Revoada)"),
            (22, 218, "#ef5350",  "Flying ant Atta spp."),
        ]
        for (ex,ey,ec,elabel) in entries:
            svg.append(f'<ellipse cx="{ex}" cy="{ey}" rx="6" ry="2.5" fill="{ec}" opacity="0.9"/>')
            svg.append(f'<text font-weight="bold" x="36" y="{ey+4}" fill="{ec}" font-size="15">{elabel}</text>')
        svg.append('</g>')

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    print(f" — Andorinha-rabo-branco ↔ Invertebrados (Migration Clock) on {CONFIG.device}...")

    sim = AndinhaInsectSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_insects_caught:,} insects caught, "
          f"{sim.total_stopover_frames:,} Stopovers in RESEX, "
          f"peak {sim.peak_swallow_count} individuals, "
          f"N-passages {sim.north_passages} / S-passages {sim.south_passages}.")

    print("Generating SVG...")
    renderer = AndinhaRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_74')
    return svg_content


if __name__ == "__main__":
    main()
