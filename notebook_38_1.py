# -*- coding: utf-8 -*-
# pyre-ignore-all-errors
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 38.1: Enhanced Climate Change Impacts with Cascading Effects
# DERIVATIVE OF: notebook_38.py (Enhanced with tipping points and ecosystem thresholds)
# ===================================================================================================

"""
Notebook Differentiation:
- Differentiation Focus: Enhanced Climate Change Impacts with Cascading Effects emphasizing storm event disruptions.
- Indicator species: Arnica-do-cerrado (Lychnophora ericoides).
- Pollination lens: riparian flowering pulse after rains.
- Human impact lens: poaching risk hot spots.
"""

from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

# ===================================================================================================
# 1. SCIENTIFIC CONTEXT & PARAMETERS
# ===================================================================================================

@dataclass
class ClimateChangeConfig:
    width: int = 1280
    height: int = 602
    frames: int = 300
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_habitats: int = 12

CONFIG = ClimateChangeConfig()

# ===================================================================================================
# 2. SPECIES & HABITAT DATABASE
# ===================================================================================================

class HabitatData(TypedDict):
    color: str
    glow: str
    temp_optimal: float
    temp_tolerance: float
    drought_tolerance: float
    fire_resistance: float
    name: str


class SpeciesData(TypedDict):
    color: str
    temp_preference: Tuple[float, float]
    habitat_preference: List[str]
    name: str


class ClimateEvent(TypedDict):
    species: str
    frame: int
    viability: float


class MigrationEvent(TypedDict):
    species: str
    frame: int
    from_viability: float


class Snapshot(TypedDict):
    temp: float
    precip: float
    fire_risk: float
    habitats: List[Tuple[float, float, bool]]


HABITAT_DB: Dict[str, HabitatData] = {
    'cerrado': {
        'color': '#8bc34a',
        'glow': '#a4d65a',
        'temp_optimal': 28,
        'temp_tolerance': 8,
        'drought_tolerance': 0.6,
        'fire_resistance': 0.5,
        'name': 'Cerrado'
    },
    'mata_seca': {
        'color': '#4caf50',
        'glow': '#66c066',
        'temp_optimal': 26,
        'temp_tolerance': 6,
        'drought_tolerance': 0.4,
        'fire_resistance': 0.3,
        'name': 'Mata Seca'
    },
    'karst': {
        'color': '#9c27b0',
        'glow': '#ba55c8',
        'temp_optimal': 25,
        'temp_tolerance': 10,
        'drought_tolerance': 0.8,
        'fire_resistance': 0.7,
        'name': 'Karst'
    },
    'vereda': {
        'color': '#00bcd4',
        'glow': '#40d4e8',
        'temp_optimal': 27,
        'temp_tolerance': 5,
        'drought_tolerance': 0.2,
        'fire_resistance': 0.2,
        'name': 'Vereda'
    }
}

SPECIES_DB: Dict[str, SpeciesData] = {
    'tucano': {
        'color': '#fe4db7',
        'temp_preference': (24, 32),
        'habitat_preference': ['mata_seca', 'cerrado'],
        'name': 'Tucano-toco'
    },
    'gralha': {
        'color': '#00ffdc',
        'temp_preference': (20, 35),
        'habitat_preference': ['cerrado', 'karst'],
        'name': 'Gralha-do-campo'
    },
    'beija_flor': {
        'color': '#ffe43f',
        'temp_preference': (22, 34),
        'habitat_preference': ['vereda', 'mata_seca'],
        'name': 'Beija-flor-tesoura'
    },
    'onca': {
        'color': '#ffeb3b',
        'temp_preference': (20, 32),
        'habitat_preference': ['mata_seca', 'karst'],
        'name': 'Onca-pintada'
    }
}

# ===================================================================================================
# 2b. POPULATION ECOLOGY PARAMETERS  (100-500 ha Cerrado reserve scale)
# ===================================================================================================
# Scientific basis:
#   Onça-pintada  – Silveira et al. (2014) camera-trap surveys: 1-3 ind./500 km²;
#                   typical home range 25-50 km²; small reserves hold 1-3 transient/resident animals.
#   Tucano-toco   – territorial pairs + juveniles; ~5-10 territories per 100 ha semi-open cerrado;
#                   carrying capacity set by fruiting-tree density.
#   Gralha-do-campo (Cyanocorax cristatellus) – cooperative breeder, groups of 4-8;
#                   several family groups sustainable per 100 ha open cerrado.
#   Beija-flor-tesoura (Eupetomena macroura) – 5-8 territories per 100 ha near veredas;
#                   constrained by nectar and nesting site availability.

# Initial population ranges (individuals) for a 100-500 ha reserve
SPECIES_POP_RANGES: Dict[str, Tuple[int, int]] = {
    'tucano':     (20, 45),   # Toco Toucan: territorial frugivore
    'gralha':     (30, 55),   # Campo Jay: gregarious, higher density per area
    'beija_flor': (10, 28),   # Swallow-tailed Hummingbird: nectar-limited pollinator
    'onca':       (1, 3),     # Jaguar: apex predator, mega-territory requirement
}

# Hard carrying-capacity ceiling K (logistic model)
SPECIES_CARRYING_CAPACITY: Dict[str, int] = {
    'tucano':     50,   # fruiting-tree density ceiling
    'gralha':     60,   # open cerrado social group ceiling
    'beija_flor': 30,   # nectar-resource ceiling in vereda/gallery
    'onca':        3,   # territorial apex predator; ≥25 km² per individual
}

# Per-frame intrinsic growth rate r (logistic); populations approach K over ~150-200 frames
SPECIES_GROWTH_RATE: Dict[str, float] = {
    'tucano':     0.05,   # ~1 breeding season/year, clutch 2-4 eggs
    'gralha':     0.06,   # cooperative breeder, higher λ
    'beija_flor': 0.04,   # 2 eggs/clutch, multiple seasons
    'onca':       0.008,  # 1-4 cubs per 2-year cycle; very slow demographic turnover
}

# Minimum count below which local extinction is declared (species-specific)
SPECIES_EXTINCTION_THRESHOLD: Dict[str, int] = {
    'tucano':     5,
    'gralha':     5,
    'beija_flor': 5,
    'onca':       1,   # single individual still counts as presence in reserve
}

# ===================================================================================================
# 3. ENHANCED CLIMATE CHANGE MODEL
# ===================================================================================================

class HabitatNode:
    def __init__(self, x: float, y: float, habitat_type: str):
        self.x = x
        self.y = y
        self.habitat_type = habitat_type
        self.data: HabitatData = HABITAT_DB[habitat_type]
        self.health = 1.0
        self.biomass = random.uniform(80, 150)
        self.water_level = 1.0
        self.tip_reached = False
        self.tip_year: Optional[int] = None
        self.stress_history: List[float] = []

class SpeciesPopulation:
    def __init__(self, species_type: str):
        self.species_type = species_type
        self.data: SpeciesData = SPECIES_DB[species_type]
        # Species-specific realistic initial count for a 100-500 ha Cerrado reserve
        self.population: float = float(random.randint(*SPECIES_POP_RANGES[species_type]))
        self.viability = 1.0

class ClimateChangeModel:
    def __init__(self, cfg: ClimateChangeConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        self.habitats: List[HabitatNode] = []
        habitat_types = list(HABITAT_DB.keys())

        for _ in range(cfg.num_habitats):
            x = random.uniform(100, cfg.width - 100)
            y = random.uniform(100, cfg.height - 100)
            htype = random.choice(habitat_types)
            self.habitats.append(HabitatNode(x, y, htype))

        self.species: List[SpeciesPopulation] = []
        for stype in SPECIES_DB.keys():
            self.species.append(SpeciesPopulation(stype))

        self.baseline_temp = 26.0
        self.current_temp = self.baseline_temp
        self.precipitation = 1.0
        self.drought_intensity = 0.0
        self.fire_risk = 0.0

        self.tipping_points_reached = 0
        self.extinction_events: List[ClimateEvent] = []
        self.migration_events: List[MigrationEvent] = []

        self.temp_history: List[float] = []
        self.health_history: List[float] = []
        self.precip_history: List[float] = []
        self.fire_risk_history: List[float] = []
        self.species_pop_history: Dict[str, List[int]] = {s: [] for s in SPECIES_DB.keys()}
        self.snapshots: List[Snapshot] = []

    def step(self, frame: int):
        temp_increase = (frame / self.cfg.frames) * 6.0
        self.current_temp = self.baseline_temp + temp_increase

        precip_decrease = (frame / self.cfg.frames) * 0.5
        self.precipitation = max(0.2, 1.0 - precip_decrease)

        if frame % 50 == 25:
            self.drought_intensity = 0.8
        else:
            self.drought_intensity = max(0.0, self.drought_intensity - 0.05)

        self.fire_risk = (self.current_temp - self.baseline_temp) / 6.0 * self.drought_intensity

        habitat_health_total = 0.0

        for habitat in self.habitats:
            temp_diff = abs(self.current_temp - habitat.data['temp_optimal'])
            temp_stress = max(0.0, (temp_diff - habitat.data['temp_tolerance']) / habitat.data['temp_tolerance'])

            drought_stress = (1 - habitat.data['drought_tolerance']) * (1 - self.precipitation)

            fire_damage = 0
            if random.random() < self.fire_risk * 0.1:
                fire_damage = (1 - habitat.data['fire_resistance']) * random.uniform(0.1, 0.3)

            habitat.stress_history.append(temp_stress + drought_stress)
            total_stress = temp_stress + drought_stress + fire_damage

            habitat.health = min(1.0, max(0.0, habitat.health - total_stress * 0.02 + 0.005))

            habitat.water_level = self.precipitation * random.uniform(0.7, 1.0)

            habitat.biomass *= (0.99 + habitat.health * 0.02)

            if not habitat.tip_reached and habitat.health < 0.3:
                habitat.tip_reached = True
                habitat.tip_year = frame
                self.tipping_points_reached += 1

            habitat_health_total += habitat.health

        for species in self.species:
            temp_min, temp_max = species.data['temp_preference']
            matching_habitat_health: List[float] = []

            for habitat in self.habitats:
                if habitat.habitat_type in species.data['habitat_preference']:
                    if temp_min <= self.current_temp <= temp_max:
                        matching_habitat_health.append(habitat.health)

            suitable_habitats = sum(matching_habitat_health)
            preference_count = max(1.0, float(len(species.data['habitat_preference'])))
            viability = suitable_habitats / preference_count
            species.viability = max(0.0, min(1.0, viability))

            K = float(SPECIES_CARRYING_CAPACITY[species.species_type])
            r = SPECIES_GROWTH_RATE[species.species_type]

            if species.viability > 0.6:
                # Logistic growth: density-dependent approach toward carrying capacity K
                growth = r * species.population * (1.0 - species.population / K)
                species.population = min(K, species.population + max(0.0, growth))
            elif species.viability < 0.3:
                # Habitat-stress decline; severity proportional to viability shortfall
                decline_rate = 0.03 * (0.3 - species.viability) / 0.3
                species.population = max(0.0, species.population - species.population * decline_rate)

            ext_threshold = SPECIES_EXTINCTION_THRESHOLD[species.species_type]
            if 0 < species.population < ext_threshold:
                self.extinction_events.append({
                    'species': species.species_type,
                    'frame': frame,
                    'viability': species.viability
                })
                species.population = 0

            if species.viability < 0.4 and species.population > 10:
                self.migration_events.append({
                    'species': species.species_type,
                    'frame': frame,
                    'from_viability': species.viability
                })

        self.temp_history.append(self.current_temp)
        self.precip_history.append(self.precipitation)
        self.fire_risk_history.append(self.fire_risk)
        avg_health = float(habitat_health_total) / float(len(self.habitats)) if self.habitats else 0.0
        self.health_history.append(avg_health)
        for sp in self.species:
            self.species_pop_history[sp.species_type].append(sp.population)

        if frame % 6 == 0:
            self.snapshots.append({
                'temp':      self.current_temp,
                'precip':    self.precipitation,
                'fire_risk': self.fire_risk,
                'habitats':  [(h.health, h.water_level, h.tip_reached) for h in self.habitats],
            })

    def get_statistics(self) -> Dict[str, Any]:
        species_status: Dict[str, Dict[str, float]] = {}
        for species in self.species:
            species_status[species.species_type] = {
                'population': float(species.population),
                'viability': species.viability
            }

        return {
            'current_temp': self.current_temp,
            'precipitation': self.precipitation,
            'drought_intensity': self.drought_intensity,
            'fire_risk': self.fire_risk,
            'avg_habitat_health': sum(h.health for h in self.habitats) / len(self.habitats) if self.habitats else 0,
            'tipping_points_reached': self.tipping_points_reached,
            'extinction_events': len(self.extinction_events),
            'species_status': species_status
        }

# ===================================================================================================
# 4. VISUALIZATION - vivid, ludic animated SVG
# ===================================================================================================

class ClimateChangeRenderer:
    def __init__(self, cfg: ClimateChangeConfig, model: ClimateChangeModel):
        self.cfg = cfg
        self.model = model

    def generate_svg(self) -> str:
        """Redesigned SVG: structured habitat-map layout, capped health labels,
        time-series dashboard, species viability panel, clear legend."""

        w, h  = self.cfg.width, self.cfg.height
        snaps = self.model.snapshots
        # Sample up to 10 evenly-spaced snapshots
        if len(snaps) > 10:
            idxs = [int(i * (len(snaps) - 1) / 9) for i in range(10)]
            snaps = [snaps[i] for i in idxs]
        n   = max(1, len(snaps))
        dur = 18.0

        # ── Layout constants ─────────────────────────────────────────────────
        # Left habitat-map panel: 0..1050; Right dashboard: 1060..1280
        MAP_W = 1050
        DASH_X = 1060
        DASH_W = w - DASH_X - 8
        HDR_H = 60
        FTR_Y = h - 60

        # ── Background ───────────────────────────────────────────────────────
        def _bg_color(temp):
            t = max(0.0, min(1.0, (temp - 26.0) / 6.0))
            r = int(12 + t * 50); g = int(8 + t * 4); b = int(20 - t * 10)
            return f'rgb({r},{g},{b})'

        bg_init = _bg_color(snaps[0]['temp']) if snaps else '#0c0814'
        bg_vals = ';'.join(_bg_color(s['temp']) for s in snaps) if snaps else bg_init

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background:#0c0814;font-family:\'Trebuchet MS\',system-ui,sans-serif;">',
            '<defs>'
            '<filter id="habBlur"><feGaussianBlur stdDeviation="3"/></filter>'
            '<radialGradient id="fireHaze38"><stop offset="0%" stop-color="#ff3d00" stop-opacity="0.3"/>'
            '<stop offset="100%" stop-color="#ff3d00" stop-opacity="0.0"/></radialGradient>'
            '</defs>',
            f'<rect width="{w}" height="{h}" fill="{bg_init}">',
            f'<animate attributeName="fill" values="{bg_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'</rect>',
        ]

        # ── Spatial habitat map: habitat circles placed in geographic clusters ─
        # Re-place habitats in ecologically meaningful horizontal bands:
        #   Top band  → Cerrado     (y: HDR_H+20 .. HDR_H+160)
        #   Middle    → Mata Seca   (y: HDR_H+170 .. HDR_H+310)
        #   Low band  → Vereda      (y: HDR_H+320 .. HDR_H+440)
        #   Bottom    → Karst       (y: HDR_H+450 .. FTR_Y-20)
        spatial_positions = [
            # Arrange 12 habitats in 4 rows of 3, within MAP_W=1050
            (166, HDR_H + 59), (534, HDR_H + 69), (900, HDR_H + 51),
            (166, HDR_H + 174), (534, HDR_H + 182), (900, HDR_H + 169),
            (166, HDR_H + 292), (534, HDR_H + 300), (900, HDR_H + 284),
            (166, HDR_H + 410), (534, HDR_H + 418), (900, HDR_H + 402),
        ]
        # Zone labels
        zone_labels = [
            (HDR_H + 130, 'Cerrado stricto sensu', '#8bc34a'),
            (HDR_H + 250, 'Mata Seca', '#4caf50'),
            (HDR_H + 370, 'Vereda / Gallery Forest', '#00bcd4'),
            (HDR_H + 470, 'Karst / Rocky Outcrops', '#9c27b0'),
        ]
        for zy, zlbl, zcol in zone_labels:
            svg.append(
                f'<line x1="10" y1="{zy}" x2="{MAP_W - 10}" y2="{zy}"'
                f' stroke="{zcol}" stroke-width="0.6" opacity="0.2" stroke-dasharray="6,4"/>'
                f'<text x="12" y="{zy - 6}" fill="{zcol}" font-size="15"'
                f' font-weight="bold" opacity="0.6">{zlbl}</text>'
            )

        # Separator between map and dashboard
        svg.append(f'<line x1="{MAP_W + 5}" y1="{HDR_H}" x2="{MAP_W + 5}" y2="{FTR_Y}"'
                   f' stroke="#2a3a5a" stroke-width="1.5" opacity="0.6"/>')

        # Fire haze overlay on map
        if snaps:
            fr_op = ';'.join(f"{min(0.35, s['fire_risk'] * 0.55):.2f}" for s in snaps)
            svg.append(
                f'<rect x="0" y="{HDR_H}" width="{MAP_W}" height="{FTR_Y - HDR_H}"'
                f' fill="#ff3d00" opacity="0">'
                f'<animate attributeName="opacity" values="{fr_op}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</rect>'
            )

        # Ember particles (fewer, contained in map area)
        random.seed(38)
        for _ in range(30):
            ex = random.uniform(20, MAP_W - 20)
            ey = random.uniform(HDR_H + 20, FTR_Y - 20)
            ed = random.uniform(2, 5)
            if snaps:
                emb_op = ';'.join(
                    f"{min(0.7, s['fire_risk'] * 1.2):.2f}" if random.random() > 0.4 else "0.0"
                    for s in snaps
                )
                svg.append(
                    f'<circle cx="{ex:.0f}" cy="{ey:.0f}" r="{random.uniform(2, 3.5):.1f}"'
                    f' fill="#ff6e40" opacity="0">'
                    f'<animate attributeName="opacity" values="{emb_op}" dur="{dur}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="cy" values="{ey:.0f};{ey - random.uniform(30, 80):.0f};{ey:.0f}"'
                    f' dur="{ed:.1f}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )

        # ── Corridor connections ─────────────────────────────────────────────
        svg.append('<g opacity="0.12">')
        for ii in range(len(spatial_positions)):
            for jj in range(ii + 1, len(spatial_positions)):
                x1, y1 = spatial_positions[ii]; x2, y2 = spatial_positions[jj]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < 390:
                    svg.append(
                        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
                        f' stroke="#4fc3f7" stroke-width="1" stroke-dasharray="4,4"/>'
                    )
        svg.append('</g>')

        # ── Habitat circles ──────────────────────────────────────────────────
        for idx, habitat in enumerate(self.model.habitats):
            if idx >= len(spatial_positions):
                break
            px, py = spatial_positions[idx]
            color  = habitat.data['color']
            # Fixed visual radius (not tied to unbounded biomass)
            radius = 27

            if snaps:
                # Health: 0-1 float → opacity
                op_init = f"{0.30 + min(1.0, snaps[0]['habitats'][idx][0]) * 0.55:.2f}"
                op_vals = ';'.join(
                    f"{0.30 + min(1.0, s['habitats'][idx][0]) * 0.55:.2f}"
                    for s in snaps
                )
                tip_vals = ';'.join('0.65' if s['habitats'][idx][2] else '0.0' for s in snaps)
            else:
                health_clamped = min(1.0, habitat.health)
                op_init = op_vals = f"{0.30 + health_clamped * 0.55:.2f}"
                tip_vals = '0.65' if habitat.tip_reached else '0.0'

            # Outer glow
            svg.append(
                f'<circle cx="{px}" cy="{py}" r="{radius + 18}" fill="{color}" opacity="0.06">'
                f'<animate attributeName="r" values="{radius + 14};{radius + 24};{radius + 14}"'
                f' dur="5s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Tipping-point ring
            svg.append(
                f'<circle cx="{px}" cy="{py}" r="{radius + 10}" fill="none"'
                f' stroke="#ff1744" stroke-width="2.5" stroke-dasharray="5,3" opacity="0">'
                f'<animate attributeName="opacity" values="{tip_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="r" values="{radius + 8};{radius + 16};{radius + 8}"'
                f' dur="2s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Main circle
            svg.append(
                f'<circle cx="{px}" cy="{py}" r="{radius}" fill="{color}" opacity="{op_init}">'
                f'<animate attributeName="opacity" values="{op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Ring border
            svg.append(
                f'<circle cx="{px}" cy="{py}" r="{radius}" fill="none"'
                f' stroke="{color}" stroke-width="1.5" opacity="0.55"/>'
            )
            # Vereda water inner ring
            if habitat.habitat_type == 'vereda' and snaps:
                wr_init = f"{radius * min(1.0, snaps[0]['habitats'][idx][1]):.1f}"
                wr_vals = ';'.join(f"{radius * min(1.0, s['habitats'][idx][1]):.1f}" for s in snaps)
                svg.append(
                    f'<circle cx="{px}" cy="{py}" r="{wr_init}" fill="#00bcd4" opacity="0.35">'
                    f'<animate attributeName="r" values="{wr_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )

            # Health label — clamped 0-100%
            h_clamped = min(1.0, habitat.health)
            h_pct = f'{h_clamped:.0%}'
            h_color = '#69f098' if h_clamped > 0.6 else '#ffb74d' if h_clamped > 0.3 else '#ef5350'
            svg.append(
                f'<text x="{px}" y="{py - radius - 8}" text-anchor="middle"'
                f' fill="{h_color}" font-size="15" font-weight="bold">Health: {h_pct}</text>'
                f'<text x="{px}" y="{py + 5}" text-anchor="middle"'
                f' fill="#e0e0e0" font-size="15" font-weight="bold">{habitat.data["name"]}</text>'
            )

        # ── Dashboard panels (right side) ─────────────────────────────────────

        # 1. Stats panel
        stats = self.model.get_statistics()
        sp_y = HDR_H + 10
        sp_h = 95
        svg.append(
            f'<rect x="{DASH_X}" y="{sp_y}" width="{DASH_W}" height="{sp_h}"'
            f' fill="#0a0812" rx="8" stroke="#ff8a00" stroke-width="1" opacity="0.95"/>'
            f'<text x="{DASH_X + 12}" y="{sp_y + 20}" fill="#ffb300" font-size="15"'
            f' font-weight="bold">Climate Impacts</text>'
            f'<text font-weight="bold" x="{DASH_X + 12}" y="{sp_y + 42}" fill="#64b5f6" font-size="15">'
            f'Precip: {stats["precipitation"]:.0%}</text>'
            f'<text font-weight="bold" x="{DASH_X + 12}" y="{sp_y + 62}" fill="#ff7043" font-size="15">'
            f'Fire: {stats["fire_risk"]:.0%}</text>'
            f'<text font-weight="bold" x="{DASH_X + 12}" y="{sp_y + 82}" fill="#90a4ae" font-size="15">'
            f'Habitats: {min(1.0, stats["avg_habitat_health"]):.0%}'
            f'  \u00b7  Tips: {stats["tipping_points_reached"]}</text>'
        )

        # 3. Thermometer
        th_x = DASH_X + DASH_W - 36; th_y = sp_y
        temp_fill = max(0.0, min(1.0, (self.model.current_temp - 26) / 6))
        th_inner_h = int((sp_h - 20) * temp_fill)
        svg.append(
            f'<rect x="{th_x}" y="{th_y + 10}" width="22" height="{sp_h - 20}" rx="11"'
            f' fill="#1a0a08" stroke="#ff5722" stroke-width="1"/>'
            f'<rect x="{th_x + 3}" y="{th_y + 10 + (sp_h - 20) - th_inner_h}" width="16"'
            f' height="{th_inner_h}" rx="8" fill="#ff5722" opacity="0.75">'
            f'<animate attributeName="opacity" values="0.55;0.85;0.55" dur="2s" repeatCount="indefinite"/>'
            f'</rect>'
            f'<text font-weight="bold" x="{th_x + 11}" y="{th_y + sp_h + 6}" text-anchor="middle"'
            f' fill="#ff8a65" font-size="15">{self.model.current_temp:.1f}\u00b0C</text>'
        )

        # 4. Species viability bars
        sv_y = sp_y + sp_h + 12
        sv_h = 30 + len(self.model.species) * 22
        max_pop = max(max(sp.population, 1) for sp in self.model.species)
        max_pop_log = math.log1p(max_pop)
        svg.append(
            f'<rect x="{DASH_X}" y="{sv_y}" width="{DASH_W}" height="{sv_h}"'
            f' fill="#0a0812" rx="8" stroke="#1e2a40" stroke-width="1" opacity="0.95"/>'
            f'<text x="{DASH_X + 12}" y="{sv_y + 20}" fill="#8090b0" font-size="15"'
            f' font-weight="bold">Species Populations</text>'
        )
        for ki, sp in enumerate(self.model.species):
            bar_y2 = sv_y + 34 + ki * 22
            bar_w = min((math.log1p(sp.population) / max_pop_log) * (DASH_W - 130), DASH_W - 130) if max_pop_log > 0 else 0
            via_color = '#69f098' if sp.viability > 0.6 else '#ffb74d' if sp.viability > 0.3 else '#ef5350'
            lbl_x = min(DASH_X + 125 + bar_w, DASH_X + DASH_W - 28)
            svg.append(
                f'<circle cx="{DASH_X + 15}" cy="{bar_y2 + 4}" r="5" fill="{sp.data["color"]}"/>'
                f'<text font-weight="bold" x="{DASH_X + 26}" y="{bar_y2 + 8}" fill="#c0c8d0" font-size="15">'
                f'{sp.data["name"]}</text>'
                f'<rect x="{DASH_X + 120}" y="{bar_y2 - 2}" width="{DASH_W - 130}" height="10" rx="4"'
                f' fill="#1a1a2e"/>'
                f'<rect x="{DASH_X + 120}" y="{bar_y2 - 2}" width="{bar_w:.0f}" height="10" rx="4"'
                f' fill="{via_color}" opacity="0.85">'
                f'<animate attributeName="opacity" values="0.65;0.9;0.65" dur="2.5s" repeatCount="indefinite"/>'
                f'</rect>'
                f'<text font-weight="bold" x="{lbl_x:.0f}" y="{bar_y2 + 8}" fill="#e0e0e0" font-size="15">'
                f' {int(sp.population)}</text>'
            )

        # 5. Legend panel
        lp_y = sv_y + sv_h + 12
        lp_h = 110
        svg.append(
            f'<rect x="{DASH_X}" y="{lp_y}" width="{DASH_W}" height="{lp_h}"'
            f' fill="#0a0812" rx="8" stroke="#1e2a40" stroke-width="1" opacity="0.95"/>'
            f'<text x="{DASH_X + 12}" y="{lp_y + 20}" fill="#8090b0" font-size="15"'
            f' font-weight="bold">Legend</text>'
        )
        legend_items38 = [
            ('#8bc34a', 'Cerrado—high health'),
            ('#4caf50', 'Mata Seca'),
            ('#00bcd4', 'Vereda / Gallery Forest'),
            ('#9c27b0', 'Karst / Rocky Outcrops'),
            ('#ff1744', 'Tipping point reached'),
        ]
        for ki, (lc, ll) in enumerate(legend_items38):
            liy2 = lp_y + 36 + ki * 16
            svg.append(
                f'<circle cx="{DASH_X + 20}" cy="{liy2}" r="6" fill="{lc}" opacity="0.85"/>'
                f'<text font-weight="bold" x="{DASH_X + 32}" y="{liy2 + 5}" fill="#c0c8d0" font-size="15">{ll}</text>'
            )

        # ── Footer ───────────────────────────────────────────────────────────
        svg.append(
            f'<rect x="0" y="{FTR_Y}" width="{w}" height="50" fill="#06040e" opacity="0.85"/>'
            f'<text x="20" y="{FTR_Y + 20}" fill="#8090b0" font-size="15" font-weight="bold">'
            f'RESEX Recanto das Araras — Terra Ronca-GO</text>'
            f'<text font-weight="bold" x="20" y="{FTR_Y + 40}" fill="#607d8b" font-size="15">'
            f'Scenario: +6°C warming over {self.cfg.frames} frames · {dur:.0f}s animated loop'
            f' · {len(self.model.habitats)} habitat nodes · {n} keyframes</text>'
        )

        # ── Header ───────────────────────────────────────────────────────────
        svg.append(
            f'<rect x="0" y="0" width="{w}" height="{HDR_H}" fill="#060410" opacity="0.80"/>'
            f'<text x="20" y="32" fill="#f5f0e8" font-size="15" font-weight="bold" letter-spacing="0.5">'
            f'ECO-SIM: Climate Change Cascading Impacts \u2014 Cerrado Habitats</text>'
            f'<text font-weight="bold" x="20" y="54" fill="#90a0b8" font-size="15">'
            f'Habitat tipping points \u00b7 fire risk \u00b7 species viability \u00b7'
            f' +6\u00b0C scenario \u00b7 {dur:.0f}s loop</text>'
            f'<line x1="20" y1="64" x2="620" y2="64" stroke="#ff8a00" stroke-width="2"'
            f' stroke-linecap="round" opacity="0.6">'
            f'<animate attributeName="x2" values="200;620;200" dur="6s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


def main():
    print(f"Initializing enhanced climate change model on {CONFIG.device}...")
    model = ClimateChangeModel(CONFIG)

    for frame in range(CONFIG.frames):
        model.step(frame)

    stats = model.get_statistics()
    print(f"Final statistics: {stats}")

    print("Simulation complete. Rendering...")
    renderer = ClimateChangeRenderer(CONFIG, model)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_38_1')
    return svg_content

if __name__ == "__main__":
    main()