# -*- coding: utf-8 -*-
# pyre-ignore-all-errors
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 11.1: Enhanced Seed Dispersal Networks with Multi-Trophic Interactions
# DERIVATIVE OF: notebook_11.py (Enhanced with gut passage viability and deposition hotspots)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Enhanced Seed Dispersal Networks with Multi-Trophic Interactions emphasizing dry-season stress refuges.
- Indicator species: Bugio-preto (Alouatta caraya).
- Pollination lens: ground-nesting bee vulnerability.
- Human impact lens: traditional harvesting pressure.

Scientific Relevance (PIGT RESEX Recanto das Araras -- 2024):
- Integrates the socio-environmental complexity of 
  de Cima, Goias, Brazil.
- Models landscape connectivity, karst vulnerability (Bacia do Rio Lapa),
  biological corridors, and seed-dispersal networks.
- Demonstrates parameters for ecological succession, biodiversity indices,
  integrated fire management (MIF), and ornithochory dynamics.
- Outputs are published via Google Sites: 
- SVG artefacts archived at: https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing
"""


from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict, cast

# ===================================================================================================
# 1. SCIENTIFIC CONTEXT & PARAMETERS
# ===================================================================================================

@dataclass
class SeedDispersalConfig:
    """Class `SeedDispersalConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 250
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_dispersers: int = 80
    num_seeds: int = 200
    grid_resolution: int = 32

CONFIG = SeedDispersalConfig()

# ===================================================================================================
# 2. SPECIES DATABASE - DISPERSERS & SEEDS
# ===================================================================================================

class DisperserData(TypedDict):
    color: str
    trail: str
    radius: int
    speed: float
    flight_range: int
    gut_passage_time: int
    seed_preference: List[str]
    viability_rate: float
    deposition_type: str
    name: str


class SeedData(TypedDict):
    color: str
    glow: str
    size: int
    germination_time: int
    success_rate: float
    nurse_plant_needed: bool
    name: str


DISPERSER_DB: Dict[str, DisperserData] = {
    'tucano': {
        'color': '#fe4db7',
        'trail': '#ff79c6',
        'radius': 6,
        'speed': 4.5,
        'flight_range': 250,
        'gut_passage_time': 45,
        'seed_preference': ['palm', 'fig'],
        'viability_rate': 0.75,
        'deposition_type': 'cluster',
        'name': 'Tucano-toco'
    },
    'gralha': {
        'color': '#00ffdc',
        'trail': '#50ffe8',
        'radius': 5,
        'speed': 5.8,
        'flight_range': 180,
        'gut_passage_time': 30,
        'seed_preference': ['palm', 'cerrado_tree'],
        'viability_rate': 0.65,
        'deposition_type': 'scattered',
        'name': 'Gralha-do-campo'
    },
    'beija_flor': {
        'color': '#ffe43f',
        'trail': '#ffed80',
        'radius': 3,
        'speed': 6.2,
        'flight_range': 50,
        'gut_passage_time': 15,
        'seed_preference': ['fig'],
        'viability_rate': 0.45,
        'deposition_type': 'scattered',
        'name': 'Beija-flor-tesoura'
    },
    'morcego': {
        'color': '#7e57c2',
        'trail': '#b39ddb',
        'radius': 4,
        'speed': 3.5,
        'flight_range': 300,
        'gut_passage_time': 60,
        'seed_preference': ['fig', 'palm'],
        'viability_rate': 0.85,
        'deposition_type': 'cluster',
        'name': 'Bat (Frugivore)'
    }
}

SEED_DB: Dict[str, SeedData] = {
    'palm': {
        'color': '#4caf50',
        'glow': '#69f098',
        'size': 4,
        'germination_time': 120,
        'success_rate': 0.3,
        'nurse_plant_needed': False,
        'name': 'Palm'
    },
    'fig': {
        'color': '#8bc34a',
        'glow': '#a8e060',
        'size': 3,
        'germination_time': 45,
        'success_rate': 0.5,
        'nurse_plant_needed': False,
        'name': 'Fig tree'
    },
    'cerrado_tree': {
        'color': '#689f38',
        'glow': '#99d066',
        'size': 5,
        'germination_time': 180,
        'success_rate': 0.2,
        'nurse_plant_needed': True,
        'name': 'Cerrado tree'
    }
}

# ===================================================================================================
# 3. ENHANCED SEED DISPERSAL MODEL
# ===================================================================================================

class Seed:
    """Class `Seed` -- simulation component."""

    def __init__(self, x: float, y: float, seed_type: str):
        self.x = x
        self.y = y
        self.seed_type = seed_type
        self.age = 0
        self.germinated = False
        self.viable = True
        self.sapling_size = 0

class DisperserAgent:
    """Class `DisperserAgent` -- simulation component."""

    def __init__(self, x: float, y: float, disperser_type: str):
        self.x = x
        self.y = y
        self.disperser_type = disperser_type
        self.data: DisperserData = DISPERSER_DB[disperser_type]
        self.vel_x = random.uniform(-1, 1) * self.data['speed']
        self.vel_y = random.uniform(-1, 1) * self.data['speed']
        self.carried_seeds: List[Seed] = []
        self.gut_timer = 0
        self.target_tree: Optional[Tuple[float, float]] = None

class SeedDispersalModel:
    """Class `SeedDispersalModel` -- simulation component."""

    def __init__(self, cfg: SeedDispersalConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Initialize food trees (seed sources)
        self.food_trees: List[Tuple[float, float, str]] = []
        tree_types = list(SEED_DB.keys())

        for _ in range(25):
            x = random.uniform(50, cfg.width - 50)
            y = random.uniform(50, cfg.height - 50)
            tree_type = random.choice(tree_types)
            self.food_trees.append((x, y, tree_type))

        # Initialize dispersers
        self.dispersers: List[DisperserAgent] = []
        disperser_types = list(DISPERSER_DB.keys())

        for _ in range(cfg.num_dispersers):
            x = random.uniform(0, cfg.width)
            y = random.uniform(0, cfg.height)
            dtype = random.choice(disperser_types)
            self.dispersers.append(DisperserAgent(x, y, dtype))

        # Initialize seeds on ground
        self.seeds: List[Seed] = []
        for _ in range(cfg.num_seeds):
            x = random.uniform(50, cfg.width - 50)
            y = random.uniform(50, cfg.height - 50)
            seed_type = random.choice(tree_types)
            self.seeds.append(Seed(x, y, seed_type))

        # Track metrics
        self.dispersal_events: List[Dict[str, Any]] = []
        self.germination_events: List[Dict[str, Any]] = []
        self.saplings: List[Seed] = []
        self.disperser_positions_history: List[List[Tuple[float, float, str, bool]]] = []

    def step(self):
        """Function `step` -- simulation component."""

        for disperser in self.dispersers:
            if disperser.carried_seeds:
                disperser.gut_timer += 1

                if disperser.gut_timer >= disperser.data['gut_passage_time']:
                    deposition_type = disperser.data['deposition_type']
                    if deposition_type == 'cluster':
                        for _ in range(len(disperser.carried_seeds)):
                            offset_x = random.uniform(-20, 20)
                            offset_y = random.uniform(-20, 20)
                            seed = disperser.carried_seeds.pop()
                            seed.x = disperser.x + offset_x
                            seed.y = disperser.y + offset_y
                            self.seeds.append(seed)
                            self.dispersal_events.append({
                                'x': seed.x, 'y': seed.y,
                                'type': seed.seed_type, 'viability': disperser.data['viability_rate']
                            })
                    else:
                        for seed in disperser.carried_seeds:
                            seed.x = disperser.x + random.uniform(-50, 50)
                            seed.y = disperser.y + random.uniform(-50, 50)
                            seed.x = max(0.0, min(float(self.cfg.width), seed.x))
                            seed.y = max(0.0, min(float(self.cfg.height), seed.y))
                            self.seeds.append(seed)
                            self.dispersal_events.append({
                                'x': seed.x, 'y': seed.y,
                                'type': seed.seed_type, 'viability': disperser.data['viability_rate']
                            })
                    disperser.carried_seeds = []
                    disperser.gut_timer = 0

            if not disperser.carried_seeds and len(disperser.carried_seeds) < 3:
                nearest_tree: Optional[Tuple[float, float]] = None
                min_dist = float('inf')

                for tx, ty, ttype in self.food_trees:
                    if ttype in disperser.data['seed_preference']:
                        dist = math.sqrt((disperser.x - tx)**2 + (disperser.y - ty)**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_tree = (tx, ty)

                if nearest_tree is not None:
                    target_tree = cast(Tuple[float, float], nearest_tree)
                    tx, ty = target_tree
                    dx, dy = tx - disperser.x, ty - disperser.y
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        disperser.vel_x += (dx / dist) * 0.5
                        disperser.vel_y += (dy / dist) * 0.5

                    if dist < 20:
                        seed_type = random.choice(disperser.data['seed_preference'])
                        new_seed = Seed(disperser.x, disperser.y, seed_type)
                        if random.random() < disperser.data['viability_rate']:
                            disperser.carried_seeds.append(new_seed)
                        disperser.gut_timer = 0
            else:
                disperser.vel_x += random.uniform(-0.5, 0.5)
                disperser.vel_y += random.uniform(-0.5, 0.5)

            speed = math.sqrt(disperser.vel_x**2 + disperser.vel_y**2)
            max_speed = disperser.data['speed']
            if speed > max_speed:
                disperser.vel_x = (disperser.vel_x / speed) * max_speed
                disperser.vel_y = (disperser.vel_y / speed) * max_speed

            disperser.x += disperser.vel_x * 0.5
            disperser.y += disperser.vel_y * 0.5

            disperser.x = disperser.x % self.cfg.width
            disperser.y = disperser.y % self.cfg.height

        for seed in list(self.seeds):
            if not seed.germinated and seed.viable:
                seed.age += 1
                germ_time = SEED_DB[seed.seed_type]['germination_time']

                if seed.age >= germ_time:
                    success_rate = SEED_DB[seed.seed_type]['success_rate']
                    if random.random() < success_rate:
                        seed.germinated = True
                        seed.sapling_size = 1
                        self.saplings.append(seed)
                        self.germination_events.append({
                            'x': seed.x, 'y': seed.y, 'type': seed.seed_type
                        })
                    else:
                        seed.viable = False

        # Record disperser positions (with carrying state)
        self.disperser_positions_history.append([
            (d.x, d.y, d.disperser_type, bool(d.carried_seeds)) for d in self.dispersers
        ])

    def get_statistics(self) -> Dict[str, Any]:
        """Function `get_statistics` -- simulation component."""

        total_seeds = len(self.seeds)
        germinated = len([s for s in self.seeds if s.germinated])
        viable = len([s for s in self.seeds if s.viable])

        seed_by_type = {}
        for seed in self.seeds:
            if seed.seed_type not in seed_by_type:
                seed_by_type[seed.seed_type] = 0
            seed_by_type[seed.seed_type] += 1

        return {
            'total_seeds': total_seeds,
            'germinated': germinated,
            'viable': viable,
            'germination_rate': germinated / total_seeds if total_seeds > 0 else 0,
            'saplings': len(self.saplings),
            'seed_by_type': seed_by_type,
            'dispersal_events': len(self.dispersal_events)
        }

# ===================================================================================================
# 4. VISUALIZATION - vivid, ludic animated SVG
# ===================================================================================================

class SeedDispersalRenderer:
    """Class `SeedDispersalRenderer` -- simulation component."""

    def __init__(self, cfg: SeedDispersalConfig, model: SeedDispersalModel):
        self.cfg = cfg
        self.model = model

    def generate_svg(self) -> str:
        """Function `generate_svg` -- simulation component."""

        w, h = self.cfg.width, self.cfg.height
        F = len(self.model.disperser_positions_history)
        dur = max(1, F) / self.cfg.fps
        step = max(1, F // 70)

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:#060d08; font-family:\'Trebuchet MS\',system-ui,sans-serif;">',
            '<defs>'
            # Tree canopy glow
            '<radialGradient id="treeGlow">'
            '<stop offset="0%" stop-color="#4caf50" stop-opacity="0.6"/>'
            '<stop offset="70%" stop-color="#2e7d32" stop-opacity="0.2"/>'
            '<stop offset="100%" stop-color="#0a2010" stop-opacity="0.0"/>'
            '</radialGradient>'
            # Seed germination glow
            '<radialGradient id="germGlow">'
            '<stop offset="0%" stop-color="#76ff03" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#33691e" stop-opacity="0.0"/>'
            '</radialGradient>'
            # Filters
            '<filter id="treeBlur"><feGaussianBlur stdDeviation="4"/></filter>'
            '<filter id="birdGlow"><feGaussianBlur stdDeviation="2" result="blur"/>'
            '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
            '<filter id="seedPulse"><feGaussianBlur stdDeviation="2.5"/></filter>'
            '</defs>',
            f'<rect width="{w}" height="{h}" fill="#060d08"/>',
        ]

        # ── Organic terrain texture ──────────────────────────────────────────
        random.seed(77)
        svg.append('<g opacity="0.08">')
        for _ in range(120):
            cx = random.uniform(0, w)
            cy = random.uniform(0, h)
            cr = random.uniform(20, 80)
            svg.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{cr:.0f}" fill="#1a4020"/>')
        svg.append('</g>')

        # ── Food trees - animated canopy with breathing ──────────────────────
        for tx, ty, ttype in self.model.food_trees:
            seed_data = SEED_DB[ttype]
            color = seed_data['color']
            glow = seed_data.get('glow', color)
            # Outer glow
            svg.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="38" fill="{glow}" '
                f'opacity="0.08" filter="url(#treeBlur)">'
                f'<animate attributeName="r" values="34;42;34" dur="4s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Canopy circle with pulsing
            svg.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="22" fill="{color}" opacity="0.45">'
                f'<animate attributeName="r" values="20;25;20" dur="5s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.35;0.55;0.35" dur="5s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Canopy ring
            svg.append(
                f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="22" fill="none" '
                f'stroke="{color}" stroke-width="1.5" opacity="0.6">'
                f'<animate attributeName="r" values="20;25;20" dur="5s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            # Tree label
            svg.append(
                f'<text font-weight="bold" x="{tx:.0f}" y="{ty + 35:.0f}" text-anchor="middle" '
                f'fill="#90c090" font-size="15" opacity="0.7">{seed_data["name"]}</text>'
            )

        # ── Seeds on ground - static or germinated ───────────────────────────
        for seed in self.model.seeds:
            if not seed.viable:
                continue
            sd = SEED_DB[seed.seed_type]
            color = sd['color']
            size = sd['size']

            if seed.germinated:
                # Germinated sapling with glow
                svg.append(
                    f'<circle cx="{seed.x:.1f}" cy="{seed.y:.1f}" r="10" '
                    f'fill="{sd.get("glow", color)}" opacity="0.15" filter="url(#seedPulse)">'
                    f'<animate attributeName="opacity" values="0.10;0.25;0.10" '
                    f'dur="3s" repeatCount="indefinite"/>'
                    f'</circle>'
                )
                svg.append(
                    f'<circle cx="{seed.x:.1f}" cy="{seed.y:.1f}" r="{size + 2}" '
                    f'fill="{color}" opacity="0.9"/>'
                )
                svg.append(
                    f'<text font-weight="bold" x="{seed.x:.1f}" y="{seed.y - 8:.1f}" text-anchor="middle" '
                    f'fill="#cceecc" font-size="15">\U0001F331</text>'
                )
            else:
                opacity = 0.3 + min(0.5, seed.age / 100)
                svg.append(
                    f'<circle cx="{seed.x:.1f}" cy="{seed.y:.1f}" r="{size}" '
                    f'fill="{color}" opacity="{opacity:.2f}"/>'
                )

        # ── Disperser agents - animated with trails ──────────────────────────
        for idx, disperser in enumerate(self.model.dispersers):
            color = disperser.data['color']
            trail_color = disperser.data.get('trail', color)
            radius = disperser.data['radius']
            carrying = bool(disperser.carried_seeds)

            if F > 1:
                # Trail
                trail_pts = []
                for fi in range(0, F, step * 2):
                    if fi < len(self.model.disperser_positions_history):
                        px, py = self.model.disperser_positions_history[fi][idx][:2]
                        trail_pts.append(f"{px:.1f},{py:.1f}")
                if len(trail_pts) > 2 and idx % 3 == 0:
                    svg.append(
                        f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                        f'stroke="{trail_color}" stroke-opacity="0.10" stroke-width="1" '
                        f'stroke-linecap="round"/>'
                    )

                # Animated position
                px_vals = ";".join(f"{self.model.disperser_positions_history[fi][idx][0]:.1f}"
                                  for fi in range(0, F, step))
                py_vals = ";".join(f"{self.model.disperser_positions_history[fi][idx][1]:.1f}"
                                  for fi in range(0, F, step))
                sub_dur = len(range(0, F, step)) * step / self.cfg.fps

                # Carrying glow
                if carrying:
                    svg.append(
                        f'<circle r="{radius + 5}" fill="#ffffff" opacity="0" '
                        f'filter="url(#seedPulse)">'
                        f'<animate attributeName="cx" values="{px_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                        f'<animate attributeName="cy" values="{py_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                        f'<animate attributeName="opacity" values="0.15;0.35;0.15" dur="1.5s" repeatCount="indefinite"/>'
                        f'</circle>'
                    )

                # Main dot
                svg.append(
                    f'<circle r="{radius}" fill="{color}" filter="url(#birdGlow)">'
                    f'<animate attributeName="cx" values="{px_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="cy" values="{py_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )
            else:
                svg.append(
                    f'<circle cx="{disperser.x:.1f}" cy="{disperser.y:.1f}" r="{radius}" '
                    f'fill="{color}" filter="url(#birdGlow)"/>'
                )

        # ── Dispersal event markers (little bursts) ──────────────────────────
        event_start = max(0, len(self.model.dispersal_events) - 30)
        for evt_idx in range(event_start, len(self.model.dispersal_events)):
            evt = self.model.dispersal_events[evt_idx]
            ex, ey = evt['x'], evt['y']
            svg.append(
                f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="#76ff03" opacity="0">'
                f'<animate attributeName="r" values="2;12;2" dur="3s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.5;0.0;0.5" dur="3s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # ── Statistics panel (glassmorphic) ──────────────────────────────────
        stats = self.model.get_statistics()
        px, py = w - 280, 290
        panel_h = 130
        svg.append(
            f'<rect x="{px}" y="{py}" width="260" height="{panel_h}" fill="#0a1810" '
            f'opacity="0.92" rx="10" stroke="#2e7d32" stroke-width="1"/>'
            f'<text x="{px + 12}" y="{py + 22}" fill="#81c784" font-size="15" '
            f'font-weight="bold" letter-spacing="0.5">Seed Dispersal Network</text>'
            f'<line x1="{px + 12}" y1="{py + 30}" x2="{px + 248}" y2="{py + 30}" '
            f'stroke="#2e5a32" stroke-width="1"/>'
            f'<text font-weight="bold" x="{px + 12}" y="{py + 48}" fill="#a5d6a7" font-size="15">'
            f'Total Seeds: {stats["total_seeds"]}</text>'
            f'<text font-weight="bold" x="{px + 12}" y="{py + 68}" fill="#a5d6a7" font-size="15">'
            f'Germinated: {stats["germinated"]} | Saplings: {stats["saplings"]}</text>'
            f'<text font-weight="bold" x="{px + 12}" y="{py + 88}" fill="#a5d6a7" font-size="15">'
            f'Dispersal Events: {stats["dispersal_events"]}</text>'
            f'<text font-weight="bold" x="{px + 12}" y="{py + 108}" fill="#a5d6a7" font-size="15">'
            f'Germination Rate: {stats["germination_rate"]:.1%}</text>'
        )

        # ── Disperser legend ─────────────────────────────────────────────────
        lx, ly = w - 220, 80
        svg.append(
            f'<rect x="{lx}" y="{ly}" width="200" height="120" fill="#0a1810" '
            f'opacity="0.90" rx="8" stroke="#2e7d32" stroke-width="1"/>'
            f'<text x="{lx + 12}" y="{ly + 20}" fill="#81c784" font-size="15" '
            f'font-weight="bold">Dispersers</text>'
        )
        y_off = ly + 40
        for dtype, data in DISPERSER_DB.items():
            svg.append(
                f'<circle cx="{lx + 18}" cy="{y_off}" r="5" fill="{data["color"]}">'
                f'<animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>'
                f'</circle>'
                f'<text font-weight="bold" x="{lx + 30}" y="{y_off + 4}" fill="#c8e6c9" font-size="15">{data["name"]}</text>'
            )
            y_off += 20

        # ── Title bar ─────────────────────────────────────────────────────────
        svg.append(
            f'<rect x="0" y="0" width="{w}" height="68" fill="#040a06" opacity="0.75"/>'
            f'<text x="20" y="{ZONES["header"]["y"] + 30}" fill="#e8f5e9" font-size="15" font-weight="bold" '
            f'letter-spacing="0.5">ECO-SIM: Enhanced Seed Dispersal Networks</text>'
            f'<text font-weight="bold" x="20" y="{ZONES["header"]["y"] + 50}" fill="#81c784" font-size="15">'
            f'Multi-trophic interactions \u00b7 gut passage viability \u00b7 ornithochory dynamics</text>'
            f'<line x1="20" y1="60" x2="480" y2="60" stroke="#69f098" stroke-width="2" '
            f'stroke-linecap="round" opacity="0.5">'
            f'<animate attributeName="x2" values="200;480;200" dur="5s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0.3;0.7;0.3" dur="5s" repeatCount="indefinite"/>'
            f'</line>'
        )


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

def main():
    """Function `main` -- simulation component."""

    print(f"Initializing enhanced seed dispersal model on {CONFIG.device}...")
    model = SeedDispersalModel(CONFIG)

    for _ in range(CONFIG.frames):
        model.step()

    stats = model.get_statistics()
    print(f"Final statistics: {stats}")

    print("Simulation complete. Rendering...")
    renderer = SeedDispersalRenderer(CONFIG, model)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_11_1')
    return svg_content

if __name__ == "__main__":
    main()
