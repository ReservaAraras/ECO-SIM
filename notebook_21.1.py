# -*- coding: utf-8 -*-
# pyre-ignore-all-errors
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 21.1: Enhanced Multi-Species Flocking with Information Transfer
# DERIVATIVE OF: notebook_21.py (Enhanced with social learning and knowledge transfer)
# ===================================================================================================

"""
Notebook Differentiation:
- Differentiation Focus: Enhanced Multi-Species Flocking with Information Transfer emphasizing movement corridor bottlenecks.
- Indicator species: Urubu-rei (Sarcoramphus papa).
- Pollination lens: bee foraging under smoke haze.
- Human impact lens: selective logging canopy loss.
"""

from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

# ===================================================================================================
# 1. SCIENTIFIC CONTEXT & PARAMETERS
# ===================================================================================================

@dataclass
class FlockingConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 250
    fps: int = 12
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_species: int = 4
    individuals_per_species: int = 40

CONFIG = FlockingConfig()

# ===================================================================================================
# 2. SPECIES DATABASE
# ===================================================================================================

class SpeciesData(TypedDict):
    color: str
    trail_color: str
    radius: int
    speed: float
    perception_radius: float
    cohesion: float
    alignment: float
    separation: float
    hierarchy: int
    name: str


SPECIES_DB: Dict[str, SpeciesData] = {
    'tucano': {
        'color': '#fe4db7',
        'trail_color': '#ff79c6',
        'radius': 6,
        'speed': 4.5,
        'perception_radius': 80,
        'cohesion': 0.03,
        'alignment': 0.05,
        'separation': 0.08,
        'hierarchy': 3,
        'name': 'Tucano-toco'
    },
    'gralha': {
        'color': '#00ffdc',
        'trail_color': '#50ffea',
        'radius': 5,
        'speed': 5.8,
        'perception_radius': 60,
        'cohesion': 0.02,
        'alignment': 0.07,
        'separation': 0.06,
        'hierarchy': 2,
        'name': 'Gralha-do-campo'
    },
    'beija_flor': {
        'color': '#ffe43f',
        'trail_color': '#ffed80',
        'radius': 3,
        'speed': 6.2,
        'perception_radius': 40,
        'cohesion': 0.01,
        'alignment': 0.03,
        'separation': 0.15,
        'hierarchy': 1,
        'name': 'Beija-flor-tesoura'
    },
    'gaviao': {
        'color': '#62fff3',
        'trail_color': '#90fff8',
        'radius': 7,
        'speed': 7.0,
        'perception_radius': 100,
        'cohesion': 0.01,
        'alignment': 0.08,
        'separation': 0.04,
        'hierarchy': 4,
        'name': 'Gaviao-carijo'
    }
}

# ===================================================================================================
# 3. ENHANCED FLOCKING MODEL WITH INFORMATION TRANSFER
# ===================================================================================================

class EnhancedFlockingModel:
    def __init__(self, cfg: FlockingConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        raw_positions: List[List[float]] = []
        raw_velocities: List[List[float]] = []
        self.species_ids: List[str] = []
        self.knowledge_levels: List[float] = []
        self.teacher_positions: List[List[float]] = []

        species_list = list(SPECIES_DB.keys())
        for species in species_list:
            for _ in range(cfg.individuals_per_species):
                raw_positions.append([
                    random.uniform(0, cfg.width),
                    random.uniform(0, cfg.height)
                ])
                raw_velocities.append([
                    random.uniform(-1, 1) * SPECIES_DB[species]['speed'],
                    random.uniform(-1, 1) * SPECIES_DB[species]['speed']
                ])
                self.species_ids.append(species)
                self.knowledge_levels.append(SPECIES_DB[species]['hierarchy'] / 4.0)

        self.num_individuals = len(raw_positions)
        self.positions: Any = torch.tensor(raw_positions, device=self.dev)
        self.velocities: Any = torch.tensor(raw_velocities, device=self.dev)
        self.species_tensor: Any = torch.tensor([hash(s) % 1000 for s in self.species_ids], device=self.dev)
        self.knowledge: Any = torch.tensor(self.knowledge_levels, device=self.dev)

        # Record trajectory at regular intervals for smoother animation
        self.trajectory_history: List[Any] = [self.positions.cpu().numpy().copy()]
        self.knowledge_history: List[Any] = [self.knowledge.cpu().numpy().copy()]

    def step(self):
        dist_matrix = torch.cdist(self.positions, self.positions)

        params_list = [SPECIES_DB[s] for s in self.species_ids]
        perception_radii = torch.tensor([p['perception_radius'] for p in params_list], device=self.dev)
        cohesions = torch.tensor([p['cohesion'] for p in params_list], device=self.dev)
        alignments = torch.tensor([p['alignment'] for p in params_list], device=self.dev)
        separations = torch.tensor([p['separation'] for p in params_list], device=self.dev)
        max_speeds = torch.tensor([p['speed'] for p in params_list], device=self.dev)

        cohesion_force = torch.zeros_like(self.positions)
        alignment_force = torch.zeros_like(self.positions)
        separation_force = torch.zeros_like(self.positions)

        for i in range(self.num_individuals):
            neighbors = dist_matrix[i] < perception_radii[i]
            if neighbors.sum() > 0:
                neighbor_pos = self.positions[neighbors]
                center = neighbor_pos.mean(dim=0)
                cohesion_force[i] = (center - self.positions[i]) * cohesions[i]

                neighbor_vel = self.velocities[neighbors]
                avg_vel = neighbor_vel.mean(dim=0)
                alignment_force[i] = (avg_vel - self.velocities[i]) * alignments[i]

                close_neighbors = dist_matrix[i] < perception_radii[i] * 0.5
                if close_neighbors.sum() > 0:
                    close_pos = self.positions[close_neighbors]
                    diff = self.positions[i] - close_pos
                    dists = torch.norm(diff, dim=1, keepdim=True).clamp(min=0.1)
                    separation_force[i] = (diff / dists).sum(dim=0) * separations[i]

        knowledge_transfer_force = torch.zeros_like(self.positions)
        for i in range(self.num_individuals):
            knowledgeable = (self.knowledge > self.knowledge[i] + 0.2) & (dist_matrix[i] < 100)
            if knowledgeable.sum() > 0:
                teacher_pos = self.positions[knowledgeable].mean(dim=0)
                knowledge_transfer_force[i] = (teacher_pos - self.positions[i]) * 0.02
                self.knowledge[i] = torch.clamp(self.knowledge[i] + 0.005, max=1.0)

        self.velocities += cohesion_force + alignment_force + separation_force + knowledge_transfer_force

        speeds = torch.norm(self.velocities, dim=1, keepdim=True)
        self.velocities = (self.velocities / speeds.clamp(min=1e-5)) * max_speeds.unsqueeze(1)

        self.positions += self.velocities * 0.5

        self.positions[:, 0] = self.positions[:, 0] % self.cfg.width
        self.positions[:, 1] = self.positions[:, 1] % self.cfg.height

        self.trajectory_history.append(self.positions.cpu().numpy().copy())
        self.knowledge_history.append(self.knowledge.cpu().numpy().copy())

    def get_knowledge_statistics(self) -> Dict[str, float]:
        species_knowledge: Dict[str, float] = {}
        for species in SPECIES_DB.keys():
            indices = [i for i, s in enumerate(self.species_ids) if s == species]
            if indices:
                avg_knowledge = self.knowledge[indices].mean().item()
                species_knowledge[species] = avg_knowledge
        return species_knowledge

# ===================================================================================================
# 4. VISUALIZATION - vivid, ludic animated SVG
# ===================================================================================================

class EnhancedFlockingRenderer:
    def __init__(self, cfg: FlockingConfig, model: EnhancedFlockingModel):
        self.cfg = cfg
        self.model = model

    def generate_svg(self) -> str:
        w, h = self.cfg.width, self.cfg.height
        F = len(self.model.trajectory_history)
        dur = F / self.cfg.fps
        # Subsample for performance (every 3rd frame)
        step = max(1, F // 80)

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:#060912; font-family:\'Trebuchet MS\',system-ui,sans-serif;">',
            '<defs>'
            # Glow filters
            '<filter id="birdGlow"><feGaussianBlur stdDeviation="3" result="blur"/>'
            '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
            '<filter id="softPulse"><feGaussianBlur stdDeviation="5"/></filter>'
            # Knowledge transfer gradient
            '<radialGradient id="knowledgeAura">'
            '<stop offset="0%" stop-color="#ffffff" stop-opacity="0.4"/>'
            '<stop offset="100%" stop-color="#ffffff" stop-opacity="0.0"/>'
            '</radialGradient>'
            # Animated dot-star background
            '<radialGradient id="starGrad">'
            '<stop offset="0%" stop-color="#8090cc" stop-opacity="0.5"/>'
            '<stop offset="100%" stop-color="#182040" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>',
            f'<rect width="{w}" height="{h}" fill="#060912"/>',
        ]

        # ── Twinkling star background ────────────────────────────────────────
        random.seed(42)
        for _ in range(60):
            sx = random.uniform(0, w)
            sy = random.uniform(0, h)
            sr = random.uniform(0.5, 2.0)
            sd = random.uniform(2.0, 5.0)
            svg.append(
                f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{sr:.1f}" fill="#8090cc" opacity="0.15">'
                f'<animate attributeName="opacity" values="0.05;0.3;0.05" '
                f'dur="{sd:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # ── Subtle grid pattern ──────────────────────────────────────────────
        svg.append('<g opacity="0.06">')
        for gx_d in range(0, w, 40):
            svg.append(f'<line x1="{gx_d}" y1="0" x2="{gx_d}" y2="{h}" stroke="#3060a0" stroke-width="0.5"/>')
        for gy_d in range(0, h, 40):
            svg.append(f'<line x1="0" y1="{gy_d}" x2="{w}" y2="{gy_d}" stroke="#3060a0" stroke-width="0.5"/>')
        svg.append('</g>')

        # ── Animated bird particles with trails ──────────────────────────────
        # First: faded trail paths for a subset (every 4th individual)
        for idx in range(0, self.model.num_individuals, 4):
            species = self.model.species_ids[idx]
            trail_color = SPECIES_DB[species].get('trail_color', SPECIES_DB[species]['color'])
            # Build trail path from history subset
            pts = []
            for fi in range(0, F, step * 2):
                p = self.model.trajectory_history[fi][idx]
                pts.append(f"{p[0]:.1f},{p[1]:.1f}")
            if len(pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(pts)}" fill="none" '
                    f'stroke="{trail_color}" stroke-opacity="0.12" stroke-width="1.2" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>'
                )

        # Main animated particles
        for idx in range(self.model.num_individuals):
            species = self.model.species_ids[idx]
            color = SPECIES_DB[species]['color']
            radius = SPECIES_DB[species]['radius']
            final_knowledge = self.model.knowledge_history[-1][idx]

            # Subsample trajectory for animate values
            px_vals = ";".join(f"{self.model.trajectory_history[fi][idx, 0]:.1f}" for fi in range(0, F, step))
            py_vals = ";".join(f"{self.model.trajectory_history[fi][idx, 1]:.1f}" for fi in range(0, F, step))
            sub_dur = len(range(0, F, step)) * step / self.cfg.fps

            # Knowledge glow halo (for knowledgeable individuals)
            if final_knowledge > 0.7:
                glow_r = radius + 6 + final_knowledge * 4
                svg.append(
                    f'<circle r="{glow_r:.1f}" fill="{color}" opacity="0" filter="url(#softPulse)">'
                    f'<animate attributeName="cx" values="{px_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="cy" values="{py_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="opacity" values="0.15;0.30;0.15" dur="2s" repeatCount="indefinite"/>'
                    f'</circle>'
                )

            # Main bird dot
            svg.append(
                f'<circle r="{radius}" fill="{color}" filter="url(#birdGlow)">'
                f'<animate attributeName="cx" values="{px_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{py_vals}" dur="{sub_dur:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # ── Knowledge level panel (glassmorphic) ─────────────────────────────
        stats = self.model.get_knowledge_statistics()
        px, py = w - 240, 80
        svg.append(
            f'<rect x="{px}" y="{py}" width="220" height="180" fill="#0d1420" '
            f'opacity="0.92" rx="10" stroke="#1e3a5f" stroke-width="1"/>'
            f'<text x="{px + 12}" y="{py + 22}" fill="#90b8d8" font-size="15" '
            f'font-weight="bold" letter-spacing="0.5">Species Knowledge Levels</text>'
            f'<line x1="{px + 12}" y1="{py + 30}" x2="{px + 208}" y2="{py + 30}" '
            f'stroke="#2a5080" stroke-width="1"/>'
        )

        y_off = py + 50
        for species, data in SPECIES_DB.items():
            knowledge = stats.get(species, 0)
            bar_w = knowledge * 120
            svg.append(
                # Species dot
                f'<circle cx="{px + 20}" cy="{y_off}" r="5" fill="{data["color"]}">'
                f'<animate attributeName="opacity" values="0.7;1;0.7" dur="2s" repeatCount="indefinite"/>'
                f'</circle>'
                # Species name
                f'<text x="{px + 32}" y="{y_off + 4}" fill="#c8dce8" font-size="15" '
                f'font-weight="bold">{data["name"]}</text>'
                # Knowledge bar background
                f'<rect x="{px + 32}" y="{y_off + 10}" width="120" height="6" rx="3" '
                f'fill="#1a2a3a"/>'
                # Knowledge bar fill
                f'<rect x="{px + 32}" y="{y_off + 10}" width="{bar_w:.0f}" height="6" rx="3" '
                f'fill="{data["color"]}" opacity="0.8">'
                f'<animate attributeName="opacity" values="0.6;0.9;0.6" dur="2.5s" repeatCount="indefinite"/>'
                f'</rect>'
                # Knowledge value
                f'<text font-weight="bold" x="{px + 158}" y="{y_off + 16}" fill="#8090a0" font-size="15">'
                f'{knowledge:.0%}</text>'
            )
            y_off += 34

        # ── Title bar ─────────────────────────────────────────────────────────
        svg.append(
            f'<rect x="0" y="0" width="{w}" height="68" fill="#060912" opacity="0.75"/>'
            f'<text x="20" y="30" fill="#f0f4f8" font-size="15" font-weight="bold" '
            f'letter-spacing="0.5">ECO-SIM: Enhanced Multi-Species Flocking</text>'
            f'<text font-weight="bold" x="20" y="50" fill="#8bb4d9" font-size="15">'
            f'Information transfer \u00b7 social learning \u00b7 knowledge hierarchy dynamics</text>'
            f'<line x1="20" y1="60" x2="480" y2="60" stroke="#62fff3" stroke-width="2" '
            f'stroke-linecap="round" opacity="0.5">'
            f'<animate attributeName="x2" values="200;480;200" dur="5s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="0.3;0.7;0.3" dur="5s" repeatCount="indefinite"/>'
            f'</line>'
        )

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

def main():
    print(f"Initializing enhanced flocking model on {CONFIG.device}...")
    model = EnhancedFlockingModel(CONFIG)

    for _ in range(CONFIG.frames):
        model.step()

    stats = model.get_knowledge_statistics()
    print(f"Knowledge statistics: {stats}")

    print("Simulation complete. Rendering...")
    renderer = EnhancedFlockingRenderer(CONFIG, model)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_21.1')
    return svg_content

if __name__ == "__main__":
    main()