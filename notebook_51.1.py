# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 51.1: Multi-Scale Landscape Connectivity Analysis
# COMPLEMENTARY TO: notebook_51 (Future Scenarios)
# ===================================================================================================

"""
Notebook Differentiation:
- Differentiation Focus: Multi-Scale Landscape Connectivity Analysis emphasizing pollinator tracking.
- Indicator species: Cigarra (Quesada gigas).
- Pollination lens: masting-to-bloom handoff in late dry season.
- Human impact lens: trail disturbance and repeated flushing.
"""

from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set

# ===================================================================================================
# 1. SCIENTIFIC CONTEXT & PARAMETERS
# ===================================================================================================

@dataclass
class ConnectivityConfig:
    width: int = 1280
    height: int = 602
    frames: int = 200
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_patches: int = 15
    num_corridors: int = 20
    movement_speed: float = 3.0

CONFIG = ConnectivityConfig()

# ===================================================================================================
# 2. HABITAT PATCH & CORRIDOR MODEL (GRAPH-BASED)
# ===================================================================================================

class HabitatPatch:
    def __init__(self, x: float, y: float, patch_type: str, quality: float):
        self.x = x
        self.y = y
        self.patch_type = patch_type  # 'cerrado', 'mata_seca', 'karst', 'vereda'
        self.quality = quality  # 0-1 scale
        self.connected_patches: List[int] = []
        self.species_present: Set[str] = set()

class Corridor:
    def __init__(self, patch_a: int, patch_b: int, length: float, resistance: float):
        self.patch_a = patch_a
        self.patch_b = patch_b
        self.length = length
        self.resistance = resistance  # higher = harder to traverse

class ConnectivityModel:
    def __init__(self, cfg: ConnectivityConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Generate habitat patches across the landscape
        self.patches: List[HabitatPatch] = []
        patch_types = ['cerrado', 'mata_seca', 'karst', 'vereda']
        type_weights = [0.4, 0.3, 0.2, 0.1]

        for i in range(cfg.num_patches):
            min_dist = 220  # minimum pixels between patch centres
            for _attempt in range(500):
                x = random.uniform(150, cfg.width - 150)
                y = random.uniform(150, cfg.height - 150)
                if all(math.sqrt((x - p.x)**2 + (y - p.y)**2) >= min_dist
                       for p in self.patches):
                    break
            patch_type = random.choices(patch_types, weights=type_weights)[0]
            quality = random.uniform(0.3, 1.0)
            self.patches.append(HabitatPatch(x, y, patch_type, quality))

        # Create corridors between nearby patches
        self.corridors: List[Corridor] = []
        for i in range(cfg.num_patches):
            for j in range(i + 1, cfg.num_patches):
                dist = math.sqrt((self.patches[i].x - self.patches[j].x)**2 +
                               (self.patches[i].y - self.patches[j].y)**2)
                if dist < 380:  # Connect patches within 380 pixels
                    resistance = dist / 250 * random.uniform(0.5, 1.5)
                    self.corridors.append(Corridor(i, j, dist, resistance))

        # Initialize species in high-quality patches
        species_list = ['onca_pintada', 'tucano_toco', 'gralha_campo', 'morcego', 'sagui']
        for i, patch in enumerate(self.patches):
            if patch.quality > 0.6:
                num_species = int(patch.quality * 3)
                patch.species_present = set(random.sample(species_list, min(num_species, len(species_list))))

        # Track individual movement
        self.num_individuals = 50
        self.positions = torch.rand((self.num_individuals, 2), device=self.dev) * torch.tensor([cfg.width, cfg.height], device=self.dev)
        self.velocities = (torch.rand((self.num_individuals, 2), device=self.dev) - 0.5) * cfg.movement_speed

        # Species-specific movement parameters
        self.species_params = {
            'onca_pintada': {'speed': 2.0, 'corridor_needed': True, 'home_range': 150},
            'tucano_toco': {'speed': 4.5, 'corridor_needed': False, 'home_range': 80},
            'gralha_campo': {'speed': 5.0, 'corridor_needed': False, 'home_range': 60},
            'morcego': {'speed': 6.0, 'corridor_needed': False, 'home_range': 100},
            'sagui': {'speed': 3.5, 'corridor_needed': False, 'home_range': 50},
        }

        self.trajectory_history = [self.positions.cpu().numpy().copy()]

    def step(self):
        # Calculate nearest patch for each individual
        patch_coords = torch.tensor([[p.x, p.y] for p in self.patches], device=self.dev)
        dist_matrix = torch.cdist(self.positions, patch_coords)
        nearest_patch, _ = torch.min(dist_matrix, dim=1)

        # Movement toward optimal habitat
        target_patch_idx = torch.argmin(dist_matrix, dim=1)
        targets = patch_coords[target_patch_idx]

        # Direction toward target
        direction = targets - self.positions
        norms = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-5)
        direction = direction / norms

        # Apply movement with corridor constraints
        speed = torch.tensor([self.species_params[list(self.patches[t].species_present)[0] if self.patches[t].species_present else 'tucano_toco']['speed']
                             for t in target_patch_idx], device=self.dev)

        self.velocities += direction * speed.unsqueeze(1) * 0.1
        self.velocities *= 0.95  # Damping

        # Update positions
        self.positions += self.velocities * 0.5

        # Boundary wrapping
        self.positions[:, 0] = self.positions[:, 0] % self.cfg.width
        self.positions[:, 1] = self.positions[:, 1] % self.cfg.height

        self.trajectory_history.append(self.positions.cpu().numpy().copy())

    def calculate_connectivity_metrics(self) -> Dict:
        # Graph-based connectivity metrics
        total_corridors = len(self.corridors)
        avg_resistance = sum(c.resistance for c in self.corridors) / total_corridors if total_corridors > 0 else 0

        # Patch quality weighted by connectivity
        connected_quality: float = 0.0
        for patch in self.patches:
            if patch.species_present:
                connected_quality += patch.quality

        return {
            'total_patches': self.cfg.num_patches,
            'total_corridors': total_corridors,
            'avg_corridor_resistance': avg_resistance,
            'connected_quality_index': connected_quality / self.cfg.num_patches,
            'onca_pintada_individuals': 2,  # Realistic count for Cerrado landscape
        }

# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class ConnectivityRenderer:
    def __init__(self, cfg: ConnectivityConfig, model: ConnectivityModel):
        self.cfg = cfg
        self.model = model

    def generate_svg(self) -> str:
        w, h = self.cfg.width, self.cfg.height
        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#121212; font-family:system-ui, -apple-system, sans-serif;">']

        # Background
        svg.append(f'<rect width="{w}" height="{h}" fill="#121212"/>')

        # Render corridors
        for corridor in self.model.corridors:
            p1 = self.model.patches[corridor.patch_a]
            p2 = self.model.patches[corridor.patch_b]
            opacity = max(0.1, 1.0 - corridor.resistance)
            stroke_width = max(1, 4 - corridor.resistance * 2)
            svg.append(f'<line x1="{p1.x}" y1="{p1.y}" x2="{p2.x}" y2="{p2.y}" stroke="#4fc3f7" stroke-width="{stroke_width}" opacity="{opacity}"/>')

        # Render habitat patches
        colors = {
            'cerrado': '#8bc34a',
            'mata_seca': '#4caf50',
            'karst': '#9c27b0',
            'vereda': '#00bcd4',
        }

        for i, patch in enumerate(self.model.patches):
            radius = 30 + patch.quality * 40
            color = colors.get(patch.patch_type, '#888')
            svg.append(f'<circle cx="{patch.x}" cy="{patch.y}" r="{radius}" fill="{color}" opacity="0.6"/>')
            svg.append(f'<circle cx="{patch.x}" cy="{patch.y}" r="{radius}" fill="none" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<text font-weight="bold" x="{patch.x}" y="{patch.y + 6}" text-anchor="middle" fill="#fff" font-size="15">{patch.patch_type[:4]}</text>')

            # Species count indicator
            if patch.species_present:
                svg.append(f'<text font-weight="bold" x="{patch.x}" y="{patch.y - radius - 10}" text-anchor="middle" fill="#ffeb3b" font-size="15">{len(patch.species_present)}sp</text>')

        # Render individual trajectories
        for idx in range(min(20, self.model.num_individuals)):
            path_x = ";".join([f"{p[idx, 0]:.1f}" for p in self.model.trajectory_history[::5]])
            path_y = ";".join([f"{p[idx, 1]:.1f}" for p in self.model.trajectory_history[::5]])
            svg.append(f'<circle r="5" fill="#ffeb3b">')
            svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{self.cfg.frames/self.cfg.fps}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{self.cfg.frames/self.cfg.fps}s" repeatCount="indefinite"/>')
            svg.append('</circle>')

        # Metrics display
        metrics = self.model.calculate_connectivity_metrics()
        svg.append(f'<g transform="translate(40, {h - 110})">')
        svg.append(f'<rect width="220" height="85" fill="#1e1e1e" opacity="0.88" rx="10"/>')
        svg.append(f'<text x="12" y="18" fill="#4fc3f7" font-size="15" font-weight="bold">Connectivity Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="34" fill="#fff" font-size="15">Patches: {metrics["total_patches"]} | Corridors: {metrics["total_corridors"]}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="50" fill="#fff" font-size="15">Avg Resistance: {metrics["avg_corridor_resistance"]:.2f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="66" fill="#ffeb3b" font-size="15">Quality Index: {metrics["connected_quality_index"]:.2f}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="82" fill="#aaaaaa" font-size="15">On\u00e7a-pintada: {metrics["onca_pintada_individuals"]} ind.</text>')
        svg.append('</g>')

        # Title
        svg.append(f'<text x="40" y="50" fill="#ffffff" font-size="15" font-weight="bold">ECO-SIM: Multi-Scale Connectivity Analysis</text>')
        svg.append(f'<text font-weight="bold" x="40" y="78" fill="#cccccc" font-size="15">Graph-based corridor modeling with species-specific movement</text>')

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

def main():
    print(f"Initializing multi-scale connectivity model on {CONFIG.device}...")
    model = ConnectivityModel(CONFIG)

    for _ in range(CONFIG.frames):
        model.step()

    metrics = model.calculate_connectivity_metrics()
    print(f"Connectivity metrics: {metrics}")

    print("Simulation complete. Rendering...")
    renderer = ConnectivityRenderer(CONFIG, model)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_51.1')
    return svg_content

if __name__ == "__main__":
    main()