# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 07: Flocking Cohesion (Dynamic Weights based on Flock Size)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Flocking Cohesion (Dynamic Weights based on Flock Size) emphasizing gallery forest refugia.
- Indicator species: Ema (Rhea americana).
- Pollination lens: orchid mimicry with specialist bees.
- Human impact lens: invasive grass spread.

Scientific Relevance (PIGT RESEX Recanto das Araras -- 2024):
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
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from eco_base import CANVAS_HEIGHT, ZONES
from dataclasses import dataclass, field
from typing import List, Dict

# ===================================================================================================
# 1. SCIENTIFIC TAXONOMY & ECOLOGICAL GUILDS (Recanto das Araras ENDEMICS)
# ===================================================================================================

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion_base": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion_base": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion_base": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion_base": 0.005, "color": "#62fff3", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 260 # Prompt requirement: 260 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    neighbor_radius: float = 80.0
    perception_radius: float = 150.0

CONFIG = SimulationConfig()

# ===================================================================================================
# 3. KINEMATIC & ECOLOGICAL ENGINE (TENSORIZED)
# ===================================================================================================

class TerraRoncaEcosystem:
    """Class `TerraRoncaEcosystem` -- simulation component."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Spatial setup
        self.pos = torch.rand((cfg.num_particles, 2), device=self.dev) * torch.tensor([cfg.width, cfg.height], device=self.dev)
        self.vel = (torch.rand((cfg.num_particles, 2), device=self.dev) - 0.5) * 10.0

        # Assign guilds based on weighting
        guilds = list(BIODIVERSITY_DB.keys())
        weights = [BIODIVERSITY_DB[g]["weight"] for g in guilds]
        indices = np.random.choice(len(guilds), size=cfg.num_particles, p=weights)

        # Vectorized Species Identifiers (For specific grouping logic)
        device = self.dev
        self.species_ids = torch.tensor(indices, device=device)

        self.speeds = torch.tensor([BIODIVERSITY_DB[guilds[i]]["speed"] for i in indices], device=self.dev)
        self.cohesions_base = torch.tensor([BIODIVERSITY_DB[guilds[i]]["cohesion_base"] for i in indices], device=self.dev).unsqueeze(1)
        self.drags = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"] for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"] for i in indices], device=self.dev)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        self.karst_nodes = torch.tensor([
            [200.0, 300.0], [500.0, 150.0], [800.0, 450.0], [350.0, 600.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.flock_size_history = [] # To store dynamic flock size for rendering
        self.dynamic_cohesion_history = [] # Store true cohesion magnitude for viz

    def step(self):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # Base environmental pull (Karst)
        dist_to_nodes = torch.cdist(self.pos, self.karst_nodes)
        min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.karst_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 0.5 # Weakened base pull

        # ---------------------------------------------------------
        # FLOCKING DYNAMICS - DYNAMIC COHESION MULTIPLIER
        # ---------------------------------------------------------
        # Distance between all particles
        dist_matrix = torch.cdist(self.pos, self.pos)
        dist_matrix.fill_diagonal_(float('inf'))

        # 1. Find neighbors of same species within perception radius
        same_species_mask = (self.species_ids.unsqueeze(1) == self.species_ids.unsqueeze(0))
        neighbor_mask = (dist_matrix < self.cfg.perception_radius) & same_species_mask

        # 2. Calculate Flock Size (Density) per particle
        flock_sizes = neighbor_mask.sum(dim=1).float()

        # Optional: Save for rendering
        self.flock_size_history.append(flock_sizes.cpu().numpy().copy())

        # 3. Calculate Cohesion Center of Mass
        # Only for particles that have neighbors
        has_neighbors = flock_sizes > 0

        cohesion_vectors = torch.zeros_like(self.pos)

        if has_neighbors.any():
             # Create positional tensors masked by neighborhood
             # neighbor_mask specifies [i, j] True if j is neighbor to i
             # Mask positions: Pos [num_particles, 2] -> [num_particles, num_particles, 2]
             pos_expanded = self.pos.unsqueeze(0).expand(self.cfg.num_particles, -1, -1)

             # Sum positions of valid neighbors
             masked_pos = pos_expanded * neighbor_mask.unsqueeze(2).float()

             # Calculate Center of Mass per particle
             center_of_mass = masked_pos.sum(dim=1)[has_neighbors] / flock_sizes[has_neighbors].unsqueeze(1)

             # Vector towards center of mass
             cohesion_vectors[has_neighbors] = center_of_mass - self.pos[has_neighbors]

             # 4. **Dynamic Scaling based on Flock Size**
             # Base cohesion ranges around 0.05.
             # We want a logarithmic or bounded scaling effect where larger groups exert MUCH
             # stronger gravitational pull to tighten the flock specifically.
             # Max scale arbitrarily capped at 30 members.
             size_multiplier = (torch.clamp(flock_sizes[has_neighbors], max=30.0) / 5.0) # Normalizes roughly around 1.0 to 6.0

             dynamic_cohesion_weight = self.cohesions_base[has_neighbors] * size_multiplier.unsqueeze(1)

             # Record for SVGs
             cohesion_magnitudes = torch.zeros(self.cfg.num_particles, device=self.dev)
             cohesion_magnitudes[has_neighbors] = torch.norm(dynamic_cohesion_weight, dim=1)
             self.dynamic_cohesion_history.append(cohesion_magnitudes.cpu().numpy().copy())

             cohesion_vectors[has_neighbors] *= dynamic_cohesion_weight
        else:
             self.dynamic_cohesion_history.append(torch.zeros(self.cfg.num_particles).numpy().copy())


        # Combine forces
        raw_new_vel = self.vel * self.drags + karst_attraction + cohesion_vectors * 8.0 + torch.randn_like(self.vel) * 0.5

        # Turn Radius (Angle Limiting)
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        clamped_diff = torch.clamp(angle_diff, -self.max_turns, self.max_turns)
        final_angles = old_angles + clamped_diff

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        # Normalize to max speeds
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)
        self.vel = (self.vel / norms) * self.speeds.unsqueeze(1)

        # Update positions
        self.pos += self.vel * self.cfg.dt

        # Periodic boundaries
        self.pos[:, 0] = self.pos[:, 0] % self.cfg.width
        self.pos[:, 1] = self.pos[:, 1] % self.cfg.height

        self.trajectory_history.append(self.pos.cpu().numpy().copy())

# ===================================================================================================
# 4. SCIENTIFIC VISUALIZATION (DATA-DRIVEN SVG)
# ===================================================================================================

class EcosystemRenderer:
    """Class `EcosystemRenderer` -- simulation component."""

    def __init__(self, cfg: SimulationConfig, sim: TerraRoncaEcosystem):
        self.cfg = cfg
        self.sim = sim

    def hex_to_rgb(self, hex_color):
        """Function `hex_to_rgb` -- simulation component."""

        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, rgb):
        """Function `rgb_to_hex` -- simulation component."""

        return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

    def blend_colors(self, col1, col2, factor):
        """Function `blend_colors` -- simulation component."""

        factor = max(0.0, min(1.0, factor))
        c1 = self.hex_to_rgb(col1)
        c2 = self.hex_to_rgb(col2)
        blended = [c1[i] * factor + c2[i] * (1 - factor) for i in range(3)]
        return self.rgb_to_hex(blended)

    def generate_svg(self) -> str:
        """Function `generate_svg` -- simulation component."""

        w, h = self.cfg.width, self.cfg.height
        dur = self.cfg.frames / self.cfg.fps

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#121212; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append('<defs><pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/></pattern></defs>')

        # Static background rect
        svg.append(f'<rect width="{w}" height="{h}" fill="#121212"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Flocking Cohesion</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Dynamic Weights Based on Density</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Animate color to visually denote dynamic flock size cohesion link.
            # Base color -> Bright White when in a massive flock
            color_vals = []
            max_flock_expected = 15.0
            for frame_idx in range(self.cfg.frames):
                 f_size = self.sim.flock_size_history[frame_idx][idx]
                 blend_factor = min(1.0, f_size / max_flock_expected)
                 color_vals.append(self.blend_colors("#ffffff", p_color, blend_factor))

            c_str = ";".join(color_vals)

            # Trail rendering
            if idx % 5 == 0:
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::4]])

                # Check average flock size over the run to assign trailing thickness
                avg_flock = sum([self.sim.flock_size_history[f][idx] for f in range(self.cfg.frames)]) / self.cfg.frames
                stroke_w = min(4.0, 1.0 + (avg_flock / 3.0))

                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.35" stroke-width="{stroke_w:.1f}"/>')

            # Entity Node
            rad = 4.5
            svg.append(f'<circle r="{rad}" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="fill" values=\"{c_str}\" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append('</circle>')

        # -- Scientific Validation Watermark --
        svg.append(f'<g transform="translate(10, {h - 15})">')

        svg.append('</g>')
        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

# ===================================================================================================
# 5. EXECUTION BOOTSTRAP
# ===================================================================================================

def save_svg_to_drive(svg_content: str, notebook_id: str):
    """Function `save_svg_to_drive` -- simulation component."""

    import os
    drive_folder = "/content/drive/MyDrive/ReservaAraras_SVGs"
    if os.path.isdir('/content/drive'):
        save_dir = drive_folder
    else:
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        save_dir = os.path.join(base_dir, 'svg_output')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'{notebook_id}.svg')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"SVG saved -> {filepath}")
    return filepath

def main():
    """Function `main` -- simulation component."""

    print(f"Initializing flocking cohesion simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for _ in range(CONFIG.frames):
        sim.step()

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_07')

if __name__ == "__main__":
    main()
