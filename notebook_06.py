# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 06: Predator Avoidance (Herbivores flee from Carnivores)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Predator Avoidance (Herbivores flee from Carnivores) emphasizing savanna grassland matrix.
- Indicator species: Seriema (Cariama cristata).
- Pollination lens: riparian flowering pulse after rains.
- Human impact lens: poaching risk hot spots.

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
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1, "is_predator": False},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15, "is_predator": False},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4, "is_predator": False},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#f44336", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2, "is_predator": True},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 250 # Prompt requirement: 250 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    fear_radius: float = 90.0 # Distance at which prey starts actively fleeing
    chase_radius: float = 120.0 # Distance at which predators lock onto prey

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

        self.speeds = torch.tensor([BIODIVERSITY_DB[guilds[i]]["speed"] for i in indices], device=self.dev)
        self.drags = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"] for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"] for i in indices], device=self.dev)
        self.is_predator = torch.tensor([BIODIVERSITY_DB[guilds[i]]["is_predator"] for i in indices], device=self.dev, dtype=torch.bool)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        self.karst_nodes = torch.tensor([
            [200.0, 300.0], [500.0, 150.0], [800.0, 450.0], [350.0, 600.0]
        ], device=self.dev)

        self.trajectory_history = []

        # Precompute boolean masks for efficient tensor ops
        self.prey_mask = ~self.is_predator

    def step(self):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # 1. Base environmental pull (Karst)
        dist_to_nodes = torch.cdist(self.pos, self.karst_nodes)
        min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.karst_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 1.5

        raw_new_vel = self.vel * self.drags + karst_attraction + torch.randn_like(self.vel) * 0.5

        # 2. PREDATOR / PREY DYNAMICS (Vectorized)

        # Get distances between ALL particles
        dist_matrix = torch.cdist(self.pos, self.pos)

        # Ignore self-distance
        dist_matrix.fill_diagonal_(float('inf'))

        # Calculate Fleeing (Preys running from Predators)
        flee_vectors = torch.zeros_like(self.vel)

        # For each prey, find distances to all predators
        if self.prey_mask.any() and self.is_predator.any():
            prey_positions = self.pos[self.prey_mask]
            predator_positions = self.pos[self.is_predator]

            # Distance from every prey to every predator
            dist_prey_to_preds = torch.cdist(prey_positions, predator_positions)

            # Find the closest predator to each prey
            min_dist_to_pred, closest_pred_idx = torch.min(dist_prey_to_preds, dim=1)

            # Create a mask for preys that are within the fear radius of their closest predator
            fleeing_preys_mask = min_dist_to_pred < self.cfg.fear_radius

            if fleeing_preys_mask.any():
                # Extract the positions of the specific preys and predators involved
                scared_prey_pos = prey_positions[fleeing_preys_mask]
                scary_pred_pos = predator_positions[closest_pred_idx[fleeing_preys_mask]]

                # Vector FROM predator TO prey (Push away)
                push_dir = scared_prey_pos - scary_pred_pos
                push_dist = min_dist_to_pred[fleeing_preys_mask].unsqueeze(1).clamp(min=1.0)

                # Force is exponentially stronger the closer the predator is
                force_magnitude = 15.0 * (1.0 - (push_dist / self.cfg.fear_radius))

                push_vectors = (push_dir / push_dist) * force_magnitude

                # Map back to the full flee_vectors tensor
                flee_vectors[self.prey_mask.nonzero(as_tuple=True)[0][fleeing_preys_mask]] = push_vectors

            # 3. PREDATOR CHASE (Predators hunting closest Prey)
            chase_vectors = torch.zeros_like(self.vel)

            dist_pred_to_preys = torch.cdist(predator_positions, prey_positions)
            min_dist_to_prey, closest_prey_idx = torch.min(dist_pred_to_preys, dim=1)

            hunting_preds_mask = min_dist_to_prey < self.cfg.chase_radius

            if hunting_preds_mask.any():
                hunting_pred_pos = predator_positions[hunting_preds_mask]
                hunted_prey_pos = prey_positions[closest_prey_idx[hunting_preds_mask]]

                # Vector FROM predator TO prey (Pull towards)
                pull_dir = hunted_prey_pos - hunting_pred_pos
                pull_dist = min_dist_to_prey[hunting_preds_mask].unsqueeze(1).clamp(min=1.0)

                force_magnitude_chase = 5.0 # Predators are heavily locked on

                pull_vectors = (pull_dir / pull_dist) * force_magnitude_chase

                # Map back
                chase_vectors[self.is_predator.nonzero(as_tuple=True)[0][hunting_preds_mask]] = pull_vectors

            # OVERRIDE Physics
            # Apply fleeing to prey, chasing to predators
            raw_new_vel += flee_vectors + chase_vectors

        # Turn Radius (Angle Limiting)
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # When fleeing or chasing, allow for desperate sharper turns (double turn radius)
        dynamic_turn_rad = self.max_turns.clone()
        if self.prey_mask.any() and self.is_predator.any():
             # Extract indices of panicked/hunting agents
             panicked_idx = self.prey_mask.nonzero(as_tuple=True)[0][fleeing_preys_mask] if fleeing_preys_mask.any() else []
             hunting_idx = self.is_predator.nonzero(as_tuple=True)[0][hunting_preds_mask] if hunting_preds_mask.any() else []

             dynamic_turn_rad[panicked_idx] *= 2.5 # Preys panic and turn instantly to survive
             dynamic_turn_rad[hunting_idx] *= 1.5   # Predators snap to prey

        clamped_diff = torch.clamp(angle_diff, -dynamic_turn_rad, dynamic_turn_rad)
        final_angles = old_angles + clamped_diff

        # Calculate speed magnitudes, give a desperate speed boost to fleeing prey
        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        # Normalize to max speeds
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        dynamic_max_speeds = self.speeds.clone()
        if self.prey_mask.any() and self.is_predator.any() and fleeing_preys_mask.any():
             dynamic_max_speeds[panicked_idx] *= 1.4 # Panic speed burst

        self.vel = (self.vel / norms) * dynamic_max_speeds.unsqueeze(1)

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
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Predator Avoidance Engine</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Carnivore vs Herbivore Dynamics</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            is_pred = self.sim.is_predator[idx].item()

            # Render threat radii for Predators
            if is_pred:
                # Add pulsing danger zone
                svg.append(f'<circle r="{self.cfg.fear_radius}" fill="none" stroke="#f44336" stroke-width="1.0" stroke-opacity="0.2" stroke-dasharray="2 4">')
                svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
                svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')
                svg.append('</circle>')

            # Trail rendering (Red for predators, subtle for prey)
            if idx % 8 == 0 or is_pred:
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::4]])

                stroke_w = 2.5 if is_pred else 1.5
                stroke_op = 0.5 if is_pred else 0.3

                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="{stroke_op}" stroke-width="{stroke_w}"/>')

            # Entity Node
            rad = 6.0 if is_pred else 4.5
            svg.append(f'<circle r="{rad}" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')
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

    print(f"Initializing predator avoidance simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for _ in range(CONFIG.frames):
        sim.step()

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_06')

if __name__ == "__main__":
    main()
