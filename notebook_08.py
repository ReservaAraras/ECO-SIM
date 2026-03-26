# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 08: Foraging Efficiency (Speed Reduction in Resource Zones)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Foraging Efficiency (Speed Reduction in Resource Zones) emphasizing campo rupestre mosaics.
- Indicator species: Tatu-canastra (Priodontes maximus).
- Pollination lens: ant-guarded nectar dynamics.
- Human impact lens: restoration weeding benefits.

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
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1, "foraging_speed": 1.2},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15, "foraging_speed": 1.5},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4, "foraging_speed": 1.0},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#f44336", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2, "foraging_speed": 3.0}, # Predators don't slow down as much
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 270 # Prompt requirement: 270 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    resource_radius: float = 60.0

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
        self.foraging_speeds = torch.tensor([BIODIVERSITY_DB[guilds[i]]["foraging_speed"] for i in indices], device=self.dev)
        self.drags = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"] for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"] for i in indices], device=self.dev)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        # Resource Hotspots
        self.resource_nodes = torch.tensor([
            [250.0, 250.0], [850.0, 180.0], [550.0, 480.0], [1050.0, 350.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.is_foraging_history = [] # Boolean mask to track when they are in the zone

    def step(self):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # Check distance to resources
        dist_to_nodes = torch.cdist(self.pos, self.resource_nodes)
        min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)

        # Mask indicating if particle is within the foraging zone
        foraging_mask = min_dist < self.cfg.resource_radius
        self.is_foraging_history.append(foraging_mask.cpu().numpy().copy())

        # Base attraction to resources
        pull_vectors = self.resource_nodes[closest_nodes] - self.pos
        resource_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 1.5

        # Once inside the resource, zero out the attraction so they wander randomly within the patch
        resource_attraction[foraging_mask] *= 0.1

        # Apply forces and random wander
        # Increase random wander when inside to simulate "searching"
        random_wander = torch.randn_like(self.vel) * 0.5
        random_wander[foraging_mask] *= 2.0 # More erratic movement while foraging

        raw_new_vel = self.vel * self.drags + resource_attraction + random_wander

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

        # Normalize to max speeds, BUT apply foraging speed reduction if in zone
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        current_max_speeds = self.speeds.clone()
        # Override with specifically lowered foraging speeds
        current_max_speeds[foraging_mask] = self.foraging_speeds[foraging_mask]

        self.vel = (self.vel / norms) * current_max_speeds.unsqueeze(1)

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

        # Render Resource Nodes (Foraging Areas)
        rnodes = self.sim.resource_nodes.cpu().numpy()
        for i, rn in enumerate(rnodes):
            svg.append(f'<circle cx="{rn[0]}" cy="{rn[1]}" r="{self.cfg.resource_radius}" fill="#8bc34a" opacity="0.25"/>')
            svg.append(f'<circle cx="{rn[0]}" cy="{rn[1]}" r="80" fill="none" stroke="#8bc34a" stroke-width="1.5" stroke-dasharray="2 6"/>')
            svg.append(f'<circle cx="{rn[0]}" cy="{rn[1]}" r="4" fill="#ffffff"/>')
            svg.append(f'<text font-weight="bold" x="{rn[0]+10}" y="{rn[1]-10}" font-size="15" fill="#cccccc">Resource Patch {i+1}</text>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Foraging Efficiency</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Restricted Kinematics in Resource Zones</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Animate color/radius depending on foraging state
            # When foraging -> slow speed implies higher focus, visual highlight (pulsing white overlay)
            color_vals = []
            for frame_idx in range(self.cfg.frames):
                 is_foraging = self.sim.is_foraging_history[frame_idx][idx]
                 if is_foraging:
                      color_vals.append("#ffffff") # Flash bright white when feeding
                 else:
                      color_vals.append(p_color)

            c_str = ";".join(color_vals)

            # Trail rendering
            if idx % 8 == 0:
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::4]])

                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.3" stroke-width="1.5"/>')

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

    print(f"Initializing foraging efficiency simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for _ in range(CONFIG.frames):
        sim.step()

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_08')

if __name__ == "__main__":
    main()
