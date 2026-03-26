# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 05: Roosting Behavior (Nighttime Tree Flocking)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Roosting Behavior (Nighttime Tree Flocking) emphasizing vereda hydrology pulses.
- Indicator species: Anta (Tapirus terrestris).
- Pollination lens: wind pollination during grass senescence.
- Human impact lens: grazing pressure on understory.

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
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#f44336", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 240 # Prompt requirement: 240 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    day_length: int = 120 # Half of the new 240 frame cycle to show a full day/night clearly
    roosting_radius: float = 30.0

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
        self.cohesions = torch.tensor([BIODIVERSITY_DB[guilds[i]]["cohesion"] for i in indices], device=self.dev)
        self.drags = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"] for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"] for i in indices], device=self.dev)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        # Karst Environmental Pull - Normal
        self.karst_nodes = torch.tensor([
            [200.0, 300.0], [500.0, 150.0], [800.0, 450.0], [350.0, 600.0]
        ], device=self.dev)

        # New: Roosting Trees (Specific coordinates for night)
        self.tree_nodes = torch.tensor([
            [100.0, 100.0], [1100.0, 150.0], [600.0, 500.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.daylight_history = []

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # Diurnal Cycle
        time_angle = (frame_idx / self.cfg.day_length) * 2 * math.pi
        sunlight = (math.sin(time_angle) + 1.0) / 2.0
        self.daylight_history.append(sunlight)

        is_night = sunlight < 0.25 # Threshold for entering roosting mode

        raw_new_vel = self.vel * self.drags + torch.randn_like(self.vel) * 0.5
        activity_multiplier = max(0.00, sunlight)

        if not is_night:
            # Daytime: Karst Environmental Pull
            dist_to_nodes = torch.cdist(self.pos, self.karst_nodes)
            min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)
            pull_vectors = self.karst_nodes[closest_nodes] - self.pos
            karst_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 1.5

            raw_new_vel += karst_attraction

        else:
            # Nighttime: Roosting Pull (Overpowers everything else)
            dist_to_trees = torch.cdist(self.pos, self.tree_nodes)
            min_dist_trees, closest_trees = torch.min(dist_to_trees, dim=1)
            pull_vectors_trees = self.tree_nodes[closest_trees] - self.pos

            # Massive magnetic pull to trees
            tree_attraction = (pull_vectors_trees / min_dist_trees.unsqueeze(1).clamp(min=1.0)) * 15.0

            # If inside roost radius, stop moving totally
            inside_roost = min_dist_trees < self.cfg.roosting_radius
            raw_new_vel[inside_roost] *= 0.0 # Stop
            tree_attraction[inside_roost] *= 0.0

            # Don't scale down speed if they haven't reached the tree yet!
            # We want them to fly fast to the tree at sunset, then stop.
            activity_multiplier = torch.ones_like(self.speeds)
            activity_multiplier[inside_roost] = 0.0  # Once inside, sleep.

            raw_new_vel += tree_attraction


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

        # Normalize to max speeds, scaled by diurnal activity
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        if isinstance(activity_multiplier, float):
             current_max_speeds = self.speeds * activity_multiplier
        else:
             current_max_speeds = self.speeds * activity_multiplier

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

        bg_colors = []
        for light in self.sim.daylight_history:
             bg_color = self.blend_colors("#15203b", "#05070a", light) # Slightly bluer "night" for visibility
             bg_colors.append(bg_color)

        bg_color_str = ";".join(bg_colors)

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#121212; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append('<defs><pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/></pattern></defs>')

        dur = self.cfg.frames / self.cfg.fps

        # Animated background rect
        svg.append(f'<rect width="{w}" height="{h}" fill="#121212">')
        svg.append(f'<animate attributeName="fill" values="{bg_color_str}" dur="{dur}s" fill="freeze" calcMode="linear"/>')
        svg.append('</rect>')

        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Render Tree Nodes (Roosting locations) - Draw these first so birds cluster OVER them
        tnodes = self.sim.tree_nodes.cpu().numpy()
        for i, tn in enumerate(tnodes):
            # Inner circle (Canopy)
            svg.append(f'<circle cx="{tn[0]}" cy="{tn[1]}" r="{self.cfg.roosting_radius}" fill="#4caf50" opacity="0.4"/>')
            # Outer boundary (Trunk marker)
            svg.append(f'<circle cx="{tn[0]}" cy="{tn[1]}" r="6" fill="#795548"/>')
            svg.append(f'<circle cx="{tn[0]}" cy="{tn[1]}" r="8" fill="none" stroke="#4caf50" stroke-width="2"/>')
            svg.append(f'<text font-weight="bold" x="{tn[0]+15}" y="{tn[1]-15}" font-size="15" fill="#cccccc">Roost Tree {i+1}</text>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Roosting Behavior</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Night Time Flocking Dynamics</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Trail rendering
            if idx % 8 == 0:
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::5]])
                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.3" stroke-width="1.5"/>')

            # Entity Node
            svg.append(f'<circle r="4.5" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append('</circle>')

        # Dynamic Sun/Moon indicator
        sun_op_vals = ";".join([f"{light:.2f}" for light in self.sim.daylight_history])
        moon_op_vals = ";".join([f"{1.0 - light:.2f}" for light in self.sim.daylight_history])

        sun_icon = f'<circle cx="1200" cy="50" r="20" fill="#ffb300"><animate attributeName="opacity" values="{sun_op_vals}" dur="{dur}s" fill="freeze"/></circle>'
        moon_icon = f'<circle cx="1200" cy="50" r="16" fill="#eceff1"><animate attributeName="opacity" values="{moon_op_vals}" dur="{dur}s" fill="freeze"/></circle>'

        svg.append(sun_icon)
        svg.append(moon_icon)

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

    print(f"Initializing roosting behavior simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_05')

if __name__ == "__main__":
    main()
