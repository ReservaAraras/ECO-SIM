# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 04: Water Dependence (Energy Recharge at Veredas)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Water Dependence (Energy Recharge at Veredas) emphasizing karst sinkhole gradients.
- Indicator species: Onca-pintada (Panthera onca).
- Pollination lens: bee foraging under smoke haze.
- Human impact lens: selective logging canopy loss.

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
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1, "water_depletion": 0.4},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15, "water_depletion": 0.6},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4, "water_depletion": 1.2},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#f44336", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2, "water_depletion": 0.2},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 230 # Prompt requirement: 230 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    day_length: int = 110
    water_threshold: float = 30.0 # When water < threshold, prioritize moving to water
    water_radius: float = 40.0 # Radius to recharge

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
        self.water_depletions = torch.tensor([BIODIVERSITY_DB[guilds[i]]["water_depletion"] for i in indices], device=self.dev)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        # New State: Water Level [0.0 to 100.0]
        self.water_levels = torch.ones(cfg.num_particles, device=self.dev) * 100.0

        self.karst_nodes = torch.tensor([
            [200.0, 300.0], [500.0, 150.0], [800.0, 450.0], [350.0, 600.0]
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
        activity_multiplier = max(0.05, sunlight)

        # Water Depletion (Faster during the day!)
        self.water_levels -= self.water_depletions * activity_multiplier * self.cfg.dt

        # Find closest water nodes
        dist_to_nodes = torch.cdist(self.pos, self.karst_nodes)
        min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)

        # Recharge if inside water radius
        in_water = min_dist < self.cfg.water_radius
        self.water_levels[in_water] += 15.0 # Fast recharge

        self.water_levels = torch.clamp(self.water_levels, 0.0, 100.0)

        # Karst Environmental Pull - Normal
        pull_vectors = self.karst_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 1.0

        # Thirsty Pull - Exponentially stronger as water level drops
        thirsty_mask = self.water_levels < self.cfg.water_threshold
        thirst_factor = (self.cfg.water_threshold - self.water_levels[thirsty_mask]) / self.cfg.water_threshold # 0.0 to 1.0

        # Override karst attraction with absolute desperation pull towards water for thirsty birds
        karst_attraction[thirsty_mask] += (pull_vectors[thirsty_mask] / min_dist[thirsty_mask].unsqueeze(1).clamp(min=1.0)) * (thirst_factor.unsqueeze(1) * 8.0)

        # Apply forces and random wander
        raw_new_vel = self.vel * self.drags + karst_attraction + torch.randn_like(self.vel) * 0.5

        # Turn Radius (Angle Limiting)
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        clamped_diff = torch.clamp(angle_diff, -self.max_turns, self.max_turns)
        final_angles = old_angles + clamped_diff

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        # Apply strict penalty if completely out of water
        dehydrated_mask = self.water_levels <= 0.0
        speed_magnitudes[dehydrated_mask] *= 0.1 # Move 90% slower if dying of thirst

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        # Normalize to max speeds, scaled by diurnal activity
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)
        current_max_speeds = self.speeds * activity_multiplier
        self.vel = (self.vel / norms) * current_max_speeds.unsqueeze(1)

        # Update positions
        self.pos += self.vel * self.cfg.dt

        # Periodic boundaries
        self.pos[:, 0] = self.pos[:, 0] % self.cfg.width
        self.pos[:, 1] = self.pos[:, 1] % self.cfg.height

        # For rendering, we want to know if particle is thirsty or dehydrated
        snapshot = torch.cat([self.pos, self.water_levels.unsqueeze(1)], dim=1)
        self.trajectory_history.append(snapshot.cpu().numpy().copy())

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

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#080e16; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append(
            '<defs>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="4" fill="#3d9df7" opacity="0.45"/>'
            '</pattern>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="waterNodeGrad" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#00d4ff" stop-opacity="0.7"/>'
            '<stop offset="60%" stop-color="#0066cc" stop-opacity="0.25"/>'
            '<stop offset="100%" stop-color="#0066cc" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="thirstGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#ff5722" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#ff5722" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>'
        )

        dur = self.cfg.frames / self.cfg.fps

        # Animated background
        svg.append(f'<rect width="{w}" height="{h}" fill="#080e16">')
        svg.append(f'<animate attributeName="fill" values="{bg_color_str}" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>')
        svg.append('</rect>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Render Karst Nodes - vivid ripple rings + gradient fill
        knodes = self.sim.karst_nodes.cpu().numpy()
        for i, kn in enumerate(knodes):
            bd = i * 0.7
            svg.append(f'<circle cx="{kn[0]}" cy="{kn[1]}" r="{self.cfg.water_radius + 40}" fill="url(#waterNodeGrad)"/>')
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="{self.cfg.water_radius}" fill="#00bbff" opacity="0.22">'
                f'<animate attributeName="r" values="{self.cfg.water_radius*0.85:.0f};{self.cfg.water_radius*1.1:.0f};{self.cfg.water_radius*0.85:.0f}" dur="2.8s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.15;0.35;0.15" dur="2.8s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            for ri, (rr, rd, rop) in enumerate([(80,3.2,0.6),(55,2.4,0.55),(35,1.8,0.5)]):
                svg.append(
                    f'<circle cx="{kn[0]}" cy="{kn[1]}" r="{rr}" fill="none" stroke="#00d4ff" stroke-width="1.5">'
                    f'<animate attributeName="r" values="{rr-16};{rr+16};{rr-16}" dur="{rd}s" begin="{bd+ri*0.55:.2f}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="opacity" values="{rop};0.0;{rop}" dur="{rd}s" begin="{bd+ri*0.55:.2f}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="7" fill="#00d4ff" filter="url(#glow)">'
                f'<animate attributeName="r" values="5;11;5" dur="1.8s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(f'<text x="{kn[0]+12}" y="{kn[1]-12}" font-size="15" fill="#66e8ff" font-weight="bold">Vereda (Water) {i+1}</text>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Water Dependence</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Energy Recharge Dynamics</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Animate color based on thirst (Normal -> Red when thirsty, pulsing)
            color_vals = []
            for frame_idx in range(self.cfg.frames):
                 water = self.sim.trajectory_history[frame_idx][idx, 2]
                 if water <= 0.0:
                      color_vals.append("#ff0000") # Dead/dying
                 elif water < self.cfg.water_threshold:
                      # If thirsty, interpolate towards orange/red
                      factor = water / self.cfg.water_threshold
                      color_vals.append(self.blend_colors(p_color, "#ff5722", factor))
                 else:
                      color_vals.append(p_color)

            c_str = ";".join(color_vals)

            # Vivid trail rendering
            if idx % 6 == 0:
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::4]])
                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.4" stroke-width="2.0"/>')

            # Glow halo (thirsty birds get desperate red glow, healthy get species glow)
            r_begin = f"{(idx % 14) * 0.18:.2f}s"
            if idx % 6 == 0:
                svg.append(f'<circle r="9" fill="{p_color}" opacity="0.25">')
                svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="r" values="6;14;6" dur="2.0s" begin="{r_begin}" repeatCount="indefinite"/>')
                svg.append('</circle>')

            # Entity Node - r-pulse faster when thirsty
            r_vals_list = []
            for fi in range(self.cfg.frames):
                water = self.sim.trajectory_history[fi][idx, 2]
                if water <= 0.0:
                    r_vals_list.append("2.5")
                elif water < self.cfg.water_threshold:
                    r_vals_list.append("6.0")
                else:
                    r_vals_list.append("4.5")
            r_vals_str = ";".join(r_vals_list)

            svg.append(f'<circle r="4.5" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="fill" values="{c_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append(f'<animate attributeName="r" values="{r_vals_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
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

    print(f"Initializing water dependence simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_04')

if __name__ == "__main__":
    main()
