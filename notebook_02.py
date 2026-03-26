# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 02: Diverse Flight Patterns (Aerodynamics & Turn Radius)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Diverse Flight Patterns (Aerodynamics & Turn Radius) emphasizing edge microclimate.
- Indicator species: Lobo-guara (Chrysocyon brachyurus).
- Pollination lens: nocturnal bat pollination window.
- Human impact lens: road mortality and barrier effects.

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

# Added 'drag' (damping factor) and 'max_turn' (radians per step) to simulate aerodynamic differences
BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#62fff3", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 210 # Prompt requirement: 210 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5

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

        # Track species for visualization
        self.is_tucano = torch.tensor([guilds[i] == "Tucano-toco (Ramphastos toco)" for i in indices], device=self.dev)
        self.is_hummingbird = torch.tensor([guilds[i] == "Beija-flor-tesoura (Eupetomena macroura)" for i in indices], device=self.dev)

        self.karst_nodes = torch.tensor([
            [200.0, 300.0], [500.0, 150.0], [800.0, 450.0], [350.0, 600.0]
        ], device=self.dev)

        self.trajectory_history = []

    def step(self):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # Karst Environmental Pull
        dist_to_nodes = torch.cdist(self.pos, self.karst_nodes)
        min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.karst_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 1.5

        # Apply forces and random wander
        raw_new_vel = self.vel * self.drags + karst_attraction + torch.randn_like(self.vel) * 0.5

        # Turn Radius (Angle Limiting)
        # Calculate current angles and target angles
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])

        angle_diff = new_angles - old_angles
        # Normalize angle diff to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Clamping turn angles
        clamped_diff = torch.clamp(angle_diff, -self.max_turns, self.max_turns)
        final_angles = old_angles + clamped_diff

        # Reconstruct velocity with clamped angles
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

    def generate_svg(self) -> str:
        """Function `generate_svg` -- simulation component."""

        w, h = self.cfg.width, self.cfg.height
        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#121212; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append(
            '<defs>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="4.5" fill="#3d9df7" opacity="0.55"/>'
            '</pattern>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="karstGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#fe4db7" stop-opacity="0.5"/>'
            '<stop offset="100%" stop-color="#fe4db7" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="hmbgGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#ffe43f" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#ffe43f" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>'
        )
        svg.append(f'<rect width="{w}" height="{h}" fill="#0a0f14"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Render Karst Nodes - pulsing rings + glow
        knodes = self.sim.karst_nodes.cpu().numpy()
        for i, kn in enumerate(knodes):
            bd = i * 0.5
            svg.append(f'<circle cx="{kn[0]}" cy="{kn[1]}" r="115" fill="url(#karstGlow)"/>')
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="80" fill="none" stroke="#fe4db7" stroke-width="2.5" stroke-dasharray="6 4">'
                f'<animate attributeName="r" values="68;96;68" dur="3.0s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="stroke-opacity" values="0.4;1.0;0.4" dur="3.0s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="52" fill="none" stroke="#00ffdc" stroke-width="1.2" stroke-dasharray="2 8">'
                f'<animate attributeName="r" values="44;66;44" dur="2.2s" begin="{bd+0.4:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.7;0.0;0.7" dur="2.2s" begin="{bd+0.4:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="9" fill="#00ffdc" filter="url(#glow)">'
                f'<animate attributeName="r" values="6;14;6" dur="2.0s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(f'<text x="{kn[0]+12}" y="{kn[1]-12}" font-size="15" fill="#80f7e0" font-weight="bold">Vereda/Cave {i+1}</text>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Diverse Flight Patterns</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Aerodynamic Drag &amp; Turn Radius</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Trail rendering - vivid, species-differentiated
            is_hmbg = self.sim.is_hummingbird[idx].item()
            is_toc = self.sim.is_tucano[idx].item()

            if (is_hmbg or is_toc) and (idx % 2 == 0):
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::3]])
                stroke_width = 3.5 if is_hmbg else 2.5
                stroke_op = 0.65 if is_hmbg else 0.4
                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="{stroke_op}" stroke-width="{stroke_width}"/>')

            # Glow halo (hummingbirds get vivid gold glow, toucans get pink glow)
            r_begin = f"{(idx % 14) * 0.15:.2f}s"
            dur_val = self.cfg.frames / self.cfg.fps
            if is_hmbg and idx % 3 == 0:
                svg.append(f'<circle r="12" fill="url(#hmbgGlow)" opacity="0.4">')
                svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur_val}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur_val}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="r" values="8;16;8" dur="0.9s" begin="{r_begin}" repeatCount="indefinite"/>')
                svg.append('</circle>')
            elif is_toc and idx % 5 == 0:
                svg.append(f'<circle r="10" fill="{p_color}" opacity="0.22">')
                svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur_val}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur_val}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="r" values="7;14;7" dur="2.4s" begin="{r_begin}" repeatCount="indefinite"/>')
                svg.append('</circle>')

            # Entity Node - fast pulse for hummingbird, slow for toucan
            r_dur = "0.7s" if is_hmbg else "2.5s"
            r_vals = "3.5;6.5;3.5" if is_hmbg else "4.0;5.8;4.0"
            svg.append(f'<circle r="4.5" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur_val}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur_val}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="r" values="{r_vals}" dur="{r_dur}" begin="{r_begin}" repeatCount="indefinite"/>')
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

    print(f"Initializing aerodynamic simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for _ in range(CONFIG.frames):
        sim.step()

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_02')

if __name__ == "__main__":
    main()
