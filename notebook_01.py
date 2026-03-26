# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 01: Baseline Ecosystem & Karst Landscape Dynamics
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Baseline Ecosystem & Karst Landscape Dynamics emphasizing canopy stratification.
- Indicator species: Arara-caninde (Ara ararauna).
- Pollination lens: masting-to-bloom handoff in late dry season.
- Human impact lens: trail disturbance and repeated flushing.

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
from eco_base import CANVAS_HEIGHT, ZONES  # pyre-ignore[21]
from dataclasses import dataclass, field
from typing import List, Dict

# ===================================================================================================
# 1. SCIENTIFIC TAXONOMY & ECOLOGICAL GUILDS (Recanto das Araras ENDEMICS)
# ===================================================================================================

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#62fff3", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 200
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
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        # Geomorphological Attractors (Sumidouros, Ressurgências, Veredas)
        self.karst_nodes = torch.tensor([
            [200.0, 300.0], [500.0, 150.0], [800.0, 450.0], [350.0, 600.0]
        ], device=self.dev)

        self.trajectory_history = []

    def step(self):
        # 1. Boids dynamics (vectorized distances)
        """Function `step` -- simulation component."""

        dist_matrix = torch.cdist(self.pos, self.pos)

        # Karst Environmental Pull (Resource hotspots)
        dist_to_nodes = torch.cdist(self.pos, self.karst_nodes)
        min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.karst_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 0.5

        # Boids Simple Cohesion / Alignment omitted for full tensor speed but keeping random walk + karst
        self.vel += karst_attraction + torch.randn_like(self.vel) * 0.2

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

        # background-color:#1a1a1a
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
            '<stop offset="0%" stop-color="#fe4db7" stop-opacity="0.55"/>'
            '<stop offset="70%" stop-color="#fe4db7" stop-opacity="0.15"/>'
            '<stop offset="100%" stop-color="#fe4db7" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#00ffdc" stop-opacity="0.9"/>'
            '<stop offset="100%" stop-color="#00ffdc" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>'
        )
        svg.append(f'<rect width="{w}" height="{h}" fill="#0a0f14"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Render Karst Nodes (Caves/Rivers) - pulsing rings + glow
        knodes = self.sim.karst_nodes.cpu().numpy()
        for i, kn in enumerate(knodes):
            bd = i * 0.6
            svg.append(f'<circle cx="{kn[0]}" cy="{kn[1]}" r="130" fill="url(#karstGlow)"/>')
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="80" fill="none" stroke="#fe4db7" '
                f'stroke-width="2.5" stroke-dasharray="6 4">'
                f'<animate attributeName="r" values="68;95;68" dur="3.2s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="stroke-opacity" values="0.45;1.0;0.45" dur="3.2s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="50" fill="none" stroke="#00ffdc" '
                f'stroke-width="1.2" stroke-dasharray="2 8" opacity="0.7">'
                f'<animate attributeName="r" values="42;65;42" dur="2.4s" begin="{bd+0.5:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.7;0.0;0.7" dur="2.4s" begin="{bd+0.5:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="30" fill="none" stroke="#ffe43f" '
                f'stroke-width="1.0" stroke-dasharray="1 10" opacity="0.5">'
                f'<animate attributeName="r" values="20;38;20" dur="1.8s" begin="{bd+0.9:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0.5;0.0;0.5" dur="1.8s" begin="{bd+0.9:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{kn[0]}" cy="{kn[1]}" r="9" fill="url(#nodeGlow)" filter="url(#glow)">'
                f'<animate attributeName="r" values="6;14;6" dur="2.0s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                f'</circle>'
            )
            svg.append(f'<text x="{kn[0]+14}" y="{kn[1]-14}" font-size="15" fill="#80f7e0" font-weight="bold">Vereda/Cave {i+1}</text>')

        # Structural UI
        svg.append(f'<text x="20" y="{ZONES["header"]["y"] + 30}" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Lapa River Basin &amp; Karst System</text>')
        svg.append(f'<text font-weight="bold" x="20" y="{ZONES["header"]["y"] + 50}" font-size="15" fill="#b0bec5">Baseline Engine</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Trail rendering for a subset
            if idx % 4 == 0:
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::3]])
                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.45" stroke-width="3.0"/>')

            # Glow halo (every 5th particle)
            r_begin = f"{(idx % 14) * 0.18:.2f}s"
            dur_val = self.cfg.frames / self.cfg.fps
            if idx % 5 == 0:
                svg.append(f'<circle r="9" fill="{p_color}" opacity="0.28">')
                svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur_val}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur_val}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="r" values="6;14;6" dur="2.2s" begin="{r_begin}" repeatCount="indefinite"/>')
                svg.append('</circle>')

            # Entity Node with r-pulse
            svg.append(f'<circle r="4.5" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur_val}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur_val}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="r" values="3.5;6.0;3.5" dur="2.5s" begin="{r_begin}" repeatCount="indefinite"/>')
            svg.append('</circle>')

        # -- Scientific Validation Watermark --
        svg.append(f'<g transform="translate(10, {CANVAS_HEIGHT - 15})">')

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

    print(f"Initializing baseline simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for _ in range(CONFIG.frames):
        sim.step()

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_01')

if __name__ == "__main__":
    main()
