# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 15: Seasonal Water (Dynamic Veredas Expansion & Contraction)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Seasonal Water (Dynamic Veredas Expansion & Contraction) emphasizing pollinator guild shifts.
- Indicator species: Perereca-verde (Hypsiboas albopunctatus).
- Pollination lens: hummingbird territorial patches.
- Human impact lens: fire management tradeoffs.

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
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1, "is_territorial": False, "mating_frequency": 0.05},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15, "is_territorial": False, "mating_frequency": 0.02},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4, "is_territorial": True, "mating_frequency": 0.08},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#f44336", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2, "is_territorial": False, "mating_frequency": 0.01},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 340 # Prompt requirement: 340 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    territory_radius: float = 70.0
    attack_force: float = 12.0
    breeding_season_start: int = 400
    breeding_season_end: int = 400
    spiral_speed_multiplier: float = 1.3
    obstacle_buffer: float = 30.0
    cave_entry_radius: float = 30.0
    cave_time_min: int = 15
    cave_time_max: int = 40

    rain_start: int = 400
    rain_end: int = 400
    rain_force_multiplier: float = 8.0
    cover_radius: float = 65.0

    # NEW: Seasonal Water (Veredas)
    vereda_min_radius: float = 20.0
    vereda_max_radius: float = 90.0
    vereda_cycle_frames: float = 200.0 # Time for a full dry/wet cycle
    vereda_base_attraction: float = 3.0

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
        self.is_territorial = torch.tensor([BIODIVERSITY_DB[guilds[i]]["is_territorial"] for i in indices], device=self.dev, dtype=torch.bool)
        self.mating_frequencies = torch.tensor([BIODIVERSITY_DB[guilds[i]]["mating_frequency"] for i in indices], device=self.dev)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        self.is_male = torch.rand(cfg.num_particles, device=self.dev) > 0.5

        self.is_displaying = torch.zeros(cfg.num_particles, device=self.dev, dtype=torch.bool)
        self.display_timer = torch.zeros(cfg.num_particles, device=self.dev)
        self.display_centers = torch.zeros((cfg.num_particles, 2), device=self.dev)

        self.is_underground = torch.zeros(cfg.num_particles, device=self.dev, dtype=torch.bool)
        self.underground_timer = torch.zeros(cfg.num_particles, device=self.dev)
        self.underground_duration = torch.zeros(cfg.num_particles, device=self.dev)

        # Environment
        self.nectar_nodes = torch.tensor([
            [250.0, 300.0]
        ], device=self.dev)

        self.obstacle_nodes = torch.tensor([
            [400.0, 300.0, 60.0]
        ], device=self.dev)

        self.cave_nodes = torch.tensor([
            [150.0, 450.0]
        ], device=self.dev)

        self.cover_nodes = torch.tensor([
            [1050.0, 200.0]
        ], device=self.dev)

        # NEW Vereda Nodes (Seasonal wetlands)
        self.vereda_nodes = torch.tensor([
            [600.0, 350.0], [900.0, 450.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.visibility_history = []
        self.sheltered_state = []

        self.vereda_radius_history = [] # Track dynamic sizes for visualization

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # Determine global conditions
        is_raining = self.cfg.rain_start <= frame_idx <= self.cfg.rain_end

        # Base environmental pulls
        dist_to_nodes = torch.cdist(self.pos, self.nectar_nodes)
        min_dist_to_node, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.nectar_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist_to_node.unsqueeze(1).clamp(min=1.0)) * 1.0

        dist_to_caves = torch.cdist(self.pos, self.cave_nodes)
        min_dist_cave, closest_caves = torch.min(dist_to_caves, dim=1)
        cave_pull_vectors = self.cave_nodes[closest_caves] - self.pos
        cave_attraction = (cave_pull_vectors / min_dist_cave.unsqueeze(1).clamp(min=1.0)) * 2.0

        rain_vector = torch.zeros_like(self.vel)
        cover_attraction = torch.zeros_like(self.vel)

        dist_to_covers = torch.cdist(self.pos, self.cover_nodes)
        min_dist_cover, closest_covers = torch.min(dist_to_covers, dim=1)
        is_sheltered = (min_dist_cover < self.cfg.cover_radius) & ~self.is_underground
        self.sheltered_state.append(is_sheltered.cpu().numpy().copy())

        if is_raining:
             exposed_mask = ~is_sheltered & ~self.is_underground
             if exposed_mask.any():
                  rain_vector[exposed_mask, 1] += self.cfg.rain_force_multiplier
                  cover_pull = self.cover_nodes[closest_covers[exposed_mask]] - self.pos[exposed_mask]
                  cover_force = (cover_pull / min_dist_cover[exposed_mask].unsqueeze(1).clamp(min=1.0)) * 6.0
                  cover_attraction[exposed_mask] = cover_force
                  karst_attraction[exposed_mask] *= 0.1
                  cave_attraction[exposed_mask] *= 0.1

        # -------------------------------------------------------------------------
        # NEW: VEREDA SEASONAL HYDROLOGY (Sinusoidal Expansion & Contraction)
        # -------------------------------------------------------------------------
        # Calculate current water level dynamically
        cycle_progress = (frame_idx % self.cfg.vereda_cycle_frames) / self.cfg.vereda_cycle_frames
        # Math: Sinusoid spanning from -1 to 1, scaled to 0 to 1
        water_level = (math.sin(cycle_progress * 2 * math.pi - math.pi/2) + 1.0) / 2.0

        current_v_radius = self.cfg.vereda_min_radius + (self.cfg.vereda_max_radius - self.cfg.vereda_min_radius) * water_level
        self.vereda_radius_history.append(current_v_radius)

        dist_to_veredas = torch.cdist(self.pos, self.vereda_nodes)
        min_dist_vereda, closest_veredas = torch.min(dist_to_veredas, dim=1)

        # Their attraction to the water scales with how much water there is!
        # If it's totally dry (radius = min), low attraction. If full, strong oasis pull.
        dynamic_attraction = self.cfg.vereda_base_attraction * water_level
        vereda_pull_vectors = self.vereda_nodes[closest_veredas] - self.pos
        vereda_attraction = (vereda_pull_vectors / min_dist_vereda.unsqueeze(1).clamp(min=1.0)) * dynamic_attraction

        # When floating directly over/inside the flooded area, wander happily
        flooded_mask = min_dist_vereda < current_v_radius
        vereda_attraction[flooded_mask] *= 0.1


        # COMBINE FORCES
        raw_new_vel = torch.zeros_like(self.vel)
        surface_mask = ~self.is_underground

        if surface_mask.any():
             base_forces = (self.vel[surface_mask] * self.drags[surface_mask] +
                          karst_attraction[surface_mask] +
                          cave_attraction[surface_mask] +
                          vereda_attraction[surface_mask] + # Add Vereda attraction
                          torch.randn_like(self.vel[surface_mask]) * 0.5)

             weather_forces = rain_vector[surface_mask] + cover_attraction[surface_mask]
             raw_new_vel[surface_mask] = base_forces + weather_forces

        if is_raining and is_sheltered.any():
             raw_new_vel[is_sheltered] *= 0.1

        raw_new_vel[self.is_underground] *= 0.0

        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        final_angles = old_angles.clone()

        if surface_mask.any():
             dynamic_max_turns = self.max_turns[surface_mask].clone()
             if is_raining:
                 dynamic_max_turns *= 1.5
             clamped_diff = torch.clamp(angle_diff[surface_mask], -dynamic_max_turns, dynamic_max_turns)
             final_angles[surface_mask] = old_angles[surface_mask] + clamped_diff

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        dynamic_max_speeds = self.speeds.clone()
        if is_raining:
             exposed_mask = ~is_sheltered & ~self.is_underground
             dynamic_max_speeds[exposed_mask] *= 0.6

        # Birds wading in deep vereda water move slower
        wading_mask = flooded_mask & surface_mask
        dynamic_max_speeds[wading_mask] *= 0.7

        self.vel = (self.vel / norms) * dynamic_max_speeds.unsqueeze(1)

        # Update positions
        self.pos += self.vel * self.cfg.dt

        # Periodic boundaries (Surface only)
        self.pos[surface_mask, 0] = self.pos[surface_mask, 0] % self.cfg.width
        self.pos[surface_mask, 1] = self.pos[surface_mask, 1] % self.cfg.height

        self.trajectory_history.append(self.pos.cpu().numpy().copy())
        self.visibility_history.append(~self.is_underground.cpu().numpy().copy())

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

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#03080f; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append(
            '<defs>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="4" fill="#1a5fa8" opacity="0.5"/>'
            '</pattern>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="waterGrad" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#00ffff" stop-opacity="0.85"/>'
            '<stop offset="45%" stop-color="#0088ff" stop-opacity="0.50"/>'
            '<stop offset="85%" stop-color="#0044cc" stop-opacity="0.20"/>'
            '<stop offset="100%" stop-color="#0033aa" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="springGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#80ffff" stop-opacity="0.9"/>'
            '<stop offset="100%" stop-color="#00ffff" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>'
        )

        svg.append(f'<rect width="{w}" height="{h}" fill="#03080f"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Vereda Nodes - vivid seasonal water pools with bubbles
        v_nodes = self.sim.vereda_nodes.cpu().numpy()
        r_vals = ";".join([f"{r:.1f}" for r in self.sim.vereda_radius_history])

        for i, vn in enumerate(v_nodes):
            bgs = i * 0.5

            # Outer ambient glow blob
            svg.append(
                f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="url(#waterGrad)">'
                f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

            # Pulsing ring at puddle edge
            svg.append(
                f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="none" stroke="#00ccff" stroke-width="2.5" stroke-dasharray="6 10">'
                f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="stroke-opacity" values="0.3;1.0;0.3" dur="2.8s" begin="{bgs}s" repeatCount="indefinite"/>'
                f'</circle>'
            )

            # Rising bubble rings (3 bubbles per vereda, staggered)
            for b in range(3):
                boff = b * 0.8
                svg.append(
                    f'<circle cx="{vn[0] + (b-1)*12}" cy="{vn[1]}" r="4" fill="none" stroke="#00ffff" stroke-width="1.5">'
                    f'<animate attributeName="r" values="2;7;2" dur="1.8s" begin="{bgs + boff:.1f}s" repeatCount="indefinite"/>'
                    f'<animate attributeName="opacity" values="0.8;0.0;0.8" dur="1.8s" begin="{bgs + boff:.1f}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )

            # Center spring glow
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" r="18" fill="url(#springGlow)"/>')
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" r="9" fill="#00ffff" filter="url(#glow)">')
            svg.append(f'<animate attributeName="r" values="6;13;6" dur="2.2s" begin="{bgs}s" repeatCount="indefinite"/>')
            svg.append('</circle>')
            svg.append(f'<text font-weight="bold" x="{vn[0]+15}" y="{vn[1]-15}" font-size="15" fill="#88ddff">Seasonal Vereda {i+1}</text>')

        # Cave Nodes - glowing dark entrance
        cnodes = self.sim.cave_nodes.cpu().numpy()
        for i, cn in enumerate(cnodes):
            svg.append(f'<circle cx="{cn[0]}" cy="{cn[1]}" r="{self.cfg.cave_entry_radius + 8}" fill="none" stroke="#00bbff" stroke-width="1.5" stroke-dasharray="3 7">')
            svg.append(f'<animate attributeName="stroke-opacity" values="0.3;0.9;0.3" dur="3.5s" begin="{i*0.7:.1f}s" repeatCount="indefinite"/>')
            svg.append('</circle>')
            svg.append(f'<circle cx="{cn[0]}" cy="{cn[1]}" r="{self.cfg.cave_entry_radius}" fill="#000000" stroke="#00bbff" stroke-width="2"/>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#00ffff" font-weight="bold">ECO-SIM: Seasonal Hydrology</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#88bbcc">Dynamic Veredas Expansion &amp; Contraction</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]

            op_vals = []
            for frame_idx in range(self.cfg.frames):
                 is_visible = self.sim.visibility_history[frame_idx][idx]
                 op_vals.append("1.0" if is_visible else "0.0")
            op_str = ";".join(op_vals)

            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            if idx % 6 == 0:
                 surface_chunks = []
                 current_chunk = []
                 for frame_idx in range(0, self.cfg.frames, 2):
                      p = self.sim.trajectory_history[frame_idx][idx]
                      is_vis = self.sim.visibility_history[frame_idx][idx]
                      if is_vis:
                           current_chunk.append(p)
                      else:
                           if len(current_chunk) > 1:
                                surface_chunks.append(current_chunk)
                           current_chunk = []
                 if len(current_chunk) > 1:
                      surface_chunks.append(current_chunk)

                 for chunk in surface_chunks:
                      d_path = "M " + " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in chunk])
                      svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.40" stroke-width="2.0"/>')

            r_begin = f"{(idx % 14) * 0.18:.2f}s"

            # Glow halo for subset
            if idx % 6 == 0:
                svg.append(f'<circle r="9" fill="{p_color}" opacity="0.25">')
                svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="r" values="6;14;6" dur="2.2s" begin="{r_begin}" repeatCount="indefinite"/>')
                svg.append('</circle>')

            # Entity Node with r-pulse
            svg.append(f'<circle r="4.5" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append(f'<animate attributeName="r" values="3.5;5.8;3.5" dur="2.5s" begin="{r_begin}" repeatCount="indefinite"/>')
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

    print(f"Initializing Seasonal Hydrology simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_15')

if __name__ == "__main__":
    main()
