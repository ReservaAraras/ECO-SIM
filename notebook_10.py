# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 10: Mating Displays (Spiral Flight Patterns)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Mating Displays (Spiral Flight Patterns) emphasizing rocky outcrop shelters.
- Indicator species: Quati (Nasua nasua).
- Pollination lens: temporal mismatch with migratory pollinators.
- Human impact lens: extreme drought thresholds.

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
    frames: int = 290 # Prompt requirement: 290 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    territory_radius: float = 70.0
    attack_force: float = 12.0
    breeding_season_start: int = 100
    breeding_season_end: int = 220
    spiral_speed_multiplier: float = 1.3

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

        # New Biological Attribute: Sex (50% Male, 50% Female)
        self.is_male = torch.rand(cfg.num_particles, device=self.dev) > 0.5

        # Mating display state tracker
        self.is_displaying = torch.zeros(cfg.num_particles, device=self.dev, dtype=torch.bool)
        self.display_timer = torch.zeros(cfg.num_particles, device=self.dev) # How long they've been doing the spiral
        self.display_centers = torch.zeros((cfg.num_particles, 2), device=self.dev) # Center of their spiral

        # Nectar Nodes (For territoriality baseline)
        self.nectar_nodes = torch.tensor([
            [250.0, 300.0], [640.0, 150.0], [1050.0, 450.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.active_defense_history = []
        self.active_display_history = [] # For rendering spirals

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # 1. Base Nectar Node pull
        dist_to_nodes = torch.cdist(self.pos, self.nectar_nodes)
        min_dist_to_node, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.nectar_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist_to_node.unsqueeze(1).clamp(min=1.0)) * 1.0

        # 2. Territoriality (from Module 09)
        defense_vectors = torch.zeros_like(self.vel)
        active_defense_status = torch.zeros(self.cfg.num_particles, dtype=torch.bool, device=self.dev)
        intruders_mask = ~self.is_territorial

        if self.is_territorial.any() and intruders_mask.any():
            defenders_pos = self.pos[self.is_territorial]
            intruders_pos = self.pos[intruders_mask]
            claiming_mask = min_dist_to_node[self.is_territorial] < self.cfg.territory_radius
            if claiming_mask.any():
                active_defenders_pos = defenders_pos[claiming_mask]
                dist_def_to_int = torch.cdist(active_defenders_pos, intruders_pos)
                min_dist_to_def, closest_def_idx = torch.min(dist_def_to_int, dim=0)
                trespassing_mask = min_dist_to_def < self.cfg.territory_radius

                if trespassing_mask.any():
                    trespassing_intruders = intruders_pos[trespassing_mask]
                    angry_defenders = active_defenders_pos[closest_def_idx[trespassing_mask]]
                    push_dir = trespassing_intruders - angry_defenders
                    push_dist = min_dist_to_def[trespassing_mask].unsqueeze(1).clamp(min=1.0)
                    force_magnitude = self.cfg.attack_force * (1.0 - (push_dist / self.cfg.territory_radius))
                    push_vectors = (push_dir / push_dist) * force_magnitude
                    defense_vectors[intruders_mask.nonzero(as_tuple=True)[0][trespassing_mask]] += push_vectors

                    global_def_idx = self.is_territorial.nonzero(as_tuple=True)[0][claiming_mask][closest_def_idx[trespassing_mask]]
                    active_defense_status[global_def_idx] = True
                    defense_vectors[global_def_idx] += (-push_dir / push_dist) * (self.cfg.attack_force * 0.5)

        # 3. MATING DISPLAYS / SPIRAL FLIGHT (Module 10)
        in_season = self.cfg.breeding_season_start <= frame_idx <= self.cfg.breeding_season_end

        # Chance to start displaying (only males)
        if in_season:
             start_display_chance = torch.rand(self.cfg.num_particles, device=self.dev) < self.mating_frequencies
             start_display_mask = start_display_chance & self.is_male & ~self.is_displaying

             if start_display_mask.any():
                  self.is_displaying[start_display_mask] = True
                  self.display_timer[start_display_mask] = 0.0
                  # Set spiral center abruptly where they are
                  self.display_centers[start_display_mask] = self.pos[start_display_mask].clone()

        # Update display timers and check bounds
        display_duration_max = 30.0 # frames
        self.display_timer[self.is_displaying] += 1.0

        end_display_mask = self.display_timer >= display_duration_max
        self.is_displaying[end_display_mask] = False

        # Calculate Spiral Vectors
        spiral_vectors = torch.zeros_like(self.vel)
        displaying_idx = self.is_displaying.nonzero(as_tuple=True)[0]

        if self.is_displaying.any():
             displays_pos = self.pos[displaying_idx]
             centers = self.display_centers[displaying_idx]
             timers = self.display_timer[displaying_idx]

             # Vector to center
             vec_to_center = centers - displays_pos
             dist_to_center = torch.norm(vec_to_center, dim=1).clamp(min=1.0)
             dir_to_center = vec_to_center / dist_to_center.unsqueeze(1)

             # Tangent vector (perpendicular)
             tangent_dir = torch.stack([-dir_to_center[:, 1], dir_to_center[:, 0]], dim=1)

             # Spiral math: Expand outward then pull inward, always rotating
             # They spiral outwards for first half, then spiral inwards
             radius_target = (torch.sin(timers / display_duration_max * math.pi) * 40.0).unsqueeze(1)

             # Push/pull to maintain the target radius while spinning rapidly
             radial_correction = (dist_to_center.unsqueeze(1) - radius_target) * -0.5 * dir_to_center
             spin_force = tangent_dir * 12.0 # High tangent speed for distinct tight loops

             spiral_vectors[displaying_idx] = spin_force + radial_correction

        # Combine physical forces (Males displaying drop all other concerns)
        # We split the physics:
        # non-displaying = normal physics
        # displaying = pure spiral physics

        raw_new_vel = torch.zeros_like(self.vel)

        # Normal
        normal_mask = ~self.is_displaying
        if normal_mask.any():
             raw_new_vel[normal_mask] = (self.vel[normal_mask] * self.drags[normal_mask] +
                                       karst_attraction[normal_mask] +
                                       defense_vectors[normal_mask] +
                                       torch.randn_like(self.vel[normal_mask]) * 0.5)

        # Spiraling
        if self.is_displaying.any():
             raw_new_vel[displaying_idx] = spiral_vectors[displaying_idx]

        # Turn Radius (Angle Limiting) - Only apply to normal birds. Spiraling birds snap exactly to spiral vector
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        final_angles = old_angles.clone()

        if normal_mask.any():
             clamped_diff = torch.clamp(angle_diff[normal_mask], -self.max_turns[normal_mask], self.max_turns[normal_mask])
             final_angles[normal_mask] = old_angles[normal_mask] + clamped_diff

        if self.is_displaying.any():
             # Mating males have zero turn restriction during their erratic spiral flight
             final_angles[displaying_idx] = new_angles[displaying_idx]

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        # Normalize to max speeds
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        dynamic_max_speeds = self.speeds.clone()
        if self.is_displaying.any():
             # Spirals are fast and flashy
             dynamic_max_speeds[displaying_idx] *= self.cfg.spiral_speed_multiplier

        self.vel = (self.vel / norms) * dynamic_max_speeds.unsqueeze(1)

        # Update positions
        self.pos += self.vel * self.cfg.dt

        # Periodic boundaries
        self.pos[:, 0] = self.pos[:, 0] % self.cfg.width
        self.pos[:, 1] = self.pos[:, 1] % self.cfg.height

        self.trajectory_history.append(self.pos.cpu().numpy().copy())
        self.active_defense_history.append(active_defense_status.cpu().numpy().copy())
        self.active_display_history.append(self.is_displaying.cpu().numpy().copy())

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

        svg.append(
            '<defs>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="4" fill="#3d9df7" opacity="0.45"/>'
            '</pattern>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="maleGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#e91e63" stop-opacity="0.85"/>'
            '<stop offset="100%" stop-color="#e91e63" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="displayGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#ffffff" stop-opacity="0.9"/>'
            '<stop offset="60%" stop-color="#e91e63" stop-opacity="0.4"/>'
            '<stop offset="100%" stop-color="#e91e63" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>'
        )

        # Dark background
        svg.append(f'<rect width="{w}" height="{h}" fill="#0c0616"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Rendering Season Indicator Bar - glowing
        bar_w = 400
        start_ratio = self.cfg.breeding_season_start / self.cfg.frames
        end_ratio = self.cfg.breeding_season_end / self.cfg.frames

        svg.append(f'<rect x="{w/2 - bar_w/2}" y="20" width="{bar_w}" height="10" fill="#2a1040" rx="5"/>')
        svg.append(
            f'<rect x="{w/2 - bar_w/2 + start_ratio*bar_w}" y="18" '
            f'width="{(end_ratio-start_ratio)*bar_w}" height="14" fill="#e91e63" rx="5">'
            f'<animate attributeName="opacity" values="0.7;1.0;0.7" dur="1.5s" repeatCount="indefinite"/>'
            f'</rect>'
        )
        svg.append(f'<text font-weight="bold" x="{w/2}" y="45" font-size="15" fill="#cc88aa" text-anchor="middle">Breeding Season Timeline</text>')

        # Animated cursor over timeline - with glow
        cursor_x_vals = ";".join([f"{w/2 - bar_w/2 + (f/self.cfg.frames)*bar_w}" for f in range(self.cfg.frames)])
        svg.append(
            f'<circle r="9" cy="25" fill="#ff4488" opacity="0.4">'
            f'<animate attributeName="cx" values="{cursor_x_vals}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="r" values="6;13;6" dur="1.2s" repeatCount="indefinite"/>'
            f'</circle>'
        )
        svg.append(f'<circle r="5" cy="25" fill="#ffffff"><animate attributeName="cx" values="{cursor_x_vals}" dur="{dur}s" repeatCount="indefinite"/></circle>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Mating Displays</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Spiral Flight Patterns &amp; Sexual Dimorphism</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            is_male = self.sim.is_male[idx].item()

            # Reduce female opacity globally so displaying males pop out massively
            base_opacity = "1.0" if is_male else "0.3"

            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Display coloring (Males turn intensely vibrant/white when spiraling)
            c_vals = []
            for frame_idx in range(self.cfg.frames):
                 is_displaying = self.sim.active_display_history[frame_idx][idx]
                 if is_displaying:
                      c_vals.append("#ffffff")
                 else:
                      c_vals.append(p_color)
            c_str = ";".join(c_vals)

            # Highlight specific path strokes when spiraling
            # Instead of one huge path, we just draw the trail heavily based on whether they were displaying
            pts_visible = []
            for frame_idx in range(0, self.cfg.frames, 2):
                 p = self.sim.trajectory_history[frame_idx][idx]
                 is_disp = self.sim.active_display_history[frame_idx][idx]
                 pts_visible.append((p, is_disp))

            # Render standard trail (brighter)
            if idx % 8 == 0 or is_male:
                 d_path = "M " + " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p, _ in pts_visible[::2]])
                 s_width = 2.0 if is_male else 1.0
                 svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.28" stroke-width="{s_width}"/>')

            # Render intense spiral overlay trails - vivid rainbow colour cycle
            spiral_chunks = []
            current_chunk = []
            for p, is_disp in pts_visible:
                 if is_disp:
                      current_chunk.append(p)
                 else:
                      if len(current_chunk) > 1:
                           spiral_chunks.append(current_chunk)
                      current_chunk = []
            if len(current_chunk) > 1:
                 spiral_chunks.append(current_chunk)

            spiral_colors = ["#e91e63", "#ff9800", "#ffffff", "#e91e63"]
            for ci, chunk in enumerate(spiral_chunks):
                 d_path = "M " + " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in chunk])
                 sc = spiral_colors[ci % len(spiral_colors)]
                 svg.append(f'<path d="{d_path}" fill="none" stroke="{sc}" stroke-opacity="0.95" stroke-width="3.5" stroke-linecap="round"/>')

            # Glow halo for displaying males
            r_begin = f"{(idx % 14) * 0.18:.2f}s"
            if is_male:
                disp_ops = ";".join([
                    "0.7" if self.sim.active_display_history[fi][idx] else "0.18"
                    for fi in range(self.cfg.frames)
                ])
                svg.append(f'<circle r="16" fill="url(#displayGlow)" opacity="0.18">')
                svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="opacity" values="{disp_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
                svg.append(f'<animate attributeName="r" values="12;22;12" dur="0.8s" begin="{r_begin}" repeatCount="indefinite"/>')
                svg.append('</circle>')

            # Entity Node - males pulse vigorously
            rad = "5.5" if is_male else "3.5"
            r_pulse = "4.5;8.0;4.5" if is_male else "3.0;4.5;3.0"
            r_dur_v  = "0.8s" if is_male else "2.5s"
            svg.append(f'<circle r="{rad}" fill="{p_color}" opacity="{base_opacity}">')
            svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="r" values="{r_pulse}" dur="{r_dur_v}" begin="{r_begin}" repeatCount="indefinite"/>')
            if is_male:
                 svg.append(f'<animate attributeName="fill" values="{c_str}" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>')
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

    print(f"Initializing mating displays simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_10')

if __name__ == "__main__":
    main()
