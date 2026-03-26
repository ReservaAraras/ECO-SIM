# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 11: Karst Obstacles (Limestone Outcrop Collision Avoidance)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Karst Obstacles (Limestone Outcrop Collision Avoidance) emphasizing wet-season resource pulses.
- Indicator species: Macaco-prego (Sapajus libidinosus).
- Pollination lens: floral resource concentration in veredas.
- Human impact lens: carbon stock monitoring incentives.

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
    frames: int = 300 # Prompt requirement: 300 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    territory_radius: float = 70.0
    attack_force: float = 12.0
    breeding_season_start: int = 100
    breeding_season_end: int = 220
    spiral_speed_multiplier: float = 1.3
    obstacle_buffer: float = 30.0 # Distance to start avoiding obstacles

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

        # Attribute: Sex (50% Male, 50% Female)
        self.is_male = torch.rand(cfg.num_particles, device=self.dev) > 0.5

        # Mating display state tracker
        self.is_displaying = torch.zeros(cfg.num_particles, device=self.dev, dtype=torch.bool)
        self.display_timer = torch.zeros(cfg.num_particles, device=self.dev)
        self.display_centers = torch.zeros((cfg.num_particles, 2), device=self.dev)

        # Nectar Nodes
        self.nectar_nodes = torch.tensor([
            [250.0, 300.0], [640.0, 150.0], [1050.0, 450.0]
        ], device=self.dev)

        # NEW: Limestone Outcrops (Obstacles: x, y, radius)
        self.obstacle_nodes = torch.tensor([
            [400.0, 300.0, 60.0], [900.0, 200.0, 80.0], [700.0, 450.0, 50.0], [150.0, 150.0, 45.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.active_defense_history = []
        self.active_display_history = []

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # 1. Base Nectar Node pull
        dist_to_nodes = torch.cdist(self.pos, self.nectar_nodes)
        min_dist_to_node, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.nectar_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist_to_node.unsqueeze(1).clamp(min=1.0)) * 1.0

        # 2. Territoriality
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

        # 3. Mating Displays
        in_season = self.cfg.breeding_season_start <= frame_idx <= self.cfg.breeding_season_end

        if in_season:
             start_display_chance = torch.rand(self.cfg.num_particles, device=self.dev) < self.mating_frequencies
             start_display_mask = start_display_chance & self.is_male & ~self.is_displaying
             if start_display_mask.any():
                  self.is_displaying[start_display_mask] = True
                  self.display_timer[start_display_mask] = 0.0
                  self.display_centers[start_display_mask] = self.pos[start_display_mask].clone()

        display_duration_max = 30.0
        self.display_timer[self.is_displaying] += 1.0
        end_display_mask = self.display_timer >= display_duration_max
        self.is_displaying[end_display_mask] = False

        spiral_vectors = torch.zeros_like(self.vel)
        displaying_idx = self.is_displaying.nonzero(as_tuple=True)[0]

        if self.is_displaying.any():
             displays_pos = self.pos[displaying_idx]
             centers = self.display_centers[displaying_idx]
             timers = self.display_timer[displaying_idx]
             vec_to_center = centers - displays_pos
             dist_to_center = torch.norm(vec_to_center, dim=1).clamp(min=1.0)
             dir_to_center = vec_to_center / dist_to_center.unsqueeze(1)
             tangent_dir = torch.stack([-dir_to_center[:, 1], dir_to_center[:, 0]], dim=1)
             radius_target = (torch.sin(timers / display_duration_max * math.pi) * 40.0).unsqueeze(1)
             radial_correction = (dist_to_center.unsqueeze(1) - radius_target) * -0.5 * dir_to_center
             spin_force = tangent_dir * 12.0
             spiral_vectors[displaying_idx] = spin_force + radial_correction

        # 4. OBSTACLE AVOIDANCE (Limestone outcrops)
        obs_centers = self.obstacle_nodes[:, :2]
        obs_radii = self.obstacle_nodes[:, 2]

        dist_to_obs = torch.cdist(self.pos, obs_centers)  # [num_particles, num_obstacles]
        combined_radii = obs_radii.unsqueeze(0) + self.cfg.obstacle_buffer

        avoid_mask = dist_to_obs < combined_radii

        obstacle_vectors = torch.zeros_like(self.vel)
        if avoid_mask.any():
            diffs = self.pos.unsqueeze(1) - obs_centers.unsqueeze(0) # [N, M, 2]
            dists = dist_to_obs.clamp(min=1.0).unsqueeze(2) # [N, M, 1]
            dirs = diffs / dists

            # Repulsion strength grows the deeper they get into the buffer zone
            forces = 30.0 * (1.0 - (dist_to_obs / combined_radii)).unsqueeze(2)
            forces = forces * avoid_mask.unsqueeze(2).float()

            # For each particle, sum the forces pushed from all obstacles
            obstacle_vectors = (dirs * forces).sum(dim=1)


        # COMBINE FORCES
        raw_new_vel = torch.zeros_like(self.vel)
        normal_mask = ~self.is_displaying

        if normal_mask.any():
             raw_new_vel[normal_mask] = (self.vel[normal_mask] * self.drags[normal_mask] +
                                       karst_attraction[normal_mask] +
                                       defense_vectors[normal_mask] +
                                       obstacle_vectors[normal_mask] + # Add Obstacle repulsion
                                       torch.randn_like(self.vel[normal_mask]) * 0.5)

        # Spiraling males might still be affected slightly by obstacles so they don't fly right through rock
        if self.is_displaying.any():
             raw_new_vel[displaying_idx] = spiral_vectors[displaying_idx] + obstacle_vectors[displaying_idx]

        # Turn Radius (Angle Limiting)
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        final_angles = old_angles.clone()

        if normal_mask.any():
             # When heavily dodging, allow sharp turning
             dynamic_max_turns = self.max_turns[normal_mask].clone()
             is_dodging = avoid_mask[normal_mask].any(dim=1)
             dynamic_max_turns[is_dodging] *= 2.0 # Can turn twice as sharp to avoid rock

             clamped_diff = torch.clamp(angle_diff[normal_mask], -dynamic_max_turns, dynamic_max_turns)
             final_angles[normal_mask] = old_angles[normal_mask] + clamped_diff

        if self.is_displaying.any():
             final_angles[displaying_idx] = new_angles[displaying_idx]

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        dynamic_max_speeds = self.speeds.clone()
        if self.is_displaying.any():
             dynamic_max_speeds[displaying_idx] *= self.cfg.spiral_speed_multiplier

        self.vel = (self.vel / norms) * dynamic_max_speeds.unsqueeze(1)

        # Update positions
        self.pos += self.vel * self.cfg.dt

        # Ensure no birds are strictly inside the obstacle radius (hard ejection)
        new_dist_to_obs = torch.cdist(self.pos, obs_centers)
        hard_eject_mask = new_dist_to_obs < obs_radii.unsqueeze(0)

        if hard_eject_mask.any():
            for p_idx in range(self.cfg.num_particles):
                for o_idx in range(len(obs_radii)):
                    if hard_eject_mask[p_idx, o_idx]:
                        # Push them strictly to edge
                        vec = self.pos[p_idx] - obs_centers[o_idx]
                        dist = torch.norm(vec).clamp(min=0.1)
                        self.pos[p_idx] = obs_centers[o_idx] + (vec / dist) * (obs_radii[o_idx] + 1.0)


        # Periodic boundaries
        self.pos[:, 0] = self.pos[:, 0] % self.cfg.width
        self.pos[:, 1] = self.pos[:, 1] % self.cfg.height

        self.trajectory_history.append(self.pos.cpu().numpy().copy())
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

        svg.append('<defs><pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/></pattern></defs>')
        svg.append('<defs><radialGradient id="rockGrad"><stop offset="60%" stop-color="#4a4a5a"/><stop offset="100%" stop-color="#2c2c36"/></radialGradient></defs>')

        # Static background rect
        svg.append(f'<rect width="{w}" height="{h}" fill="#121212"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Limestone Obstacles
        obs_nodes = self.sim.obstacle_nodes.cpu().numpy()
        for i, obs in enumerate(obs_nodes):
            svg.append(f'<circle cx="{obs[0]}" cy="{obs[1]}" r="{obs[2]}" fill="url(#rockGrad)" stroke="#78788a" stroke-width="2"/>')
            # Render Buffer zone
            svg.append(f'<circle cx="{obs[0]}" cy="{obs[1]}" r="{obs[2] + self.cfg.obstacle_buffer}" fill="none" stroke="#78788a" stroke-width="1" stroke-dasharray="2 4" opacity="0.5"/>')
            svg.append(f'<text font-weight="bold" x="{obs[0]}" y="{obs[1]+5}" font-size="15" fill="#cccccc" text-anchor="middle">Karst Outcrop</text>')


        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Karst Obstacles</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Static Outcrop Collision Avoidance</text>')

        # Nectar Nodes (Territories)
        nnodes = self.sim.nectar_nodes.cpu().numpy()
        for i, nn in enumerate(nnodes):
            svg.append(f'<circle cx="{nn[0]}" cy="{nn[1]}" r="{self.cfg.territory_radius}" fill="#ffe43f" opacity="0.05"/>')
            svg.append(f'<circle cx="{nn[0]}" cy="{nn[1]}" r="6" fill="#ffe43f" opacity="0.5"/>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            is_male = self.sim.is_male[idx].item()

            base_opacity = "1.0" if is_male else "0.3"

            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            c_vals = []
            for frame_idx in range(self.cfg.frames):
                 is_displaying = self.sim.active_display_history[frame_idx][idx]
                 if is_displaying:
                      c_vals.append("#ffffff")
                 else:
                      c_vals.append(p_color)
            c_str = ";".join(c_vals)

            pts_visible = []
            for frame_idx in range(0, self.cfg.frames, 2):
                 p = self.sim.trajectory_history[frame_idx][idx]
                 is_disp = self.sim.active_display_history[frame_idx][idx]
                 pts_visible.append((p, is_disp))

            # Render standard trail
            if idx % 8 == 0 or is_male:
                 d_path = "M " + " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p, _ in pts_visible[::2]])
                 s_width = 2.0 if is_male else 1.0
                 svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.15" stroke-width="{s_width}"/>')

            # Render intense spiral overlay trail
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

            for chunk in spiral_chunks:
                 d_path = "M " + " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in chunk])
                 svg.append(f'<path d="{d_path}" fill="none" stroke="#e91e63" stroke-opacity="0.8" stroke-width="2.5"/>')

            # Entity Node
            rad = "5.5" if is_male else "3.5"
            svg.append(f'<circle r="{rad}" fill="{p_color}" opacity="{base_opacity}">')
            svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')

            if is_male:
                 svg.append(f'<animate attributeName="fill" values=\"{c_str}\" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>')
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

    print(f"Initializing karst obstacles simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_11')

if __name__ == "__main__":
    main()
