# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 12: Cave Entry/Exit (Sinkhole Teleportation)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Cave Entry/Exit (Sinkhole Teleportation) emphasizing nutrient limitation in soils.
- Indicator species: Jiboia (Boa constrictor).
- Pollination lens: pollinator dilution across open fields.
- Human impact lens: apiary placement competition.

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
    frames: int = 310 # Prompt requirement: 310 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    territory_radius: float = 70.0
    attack_force: float = 12.0
    breeding_season_start: int = 100
    breeding_season_end: int = 220
    spiral_speed_multiplier: float = 1.3
    obstacle_buffer: float = 30.0
    cave_entry_radius: float = 30.0 # Distance to trigger "falling" into cave
    cave_time_min: int = 15 # Frames spent "underground"
    cave_time_max: int = 40

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

        # Mating states
        self.is_displaying = torch.zeros(cfg.num_particles, device=self.dev, dtype=torch.bool)
        self.display_timer = torch.zeros(cfg.num_particles, device=self.dev)
        self.display_centers = torch.zeros((cfg.num_particles, 2), device=self.dev)

        # Caving states
        self.is_underground = torch.zeros(cfg.num_particles, device=self.dev, dtype=torch.bool)
        self.underground_timer = torch.zeros(cfg.num_particles, device=self.dev)
        self.underground_duration = torch.zeros(cfg.num_particles, device=self.dev)

        # Environment
        self.nectar_nodes = torch.tensor([
            [250.0, 300.0], [640.0, 150.0]
        ], device=self.dev)

        self.obstacle_nodes = torch.tensor([
            [400.0, 300.0, 60.0], [900.0, 200.0, 80.0]
        ], device=self.dev)

        # NEW: Sinkhole Caves (Connected network)
        # Birds entering one cave will exit from another
        self.cave_nodes = torch.tensor([
            [150.0, 450.0], [700.0, 500.0], [1050.0, 150.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.visibility_history = [] # Toggles off when underground to cut SMIL trailing correctly

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # 1. Base Nectar Node pull
        dist_to_nodes = torch.cdist(self.pos, self.nectar_nodes)
        min_dist_to_node, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.nectar_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist_to_node.unsqueeze(1).clamp(min=1.0)) * 1.0

        # NEW: Base Cave Pull (Draw them towards the sinkholes!)
        dist_to_caves = torch.cdist(self.pos, self.cave_nodes)
        min_dist_cave, closest_caves = torch.min(dist_to_caves, dim=1)
        cave_pull_vectors = self.cave_nodes[closest_caves] - self.pos
        cave_attraction = (cave_pull_vectors / min_dist_cave.unsqueeze(1).clamp(min=1.0)) * 2.0


        # 2. Territoriality
        defense_vectors = torch.zeros_like(self.vel)
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
                    defense_vectors[global_def_idx] += (-push_dir / push_dist) * (self.cfg.attack_force * 0.5)

        # 3. Spirals
        in_season = self.cfg.breeding_season_start <= frame_idx <= self.cfg.breeding_season_end
        if in_season:
             start_display_chance = torch.rand(self.cfg.num_particles, device=self.dev) < self.mating_frequencies
             start_display_mask = start_display_chance & self.is_male & ~self.is_displaying & ~self.is_underground
             if start_display_mask.any():
                  self.is_displaying[start_display_mask] = True
                  self.display_timer[start_display_mask] = 0.0
                  self.display_centers[start_display_mask] = self.pos[start_display_mask].clone()

        self.display_timer[self.is_displaying] += 1.0
        end_display_mask = self.display_timer >= 30.0
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
             radius_target = (torch.sin(timers / 30.0 * math.pi) * 40.0).unsqueeze(1)
             radial_correction = (dist_to_center.unsqueeze(1) - radius_target) * -0.5 * dir_to_center
             spin_force = tangent_dir * 12.0
             spiral_vectors[displaying_idx] = spin_force + radial_correction

        # 4. Obstacles
        obs_centers = self.obstacle_nodes[:, :2]
        obs_radii = self.obstacle_nodes[:, 2]
        dist_to_obs = torch.cdist(self.pos, obs_centers)
        combined_radii = obs_radii.unsqueeze(0) + self.cfg.obstacle_buffer
        avoid_mask = dist_to_obs < combined_radii
        obstacle_vectors = torch.zeros_like(self.vel)
        if avoid_mask.any():
            diffs = self.pos.unsqueeze(1) - obs_centers.unsqueeze(0)
            dists = dist_to_obs.clamp(min=1.0).unsqueeze(2)
            dirs = diffs / dists
            forces = 30.0 * (1.0 - (dist_to_obs / combined_radii)).unsqueeze(2)
            forces = forces * avoid_mask.unsqueeze(2).float()
            obstacle_vectors = (dirs * forces).sum(dim=1)


        # 5. CAVE ENTRY/EXIT (Teleportation Matrix)
        # Check if surface birds fell into a cave
        falling_mask = (min_dist_cave < self.cfg.cave_entry_radius) & (~self.is_underground)
        if falling_mask.any():
             self.is_underground[falling_mask] = True
             # Assign a random duration to spend underground
             durations = torch.randint(self.cfg.cave_time_min, self.cfg.cave_time_max, (falling_mask.sum().item(),), device=self.dev).float()
             self.underground_duration[falling_mask] = durations
             self.underground_timer[falling_mask] = 0.0

             # Pre-select an exit cave now! (Different from the one they entered)
             entering_cave_idx = closest_caves[falling_mask]

             exit_cave_idx = torch.zeros_like(entering_cave_idx)
             for i, ent_idx in enumerate(entering_cave_idx):
                  # Pick a random cave idx from available caves that is NOT ent_idx
                  available = [c for c in range(len(self.cave_nodes)) if c != ent_idx.item()]
                  exit_cave_idx[i] = random.choice(available)

             # Instantly teleport their mathematical coordinates into the exit cave void so they are ready
             # But they remain marked 'underground' so they don't render or interact
             exit_caves = self.cave_nodes[exit_cave_idx]
             self.pos[falling_mask] = exit_caves

        # Advance underground timers
        self.underground_timer[self.is_underground] += 1.0

        # Emerging birds
        emerging_mask = self.is_underground & (self.underground_timer >= self.underground_duration)
        if emerging_mask.any():
             self.is_underground[emerging_mask] = False

             # Give them an explosive velocity vector straight OUT of the cave to clear the radius
             # Add tiny randomized angle
             explosion_angles = torch.rand(emerging_mask.sum(), device=self.dev) * 2 * math.pi
             exp_vel_x = torch.cos(explosion_angles) * 15.0 # Max speed exit
             exp_vel_y = torch.sin(explosion_angles) * 15.0

             self.vel[emerging_mask, 0] = exp_vel_x
             self.vel[emerging_mask, 1] = exp_vel_y


        # COMBINE FORCES
        raw_new_vel = torch.zeros_like(self.vel)

        # Only birds NOT underground process regular physics
        surface_mask = ~self.is_underground
        normal_mask = surface_mask & ~self.is_displaying

        if normal_mask.any():
             raw_new_vel[normal_mask] = (self.vel[normal_mask] * self.drags[normal_mask] +
                                       karst_attraction[normal_mask] +
                                       cave_attraction[normal_mask] +
                                       defense_vectors[normal_mask] +
                                       obstacle_vectors[normal_mask] +
                                       torch.randn_like(self.vel[normal_mask]) * 0.5)

        # Spiraling males still active on surface
        displaying_surface_idx = (self.is_displaying & surface_mask).nonzero(as_tuple=True)[0]
        if displaying_surface_idx.numel() > 0:
             raw_new_vel[displaying_surface_idx] = spiral_vectors[displaying_surface_idx] + obstacle_vectors[displaying_surface_idx]

        # Underground birds have Zero velocity calculated (they just wait)
        raw_new_vel[self.is_underground] *= 0.0

        # Turn Radius (Angle Limiting)
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        final_angles = old_angles.clone()

        if normal_mask.any():
             dynamic_max_turns = self.max_turns[normal_mask].clone()
             is_dodging = avoid_mask[normal_mask].any(dim=1)
             dynamic_max_turns[is_dodging] *= 2.0
             clamped_diff = torch.clamp(angle_diff[normal_mask], -dynamic_max_turns, dynamic_max_turns)
             final_angles[normal_mask] = old_angles[normal_mask] + clamped_diff

        if displaying_surface_idx.numel() > 0:
             final_angles[displaying_surface_idx] = new_angles[displaying_surface_idx]

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        dynamic_max_speeds = self.speeds.clone()
        if displaying_surface_idx.numel() > 0:
             dynamic_max_speeds[displaying_surface_idx] *= self.cfg.spiral_speed_multiplier

        self.vel = (self.vel / norms) * dynamic_max_speeds.unsqueeze(1)

        # Ensure emerging birds don't get normalized down on this specific frame
        if emerging_mask.any():
            self.vel[emerging_mask, 0] = exp_vel_x
            self.vel[emerging_mask, 1] = exp_vel_y

        # Update positions
        self.pos += self.vel * self.cfg.dt

        new_dist_to_obs = torch.cdist(self.pos, obs_centers)
        hard_eject_mask = new_dist_to_obs < obs_radii.unsqueeze(0)
        hard_eject_mask &= surface_mask.unsqueeze(1) # Only eject surface birds

        if hard_eject_mask.any():
            for p_idx in range(self.cfg.num_particles):
                for o_idx in range(len(obs_radii)):
                    if hard_eject_mask[p_idx, o_idx]:
                        vec = self.pos[p_idx] - obs_centers[o_idx]
                        dist = torch.norm(vec).clamp(min=0.1)
                        self.pos[p_idx] = obs_centers[o_idx] + (vec / dist) * (obs_radii[o_idx] + 1.0)

        # Periodic boundaries (Surface only)
        self.pos[surface_mask, 0] = self.pos[surface_mask, 0] % self.cfg.width
        self.pos[surface_mask, 1] = self.pos[surface_mask, 1] % self.cfg.height

        self.trajectory_history.append(self.pos.cpu().numpy().copy())

        # Track who is visible
        # (NOT underground ensures they just blip out of existence until emerging)
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

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#121212; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append('<defs><pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/></pattern></defs>')
        svg.append('<defs><radialGradient id="rockGrad"><stop offset="60%" stop-color="#4a4a5a"/><stop offset="100%" stop-color="#2c2c36"/></radialGradient></defs>')

        # True Black Abyss gradient for Caves
        svg.append('<defs><radialGradient id="caveGrad"><stop offset="0%" stop-color="#000000"/><stop offset="80%" stop-color="#05070a"/><stop offset="100%" stop-color="#00bbff" stop-opacity="0.3"/></radialGradient></defs>')

        # Static background rect
        svg.append(f'<rect width="{w}" height="{h}" fill="#121212"/>')
        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # Sinkhole Caves (Draw UNDER everything else)
        cnodes = self.sim.cave_nodes.cpu().numpy()
        for i, cn in enumerate(cnodes):
            # Outer mysterious aura
            svg.append(f'<circle cx="{cn[0]}" cy="{cn[1]}" r="80" fill="url(#caveGrad)"/>')
            # The literal dark hole
            svg.append(f'<circle cx="{cn[0]}" cy="{cn[1]}" r="{self.cfg.cave_entry_radius}" fill="#000000" stroke="#00bbff" stroke-width="2"/>')
            svg.append(f'<text font-weight="bold" x="{cn[0]}" y="{cn[1]-40}" font-size="15" fill="#cccccc" text-anchor="middle">Sinkhole {i+1}</text>')

        # Limestone Obstacles
        obs_nodes = self.sim.obstacle_nodes.cpu().numpy()
        for i, obs in enumerate(obs_nodes):
            svg.append(f'<circle cx="{obs[0]}" cy="{obs[1]}" r="{obs[2]}" fill="url(#rockGrad)" stroke="#78788a" stroke-width="2"/>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Cave Network Dynamics</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Sinkhole Teleportation Events</text>')

        # Nectar Nodes
        nnodes = self.sim.nectar_nodes.cpu().numpy()
        for i, nn in enumerate(nnodes):
            svg.append(f'<circle cx="{nn[0]}" cy="{nn[1]}" r="6" fill="#ffe43f" opacity="0.3"/>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            is_male = self.sim.is_male[idx].item()

            # Animate OPACITY to perfectly handle cave entry!
            # If visible: Normal map. If underground: Force opacity to 0.0 rendering them invisible.
            op_vals = []
            for frame_idx in range(self.cfg.frames):
                 is_visible = self.sim.visibility_history[frame_idx][idx]
                 if is_visible:
                      op_vals.append("1.0" if is_male else "0.3")
                 else:
                      op_vals.append("0.0")
            op_str = ";".join(op_vals)

            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Trail rendering (Cut paths when they teleport to avoid massive straight lines across the screen!)
            # We break the path into continuous "surface" chunks.
            if idx % 8 == 0 or is_male:
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

                 s_width = 2.0 if is_male else 1.0
                 for chunk in surface_chunks:
                      d_path = "M " + " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in chunk])
                      svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.15" stroke-width="{s_width}"/>')


            # Entity Node
            rad = "5.5" if is_male else "3.5"
            svg.append(f'<circle r="{rad}" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')
            # Toggle opacity sharply when entering/exiting caves
            svg.append(f'<animate attributeName="opacity" values=\"{op_str}\" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
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

    print(f"Initializing cave network simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_12')

if __name__ == "__main__":
    main()
