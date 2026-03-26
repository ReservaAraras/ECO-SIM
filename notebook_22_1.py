# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 22: Mortality (Old Age & Starvation Physics)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Mortality (Old Age & Starvation Physics) emphasizing agroforestry edge subsidies.
- Indicator species: Jacu (Penelope obscura).
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


from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from dataclasses import dataclass, field
from typing import List, Dict

# ===================================================================================================
# 1. SCIENTIFIC TAXONOMY & ECOLOGICAL GUILDS (Recanto das Araras ENDEMICS)
# ===================================================================================================

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1, "is_territorial": False, "mating_frequency": 0.05, "lifespan_base": 300},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15, "is_territorial": False, "mating_frequency": 0.02, "lifespan_base": 250},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4, "is_territorial": True, "mating_frequency": 0.08, "lifespan_base": 180},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#f44336", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2, "is_territorial": False, "mating_frequency": 0.01, "lifespan_base": 350},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 410 # Prompt requirement: 410 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    initial_particles: int = 250
    max_particles: int = 350

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

    vereda_min_radius: float = 20.0
    vereda_max_radius: float = 90.0
    vereda_cycle_frames: float = 200.0
    vereda_base_attraction: float = 3.0

    fruiting_cycle_frames: float = 150.0
    fruiting_radius: float = 50.0
    fruiting_base_attraction: float = 4.0

    seed_carry_duration: float = 60.0
    seed_pickup_chance: float = 0.05
    seed_germination_delay: float = 70.0

    sand_zone_x: float = 640.0
    sand_speed_modifier: float = 0.6

    fire_start_frame: int = 80
    fire_spread_prob: float = 0.08
    fire_flee_radius: float = 150.0
    fire_flee_force: float = 12.0
    max_fire_nodes: int = 150

    mating_energy_threshold: float = 85.0
    mating_radius: float = 30.0
    energy_decay: float = 0.15      # Slightly increased so starvation happens
    energy_gain_fruiting: float = 6.0
    energy_gain_vereda: float = 2.5

    # NEW: Mortality Dynamics
    lifespan_variance: float = 50.0 # Frames of variance applied to base lifespan

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
        self.pos = torch.rand((cfg.max_particles, 2), device=self.dev) * torch.tensor([cfg.width, cfg.height], device=self.dev)
        self.vel = (torch.rand((cfg.max_particles, 2), device=self.dev) - 0.5) * 10.0

        # Assign guilds based on weighting
        guilds = list(BIODIVERSITY_DB.keys())
        weights = [BIODIVERSITY_DB[g]["weight"] for g in guilds]
        indices = np.random.choice(len(guilds), size=cfg.max_particles, p=weights)

        self.species_id = torch.tensor(indices, device=self.dev)
        self.speeds = torch.tensor([BIODIVERSITY_DB[guilds[i]]["speed"] for i in indices], device=self.dev)
        self.drags = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"] for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"] for i in indices], device=self.dev)
        self.is_territorial = torch.tensor([BIODIVERSITY_DB[guilds[i]]["is_territorial"] for i in indices], device=self.dev, dtype=torch.bool)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        self.is_frugivore = torch.tensor([BIODIVERSITY_DB[guilds[i]]["diet"] in ["Frugivore", "Omnivore"] for i in indices], device=self.dev, dtype=torch.bool)
        self.is_male = torch.rand(cfg.max_particles, device=self.dev) > 0.5

        # State Arrays
        self.is_active = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.is_active[:cfg.initial_particles] = True

        self.energy = torch.ones(cfg.max_particles, device=self.dev) * 60.0  # Start with decent energy

        # NEW: Age and Lifespan
        self.age = torch.zeros(cfg.max_particles, device=self.dev)
        # Random initial ages so the initial population doesn't all die at once
        self.age[:cfg.initial_particles] = torch.rand(cfg.initial_particles, device=self.dev) * 200.0

        base_lifespans = torch.tensor([BIODIVERSITY_DB[guilds[i]]["lifespan_base"] for i in indices], device=self.dev, dtype=torch.float32)
        # Add random scatter to exact lifespans globally per tensor element
        self.lifespan = base_lifespans + (torch.randn(cfg.max_particles, device=self.dev) * cfg.lifespan_variance)

        self.is_underground = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)

        # Environment
        self.nectar_nodes = torch.tensor([[250.0, 300.0]], device=self.dev)
        self.obstacle_nodes = torch.tensor([[400.0, 300.0, 60.0]], device=self.dev)
        self.cave_nodes = torch.tensor([[150.0, 450.0]], device=self.dev)
        self.cover_nodes = torch.tensor([[1050.0, 200.0]], device=self.dev)
        self.vereda_nodes = torch.tensor([[600.0, 350.0], [900.0, 450.0]], device=self.dev)
        self.fruiting_nodes = torch.tensor([[300.0, 150.0], [850.0, 250.0]], device=self.dev)

        self.has_seed = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.seed_timers = torch.zeros(cfg.max_particles, device=self.dev)
        self.dropped_seeds = []
        self.fire_nodes = torch.zeros((0, 2), device=self.dev)

        # History
        self.trajectory_history = []
        self.visibility_history = []
        self.active_history = []
        self.carrying_seed_history = []
        self.fire_nodes_history = []
        self.vereda_radius_history = []
        self.fruiting_intensity_history = []

        self.birth_events = []
        self.death_events = [] # NEW: for visualizing mortality

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        active_mask = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground
        is_raining = self.cfg.rain_start <= frame_idx <= self.cfg.rain_end

        # -------------------------------------------------------------------------
        # NEW: MORTALITY & AGING LOGIC
        # -------------------------------------------------------------------------
        self.age[active_mask] += 1.0
        self.energy[active_mask] -= self.cfg.energy_decay

        # Determine death: old age OR starvation (energy <= 0.0)
        starved_mask = active_mask & (self.energy <= 0.0)
        old_age_mask = active_mask & (self.age >= self.lifespan)
        dead_mask = starved_mask | old_age_mask

        if dead_mask.any():
            dead_idx = dead_mask.nonzero().squeeze(1)
            for idx in dead_idx:
                self.is_active[idx] = False
                self.has_seed[idx] = False

                # Log death mapping
                reason = "old_age" if old_age_mask[idx] else "starvation"
                self.death_events.append({
                    "pos": self.pos[idx].cpu().numpy().copy(),
                    "frame": frame_idx,
                    "color": self.colors[idx],
                    "reason": reason
                })

        # Re-fetch active_mask just in case some particles died
        active_mask = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground

        if not active_mask.any():
            # If everything dies, we still record frames safely without crashing
            self._log_history()
            return

        # Base environmental pulls
        dist_to_nodes = torch.cdist(self.pos, self.nectar_nodes)
        min_dist_to_node, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.nectar_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist_to_node.unsqueeze(1).clamp(min=1.0)) * 1.0

        dist_to_caves = torch.cdist(self.pos, self.cave_nodes)
        min_dist_cave, closest_caves = torch.min(dist_to_caves, dim=1)
        cave_pull_vectors = self.cave_nodes[closest_caves] - self.pos
        cave_attraction = (cave_pull_vectors / min_dist_cave.unsqueeze(1).clamp(min=1.0)) * 2.0

        dist_to_covers = torch.cdist(self.pos, self.cover_nodes)
        min_dist_cover, closest_covers = torch.min(dist_to_covers, dim=1)
        is_sheltered = (min_dist_cover < self.cfg.cover_radius) & ~self.is_underground

        # Vereda Hydrology
        cycle_progress = (frame_idx % self.cfg.vereda_cycle_frames) / self.cfg.vereda_cycle_frames
        water_level = (math.sin(cycle_progress * 2 * math.pi - math.pi/2) + 1.0) / 2.0
        current_v_radius = self.cfg.vereda_min_radius + (self.cfg.vereda_max_radius - self.cfg.vereda_min_radius) * water_level
        self.vereda_radius_history.append(current_v_radius)

        dist_to_veredas = torch.cdist(self.pos, self.vereda_nodes)
        min_dist_vereda, closest_veredas = torch.min(dist_to_veredas, dim=1)
        dynamic_attraction = self.cfg.vereda_base_attraction * water_level
        vereda_pull_vectors = self.vereda_nodes[closest_veredas] - self.pos
        vereda_attraction = (vereda_pull_vectors / min_dist_vereda.unsqueeze(1).clamp(min=1.0)) * dynamic_attraction

        flooded_mask = min_dist_vereda < current_v_radius
        vereda_attraction[flooded_mask] *= 0.1

        # Gain energy in oasis
        drinking_mask = flooded_mask & surface_mask
        if drinking_mask.any():
            self.energy[drinking_mask] += self.cfg.energy_gain_vereda

        # Fruiting Phenology
        fruit_cycle = (frame_idx % self.cfg.fruiting_cycle_frames) / self.cfg.fruiting_cycle_frames
        raw_fruit_val = math.sin(fruit_cycle * 2 * math.pi)
        fruiting_intensity = max(0.0, raw_fruit_val)
        self.fruiting_intensity_history.append(fruiting_intensity)

        dist_to_fruit = torch.cdist(self.pos, self.fruiting_nodes)
        min_dist_fruit, closest_fruits = torch.min(dist_to_fruit, dim=1)

        fruiting_attraction = torch.zeros_like(self.vel)
        feeding_mask = torch.zeros(self.cfg.max_particles, device=self.dev, dtype=torch.bool)

        if fruiting_intensity > 0.0:
            f_pull_vectors = self.fruiting_nodes[closest_fruits] - self.pos
            smell_mask = (min_dist_fruit < 200.0) & self.is_frugivore & surface_mask
            if smell_mask.any():
                f_attract_magnitude = (self.cfg.fruiting_base_attraction * fruiting_intensity) / min_dist_fruit[smell_mask].unsqueeze(1).clamp(min=1.0)
                fruiting_attraction[smell_mask] = f_pull_vectors[smell_mask] * f_attract_magnitude

                foraging_mask = smell_mask & (min_dist_fruit < self.cfg.fruiting_radius)
                feeding_mask = foraging_mask & (fruiting_intensity > 0.1)
                if foraging_mask.any():
                    fruiting_attraction[foraging_mask] += torch.randn_like(fruiting_attraction[foraging_mask]) * 2.0
                    self.energy[foraging_mask] += self.cfg.energy_gain_fruiting * fruiting_intensity

        self.energy.clamp_(0.0, 100.0)

        # Seed Dispersal Logic
        can_grab_mask = feeding_mask & ~self.has_seed
        if can_grab_mask.any():
            grab_chance = torch.rand(self.cfg.max_particles, device=self.dev) < self.cfg.seed_pickup_chance
            successful_grabs = can_grab_mask & grab_chance
            if successful_grabs.any():
                self.has_seed[successful_grabs] = True
                jitter = torch.randn(self.cfg.max_particles, device=self.dev) * 5.0
                self.seed_timers[successful_grabs] = jitter[successful_grabs]

        self.seed_timers[self.has_seed] += 1.0
        drop_mask = self.has_seed & (self.seed_timers >= self.cfg.seed_carry_duration) & surface_mask
        if drop_mask.any():
            self.has_seed[drop_mask] = False
            for p in self.pos[drop_mask]:
                self.dropped_seeds.append({"pos": p.cpu().numpy().copy(), "frame": frame_idx})

        # Fire Logic
        fire_repulsion = torch.zeros_like(self.vel)
        flee_mask = torch.zeros(self.cfg.max_particles, device=self.dev, dtype=torch.bool)

        if frame_idx >= self.cfg.fire_start_frame:
            if len(self.fire_nodes) == 0:
                self.fire_nodes = torch.tensor([[1000.0, 300.0]], device=self.dev)
            else:
                if frame_idx % 2 == 0 and len(self.fire_nodes) < self.cfg.max_fire_nodes:
                    spread_mask = torch.rand(len(self.fire_nodes), device=self.dev) < self.cfg.fire_spread_prob
                    if spread_mask.any():
                        new_sparks = self.fire_nodes[spread_mask] + (torch.randn((spread_mask.sum().item(), 2), device=self.dev) * 30.0)
                        new_sparks[:, 0] = new_sparks[:, 0].clamp(0, self.cfg.width)
                        new_sparks[:, 1] = new_sparks[:, 1].clamp(0, self.cfg.height)
                        self.fire_nodes = torch.cat([self.fire_nodes, new_sparks], dim=0)

            if len(self.fire_nodes) > 0:
                dist_to_fires = torch.cdist(self.pos, self.fire_nodes)
                min_dist_fire, closest_fire_idx = torch.min(dist_to_fires, dim=1)

                flee_mask = (min_dist_fire < self.cfg.fire_flee_radius) & surface_mask
                if flee_mask.any():
                    flee_vectors = self.pos[flee_mask] - self.fire_nodes[closest_fire_idx[flee_mask]]
                    flee_force = (flee_vectors / min_dist_fire[flee_mask].unsqueeze(1).clamp(min=1.0)**1.5) * self.cfg.fire_flee_force * 30.0
                    fire_repulsion[flee_mask] = flee_force

                    karst_attraction[flee_mask] *= 0.1
                    vereda_attraction[flee_mask] *= 0.1
                    fruiting_attraction[flee_mask] *= 0.1

        self.fire_nodes_history.append(self.fire_nodes.cpu().numpy().copy())

        # REPRODUCTION LOGIC
        ready_to_mate = surface_mask & ~flee_mask & (self.energy > self.cfg.mating_energy_threshold)
        if ready_to_mate.any():
            ready_idx = ready_to_mate.nonzero().squeeze(1)
            for i in ready_idx:
                if self.energy[i] < self.cfg.mating_energy_threshold: continue

                dist = torch.norm(self.pos[active_mask] - self.pos[i], dim=1)
                active_indices = active_mask.nonzero().squeeze(1)

                close_mask = (dist < self.cfg.mating_radius) & (dist > 0.1)
                candidates = active_indices[close_mask]

                for j in candidates:
                    if (self.energy[j] > self.cfg.mating_energy_threshold and
                        self.is_male[i] != self.is_male[j] and
                        self.species_id[i] == self.species_id[j] and
                        not flee_mask[j]):

                        inactive_indices = (~self.is_active).nonzero().squeeze(1)
                        if len(inactive_indices) > 0:
                            child_idx = inactive_indices[0]
                            self.is_active[child_idx] = True
                            self.pos[child_idx] = self.pos[i].clone() + (torch.randn(2, device=self.dev) * 10.0)
                            self.vel[child_idx] = self.vel[i].clone() * -1.0
                            self.energy[child_idx] = 40.0
                            self.age[child_idx] = 0.0 # Newborn age 0
                            self.is_underground[child_idx] = False

                            self.energy[i] -= 45.0
                            self.energy[j] -= 45.0

                            self.birth_events.append({"pos": self.pos[child_idx].cpu().numpy().copy(), "frame": frame_idx, "color": self.colors[child_idx]})
                            break

        active_mask = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground

        # COMBINE FORCES
        raw_new_vel = torch.zeros_like(self.vel)

        if surface_mask.any():
             base_forces = (self.vel[surface_mask] * self.drags[surface_mask] +
                          karst_attraction[surface_mask] +
                          cave_attraction[surface_mask] +
                          vereda_attraction[surface_mask] +
                          fruiting_attraction[surface_mask] +
                          fire_repulsion[surface_mask] +
                          torch.randn_like(self.vel[surface_mask]) * 0.5)
             raw_new_vel[surface_mask] = base_forces

        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        final_angles = old_angles.clone()

        if surface_mask.any():
             dynamic_max_turns = self.max_turns[surface_mask].clone()
             if flee_mask.any():
                 flee_mask_surface = flee_mask[surface_mask]
                 dynamic_max_turns[flee_mask_surface] *= 3.0

             clamped_diff = torch.clamp(angle_diff[surface_mask], -dynamic_max_turns, dynamic_max_turns)
             final_angles[surface_mask] = old_angles[surface_mask] + clamped_diff

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)

        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)

        dynamic_max_speeds = self.speeds.clone()
        wading_mask = flooded_mask & surface_mask
        dynamic_max_speeds[wading_mask] *= 0.7

        if feeding_mask.any(): dynamic_max_speeds[feeding_mask] *= 0.4

        in_sand_mask = (self.pos[:, 0] < self.cfg.sand_zone_x) & surface_mask
        if in_sand_mask.any(): dynamic_max_speeds[in_sand_mask] *= self.cfg.sand_speed_modifier

        if flee_mask.any(): dynamic_max_speeds[flee_mask] *= 1.5

        self.vel = (self.vel / norms) * dynamic_max_speeds.unsqueeze(1)

        # Update positions ONLY for active agents
        self.pos[active_mask] += self.vel[active_mask] * self.cfg.dt

        # Periodic boundaries (Surface only)
        self.pos[surface_mask, 0] = self.pos[surface_mask, 0] % self.cfg.width
        self.pos[surface_mask, 1] = self.pos[surface_mask, 1] % self.cfg.height

        self._log_history()

    def _log_history(self):
        self.trajectory_history.append(self.pos.cpu().numpy().copy())
        self.visibility_history.append((~self.is_underground).cpu().numpy().copy())
        self.active_history.append(self.is_active.cpu().numpy().copy())
        self.carrying_seed_history.append(self.has_seed.cpu().numpy().copy())


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
        dur = self.cfg.frames / self.cfg.fps

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#121212; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append('<defs><pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/></pattern></defs>')
        svg.append('<defs><radialGradient id="waterGrad"><stop offset="0%" stop-color="#00ffff" stop-opacity="0.6"/><stop offset="80%" stop-color="#0066ff" stop-opacity="0.3"/><stop offset="100%" stop-color="#0033aa" stop-opacity="0.0"/></radialGradient></defs>')

        # Soil Diversity Background Setup
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="#1c1611"/>')
        svg.append('<defs><pattern id="sandDot" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="5" cy="5" r="1.5" fill="#d4a373" opacity="0.15"/><circle cx="15" cy="15" r="1.0" fill="#e2b488" opacity="0.1"/></pattern></defs>')
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="url(#sandDot)"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w - self.cfg.sand_zone_x}" height="{h}" fill="#05070a"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w - self.cfg.sand_zone_x}" height="{h}" fill="url(#dotGrid)"/>')

        # Vereda Nodes
        v_nodes = self.sim.vereda_nodes.cpu().numpy()
        r_vals = ";".join([f"{r:.1f}" for r in self.sim.vereda_radius_history])
        for i, vn in enumerate(v_nodes):
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="url(#waterGrad)">')
            svg.append(f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite" />')
            svg.append('</circle>')
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" r="8" fill="#00bbff"/>')

        # Cave Nodes
        cnodes = self.sim.cave_nodes.cpu().numpy()
        for i, cn in enumerate(cnodes):
            svg.append(f'<circle cx="{cn[0]}" cy="{cn[1]}" r="{self.cfg.cave_entry_radius}" fill="#000000" stroke="#00bbff" stroke-width="2"/>')

        # Fruiting Nodes
        f_nodes = self.sim.fruiting_nodes.cpu().numpy()
        f_r_vals = ";".join([f"{20 + (30 * f):.1f}" for f in self.sim.fruiting_intensity_history])
        f_op_vals = ";".join([f"{0.2 + (0.6 * f):.2f}" for f in self.sim.fruiting_intensity_history])
        f_c_vals = ";".join(["#fe4db7" if f > 0.0 else "#4c4c5e" for f in self.sim.fruiting_intensity_history])
        f_stroke_vals = ";".join(["#ff007f" if f > 0.5 else "#333333" for f in self.sim.fruiting_intensity_history])

        for i, fn in enumerate(f_nodes):
            svg.append(f'<circle cx="{fn[0]}" cy="{fn[1]}">')
            svg.append(f'<animate attributeName="r" values="{f_r_vals}" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="fill" values="{f_c_vals}" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="opacity" values="{f_op_vals}" dur="{dur}s" repeatCount="indefinite" />')
            svg.append('</circle>')
            svg.append(f'<circle cx="{fn[0]}" cy="{fn[1]}" r="12" stroke-width="3">')
            svg.append(f'<animate attributeName="fill" values="{f_c_vals}" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="stroke" values="{f_stroke_vals}" dur="{dur}s" repeatCount="indefinite" />')
            svg.append('</circle>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Ecosystem Mortality</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Death by Starving &amp; Old Age Limits</text>')

        # Fire Nodes Rendering
        final_fire_nodes = self.sim.fire_nodes_history[-1]
        if len(final_fire_nodes) > 0:
             birth_frames = [-1] * len(final_fire_nodes)
             for f_idx in range(self.cfg.frames):
                  frame_f_nodes = self.sim.fire_nodes_history[f_idx]
                  if len(frame_f_nodes) == 0: continue
                  for i in range(len(final_fire_nodes)):
                      if birth_frames[i] == -1 and len(frame_f_nodes) > i:
                          birth_frames[i] = f_idx

             for i, fn in enumerate(final_fire_nodes):
                  spawn_f = birth_frames[i]
                  if spawn_f == -1: spawn_f = self.cfg.frames

                  op_vals = ["0.0" if fi < spawn_f else "0.85" for fi in range(self.cfg.frames)]
                  op_str = ";".join(op_vals)

                  svg.append(f'<circle cx="{fn[0]:.1f}" cy="{fn[1]:.1f}" r="18" fill="#ff4c4c" stroke="#ffaa00" stroke-width="2">')
                  svg.append(f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
                  r_vals = ";".join([f"{14 + math.sin(f*0.5)*4:.1f}" for f in range(self.cfg.frames)])
                  svg.append(f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite" />')
                  svg.append('</circle>')

        # Seeds
        for seed_data in self.sim.dropped_seeds:
             spos = seed_data["pos"]
             s_frame = seed_data["frame"]
             op_vals = []
             col_vals = []
             rad_vals = []
             for f_idx in range(self.cfg.frames):
                  if f_idx < s_frame:
                      op_vals.append("0.0"); col_vals.append("#ffe43f"); rad_vals.append("3.5")
                  elif f_idx < s_frame + self.cfg.seed_germination_delay:
                      op_vals.append("1.0"); col_vals.append("#ffe43f"); rad_vals.append("3.5")
                  else:
                      op_vals.append("1.0"); col_vals.append("#4caf50"); rad_vals.append("6.0")
             svg.append(f'<circle cx="{spos[0]:.1f}" cy="{spos[1]:.1f}">')
             svg.append(f'<animate attributeName="r" values="{";".join(rad_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
             svg.append(f'<animate attributeName="fill" values="{";".join(col_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
             svg.append(f'<animate attributeName="opacity" values="{";".join(op_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
             svg.append('</circle>')

        # Birth Events
        for birth in self.sim.birth_events:
             bpos = birth["pos"]
             bframe = birth["frame"]
             op_vals = ["0.0"] * self.cfg.frames
             for i in range(bframe, min(bframe+20, self.cfg.frames)): op_vals[i] = "0.9"
             r_vals = ["1.0"] * self.cfg.frames
             for i in range(bframe, min(bframe+20, self.cfg.frames)): r_vals[i] = f"{(i-bframe)*2 + 5:.1f}"

             svg.append(f'<circle cx="{bpos[0]:.1f}" cy="{bpos[1]:.1f}" fill="none" stroke="{birth["color"]}" stroke-width="3">')
             svg.append(f'<animate attributeName="r" values="{";".join(r_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
             svg.append(f'<animate attributeName="opacity" values="{";".join(op_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
             svg.append('</circle>')

        # NEW: Death Events (Mortality visualization - fading X or ghost)
        for death in self.sim.death_events:
            dpos = death["pos"]
            dframe = death["frame"]
            op_vals = ["0.0"] * self.cfg.frames
            for i in range(dframe, min(dframe+30, self.cfg.frames)):
                op_vals[i] = f"{max(0.0, 1.0 - (i-dframe)/30.0):.2f}"

            d_col = "#616161" if death["reason"] == "old_age" else "#d32f2f" # Grey string for old age, red cross for starving
            svg.append(f'<g>')
            svg.append(f'<line x1="{dpos[0]-5:.1f}" y1="{dpos[1]-5:.1f}" x2="{dpos[0]+5:.1f}" y2="{dpos[1]+5:.1f}" stroke="{d_col}" stroke-width="3">')
            svg.append(f'<animate attributeName="opacity" values="{";".join(op_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></line>')
            svg.append(f'<line x1="{dpos[0]+5:.1f}" y1="{dpos[1]-5:.1f}" x2="{dpos[0]-5:.1f}" y2="{dpos[1]+5:.1f}" stroke="{d_col}" stroke-width="3">')
            svg.append(f'<animate attributeName="opacity" values="{";".join(op_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></line>')
            svg.append(f'</g>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.max_particles):
            was_ever_active = any(self.sim.active_history[f][idx] for f in range(self.cfg.frames))
            if not was_ever_active: continue

            p_color = self.sim.colors[idx]
            is_frugivore = self.sim.is_frugivore[idx].item()

            op_vals = []
            seed_indicator_op_vals = []
            for frame_idx in range(self.cfg.frames):
                 is_act = self.sim.active_history[frame_idx][idx]
                 is_vis = self.sim.visibility_history[frame_idx][idx]
                 op_vals.append("1.0" if (is_act and is_vis) else "0.0")

                 has_s = self.sim.carrying_seed_history[frame_idx][idx]
                 seed_indicator_op_vals.append("1.0" if (is_act and is_vis and has_s) else "0.0")

            op_str = ";".join(op_vals)
            seed_op_str = ";".join(seed_indicator_op_vals)

            path_x  = ";".join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";".join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Trace paths for older elements or a sample
            if idx % 8 == 0 or idx >= self.cfg.initial_particles:
                 surface_chunks = []
                 current_chunk = []
                 for frame_idx in range(0, self.cfg.frames, 2):
                      p = self.sim.trajectory_history[frame_idx][idx]
                      is_v_act = self.sim.active_history[frame_idx][idx] and self.sim.visibility_history[frame_idx][idx]
                      if is_v_act:
                           current_chunk.append(p)
                      else:
                           if len(current_chunk) > 1:
                                surface_chunks.append(current_chunk)
                           current_chunk = []
                 if len(current_chunk) > 1:
                      surface_chunks.append(current_chunk)

                 s_w = 2.0 if is_frugivore else 1.0
                 for chunk in surface_chunks:
                      d_path = "M " + " L ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in chunk])
                      # Draw newly born lines brighter so they stand out
                      base_op = "0.5" if idx >= self.cfg.initial_particles else "0.25"
                      svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="{base_op}" stroke-width="{s_w}"/>')

            # Entity Node
            rad = "5.5" if is_frugivore else "3.5"
            svg.append(f'<circle r="{rad}" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="opacity" values=\"{op_str}\" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append('</circle>')

            # Seed indicator
            if is_frugivore:
                 svg.append(f'<circle r="2.5" fill="#ffe43f">')
                 svg.append(f'<animate attributeName="cx" values=\"{path_x}\" dur="{dur}s" repeatCount="indefinite" />')
                 svg.append(f'<animate attributeName="cy" values=\"{path_y}\" dur="{dur}s" repeatCount="indefinite" />')
                 svg.append(f'<animate attributeName="opacity" values=\"{seed_op_str}\" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
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

    print(f"Initializing Mortality simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    old_age_deaths = sum(1 for d in sim.death_events if d['reason'] == 'old_age')
    starv_deaths = sum(1 for d in sim.death_events if d['reason'] == 'starvation')
    print(f"Simulation complete. Deaths - Old Age: {old_age_deaths}, Starvation: {starv_deaths}. Generating SVG...")

    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg(svg_content, 'notebook_22_1')
    

if __name__ == "__main__":
    main()
