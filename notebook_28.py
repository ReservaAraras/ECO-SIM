# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 28: Satiation (Predators Ignore Prey After Successful Hunt)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Satiation (Predators Ignore Prey After Successful Hunt) emphasizing carbon flux partitioning.
- Indicator species: Jatoba (Hymenaea courbaril).
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
from dataclasses import dataclass
from typing import List, Dict, Optional

# ===================================================================================================
# 1. SCIENTIFIC TAXONOMY & ECOLOGICAL GUILDS
# ===================================================================================================

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)":             {"speed": 4.5, "color": "#fe4db7", "weight": 0.28, "diet": "Frugivore",   "cohesion": 0.05, "drag": 0.85, "max_turn": 0.10, "lifespan_base": 300},
    "Gralha-do-campo (Cyanocorax cristatellus)":  {"speed": 5.8, "color": "#00ffdc", "weight": 0.18, "diet": "Insectivore", "cohesion": 0.08, "drag": 0.90, "max_turn": 0.15, "lifespan_base": 250},
    "Beija-flor-tesoura (Eupetomena macroura)":   {"speed": 6.2, "color": "#ffe43f", "weight": 0.14, "diet": "Insectivore", "cohesion": 0.01, "drag": 0.98, "max_turn": 0.40, "lifespan_base": 180},
    "Gavião-carijó (Rupornis magnirostris)":      {"speed": 7.0, "color": "#f44336", "weight": 0.15, "diet": "Carnivore",   "cohesion": 0.00, "drag": 0.95, "max_turn": 0.20, "lifespan_base": 350},
    "Urubu-rei (Sarcoramphus papa)":              {"speed": 5.5, "color": "#b39ddb", "weight": 0.25, "diet": "Scavenger",   "cohesion": 0.02, "drag": 0.92, "max_turn": 0.08, "lifespan_base": 400},
}

# ===================================================================================================
# 2. CONFIGURATION
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width:  int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 470          # Prompt 28: 470 frames
    fps:    int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    initial_particles: int = 240
    max_particles:     int = 400
    carrying_capacity: int = 300

    dt: float = 0.5
    cave_entry_radius: float = 30.0

    vereda_min_radius:      float = 20.0
    vereda_max_radius:      float = 90.0
    vereda_cycle_frames:    float = 200.0
    vereda_base_attraction: float = 3.0

    fruiting_cycle_frames:    float = 150.0
    fruiting_radius:          float = 50.0
    fruiting_base_attraction: float = 4.0

    seed_carry_duration:    float = 60.0
    seed_pickup_chance:     float = 0.05
    seed_germination_delay: float = 70.0

    sand_zone_x:        float = 640.0
    sand_speed_modifier: float = 0.6

    fire_start_frame:  int   = 80
    fire_spread_prob:  float = 0.08
    fire_flee_radius:  float = 150.0
    fire_flee_force:   float = 12.0
    max_fire_nodes:    int   = 120

    mating_energy_threshold: float = 85.0
    mating_radius:           float = 30.0
    energy_decay:            float = 0.12
    energy_gain_fruiting:    float = 6.0
    energy_gain_vereda:      float = 2.5
    lifespan_variance:       float = 50.0

    swarm_spawn_interval: int   = 45
    swarm_lifetime:       int   = 55
    swarm_radius:         float = 50.0
    swarm_drift:          float = 0.3
    insect_chase_radius:  float = 200.0
    insect_chase_force:   float = 8.0
    insect_energy_gain:   float = 12.0

    carrion_attract_radius: float = 220.0
    carrion_circle_radius:  float = 60.0
    carrion_linger_frames:  int   = 120
    scavenger_energy_gain:  float = 8.0

    num_nest_nodes:         int   = 6
    nest_attract_radius:    float = 180.0
    nest_attract_force:     float = 5.0
    nest_capacity:          int   = 1
    nest_energy_bonus:      float = 1.5
    homeless_restless_force: float = 2.5

    keystone_removal_frame:   int   = 230
    flock_cohesion_radius:    float = 80.0
    flock_cohesion_multiplier: float = 6.0

    # NEW: Satiation Mechanics
    hunting_chase_radius:     float = 160.0
    hunting_capture_radius:   float = 12.0
    hunting_chase_force:      float = 14.0
    predator_energy_gain:     float = 40.0
    satiation_duration:       int   = 90
    satiation_speed_penalty:  float = 0.4

CONFIG = SimulationConfig()

# ===================================================================================================
# 3. KINEMATIC ENGINE
# ===================================================================================================

class TerraRoncaEcosystem:
    """Class `TerraRoncaEcosystem` -- simulation component."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        self.pos = torch.rand((cfg.max_particles, 2), device=self.dev) * torch.tensor([cfg.width, cfg.height], device=self.dev)
        self.vel = (torch.rand((cfg.max_particles, 2), device=self.dev) - 0.5) * 10.0

        self.guilds = list(BIODIVERSITY_DB.keys())
        weights     = [BIODIVERSITY_DB[g]["weight"] for g in self.guilds]
        indices     = np.random.choice(len(self.guilds), size=cfg.max_particles, p=weights)

        self.species_id     = torch.tensor(indices, device=self.dev)
        self.speeds         = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["speed"]     for i in indices], device=self.dev)
        self.drags          = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["drag"]      for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns      = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["max_turn"]  for i in indices], device=self.dev)
        self.cohesions      = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["cohesion"]  for i in indices], device=self.dev)
        self.colors         = [BIODIVERSITY_DB[self.guilds[i]]["color"] for i in indices]
        self.is_frugivore   = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Frugivore"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_insectivore = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Insectivore" for i in indices], device=self.dev, dtype=torch.bool)
        self.is_scavenger   = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Scavenger"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_carnivore   = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Carnivore"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_male        = torch.rand(cfg.max_particles, device=self.dev) > 0.5

        self.is_active = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.is_active[:cfg.initial_particles] = True
        self.energy    = torch.ones(cfg.max_particles, device=self.dev) * 60.0
        self.age       = torch.zeros(cfg.max_particles, device=self.dev)
        self.age[:cfg.initial_particles] = torch.rand(cfg.initial_particles, device=self.dev) * 200.0
        base_ls        = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["lifespan_base"] for i in indices], device=self.dev, dtype=torch.float32)
        self.lifespan  = base_ls + torch.randn(cfg.max_particles, device=self.dev) * cfg.lifespan_variance
        self.is_underground = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)

        # NEW: Satiation states
        self.satiation_timers = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.int32)

        nest_xs = [160.0, 420.0, 680.0, 820.0, 1050.0, 320.0]
        nest_ys = [120.0, 480.0, 200.0, 430.0, 300.0,  330.0]
        self.nest_nodes    = torch.tensor(list(zip(nest_xs, nest_ys)), device=self.dev, dtype=torch.float32)
        self.nest_occupant = [-1] * cfg.num_nest_nodes
        self.particle_nest = torch.full((cfg.max_particles,), -1, device=self.dev, dtype=torch.long)

        self.nectar_nodes   = torch.tensor([[250.0, 300.0]], device=self.dev)
        self.cave_nodes     = torch.tensor([[150.0, 450.0]], device=self.dev)
        self.vereda_nodes   = torch.tensor([[600.0, 350.0], [900.0, 450.0]], device=self.dev)
        self.fruiting_nodes = torch.tensor([[300.0, 150.0], [850.0, 250.0]], device=self.dev)

        self.has_seed       = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.seed_timers    = torch.zeros(cfg.max_particles, device=self.dev)
        self.dropped_seeds: List[Dict] = []
        self.fire_nodes     = torch.zeros((0, 2), device=self.dev)
        self.swarms: List[Dict] = []
        self.carrion_sites: List[Dict] = []

        self.keystone_event: Optional[Dict] = None

        # Render logs
        self.swarm_render_log: List[Dict] = []
        self.trajectory_history         = []
        self.visibility_history         = []
        self.active_history             = []
        self.carrying_seed_history      = []
        self.fire_nodes_history         = []
        self.vereda_radius_history      = []
        self.fruiting_intensity_history = []
        self.population_history         = []
        self.nest_state_history         = []
        self.cohesion_stats             = []
        self.satiation_history          = []

        self.birth_events:  List[Dict] = []
        self.death_events:  List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel      = self.vel.clone()
        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground

        if frame_idx == self.cfg.keystone_removal_frame:
            counts = torch.bincount(self.species_id[active_mask])
            keystone_id = torch.argmax(counts).item()
            target_mask = (self.species_id == keystone_id) & active_mask
            num_removed = target_mask.sum().item()
            self.is_active[target_mask] = False
            self.has_seed[target_mask]  = False
            for idx in target_mask.nonzero().squeeze(1):
                n = int(self.particle_nest[idx])
                if n >= 0:
                    self.nest_occupant[n] = -1
                    self.particle_nest[idx] = -1
                p_np = self.pos[idx].cpu().numpy().copy()
                self.death_events.append({"pos": p_np, "frame": frame_idx, "color": self.colors[idx], "reason": "extinction"})
                self.carrion_sites.append({"pos": p_np, "spawn_frame": frame_idx, "expire_frame": frame_idx + self.cfg.carrion_linger_frames})
            self.keystone_event = {"frame": frame_idx, "species": self.guilds[keystone_id], "count": num_removed}
            active_mask = self.is_active.clone()
            surface_mask = active_mask & ~self.is_underground

        self.age[active_mask]    += 1.0
        self.energy[active_mask] -= self.cfg.energy_decay

        # Decrement satiation timers
        self.satiation_timers[self.satiation_timers > 0] -= 1

        dead = (active_mask & (self.energy <= 0.0)) | (active_mask & (self.age >= self.lifespan))
        if dead.any():
            for idx in dead.nonzero().squeeze(1):
                self.is_active[idx] = False
                self.has_seed[idx]  = False
                n = int(self.particle_nest[idx])
                if n >= 0:
                    self.nest_occupant[n] = -1
                    self.particle_nest[idx] = -1
                reason = "old_age" if self.age[idx] >= self.lifespan[idx] else "starvation"
                pos_np = self.pos[idx].cpu().numpy().copy()
                self.death_events.append({"pos": pos_np, "frame": frame_idx, "color": self.colors[idx], "reason": reason})
                self.carrion_sites.append({"pos": pos_np, "spawn_frame": frame_idx, "expire_frame": frame_idx + self.cfg.carrion_linger_frames})

        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground
        current_pop  = int(active_mask.sum().item())
        if not active_mask.any():
            self._log(current_pop, frame_idx)
            return

        # ── Flocking / Cohesion ───────────────────────────────────────────────
        flock_attraction = torch.zeros_like(self.vel)
        if surface_mask.any():
            surf_idx = surface_mask.nonzero().squeeze(1)
            dist_matrix = torch.cdist(self.pos[surface_mask], self.pos[surface_mask])
            in_range = dist_matrix < self.cfg.flock_cohesion_radius
            in_range.fill_diagonal_(False)
            neighbor_counts = in_range.sum(dim=1)
            has_neighbors = neighbor_counts > 0
            if has_neighbors.any():
                sum_pos = torch.matmul(in_range.float(), self.pos[surface_mask])
                avg_pos = sum_pos[has_neighbors] / neighbor_counts[has_neighbors].unsqueeze(1).float()
                pull = avg_pos - self.pos[surface_mask][has_neighbors]
                pull_mag = self.cohesions[surf_idx[has_neighbors]].unsqueeze(1) * self.cfg.flock_cohesion_multiplier
                flock_attraction[surf_idx[has_neighbors]] = (pull / pull.norm(dim=1, keepdim=True).clamp(min=1.0)) * pull_mag
            self.cohesion_stats.append((neighbor_counts.float().mean().item()) if len(neighbor_counts) > 0 else 0.0)
        else:
            self.cohesion_stats.append(0.0)

        # ── Hunting & Satiation ───────────────────────────────────────────────
        hunting_attraction = torch.zeros_like(self.vel)
        predator_mask = surface_mask & self.is_carnivore
        prey_mask     = surface_mask & (self.is_frugivore | self.is_insectivore)

        # Only unsatiated predators hunt
        hungry_predators = predator_mask & (self.satiation_timers <= 0)

        if hungry_predators.any() and prey_mask.any():
            pred_idx = hungry_predators.nonzero().squeeze(1)
            prey_idx = prey_mask.nonzero().squeeze(1)

            d_hunt = torch.cdist(self.pos[hungry_predators], self.pos[prey_mask])
            min_d, target_idx = torch.min(d_hunt, dim=1)

            # Chase
            chase = min_d < self.cfg.hunting_chase_radius
            if chase.any():
                targets = self.pos[prey_idx[target_idx[chase]]]
                hunters = self.pos[pred_idx[chase]]
                pull = targets - hunters
                hunting_attraction[pred_idx[chase]] = (pull / min_d[chase].unsqueeze(1).clamp(min=1.0)) * self.cfg.hunting_chase_force

                # Capture
                capture = min_d[chase] < self.cfg.hunting_capture_radius
                if capture.any():
                    captured_hunters = pred_idx[chase][capture]
                    captured_prey = prey_idx[target_idx[chase][capture]]

                    for i in range(len(captured_hunters)):
                        h_id = captured_hunters[i]
                        p_id = captured_prey[i]

                        if self.is_active[p_id]:
                            # Kill prey
                            self.is_active[p_id] = False
                            self.has_seed[p_id] = False
                            # Satiate predator
                            self.energy[h_id] += self.cfg.predator_energy_gain
                            self.satiation_timers[h_id] = self.cfg.satiation_duration

                            p_np = self.pos[p_id].cpu().numpy().copy()
                            self.death_events.append({"pos": p_np, "frame": frame_idx, "color": self.colors[p_id], "reason": "predation"})
                            self.carrion_sites.append({"pos": p_np, "spawn_frame": frame_idx, "expire_frame": frame_idx + self.cfg.carrion_linger_frames})
                            current_pop -= 1

        self.energy.clamp_(0.0, 100.0)

        # ── Environmental forces ──────────────────────────────────────────────
        d_nec = torch.cdist(self.pos, self.nectar_nodes)
        mn_nec, cl_nec = torch.min(d_nec, dim=1)
        karst_attraction = (self.nectar_nodes[cl_nec] - self.pos) / mn_nec.unsqueeze(1).clamp(min=1.0)

        cycp = (frame_idx % self.cfg.vereda_cycle_frames) / self.cfg.vereda_cycle_frames
        wlvl = (math.sin(cycp * 2*math.pi - math.pi/2) + 1.0) / 2.0
        cvrad = self.cfg.vereda_min_radius + (self.cfg.vereda_max_radius - self.cfg.vereda_min_radius) * wlvl
        self.vereda_radius_history.append(cvrad)
        d_ver = torch.cdist(self.pos, self.vereda_nodes)
        mn_ver, cl_ver = torch.min(d_ver, dim=1)
        vereda_attraction = (self.vereda_nodes[cl_ver] - self.pos) / mn_ver.unsqueeze(1).clamp(min=1.0) * (self.cfg.vereda_base_attraction * wlvl)
        flooded = mn_ver < cvrad
        vereda_attraction[flooded] *= 0.1
        if (flooded & surface_mask).any():
            self.energy[flooded & surface_mask] += self.cfg.energy_gain_vereda

        fic = (frame_idx % self.cfg.fruiting_cycle_frames) / self.cfg.fruiting_cycle_frames
        fi_int = max(0.0, math.sin(fic * 2*math.pi))
        self.fruiting_intensity_history.append(fi_int)
        d_fr = torch.cdist(self.pos, self.fruiting_nodes)
        mn_fr, cl_fr = torch.min(d_fr, dim=1)
        fruiting_attraction = torch.zeros_like(self.vel)
        feeding_mask        = torch.zeros(self.cfg.max_particles, device=self.dev, dtype=torch.bool)
        if fi_int > 0.0:
            smell = (mn_fr < 200.0) & self.is_frugivore & surface_mask
            if smell.any():
                fruiting_attraction[smell] = (self.fruiting_nodes[cl_fr[smell]] - self.pos[smell]) / mn_fr[smell].unsqueeze(1).clamp(min=1.0) * (self.cfg.fruiting_base_attraction * fi_int)
                forage = smell & (mn_fr < self.cfg.fruiting_radius)
                feeding_mask = forage & (fi_int > 0.1)
                if forage.any():
                    fruiting_attraction[forage] += torch.randn_like(fruiting_attraction[forage]) * 2.0
                    self.energy[forage] += self.cfg.energy_gain_fruiting * fi_int

        # ── Seeds ─────────────────────────────────────────────────────────────
        grabs = feeding_mask & ~self.has_seed
        if grabs.any():
            ok = grabs & (torch.rand(self.cfg.max_particles, device=self.dev) < self.cfg.seed_pickup_chance)
            if ok.any():
                self.has_seed[ok]    = True
                self.seed_timers[ok] = torch.randn(self.cfg.max_particles, device=self.dev)[ok] * 5.0
        self.seed_timers[self.has_seed] += 1.0
        drop = self.has_seed & (self.seed_timers >= self.cfg.seed_carry_duration) & surface_mask
        if drop.any():
            self.has_seed[drop] = False
            for p in self.pos[drop]:
                self.dropped_seeds.append({"pos": p.cpu().numpy().copy(), "frame": frame_idx})

        # ── Fire ──────────────────────────────────────────────────────────────
        fire_repulsion = torch.zeros_like(self.vel)
        flee_mask      = torch.zeros(self.cfg.max_particles, device=self.dev, dtype=torch.bool)
        if frame_idx >= self.cfg.fire_start_frame:
            if len(self.fire_nodes) == 0:
                self.fire_nodes = torch.tensor([[1000.0, 300.0]], device=self.dev)
            elif frame_idx % 2 == 0 and len(self.fire_nodes) < self.cfg.max_fire_nodes:
                sp = torch.rand(len(self.fire_nodes), device=self.dev) < self.cfg.fire_spread_prob
                if sp.any():
                    ns = self.fire_nodes[sp] + torch.randn((int(sp.sum()), 2), device=self.dev) * 30.0
                    ns[:, 0] = ns[:, 0].clamp(0, self.cfg.width)
                    ns[:, 1] = ns[:, 1].clamp(0, self.cfg.height)
                    self.fire_nodes = torch.cat([self.fire_nodes, ns], dim=0)
            if len(self.fire_nodes) > 0:
                d_fire = torch.cdist(self.pos, self.fire_nodes)
                mn_fire, cl_fire = torch.min(d_fire, dim=1)
                flee_mask = (mn_fire < self.cfg.fire_flee_radius) & surface_mask
                if flee_mask.any():
                    fv = self.pos[flee_mask] - self.fire_nodes[cl_fire[flee_mask]]
                    fire_repulsion[flee_mask] = (fv / mn_fire[flee_mask].unsqueeze(1).clamp(min=1.0)**1.5) * self.cfg.fire_flee_force * 30.0

        self.fire_nodes_history.append(self.fire_nodes.cpu().numpy().copy())

        # ── Insect swarms ─────────────────────────────────────────────────────
        if frame_idx % self.cfg.swarm_spawn_interval == 0:
            cx, cy = random.uniform(100, self.cfg.width - 100), random.uniform(80,  self.cfg.height - 80)
            centre = torch.tensor([cx, cy], device=self.dev)
            drift  = (torch.rand(2, device=self.dev) - 0.5) * self.cfg.swarm_drift * 2.0
            self.swarms.append({"centre": centre, "drift": drift, "spawn_frame": frame_idx, "die_frame": frame_idx + self.cfg.swarm_lifetime})
            self.swarm_render_log.append({"centre": centre.cpu().numpy().copy(), "spawn_frame": frame_idx, "die_frame": frame_idx + self.cfg.swarm_lifetime, "radius": self.cfg.swarm_radius})
        insect_attraction = torch.zeros_like(self.vel)
        live_sw = []
        for sw in self.swarms:
            if frame_idx >= sw["die_frame"]: continue
            sw["centre"] += sw["drift"]
            sw["centre"][0] = sw["centre"][0].clamp(50, self.cfg.width  - 50)
            sw["centre"][1] = sw["centre"][1].clamp(50, self.cfg.height - 50)
            live_sw.append(sw)
            d_sw = torch.norm(self.pos - sw["centre"], dim=1)
            chase = (d_sw < self.cfg.insect_chase_radius) & self.is_insectivore & surface_mask & ~flee_mask
            if chase.any():
                insect_attraction[chase] += (sw["centre"] - self.pos[chase]) / d_sw[chase].unsqueeze(1).clamp(min=1.0) * self.cfg.insect_chase_force
            feed_sw = (d_sw < self.cfg.swarm_radius) & self.is_insectivore & surface_mask
            if feed_sw.any():
                self.energy[feed_sw] += self.cfg.insect_energy_gain
        self.swarms = live_sw

        # ── Scavenging ────────────────────────────────────────────────────────
        scav_attraction = torch.zeros_like(self.vel)
        self.carrion_sites = [c for c in self.carrion_sites if c["expire_frame"] > frame_idx]
        if self.carrion_sites and self.is_scavenger.any():
            carr_pos = torch.tensor([c["pos"] for c in self.carrion_sites], device=self.dev, dtype=torch.float32)
            d_carr = torch.cdist(self.pos, carr_pos)
            mn_carr, cl_carr = torch.min(d_carr, dim=1)
            scav_surf = self.is_scavenger & surface_mask & ~flee_mask
            if scav_surf.any():
                closest = carr_pos[cl_carr[scav_surf]]
                dist    = mn_carr[scav_surf]
                si      = scav_surf.nonzero().squeeze(1)
                far     = dist > self.cfg.carrion_circle_radius
                if far.any():
                    scav_attraction[si[far]] = (closest[far] - self.pos[si[far]]) / dist[far].unsqueeze(1).clamp(min=1.0) * 6.0
                near = ~far
                if near.any():
                    to_c    = self.pos[si[near]] - closest[near]
                    tangent = torch.stack([-to_c[:, 1], to_c[:, 0]], dim=1)
                    tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp(min=1e-5) * 5.0
                    scav_attraction[si[near]] = tangent
                    self.energy[si[near]] += self.cfg.scavenger_energy_gain

        # ── Nesting ───────────────────────────────────────────────────────────
        nest_attraction = torch.zeros_like(self.vel)
        for ni in range(self.cfg.num_nest_nodes):
            occ = self.nest_occupant[ni]
            if occ >= 0 and not self.is_active[occ]:
                self.nest_occupant[ni] = -1; self.particle_nest[occ] = -1

        has_nest_mask = self.particle_nest >= 0
        homeless_mask = active_mask & ~has_nest_mask
        nested_mask   = active_mask & has_nest_mask
        if nested_mask.any():
            self.energy[nested_mask] += self.cfg.nest_energy_bonus
            nest_idxs = self.particle_nest[nested_mask]
            nest_positions = self.nest_nodes[nest_idxs]
            d_own_nest = torch.norm(self.pos[nested_mask] - nest_positions, dim=1)
            nest_attraction[nested_mask] = (nest_positions - self.pos[nested_mask]) / d_own_nest.unsqueeze(1).clamp(min=1.0) * 2.0

        free_nests = [ni for ni in range(self.cfg.num_nest_nodes) if self.nest_occupant[ni] == -1]
        if free_nests and homeless_mask.any():
            free_nest_pos = self.nest_nodes[torch.tensor(free_nests, device=self.dev)]
            d_free = torch.cdist(self.pos[homeless_mask], free_nest_pos)
            mn_free, best_free = torch.min(d_free, dim=1)
            homeless_idx = homeless_mask.nonzero().squeeze(1)
            close_enough = mn_free < self.cfg.nest_attract_radius
            if close_enough.any():
                targets = free_nest_pos[best_free[close_enough]]
                dist_t  = mn_free[close_enough]
                nest_attraction[homeless_idx[close_enough]] = (targets - self.pos[homeless_idx[close_enough]]) / dist_t.unsqueeze(1).clamp(min=1.0) * self.cfg.nest_attract_force
                arrived = dist_t < 20.0
                if arrived.any():
                    for k_local, k_global in enumerate(homeless_idx[close_enough][arrived]):
                        best_ni = best_free[close_enough][arrived][k_local].item()
                        real_ni = free_nests[best_ni]
                        if self.nest_occupant[real_ni] == -1:
                            self.nest_occupant[real_ni] = int(k_global); self.particle_nest[int(k_global)] = real_ni

        if homeless_mask.any():
            nest_attraction[homeless_mask] += torch.randn(int(homeless_mask.sum().item()), 2, device=self.dev) * self.cfg.homeless_restless_force

        # ── Reproduction ──────────────────────────────────────────────────────
        if current_pop < self.cfg.carrying_capacity:
            ready = surface_mask & ~flee_mask & (self.energy > self.cfg.mating_energy_threshold)
            if ready.any():
                for i in ready.nonzero().squeeze(1):
                    if self.energy[i] < self.cfg.mating_energy_threshold: continue
                    ai    = active_mask.nonzero().squeeze(1)
                    d_all = torch.norm(self.pos[ai] - self.pos[i], dim=1)
                    for j in ai[(d_all < self.cfg.mating_radius) & (d_all > 0.1)]:
                        if (self.energy[j] > self.cfg.mating_energy_threshold and self.is_male[i] != self.is_male[j] and self.species_id[i] == self.species_id[j] and not flee_mask[j]):
                            free = (~self.is_active).nonzero().squeeze(1)
                            if len(free):
                                ci = free[0]
                                self.is_active[ci] = True
                                self.pos[ci] = self.pos[i] + torch.randn(2, device=self.dev) * 10.0
                                self.vel[ci] = self.vel[i] * -1.0
                                self.energy[ci] = 40.0; self.age[ci] = 0.0
                                self.particle_nest[ci] = -1; self.is_underground[ci] = False
                                self.energy[i] -= 45.0; self.energy[j] -= 45.0
                                self.birth_events.append({"pos": self.pos[ci].cpu().numpy().copy(), "frame": frame_idx, "color": self.colors[ci]})
                                current_pop += 1
                            break
                    if current_pop >= self.cfg.carrying_capacity: break

        # ── Velocity & Position ───────────────────────────────────────────────
        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground
        raw = torch.zeros_like(self.vel)
        if surface_mask.any():
            raw[surface_mask] = (
                self.vel[surface_mask] * self.drags[surface_mask]
                + karst_attraction[surface_mask]
                + vereda_attraction[surface_mask]
                + fruiting_attraction[surface_mask]
                + fire_repulsion[surface_mask]
                + insect_attraction[surface_mask]
                + scav_attraction[surface_mask]
                + nest_attraction[surface_mask]
                + flock_attraction[surface_mask]
                + hunting_attraction[surface_mask]
                + torch.randn_like(self.vel[surface_mask]) * 0.5
            )

        new_ang = torch.atan2(raw[:, 1], raw[:, 0])
        old_ang = torch.atan2(old_vel[:, 1], old_vel[:, 0])
        final_ang = old_ang.clone()
        if surface_mask.any():
            dyn_t = self.max_turns[surface_mask].clone()
            if flee_mask.any(): dyn_t[flee_mask[surface_mask]] *= 3.0
            diff = ((new_ang - old_ang + math.pi) % (2*math.pi) - math.pi)[surface_mask]
            final_ang[surface_mask] = old_ang[surface_mask] + torch.clamp(diff, -dyn_t, dyn_t)

        spd = torch.norm(raw, dim=1).clamp(min=0.1)
        self.vel[:, 0] = torch.cos(final_ang) * spd
        self.vel[:, 1] = torch.sin(final_ang) * spd
        nrm = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)
        ds  = self.speeds.clone()
        ds[flooded & surface_mask] *= 0.7
        if feeding_mask.any():   ds[feeding_mask] *= 0.4

        # Satiated predators walk very slow
        satiated = (self.satiation_timers > 0) & surface_mask
        if satiated.any(): ds[satiated] *= self.cfg.satiation_speed_penalty

        in_sand = (self.pos[:, 0] < self.cfg.sand_zone_x) & surface_mask
        if in_sand.any():        ds[in_sand] *= self.cfg.sand_speed_modifier
        if flee_mask.any():      ds[flee_mask] *= 1.5
        self.vel = (self.vel / nrm) * ds.unsqueeze(1)

        self.pos[active_mask] += self.vel[active_mask] * self.cfg.dt
        self.pos[surface_mask, 0] = self.pos[surface_mask, 0] % self.cfg.width
        self.pos[surface_mask, 1] = self.pos[surface_mask, 1] % self.cfg.height
        self._log(current_pop, frame_idx)

    def _log(self, pop, frame_idx):
        self.trajectory_history.append(self.pos.cpu().numpy().copy())
        self.visibility_history.append((~self.is_underground).cpu().numpy().copy())
        self.active_history.append(self.is_active.cpu().numpy().copy())
        self.satiation_history.append(self.satiation_timers.cpu().numpy().copy())
        self.population_history.append(pop)
        self.nest_state_history.append(list(self.nest_occupant))

# ===================================================================================================
# 4. VISUALIZATION
# ===================================================================================================

class EcosystemRenderer:
    """Class `EcosystemRenderer` -- simulation component."""

    def __init__(self, cfg: SimulationConfig, sim: TerraRoncaEcosystem):
        self.cfg = cfg
        self.sim = sim

    def generate_svg(self) -> str:
        """Function `generate_svg` -- simulation component."""

        w, h   = self.cfg.width, self.cfg.height
        dur    = self.cfg.frames / self.cfg.fps
        frames = self.cfg.frames

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#0a040a;font-family:system-ui, -apple-system, sans-serif;">']
        svg.append('<defs>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
              '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
              '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="waterGrad"><stop offset="0%" stop-color="#00ffff" stop-opacity="0.6"/><stop offset="100%" stop-color="#0033aa" stop-opacity="0.0"/></radialGradient>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="4" fill="#5a1a5a" opacity="0.45"/></pattern>'
            '<pattern id="sandDot" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="5" cy="5" r="1.5" fill="#d4a373" opacity="0.25"/><circle cx="15" cy="15" r="1.0" fill="#e2b488" opacity="0.15"/></pattern>'
            '<radialGradient id="carrionGrad"><stop offset="0%" stop-color="#8d6e63" stop-opacity="0.90"/><stop offset="100%" stop-color="#3e2723" stop-opacity="0.0"/></radialGradient>'
            '<radialGradient id="predatorGlow">'
              '<stop offset="0%"   stop-color="#ff5252" stop-opacity="0.70"/>'
              '<stop offset="60%"  stop-color="#d50000" stop-opacity="0.30"/>'
              '<stop offset="100%" stop-color="#7f0000" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>')

        # Background
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="#1c1611"/>')
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="url(#sandDot)"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="#05070a"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="url(#dotGrid)"/>')

        # Vereda
        r_vals = ";".join(f"{r:.1f}" for r in self.sim.vereda_radius_history)
        for vn in self.sim.vereda_nodes.cpu().numpy():
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="url(#waterGrad)"><animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/></circle>'
                       f'<circle cx="{vn[0]}" cy="{vn[1]}" r="16" fill="none" stroke="#00ccff" stroke-width="1.5" stroke-dasharray="6,4"><animate attributeName="r" values="14;22;14" dur="3.2s" repeatCount="indefinite"/></circle>'
                       f'<circle cx="{vn[0]}" cy="{vn[1]}" r="8" fill="#00bbff" filter="url(#glow)"/>')

        # Title
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ff5252" font-weight="bold">ECO-SIM: Satiation &amp; Predation</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Predators Ignore Prey After Successful Hunt</text>')

        # Carrion - pulsing brown decay blobs
        for c in self.sim.death_events:
            cpos, cf, ef = c["pos"], c["frame"], c["frame"] + self.cfg.carrion_linger_frames
            op_vals = ["0.0" if fi<cf or fi>=ef else (f"{(fi-cf)/8.0:.2f}" if fi-cf<8 else (f"{(ef-fi)/20.0:.2f}" if ef-fi<20 else "1.0")) for fi in range(frames)]
            op_str = ";".join(op_vals)
            svg.append(f'<circle cx="{cpos[0]:.1f}" cy="{cpos[1]:.1f}" r="{self.cfg.carrion_circle_radius*1.4:.1f}" fill="url(#carrionGrad)">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{self.cfg.carrion_circle_radius:.1f};{self.cfg.carrion_circle_radius*1.8:.1f};{self.cfg.carrion_circle_radius:.1f}" dur="3.5s" repeatCount="indefinite"/>'
                       f'</circle>')

        # TRACES
        for idx in range(self.cfg.max_particles):
            if not any(self.sim.active_history[fi][idx] for fi in range(frames)): continue
            p_col = self.sim.colors[idx]
            is_carnivore = self.sim.is_carnivore[idx].item()

            px = ";".join(f"{p[idx,0]:.1f}" for p in self.sim.trajectory_history)
            py = ";".join(f"{p[idx,1]:.1f}" for p in self.sim.trajectory_history)

            op_vals = []
            sleep_ops = []
            for fi in range(frames):
                act = self.sim.active_history[fi][idx]
                vis = self.sim.visibility_history[fi][idx]
                sat = self.sim.satiation_history[fi][idx] > 0

                op_vals.append("1.0" if (act and vis) else "0.0")
                sleep_ops.append("1.0" if (act and vis and is_carnivore and sat) else "0.0")

            if idx % 8 == 0 or idx >= self.cfg.initial_particles or is_carnivore:
                chunks, cur = [], []
                for fi in range(0, frames, 2):
                    if self.sim.active_history[fi][idx]: cur.append(self.sim.trajectory_history[fi][idx])
                    elif len(cur)>1: chunks.append(cur); cur=[]
                if len(cur)>1: chunks.append(cur)
                sw = 2.0 if is_carnivore else 1.5
                for chunk in chunks:
                    svg.append(f'<path d="M {" L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in chunk)}" fill="none" stroke="{p_col}" stroke-opacity="0.35" stroke-width="{sw}"/>')

            # Vivid r-pulse; carnivores get predatorGlow halo
            rad = 6.0 if is_carnivore else 3.5
            r_begin = f"{(idx % 14) * 0.18:.2f}s"
            r_pulse = "5;9;5" if is_carnivore else "3;5.2;3"

            if is_carnivore or idx % 6 == 0:
                halo_fill = "url(#predatorGlow)" if is_carnivore else p_col
                halo_r = "12;20;12" if is_carnivore else "6;12;6"
                halo_op_str = ";".join("0.0" if v == "0.0" else ("0.40" if is_carnivore else "0.25") for v in op_vals)
                svg.append(f'<circle r="14" fill="{halo_fill}">'
                           f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="opacity" values="{halo_op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="r" values="{halo_r}" dur="{1.4 if is_carnivore else 2.0}s" begin="{r_begin}" repeatCount="indefinite"/>'
                           f'</circle>')

            svg.append(f'<circle r="{rad}" fill="{p_col}">'
                       f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{";".join(op_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{r_pulse}" dur="{1.6 if is_carnivore else 2.2}s" begin="{r_begin}" repeatCount="indefinite"/>'
                       f'</circle>')

            # Satiation sleep icon "Z" with animated position
            if is_carnivore:
                svg.append(f'<text font-size="15" fill="#ffcccc" font-weight="bold">'
                           f'<animate attributeName="x" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="y" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="opacity" values="{";".join(sleep_ops)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>Z</text>')

        # Educational card
        cw, ch = 375, 140
        svg.append(f'<g transform="translate({w-cw-20},340)">'
                   f'<rect width="{cw}" height="{ch}" fill="#1a1a2e" rx="8" ry="8" stroke="#f44336" stroke-width="1" opacity="0.95"/>'
                   f'<text x="15" y="25" fill="#cccccc" font-size="15" font-weight="bold">Feature: Satiation &amp; Predation</text>'
                   f'<text font-weight="bold" x="15" y="48" fill="#cccccc" font-size="15">Carnivores (Gavião-carijó) hunt prey actively.</text>'
                   f'<text font-weight="bold" x="15" y="68" fill="#cccccc" font-size="15">After a successful kill, they become Satiated</text>'
                   f'<text font-weight="bold" x="15" y="88" fill="#cccccc" font-size="15">while satiated. Predators ignore prey,</text>'
                   f'<text font-weight="bold" x="15" y="108" fill="#cccccc" font-size="15">glow with a "Z" icon, and move {self.cfg.satiation_speed_penalty*100}% slower.</text>'
                   f'</g>')

        # -- Scientific Validation Watermark --
        svg.append(f'<g transform="translate(10, {h - 15})">')

        svg.append('</g>')
        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

# ===================================================================================================
# 5. EXECUTION
# ===================================================================================================

def save_svg_to_drive(svg_content: str, notebook_id: str):
    """Function `save_svg_to_drive` -- simulation component."""

    drive_folder = "/content/drive/MyDrive/ReservaAraras_SVGs"
    save_dir = drive_folder if os.path.isdir('/content/drive') else os.path.join(
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd(), 'svg_output')
    os.makedirs(save_dir, exist_ok=True)
    fp = os.path.join(save_dir, f'{notebook_id}.svg')
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"SVG saved -> {fp}")

def main():
    """Function `main` -- simulation component."""

    print(f"Initializing Satiation simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for fi in range(CONFIG.frames):
        sim.step(fi)
    print(f"Simulation complete. Pop: {sim.population_history[-1]}. Generating SVG...")
    renderer    = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_28')

if __name__ == "__main__":
    main()
