# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 25: Scavenging (Vultures Circle Death Sites)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Scavenging (Vultures Circle Death Sites) emphasizing restoration patch recovery.
- Indicator species: Lontra (Lontra longicaudis).
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
from dataclasses import dataclass
from typing import List, Dict

# ===================================================================================================
# 1. SCIENTIFIC TAXONOMY & ECOLOGICAL GUILDS
# ===================================================================================================

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)":             {"speed": 4.5, "color": "#fe4db7", "weight": 0.30, "diet": "Frugivore",   "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.10, "lifespan_base": 300},
    "Gralha-do-campo (Cyanocorax cristatellus)":  {"speed": 5.8, "color": "#00ffdc", "weight": 0.20, "diet": "Insectivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15, "lifespan_base": 250},
    "Beija-flor-tesoura (Eupetomena macroura)":   {"speed": 6.2, "color": "#ffe43f", "weight": 0.15, "diet": "Insectivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.40, "lifespan_base": 180},
    "Gavião-carijó (Rupornis magnirostris)":      {"speed": 7.0, "color": "#f44336", "weight": 0.15, "diet": "Carnivore",   "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.20, "lifespan_base": 350},
    # NEW: Urubu-rei — Scavenger guild
    "Urubu-rei (Sarcoramphus papa)":              {"speed": 5.5, "color": "#b39ddb", "weight": 0.20, "diet": "Scavenger",   "seed_drop_prob": 0.000, "drag": 0.92, "max_turn": 0.08, "lifespan_base": 400},
}

# ===================================================================================================
# 2. CONFIGURATION
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width:  int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 440          # Prompt 25: 440 frames
    fps:    int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    initial_particles: int = 220
    max_particles:     int = 400
    carrying_capacity: int = 300

    dt: float = 0.5
    cave_entry_radius: float = 30.0
    cover_radius:      float = 65.0

    vereda_min_radius:     float = 20.0
    vereda_max_radius:     float = 90.0
    vereda_cycle_frames:   float = 200.0
    vereda_base_attraction: float = 3.0

    fruiting_cycle_frames:    float = 150.0
    fruiting_radius:          float = 50.0
    fruiting_base_attraction: float = 4.0

    seed_carry_duration:    float = 60.0
    seed_pickup_chance:     float = 0.05
    seed_germination_delay: float = 70.0

    sand_zone_x:       float = 640.0
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
    swarm_count:          int   = 80
    swarm_radius:         float = 50.0
    swarm_drift:          float = 0.3
    insect_chase_radius:  float = 200.0
    insect_chase_force:   float = 8.0
    insect_energy_gain:   float = 12.0

    # NEW: Scavenging parameters
    carrion_attract_radius: float = 220.0  # How far scavengers sense a carcass
    carrion_circle_radius:  float = 60.0   # Orbital radius of circling behaviour
    carrion_linger_frames:  int   = 120    # How long a carcass stays attractive
    scavenger_energy_gain:  float = 8.0    # Energy per frame from feeding on carcass

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

        guilds  = list(BIODIVERSITY_DB.keys())
        weights = [BIODIVERSITY_DB[g]["weight"] for g in guilds]
        indices = np.random.choice(len(guilds), size=cfg.max_particles, p=weights)

        self.species_id    = torch.tensor(indices, device=self.dev)
        self.speeds        = torch.tensor([BIODIVERSITY_DB[guilds[i]]["speed"]     for i in indices], device=self.dev)
        self.drags         = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"]      for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns     = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"]  for i in indices], device=self.dev)
        self.colors        = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]
        self.is_frugivore  = torch.tensor([BIODIVERSITY_DB[guilds[i]]["diet"] == "Frugivore"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_insectivore = torch.tensor([BIODIVERSITY_DB[guilds[i]]["diet"] == "Insectivore" for i in indices], device=self.dev, dtype=torch.bool)
        self.is_scavenger  = torch.tensor([BIODIVERSITY_DB[guilds[i]]["diet"] == "Scavenger"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_male       = torch.rand(cfg.max_particles, device=self.dev) > 0.5

        self.is_active     = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.is_active[:cfg.initial_particles] = True
        self.energy        = torch.ones(cfg.max_particles, device=self.dev) * 60.0
        self.age           = torch.zeros(cfg.max_particles, device=self.dev)
        self.age[:cfg.initial_particles] = torch.rand(cfg.initial_particles, device=self.dev) * 200.0
        base_ls            = torch.tensor([BIODIVERSITY_DB[guilds[i]]["lifespan_base"] for i in indices], device=self.dev, dtype=torch.float32)
        self.lifespan      = base_ls + torch.randn(cfg.max_particles, device=self.dev) * cfg.lifespan_variance
        self.is_underground = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)

        self.nectar_nodes   = torch.tensor([[250.0, 300.0]], device=self.dev)
        self.cave_nodes     = torch.tensor([[150.0, 450.0]], device=self.dev)
        self.cover_nodes    = torch.tensor([[1050.0, 200.0]], device=self.dev)
        self.vereda_nodes   = torch.tensor([[600.0, 350.0], [900.0, 450.0]], device=self.dev)
        self.fruiting_nodes = torch.tensor([[300.0, 150.0], [850.0, 250.0]], device=self.dev)

        self.has_seed       = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.seed_timers    = torch.zeros(cfg.max_particles, device=self.dev)
        self.dropped_seeds: List[Dict] = []
        self.fire_nodes     = torch.zeros((0, 2), device=self.dev)
        self.swarms: List[Dict] = []

        # NEW: Carrion (carcass) list — created when agents die
        # Each: {"pos": np.array, "spawn_frame": int, "expire_frame": int}
        self.carrion_sites: List[Dict] = []
        # Scavenger orbit angles (per particle), so each vulture circles a unique arc
        self.scav_orbit_angle = torch.rand(cfg.max_particles, device=self.dev) * 2 * math.pi

        # Render logs
        self.swarm_render_log:   List[Dict] = []
        self.trajectory_history         = []
        self.visibility_history         = []
        self.active_history             = []
        self.carrying_seed_history      = []
        self.fire_nodes_history         = []
        self.vereda_radius_history      = []
        self.fruiting_intensity_history = []
        self.population_history         = []
        self.birth_events:  List[Dict]  = []
        self.death_events:  List[Dict]  = []

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel      = self.vel.clone()
        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground

        # ── Aging + decay ─────────────────────────────────────────────────────
        self.age[active_mask]    += 1.0
        self.energy[active_mask] -= self.cfg.energy_decay

        dead_mask = (active_mask & (self.energy <= 0.0)) | (active_mask & (self.age >= self.lifespan))
        if dead_mask.any():
            for idx in dead_mask.nonzero().squeeze(1):
                self.is_active[idx] = False
                self.has_seed[idx]  = False
                reason = "old_age" if self.age[idx] >= self.lifespan[idx] else "starvation"
                pos_np = self.pos[idx].cpu().numpy().copy()
                self.death_events.append({"pos": pos_np, "frame": frame_idx,
                                          "color": self.colors[idx], "reason": reason})
                # NEW: Register carcass
                self.carrion_sites.append({
                    "pos":          pos_np,
                    "spawn_frame":  frame_idx,
                    "expire_frame": frame_idx + self.cfg.carrion_linger_frames,
                })

        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground
        current_pop  = int(active_mask.sum().item())

        if not active_mask.any():
            self._log(current_pop)
            return

        # ── Environmental attractions ─────────────────────────────────────────
        d_nec = torch.cdist(self.pos, self.nectar_nodes)
        mn_nec, cl_nec = torch.min(d_nec, dim=1)
        karst_attraction = (self.nectar_nodes[cl_nec] - self.pos) / mn_nec.unsqueeze(1).clamp(min=1.0)

        d_cave = torch.cdist(self.pos, self.cave_nodes)
        mn_cave, cl_cave = torch.min(d_cave, dim=1)
        cave_attraction  = (self.cave_nodes[cl_cave] - self.pos) / mn_cave.unsqueeze(1).clamp(min=1.0) * 2.0

        cyc_p    = (frame_idx % self.cfg.vereda_cycle_frames) / self.cfg.vereda_cycle_frames
        wlvl     = (math.sin(cyc_p * 2*math.pi - math.pi/2) + 1.0) / 2.0
        cur_vrad = self.cfg.vereda_min_radius + (self.cfg.vereda_max_radius - self.cfg.vereda_min_radius) * wlvl
        self.vereda_radius_history.append(cur_vrad)

        d_ver = torch.cdist(self.pos, self.vereda_nodes)
        mn_ver, cl_ver = torch.min(d_ver, dim=1)
        vereda_attraction = (self.vereda_nodes[cl_ver] - self.pos) / mn_ver.unsqueeze(1).clamp(min=1.0) * (self.cfg.vereda_base_attraction * wlvl)
        flooded = mn_ver < cur_vrad
        vereda_attraction[flooded] *= 0.1
        if (flooded & surface_mask).any():
            self.energy[flooded & surface_mask] += self.cfg.energy_gain_vereda

        fi_cyc = (frame_idx % self.cfg.fruiting_cycle_frames) / self.cfg.fruiting_cycle_frames
        fi_int = max(0.0, math.sin(fi_cyc * 2*math.pi))
        self.fruiting_intensity_history.append(fi_int)

        d_fr = torch.cdist(self.pos, self.fruiting_nodes)
        mn_fr, cl_fr = torch.min(d_fr, dim=1)
        fruiting_attraction = torch.zeros_like(self.vel)
        feeding_mask        = torch.zeros(self.cfg.max_particles, device=self.dev, dtype=torch.bool)
        if fi_int > 0.0:
            f_pull = self.fruiting_nodes[cl_fr] - self.pos
            smell  = (mn_fr < 200.0) & self.is_frugivore & surface_mask
            if smell.any():
                fruiting_attraction[smell] = f_pull[smell] / mn_fr[smell].unsqueeze(1).clamp(min=1.0) * (self.cfg.fruiting_base_attraction * fi_int)
                forage = smell & (mn_fr < self.cfg.fruiting_radius)
                feeding_mask = forage & (fi_int > 0.1)
                if forage.any():
                    fruiting_attraction[forage] += torch.randn_like(fruiting_attraction[forage]) * 2.0
                    self.energy[forage] += self.cfg.energy_gain_fruiting * fi_int
        self.energy.clamp_(0.0, 100.0)

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
                    ns[:,0] = ns[:,0].clamp(0, self.cfg.width)
                    ns[:,1] = ns[:,1].clamp(0, self.cfg.height)
                    self.fire_nodes = torch.cat([self.fire_nodes, ns], dim=0)
            if len(self.fire_nodes) > 0:
                d_fire = torch.cdist(self.pos, self.fire_nodes)
                mn_fire, cl_fire = torch.min(d_fire, dim=1)
                flee_mask = (mn_fire < self.cfg.fire_flee_radius) & surface_mask
                if flee_mask.any():
                    fv = self.pos[flee_mask] - self.fire_nodes[cl_fire[flee_mask]]
                    fire_repulsion[flee_mask] = (fv / mn_fire[flee_mask].unsqueeze(1).clamp(min=1.0)**1.5) * self.cfg.fire_flee_force * 30.0
                    karst_attraction[flee_mask]    *= 0.1
                    vereda_attraction[flee_mask]   *= 0.1
                    fruiting_attraction[flee_mask] *= 0.1
        self.fire_nodes_history.append(self.fire_nodes.cpu().numpy().copy())

        # ── Insect swarms ─────────────────────────────────────────────────────
        if frame_idx % self.cfg.swarm_spawn_interval == 0:
            cx = random.uniform(100, self.cfg.width  - 100)
            cy = random.uniform(80,  self.cfg.height - 80)
            centre = torch.tensor([cx, cy], device=self.dev)
            drift  = (torch.rand(2, device=self.dev) - 0.5) * self.cfg.swarm_drift * 2.0
            self.swarms.append({"centre": centre, "drift": drift,
                                "spawn_frame": frame_idx,
                                "die_frame":   frame_idx + self.cfg.swarm_lifetime})
            self.swarm_render_log.append({"centre": centre.cpu().numpy().copy(),
                                          "spawn_frame": frame_idx,
                                          "die_frame":   frame_idx + self.cfg.swarm_lifetime,
                                          "radius": self.cfg.swarm_radius})

        insect_attraction = torch.zeros_like(self.vel)
        live_swarms = []
        for sw in self.swarms:
            if frame_idx >= sw["die_frame"]: continue
            sw["centre"] += sw["drift"]
            sw["centre"][0] = sw["centre"][0].clamp(50, self.cfg.width  - 50)
            sw["centre"][1] = sw["centre"][1].clamp(50, self.cfg.height - 50)
            live_swarms.append(sw)
            d_sw = torch.norm(self.pos - sw["centre"], dim=1)
            chase = (d_sw < self.cfg.insect_chase_radius) & self.is_insectivore & surface_mask & ~flee_mask
            if chase.any():
                insect_attraction[chase] += (sw["centre"] - self.pos[chase]) / d_sw[chase].unsqueeze(1).clamp(min=1.0) * self.cfg.insect_chase_force
            feed_sw = (d_sw < self.cfg.swarm_radius) & self.is_insectivore & surface_mask
            if feed_sw.any():
                self.energy[feed_sw] += self.cfg.insect_energy_gain
                self.energy.clamp_(0.0, 100.0)
        self.swarms = live_swarms

        # ── NEW: SCAVENGING — Vultures circle carcasses ───────────────────────
        scav_attraction  = torch.zeros_like(self.vel)
        active_carrion   = [c for c in self.carrion_sites if c["expire_frame"] > frame_idx]
        self.carrion_sites = active_carrion  # prune expired sites

        if active_carrion and self.is_scavenger.any():
            # Build a tensor of active carcass positions
            carrion_pos = torch.tensor(
                [c["pos"] for c in active_carrion], device=self.dev, dtype=torch.float32)

            d_carr = torch.cdist(self.pos, carrion_pos)      # (max_particles, num_carrion)
            mn_carr, cl_carr = torch.min(d_carr, dim=1)

            scav_surf = self.is_scavenger & surface_mask & ~flee_mask
            if scav_surf.any():
                closest_carcass = carrion_pos[cl_carr[scav_surf]]   # (N_scav, 2)
                dist_to_carcass = mn_carr[scav_surf]

                # Two behaviours based on distance:
                # > circle_radius: fly directly toward carcass
                # <= circle_radius: orbit in a tangent direction (circling)
                far_mask  = dist_to_carcass > self.cfg.carrion_circle_radius
                near_mask = ~far_mask

                scav_indices = scav_surf.nonzero().squeeze(1)

                if far_mask.any():
                    pull = closest_carcass[far_mask] - self.pos[scav_indices[far_mask]]
                    scav_attraction[scav_indices[far_mask]] = pull / dist_to_carcass[far_mask].unsqueeze(1).clamp(min=1.0) * 6.0

                if near_mask.any():
                    # Tangent direction: perpendicular to the radius vector → circling
                    to_carcass = self.pos[scav_indices[near_mask]] - closest_carcass[near_mask]
                    tangent = torch.stack([-to_carcass[:, 1], to_carcass[:, 0]], dim=1)  # 90° CCW
                    tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp(min=1e-5) * 5.0
                    scav_attraction[scav_indices[near_mask]] = tangent
                    # Gain energy while circling (represents feeding)
                    self.energy[scav_indices[near_mask]] += self.cfg.scavenger_energy_gain
                    self.energy.clamp_(0.0, 100.0)

        # ── Carrying-capacity reproduction gate ───────────────────────────────
        if current_pop < self.cfg.carrying_capacity:
            ready = surface_mask & ~flee_mask & (self.energy > self.cfg.mating_energy_threshold)
            if ready.any():
                for i in ready.nonzero().squeeze(1):
                    if self.energy[i] < self.cfg.mating_energy_threshold: continue
                    ai   = active_mask.nonzero().squeeze(1)
                    d_all = torch.norm(self.pos[ai] - self.pos[i], dim=1)
                    for j in ai[(d_all < self.cfg.mating_radius) & (d_all > 0.1)]:
                        if (self.energy[j] > self.cfg.mating_energy_threshold and
                                self.is_male[i] != self.is_male[j] and
                                self.species_id[i] == self.species_id[j] and
                                not flee_mask[j]):
                            free = (~self.is_active).nonzero().squeeze(1)
                            if len(free):
                                ci = free[0]
                                self.is_active[ci]      = True
                                self.pos[ci]            = self.pos[i] + torch.randn(2, device=self.dev) * 10.0
                                self.vel[ci]            = self.vel[i] * -1.0
                                self.energy[ci]         = 40.0
                                self.age[ci]            = 0.0
                                self.is_underground[ci] = False
                                self.energy[i] -= 45.0
                                self.energy[j] -= 45.0
                                self.birth_events.append({"pos": self.pos[ci].cpu().numpy().copy(),
                                                          "frame": frame_idx,
                                                          "color": self.colors[ci]})
                                current_pop += 1
                            break
                    if current_pop >= self.cfg.carrying_capacity:
                        break

        # ── Combine forces → velocity → position ──────────────────────────────
        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground
        raw = torch.zeros_like(self.vel)
        if surface_mask.any():
            raw[surface_mask] = (
                self.vel[surface_mask] * self.drags[surface_mask]
                + karst_attraction[surface_mask]
                + cave_attraction[surface_mask]
                + vereda_attraction[surface_mask]
                + fruiting_attraction[surface_mask]
                + fire_repulsion[surface_mask]
                + insect_attraction[surface_mask]
                + scav_attraction[surface_mask]       # NEW
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
        ds = self.speeds.clone()
        ds[flooded & surface_mask] *= 0.7
        if feeding_mask.any():  ds[feeding_mask] *= 0.4
        in_sand = (self.pos[:, 0] < self.cfg.sand_zone_x) & surface_mask
        if in_sand.any():       ds[in_sand] *= self.cfg.sand_speed_modifier
        if flee_mask.any():     ds[flee_mask] *= 1.5
        self.vel = (self.vel / nrm) * ds.unsqueeze(1)

        self.pos[active_mask] += self.vel[active_mask] * self.cfg.dt
        self.pos[surface_mask, 0] = self.pos[surface_mask, 0] % self.cfg.width
        self.pos[surface_mask, 1] = self.pos[surface_mask, 1] % self.cfg.height
        self._log(current_pop)

    def _log(self, pop):
        self.trajectory_history.append(self.pos.cpu().numpy().copy())
        self.visibility_history.append((~self.is_underground).cpu().numpy().copy())
        self.active_history.append(self.is_active.cpu().numpy().copy())
        self.carrying_seed_history.append(self.has_seed.cpu().numpy().copy())
        self.population_history.append(pop)


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

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"'
               f' style="background-color:#0a0702; font-family:system-ui, -apple-system, sans-serif;">']

        # ── defs ──────────────────────────────────────────────────────────────
        svg.append('<defs>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
              '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
              '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="waterGrad">'
              '<stop offset="0%"   stop-color="#00ffff" stop-opacity="0.6"/>'
              '<stop offset="80%"  stop-color="#0066ff" stop-opacity="0.3"/>'
              '<stop offset="100%" stop-color="#0033aa" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
              '<circle cx="2" cy="2" r="4" fill="#6a4820" opacity="0.45"/>'
            '</pattern>'
            '<pattern id="sandDot" width="20" height="20" patternUnits="userSpaceOnUse">'
              '<circle cx="5"  cy="5"  r="1.5" fill="#d4a373" opacity="0.30"/>'
              '<circle cx="15" cy="15" r="1.0" fill="#e2b488" opacity="0.20"/>'
            '</pattern>'
            '<radialGradient id="swarmGlow">'
              '<stop offset="0%"   stop-color="#ffff88" stop-opacity="0.55"/>'
              '<stop offset="60%"  stop-color="#ffcc00" stop-opacity="0.25"/>'
              '<stop offset="100%" stop-color="#ff9900" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="carrionGrad">'
              '<stop offset="0%"   stop-color="#8d6e63" stop-opacity="0.90"/>'
              '<stop offset="55%"  stop-color="#5d4037" stop-opacity="0.55"/>'
              '<stop offset="100%" stop-color="#3e2723" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="vultureGlow">'
              '<stop offset="0%"   stop-color="#ce93d8" stop-opacity="0.70"/>'
              '<stop offset="60%"  stop-color="#9c27b0" stop-opacity="0.30"/>'
              '<stop offset="100%" stop-color="#4a148c" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>')

        # ── soil background ───────────────────────────────────────────────────
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="#1c1611"/>')
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="url(#sandDot)"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="#05070a"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="url(#dotGrid)"/>')

        # ── vereda ───────────────────────────────────────────────────────────
        r_vals = ";".join(f"{r:.1f}" for r in self.sim.vereda_radius_history)
        for vn in self.sim.vereda_nodes.cpu().numpy():
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="url(#waterGrad)">'
                       f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'</circle>'
                       f'<circle cx="{vn[0]}" cy="{vn[1]}" r="16" fill="none" stroke="#00ccff" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.6">'
                       f'<animate attributeName="r" values="14;22;14" dur="3.2s" repeatCount="indefinite"/>'
                       f'<animate attributeName="stroke-opacity" values="0.6;0.2;0.6" dur="3.2s" repeatCount="indefinite"/>'
                       f'</circle>'
                       f'<circle cx="{vn[0]}" cy="{vn[1]}" r="8" fill="#00bbff" filter="url(#glow)"/>')

        # ── fruiting ─────────────────────────────────────────────────────────
        fi = self.sim.fruiting_intensity_history
        f_r  = ";".join(f"{20+30*v:.1f}" for v in fi)
        f_c  = ";".join("#fe4db7" if v > 0 else "#4c4c5e" for v in fi)
        f_op = ";".join(f"{0.2+0.6*v:.2f}" for v in fi)
        f_st = ";".join("#ff007f" if v > 0.5 else "#333333" for v in fi)
        for fn in self.sim.fruiting_nodes.cpu().numpy():
            svg.append(f'<circle cx="{fn[0]}" cy="{fn[1]}">'
                       f'<animate attributeName="r" values="{f_r}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="fill" values="{f_c}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{f_op}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'</circle>')

        # ── title ─────────────────────────────────────────────────────────────
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ce93d8" font-weight="bold">'
                   f'ECO-SIM: Scavenging &amp; Carrion Dynamics</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">'
                   f'Vultures Circle Death Sites</text>')

        # ── NEW: Carrion / carcass visualisation ──────────────────────────────
        for c in self.sim.death_events:
            cpos, cf   = c["pos"], c["frame"]
            expire_f   = cf + self.cfg.carrion_linger_frames
            cr         = self.cfg.carrion_circle_radius

            op_vals = []
            for fi_idx in range(frames):
                if fi_idx < cf or fi_idx >= expire_f:
                    op_vals.append("0.0")
                else:
                    age   = fi_idx - cf
                    total = expire_f - cf
                    if age < 8:   op_vals.append(f"{age/8.0:.2f}")
                    elif total - age < 20: op_vals.append(f"{(total-age)/20.0:.2f}")
                    else:         op_vals.append("1.0")
            op_str = ";".join(op_vals)

            # Carcass decay blob with pulsing ring and circling glow
            svg.append(f'<circle cx="{cpos[0]:.1f}" cy="{cpos[1]:.1f}" r="{cr*1.5:.1f}" fill="url(#carrionGrad)">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{cr*1.2:.1f};{cr*1.8:.1f};{cr*1.2:.1f}" dur="3.5s" repeatCount="indefinite"/>'
                       f'</circle>')
            svg.append(f'<circle cx="{cpos[0]:.1f}" cy="{cpos[1]:.1f}" r="{cr:.1f}" fill="none" stroke="#5d4037" stroke-width="2" stroke-dasharray="5,4">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{cr*0.9:.1f};{cr*1.3:.1f};{cr*0.9:.1f}" dur="2.2s" repeatCount="indefinite"/>'
                       f'</circle>')
            # Bone × symbol
            svg.append(f'<g>'
                       f'<line x1="{cpos[0]-9:.1f}" y1="{cpos[1]-9:.1f}" x2="{cpos[0]+9:.1f}" y2="{cpos[1]+9:.1f}" stroke="#8d6e63" stroke-width="3" stroke-linecap="round">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></line>'
                       f'<line x1="{cpos[0]+9:.1f}" y1="{cpos[1]-9:.1f}" x2="{cpos[0]-9:.1f}" y2="{cpos[1]+9:.1f}" stroke="#8d6e63" stroke-width="3" stroke-linecap="round">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></line>'
                       f'</g>')

        # ── Swarms ────────────────────────────────────────────────────────────
        for sw in self.sim.swarm_render_log:
            cx, cy = sw["centre"]
            sf, df = sw["spawn_frame"], sw["die_frame"]
            rad    = sw["radius"]
            op_vals = []
            for fi_idx in range(frames):
                if fi_idx < sf or fi_idx >= df: op_vals.append("0.0")
                else:
                    age = fi_idx - sf; life = df - sf
                    if age < 10: op_vals.append(f"{age/10.0:.2f}")
                    elif life - age < 15: op_vals.append(f"{(life-age)/15.0:.2f}")
                    else: op_vals.append("1.0")
            op_str = ";".join(op_vals)
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad*1.4:.1f}" fill="url(#swarmGlow)">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # ── Fire ──────────────────────────────────────────────────────────────
        final_fn = self.sim.fire_nodes_history[-1]
        if len(final_fn) > 0:
            bf = [-1] * len(final_fn)
            for fi_idx in range(frames):
                ff = self.sim.fire_nodes_history[fi_idx]
                for i in range(len(final_fn)):
                    if bf[i] == -1 and len(ff) > i: bf[i] = fi_idx
            rp = ";".join(f"{14+math.sin(fi*0.5)*4:.1f}" for fi in range(frames))
            for i, fn in enumerate(final_fn):
                sf = bf[i] if bf[i] != -1 else frames
                op = ";".join("0.0" if fi < sf else "0.85" for fi in range(frames))
                svg.append(f'<circle cx="{fn[0]:.1f}" cy="{fn[1]:.1f}" fill="#ff4c4c" stroke="#ffaa00" stroke-width="2">'
                           f'<animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="r" values="{rp}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'</circle>')

        # ── Seeds ─────────────────────────────────────────────────────────────
        for sd in self.sim.dropped_seeds:
            sp, sf = sd["pos"], sd["frame"]
            ov, cv, rv = [], [], []
            for fi_idx in range(frames):
                if fi_idx < sf:                          ov.append("0.0"); cv.append("#ffe43f"); rv.append("3.5")
                elif fi_idx < sf+self.cfg.seed_germination_delay: ov.append("1.0"); cv.append("#ffe43f"); rv.append("3.5")
                else:                                    ov.append("1.0"); cv.append("#4caf50"); rv.append("6.0")
            svg.append(f'<circle cx="{sp[0]:.1f}" cy="{sp[1]:.1f}">'
                       f'<animate attributeName="r"       values="{";".join(rv)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="fill"    values="{";".join(cv)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{";".join(ov)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # ── Births ────────────────────────────────────────────────────────────
        for b in self.sim.birth_events:
            bp, bf2 = b["pos"], b["frame"]
            ov = ["0.9" if bf2 <= fi < bf2+20 else "0.0" for fi in range(frames)]
            rv = [f"{(fi-bf2)*2+5:.1f}" if bf2 <= fi < bf2+20 else "1.0" for fi in range(frames)]
            svg.append(f'<circle cx="{bp[0]:.1f}" cy="{bp[1]:.1f}" fill="none" stroke="{b["color"]}" stroke-width="3">'
                       f'<animate attributeName="r"       values="{";".join(rv)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{";".join(ov)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # ── Agent trajectories ────────────────────────────────────────────────
        for idx in range(self.cfg.max_particles):
            if not any(self.sim.active_history[f][idx] for f in range(frames)): continue
            p_color     = self.sim.colors[idx]
            is_frugivore = self.sim.is_frugivore[idx].item()
            is_insect   = self.sim.is_insectivore[idx].item()
            is_scav     = self.sim.is_scavenger[idx].item()

            op_vals, seed_op = [], []
            for fi in range(frames):
                act = self.sim.active_history[fi][idx]
                vis = self.sim.visibility_history[fi][idx]
                op_vals.append("1.0" if (act and vis) else "0.0")
                seed_op.append("1.0" if (act and vis and self.sim.carrying_seed_history[fi][idx]) else "0.0")

            px = ";".join(f"{p[idx,0]:.1f}" for p in self.sim.trajectory_history)
            py = ";".join(f"{p[idx,1]:.1f}" for p in self.sim.trajectory_history)
            op_str = ";".join(op_vals)

            # Traces
            if idx % 8 == 0 or idx >= self.cfg.initial_particles or is_scav or is_insect:
                chunks, cur = [], []
                for fi in range(0, frames, 2):
                    alive = self.sim.active_history[fi][idx] and self.sim.visibility_history[fi][idx]
                    if alive: cur.append(self.sim.trajectory_history[fi][idx])
                    elif len(cur) > 1: chunks.append(cur); cur = []
                if len(cur) > 1: chunks.append(cur)
                sw = 2.5 if is_scav else (2.0 if is_insect else (1.5 if is_frugivore else 1.0))
                base_op = "0.6" if is_scav else ("0.5" if idx >= self.cfg.initial_particles else "0.25")
                for chunk in chunks:
                    svg.append(f'<path d="M {" L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in chunk)}"'
                               f' fill="none" stroke="{p_color}" stroke-opacity="{base_op}" stroke-width="{sw}"/>')

            # Scavengers get vivid vulture glow + r-pulse; all particles get r-pulse
            rad = "7.0" if is_scav else ("5.5" if is_frugivore or is_insect else "3.5")
            r_begin = f"{(idx % 14) * 0.18:.2f}s"
            r_pulse = "6;10;6" if is_scav else ("4;7;4" if is_insect else ("3.5;6;3.5" if is_frugivore else "3;5.2;3"))

            if is_scav or idx % 6 == 0:
                halo_r = "12;22;12" if is_scav else "6;12;6"
                halo_fill = "url(#vultureGlow)" if is_scav else p_color
                halo_op = "0.40" if is_scav else "0.25"
                svg.append(f'<circle r="14" fill="{halo_fill}" opacity="{halo_op}">'
                           f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="opacity" values="{";".join("0.0" if v=="0.0" else halo_op for v in op_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="r" values="{halo_r}" dur="{1.6 if is_scav else 2.0}s" begin="{r_begin}" repeatCount="indefinite"/>'
                           f'</circle>')

            svg.append(f'<circle r="{rad}" fill="{p_color}">'
                       f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{r_pulse}" dur="{1.8 if is_scav else 2.2}s" begin="{r_begin}" repeatCount="indefinite"/>'
                       f'</circle>')
            if is_frugivore:
                svg.append(f'<circle r="3" fill="#ffe43f">'
                           f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="opacity" values="{";".join(seed_op)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'</circle>')

        # ── Educational card ──────────────────────────────────────────────────
        cw, ch = 375, 175
        svg.append(f'<g transform="translate({w-cw-20},340)">'
                   f'<rect width="{cw}" height="{ch}" fill="#1a1a2e" rx="8" ry="8" stroke="#b39ddb" stroke-width="1" opacity="0.95"/>'
                   f'<text x="15" y="25" fill="#cccccc" font-size="15" font-weight="bold">Feature: Scavenging (Urubu-rei Guild)</text>'
                   f'<text font-weight="bold" x="15" y="48" fill="#cccccc" font-size="15">When any agent dies, a Carrion site is created</text>'
                   f'<text font-weight="bold" x="15" y="66" fill="#cccccc" font-size="15">and lingers (brown decay blob).</text>'
                   f'<text font-weight="bold" x="15" y="88" fill="#cccccc" font-size="15">Urubu-rei scavengers within {self.cfg.carrion_attract_radius:.0f} units are attracted.</text>'
                   f'<text font-weight="bold" x="15" y="106" fill="#cccccc" font-size="15">On approach they switch to orbital circling</text>'
                   f'<text font-weight="bold" x="15" y="124" fill="#cccccc" font-size="15">(tangent force vectors), gaining +{self.cfg.scavenger_energy_gain:.0f} energy per step</text>'
                   f'<text font-weight="bold" x="15" y="150" fill="#cccccc" font-size="15">Adapted from: Sarcoramphus papa thermal soaring</text>'
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
    return fp

def main():
    """Function `main` -- simulation component."""

    print(f"Initializing Scavenging simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for fi in range(CONFIG.frames):
        sim.step(fi)
    scavs = int(sim.is_scavenger.sum())
    print(f"Simulation complete. Deaths recorded: {len(sim.death_events)}, "
          f"Scavenger agents: {scavs}, Final pop: {sim.population_history[-1]}. Generating SVG...")
    renderer    = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_25')

if __name__ == "__main__":
    main()
