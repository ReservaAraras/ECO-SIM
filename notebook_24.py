# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 24: Insect Swarms (Temporary Prey Clouds & Insectivore Chase)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Insect Swarms (Temporary Prey Clouds & Insectivore Chase) emphasizing invasive front expansion.
- Indicator species: Capivara (Hydrochoerus hydrochaeris).
- Pollination lens: pollen limitation in fragmented edges.
- Human impact lens: climate warming on water balance.

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
# 1. SCIENTIFIC TAXONOMY & ECOLOGICAL GUILDS
# ===================================================================================================

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)":        {"speed": 4.5,  "color": "#fe4db7", "weight": 0.35, "diet": "Frugivore",   "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.10, "lifespan_base": 300},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "color": "#00ffdc", "weight": 0.25, "diet": "Insectivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15, "lifespan_base": 250},
    "Beija-flor-tesoura (Eupetomena macroura)":  {"speed": 6.2, "color": "#ffe43f", "weight": 0.20, "diet": "Insectivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.40, "lifespan_base": 180},
    "Gavião-carijó (Rupornis magnirostris)":     {"speed": 7.0, "color": "#f44336", "weight": 0.20, "diet": "Carnivore",   "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.20, "lifespan_base": 350},
}

# ===================================================================================================
# 2. CONFIGURATION
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int  = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 430          # Prompt 24: 430 frames
    fps: int    = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    initial_particles: int = 220
    max_particles: int     = 350
    carrying_capacity: int = 280

    dt: float  = 0.5
    cave_entry_radius: float = 30.0
    cover_radius: float      = 65.0

    vereda_min_radius: float  = 20.0
    vereda_max_radius: float  = 90.0
    vereda_cycle_frames: float = 200.0
    vereda_base_attraction: float = 3.0

    fruiting_cycle_frames: float   = 150.0
    fruiting_radius: float         = 50.0
    fruiting_base_attraction: float = 4.0

    seed_carry_duration: float   = 60.0
    seed_pickup_chance: float    = 0.05
    seed_germination_delay: float = 70.0

    sand_zone_x: float       = 640.0
    sand_speed_modifier: float = 0.6

    fire_start_frame: int  = 80
    fire_spread_prob: float = 0.08
    fire_flee_radius: float = 150.0
    fire_flee_force: float  = 12.0
    max_fire_nodes: int     = 120

    mating_energy_threshold: float = 85.0
    mating_radius: float           = 30.0
    energy_decay: float            = 0.12
    energy_gain_fruiting: float    = 6.0
    energy_gain_vereda: float      = 2.5
    lifespan_variance: float       = 50.0

    # NEW: Insect Swarm parameters
    swarm_spawn_interval: int   = 40    # Frames between new swarms
    swarm_lifetime: int         = 60    # Frames each swarm stays active
    swarm_count: int            = 80    # Insect particles per swarm
    swarm_radius: float         = 50.0  # Radius of the cloud blob
    swarm_drift: float          = 0.3   # How fast the whole swarm drifts
    insect_chase_radius: float  = 200.0 # How far insectivores can sense a swarm
    insect_chase_force: float   = 8.0   # Attraction force toward swarm centre
    insect_energy_gain: float   = 12.0  # Energy gained per frame when feeding in swarm

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

        self.species_id   = torch.tensor(indices, device=self.dev)
        self.speeds       = torch.tensor([BIODIVERSITY_DB[guilds[i]]["speed"]       for i in indices], device=self.dev)
        self.drags        = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"]        for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns    = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"]    for i in indices], device=self.dev)
        self.colors       = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]
        self.is_frugivore = torch.tensor([BIODIVERSITY_DB[guilds[i]]["diet"] == "Frugivore" for i in indices], device=self.dev, dtype=torch.bool)
        self.is_insectivore = torch.tensor([BIODIVERSITY_DB[guilds[i]]["diet"] == "Insectivore" for i in indices], device=self.dev, dtype=torch.bool)
        self.is_male      = torch.rand(cfg.max_particles, device=self.dev) > 0.5

        self.is_active    = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.is_active[:cfg.initial_particles] = True
        self.energy       = torch.ones(cfg.max_particles, device=self.dev) * 60.0
        self.age          = torch.zeros(cfg.max_particles, device=self.dev)
        self.age[:cfg.initial_particles] = torch.rand(cfg.initial_particles, device=self.dev) * 200.0
        base_ls           = torch.tensor([BIODIVERSITY_DB[guilds[i]]["lifespan_base"] for i in indices], device=self.dev, dtype=torch.float32)
        self.lifespan     = base_ls + torch.randn(cfg.max_particles, device=self.dev) * cfg.lifespan_variance
        self.is_underground = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)

        self.nectar_nodes  = torch.tensor([[250.0, 300.0]], device=self.dev)
        self.cave_nodes    = torch.tensor([[150.0, 450.0]], device=self.dev)
        self.cover_nodes   = torch.tensor([[1050.0, 200.0]], device=self.dev)
        self.vereda_nodes  = torch.tensor([[600.0, 350.0], [900.0, 450.0]], device=self.dev)
        self.fruiting_nodes = torch.tensor([[300.0, 150.0], [850.0, 250.0]], device=self.dev)

        self.has_seed      = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.seed_timers   = torch.zeros(cfg.max_particles, device=self.dev)
        self.dropped_seeds = []
        self.fire_nodes    = torch.zeros((0, 2), device=self.dev)

        # NEW: Insect swarm state
        # Each swarm: centre(x,y), velocity(vx,vy), spawn_frame, alive_until_frame
        self.swarms: List[Dict] = []   # list of dicts with tensor fields
        self.swarm_render_log  = []    # for SVG rendering: list per swarm of (centre, spawn_f, die_f)

        # History
        self.trajectory_history    = []
        self.visibility_history    = []
        self.active_history        = []
        self.carrying_seed_history = []
        self.fire_nodes_history    = []
        self.vereda_radius_history = []
        self.fruiting_intensity_history = []
        self.population_history    = []
        self.swarm_centre_history  = []  # [(centre_array_or_None, alive) per frame]

        self.birth_events = []
        self.death_events = []

    # ------------------------------------------------------------------ helpers
    def _count_active(self) -> int:
        return int(self.is_active.sum().item())

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel    = self.vel.clone()
        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground

        # ── Aging & energy decay ──────────────────────────────────────────────
        self.age[active_mask]    += 1.0
        self.energy[active_mask] -= self.cfg.energy_decay

        starved   = active_mask & (self.energy <= 0.0)
        old_age   = active_mask & (self.age >= self.lifespan)
        dead_mask = starved | old_age
        if dead_mask.any():
            for idx in dead_mask.nonzero().squeeze(1):
                self.is_active[idx] = False
                self.has_seed[idx]  = False
                reason = "old_age" if old_age[idx] else "starvation"
                self.death_events.append({"pos": self.pos[idx].cpu().numpy().copy(),
                                          "frame": frame_idx,
                                          "color": self.colors[idx],
                                          "reason": reason})

        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground
        current_pop  = self._count_active()

        if not active_mask.any():
            self._log_history(current_pop, frame_idx)
            return

        # ── Environmental forces ──────────────────────────────────────────────
        d_nec = torch.cdist(self.pos, self.nectar_nodes)
        mn_nec, cl_nec = torch.min(d_nec, dim=1)
        karst_attraction = (self.nectar_nodes[cl_nec] - self.pos) / mn_nec.unsqueeze(1).clamp(min=1.0)

        d_cave = torch.cdist(self.pos, self.cave_nodes)
        mn_cave, cl_cave = torch.min(d_cave, dim=1)
        cave_attraction  = (self.cave_nodes[cl_cave] - self.pos) / mn_cave.unsqueeze(1).clamp(min=1.0) * 2.0

        cycle_p   = (frame_idx % self.cfg.vereda_cycle_frames) / self.cfg.vereda_cycle_frames
        water_lvl = (math.sin(cycle_p * 2 * math.pi - math.pi/2) + 1.0) / 2.0
        cur_v_rad = self.cfg.vereda_min_radius + (self.cfg.vereda_max_radius - self.cfg.vereda_min_radius) * water_lvl
        self.vereda_radius_history.append(cur_v_rad)

        d_ver = torch.cdist(self.pos, self.vereda_nodes)
        mn_ver, cl_ver = torch.min(d_ver, dim=1)
        dyn_attr = self.cfg.vereda_base_attraction * water_lvl
        vereda_attraction = (self.vereda_nodes[cl_ver] - self.pos) / mn_ver.unsqueeze(1).clamp(min=1.0) * dyn_attr
        flooded_mask = mn_ver < cur_v_rad
        vereda_attraction[flooded_mask] *= 0.1
        if (flooded_mask & surface_mask).any():
            self.energy[flooded_mask & surface_mask] += self.cfg.energy_gain_vereda

        fruit_cycle = (frame_idx % self.cfg.fruiting_cycle_frames) / self.cfg.fruiting_cycle_frames
        fruit_int   = max(0.0, math.sin(fruit_cycle * 2 * math.pi))
        self.fruiting_intensity_history.append(fruit_int)

        d_fruit = torch.cdist(self.pos, self.fruiting_nodes)
        mn_fruit, cl_fruit = torch.min(d_fruit, dim=1)
        fruiting_attraction = torch.zeros_like(self.vel)
        feeding_mask        = torch.zeros(self.cfg.max_particles, device=self.dev, dtype=torch.bool)

        if fruit_int > 0.0:
            f_pull = self.fruiting_nodes[cl_fruit] - self.pos
            smell  = (mn_fruit < 200.0) & self.is_frugivore & surface_mask
            if smell.any():
                fruiting_attraction[smell] = f_pull[smell] / mn_fruit[smell].unsqueeze(1).clamp(min=1.0) * (self.cfg.fruiting_base_attraction * fruit_int)
                forage = smell & (mn_fruit < self.cfg.fruiting_radius)
                feeding_mask = forage & (fruit_int > 0.1)
                if forage.any():
                    fruiting_attraction[forage] += torch.randn_like(fruiting_attraction[forage]) * 2.0
                    self.energy[forage] += self.cfg.energy_gain_fruiting * fruit_int

        self.energy.clamp_(0.0, 100.0)

        # ── Seed dispersal ────────────────────────────────────────────────────
        grabs = feeding_mask & ~self.has_seed
        if grabs.any():
            chance = torch.rand(self.cfg.max_particles, device=self.dev) < self.cfg.seed_pickup_chance
            ok = grabs & chance
            if ok.any():
                self.has_seed[ok]     = True
                self.seed_timers[ok]  = torch.randn(self.cfg.max_particles, device=self.dev)[ok] * 5.0
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
                    ns = self.fire_nodes[sp] + torch.randn((sp.sum().item(), 2), device=self.dev) * 30.0
                    ns[:, 0] = ns[:, 0].clamp(0, self.cfg.width)
                    ns[:, 1] = ns[:, 1].clamp(0, self.cfg.height)
                    self.fire_nodes = torch.cat([self.fire_nodes, ns], dim=0)
            if len(self.fire_nodes) > 0:
                d_fire = torch.cdist(self.pos, self.fire_nodes)
                mn_fire, cl_fire = torch.min(d_fire, dim=1)
                flee_mask = (mn_fire < self.cfg.fire_flee_radius) & surface_mask
                if flee_mask.any():
                    fvec = self.pos[flee_mask] - self.fire_nodes[cl_fire[flee_mask]]
                    fire_repulsion[flee_mask] = (fvec / mn_fire[flee_mask].unsqueeze(1).clamp(min=1.0)**1.5) * self.cfg.fire_flee_force * 30.0
                    karst_attraction[flee_mask]   *= 0.1
                    vereda_attraction[flee_mask]  *= 0.1
                    fruiting_attraction[flee_mask] *= 0.1

        self.fire_nodes_history.append(self.fire_nodes.cpu().numpy().copy())

        # ── NEW: INSECT SWARM MECHANICS ───────────────────────────────────────
        # 1. Spawn new swarm every `swarm_spawn_interval` frames
        if frame_idx % self.cfg.swarm_spawn_interval == 0:
            cx = random.uniform(100, self.cfg.width - 100)
            cy = random.uniform(80, self.cfg.height - 80)
            centre = torch.tensor([cx, cy], device=self.dev)
            drift  = (torch.rand(2, device=self.dev) - 0.5) * self.cfg.swarm_drift * 2.0
            # Particle offsets around centre (jitter cloud)
            offsets = torch.randn(self.cfg.swarm_count, 2, device=self.dev) * (self.cfg.swarm_radius / 3.0)
            swarm_positions = (centre + offsets).clamp(
                torch.tensor([0.0, 0.0], device=self.dev),
                torch.tensor([float(self.cfg.width), float(self.cfg.height)], device=self.dev))

            self.swarms.append({
                "positions":   swarm_positions,
                "centre":      centre.clone(),
                "drift":       drift,
                "spawn_frame": frame_idx,
                "die_frame":   frame_idx + self.cfg.swarm_lifetime,
            })
            self.swarm_render_log.append({
                "centre":      centre.cpu().numpy().copy(),
                "spawn_frame": frame_idx,
                "die_frame":   frame_idx + self.cfg.swarm_lifetime,
                "radius":      self.cfg.swarm_radius,
            })

        # 2. Step active swarms (drift + diffuse) and build insectivore attraction
        insect_attraction = torch.zeros_like(self.vel)
        active_swarms = []
        for sw in self.swarms:
            if frame_idx >= sw["die_frame"]:
                continue   # Swarm dissipated
            # Drift entire cloud
            sw["centre"] += sw["drift"]
            sw["centre"][0] = sw["centre"][0].clamp(50, self.cfg.width  - 50)
            sw["centre"][1] = sw["centre"][1].clamp(50, self.cfg.height - 50)
            # Diffuse individual positions slightly
            sw["positions"] += torch.randn_like(sw["positions"]) * 1.5
            active_swarms.append(sw)

            # Insectivore attraction: towards swarm centre if close enough
            d_centre = torch.norm(self.pos - sw["centre"], dim=1)
            chase_mask = (d_centre < self.cfg.insect_chase_radius) & self.is_insectivore & surface_mask & ~flee_mask
            if chase_mask.any():
                pull = sw["centre"] - self.pos[chase_mask]
                insect_attraction[chase_mask] += (pull / d_centre[chase_mask].unsqueeze(1).clamp(min=1.0)) * self.cfg.insect_chase_force

            # Energy reward for insectivores actually inside swarm zone
            feeding_range = (d_centre < self.cfg.swarm_radius) & self.is_insectivore & surface_mask
            if feeding_range.any():
                self.energy[feeding_range] += self.cfg.insect_energy_gain
                self.energy.clamp_(0.0, 100.0)

        self.swarms = active_swarms

        # ── Carrying-capacity gate on reproduction ────────────────────────────
        if current_pop < self.cfg.carrying_capacity:
            ready = surface_mask & ~flee_mask & (self.energy > self.cfg.mating_energy_threshold)
            if ready.any():
                ready_idx = ready.nonzero().squeeze(1)
                for i in ready_idx:
                    if self.energy[i] < self.cfg.mating_energy_threshold: continue
                    d_all = torch.norm(self.pos[active_mask] - self.pos[i], dim=1)
                    act_idx = active_mask.nonzero().squeeze(1)
                    cands = act_idx[(d_all < self.cfg.mating_radius) & (d_all > 0.1)]
                    for j in cands:
                        if (self.energy[j] > self.cfg.mating_energy_threshold and
                                self.is_male[i] != self.is_male[j] and
                                self.species_id[i] == self.species_id[j] and
                                not flee_mask[j]):
                            free = (~self.is_active).nonzero().squeeze(1)
                            if len(free) > 0:
                                ci = free[0]
                                self.is_active[ci]    = True
                                self.pos[ci]          = self.pos[i] + torch.randn(2, device=self.dev) * 10.0
                                self.vel[ci]          = self.vel[i] * -1.0
                                self.energy[ci]       = 40.0
                                self.age[ci]          = 0.0
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

        # ── Forces → velocity → position ─────────────────────────────────────
        active_mask  = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground
        raw_new_vel  = torch.zeros_like(self.vel)

        if surface_mask.any():
            raw_new_vel[surface_mask] = (
                self.vel[surface_mask] * self.drags[surface_mask]
                + karst_attraction[surface_mask]
                + cave_attraction[surface_mask]
                + vereda_attraction[surface_mask]
                + fruiting_attraction[surface_mask]
                + fire_repulsion[surface_mask]
                + insect_attraction[surface_mask]   # NEW
                + torch.randn_like(self.vel[surface_mask]) * 0.5
            )

        new_ang = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])
        old_ang = torch.atan2(old_vel[:, 1], old_vel[:, 0])
        if surface_mask.any():
            dyn_turns = self.max_turns[surface_mask].clone()
            if flee_mask.any():
                dyn_turns[flee_mask[surface_mask]] *= 3.0
            diff = ((new_ang - old_ang + math.pi) % (2 * math.pi) - math.pi)[surface_mask]
            diff = torch.clamp(diff, -dyn_turns, dyn_turns)
            final_ang = old_ang.clone()
            final_ang[surface_mask] = old_ang[surface_mask] + diff
        else:
            final_ang = old_ang.clone()

        spd_mag = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)
        self.vel[:, 0] = torch.cos(final_ang) * spd_mag
        self.vel[:, 1] = torch.sin(final_ang) * spd_mag
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)
        dyn_spd = self.speeds.clone()
        dyn_spd[flooded_mask & surface_mask] *= 0.7
        if feeding_mask.any():  dyn_spd[feeding_mask] *= 0.4
        in_sand = (self.pos[:, 0] < self.cfg.sand_zone_x) & surface_mask
        if in_sand.any():       dyn_spd[in_sand] *= self.cfg.sand_speed_modifier
        if flee_mask.any():     dyn_spd[flee_mask] *= 1.5
        self.vel = (self.vel / norms) * dyn_spd.unsqueeze(1)

        self.pos[active_mask] += self.vel[active_mask] * self.cfg.dt
        self.pos[surface_mask, 0] = self.pos[surface_mask, 0] % self.cfg.width
        self.pos[surface_mask, 1] = self.pos[surface_mask, 1] % self.cfg.height

        self._log_history(current_pop, frame_idx)

    def _log_history(self, current_pop, frame_idx):
        self.trajectory_history.append(self.pos.cpu().numpy().copy())
        self.visibility_history.append((~self.is_underground).cpu().numpy().copy())
        self.active_history.append(self.is_active.cpu().numpy().copy())
        self.carrying_seed_history.append(self.has_seed.cpu().numpy().copy())
        self.population_history.append(current_pop)


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

        w, h  = self.cfg.width, self.cfg.height
        dur   = self.cfg.frames / self.cfg.fps
        frames = self.cfg.frames

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"'
               f' style="background-color:#121212; font-family:system-ui, -apple-system, sans-serif;">']

        # ── defs ──
        svg.append('<defs>'
                   '<radialGradient id="waterGrad">'
                   '<stop offset="0%" stop-color="#00ffff" stop-opacity="0.6"/>'
                   '<stop offset="80%" stop-color="#0066ff" stop-opacity="0.3"/>'
                   '<stop offset="100%" stop-color="#0033aa" stop-opacity="0.0"/>'
                   '</radialGradient>'
                   '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
                   '<circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/>'
                   '</pattern>'
                   '<radialGradient id="swarmGlow" cx="50%" cy="50%" r="50%">'
                   '<stop offset="0%"   stop-color="#ffff88" stop-opacity="0.55"/>'
                   '<stop offset="60%"  stop-color="#ffcc00" stop-opacity="0.25"/>'
                   '<stop offset="100%" stop-color="#ff9900" stop-opacity="0.0"/>'
                   '</radialGradient>'
                   '</defs>')

        # ── soil background ──
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="#1c1611"/>')
        svg.append('<defs><pattern id="sandDot" width="20" height="20" patternUnits="userSpaceOnUse">'
                   '<circle cx="5" cy="5" r="1.5" fill="#d4a373" opacity="0.15"/>'
                   '<circle cx="15" cy="15" r="1.0" fill="#e2b488" opacity="0.1"/>'
                   '</pattern></defs>')
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="url(#sandDot)"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="#05070a"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="url(#dotGrid)"/>')

        # ── vereda ──
        r_vals = ";".join(f"{r:.1f}" for r in self.sim.vereda_radius_history)
        for vn in self.sim.vereda_nodes.cpu().numpy():
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="url(#waterGrad)">'
                       f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'</circle>')
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" r="8" fill="#00bbff"/>')

        # ── fruiting ──
        fi = self.sim.fruiting_intensity_history
        f_r  = ";".join(f"{20+30*v:.1f}" for v in fi)
        f_op = ";".join(f"{0.2+0.6*v:.2f}" for v in fi)
        f_c  = ";".join("#fe4db7" if v > 0 else "#4c4c5e" for v in fi)
        f_st = ";".join("#ff007f" if v > 0.5 else "#333333" for v in fi)
        for fn in self.sim.fruiting_nodes.cpu().numpy():
            svg.append(f'<circle cx="{fn[0]}" cy="{fn[1]}">'
                       f'<animate attributeName="r" values="{f_r}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="fill" values="{f_c}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{f_op}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'</circle>')

        # ── title ──
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">'
                   f'ECO-SIM: Insect Swarm Dynamics</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">'
                   f'Temporary Prey Clouds &amp; Insectivore Chase</text>')

        # ── NEW: Insect swarm glowing blobs ──────────────────────────────────
        for sw in self.sim.swarm_render_log:
            cx, cy   = sw["centre"]
            sf, df   = sw["spawn_frame"], sw["die_frame"]
            rad      = sw["radius"]
            # Opacity: appears at spawn_frame, fades near die_frame
            op_vals = []
            for fi_idx in range(frames):
                if fi_idx < sf or fi_idx >= df:
                    op_vals.append("0.0")
                else:
                    age = fi_idx - sf
                    life = df - sf
                    # fade-in first 10 frames, fade-out last 15 frames
                    if age < 10:
                        op_vals.append(f"{age/10.0:.2f}")
                    elif life - age < 15:
                        op_vals.append(f"{(life-age)/15.0:.2f}")
                    else:
                        op_vals.append("1.0")
            op_str = ";".join(op_vals)
            # Outer glow blob (pulsing)
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad*1.4:.1f}" fill="url(#swarmGlow)">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{rad*1.2:.1f};{rad*1.6:.1f};{rad*1.2:.1f}" dur="1.5s" repeatCount="indefinite"/>'
                       f'</circle>')
            # Dense buzzing core
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rad*0.6:.1f}" fill="#ffee44" opacity="0.45">'
                       f'<animate attributeName="opacity" values="{";" .join(["0.0" if v=="0.0" else "0.45" for v in op_vals])}"'
                       f' dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{rad*0.4:.1f};{rad*0.7:.1f};{rad*0.4:.1f}" dur="0.8s" repeatCount="indefinite"/>'
                       f'</circle>')
            # Individual insect dots (6 dots at scattered positions in cloud)
            for ins in range(6):
                import math as _m
                angle = ins * _m.pi / 3.0
                ix = cx + _m.cos(angle) * rad * 0.55
                iy = cy + _m.sin(angle) * rad * 0.55
                ioff = ins * 0.22
                dot_op = ";".join(["0.0" if v == "0.0" else "0.85" for v in op_vals])
                svg.append(
                    f'<circle cx="{ix:.1f}" cy="{iy:.1f}" r="2.5" fill="#ffffcc">'
                    f'<animate attributeName="opacity" values="{dot_op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                    f'<animate attributeName="r" values="1;4;1" dur="0.5s" begin="{ioff:.2f}s" repeatCount="indefinite"/>'
                    f'</circle>'
                )
            # Tiny swarm label
            svg.append(f'<text font-weight="bold" x="{cx:.1f}" y="{cy-rad*1.5:.1f}" text-anchor="middle" '
                       f'font-size="15" fill="#ffdd44">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'Swarm</text>')

        # ── fire nodes ──
        final_fn = self.sim.fire_nodes_history[-1]
        if len(final_fn) > 0:
            birth_f = [-1] * len(final_fn)
            for fi_idx in range(frames):
                ff = self.sim.fire_nodes_history[fi_idx]
                for i in range(len(final_fn)):
                    if birth_f[i] == -1 and len(ff) > i:
                        birth_f[i] = fi_idx
            r_pulse = ";".join(f"{14+math.sin(fi*0.5)*4:.1f}" for fi in range(frames))
            for i, fn in enumerate(final_fn):
                sf = birth_f[i] if birth_f[i] != -1 else frames
                op = ";".join("0.0" if fi < sf else "0.85" for fi in range(frames))
                svg.append(f'<circle cx="{fn[0]:.1f}" cy="{fn[1]:.1f}" fill="#ff4c4c" stroke="#ffaa00" stroke-width="2">'
                           f'<animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="r" values="{r_pulse}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'</circle>')

        # ── seeds ──
        for sd in self.sim.dropped_seeds:
            sp, sf = sd["pos"], sd["frame"]
            ov, cv, rv = [], [], []
            for fi_idx in range(frames):
                if fi_idx < sf:               ov.append("0.0"); cv.append("#ffe43f"); rv.append("3.5")
                elif fi_idx < sf + self.cfg.seed_germination_delay: ov.append("1.0"); cv.append("#ffe43f"); rv.append("3.5")
                else:                          ov.append("1.0"); cv.append("#4caf50"); rv.append("6.0")
            svg.append(f'<circle cx="{sp[0]:.1f}" cy="{sp[1]:.1f}">'
                       f'<animate attributeName="r" values="{";".join(rv)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="fill" values="{";".join(cv)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{";".join(ov)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # ── births ──
        for b in self.sim.birth_events:
            bp, bf = b["pos"], b["frame"]
            ov = ["0.9" if bf <= fi_idx < bf+20 else "0.0" for fi_idx in range(frames)]
            rv = [f"{(fi_idx-bf)*2+5:.1f}" if bf <= fi_idx < bf+20 else "1.0" for fi_idx in range(frames)]
            svg.append(f'<circle cx="{bp[0]:.1f}" cy="{bp[1]:.1f}" fill="none" stroke="{b["color"]}" stroke-width="3">'
                       f'<animate attributeName="r" values="{";".join(rv)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{";".join(ov)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # ── deaths ──
        for d in self.sim.death_events:
            dp, df = d["pos"], d["frame"]
            ov = [f"{max(0.0, 1.0-(fi_idx-df)/30.0):.2f}" if df <= fi_idx < df+30 else "0.0" for fi_idx in range(frames)]
            dc = "#616161" if d["reason"] == "old_age" else "#d32f2f"
            op_str = ";".join(ov)
            svg.append(f'<g>'
                       f'<line x1="{dp[0]-5:.1f}" y1="{dp[1]-5:.1f}" x2="{dp[0]+5:.1f}" y2="{dp[1]+5:.1f}" stroke="{dc}" stroke-width="3">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></line>'
                       f'<line x1="{dp[0]+5:.1f}" y1="{dp[1]-5:.1f}" x2="{dp[0]-5:.1f}" y2="{dp[1]+5:.1f}" stroke="{dc}" stroke-width="3">'
                       f'<animate attributeName="opacity" values="{op_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></line>'
                       f'</g>')

        # ── agent trajectories ──
        for idx in range(self.cfg.max_particles):
            if not any(self.sim.active_history[f][idx] for f in range(frames)):
                continue
            p_color     = self.sim.colors[idx]
            is_frugivore = self.sim.is_frugivore[idx].item()
            is_insect   = self.sim.is_insectivore[idx].item()

            op_vals, seed_op = [], []
            for fi_idx in range(frames):
                act = self.sim.active_history[fi_idx][idx]
                vis = self.sim.visibility_history[fi_idx][idx]
                op_vals.append("1.0" if (act and vis) else "0.0")
                seed_op.append("1.0" if (act and vis and self.sim.carrying_seed_history[fi_idx][idx]) else "0.0")

            px = ";".join(f"{p[idx,0]:.1f}" for p in self.sim.trajectory_history)
            py = ";".join(f"{p[idx,1]:.1f}" for p in self.sim.trajectory_history)

            # Traces for insectivores (chase trails) + sample others
            if idx % 8 == 0 or idx >= self.cfg.initial_particles or is_insect:
                chunks, cur = [], []
                for fi_idx in range(0, frames, 2):
                    alive = self.sim.active_history[fi_idx][idx] and self.sim.visibility_history[fi_idx][idx]
                    if alive: cur.append(self.sim.trajectory_history[fi_idx][idx])
                    elif len(cur) > 1: chunks.append(cur); cur = []
                if len(cur) > 1: chunks.append(cur)

                s_w = 2.5 if is_insect else (2.0 if is_frugivore else 1.0)
                base_op = "0.55" if is_insect else ("0.5" if idx >= self.cfg.initial_particles else "0.25")
                for chunk in chunks:
                    d_path = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in chunk)
                    svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" '
                               f'stroke-opacity="{base_op}" stroke-width="{s_w}"/>')

            # Dot - insectivores get vivid halo + r-pulse
            rad = "6.0" if is_insect else ("5.5" if is_frugivore else "3.5")
            r_begin = f"{(idx % 14) * 0.18:.2f}s"
            r_pulse = "5;8;5" if is_insect else ("4;7;4" if is_frugivore else "3;5.2;3")

            if is_insect or idx % 6 == 0:
                halo_r = "10;18;10" if is_insect else "6;12;6"
                svg.append(f'<circle r="12" fill="{p_color}" opacity="0.30">'
                           f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="opacity" values="{";".join(["0.0" if v=="0.0" else "0.30" for v in op_vals])}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="r" values="{halo_r}" dur="{1.0 if is_insect else 2.0}s" begin="{r_begin}" repeatCount="indefinite"/>'
                           f'</circle>')
            svg.append(f'<circle r="{rad}" fill="{p_color}">'
                       f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{";".join(op_vals)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="r" values="{r_pulse}" dur="{1.2 if is_insect else 2.2}s" begin="{r_begin}" repeatCount="indefinite"/>'
                       f'</circle>')
            if is_frugivore:
                svg.append(f'<circle r="3" fill="#ffe43f">'
                           f'<animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/>'
                           f'<animate attributeName="opacity" values="{";".join(seed_op)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'</circle>')

        # ── educational card ──
        cw, ch = 360, 170
        cx_c = w - cw - 20
        cy_c = h - ch - 20
        svg.append(f'<g transform="translate({cx_c},{cy_c})">'
                   f'<rect width="{cw}" height="{ch}" fill="#1a1a2e" rx="8" ry="8" stroke="#ffcc00" stroke-width="1" opacity="0.95"/>'
                   f'<text x="15" y="25" fill="#cccccc" font-size="15" font-weight="bold">Feature: Insect Swarm Dynamics</text>'
                   f'<text font-weight="bold" x="15" y="48" fill="#cccccc" font-size="15">Periodically a new swarm cloud</text>'
                   f'<text font-weight="bold" x="15" y="66" fill="#cccccc" font-size="15">drifts randomly across the map before dispersing.</text>'
                   f'<text font-weight="bold" x="15" y="88" fill="#cccccc" font-size="15">Insectivore agents (Gralha, Beija-flor)  detect</text>'
                   f'<text font-weight="bold" x="15" y="106" fill="#cccccc" font-size="15">swarms within {self.cfg.insect_chase_radius:.0f} units and aggressively chase.</text>'
                   f'<text font-weight="bold" x="15" y="126" fill="#cccccc" font-size="15">Successfully entering a cloud grants +{self.cfg.insect_energy_gain:.0f} energy</text>'
                   f'<text font-weight="bold" x="15" y="146" fill="#cccccc" font-size="15">per step, accelerating their reproduction cycle.</text>'
                   f'</g>')

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

    drive_folder = "/content/drive/MyDrive/ReservaAraras_SVGs"
    save_dir = drive_folder if os.path.isdir('/content/drive') else os.path.join(
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd(), 'svg_output')
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f'{notebook_id}.svg')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    print(f"SVG saved -> {filepath}")
    return filepath

def main():
    """Function `main` -- simulation component."""

    print(f"Initializing Insect Swarm simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)
    print(f"Simulation complete. Swarms spawned: {len(sim.swarm_render_log)}. "
          f"Final pop: {sim.population_history[-1]}. Generating SVG...")
    renderer    = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_24')

if __name__ == "__main__":
    main()
