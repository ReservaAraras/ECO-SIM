# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 33: Fencing (Hard boundaries for ground feeders; permeable for high flyers)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Fencing (Hard boundaries for ground feeders; permeable for high flyers) emphasizing succession acceleration.
- Indicator species: Macauba (Acrocomia aculeata).
- Pollination lens: butterfly host-plant coupling.
- Human impact lens: pesticide drift from farms.

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


import os, torch, numpy as np, random, math
from IPython.display import display, HTML # pyre-ignore[21]
from eco_base import CANVAS_HEIGHT, ZONES
from dataclasses import dataclass
from typing import List, Dict, Optional

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)":             {"speed": 4.5, "color": "#fe4db7", "weight": 0.20, "diet": "Frugivore",   "cohesion": 0.05, "drag": 0.85, "max_turn": 0.10, "lifespan_base": 300, "flies": True},
    "Gralha-do-campo (Cyanocorax cristatellus)":  {"speed": 5.8, "color": "#00ffdc", "weight": 0.15, "diet": "Insectivore", "cohesion": 0.08, "drag": 0.90, "max_turn": 0.15, "lifespan_base": 250, "flies": True},
    "Beija-flor-tesoura (Eupetomena macroura)":   {"speed": 6.2, "color": "#ffe43f", "weight": 0.12, "diet": "Insectivore", "cohesion": 0.01, "drag": 0.98, "max_turn": 0.40, "lifespan_base": 180, "flies": True},
    "Gavião-carijó (Rupornis magnirostris)":      {"speed": 7.0, "color": "#f44336", "weight": 0.10, "diet": "Carnivore",   "cohesion": 0.00, "drag": 0.95, "max_turn": 0.20, "lifespan_base": 350, "flies": True},
    "Urubu-rei (Sarcoramphus papa)":              {"speed": 5.5, "color": "#b39ddb", "weight": 0.13, "diet": "Scavenger",   "cohesion": 0.02, "drag": 0.92, "max_turn": 0.08, "lifespan_base": 400, "flies": True},
    "Andorinha-do-rio (Tachycineta albiventer)":  {"speed": 8.5, "color": "#4fc3f7", "weight": 0.15, "diet": "Migrant",     "cohesion": 0.15, "drag": 0.98, "max_turn": 0.05, "lifespan_base": 600, "flies": True},
    "Gado Nelore (Bos indicus)":                  {"speed": 1.2, "color": "#8d6e63", "weight": 0.15, "diet": "Grazer",      "cohesion": 0.35, "drag": 0.80, "max_turn": 0.02, "lifespan_base": 800, "flies": False},
}

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width:  int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 520
    fps:    int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    initial_particles: int = 240
    max_particles:     int = 450
    carrying_capacity: int = 350
    dt: float = 0.5
    vereda_min_radius: float = 20.0; vereda_max_radius: float = 90.0
    vereda_cycle_frames: float = 200.0; vereda_base_attraction: float = 3.0
    fruiting_cycle_frames: float = 150.0; fruiting_radius: float = 50.0
    fruiting_base_attraction: float = 4.0
    seed_carry_duration: float = 60.0; seed_pickup_chance: float = 0.05
    sand_zone_x: float = 640.0; sand_speed_modifier: float = 0.6
    fire_start_frame: int = 80; fire_spread_prob: float = 0.08
    fire_flee_radius: float = 150.0; fire_flee_force: float = 12.0; max_fire_nodes: int = 120
    mating_energy_threshold: float = 85.0; mating_radius: float = 30.0
    energy_decay: float = 0.12; energy_gain_fruiting: float = 6.0; energy_gain_vereda: float = 2.5
    lifespan_variance: float = 50.0
    swarm_spawn_interval: int = 45; swarm_lifetime: int = 55; swarm_radius: float = 50.0
    swarm_drift: float = 0.3; insect_chase_radius: float = 200.0; insect_chase_force: float = 8.0
    insect_energy_gain: float = 12.0
    carrion_circle_radius: float = 60.0; carrion_linger_frames: int = 120; scavenger_energy_gain: float = 8.0
    num_nest_nodes: int = 6; nest_attract_radius: float = 180.0; nest_attract_force: float = 5.0
    nest_energy_bonus: float = 1.5; homeless_restless_force: float = 2.5
    keystone_removal_frame: int = 230; flock_cohesion_radius: float = 80.0; flock_cohesion_multiplier: float = 6.0
    hunting_chase_radius: float = 160.0; hunting_capture_radius: float = 12.0
    hunting_chase_force: float = 14.0; predator_energy_gain: float = 40.0
    satiation_duration: int = 90; satiation_speed_penalty: float = 0.4
    alarm_propagation_radius: float = 90.0; alarm_decay_rate: float = 0.05
    alarm_panic_multiplier: float = 1.8; alarm_panic_force: float = 8.0
    migration_start_frame: int = 40; migration_spawn_rate: float = 0.8
    migration_flow_force: float = 6.0; migration_span_y_min: float = 150.0; migration_span_y_max: float = 450.0
    road_x_center: float = 780.0; road_width: float = 20.0
    road_fear_radius: float = 60.0; road_fear_force_base: float = 15.0; road_crossing_courage: float = 0.05
    cattle_graze_radius: float = 45.0; cattle_resource_depletion_rate: float = 0.01
    cattle_chase_vereda_force: float = 2.0; cattle_graze_gain: float = 0.8
    # NEW: Fencing
    fence_y_positions: tuple = (180.0, 420.0)   # Two horizontal fences
    fence_fear_radius: float = 18.0              # Hard physical buffer
    fence_repulsion_force: float = 22.0          # Very strong push for ground animals

CONFIG = SimulationConfig()

class TerraRoncaEcosystem:
    """Class `TerraRoncaEcosystem` -- simulation component."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        self.pos = torch.rand((cfg.max_particles, 2), device=self.dev) * torch.tensor([cfg.width, cfg.height], device=self.dev)
        on_road = (self.pos[:, 0] > cfg.road_x_center - cfg.road_width/2) & (self.pos[:, 0] < cfg.road_x_center + cfg.road_width/2)
        if on_road.any(): self.pos[on_road, 0] -= cfg.road_width * 2
        self.vel = (torch.rand((cfg.max_particles, 2), device=self.dev) - 0.5) * 10.0

        self.guilds = list(BIODIVERSITY_DB.keys())
        weights     = [BIODIVERSITY_DB[g]["weight"] for g in self.guilds]
        indices     = np.random.choice(len(self.guilds), size=cfg.max_particles, p=weights)

        self.species_id     = torch.tensor(indices, device=self.dev)
        self.speeds         = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["speed"]    for i in indices], device=self.dev)
        self.drags          = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["drag"]     for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns      = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["max_turn"] for i in indices], device=self.dev)
        self.cohesions      = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["cohesion"] for i in indices], device=self.dev)
        self.colors         = [BIODIVERSITY_DB[self.guilds[i]]["color"] for i in indices]
        self.is_frugivore   = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Frugivore"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_insectivore = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Insectivore" for i in indices], device=self.dev, dtype=torch.bool)
        self.is_scavenger   = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Scavenger"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_carnivore   = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Carnivore"   for i in indices], device=self.dev, dtype=torch.bool)
        self.is_migrant     = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Migrant"     for i in indices], device=self.dev, dtype=torch.bool)
        self.is_grazer      = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["diet"] == "Grazer"      for i in indices], device=self.dev, dtype=torch.bool)
        # NEW: flies flag
        self.flies          = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["flies"]                 for i in indices], device=self.dev, dtype=torch.bool)

        self.is_male   = torch.rand(cfg.max_particles, device=self.dev) > 0.5
        self.courage   = torch.rand(cfg.max_particles, device=self.dev)

        self.is_active = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.is_active[:cfg.initial_particles] = True
        self.energy    = torch.ones(cfg.max_particles, device=self.dev) * 80.0
        self.age       = torch.zeros(cfg.max_particles, device=self.dev)
        self.age[:cfg.initial_particles] = torch.rand(cfg.initial_particles, device=self.dev) * 200.0
        base_ls        = torch.tensor([BIODIVERSITY_DB[self.guilds[i]]["lifespan_base"] for i in indices], device=self.dev, dtype=torch.float32)
        self.lifespan  = base_ls + torch.randn(cfg.max_particles, device=self.dev) * cfg.lifespan_variance
        self.is_underground = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.satiation_timers = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.int32)
        self.alarm_level   = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.float32)
        self.alarm_vectors = torch.zeros((cfg.max_particles, 2), device=self.dev, dtype=torch.float32)

        nest_xs = [160.0, 420.0, 680.0, 820.0, 1050.0, 320.0]; nest_ys = [120.0, 480.0, 200.0, 430.0, 300.0, 330.0]
        self.nest_nodes    = torch.tensor(list(zip(nest_xs, nest_ys)), device=self.dev, dtype=torch.float32)
        self.nest_occupant = [-1] * cfg.num_nest_nodes
        self.particle_nest = torch.full((cfg.max_particles,), -1, device=self.dev, dtype=torch.long)

        self.nectar_nodes   = torch.tensor([[250.0, 300.0]], device=self.dev)
        self.vereda_nodes   = torch.tensor([[600.0, 350.0], [900.0, 450.0]], device=self.dev)
        self.vereda_health  = torch.ones(len(self.vereda_nodes), device=self.dev, dtype=torch.float32)
        self.fruiting_nodes = torch.tensor([[300.0, 150.0], [850.0, 250.0]], device=self.dev)

        self.has_seed    = torch.zeros(cfg.max_particles, device=self.dev, dtype=torch.bool)
        self.seed_timers = torch.zeros(cfg.max_particles, device=self.dev)
        self.dropped_seeds: List[Dict] = []
        self.fire_nodes  = torch.zeros((0, 2), device=self.dev)
        self.swarms: List[Dict] = []; self.carrion_sites: List[Dict] = []
        self.keystone_event: Optional[Dict] = None
        self.migrants_spawned = 0; self.migrants_exited = 0

        self.swarm_render_log = []; self.trajectory_history = []; self.visibility_history = []
        self.active_history = []; self.fire_nodes_history = []; self.vereda_radius_history = []
        self.fruiting_intensity_history = []; self.population_history = []; self.nest_state_history = []
        self.cohesion_stats = []; self.satiation_history = []; self.alarm_history = []; self.vereda_health_history = []
        self.birth_events = []; self.death_events = []; self.alarm_events = []
        self.migration_spawn_events = []; self.migration_exit_events = []

    def _spawn_migrant(self, frame_idx):
        free = (~self.is_active).nonzero().squeeze(1)
        if len(free) == 0: return
        ni  = free[0]; ms = "Andorinha-do-rio (Tachycineta albiventer)"; mid = self.guilds.index(ms)
        self.species_id[ni]=mid; self.speeds[ni]=BIODIVERSITY_DB[ms]["speed"]; self.drags[ni]=BIODIVERSITY_DB[ms]["drag"]
        self.max_turns[ni]=BIODIVERSITY_DB[ms]["max_turn"]; self.cohesions[ni]=BIODIVERSITY_DB[ms]["cohesion"]
        self.colors[ni]=BIODIVERSITY_DB[ms]["color"]
        self.is_frugivore[ni]=False; self.is_insectivore[ni]=False; self.is_scavenger[ni]=False
        self.is_carnivore[ni]=False; self.is_migrant[ni]=True; self.is_grazer[ni]=False; self.flies[ni]=True
        self.is_active[ni]=True; y=random.uniform(self.cfg.migration_span_y_min, self.cfg.migration_span_y_max)
        self.pos[ni]=torch.tensor([-20.0, y], device=self.dev); self.vel[ni]=torch.tensor([8.0, 0.0], device=self.dev)
        self.energy[ni]=100.0; self.age[ni]=0.0; self.alarm_level[ni]=0.0
        self.is_underground[ni]=False; self.has_seed[ni]=False; self.courage[ni]=1.0
        self.migrants_spawned += 1

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel     = self.vel.clone()
        active_mask = self.is_active.clone()
        surface_mask = active_mask & ~self.is_underground

        # Keystone removal
        if frame_idx == self.cfg.keystone_removal_frame:
            lm = active_mask & ~self.is_migrant & ~self.is_grazer
            if lm.any():
                counts = torch.bincount(self.species_id[lm])
                kid = torch.argmax(counts).item()
                tm = (self.species_id == kid) & active_mask
                self.is_active[tm]=False; self.has_seed[tm]=False; self.alarm_level[tm]=0.0
                for idx in tm.nonzero().squeeze(1):
                    n=int(self.particle_nest[idx])
                    if n>=0: self.nest_occupant[n]=-1; self.particle_nest[idx]=-1
                    pn=self.pos[idx].cpu().numpy().copy()
                    self.death_events.append({"pos":pn,"frame":frame_idx,"color":self.colors[idx],"reason":"extinction"})
                    self.carrion_sites.append({"pos":pn,"spawn_frame":frame_idx,"expire_frame":frame_idx+self.cfg.carrion_linger_frames})
                self.keystone_event={"frame":frame_idx,"species":self.guilds[kid],"count":int(tm.sum())}
            active_mask=self.is_active.clone(); surface_mask=active_mask & ~self.is_underground

        self.age[active_mask]+=1.0; self.energy[active_mask]-=self.cfg.energy_decay
        self.satiation_timers[self.satiation_timers>0]-=1
        self.alarm_level=torch.clamp(self.alarm_level-self.cfg.alarm_decay_rate, min=0.0)

        # Migration spawn/exit
        if frame_idx >= self.cfg.migration_start_frame and random.random() < self.cfg.migration_spawn_rate:
            self._spawn_migrant(frame_idx)
        active_mask=self.is_active.clone(); surface_mask=active_mask & ~self.is_underground
        mig_act = active_mask & self.is_migrant
        if mig_act.any():
            ex = mig_act & (self.pos[:,0] > self.cfg.width+10.0)
            if ex.any():
                for idx in ex.nonzero().squeeze(1): self.is_active[idx]=False; self.migrants_exited+=1
        active_mask=self.is_active.clone(); surface_mask=active_mask & ~self.is_underground

        # Mortality
        dead = (active_mask & (self.energy<=0.0)) | (active_mask & (self.age>=self.lifespan))
        if dead.any():
            for idx in dead.nonzero().squeeze(1):
                self.is_active[idx]=False; self.has_seed[idx]=False; self.alarm_level[idx]=0.0
                n=int(self.particle_nest[idx])
                if n>=0: self.nest_occupant[n]=-1; self.particle_nest[idx]=-1
                pn=self.pos[idx].cpu().numpy().copy()
                self.death_events.append({"pos":pn,"frame":frame_idx,"color":self.colors[idx],"reason":"old_age" if self.age[idx]>=self.lifespan[idx] else "starvation"})
                self.carrion_sites.append({"pos":pn,"spawn_frame":frame_idx,"expire_frame":frame_idx+self.cfg.carrion_linger_frames})

        active_mask=self.is_active.clone(); surface_mask=active_mask & ~self.is_underground
        current_pop=int(active_mask.sum().item())
        if not active_mask.any():
            self._log(current_pop, frame_idx); return

        self.vereda_health=torch.clamp(self.vereda_health+0.002, 0.0, 1.0)
        self.vereda_health_history.append(self.vereda_health.cpu().numpy().copy())

        # Flocking
        flock_attraction=torch.zeros_like(self.vel)
        if surface_mask.any():
            sidx=surface_mask.nonzero().squeeze(1); dm=torch.cdist(self.pos[surface_mask],self.pos[surface_mask])
            ir=dm<self.cfg.flock_cohesion_radius; ir.fill_diagonal_(False); nc=ir.sum(dim=1); hn=nc>0
            if hn.any():
                sp2=torch.matmul(ir.float(),self.pos[surface_mask]); ap=sp2[hn]/nc[hn].unsqueeze(1).float()
                pull=ap-self.pos[surface_mask][hn]; pm=self.cohesions[sidx[hn]].unsqueeze(1)*self.cfg.flock_cohesion_multiplier
                flock_attraction[sidx[hn]]=(pull/pull.norm(dim=1,keepdim=True).clamp(min=1.0))*pm
            self.cohesion_stats.append(nc.float().mean().item() if len(nc)>0 else 0.0)
        else: self.cohesion_stats.append(0.0)

        # ── NEW: FENCE REPULSION (only non-flyers) ────────────────────────────
        fence_repulsion = torch.zeros_like(self.vel)
        ground_mask = surface_mask & ~self.flies   # Only ground-bound animals feel the fence
        if ground_mask.any():
            for fy in self.cfg.fence_y_positions:
                dist_y = torch.abs(self.pos[:, 1] - fy)
                near_fence = ground_mask & (dist_y < self.cfg.fence_fear_radius)
                if near_fence.any():
                    direction = torch.sign(self.pos[near_fence, 1] - fy)
                    strength  = (1.0 - dist_y[near_fence] / self.cfg.fence_fear_radius) ** 2
                    fence_repulsion[near_fence, 1] += direction * strength * self.cfg.fence_repulsion_force

        # Road repulsion
        road_repulsion = torch.zeros_like(self.vel)
        dist_to_road = torch.abs(self.pos[:,0]-self.cfg.road_x_center)
        scared = surface_mask & (dist_to_road<self.cfg.road_fear_radius) & (self.courage<(1.0-self.cfg.road_crossing_courage)) & ~self.is_grazer
        if scared.any():
            daw=torch.sign(self.pos[scared,0]-self.cfg.road_x_center)
            pm2=(1.0-(dist_to_road[scared]/self.cfg.road_fear_radius))**2
            road_repulsion[scared,0]=daw*pm2*self.cfg.road_fear_force_base

        # Hunting
        hunting_attraction=torch.zeros_like(self.vel)
        pred_m=surface_mask & self.is_carnivore; prey_m=surface_mask & (self.is_frugivore|self.is_insectivore|self.is_migrant)
        hungry=pred_m & (self.satiation_timers<=0)
        if hungry.any() and prey_m.any():
            pi2=hungry.nonzero().squeeze(1); qi=prey_m.nonzero().squeeze(1)
            dh=torch.cdist(self.pos[hungry],self.pos[prey_m]); md,ti=torch.min(dh,dim=1)
            chase=md<self.cfg.hunting_chase_radius
            if chase.any():
                tgt=self.pos[qi[ti[chase]]]; hnt=self.pos[pi2[chase]]; pull2=tgt-hnt
                hunting_attraction[pi2[chase]]=(pull2/md[chase].unsqueeze(1).clamp(min=1.0))*self.cfg.hunting_chase_force
                self.alarm_level[qi[ti[chase]]]=1.0; self.alarm_vectors[qi[ti[chase]]]=(tgt-hnt)/md[chase].unsqueeze(1).clamp(min=1.0)
                cap=md[chase]<self.cfg.hunting_capture_radius
                if cap.any():
                    ch2=pi2[chase][cap]; cp2=qi[ti[chase][cap]]
                    for k in range(len(ch2)):
                        hid=ch2[k]; pid=cp2[k]
                        if self.is_active[pid]:
                            self.is_active[pid]=False; self.has_seed[pid]=False; self.alarm_level[pid]=0.0
                            self.energy[hid]+=self.cfg.predator_energy_gain; self.satiation_timers[hid]=self.cfg.satiation_duration
                            pn=self.pos[pid].cpu().numpy().copy()
                            self.death_events.append({"pos":pn,"frame":frame_idx,"color":self.colors[pid],"reason":"predation"})
                            self.carrion_sites.append({"pos":pn,"spawn_frame":frame_idx,"expire_frame":frame_idx+self.cfg.carrion_linger_frames})
                            current_pop-=1
        self.energy.clamp_(0.0, 100.0)

        # Environmental
        d_nec=torch.cdist(self.pos,self.nectar_nodes); mn_nec,cl_nec=torch.min(d_nec,dim=1)
        karst_attraction=(self.nectar_nodes[cl_nec]-self.pos)/mn_nec.unsqueeze(1).clamp(min=1.0)

        cycp=(frame_idx%self.cfg.vereda_cycle_frames)/self.cfg.vereda_cycle_frames
        wlvl=(math.sin(cycp*2*math.pi-math.pi/2)+1.0)/2.0
        cvrad=self.cfg.vereda_min_radius+(self.cfg.vereda_max_radius-self.cfg.vereda_min_radius)*wlvl
        self.vereda_radius_history.append(cvrad)
        d_ver=torch.cdist(self.pos,self.vereda_nodes); mn_ver,cl_ver=torch.min(d_ver,dim=1)
        hmod=self.vereda_health[cl_ver]
        vereda_attraction=(self.vereda_nodes[cl_ver]-self.pos)/mn_ver.unsqueeze(1).clamp(min=1.0)*(self.cfg.vereda_base_attraction*wlvl*hmod).unsqueeze(1)
        flooded=mn_ver<cvrad; vereda_attraction[flooded]*=0.1
        if (flooded & surface_mask & ~self.is_grazer).any():
            self.energy[flooded & surface_mask & ~self.is_grazer]+=self.cfg.energy_gain_vereda*hmod[flooded & surface_mask & ~self.is_grazer]
        gz=surface_mask & self.is_grazer
        if gz.any():
            pv=self.vereda_nodes[cl_ver[gz]]-self.pos[gz]; vereda_attraction[gz]+=(pv/mn_ver[gz].unsqueeze(1).clamp(min=1.0))*self.cfg.cattle_chase_vereda_force
            eating=gz & (mn_ver<self.cfg.cattle_graze_radius)
            if eating.any():
                self.energy[eating]+=self.cfg.cattle_graze_gain
                for ni in cl_ver[eating]: self.vereda_health[ni]-=self.cfg.cattle_resource_depletion_rate
                self.vereda_health=torch.clamp(self.vereda_health, 0.0, 1.0)

        fic=(frame_idx%self.cfg.fruiting_cycle_frames)/self.cfg.fruiting_cycle_frames
        fi_int=max(0.0,math.sin(fic*2*math.pi)); self.fruiting_intensity_history.append(fi_int)
        d_fr=torch.cdist(self.pos,self.fruiting_nodes); mn_fr,cl_fr=torch.min(d_fr,dim=1)
        fruiting_attraction=torch.zeros_like(self.vel); feeding_mask=torch.zeros(self.cfg.max_particles,device=self.dev,dtype=torch.bool)
        if fi_int>0.0:
            smell=(mn_fr<200.0)&self.is_frugivore&surface_mask
            if smell.any():
                fruiting_attraction[smell]=(self.fruiting_nodes[cl_fr[smell]]-self.pos[smell])/mn_fr[smell].unsqueeze(1).clamp(min=1.0)*(self.cfg.fruiting_base_attraction*fi_int)
                forage=smell&(mn_fr<self.cfg.fruiting_radius); feeding_mask=forage&(fi_int>0.1)
                if forage.any(): fruiting_attraction[forage]+=torch.randn_like(fruiting_attraction[forage])*2.0; self.energy[forage]+=self.cfg.energy_gain_fruiting*fi_int

        # Seeds
        if feeding_mask.any():
            ok=feeding_mask & ~self.has_seed & (torch.rand(self.cfg.max_particles,device=self.dev)<self.cfg.seed_pickup_chance)
            if ok.any(): self.has_seed[ok]=True; self.seed_timers[ok]=0.0
        self.seed_timers[self.has_seed]+=1.0
        drop=self.has_seed&(self.seed_timers>=self.cfg.seed_carry_duration)&surface_mask
        if drop.any():
            self.has_seed[drop]=False
            for p in self.pos[drop]: self.dropped_seeds.append({"pos":p.cpu().numpy().copy(),"frame":frame_idx})

        # Fire
        fire_repulsion=torch.zeros_like(self.vel); flee_mask=torch.zeros(self.cfg.max_particles,device=self.dev,dtype=torch.bool)
        if frame_idx>=self.cfg.fire_start_frame:
            if len(self.fire_nodes)==0: self.fire_nodes=torch.tensor([[1000.0,300.0]],device=self.dev)
            elif frame_idx%2==0 and len(self.fire_nodes)<self.cfg.max_fire_nodes:
                sp=torch.rand(len(self.fire_nodes),device=self.dev)<self.cfg.fire_spread_prob
                if sp.any():
                    ns=self.fire_nodes[sp]+torch.randn((int(sp.sum()),2),device=self.dev)*30.0
                    ns[:,0]=ns[:,0].clamp(0,self.cfg.width); ns[:,1]=ns[:,1].clamp(0,self.cfg.height); self.fire_nodes=torch.cat([self.fire_nodes,ns],dim=0)
            if len(self.fire_nodes)>0:
                df=torch.cdist(self.pos,self.fire_nodes); mn_f,cl_f=torch.min(df,dim=1); flee_mask=(mn_f<self.cfg.fire_flee_radius)&surface_mask
                if flee_mask.any():
                    fv=self.pos[flee_mask]-self.fire_nodes[cl_f[flee_mask]]
                    fire_repulsion[flee_mask]=(fv/mn_f[flee_mask].unsqueeze(1).clamp(min=1.0)**1.5)*self.cfg.fire_flee_force*30.0
                    self.alarm_level[flee_mask]=1.0; self.alarm_vectors[flee_mask]=fv/mn_f[flee_mask].unsqueeze(1).clamp(min=1.0)
        self.fire_nodes_history.append(self.fire_nodes.cpu().numpy().copy())

        # Alarm wave
        alarm_repulsion=torch.zeros_like(self.vel)
        ial=(self.alarm_level>0.5)&surface_mask; nal=(self.alarm_level<=0.5)&surface_mask
        if ial.any() and nal.any():
            da2=torch.cdist(self.pos[nal],self.pos[ial]); md2,ti2=torch.min(da2,dim=1); inf=md2<self.cfg.alarm_propagation_radius
            if inf.any():
                ig=nal.nonzero().squeeze(1)[inf]; sg=ial.nonzero().squeeze(1)[ti2[inf]]; self.alarm_level[ig]=1.0
                cv2=(self.alarm_vectors[sg]+(self.pos[ig]-self.pos[sg])/md2[inf].unsqueeze(1).clamp(min=1.0))/2.0
                self.alarm_vectors[ig]=cv2/cv2.norm(dim=1,keepdim=True).clamp(min=1e-5)
                for idx in ig:
                    if random.random()<0.2: self.alarm_events.append({"pos":self.pos[idx].cpu().numpy().copy(),"frame":frame_idx})
        aal=(self.alarm_level>0.1)&surface_mask&~flee_mask&~self.is_carnivore
        if aal.any(): alarm_repulsion[aal]=self.alarm_vectors[aal]*self.cfg.alarm_panic_force*self.alarm_level[aal].unsqueeze(1)

        # Swarms
        if frame_idx%self.cfg.swarm_spawn_interval==0:
            cx,cy=random.uniform(100,self.cfg.width-100),random.uniform(80,self.cfg.height-80)
            self.swarms.append({"centre":torch.tensor([cx,cy],device=self.dev),"drift":(torch.rand(2,device=self.dev)-0.5)*self.cfg.swarm_drift*2.0,"spawn_frame":frame_idx,"die_frame":frame_idx+self.cfg.swarm_lifetime})
            self.swarm_render_log.append({"centre":np.array([cx,cy]),"spawn_frame":frame_idx,"die_frame":frame_idx+self.cfg.swarm_lifetime,"radius":self.cfg.swarm_radius})
        insect_attraction=torch.zeros_like(self.vel); lsw=[]
        for sw in self.swarms:
            if frame_idx>=sw["die_frame"]: continue
            sw["centre"]+=sw["drift"]; sw["centre"][0]=sw["centre"][0].clamp(50,self.cfg.width-50); sw["centre"][1]=sw["centre"][1].clamp(50,self.cfg.height-50); lsw.append(sw)
            dsw=torch.norm(self.pos-sw["centre"],dim=1)
            ch3=(dsw<self.cfg.insect_chase_radius)&self.is_insectivore&surface_mask&~flee_mask&(self.alarm_level<0.5)
            if ch3.any(): insect_attraction[ch3]+=(sw["centre"]-self.pos[ch3])/dsw[ch3].unsqueeze(1).clamp(min=1.0)*self.cfg.insect_chase_force
            fsw=(dsw<self.cfg.swarm_radius)&self.is_insectivore&surface_mask
            if fsw.any(): self.energy[fsw]+=self.cfg.insect_energy_gain
        self.swarms=lsw

        # Scavenging
        scav_attraction=torch.zeros_like(self.vel)
        self.carrion_sites=[c for c in self.carrion_sites if c["expire_frame"]>frame_idx]
        if self.carrion_sites and self.is_scavenger.any():
            cp3=torch.tensor([c["pos"] for c in self.carrion_sites],device=self.dev,dtype=torch.float32)
            dc=torch.cdist(self.pos,cp3); mc,cc=torch.min(dc,dim=1)
            ss=self.is_scavenger&surface_mask&~flee_mask
            if ss.any():
                cl2=cp3[cc[ss]]; ds2=mc[ss]; si2=ss.nonzero().squeeze(1); farp=ds2>self.cfg.carrion_circle_radius
                if farp.any(): scav_attraction[si2[farp]]=(cl2[farp]-self.pos[si2[farp]])/ds2[farp].unsqueeze(1).clamp(min=1.0)*6.0
                np2=~farp
                if np2.any():
                    tc2=self.pos[si2[np2]]-cl2[np2]; tg=torch.stack([-tc2[:,1],tc2[:,0]],dim=1)
                    scav_attraction[si2[np2]]=tg/tg.norm(dim=1,keepdim=True).clamp(min=1e-5)*5.0; self.energy[si2[np2]]+=self.cfg.scavenger_energy_gain

        # Nesting
        nest_attraction=torch.zeros_like(self.vel)
        for ni in range(self.cfg.num_nest_nodes):
            occ=self.nest_occupant[ni]
            if occ>=0 and not self.is_active[occ]: self.nest_occupant[ni]=-1; self.particle_nest[occ]=-1
        hm=self.particle_nest>=0; homeless=active_mask&~hm&(self.alarm_level<0.5)&~self.is_migrant&~self.is_grazer
        nested=active_mask&hm
        if nested.any():
            self.energy[nested]+=self.cfg.nest_energy_bonus; ni2=self.particle_nest[nested]; np3=self.nest_nodes[ni2]
            d2=torch.norm(self.pos[nested]-np3,dim=1); nest_attraction[nested]=(np3-self.pos[nested])/d2.unsqueeze(1).clamp(min=1.0)*2.0
        fn2=[ni for ni in range(self.cfg.num_nest_nodes) if self.nest_occupant[ni]==-1]
        if fn2 and homeless.any():
            fnp=self.nest_nodes[torch.tensor(fn2,device=self.dev)]; df2=torch.cdist(self.pos[homeless],fnp)
            mf2,bf2=torch.min(df2,dim=1); hli=homeless.nonzero().squeeze(1); ce=mf2<self.cfg.nest_attract_radius
            if ce.any():
                tgt2=fnp[bf2[ce]]; dt2=mf2[ce]
                nest_attraction[hli[ce]]=(tgt2-self.pos[hli[ce]])/dt2.unsqueeze(1).clamp(min=1.0)*self.cfg.nest_attract_force
                arr=dt2<20.0
                if arr.any():
                    for kl,kg in enumerate(hli[ce][arr]):
                        bn=bf2[ce][arr][kl].item(); rn=fn2[bn]
                        if self.nest_occupant[rn]==-1: self.nest_occupant[rn]=int(kg); self.particle_nest[int(kg)]=rn
        if homeless.any(): nest_attraction[homeless]+=torch.randn(int(homeless.sum()),2,device=self.dev)*self.cfg.homeless_restless_force

        # Migration flow
        mig_attraction=torch.zeros_like(self.vel)
        am2=active_mask&self.is_migrant&~flee_mask&(self.alarm_level<0.5)
        if am2.any(): mig_attraction[am2]=torch.tensor([self.cfg.migration_flow_force,0.0],device=self.dev)

        # Reproduction
        if current_pop<self.cfg.carrying_capacity:
            rdy=surface_mask&~flee_mask&(self.energy>self.cfg.mating_energy_threshold)&(self.alarm_level<0.5)&~self.is_migrant
            if rdy.any():
                for i in rdy.nonzero().squeeze(1):
                    if self.energy[i]<self.cfg.mating_energy_threshold: continue
                    ai2=active_mask.nonzero().squeeze(1); da3=torch.norm(self.pos[ai2]-self.pos[i],dim=1)
                    for j in ai2[(da3<self.cfg.mating_radius)&(da3>0.1)]:
                        if self.energy[j]>self.cfg.mating_energy_threshold and self.is_male[i]!=self.is_male[j] and self.species_id[i]==self.species_id[j] and not flee_mask[j]:
                            fr3=(~self.is_active).nonzero().squeeze(1)
                            if len(fr3):
                                ci=fr3[0]; self.is_active[ci]=True; self.pos[ci]=self.pos[i]+torch.randn(2,device=self.dev)*10.0
                                self.vel[ci]=self.vel[i]*-1.0; self.energy[ci]=40.0; self.age[ci]=0.0; self.alarm_level[ci]=0.0
                                self.particle_nest[ci]=-1; self.is_underground[ci]=False; self.courage[ci]=torch.rand(1,device=self.dev).item()
                                self.flies[ci]=self.flies[i]  # Inherit flight ability
                                self.energy[i]-=45.0; self.energy[j]-=45.0
                                self.birth_events.append({"pos":self.pos[ci].cpu().numpy().copy(),"frame":frame_idx,"color":self.colors[ci]})
                                current_pop+=1
                            break
                    if current_pop>=self.cfg.carrying_capacity: break

        active_mask=self.is_active.clone(); surface_mask=active_mask&~self.is_underground
        raw=torch.zeros_like(self.vel)
        if surface_mask.any():
            raw[surface_mask]=(
                self.vel[surface_mask]*self.drags[surface_mask]
                + karst_attraction[surface_mask]
                + vereda_attraction[surface_mask]
                + fruiting_attraction[surface_mask]
                + fire_repulsion[surface_mask]
                + insect_attraction[surface_mask]
                + scav_attraction[surface_mask]
                + nest_attraction[surface_mask]
                + flock_attraction[surface_mask]
                + hunting_attraction[surface_mask]
                + alarm_repulsion[surface_mask]
                + mig_attraction[surface_mask]
                + road_repulsion[surface_mask]
                + fence_repulsion[surface_mask]
                + torch.randn_like(self.vel[surface_mask])*0.5
            )

        na=torch.atan2(raw[:,1],raw[:,0]); oa=torch.atan2(old_vel[:,1],old_vel[:,0]); fa=oa.clone()
        if surface_mask.any():
            dt=self.max_turns[surface_mask].clone()
            ip=(self.alarm_level>0.5)&surface_mask
            if ip.any(): dt[ip[surface_mask]]*=2.5
            if flee_mask.any(): dt[flee_mask[surface_mask]]*=3.0
            mg=surface_mask&self.is_migrant
            if mg.any(): dt[mg[surface_mask]]*=0.1
            if scared.any(): dt[scared[surface_mask]]*=4.0
            if ground_mask.any():
                near_any_fence = torch.zeros(self.cfg.max_particles, device=self.dev, dtype=torch.bool)
                for fy in self.cfg.fence_y_positions:
                    near_any_fence |= (torch.abs(self.pos[:,1]-fy) < self.cfg.fence_fear_radius*1.5)
                gf = ground_mask & near_any_fence
                if gf.any(): dt[gf[surface_mask]] *= 5.0
            diff=((na-oa+math.pi)%(2*math.pi)-math.pi)[surface_mask]
            fa[surface_mask]=oa[surface_mask]+torch.clamp(diff,-dt,dt)

        spd=torch.norm(raw,dim=1).clamp(min=0.1)
        self.vel[:,0]=torch.cos(fa)*spd; self.vel[:,1]=torch.sin(fa)*spd
        nrm=torch.norm(self.vel,dim=1,keepdim=True).clamp(min=1e-5)
        ds=self.speeds.clone()
        ds[flooded&surface_mask]*=0.7
        if feeding_mask.any(): ds[feeding_mask]*=0.4
        if ip.any(): ds[ip]*=self.cfg.alarm_panic_multiplier
        sat=(self.satiation_timers>0)&surface_mask
        if sat.any(): ds[sat]*=self.cfg.satiation_speed_penalty
        ins=(self.pos[:,0]<self.cfg.sand_zone_x)&surface_mask
        if ins.any(): ds[ins]*=self.cfg.sand_speed_modifier
        if flee_mask.any(): ds[flee_mask]*=1.5
        self.vel=(self.vel/nrm)*ds.unsqueeze(1)

        self.pos[active_mask]+=self.vel[active_mask]*self.cfg.dt
        wm=surface_mask&~self.is_migrant
        if wm.any():
            self.pos[wm,0]=self.pos[wm,0]%self.cfg.width
            self.pos[wm,1]=self.pos[wm,1]%self.cfg.height
        self._log(current_pop, frame_idx)

    def _log(self, pop, frame_idx):
        self.trajectory_history.append(self.pos.cpu().numpy().copy())
        self.visibility_history.append((~self.is_underground).cpu().numpy().copy())
        self.active_history.append(self.is_active.cpu().numpy().copy())
        self.satiation_history.append(self.satiation_timers.cpu().numpy().copy())
        self.alarm_history.append(self.alarm_level.cpu().numpy().copy())
        self.population_history.append(pop)
        self.nest_state_history.append(list(self.nest_occupant))

class EcosystemRenderer:
    """Class `EcosystemRenderer` -- simulation component."""

    def __init__(self, cfg: SimulationConfig, sim: TerraRoncaEcosystem):
        self.cfg = cfg; self.sim = sim

    def generate_svg(self) -> str:
        """Function `generate_svg` -- simulation component."""

        w, h  = self.cfg.width, self.cfg.height
        dur   = self.cfg.frames / self.cfg.fps
        frames = self.cfg.frames

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#121212;font-family:system-ui, -apple-system, sans-serif;">']
        svg.append('<defs>'
            '<radialGradient id="waterGrad"><stop offset="0%" stop-color="#00ffff" stop-opacity="0.6"/><stop offset="100%" stop-color="#0033aa" stop-opacity="0.0"/></radialGradient>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/></pattern>'
            '<pattern id="sandDot" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="5" cy="5" r="1.5" fill="#d4a373" opacity="0.15"/></pattern>'
            '</defs>')

        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="#1c1611"/>')
        svg.append(f'<rect x="0" y="0" width="{self.cfg.sand_zone_x}" height="{h}" fill="url(#sandDot)"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="#05070a"/>')
        svg.append(f'<rect x="{self.cfg.sand_zone_x}" y="0" width="{w-self.cfg.sand_zone_x}" height="{h}" fill="url(#dotGrid)"/>')

        # Vereda with health
        r_vals = ";".join(f"{r:.1f}" for r in self.sim.vereda_radius_history)
        for i, vn in enumerate(self.sim.vereda_nodes.cpu().numpy()):
            ops = ";".join(f"{h2[i]*0.6:.2f}" for h2 in self.sim.vereda_health_history)
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="url(#waterGrad)"><animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite"/></circle>')

        # Road
        svg.append(f'<rect x="{self.cfg.road_x_center-self.cfg.road_width/2}" y="0" width="{self.cfg.road_width}" height="{h}" fill="#424242" opacity="0.85"/>')
        svg.append(f'<line x1="{self.cfg.road_x_center}" y1="0" x2="{self.cfg.road_x_center}" y2="{h}" stroke="#ffeb3b" stroke-width="2" stroke-dasharray="10" opacity="0.75"/>')

        # NEW: Fence lines (horizontal)
        for fy in self.cfg.fence_y_positions:
            svg.append(f'<line x1="0" y1="{fy}" x2="{w}" y2="{fy}" stroke="#a5d6a7" stroke-width="3" stroke-dasharray="6,3" opacity="0.85"/>')
            svg.append(f'<text font-weight="bold" x="8" y="{fy-6}" font-size="15" fill="#999999" opacity="0.8">FENCE (ground-impermeable)</text>')
            # Fear buffer zone
            svg.append(f'<rect x="0" y="{fy-self.cfg.fence_fear_radius}" width="{w}" height="{self.cfg.fence_fear_radius*2}" fill="#1b5e20" opacity="0.06"/>')

        # Title
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Fencing (Permeable to Flyers)</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Ground feeders blocked; aerial species cross freely</text>')

        # Traces
        for idx in range(self.cfg.max_particles):
            if not any(self.sim.active_history[fi][idx] for fi in range(frames)): continue
            p_col = self.sim.colors[idx]
            is_migrant = self.sim.is_migrant[idx].item()
            is_grazer  = self.sim.is_grazer[idx].item()
            flies_flag = self.sim.flies[idx].item()

            px = ";".join(f"{p[idx,0]:.1f}" for p in self.sim.trajectory_history)
            py = ";".join(f"{p[idx,1]:.1f}" for p in self.sim.trajectory_history)
            op = ";".join("1.0" if (self.sim.active_history[fi][idx] and self.sim.visibility_history[fi][idx]) else "0.0" for fi in range(frames))

            if idx % 8 == 0 or is_migrant or is_grazer:
                chunks, cur = [], []
                for fi in range(0, frames, 2):
                    if self.sim.active_history[fi][idx]: cur.append(self.sim.trajectory_history[fi][idx])
                    elif len(cur)>1: chunks.append(cur); cur=[]
                if len(cur)>1: chunks.append(cur)
                dash = 'stroke-dasharray="2,2"' if is_migrant else ""
                sw = 1.8 if is_migrant else (3.0 if is_grazer else 1.0)
                opa = 0.6 if is_migrant else (0.8 if is_grazer else 0.3)
                for chunk in chunks:
                    svg.append(f'<path d="M {" L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in chunk)}" fill="none" stroke="{p_col}" stroke-opacity="{opa}" stroke-width="{sw}" {dash}/>')

            rad = 4.5 if is_migrant else (7.0 if is_grazer else 4.0)
            svg.append(f'<circle r="{rad}" fill="{p_col}"><animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Educational card
        cw, ch = 375, 155
        svg.append(f'<g transform="translate({w-cw-20},340)">'
                   f'<rect width="{cw}" height="{ch}" fill="#1a1a2e" rx="8" ry="8" stroke="#a5d6a7" stroke-width="1" opacity="0.95"/>'
                   f'<text x="15" y="25" fill="#cccccc" font-size="15" font-weight="bold">Feature: Fencing</text>'
                   f'<text font-weight="bold" x="15" y="48" fill="#cccccc" font-size="15">Two dashed green horizontal fences span the map.</text>'
                   f'<text font-weight="bold" x="15" y="68" fill="#cccccc" font-size="15">Ground feeders (Gado Nelore) cannot cross them.</text>'
                   f'<text font-weight="bold" x="15" y="88" fill="#cccccc" font-size="15">Their "flies=False" flag means the fence repulsion</text>'
                   f'<text font-weight="bold" x="15" y="108" fill="#cccccc" font-size="15">force bounces them back. Aerial species ignore the</text>'
                   f'<text font-weight="bold" x="15" y="128" fill="#cccccc" font-size="15">fence entirely (flies=True) and pass through freely.</text>'
                   f'<text font-weight="bold" x="15" y="148" fill="#cccccc" font-size="15">Birdlife is unaffected; cattle are corralled.</text>'
                   f'</g>')

        # -- Scientific Validation Watermark --
        svg.append(f'<g transform="translate(10, {h - 15})">')

        svg.append('</g>')
        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

def save_svg_to_drive(svg_content, notebook_id):
    """Function `save_svg_to_drive` -- simulation component."""

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd(), 'svg_output')
    os.makedirs(save_dir, exist_ok=True)
    fp = os.path.join(save_dir, f'{notebook_id}.svg')
    with open(fp, 'w', encoding='utf-8') as f: f.write(svg_content)
    print(f"SVG saved -> {fp}")

def main():
    """Function `main` -- simulation component."""

    print(f"Initializing Fencing simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for fi in range(CONFIG.frames): sim.step(fi)
    print("Simulation complete. Generating SVG...")
    svg = EcosystemRenderer(CONFIG, sim).generate_svg()
    display(HTML(svg))
    save_svg_to_drive(svg, 'notebook_33')

if __name__ == "__main__":
    main()
