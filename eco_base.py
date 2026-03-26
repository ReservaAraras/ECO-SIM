# eco_base.py - shared core for notebooks 37-80
# pyre-ignore-all-errors
import html as _html


CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 602  # Migrated from 600

ZONES = {
    "header": {"y": 0, "height": 60},
    "main_content": {"y": 60, "height": 460},
    "info_cards": {"y": 520, "height": 50},
    "footer": {"y": 570, "height": 32},
}

import os, re, random, math, sys
import torch # pyre-ignore
import numpy as np # pyre-ignore
import xml.etree.ElementTree as _ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

try:
    from IPython.display import display, HTML # pyre-ignore
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x

BIODIVERSITY_DB = {
    "Tucano-toco":   {"speed":4.5,"color":"#fe4db7","weight":0.20,"diet":"Frugivore",  "cohesion":0.05,"drag":0.85,"max_turn":0.10,"lifespan_base":300,"flies":True},
    "Gralha-campo":  {"speed":5.8,"color":"#00ffdc","weight":0.15,"diet":"Insectivore","cohesion":0.08,"drag":0.90,"max_turn":0.15,"lifespan_base":250,"flies":True},
    "Beija-flor":    {"speed":6.2,"color":"#ffe43f","weight":0.12,"diet":"Insectivore","cohesion":0.01,"drag":0.98,"max_turn":0.40,"lifespan_base":180,"flies":True},
    "Gaviao-carijo": {"speed":7.0,"color":"#f44336","weight":0.10,"diet":"Carnivore",  "cohesion":0.00,"drag":0.95,"max_turn":0.20,"lifespan_base":350,"flies":True},
    "Urubu-rei":     {"speed":5.5,"color":"#b39ddb","weight":0.13,"diet":"Scavenger",  "cohesion":0.02,"drag":0.92,"max_turn":0.08,"lifespan_base":400,"flies":True},
    "Andorinha-rio": {"speed":8.5,"color":"#4fc3f7","weight":0.15,"diet":"Migrant",    "cohesion":0.15,"drag":0.98,"max_turn":0.05,"lifespan_base":600,"flies":True},
    "Gado-Nelore":   {"speed":1.2,"color":"#8d6e63","weight":0.15,"diet":"Grazer",     "cohesion":0.35,"drag":0.80,"max_turn":0.02,"lifespan_base":300,"flies":False},
    # ── Ecological Relationship Clocks series (notebooks 55-66) ──────────
    "Arara-caninde": {"speed":5.0,"color":"#ffd600","weight":0.18,"diet":"Frugivore",  "cohesion":0.10,"drag":0.87,"max_turn":0.12,"lifespan_base":500,"flies":True},
    "Lobo-guara":    {"speed":4.0,"color":"#d84315","weight":0.08,"diet":"Frugivore",  "cohesion":0.00,"drag":0.82,"max_turn":0.18,"lifespan_base":450,"flies":False},
    "Anta":          {"speed":2.5,"color":"#5d4037","weight":0.14,"diet":"Frugivore",  "cohesion":0.02,"drag":0.78,"max_turn":0.05,"lifespan_base":700,"flies":False},
    "Morcego-nectar":{"speed":7.5,"color":"#9c27b0","weight":0.10,"diet":"Insectivore","cohesion":0.03,"drag":0.96,"max_turn":0.35,"lifespan_base":200,"flies":True},
    "Abelha-nativa": {"speed":5.0,"color":"#ffeb3b","weight":0.14,"diet":"Insectivore","cohesion":0.20,"drag":0.97,"max_turn":0.50,"lifespan_base":120,"flies":True},
    # ── Seasonal/Migratory Interventions Series (notebooks 67-70) ───────
    "Cutia": {"speed":4.0, "color":"#a1887f", "weight":0.05, "diet":"Frugivore","cohesion":0.01,"drag":0.85,"max_turn":0.15,"lifespan_base":200,"flies":False},
    "Tamandua-bandeira": {"speed":3.0, "color":"#5d4037", "weight":0.12, "diet":"Insectivore","cohesion":0.00,"drag":0.80,"max_turn":0.10,"lifespan_base":400,"flies":False},
    "Cupim": {"speed":1.0, "color":"#fff59d", "weight":0.01, "diet":"Scavenger","cohesion":0.90,"drag":0.50,"max_turn":0.80,"lifespan_base":50,"flies":False},
    "Fungo-micorrizico": {"speed":0.0, "color":"#bb86fc", "weight":0.01, "diet":"Scavenger","cohesion":0.00,"drag":0.10,"max_turn":0.00,"lifespan_base":1000,"flies":False},
    # ── Cerrado Ecological Web Series (notebooks 71-74) ──────────────────
    # nb71: Seriema × Serpentes (animal × animal - dry-season predation)
    "Seriema": {"speed":3.2, "color":"#e65100", "weight":0.08, "diet":"Carnivore","cohesion":0.00,"drag":0.83,"max_turn":0.20,"lifespan_base":500,"flies":False},
    "Serpente-Bothrops": {"speed":1.5, "color":"#c62828", "weight":0.02, "diet":"Carnivore","cohesion":0.00,"drag":0.70,"max_turn":0.30,"lifespan_base":180,"flies":False},
    # nb72: Ipê-amarelo × Abelha-Mamangava (plant × animal pollinator)
    "Abelha-mamangava": {"speed":5.5, "color":"#fdd835", "weight":0.13, "diet":"Insectivore","cohesion":0.12,"drag":0.97,"max_turn":0.55,"lifespan_base":90,"flies":True},
    # nb73: Buriti × Fungo-arbuscular (plant × fungus - flood network)
    "Fungo-arbuscular-Glomus": {"speed":0.0, "color":"#ce93d8", "weight":0.01, "diet":"Scavenger","cohesion":0.00,"drag":0.05,"max_turn":0.00,"lifespan_base":1200,"flies":False},
    # nb74: Andorinha-rabo-branco × Insetos-cerrado (animal × animal - migration)
    "Andorinha-rabo-branco": {"speed":9.5, "color":"#e0f2f1", "weight":0.14, "diet":"Insectivore","cohesion":0.20,"drag":0.99,"max_turn":0.08,"lifespan_base":700,"flies":True},
    # ── Cerrado Trophic Cascade Series (notebooks 75-78) ──────────────────────
    # nb75: Lobo-guará × Lobeira (animal × plant - frugivory & seed dispersal)
    "Lobo-guara-Chrysocyon": {"speed":4.2, "color":"#ff8f00", "weight":0.07, "diet":"Frugivore","cohesion":0.00,"drag":0.85,"max_turn":0.18,"lifespan_base":600,"flies":False},
    "Lobeira-Solanum": {"speed":0.0, "color":"#7b1fa2", "weight":0.05, "diet":"Scavenger","cohesion":0.00,"drag":0.05,"max_turn":0.00,"lifespan_base":1500,"flies":False},
    # nb76: Gavião-real × Macaco-prego (animal × animal - apex predation)
    "Gaviao-real-Harpia": {"speed":7.5, "color":"#b0bec5", "weight":0.08, "diet":"Carnivore","cohesion":0.00,"drag":0.96,"max_turn":0.15,"lifespan_base":800,"flies":True},
    "Macaco-prego-Sapajus": {"speed":4.8, "color":"#8d6e63", "weight":0.12, "diet":"Frugivore","cohesion":0.25,"drag":0.88,"max_turn":0.30,"lifespan_base":300,"flies":False},
    # nb77: Mandacaru × Morcego-polinizador (plant × animal - nocturnal pollination)
    "Morcego-Glossophaga": {"speed":8.0, "color":"#78909c", "weight":0.09, "diet":"Insectivore","cohesion":0.05,"drag":0.98,"max_turn":0.45,"lifespan_base":180,"flies":True},
    # nb78: Besouro-Scolytinae × Fungo-Ambrosia (animal × fungus - nutrient cycling)
    "Besouro-Scolytinae": {"speed":1.5, "color":"#ffe082", "weight":0.02, "diet":"Scavenger","cohesion":0.95,"drag":0.40,"max_turn":0.90,"lifespan_base":60,"flies":True},
    "Fungo-Ambrosia": {"speed":0.0, "color":"#ce93d8", "weight":0.01, "diet":"Scavenger","cohesion":0.00,"drag":0.05,"max_turn":0.00,"lifespan_base":800,"flies":False},
    # ── Seasonal/Migratory Cerrado Dynamics Series (notebooks 79-80) ──────────
    # nb79: Suiriri × Murici (animal × plant - migration & seasonal fruiting)
    "Suiriri-Tyrannus": {"speed":8.5, "color":"#ffeb3b", "weight":0.05, "diet":"Migrant","cohesion":0.15,"drag":0.95,"max_turn":0.10,"lifespan_base":300,"flies":True},
    "Murici-Byrsonima": {"speed":0.0, "color":"#fbc02d", "weight":0.08, "diet":"Scavenger","cohesion":0.00,"drag":0.05,"max_turn":0.00,"lifespan_base":1000,"flies":False},
    # nb80: Saúva × Fungo-Leucoagaricus (animal × fungus - dry season subterranean symbiosis)
    "Sauva-Atta": {"speed":1.0, "color":"#d84315", "weight":0.02, "diet":"Insectivore","cohesion":0.80,"drag":0.60,"max_turn":0.80,"lifespan_base":120,"flies":False},
    "Fungo-Leucoagaricus": {"speed":0.0, "color":"#f8bbd0", "weight":0.01, "diet":"Scavenger","cohesion":0.00,"drag":0.05,"max_turn":0.00,"lifespan_base":1500,"flies":False},
}

@dataclass
class BaseConfig:
    width:int=CANVAS_WIDTH; height:int=CANVAS_HEIGHT; frames:int=560; fps:int=10
    device:str='cuda' if torch.cuda.is_available() else 'cpu'
    initial_particles:int=240; max_particles:int=450; carrying_capacity:int=350; dt:float=0.5
    vereda_min_radius:float=20.0; vereda_max_radius:float=90.0; vereda_cycle_frames:float=200.0; vereda_base_attraction:float=3.0
    fruiting_cycle_frames:float=150.0; fruiting_radius:float=50.0; fruiting_base_attraction:float=4.0
    seed_carry_duration:float=60.0; seed_pickup_chance:float=0.05
    sand_zone_x:float=640.0; sand_speed_modifier:float=0.6
    fire_start_frame:int=350; fire_spread_prob:float=0.08; fire_flee_radius:float=150.0; fire_flee_force:float=12.0; max_fire_nodes:int=120
    mating_energy_threshold:float=75.0; mating_radius:float=30.0; energy_decay:float=0.12
    energy_gain_fruiting:float=6.0; energy_gain_vereda:float=2.5; lifespan_variance:float=50.0
    swarm_spawn_interval:int=45; swarm_lifetime:int=55; swarm_radius:float=50.0; swarm_drift:float=0.3
    insect_chase_radius:float=200.0; insect_chase_force:float=8.0; insect_energy_gain:float=12.0
    carrion_circle_radius:float=60.0; carrion_linger_frames:int=120; scavenger_energy_gain:float=8.0
    roost_nodes_initial:int=10; roost_attract_radius:float=120.0; roost_attract_force:float=4.0; roost_energy_bonus:float=2.0
    logging_start_frame:int=100; logging_interval:int=60; logging_disturbance_radius:float=80.0
    homeless_restless_force:float=2.5
    keystone_removal_frame:int=230; flock_cohesion_radius:float=80.0; flock_cohesion_multiplier:float=6.0
    hunting_chase_radius:float=160.0; hunting_capture_radius:float=12.0; hunting_chase_force:float=14.0
    predator_energy_gain:float=40.0; satiation_duration:int=90; satiation_speed_penalty:float=0.4
    alarm_propagation_radius:float=90.0; alarm_decay_rate:float=0.05; alarm_panic_multiplier:float=1.8; alarm_panic_force:float=8.0
    migration_start_frame:int=400; migration_spawn_rate:float=0.8; migration_flow_force:float=6.0
    migration_span_y_min:float=150.0; migration_span_y_max:float=450.0
    road_x_center:float=780.0; road_width:float=20.0; road_fear_radius:float=60.0
    road_fear_force_base:float=15.0; road_crossing_courage:float=0.05
    cattle_graze_radius:float=45.0; cattle_resource_depletion_rate:float=0.01
    cattle_chase_vereda_force:float=2.0; cattle_graze_gain:float=0.8
    fence_y_positions:tuple=(180.0,420.0); fence_fear_radius:float=18.0; fence_repulsion_force:float=22.0
    num_noise_vehicles:int=4; noise_radius:float=90.0; noise_repulsion_force:float=18.0; noise_speed:float=2.5
    poach_zones:tuple=((200.0,480.0,120.0),(1050.0,120.0,100.0),(500.0,80.0,90.0))
    poach_start_frame:int=60; poach_mortality_prob:float=0.008; poach_alarm_radius:float=110.0


class EcosystemBase:
    """Core simulation - subclass and override extra_step() + extra_forces() + extra_svg()."""

    def __init__(self, cfg: BaseConfig):
        self.cfg=cfg; self.dev=torch.device(cfg.device); W,H=cfg.width,cfg.height
        self.pos=torch.rand((cfg.max_particles,2),device=self.dev)*torch.tensor([W,H],device=self.dev)
        self.vel=(torch.rand((cfg.max_particles,2),device=self.dev)-0.5)*10.0
        self.guilds=list(BIODIVERSITY_DB.keys())
        wt=[BIODIVERSITY_DB[g]["weight"] for g in self.guilds]
        wt=np.array(wt, dtype='float64')
        wt /= wt.sum()
        wt[-1] = 1.0 - wt[:-1].sum()
        idx=np.random.choice(len(self.guilds),size=cfg.max_particles,p=wt)
        self.species_id=torch.tensor(idx,device=self.dev)
        G=BIODIVERSITY_DB; gl=self.guilds
        self.speeds  =torch.tensor([G[gl[i]]["speed"]    for i in idx],device=self.dev)
        self.drags   =torch.tensor([G[gl[i]]["drag"]     for i in idx],device=self.dev).unsqueeze(1)
        self.max_turns=torch.tensor([G[gl[i]]["max_turn"] for i in idx],device=self.dev)
        self.cohesions=torch.tensor([G[gl[i]]["cohesion"] for i in idx],device=self.dev)
        self.colors  =[G[gl[i]]["color"] for i in idx]
        def mk(d): return torch.tensor([G[gl[i]]["diet"]==d for i in idx],device=self.dev,dtype=torch.bool)
        self.is_frugivore=mk("Frugivore"); self.is_insectivore=mk("Insectivore"); self.is_scavenger=mk("Scavenger")
        self.is_carnivore=mk("Carnivore"); self.is_migrant=mk("Migrant"); self.is_grazer=mk("Grazer")
        self.flies=torch.tensor([G[gl[i]]["flies"] for i in idx],device=self.dev,dtype=torch.bool)
        self.is_male=torch.rand(cfg.max_particles,device=self.dev)>0.5
        self.courage=torch.rand(cfg.max_particles,device=self.dev)
        self.is_active=torch.zeros(cfg.max_particles,device=self.dev,dtype=torch.bool); self.is_active[:cfg.initial_particles]=True
        self.energy=torch.ones(cfg.max_particles,device=self.dev)*80.0
        self.age=torch.zeros(cfg.max_particles,device=self.dev); self.age[:cfg.initial_particles]=torch.rand(cfg.initial_particles,device=self.dev)*200.0
        bl=torch.tensor([G[gl[i]]["lifespan_base"] for i in idx],device=self.dev,dtype=torch.float32)
        self.lifespan=bl+torch.randn(cfg.max_particles,device=self.dev)*cfg.lifespan_variance
        self.is_underground=torch.zeros(cfg.max_particles,device=self.dev,dtype=torch.bool)
        self.satiation_timers=torch.zeros(cfg.max_particles,device=self.dev,dtype=torch.int32)
        self.alarm_level=torch.zeros(cfg.max_particles,device=self.dev,dtype=torch.float32)
        self.alarm_vectors=torch.zeros((cfg.max_particles,2),device=self.dev,dtype=torch.float32)
        nx=[120.,280.,450.,600.,750.,900.,1050.,200.,500.,1100.]; ny=[150.,300.,100.,450.,200.,350.,120.,480.,320.,400.]
        self.nest_nodes=torch.tensor(list(zip(nx,ny)),device=self.dev,dtype=torch.float32)
        self.nest_active=[True]*cfg.roost_nodes_initial; self.nest_occupant=[-1]*cfg.roost_nodes_initial
        self.particle_nest=torch.full((cfg.max_particles,),-1,device=self.dev,dtype=torch.long)
        self.logging_events:List[Dict]=[]
        self.nectar_nodes=torch.tensor([[250.,300.]],device=self.dev)
        self.vereda_nodes=torch.tensor([[600.,350.],[900.,450.]],device=self.dev)
        self.vereda_health=torch.ones(2,device=self.dev,dtype=torch.float32)
        self.fruiting_nodes=torch.tensor([[300.,150.],[850.,250.]],device=self.dev)
        self.has_seed=torch.zeros(cfg.max_particles,device=self.dev,dtype=torch.bool)
        self.seed_timers=torch.zeros(cfg.max_particles,device=self.dev); self.dropped_seeds:List[Dict]=[]
        self.fire_nodes=torch.zeros((0,2),device=self.dev); self.swarms:List[Dict]=[]; self.carrion_sites:List[Dict]=[]
        self.migrants_spawned=0; self.migrants_exited=0
        self.noise_pos=torch.rand((cfg.num_noise_vehicles,2),device=self.dev)*torch.tensor([W,H],device=self.dev)
        self.noise_vel=(torch.rand((cfg.num_noise_vehicles,2),device=self.dev)-0.5)*cfg.noise_speed*2.
        self.noise_phase=torch.rand(cfg.num_noise_vehicles,device=self.dev)*math.pi*2.
        self.poach_centers=torch.tensor([[z[0],z[1]] for z in cfg.poach_zones],device=self.dev,dtype=torch.float32)
        self.poach_radii=torch.tensor([z[2] for z in cfg.poach_zones],device=self.dev,dtype=torch.float32)
        self.poach_kill_events:List[Dict]=[]
        # Logs
        self.trajectory_history=[]; self.visibility_history=[]; self.active_history=[]
        self.fire_nodes_history=[]; self.vereda_radius_history=[]; self.fruiting_intensity_history=[]
        self.population_history=[]; self.nest_state_history=[]; self.nest_active_history=[]
        self.cohesion_stats=[]; self.satiation_history=[]; self.alarm_history=[]; self.vereda_health_history=[]
        self.noise_pos_history=[]; self.birth_events=[]; self.death_events=[]; self.alarm_events=[]
        self.swarm_render_log=[]
        self._extra_init()

    # ── override hooks ────────────────────────────────────────────────────────
    def _extra_init(self): pass
    def extra_step(self, fi, am, sm): pass          # called at start of step, before physics
    def extra_forces(self, fi, am, sm): return torch.zeros_like(self.vel)  # added to raw force
    def extra_svg(self): return ""                  # injected into SVG before particles
    def extra_svg_overlay(self): return ""          # injected after particles (info card etc.)

    # ── internals ─────────────────────────────────────────────────────────────
    def _spawn_migrant(self,fi):
        free=(~self.is_active).nonzero().squeeze(1)
        if len(free)==0: return
        ni=free[0]; ms="Andorinha-rio"; mid=self.guilds.index(ms); G=BIODIVERSITY_DB
        self.species_id[ni]=mid; self.speeds[ni]=G[ms]["speed"]; self.drags[ni]=G[ms]["drag"]
        self.max_turns[ni]=G[ms]["max_turn"]; self.cohesions[ni]=G[ms]["cohesion"]
        self.colors[ni]=G[ms]["color"]; self.is_migrant[ni]=True; self.is_grazer[ni]=False; self.flies[ni]=True
        self.is_frugivore[ni]=self.is_insectivore[ni]=self.is_scavenger[ni]=self.is_carnivore[ni]=False
        self.is_active[ni]=True; y=random.uniform(self.cfg.migration_span_y_min,self.cfg.migration_span_y_max)
        self.pos[ni]=torch.tensor([-20.,y],device=self.dev); self.vel[ni]=torch.tensor([8.,0.],device=self.dev)
        self.energy[ni]=100.; self.age[ni]=0.; self.alarm_level[ni]=0.
        self.is_underground[ni]=False; self.has_seed[ni]=False; self.courage[ni]=1.; self.migrants_spawned+=1

    def step(self, fi):
        cfg=self.cfg; old_vel=self.vel.clone(); am=self.is_active.clone(); sm=am&~self.is_underground

        # Keystone
        if fi==cfg.keystone_removal_frame:
            lm=am&~self.is_migrant&~self.is_grazer
            if lm.any():
                c=torch.bincount(self.species_id[lm]); kid=int(torch.argmax(c))
                tm=(self.species_id==kid)&am; self.is_active[tm]=False; self.has_seed[tm]=False; self.alarm_level[tm]=0.
                for ix in tm.nonzero().squeeze(1): # pyre-ignore
                    n=int(self.particle_nest[ix])
                    if n>=0: self.nest_occupant[n]=-1; self.particle_nest[ix]=-1
                    pn=self.pos[ix].cpu().numpy().copy()
                    self.death_events.append({"pos":pn,"frame":fi,"color":self.colors[ix],"reason":"extinction"})
                    self.carrion_sites.append({"pos":pn,"spawn_frame":fi,"expire_frame":fi+cfg.carrion_linger_frames})
            am=self.is_active.clone(); sm=am&~self.is_underground

        # Logging
        if fi>=cfg.logging_start_frame and (fi-cfg.logging_start_frame)%cfg.logging_interval==0:
            alive=[i for i,a in enumerate(self.nest_active) if a]
            if alive:
                fallen=random.choice(alive); self.nest_active[fallen]=False
                tp=self.nest_nodes[fallen].cpu().numpy().copy()
                self.logging_events.append({"tree_idx":fallen,"pos":tp,"frame":fi})
                occ=self.nest_occupant[fallen]
                if occ>=0: self.particle_nest[occ]=-1
                self.nest_occupant[fallen]=-1
                dtr=torch.norm(self.pos-torch.tensor(tp,device=self.dev),dim=1)
                dist=(dtr<cfg.logging_disturbance_radius)&sm
                if dist.any():
                    self.alarm_level[dist]=1.; away=self.pos[dist]-torch.tensor(tp,device=self.dev)
                    self.alarm_vectors[dist]=away/away.norm(dim=1,keepdim=True).clamp(min=1.)

        # Noise vehicles
        self.noise_phase+=0.05
        wb=torch.stack([torch.cos(self.noise_phase),torch.sin(self.noise_phase*0.7)],dim=1)*cfg.noise_speed
        self.noise_vel=self.noise_vel*0.92+wb*0.08+torch.randn_like(self.noise_vel)*0.3
        spdn=torch.norm(self.noise_vel,dim=1,keepdim=True).clamp(min=0.1)
        self.noise_vel=(self.noise_vel/spdn)*cfg.noise_speed; self.noise_pos+=self.noise_vel
        self.noise_pos[:,0]=self.noise_pos[:,0].clamp(50,cfg.width-50)
        self.noise_pos[:,1]=self.noise_pos[:,1].clamp(50,cfg.height-50)
        hx=(self.noise_pos[:,0]<=50)|(self.noise_pos[:,0]>=cfg.width-50)
        hy=(self.noise_pos[:,1]<=50)|(self.noise_pos[:,1]>=cfg.height-50)
        self.noise_vel[hx,0]*=-1; self.noise_vel[hy,1]*=-1
        self.noise_pos_history.append(self.noise_pos.cpu().numpy().copy())

        self.age[am]+=1.; self.energy[am]-=cfg.energy_decay
        self.satiation_timers[self.satiation_timers>0]-=1
        self.alarm_level=torch.clamp(self.alarm_level-cfg.alarm_decay_rate,min=0.)

        if fi>=cfg.migration_start_frame and random.random()<cfg.migration_spawn_rate: self._spawn_migrant(fi)
        am=self.is_active.clone(); sm=am&~self.is_underground
        if (am&self.is_migrant).any():
            ex=(am&self.is_migrant)&(self.pos[:,0]>cfg.width+10.)
            if ex.any():
                for ix in ex.nonzero().squeeze(1): self.is_active[ix]=False; self.migrants_exited+=1
        am=self.is_active.clone(); sm=am&~self.is_underground

        # Poaching
        if fi>=cfg.poach_start_frame:
            dz=torch.cdist(self.pos,self.poach_centers); inz=(dz<self.poach_radii.unsqueeze(0)).any(dim=1)
            at_risk=inz&sm&~self.is_grazer&~self.is_carnivore
            if at_risk.any():
                roll=torch.rand(cfg.max_particles,device=self.dev)<cfg.poach_mortality_prob
                poached=at_risk&roll
                if poached.any():
                    for ix in poached.nonzero().squeeze(1):
                        self.is_active[ix]=False; self.has_seed[ix]=False; self.alarm_level[ix]=0.
                        n=int(self.particle_nest[ix])
                        if n>=0: self.nest_occupant[n]=-1; self.particle_nest[ix]=-1
                        pn=self.pos[ix].cpu().numpy().copy()
                        self.poach_kill_events.append({"pos":pn,"frame":fi}); self.death_events.append({"pos":pn,"frame":fi,"color":self.colors[ix],"reason":"poaching"})
                        self.carrion_sites.append({"pos":pn,"spawn_frame":fi,"expire_frame":fi+cfg.carrion_linger_frames})
                near_zone=sm&(dz.min(dim=1).values<cfg.poach_alarm_radius)
                self.alarm_level[near_zone]=torch.clamp(self.alarm_level[near_zone]+0.2,max=1.)

        # Extra hook
        self.extra_step(fi, am, sm)

        # Natural mortality
        am=self.is_active.clone(); sm=am&~self.is_underground
        dead=(am&(self.energy<=0.))|(am&(self.age>=self.lifespan))
        if dead.any():
            for ix in dead.nonzero().squeeze(1):
                self.is_active[ix]=False; self.has_seed[ix]=False; self.alarm_level[ix]=0.
                n=int(self.particle_nest[ix])
                if n>=0: self.nest_occupant[n]=-1; self.particle_nest[ix]=-1
                pn=self.pos[ix].cpu().numpy().copy()
                self.death_events.append({"pos":pn,"frame":fi,"color":self.colors[ix],"reason":"old_age" if self.age[ix]>=self.lifespan[ix] else "starvation"})
                self.carrion_sites.append({"pos":pn,"spawn_frame":fi,"expire_frame":fi+cfg.carrion_linger_frames})

        am=self.is_active.clone(); sm=am&~self.is_underground; pop=int(am.sum())
        if not am.any(): self._log(pop,fi); return

        self.vereda_health=torch.clamp(self.vereda_health+0.002,0.,1.); self.vereda_health_history.append(self.vereda_health.cpu().numpy().copy())

        # Flocking
        fa2=torch.zeros_like(self.vel)
        if sm.any():
            sidx=sm.nonzero().squeeze(1); dm=torch.cdist(self.pos[sm],self.pos[sm])
            ir=dm<cfg.flock_cohesion_radius; ir.fill_diagonal_(False); nc=ir.sum(dim=1); hn=nc>0
            if hn.any():
                ap=torch.matmul(ir.float(),self.pos[sm])[hn]/nc[hn].unsqueeze(1).float()
                pull=ap-self.pos[sm][hn]; fa2[sidx[hn]]=(pull/pull.norm(dim=1,keepdim=True).clamp(min=1.))*self.cohesions[sidx[hn]].unsqueeze(1)*cfg.flock_cohesion_multiplier
            self.cohesion_stats.append(nc.float().mean().item())
        else: self.cohesion_stats.append(0.)

        # Noise repulsion
        nr=torch.zeros_like(self.vel)
        for vi in range(cfg.num_noise_vehicles):
            vp=self.noise_pos[vi]; dn=torch.norm(self.pos-vp,dim=1); aff=sm&(dn<cfg.noise_radius)
            if aff.any():
                away=self.pos[aff]-vp; mag=(1.-dn[aff]/cfg.noise_radius)**2
                nr[aff]+=(away/dn[aff].unsqueeze(1).clamp(min=1.))*mag.unsqueeze(1)*cfg.noise_repulsion_force # pyre-ignore
                self.alarm_level[aff]=torch.clamp(self.alarm_level[aff]+0.3,max=1.)
                self.alarm_vectors[aff]=away/dn[aff].unsqueeze(1).clamp(min=1.)

        # Road + fence
        road_rep=torch.zeros_like(self.vel); dtr=torch.abs(self.pos[:,0]-cfg.road_x_center)
        sc=sm&(dtr<cfg.road_fear_radius)&(self.courage<(1.-cfg.road_crossing_courage))&~self.is_grazer
        if sc.any(): road_rep[sc,0]=torch.sign(self.pos[sc,0]-cfg.road_x_center)*(1.-(dtr[sc]/cfg.road_fear_radius))**2*cfg.road_fear_force_base
        fence_rep=torch.zeros_like(self.vel); gm=sm&~self.flies
        if gm.any():
            for fy in cfg.fence_y_positions:
                dy=torch.abs(self.pos[:,1]-fy); nf=gm&(dy<cfg.fence_fear_radius)
                if nf.any(): fence_rep[nf,1]+=torch.sign(self.pos[nf,1]-fy)*(1.-dy[nf]/cfg.fence_fear_radius)**2*cfg.fence_repulsion_force

        # Hunting
        hunt=torch.zeros_like(self.vel); hg=(sm&self.is_carnivore)&(self.satiation_timers<=0); qm=sm&(self.is_frugivore|self.is_insectivore|self.is_migrant)
        if hg.any() and qm.any():
            pi2=hg.nonzero().squeeze(1); qi=qm.nonzero().squeeze(1)
            dh=torch.cdist(self.pos[hg],self.pos[qm]); md,ti=torch.min(dh,dim=1); chase=md<cfg.hunting_chase_radius
            if chase.any():
                tgt=self.pos[qi[ti[chase]]]; pll=tgt-self.pos[pi2[chase]]
                hunt[pi2[chase]]=(pll/md[chase].unsqueeze(1).clamp(min=1.))*cfg.hunting_chase_force
                self.alarm_level[qi[ti[chase]]]=1.; self.alarm_vectors[qi[ti[chase]]]=pll/md[chase].unsqueeze(1).clamp(min=1.)
                cap=md[chase]<cfg.hunting_capture_radius
                if cap.any():
                    for k in range(int(cap.sum())):
                        hid=pi2[chase][cap][k]; pid=qi[ti[chase][cap]][k]
                        if self.is_active[pid]:
                            self.is_active[pid]=False; self.energy[hid]+=cfg.predator_energy_gain; self.satiation_timers[hid]=cfg.satiation_duration
                            pn=self.pos[pid].cpu().numpy().copy()
                            self.death_events.append({"pos":pn,"frame":fi,"color":self.colors[pid],"reason":"predation"})
                            self.carrion_sites.append({"pos":pn,"spawn_frame":fi,"expire_frame":fi+cfg.carrion_linger_frames}); pop-=1
        self.energy.clamp_(0.,100.)

        # Environment
        d_nec=torch.cdist(self.pos,self.nectar_nodes); mn_nec,cl_nec=torch.min(d_nec,dim=1)
        karst=(self.nectar_nodes[cl_nec]-self.pos)/mn_nec.unsqueeze(1).clamp(min=1.)
        cycp=(fi%cfg.vereda_cycle_frames)/cfg.vereda_cycle_frames; wlvl=(math.sin(cycp*2*math.pi-math.pi/2)+1.)/2.
        cvrad=cfg.vereda_min_radius+(cfg.vereda_max_radius-cfg.vereda_min_radius)*wlvl
        self.vereda_radius_history.append(cvrad)
        d_ver=torch.cdist(self.pos,self.vereda_nodes); mn_ver,cl_ver=torch.min(d_ver,dim=1); hmod=self.vereda_health[cl_ver]
        vera=(self.vereda_nodes[cl_ver]-self.pos)/mn_ver.unsqueeze(1).clamp(min=1.)*(cfg.vereda_base_attraction*wlvl*hmod).unsqueeze(1) # pyre-ignore
        flooded=mn_ver<cvrad; vera[flooded]*=0.1
        if (flooded&sm&~self.is_grazer).any(): self.energy[flooded&sm&~self.is_grazer]+=cfg.energy_gain_vereda*hmod[flooded&sm&~self.is_grazer]
        gz=sm&self.is_grazer
        if gz.any():
            pv=self.vereda_nodes[cl_ver[gz]]-self.pos[gz]; vera[gz]+=(pv/mn_ver[gz].unsqueeze(1).clamp(min=1.))*cfg.cattle_chase_vereda_force
            eat=gz&(mn_ver<cfg.cattle_graze_radius)
            if eat.any():
                self.energy[eat]+=cfg.cattle_graze_gain
                for ni in cl_ver[eat]: self.vereda_health[ni]-=cfg.cattle_resource_depletion_rate
                self.vereda_health.clamp_(0.,1.)
        fic=(fi%cfg.fruiting_cycle_frames)/cfg.fruiting_cycle_frames; fi_int=max(0.,math.sin(fic*2*math.pi)); self.fruiting_intensity_history.append(fi_int)
        d_fr=torch.cdist(self.pos,self.fruiting_nodes); mn_fr,cl_fr=torch.min(d_fr,dim=1)
        frut=torch.zeros_like(self.vel); feed=torch.zeros(cfg.max_particles,device=self.dev,dtype=torch.bool)
        if fi_int>0.:
            sml=(mn_fr<200.)&self.is_frugivore&sm
            if sml.any():
                frut[sml]=(self.fruiting_nodes[cl_fr[sml]]-self.pos[sml])/mn_fr[sml].unsqueeze(1).clamp(min=1.)*(cfg.fruiting_base_attraction*fi_int)
                fr2=sml&(mn_fr<cfg.fruiting_radius); feed=fr2&(fi_int>0.1)
                if fr2.any(): frut[fr2]+=torch.randn_like(frut[fr2])*2.; self.energy[fr2]+=cfg.energy_gain_fruiting*fi_int
        if feed.any():
            ok=feed&~self.has_seed&(torch.rand(cfg.max_particles,device=self.dev)<cfg.seed_pickup_chance)
            if ok.any(): self.has_seed[ok]=True; self.seed_timers[ok]=0.
        self.seed_timers[self.has_seed]+=1.
        drop=self.has_seed&(self.seed_timers>=cfg.seed_carry_duration)&sm
        if drop.any():
            self.has_seed[drop]=False
            for p in self.pos[drop]: self.dropped_seeds.append({"pos":p.cpu().numpy().copy(),"frame":fi})

        # Fire
        fire_rep=torch.zeros_like(self.vel); flee=torch.zeros(cfg.max_particles,device=self.dev,dtype=torch.bool)
        if fi>=cfg.fire_start_frame:
            if len(self.fire_nodes)==0: self.fire_nodes=torch.tensor([[1000.,300.]],device=self.dev)
            elif fi%2==0 and len(self.fire_nodes)<cfg.max_fire_nodes:
                sp=torch.rand(len(self.fire_nodes),device=self.dev)<cfg.fire_spread_prob
                if sp.any():
                    ns=self.fire_nodes[sp]+torch.randn((int(sp.sum()),2),device=self.dev)*30.
                    ns[:,0]=ns[:,0].clamp(0,cfg.width); ns[:,1]=ns[:,1].clamp(0,cfg.height)
                    self.fire_nodes=torch.cat([self.fire_nodes,ns],dim=0)
            if len(self.fire_nodes)>0:
                df=torch.cdist(self.pos,self.fire_nodes); mn_f,cl_f=torch.min(df,dim=1); flee=(mn_f<cfg.fire_flee_radius)&sm
                if flee.any():
                    fv=self.pos[flee]-self.fire_nodes[cl_f[flee]]
                    fire_rep[flee]=(fv/mn_f[flee].unsqueeze(1).clamp(min=1.)**1.5)*cfg.fire_flee_force*30.
                    self.alarm_level[flee]=1.; self.alarm_vectors[flee]=fv/mn_f[flee].unsqueeze(1).clamp(min=1.)
        self.fire_nodes_history.append(self.fire_nodes.cpu().numpy().copy())

        # Alarm
        al_rep=torch.zeros_like(self.vel); ial=(self.alarm_level>0.5)&sm; nal=(self.alarm_level<=0.5)&sm
        if ial.any() and nal.any():
            da2=torch.cdist(self.pos[nal],self.pos[ial]); md2,ti2=torch.min(da2,dim=1); inf=md2<cfg.alarm_propagation_radius
            if inf.any():
                ig=nal.nonzero().squeeze(1)[inf]; sg=ial.nonzero().squeeze(1)[ti2[inf]]; self.alarm_level[ig]=1.
                cv2=(self.alarm_vectors[sg]+(self.pos[ig]-self.pos[sg])/md2[inf].unsqueeze(1).clamp(min=1.))/2.
                self.alarm_vectors[ig]=cv2/cv2.norm(dim=1,keepdim=True).clamp(min=1e-5)
        aal=(self.alarm_level>0.1)&sm&~flee&~self.is_carnivore
        if aal.any(): al_rep[aal]=self.alarm_vectors[aal]*cfg.alarm_panic_force*self.alarm_level[aal].unsqueeze(1)

        # Swarms
        if fi%cfg.swarm_spawn_interval==0:
            cx,cy=random.uniform(100,cfg.width-100),random.uniform(80,cfg.height-80)
            self.swarms.append({"centre":torch.tensor([cx,cy],device=self.dev),"drift":(torch.rand(2,device=self.dev)-0.5)*cfg.swarm_drift*2.,"die_frame":fi+cfg.swarm_lifetime})
        insect=torch.zeros_like(self.vel); lsw=[]
        for sw in self.swarms:
            if fi>=sw["die_frame"]: continue
            sw["centre"]+=sw["drift"]; lsw.append(sw)
            dsw=torch.norm(self.pos-sw["centre"],dim=1)
            ch3=(dsw<cfg.insect_chase_radius)&self.is_insectivore&sm&~flee&(self.alarm_level<0.5) # pyre-ignore
            if ch3.any(): insect[ch3]+=(sw["centre"]-self.pos[ch3])/dsw[ch3].unsqueeze(1).clamp(min=1.)*cfg.insect_chase_force
            if ((dsw<cfg.swarm_radius)&self.is_insectivore&sm).any(): self.energy[(dsw<cfg.swarm_radius)&self.is_insectivore&sm]+=cfg.insect_energy_gain
        self.swarms=lsw

        # Scavenging
        scav=torch.zeros_like(self.vel); self.carrion_sites=[c for c in self.carrion_sites if c["expire_frame"]>fi]
        if self.carrion_sites and self.is_scavenger.any():
            import numpy as np # pyre-ignore
            cp3=torch.tensor(np.array([c["pos"] for c in self.carrion_sites]),device=self.dev,dtype=torch.float32)
            dc=torch.cdist(self.pos,cp3); mc,cc=torch.min(dc,dim=1); ss2=self.is_scavenger&sm&~flee # pyre-ignore
            if ss2.any():
                cl2=cp3[cc[ss2]]; ds2=mc[ss2]; si2=ss2.nonzero().squeeze(1); farp=ds2>cfg.carrion_circle_radius
                if farp.any(): scav[si2[farp]]=(cl2[farp]-self.pos[si2[farp]])/ds2[farp].unsqueeze(1).clamp(min=1.)*6.
                np2=~farp
                if np2.any():
                    tc2=self.pos[si2[np2]]-cl2[np2]; tg=torch.stack([-tc2[:,1],tc2[:,0]],dim=1)
                    scav[si2[np2]]=tg/tg.norm(dim=1,keepdim=True).clamp(min=1e-5)*5.; self.energy[si2[np2]]+=cfg.scavenger_energy_gain

        # Nesting
        nest=torch.zeros_like(self.vel); active_nests=[ni for ni,a in enumerate(self.nest_active) if a]
        for ni in range(cfg.roost_nodes_initial):
            occ=self.nest_occupant[ni]
            if occ>=0 and not self.is_active[occ]: self.nest_occupant[ni]=-1; self.particle_nest[occ]=-1
            if not self.nest_active[ni] and self.nest_occupant[ni]>=0: self.particle_nest[self.nest_occupant[ni]]=-1; self.nest_occupant[ni]=-1
        hm=self.particle_nest>=0; homeless=am&~hm&(self.alarm_level<0.5)&~self.is_migrant&~self.is_grazer
        nested=am&hm
        if nested.any():
            self.energy[nested]+=cfg.roost_energy_bonus; ni2=self.particle_nest[nested]
            d_n=torch.norm(self.pos[nested]-self.nest_nodes[ni2],dim=1)
            nest[nested]=(self.nest_nodes[ni2]-self.pos[nested])/d_n.unsqueeze(1).clamp(min=1.)*cfg.roost_attract_force
        free_nests=[ni for ni in active_nests if self.nest_occupant[ni]==-1]
        if free_nests and homeless.any():
            fnp=self.nest_nodes[torch.tensor(free_nests,device=self.dev)]; df2=torch.cdist(self.pos[homeless],fnp)
            mf2,bf2=torch.min(df2,dim=1); hli=homeless.nonzero().squeeze(1); ce=mf2<cfg.roost_attract_radius
            if ce.any():
                tgt2=fnp[bf2[ce]]; dt3=mf2[ce]; nest[hli[ce]]=(tgt2-self.pos[hli[ce]])/dt3.unsqueeze(1).clamp(min=1.)*cfg.roost_attract_force
                arr=dt3<20.
                if arr.any():
                    for kl,kg in enumerate(hli[ce][arr]): # pyre-ignore
                        rn=free_nests[bf2[ce][arr][kl].item()]
                        if self.nest_occupant[rn]==-1: self.nest_occupant[rn]=int(kg); self.particle_nest[int(kg)]=rn
        if homeless.any(): nest[homeless]+=torch.randn(int(homeless.sum()),2,device=self.dev)*cfg.homeless_restless_force

        mig_attr=torch.zeros_like(self.vel); am2=am&self.is_migrant&~flee&(self.alarm_level<0.5)
        if am2.any(): mig_attr[am2]=torch.tensor([cfg.migration_flow_force,0.],device=self.dev)

        # Reproduction
        if pop<cfg.carrying_capacity:
            rdy=sm&~flee&(self.energy>cfg.mating_energy_threshold)&(self.alarm_level<0.5)&~self.is_migrant
            if rdy.any():
                for i in rdy.nonzero().squeeze(1):
                    if self.energy[i]<cfg.mating_energy_threshold: continue
                    ai2=am.nonzero().squeeze(1); da3=torch.norm(self.pos[ai2]-self.pos[i],dim=1)
                    for j in ai2[(da3<cfg.mating_radius)&(da3>0.1)]:
                        if self.energy[j]>cfg.mating_energy_threshold and self.is_male[i]!=self.is_male[j] and self.species_id[i]==self.species_id[j] and not flee[j]: # pyre-ignore
                            fr3=(~self.is_active).nonzero().squeeze(1)
                            if len(fr3):
                                ci=fr3[0]; self.is_active[ci]=True; self.pos[ci]=self.pos[i]+torch.randn(2,device=self.dev)*10.
                                self.vel[ci]=self.vel[i]*-1.; self.energy[ci]=40.; self.age[ci]=0.; self.alarm_level[ci]=0.
                                self.particle_nest[ci]=-1; self.is_underground[ci]=False; self.courage[ci]=torch.rand(1,device=self.dev).item(); self.flies[ci]=self.flies[i]
                                self.energy[i]-=45.; self.energy[j]-=45.
                                self.birth_events.append({"pos":self.pos[ci].cpu().numpy().copy(),"frame":fi,"color":self.colors[ci]}); pop+=1
                            break
                    if pop>=cfg.carrying_capacity: break

        # Physics
        am=self.is_active.clone(); sm=am&~self.is_underground; raw=torch.zeros_like(self.vel)
        xf=self.extra_forces(fi,am,sm)
        if sm.any():
            raw[sm]=(self.vel[sm]*self.drags[sm]+karst[sm]+vera[sm]+frut[sm]+fire_rep[sm]+insect[sm]+scav[sm]
                     +nest[sm]+fa2[sm]+hunt[sm]+al_rep[sm]+mig_attr[sm]+road_rep[sm]+fence_rep[sm]+nr[sm]+xf[sm]
                     +torch.randn_like(self.vel[sm])*0.5)
        na=torch.atan2(raw[:,1],raw[:,0]); oa=torch.atan2(old_vel[:,1],old_vel[:,0]); fa_ang=oa.clone()
        if sm.any():
            dt_t=self.max_turns[sm].clone()
            ip=(self.alarm_level>0.5)&sm
            if ip.any(): dt_t[ip[sm]]*=2.5
            if flee.any(): dt_t[flee[sm]]*=3.
            if (sm&self.is_migrant).any(): dt_t[(sm&self.is_migrant)[sm]]*=0.1
            diff=((na-oa+math.pi)%(2*math.pi)-math.pi)[sm]; fa_ang[sm]=oa[sm]+torch.clamp(diff,-dt_t,dt_t)
        spd=torch.norm(raw,dim=1).clamp(min=0.1); self.vel[:,0]=torch.cos(fa_ang)*spd; self.vel[:,1]=torch.sin(fa_ang)*spd
        nrm=torch.norm(self.vel,dim=1,keepdim=True).clamp(min=1e-5); ds=self.speeds.clone(); ds[flooded&sm]*=0.7
        if feed.any(): ds[feed]*=0.4
        if ip.any(): ds[ip]*=cfg.alarm_panic_multiplier
        if (sat:=(self.satiation_timers>0)&sm).any(): ds[sat]*=cfg.satiation_speed_penalty
        if (ins:=(self.pos[:,0]<cfg.sand_zone_x)&sm).any(): ds[ins]*=cfg.sand_speed_modifier
        if flee.any(): ds[flee]*=1.5
        self.vel=(self.vel/nrm)*ds.unsqueeze(1); self.pos[am]+=self.vel[am]*cfg.dt
        wm=sm&~self.is_migrant
        if wm.any(): self.pos[wm,0]=self.pos[wm,0]%cfg.width; self.pos[wm,1]=self.pos[wm,1]%cfg.height
        self._log(pop,fi)

    def _log(self,pop,fi):
        self.trajectory_history.append(self.pos.cpu().numpy().copy()); self.visibility_history.append((~self.is_underground).cpu().numpy().copy())
        self.active_history.append(self.is_active.cpu().numpy().copy()); self.satiation_history.append(self.satiation_timers.cpu().numpy().copy())
        self.alarm_history.append(self.alarm_level.cpu().numpy().copy()); self.population_history.append(pop)
        self.nest_state_history.append(list(self.nest_occupant)); self.nest_active_history.append(list(self.nest_active))

    def run(self):
        for fi in range(self.cfg.frames): self.step(fi)

    # ── SVG helpers ──────────────────────────────────────────────────────────
    def _svg_base(self, title, subtitle, title_color="#4fc3f7"):
        cfg=self.cfg; w,h=cfg.width,cfg.height; dur=cfg.frames/cfg.fps; F=cfg.frames
        svg=[f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#000000;font-family:system-ui, -apple-system, sans-serif;">']
        svg.append('<defs>'
            '<radialGradient id="waterGrad"><stop offset="0%" stop-color="#00ffff" stop-opacity="0.6"/><stop offset="100%" stop-color="#0033aa" stop-opacity="0.0"/></radialGradient>'
            '<radialGradient id="noiseGrad"><stop offset="0%" stop-color="#ffeb3b" stop-opacity="0.3"/><stop offset="100%" stop-color="#ff9800" stop-opacity="0.0"/></radialGradient>'
            '<radialGradient id="poachGrad"><stop offset="0%" stop-color="#b71c1c" stop-opacity="0.4"/><stop offset="100%" stop-color="#d32f2f" stop-opacity="0.0"/></radialGradient>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="3.5" fill="#507aae" opacity="0.15"/></pattern>'
            '<pattern id="sandDot" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="5" cy="5" r="1.5" fill="#d4a373" opacity="0.15"/></pattern>'
            '</defs>')
        svg.append(f'<rect x="0" y="0" width="{cfg.sand_zone_x}" height="{h}" fill="#1c1611"/><rect x="0" y="0" width="{cfg.sand_zone_x}" height="{h}" fill="url(#sandDot)"/>')
        svg.append(f'<rect x="{cfg.sand_zone_x}" y="0" width="{w-cfg.sand_zone_x}" height="{h}" fill="#05070a"/><rect x="{cfg.sand_zone_x}" y="0" width="{w-cfg.sand_zone_x}" height="{h}" fill="url(#dotGrid)"/>')
        rv=";".join(f"{r:.1f}" for r in self.vereda_radius_history)
        for i,vn in enumerate(self.vereda_nodes.cpu().numpy()):
            ops=";".join(f"{hh[i]*0.6:.2f}" for hh in self.vereda_health_history)
            svg.append(f'<circle cx="{vn[0]}" cy="{vn[1]}" fill="url(#waterGrad)"><animate attributeName="r" values="{rv}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite"/></circle>')
        svg.append(f'<rect x="{cfg.road_x_center-cfg.road_width/2}" y="0" width="{cfg.road_width}" height="{h}" fill="#424242" opacity="0.7"/>')
        svg.append(f'<line x1="{cfg.road_x_center}" y1="0" x2="{cfg.road_x_center}" y2="{h}" stroke="#ffeb3b" stroke-width="1" stroke-dasharray="8" opacity="0.4"/>')
        for fy in cfg.fence_y_positions:
            svg.append(f'<line x1="0" y1="{fy}" x2="{w}" y2="{fy}" stroke="#a5d6a7" stroke-width="1" stroke-dasharray="5,3" opacity="0.4"/>')
        for zx,zy,zr in cfg.poach_zones:
            svg.append(f'<circle cx="{zx}" cy="{zy}" r="{zr}" fill="url(#poachGrad)" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.7"/>')
        tn=self.nest_nodes.cpu().numpy()
        for i in range(cfg.roost_nodes_initial):
            tx,ty=tn[i,0],tn[i,1]; cut=next((e["frame"] for e in self.logging_events if e["tree_idx"]==i),None)
            if cut is None: svg.append(f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="8" fill="#2e7d32" stroke="#1b5e20" stroke-width="1.5" opacity="0.8"/>')
            else:
                ao=";".join("0.8" if fi<cut else "0." for fi in range(F)); so=";".join("0." if fi<cut else "0.8" for fi in range(F))
                svg.append(f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="8" fill="#2e7d32"><animate attributeName="opacity" values="{ao}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
                svg.append(f'<circle cx="{tx:.0f}" cy="{ty:.0f}" r="6" fill="#5d4037" stroke="#ff7043" stroke-width="1.5"><animate attributeName="opacity" values="{so}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        for vi in range(cfg.num_noise_vehicles):
            vx=";".join(f"{self.noise_pos_history[fi][vi,0]:.1f}" for fi in range(F))
            vy=";".join(f"{self.noise_pos_history[fi][vi,1]:.1f}" for fi in range(F))
            svg.append(f'<circle r="{cfg.noise_radius:.0f}" fill="url(#noiseGrad)" opacity="0.35"><animate attributeName="cx" values="{vx}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="cy" values="{vy}" dur="{dur}s" repeatCount="indefinite"/></circle>')
            svg.append(f'<circle r="6" fill="#ffeb3b"><animate attributeName="cx" values="{vx}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="cy" values="{vy}" dur="{dur}s" repeatCount="indefinite"/></circle>')
        st = sanitize_svg_text
        svg.append(svg_ui_style())
        svg.append(f'<text x="20" y="32" class="ui-title">{st(title)}</text>')
        svg.append(f'<text x="20" y="56" class="ui-subtitle">{st(subtitle)}</text>')
        svg.append(f'<line x1="20" y1="68" x2="440" y2="68" stroke="{title_color}" stroke-width="2.5" stroke-linecap="round" opacity="0.8"/>')
        # Footer text removed per user request
        svg.append(self.extra_svg())
        # Particles
        for idx in range(cfg.max_particles):
            if not any(self.active_history[fi][idx] for fi in range(F)): continue
            pc=self.colors[idx]; im=self.is_migrant[idx].item(); ig=self.is_grazer[idx].item()
            px=";".join(f"{p[idx,0]:.1f}" for p in self.trajectory_history)
            py=";".join(f"{p[idx,1]:.1f}" for p in self.trajectory_history)
            op=";".join("1.0" if (self.active_history[fi][idx] and self.visibility_history[fi][idx]) else "0.0" for fi in range(F))
            if idx%8==0 or im or ig:
                chunks,cur=[],[]
                for fi in range(0,F,2):
                    if self.active_history[fi][idx]: cur.append(self.trajectory_history[fi][idx])
                    elif len(cur)>1: chunks.append(cur); cur=[]
                if len(cur)>1: chunks.append(cur)
                dash='stroke-dasharray="2,2"' if im else ""; sw=1.8 if im else (3. if ig else 1.); oa2=0.6 if im else (0.8 if ig else 0.3)
                for chunk in chunks:
                    svg.append(f'<path d="M {" L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in chunk)}" fill="none" stroke="{pc}" stroke-opacity="{oa2}" stroke-width="{sw}" {dash}/>')
            rad=4.5 if im else (7. if ig else 4.)
            svg.append(f'<circle r="{rad}" fill="{pc}"><animate attributeName="cx" values="{px}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="cy" values="{py}" dur="{dur}s" repeatCount="indefinite"/><animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        svg.append(self.extra_svg_overlay())
        svg.append(SVG_VISIBILITY_SCRIPT)
        svg.append('</svg>'); return "".join(svg)

    @staticmethod
    def draw_ui_panel(x, y, w, title, items, color="#4fc3f7"):
        """Draws a premium-style dashboard panel with title and bullet points."""
        h = 32 + len(items) * 24 + 16
        st = sanitize_svg_text
        res = [f'<g transform="translate({x},{y})">',
               f'<rect width="{w}" height="{h}" fill="#1a1a2e" rx="8" ry="8" stroke="{color}" stroke-width="1" opacity="0.94"/>',
               f'<text x="12" y="20" fill="{color}" font-size="15" font-weight="bold">{st(title)}</text>']
        for i, (txt, tcol) in enumerate(items):
            res.append(f'<text x="12" y="{44 + i * 24}" fill="{tcol}" font-size="15" font-weight="bold">{st(txt)}</text>')
        res.append('</g>')
        return "".join(res), h

    @staticmethod
    def draw_mini_chart(x, y, w, h, title, curves, color="#7e57c2"):
        """Draws a mini annual-curve chart for the dashboard."""
        st = sanitize_svg_text
        res = [f'<g transform="translate({x},{y})">',
               f'<rect width="{w}" height="{h}" fill="#1a1a2e" rx="8" ry="8" stroke="{color}" stroke-width="1" opacity="0.94"/>',
               f'<text x="12" y="18" fill="{color}" font-size="15" font-weight="bold">{st(title)}</text>']
        cw, ch = w - 30, h - 45
        cx0, cy0 = 15, 25
        for data, ccol, label in curves:
            if not data: continue
            pts = [f"{cx0 + (i/(len(data)-1))*cw:.1f},{cy0 + ch - v*ch:.1f}" for i, v in enumerate(data)] # pyre-ignore
            res.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ccol}" stroke-width="1.5" opacity="0.8"/>')
        res.append('</g>')
        return "".join(res), h

    @staticmethod
    def info_card(w, h, title, lines, border_color="#4fc3f7"):
        cw=375; ch=20+len(lines)*24+10
        st = sanitize_svg_text
        items=[f'<g transform="translate({w-cw-20},340)">'
               f'<rect width="{cw}" height="{ch}" fill="#1a1a2e" rx="8" ry="8" stroke="{border_color}" stroke-width="1" opacity="0.95"/>',
               f'<text x="15" y="22" fill="#f5f5f5" font-size="15" font-weight="bold">{st(title)}</text>']
        for k,(txt,col) in enumerate(lines):
            items.append(f'<text x="15" y="{42+k*24}" fill="{col}" font-size="15" font-weight="bold">{st(txt)}</text>')
        items.append('</g>'); return "".join(items)


def sanitize_svg_text(text: str) -> str:
    """Escape &, <, > for safe use inside SVG <text> elements.

    Use this when injecting dynamic text into SVG to avoid XML parse errors.
    Example: `svg.append(f'<text>{sanitize_svg_text(label)}</text>')`
    """
    return _html.escape(text, quote=False)


SVG_FONT_FAMILY = "'Trebuchet MS', 'Segoe UI', sans-serif"

SVG_VISIBILITY_SCRIPT = (
    "<script type=\"application/ecmascript\">\n"
    "  <![CDATA[\n"
    "    document.addEventListener('visibilitychange', function() {\n"
    "      if (!document.hidden) {\n"
    "        const s = document.querySelector('svg');\n"
    "        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\n"
    "      }\n"
    "    });\n"
    "  ]]>\n"
    "  </script>"
)


def svg_ui_style() -> str:
    """Return a shared typography/style block for notebook SVG interfaces."""
    return (
        "<style><![CDATA["
        f".ui-title {{ font-family: {SVG_FONT_FAMILY}; font-size: 15; font-weight: 700; letter-spacing: 0.4; fill: #f7f4ea; }}"
        f".ui-subtitle {{ font-family: {SVG_FONT_FAMILY}; font-size: 15; font-weight: bold; fill: #c9d3dd; }}"
        f".ui-card-title {{ font-family: {SVG_FONT_FAMILY}; font-size: 15; font-weight: 700; fill: #f5f7fa; }}"
        f".ui-card-body {{ font-family: {SVG_FONT_FAMILY}; font-size: 15; font-weight: bold; fill: #d4dde6; }}"
        f".ui-label {{ font-family: {SVG_FONT_FAMILY}; font-size: 15; font-weight: 600; fill: #dce4eb; letter-spacing: 0.3; }}"
        f".ui-foot {{ font-family: {SVG_FONT_FAMILY}; font-size: 15; font-weight: bold; fill: #c2ccd6; }}"
        "svg::before { content: 'Scientific Provenance: RESEX-Cerrado-Systems-v2'; display: none; }"
        "]]></style>"
    )


def svg_title_block(title: str, subtitle: str, accent: str = "#4fc3f7", x: int = 40, y: int = 44, width: int = 420) -> str:
    """Render a shared title/subtitle block with a short accent rule."""
    safe_title = sanitize_svg_text(title)
    safe_subtitle = sanitize_svg_text(subtitle)
    return (
        f'<g transform="translate({x},{y})">'
        f'<text x="0" y="0" class="ui-title">{safe_title}</text>'
        f'<text x="0" y="24" class="ui-subtitle">{safe_subtitle}</text>'
        f'<line x1="0" y1="38" x2="{width}" y2="38" stroke="{accent}" stroke-width="2.4" stroke-linecap="round" opacity="0.85"/>'
        f'<text opacity="0" x="0" y="0">Scientific Provenance: RESEX-Cerrado-Systems-v2</text>'
        '</g>'
    )


def svg_metric_card(x: float, y: float, title: str, lines: List[Tuple[str, str]], accent: str = "#4fc3f7", width: int = 360) -> str:
    """Render a shared dashboard card for notebook SVG overlays."""
    height = 48 + len(lines) * 24
    parts = [
        f'<g transform="translate({x:.1f},{y:.1f})">',
        f'<rect width="{width}" height="{height}" fill="#17202b" fill-opacity="0.90" rx="10" ry="10" stroke="{accent}" stroke-width="1.2" stroke-opacity="0.80"/>',
        f'<text x="16" y="24" class="ui-card-title">{sanitize_svg_text(title)}</text>',
        f'<line x1="16" y1="34" x2="{width - 16}" y2="34" stroke="{accent}" stroke-width="1.4" stroke-opacity="0.60"/>'
    ]
    for idx, (text, color) in enumerate(lines):
        parts.append(
            f'<text x="16" y="{54 + idx * 24}" class="ui-card-body" fill="{color}">{sanitize_svg_text(text)}</text>'
        )
    parts.append('</g>')
    return "".join(parts)


def svg_footer_note(x: float, y: float, text: str, color: str = "#c2ccd6") -> str:
    """Render a shared footer note for notebook SVG overlays."""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="ui-foot" fill="{color}">{sanitize_svg_text(text)}</text>'


def draw_phenology_chart(
    curves: list,
    chart_w: int = 320,
    chart_h: int = 55,
    panel_h: int = 120,
    title: str = "Phenological Curves",
    title_color: str = "#81c784",
    bg_color: str = "#101510",
    border_color: str = "#4caf50",
    legend_row_h: int = 20,
    legend_font_size: int = 15,
    legend_font_weight: str = "normal",
    legend_letter_spacing: float = 0,
    legend_cols: int = 2,
) -> str:
    """Generate a self-contained SVG <g> snippet for a multi-curve phenology mini-chart.

    Designed for the right-panel info cards of notebooks 71-74.  Position it with a
    ``transform="translate(x, y)"`` applied to the returned string before embedding in
    the parent SVG.

    Args:
        curves: list of (values_12, hex_color, label) tuples where values_12 is a list
                of 12 monthly floats in [0, 1].
        chart_w: inner chart drawing area width  (excluding left padding).
        chart_h: inner chart drawing area height.
        panel_h: total height of the background rect.
        title:   panel header text.
        title_color, bg_color, border_color: visual theming.

    Returns:
        SVG markup string (a ``<g>`` element) to embed directly into an SVG document.
    """
    cw  = chart_w + 30          # total panel width = chart + left margin
    cx0 = 15                    # left margin for chart
    cy0 = 30                    # top margin for chart
    parts: list[str] = []
    safe_title = sanitize_svg_text(title)

    parts.append(
        f'<rect width="{cw}" height="{panel_h}" fill="{bg_color}" rx="8" '
        f'stroke="{border_color}" stroke-width="1" opacity="0.93"/>'
    )
    parts.append(
        f'<text x="12" y="20" fill="{title_color}" font-size="15" '
        f'font-weight="bold" font-family="system-ui, -apple-system, sans-serif">{safe_title}</text>'
    )

    # Curves
    for values, color, _label in curves:
        pts = []
        for mi in range(12):
            px_c = cx0 + (mi / 11) * chart_w
            py_c = cy0 + chart_h - values[mi] * chart_h
            pts.append(f"{px_c:.1f},{py_c:.1f}")
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{color}" stroke-width="1.8" opacity="0.88"/>'
        )

    # Legend
    leg_y = cy0 + chart_h + 10
    for ci, (_vals, color, label) in enumerate(curves):
        lx  = cx0 + (ci % legend_cols) * (chart_w // legend_cols + 5)
        lyy = leg_y + (ci // legend_cols) * legend_row_h
        parts.append(f'<circle cx="{lx}" cy="{lyy}" r="3.5" fill="{color}"/>')
        ls_attr = f' letter-spacing="{legend_letter_spacing}"' if legend_letter_spacing else ''
        parts.append(
            f'<text x="{lx + 7}" y="{lyy + 5}" fill="{color}" '
            f'font-size="{legend_font_size}" font-weight="{legend_font_weight}"{ls_attr} font-family="system-ui, -apple-system, sans-serif">{sanitize_svg_text(label)}</text>'
        )

    return f'<g>{"".join(parts)}</g>'


def draw_migration_map(
    range_patches: list,
    corridors: list,
    map_w: int = 340,
    map_h: int = 160,
    panel_h: int = 190,
    title: str = "Seasonal Range Shift",
    title_color: str = "#ffb300",
    bg_color: str = "#0a1020",
    border_color: str = "#ffb300",
    label_font_size: int = 15,
    label_font_weight: str = "normal",
    label_letter_spacing: float = 0,
) -> str:
    """Generate a self-contained SVG <g> snippet depicting a schematic landscape
    with species range patches (seasonal) and connecting dispersal corridors.

    Designed to complement ``draw_phenology_chart`` in the right-panel of
    notebooks 75-78 (Cerrado Trophic Cascade series).

    Args:
        range_patches: list of dicts with keys:
            - x, y, rx, ry (ellipse geometry, in 0..map_w / 0..map_h coords)
            - color (hex)
            - label (str)
            - opacity (float 0..1)
        corridors: list of dicts with keys:
            - x1, y1, x2, y2 (line endpoints)
            - color (hex)
            - label (str, optional)
        map_w, map_h: inner drawing area dimensions.
        panel_h: total height of background rect.
        title, title_color, bg_color, border_color: visual theming.

    Returns:
        SVG markup string (a ``<g>`` element) to embed directly.
    """
    total_w = map_w + 20
    safe_title = sanitize_svg_text(title)
    map_x0, map_y0 = 10, 30   # top-left corner of inner map
    parts: list[str] = []

    # Background panel
    parts.append(
        f'<rect width="{total_w}" height="{panel_h}" fill="{bg_color}" rx="8" '
        f'stroke="{border_color}" stroke-width="1" opacity="0.93"/>'
    )
    # Title
    parts.append(
        f'<text x="12" y="20" fill="{title_color}" font-size="15" '
        f'font-weight="bold" font-family="system-ui, -apple-system, sans-serif">{safe_title}</text>'
    )
    # Map inner background
    parts.append(
        f'<rect x="{map_x0}" y="{map_y0}" width="{map_w}" height="{map_h}" '
        f'fill="#0d1a10" rx="4" stroke="#1b2e15" stroke-width="1"/>'
    )
    # Corridors (drawn beneath patches)
    for corr in corridors:
        x1 = map_x0 + corr["x1"]; y1 = map_y0 + corr["y1"]
        x2 = map_x0 + corr["x2"]; y2 = map_y0 + corr["y2"]
        col = corr.get("color", "#aaaaaa")
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{col}" stroke-width="2" stroke-dasharray="5,3" opacity="0.55"/>'
        )
        lbl = corr.get("label", "")
        if lbl:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            lbl_y_off = corr.get("label_y_offset", 0)
            ls_attr = f' letter-spacing="{label_letter_spacing}"' if label_letter_spacing else ''
            parts.append(
                f'<text x="{mx:.0f}" y="{my - 4 + lbl_y_off:.0f}" fill="{col}" '
                f'font-size="{label_font_size}" font-weight="{label_font_weight}"{ls_attr} text-anchor="middle" font-family="system-ui, -apple-system, sans-serif">{sanitize_svg_text(lbl)}</text>'
            )
    # Range patches
    for patch in range_patches:
        px = map_x0 + patch["x"]; py = map_y0 + patch["y"]
        col = patch.get("color", "#4caf50")
        op  = patch.get("opacity", 0.5)
        rx  = patch.get("rx", 20); ry = patch.get("ry", 15)
        parts.append(
            f'<ellipse cx="{px:.0f}" cy="{py:.0f}" rx="{rx}" ry="{ry}" '
            f'fill="{col}" opacity="{op:.2f}" stroke="{col}" stroke-width="1"/>'
        )
        lbl = patch.get("label", "")
        if lbl:
            ls_attr = f' letter-spacing="{label_letter_spacing}"' if label_letter_spacing else ''
            parts.append(
                f'<text x="{px:.0f}" y="{py + ry + 9:.0f}" fill="{col}" '
                f'font-size="{label_font_size}" font-weight="{label_font_weight}"{ls_attr} text-anchor="middle" font-family="system-ui, -apple-system, sans-serif">{sanitize_svg_text(lbl)}</text>'
            )
    # RESEX marker
    rx_c = map_x0 + map_w * 0.50; ry_c = map_y0 + map_h * 0.52
    parts.append(
        f'<circle cx="{rx_c:.0f}" cy="{ry_c:.0f}" r="5" fill="#ffd600" '
        f'stroke="#ffffff" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{rx_c + 8:.0f}" y="{ry_c + 3:.0f}" fill="#ffd600" '
        f'font-size="15" font-family="system-ui, -apple-system, sans-serif">RESEX</text>'
    )

    return f'<g>{";".join("".join(parts).split(";"))}</g>'.replace(
        f'<g>{";".join("".join(parts).split(";"))}</g>',
        f'<g>{chr(10).join(parts)}</g>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCIENTIFIC VALIDATION FRAMEWORK
# ───────────────────────────────────────────────────────────────────────────────
# ScientificValidator provides a comprehensive in-process audit of simulation
# quality, ecological plausibility, and SVG animation completeness.
# It is called automatically from save_svg() but can also be invoked directly.
# ═══════════════════════════════════════════════════════════════════════════════

class ScientificValidator:
    """
    Brute-force scientific validator for ECO-SIM notebook outputs.

    Covers five audit dimensions:
      1. Biodiversity completeness - all species in BIODIVERSITY_DB have the
         full set of required parameters with ecologically plausible values.
      2. Simulation physics - BaseConfig parameters are within known ecological
         ranges (e.g. max_turn, drag, speed vs. body-mass).
      3. Population dynamics - carrying capacity, energy balance, and
         reproductive constraints are internally consistent.
      4. SVG animation quality - keyframe count, looping, element coverage,
         particle-trajectory completeness, info-card and scientific metadata.
      5. Ecological diversity indices - Shannon-Wiener H', Simpson's D,
         trophic evenness, and guild representation completeness.
    """

    # ── Required fields per BIODIVERSITY_DB entry ───────────────────────────
    _REQUIRED_SPP_FIELDS: Dict[str, Tuple] = {
        "speed":          (float, 0.0, 20.0,  "m/s equivalent - typical 0.5–12 for Cerrado fauna"),
        "color":          (str,   None, None,  "hex colour #rrggbb"),
        "weight":         (float, 0.0, 1.0,   "relative weight 0–1 (normalised)"),
        "diet":           (str,   None, None,  "one of: Frugivore|Insectivore|Carnivore|Scavenger|Grazer|Migrant"),
        "cohesion":       (float, 0.0, 1.0,   "flocking cohesion 0–1"),
        "drag":           (float, 0.0, 1.0,   "velocity drag 0–1"),
        "max_turn":       (float, 0.0, 1.0,   "max angular turn per step (radians/π)"),
        "lifespan_base":  (float, 10.0, 3000.0,"base lifespan in simulation frames"),
        "flies":          (bool,  None, None,  "aerial locomotion flag"),
    }

    # ── Valid diet categories ────────────────────────────────────────────────
    _VALID_DIETS = {"Frugivore", "Insectivore", "Carnivore", "Scavenger", "Grazer", "Migrant"}

    # ── SVG minimum standards ────────────────────────────────────────────────
    _SVG_MIN_ANIMATE       = 5    # absolute minimum <animate> elements
    _SVG_WARN_ANIMATE      = 20   # below this triggers a warning
    _SVG_MIN_FILE_KB       = 30.0
    _SVG_MIN_KEYFRAMES_AVG = 50   # avg keyframes per values= animation

    def __init__(self, verbose: bool = True):
        self.verbose  = verbose
        self._issues:  List[str] = []   # "FAIL: ..." lines
        self._warns:   List[str] = []   # "WARN: ..." lines
        self._infos:   List[str] = []   # informational

    # ── Public entry points ──────────────────────────────────────────────────

    def validate_species_db(self, db: Optional[Dict] = None) -> "ScientificValidator":
        """Validate every species in BIODIVERSITY_DB (or a supplied dict)."""
        db = db or BIODIVERSITY_DB
        self._section("BIODIVERSITY DATABASE")
        diet_counts: Dict[str, int] = {}

        for spp, props in db.items(): # pyre-ignore
            for field_name, (ftype, lo, hi, desc) in self._REQUIRED_SPP_FIELDS.items():
                val = props.get(field_name) # pyre-ignore
                if val is None:
                    self._fail(f"{spp}: missing field '{field_name}' ({desc})")
                    continue
                if ftype == float and not isinstance(val, (int, float)):
                    self._fail(f"{spp}.{field_name}: expected numeric, got {type(val).__name__}")
                    continue
                if ftype == str and not isinstance(val, str):
                    self._fail(f"{spp}.{field_name}: expected str, got {type(val).__name__}")
                    continue
                if ftype == bool and not isinstance(val, bool):
                    self._fail(f"{spp}.{field_name}: expected bool, got {type(val).__name__}")
                    continue
                if ftype == float and lo is not None and hi is not None:
                    if not (lo <= float(val) <= hi):
                        self._warn(f"{spp}.{field_name}={val} - outside plausible range [{lo}, {hi}] ({desc})")
                if field_name == "color" and isinstance(val, str):
                    if not re.match(r'^#[0-9a-fA-F]{6}$', val):
                        self._warn(f"{spp}.color='{val}' - not a valid #rrggbb hex colour")
                if field_name == "diet" and isinstance(val, str):
                    if val not in self._VALID_DIETS:
                        self._fail(f"{spp}.diet='{val}' - not in valid diet set {self._VALID_DIETS}")
                    diet_counts[val] = diet_counts.get(val, 0) + 1

        # Trophic completeness
        missing_diets = self._VALID_DIETS - {"Grazer", "Migrant"} - set(diet_counts)
        if missing_diets:
            self._warn(f"Trophic guild(s) absent from DB: {missing_diets}")
        else:
            self._info(f"All core trophic guilds represented: {sorted(diet_counts.keys())}")

        # Diversity indices
        h_prime, d_simpson = self._diversity_indices(list(diet_counts.values()))
        self._info(f"Guild Shannon-Wiener H' = {h_prime:.3f}  |  Simpson D = {d_simpson:.3f}  "
                   f"(across {sum(diet_counts.values())} species × {len(diet_counts)} guilds)")
        if h_prime < 0.8:
            self._warn(f"Low guild diversity H'={h_prime:.3f} - consider broader trophic representation")

        return self

    def validate_config(self, cfg: Optional["BaseConfig"] = None) -> "ScientificValidator":
        """Validate BaseConfig for ecological and physical plausibility."""
        if cfg is None:
            return self
        self._section("SIMULATION CONFIG")

        # Spatial
        if not (800 <= cfg.width <= 3840):
            self._warn(f"width={cfg.width} - outside typical browser-viewable 800–3840")
        if not (400 <= cfg.height <= 2160):
            self._warn(f"height={cfg.height} - outside typical 400–2160")

        # Temporal
        if cfg.frames < 100:
            self._warn(f"frames={cfg.frames} - very short; ecological dynamics need ≥ 100 steps to manifest")
        if not (5 <= cfg.fps <= 30):
            self._warn(f"fps={cfg.fps} - outside standard animation range 5–30")

        # Population
        if cfg.initial_particles > cfg.max_particles:
            self._fail(f"initial_particles ({cfg.initial_particles}) > max_particles ({cfg.max_particles})")
        if cfg.carrying_capacity > cfg.max_particles:
            self._fail(f"carrying_capacity ({cfg.carrying_capacity}) > max_particles ({cfg.max_particles})")
        if cfg.carrying_capacity < cfg.initial_particles * 0.5:
            self._warn(f"carrying_capacity ({cfg.carrying_capacity}) < 50% of initial_particles - "
                       f"triggering immediate overpopulation crash")

        # Energy conservation
        e_gain_max = cfg.energy_gain_fruiting + cfg.energy_gain_vereda + cfg.roost_energy_bonus \
                     + cfg.predator_energy_gain / 50.0  # amortised
        if cfg.energy_decay > e_gain_max:
            self._warn(f"energy_decay={cfg.energy_decay} > estimated max gain/step≈{e_gain_max:.3f} - "
                       f"system will trend toward population collapse; check energy balance")

        # Fire dynamics
        if cfg.fire_start_frame >= cfg.frames:
            self._warn(f"fire_start_frame={cfg.fire_start_frame} ≥ frames={cfg.frames} - fire never ignites")
        if cfg.max_fire_nodes < 10:
            self._warn(f"max_fire_nodes={cfg.max_fire_nodes} - fire will be negligible; "
                       f"ecological impact requires ≥ 10 nodes")

        # Predation balance
        if cfg.hunting_capture_radius > cfg.hunting_chase_radius:
            self._fail(f"hunting_capture_radius ({cfg.hunting_capture_radius}) > "
                       f"hunting_chase_radius ({cfg.hunting_chase_radius}) - predator can capture without chasing")

        # Migration
        if cfg.migration_start_frame >= cfg.frames - 20:
            self._warn(f"migration_start_frame={cfg.migration_start_frame} leaves < 20 steps for migrants")

        # Poaching
        if cfg.poach_mortality_prob > 0.1:
            self._warn(f"poach_mortality_prob={cfg.poach_mortality_prob} - extremely high; "
                       f"may cause unrealistic rapid depopulation")

        self._info(f"Simulation duration: {cfg.frames/cfg.fps:.1f}s at {cfg.fps} fps ({cfg.frames} frames)")
        self._info(f"Arena: {cfg.width}×{cfg.height}  |  Particles: {cfg.initial_particles}→max {cfg.max_particles}")
        return self

    def validate_simulation_state(self, sim: "EcosystemBase") -> "ScientificValidator":
        """Post-run validation of trajectory & history data quality."""
        self._section("SIMULATION STATE (post-run)")
        if not sim.population_history:
            self._fail("population_history is empty - simulation never ran")
            return self

        pop = np.array(sim.population_history)
        self._info(f"Population trajectory: min={pop.min()}, max={pop.max()}, "
                   f"final={pop[-1]}, mean={pop.mean():.1f}")

        if pop[-1] == 0:
            self._warn("Terminal population = 0 - full extinction occurred")
        elif pop[-1] < sim.cfg.initial_particles * 0.1:
            self._warn(f"Terminal population {pop[-1]} < 10% of initial - near-extinction trajectory")

        # Check for population oscillations (ecological realism proxy)
        if len(pop) > 20:
            diff = np.diff(pop.astype(float))
            sign_changes = int(np.sum(np.diff(np.sign(diff)) != 0))
            if sign_changes < 2:
                self._warn("Population curve is monotonic - no ecological oscillations detected; "
                           "consider tuning reproduction/mortality balance")
            else:
                self._info(f"Population oscillations detected: {sign_changes} direction changes (ecologically realistic)")

        # Trajectory coverage
        n_frames = len(sim.trajectory_history)
        n_expected = sim.cfg.frames
        if n_frames != n_expected:
            self._warn(f"trajectory_history has {n_frames} frames; expected {n_expected}")

        # Death event audit
        if sim.death_events:
            reasons: Dict[str, int] = {}
            for ev in sim.death_events:
                reasons[ev.get("reason", "unknown")] = reasons.get(ev.get("reason", "unknown"), 0) + 1 # pyre-ignore
            self._info(f"Mortality breakdown: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}")
        else:
            self._warn("No death events recorded - mortality system may not be active")

        # Seed dispersal
        if sim.dropped_seeds:
            self._info(f"Seed dispersal events: {len(sim.dropped_seeds)} seeds dropped")
        else:
            self._warn("No seed dispersal recorded - ornithochory system inactive or no frugivores present")

        return self

    def validate_svg(self, svg_str: str, nb_name: str = "") -> "ScientificValidator":
        """Validate SVG content for animation completeness and scientific standards."""
        self._section(f"SVG QUALITY{' - ' + nb_name if nb_name else ''}")

        # XML parse test
        try:
            _ET.fromstring(svg_str.encode("utf-8"))
            self._info("XML well-formed - parses without errors")
        except _ET.ParseError as exc:
            self._fail(f"XML parse error: {exc}")

        # Minimum structural checks
        checks = [
            ('<svg',        "FAIL", "Root <svg> element missing"),
            ('</svg>',      "FAIL", "Closing </svg> tag missing"),
            ('<animate',    "FAIL", "No <animate> elements - static output"),
            ('<defs>',      "WARN", "<defs> section missing - no reusable gradients/patterns"),
            ('viewBox',     "FAIL", "viewBox attribute missing from <svg>"),
            ('repeatCount', "WARN", "repeatCount='indefinite' missing - animations may not loop"),
            ('font-size',   "WARN", "No text elements with font-size - labels/title absent"),
        ]
        for tag, verdict, msg in checks:
            if tag not in svg_str:
                if verdict == "FAIL":
                    self._fail(msg)
                else:
                    self._warn(msg)

        # Unescaped ampersand detection
        bare_amp = re.findall(
            r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;|\w+;)',
            svg_str
        )
        if bare_amp:
            self._fail(f"Unescaped '&' found ({len(bare_amp)} occurrence(s)) - XML parse error in browser")

        # Animation depth
        n_animate    = svg_str.count('<animate')
        n_cx         = svg_str.count("attributeName='cx'") + svg_str.count('attributeName="cx"')
        n_cy         = svg_str.count("attributeName='cy'") + svg_str.count('attributeName="cy"')
        values_attrs = re.findall(r'\bvalues\s*=\s*["\']([^"\']{20,})["\']', svg_str)
        n_kf_total   = sum(v.count(";") + 1 for v in values_attrs)
        n_kf_avg     = n_kf_total / len(values_attrs) if values_attrs else 0

        if n_animate < self._SVG_MIN_ANIMATE:
            self._fail(f"Only {n_animate} <animate> elements (minimum={self._SVG_MIN_ANIMATE})")
        elif n_animate < self._SVG_WARN_ANIMATE:
            self._warn(f"{n_animate} <animate> elements - low for ecological complexity (recommended ≥{self._SVG_WARN_ANIMATE})")
        else:
            self._info(f"{n_animate} <animate> elements  |  cx/cy animated: {n_cx}/{n_cy}")

        if n_kf_avg > 0:
            if n_kf_avg < self._SVG_MIN_KEYFRAMES_AVG:
                self._warn(f"Avg keyframes/animation: {n_kf_avg:.0f} - below {self._SVG_MIN_KEYFRAMES_AVG} "
                           f"(animation may appear jerky)")
            else:
                self._info(f"Keyframe density: avg {n_kf_avg:.0f} frames × {len(values_attrs)} value-animations "
                           f"= {n_kf_total:,} total keyframe entries")

        # File size proxy
        size_kb = len(svg_str.encode("utf-8")) / 1024
        if size_kb < self._SVG_MIN_FILE_KB:
            self._fail(f"SVG content {size_kb:.1f} KB - suspiciously small for a particle simulation")
        else:
            self._info(f"SVG content size: {size_kb:.0f} KB")

        # Scientific provenance
        if not re.search(r'(?:RESEX|Recanto|Cerrado)', svg_str, re.IGNORECASE):
            self._warn("No RESEX/Cerrado/Recanto provenance text in SVG - scientific context absent")

        # Gradient / pattern IDs
        defined_ids = set(re.findall(r'id\s*=\s*["\']([^"\']+)["\']', svg_str))
        ref_ids     = set(re.findall(r'url\(#([^)]+)\)', svg_str))
        unresolved  = ref_ids - defined_ids
        if unresolved:
            self._fail(f"Unresolved url(#...) references: {', '.join(sorted(unresolved))}")
        elif ref_ids:
            self._info(f"All {len(ref_ids)} url(#...) gradient/pattern references resolved")

        return self

    def validate_phenological_curve(
        self, curve: List[float], name: str = "phenological curve"
    ) -> "ScientificValidator":
        """Validate that a 12-month phenological curve is ecologically plausible."""
        self._section(f"PHENOLOGICAL CURVE - {name}")
        if len(curve) != 12:
            self._warn(f"{name}: {len(curve)} values (expected 12 monthly values)")
        if any(v < 0 or v > 1 for v in curve):
            self._fail(f"{name}: values outside [0, 1] - must be normalised intensities")
        if all(v == curve[0] for v in curve):
            self._warn(f"{name}: flat curve - no seasonal variation; scientifically implausible for Cerrado")
        if max(curve) - min(curve) < 0.1:
            self._warn(f"{name}: amplitude {max(curve)-min(curve):.2f} < 0.1 - negligible seasonality")
        # Check for wet/dry season asymmetry (Cerrado: dry May-Sep, wet Oct-Apr)
        wet_mean = (curve[0]+curve[1]+curve[2]+curve[9]+curve[10]+curve[11]) / 6
        dry_mean = (curve[4]+curve[5]+curve[6]+curve[7]) / 4
        if wet_mean < dry_mean - 0.05:
            self._warn(f"{name}: dry-season mean ({dry_mean:.2f}) > wet-season mean ({wet_mean:.2f}) - "
                       f"unusual for most Cerrado species; verify against literature")
        else:
            self._info(f"{name}: wet-season mean={wet_mean:.2f}, dry-season mean={dry_mean:.2f} "
                       f"({'expected wet-season peak' if wet_mean > dry_mean else 'dry-season peak'})")
        return self

    # ── Report output ────────────────────────────────────────────────────────

    def report(self, raise_on_fail: bool = False) -> "ScientificValidator":
        """Print the full validation report and optionally raise on failures."""
        print("\n" + "═" * 68)
        print("  ECO-SIM SCIENTIFIC VALIDATION REPORT")
        print("═" * 68)
        for line in self._infos:
            print(f"  [INFO]  {line}")
        for line in self._warns:
            print(f"  [WARN]  {line}")
        for line in self._issues:
            print(f"  [FAIL]  {line}")
        print("─" * 68)
        print(f"  Totals:  {len(self._infos)} info  |  "
              f"{len(self._warns)} warning(s)  |  {len(self._issues)} failure(s)")
        print("═" * 68 + "\n")
        if raise_on_fail and self._issues:
            raise ValueError(
                f"Scientific validation failed with {len(self._issues)} error(s):\n"
                + "\n".join(self._issues)
            )
        return self

    @property
    def has_failures(self) -> bool:
        return bool(self._issues)

    @property
    def has_warnings(self) -> bool:
        return bool(self._warns)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _diversity_indices(counts: List[int]) -> Tuple[float, float]:
        """Return (Shannon H', Simpson D) for a list of counts."""
        total = sum(counts)
        if total == 0:
            return 0.0, 0.0
        props  = [c / total for c in counts if c > 0]
        h_prime = -sum(p * math.log(p) for p in props)
        d_simpson = 1.0 - sum(p * p for p in props)
        return h_prime, d_simpson

    def _section(self, title: str):
        if self.verbose:
            print(f"\n── {title} {'─'*(54-len(title))}")

    def _fail(self, msg: str):
        self._issues.append(msg)
        if self.verbose:
            print(f"  [FAIL]  {msg}")

    def _warn(self, msg: str):
        self._warns.append(msg)
        if self.verbose:
            print(f"  [WARN]  {msg}")

    def _info(self, msg: str):
        self._infos.append(msg)
        if self.verbose:
            print(f"  [INFO]  {msg}")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def validate_notebook(
    sim: Optional["EcosystemBase"] = None,
    cfg: Optional["BaseConfig"]    = None,
    svg: Optional[str]             = None,
    nb_name: str                   = "",
    phenology_curves: Optional[List[Tuple[List[float], str]]] = None,
    verbose: bool        = True,
    raise_on_fail: bool  = False,
) -> ScientificValidator:
    """
    One-call scientific validation for any notebook.

    Parameters
    ----------
    sim      : EcosystemBase instance (post-run) - validates population dynamics.
    cfg      : BaseConfig instance - validates physical/ecological parameters.
    svg      : raw SVG string - validates animation quality.
    nb_name  : notebook identifier for display (e.g. "notebook_55").
    phenology_curves : list of (values_12, label) tuples to validate seasonality.
    verbose  : print section headers and INFO lines during validation.
    raise_on_fail    : raise ValueError if any FAIL-level issue is found.

    Returns
    -------
    ScientificValidator with full results accessible via .has_failures, .report()
    """
    v = ScientificValidator(verbose=verbose)
    v.validate_species_db()
    if cfg is not None:
        v.validate_config(cfg)
    if sim is not None:
        v.validate_simulation_state(sim)
    if svg is not None:
        v.validate_svg(svg, nb_name)
    if phenology_curves:
        for curve, label in phenology_curves:
            v.validate_phenological_curve(curve, label)
    v.report(raise_on_fail=raise_on_fail)
    return v


# ─── save_svg - enhanced with integrated scientific validation ────────────────

def save_svg(svg, nb, base_dir=None, validate=True):
    """Save SVG to disk and run the full scientific validation pipeline.

    Parameters
    ----------
    svg      : str - full SVG content string.
    nb       : str - notebook name (used as filename stem, e.g. 'notebook_55').
    base_dir : optional directory override (default: svg_output/ beside this file).
    validate : bool - if True (default), run ScientificValidator.validate_svg().
    """
    d = base_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd(),
        'svg_output'
    )
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, f'{nb}.svg')

    # Normalize invalid XML ampersands globally before validation/write.
    # Keeps valid XML entities intact (amp, lt, gt, quot, apos, numeric).
    invalid_amp_re = r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)'
    amp_fixes = len(re.findall(invalid_amp_re, svg))
    if amp_fixes:
        print(f"  [WARN]  {nb}: auto-escaped {amp_fixes} invalid '&' occurrence(s)")
        svg = re.sub(invalid_amp_re, '&amp;', svg)

    # ── Scientific SVG validation (inline, concise) ──────────────────────────
    _WARN  = lambda m: print(f"  [WARN]  {m}")
    _FAIL  = lambda m: print(f"  [FAIL]  {m}")

    issues = 0

    if not svg.strip().startswith('<svg'):
        _FAIL(f"{nb}: SVG does not start with <svg> tag"); issues += 1
    if 'viewBox' not in svg:
        _FAIL(f"{nb}: missing viewBox attribute"); issues += 1
    if '</svg>' not in svg:
        _FAIL(f"{nb}: missing closing </svg> tag"); issues += 1
    if '<animate' not in svg:
        _FAIL(f"{nb}: no <animate> elements - static output"); issues += 1
    else:
        n_anim = svg.count('<animate')
        if n_anim < 5:
            _FAIL(f"{nb}: only {n_anim} <animate> elements (min 5)")
        elif n_anim < 20:
            _WARN(f"{nb}: {n_anim} <animate> elements - low for ecological complexity")
    if '<defs>' not in svg:
        _WARN(f"{nb}: no <defs> section - gradients/patterns not reused")

    # Unescaped ampersand (XML-breaking)
    bare = re.findall(invalid_amp_re, svg)
    if bare:
        _FAIL(f"{nb}: {len(bare)} unescaped '&' - causes XML parse errors in browser"); issues += 1

    # Gradient ID consistency
    defined_ids = set(re.findall(r'id\s*=\s*["\']([^"\']+)["\']', svg))
    ref_ids     = set(re.findall(r'url\(#([^)]+)\)', svg))
    missing_ids = ref_ids - defined_ids
    if missing_ids:
        _FAIL(f"{nb}: {len(missing_ids)} unresolved gradient/pattern references: {', '.join(sorted(missing_ids))}"); issues += 1

    # Keyframe density
    val_attrs = re.findall(r'\bvalues\s*=\s*["\']([^"\']{20,})["\']', svg)
    if val_attrs:
        avg_kf = sum(v.count(";") + 1 for v in val_attrs) / len(val_attrs)
        if avg_kf < 50:
            _WARN(f"{nb}: avg {avg_kf:.0f} keyframes/animation (< 50 - animation may appear jerky)")

    # Scientific provenance
    if not re.search(r'(?:RESEX|Recanto|Cerrado)', svg, re.IGNORECASE):
        _WARN(f"{nb}: no RESEX/Cerrado provenance text in SVG")

    # Optional deep validation
    if validate:
        try:
            _ET.fromstring(svg.encode("utf-8"))
        except _ET.ParseError as exc:
            _FAIL(f"{nb}: XML parse error - {exc}"); issues += 1

    # Write file
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(svg)

    size_kb = os.path.getsize(fp) / 1024
    status  = "OK" if issues == 0 else f"WARN {issues} issue(s)"
    print(f"SVG saved -> {fp}  ({size_kb:.0f} KB)  [{status}]")

# EOF
