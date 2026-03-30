# -*- coding: utf-8 -*-
# MODULE 54: The Living Landscape (Grand Finale - all modules combined) - 730 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: The Living Landscape (Grand Finale - all modules combined) - 730 frames emphasizing hunting pressure corridors.
- Indicator species: Fungo-micorrizico (Pisolithus sp.).
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

import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, math, random # pyre-ignore[21]
from dataclasses import dataclass
from typing import List, Dict, Set
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 730
    # ── Agroforestry Rows (37) ──────────────────────────────────────────────
    agro_rows: tuple = (100.0, 300.0, 500.0)
    agro_cols: int = 7
    agro_x_start: float = 100.0; agro_x_end: float = 1180.0
    agro_attract_radius: float = 110.0; agro_attract_force: float = 3.0
    agro_energy_gain: float = 2.5
    # ── Corridor (39) ───────────────────────────────────────────────────────
    corridor_y: float = 300.0
    corridor_x_start: float = 300.0; corridor_x_end: float = 980.0
    corridor_nodes: int = 7
    corridor_reveal_interval: int = 50; corridor_reveal_start: int = 80
    corridor_attract_radius: float = 100.0; corridor_attract_force: float = 2.8
    # ── Camera Traps (46) ───────────────────────────────────────────────────
    camera_locations: tuple = ((300.0,200.0),(750.0,450.0),(1100.0,150.0))
    camera_radius: float = 60.0; camera_flash_frames: int = 4
    # ── Jaguar (49) ─────────────────────────────────────────────────────────
    jaguar_start_frame: int = 200
    jaguar_speed: float = 4.5; jaguar_fear_radius: float = 180.0
    jaguar_fear_force: float = 18.0; jaguar_alarm_radius: float = 260.0
    jaguar_chase_radius: float = 90.0; jaguar_kill_radius: float = 12.0
    jaguar_satiation_dur: int = 100; jaguar_energy_gain: float = 60.0
    # ── Maned Wolf (50) ─────────────────────────────────────────────────────
    wolf_start_frame: int = 60
    wolf_speed: float = 5.0; wolf_fear_radius: float = 80.0; wolf_fear_force: float = 7.0
    wolf_seed_drop_interval: int = 40; wolf_seed_germinate_delay: int = 70
    wolf_lobeira_attract_radius: float = 120.0
    # ── Controlled Burn (38) ────────────────────────────────────────────────
    burn_interval: int = 120
    burn_radius: float = 50.0; burn_duration: int = 25
    burn_flee_radius: float = 80.0; burn_flee_force: float = 9.0
    # ── Zoning (52) - (static bkg) ──────────────────────────────────────────
    pres_zone: tuple = (20, 50, 280, 500)    # preservation (left)
    use_zone:  tuple = (980, 50, 280, 500)   # sustainable use (right)

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg

        # Agroforestry nodes
        xs_agro = [cfg.agro_x_start + i*(cfg.agro_x_end-cfg.agro_x_start)/(cfg.agro_cols-1) for i in range(cfg.agro_cols)]
        self.agro_nodes = torch.tensor([[x, y] for y in cfg.agro_rows for x in xs_agro], device=self.dev, dtype=torch.float32)

        # Corridor nodes (progressive)
        xs_cor = [cfg.corridor_x_start + i*(cfg.corridor_x_end-cfg.corridor_x_start)/(cfg.corridor_nodes-1) for i in range(cfg.corridor_nodes)]
        self.corridor_pts = torch.tensor([[x, cfg.corridor_y] for x in xs_cor], device=self.dev, dtype=torch.float32)
        self.corridor_xs = xs_cor
        self.active_corridor: List[bool] = [False]*cfg.corridor_nodes

        # Camera traps
        self.cameras = torch.tensor(cfg.camera_locations, device=self.dev, dtype=torch.float32)
        self.identified: Set[int] = set()
        self.camera_flashes: List[Dict] = []
        self.identified_history: List[List[bool]] = []

        # Jaguar
        self.jaguar = {"pos": torch.tensor([640., 300.], device=self.dev, dtype=torch.float32),
                       "angle": 0.5, "satiation": 0}
        self.jaguar_history: List[Dict] = []
        self.jaguar_kills = 0

        # Maned Wolf
        self.wolf = {"pos": torch.tensor([200., 400.], device=self.dev, dtype=torch.float32),
                     "angle": 1.0, "last_seed": 0}
        self.wolf_history: List[Dict] = []
        self.lobeira_sites: List[Dict] = []

        # Controlled burns
        self.burn_history: List[Dict] = []
        self.active_burns: List[Dict] = []

        # Population history for sparkline
        self.pop_history: List[int] = []

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg

        # ── Corridor reveal ──────────────────────────────────────────────────
        for i in range(cfg.corridor_nodes):
            if fi >= cfg.corridor_reveal_start + i * cfg.corridor_reveal_interval:
                self.active_corridor[i] = True

        # ── Camera traps ─────────────────────────────────────────────────────
        if am.any() and self.flies.any():
            birds = am & self.flies
            if birds.any():
                bird_idx = birds.nonzero().squeeze(1)
                dm = torch.cdist(self.pos[birds], self.cameras)
                for ci in range(len(cfg.camera_locations)):
                    in_range = dm[:, ci] < cfg.camera_radius
                    if in_range.any():
                        new_finds = [int(bird_idx[k]) for k in in_range.nonzero().squeeze(1) if int(bird_idx[k]) not in self.identified]
                        if new_finds:
                            self.identified.update(new_finds)
                            self.camera_flashes.append({"cam_idx": ci, "frame": fi})
        self.identified_history.append([i in self.identified for i in range(cfg.max_particles)])

        # ── Jaguar ───────────────────────────────────────────────────────────
        if fi >= cfg.jaguar_start_frame:
            j = self.jaguar
            j["satiation"] = max(0, j["satiation"] - 1)
            prey_mask = sm & ~self.is_grazer & ~self.is_carnivore & ~self.is_migrant
            if j["satiation"] == 0 and prey_mask.any():
                d_p = torch.norm(self.pos[prey_mask] - j["pos"], dim=1)
                md, mi = torch.min(d_p, dim=0)
                pi = prey_mask.nonzero().squeeze(1)[mi]
                v = self.pos[pi] - j["pos"]
                dist = torch.norm(v).item()
                j["angle"] = math.atan2(v[1].item(), v[0].item())
                j["pos"] += (v / max(dist, 1e-5)) * (cfg.jaguar_speed * (1.5 if dist < cfg.jaguar_chase_radius else 0.8))
                if dist < cfg.jaguar_kill_radius and self.is_active[pi]:
                    self.is_active[pi] = False
                    j["satiation"] = cfg.jaguar_satiation_dur
                    self.jaguar_kills += 1
                    pn = self.pos[pi].cpu().numpy().copy()
                    self.carrion_sites.append({"pos": pn, "spawn_frame": fi, "expire_frame": fi + cfg.carrion_linger_frames})
            else:
                j["angle"] += random.uniform(-0.1, 0.1)
                j["pos"] += torch.tensor([math.cos(j["angle"]), math.sin(j["angle"])], device=self.dev) * 1.5
            j["pos"][0] = torch.clamp(j["pos"][0], 20, cfg.width-20)
            j["pos"][1] = torch.clamp(j["pos"][1], 20, cfg.height-20)
            self.jaguar_history.append({"x": j["pos"][0].item(), "y": j["pos"][1].item()})
        else:
            self.jaguar_history.append({"x": 0.0, "y": 0.0})

        # ── Maned Wolf ───────────────────────────────────────────────────────
        if fi >= cfg.wolf_start_frame:
            w = self.wolf
            w["angle"] += random.uniform(-0.18, 0.18) + (0.5 if random.random() < 0.02 else 0)
            w["pos"] += torch.tensor([math.cos(w["angle"]), math.sin(w["angle"])], device=self.dev) * cfg.wolf_speed
            w["pos"][0] = w["pos"][0] % cfg.width
            w["pos"][1] = w["pos"][1] % cfg.height
            if fi - w["last_seed"] >= cfg.wolf_seed_drop_interval:
                self.lobeira_sites.append({"pos": w["pos"].clone(), "x": w["pos"][0].item(), "y": w["pos"][1].item(),
                                           "plant_frame": fi, "sprout_frame": fi + cfg.wolf_seed_germinate_delay})
                w["last_seed"] = fi
            self.wolf_history.append({"x": w["pos"][0].item(), "y": w["pos"][1].item()})
        else:
            self.wolf_history.append({"x": cfg.width/4, "y": cfg.height/2})

        # ── Controlled Burns ─────────────────────────────────────────────────
        if fi > 0 and fi % cfg.burn_interval == 0:
            bx, by = random.uniform(100, cfg.width-100), random.uniform(80, cfg.height-80)
            self.active_burns.append({"x": bx, "y": by, "start": fi, "end": fi + cfg.burn_duration})
            self.burn_history.append({"x": bx, "y": by, "frame": fi})
        self.active_burns = [b for b in self.active_burns if b["end"] > fi]

        # Lobeira feeding
        for site in self.lobeira_sites:
            if fi >= site["sprout_frame"]:
                d = torch.norm(self.pos - site["pos"], dim=1)
                near = sm & (d < 15.0) & (self.is_frugivore | self.is_insectivore)
                if near.any():
                    self.energy[near] += 2.0

        self.pop_history.append(int(am.sum()))

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg

        # Agroforestry rows
        d_agro = torch.cdist(self.pos, self.agro_nodes)
        mn_a, cl_a = torch.min(d_agro, dim=1)
        near_a = sm & (mn_a < cfg.agro_attract_radius) & (self.is_frugivore | self.is_insectivore)
        if near_a.any():
            pull = self.agro_nodes[cl_a[near_a]] - self.pos[near_a]
            f[near_a] += (pull / mn_a[near_a].unsqueeze(1).clamp(min=1.)) * cfg.agro_attract_force
            feeding = near_a & (mn_a < 18.0)
            if feeding.any():
                self.energy[feeding] += cfg.agro_energy_gain

        # Corridor
        act_idx = [i for i, a in enumerate(self.active_corridor) if a]
        if act_idx:
            act_pts = self.corridor_pts[torch.tensor(act_idx, device=self.dev)]
            d_c = torch.cdist(self.pos, act_pts)
            mn_c, cl_c = torch.min(d_c, dim=1)
            near_c = sm & ~self.is_grazer & ~self.is_migrant & (mn_c < cfg.corridor_attract_radius)
            if near_c.any():
                pull = act_pts[cl_c[near_c]] - self.pos[near_c]
                f[near_c] += (pull / mn_c[near_c].unsqueeze(1).clamp(min=1.)) * cfg.corridor_attract_force

        # Jaguar flee
        if fi >= cfg.jaguar_start_frame:
            jp = self.jaguar["pos"]
            d_j = torch.norm(self.pos - jp, dim=1)
            flee_j = sm & (d_j < cfg.jaguar_fear_radius)
            if flee_j.any():
                away = self.pos[flee_j] - jp
                strength = (1.0 - d_j[flee_j] / cfg.jaguar_fear_radius) ** 1.5
                f[flee_j] += (away / d_j[flee_j].unsqueeze(1).clamp(min=1.)) * strength.unsqueeze(1) * cfg.jaguar_fear_force
                self.alarm_level[flee_j] = torch.clamp(self.alarm_level[flee_j] + 0.6, max=1.0)
            ripple = sm & (d_j >= cfg.jaguar_fear_radius) & (d_j < cfg.jaguar_alarm_radius)
            if ripple.any():
                self.alarm_level[ripple] = torch.clamp(self.alarm_level[ripple] + 0.12, max=1.0)

        # Wolf shy fear
        if fi >= cfg.wolf_start_frame:
            wp = self.wolf["pos"]
            d_w = torch.norm(self.pos - wp, dim=1)
            flee_w = sm & (d_w < cfg.wolf_fear_radius)
            if flee_w.any():
                away = self.pos[flee_w] - wp
                f[flee_w] += (away / d_w[flee_w].unsqueeze(1).clamp(min=1.)) * cfg.wolf_fear_force
                self.alarm_level[flee_w] = torch.clamp(self.alarm_level[flee_w] + 0.2, max=1.0)

        # Lobeira attraction
        live_sites = [s for s in self.lobeira_sites if fi >= s["sprout_frame"]]
        if live_sites and (sm & (self.is_frugivore | self.is_insectivore)).any():
            lp = torch.stack([s["pos"] for s in live_sites])
            d_l = torch.cdist(self.pos, lp)
            mn_l, cl_l = torch.min(d_l, dim=1)
            near_l = sm & (self.is_frugivore | self.is_insectivore) & (mn_l < cfg.wolf_lobeira_attract_radius)
            if near_l.any():
                pull = lp[cl_l[near_l]] - self.pos[near_l]
                f[near_l] += (pull / mn_l[near_l].unsqueeze(1).clamp(min=1.)) * 2.0

        # Controlled burn flee
        for b in self.active_burns:
            bp = torch.tensor([b["x"], b["y"]], device=self.dev)
            d_b = torch.norm(self.pos - bp, dim=1)
            fb = sm & (d_b < cfg.burn_flee_radius)
            if fb.any():
                away = self.pos[fb] - bp
                f[fb] += (away / d_b[fb].unsqueeze(1).clamp(min=1.)) * cfg.burn_flee_force
                self.alarm_level[fb] = torch.clamp(self.alarm_level[fb] + 0.35, max=1.0)

        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Zone backgrounds (preservation green / sustainable use amber)
        for (zx, zy, zw, zh), col, lbl, tc in [
            (cfg.pres_zone, "#1b5e20", "Preservation", "#c8e6c9"),
            (cfg.use_zone,  "#f57f17", "Sustainable Use", "#ffe082"),
        ]:
            out.append(f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" fill="{col}" rx="6" opacity="0.14"/>')
            out.append(f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" fill="none" stroke="{col}" stroke-width="2" rx="6" opacity="0.5"/>')
            out.append(f'<text x="{zx+zw/2:.0f}" y="{zy+22:.0f}" text-anchor="middle" font-size="15" font-weight="bold" fill="{tc}" opacity="0.8">{lbl}</text>')

        # Agroforestry rows
        for ry in cfg.agro_rows:
            out.append(f'<line x1="{cfg.agro_x_start}" y1="{ry}" x2="{cfg.agro_x_end}" y2="{ry}" stroke="#8bc34a" stroke-width="1" stroke-dasharray="5,5" opacity="0.35"/>')
        for p in self.agro_nodes.cpu().numpy():
            out.append(f'<circle cx="{p[0]:.0f}" cy="{p[1]:.0f}" r="5" fill="#8bc34a" stroke="#33691e" stroke-width="1.2" opacity="0.75"/>')

        # Corridor progressive reveal
        for i, x in enumerate(self.corridor_xs):
            reveal_at = cfg.corridor_reveal_start + i*cfg.corridor_reveal_interval
            ops = ";".join("0.0" if fi < reveal_at else "0.9" for fi in range(F))
            out.append(f'<circle cx="{x:.0f}" cy="{cfg.corridor_y:.0f}" r="7" fill="#4db6ac" stroke="#00695c" stroke-width="1.5" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Lobeira sprouts
        for site in self.lobeira_sites:
            sf = site["sprout_frame"]
            ops_s = ";".join("0.0" if fi < sf else ("0.4" if fi < sf+10 else "0.8") for fi in range(F))
            rv = ";".join("0" if fi < sf else f"{min(8.0,(fi-sf)/12.0*8.0):.1f}" for fi in range(F))
            out.append(f'<circle cx="{site["x"]:.0f}" cy="{site["y"]:.0f}" fill="#7cb342" opacity="0.0">'
                       f'<animate attributeName="r" values="{rv}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{ops_s}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Controlled burns
        for bh in self.burn_history:
            bx, by, bf = bh["x"], bh["y"], bh["frame"]
            bops = ";".join(f"{max(0.,0.45*math.sin((fi-bf)/cfg.burn_duration*math.pi)):.2f}" if bf <= fi < bf+cfg.burn_duration else "0.0" for fi in range(F))
            out.append(f'<circle cx="{bx:.0f}" cy="{by:.0f}" r="{cfg.burn_radius:.0f}" fill="#ff6f00" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{bops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Camera traps
        for ci, (cx, cy) in enumerate(cfg.camera_locations):
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{cfg.camera_radius}" fill="none" stroke="#e0e0e0" stroke-dasharray="4,4" stroke-width="1" opacity="0.3"/>')
            out.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#90caf9" opacity="0.8"/>')
            flash_ops = ";".join(f"{max(0.,1.0-(fi - next((f['frame'] for f in self.camera_flashes if f['cam_idx']==ci and 0<=fi-f['frame']<cfg.camera_flash_frames), fi))/cfg.camera_flash_frames):.2f}" if any(f["cam_idx"]==ci and 0<=fi-f["frame"]<cfg.camera_flash_frames for f in self.camera_flashes) else "0.0" for fi in range(F))
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{cfg.camera_radius}" fill="#fff" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{flash_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Tracking brackets on identified birds
        for idx in range(cfg.max_particles):
            if not self.flies[idx].item() or not any(self.identified_history[fi][idx] for fi in range(F)): continue
            xs = ";".join(f"{self.trajectory_history[fi][idx,0]:.1f}" for fi in range(F))
            ys = ";".join(f"{self.trajectory_history[fi][idx,1]:.1f}" for fi in range(F))
            ops = ";".join("1.0" if (self.active_history[fi][idx] and self.identified_history[fi][idx]) else "0.0" for fi in range(F))
            out.append(f'<circle r="9" fill="none" stroke="#fff" stroke-width="1.2" stroke-dasharray="3,2" opacity="0.0">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Jaguar
        jxs = ";".join(f"{self.jaguar_history[fi]['x']:.1f}" for fi in range(F))
        jys = ";".join(f"{self.jaguar_history[fi]['y']:.1f}" for fi in range(F))
        jvis = ";".join("1.0" if fi >= cfg.jaguar_start_frame else "0.0" for fi in range(F))
        jvis_z = jvis.replace("1.0", "0.3")
        out.append(f'<circle r="{cfg.jaguar_fear_radius:.0f}" fill="#b71c1c" opacity="0.0">'
                   f'<animate attributeName="cx" values="{jxs}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="cy" values="{jys}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="{jvis_z}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        out.append(f'<circle r="11" fill="#e65100" stroke="#ffd54f" stroke-width="2">'
                   f'<animate attributeName="cx" values="{jxs}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="cy" values="{jys}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="{jvis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Maned Wolf
        wxs = ";".join(f"{self.wolf_history[fi]['x']:.1f}" for fi in range(F))
        wys = ";".join(f"{self.wolf_history[fi]['y']:.1f}" for fi in range(F))
        wvis = ";".join("1.0" if fi >= cfg.wolf_start_frame else "0.0" for fi in range(F))
        out.append(f'<circle r="9" fill="#795548" stroke="#ffd54f" stroke-width="2">'
                   f'<animate attributeName="cx" values="{wxs}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="cy" values="{wys}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="opacity" values="{wvis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Population sparkline (bottom strip)
        max_pop = max(self.pop_history) if self.pop_history else 1
        bar_w = cfg.width / F
        for fi in range(F):
            ph = (self.pop_history[fi] / max_pop) * 40
            col = "#4caf50"
            out.append(f'<rect x="{fi*bar_w:.1f}" y="{cfg.height-ph:.1f}" width="{bar_w+0.5:.1f}" height="{ph:.1f}" fill="{col}" opacity="0.6"/>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg; F = cfg.frames
        final_pop = self.pop_history[-1] if self.pop_history else 0
        seeds = len(self.lobeira_sites)
        sparked = sum(1 for s in self.lobeira_sites if s["sprout_frame"] <= F)
        identified = len(self.identified)
        # Grand summary card
        lines = [
            ("Agroforestry rows attract frugivores and insectivores.", "#dcedc8"),
            (f"{cfg.corridor_nodes} corridor nodes bridge habitat patches progressively.", "#b2dfdb"),
            (f"Camera traps: {identified} birds identified.", "#bbdefb"),
            (f"Jaguar roams from frame {cfg.jaguar_start_frame}, kills: {self.jaguar_kills}.", "#ffccbc"),
            (f"Maned Wolf seeds {seeds} Lobeira; {sparked} sprouted.", "#fff9c4"),
            (f"{len(self.burn_history)} controlled burns pulse across landscape.", "#ffe0b2"),
            (f"Final population: {final_pop} | Green sparkline = bottom strip.", "#e0e0e0"),
        ]
        return self.info_card(cfg.width, cfg.height, "The Living Landscape - Grand Finale", lines, "#4caf50")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: The Living Landscape", "All features unified - Recanto das Araras in full ecological complexity", "#4caf50")


# -- SVG Export to Google Drive --------------------------------------------
def save_svg_to_drive(svg_content: str, notebook_id: str):
    """Persist SVG artefact to Google Drive for publication on Google Sites."""
    import os
    drive_folder = "/content/drive/MyDrive/ReservaAraras_SVGs"
    # Colab: mount drive first; local: use fallback folder
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
    print(f"SVG saved ->→ {filepath}")
    return filepath

def main():
    """Function `main` -- simulation component."""

    print(f" - The Living Landscape on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print(f"Done: pop={sim.pop_history[-1]}, jaguar kills={sim.jaguar_kills},"
          f" wolf seeds={len(sim.lobeira_sites)}, birds tagged={len(sim.identified)}")
    print("Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_54')

if __name__ == "__main__": main()
