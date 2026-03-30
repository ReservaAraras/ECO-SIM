# -*- coding: utf-8 -*-
# MODULE 50: Maned Wolf Dispersal (Wolf agent moves far, planting Lobeira seeds) - 690 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Maned Wolf Dispersal (Wolf agent moves far, planting Lobeira seeds) - 690 frames emphasizing forest-savanna ecotones.
- Indicator species: Besouro-rola-bosta (Scarabaeinae sp.).
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

import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, math, random # pyre-ignore[21]
from dataclasses import dataclass
from typing import List, Dict
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 690
    # Maned Wolf parameters
    wolf_count: int = 2
    wolf_speed: float = 5.5
    wolf_fear_radius: float = 90.0    # smaller than jaguar, wolf is shy
    wolf_fear_force: float = 8.0
    wolf_seed_drop_interval: int = 35  # frames between seed drops
    wolf_seed_germinate_delay: int = 80 # frames before a lobeira sprout appears
    wolf_lobeira_attract_radius: float = 130.0
    wolf_start_frame: int = 30
    # Lobeira fruit can be eaten by birds for bonus energy
    lobeira_energy_bonus: float = 5.0

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        self.wolves: List[Dict] = []
        for i in range(cfg.wolf_count):
            self.wolves.append({
                "pos": torch.tensor([
                    cfg.width * (i + 1) / (cfg.wolf_count + 1),
                    cfg.height * 0.5
                ], device=self.dev, dtype=torch.float32),
                "vel": torch.zeros(2, device=self.dev, dtype=torch.float32),
                "angle": random.uniform(0, math.pi * 2),
                "last_seed_drop": 0,
            })
        self.wolf_history: List[List[Dict]] = []

        # Lobeira plants: each has a position, spawn_frame, and active state
        self.lobeira_sites: List[Dict] = []

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        if fi < cfg.wolf_start_frame:
            self.wolf_history.append([{"x": w["pos"][0].item(), "y": w["pos"][1].item()} for w in self.wolves])
            return

        frame_pos = []
        for w in self.wolves:
            # Smooth wandering with occasional sharp turns (simulates long-range travel)
            turn = random.uniform(-0.15, 0.15)
            if random.random() < 0.02:  # 2% chance of a big course change
                turn = random.uniform(-1.0, 1.0)
            w["angle"] += turn
            w["pos"] += torch.tensor([math.cos(w["angle"]), math.sin(w["angle"])], device=self.dev) * cfg.wolf_speed

            # Wrap around map edges (mimics wide-ranging territory)
            w["pos"][0] = w["pos"][0] % cfg.width
            w["pos"][1] = w["pos"][1] % cfg.height

            # Drop a lobeira seed periodically
            if (fi - w["last_seed_drop"]) >= cfg.wolf_seed_drop_interval:
                site = {
                    "pos": w["pos"].clone(),
                    "plant_frame": fi,
                    "sprout_frame": fi + cfg.wolf_seed_germinate_delay,
                    "x": w["pos"][0].item(),
                    "y": w["pos"][1].item(),
                }
                self.lobeira_sites.append(site)
                w["last_seed_drop"] = fi

            frame_pos.append({"x": w["pos"][0].item(), "y": w["pos"][1].item()})

        self.wolf_history.append(frame_pos)

        # Lobeira frugivore attraction (after sprout)
        for site in self.lobeira_sites:
            if fi < site["sprout_frame"]:
                continue
            sp = site["pos"]
            d = torch.norm(self.pos - sp, dim=1)
            near = sm & (d < cfg.wolf_lobeira_attract_radius) & (self.is_frugivore | self.is_insectivore)
            if near.any():
                self.energy[near & (d < 18.0)] += cfg.lobeira_energy_bonus * 0.1

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg
        if fi < cfg.wolf_start_frame:
            return f

        for w in self.wolves:
            wp = w["pos"]
            d = torch.norm(self.pos - wp, dim=1)
            flee = sm & (d < cfg.wolf_fear_radius)
            if flee.any():
                away = self.pos[flee] - wp
                str_ = (1.0 - d[flee] / cfg.wolf_fear_radius) ** 2
                f[flee] += (away / d[flee].unsqueeze(1).clamp(min=1.)) * str_.unsqueeze(1) * cfg.wolf_fear_force
                self.alarm_level[flee] = torch.clamp(self.alarm_level[flee] + 0.25, max=1.0)
                self.alarm_vectors[flee] = away / d[flee].unsqueeze(1).clamp(min=1.)

        # Lobeira attraction for nearby birds/frugivores
        active_sites = [s for s in self.lobeira_sites if fi >= s["sprout_frame"]]
        if active_sites and (sm & (self.is_frugivore | self.is_insectivore)).any():
            sp_pts = torch.stack([s["pos"] for s in active_sites])
            d2 = torch.cdist(self.pos, sp_pts)
            mn, cl = torch.min(d2, dim=1)
            eligible = sm & (self.is_frugivore | self.is_insectivore) & (mn < cfg.wolf_lobeira_attract_radius)
            if eligible.any():
                pull = sp_pts[cl[eligible]] - self.pos[eligible]
                f[eligible] += (pull / mn[eligible].unsqueeze(1).clamp(min=1.)) * 2.5

        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Lobeira sprouts
        for site in self.lobeira_sites:
            sx, sy = site["x"], site["y"]
            sf = site["sprout_frame"]
            ops = ";".join("0.0" if fi < sf else "0.85" for fi in range(F))
            r_vals = ";".join("0" if fi < sf else f"{min(9.0, (fi - sf) / 15.0 * 9.0):.1f}" for fi in range(F))
            # Seed dot (appears at planting time, faint)
            seed_ops = ";".join("0.0" if fi < site["plant_frame"] else ("0.4" if fi < sf else "0.0") for fi in range(F))
            out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="#ffb300" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{seed_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            # Sprout (grows after delay)
            out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" fill="#7cb342" stroke="#33691e" stroke-width="1.5" opacity="0.0">'
                       f'<animate attributeName="r" values="{r_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            # Attract halo (very faint)
            out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{cfg.wolf_lobeira_attract_radius:.0f}" fill="none" stroke="#aed581" stroke-width="0.8" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ops.replace("0.85","0.1")}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Wolf trails + body
        for wi, w in enumerate(self.wolves):
            xs = ";".join(f"{self.wolf_history[fi][wi]['x']:.1f}" for fi in range(F))
            ys = ";".join(f"{self.wolf_history[fi][wi]['y']:.1f}" for fi in range(F))
            vis = ";".join("1.0" if fi >= cfg.wolf_start_frame else "0.0" for fi in range(F))
            vis_faint = vis.replace("1.0", "0.3")

            # Trail line (very faint)
            trail_x = [self.wolf_history[fi][wi]["x"] for fi in range(0, F, 3)]
            trail_y = [self.wolf_history[fi][wi]["y"] for fi in range(0, F, 3)]
            trail_pts = " ".join(f"{x:.0f},{y:.0f}" for x, y in zip(trail_x, trail_y))
            out.append(f'<polyline points="{trail_pts}" fill="none" stroke="#ffb300" stroke-width="1" opacity="0.2"/>')

            # Fear zone ring
            out.append(f'<circle r="{cfg.wolf_fear_radius:.0f}" fill="none" stroke="#ffb300" stroke-dasharray="6,4" stroke-width="1.2" opacity="0.0">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis_faint}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

            # Wolf body
            out.append(f'<circle r="9" fill="#795548" stroke="#ffd54f" stroke-width="2">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            # Mane highlight
            out.append(f'<circle r="4" fill="#e65100">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        total_sites = len(self.lobeira_sites)
        sprouted = sum(1 for s in self.lobeira_sites if s["sprout_frame"] <= cfg.frames)
        lines = [
            (f"{cfg.wolf_count} Maned Wolves roam the landscape.", "#ffe0b2"),
            (f"Each wolf drops a Lobeira seed every {cfg.wolf_seed_drop_interval} steps.", "#ffcc80"),
            (f"Seeds sprout after {cfg.wolf_seed_germinate_delay} frames (yellow→green).", "#ffb74d"),
            (f"{total_sites} seeds dropped; {sprouted} sprouted into fruiting plants.", "#ffb74d"),
            ("Birds are drawn to Lobeira plants for energy.", "#ffa726"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Maned Wolf Dispersal", lines, "#f57f17")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Maned Wolf Dispersal", "Wide-ranging wolves act as keystone seed dispersers for Wolf Apple plants", "#f57f17")


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

    print(f" - Maned Wolf Dispersal on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print(f"Lobeira sites seeded: {len(sim.lobeira_sites)}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_50')

if __name__ == "__main__": main()
