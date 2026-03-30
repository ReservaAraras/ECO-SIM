# -*- coding: utf-8 -*-
# MODULE 42: Restoration Weeding (Clear invasive texture in management zones) - 610 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Restoration Weeding (Clear invasive texture in management zones) - 610 frames emphasizing phenology lag effects.
- Indicator species: Sucupira-preta (Bowdichia virgilioides).
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
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, math, random # pyre-ignore[21]
from dataclasses import dataclass
from typing import List, Dict
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 610
    # Invasive grass parameters (inherited from 41)
    grass_seed_count: int = 4
    grass_spread_interval: int = 15
    grass_spread_prob: float = 0.45
    grass_spread_dist: float = 40.0
    grass_max_patches: int = 180
    grass_choke_radius: float = 38.0
    grass_energy_penalty: float = 0.08
    grass_start_frame: int = 40
    # NEW: Restoration Weeding parameters
    weed_zones_count: int = 3          # number of active weeding crews
    weed_zone_radius: float = 65.0     # radius cleared per frame
    weed_speed: float = 4.5            # movement speed of weeding crews
    weed_start_frame: int = 180        # when restoration starts
    weed_turn_rate: float = 0.15       # random wander per frame

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        # Grass
        self.grass_patches: List[Dict] = []
        for _ in range(cfg.grass_seed_count):
            self.grass_patches.append({"x": random.uniform(200, cfg.width-200),
                                       "y": random.uniform(100, cfg.height-100), "frame": 0})
        self._rebuild_grass_tensor()
        self.grass_patch_history: List[int] = []

        # Weeding crews (start offscreen or random, will activate at weed_start_frame)
        self.weeders = []
        for _ in range(cfg.weed_zones_count):
            self.weeders.append({"x": random.uniform(100, cfg.width-100),
                                 "y": random.uniform(80, cfg.height-80),
                                 "angle": random.uniform(0, math.pi*2)})
        self.weed_history = []  # track positions for SVG

    def _rebuild_grass_tensor(self):
        if self.grass_patches:
            self.grass_tensor = torch.tensor([[g["x"], g["y"]] for g in self.grass_patches],
                                              device=self.dev, dtype=torch.float32)
        else:
            self.grass_tensor = torch.zeros((0, 2), device=self.dev, dtype=torch.float32)

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg

        # 1. Grass Spread
        if fi >= cfg.grass_start_frame and fi % cfg.grass_spread_interval == 0 and len(self.grass_patches) < cfg.grass_max_patches:
            new_patches = []
            for g in self.grass_patches:
                if random.random() < cfg.grass_spread_prob and len(self.grass_patches)+len(new_patches) < cfg.grass_max_patches:
                    a = random.uniform(0, 2*math.pi)
                    nx = max(10, min(cfg.width-10, g["x"] + math.cos(a)*cfg.grass_spread_dist))
                    ny = max(10, min(cfg.height-10, g["y"] + math.sin(a)*cfg.grass_spread_dist))
                    new_patches.append({"x": nx, "y": ny, "frame": fi})
            if new_patches:
                self.grass_patches.extend(new_patches)
                self._rebuild_grass_tensor()

        # Grass effects
        if self.grass_patches:
            # Choke seeds
            if self.dropped_seeds:
                gt = self.grass_tensor
                self.dropped_seeds = [s for s in self.dropped_seeds if torch.norm(gt - torch.tensor(s["pos"], device=self.dev), dim=1).min().item() > cfg.grass_choke_radius]
            # Drain energy
            if sm.any():
                d_grass = torch.cdist(self.pos, self.grass_tensor)
                in_grass = (d_grass.min(dim=1).values < cfg.grass_choke_radius) & sm & ~self.is_grazer
                if in_grass.any():
                    self.energy[in_grass] -= cfg.grass_energy_penalty

        # 2. Weeding Crews (Restoration)
        if fi >= cfg.weed_start_frame:
            frame_pos = []
            for w in self.weeders:
                # Wander
                w["angle"] += random.uniform(-cfg.weed_turn_rate, cfg.weed_turn_rate)
                w["x"] += math.cos(w["angle"]) * cfg.weed_speed
                w["y"] += math.sin(w["angle"]) * cfg.weed_speed
                # Bounce
                if w["x"] < 50 or w["x"] > cfg.width-50:
                    w["angle"] = math.pi - w["angle"]
                    w["x"] = max(50, min(cfg.width-50, w["x"]))
                if w["y"] < 50 or w["y"] > cfg.height-50:
                    w["angle"] = -w["angle"]
                    w["y"] = max(50, min(cfg.height-50, w["y"]))
                frame_pos.append({"x": w["x"], "y": w["y"]})

                # Clear grass inside weed_zone_radius
                if self.grass_patches:
                    surviving = []
                    for g in self.grass_patches:
                        dx = g["x"] - w["x"]; dy = g["y"] - w["y"]
                        if math.sqrt(dx*dx + dy*dy) > cfg.weed_zone_radius:
                            surviving.append(g)
                    if len(surviving) < len(self.grass_patches):
                        self.grass_patches = surviving
                        self._rebuild_grass_tensor()

            self.weed_history.append(frame_pos)
        else:
            self.weed_history.append([{"x": w["x"], "y": w["y"]} for w in self.weeders])

        self.grass_patch_history.append(len(self.grass_patches))

    def extra_forces(self, fi, am, sm):
        # Weeding crews don't repel animals, but instead attract native frugivores looking for newly cleared ground?
        # Let's just keep it simple: no extra forces. Animals naturally re-colonize cleared areas because the energy drain is gone.
        """Function `extra_forces` -- simulation component."""

        return torch.zeros_like(self.vel)

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Weeding trail footprints (draw a faded path of where they've been)
        for i in range(cfg.weed_zones_count):
            pts = []
            for fi in range(cfg.weed_start_frame, F, 3):
                if fi < len(self.weed_history):
                    w = self.weed_history[fi][i]
                    pts.append(f"{w['x']:.1f},{w['y']:.1f}")
            if pts:
                out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#29b6f6" stroke-width="{cfg.weed_zone_radius*2}" stroke-linecap="round" stroke-linejoin="round" opacity="0.1"/>')

        # Grass patches (drawn under the active crews)
        # Note: since they might get destroyed, we only show standard opacity until they are removed.
        # But wait, SVG animation doesn't easily hide dynamically removed elements mid-sim unless we track 'death_frame'.
        # Since we modify the grass_patches list directly, SVG only sees the surviving patches at the *end*.
        # To show grass being cleared, we must track the death frame.
        pass # implemented in extra_svg override

        return "".join(out)

class WeedingSim(Sim):
    """Class `WeedingSim` -- simulation component."""

    def _extra_init(self):
        super()._extra_init()
        # To animate weeding properly, we need to never delete grass from the master list, just mark it dead.
        self.all_grass_spawned = []
        for g in self.grass_patches:
            ag = {"x": g["x"], "y": g["y"], "start": 0, "end": 9999}
            g["ref"] = ag
            self.all_grass_spawned.append(ag)

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        # Spread (Custom logic so we can track lifetimes)
        if fi >= cfg.grass_start_frame and fi % cfg.grass_spread_interval == 0 and len(self.grass_patches) < cfg.grass_max_patches:
            new_patches = []
            for g in self.grass_patches:
                if random.random() < cfg.grass_spread_prob and len(self.grass_patches)+len(new_patches) < cfg.grass_max_patches:
                    a = random.uniform(0, 2*math.pi)
                    nx = max(10, min(cfg.width-10, g["x"] + math.cos(a)*cfg.grass_spread_dist))
                    ny = max(10, min(cfg.height-10, g["y"] + math.sin(a)*cfg.grass_spread_dist))
                    ag = {"x": nx, "y": ny, "start": fi, "end": 9999}
                    self.all_grass_spawned.append(ag)
                    new_patches.append({"x": nx, "y": ny, "frame": fi, "ref": ag})
            if new_patches:
                self.grass_patches.extend(new_patches)
                self._rebuild_grass_tensor()

        # Choke
        if self.grass_patches and self.dropped_seeds:
            gt = self.grass_tensor
            self.dropped_seeds = [s for s in self.dropped_seeds if torch.norm(gt - torch.tensor(s["pos"], device=self.dev), dim=1).min().item() > cfg.grass_choke_radius]
        if self.grass_patches and sm.any():
            d_grass = torch.cdist(self.pos, self.grass_tensor)
            in_grass = (d_grass.min(dim=1).values < cfg.grass_choke_radius) & sm & ~self.is_grazer
            if in_grass.any(): self.energy[in_grass] -= cfg.grass_energy_penalty

        # Weeding (kill elements)
        if fi >= cfg.weed_start_frame:
            frame_pos = []
            for w in self.weeders:
                w["angle"] += random.uniform(-cfg.weed_turn_rate, cfg.weed_turn_rate)
                w["x"] += math.cos(w["angle"]) * cfg.weed_speed
                w["y"] += math.sin(w["angle"]) * cfg.weed_speed
                if w["x"] < 50 or w["x"] > cfg.width-50: w["angle"] = math.pi - w["angle"]; w["x"] = max(50, min(cfg.width-50, w["x"]))
                if w["y"] < 50 or w["y"] > cfg.height-50: w["angle"] = -w["angle"]; w["y"] = max(50, min(cfg.height-50, w["y"]))
                frame_pos.append({"x": w["x"], "y": w["y"]})

                surviving = []
                for g in self.grass_patches:
                    dx = g["x"] - w["x"]; dy = g["y"] - w["y"]
                    if math.sqrt(dx*dx + dy*dy) <= cfg.weed_zone_radius:
                        g["ref"]["end"] = fi  # mark death frame
                    else:
                        surviving.append(g)
                if len(surviving) < len(self.grass_patches):
                    self.grass_patches = surviving
                    self._rebuild_grass_tensor()
            self.weed_history.append(frame_pos)
        else:
            self.weed_history.append([{"x": w["x"], "y": w["y"]} for w in self.weeders])
        self.grass_patch_history.append(len(self.grass_patches))

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Footprints (swept areas)
        for i in range(cfg.weed_zones_count):
            pts = []
            for fi in range(cfg.weed_start_frame, F, 3):
                if fi < len(self.weed_history):
                    w = self.weed_history[fi][i]
                    pts.append(f"{w['x']:.1f},{w['y']:.1f}")
            if pts: out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#29b6f6" stroke-width="{cfg.weed_zone_radius*2}" stroke-linecap="round" stroke-linejoin="round" opacity="0.1"/>')

        # Grass patches
        for ag in self.all_grass_spawned:
            sf, ef = ag["start"], ag["end"]
            ops = ";".join("0.0" if (fi < sf or fi > ef) else "0.55" for fi in range(F))
            out.append(f'<circle cx="{ag["x"]:.1f}" cy="{ag["y"]:.1f}" r="{cfg.grass_choke_radius:.0f}" fill="#827717" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            out.append(f'<circle cx="{ag["x"]:.1f}" cy="{ag["y"]:.1f}" r="3" fill="#cddc39" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Active weeding crews (cyan sweepers)
        for i in range(cfg.weed_zones_count):
            xs = ";".join(f"{self.weed_history[fi][i]['x']:.1f}" for fi in range(F))
            ys = ";".join(f"{self.weed_history[fi][i]['y']:.1f}" for fi in range(F))
            vis = ";".join("1.0" if fi >= cfg.weed_start_frame else "0.0" for fi in range(F))
            # Ring
            out.append(f'<circle r="{cfg.weed_zone_radius:.0f}" fill="#29b6f6" fill-opacity="0.2" stroke="#4fc3f7" stroke-width="2">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            out.append(f'<circle r="5" fill="#4dd0e1">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg; final = self.grass_patch_history[-1] if self.grass_patch_history else 0
        wstarted = (cfg.frames - cfg.weed_start_frame) > 0
        lines = [
            (f"Invasive grass spreads exponentially up to step {cfg.weed_start_frame}.", "#f0f4c3"),
            (f"At step {cfg.weed_start_frame}, {cfg.weed_zones_count} restoration crews deploy.", "#81d4fa"),
            (f"Crews roam the map, sweeping a {cfg.weed_zone_radius:.0f} zone.", "#81d4fa"),
            ("Weeded grass patches are instantly deleted.", "#4fc3f7"),
            (f"Final patch count: {final} / {cfg.grass_max_patches} max.", "#cddc39"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Restoration Weeding", lines, "#29b6f6")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Restoration Weeding", "Management crews roam the map to eradicate invasive textures", "#29b6f6")


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

    print(f" - Restoration Weeding on {CONFIG.device}...")
    sim = WeedingSim(CONFIG); sim.run()
    final = sim.grass_patch_history[-1] if sim.grass_patch_history else 0
    print(f"Grass cleared. Final count: {final}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_42')

if __name__ == "__main__": main()
