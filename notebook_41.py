# -*- coding: utf-8 -*-
# MODULE 41: Invasive Grasses (Fast-spreading texture chokes native saplings) - 600 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Invasive Grasses (Fast-spreading texture chokes native saplings) - 600 frames emphasizing symbiosis stability.
- Indicator species: Aroeira (Myracrodruon urundeuva).
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
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, math, random # pyre-ignore[21]
from dataclasses import dataclass
from typing import List, Dict
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 600
    # Invasive grass parameters
    grass_seed_count: int = 3           # initial invasion points
    grass_spread_interval: int = 12     # frames between spread attempts
    grass_spread_prob: float = 0.35     # chance each patch spawns a neighbour
    grass_spread_dist: float = 45.0     # distance of new patch from parent
    grass_max_patches: int = 200        # cap
    grass_choke_radius: float = 35.0    # radius within which saplings/seeds are destroyed
    grass_choke_prob: float = 0.015     # per-frame chance of killing a sapling inside radius
    grass_energy_penalty: float = 0.08  # energy drain for native fauna inside grass
    grass_start_frame: int = 50

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        # Seed patches at random locations
        self.grass_patches: List[Dict] = []
        for _ in range(cfg.grass_seed_count):
            x = random.uniform(100, cfg.width-100)
            y = random.uniform(80, cfg.height-80)
            self.grass_patches.append({"x": x, "y": y, "frame": 0})
        self.grass_tensor = torch.tensor([[g["x"], g["y"]] for g in self.grass_patches],
                                          device=self.dev, dtype=torch.float32)
        self.grass_patch_history: List[int] = []  # count per frame

    def _rebuild_grass_tensor(self):
        if self.grass_patches:
            self.grass_tensor = torch.tensor([[g["x"], g["y"]] for g in self.grass_patches],
                                              device=self.dev, dtype=torch.float32)
        else:
            self.grass_tensor = torch.zeros((0, 2), device=self.dev, dtype=torch.float32)

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        if fi < cfg.grass_start_frame:
            self.grass_patch_history.append(len(self.grass_patches))
            return

        # Spread
        if fi % cfg.grass_spread_interval == 0 and len(self.grass_patches) < cfg.grass_max_patches:
            new_patches = []
            for g in self.grass_patches:
                if random.random() < cfg.grass_spread_prob and len(self.grass_patches)+len(new_patches) < cfg.grass_max_patches:
                    angle = random.uniform(0, 2*math.pi)
                    nx = g["x"] + math.cos(angle)*cfg.grass_spread_dist
                    ny = g["y"] + math.sin(angle)*cfg.grass_spread_dist
                    nx = max(10, min(cfg.width-10, nx))
                    ny = max(10, min(cfg.height-10, ny))
                    new_patches.append({"x": nx, "y": ny, "frame": fi})
            self.grass_patches.extend(new_patches)
            if new_patches:
                self._rebuild_grass_tensor()

        # Choke: destroy dropped seeds near grass
        if self.grass_patches and self.dropped_seeds:
            gt = self.grass_tensor
            to_keep = []
            for s in self.dropped_seeds:
                sp = torch.tensor(s["pos"], device=self.dev)
                d = torch.norm(gt - sp, dim=1)
                if d.min().item() > cfg.grass_choke_radius:
                    to_keep.append(s)
            self.dropped_seeds = to_keep

        # Energy penalty for natives in grass
        if self.grass_patches and sm.any():
            d_grass = torch.cdist(self.pos, self.grass_tensor)
            in_grass = (d_grass.min(dim=1).values < cfg.grass_choke_radius) & sm & ~self.is_grazer
            if in_grass.any():
                self.energy[in_grass] -= cfg.grass_energy_penalty

        self.grass_patch_history.append(len(self.grass_patches))

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        return torch.zeros_like(self.vel)

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []
        # Draw all grass patches with fade-in opacity
        for g in self.grass_patches:
            gf = g["frame"]
            ops = ";".join("0.0" if fi < max(gf, cfg.grass_start_frame) else f"{min(0.55, (fi-gf)/40*0.55):.2f}" for fi in range(F))
            out.append(f'<circle cx="{g["x"]:.1f}" cy="{g["y"]:.1f}" r="{cfg.grass_choke_radius:.0f}" fill="#827717" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite"/></circle>')
            # Small dot centre
            out.append(f'<circle cx="{g["x"]:.1f}" cy="{g["y"]:.1f}" r="3" fill="#cddc39" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite"/></circle>')
        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg; final = self.grass_patch_history[-1] if self.grass_patch_history else 0
        lines = [
            (f"Starts with {cfg.grass_seed_count} invasion points at frame {cfg.grass_start_frame}.", "#f0f4c3"),
            (f"Spreads every {cfg.grass_spread_interval} steps ({cfg.grass_spread_prob*100:.0f}% chance).", "#f0f4c3"),
            (f"Chokes native seed drops within {cfg.grass_choke_radius:.0f} units.", "#dce775"),
            (f"Native animals lose energy inside invaded patches.", "#dce775"),
            (f"Final patch count: {final} / {cfg.grass_max_patches} max.", "#cddc39"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Invasive Grasses", lines, "#cddc39")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Invasive Grasses", "Fast-spreading grass chokes native saplings and drains fauna energy", "#cddc39")


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

    print(f" - Invasive Grasses on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    final = sim.grass_patch_history[-1] if sim.grass_patch_history else 0
    print(f"Final grass patches: {final}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_41')

if __name__ == "__main__": main()
