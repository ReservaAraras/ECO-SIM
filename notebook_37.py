# -*- coding: utf-8 -*-
# MODULE 37: Agroforestry Rows (Linear resource nodes attract biodiversity) - 560 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Agroforestry Rows (Linear resource nodes attract biodiversity) - 560 frames emphasizing nocturnal pollination windows.
- Indicator species: Capim-dourado (Syngonanthus nitens).
- Pollination lens: bee foraging under smoke haze.
- Human impact lens: selective logging canopy loss.

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
import torch, math # pyre-ignore[21]
from dataclasses import dataclass
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 560
    # Agroforestry rows: 3 horizontal rows of evenly-spaced fruiting nodes
    agro_rows: tuple = (100.0, 300.0, 500.0)   # Y positions
    agro_cols: int = 8                           # nodes per row
    agro_x_start: float = 80.0
    agro_x_end: float = 1200.0
    agro_attract_radius: float = 140.0
    agro_attract_force: float = 3.5
    agro_energy_gain: float = 3.0

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        import numpy as np
        cfg = self.cfg
        xs = [cfg.agro_x_start + i*(cfg.agro_x_end-cfg.agro_x_start)/(cfg.agro_cols-1) for i in range(cfg.agro_cols)]
        pts = [[x, y] for y in cfg.agro_rows for x in xs]
        self.agro_nodes = torch.tensor(pts, device=self.dev, dtype=torch.float32)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg
        d = torch.cdist(self.pos, self.agro_nodes)
        mn, cl = torch.min(d, dim=1)
        near = sm & (mn < cfg.agro_attract_radius) & (self.is_frugivore | self.is_insectivore)
        if near.any():
            pull = self.agro_nodes[cl[near]] - self.pos[near]
            f[near] = (pull / mn[near].unsqueeze(1).clamp(min=1.)) * cfg.agro_attract_force
            feeding = near & (mn < 20.0)
            if feeding.any():
                self.energy[feeding] += cfg.agro_energy_gain
        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; pts = self.agro_nodes.cpu().numpy()
        out = []
        # Draw row lines
        xs = [cfg.agro_x_start + i*(cfg.agro_x_end-cfg.agro_x_start)/(cfg.agro_cols-1) for i in range(cfg.agro_cols)]
        for ry in cfg.agro_rows:
            out.append(f'<line x1="{cfg.agro_x_start}" y1="{ry}" x2="{cfg.agro_x_end}" y2="{ry}" stroke="#8bc34a" stroke-width="1" stroke-dasharray="4,4" opacity="0.4"/>')
        # Draw nodes
        for p in pts:
            out.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="6" fill="#8bc34a" stroke="#33691e" stroke-width="1.5" opacity="0.85"/>')
        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        lines = [
            (f"{cfg.agro_cols} nodes × {len(cfg.agro_rows)} rows of agroforestry trees.", "#dcedc8"),
            (f"Frugivores & insectivores attracted within {cfg.agro_attract_radius:.0f} units.", "#dcedc8"),
            (f"Feeding agents gain +{cfg.agro_energy_gain} energy and slow down.", "#aed581"),
            ("Rows act as corridors bridging habitat patches.", "#aed581"),
            ("Biodiversity clusters visibly along planted rows.", "#c5e1a5"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Agroforestry Rows", lines, "#8bc34a")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Agroforestry Rows", "Linear resource rows attract and sustain fauna", "#8bc34a")


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

    print(f" - Agroforestry Rows on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_37')

if __name__ == "__main__": main()
