# -*- coding: utf-8 -*-
# MODULE 39: Corridor Planting (Nodes connecting two separated habitat patches) - 580 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Corridor Planting (Nodes connecting two separated habitat patches) - 580 frames emphasizing soil crust recovery.
- Indicator species: Barbatimao (Stryphnodendron adstringens).
- Pollination lens: orchid mimicry with specialist bees.
- Human impact lens: invasive grass spread.

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
from typing import List, Dict
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 580
    # Two habitat patches (left & right)
    patch_a_center: tuple = (160.0, 300.0)
    patch_a_radius: float = 100.0
    patch_b_center: tuple = (1120.0, 300.0)
    patch_b_radius: float = 100.0
    patch_attract_force: float = 2.5
    patch_energy_bonus: float = 1.5
    # Corridor nodes planted between the patches
    corridor_y: float = 300.0
    corridor_x_start: float = 280.0
    corridor_x_end: float = 1000.0
    corridor_nodes: int = 8
    corridor_attract_radius: float = 110.0
    corridor_attract_force: float = 3.0
    corridor_energy_gain: float = 2.5
    # Corridor appears progressively
    corridor_reveal_start: int = 60   # first node planted
    corridor_reveal_interval: int = 40  # frames between each node appearing

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        xs = [cfg.corridor_x_start + i*(cfg.corridor_x_end-cfg.corridor_x_start)/(cfg.corridor_nodes-1)
              for i in range(cfg.corridor_nodes)]
        self.corridor_pts = torch.tensor([[x, cfg.corridor_y] for x in xs], device=self.dev, dtype=torch.float32)
        self.corridor_xs = xs
        self.active_corridor: List[bool] = [False] * cfg.corridor_nodes
        self.patch_a = torch.tensor(cfg.patch_a_center, device=self.dev, dtype=torch.float32)
        self.patch_b = torch.tensor(cfg.patch_b_center, device=self.dev, dtype=torch.float32)

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        # Reveal corridor nodes progressively
        for i in range(cfg.corridor_nodes):
            reveal_at = cfg.corridor_reveal_start + i * cfg.corridor_reveal_interval
            if fi >= reveal_at:
                self.active_corridor[i] = True

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg

        # Patch A & B attract all non-grazer surface animals
        eligible = sm & ~self.is_grazer & ~self.is_migrant
        for patch in [self.patch_a, self.patch_b]:
            d = torch.norm(self.pos - patch, dim=1)
            near = eligible & (d < cfg.patch_a_radius * 2.5)
            if near.any():
                pull = patch - self.pos[near]
                f[near] += (pull / d[near].unsqueeze(1).clamp(min=1.)) * cfg.patch_attract_force
            inside = eligible & (d < cfg.patch_a_radius)
            if inside.any():
                self.energy[inside] += cfg.patch_energy_bonus

        # Active corridor nodes attract animals bridging the two patches
        active_idx = [i for i, a in enumerate(self.active_corridor) if a]
        if active_idx:
            active_pts = self.corridor_pts[torch.tensor(active_idx, device=self.dev)]
            d2 = torch.cdist(self.pos, active_pts)
            mn2, cl2 = torch.min(d2, dim=1)
            near2 = eligible & (mn2 < cfg.corridor_attract_radius)
            if near2.any():
                pull2 = active_pts[cl2[near2]] - self.pos[near2]
                f[near2] += (pull2 / mn2[near2].unsqueeze(1).clamp(min=1.)) * cfg.corridor_attract_force
                feeding = near2 & (mn2 < 18.0)
                if feeding.any():
                    self.energy[feeding] += cfg.corridor_energy_gain
        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []
        # Patch A
        ax, ay = cfg.patch_a_center
        out.append(f'<circle cx="{ax}" cy="{ay}" r="{cfg.patch_a_radius}" fill="#1b5e20" stroke="#69f0ae" stroke-width="2" opacity="0.25"/>')
        out.append(f'<text font-weight="bold" x="{ax}" y="{ay+4}" text-anchor="middle" font-size="15" fill="#cccccc" opacity="0.9">Patch A</text>')
        # Patch B
        bx, by = cfg.patch_b_center
        out.append(f'<circle cx="{bx}" cy="{by}" r="{cfg.patch_b_radius}" fill="#1b5e20" stroke="#69f0ae" stroke-width="2" opacity="0.25"/>')
        out.append(f'<text font-weight="bold" x="{bx}" y="{by+4}" text-anchor="middle" font-size="15" fill="#cccccc" opacity="0.9">Patch B</text>')
        # Corridor line (dashed, always shown)
        out.append(f'<line x1="{cfg.corridor_x_start}" y1="{cfg.corridor_y}" x2="{cfg.corridor_x_end}" y2="{cfg.corridor_y}" stroke="#33691e" stroke-width="1" stroke-dasharray="6,4" opacity="0.3"/>')
        # Corridor nodes - progressive reveal
        for i, x in enumerate(self.corridor_xs):
            reveal_at = cfg.corridor_reveal_start + i * cfg.corridor_reveal_interval
            ops = ";".join("0.0" if fi < reveal_at else "0.9" for fi in range(F))
            out.append(f'<circle cx="{x:.1f}" cy="{cfg.corridor_y}" r="7" fill="#8bc34a" stroke="#33691e" stroke-width="1.5">'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            # Attract radius ring (faint)
            out.append(f'<circle cx="{x:.1f}" cy="{cfg.corridor_y}" r="{cfg.corridor_attract_radius:.0f}" fill="none" stroke="#8bc34a" stroke-width="0.5" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ops.replace("0.9","0.12")}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        lines = [
            ("Two habitat patches (A & B) bookend the map.", "#c8e6c9"),
            (f"{cfg.corridor_nodes} nodes planted every {cfg.corridor_reveal_interval} steps.", "#c8e6c9"),
            ("Each node attracts fauna bridging the patch gap.", "#a5d6a7"),
            ("Animals flow through once enough nodes exist.", "#a5d6a7"),
            ("Connectivity restored → gene flow & recolonisation.", "#69f0ae"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Corridor Planting", lines, "#8bc34a")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Corridor Planting", "Progressive node planting restores habitat connectivity", "#8bc34a")


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

    print(f" - Corridor Planting on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    active = sum(sim.active_corridor)
    print(f"Corridor nodes active at end: {active}/{CONFIG.corridor_nodes}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_39')

if __name__ == "__main__": main()
