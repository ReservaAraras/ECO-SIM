# -*- coding: utf-8 -*-
# MODULE 51: Rapid Succession (Open field → Shrub → Forest, accelerated time-lapse) - 700 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Rapid Succession (Open field → Shrub → Forest, accelerated time-lapse) - 700 frames emphasizing resource depletion fronts.
- Indicator species: Grilo-do-campo (Gryllus sp.).
- Pollination lens: flowering in fire-following shrubs.
- Human impact lens: illegal extraction nodes.

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

    frames: int = 700
    # Succession Zones: [x, y] cells that advance through phases
    succn_grid_cols: int = 9
    succn_grid_rows: int = 4
    succn_cell_w: float = 120.0
    succn_cell_h: float = 120.0
    succn_x_offset: float = 80.0
    succn_y_offset: float = 60.0
    # Phase durations (in frames)
    phase_open_dur: int = 100      # open / degraded soil
    phase_shrub_dur: int = 150     # shrub / pioneer colonisation
    # forest = remaining frames
    # Each cell gets a random stagger so they don't all change at once
    succn_stagger_max: int = 80    # max frame offset between cells
    # Ecology effects by phase
    open_attract_mult: float = 0.3   # grazers prefer open
    shrub_attract_mult: float = 0.7
    forest_attract_mult: float = 1.0
    forest_energy_bonus: float = 3.0
    forest_nest_bonus: float = 2.5

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        self.cells: List[Dict] = []
        for row in range(cfg.succn_grid_rows):
            for col in range(cfg.succn_grid_cols):
                cx = cfg.succn_x_offset + col * cfg.succn_cell_w + cfg.succn_cell_w/2
                cy = cfg.succn_y_offset + row * cfg.succn_cell_h + cfg.succn_cell_h/2
                # Stagger start times so succession ripples visually
                stagger = random.randint(0, cfg.succn_stagger_max)
                self.cells.append({
                    "cx": cx, "cy": cy,
                    "start": stagger,
                    "shrub_frame": stagger + cfg.phase_open_dur,
                    "forest_frame": stagger + cfg.phase_open_dur + cfg.phase_shrub_dur,
                })
        self.cell_phases: List[List[int]] = []  # phase per cell per frame (0=open,1=shrub,2=forest)

    def _get_phase(self, cell: Dict, fi: int) -> int:
        if fi < cell["shrub_frame"]:
            return 0
        elif fi < cell["forest_frame"]:
            return 1
        else:
            return 2

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        phases = [self._get_phase(c, fi) for c in self.cells]
        self.cell_phases.append(phases)

        # Ecology: cells in forest phase attract birds and provide energy
        forest_cells = [c for c, p in zip(self.cells, phases) if p == 2]
        if forest_cells and sm.any():
            fps = torch.tensor([[c["cx"], c["cy"]] for c in forest_cells], device=self.dev, dtype=torch.float32)
            d = torch.cdist(self.pos, fps)
            mn = d.min(dim=1).values
            in_forest = sm & (mn < CONFIG.succn_cell_w / 2)
            if in_forest.any():
                self.energy[in_forest] += CONFIG.forest_energy_bonus * 0.05
                self.vereda_health[:] = torch.clamp(self.vereda_health + 0.001, 0., 1.)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg
        phases = self.cell_phases[-1] if self.cell_phases else [0]*len(self.cells)

        # Grazers prefer open cells; birds prefer forest
        for cell, phase in zip(self.cells, phases):
            cp = torch.tensor([cell["cx"], cell["cy"]], device=self.dev)
            d = torch.norm(self.pos - cp, dim=1)
            nearby = sm & (d < cfg.succn_cell_w * 0.7)
            if not nearby.any():
                continue
            if phase == 0:  # open → attract grazers
                gm = nearby & self.is_grazer
                if gm.any():
                    pull = cp - self.pos[gm]
                    f[gm] += (pull / d[gm].unsqueeze(1).clamp(min=1.)) * cfg.open_attract_mult
            elif phase == 2:  # forest → attract birds
                bm = nearby & self.flies & ~self.is_carnivore
                if bm.any():
                    pull = cp - self.pos[bm]
                    f[bm] += (pull / d[bm].unsqueeze(1).clamp(min=1.)) * cfg.forest_attract_mult

        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []
        # Map grid background so cells are shown as a subtle grid
        gw = cfg.succn_x_offset + cfg.succn_grid_cols * cfg.succn_cell_w
        gh = cfg.succn_y_offset + cfg.succn_grid_rows * cfg.succn_cell_h
        out.append(f'<rect x="{cfg.succn_x_offset}" y="{cfg.succn_y_offset}" width="{gw-cfg.succn_x_offset}" height="{gh-cfg.succn_y_offset}" fill="none" stroke="#333" stroke-width="0.5" opacity="0.4"/>')

        # Each cell animated through phases
        phase_colors = ["#5d4037", "#9ccc65", "#2e7d32"]  # brown, lime, deep green
        for ci, cell in enumerate(self.cells):
            x = cell["cx"] - cfg.succn_cell_w / 2
            y = cell["cy"] - cfg.succn_cell_h / 2
            w = cfg.succn_cell_w - 2
            h = cfg.succn_cell_h - 2

            colors_seq = ";".join(phase_colors[self.cell_phases[fi][ci]] for fi in range(F))
            ops_seq = ";".join(
                "0.18" if self.cell_phases[fi][ci] == 0 else
                "0.28" if self.cell_phases[fi][ci] == 1 else
                "0.45"
                for fi in range(F)
            )

            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" opacity="0.0">'
                       f'<animate attributeName="fill" values="{colors_seq}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{ops_seq}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</rect>')

            # Phase label - show state text for just a few central cells
            if ci % (cfg.succn_grid_cols * cfg.succn_grid_rows // 4) == 0:
                labels = ";".join(
                    ("Shrub" if self.cell_phases[fi][ci] == 1 else
                     "Forest" if self.cell_phases[fi][ci] == 2 else
                     "○ Open")
                    for fi in range(F)
                )
                out.append(f'<text font-weight="bold" x="{cell["cx"]:.0f}" y="{cell["cy"]+4:.0f}" text-anchor="middle" font-size="15" fill="#999999" opacity="0.7">'
                           f'<animate attributeName="fill" values="#a1887f;#c5e1a5;#81c784" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                           f'<animate attributeName="textContent" values="{labels}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></text>')

        # Phase legend
        for pi, (lbl, col) in enumerate([("Open/Degraded", "#5d4037"), ("Shrub/Pioneer", "#9ccc65"), ("Closed Forest", "#2e7d32")]):
            out.append(f'<rect x="20" y="{510+pi*22}" width="14" height="14" fill="{col}" rx="3" opacity="0.8"/>')
            out.append(f'<text font-weight="bold" x="40" y="{522+pi*22}" font-size="15" fill="#cccccc">{lbl}</text>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        F = cfg.frames
        if self.cell_phases:
            final_phases = self.cell_phases[-1]
            open_n = sum(1 for p in final_phases if p == 0)
            shrub_n = sum(1 for p in final_phases if p == 1)
            forest_n = sum(1 for p in final_phases if p == 2)
        else:
            open_n = shrub_n = forest_n = 0
        total_cells = len(self.cells)
        lines = [
            (f"{total_cells} cells on a {cfg.succn_grid_cols}×{cfg.succn_grid_rows} grid, staggered start times.", "#c8e6c9"),
            ("Each cell passes: Open → Shrub → Forest over time.", "#a5d6a7"),
            (f"Phase durations: Open {cfg.phase_open_dur}f, Shrub {cfg.phase_shrub_dur}f, Forest forever.", "#81c784"),
            (f"Final state: {open_n} Open | {shrub_n} Shrub | {forest_n} Forest cells.", "#66bb6a"),
            ("Forest cells boost bird energy and Vereda recovery.", "#4caf50"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Rapid Succession", lines, "#2e7d32")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Rapid Succession", "Time-lapse succession: open field → shrub colonisation → closed canopy forest", "#388e3c")


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

    print(f" - Rapid Succession on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    if sim.cell_phases:
        fp = sim.cell_phases[-1]
        print(f"Final: {fp.count(0)} open | {fp.count(1)} shrub | {fp.count(2)} forest cells")
    print("Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_51')

if __name__ == "__main__": main()
