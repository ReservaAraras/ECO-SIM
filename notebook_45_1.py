# -*- coding: utf-8 -*-
# MODULE 45: Carbon Tracking (Dual-axis line chart of biomass: trees + animals) - 640 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Carbon Tracking (Dual-axis line chart of biomass: trees + animals) - 640 frames emphasizing firebreak network geometry.
- Indicator species: Formiga-sauva (Atta laevigata).
- Pollination lens: stingless bee corridor dependency.
- Human impact lens: fence permeability conflicts.

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

from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
from dataclasses import dataclass
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 640
    # Average carbon values (arbitrary units for scale)
    carbon_per_tree: float = 400.0
    carbon_per_animal: float = 2.5
    carbon_baseline: float = 0.0

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        self.carbon_history_trees = []
        self.carbon_history_animals = []
        self.carbon_history_total = []

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg

        # Calculate carbon stored in active trees
        active_trees = sum(1 for t in self.nest_active if t)
        # Assuming fruiting/nectar/vereda nodes are permanent, we just count nest trees for variance
        tree_carbon = active_trees * cfg.carbon_per_tree

        # Calculate carbon stored in active animals
        active_animals_count = int(am.sum())
        animal_carbon = active_animals_count * cfg.carbon_per_animal

        total_carbon = tree_carbon + animal_carbon

        self.carbon_history_trees.append(tree_carbon)
        self.carbon_history_animals.append(animal_carbon)
        self.carbon_history_total.append(total_carbon)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        return torch.zeros_like(self.vel)

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        return ""

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg; out = []

        if not self.carbon_history_total:
            return ""

        # Dual Y-axis line chart: left scale for flora & total, right scale for fauna
        max_left  = max(max(self.carbon_history_total), 1.0) * 1.1
        max_right = max(max(self.carbon_history_animals), 1.0) * 1.1

        chart_w, chart_h = 260, 150
        tx, ty = cfg.width - (chart_w + 20) - 20, 300
        CX0 = 40; CY0 = 30; CY1 = chart_h + 10; CW = 200; CIH = chart_h - 20

        out.append(f'<g transform="translate({tx},{ty})">')
        out.append(f'<rect width="{chart_w + 20}" height="{chart_h + 40}" fill="#1a1a2e" rx="8" stroke="#4db6ac" opacity="0.95"/>')
        out.append(f'<text x="10" y="20" fill="#cccccc" font-size="15" font-weight="bold">CARBON SEQUESTRATION</text>')

        # Left Y-axis (flora / total scale)
        out.append(f'<line x1="{CX0}" y1="{CY0}" x2="{CX0}" y2="{CY1}" stroke="#555" stroke-width="2"/>')
        out.append(f'<text font-weight="bold" x="{CX0 - 3}" y="{CY0 + 4}" fill="#4caf50" font-size="15" text-anchor="end">{int(max_left)}</text>')
        out.append(f'<text font-weight="bold" x="{CX0 - 3}" y="{CY1}" fill="#4caf50" font-size="15" text-anchor="end">0</text>')

        # X-axis
        out.append(f'<line x1="{CX0}" y1="{CY1}" x2="{CX0 + CW}" y2="{CY1}" stroke="#555" stroke-width="2"/>')

        # Right Y-axis (fauna scale, dashed pink)
        out.append(f'<line x1="{CX0 + CW}" y1="{CY0}" x2="{CX0 + CW}" y2="{CY1}" stroke="#ec407a" stroke-width="1" stroke-dasharray="3,3" opacity="0.7"/>')
        out.append(f'<text font-weight="bold" x="{CX0 + CW + 4}" y="{CY0 + 4}" fill="#ec407a" font-size="15">{int(max_right)}</text>')
        out.append(f'<text font-weight="bold" x="{CX0 + CW + 4}" y="{CY1}" fill="#ec407a" font-size="15">0</text>')

        def _pts(series, scale):
            n = max(1, len(series) - 1)
            return " ".join(
                f"{CX0 + (i / n) * CW:.1f},{CY1 - (v / scale) * CIH:.1f}"
                for i, v in enumerate(series)
            )

        out.append(f'<polyline points="{_pts(self.carbon_history_total, max_left)}" fill="none" stroke="#ffeb3b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>')
        out.append(f'<polyline points="{_pts(self.carbon_history_trees, max_left)}" fill="none" stroke="#4caf50" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>')
        out.append(f'<polyline points="{_pts(self.carbon_history_animals, max_right)}" fill="none" stroke="#ec407a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>')

        # Legend
        out.append(f'<rect x="43" y="{chart_h+20}" width="10" height="10" fill="#4caf50"/><text font-weight="bold" x="56" y="{chart_h+29}" fill="#cccccc" font-size="15">Flora</text>')
        out.append(f'<rect x="110" y="{chart_h+20}" width="10" height="10" fill="#ec407a"/><text font-weight="bold" x="123" y="{chart_h+29}" fill="#cccccc" font-size="15">Fauna</text>')
        out.append(f'<rect x="185" y="{chart_h+20}" width="10" height="10" fill="#ffeb3b"/><text font-weight="bold" x="198" y="{chart_h+29}" fill="#cccccc" font-size="15">Total</text>')
        out.append('</g>')

        return "".join(out)

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Carbon Tracking", "Dual-axis line chart: flora & total (left) vs fauna (right) carbon in biomass", "#4db6ac")


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

    print(f" - Carbon Tracking on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()

    start_c = sim.carbon_history_total[0]
    end_c = sim.carbon_history_total[-1]
    diff = end_c - start_c
    print(f"Done. Carbon change: {start_c:.0f} -> {end_c:.0f} ({diff:+.0f}). Generating SVG...")

    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_45_1')

if __name__ == "__main__": main()
