# -*- coding: utf-8 -*-
# MODULE 45: Carbon Tracking (Visual bar chart of biomass: trees + animals) - 640 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Carbon Tracking (Visual bar chart of biomass: trees + animals) - 640 frames emphasizing land-use boundary frictions.
- Indicator species: Cupim (Nasutitermes sp.).
- Pollination lens: pollinator dilution across open fields.
- Human impact lens: apiary placement competition.

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

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Carbon Tracking Bar Chart UI
        max_c = max(max(self.carbon_history_total), 1.0) * 1.1 # 10% headroom

        chart_w, chart_h = 250, 150
        tx, ty = cfg.width - (chart_w + 20) - 20, 300

        out.append(f'<g transform="translate({tx},{ty})">')
        out.append(f'<rect width="{chart_w+20}" height="{chart_h+40}" fill="#1a1a2e" rx="8" stroke="#4db6ac" opacity="0.95"/>')
        out.append(f'<text x="10" y="20" fill="#cccccc" font-size="15" font-weight="bold">CARBON SEQUESTRATION</text>')

        # Y axis max label
        out.append(f'<text font-weight="bold" x="10" y="40" fill="#cccccc" font-size="15">{int(max_c)}C</text>')
        # X/Y axes lines
        out.append(f'<line x1="35" y1="30" x2="35" y2="{chart_h+10}" stroke="#555" stroke-width="2"/>')
        out.append(f'<line x1="35" y1="{chart_h+10}" x2="{chart_w+10}" y2="{chart_h+10}" stroke="#555" stroke-width="2"/>')

        # Animating the chart bar using scaling
        # We will split into two stacked rects: bottom=trees, top=animals
        h_trees = ";".join(f"{(t/max_c)*(chart_h-20):.1f}" for t in self.carbon_history_trees)
        y_trees = ";".join(f"{(chart_h+10) - (t/max_c)*(chart_h-20):.1f}" for t in self.carbon_history_trees)

        h_animals = ";".join(f"{(a/max_c)*(chart_h-20):.1f}" for a in self.carbon_history_animals)
        y_animals = ";".join(f"{(chart_h+10) - ((t+a)/max_c)*(chart_h-20):.1f}" for t, a in zip(self.carbon_history_trees, self.carbon_history_animals))

        val_txt = ";".join(f"{int(t)} C-units" for t in self.carbon_history_total)

        bar_x = 80
        bar_w = 60
        # Tree Bar (Green)
        out.append(f'<rect x="{bar_x}" y="{chart_h+10}" width="{bar_w}" height="0" fill="#4caf50">'
                   f'<animate attributeName="height" values="{h_trees}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'<animate attributeName="y" values="{y_trees}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'</rect>')
        # Animal Bar (Pinkish)
        out.append(f'<rect x="{bar_x}" y="{chart_h+10}" width="{bar_w}" height="0" fill="#ec407a">'
                   f'<animate attributeName="height" values="{h_animals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'<animate attributeName="y" values="{y_animals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'</rect>')

        # Current Value Text
        out.append(f'<text x="{bar_x + bar_w + 10}" y="{chart_h/2 + 20}" fill="#cccccc" font-size="15" font-weight="bold">'
                   f'<animate attributeName="textContent" values="{val_txt}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'</text>')

        # Legend
        out.append(f'<rect x="45" y="{chart_h+20}" width="10" height="10" fill="#4caf50"/><text font-weight="bold" x="60" y="{chart_h+29}" fill="#cccccc" font-size="15">Flora</text>')
        out.append(f'<rect x="110" y="{chart_h+20}" width="10" height="10" fill="#ec407a"/><text font-weight="bold" x="125" y="{chart_h+29}" fill="#cccccc" font-size="15">Fauna</text>')
        out.append('</g>')

        return "".join(out)

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Carbon Tracking", "Live stacked bar chart tracks total carbon sequestered in biomass", "#4db6ac")


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
    display(HTML(svg)); save_svg(svg, 'notebook_45')

if __name__ == "__main__": main()
