# -*- coding: utf-8 -*-
# MODULE 43: Climate Warming (Increase evaporation rate, water shrinks, energy demands rise) - 620 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Climate Warming (Increase evaporation rate, water shrinks, energy demands rise) - 620 frames emphasizing gene flow connectors.
- Indicator species: Fava-d-anta (Dimorphandra mollis).
- Pollination lens: floral resource concentration in veredas.
- Human impact lens: carbon stock monitoring incentives.

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

    frames: int = 620
    # Climate Warming Parameters
    warming_start_frame: int = 150
    temp_base: float = 25.0
    temp_max: float = 38.0
    evaporation_multiplier: float = 0.6  # how much max_radius shrinks by the end
    decay_modifier: float = 0.15         # extra energy decay due to heat exhaustion

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        self.temp_history = []

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        if fi >= cfg.warming_start_frame:
            # Linear temperature increase
            progress = (fi - cfg.warming_start_frame) / (cfg.frames - cfg.warming_start_frame)
            self.temperature = cfg.temp_base + (cfg.temp_max - cfg.temp_base) * progress

            # 1. Shrink water (evaporation)
            # base max_radius is 90.0, min is 20.0
            cfg.vereda_max_radius = 90.0 * (1.0 - cfg.evaporation_multiplier * progress)
            cfg.vereda_min_radius = 20.0 * (1.0 - cfg.evaporation_multiplier * 0.8 * progress)

            # 2. Dry out vereda health actively and counter the natural regeneration
            self.vereda_health = torch.clamp(self.vereda_health - 0.003 * progress, 0.0, 1.0)

            # 3. Heat exhaustion: increased energy decay for animals
            # (ecosystem base resets it? No, we modify the config)
            cfg.energy_decay = 0.12 + cfg.decay_modifier * progress
        else:
            self.temperature = cfg.temp_base

        self.temp_history.append(self.temperature)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        return torch.zeros_like(self.vel)

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        return ""

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Red/Orange temperature overlay
        ops = ";".join(f"{max(0.0, (t-cfg.temp_base)/(cfg.temp_max-cfg.temp_base) * 0.25):.2f}" for t in self.temp_history)
        out.append(f'<rect width="{cfg.width}" height="{cfg.height}" fill="#ff3d00" opacity="0.0" pointer-events="none">'
                   f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></rect>')

        # Thermometer UI
        ops_h = ";".join(f"{max(0.0, (t-20.0)/20.0 * 100):.1f}" for t in self.temp_history) # % height mapping
        ops_y = ";".join(f"{120 - max(0.0, (t-20.0)/20.0 * 100):.1f}" for t in self.temp_history)
        ops_txt = ";".join(f"{t:.1f}°C" for t in self.temp_history)

        tx, ty = cfg.width - 450, cfg.height - 180
        out.append(f'<g transform="translate({tx},{ty})">'
                   f'<rect x="0" y="0" width="15" height="120" fill="#1e1e1e" rx="7" ry="7" stroke="#666" stroke-width="2"/>'
                   f'<rect x="2" y="120" width="11" height="0" fill="#ff5252" rx="5" ry="5">'
                   f'<animate attributeName="height" values="{ops_h}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'<animate attributeName="y" values="{ops_y}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'</rect>'
                   f'<circle cx="7.5" cy="120" r="12" fill="#ff5252"/>'
                   f'<text x="25" y="10" fill="#cccccc" font-size="15" font-weight="bold">TEMP</text>'
                   f'<text x="25" y="24" fill="#cccccc" font-size="15" font-weight="bold">'
                   f'<animate attributeName="textContent" values="{ops_txt}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'</text>'
                   f'</g>')

        final_temp = self.temp_history[-1] if self.temp_history else cfg.temp_base
        radius_drop = (1.0 - cfg.vereda_max_radius / 90.0) * 100
        lines = [
            (f"Global temperature rises smoothly to {cfg.temp_max:.1f}°C.", "#ffcc80"),
            (f"Heat accelerates evaporation; Veredas shrink by {radius_drop:.0f}%.", "#ffb74d"),
            (f"Heat exhaustion raises energy decay by {cfg.decay_modifier*100:.0f}%.", "#ffb74d"),
            ("Starvation and dehydration cascade the population.", "#ffa726"),
            (f"Final temperature: {final_temp:.1f}°C (shown by thermometer).", "#ffcc80"),
        ]
        out.append(self.info_card(cfg.width, cfg.height, "Feature: Climate Warming", lines, "#ff7043"))
        return "".join(out)

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Climate Warming", "Temperatures rise, evaporating water sources and exhausting wildlife", "#ff7043")


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

    print(f" - Climate Warming on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print("Done. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_43')

if __name__ == "__main__": main()
