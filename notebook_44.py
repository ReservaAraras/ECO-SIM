# -*- coding: utf-8 -*-
# MODULE 44: Extreme Drought (Water nodes vanish entirely; movement cost spikes) - 630 frames
"""
notebook_44.py -- Extreme Drought (Water nodes vanish entirely; movement cost spikes)
Notebook Differentiation:
- Differentiation Focus: Extreme Drought (Water nodes vanish entirely; movement cost spikes) - 630 frames emphasizing evolutionary refuge pockets.
- Indicator species: Buriti-do-campo (Mauritiella armata).
- Pollination lens: ground-nesting bee vulnerability.
- Human impact lens: traditional harvesting pressure.

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
import torch, math  # pyre-ignore[21]
from dataclasses import dataclass
from eco_base import EcosystemBase, BaseConfig, save_svg, svg_metric_card, CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML  # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 630
    # Extreme Drought Parameters
    drought_start_frame: int = 250
    drought_peak_frame: int = 400
    drought_speed_penalty: float = 0.45    # animals slow down significantly in deep drought
    drought_sand_expansion: float = 300.0  # sand zone advances eastward

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        self.drought_intensity = []
        self.sand_zone_history = []

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        if fi >= cfg.drought_start_frame:
            # Scale 0 to 1
            intensity = min(1.0, (fi - cfg.drought_start_frame) / (cfg.drought_peak_frame - cfg.drought_start_frame))

            # 1. Water nodes completely vanish
            self.vereda_health *= (1.0 - intensity * 0.1) # Rapidly decays to 0
            cfg.vereda_max_radius = max(0.0, 90.0 * (1.0 - intensity * 1.5))
            cfg.vereda_min_radius = max(0.0, 20.0 * (1.0 - intensity * 1.5))

            # 2. Fruiting nodes die off
            cfg.fruiting_base_attraction = max(0.0, 4.0 * (1.0 - intensity))

            # 3. Sand zone expands (desertification)
            cfg.sand_zone_x = 640.0 + intensity * cfg.drought_sand_expansion

            # 4. Movement cost spikes (we just penalize speeds via drag scaling)
            # Implemented in extra_forces or by directly modifying speeds in base but base
            # restores speeds every step based on constants.
            # We can't modify self.speeds easily since it's used as base,
            # but we CAN modify cfg.sand_speed_modifier (which was originally 0.6)
            cfg.sand_speed_modifier = max(0.1, 0.6 - intensity * cfg.drought_speed_penalty)

            self.drought_intensity.append(intensity)
            self.sand_zone_history.append(cfg.sand_zone_x)
        else:
            self.drought_intensity.append(0.0)
            self.sand_zone_history.append(cfg.sand_zone_x)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        return torch.zeros_like(self.vel)

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out: list[str] = []

        # Drought overlay: a dusty yellow/brown overlay that fades in
        ops = ";".join(f"{ds * 0.35:.2f}" for ds in self.drought_intensity)
        out.append(f'<rect width="{cfg.width}" height="{cfg.height}" fill="#8D6E63" opacity="0.0" pointer-events="none">'
                   f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></rect>')

        # The sand zone actually expanding.
        # Start at y=80 to leave the main title/subtitle area unobscured.
        w_vals = ";".join(f"{sz:.1f}" for sz in self.sand_zone_history)
        out.append(f'<rect x="0" y="80" width="640" height="{cfg.height - 80}" fill="#1c1611">'
                   f'<animate attributeName="width" values="{w_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></rect>')
        out.append(f'<rect x="0" y="80" width="640" height="{cfg.height - 80}" fill="url(#sandDot)">'
                   f'<animate attributeName="width" values="{w_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></rect>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg

        # Drought meter UI
        ops_h = ";".join(f"{ds * 100:.1f}" for ds in self.drought_intensity)
        ops_y = ";".join(f"{120 - ds * 100:.1f}" for ds in self.drought_intensity)
        ops_op = ";".join("0.0" if ds == 0.0 else "1.0" for ds in self.drought_intensity)

        tx, ty = cfg.width - 250, cfg.height - 180
        out: list[str] = []
        out.append(f'<g transform="translate({tx},{ty})" opacity="0.0">'
                   f'<animate attributeName="opacity" values="{ops_op}" dur="{cfg.frames/cfg.fps}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'<rect x="0" y="0" width="15" height="120" fill="#1e1e1e" rx="7" ry="7" stroke="#666" stroke-width="2"/>'
                   f'<rect x="2" y="120" width="11" height="0" fill="#ffb300" rx="5" ry="5">'
                   f'<animate attributeName="height" values="{ops_h}" dur="{cfg.frames/cfg.fps}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'<animate attributeName="y" values="{ops_y}" dur="{cfg.frames/cfg.fps}s" repeatCount="indefinite" calcMode="discrete"/>'
                   f'</rect>'
                   f'<text x="25" y="10" fill="#cccccc" font-size="15" font-weight="bold">DROUGHT</text>'
                   f'<text x="25" y="24" fill="#cccccc" font-size="15" font-weight="bold">INTENSITY</text>'
                   f'</g>')

        final_sz = self.sand_zone_history[-1] if self.sand_zone_history else 640
        lines = [
            ("Extreme drought starts.", "#ffe082"),
            ("Water nodes vanish; veredas drop to 0 radius.", "#ffb300"),
            ("Desertification: Sand zone advances.", "#ffa000"),
            ("Sand movement penalized; speed drops to 10%.", "#ff8f00"),
        ]
        out.append(svg_metric_card(cfg.width - 395, 20, "Feature: Extreme Drought", lines, "#ffb300", width=375))
        return "".join(out)

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Extreme Drought", "Water sources vanish entirely and desertification claims the landscape", "#ffb300")


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

    print(f" - Extreme Drought on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()  # type: ignore[call-arg]
    print("Done. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_44')

if __name__ == "__main__": main()
