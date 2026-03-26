# pyre-ignore-all-errors
# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 13: Mastofauna & Flora - Tapir Browsing and Trail Making
# ===================================================================================================
"""
notebook_13.py - Mammals and Flora Interaction.
Notebook Differentiation:
- Differentiation Focus: Mastofauna & Flora - Tapir Browsing and Trail Making emphasizing post-fire recovery scars.
- Indicator species: Jararaca-do-cerrado (Bothrops moojeni).
- Pollination lens: stingless bee corridor dependency.
- Human impact lens: fence permeability conflicts.

Simulates the Tapir (Anta), the 'gardener of the Cerrado'.
Tapirs slowly browse through dense vegetation, creating trails (trampling)
and dispersing large seeds via their droppings, which promotes flora diversity.


Scientific Relevance (PIGT RESEX Recanto das Araras -- 2024):
    - Integrates the socio-environmental complexity of 
  de Cima, Goiás, Brazil.
- Models landscape connectivity, karst vulnerability (Bacia do Rio Lapa),
  biological corridors, and seed-dispersal networks.
- Demonstrates parameters for ecological succession, biodiversity indices,
  integrated fire management (MIF), and ornithochory dynamics.
- Outputs are published via Google Sites: 
- SVG artefacts archived at: https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing

Mechanics:
    - Slow Random Walk: Simulating the heavy, deliberate movement of Tapirs.
- Trail Clearing: Modifying background vegetation where Tapirs walk.
- Seed Deposition: Periodic dropping of nutrient-rich seeds.
"""
import os
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]
from eco_base import CANVAS_HEIGHT, ZONES
from dataclasses import dataclass

@dataclass
class TapirFloraConfig:
    "Class `TapirFloraConfig` -- simulation component."

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 250
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_tapirs: int = 5

CONFIG = TapirFloraConfig()

class TapirSim:
    "Class `TapirSim` -- simulation component."

    def __init__(self, cfg: TapirFloraConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        self.pos = torch.rand((cfg.num_tapirs, 2), device=self.dev) * torch.tensor([cfg.width, cfg.height], device=self.dev)
        self.vel = torch.randn((cfg.num_tapirs, 2), device=self.dev) * 1.5

        self.seeds = []
        self.hist_tapirs = []

    def step(self):
        # Tapir movement
        # Tapir movement
        "Function `step` -- simulation component."

        self.vel += torch.randn_like(self.vel) * 0.2
        v_norm = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)
        self.vel = (self.vel / v_norm) * 2.0  # Slow constant speed

        self.pos = (self.pos + self.vel) % torch.tensor([self.cfg.width, self.cfg.height], device=self.dev)

        # Dropping seeds (2% chance)
        drops = torch.rand(self.cfg.num_tapirs, device=self.dev) < 0.02
        if drops.any():
            for sp in self.pos[drops]:
                self.seeds.append(sp.cpu().numpy().copy())

        self.hist_tapirs.append(self.pos.cpu().numpy().copy())

class TapirRenderer:
    "Class `TapirRenderer` -- simulation component."

    def __init__(self, cfg: TapirFloraConfig, sim: TapirSim):
        self.cfg = cfg
        self.sim = sim

    def generate_svg(self) -> str:
        """Function `generate_svg` -- simulation component."""

        color = "#7df9ff"
        "Function `generate_svg` -- simulation component."

        w, h = self.cfg.width, self.cfg.height
# <rect width="100%" height="100%" fill="url(#dotGrid)"/>']
        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:#0a0c0a; font-family:system-ui, -apple-system, sans-serif;">'
            '<defs>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="4.5" fill="#38804a" opacity="0.55"/>'
            '</pattern>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="tapirGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#fe4db7" stop-opacity="0.7"/>'
            '<stop offset="100%" stop-color="#fe4db7" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="seedGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#ffe43f" stop-opacity="0.9"/>'
            '<stop offset="100%" stop-color="#ffe43f" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>'
            '<rect width="100%" height="100%" fill="#0a0c0a"/>'
            '<rect width="100%" height="100%" fill="url(#dotGrid)"/>'
        ]

        svg.append(f'<text x="20" y="30" font-size="15" fill="#7fffaa" font-weight="bold">ECO-SIM: Mastofauna &amp; Flora Interaction</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#88cc99">Tapir (Anta) Browsing and Seed Deposition</text>')

        # Browse trails - vivid gradient-like polylines
        trail_colors = ["#fe4db7", "#d63fab", "#b02f8d"]
        for i in range(self.cfg.num_tapirs):
            pts = " ".join([f"{p[i,0]:.1f},{p[i,1]:.1f}" for p in self.sim.hist_tapirs])
            tc = trail_colors[i % len(trail_colors)]
            svg.append(f'<polyline points="{pts}" stroke="{tc}" stroke-width="9" fill="none" stroke-linecap="round" opacity="0.65"/>')
            # Inner highlight trail
            svg.append(f'<polyline points="{pts}" stroke="#ffaadd" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.35"/>')

        # Seeds with sparkle rings
        dur_v = self.cfg.frames / self.cfg.fps
        for si, (sx, sy) in enumerate(self.sim.seeds):
            bd = (si % 12) * 0.2
            svg.append(f'<circle cx="{sx}" cy="{sy}" r="14" fill="url(#seedGlow)" opacity="0.0">'
                       f'<animate attributeName="r" values="8;20;8" dur="3.0s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="0.0;0.6;0.0" dur="3.0s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                       f'</circle>')
            svg.append(f'<circle cx="{sx}" cy="{sy}" r="4.5" fill="#ffe43f">'
                       f'<animate attributeName="r" values="3.5;6.5;3.5" dur="2.5s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                       f'</circle>')
            # 4 sparkle lines radiating outward
            for li in range(4):
                ang = li * 45
                import math as _m
                lx1 = sx + _m.cos(_m.radians(ang)) * 7
                ly1 = sy + _m.sin(_m.radians(ang)) * 7
                lx2 = sx + _m.cos(_m.radians(ang)) * 14
                ly2 = sy + _m.sin(_m.radians(ang)) * 14
                svg.append(
                    f'<line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" '
                    f'stroke="#ffe43f" stroke-width="1.5" stroke-linecap="round">'
                    f'<animate attributeName="opacity" values="0.0;1.0;0.0" dur="2.5s" begin="{bd+li*0.18:.2f}s" repeatCount="indefinite"/>'
                    f'</line>'
                )

        # Tapirs - with glow halo + r pulse
        for i in range(self.cfg.num_tapirs):
            path_vals = ";".join([f"{p[i,0]:.1f},{p[i,1]:.1f}" for p in self.sim.hist_tapirs])
            bd = i * 0.4
            svg.append(f'<circle r="18" fill="url(#tapirGlow)" opacity="0.5">'
                       f'<animateMotion values="{path_vals}" dur="{dur_v}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="r" values="14;26;14" dur="2.2s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                       f'</circle>')
            svg.append(f'<circle r="7" fill="#fe4db7" filter="url(#glow)">'
                       f'<animateMotion values="{path_vals}" dur="{dur_v}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="r" values="5;10;5" dur="1.8s" begin="{bd:.1f}s" repeatCount="indefinite"/>'
                       f'</circle>')

        # HUD Dashboard
        card_w = 320
        rx = w - card_w - 20
        svg.append(f'<g transform="translate({rx}, 340)">')
        svg.append('<rect width="320" height="160" fill="#1e1e1e" rx="5" ry="5"/>')
        svg.append('<text font-weight="bold" x="10" y="25" fill="#cccccc" font-size="15" >Interactions</text>')
        svg.append('<circle cx="15" cy="45" r="5" fill="#fe4db7"/>')
        svg.append('<text font-weight="bold" x="25" y="49" fill="#cccccc" font-size="15">Anta (Tapir)</text>')
        svg.append('<polyline points="10,65 20,65" stroke="#fe4db7" stroke-width="4"/>')
        svg.append('<text font-weight="bold" x="25" y="69" fill="#cccccc" font-size="15">Browsing Trail &amp; Seeds</text>')
        svg.append('</g>')

        svg.append('<g transform="translate(10, ' + str(self.cfg.height - 20) + ')">')

        svg.append('</g>')
        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

# -- SVG Export to Google Drive --------------------------------------------
def save_svg_to_drive(svg_content: str, notebook_id: str):
    "Persist SVG artefact to Google Drive for publication on Google Sites."
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
    print(f"SVG saved -> {filepath}")
    return filepath

def main():
    "Function `main` -- simulation component."

    sim = TapirSim(CONFIG)
    for _ in range(CONFIG.frames):
        sim.step()
    renderer = TapirRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_13')

if __name__ == "__main__":
    main()
