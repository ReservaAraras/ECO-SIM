# -*- coding: utf-8 -*-
# MODULE 38: Controlled Burns (Small contained fires that regenerate resources fast) - 570 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Controlled Burns (Small contained fires that regenerate resources fast) - 570 frames emphasizing migratory stopover timing.
- Indicator species: Alecrim-do-campo (Vernonia sp.).
- Pollination lens: wind pollination during grass senescence.
- Human impact lens: grazing pressure on understory.

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
from dataclasses import dataclass, field
from typing import List, Dict
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 570
    burn_interval: int = 80          # frames between controlled burns
    burn_radius: float = 55.0        # size of each burn patch
    burn_duration: int = 30          # frames the fire is active
    burn_flee_radius: float = 90.0
    burn_flee_force: float = 10.0
    burn_regen_bonus: float = 0.04   # vereda/fruiting health boost after burn
    burn_new_node_chance: float = 0.6  # chance a burned site spawns a temp fruiting node
    burn_node_lifetime: int = 120    # frames the post-burn fruiting bonus lasts

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        self.burn_sites: List[Dict] = []       # active burn patches
        self.regen_nodes: List[Dict] = []      # post-burn fruiting bonuses
        self.burn_history: List[Dict] = []     # for SVG

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        # Spawn a new controlled burn periodically
        if fi > 0 and fi % cfg.burn_interval == 0:
            bx = random.uniform(80, cfg.width - 80)
            by = random.uniform(80, cfg.height - 80)
            self.burn_sites.append({"x": bx, "y": by, "start": fi, "end": fi + cfg.burn_duration})
            self.burn_history.append({"x": bx, "y": by, "frame": fi})
            # After burn: maybe spawn a regen node
            if random.random() < cfg.burn_new_node_chance:
                self.regen_nodes.append({"pos": torch.tensor([bx, by], device=self.dev, dtype=torch.float32),
                                         "expire": fi + cfg.burn_duration + cfg.burn_node_lifetime,
                                         "x": bx, "y": by, "start": fi + cfg.burn_duration})
            # Boost vereda health near burn
            for vi, vn in enumerate(self.vereda_nodes):
                if torch.norm(vn - torch.tensor([bx, by], device=self.dev)).item() < 250:
                    self.vereda_health[vi] = min(1.0, self.vereda_health[vi].item() + cfg.burn_regen_bonus)

        # Expire old regen nodes
        self.regen_nodes = [n for n in self.regen_nodes if n["expire"] > fi]
        # Expire old burn sites
        self.burn_sites = [b for b in self.burn_sites if b["end"] > fi]

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg

        # Flee active burns
        for b in self.burn_sites:
            bp = torch.tensor([b["x"], b["y"]], device=self.dev)
            d = torch.norm(self.pos - bp, dim=1)
            flee = sm & (d < cfg.burn_flee_radius)
            if flee.any():
                away = self.pos[flee] - bp
                f[flee] += (away / d[flee].unsqueeze(1).clamp(min=1.)) * cfg.burn_flee_force
                self.alarm_level[flee] = torch.clamp(self.alarm_level[flee] + 0.4, max=1.)
                self.alarm_vectors[flee] = away / d[flee].unsqueeze(1).clamp(min=1.)

        # Attract frugivores/insectivores to regen nodes (post-burn bloom)
        for rn in self.regen_nodes:
            rp = rn["pos"]
            d = torch.norm(self.pos - rp, dim=1)
            near = sm & (d < 130.0) & (self.is_frugivore | self.is_insectivore)
            if near.any():
                pull = rp - self.pos[near]
                f[near] += (pull / d[near].unsqueeze(1).clamp(min=1.)) * 3.5
                feeding = near & (d < 20.0)
                if feeding.any():
                    self.energy[feeding] += 4.0
        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []
        # Regen nodes - green bloom circles
        for rn in self.regen_nodes:
            # always draw at fixed pos (post-burn)
            pass  # handled in overlay via burn_history

        # Animated burn flash for each recorded burn event
        for bh in self.burn_history:
            bx, by, bf = bh["x"], bh["y"], bh["frame"]
            # Opacity: flicker during burn, fade after
            ops = []
            for fi in range(F):
                age = fi - bf
                if age < 0: ops.append("0.0")
                elif age < cfg.burn_duration:
                    ops.append(f"{0.5 + 0.4*math.sin(age*0.6):.2f}")
                elif age < cfg.burn_duration + 60:
                    ops.append(f"{max(0., 0.4*(1-(age-cfg.burn_duration)/60)):.2f}")
                else: ops.append("0.0")
            ov = ";".join(ops)
            # Fire glow
            out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{cfg.burn_radius:.0f}" fill="#ff6f00" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{ov}" dur="{dur}s" repeatCount="indefinite"/></circle>')
            # Post-burn regen bloom
            rops = []
            for fi in range(F):
                age = fi - bf - cfg.burn_duration
                if age < 0: rops.append("0.0")
                elif age < cfg.burn_node_lifetime: rops.append(f"{min(0.6, age/30*0.6):.2f}")
                else: rops.append("0.0")
            rov = ";".join(rops)
            out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="40" fill="#76ff03" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{rov}" dur="{dur}s" repeatCount="indefinite"/></circle>')
            out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8" fill="#76ff03" stroke="#33691e" stroke-width="1.5" opacity="0.0">'
                       f'<animate attributeName="opacity" values="{rov}" dur="{dur}s" repeatCount="indefinite"/></circle>')
        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        nb = len(self.burn_history)
        lines = [
            (f"Controlled burns ignite every {cfg.burn_interval} steps.", "#ffe0b2"),
            (f"Each burn ({cfg.burn_radius:.0f} radius).", "#ffe0b2"),
            (f"{nb} burns recorded; animals flee the orange flash.", "#ffb74d"),
            (f"Post-burn: green bloom node boosts fruiting.", "#c5e1a5"),
            ("Nearby Vereda health is also restored after each burn.", "#a5d6a7"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Controlled Burns", lines, "#ff6f00")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Controlled Burns", "Contained fires → fast regeneration blooms", "#ff8f00")


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

    print(f" - Controlled Burns on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print(f"Burns triggered: {len(sim.burn_history)}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_38')

if __name__ == "__main__": main()
