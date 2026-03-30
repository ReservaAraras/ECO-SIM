# -*- coding: utf-8 -*-
# MODULE 49: Jaguar Presence (Large predator causes massive avoidance behavior) - 680 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Jaguar Presence (Large predator causes massive avoidance behavior) - 680 frames emphasizing seedling recruitment.
- Indicator species: Mariposa-esfinge (Manduca sp.).
- Pollination lens: secondary pollination by beetles.
- Human impact lens: light pollution near settlements.

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

    frames: int = 680
    jaguar_count: int = 1
    jaguar_speed: float = 4.8
    jaguar_fear_radius: float = 200.0    # massive panic zone
    jaguar_fear_force: float = 20.0
    jaguar_alarm_radius: float = 280.0  # slightly wider alarm spread
    jaguar_chase_radius: float = 100.0  # distance at which jaguar targets prey
    jaguar_kill_radius: float = 12.0
    jaguar_energy_gain: float = 60.0
    jaguar_satiation_dur: int = 120
    jaguar_start_frame: int = 50

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        self.jaguars: List[Dict] = []
        for _ in range(cfg.jaguar_count):
            self.jaguars.append({
                "pos": torch.tensor([
                    random.uniform(200, cfg.width - 200),
                    random.uniform(100, cfg.height - 100)
                ], device=self.dev, dtype=torch.float32),
                "vel": torch.zeros(2, device=self.dev, dtype=torch.float32),
                "satiation": 0,
                "angle": random.uniform(0, math.pi * 2),
            })
        self.jaguar_history: List[List[Dict]] = []
        self.jaguar_kills = 0

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        if fi < cfg.jaguar_start_frame:
            self.jaguar_history.append([{"x": j["pos"][0].item(), "y": j["pos"][1].item()} for j in self.jaguars])
            return

        prey_mask = sm & ~self.is_grazer & ~self.is_carnivore & ~self.is_migrant & (self.alarm_level < 0.9)

        frame_pos = []
        for j in self.jaguars:
            j["satiation"] = max(0, j["satiation"] - 1)

            if j["satiation"] > 0:
                # Resting - wander slowly
                j["angle"] += random.uniform(-0.1, 0.1)
                j["pos"] += torch.tensor([math.cos(j["angle"]), math.sin(j["angle"])], device=self.dev) * 1.2
            else:
                # Hunt: find closest prey
                if prey_mask.any():
                    d_prey = torch.norm(self.pos[prey_mask] - j["pos"], dim=1)
                    md, mi = torch.min(d_prey, dim=0)
                    if md.item() < cfg.jaguar_chase_radius:
                        prey_idx = prey_mask.nonzero().squeeze(1)[mi]
                        vec = self.pos[prey_idx] - j["pos"]
                        dist = torch.norm(vec).item()
                        j["pos"] += (vec / max(dist, 1e-5)) * cfg.jaguar_speed * 1.5
                        if dist < cfg.jaguar_kill_radius and self.is_active[prey_idx]:
                            self.is_active[prey_idx] = False
                            self.has_seed[prey_idx] = False
                            pn = self.pos[prey_idx].cpu().numpy().copy()
                            self.death_events.append({"pos": pn, "frame": fi, "color": self.colors[prey_idx], "reason": "jaguar"})
                            self.carrion_sites.append({"pos": pn, "spawn_frame": fi, "expire_frame": fi + self.cfg.carrion_linger_frames})
                            j["satiation"] = cfg.jaguar_satiation_dur
                            self.jaguar_kills += 1
                    else:
                        # Stalk: drift towards closest prey with noise
                        j["angle"] = math.atan2(vec[1].item() if 'vec' in dir() else 0, vec[0].item() if 'vec' in dir() else 1)
                        j["pos"] += torch.tensor([math.cos(j["angle"]), math.sin(j["angle"])], device=self.dev) * cfg.jaguar_speed
                else:
                    j["angle"] += random.uniform(-0.2, 0.2)
                    j["pos"] += torch.tensor([math.cos(j["angle"]), math.sin(j["angle"])], device=self.dev) * cfg.jaguar_speed * 0.7

            # Bounds
            j["pos"][0] = torch.clamp(j["pos"][0], 20, cfg.width - 20)
            j["pos"][1] = torch.clamp(j["pos"][1], 20, cfg.height - 20)
            frame_pos.append({"x": j["pos"][0].item(), "y": j["pos"][1].item()})

        self.jaguar_history.append(frame_pos)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg
        if fi < cfg.jaguar_start_frame:
            return f

        for j in self.jaguars:
            jp = j["pos"]
            d = torch.norm(self.pos - jp, dim=1)

            # Panic flee - very large radius, very strong force
            flee = sm & (d < cfg.jaguar_fear_radius)
            if flee.any():
                away = self.pos[flee] - jp
                strength = (1.0 - d[flee] / cfg.jaguar_fear_radius) ** 1.5
                f[flee] += (away / d[flee].unsqueeze(1).clamp(min=1.)) * strength.unsqueeze(1) * cfg.jaguar_fear_force
                self.alarm_level[flee] = torch.clamp(self.alarm_level[flee] + 0.6, max=1.0)
                self.alarm_vectors[flee] = away / d[flee].unsqueeze(1).clamp(min=1.)

            # Alarm ripple just outside the fear zone
            ripple = sm & (d >= cfg.jaguar_fear_radius) & (d < cfg.jaguar_alarm_radius)
            if ripple.any():
                self.alarm_level[ripple] = torch.clamp(self.alarm_level[ripple] + 0.15, max=1.0)

        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []
        n = len(self.jaguars)

        for ji in range(n):
            xs = ";".join(f"{self.jaguar_history[fi][ji]['x']:.1f}" for fi in range(F))
            ys = ";".join(f"{self.jaguar_history[fi][ji]['y']:.1f}" for fi in range(F))
            vis = ";".join("1.0" if fi >= cfg.jaguar_start_frame else "0.0" for fi in range(F))
            vis_zone = vis.replace("1.0", "0.35")
            alarm_vis = vis.replace("1.0", "0.12")

            # Fear zone halo
            out.append(f'<circle r="{cfg.jaguar_fear_radius:.0f}" fill="#b71c1c" opacity="0.0">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis_zone}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

            # Alarm outer ring
            out.append(f'<circle r="{cfg.jaguar_alarm_radius:.0f}" fill="none" stroke="#ff6d00" stroke-width="2" stroke-dasharray="8,6" opacity="0.0">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{alarm_vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

            # Trail
            trail_pts = ";".join(f"{self.jaguar_history[fi][ji]['x']:.0f},{self.jaguar_history[fi][ji]['y']:.0f}" for fi in range(0, F, 2))
            out.append(f'<polyline points="{trail_pts}" fill="none" stroke="#e64a19" stroke-width="1.5" opacity="0.35"/>')

            # Jaguar body (large tawny dot)
            out.append(f'<circle r="11" fill="#e65100" stroke="#ffd54f" stroke-width="2">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

            # Spot rosette overlay
            out.append(f'<circle r="4" fill="#bf360c" opacity="0.0">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        lines = [
            ("1 Jaguar roams the landscape.", "#ffccbc"),
            ("200px panic zone — nearby fauna scatter instantly.", "#ff8a65"),
            ("Alarm ripple propagates 280px ahead of the jaguar.", "#ff7043"),
            ("Jaguar rests 120 steps after each kill.", "#ef9a9a"),
            (f"Total kills this run: {self.jaguar_kills}.", "#ff5252"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Jaguar Presence", lines, "#e64a19")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Jaguar Presence", "Apex predator triggers cascading panic across the ecosystem", "#e64a19")


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

    print(f" - Jaguar Presence on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print(f"Jaguar kills: {sim.jaguar_kills}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_49')

if __name__ == "__main__": main()
