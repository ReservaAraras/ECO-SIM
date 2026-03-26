# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 03: Diurnal Cycles (Background & Activity Modulation)
# ===================================================================================================
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Diurnal Cycles (Background & Activity Modulation) emphasizing riparian corridor dynamics.
- Indicator species: Tamandua-bandeira (Myrmecophaga tridactyla).
- Pollination lens: moth visitation during cold fronts.
- Human impact lens: noise from vehicle corridors.

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
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import random
import math
from IPython.display import display, HTML # pyre-ignore[21]
from eco_base import CANVAS_HEIGHT, ZONES
from dataclasses import dataclass, field
from typing import List, Dict

# ===================================================================================================
# 1. SCIENTIFIC TAXONOMY & ECOLOGICAL GUILDS (Recanto das Araras ENDEMICS)
# ===================================================================================================

BIODIVERSITY_DB = {
    "Tucano-toco (Ramphastos toco)": {"speed": 4.5, "cohesion": 0.03, "color": "#fe4db7", "weight": 0.4, "diet": "Frugivore", "seed_drop_prob": 0.073, "drag": 0.85, "max_turn": 0.1},
    "Gralha-do-campo (Cyanocorax cristatellus)": {"speed": 5.8, "cohesion": 0.01, "color": "#00ffdc", "weight": 0.3, "diet": "Omnivore", "seed_drop_prob": 0.041, "drag": 0.90, "max_turn": 0.15},
    "Beija-flor-tesoura (Eupetomena macroura)": {"speed": 6.2, "cohesion": 0.05, "color": "#ffe43f", "weight": 0.2, "diet": "Nectarivore", "seed_drop_prob": 0.012, "drag": 0.98, "max_turn": 0.4},
    "Gavião-carijó (Rupornis magnirostris)": {"speed": 7.0, "cohesion": 0.005, "color": "#62fff3", "weight": 0.1, "diet": "Carnivore", "seed_drop_prob": 0.001, "drag": 0.95, "max_turn": 0.2},
}

# ===================================================================================================
# 2. CONFIGURATION & SPATIAL PARAMETERS
# ===================================================================================================

@dataclass
class SimulationConfig:
    """Class `SimulationConfig` -- simulation component."""

    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 220 # Prompt requirement: 220 frames
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_particles: int = 250
    dt: float = 0.5
    day_length: int = 110 # Frames per full day cycle (one day, one night, etc or half day half night)

CONFIG = SimulationConfig()

# ===================================================================================================
# 3. KINEMATIC & ECOLOGICAL ENGINE (TENSORIZED)
# ===================================================================================================

class TerraRoncaEcosystem:
    """Class `TerraRoncaEcosystem` -- simulation component."""

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Spatial setup
        self.pos = torch.rand((cfg.num_particles, 2), device=self.dev) * torch.tensor([cfg.width, cfg.height], device=self.dev)
        self.vel = (torch.rand((cfg.num_particles, 2), device=self.dev) - 0.5) * 10.0

        # Assign guilds based on weighting
        guilds = list(BIODIVERSITY_DB.keys())
        weights = [BIODIVERSITY_DB[g]["weight"] for g in guilds]
        indices = np.random.choice(len(guilds), size=cfg.num_particles, p=weights)

        self.speeds = torch.tensor([BIODIVERSITY_DB[guilds[i]]["speed"] for i in indices], device=self.dev)
        self.cohesions = torch.tensor([BIODIVERSITY_DB[guilds[i]]["cohesion"] for i in indices], device=self.dev)
        self.drags = torch.tensor([BIODIVERSITY_DB[guilds[i]]["drag"] for i in indices], device=self.dev).unsqueeze(1)
        self.max_turns = torch.tensor([BIODIVERSITY_DB[guilds[i]]["max_turn"] for i in indices], device=self.dev)
        self.colors = [BIODIVERSITY_DB[guilds[i]]["color"] for i in indices]

        self.karst_nodes = torch.tensor([
            [200.0, 300.0], [500.0, 150.0], [800.0, 450.0], [350.0, 600.0]
        ], device=self.dev)

        self.trajectory_history = []
        self.daylight_history = [] # 1.0 = noon, 0.0 = midnight

    def step(self, frame_idx: int):
        """Function `step` -- simulation component."""

        old_vel = self.vel.clone()

        # Diurnal Cycle: Calculate sunlight intensity using a sine wave
        # Sine starts at 0 (sunrise), goes to 1 (noon), back to 0 (sunset), to -1 (midnight)
        time_angle = (frame_idx / self.cfg.day_length) * 2 * math.pi
        sunlight = (math.sin(time_angle) + 1.0) / 2.0 # Maps to 0.0 (midnight) to 1.0 (noon)
        self.daylight_history.append(sunlight)

        # Activity modulation: Birds sleep at night, active during day
        # Apply a multiplier to speed based on sunlight (e.g., highly active when sun > 0.3)
        activity_multiplier = max(0.05, sunlight) # Cap minimum so they don't completely freeze in air, but very slow

        # Karst Environmental Pull
        dist_to_nodes = torch.cdist(self.pos, self.karst_nodes)
        min_dist, closest_nodes = torch.min(dist_to_nodes, dim=1)
        pull_vectors = self.karst_nodes[closest_nodes] - self.pos
        karst_attraction = (pull_vectors / min_dist.unsqueeze(1).clamp(min=1.0)) * 1.5

        # Apply forces and random wander
        raw_new_vel = self.vel * self.drags + karst_attraction + torch.randn_like(self.vel) * 0.5

        # Turn Radius (Angle Limiting)
        old_angles = torch.atan2(old_vel[:, 1], old_vel[:, 0])
        new_angles = torch.atan2(raw_new_vel[:, 1], raw_new_vel[:, 0])

        angle_diff = new_angles - old_angles
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        clamped_diff = torch.clamp(angle_diff, -self.max_turns, self.max_turns)
        final_angles = old_angles + clamped_diff

        speed_magnitudes = torch.norm(raw_new_vel, dim=1).clamp(min=0.1)
        self.vel[:, 0] = torch.cos(final_angles) * speed_magnitudes
        self.vel[:, 1] = torch.sin(final_angles) * speed_magnitudes

        # Normalize to max speeds, scaled by diurnal activity
        norms = torch.norm(self.vel, dim=1, keepdim=True).clamp(min=1e-5)
        current_max_speeds = self.speeds * activity_multiplier
        self.vel = (self.vel / norms) * current_max_speeds.unsqueeze(1)

        # Update positions
        self.pos += self.vel * self.cfg.dt

        # Periodic boundaries
        self.pos[:, 0] = self.pos[:, 0] % self.cfg.width
        self.pos[:, 1] = self.pos[:, 1] % self.cfg.height

        self.trajectory_history.append(self.pos.cpu().numpy().copy())

# ===================================================================================================
# 4. SCIENTIFIC VISUALIZATION (DATA-DRIVEN SVG)
# ===================================================================================================

class EcosystemRenderer:
    """Class `EcosystemRenderer` -- simulation component."""

    def __init__(self, cfg: SimulationConfig, sim: TerraRoncaEcosystem):
        self.cfg = cfg
        self.sim = sim

    def hex_to_rgb(self, hex_color):
        """Function `hex_to_rgb` -- simulation component."""

        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(self, rgb):
        """Function `rgb_to_hex` -- simulation component."""

        return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

    def blend_colors(self, col1, col2, factor):
        """Function `blend_colors` -- simulation component."""

        c1 = self.hex_to_rgb(col1)
        c2 = self.hex_to_rgb(col2)
        blended = [c1[i] * factor + c2[i] * (1 - factor) for i in range(3)]
        return self.rgb_to_hex(blended)

    def generate_svg(self) -> str:
        """Function `generate_svg` -- simulation component."""

        w, h = self.cfg.width, self.cfg.height

        # We need to construct the animation string for the background color
        # 0.0 (midnight) -> #000b1a
        # 1.0 (noon) -> #000000 (We keep it space black but tint it dawn/dusk)
        # Actually, let's use rich visual colors for day/night
        # Day: #0a1128 (Dark blue) -> Night: #000000 (Pitch black)
        # To make it vibrant, let's just animate between two states based on `daylight_history`
        # Richer diurnal palette: midnight → dawn-orange → noon-blue → dusk-amber → midnight
        bg_colors = []
        for light in self.sim.daylight_history:
            if light < 0.2:
                bg_color = self.blend_colors("#000510", "#000000", light / 0.2)
            elif light < 0.5:
                t = (light - 0.2) / 0.3
                bg_color = self.blend_colors("#1a0a0a", "#000510", t)  # dawn orange tint
            else:
                bg_color = self.blend_colors("#0d1f38", "#1a0a0a", (light - 0.5) / 0.5)
            bg_colors.append(bg_color)

        # Sun colour also animates: dawn=orange, noon=yellow, dusk=red
        sun_colors = []
        for light in self.sim.daylight_history:
            if light < 0.3:
                sun_colors.append("#ff6a00")
            elif light < 0.7:
                sun_colors.append("#ffe066")
            else:
                sun_colors.append("#ff4500")
        bg_color_str = ";".join(bg_colors)
        sun_color_str = ";".join(sun_colors)

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#000510; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append(
            '<defs>'
            '<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse">'
            '<circle cx="2" cy="2" r="4" fill="#3d9df7" opacity="0.45"/>'
            '</pattern>'
            '<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
            '</filter>'
            '<radialGradient id="sunGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#ffe066" stop-opacity="0.9"/>'
            '<stop offset="60%" stop-color="#ff6a00" stop-opacity="0.4"/>'
            '<stop offset="100%" stop-color="#ff6a00" stop-opacity="0.0"/>'
            '</radialGradient>'
            '<radialGradient id="moonGlow" cx="50%" cy="50%" r="50%">'
            '<stop offset="0%" stop-color="#ddeeff" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#ddeeff" stop-opacity="0.0"/>'
            '</radialGradient>'
            '</defs>'
        )

        dur = self.cfg.frames / self.cfg.fps

        # Animated background rect - vivid dawn/dusk palette
        svg.append(f'<rect width="{w}" height="{h}" fill="#000510">')
        svg.append(f'<animate attributeName="fill" values="{bg_color_str}" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>')
        svg.append('</rect>')

        svg.append(f'<rect width="{w}" height="{h}" fill="url(#dotGrid)"/>')

        # ── Star field (visible at night) ──────────────────────────────────
        import random as _rnd
        _rnd.seed(42)
        for si in range(22):
            sx = _rnd.randint(40, w - 40)
            sy = _rnd.randint(20, h // 2)
            sr = _rnd.uniform(1.2, 2.5)
            # Star blink tied to inverse of daylight
            star_ops = ";".join([
                f"{max(0.0, (1.0 - light) * _rnd.uniform(0.6, 1.0)):.2f}"
                for light in self.sim.daylight_history
            ])
            beg = f"{si * 0.12:.2f}s"
            svg.append(
                f'<circle cx="{sx}" cy="{sy}" r="{sr:.1f}" fill="#e8f0ff" opacity="0.0">'
                f'<animate attributeName="opacity" values="{star_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="r" values="{sr:.1f};{sr*1.6:.1f};{sr:.1f}" dur="{_rnd.uniform(1.5, 3.5):.1f}s" begin="{beg}" repeatCount="indefinite"/>'
                f'</circle>'
            )

        # Render Karst Nodes (Caves/Rivers)
        knodes = self.sim.karst_nodes.cpu().numpy()
        for i, kn in enumerate(knodes):
            svg.append(f'<circle cx="{kn[0]}" cy="{kn[1]}" r="80" fill="none" stroke="#fe4db7" stroke-width="2.5" stroke-dasharray="4 4"/>')
            svg.append(f'<circle cx="{kn[0]}" cy="{kn[1]}" r="4" fill="#00ffdc"/>')
            svg.append(f'<text font-weight="bold" x="{kn[0]+10}" y="{kn[1]-10}" font-size="15" fill="#cccccc">Vereda/Cave {i+1}</text>')

        # Structural UI
        svg.append(f'<text x="20" y="30" font-size="15" fill="#ffffff" font-weight="bold">ECO-SIM: Diurnal Cycles</text>')
        svg.append(f'<text font-weight="bold" x="20" y="50" font-size="15" fill="#b0bec5">Background &amp; Activity Modulation</text>')

        # Render Trajectories (SMIL <animate>)
        for idx in range(self.cfg.num_particles):
            p_color = self.sim.colors[idx]
            path_x  = ";" .join([f"{p[idx, 0]:.1f}" for p in self.sim.trajectory_history])
            path_y  = ";" .join([f"{p[idx, 1]:.1f}" for p in self.sim.trajectory_history])

            # Vivid trail rendering
            if idx % 6 == 0:
                d_path = "M " + " L ".join([f"{p[idx,0]:.1f},{p[idx,1]:.1f}" for p in self.sim.trajectory_history[::4]])
                svg.append(f'<path d="{d_path}" fill="none" stroke="{p_color}" stroke-opacity="0.45" stroke-width="2.5"/>')

            # Glow halo (night birds glow more)
            r_begin = f"{(idx % 14) * 0.18:.2f}s"
            if idx % 5 == 0:
                svg.append(f'<circle r="9" fill="{p_color}" opacity="0.28">')
                svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
                svg.append(f'<animate attributeName="r" values="6;14;6" dur="2.0s" begin="{r_begin}" repeatCount="indefinite"/>')
                svg.append('</circle>')

            # Entity Node with r-pulse
            svg.append(f'<circle r="4.5" fill="{p_color}">')
            svg.append(f'<animate attributeName="cx" values="{path_x}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{path_y}" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="r" values="3.5;5.8;3.5" dur="2.2s" begin="{r_begin}" repeatCount="indefinite"/>')
            svg.append('</circle>')

        # Dynamic Sun/Moon indicator - enhanced with glow, ray spokes, and pulsing
        sun_op_vals  = ";".join([f"{light:.2f}" for light in self.sim.daylight_history])
        moon_op_vals = ";".join([f"{max(0.0, 1.0 - light * 2.5):.2f}" for light in self.sim.daylight_history])

        # Sun glow halo
        svg.append(
            f'<circle cx="1200" cy="52" r="38" fill="url(#sunGlow)">'
            f'<animate attributeName="opacity" values="{sun_op_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="r" values="32;46;32" dur="2.8s" repeatCount="indefinite"/>'
            f'</circle>'
        )
        # Sun disc
        svg.append(
            f'<circle cx="1200" cy="52" r="20" fill="#ffe066">'
            f'<animate attributeName="fill" values="{sun_color_str}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'<animate attributeName="opacity" values="{sun_op_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="r" values="18;24;18" dur="2.2s" repeatCount="indefinite"/>'
            f'</circle>'
        )
        # Sun rays (6 short lines radiating outward)
        for ri in range(6):
            ang = ri * 60
            rx1 = 1200 + math.cos(math.radians(ang)) * 26
            ry1 = 52   + math.sin(math.radians(ang)) * 26
            rx2 = 1200 + math.cos(math.radians(ang)) * 38
            ry2 = 52   + math.sin(math.radians(ang)) * 38
            svg.append(
                f'<line x1="{rx1:.0f}" y1="{ry1:.0f}" x2="{rx2:.0f}" y2="{ry2:.0f}" '
                f'stroke="#ffe066" stroke-width="2.5" stroke-linecap="round">'
                f'<animate attributeName="opacity" values="{sun_op_vals}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="stroke-width" values="1.5;3.5;1.5" dur="1.8s" begin="{ri*0.3:.1f}s" repeatCount="indefinite"/>'
                f'</line>'
            )
        # Moon glow halo
        svg.append(
            f'<circle cx="1200" cy="52" r="24" fill="url(#moonGlow)">'
            f'<animate attributeName="opacity" values="{moon_op_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="r" values="20;32;20" dur="3.5s" repeatCount="indefinite"/>'
            f'</circle>'
        )
        # Moon disc
        svg.append(
            f'<circle cx="1200" cy="52" r="16" fill="#ddeeff">'
            f'<animate attributeName="opacity" values="{moon_op_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="linear"/>'
            f'</circle>'
        )

        # -- Scientific Validation Watermark --
        svg.append(f'<g transform="translate(10, {h - 15})">')

        svg.append('</g>')
        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

# ===================================================================================================
# 5. EXECUTION BOOTSTRAP
# ===================================================================================================

def save_svg_to_drive(svg_content: str, notebook_id: str):
    """Function `save_svg_to_drive` -- simulation component."""

    import os
    drive_folder = "/content/drive/MyDrive/ReservaAraras_SVGs"
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
    """Function `main` -- simulation component."""

    print(f"Initializing diurnal simulator on {CONFIG.device}...")
    sim = TerraRoncaEcosystem(CONFIG)
    for frame_idx in range(CONFIG.frames):
        sim.step(frame_idx)

    print("Simulation complete. Generating SVG...")
    renderer = EcosystemRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))
    save_svg_to_drive(svg_content, 'notebook_03')

if __name__ == "__main__":
    main()
