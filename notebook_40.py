# -*- coding: utf-8 -*-
# MODULE 40: Ecotourism Trails (Tourist agents on fixed paths, birds flush at short range) - 590 frames
"""
Notebook Differentiation:
- Differentiation Focus: Ecotourism Trails (Tourist agents on fixed paths, birds flush at short range) - 590 frames emphasizing mutualism strength shifts.
- Indicator species: Paineira (Ceiba speciosa).
- Pollination lens: ant-guarded nectar dynamics.
- Human impact lens: restoration weeding benefits.

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

    frames: int = 590
    # Trail waypoints: two loop trails crossing the map
    trail_a: tuple = ((100,150),(400,120),(700,200),(1000,140),(1180,200),(1000,300),(700,350),(400,300),(100,250),(100,150))
    trail_b: tuple = ((100,450),(400,480),(700,400),(1000,460),(1180,400),(1000,500),(700,540),(400,510),(100,500),(100,450))
    num_tourists_per_trail: int = 3
    tourist_speed: float = 3.2
    tourist_flush_radius: float = 70.0    # birds flush if tourist this close
    tourist_flush_force: float = 14.0
    tourist_alarm_add: float = 0.35
    tourist_start_frame: int = 30

CONFIG = Config()


def _trail_length(wps):
    total = 0.0
    for i in range(len(wps)-1):
        dx = wps[i+1][0]-wps[i][0]; dy = wps[i+1][1]-wps[i][1]
        total += math.sqrt(dx*dx+dy*dy)
    return total

def _trail_pos_at(wps, dist_along):
    """Return (x,y) at `dist_along` along the trail (wraps).
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
    total = _trail_length(wps)
    dist_along = dist_along % total
    acc = 0.0
    for i in range(len(wps)-1):
        dx = wps[i+1][0]-wps[i][0]; dy = wps[i+1][1]-wps[i][1]
        seg = math.sqrt(dx*dx+dy*dy)
        if acc+seg >= dist_along:
            t = (dist_along-acc)/max(seg,1e-5)
            return wps[i][0]+dx*t, wps[i][1]+dy*t
        acc += seg
    return wps[-1]


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        self.trails = [cfg.trail_a, cfg.trail_b]
        # Each tourist: (trail_idx, phase_offset) → determines position along trail
        self.tourists: List[Dict] = []
        tl_a = _trail_length(cfg.trail_a); tl_b = _trail_length(cfg.trail_b)
        for t in range(cfg.num_tourists_per_trail):
            self.tourists.append({"trail": 0, "dist": tl_a * t / cfg.num_tourists_per_trail, "tl": tl_a})
            self.tourists.append({"trail": 1, "dist": tl_b * t / cfg.num_tourists_per_trail, "tl": tl_b})
        self.tourist_pos_history: List[List] = []  # list of list of (x,y) per frame

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        if fi < cfg.tourist_start_frame:
            # Still record dummy positions so history aligns with frames
            self.tourist_pos_history.append([(t["trail"]*1000, 0) for t in self.tourists])
            return
        # Advance each tourist along its trail
        for t in self.tourists:
            t["dist"] = (t["dist"] + cfg.tourist_speed) % t["tl"]
        frame_pos = []
        for t in self.tourists:
            x, y = _trail_pos_at(self.trails[t["trail"]], t["dist"])
            frame_pos.append((x, y))
        self.tourist_pos_history.append(frame_pos)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        if fi < self.cfg.tourist_start_frame or not self.tourist_pos_history: return f
        cfg = self.cfg
        frame_pos = self.tourist_pos_history[-1]
        for (tx, ty) in frame_pos:
            tp = torch.tensor([tx, ty], device=self.dev)
            d = torch.norm(self.pos - tp, dim=1)
            flush = sm & (d < cfg.tourist_flush_radius)
            if flush.any():
                away = self.pos[flush] - tp
                f[flush] += (away / d[flush].unsqueeze(1).clamp(min=1.)) * cfg.tourist_flush_force
                self.alarm_level[flush] = torch.clamp(self.alarm_level[flush] + cfg.tourist_alarm_add, max=1.)
                self.alarm_vectors[flush] = away / d[flush].unsqueeze(1).clamp(min=1.)
        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []
        # Draw trails
        colors = ["#80cbc4", "#ffcc80"]
        for ti, (trail, col) in enumerate(zip(self.trails, colors)):
            pts = " ".join(f"{x},{y}" for x,y in trail)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5" stroke-dasharray="8,4" opacity="0.45"/>')
            # waypoint dots
            for x, y in trail[:-1]:
                out.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{col}" opacity="0.5"/>')

        # Animate tourists - one circle per tourist
        for ti_i, t in enumerate(self.tourists):
            col = colors[t["trail"]]
            xs = ";".join(f"{self.tourist_pos_history[fi][ti_i][0]:.1f}" if fi < len(self.tourist_pos_history) else "0" for fi in range(F))
            ys = ";".join(f"{self.tourist_pos_history[fi][ti_i][1]:.1f}" if fi < len(self.tourist_pos_history) else "0" for fi in range(F))
            vis = ";".join("1.0" if fi >= cfg.tourist_start_frame else "0.0" for fi in range(F))
            # Flush radius halo
            out.append(f'<circle r="{cfg.tourist_flush_radius:.0f}" fill="{col}" fill-opacity="0.07" stroke="{col}" stroke-width="1" stroke-opacity="0.25">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            # Tourist dot
            out.append(f'<circle r="7" fill="{col}" stroke="#fff" stroke-width="1.5">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg; nt = len(self.tourists)
        lines = [
            (f"{nt} tourists walk 2 looping trails from frame {cfg.tourist_start_frame}.", "#b2dfdb"),
            (f"Birds flush when a tourist enters {cfg.tourist_flush_radius:.0f} radius.", "#b2dfdb"),
            ("Alarm propagates outward from flushed birds.", "#80cbc4"),
            ("Trails shown as dashed coloured polylines.", "#80cbc4"),
            ("Wildlife avoids trail corridors, stays in quiet zones.", "#4db6ac"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Ecotourism Trails", lines, "#80cbc4")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Ecotourism Trails", "Tourist agents flush fauna along fixed loop paths", "#80cbc4")


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

    print(f" - Ecotourism Trails on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print("Done. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_40')

if __name__ == "__main__": main()
