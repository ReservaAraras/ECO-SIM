# -*- coding: utf-8 -*-
# MODULE 52: Zoning Policy (Preservation, Sustainable Use, Buffer zones) - 710 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Zoning Policy (Preservation, Sustainable Use, Buffer zones) - 710 frames emphasizing human-wildlife conflict zones.
- Indicator species: Gafanhoto (Schistocerca sp.).
- Pollination lens: nocturnal bat pollination window.
- Human impact lens: road mortality and barrier effects.

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
from typing import List, Tuple
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 710
    # Zones are defined as (x, y, w, h, zone_type)
    # zone_type: 0=Preservation, 1=Sustainable Use, 2=Buffer
    zones: tuple = (
        (30,  60,  360, 480, 0),   # Preservation - left block
        (420, 60,  440, 480, 2),   # Buffer - center
        (880, 60,  370, 480, 1),   # Sustainable Use - right block
    )
    # Per-zone parameters
    pres_energy_bonus: float = 4.0      # Preservation: boosts fauna strongly
    pres_mortality_reduction: float = 0.5  # 50% less predation success inside
    pres_alarm_decay_boost: float = 0.08   # alarm fades faster (calm zone)
    use_energy_drain: float = 0.05         # Sustainable Use: slight drain (controlled harvesting)
    use_grazer_attract: float = 3.0        # cattle allowed here
    buf_alarm_add: float = 0.1             # Buffer: mild stress from edge effects
    buf_repel_grazers: float = 5.0         # grazers pushed back from preservation/buffer

CONFIG = Config()

ZONE_COLORS = ["#1b5e20", "#f57f17", "#0d47a1"]  # dark green, amber, dark blue
ZONE_LABELS = ["Preservation", "Sustainable Use", "Buffer Zone"]
ZONE_TEXT_COLORS = ["#c8e6c9", "#ffe082", "#bbdefb"]


def _in_zone(pos_x, pos_y, zone):
    zx, zy, zw, zh, _ = zone
    return (zx <= pos_x <= zx + zw) and (zy <= pos_y <= zy + zh)


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        self.zone_tensors: List[Tuple] = []
        for (zx, zy, zw, zh, zt) in cfg.zones:
            mins = torch.tensor([zx, zy], device=self.dev, dtype=torch.float32)
            maxs = torch.tensor([zx+zw, zy+zh], device=self.dev, dtype=torch.float32)
            self.zone_tensors.append((mins, maxs, zt))

        # For counting: record zone population each frame
        self.zone_pop_history: List[List[int]] = []

    def _in_zone_mask(self, zone_idx):
        mins, maxs, _ = self.zone_tensors[zone_idx]
        return ((self.pos[:, 0] >= mins[0]) & (self.pos[:, 0] <= maxs[0]) &
                (self.pos[:, 1] >= mins[1]) & (self.pos[:, 1] <= maxs[1]))

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg

        frame_pops = []
        for zi, zone in enumerate(cfg.zones):
            _, _, zt = self.zone_tensors[zi]
            in_zone = self._in_zone_mask(zi) & am

            # Preservation zone
            if zt == 0:
                active_in = in_zone & sm
                if active_in.any():
                    self.energy[active_in] += cfg.pres_energy_bonus * 0.05
                    # Alarm decays faster
                    self.alarm_level[active_in] = torch.clamp(
                        self.alarm_level[active_in] - cfg.pres_alarm_decay_boost, min=0.)

            # Sustainable Use zone
            elif zt == 1:
                active_in = in_zone & sm
                if active_in.any():
                    self.energy[active_in] -= cfg.use_energy_drain

            # Buffer zone: slight stress for all
            elif zt == 2:
                active_in = in_zone & sm
                if active_in.any():
                    self.alarm_level[active_in] = torch.clamp(
                        self.alarm_level[active_in] + cfg.buf_alarm_add * 0.1, max=1.)

            frame_pops.append(int(in_zone.sum()))

        self.zone_pop_history.append(frame_pops)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg

        # Grazers pushed out of Preservation zone (0) and Buffer (2)
        for zi, zone in enumerate(cfg.zones):
            _, _, zt = self.zone_tensors[zi]
            if zt in (0, 2):
                mins, maxs, _ = self.zone_tensors[zi]
                in_zone = self._in_zone_mask(zi) & sm & self.is_grazer
                if in_zone.any():
                    center = (mins + maxs) / 2
                    away = self.pos[in_zone] - center
                    f[in_zone] += (away / away.norm(dim=1, keepdim=True).clamp(min=1.)) * cfg.buf_repel_grazers

        # Birds attracted to Preservation zone
        mins_p, maxs_p, _ = self.zone_tensors[0]
        center_p = (mins_p + maxs_p) / 2
        d_pres = torch.norm(self.pos - center_p, dim=1)
        near_pres = sm & self.flies & ~self.is_carnivore & (d_pres < 300.)
        if near_pres.any():
            pull = center_p - self.pos[near_pres]
            f[near_pres] += (pull / d_pres[near_pres].unsqueeze(1).clamp(min=1.)) * 1.8

        # Grazers attracted to Sustainable Use zone
        mins_u, maxs_u, _ = self.zone_tensors[1]
        center_u = (mins_u + maxs_u) / 2
        d_use = torch.norm(self.pos - center_u, dim=1)
        near_use = sm & self.is_grazer & (d_use < 300.)
        if near_use.any():
            pull = center_u - self.pos[near_use]
            f[near_use] += (pull / d_use[near_use].unsqueeze(1).clamp(min=1.)) * cfg.use_grazer_attract

        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; out = []
        # Draw zones (static - they don't move)
        for zi, (zx, zy, zw, zh, zt) in enumerate(cfg.zones):
            col = ZONE_COLORS[zt]
            lbl = ZONE_LABELS[zt]
            tcol = ZONE_TEXT_COLORS[zt]
            # Fill rect
            out.append(f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" fill="{col}" rx="6" opacity="0.18"/>')
            # Border
            out.append(f'<rect x="{zx}" y="{zy}" width="{zw}" height="{zh}" fill="none" stroke="{col}" stroke-width="2.5" rx="6" opacity="0.7"/>')
            # Label
            out.append(f'<text x="{zx + zw/2:.0f}" y="{zy + 22:.0f}" text-anchor="middle" font-size="15" font-weight="bold" fill="{tcol}" opacity="0.9">{lbl}</text>')
            # Population counter
            F = cfg.frames; dur = F / cfg.fps
            pop_vals = ";".join(str(self.zone_pop_history[fi][zi]) for fi in range(F))
            out.append(f'<text font-weight="bold" x="{zx + zw/2:.0f}" y="{zy + 40:.0f}" text-anchor="middle" font-size="15" fill="{tcol}" opacity="0.7">'
                       f'<animate attributeName="textContent" values="{pop_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</text>')

        # Dashed boundary lines between zones
        for zi in range(len(cfg.zones) - 1):
            zx, zy, zw, zh, _ = cfg.zones[zi]
            bx = zx + zw
            out.append(f'<line x1="{bx}" y1="{zy}" x2="{bx}" y2="{zy+zh}" stroke="#fff" stroke-dasharray="6,4" stroke-width="1.5" opacity="0.3"/>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        if self.zone_pop_history:
            fp = self.zone_pop_history[-1]
        else:
            fp = [0, 0, 0]
        lines = [
            ("Protection (green): energy up, alarm gone, no grazers.", ZONE_TEXT_COLORS[0]),
            ("Buffer (blue): edge-effect stress, grazers excluded.", ZONE_TEXT_COLORS[2]),
            ("Sustainable Use (amber): grazers allowed, slight drain.", ZONE_TEXT_COLORS[1]),
            (f"Final populations - Pres:{fp[0]} | Buf:{fp[2]} | SustUse:{fp[1]}", "#e0e0e0"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Zoning Policy", lines, "#4caf50")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Zoning Policy", "Conservation zones define differential rules across the landscape", "#388e3c")


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

    print(f" - Zoning Policy on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    if sim.zone_pop_history:
        fp = sim.zone_pop_history[-1]
        print(f"Final zone populations: Preservation={fp[0]}, Sustainable Use={fp[1]}, Buffer={fp[2]}")
    print("Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_52')

if __name__ == "__main__": main()
