# -*- coding: utf-8 -*-
# MODULE 48: Apiary Pollination (Bee nodes increase fruit regeneration rate) - 670 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Apiary Pollination (Bee nodes increase fruit regeneration rate) - 670 frames emphasizing microhabitat refugia.
- Indicator species: Borboleta-88 (Diaethria clymena).
- Pollination lens: flowering phenology under drought stress.
- Human impact lens: tourism peak season stress.

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

    frames: int = 670
    # Harvesting parameters (from 47)
    harvester_count: int = 2
    harvester_speed: float = 2.5
    harvester_collect_radius: float = 30.0
    harvester_deplete_amount: float = 1.0
    harvester_fear_radius: float = 50.0
    harvester_fear_force: float = 6.0
    harvest_start_frame: int = 40
    # Apiary Parameters (NEW 48)
    apiary_locations: tuple = ((200.0, 300.0), (1000.0, 250.0))
    pollination_radius: float = 200.0
    pollination_regen_boost: float = 0.015  # Additional regeneration rate per frame
    base_regen_rate: float = 0.005

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg

        # 1. Harvest Nodes (from 47)
        self.harvest_nodes = torch.tensor([[200., 200.], [250., 450.], [850., 350.], [1100., 250.]], device=self.dev, dtype=torch.float32)
        self.node_fruits = torch.ones(len(self.harvest_nodes), device=self.dev, dtype=torch.float32)

        # 2. Harvesters (from 47)
        self.harvesters = []
        for _ in range(cfg.harvester_count):
            self.harvesters.append({"pos": torch.tensor([cfg.width/2, cfg.height/2], device=self.dev, dtype=torch.float32),
                                    "target": -1, "state": "wandering", "basket": 0.0, "angle": random.uniform(0, math.pi*2)})
        self.harvester_history = []
        self.fruit_history = []

        # 3. Apiaries (from 48)
        self.apiaries = torch.tensor(cfg.apiary_locations, device=self.dev, dtype=torch.float32)

        # Precompute which harvest nodes get boosted
        d_apiary = torch.cdist(self.harvest_nodes, self.apiaries)
        min_d, _ = torch.min(d_apiary, dim=1)
        self.is_boosted = min_d < cfg.pollination_radius

        # Just for SVG animation visual
        self.apiary_bee_angles = torch.rand((len(self.apiaries), 5), device=self.dev) * math.pi * 2

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg

        # Regeneration (base + boosted)
        regen_rates = torch.full_like(self.node_fruits, cfg.base_regen_rate)
        if self.is_boosted.any():
            regen_rates[self.is_boosted] += cfg.pollination_regen_boost

        self.node_fruits = torch.clamp(self.node_fruits + regen_rates, 0.0, 1.0)

        # Harvester state machine
        if fi >= cfg.harvest_start_frame:
            frame_h = []
            for h in self.harvesters:
                hp = h["pos"]
                if h["state"] == "wandering":
                    ripe_idx = (self.node_fruits > 0.8).nonzero().squeeze(1)
                    if len(ripe_idx) > 0:
                        dists = torch.norm(self.harvest_nodes[ripe_idx] - hp, dim=1)
                        best_i = ripe_idx[torch.argmin(dists)].item()
                        h["target"] = best_i
                        h["state"] = "moving"
                    else:
                        h["angle"] += random.uniform(-0.2, 0.2)
                        h["pos"][0] += math.cos(h["angle"]) * cfg.harvester_speed * 0.5
                        h["pos"][1] += math.sin(h["angle"]) * cfg.harvester_speed * 0.5
                elif h["state"] == "moving":
                    target_pos = self.harvest_nodes[h["target"]]
                    vec = target_pos - hp
                    dist = torch.norm(vec).item()
                    if dist < 10.0: h["state"] = "gathering"
                    else:
                        h["angle"] = math.atan2(vec[1].item(), vec[0].item())
                        h["pos"] += (vec / max(dist, 1e-5)) * cfg.harvester_speed
                elif h["state"] == "gathering":
                    t = h["target"]
                    if self.node_fruits[t] > 0.1:
                        self.node_fruits[t] -= 0.02
                        h["basket"] += 0.02
                    else:
                        h["target"] = -1; h["state"] = "wandering"; h["basket"] = 0.0

                h["pos"][0] = torch.clamp(h["pos"][0], 50, cfg.width-50)
                h["pos"][1] = torch.clamp(h["pos"][1], 50, cfg.height-50)
                frame_h.append({"x": h["pos"][0].item(), "y": h["pos"][1].item()})
            self.harvester_history.append(frame_h)
        else:
            self.harvester_history.append([{"x": h["pos"][0].item(), "y": h["pos"][1].item()} for h in self.harvesters])

        self.fruit_history.append(self.node_fruits.cpu().numpy().copy())

        # Advance bee angles for rendering
        self.apiary_bee_angles += torch.rand_like(self.apiary_bee_angles) * 0.2 + 0.1

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg

        # Harvest Node attraction
        ripe = self.node_fruits > 0.3
        if ripe.any() and (sm & ~self.is_grazer).any():
            eligible = sm & ~self.is_grazer
            valid_nodes = self.harvest_nodes[ripe]
            valid_amounts = self.node_fruits[ripe]
            d2 = torch.cdist(self.pos, valid_nodes)
            mn, cl = torch.min(d2, dim=1)
            near = eligible & (mn < 100.0)
            if near.any():
                pull = valid_nodes[cl[near]] - self.pos[near]
                f[near] += (pull / mn[near].unsqueeze(1).clamp(min=1.)) * 2.5 * valid_amounts[cl[near]].unsqueeze(1)
                feeding = near & (mn < 15.0)
                if feeding.any():
                    self.energy[feeding] += 2.0
                    for ci in cl[feeding]:
                        real_idx = ripe.nonzero().squeeze(1)[ci]
                        self.node_fruits[real_idx] = max(0.0, self.node_fruits[real_idx].item() - 0.002)

        # Harvester fear
        if fi >= cfg.harvest_start_frame:
            for h in self.harvesters:
                hp = h["pos"]
                d = torch.norm(self.pos - hp, dim=1)
                fear = sm & (d < cfg.harvester_fear_radius)
                if fear.any():
                    away = self.pos[fear] - hp
                    f[fear] += (away / d[fear].unsqueeze(1).clamp(min=1.)) * cfg.harvester_fear_force

        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # 1. Apiaries & Pollination Rings
        for i, (ax, ay) in enumerate(cfg.apiary_locations):
            # Outer pollination area
            out.append(f'<circle cx="{ax}" cy="{ay}" r="{cfg.pollination_radius}" fill="url(#noiseGrad)" opacity="0.15"/>')
            out.append(f'<circle cx="{ax}" cy="{ay}" r="{cfg.pollination_radius}" fill="none" stroke="#fbc02d" stroke-dasharray="8,4" stroke-width="1.5" opacity="0.5"/>')

            # Apiary box
            out.append(f'<rect x="{ax-6}" y="{ay-6}" width="12" height="12" fill="#1e1e1e" rx="2" stroke="#fff" stroke-width="1"/>')
            out.append(f'<rect x="{ax-4}" y="{ay-2}" width="8" height="2" fill="#212121"/>')

            # Floating bees (purely animated in SVG via spinning orbit)
            for b in range(5):
                out.append(f'<circle r="1.5" fill="#fbc02d">'
                           f'<animateTransform attributeName="transform" type="rotate" from="0 {ax} {ay}" to="{360 if b%2==0 else -360} {ax} {ay}" dur="{dur / (b+2)}s" repeatCount="indefinite"/>'
                           f'<animateTransform attributeName="transform" type="translate" values="{ax+10+(b*2)},{ay}; {ax+5}, {ay+10+(b*2)}; {ax+10+(b*2)},{ay}" dur="{(b+1)*0.5}s" repeatCount="indefinite" additive="sum"/>'
                           f'</circle>')

        # 2. Harvest Trees (Pequizeiros)
        for i, (hx, hy) in enumerate(self.harvest_nodes.cpu().numpy()):
            # Highlight border if boosted
            if self.is_boosted[i]:
                out.append(f'<circle cx="{hx}" cy="{hy}" r="17" fill="none" stroke="#fdd835" stroke-dasharray="2,2" stroke-width="2" opacity="0.8"/>')

            # Outer tree crown
            out.append(f'<circle cx="{hx}" cy="{hy}" r="14" fill="#689f38" opacity="0.6"/>')

            # Inner yellow fruit indicator
            rads = ";".join(f"{1.0 + self.fruit_history[fi][i]*8.0:.1f}" for fi in range(F))
            ops = ";".join(f"{0.2 + self.fruit_history[fi][i]*0.8:.2f}" for fi in range(F))
            out.append(f'<circle cx="{hx}" cy="{hy}" fill="#fbc02d" stroke="#f9a825" stroke-width="1.5">'
                       f'<animate attributeName="r" values="{rads}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # 3. Harvesters
        for i in range(cfg.harvester_count):
            xs = ";".join(f"{self.harvester_history[fi][i]['x']:.1f}" for fi in range(F))
            ys = ";".join(f"{self.harvester_history[fi][i]['y']:.1f}" for fi in range(F))
            vis = ";".join("1.0" if fi >= cfg.harvest_start_frame else "0.0" for fi in range(F))
            out.append(f'<circle r="6" fill="#f57c00" stroke="#ffcc80" stroke-width="2">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
            vis_faint = vis.replace("1.0", "0.3")
            out.append(f'<circle r="{cfg.harvester_fear_radius:.0f}" fill="none" stroke="#f57c00" stroke-width="1" stroke-dasharray="4,4" opacity="0.0">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis_faint}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        lines = [
            (f"{len(cfg.apiary_locations)} Apiaries placed, radiating a {cfg.pollination_radius:.0f} pollination zone.", "#fff59d"),
            (f"Trees in pollination zones yield {cfg.pollination_regen_boost/cfg.base_regen_rate:.0f}x faster.", "#ffeb3b"),
            ("Boosted trees have a dashed yellow ring.", "#fbc02d"),
            ("Harvesters gain from the increased yield.", "#fbc02d"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Apiary Pollination", lines, "#fbc02d")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Apiary Pollination", "Bee nodes continuously increase the fruit regeneration rate of nearby trees", "#fbc02d")


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

    print(f" - Apiary Pollination on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    boost_count = int(sim.is_boosted.sum().item())
    print(f"Done. Boosted trees: {boost_count}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_48')

if __name__ == "__main__": main()
