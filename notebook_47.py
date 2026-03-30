# -*- coding: utf-8 -*-
# MODULE 47: Traditional Harvesting (Human agents collect fruit without destroying nodes) - 660 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Traditional Harvesting — RESEX extractivists sustainably collect Pequi
  fruits following a traditional ecological calendar explicitly coupled to migratory bird phenology -
  660 frames emphasising siltation impacts and corridor-aware harvest scheduling.
- Indicator species: Abelha-urucu (Melipona scutellaris) — stingless bee dependent on Pequi flowers.
- Pollination lens: hummingbird territorial patches; bee-mediated Pequi pollination.
- Human impact lens: fire management tradeoffs; migration-calendar harvest scheduling.

Social-Migratory Coherence (RESEX Recanto das Araras, São Domingos — Goiás):
- Pequi (Caryocar brasiliense) fruiting occurs June-August in the Cerrado, overlapping with the
  dry season and coinciding with late-stage migratory bird departures from the landscape.
- Traditional extractivist families of the RESEX maintain a harvest calendar aligned with the
  bird migration cycle: during the peak migration window (Oct-Nov), harvesters voluntarily
  reduce activity near vereda-adjacent Pequí trees to minimise disturbance to corridor species.
  This practice is embedded in the RESEX Plano de Utilização (PU) and reflects centuries of
  co-evolved traditional ecological knowledge (TEK).
- After the migration peak passes, harvesters can freely access all Pequi nodes, consistent with
  the traditional management principle of 'waiting for the birds to pass before harvesting'.
- This notebook models the social-migratory coupling: extractivists target non-corridor Pequi
  nodes during migration peak frames (corridor node shown with blue dashed ring), then resume
  all-node harvesting once migrants have departed.

Scientific Relevance (PIGT RESEX Recanto das Araras — 2024):
- Integrates the socio-environmental complexity of the RESEX Recanto das Araras /
  São Domingos de Goiás, Goiás, Brazil.
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

    frames: int = 660
    # Harvesting parameters
    harvester_count: int = 2
    harvester_speed: float = 2.5
    harvester_collect_radius: float = 30.0
    harvester_deplete_amount: float = 1.0  # how much fruit 'intensity' they remove per frame while gathering
    harvester_fear_radius: float = 50.0    # animals still keep a slight distance, but less than tourists
    harvester_fear_force: float = 6.0
    harvest_start_frame: int = 40
    # Social-migratory calendar coupling: harvesters respect corridor during migration peak
    migration_peak_start_frame: int = 100  # frame when traditional families note migrants arriving
    migration_peak_end_frame:   int = 260  # frame when migration peak ends; all-node harvest resumes
    corridor_node_idx:          int = 1    # index of harvest node adjacent to the vereda corridor

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        # We need independent 'fruit levels' for our fruiting nodes so they can be depleted by humans and regenerate
        # The base logic uses a global sine wave `fi_int` for fruiting.
        # We'll override the base logic by adding our own specific harvestable nodes, or tracking the base ones.
        # Let's add 4 specific "Pequi" trees for traditional harvesting.
        self.harvest_nodes = torch.tensor([[200., 200.], [250., 450.], [850., 350.], [1100., 250.]], device=self.dev, dtype=torch.float32)
        # 1.0 = fully loaded with fruit, 0.0 = empty
        self.node_fruits = torch.ones(len(self.harvest_nodes), device=self.dev, dtype=torch.float32)

        # Harvesters
        self.harvesters = []
        for _ in range(cfg.harvester_count):
            self.harvesters.append({
                "pos": torch.tensor([cfg.width/2, cfg.height/2], device=self.dev, dtype=torch.float32),
                "target": -1,  # index of harvest node they are walking to
                "state": "wandering", # wandering, moving, gathering
                "basket": 0.0,
                "angle": random.uniform(0, math.pi*2)
            })

        self.harvester_history = []
        self.fruit_history = []
        # Social-migratory coupling: track per-frame corridor respect status
        self.corridor_respect_history: List[bool] = []

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg

        # Natural regeneration of fruit
        self.node_fruits = torch.clamp(self.node_fruits + 0.005, 0.0, 1.0)

        if fi >= cfg.harvest_start_frame:
            frame_h = []
            # Social-migratory calendar: determine whether we are in the migration peak window
            in_migration_peak = cfg.migration_peak_start_frame <= fi < cfg.migration_peak_end_frame
            for h in self.harvesters:
                hp = h["pos"]

                # State machine
                if h["state"] == "wandering":
                    # Look for ripe fruit nodes
                    ripe_idx = (self.node_fruits > 0.8).nonzero().squeeze(1)
                    # During migration peak: avoid the corridor-adjacent Pequi node (TEK protocol)
                    if in_migration_peak and len(ripe_idx) > 0:
                        ripe_idx = ripe_idx[ripe_idx != cfg.corridor_node_idx]
                    if len(ripe_idx) > 0:
                        # pick closest
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
                    if dist < 10.0:
                        h["state"] = "gathering"
                    else:
                        h["angle"] = math.atan2(vec[1].item(), vec[0].item())
                        h["pos"] += (vec / max(dist, 1e-5)) * cfg.harvester_speed

                elif h["state"] == "gathering":
                    t = h["target"]
                    if self.node_fruits[t] > 0.1:
                        self.node_fruits[t] -= 0.02 # deplete
                        h["basket"] += 0.02
                    else:
                        h["target"] = -1
                        h["state"] = "wandering"
                        h["basket"] = 0.0 # dropped off / reset

                # bounds
                h["pos"][0] = torch.clamp(h["pos"][0], 50, cfg.width-50)
                h["pos"][1] = torch.clamp(h["pos"][1], 50, cfg.height-50)
                frame_h.append({"x": h["pos"][0].item(), "y": h["pos"][1].item()})
            self.harvester_history.append(frame_h)
        else:
            in_migration_peak = False
            self.harvester_history.append([{"x": h["pos"][0].item(), "y": h["pos"][1].item()} for h in self.harvesters])

        self.corridor_respect_history.append(in_migration_peak)
        self.fruit_history.append(self.node_fruits.cpu().numpy().copy())

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg

        # Frugivores attracted to harvest nodes if they have fruit
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
                # Attract proportional to fruit left
                f[near] += (pull / mn[near].unsqueeze(1).clamp(min=1.)) * 2.5 * valid_amounts[cl[near]].unsqueeze(1)

                # Animals deplete fruit too
                feeding = near & (mn < 15.0)
                if feeding.any():
                    self.energy[feeding] += 2.0
                    for ci in cl[feeding]:
                        real_idx = ripe.nonzero().squeeze(1)[ci]
                        self.node_fruits[real_idx] = max(0.0, self.node_fruits[real_idx].item() - 0.002)

        # Fauna slight repel from harvesters
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

        # Draw harvest nodes (Pequizeiros)
        for i, (hx, hy) in enumerate(self.harvest_nodes.cpu().numpy()):
            # Outer tree crown
            out.append(f'<circle cx="{hx}" cy="{hy}" r="14" fill="#689f38" opacity="0.6"/>')
            # Inner yellow fruit indicator, scales with fruit level
            rads = ";".join(f"{1.0 + self.fruit_history[fi][i]*8.0:.1f}" for fi in range(F))
            ops = ";".join(f"{0.2 + self.fruit_history[fi][i]*0.8:.2f}" for fi in range(F))
            out.append(f'<circle cx="{hx}" cy="{hy}" fill="#fbc02d" stroke="#f9a825" stroke-width="1.5">'
                       f'<animate attributeName="r" values="{rads}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'</circle>')

        # Migration corridor protection: animated blue dashed ring on corridor-adjacent node during peak
        cn_pos = self.harvest_nodes[cfg.corridor_node_idx].cpu().numpy()
        cnx, cny = float(cn_pos[0]), float(cn_pos[1])
        corridor_ops = ";".join(
            "0.75" if cfg.migration_peak_start_frame <= fi < cfg.migration_peak_end_frame else "0.0"
            for fi in range(F)
        )
        out.append(
            f'<circle cx="{cnx:.1f}" cy="{cny:.1f}" r="22" fill="none" stroke="#4fc3f7" '
            f'stroke-width="2.5" stroke-dasharray="5,3" opacity="0.0">'
            f'<animate attributeName="opacity" values="{corridor_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</circle>'
        )
        out.append(
            f'<text font-weight="bold" x="{cnx + 28:.1f}" y="{cny:.1f}" font-size="15" fill="#4fc3f7" '
            f'dominant-baseline="middle" opacity="0.0">migration corridor'
            f'<animate attributeName="opacity" values="{corridor_ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
            f'</text>'
        )

        # Draw Harvesters (Orange icons)
        for i in range(cfg.harvester_count):
            xs = ";".join(f"{self.harvester_history[fi][i]['x']:.1f}" for fi in range(F))
            ys = ";".join(f"{self.harvester_history[fi][i]['y']:.1f}" for fi in range(F))
            vis = ";".join("1.0" if fi >= cfg.harvest_start_frame else "0.0" for fi in range(F))

            # Harvester dot
            out.append(f'<circle r="6" fill="#f57c00" stroke="#ffcc80" stroke-width="2">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{vis}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

            # Fear radius ring
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
            (f"{cfg.harvester_count} extractivists harvest Pequi by tradition.", "#ffe0b2"),
            ("Migration peak: harvester avoids corridor Pequi node.", "#ffb74d"),
            ("Blue ring marks corridor nodes as off-limits.", "#fbc02d"),
            ("Shared resources; minimal avifauna disturbance.", "#ff9800"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Traditional Harvesting & Migration Calendar", lines, "#f57c00")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Traditional Harvesting", "Sustainable extraction by local populations shares resources with fauna", "#f57c00")


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

    print(f" - Traditional Harvesting on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print("Done. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_47')

if __name__ == "__main__": main()
