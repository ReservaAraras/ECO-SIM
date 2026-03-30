# -*- coding: utf-8 -*-
# MODULE 46: Community Monitoring (Camera traps highlight birds passing through zones) - 650 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Community Monitoring — Camera traps placed by RESEX extractivist families
  along vereda migration corridors, detecting both resident and migratory avifauna - 650 frames
  emphasising water table fluctuations and seasonal migratory pulse tracking.
- Indicator species: Abelha-mamangava (Xylocopa sp.) — proxy for pollination corridor health.
- Pollination lens: nectar robbing effects along camera-monitored corridors.
- Human impact lens: agroforestry edge subsidies; community-based adaptive management response
  to migratory detections.

Social-Migratory Coherence (RESEX Recanto das Araras, São Domingos — Goiás):
- Camera traps are positioned by local RESEX families at known bird passage points along the
  vereda corridors (Mauritia flexuosa gallery strips) and karst sinkhole margins, reflecting
  traditional ecological knowledge of migration routes through the São Domingos uplands.
- When migratory species are detected above a community-defined threshold, a 'community alert'
  is triggered — signalling management actions: increased patrol frequency, temporary livestock
  exclusion from corridor zones, and notification of the RESEX governing council (Conselho
  Deliberativo).
- This notebook models the social feedback loop: migratory detections by community-operated
  cameras inform adaptive stewardship decisions, operationalising the Plano de Gestão Integrado
  (PGI) of the RESEX Recanto das Araras.

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
import torch # pyre-ignore[21]
from dataclasses import dataclass
from typing import List, Dict, Set
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 650
    # Camera Trap Parameters
    camera_locations: tuple = ((300.0, 300.0), (800.0, 150.0), (1050.0, 450.0))
    camera_radius: float = 65.0
    camera_flash_frames: int = 4
    # Social-migratory coupling: threshold to trigger RESEX community migration alert
    migration_detection_threshold: int = 3   # migratory individuals that must be photographed
    # We want to draw a highlight around identified birds permanently

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        cfg = self.cfg
        self.cameras = torch.tensor(cfg.camera_locations, device=self.dev, dtype=torch.float32)

        # Track which particles have been photographed (identified)
        self.identified_particles: Set[int] = set()
        # Migrant-specific tracking for RESEX community alert logic
        self.migrant_detections: Set[int] = set()        # photographed migratory individuals
        self.community_alert_frame: Optional[int] = None # frame when community alert was triggered

        # Track flash events for SVG: [{"cam_idx": int, "frame": int}]
        self.camera_flashes: List[Dict] = []

        # To make it visible, we will record the set of identified particles per frame
        self.identified_history: List[List[bool]] = []

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg

        # Check active birds in camera ranges
        if am.any() and self.flies.any():
            birds = am & self.flies
            if birds.any():
                bird_pos = self.pos[birds]
                bird_indices = birds.nonzero().squeeze(1)

                # Distance from birds to cameras
                dist_matrix = torch.cdist(bird_pos, self.cameras)

                for c_idx in range(len(self.cameras)):
                    # Birds within radius of camera c_idx
                    in_range = dist_matrix[:, c_idx] < cfg.camera_radius
                    if in_range.any():
                        caught = bird_indices[in_range]
                        flashed = False
                        for b_id in caught.cpu().numpy():
                            if b_id not in self.identified_particles:
                                self.identified_particles.add(int(b_id))
                                flashed = True
                                # Distinguish migratory vs resident detections for community alert
                                if self.is_migrant[b_id].item():
                                    self.migrant_detections.add(int(b_id))

                        if flashed:
                            self.camera_flashes.append({"cam_idx": c_idx, "frame": fi})
                        # RESEX community migration alert: triggered when migratory detections
                        # reach the community-defined threshold (cfg.migration_detection_threshold)
                        if (self.community_alert_frame is None and
                                len(self.migrant_detections) >= cfg.migration_detection_threshold):
                            self.community_alert_frame = fi

        # Store boolean mask of currently identified particles for rendering
        identified_mask = [i in self.identified_particles for i in range(cfg.max_particles)]
        self.identified_history.append(identified_mask)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        return torch.zeros_like(self.vel)

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Draw Camera Zones
        for ci, (cx, cy) in enumerate(cfg.camera_locations):
            # Zone border
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{cfg.camera_radius}" fill="none" stroke="#fff" stroke-dasharray="4,4" stroke-width="1.5" opacity="0.3"/>')
            # Camera icon (small triangle/box)
            out.append(f'<rect x="{cx-6}" y="{cy-4}" width="12" height="8" fill="#1e1e1e" rx="2" stroke="#fff" stroke-width="1"/>')
            out.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="#90caf9"/>')

            # Flashes: animate opacity spiked on frames where a new bird was caught
            flash_ops = []
            for fi in range(F):
                age = next((fi - f["frame"] for f in self.camera_flashes if f["cam_idx"] == ci and 0 <= fi - f["frame"] <= cfg.camera_flash_frames), -1)
                if age >= 0:
                    flash_ops.append(f"{1.0 - (age/cfg.camera_flash_frames):.2f}")
                else:
                    flash_ops.append("0.0")
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{cfg.camera_radius}" fill="#fff" opacity="0.0" pointer-events="none">'
                       f'<animate attributeName="opacity" values="{";".join(flash_ops)}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Draw tracking halos around identified birds
        # EcosystemBase draws particles *after* extra_svg, but *before* extra_svg_overlay.
        # We can draw halos exactly on top of them here.
        for idx in range(cfg.max_particles):
            if not self.flies[idx].item() or not any(self.identified_history[fi][idx] for fi in range(F)):
                continue

            xs = ";".join(f"{self.trajectory_history[fi][idx,0]:.1f}" for fi in range(F))
            ys = ";".join(f"{self.trajectory_history[fi][idx,1]:.1f}" for fi in range(F))

            # Opacity rules: visible if active AND identified
            ops = ";".join("1.0" if (self.active_history[fi][idx] and self.visibility_history[fi][idx] and self.identified_history[fi][idx]) else "0.0" for fi in range(F))

            rad = 7 if self.is_migrant[idx].item() else 6
            pc = self.colors[idx]

            # Outer white tracking crosshair brackets
            out.append(f'<circle r="{rad+3}" fill="none" stroke="#fff" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.0">'
                       f'<animate attributeName="cx" values="{xs}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="cy" values="{ys}" dur="{dur}s" repeatCount="indefinite"/>'
                       f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Info Card — community monitoring with migration corridor awareness
        total_id = len(self.identified_particles)
        migrant_id = len(self.migrant_detections)
        if self.community_alert_frame is not None:
            alert_str = f"Migration alert at frame {self.community_alert_frame}: corridor patrol activated."
        else:
            alert_str = "Awaiting community migration alert threshold."
        lines = [
            (f"{len(cfg.camera_locations)} cameras placed along vereda migration corridors.", "#e3f2fd"),
            (f"Total unique birds identified: {total_id} (residents + migrants).", "#bbdefb"),
            (f"Migratory species photographed: {migrant_id} individuals.", "#90caf9"),
            (alert_str, "#64b5f6"),
        ]
        out.append(self.info_card(cfg.width, cfg.height, "Feature: Community Monitoring & Migration", lines, "#90caf9"))
        return "".join(out)

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Community Monitoring", "Camera traps photograph and tag birds flying through observation zones", "#90caf9")


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

    print(f" - Community Monitoring on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    print(f"Total birds identified: {len(sim.identified_particles)}. Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_46')

if __name__ == "__main__": main()
