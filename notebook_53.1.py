# -*- coding: utf-8 -*-
# MODULE 53: Resilience Test (Fire + Drought shock, then measure recovery) - 720 frames
"""
Module documentation.
Notebook Differentiation:
- Differentiation Focus: Resilience Test (Fire + Drought shock, then measure recovery) - 720 frames emphasizing noise pollution zones.
- Indicator species: Minhoca-gigante (Glossoscolex sp.).
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

from eco_base import save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
import os
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, math # pyre-ignore[21]
from dataclasses import dataclass
from typing import List
from eco_base import EcosystemBase, BaseConfig, save_svg , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
from IPython.display import display, HTML # pyre-ignore[21]

@dataclass
class Config(BaseConfig):
    """Class `Config` -- simulation component."""

    frames: int = 720
    # ── Shock window ────────────────────────────────────────────────────────
    shock_start: int = 120          # when fire+drought strike simultaneously
    shock_end:   int = 300          # when both pressures lift
    # Fire shock parameters
    shock_fire_x: float = 640.0
    shock_fire_y: float = 300.0
    shock_fire_radius: float = 220.0
    shock_fire_flee_force: float = 18.0
    shock_fire_mortality: float = 0.006   # per-frame kill chance in fire
    # Drought shock parameters
    shock_drought_intensity: float = 0.9  # 0→1, shrinks veredas and drains energy
    shock_drain_rate: float = 0.25        # extra energy decay during shock
    # ── Recovery tracking ───────────────────────────────────────────────────
    recovery_threshold: float = 0.6      # fraction of initial pop → "recovered"

CONFIG = Config()


class Sim(EcosystemBase):
    """Class `Sim` -- simulation component."""

    def _extra_init(self):
        self._initial_pop = CONFIG.initial_particles
        self.shock_active = False
        self.pop_history: List[int] = []
        self.vereda_radius_shock: List[float] = []
        self.recovery_frame: int = -1
        self.fire_pos = torch.tensor(
            [self.cfg.shock_fire_x, self.cfg.shock_fire_y],
            device=self.dev, dtype=torch.float32
        )

    def extra_step(self, fi, am, sm):
        """Function `extra_step` -- simulation component."""

        cfg = self.cfg
        self.pop_history.append(int(am.sum()))

        # ── Shock window ──────────────────────────────────────────────────
        if cfg.shock_start <= fi < cfg.shock_end:
            self.shock_active = True
            progress = (fi - cfg.shock_start) / (cfg.shock_end - cfg.shock_start)

            # Drought: shrink veredas
            factor = 1.0 - cfg.shock_drought_intensity * math.sin(progress * math.pi)
            self.cfg.vereda_max_radius = max(5.0, 90.0 * factor)
            self.cfg.vereda_min_radius = max(2.0, 20.0 * factor)
            self.cfg.energy_decay = 0.12 + cfg.shock_drain_rate * math.sin(progress * math.pi)
            self.vereda_health[:] = torch.clamp(self.vereda_health - 0.01, 0., 1.)

            # Fire: mortality inside radius
            d_fire = torch.norm(self.pos - self.fire_pos, dim=1)
            in_fire = sm & (d_fire < cfg.shock_fire_radius)
            if in_fire.any():
                roll = torch.rand(self.cfg.max_particles, device=self.dev) < cfg.shock_fire_mortality
                killed = in_fire & roll
                if killed.any():
                    for ix in killed.nonzero().squeeze(1):
                        self.is_active[ix] = False
                        n = int(self.particle_nest[ix])
                        if n >= 0:
                            self.nest_occupant[n] = -1
                            self.particle_nest[ix] = -1
                        pn = self.pos[ix].cpu().numpy().copy()
                        self.death_events.append({"pos": pn, "frame": fi, "color": self.colors[ix], "reason": "fire_shock"})
                        self.carrion_sites.append({"pos": pn, "spawn_frame": fi, "expire_frame": fi + self.cfg.carrion_linger_frames})

        else:
            # ── Recovery: restore baselines ────────────────────────────────
            self.shock_active = False
            if fi >= cfg.shock_end:
                self.cfg.vereda_max_radius = min(90.0, self.cfg.vereda_max_radius + 0.3)
                self.cfg.vereda_min_radius = min(20.0, self.cfg.vereda_min_radius + 0.05)
                self.cfg.energy_decay = max(0.12, self.cfg.energy_decay - 0.002)
                self.vereda_health[:] = torch.clamp(self.vereda_health + 0.006, 0., 1.)

            # Look for first recovery frame
            if self.recovery_frame == -1 and fi > cfg.shock_end:
                if int(am.sum()) >= int(self._initial_pop * cfg.recovery_threshold):
                    self.recovery_frame = fi

        self.vereda_radius_shock.append(self.cfg.vereda_max_radius)

    def extra_forces(self, fi, am, sm):
        """Function `extra_forces` -- simulation component."""

        f = torch.zeros_like(self.vel)
        cfg = self.cfg
        if not (cfg.shock_start <= fi < cfg.shock_end):
            return f

        # Fire flee force
        d_fire = torch.norm(self.pos - self.fire_pos, dim=1)
        flee = sm & (d_fire < cfg.shock_fire_radius)
        if flee.any():
            progress = (fi - cfg.shock_start) / (cfg.shock_end - cfg.shock_start)
            strength = math.sin(progress * math.pi)
            away = self.pos[flee] - self.fire_pos
            f[flee] += (away / d_fire[flee].unsqueeze(1).clamp(min=1.)) * cfg.shock_fire_flee_force * strength
            self.alarm_level[flee] = torch.clamp(self.alarm_level[flee] + 0.5, max=1.0)

        return f

    def extra_svg(self):
        """Function `extra_svg` -- simulation component."""

        cfg = self.cfg; F = cfg.frames; dur = F / cfg.fps; out = []

        # Fire glow (active during shock window)
        fx, fy = cfg.shock_fire_x, cfg.shock_fire_y
        ops_fire = []
        for fi in range(F):
            if cfg.shock_start <= fi < cfg.shock_end:
                p = (fi - cfg.shock_start) / (cfg.shock_end - cfg.shock_start)
                ops_fire.append(f"{0.35 * math.sin(p * math.pi):.2f}")
            else:
                ops_fire.append("0.0")
        ov = ";".join(ops_fire)
        # outer glow
        out.append(f'<circle cx="{fx}" cy="{fy}" r="{cfg.shock_fire_radius:.0f}" fill="#ff6f00" opacity="0.0">'
                   f'<animate attributeName="opacity" values="{ov}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')
        # inner core
        out.append(f'<circle cx="{fx}" cy="{fy}" r="40" fill="#fff176" opacity="0.0">'
                   f'<animate attributeName="opacity" values="{ov}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Drought: red tint overlay
        ops_drought = []
        for fi in range(F):
            if cfg.shock_start <= fi < cfg.shock_end:
                p = (fi - cfg.shock_start) / (cfg.shock_end - cfg.shock_start)
                ops_drought.append(f"{0.22 * math.sin(p * math.pi):.2f}")
            else:
                ops_drought.append("0.0")
        dv = ";".join(ops_drought)
        out.append(f'<rect width="{cfg.width}" height="{cfg.height}" fill="#8d6e63" opacity="0.0" pointer-events="none">'
                   f'<animate attributeName="opacity" values="{dv}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></rect>')

        # Population sparkline
        max_pop = max(self.pop_history) if self.pop_history else 1
        bar_y_base = cfg.height - 30
        bar_w = cfg.width / F
        for fi in range(F):
            pop_h = (self.pop_history[fi] / max_pop) * 60
            col = "#ef5350" if cfg.shock_start <= fi < cfg.shock_end else "#66bb6a"
            out.append(f'<rect x="{fi*bar_w:.1f}" y="{bar_y_base - pop_h:.1f}" width="{bar_w+0.5:.1f}" height="{pop_h:.1f}" fill="{col}" opacity="0.65"/>')

        # Recovery marker line
        if self.recovery_frame > 0:
            rx = self.recovery_frame * bar_w
            out.append(f'<line x1="{rx:.1f}" y1="{bar_y_base - 75}" x2="{rx:.1f}" y2="{bar_y_base}" stroke="#fff" stroke-width="1.5" stroke-dasharray="4,3"/>')
            out.append(f'<text font-weight="bold" x="{rx + 4:.1f}" y="{bar_y_base - 60}" fill="#cccccc" font-size="15">Recovery marker</text>')

        # Shock window bracket
        sx = cfg.shock_start * bar_w; ex = cfg.shock_end * bar_w
        out.append(f'<rect x="{sx:.1f}" y="{bar_y_base - 70}" width="{ex-sx:.1f}" height="70" fill="none" stroke="#ff7043" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.8"/>')
        out.append(f'<text font-weight="bold" x="{(sx+ex)/2:.0f}" y="{bar_y_base - 55}" text-anchor="middle" fill="#cccccc" font-size="15">← SHOCK WINDOW →</text>')

        return "".join(out)

    def extra_svg_overlay(self):
        """Function `extra_svg_overlay` -- simulation component."""

        cfg = self.cfg
        min_pop = min(self.pop_history) if self.pop_history else 0
        final_pop = self.pop_history[-1] if self.pop_history else 0
        rf = f"frame {self.recovery_frame}" if self.recovery_frame > 0 else "not reached"
        lines = [
            (f"Fire + Drought shock.", "#ffccbc"),
            (f"Fire kills {cfg.shock_fire_mortality*100:.1f}%/event in a {cfg.shock_fire_radius:.0f} radius.", "#ff8a65"),
            (f"Population minimum during shock: {min_pop} animals.", "#ef9a9a"),
            (f"Final population: {final_pop}.", "#a5d6a7"),
            (f"Ecosystem recovery (>{cfg.recovery_threshold*100:.0f}% of initial) at: {rf}.", "#80cbc4"),
        ]
        return self.info_card(cfg.width, cfg.height, "Feature: Resilience Test", lines, "#ef5350")

    def generate_svg(self):
        """Function `generate_svg` -- simulation component."""

        return self._svg_base("ECO-SIM: Resilience Test", "Combined fire + drought shock → ecosystem crash → measured recovery", "#e64a19")


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

    print(f" - Resilience Test on {CONFIG.device}...")
    sim = Sim(CONFIG); sim.run()
    min_p = min(sim.pop_history) if sim.pop_history else 0
    rf = sim.recovery_frame
    print(f"Shock: frames {CONFIG.shock_start}-{CONFIG.shock_end}. Min pop: {min_p}. Recovery frame: {rf}.")
    print("Generating SVG...")
    svg = sim.generate_svg()
    display(HTML(svg)); save_svg(svg, 'notebook_53.1')

if __name__ == "__main__": main()
