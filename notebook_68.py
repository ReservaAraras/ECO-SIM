# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 68: Pequizeiro (Caryocar brasiliense) ↔ Morcego-nectarívoro — Spatial Chiropterophily Map
# INTERVENTION 2/4: Seasonal & Migratory Interventions Series
# ===================================================================================================
"""
notebook_68.py — Pequi Tree ↔ Nectar Bats (Chiropterophily):
Notebook Differentiation:
- Differentiation Focus: Pequizeiro (Caryocar brasiliense) ↔ Morcego-nectarívoro — Spatial Chiropterophily Map emphasizing trophic mismatch risk.
- Indicator species: Suiriri (Tyrannus melancholicus).
- Pollination lens: flowering in fire-following shrubs.
- Human impact lens: illegal extraction nodes.

                 Nocturnal Pollination & Migratory Phenology (Spatial Pattern)

Models the highly specialized symbiotic relationship between the iconic Cerrado 
Pequizeiro (Caryocar brasiliense) and migratory vector-pollinators like the 
nectar-feeding bats (Glossophaga spp.) on a spatial map.

The Pequi drops leaves during the peak dry season and produces massive, nectar-rich 
white flowers exclusively at night between August and November. This perfectly 
coincides with the massive influx of migratory bats seeking high-energy nectar sources.

To stay consistent with the evidence-based framing used in the rest of the
series, the map treats this as a spatial coordination problem: flowering timing,
bat arrival, and access to dark movement corridors must overlap for visits to
become effective pollination rather than short-lived nectar extraction.

Scientific Relevance:
  - Demonstrates macro-floral reproductive timing matched to animal migration.
  - Highlights the importance of nocturnal ecological networks in the Cerrado.
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from typing import List, Tuple
from eco_base import save_svg, sanitize_svg_text , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x

# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES
# ===================================================================================================

# Pequi Leaf Canopy (Drops late dry season, rapid flush in spring)
PEQUI_LEAF_CURVE = [
    1.00, 1.00, 1.00, 0.90, 0.60, 0.40, 0.20, 0.10, 0.40, 0.80, 1.00, 1.00
]

# Pequi Nocturnal Bloom (Aug, Sep, Oct, Nov)
PEQUI_BLOOM_CURVE = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.60, 1.00, 0.80, 0.30, 0.00
]

# Pequi Fruiting (Nov, Dec, Jan, Feb)
PEQUI_FRUIT_CURVE = [
    0.70, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.50, 0.90
]

# Migratory Nectar Bat Influx (Arrive in late dry season driven by bloom)
BAT_MIGRATION_CURVE = [0.10, 0.10, 0.15, 0.30, 0.60, 0.85, 0.90, 0.85, 0.70, 0.40, 0.15, 0.10]


@dataclass
class BatPollinationSpatialConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_trees: int = 12
    num_bats: int = 30
    bat_speed: float = 7.0


CONFIG = BatPollinationSpatialConfig()


class SpatialBatPollinationSim:

    def __init__(self, cfg: BatPollinationSpatialConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Trees
        self.trees = []
        for _ in range(cfg.num_trees):
            x = random.uniform(50, cfg.width - 400)
            y = random.uniform(50, cfg.height - 50)
            self.trees.append({
                "pos": (x, y),
                "nectar_load": 0.0,
                "pollinated_count": 0
            })

        # Bats
        self.bats = []
        for _ in range(cfg.num_bats):
            self.bats.append({
                "pos": torch.tensor([
                    random.uniform(50, cfg.width - 400), 
                    random.uniform(50, cfg.height - 50)
                ], device=self.dev, dtype=torch.float32),
                "target_tree": -1,
                "energy": random.uniform(50.0, 100.0)
            })

        self.hist_month = []
        self.hist_bats_xy = [[] for _ in range(cfg.num_bats)]
        
        self.total_nectar_extracted = 0.0
        self.total_flowers_pollinated = 0

    def _interp(self, curve: list, month_frac: float) -> float:
        m = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        month_frac = (frame / cfg.frames) * 12.0
        self.hist_month.append(month_frac)
        
        bloom = self._interp(PEQUI_BLOOM_CURVE, month_frac)
        bat_presence = self._interp(BAT_MIGRATION_CURVE, month_frac)

        # 1. Trees generate nectar during bloom
        for t in self.trees:
            if bloom > 0.1:
                t["nectar_load"] = min(100.0, t["nectar_load"] + bloom * 5.0)
            else:
                t["nectar_load"] *= 0.9

        # 2. Bats forage (Migratory presence dictates how many are active)
        active_bats_count = int(cfg.num_bats * bat_presence)
        
        for i, bat in enumerate(self.bats):
            is_active = i < active_bats_count
            pos = bat["pos"]
            
            if not is_active:
                # Roosting / absent (hide off screen)
                pos[0] = -100.0
                pos[1] = -100.0
                opacity = 0.0
            else:
                opacity = 1.0
                bat["energy"] -= 1.0
                
                # If hungry, find the highest nectar tree
                if bat["target_tree"] == -1 and bat["energy"] < 80.0:
                    best_t = -1
                    best_nectar = 0.0
                    for ti, t in enumerate(self.trees):
                        if t["nectar_load"] > best_nectar:
                            best_nectar = t["nectar_load"]
                            best_t = ti
                    
                    if best_t != -1 and best_nectar > 10.0:
                        bat["target_tree"] = best_t
                
                # Move
                if bat["target_tree"] != -1:
                    t = self.trees[bat["target_tree"]]
                    tgt = torch.tensor(t["pos"], device=self.dev, dtype=torch.float32)
                    vec = tgt - pos
                    dist = torch.norm(vec).item()
                    
                    if dist > 5.0:
                        dir_v = vec / dist
                        jitter = torch.randn(2, device=self.dev) * 1.5
                        pos += dir_v * cfg.bat_speed + jitter
                    else:
                        # Feed & Pollinate
                        drunk = min(t["nectar_load"], 25.0)
                        t["nectar_load"] -= drunk
                        bat["energy"] += drunk
                        self.total_nectar_extracted += drunk
                        self.total_flowers_pollinated += 1
                        t["pollinated_count"] += 1
                        bat["target_tree"] = -1
                else:
                    # Random forage flight
                    pos += torch.randn(2, device=self.dev) * (cfg.bat_speed * 0.8)

                # Clamp to screen area
                if pos[0] < 20: pos[0] = 20
                if pos[0] > cfg.width - 380: pos[0] = cfg.width - 380
                if pos[1] < 20: pos[1] = 20
                if pos[1] > cfg.height - 20: pos[1] = cfg.height - 20
            
            self.hist_bats_xy[i].append((pos[0].item(), pos[1].item(), opacity))


class SpatialBatPollinationRenderer:

    def __init__(self, cfg: BatPollinationSpatialConfig, sim: SpatialBatPollinationSim):
        self.cfg = cfg
        self.sim = sim

    def generate_svg(self) -> str:
        cfg = self.cfg
        sim = self.sim
        w, h = cfg.width, cfg.height
        F = cfg.frames
        dur = F / cfg.fps

        svg = [
            f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="background-color:#11151c; font-family:system-ui, -apple-system, sans-serif;">'
        ]

        svg.append('<defs>')
        svg.append(
            '<radialGradient id="nightBg">'
            '<stop offset="0%" stop-color="#1b263b" stop-opacity="0.9"/>'
            '<stop offset="80%" stop-color="#0d1b2a" stop-opacity="0.8"/>'
            '<stop offset="100%" stop-color="#11151c" stop-opacity="0.0"/>'
            '</radialGradient>'
        )
        svg.append('</defs>')

        # Background Night Sky
        svg.append(f'<rect width="{w}" height="{h}" fill="#11151c"/>')
        
        # Center the radial gradient on the forest
        bg_cx, bg_cy = (w-380)/2, h/2
        svg.append(f'<circle cx="{bg_cx}" cy="{bg_cy}" r="{h}" fill="url(#nightBg)"/>')

        # Title
        svg.append(
            f'<text x="20" y="30" font-size="15" fill="#e0e1dd" font-weight="bold">'
            f'ECO-SIM: Pequi × Nectarivorous Bat    - Chiropterophily Spatial Map</text>'
        )
        svg.append(
            f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#778da9">'
            f'Temporal Shift: 12 Months of Nocturnal Flowering & Migratory Bat Influx</text>'
        )

        # --- Pequi Trees ---
        for ti, t in enumerate(sim.trees):
            px, py = t["pos"]
            
            # Trunk
            svg.append(f'<path d="M{px},{py+10} Q{px-5},{py+5} {px},{py} Q{px+5},{py-5} {px},{py-10}" fill="none" stroke="#5d4037" stroke-width="4" stroke-linecap="round"/>')
            
            # Canopy animation
            l_vals = ";".join(
                f"{max(0.1, sim._interp(PEQUI_LEAF_CURVE, (fi/F)*12)) * 18:.1f}" for fi in range(F)
            )
            c_vals = ";".join(
                "#8d6e63" if sim._interp(PEQUI_LEAF_CURVE, (fi/F)*12) < 0.3 else "#81b29a" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py-10:.0f}" fill="#81b29a" opacity="0.8">'
                f'<animate attributeName="r" values="{l_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'<animate attributeName="fill" values="{c_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

            # Nocturnal Flowers
            b_vals = ";".join(
                f"{sim._interp(PEQUI_BLOOM_CURVE, (fi/F)*12) * 8:.1f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px-5:.0f}" cy="{py-12:.0f}" fill="#fff3b0" opacity="0.9">'
                f'<animate attributeName="r" values="{b_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            svg.append(
                f'<circle cx="{px+5:.0f}" cy="{py-8:.0f}" fill="#ffffff" opacity="0.9">'
                f'<animate attributeName="r" values="{b_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

            # Fruits
            fr_vals = ";".join(
                f"{sim._interp(PEQUI_FRUIT_CURVE, (fi/F)*12) * 6:.1f}" for fi in range(F)
            )
            svg.append(
                f'<circle cx="{px:.0f}" cy="{py-5:.0f}" fill="#c5e1a5" stroke="#33691e" stroke-width="0.5">'
                f'<animate attributeName="r" values="{fr_vals}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )

        # --- Nectar Bats (Glossophaga) ---
        for i in range(cfg.num_bats):
            hist = sim.hist_bats_xy[i]
            cxs = ";".join(str(round(h[0], 1)) for h in hist)
            cys = ";".join(str(round(h[1], 1)) for h in hist)
            ops = ";".join(str(round(h[2], 2)) for h in hist)
            
            # Flight paths (dark purple trails) - sparingly for performance
            trail_pts = [f"{h[0]:.0f},{h[1]:.0f}" for i, h in enumerate(hist) if i % 4 == 0 and h[2] > 0.5]
            if len(trail_pts) > 2:
                svg.append(
                    f'<polyline points="{" ".join(trail_pts)}" fill="none" '
                    f'stroke="#9c27b0" stroke-width="0.6" opacity="0.25"/>'
                )
            
            # Bat Body
            svg.append(
                f'<circle r="3" fill="#4a148c">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</circle>'
            )
            # Bat Wings (rapid flapping)
            w_ry = ";".join("5" if fi % 2 == 0 else "1" for fi in range(F))
            svg.append(
                f'<ellipse rx="6" ry="4" fill="#7b1fa2" opacity="0.8">'
                f'<animate attributeName="cx" values="{cxs}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cys}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="ry" values="{w_ry}" dur="{dur}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="{ops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                f'</ellipse>'
            )

        # Right Panel UI
        card_w = 350
        rx = w - card_w - 20
        svg.append(f'<g transform="translate({rx}, 80)">')
        svg.append('<rect width="350" height="200" fill="#0d1b2a" rx="8" ry="8" stroke="#9c27b0" stroke-width="1.5" opacity="0.95"/>')
        svg.append('<text x="15" y="30" font-size="15" fill="#e0e1dd" font-weight="bold">Current Month Simulator:</text>')
        
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12
            vals[m_idx] = "1"
            op_str  = ";".join(vals + ["0"])
            
            # Highlight month text
            svg.append(f'<text x="15" y="60" font-size="15" fill="#fff3b0" font-weight="bold">')
            svg.append(f'{m_name}')
            svg.append(f'<animate attributeName="opacity" values="{op_str}" calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')

            # Show context based on biological curves
            bloom = sim._interp(PEQUI_BLOOM_CURVE, m_idx)
            bats = sim._interp(BAT_MIGRATION_CURVE, m_idx)
            fruit = sim._interp(PEQUI_FRUIT_CURVE, m_idx)
            
            if bloom > 0.1:
                txt1 = "Night flowering opens a pollination window"
                txt2 = "Bat influx overlaps bloom" if bats > 0.4 else "Bloom starts before full bat arrival"
                color1 = "#fff3b0"
                color2 = "#9c27b0"
            elif fruit > 0.1:
                txt1 = "Flowering ended; fruits now hold the carbon gain"
                txt2 = "Most pollinating bats have shifted away"
                color1 = "#ffb74d"
                color2 = "#778da9"
            else:
                txt1 = "Dormancy or leaf recovery limits nocturnal rewards"
                txt2 = "Low bat presence means low pollination service"
                color1 = "#81b29a"
                color2 = "#778da9"

            svg.append(f'<text x="15" y="85" font-size="15" fill="{color1}" font-weight="bold">')
            svg.append(f'{txt1}')
            svg.append(f'<animate attributeName="opacity" values="{op_str}" calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')
            svg.append(f'<text x="15" y="105" font-size="15" fill="{color2}" font-weight="bold">')
            svg.append(f'{txt2}')
            svg.append(f'<animate attributeName="opacity" values="{op_str}" calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')
            
        svg.append('<text x="15" y="145" fill="#415a77" font-size="15" font-weight="bold">Map Legend:</text>')
        svg.append('<circle cx="25" cy="165" r="5" fill="#7b1fa2"/><text font-weight="bold" x="45" y="169" fill="#e0e1dd" font-size="15">Nectar Bat (Glossophaga)</text>')
        svg.append('<circle cx="25" cy="185" r="5" fill="#fff3b0"/><text font-weight="bold" x="45" y="189" fill="#fff3b0" font-size="15">Pequi Nocturnal Bloom</text>')
        svg.append('</g>')

        svg.append(f'<g transform="translate({rx}, 300)">')
        svg.append('<rect width="350" height="114" fill="#0d1b2a" rx="8" ry="8" stroke="#ffb74d" stroke-width="1" opacity="0.9"/>')
        svg.append('<text x="15" y="25" font-size="15" fill="#ffb74d" font-weight="bold">Pollination Metrics</text>')
        svg.append(f'<text font-weight="bold" x="15" y="45" font-size="15" fill="#fff3b0">Pequi Flowers Pollinated: {sim.total_flowers_pollinated:,.0f} visits</text>')
        svg.append(f'<text font-weight="bold" x="15" y="65" font-size="15" fill="#9c27b0">Total Nectar Extracted: {sim.total_nectar_extracted:,.1f} ml</text>')
        svg.append('<text font-weight="bold" x="15" y="85" font-size="15" fill="#81b29a">Service needs bloom-migration spatial overlap.</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


def main():
    print(f" (Spatial Map Pattern) — Pequi ↔ Morcego-nectar on {CONFIG.device}...")

    sim = SpatialBatPollinationSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_flowers_pollinated} flowers pollinated, {sim.total_nectar_extracted:.1f} ml extracted.")

    print("Generating SVG...")
    renderer = SpatialBatPollinationRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_68')
    return svg_content


if __name__ == "__main__":
    main()
