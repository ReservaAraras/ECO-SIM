# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 67: Orquídea (Cyrtopodium) ↔ Fungo Micorrízico - Symbiotic Survival Spatial Map
# INTERVENTION 1/4: Seasonal & Migratory Interventions Series
# ===================================================================================================
"""
notebook_67.py - Terrestrial Orchid × Mycorrhizal Fungi:
Notebook Differentiation:
- Differentiation Focus: Orquídea (Cyrtopodium) ↔ Fungo Micorrízico - Symbiotic Survival Spatial Map emphasizing nutrient cycling loops.
- Indicator species: Tatu-peba (Euphractus sexcinctus).
- Pollination lens: butterfly host-plant coupling.
- Human impact lens: pesticide drift from farms.

                 Drought Survival & Resource Sharing (Spatial Pattern)

Models the profound symbiosis between terrestrial Cerrado orchids (Cyrtopodium sp.) 
and subterranean Mycorrhizal fungal networks. Following the pattern of notebook.py,
this is an animated spatial map.

During the harsh 6-month dry season, the orchid sheds leaves to prevent water loss. 
It survives entirely on hydraulic lift and nutrient subsidies provided by the vast 
underground fungal mycelium.

Scientific Relevance:
  - Demonstrates plant-fungus mutualism crucial for Cerrado resilience.
  - Highlights seasonal resource flows (Carbon down vs Water up) across a landscape.
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # pyre-ignore[21]
import numpy as np  # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from eco_base import save_svg, sanitize_svg_text  , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML  # pyre-ignore[21]
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x


@dataclass
class SymbioticSpatialConfig:
    width: int = 1280
    height: int = CANVAS_HEIGHT
    frames: int = 360
    fps: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_orchids: int = 40
    num_fungi_nodes: int = 80

CONFIG = SymbioticSpatialConfig()


class SpatialSymbioticSim:

    def __init__(self, cfg: SymbioticSpatialConfig):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)

        # Orchids: [x, y, stored_water, stored_carbon]
        self.orchids = torch.rand((cfg.num_orchids, 4), device=self.dev)
        self.orchids[:, 0] = self.orchids[:, 0] * (cfg.width - 400) + 50 # keep away from UI
        self.orchids[:, 1] = self.orchids[:, 1] * (cfg.height - 100) + 50
        self.orchids[:, 2] = 50.0
        self.orchids[:, 3] = 50.0

        # Fungal Nodes (Underground network hubs)
        self.fungi = torch.rand((cfg.num_fungi_nodes, 2), device=self.dev)
        self.fungi[:, 0] = self.fungi[:, 0] * (cfg.width - 400) + 50
        self.fungi[:, 1] = self.fungi[:, 1] * (cfg.height - 100) + 50

        # Connections (Orchid index to Fungi index)
        self.connections = []
        for i in range(cfg.num_orchids):
            # Connect each orchid to 2 closest fungi nodes
            dists = torch.norm(self.fungi - self.orchids[i, :2], dim=1)
            closest = torch.topk(dists, 2, largest=False).indices
            for c in closest:
                self.connections.append((i, int(c.item())))

        # Extra fungi connections to form a web
        self.fungi_edges = []
        for i in range(cfg.num_fungi_nodes):
            dists = torch.norm(self.fungi - self.fungi[i], dim=1)
            closest = torch.topk(dists, 3, largest=False).indices
            for c in closest:
                if c.item() != i:
                    self.fungi_edges.append((i, int(c.item())))

        self.hist_month = []
        # list of packets over frames
        self.hist_packets = []
        
        self.total_carbon = 0.0
        self.total_water = 0.0

    def step(self, frame_idx: int):
        month = (frame_idx // 30) % 12
        
        # Wet season (Nov-Mar, approx month 10,11,0,1,2,3) -> Orchids send Carbon down
        # Dry season (May-Sep, approx month 4,5,6,7,8,9) -> Fungi send Water up
        is_wet = month in [10, 11, 0, 1, 2, 3]
        
        new_packets = []
        spawn_rate = 0.2
        
        # Track packets frame by frame. Since we want an animated view, 
        # we can just probabilistically generate trace dots that move along connections.
        packet_visuals = []

        if is_wet:
            # Carbon down
            for (o_idx, f_idx) in self.connections:
                if random.random() < spawn_rate:
                    ox, oy = self.orchids[o_idx, 0].item(), self.orchids[o_idx, 1].item()
                    fx, fy = self.fungi[f_idx, 0].item(), self.fungi[f_idx, 1].item()
                    packet_visuals.append([ox, oy, fx, fy, 0]) # 0 = Carbon
                    self.total_carbon += 1.0
        else:
            # Water up
            for (o_idx, f_idx) in self.connections:
                if random.random() < spawn_rate * 1.5:
                    ox, oy = self.orchids[o_idx, 0].item(), self.orchids[o_idx, 1].item()
                    fx, fy = self.fungi[f_idx, 0].item(), self.fungi[f_idx, 1].item()
                    packet_visuals.append([fx, fy, ox, oy, 1]) # 1 = Water
                    self.total_water += 1.0
                    
        self.hist_month.append(month)
        self.hist_packets.append(packet_visuals)


class SpatialSymbioticRenderer:

    def __init__(self, cfg: SymbioticSpatialConfig, sim: SpatialSymbioticSim):
        self.cfg = cfg
        self.sim = sim

    def generate_svg(self) -> str:
        cfg = self.cfg
        w, h = cfg.width, cfg.height
        F = cfg.frames
        dur = F / cfg.fps
        
        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="background-color:#1c110a; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append('<defs>')
        svg.append('<pattern id="dotGrid" width="40" height="40" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="2.5" fill="#3e2723" opacity="0.3"/></pattern>')
        svg.append('</defs>')

        svg.append(f'<rect width="100%" height="100%" fill="#1c110a"/>')
        svg.append(f'<rect width="100%" height="100%" fill="url(#dotGrid)"/>')

        svg.append(f'<text x="20" y="30" font-size="15" fill="#dcedc8" font-weight="bold">ECO-SIM: Orquídea × Fungo Micorrízico - Symbiotic Survival Spatial Map</text>')
        svg.append(f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#aed581">Temporal Shift: 12 Months of Subterranean Resource Sharing</text>')

        # Draw Fungal Network (Hyphae) Base
        for (i, j) in self.sim.fungi_edges:
            fx1, fy1 = self.sim.fungi[i, 0].item(), self.sim.fungi[i, 1].item()
            fx2, fy2 = self.sim.fungi[j, 0].item(), self.sim.fungi[j, 1].item()
            svg.append(f'<line x1="{fx1:.1f}" y1="{fy1:.1f}" x2="{fx2:.1f}" y2="{fy2:.1f}" stroke="#4a148c" stroke-width="0.8" opacity="0.3"/>')

        for (o_idx, f_idx) in self.sim.connections:
            ox, oy = self.sim.orchids[o_idx, 0].item(), self.sim.orchids[o_idx, 1].item()
            fx, fy = self.sim.fungi[f_idx, 0].item(), self.sim.fungi[f_idx, 1].item()
            svg.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{fx:.1f}" y2="{fy:.1f}" stroke="#5d4037" stroke-width="1.5" opacity="0.5"/>')

        # Draw Fungal Nodes
        for i in range(cfg.num_fungi_nodes):
            fx, fy = self.sim.fungi[i, 0].item(), self.sim.fungi[i, 1].item()
            svg.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="4" fill="#bb86fc" opacity="0.6"/>')

        # Orchids (Animated based on month)
        # Wet month -> Bloom/Leaves visible, Dry month -> Dormant
        for i in range(cfg.num_orchids):
            ox, oy = self.sim.orchids[i, 0].item(), self.sim.orchids[i, 1].item()
            
            # Pseudobulb (always visible)
            svg.append(f'<ellipse cx="{ox:.1f}" cy="{oy:.1f}" rx="7" ry="10" fill="#a1887f" stroke="#3e2723" stroke-width="1.5"/>')
            
            # Leaves (Expand in wet season)
            l_scales = ";".join("1.2" if m in [10, 11, 0, 1, 2, 3] else "0" for m in self.sim.hist_month)
            
            svg.append(f'<g transform="translate({ox:.1f},{oy:.1f})">')
            svg.append(f'<path d="M0,0 Q-15,-15 0,-30 Q5,-15 0,0" fill="#81c784" opacity="0.9">')
            svg.append(f'<animateTransform attributeName="transform" type="scale" values="{l_scales}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append('</path>')
            svg.append(f'<path d="M0,0 Q15,-15 0,-30 Q-5,-15 0,0" fill="#aed581" opacity="0.9">')
            svg.append(f'<animateTransform attributeName="transform" type="scale" values="{l_scales}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append('</path>')
            svg.append('</g>')

        # Animated Resource Packets 
        # (Since adding all traces is expensive, we'll pick representative connections)
        max_traces = 50
        trace_count = 0
        for (o_idx, f_idx) in self.sim.connections:
            if trace_count >= max_traces: break
            trace_count += 1
            ox, oy = self.sim.orchids[o_idx, 0].item(), self.sim.orchids[o_idx, 1].item()
            fx, fy = self.sim.fungi[f_idx, 0].item(), self.sim.fungi[f_idx, 1].item()
            
            # Carbon Tracer (Downwards - Green to Purple)
            c_op = ";".join("0.8" if m in [10,11,0,1,2,3] and random.random()>0.3 else "0" for m in self.sim.hist_month)
            move_dur = random.uniform(0.5, 1.5)
            svg.append(f'<circle r="3.5" fill="#81c784">')
            svg.append(f'<animate attributeName="cx" values="{ox:.1f};{fx:.1f}" dur="{move_dur:.1f}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{oy:.1f};{fy:.1f}" dur="{move_dur:.1f}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="opacity" values="{c_op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append('</circle>')

            # Water Tracer (Upwards - Blue to Brown)
            w_op = ";".join("0.8" if m in [4,5,6,7,8,9] and random.random()>0.3 else "0" for m in self.sim.hist_month)
            move_dur2 = random.uniform(0.5, 1.5)
            svg.append(f'<circle r="3.5" fill="#4fc3f7">')
            svg.append(f'<animate attributeName="cx" values="{fx:.1f};{ox:.1f}" dur="{move_dur2:.1f}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="cy" values="{fy:.1f};{oy:.1f}" dur="{move_dur2:.1f}s" repeatCount="indefinite"/>')
            svg.append(f'<animate attributeName="opacity" values="{w_op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>')
            svg.append('</circle>')

        # Right Panel UI
        card_w = 340
        rx = w - card_w - 20
        svg.append(f'<g transform="translate({rx}, 80)">')
        svg.append('<rect width="340" height="200" fill="#2d1d16" rx="8" ry="8" stroke="#bb86fc" stroke-width="1.5" opacity="0.95"/>')
        svg.append('<text x="15" y="30" font-size="15" fill="#4fc3f7" font-weight="bold">Current Month Simulator:</text>')
        
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        
        for m_idx, m_name in enumerate(month_names):
            vals = ["0"] * 12
            vals[m_idx] = "1"
            op_str  = ";".join(vals + ["0"])
            
            # Highlight month text
            svg.append(f'<text x="15" y="60" font-size="15" fill="#dcedc8" font-weight="bold">')
            svg.append(f'{m_name}')
            svg.append(f'<animate attributeName="opacity" values="{op_str}" calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')
            
            # Show season context and logic
            is_wet = m_idx in [10, 11, 0, 1, 2, 3]
            txt1 = "WET SEASON: Orchid Photosynthesizes" if is_wet else "DRY SEASON: Orchid is Dormant"
            txt2 = "→ Giving Carbon (Sugars) to Fungi" if is_wet else "→ Receiving Water (Hydraulic Lift)"
            color = "#81c784" if is_wet else "#4fc3f7"
            
            svg.append(f'<text x="15" y="85" font-size="15" fill="{color}" font-weight="bold">')
            svg.append(f'{txt1}')
            svg.append(f'<animate attributeName="opacity" values="{op_str}" calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')
            svg.append(f'<text x="15" y="105" font-size="15" fill="{color}" font-weight="bold">')
            svg.append(f'{txt2}')
            svg.append(f'<animate attributeName="opacity" values="{op_str}" calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')
            svg.append('</text>')
            
        svg.append('<text x="15" y="145" fill="#a1887f" font-size="15" font-weight="bold">Symbiosis Legend:</text>')
        svg.append('<circle cx="25" cy="165" r="5" fill="#a1887f"/><text font-weight="bold" x="45" y="169" fill="#dcedc8" font-size="15">Orchid Pseudobulb/Leaves</text>')
        svg.append('<circle cx="25" cy="185" r="5" fill="#bb86fc"/><text font-weight="bold" x="45" y="189" fill="#bb86fc" font-size="15">Mycorrhizal Fungal Hub</text>')
        svg.append('</g>')

        svg.append(f'<g transform="translate({rx}, 300)">')
        svg.append('<rect width="340" height="90" fill="#1c110a" rx="8" ry="8" stroke="#81c784" stroke-width="1" opacity="0.9"/>')
        svg.append('<text x="15" y="25" font-size="15" fill="#81c784" font-weight="bold">Metrics Tracker</text>')
        svg.append(f'<text font-weight="bold" x="15" y="45" font-size="15" fill="#dcedc8">Total Carbon Exchanged: {self.sim.total_carbon:.0f} units</text>')
        svg.append(f'<text font-weight="bold" x="15" y="65" font-size="15" fill="#4fc3f7">Total Water Uplifted: {self.sim.total_water:.0f} units</text>')
        svg.append('</g>')


        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)


def main():
    print(f" (Spatial Map Pattern) - Orquídea × Fungo Micorrízico on {CONFIG.device}...")

    sim = SpatialSymbioticSim(CONFIG)
    for frame in range(CONFIG.frames):
        sim.step(frame)

    print(f"Done: {sim.total_carbon:.0f} units carbon, {sim.total_water:.0f} units water exchanged.")

    print("Generating SVG...")
    renderer = SpatialSymbioticRenderer(CONFIG, sim)
    svg_content = renderer.generate_svg()
    display(HTML(svg_content))

    save_svg(svg_content, 'notebook_67')
    return svg_content


if __name__ == "__main__":
    main()
