# -*- coding: utf-8 -*-
# ===================================================================================================
# PROJECT: Recanto das Araras Extractive Reserve Eco-Simulator
# MODULE 77: Mandacaru (Cereus jamacaru) ↔ Morcego-polinizador (Glossophaga soricina)
#            Nocturnal Pollination · Nectarivorous Migration · Bloom Pulse
# INTERVENTION 3/4: Cerrado Trophic Cascade & Seasonal Connectivity Series
# ===================================================================================================
"""
notebook_77.py — Mandacaru Cactus ↔ Pallas's Long-tongued Bat (Plant ↔ Animal):
Notebook Differentiation:
- Differentiation Focus: Mandacaru (Cereus jamacaru) ↔ Morcego-polinizador (Glossophaga soricina) emphasizing termite mound networks.
- Indicator species: Gato-maracaja (Leopardus wiedii).
- Pollination lens: pollen limitation in fragmented edges.
- Human impact lens: climate warming on water balance.

              Nocturnal Pollination Network, Nectar Corridor Migration

The Mandacaru (Cereus jamacaru) is a columnar cactus typical of the Caatinga,
but it occurs in rocky outcrops (afloramentos rochosos) within the Cerrado
transition zone of the RESEX Recanto das Araras.

Glossophaga soricina is a small nectarivorous bat. While some populations
are resident, others undertake regional migrations tracking the sequential
blooming of chiropterophilous (bat-pollinated) plants across biomes.

SEASONAL DYNAMICS at  (Goiás):

  BLOOM PULSE (Sep–Nov):
    Mandacaru flowers open exclusively at night. Each flower lasts only a single
    night, opening around 19:00 and wilting by dawn. The peak blooming period
    is synchronized at the end of the dry season, just before the first rains.

  MIGRATORY NECTAR CORRIDOR:
    The bats act as a mobile link in the ecosystem. As the dry season progresses,
    they move through the landscape following nectar availability. Their arrival
    at the RESEX's rocky outcrops coincides precisely with the Mandacaru
    bloom pulse.

  POLLINATION NETWORK:
    By flying up to 10-15 km in a single night, the bats cross-pollinate
    isolated Mandacaru plants. This high-mobility pollen dispersal is
    essential for the genetic diversity of plants separated by dense Cerrado.

Scientific references:
  • Fleming et al. (1993): The evolution of bat pollination.
  • Rojas-Martínez et al. (1999): Seasonal distribution of nectar-feeding bats.
  • Silva et al. (2010): Reproductive biology of Cereus jamacaru.
  • PIGT  field observations, Goiás (2022–2024).
"""

import os
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch # pyre-ignore[21]
import numpy as np # pyre-ignore[21]
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
from eco_base import save_svg, sanitize_svg_text, draw_phenology_chart, draw_migration_map , CANVAS_HEIGHT, ZONES # pyre-ignore[21]
try:
    from IPython.display import display, HTML
except ImportError:
    def display(*a, **kw): pass
    def HTML(x): return x

# ===================================================================================================
# 1. SCIENTIFIC PARAMETERS & PHENOLOGICAL CURVES  (month 0 = January)
# ===================================================================================================

# Mandacaru blooming intensity (peaks late dry season / early wet: Oct-Nov)
MANDACARU_BLOOM_CURVE = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.10, 0.15, 0.30, 0.60, 0.90, 0.70]

# Bat regional presence / migration into RESEX (tracks bloom)
BAT_MIGRATION_CURVE = [0.10, 0.10, 0.15, 0.30, 0.60, 0.85, 0.90, 0.85, 0.70, 0.40, 0.15, 0.10]

# Nightly foraging activity index (longest flights during peak bloom)
BAT_FORAGING_CURVE = [
    0.40, 0.35, 0.30, 0.35, 0.30, 0.25, 0.40, 0.60, 0.75, 1.00, 0.90, 0.60
]

# Pollination success rate (cross-pollination probability)
POLLINATION_CURVE = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.10, 0.15, 0.30, 0.60, 0.90, 0.70]

# Nectar abundance/energy available
NECTAR_ENERGY_CURVE = [0.40, 0.30, 0.20, 0.10, 0.05, 0.05, 0.05, 0.10, 0.30, 0.60, 0.90, 0.65]

@dataclass
class BatCfg:
    width:  int   = 1280
    height: int = CANVAS_HEIGHT
    frames: int   = 360
    fps:    int   = 10
    device: str   = 'cuda' if torch.cuda.is_available() else 'cpu'

    clock_cx:     float = 420.0
    clock_cy:     float = 310.0
    clock_radius: float = 240.0

    num_bats:      int   = 15   # migrating bat swarm
    num_cacti:     int   = 20   # isolated cacti
    bat_speed:     float = 8.0
    visit_radius:  float = 15.0 # proximity to pollinate

CONFIG = BatCfg()

# ===================================================================================================
# 2. SIMULATION MODEL
# ===================================================================================================

class BatPollinationSim:
    def __init__(self, cfg: BatCfg):
        self.cfg = cfg
        self.dev = torch.device(cfg.device)
        cx, cy, R = cfg.clock_cx, cfg.clock_cy, cfg.clock_radius

        # Cacti
        self.cacti: List[Dict] = []
        for k in range(cfg.num_cacti):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(R * 0.2, R * 0.85)
            self.cacti.append({
                "pos": [cx + math.cos(angle) * r, cy + math.sin(angle) * r],
                "flowers": 0.0,
                "pollinated": 0,
            })

        # Bats
        self.bats: List[Dict] = []
        for k in range(cfg.num_bats):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(R+20, R+80) # start outside
            self.bats.append({
                "pos": torch.tensor([cx + math.cos(angle) * r, cy + math.sin(angle) * r],
                                    device=self.dev, dtype=torch.float32),
                "target_cactus": -1,
                "pollen_load": 0,
                "energy": 50.0,
            })

        self.hist_month: List[float] = []
        self.hist_bat_xy: List[List[Tuple[float,float,float]]] = [[] for _ in range(cfg.num_bats)]
        self.hist_pollination_events: List[Tuple[float,float,int]] = []

        self.total_visits = 0
        self.total_pollinations = 0

    def _interp(self, curve: list, month_frac: float) -> float:
        m  = month_frac % 12
        lo = int(m) % 12
        hi = (lo + 1) % 12
        t  = m - int(m)
        return curve[lo] * (1 - t) + curve[hi] * t

    def step(self, frame: int):
        cfg = self.cfg
        cx, cy, R = cfg.clock_cx, cfg.clock_cy, cfg.clock_radius
        month_frac = (frame / cfg.frames) * 12.0
        self.hist_month.append(month_frac)

        bloom      = self._interp(MANDACARU_BLOOM_CURVE, month_frac)
        migration  = self._interp(BAT_MIGRATION_CURVE, month_frac)
        activity   = self._interp(BAT_FORAGING_CURVE, month_frac)
        poll_prob  = self._interp(POLLINATION_CURVE, month_frac)

        # Update cacti flowers
        active_cacti = []
        for ci, c in enumerate(self.cacti):
            c["flowers"] = bloom * (0.5 + random.random()*0.5)
            if c["flowers"] > 0.2:
                active_cacti.append(ci)

        active_bat_count = int(cfg.num_bats * migration)

        # Update bats
        for bi, bat in enumerate(self.bats):
            pos = bat["pos"]
            is_active = (bi < active_bat_count)
            opacity = 1.0 if is_active else 0.0

            if not is_active:
                # Drift outside bounds
                dx_ = pos[0].item() - cx
                dy_ = pos[1].item() - cy
                dr_ = math.sqrt(dx_*dx_ + dy_*dy_)
                if dr_ < R + 30:
                    pos[0] += (dx_/dr_)*2.0
                    pos[1] += (dy_/dr_)*2.0
                self.hist_bat_xy[bi].append((pos[0].item(), pos[1].item(), opacity))
                continue

            speed = cfg.bat_speed * activity

            # Select target
            if bat["target_cactus"] == -1 and active_cacti:
                if random.random() < 0.4:
                    bat["target_cactus"] = random.choice(active_cacti)

            if bat["target_cactus"] != -1:
                tc = bat["target_cactus"]
                tgt_pos = self.cacti[tc]["pos"]
                tgt = torch.tensor(tgt_pos, device=self.dev, dtype=torch.float32)
                vec = tgt - pos
                dist = torch.norm(vec).item()

                if dist > cfg.visit_radius:
                    jitter = torch.randn(2, device=self.dev) * speed * 0.3
                    pos += (vec / max(dist, 1e-5)) * speed + jitter
                else:
                    self.total_visits += 1
                    if bat["pollen_load"] > 0 and random.random() < poll_prob:
                        self.cacti[tc]["pollinated"] += 1
                        self.total_pollinations += 1
                        self.hist_pollination_events.append((tgt_pos[0], tgt_pos[1], frame))
                    bat["pollen_load"] = 1 # pick up pollen
                    bat["target_cactus"] = -1
            else:
                # Random foraging flight inside clock
                jitter = torch.randn(2, device=self.dev) * speed
                pos += jitter
                dx_ = pos[0].item() - cx
                dy_ = pos[1].item() - cy
                dr_ = math.sqrt(dx_*dx_ + dy_*dy_)
                if dr_ > R - 10:
                    pos[0] -= (dx_/dr_)*speed
                    pos[1] -= (dy_/dr_)*speed

            self.hist_bat_xy[bi].append((pos[0].item(), pos[1].item(), opacity))


# ===================================================================================================
# 3. VISUALIZATION
# ===================================================================================================

class BatRenderer:
    def __init__(self, cfg: BatCfg, sim: BatPollinationSim):
        self.cfg = cfg
        self.sim = sim

    def _arc_path(self, cx, cy, radius, start_m, end_m):
        a1 = (start_m / 12) * 2 * math.pi - math.pi / 2
        a2 = (end_m   / 12) * 2 * math.pi - math.pi / 2
        x1 = cx + math.cos(a1) * radius; y1 = cy + math.sin(a1) * radius
        x2 = cx + math.cos(a2) * radius; y2 = cy + math.sin(a2) * radius
        span  = (end_m - start_m) % 12
        large = 1 if span > 6 else 0
        return f"M {x1:.0f} {y1:.0f} A {radius} {radius} 0 {large} 1 {x2:.0f} {y2:.0f}"

    def generate_svg(self) -> str:
        cfg = self.cfg
        sim = self.sim
        w, h = cfg.width, cfg.height
        cx, cy = cfg.clock_cx, cfg.clock_cy
        R = cfg.clock_radius
        F = cfg.frames
        dur = F / cfg.fps

        svg = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
               f'style="background-color:#05070a; font-family:system-ui, -apple-system, sans-serif;">']

        svg.append('<defs>')
        svg.append('<radialGradient id="nightBg">'
                   '<stop offset="0%" stop-color="#141a29" stop-opacity="0.95"/>'
                   '<stop offset="70%" stop-color="#090d14" stop-opacity="0.85"/>'
                   '<stop offset="100%" stop-color="#05070a" stop-opacity="0.0"/>'
                   '</radialGradient>')
        svg.append('</defs>')

        # Background
        svg.append(f'<rect width="{w}" height="{h}" fill="#05070a"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R+55}" fill="url(#nightBg)"/>')

        bloom_fills = ";".join(
            f"rgba(255,255,255,{sim._interp(MANDACARU_BLOOM_CURVE, (f/F)*12)*0.1:.2f})"
            for f in range(F)
        )
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R-8}" fill="transparent">'
                   f'<animate attributeName="fill" values="{bloom_fills}" '
                   f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Title
        svg.append(f'<text x="20" y="30" font-size="15" fill="#e0e0e0" font-weight="bold">'
                   f'ECO-SIM: Mandacaru × Glossophaga Pollination</text>')
        svg.append(f'<text font-weight="bold" x="20" y="52" font-size="15" fill="#90caf9">'
                   f'Bloom Pulse · Corridor Migration · Pollination</text>')

        # Clock Face
        months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        month_cols = {0:"#37474f", 1:"#37474f", 2:"#37474f", 3:"#37474f", 4:"#37474f", 5:"#37474f",
                      6:"#78909c", 7:"#90caf9", 8:"#bbdefb", 9:"#ffffff", 10:"#e3f2fd", 11:"#90caf9"}
        for i, m in enumerate(months):
            angle = (i/12)*2*math.pi - math.pi/2
            tx = cx + math.cos(angle)*(R+35); ty = cy + math.sin(angle)*(R+35)
            svg.append(f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="15" fill="{month_cols[i]}" '
                       f'text-anchor="middle" alignment-baseline="middle" font-weight="bold">{m}</text>')
            svg.append(f'<line x1="{cx+math.cos(angle)*R:.0f}" y1="{cy+math.sin(angle)*R:.0f}" '
                       f'x2="{cx+math.cos(angle)*(R-6):.0f}" y2="{cy+math.sin(angle)*(R-6):.0f}" '
                       f'stroke="#141a29" stroke-width="2"/>')

        # Arcs
        d1 = self._arc_path(cx, cy, R+11, 8.5, 11)
        svg.append(f'<path d="{d1}" fill="none" stroke="#ffffff" stroke-width="8" opacity="0.6"/>')
        mid1 = ((8.5+11)/2/12)*2*math.pi - math.pi/2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid1)*(R+24):.0f}" y="{cy+math.sin(mid1)*(R+24):.0f}" '
                   f'font-size="15" fill="#ffffff" text-anchor="middle">Peak Bloom</text>')

        d2 = self._arc_path(cx, cy, R+22, 6, 11)
        svg.append(f'<path d="{d2}" fill="none" stroke="#90caf9" stroke-width="4" stroke-dasharray="4,4" opacity="0.6"/>')
        mid2 = ((6+11)/2/12)*2*math.pi - math.pi/2
        svg.append(f'<text font-weight="bold" x="{cx+math.cos(mid2)*(R+34):.0f}" y="{cy+math.sin(mid2)*(R+34):.0f}" '
                   f'font-size="15" fill="#90caf9" text-anchor="middle">Bat Arrival</text>')

        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="#263238" stroke-width="1.5"/>')

        # Cacti
        for ci, c in enumerate(sim.cacti):
            px, py = c["pos"]
            # Base plant
            svg.append(f'<rect x="{px-2:.0f}" y="{py-8:.0f}" width="4" height="16" fill="#2e7d32" opacity="0.8"/>')
            
            c_ops_str = ";".join(f"{sim._interp(MANDACARU_BLOOM_CURVE, (fi/F)*12)*0.9+0.1:.2f}" for fi in range(F))
            svg.append(f'<circle cx="{px:.0f}" cy="{py-8:.0f}" r="4" fill="#ffffff" opacity="0">'
                       f'<animate attributeName="opacity" values="{c_ops_str}" '
                       f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Pollination Events
        MAX_EVTS = min(60, len(sim.hist_pollination_events))
        for px, py, pf in sim.hist_pollination_events[:MAX_EVTS]:
            p_ops = ["0.0"]*F
            for fi in range(pf, min(pf+30, F)):
                p_ops[fi] = f"{(1.0 - (fi-pf)/30)*0.8:.2f}"
            svg.append(f'<circle cx="{px:.0f}" cy="{py:.0f}" r="6" fill="#ffd54f" opacity="0">'
                       f'<animate attributeName="opacity" values="{";".join(p_ops)}" '
                       f'dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></circle>')

        # Bats
        for bi in range(cfg.num_bats):
            hist = sim.hist_bat_xy[bi]
            if not hist: continue
            bxs = ";".join(f"{h[0]:.1f}" for h in hist)
            bys = ";".join(f"{h[1]:.1f}" for h in hist)
            bops = ";".join(f"{h[2]:.2f}" for h in hist)

            svg.append(f'<polygon points="-3,-2 3,0 -3,2 0,0" fill="#90a4ae" opacity="0">'
                       f'<animate attributeName="opacity" values="{bops}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/>'
                       f'<animateTransform attributeName="transform" type="translate" values="{";".join(f"{h[0]:.1f} {h[1]:.1f}" for h in hist)}" '
                       f'dur="{dur}s" repeatCount="indefinite"/></polygon>')

        # Hand
        hx = ";".join(f"{cx+math.cos((m/12)*2*math.pi-math.pi/2)*(R-10):.1f}" for m in sim.hist_month)
        hy = ";".join(f"{cy+math.sin((m/12)*2*math.pi-math.pi/2)*(R-10):.1f}" for m in sim.hist_month)
        svg.append(f'<line x1="{cx}" y1="{cy}" stroke="#ffffff" stroke-width="2.5" opacity="0.8">'
                   f'<animate attributeName="x2" values="{hx}" dur="{dur}s" repeatCount="indefinite"/>'
                   f'<animate attributeName="y2" values="{hy}" dur="{dur}s" repeatCount="indefinite"/></line>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#05070a" stroke="#ffffff" stroke-width="2"/>')

        # PANELS
        px = w - 395; pw = 375
        
        # P1
        svg.append(f'<g transform="translate({px}, 20)"><rect width="{pw}" height="196" fill="#080c14" rx="8" stroke="#bbdefb" stroke-width="1" opacity="0.95"/>')
        svg.append(f'<text x="12" y="22" fill="#bbdefb" font-size="15" font-weight="bold">Nocturnal Pollination Network</text>')
        svg.append(f'<text font-weight="bold" x="12" y="42" fill="#90a4ae" font-size="15">Cereus jamacaru blooms at end of dry season.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="56" fill="#90a4ae" font-size="15">Flowers open at night across all outcroppings.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="70" fill="#90caf9" font-size="15">G. soricina tracks blooms, moving into the</text>')
        svg.append(f'<text font-weight="bold" x="12" y="84" fill="#90caf9" font-size="15">RESEX through migratory nectar corridors.</text>')
        svg.append(f'<text font-weight="bold" x="12" y="100" fill="#ffffff" font-size="15">Bat mobility sustains gene flow among cacti.</text>')
        svg.append('</g>')
        svg.append(f'<g transform="translate({px}, 226)"><rect width="{pw}" height="94" fill="#080c14" rx="8" stroke="#37474f"/>')
        svg.append(f'<text x="12" y="22" fill="#90a4ae" font-size="15" font-weight="bold">Foraging Metrics</text>')
        svg.append(f'<text font-weight="bold" x="12" y="44" fill="#90caf9" font-size="15">Total Flower Visits: {sim.total_visits}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="62" fill="#ffd54f" font-size="15">Cross-Pollinations: {sim.total_pollinations}</text>')
        svg.append(f'<text font-weight="bold" x="12" y="80" fill="#546e7a" font-size="15">Bats: {cfg.num_bats} | Cacti: {cfg.num_cacti}</text></g>')

        # P3 Chart
        cvs = [(MANDACARU_BLOOM_CURVE, "#ffffff", "Cactus Bloom"),
               (BAT_MIGRATION_CURVE, "#90caf9", "Bat Migration Presence"),
               (POLLINATION_CURVE, "#ffd54f", "Pollination Success")]
        chart = draw_phenology_chart(cvs, 330, 58, 125, "Pollination Phenology", "#bbdefb", "#080c14", "#bbdefb")
        svg.append(f'<g transform="translate({px}, 326)">{chart}</g>')

        # Status
        svg.append(f'<g transform="translate(545, 362)"><rect width="260" height="230" fill="#080c14" rx="8" stroke="#bbdefb" opacity="0.95"/>')
        svg.append(f'<text x="12" y="22" fill="#bbdefb" font-weight="bold">Active Season Status:</text>')
        mnames = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        for mi, m in enumerate(mnames):
            vs = ["0"]*12; vs[mi] = "1"
            op = ";".join(vs+["0"])
            svg.append(f'<text x="12" y="52" font-size="15" fill="#ffffff" font-weight="bold">{m}'
                       f'<animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></text>')
            bl = sim._interp(MANDACARU_BLOOM_CURVE, mi)
            if bl > 0.5:
                st1, c1 = "MANDACARU BLOOM PEAK", "#ffffff"
                st2, c2 = "Synchronised nocturnal flowers", "#e0e0e0"
            else:
                st1, c1 = "BAT FORAGING", "#90caf9"
                st2, c2 = "Bats dispersed / low residency", "#78909c"
            
            for yo, txt, co in [(80, st1, c1), (100, st2, c2)]:
                svg.append(f'<text x="12" y="{yo}" font-size="15" fill="{co}" font-weight="bold">{txt}'
                           f'<animate attributeName="opacity" values="{op}" dur="{dur}s" repeatCount="indefinite" calcMode="discrete"/></text>')
        
        svg.append(f'<text x="12" y="140" fill="#546e7a" font-size="15" font-weight="bold">Legend:</text>')
        leg = [(155,"#90a4ae","Bat (Glossophaga)"),(170,"#2e7d32","Cactus (Mandacaru)"),
               (185,"#ffffff","Flower (open)"),(200,"#ffd54f","Pollination event")]
        for ey, ec, el in leg:
            svg.append(f'<circle cx="22" cy="{ey}" r="5" fill="{ec}"/><text font-weight="bold" x="36" y="{ey+4}" fill="{ec}" font-size="15">{el}</text>')
        svg.append('</g>')

        svg.append('<script type="application/ecmascript">\\n  <![CDATA[\\n    document.addEventListener(\'visibilitychange\', function() {\\n      if (!document.hidden) {\\n        const s = document.querySelector(\'svg\');\\n        if (s) { s.pauseAnimations(); s.setCurrentTime(0); s.unpauseAnimations(); }\\n      }\\n    });\\n  ]]>\\n  </script>')
        svg.append('</svg>')
        return "".join(svg)

# ===================================================================================================
# 4. MAIN
# ===================================================================================================

def main():
    sim = BatPollinationSim(CONFIG)
    for frame in range(CONFIG.frames): sim.step(frame)
    print(f"Done: {sim.total_visits} visits, {sim.total_pollinations} pollinations.")
    svg_content = BatRenderer(CONFIG, sim).generate_svg()
    save_svg(svg_content, 'notebook_77')

if __name__ == "__main__": main()
