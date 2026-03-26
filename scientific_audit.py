#!/usr/bin/env python3
# pyre-ignore-all-errors
"""
scientific_audit.py
===================
Automated numerical, statistical, and ecological-consistency auditor for the
ECO-SIM Cerrado notebook suite (notebooks 37–80).

Checks performed
----------------
1. BIODIVERSITY_DB (eco_base.py)
   • Weight rank-order plausibility vs real body masses (Anta/Tapir historically
     misweighted below Maned Wolf and other species)
   • Lifespan ordering vs real longevity records (Macaw, King Vulture, etc.)
   • Numeric field bounds: drag ∈ (0, 1], cohesion ∈ [0,1], max_turn ∈ [0,1]

2. BaseConfig timing parameters (eco_base.py)
   • fire_start_frame: frame 80 / 560 ≈ February — Cerrado fires peak Aug–Sep
   • migration_start_frame: frame 40 / 560 ≈ late January — should be Sep–Oct or Mar–Apr
   • Population ordering: initial_particles ≤ max_particles; carrying_capacity ≤ max_particles
   • Energy system: mating_energy_threshold vs initial energy (80.0) and energy_gain_fruiting
   • All probability-type fields ∈ [0, 1]

3. Per-notebook Config fields
   • frames > 0, fps > 0
   • Event-frame references (fire_start_frame, weed_start_frame, …) < frames
   • Probability fields ∈ [0, 1]
   • Population ordering

4. Phenological curves (module-level *_CURVE lists of 12 floats)
   • Exactly 12 entries (one per month, January = index 0)
   • All values ∈ [0.0, 1.0]
   • Seasonal alignment: wet-season curves should peak Oct–Mar;
     dry-season curves should peak Apr–Sep
   • Abrupt jumps: |Δ| > 0.5 between consecutive months flagged
   • Pearson correlation for ecologically linked curve pairs

5. Cross-notebook consistency
   • Common curves (e.g., RAINFALL_CURVE) must be identical across all notebooks

Usage
-----
    python scientific_audit.py              # scan current directory, print report
    python scientific_audit.py --dir .      # explicit directory
    python scientific_audit.py --json       # also write audit.json
    python scientific_audit.py --md         # also write Scientific Validation Report_03.md
"""

import ast
import glob
import json
import math
import os
import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data model
# ─────────────────────────────────────────────────────────────────────────────

_SEV_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


@dataclass
class Finding:
    severity: str   # ERROR | WARN | INFO
    source:   str   # filename (or "filename::ClassName")
    check:    str   # short category label
    message:  str   # one-line summary
    detail:   str = ""  # optional citation / remediation hint

    def __lt__(self, other: "Finding") -> bool:
        return (_SEV_ORDER[self.severity], self.source) < \
               (_SEV_ORDER[other.severity], other.source)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AST helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_literal(node: ast.expr) -> Any:
    """Return Python value for a literal AST node, or None on failure."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_module_toplevel_assignments(tree: ast.Module) -> Dict[str, Any]:
    """Return {name: value} for all module-level assignments with literal values."""
    result: Dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            val = _safe_literal(node.value)
            if val is not None:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        result[t.id] = val
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            val = _safe_literal(node.value)
            if val is not None and isinstance(node.target, ast.Name):
                result[node.target.id] = val
    return result


def _extract_dataclass_fields(
    tree: ast.Module, name_contains: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Returns {ClassName: {field_name: default_value}} for @dataclass classes.
    Filters by class-name substring when name_contains is given.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if name_contains and name_contains not in node.name:
            continue
        is_dc = any(
            (isinstance(d, ast.Name) and d.id == "dataclass") or
            (isinstance(d, ast.Call) and getattr(d.func, "id", "") == "dataclass")
            for d in node.decorator_list
        )
        if not is_dc:
            continue
        fields: Dict[str, Any] = {}
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and item.value is not None:
                val = _safe_literal(item.value)
                if val is not None and isinstance(item.target, ast.Name):
                    fields[item.target.id] = val
            elif isinstance(item, ast.Assign):
                val = _safe_literal(item.value)
                if val is not None:
                    for t in item.targets:
                        if isinstance(t, ast.Name):
                            fields[t.id] = val
        result[node.name] = fields
    return result


def _parse_file(path: str) -> Optional[ast.Module]:
    try:
        with open(path, encoding="utf-8") as fh:
            return ast.parse(fh.read())
    except Exception:
        return None


def _extract_curve_lists(tree: ast.Module) -> Dict[str, List[float]]:
    """Extract module-level name = [float, …] assignments of any length."""
    curves: Dict[str, List[float]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        val = _safe_literal(node.value)
        if isinstance(val, list) and val and all(
            isinstance(v, (int, float)) for v in val
        ):
            curves[node.targets[0].id] = [float(v) for v in val]
    return curves


def _extract_biodiversity_db(tree: ast.Module) -> Optional[Dict[str, Dict[str, Any]]]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "BIODIVERSITY_DB":
                    val = _safe_literal(node.value)
                    if isinstance(val, dict):
                        return val  # type: ignore[return-value]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ecological ground truth
# ─────────────────────────────────────────────────────────────────────────────

# Cerrado season indices (month 0 = January)
_WET = [9, 10, 11, 0, 1, 2]   # Oct Nov Dec Jan Feb Mar  (85% of annual rain)
_DRY = [3,  4,  5, 6, 7, 8]   # Apr May Jun Jul Aug Sep

_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# Keyword → expected peak season for phenological curves
# Key = lowercase substring of the curve variable name
_CURVE_SEASON: Dict[str, str] = {
    # Dry-season fruiters must appear BEFORE the generic "fruit" / "fruiting"
    # catch-all below, since matching is first-hit on insertion order.
    "jatoba":           "dry",   # Hymenaea pods ripen Jul–Sep (dry season)
    "macauba_drop":     "dry",   # Macaúba fruit drop peaks at end of dry season
    "rain":             "wet",
    "fruiting":         "wet",
    "fruit":            "wet",
    "moisture":         "wet",
    "flood":            "wet",
    "lobeira":          "wet",
    "pequi":            "wet",
    "nuptial":          "wet",   # ant/termite nuptial flights: first Oct rains
    "swarm_fruit":      "wet",   # Termitomyces fruiting pulse: Oct-Nov
    "grass_cover":      "wet",   # dense grass in wet season
    "breeding":         "wet",   # Seriema/Macaw breeding onset at wet season
    "wolf_lobeira_diet": "wet",  # Maned wolf fruit intake tracks wet season
    "foraging_surface": "wet",   # Ant/bee surface foraging: wet season peak
    "productivity":     "wet",   # Fungus productivity tied to leaf input (wet)
    "fire":             "dry",   # Cerrado fires: Aug-Sep
    "drought":          "dry",
    "exposure":         "dry",   # Reptile exposure: dry season
    "hunt":             "dry",   # Seriema hunting: dry season
    "predation":        "dry",
    "colony_depth":     "dry",   # Ant colony retreats deeper in dry season
    "depth":            "dry",
    "metabolism":       "dry",   # fungal metabolism peaks dry season inside mound
}

# Real-world body-mass ordering: (heavier_species_key, lighter_species_key, citation)
_WEIGHT_RANKS: List[Tuple[str, str, str]] = [
    ("Gado-Nelore",    "Anta",
     "Nelore cattle ≈500 kg vs Lowland Tapir ≈200 kg. [ABCZ 2021 / Medici 2012]"),
    ("Anta",           "Lobo-guara",
     "Tapir ≈200 kg vs Maned Wolf ≈23 kg. [Medici 2012 / Paula 2008]"),
    ("Anta",           "Lobo-guara-Chrysocyon",
     "Tapir ≈200 kg vs Maned Wolf ≈23 kg. [Medici 2012 / Paula 2008]"),
    ("Anta",           "Tamandua-bandeira",
     "Tapir ≈200 kg vs Giant Anteater ≈35 kg. [IUCN 2014]"),
    ("Gado-Nelore",    "Tamandua-bandeira",
     "Cattle ≈500 kg vs Giant Anteater ≈35 kg. [IUCN 2014]"),
    ("Anta",           "Seriema",
     "Tapir ≈200 kg vs Red-legged Seriema ≈1.5 kg. [Sick 1997]"),
    ("Anta",           "Macaco-prego-Sapajus",
     "Tapir ≈200 kg vs Capuchin monkey ≈3-4 kg. [de Waal 2001]"),
    ("Tamandua-bandeira", "Serpente-Bothrops",
     "Giant Anteater ≈35 kg vs Bothrops pit-viper ≈1-3 kg. [Campbell 1992]"),
]

# Lifespan ordering: (longer_lived_key, shorter_lived_key, citation)
_LIFESPAN_RANKS: List[Tuple[str, str, str]] = [
    ("Arara-caninde",  "Gado-Nelore",
     "Hyacinth Macaw 50–60 yrs vs Nelore cattle 15–20 yrs. [Sick 1997 / ABCZ 2021]"),
    ("Urubu-rei",      "Gado-Nelore",
     "King Vulture 30–40 yrs vs Nelore cattle 15–20 yrs. [BirdLife 2023]"),
    ("Arara-caninde",  "Lobo-guara",
     "Hyacinth Macaw 50–60 yrs vs Maned Wolf 12–15 yrs. [Sick 1997 / Paula 2008]"),
    ("Fungo-micorrizico", "Cupim",
     "Mycorrhizal fungal networks can persist decades vs termite colony lifespan of years."),
]

# Curve pairs expected to be positively or negatively correlated
# (curve_A, curve_B, expected_sign (+1 or -1), description)
_CORR_PAIRS: List[Tuple[str, str, int, str]] = [
    ("WOLF_LOBEIRA_DIET_CURVE", "LOBEIRA_FRUIT_CURVE", +1,
     "Wolf's Lobeira fruit intake should track Lobeira availability (wet season)"),
    ("SERIEMA_HUNT_CURVE",      "REPTILE_EXPOSURE_CURVE", +1,
     "Seriema hunting success should track reptile exposure (dry season)"),
    ("SWARM_FRUIT_PULSE",       "RAINFALL_CURVE", +1,
     "Termite swarming / fungal fruiting pulse is triggered by first rains (Oct–Nov)"),
    ("NUPTIAL_FLIGHT",          "RAINFALL_CURVE", +1,
     "Ant nuptial flights triggered by first rains (Sep–Oct)"),
    ("GRASS_COVER_CURVE",       "RAINFALL_CURVE", +1,
     "Grass cover density tracks rainfall (wet = dense)"),
    ("COLONY_DEPTH",            "GRASS_COVER_CURVE", -1,
     "Colony retreats deeper when surface is dry (less grass cover)"),
    ("FUNGAL_METABOLISM_CURVE", "COLONY_DEPTH", +1,
     "Fungal metabolism peaks in dry season when sealed mound depth is greatest"),
    ("P_FORAGING_CURVE",        "RAINFALL_CURVE", -1,
     "Termite surface foraging peaks in dry season (inverse of rainfall)"),
    ("SOIL_MOISTURE_CURVE",     "RAINFALL_CURVE", +1,
     "Soil moisture should track rainfall closely"),
    ("BREEDING_CURVE",          "RAINFALL_CURVE", +1,
     "Seriema / bird breeding onset coincides with wet-season arrival"),
]

# Correct ecological months (0-indexed from Jan) for BaseConfig timing fields
_FIRE_CORRECT_MONTHS    = {6, 7, 8}    # Jul Aug Sep (late dry season)
_MIGRATION_OK_MONTHS    = {8, 9, 2, 3} # Sep Oct (arrival) or Mar Apr (departure)
_DEFAULT_FRAMES = 560
_INITIAL_ENERGY = 80.0   # hard-coded in EcosystemBase.__init__


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Check functions
# ─────────────────────────────────────────────────────────────────────────────

def _season_mean(curve: List[float], idx: List[int]) -> float:
    return sum(curve[i] for i in idx) / len(idx)


def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a)/n, sum(b)/n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    den = math.sqrt(
        sum((a[i]-ma)**2 for i in range(n)) *
        sum((b[i]-mb)**2 for i in range(n))
    )
    return num / den if den > 1e-9 else 0.0


def check_biodiversity_db(
    db: Dict[str, Dict[str, Any]], source: str
) -> List[Finding]:
    findings: List[Finding] = []

    # 4a. Numeric field bounds
    _bounds: Dict[str, Tuple[float, float, bool]] = {
        # field: (lo, hi, lo_exclusive)
        "drag":     (0.0, 1.0, True),
        "cohesion": (0.0, 1.0, False),
        "max_turn": (0.0, 1.0, False),
        "weight":   (0.0, 1.0, False),
    }
    for sp, attrs in db.items():
        for fname, (lo, hi, lo_excl) in _bounds.items():
            val = attrs.get(fname)
            if not isinstance(val, (int, float)):
                continue
            fails = (val <= lo) if lo_excl else (val < lo)
            if fails or val > hi:
                findings.append(Finding(
                    "ERROR", source, "DB bounds",
                    f"{sp}.{fname} = {val} is outside {'(' if lo_excl else '['}{lo}, {hi}]",
                ))
        speed = attrs.get("speed")
        if isinstance(speed, (int, float)) and speed < 0:
            findings.append(Finding(
                "ERROR", source, "DB bounds",
                f"{sp}.speed = {speed} cannot be negative",
            ))

    # 4b. Weight rank ordering
    for (heavier, lighter, citation) in _WEIGHT_RANKS:
        if heavier not in db or lighter not in db:
            continue
        wh = db[heavier].get("weight", 0.0)
        wl = db[lighter].get("weight", 0.0)
        if not isinstance(wh, (int, float)) or not isinstance(wl, (int, float)):
            continue
        if wh <= wl:
            findings.append(Finding(
                "ERROR", source, "DB weight rank",
                f"{heavier}.weight ({wh}) ≤ {lighter}.weight ({wl}) — "
                f"inverted vs real biology",
                citation,
            ))

    # 4c. Lifespan rank ordering
    for (longer_sp, shorter_sp, citation) in _LIFESPAN_RANKS:
        if longer_sp not in db or shorter_sp not in db:
            continue
        ll = db[longer_sp].get("lifespan_base", 0)
        ls = db[shorter_sp].get("lifespan_base", 0)
        if not isinstance(ll, (int, float)) or not isinstance(ls, (int, float)):
            continue
        if ll <= ls:
            findings.append(Finding(
                "WARN", source, "DB lifespan rank",
                f"{longer_sp}.lifespan_base ({ll}) ≤ {shorter_sp}.lifespan_base ({ls}) — "
                f"real {longer_sp} lives longer than {shorter_sp}",
                citation,
            ))

    # 4d. Sessile species (speed=0) should not fly
    for sp, attrs in db.items():
        speed = attrs.get("speed", -1)
        flies = attrs.get("flies")
        if isinstance(speed, (int, float)) and speed == 0.0 and flies is True:
            findings.append(Finding(
                "WARN", source, "DB logical consistency",
                f"{sp}: speed=0.0 but flies=True — sessile species cannot fly",
            ))

    return findings


def check_base_config(cfg: Dict[str, Any], source: str) -> List[Finding]:
    findings: List[Finding] = []
    frames = cfg.get("frames", _DEFAULT_FRAMES)
    if not isinstance(frames, (int, float)):
        frames = _DEFAULT_FRAMES
    fpm = float(frames) / 12.0   # frames per month

    # 4e. Fire start frame → ecological month
    fsf = cfg.get("fire_start_frame")
    if isinstance(fsf, (int, float)):
        fire_month_idx = int(fsf / fpm) % 12
        if fire_month_idx not in _FIRE_CORRECT_MONTHS:
            findings.append(Finding(
                "ERROR", source, "Fire seasonality",
                f"fire_start_frame={fsf} maps to {_MONTH_NAMES[fire_month_idx]} "
                f"({fsf/fpm:.1f} months) — Cerrado fires peak Aug–Sep (late dry season)",
                f"Pivello (2011): natural/anthropogenic fires in Cerrado peak Aug-Sep. "
                f"Suggested frames: {6*fpm:.0f}–{9*fpm:.0f} (Jul–Sep).",
            ))

    # 4f. Migration start frame → ecological month
    msf = cfg.get("migration_start_frame")
    if isinstance(msf, (int, float)):
        mig_month_idx = int(msf / fpm) % 12
        if mig_month_idx not in _MIGRATION_OK_MONTHS:
            findings.append(Finding(
                "WARN", source, "Migration seasonality",
                f"migration_start_frame={msf} maps to {_MONTH_NAMES[mig_month_idx]} "
                f"({msf/fpm:.1f} months) — migrants arrive Sep–Oct or depart Mar–Apr",
                "Alves et al. (2004): Cerrado migratory birds arrive at wet season onset "
                "(Sep-Oct) and depart at wet season end (Mar-Apr).",
            ))

    # 4g. Population ordering
    init_p  = cfg.get("initial_particles")
    max_p   = cfg.get("max_particles")
    carry_p = cfg.get("carrying_capacity")
    if isinstance(init_p, int) and isinstance(max_p, int) and init_p > max_p:
        findings.append(Finding(
            "ERROR", source, "Population bounds",
            f"initial_particles ({init_p}) > max_particles ({max_p}) — "
            f"initialisation will exceed allocated array size",
        ))
    if isinstance(carry_p, int) and isinstance(max_p, int) and carry_p > max_p:
        findings.append(Finding(
            "ERROR", source, "Population bounds",
            f"carrying_capacity ({carry_p}) > max_particles ({max_p}) — "
            f"carrying capacity is unreachable within allocated array",
        ))

    # 4h. Energy system consistency
    mat_thresh = cfg.get("mating_energy_threshold")
    e_gain_fr  = cfg.get("energy_gain_fruiting", 6.0)
    e_decay    = cfg.get("energy_decay", 0.12)
    if isinstance(mat_thresh, (int, float)):
        if mat_thresh > _INITIAL_ENERGY + (e_gain_fr if isinstance(e_gain_fr, (int,float)) else 6.0):
            findings.append(Finding(
                "WARN", source, "Energy system",
                f"mating_energy_threshold ({mat_thresh}) > initial_energy "
                f"({_INITIAL_ENERGY}) + energy_gain_fruiting ({e_gain_fr}): "
                f"particles cannot mate even after one feeding event",
                f"Increase energy_gain_fruiting or decrease mating_energy_threshold.",
            ))
        elif mat_thresh > _INITIAL_ENERGY:
            findings.append(Finding(
                "INFO", source, "Energy system",
                f"mating_energy_threshold ({mat_thresh}) > initial_energy "
                f"({_INITIAL_ENERGY}): animals must feed before first mating — "
                f"ecologically plausible but reduces early-generation reproduction",
            ))

    # 4i. Probability fields in [0, 1]
    _prob_fields = [
        "fire_spread_prob", "migration_spawn_rate", "road_crossing_courage",
        "poach_mortality_prob", "seed_pickup_chance",
    ]
    for pf in _prob_fields:
        val = cfg.get(pf)
        if isinstance(val, (int, float)) and not (0.0 <= val <= 1.0):
            findings.append(Finding(
                "ERROR", source, "Config probability",
                f"{pf} = {val} is outside valid probability range [0.0, 1.0]",
            ))

    return findings


def check_notebook_config(cfg: Dict[str, Any], source: str) -> List[Finding]:
    """Generic per-notebook Config dataclass checks."""
    findings: List[Finding] = []
    frames = cfg.get("frames")
    fps    = cfg.get("fps")

    # 4j. Basic positive values
    if isinstance(frames, (int, float)) and frames <= 0:
        findings.append(Finding(
            "ERROR", source, "Config basic",
            f"frames = {frames} must be > 0",
        ))
    if isinstance(fps, (int, float)) and fps <= 0:
        findings.append(Finding(
            "ERROR", source, "Config basic",
            f"fps = {fps} must be > 0",
        ))

    # 4k. Event frames must fire before simulation ends
    _frame_ref_fields = [
        "fire_start_frame", "migration_start_frame", "logging_start_frame",
        "poach_start_frame", "keystone_removal_frame", "weed_start_frame",
        "grass_start_frame",
    ]
    if isinstance(frames, (int, float)) and frames > 0:
        for frf in _frame_ref_fields:
            val = cfg.get(frf)
            if isinstance(val, (int, float)) and val >= frames:
                findings.append(Finding(
                    "WARN", source, "Config timing",
                    f"{frf} = {val} ≥ frames = {frames} — event will never trigger",
                ))

    # 4l. Probability fields
    _prob_fields = [
        "grass_spread_prob", "fire_spread_prob", "poach_mortality_prob",
        "migration_spawn_rate", "road_crossing_courage", "seed_pickup_chance",
    ]
    for pf in _prob_fields:
        val = cfg.get(pf)
        if isinstance(val, (int, float)) and not (0.0 <= val <= 1.0):
            findings.append(Finding(
                "ERROR", source, "Config probability",
                f"{pf} = {val} is outside valid probability range [0.0, 1.0]",
            ))

    # 4m. Carrying capacity ≤ max_particles
    cc = cfg.get("carrying_capacity")
    mp = cfg.get("max_particles")
    ip = cfg.get("initial_particles")
    if isinstance(cc, int) and isinstance(mp, int) and cc > mp:
        findings.append(Finding(
            "ERROR", source, "Population bounds",
            f"carrying_capacity ({cc}) > max_particles ({mp}) — unreachable",
        ))
    if isinstance(ip, int) and isinstance(mp, int) and ip > mp:
        findings.append(Finding(
            "ERROR", source, "Population bounds",
            f"initial_particles ({ip}) > max_particles ({mp}) — array overflow",
        ))

    # 4n. Clock/display geometry checks
    cx, cy = cfg.get("clock_cx"), cfg.get("clock_cy")
    cr     = cfg.get("clock_radius")
    w, h   = cfg.get("width", 1280), cfg.get("height", 602)
    if isinstance(cx, (int,float)) and isinstance(cr, (int,float)):
        if cx - cr < 0 or cx + cr > float(w):
            findings.append(Finding(
                "INFO", source, "Geometry",
                f"Clock circle (cx={cx}, r={cr}) extends outside canvas width {w}",
            ))
    if isinstance(cy, (int,float)) and isinstance(cr, (int,float)):
        if cy - cr < 0 or cy + cr > float(h):
            findings.append(Finding(
                "INFO", source, "Geometry",
                f"Clock circle (cy={cy}, r={cr}) extends outside canvas height {h}",
            ))

    return findings


def check_curves(curves: Dict[str, List[float]], source: str) -> List[Finding]:
    findings: List[Finding] = []

    for name, curve in curves.items():
        # 4o. Length
        if len(curve) != 12:
            findings.append(Finding(
                "ERROR", source, "Curve length",
                f"{name}: {len(curve)} entries — expected 12 (one per month, Jan=0)",
            ))
            continue   # remaining checks require 12 entries

        # 4p. All values in [0, 1]
        bad = [(i, v) for i, v in enumerate(curve) if not 0.0 <= v <= 1.0]
        for i, v in bad:
            findings.append(Finding(
                "ERROR", source, "Curve bounds",
                f"{name}[{i}] ({_MONTH_NAMES[i]}) = {v:.4f} is outside [0.0, 1.0]",
            ))

        # 4q. Seasonal alignment (keyword heuristics)
        low = name.lower()
        expected: Optional[str] = None
        for keyword, season in _CURVE_SEASON.items():
            if keyword in low:
                expected = season
                break

        if expected:
            wet_m = _season_mean(curve, _WET)
            dry_m = _season_mean(curve, _DRY)
            margin = 0.10
            if expected == "wet" and dry_m > wet_m + margin:
                findings.append(Finding(
                    "WARN", source, "Curve seasonality",
                    f"{name}: expected peak WET (Oct–Mar avg={wet_m:.2f}) "
                    f"but DRY (Apr–Sep) avg={dry_m:.2f} is higher by {dry_m-wet_m:.2f}",
                    "Verify that peak months match documented Cerrado phenology.",
                ))
            elif expected == "dry" and wet_m > dry_m + margin:
                findings.append(Finding(
                    "WARN", source, "Curve seasonality",
                    f"{name}: expected peak DRY (Apr–Sep avg={dry_m:.2f}) "
                    f"but WET (Oct–Mar) avg={wet_m:.2f} is higher by {wet_m-dry_m:.2f}",
                    "Verify that peak months match documented Cerrado phenology.",
                ))

        # 4r. Abrupt month-to-month jumps
        for i in range(12):
            delta = abs(curve[(i+1) % 12] - curve[i])
            if delta > 0.50:
                findings.append(Finding(
                    "INFO", source, "Curve smoothness",
                    f"{name}: |Δ| = {delta:.2f} between "
                    f"{_MONTH_NAMES[i]} → {_MONTH_NAMES[(i+1)%12]} — abrupt transition",
                ))

    return findings


def check_curve_correlations(
    curves: Dict[str, List[float]], source: str
) -> List[Finding]:
    findings: List[Finding] = []
    for (a_name, b_name, expected_sign, desc) in _CORR_PAIRS:
        if a_name not in curves or b_name not in curves:
            continue
        a, b = curves[a_name], curves[b_name]
        if len(a) != 12 or len(b) != 12:
            continue
        r = _pearson(a, b)
        if expected_sign == +1 and r < 0.50:
            findings.append(Finding(
                "WARN", source, "Curve correlation",
                f"{a_name} ↔ {b_name}: Pearson r = {r:.2f} (expected ≥ 0.50, positive)",
                desc,
            ))
        elif expected_sign == -1 and r > -0.30:
            findings.append(Finding(
                "WARN", source, "Curve correlation",
                f"{a_name} ↔ {b_name}: Pearson r = {r:.2f} (expected ≤ -0.30, negative)",
                desc,
            ))
    return findings


def check_cross_notebook(
    all_curves: Dict[str, Dict[str, List[float]]]
) -> List[Finding]:
    """Flag same-named curves that differ across notebooks."""
    findings: List[Finding] = []
    by_name: Dict[str, List[Tuple[str, List[float]]]] = defaultdict(list)
    for fname, curves in all_curves.items():
        for cname, cvals in curves.items():
            if len(cvals) == 12:
                by_name[cname].append((fname, cvals))

    for cname, entries in by_name.items():
        if len(entries) < 2:
            continue
        ref_fname, ref_curve = entries[0]
        for other_fname, other_curve in entries[1:]:
            diffs = [abs(a - b) for a, b in zip(ref_curve, other_curve)]
            max_diff = max(diffs)
            if max_diff > 1e-6:
                findings.append(Finding(
                    "WARN",
                    f"{ref_fname} vs {other_fname}",
                    "Cross-notebook consistency",
                    f"{cname} disagrees across notebooks (max monthly Δ = {max_diff:.4f})",
                    "Shared ecological backbone curves should be identical across the series.",
                ))
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(base_dir: str = ".") -> List[Finding]:
    findings: List[Finding] = []
    all_curves: Dict[str, Dict[str, List[float]]] = {}

    # ── eco_base.py ────────────────────────────────────────────────────────
    base_path = os.path.join(base_dir, "eco_base.py")
    base_tree = _parse_file(base_path)
    if base_tree is None:
        findings.append(Finding(
            "ERROR", "eco_base.py", "Parse",
            "Could not parse eco_base.py — all dependent checks skipped",
        ))
    else:
        db = _extract_biodiversity_db(base_tree)
        if db:
            findings += check_biodiversity_db(db, "eco_base.py")
        dc_map = _extract_dataclass_fields(base_tree, "BaseConfig")
        for cls_name, fields in dc_map.items():
            findings += check_base_config(fields, f"eco_base.py :: {cls_name}")

    # ── Notebooks ──────────────────────────────────────────────────────────
    nb_paths = sorted(glob.glob(os.path.join(base_dir, "notebook_*.py")))
    for nb_path in nb_paths:
        fname = os.path.basename(nb_path)
        tree = _parse_file(nb_path)
        if tree is None:
            findings.append(Finding(
                "WARN", fname, "Parse",
                f"Could not parse {fname}",
            ))
            continue

        # Phenological curves
        curves = _extract_curve_lists(tree)
        pheno = {k: v for k, v in curves.items() if len(v) == 12}
        if pheno:
            all_curves[fname] = pheno
            findings += check_curves(pheno, fname)
            findings += check_curve_correlations(pheno, fname)

        # Config dataclasses
        dc_map = _extract_dataclass_fields(tree)
        for cls_name, fields in dc_map.items():
            findings += check_notebook_config(fields, f"{fname} :: {cls_name}")

    # ── Cross-notebook ─────────────────────────────────────────────────────
    findings += check_cross_notebook(all_curves)

    findings.sort()
    return findings


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _icon(sev: str) -> str:
    return {"ERROR": "[ERROR]", "WARN": "[WARN]", "INFO": "[INFO]"}.get(sev, "[ ?? ]")


def console_report(findings: List[Finding]) -> None:
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    _color = {"ERROR": "\033[91m", "WARN": "\033[93m", "INFO": "\033[96m"}
    _reset = "\033[0m"
    _bold  = "\033[1m"

    counts: Dict[str, int] = {"ERROR": 0, "WARN": 0, "INFO": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  ECO-SIM Scientific Audit — {sum(counts.values())} findings  "
          f"[{counts['ERROR']} errors · {counts['WARN']} warnings · {counts['INFO']} info]")
    print(sep)

    prev_src = None
    for f in findings:
        if f.source != prev_src:
            prev_src = f.source
            bold = _bold if use_color else ""
            reset = _reset if use_color else ""
            print(f"\n  ─── {bold}{f.source}{reset}")
        color = _color.get(f.severity, "") if use_color else ""
        reset = _reset if use_color else ""
        print(f"  {color}{_icon(f.severity)}{reset} [{f.check}]  {f.message}")
        if f.detail:
            print(f"           → {f.detail}")

    print(f"\n{sep}")
    print(f"  {counts['ERROR']} ERROR   {counts['WARN']} WARN   {counts['INFO']} INFO")
    print(f"{sep}\n")


def markdown_report(findings: List[Finding]) -> str:
    lines: List[str] = []
    counts: Dict[str, int] = {"ERROR": 0, "WARN": 0, "INFO": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    _md_icon = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🔵"}

    lines += [
        "# Scientific Validation Report_03: Automated Audit",
        "",
        "> **Tool:** `scientific_audit.py`  ",
        "> **Scope:** eco_base.py + all notebook_\\*.py  ",
        "> **Checks:** BIODIVERSITY_DB integrity · BaseConfig timing · "
        "phenological curves · cross-notebook consistency",
        "",
        "## Executive Summary",
        "",
        f"| Severity | Count |",
        f"| --- | --- |",
        f"| 🔴 ERROR | {counts['ERROR']} |",
        f"| 🟡 WARN  | {counts['WARN']}  |",
        f"| 🔵 INFO  | {counts['INFO']}  |",
        f"| **Total** | **{sum(counts.values())}** |",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| # | Sev | Source | Check | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for i, f in enumerate(findings, 1):
        icon = _md_icon.get(f.severity, "")
        msg  = f.message.replace("|", "\\|")
        src  = f"`{f.source}`"
        lines.append(f"| {i} | {icon} {f.severity} | {src} | {f.check} | {msg} |")

    lines += ["", "---", "", "## Detailed Findings", ""]

    prev_sev: Optional[str] = None
    for i, f in enumerate(findings, 1):
        if f.severity != prev_sev:
            prev_sev = f.severity
            label = {"ERROR": "Errors", "WARN": "Warnings", "INFO": "Info"}.get(f.severity, f.severity)
            lines += [f"### {_md_icon.get(f.severity,'')} {label}", ""]

        lines.append(f"**{i}. [{f.check}]** `{f.source}`")
        lines.append(f"")
        lines.append(f"{f.message}")
        if f.detail:
            lines.append(f"")
            lines.append(f"> {f.detail}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Methodology",
        "",
        "### Seasonal Framework",
        "The Cerrado exhibits a pronounced bimodal climate:",
        "- **Wet season**: October–March (indices 9–11, 0–2) — 85% of annual precipitation.",
        "- **Dry season**: April–September (indices 3–8) — fire season, dormancy, "
        "reduced food availability.",
        "",
        "### Frame-to-Month Conversion",
        "For a 560-frame simulation spanning one full year:",
        "- 1 month ≈ 46.7 frames",
        "- fire_start_frame = 80 → February (month 1.7) — **ecologically wrong**;",
        "  should be ~frame 326–420 (Jul–Sep).",
        "",
        "### Weight-Rank Validation",
        "Species weights in BIODIVERSITY_DB are validated against IUCN field data:",
        "- Tapir (*Tapirus terrestris*): 150–300 kg",
        "- Maned Wolf (*Chrysocyon brachyurus*): 20–30 kg",
        "- Nelore Cattle: 450–600 kg",
        "",
        "### Curve Correlation",
        "Pearson r computed over 12 monthly values; thresholds: r ≥ 0.50 (positive), "
        "r ≤ −0.30 (negative).",
        "",
        "---",
        "",
        "## References",
        "",
        "1. Medici, E.P. et al. (2012). *Tapirus terrestris*. IUCN Red List.",
        "2. Paula, R.C. et al. (2008). *Chrysocyon brachyurus*. IUCN Red List.",
        "3. Pivello, V.R. (2011). The use of fire in the Cerrado. *Fire Ecology*.",
        "4. Alves, M.A.S. et al. (2004). Migratory birds of the Brazilian Cerrado. "
        "*Bird Conservation International*.",
        "5. Sick, H. (1997). *Ornitologia Brasileira*. Editora Nova Fronteira.",
        "6. ABCZ (2021). Padrão Racial da Raça Nelore.",
        "7. López-Bao & González-Varo (2011). Frugivory by the maned wolf. "
        "*J. Biogeography*.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ECO-SIM Scientific Audit — numerical, statistical & ecological checks"
    )
    parser.add_argument(
        "--dir", default=".",
        help="Directory containing eco_base.py and notebook_*.py (default: current dir)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Write detailed findings to audit.json",
    )
    parser.add_argument(
        "--md", action="store_true",
        help="Write markdown report to Scientific Validation Report_03.md",
    )
    args = parser.parse_args()

    findings = run_audit(args.dir)
    console_report(findings)

    if args.json:
        out = [
            {"severity": f.severity, "source": f.source, "check": f.check,
             "message": f.message, "detail": f.detail}
            for f in findings
        ]
        jpath = os.path.join(args.dir, "audit.json")
        with open(jpath, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"JSON written  →  {jpath}")

    if args.md:
        md   = markdown_report(findings)
        mpath = os.path.join(args.dir, "Scientific Validation Report_03.md")
        with open(mpath, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"Markdown written  →  {mpath}")


if __name__ == "__main__":
    main()
