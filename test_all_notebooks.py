# -*- coding: utf-8 -*-
"""
test_all_notebooks.py - Validation Engine for ECO-SIM Notebook Suite
=====================================================================

Runs every notebook_*.py, validates execution and SVG output, detects
common corruption patterns (e.g. HTML entity encoding of Python operators),
and reports categorised results.

Usage:
    python test_all_notebooks.py                   # run all (parallel)
    python test_all_notebooks.py --quick           # syntax-only (no execution)
    python test_all_notebooks.py --fix             # auto-fix HTML entity corruption
    python test_all_notebooks.py --stub-svgs       # create placeholder SVGs for missing outputs
    python test_all_notebooks.py --filter 3        # only notebooks matching "3"
    python test_all_notebooks.py --workers 8       # set concurrency (default: CPU count)
    python test_all_notebooks.py --no-parallel     # serial execution (good for debugging)
    python test_all_notebooks.py --narrative       # run in ecological narrative order
    python test_all_notebooks.py --regen           # only re-run notebooks with missing/invalid SVG
    python test_all_notebooks.py --svg-only        # validate existing SVGs without execution
    python test_all_notebooks.py --list            # list notebooks in narrative order and exit
    python test_all_notebooks.py --fail-fast       # stop on first non-empty failure
    python test_all_notebooks.py --quiet           # dashboard only (no per-notebook rows)
    python test_all_notebooks.py --verbose         # show full stderr inline for failures
    python test_all_notebooks.py --category svg    # filter report to a specific error category
    python test_all_notebooks.py --top-slow 10     # show N slowest notebooks (default: 5)
    python test_all_notebooks.py --color           # force ANSI colour output
    python test_all_notebooks.py --no-color        # disable ANSI colour output
"""
import ast
import glob
from importlib import import_module
import json
import os
import re
import subprocess
import sys
import time
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from types import ModuleType
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Any

pytest: ModuleType | None
try:
    pytest = import_module("py" + "test")
except ImportError:
    pytest = None  # type: ignore[assignment]

_prompt_tool: ModuleType | None
try:
    _prompt_tool = import_module("prompt_" + "tool")
except ImportError:
    _prompt_tool = None

generate_prompt: Callable[[str], str] | None = None
if _prompt_tool is not None:
    prompt_candidate = getattr(_prompt_tool, "generate_prompt", None)
    if callable(prompt_candidate):
        generate_prompt = prompt_candidate

# ─── ANSI Colour Support ─────────────────────────────────────────────────────

class _Clr:
    """Lazy ANSI escape codes; populated by _Clr.init() based on TTY/env detection."""
    GREEN  = ""; RED    = ""; YELLOW = ""; CYAN   = ""
    MAGENTA= ""; BOLD   = ""; DIM    = ""; RESET  = ""
    OK     = ""; FAIL   = ""; WARN   = ""; INFO   = ""

    @classmethod
    def init(cls, force_on: bool = False, force_off: bool = False) -> None:
        """Enable colours when stdout is a real TTY (or forced on)."""
        enabled = False
        if force_off:
            enabled = False
        elif force_on:
            enabled = True
        else:
            enabled = (
                sys.stdout.isatty()
                and os.environ.get("NO_COLOR") is None
                and os.environ.get("TERM") != "dumb"
            )
        if enabled:
            cls.GREEN   = "\033[92m"
            cls.RED     = "\033[91m"
            cls.YELLOW  = "\033[93m"
            cls.CYAN    = "\033[96m"
            cls.MAGENTA = "\033[95m"
            cls.BOLD    = "\033[1m"
            cls.DIM     = "\033[2m"
            cls.RESET   = "\033[0m"
        else:
            cls.GREEN = cls.RED = cls.YELLOW = cls.CYAN = ""
            cls.MAGENTA = cls.BOLD = cls.DIM = cls.RESET = ""
        cls.OK   = cls.GREEN  + cls.BOLD
        cls.FAIL = cls.RED    + cls.BOLD
        cls.WARN = cls.YELLOW + cls.BOLD
        cls.INFO = cls.CYAN

    @staticmethod
    def strip(text: str) -> str:
        """Remove ANSI codes from a string (for log files)."""
        return re.sub(r'\033\[[0-9;]*m', '', text)


# Ensure UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT = 300   # seconds per notebook
LOG_DIR         = "test_logs"
SVG_DIR         = "svg_output"


# ─── Ecological Narrative Order ────────────────────────────────────────────────────────────────────
# Pedagogical sequence from agent physics through ecosystem conservation to phenological synthesis.
# Seven acts covering all 92 notebooks: foundations → environment → populations →
#   human pressures → restoration → seed/pollination clocks → trophic/migration clocks
NARRATIVE_ORDER: List[Tuple[str, str]] = [
    # 0. FÍSICA DO VOO - Movement & Physics Foundations
    ("notebook_01",   "Karst landscape baseline - ecological foundation & agent setup"),
    ("notebook_02",   "Aerodynamics & Turn Radius - diverse flight patterns"),
    ("notebook_05",   "Roosting Behavior - nighttime collective tree flocking"),
    ("notebook_07",   "Flocking Cohesion - dynamic weights based on flock size"),
    ("notebook_08",   "Foraging Efficiency - speed reduction in resource zones"),
    ("notebook_09",   "Territoriality - hummingbirds defending nectar patches"),
    ("notebook_10",   "Mating Displays - spiral flight patterns"),
    # I. AMBIENTE FÍSICO - Landscape, Resources & Cycles
    ("notebook_03",   "Diurnal Cycles - background activity modulation"),
    ("notebook_04",   "Water Dependence - energy recharge at veredas"),
    ("notebook_06",   "Predator Avoidance - herbivores flee from carnivores"),
    ("notebook_11",   "Karst Obstacles - limestone outcrop collision avoidance"),
    ("notebook_11_1", "Enhanced Seed Dispersal Networks - multi-trophic interactions"),
    ("notebook_12",   "Cave Entry/Exit - sinkhole teleportation"),
    ("notebook_14",   "Rain Events - global downward force & cover seeking"),
    ("notebook_15",   "Seasonal Water - dynamic vereda expansion & contraction"),
    ("notebook_15.1", "Enhanced Invasive Competition - spatial dynamics & control"),
    ("notebook_19",   "Soil Diversity - sand vs clay biome partitioning"),
    # II. DINÂMICAS POPULACIONAIS - Species Interactions & Population
    ("notebook_13",   "Mastofauna & Flora - tapir browsing and trail making"),
    ("notebook_16",   "Fruiting Phenology - resource node pulsing"),
    ("notebook_17",   "Seed Dispersal - ornithochory"),
    ("notebook_18",   "Seed Germination - sapling growth dynamics"),
    ("notebook_20",   "Natural Fire - propagation & avoidance behavior"),
    ("notebook_21",   "Reproduction - energy-based spawning"),
    ("notebook_21.1", "Enhanced Multi-species Flocking - information transfer"),
    ("notebook_22",   "Mortality - old age & starvation physics"),
    ("notebook_22_1", "Plant-pollinator Mutualism - phenology and floral fidelity"),
    ("notebook_23",   "Carrying Capacity - population control mechanisms"),
    ("notebook_23_1", "Carrying Capacity Enhanced - spatial regulation"),
    ("notebook_24",   "Insect Swarms - temporary prey clouds & insectivore chase"),
    ("notebook_25",   "Scavenging - vultures circling death sites"),
    ("notebook_26",   "Nesting Competition - limited nest nodes"),
    ("notebook_27",   "Keystone Removal - flock collapse & ecological void"),
    ("notebook_28",   "Satiation - predators ignore prey after successful hunt"),
    ("notebook_29",   "Alarm Propagation - flee vector wave dynamics"),
    ("notebook_30",   "Migration Flow - stream enters left, exits right"),
    # III. PERTURBAÇÕES HUMANAS - Threats & Landscape Pressures
    ("notebook_31",   "Road Fragmentation - linear barrier with high fear cost"),
    ("notebook_31.1", "Enhanced Integrated Fire Management (MIF) - suppression sim"),
    ("notebook_32",   "Cattle Grazing - slow obstacles deplete resource nodes"),
    ("notebook_33",   "Fencing - hard boundaries for ground feeders"),
    ("notebook_33_1", "Fencing Enhanced - permeability & compliance dynamics"),
    ("notebook_34",   "Selective Logging - random removal of roosting tree nodes"),
    ("notebook_35",   "Noise Zones - circular repulsion from tourist vehicles"),
    ("notebook_36",   "Poaching Risk - high mortality in unprotected zones"),
    ("notebook_43",   "Climate Warming - evaporation increase, energy demands rise"),
    ("notebook_44",   "Extreme Drought - water nodes vanish; movement costs spike"),
    ("notebook_38",   "Controlled Burns - small contained fires, fast regeneration"),
    ("notebook_38_1", "Enhanced Climate Change Impacts - cascading tipping points"),
    # IV. CONSERVAÇÃO & RESTAURAÇÃO - Management & Recovery
    ("notebook_37",   "Agroforestry Rows - linear resource nodes attract biodiversity"),
    ("notebook_39",   "Corridor Planting - connecting separated habitat patches"),
    ("notebook_40",   "Ecotourism Trails - tourist agents on fixed paths"),
    ("notebook_41",   "Invasive Grasses - fast-spreading texture chokes saplings"),
    ("notebook_42",   "Restoration Weeding - clearing invasive texture zones"),
    ("notebook_45",   "Carbon Tracking - visual biomass bar chart"),
    ("notebook_45_1", "Carbon Tracking Enhanced - biomass dynamics"),
    ("notebook_46",   "Community Monitoring - camera traps highlight bird passage"),
    ("notebook_47",   "Traditional Harvesting - human agents collect fruit sustainably"),
    ("notebook_48",   "Apiary Pollination - bee nodes increase fruit regeneration"),
    ("notebook_49",   "Jaguar Presence - large predator causes massive avoidance"),
    ("notebook_50",   "Maned Wolf Dispersal - wolf agent plants Lobeira seeds"),
    ("notebook_51",   "Rapid Succession - open field \u2192 shrub \u2192 forest time-lapse"),
    ("notebook_51.1", "Multi-Scale Landscape Connectivity Analysis"),
    ("notebook_52",   "Zoning Policy - preservation, sustainable use, buffer zones"),
    ("notebook_52.1", "Zoning Policy Enhanced"),
    ("notebook_53",   "Resilience Test - fire + drought shock, measure recovery"),
    ("notebook_53.1", "Resilience Test Enhanced"),
    ("notebook_54",   "The Living Landscape - grand finale, all modules combined"),
    # V. RELÓGIOS I - Seed Dispersal, Pollination & Mycorrhizal Clocks
    ("notebook_55",   "Tucano-toco \u2194 Pequi Tree - ornithochory phenological clock"),
    ("notebook_56",   "Abelha-nativa \u2194 Cerrado Wildflowers - pollination clock"),
    ("notebook_57",   "Lobo-guar\u00e1 \u2194 Lobeira - endozoochory phenological clock"),
    ("notebook_58",   "Morcego-nect\u00e1rivo \u2194 Babassu Palm - chiropterophily clock"),
    ("notebook_59",   "AMF Fungi \u2194 Cerrado Root Network - mycorrhizal seasonality clock"),
    ("notebook_60",   "Arara-canind\u00e9 \u2194 Buriti Palm - vereda phenological clock"),
    ("notebook_61",   "Anta \u2194 Maca\u00faba/Jatob\u00e1 - megaherbivore endozoochory clock"),
    ("notebook_67",   "Orqu\u00eddea (Cyrtopodium) \u2194 Fungo Micorr\u00edzico - symbiotic spatial map"),
    ("notebook_68",   "Pequizeiro \u2194 Morcego-nect\u00e1rivo - spatial chiropterophily clock"),
    # VI. RELÓGIOS II - Trophic, Mutualism & Migration Clocks
    ("notebook_62",   "Tamandu\u00e1-bandeira \u2194 Termite Mounds - myrmecophagy phenological clock"),
    ("notebook_63",   "Formiga-Cerrado \u2194 EFN - mutualistic ant-plant defense clock"),
    ("notebook_64",   "Sapo-cururu & Pererecas \u2194 Ephemeral Ponds - explosive breeding clock"),
    ("notebook_65",   "Beija-flor \u2194 Canela-de-ema - pyrophytic pollination clock"),
    ("notebook_66",   "On\u00e7a-pintada \u2194 Queixadas - apex predator-prey karst clock"),
    ("notebook_69",   "Tamandu\u00e1-bandeira \u2194 Cupinzeiros - macro-predator regulation clock"),
    ("notebook_70",   "Jatob\u00e1 \u2194 Cutia - seed scatter-hoarding seasonal clock"),
    ("notebook_71",   "Seriema \u2194 Serpentes do Cerrado - seasonal predation clock"),
    ("notebook_72",   "Ip\u00ea-amarelo \u2194 Abelha-Mamangava - bloom synchrony migration clock"),
    ("notebook_73",   "Buriti \u2194 Fungo-Ectomicorr\u00edzico - vereda flood peak clock"),
    ("notebook_74",   "Andorinha-rabo-branco \u2194 Invertebrados - trans-continental migration clock"),
    ("notebook_75",   "Lobo-guar\u00e1 \u2194 Lobeira II - territory & den phenology clock"),
    ("notebook_76",   "Gavi\u00e3o-real \u2194 Macaco-prego - apex raptor-prey clock"),
    ("notebook_77",   "Mandacaru \u2194 Morcego-polinizador - night bloom phenological clock"),
    ("notebook_78",   "Cupim \u2194 Termitomyces - underground fungal symbiosis clock"),
    ("notebook_79",   "Suiriri \u2194 Murici - wet-season frugivory phenological clock"),
    ("notebook_80",   "Sa\u00fava \u2194 Leucoagaricus - seasonal subterranean symbiosis clock"),
]

# Fast lookup: notebook stem → narrative position (1-based)
_NARRATIVE_POS: Dict[str, int] = {s: i + 1 for i, (s, _) in enumerate(NARRATIVE_ORDER)}

# Narrative section boundaries (start_pos, end_pos, label)
_NARRATIVE_SECTIONS = [
    (1,  7,  "0.   FÍS. DO VOO - Movement & Physics Foundations"),
    (8,  17, "I.   AMBIENTE FÍSICO - Landscape, Resources & Cycles"),
    (18, 35, "II.  DINÂMICAS POPULACIONAIS - Species Interactions & Population"),
    (36, 47, "III. PERTURBAÇÕES HUMANAS - Threats & Landscape Pressures"),
    (48, 66, "IV.  CONSERVAÇÃO & RESTAURAÇÃO - Management & Recovery"),
    (67, 75, "V.   RELÓGIOS I - Seed Dispersal, Pollination & Mycorrhizal Clocks"),
    (76, 92, "VI.  RELÓGIOS II - Trophic, Mutualism & Migration Clocks"),
]


# ─── Result Model ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NotebookResult:
    filename:              str
    passed:                bool  = False
    duration_s:            float = 0.0
    error_category:        str   = ""    # syntax|import|runtime|svg|corruption|timeout|empty
    error_summary:         str   = ""
    stderr_snippet:        str   = ""
    stdout_snippet:        str   = ""
    svg_generated:         bool  = False
    svg_valid:             bool  = False   # True if file has <svg tag
    svg_xml_valid:         bool  = False   # True if full XML parse succeeds
    svg_animate_count:     int   = 0       # number of <animate> elements
    svg_size_kb:           float = 0.0
    svg_title:             str   = ""      # extracted from SVG text
    has_entity_corruption: bool  = False
    uses_eco_base:         bool  = False
    is_empty:              bool  = False
    narrative_pos:         int   = 0       # position in NARRATIVE_ORDER (0 = not listed)
    health_score:          int   = 0       # 0-100 composite quality score


# ─── SVG Info NamedTuple ──────────────────────────────────────────────────────────────────────

class SvgInfo(NamedTuple):
    exists:        bool
    valid_tag:     bool    # has <svg tag
    xml_valid:     bool    # full XML parse passes
    animate_count: int
    size_kb:       float
    title:         str
    amp_errors:    int     # unescaped & count


_BARE_AMP_RE  = re.compile(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)')
_SVG_TITLE_RE = re.compile(r'class="ui-title">([^<]{1,120})</text>', re.IGNORECASE)
_SVG_MODULE_RE = re.compile(r'MODULE[^<]{0,100}')


# ─── Python Interpreter Detection ────────────────────────────────────────────

def _get_notebook_python() -> str:
    """Return the best Python interpreter for running notebooks.

    Priority:
      1. NOTEBOOK_PYTHON environment variable (explicit override)
      2. .venv next to this script (Windows Scripts/ or Unix bin/)
      3. sys.executable (the interpreter running this test script)
    """
    env_python = os.environ.get('NOTEBOOK_PYTHON', '')
    if env_python and os.path.isfile(env_python):
        return env_python

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, '.venv', 'Scripts', 'python.exe'),  # Windows
        os.path.join(here, '.venv', 'bin', 'python'),          # Unix/macOS
        os.path.join(here, 'venv',  'Scripts', 'python.exe'),
        os.path.join(here, 'venv',  'bin', 'python'),
    ):
        if os.path.isfile(candidate):
            return candidate

    return sys.executable


# Resolved once at import time so all workers share the same value.
_NOTEBOOK_PYTHON = _get_notebook_python()


# ─── Pre-Flight Checks ────────────────────────────────────────────────────────

HTML_ENTITY_PATTERN = re.compile(r'&amp;|&lt;|&gt;|&#39;|&quot;')


def check_html_entity_corruption(filepath: str) -> List[Tuple[int, str]]:
    """Return list of (line_number, line) where HTML entity corruption is found."""
    hits = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f, start=1):
            if HTML_ENTITY_PATTERN.search(line):
                hits.append((i, line.rstrip()))
    return hits


def check_syntax(filepath: str) -> Optional[str]:
    """Return None if syntax is valid, else error message string."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        source = f.read()
    try:
        ast.parse(source, filename=filepath)
        return None
    except SyntaxError as e:
        return f"Line {e.lineno}: {e.msg}"


def check_imports_eco_base(filepath: str) -> bool:
    """Check if notebook imports from eco_base."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    return bool(re.search(r'from\s+eco_base\s+import|import\s+eco_base', content))


def check_empty(filepath: str) -> bool:
    """Return True if the file is missing, 0-byte, or contains only whitespace."""
    try:
        if os.path.getsize(filepath) == 0:
            return True
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return not f.read().strip()
    except Exception:
        return True


def check_svg_full(notebook_name: str, svg_dir: str) -> SvgInfo:
    """Full SVG audit: existence, tag, XML validity, animation count, title, bare-& count."""
    stem     = os.path.splitext(notebook_name)[0]
    svg_path = os.path.join(svg_dir, f"{stem}.svg")
    _empty   = SvgInfo(False, False, False, 0, 0.0, "", 0)
    if not os.path.isfile(svg_path):
        return _empty
    try:
        size_kb = os.path.getsize(svg_path) / 1024
        with open(svg_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        valid_tag     = '<svg' in content[:1024].lower()  # type: ignore
        animate_count = content.count('<animate')
        amp_errors    = len(_BARE_AMP_RE.findall(content))
        title_m = _SVG_TITLE_RE.search(content)
        title   = title_m.group(1).strip() if title_m else _extract_module_title(content)
        xml_valid = False
        if valid_tag and amp_errors == 0:
            try:
                ET.fromstring(content.encode('utf-8'))
                xml_valid = True
            except ET.ParseError:
                pass
        return SvgInfo(True, valid_tag, xml_valid, animate_count, size_kb, title, amp_errors)
    except Exception:
        return SvgInfo(True, False, False, 0, 0.0, "", 0)


def _extract_module_title(content: str) -> str:
    """Fallback title extraction from SVG when ui-title class is absent."""
    m = _SVG_MODULE_RE.search(content)
    if m:
        raw = m.group(0).strip()
        raw = raw.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return raw[:80]  # type: ignore
    return ""


# ─── Error Categoriser ───────────────────────────────────────────────────────

# Ordered from most-specific to least-specific
_ERROR_PATTERNS: List[Tuple[str, str]] = [
    (r'SyntaxError:\s*(.+)',                 "syntax"),
    (r'ModuleNotFoundError:\s*(.+)',         "import"),
    (r'ImportError:\s*(.+)',                 "import"),
    (r'NameError:\s*(.+)',                   "runtime"),
    (r'TypeError:\s*(.+)',                   "runtime"),
    (r'AttributeError:\s*(.+)',              "runtime"),
    (r'ValueError:\s*(.+)',                  "runtime"),
    (r'IndexError:\s*(.+)',                  "runtime"),
    (r'KeyError:\s*(.+)',                    "runtime"),
    (r'ZeroDivisionError:\s*(.+)',           "runtime"),
    (r'RecursionError:\s*(.+)',              "runtime"),
    (r'MemoryError',                         "runtime"),
    (r'FileNotFoundError:\s*(.+)',           "runtime"),
    (r'OSError:\s*(.+)',                     "runtime"),
    (r'RuntimeError:\s*(.+)',               "runtime"),
]


def categorise_error(stderr: str) -> Tuple[str, str]:
    """Classify error from stderr → (category, short_summary)."""
    if not stderr.strip():
        return "unknown", "No stderr output"

    for pattern, category in _ERROR_PATTERNS:
        m = re.search(pattern, stderr)
        if m:
            summary = m.group(1).strip() if m.lastindex else category.upper()
            # Flag HTML-entity corruption that shows up as NameError(&amp;...)
            if "amp;" in summary or "&lt;" in summary:
                return "corruption", f"HTML entity corruption: {summary}"
            return category, summary

    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    return "runtime", lines[-1] if lines else "Unknown error"


# ─── Auto-Fix ────────────────────────────────────────────────────────────────

_ENTITY_MAP = {
    '&amp;':  '&',
    '&lt;':   '<',
    '&gt;':   '>',
    '&#39;':  "'",
    '&quot;': '"',
}


def fix_html_entities(filepath: str) -> bool:
    """Replace HTML entities with plaintext equivalents. Returns True if changed."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    new_content = content
    for entity, char in _ENTITY_MAP.items():
        new_content = new_content.replace(entity, char)
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


def fix_svg_entities(svg_dir: str) -> None:
    """Sanitize all SVG files in svg_dir: escape bare & that break XML parsers.

    Keeps valid XML references (&amp; &lt; &gt; &quot; &apos; &#N; &#xN;) intact.
    Reports any remaining XML errors that are NOT ampersand-related.
    """
    _RE  = re.compile(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)')
    files = sorted(glob.glob(os.path.join(svg_dir, '*.svg')))
    if not files:
        print(f"  No SVG files found in: {svg_dir}")
        return
    fixed_count = 0
    error_count = 0
    clean_count = 0
    for fp in files:
        name = os.path.basename(fp)
        try:
            content = open(fp, 'r', encoding='utf-8', errors='replace').read()
            matches = _RE.findall(content)
            if matches:
                fixed = _RE.sub('&amp;', content)
                with open(fp, 'w', encoding='utf-8') as fh:
                    fh.write(fixed)
                print(f"    [fixed] {len(matches):>3} bare '&' -> &amp;  in {name}")
                fixed_count += 1
                content = fixed  # validate the fixed version
            # Secondary: verify full XML parse
            try:
                ET.fromstring(content.encode('utf-8'))
                clean_count += 1
            except ET.ParseError as exc:
                print(f"    [ERROR] XML parse still fails: {name}  -- {str(exc)[:100]}")  # type: ignore
                error_count += 1
        except Exception as exc:
            print(f"    [ERROR] Cannot read/write {name}: {exc}")
            error_count += 1
    print(f"\n  SVG fix complete: {fixed_count} fixed, {error_count} XML errors, "
          f"{clean_count}/{len(files)} fully valid.")


def create_stub_svgs(notebooks: List[str], svg_dir: str) -> None:
        """Create valid placeholder SVGs for notebooks that have no SVG output.

        This is a practical fallback for workflows where many notebook files are
        currently empty placeholders but downstream tools expect one SVG per notebook.
        Existing SVGs are preserved.
        """
        os.makedirs(svg_dir, exist_ok=True)

        created = 0
        skipped = 0
        errors = 0

        for nb in notebooks:
                filename = os.path.basename(nb)
                stem = os.path.splitext(filename)[0]
                svg_path = os.path.join(svg_dir, f"{stem}.svg")

                if os.path.isfile(svg_path):
                        skipped += 1
                        continue

                is_empty_nb = check_empty(nb)
                nb_state = "empty" if is_empty_nb else "failed/no-output"
                title = f"ECO-SIM Placeholder: {stem}"

                # Keep output deterministic and XML-safe.
                svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900" role="img" aria-labelledby="title desc">
    <title>{title}</title>
    <desc>Auto-generated placeholder SVG for {stem}. Notebook state: {nb_state}.</desc>
    <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#10221A"/>
            <stop offset="100%" stop-color="#1E3A2F"/>
        </linearGradient>
    </defs>

    <rect width="1400" height="900" fill="url(#bg)"/>
    <rect x="80" y="80" width="1240" height="740" rx="18" fill="none" stroke="#86C29A" stroke-width="2" opacity="0.9"/>

    <text x="110" y="180" fill="#EAF7EE" font-size="46" font-family="Segoe UI, Arial, sans-serif">{title}</text>
    <text x="110" y="240" fill="#BFE5CC" font-size="26" font-family="Segoe UI, Arial, sans-serif">Notebook status: {nb_state}</text>
    <text x="110" y="290" fill="#A7DDB9" font-size="22" font-family="Segoe UI, Arial, sans-serif">Reason: no generated SVG was available at validation time.</text>
    <text x="110" y="360" fill="#D2F0DC" font-size="20" font-family="Segoe UI, Arial, sans-serif">Generated by test_all_notebooks.py --stub-svgs</text>

    <circle cx="1180" cy="250" r="22" fill="#7ED6A2" opacity="0.95">
        <animate attributeName="opacity" values="0.25;1;0.25" dur="2.2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="1240" cy="250" r="14" fill="#C8F1D8" opacity="0.8">
        <animate attributeName="r" values="8;20;8" dur="2.6s" repeatCount="indefinite"/>
    </circle>
</svg>
'''

                try:
                        with open(svg_path, 'w', encoding='utf-8') as fh:
                                fh.write(svg)
                        created += 1
                        print(f"    [created] {os.path.basename(svg_path)}")
                except Exception as exc:
                        errors += 1
                        print(f"    [ERROR] could not write {os.path.basename(svg_path)}: {exc}")

        print(
                f"\n  Stub SVG generation complete: {created} created, "
                f"{skipped} already existed, {errors} errors."
        )


# ─── Execution ────────────────────────────────────────────────────────────────

def run_notebook(filepath: str, timeout: int = DEFAULT_TIMEOUT) -> NotebookResult:
    """Execute a single notebook and return structured results."""
    filename = os.path.basename(filepath)
    stem     = os.path.splitext(filename)[0]
    result   = NotebookResult(filename=filename)
    base_dir = os.path.dirname(os.path.abspath(filepath))
    svg_dir  = os.path.join(base_dir, SVG_DIR)

    result.narrative_pos = _NARRATIVE_POS.get(stem, 0)
    result.uses_eco_base = check_imports_eco_base(filepath)
    result.is_empty      = check_empty(filepath)

    if result.is_empty:
        result.error_category = "empty"
        result.error_summary  = "File is empty (0 bytes or whitespace only)"
    else:
        # Pre-flight: entity corruption
        if check_html_entity_corruption(filepath):
            result.has_entity_corruption = True

        # Pre-flight: syntax
        syntax_err = check_syntax(filepath)
        if syntax_err:
            result.error_category = "syntax"
            result.error_summary  = syntax_err
        else:
            # Execute
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [_NOTEBOOK_PYTHON, filepath],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=timeout,
                    cwd=base_dir,
                )
                result.duration_s     = time.perf_counter() - t0
                result.stdout_snippet = (proc.stdout or "")[-800:]  # type: ignore
                result.stderr_snippet = (proc.stderr or "")[-1500:]  # type: ignore
                if proc.returncode == 0:
                    result.passed = True
                else:
                    cat, summary          = categorise_error(proc.stderr or "")
                    result.error_category = cat
                    result.error_summary  = summary
            except subprocess.TimeoutExpired:
                result.duration_s     = time.perf_counter() - t0
                result.error_category = "timeout"
                result.error_summary  = f"Exceeded {timeout}s timeout"

    # SVG audit -- ALWAYS, whether notebook ran, was empty, had syntax errors, etc.
    _fill_svg(result, check_svg_full(filename, svg_dir))

    # Passed notebook with no SVG -> soft warning
    if result.passed and not result.svg_generated:
        result.error_category = "svg"
        result.error_summary  = "Notebook ran OK but produced no SVG output"

    return result


def _compute_health_score(r: 'NotebookResult') -> int:
    """Return a 0-100 composite quality score for a notebook result."""
    score = 0
    if r.passed:                    score += 40
    if r.svg_xml_valid:             score += 25
    elif r.svg_valid:               score += 10
    if r.svg_animate_count >= 20:   score += 20
    elif r.svg_animate_count >= 10: score += 12
    elif r.svg_animate_count >= 5:  score += 6
    if r.svg_size_kb >= 100:        score += 10
    elif r.svg_size_kb >= 40:       score += 5
    if r.has_entity_corruption:     score -= 20
    if r.is_empty:                  score  = 0
    return max(0, min(100, score))


def _health_bar(score: int) -> str:
    """Return a compact coloured bar string representing a health score."""
    filled = score // 10
    bar = "█" * filled + "░" * (10 - filled)
    if score >= 75:
        color = _Clr.GREEN
    elif score >= 40:
        color = _Clr.YELLOW
    else:
        color = _Clr.RED
    return f"{color}{bar}{_Clr.RESET} {score:>3}"


def _load_previous_summary(log_dir: str) -> Optional[Dict[str, Any]]:
    """Load the last saved summary.json for trend comparison; returns None if unavailable."""
    path = os.path.join(log_dir, "summary.json")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data  # type: ignore[return-value]
    except Exception:
        return None


def _fill_svg(result: NotebookResult, info: SvgInfo) -> None:
    """Copy SvgInfo fields into a NotebookResult and compute the health score."""
    result.svg_generated     = info.exists
    result.svg_valid         = info.valid_tag
    result.svg_xml_valid     = info.xml_valid
    result.svg_animate_count = info.animate_count
    result.svg_size_kb       = info.size_kb
    result.svg_title         = info.title
    result.health_score      = _compute_health_score(result)


# \u2500\u2500\u2500 SVG-Only Audit \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def run_svg_audit(notebooks: List[str], svg_dir: str) -> List[NotebookResult]:
    """Audit existing SVGs without executing notebooks."""
    results = []
    for nb in notebooks:
        filename = os.path.basename(nb)
        stem     = os.path.splitext(filename)[0]
        r        = NotebookResult(filename=filename)
        r.narrative_pos = _NARRATIVE_POS.get(stem, 0)
        r.uses_eco_base = check_imports_eco_base(nb) if not check_empty(nb) else False
        r.is_empty      = check_empty(nb)
        info = check_svg_full(filename, svg_dir)
        _fill_svg(r, info)
        r.passed = info.exists and info.valid_tag and info.xml_valid
        if not r.passed:
            if not info.exists:
                r.error_category = "svg"; r.error_summary = "SVG file missing"
            elif not info.valid_tag:
                r.error_category = "svg"; r.error_summary = "SVG has no <svg> tag"
            elif info.amp_errors:
                r.error_category = "svg"
                r.error_summary  = f"SVG fails XML parse \u2014 {info.amp_errors} unescaped '&'"
            else:
                r.error_category = "svg"; r.error_summary = "SVG fails XML parse"
        results.append(r)
    return results


# ─── Progress Tracker (thread-safe) ──────────────────────────────────────────

class Progress:
    def __init__(self, total: int):
        self.total    = total
        self.done     = 0
        self._lock    = threading.Lock()
        self._t_start = time.perf_counter()
        self.first_fail: Optional[str] = None  # name of first failure

    def increment(self, name: str, ok: bool) -> None:
        with self._lock:
            self.done += 1
            if not ok and self.first_fail is None:
                self.first_fail = name
            elapsed   = time.perf_counter() - self._t_start
            avg       = elapsed / self.done
            remaining = avg * (self.total - self.done)
            pct       = 100 * self.done / self.total
            icon      = f"{_Clr.OK}\u2705{_Clr.RESET}" if ok else f"{_Clr.FAIL}\u274c{_Clr.RESET}"
            eta_str   = f"ETA {remaining:.0f}s" if self.done < self.total else "done"
            bar_len   = 28
            filled    = int(bar_len * self.done / self.total)
            bar_fill  = f"{_Clr.GREEN}" + "\u2588" * filled + f"{_Clr.DIM}" + "\u2591" * (bar_len - filled) + f"{_Clr.RESET}"
            print(
                f"  [{bar_fill}] {pct:5.1f}%  {self.done}/{self.total}  "
                f"{_Clr.DIM}{eta_str}{_Clr.RESET}  {icon} {name}",
                flush=True,
            )


# ─── Report ───────────────────────────────────────────────────────────────────

CATEGORY_ICONS: Dict[str, str] = {
    "syntax":     "[!] SYNTAX ",
    "import":     "[I] IMPORT ",
    "runtime":    "[X] RUNTIME",
    "corruption": "[C] CORRUPT",
    "timeout":    "[T] TIMEOUT",
    "svg":        "[S]  SVG   ",
    "empty":      "[0] EMPTY  ",
    "unknown":    "[?] UNKNOWN",
}

_SECT_WIDTH = 76


def _sect_header(label: str) -> str:
    pad = max(0, _SECT_WIDTH - 4 - len(label))
    return f"  \u2502 {label} {'':>{pad}}"


def print_narrative_list(svg_dir: str) -> None:
    """Print all notebooks in ecological narrative order with SVG status."""
    border = "=" * _SECT_WIDTH
    thin   = "-" * _SECT_WIDTH
    print(f"\n{_Clr.BOLD}{border}{_Clr.RESET}")
    print(f"  {_Clr.CYAN}ECO-SIM \u2500 NARRATIVE ORDER{_Clr.RESET} (ecological argument sequence)")
    print(f"{_Clr.BOLD}{border}{_Clr.RESET}\n")
    print(f"  {'#':>2}  {'notebook':<22s}  {'SVG':8}  {'anim':>5}  {'size':>8}  description")
    print(f"  {thin}")

    for pos, (stem, description) in enumerate(NARRATIVE_ORDER, start=1):
        for s_start, _s_end, s_label in _NARRATIVE_SECTIONS:
            if pos == s_start:
                print(f"\n  {_Clr.INFO}{s_label}{_Clr.RESET}")
                print(f"  {thin}")
        info = check_svg_full(f"{stem}.py", svg_dir)
        py_exists = os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{stem}.py"))
        if info.exists and info.xml_valid:
            svg_status = f"{_Clr.GREEN}\u2705 SVG OK {_Clr.RESET}"
        elif info.exists and info.valid_tag:
            svg_status = f"{_Clr.YELLOW}\u26a0  SVG?xml{_Clr.RESET}"
        elif info.exists:
            svg_status = f"{_Clr.RED}\u274c SVG bad{_Clr.RESET}"
        elif not py_exists:
            svg_status = f"{_Clr.DIM}-- MISSING{_Clr.RESET}"
        else:
            svg_status = f"{_Clr.DIM}-- no SVG{_Clr.RESET}"
        anim = f"  {info.animate_count:>4}\u25b6" if info.exists else "       "
        size = f"  {info.size_kb:>6.0f}KB" if info.exists else "          "
        print(f"  {pos:>2}.  {stem:<22s}  {svg_status}  {anim}{size}  {description}")

    # Missing .py files
    base = os.path.dirname(os.path.abspath(__file__))
    missing = [s for s, _ in NARRATIVE_ORDER if not os.path.isfile(os.path.join(base, f"{s}.py"))]
    if missing:
        print(f"\n  {_Clr.WARN}[!] {len(missing)} notebook(s) listed in NARRATIVE_ORDER but not on disk:{_Clr.RESET}")
        for s in missing:
            print(f"      \u2022 {s}.py")

    print(f"\n{_Clr.BOLD}{border}{_Clr.RESET}\n")


def print_report(
    results:    List[NotebookResult],
    elapsed:    float,
    log_dir:    str,
    narrative:  bool = False,
    quiet:      bool = False,
    verbose:    bool = False,
    top_slow:   int  = 5,
    category:   Optional[str] = None,
    prev:       Optional[Dict[str, Any]] = None,
) -> None:
    border = _Clr.BOLD + "=" * _SECT_WIDTH + _Clr.RESET
    thin   = "-" * _SECT_WIDTH

    passed       = [r for r in results if r.passed]
    failed       = [r for r in results if not r.passed]
    empty_nb     = [r for r in results if r.is_empty]
    non_empty_f  = [r for r in failed if not r.is_empty]
    svg_ok       = [r for r in results if r.svg_xml_valid]
    svg_tag_only = [r for r in results if r.svg_valid and not r.svg_xml_valid]
    svg_invalid  = [r for r in results if r.svg_generated and not r.svg_valid]
    svg_missing  = [r for r in results if not r.svg_generated]
    eco_users    = [r for r in results if r.uses_eco_base]
    corrupted    = [r for r in results if r.has_entity_corruption]
    animated     = [r for r in results if r.svg_animate_count > 0]
    avg_anim     = (
        sum(r.svg_animate_count for r in animated) / len(animated)
        if animated else 0
    )
    total_kb     = sum(r.svg_size_kb for r in results)
    pass_rate    = 100 * len(passed) / len(results) if results else 0.0
    avg_health   = int(sum(r.health_score for r in results) / len(results)) if results else 0

    print(f"\n{border}")
    print(f"  {_Clr.BOLD}ECO-SIM \u2500 NOTEBOOK VALIDATION REPORT{_Clr.RESET}")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{border}\n")

    # ── Dashboard ────────────────────────────────────────────────────────────
    p_col = _Clr.GREEN if len(non_empty_f) == 0 else (_Clr.YELLOW if pass_rate >= 80 else _Clr.RED)
    print(f"  Total notebooks      : {_Clr.BOLD}{len(results)}{_Clr.RESET}")
    print(f"  Pass rate            : {p_col}{_Clr.BOLD}{pass_rate:.1f}%{_Clr.RESET}")
    print(f"  {_Clr.GREEN}\u2705 Passed{_Clr.RESET}             : {len(passed)}")
    print(f"  {_Clr.RED}\u274c Failed (non-empty){_Clr.RESET} : {len(non_empty_f)}")
    print(f"  [0] Empty .py files  : {len(empty_nb)}{_Clr.DIM}  (need regeneration){_Clr.RESET}")
    print(f"  --- eco_base users   : {len(eco_users)}")
    print(f"  [S] SVG xml-valid    : {_Clr.GREEN}{len(svg_ok)}{_Clr.RESET}")
    print(f"  [S] SVG tag-only     : {len(svg_tag_only)}")
    print(f"  [S] SVG invalid      : {len(svg_invalid)}")
    print(f"  [S] SVG missing      : {_Clr.YELLOW if svg_missing else ''}{len(svg_missing)}{_Clr.RESET}")
    print(f"  ... Avg <animate>/SVG: {avg_anim:.0f}")
    print(f"  ... Total SVG (KB)   : {total_kb:.0f}")
    print(f"  [C] Entity corrupt   : {len(corrupted)}")
    print(f"  Avg health score     : {_health_bar(avg_health)}")
    print(f"  Wall time (s)        : {elapsed:.1f}")

    # ── Trend comparison ─────────────────────────────────────────────────────
    if prev is not None:
        prev_passed  = prev.get("passed", 0)
        prev_failed  = prev.get("failed", 0)
        prev_total   = prev.get("total",  0)
        prev_ts      = prev.get("timestamp", "?")
        delta_passed = len(passed)      - prev_passed
        delta_failed = len(non_empty_f) - prev_failed
        def _delta(n: int) -> str:
            if n > 0:  return f"{_Clr.GREEN}+{n}{_Clr.RESET}"
            if n < 0:  return f"{_Clr.RED}{n}{_Clr.RESET}"
            return f"{_Clr.DIM}+0{_Clr.RESET}"
        print(f"\n  {_Clr.INFO}Trend vs last run{_Clr.RESET} ({prev_ts[:16]}):")
        print(f"    Passed  : {prev_passed} \u2192 {len(passed)}  ({_delta(delta_passed)})")
        print(f"    Failed  : {prev_failed} \u2192 {len(non_empty_f)}  ({_delta(-delta_failed)})")
        print(f"    Total   : {prev_total}  \u2192 {len(results)}")

    print(f"\n{thin}")

    if quiet:
        # Dashboard only, skip per-notebook rows
        _write_summary_files(results, passed, non_empty_f, empty_nb, svg_ok, svg_missing,
                             total_kb, avg_anim, elapsed, log_dir, verbose)
        print(f"\n{border}\n")
        return

    # ── Narrative order results ───────────────────────────────────────────────
    pos_map = {r.filename: r for r in results}
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n  {_Clr.BOLD}>> RESULTS IN NARRATIVE ORDER:{_Clr.RESET}")
    print(f"     {'#':>2}  {'notebook':<28s}  {'st':2}  {'SVG':6}  {'anim':>4}  {'dur':>6}  {'health':>10}  title")
    print(f"     {thin}")

    prev_s    = -1
    # Per-section pass counters
    sect_stats: Dict[int, Dict[str, int]] = {}
    for pos, (stem, _desc) in enumerate(NARRATIVE_ORDER, start=1):
        for s_start, s_end, s_label in _NARRATIVE_SECTIONS:
            if pos == s_start and s_start != prev_s:
                # Print section header with pass summary when starting a new section
                if prev_s >= 0:
                    st = sect_stats.get(prev_s, {})
                    _print_section_summary(st)
                print(f"\n     {_Clr.INFO}{s_label}{_Clr.RESET}")
                prev_s = s_start
                sect_stats[s_start] = {"pass": 0, "fail": 0, "empty": 0, "missing_py": 0}

        filename = f"{stem}.py"
        r = pos_map.get(filename)
        py_exists = os.path.isfile(os.path.join(base_dir, filename))

        # Update section stats
        st = sect_stats.get(prev_s, {"pass": 0, "fail": 0, "empty": 0, "missing_py": 0})
        if r is None:
            if not py_exists:
                st["missing_py"] = st.get("missing_py", 0) + 1
                print(f"     {pos:>2}. {filename:<28s}  {_Clr.DIM}-- .py file missing{_Clr.RESET}")
            else:
                print(f"     {pos:>2}. {filename:<28s}  {_Clr.DIM}-- not tested{_Clr.RESET}")
            continue

        if r.passed:     st["pass"]  = st.get("pass",  0) + 1
        elif r.is_empty: st["empty"] = st.get("empty", 0) + 1
        else:            st["fail"]  = st.get("fail",  0) + 1

        # Skip if --category filter is active and doesn't match
        if category and r.error_category and r.error_category != category and r.passed:
            # Still count but only print fails matching category
            pass

        icon   = f"{_Clr.GREEN}\u2705{_Clr.RESET}" if r.passed else (
                 f"{_Clr.DIM}[0]{_Clr.RESET}" if r.is_empty else f"{_Clr.RED}\u274c{_Clr.RESET}")
        s_tag  = (f"{_Clr.GREEN}xml\u2714{_Clr.RESET}" if r.svg_xml_valid else
                 (f"{_Clr.YELLOW}tag\u26a0{_Clr.RESET}" if r.svg_valid else f"{_Clr.DIM} -- {_Clr.RESET}"))
        anim   = f"{r.svg_animate_count:>4}" if r.svg_generated else "    "
        dur    = f"{r.duration_s:>5.1f}s" if r.duration_s > 0 else "     -"
        hbar   = _health_bar(r.health_score)
        t_str  = f"  {_Clr.DIM}'{r.svg_title[:38]}'{_Clr.RESET}" if r.svg_title else ""

        if category is None or (r.error_category == category) or r.passed:
            print(f"     {pos:>2}. {filename:<28s}  {icon}  [{s_tag}] {anim}  {dur}  {hbar}{t_str}")
            if not r.passed and r.error_summary and not r.is_empty:
                cat_icon = CATEGORY_ICONS.get(r.error_category, "[?]")
                print(f"          {_Clr.RED}+-- {cat_icon}: {r.error_summary[:80]}{_Clr.RESET}")
                if verbose and r.stderr_snippet:
                    for ln in r.stderr_snippet.strip().splitlines()[-6:]:
                        print(f"             {_Clr.DIM}{ln}{_Clr.RESET}")

    # Print final section summary
    if prev_s >= 0:
        _print_section_summary(sect_stats.get(prev_s, {}))

    # ── Missing .py files from NARRATIVE_ORDER ───────────────────────────────
    missing_py = [s for s, _ in NARRATIVE_ORDER
                  if not os.path.isfile(os.path.join(base_dir, f"{s}.py"))]
    if missing_py:
        print(f"\n  {_Clr.WARN}[!] {len(missing_py)} NARRATIVE entry(ies) have no .py file on disk:{_Clr.RESET}")
        for s in missing_py:
            print(f"       \u2022 {s}.py")

    # ── Non-narrative notebooks ──────────────────────────────────────────────
    narrative_files = {f"{s}.py" for s, _ in NARRATIVE_ORDER}
    extra = [r for r in results if r.filename not in narrative_files]
    if extra:
        print(f"\n  {_Clr.INFO}[+] OTHER NOTEBOOKS ({len(extra)}):{_Clr.RESET}")
        for r in sorted(extra, key=lambda x: x.filename):
            icon   = f"{_Clr.GREEN}\u2705{_Clr.RESET}" if r.passed else (
                     f"{_Clr.DIM}[0]{_Clr.RESET}" if r.is_empty else f"{_Clr.RED}\u274c{_Clr.RESET}")
            s_tag  = (f"{_Clr.GREEN}xml\u2714{_Clr.RESET}" if r.svg_xml_valid else
                     (f"{_Clr.YELLOW}tag\u26a0{_Clr.RESET}" if r.svg_valid else f"{_Clr.DIM} -- {_Clr.RESET}"))
            anim   = f"{r.svg_animate_count:>4}" if r.svg_generated else "    "
            dur    = f"{r.duration_s:>5.1f}s" if r.duration_s > 0 else "     -"
            print(f"     {r.filename:<34s}  {icon}  [{s_tag}] {anim}  {dur}")
            if not r.passed and not r.is_empty and r.error_summary:
                print(f"          {_Clr.RED}+-- {r.error_summary[:80]}{_Clr.RESET}")

    # ── Failure breakdown by category ────────────────────────────────────────
    if non_empty_f:
        print(f"\n{thin}")
        print(f"\n  {_Clr.FAIL}\u274c FAILURES BY CATEGORY:{_Clr.RESET}")
        categories: Dict[str, List[NotebookResult]] = {}
        for r in non_empty_f:
            categories.setdefault(r.error_category or "unknown", []).append(r)
        for cat in ["corruption", "syntax", "import", "runtime", "svg", "timeout", "unknown"]:
            group = categories.get(cat, [])
            if not group:
                continue
            ci = CATEGORY_ICONS.get(cat, cat.upper())
            print(f"\n     {_Clr.YELLOW}{ci}{_Clr.RESET} ({len(group)}):")
            for r in sorted(group, key=lambda x: x.filename):
                eco_tag = f" {_Clr.DIM}[eco_base]{_Clr.RESET}" if r.uses_eco_base else ""
                print(f"       \u2022 {r.filename:<30s}{eco_tag}")
                print(f"         \u2514\u2500 {_Clr.RED}{r.error_summary}{_Clr.RESET}")
                if verbose and r.stderr_snippet:
                    for ln in r.stderr_snippet.strip().splitlines()[-5:]:
                        print(f"            {_Clr.DIM}{ln}{_Clr.RESET}")

    # ── SVG detail warnings ──────────────────────────────────────────────────
    if svg_tag_only:
        print(f"\n  [!] {len(svg_tag_only)} SVG(s) have <svg> tag but FAIL XML parse:")
        for r in svg_tag_only:
            print(f"     * {r.filename}")

    if empty_nb and not quiet:
        print(f"\n  {_Clr.DIM}[0] {len(empty_nb)} empty .py files cannot generate SVGs.")
        print("     To regenerate: implement the notebook code, then run:")
        print("       python test_all_notebooks.py --regen")
        print("     To audit existing SVGs only:")
        print("       python test_all_notebooks.py --svg-only")
        print(f"     To fix XML entity issues in SVGs already on disk:{_Clr.RESET}")
        print("       python test_all_notebooks.py --fix-svgs")

    if corrupted:
        print(f"\n  {_Clr.WARN}[C] HTML ENTITY CORRUPTION in {len(corrupted)} source file(s).{_Clr.RESET}")
        print("     Run: python test_all_notebooks.py --fix")

    # ── Slowest notebooks ────────────────────────────────────────────────────
    timed = sorted([r for r in results if r.duration_s > 0], key=lambda r: -r.duration_s)
    if timed and top_slow > 0:
        print(f"\n{thin}")
        n = min(top_slow, len(timed))
        print(f"\n  {_Clr.INFO}\u23f1  TOP {n} SLOWEST NOTEBOOKS:{_Clr.RESET}")
        for rank, r in enumerate(timed[:n], start=1):
            icon = f"{_Clr.GREEN}\u2705{_Clr.RESET}" if r.passed else f"{_Clr.RED}\u274c{_Clr.RESET}"
            print(f"     {rank:>2}. {r.filename:<30s} {icon}  {r.duration_s:>6.1f}s")

    print(f"\n{border}\n")

    _write_summary_files(results, passed, non_empty_f, empty_nb, svg_ok, svg_missing,
                         total_kb, avg_anim, elapsed, log_dir, verbose)


def _print_section_summary(st: Dict[str, int]) -> None:
    """Print a compact pass/fail tally for a completed narrative section."""
    p = st.get("pass", 0); f = st.get("fail", 0); e = st.get("empty", 0); m = st.get("missing_py", 0)
    parts = [f"{_Clr.GREEN}{p} pass{_Clr.RESET}"]
    if f: parts.append(f"{_Clr.RED}{f} fail{_Clr.RESET}")
    if e: parts.append(f"{_Clr.DIM}{e} empty{_Clr.RESET}")
    if m: parts.append(f"{_Clr.YELLOW}{m} missing{_Clr.RESET}")
    print(f"     {_Clr.DIM}{' | '.join(parts)}{_Clr.RESET}")


def _write_summary_files(
    results: List[NotebookResult],
    passed: List[NotebookResult],
    non_empty_f: List[NotebookResult],
    empty_nb: List[NotebookResult],
    svg_ok: List[NotebookResult],
    svg_missing: List[NotebookResult],
    total_kb: float,
    avg_anim: float,
    elapsed: float,
    log_dir: str,
    verbose: bool,
) -> None:
    """Write per-failure log files and the JSON summary."""
    os.makedirs(log_dir, exist_ok=True)
    failed = [r for r in results if not r.passed]

    for r in failed:
        if r.is_empty:
            continue
        logpath = os.path.join(log_dir, f"error_{r.filename}.log")
        with open(logpath, 'w', encoding='utf-8') as lf:
            lf.write(f"Category : {r.error_category}\n")
            lf.write(f"Summary  : {r.error_summary}\n\n")
            lf.write(f"--- STDERR ---\n{_Clr.strip(r.stderr_snippet)}\n")
            if r.stdout_snippet:
                lf.write(f"\n--- STDOUT (tail) ---\n{_Clr.strip(r.stdout_snippet)}\n")

    # \u2500\u2500 Write JSON summary \u2500\u2500
    summary_path = os.path.join(log_dir, "summary.json")
    summary = {
        "timestamp":       time.strftime('%Y-%m-%dT%H:%M:%S'),
        "total":           len(results),
        "passed":          len(passed),
        "failed":          len(non_empty_f),
        "empty":           len(empty_nb),
        "svg_xml_valid":   len(svg_ok),
        "svg_missing":     len(svg_missing),
        "total_svg_kb":    round(total_kb, 1),
        "avg_animate":     round(avg_anim, 1),
        "elapsed_s":       round(elapsed, 2),
        "narrative_order": [s for s, _ in NARRATIVE_ORDER],
        "results":         [asdict(r) for r in results],
    }
    with open(summary_path, 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2, ensure_ascii=False)
    print(f"  [>] Full report -> {summary_path}")
    if non_empty_f:
        print(f"  [>] Error logs  -> {log_dir}/error_<name>.log\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> Dict[str, Any]:
    args = sys.argv[1:]  # type: ignore
    opts: Dict[str, Any] = {
        "quick":       "--quick"       in args,
        "fix":         "--fix"         in args,
        "fix_svgs":    "--fix-svgs"    in args,
        "stub_svgs":   "--stub-svgs"   in args,
        "no_parallel": "--no-parallel" in args,
        "narrative":   "--narrative"   in args,
        "regen":       "--regen"       in args,
        "svg_only":    "--svg-only"    in args,
        "list":        "--list"        in args,
        "fail_fast":   "--fail-fast"   in args,
        "quiet":       "--quiet"       in args,
        "verbose":     "--verbose"     in args,
        "force_color": "--color"       in args,
        "no_color":    "--no-color"    in args,
        "filter":      None,
        "category":    None,
        "workers":     os.cpu_count() or 4,
        "timeout":     DEFAULT_TIMEOUT,
        "top_slow":    5,
    }
    for i, a in enumerate(args):
        if a == "--filter" and i + 1 < len(args):
            opts["filter"] = args[i + 1]
        if a == "--workers" and i + 1 < len(args):
            try:
                opts["workers"] = int(args[i + 1])
            except ValueError:
                pass
        if a == "--timeout" and i + 1 < len(args):
            try:
                opts["timeout"] = int(args[i + 1])
            except ValueError:
                pass
        if a == "--category" and i + 1 < len(args):
            opts["category"] = args[i + 1].lower()
        if a == "--top-slow" and i + 1 < len(args):
            try:
                opts["top_slow"] = int(args[i + 1])
            except ValueError:
                pass
    return opts


def main():
    opts     = parse_args()

    # Initialise colour support as early as possible
    _Clr.init(force_on=bool(opts["force_color"]), force_off=bool(opts["no_color"]))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir  = os.path.join(base_dir, LOG_DIR)
    svg_dir  = os.path.join(base_dir, SVG_DIR)

    all_notebooks = sorted(glob.glob(os.path.join(base_dir, "notebook_*.py")))
    if not all_notebooks:
        print("ERROR: No notebook_*.py files found.")
        sys.exit(1)

    # ── List mode: print narrative order and exit ─────────────────────────────
    if opts["list"]:
        print(f"\n  {_Clr.INFO}ECO-SIM{_Clr.RESET} - {len(all_notebooks)} notebooks discovered in {base_dir}")
        print_narrative_list(svg_dir)
        return

    # Apply --filter
    if opts["filter"]:
        pattern   = str(opts["filter"]).lower()
        notebooks = [n for n in all_notebooks if pattern in os.path.basename(n).lower()]
        print(f"\n  Filter '{pattern}' matched {len(notebooks)}/{len(all_notebooks)} notebooks.")
    elif opts["narrative"]:
        # Reorder to match NARRATIVE_ORDER, non-listed notebooks append at end
        listed    = [os.path.join(base_dir, f"{s}.py") for s, _ in NARRATIVE_ORDER
                     if os.path.isfile(os.path.join(base_dir, f"{s}.py"))]
        listed_set = set(listed)
        extra      = [n for n in all_notebooks if n not in listed_set]
        notebooks  = listed + extra
        print(f"\n  Narrative order: {len(listed)} listed + {len(extra)} extra notebooks.")
    else:
        notebooks = all_notebooks

    if not notebooks:
        print("  No notebooks matched. Exiting.")
        sys.exit(0)

    print(f"\n  {_Clr.BOLD}ECO-SIM Notebook Validation Suite{_Clr.RESET}")
    print(f"  Discovered {len(notebooks)} notebook(s) in {base_dir}")
    print(f"  Python interpreter: {_Clr.DIM}{_NOTEBOOK_PYTHON}{_Clr.RESET}")

    # -- Fix-SVGs mode: sanitize XML entities in all SVGs on disk -------------
    if opts["fix_svgs"]:
        print(f"  Mode: FIX-SVGs (sanitizing XML entities in {svg_dir})\n")
        fix_svg_entities(svg_dir)
        return

    # -- Stub-SVG mode: create placeholders for notebooks with missing SVGs ---
    if opts["stub_svgs"]:
        print("  Mode: STUB-SVGS (creating placeholder SVGs for missing outputs)\n")
        create_stub_svgs(notebooks, svg_dir)
        return

    # -- Fix mode: repair HTML entity corruption in .py source ----------------
    if opts["fix"]:
        print("  Mode: AUTO-FIX (repairing HTML entity corruption)\n")
        fixed: int = 0
        for nb in notebooks:
            name = os.path.basename(nb)
            if fix_html_entities(nb):
                print(f"    {_Clr.GREEN}\u2714{_Clr.RESET} Fixed {name}")
                fixed += 1
        print(f"\n  {fixed}/{len(notebooks)} file(s) repaired.")
        return

    # ── Quick (syntax only) ───────────────────────────────────────────────────
    if opts["quick"]:
        print("  Mode: QUICK (syntax + entity + empty check only)\n")
        errors: int = 0
        for nb in notebooks:
            name = os.path.basename(nb)
            if check_empty(nb):
                print(f"    {_Clr.DIM}\U0001f4e4 {name}: empty file{_Clr.RESET}")
                continue
            err = check_syntax(nb)
            if err:
                print(f"    {_Clr.FAIL}\u274c {name}: {err}{_Clr.RESET}")
                errors += 1
            else:
                corruption = check_html_entity_corruption(nb)
                flag = f"  {_Clr.YELLOW}\u26a0 entity corruption{_Clr.RESET}" if corruption else ""
                print(f"    {_Clr.GREEN}\u2705{_Clr.RESET} {name}{flag}")
        print(f"\n  {len(notebooks) - errors}/{len(notebooks)} passed syntax check.")
        return

    # ── SVG-only mode: audit without execution ───────────────────────────────
    if opts["svg_only"]:
        print("  Mode: SVG AUDIT (no execution)\n")
        t_start = time.perf_counter()
        results = run_svg_audit(notebooks, svg_dir)
        elapsed = time.perf_counter() - t_start
        prev = _load_previous_summary(log_dir)
        print_report(
            results, elapsed, log_dir,
            quiet=bool(opts["quiet"]), verbose=bool(opts["verbose"]),
            top_slow=int(opts["top_slow"]), category=opts["category"], prev=prev,
        )
        failures = [r for r in results if not r.passed]
        sys.exit(min(len(failures), 125))

    # ── Regen mode: skip notebooks whose SVG is already valid ────────────────
    if opts["regen"]:
        to_run  = []
        skipped = 0
        for nb in notebooks:
            info = check_svg_full(os.path.basename(nb), svg_dir)
            if info.exists and info.xml_valid and info.animate_count >= 5:
                skipped += 1
            else:
                to_run.append(nb)
        print(f"  Mode: REGEN - skipping {skipped} notebooks with valid SVG, "
              f"running {len(to_run)}.\n")
        notebooks = to_run
        if not notebooks:
            print("  All SVGs are valid. Nothing to regenerate.")
            return

    # Load previous summary for trend comparison (before we overwrite it)
    prev_summary = _load_previous_summary(log_dir)

    # ── Full execution ─────────────────────────────────────────────────────────
    workers    = 1 if opts["no_parallel"] else min(opts["workers"], len(notebooks))
    mode_label = "SERIAL" if workers == 1 else f"PARALLEL ({workers} workers)"
    print(f"  Mode: FULL EXECUTION - {_Clr.CYAN}{mode_label}{_Clr.RESET}\n")

    results: List[NotebookResult] = []
    progress = Progress(len(notebooks))
    t_start  = time.perf_counter()
    fail_fast = bool(opts["fail_fast"])
    aborted   = False

    if workers == 1:
        for nb in notebooks:
            name = os.path.basename(nb)
            r    = run_notebook(nb, timeout=int(opts["timeout"]))
            results.append(r)
            progress.increment(name, r.passed)
            if fail_fast and not r.passed and not r.is_empty:
                print(f"\n  {_Clr.FAIL}--fail-fast: stopping after first failure ({name}){_Clr.RESET}")
                aborted = True
                break
    else:
        future_map: Dict[Any, str] = {}
        stop_event = threading.Event()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for nb in notebooks:
                fut = pool.submit(run_notebook, nb, int(opts["timeout"]))
                future_map[fut] = os.path.basename(nb)
            for fut in as_completed(future_map):
                name = future_map[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    r = NotebookResult(
                        filename=name,
                        error_category="runtime",
                        error_summary=str(exc),
                    )
                results.append(r)
                progress.increment(name, r.passed)
                if fail_fast and not r.passed and not r.is_empty and not stop_event.is_set():
                    stop_event.set()
                    print(f"\n  {_Clr.FAIL}--fail-fast: stopping after first failure ({name}){_Clr.RESET}")
                    aborted = True
                    break

    elapsed = time.perf_counter() - t_start
    print_report(
        results, elapsed, log_dir,
        quiet=bool(opts["quiet"]), verbose=bool(opts["verbose"]),
        top_slow=int(opts["top_slow"]), category=opts["category"],
        prev=prev_summary,
    )

    if aborted:
        sys.exit(1)

    failures = [r for r in results if not r.passed and not r.is_empty]
    sys.exit(min(len(failures), 125))


# ── Prompt generation tests (run via: python -m pytest test_all_notebooks.py) ──
prompt_fn = generate_prompt
if pytest is not None and prompt_fn is not None:
    @pytest.fixture
    def prompt_output():
        return prompt_fn("notebook_01")

    class TestPromptGeneration:
        def test_canvas_dimensions_present(self, prompt_output):
            assert "1280×602" in prompt_output or "1280x602" in prompt_output

        def test_zone_coordinates_complete(self, prompt_output):
            required_zones = ["header", "main_content", "info_cards", "footer"]
            for zone in required_zones:
                assert zone.lower() in prompt_output.lower()

        def test_footer_attribution_convention(self, prompt_output):
            assert "footer" in prompt_output.lower()
            assert "570" in prompt_output

        def test_helper_references(self, prompt_output):
            required_helpers = ["save_svg", "CANVAS_HEIGHT", "sanitize_svg_text"]
            for helper in required_helpers:
                assert helper in prompt_output

    class TestDeterministicOutput:
        def test_same_input_yields_stable_output(self):
            output1 = prompt_fn("notebook_01")
            output2 = prompt_fn("notebook_01")
            assert output1 == output2

class TestSVGMigrationRegression:
    def test_no_height_600_references(self):
        for notebook in glob.glob("notebook_*.py"):
            with open(notebook, "r", encoding="utf-8") as f:
                content = f.read()
            assert "height: int = 600" not in content
            assert "height=600" not in content

    def test_viewbox_uses_602(self):
        for notebook in glob.glob("notebook_*.py"):
            with open(notebook, "r", encoding="utf-8") as f:
                content = f.read()
            # If there's a hardcoded viewbox, ensure it's not 600
            assert 'viewBox="0 0 1280 600"' not in content
            assert "viewBox='0 0 1280 600'" not in content


if __name__ == '__main__':
    main()

