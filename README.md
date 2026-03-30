# ECO-SIM — Recanto das Araras Extractive Reserve Eco-Simulator

> **PyTorch-powered ecosystem simulator for the Brazilian Cerrado.**  
> 80 species-interaction modules · animated SVG phenological dashboards · automated scientific validation.

---

## Overview

ECO-SIM is a computational ecology simulation suite built for the **Recanto das Araras Extractive Reserve (RESEX)**, located in Três de Cima, Goiás, Brazil, within the Bacia do Rio Lapa karst watershed.

Each of the 80+ notebooks simulates a distinct ecological relationship — predation, mutualism, pollination, seed dispersal, fire dynamics, seasonal migration, and underground symbioses — grounded in peer-reviewed Cerrado literature and the **PIGT RESEX Recanto das Araras 2024** reserve management plan.

The engine runs fully on **PyTorch tensors** (CPU or CUDA), produces animated **SVG dashboards** as output artefacts, and includes automated scientific auditing tools to validate biological plausibility across the entire suite.

---

## Features

- **GPU-accelerated particle/agent physics** — all positions, velocities, and forces computed as batched PyTorch tensors.
- **Phenological clocks** — month-by-month seasonal curves drive behaviour (rainfall, fruiting, fire risk, migration pulses, nuptial flights, etc.).
- **BIODIVERSITY_DB** — 30+ Cerrado species with calibrated parameters: speed, mass, diet, cohesion, aerodynamic drag, lifespan, and flight capability.
- **Animated SVG output** — every module renders a 1280 × 602 pixel dashboard with SMIL animations, gradient overlays, UI panels, mini-charts, and phenology wheels.
- **Scientific audit toolchain** — `scientific_validator.py` and `scientific_audit.py` perform AST-level extraction and cross-notebook consistency checks, producing `audit.json` and a human-readable `Scientific Validation Report.md`.
- **Extensible base** — `eco_base.py` provides `EcosystemBase` and `BaseConfig`, with hook methods (`extra_step`, `extra_forces`, `extra_svg`) for clean subclassing.

---

## Module Catalogue

### Part I — Behavioral Mechanics (01 – 23)
Foundational physics and ethology of the particle agents.

| Module | Topic |
|--------|-------|
| 01 | Baseline Ecosystem & Karst Landscape Dynamics |
| 02 | Diverse Flight Patterns (Aerodynamics & Turn Radius) |
| 03 | Diurnal Cycles (Background & Activity Modulation) |
| 04 | Water Dependence (Energy Recharge at Veredas) |
| 05 | Roosting Behavior (Nighttime Tree Flocking) |
| 06 | Predator Avoidance (Herbivores Flee Carnivores) |
| 07 | Flocking Cohesion (Dynamic Weights by Flock Size) |
| 08 | Foraging Efficiency (Speed Reduction in Resource Zones) |
| 09 | Territoriality (Hummingbirds Defending Nectar Patches) |
| 10 | Mating Displays (Spiral Flight Patterns) |
| 11 | Karst Obstacles (Limestone Outcrop Collision Avoidance) |
| 12 | Cave Entry / Exit (Sinkhole Teleportation) |
| 13 | Mastofauna & Flora — Tapir Browsing and Trail Making |
| 14 | Rain Events (Global Downward Force & Cover Seeking) |
| 15 | Seasonal Water (Dynamic Veredas Expansion & Contraction) |
| 16 | Fruiting Phenology (Resource Node Pulsing) |
| 17 | Seed Dispersal (Ornithochory) |
| 18 | Seed Germination (Sapling Growth) |
| 19 | Soil Diversity (Sand vs Clay Biomes) |
| 20 | Natural Fire (Propagation & Avoidance) |
| 21 | Reproduction (Energy & Spawning) |
| 22 | Mortality (Old Age & Starvation Physics) |
| 23 | Carrying Capacity (Population Control) |

### Part II — Ecological Pressures & Disturbances (24 – 36)

| Module | Topic |
|--------|-------|
| 24 | Insect Swarms (Temporary Prey Clouds & Insectivore Chase) |
| 25 | Scavenging (Vultures Circle Death Sites) |
| 26 | Nesting Competition (Limited Nest Nodes) |
| 27 | Keystone Removal (Flock Collapse & Ecological Void) |
| 28 | Satiation (Predators Ignore Prey After Successful Hunt) |
| 29 | Alarm Propagation (Flee Vector Wave) |
| 30 | Migration Flow (Stream Enters Left, Exits Right) |
| 31 | Road Fragmentation (Linear Barrier with High Fear Cost) |
| 32 | Cattle Grazing (Slow Brown Obstacles Deplete Resource Nodes) |
| 33 | Fencing (Hard Boundaries for Ground Feeders; Permeable for Flyers) |
| 34 | Selective Logging (Random Removal of Roosting Tree Nodes) |
| 35 | Noise Zones (Circular Repulsion — Tourist Vehicles) |
| 36 | Poaching Risk (High Mortality in Unprotected Zones) |

### Part III — Conservation & Management Interventions (37 – 54)

| Module | Topic |
|--------|-------|
| 37 | Agroforestry Rows (Linear Resource Nodes Attract Biodiversity) |
| 38 | Controlled Burns (Small Fires That Regenerate Resources Fast) |
| 39 | Corridor Planting (Nodes Connecting Separated Habitat Patches) |
| 40 | Ecotourism Trails (Tourist Agents on Fixed Paths, Birds Flush) |
| 41 | Invasive Grasses (Fast-Spreading Texture Chokes Native Saplings) |
| 42 | Restoration Weeding (Clear Invasive Texture in Management Zones) |
| 43 | Climate Warming (Increased Evaporation; Water Shrinks; Energy Demands Rise) |
| 44 | Extreme Drought (Water Nodes Vanish; Movement Cost Spikes) |
| 45 | Carbon Tracking (Biomass Bar / Dual-Axis Line Chart) |
| 46 | Community Monitoring (Camera Traps Highlight Birds in Zones) |
| 47 | Traditional Harvesting (Human Agents Collect Fruit Without Destroying Nodes) |
| 48 | Apiary Pollination (Bee Nodes Increase Fruit Regeneration Rate) |
| 49 | Jaguar Presence (Large Predator Causes Massive Avoidance Behavior) |
| 50 | Maned Wolf Dispersal (Wolf Agent Moves Far, Planting Lobeira Seeds) |
| 51 | Rapid Succession (Open Field → Shrub → Forest, Accelerated Time-Lapse) |
| 52 | Zoning Policy (Preservation, Sustainable Use, Buffer Zones) |
| 53 | Resilience Test (Fire + Drought Shock, Then Measure Recovery) |
| 54 | The Living Landscape (Grand Finale — All Modules Combined) |

### Part IV — Ecological Relationship Clocks (55 – 66)
Radial phenological clock visualizations of 12 interspecific interactions.

| Module | Interaction |
|--------|-------------|
| 55 | Tucano-toco ↔ Pequi (*Caryocar brasiliense*) — Ornithochory |
| 56 | Abelha-nativa ↔ Cerrado Wildflowers — Pollination |
| 57 | Lobo-guará ↔ Lobeira (*Solanum lycocarpum*) — Endozoochory |
| 58 | Morcego-nectarívoro ↔ Babassu Palm — Chiropterophily |
| 59 | AMF Fungi ↔ Cerrado Root Network — Mycorrhizal Seasonality |
| 60 | Arara-canindé ↔ Buriti Palm (*Mauritia flexuosa*) — Vereda Phenology |
| 61 | Anta ↔ Macaúba / Jatobá — Megaherbivore Endozoochory |
| 62 | Tamanduá-bandeira ↔ Termite Mounds — Myrmecophagy |
| 63 | Formiga-Cerrado ↔ Extrafloral Nectaries (EFN) — Mutualistic Defense |
| 64 | Sapo-cururu & Pererecas ↔ Ephemeral Ponds — Explosive Breeding |
| 65 | Beija-flor ↔ Canela-de-ema — Pyrophytic Pollination |
| 66 | Onça-pintada ↔ Queixadas — Apex Predator-Prey & Karst |

### Part V — Seasonal / Spatial Maps (67 – 70)

| Module | Interaction |
|--------|-------------|
| 67 | Orquídea (*Cyrtopodium*) ↔ Fungo Micorrízico — Symbiotic Survival Map |
| 68 | Pequizeiro ↔ Morcego-nectarívoro — Spatial Chiropterophily Map |
| 69 | Tamanduá-bandeira ↔ Cupinzeiros — Macropredator Regulation Clock |
| 70 | Jatobá (*Hymenaea courbaril*) ↔ Cutia — Seed Scatter-Hoarding |

### Part VI — Cerrado Ecological Web (71 – 74)

| Module | Interaction |
|--------|-------------|
| 71 | Seriema (*Cariama cristata*) ↔ Serpentes — Dry-Season Predation |
| 72 | Ipê-amarelo (*Handroanthus ochraceus*) ↔ Abelha-Mamangava (*Xylocopa* spp.) |
| 73 | Buriti ↔ Fungo-Ectomicorrízico (*Pisolithus* / *Scleroderma*) |
| 74 | Andorinha-rabo-branco ↔ Surto de Invertebrados do Cerrado |

### Part VII — Passage of Months Series (75 – 80)
Full-year seasonal subterranean and canopy dynamics.

| Module | Interaction |
|--------|-------------|
| 75 | Lobo-guará (*Chrysocyon brachyurus*) ↔ Lobeira (*Solanum lycocarpum*) |
| 76 | Gavião-real (*Harpia harpyja*) ↔ Macaco-prego (*Sapajus libidinosus*) |
| 77 | Mandacaru (*Cereus jamacaru*) ↔ Morcego-polinizador (*Glossophaga soricina*) |
| 78 | Cupim (*Nasutitermes* sp.) ↔ Fungo (*Termitomyces* sp.) — Subterranean Mutualism |
| 79 | Suiriri (*Tyrannus melancholicus*) ↔ Murici (*Byrsonima crassifolia*) |
| 80 | Saúva (*Atta* spp.) ↔ *Leucoagaricus* — Seasonal Fungus-Garden Symbiosis |

---

## Architecture

```
ECO-SIM/
├── eco_base.py              # Shared engine: EcosystemBase, BaseConfig, BIODIVERSITY_DB,
│                            #   save_svg, draw_phenology_chart, draw_migration_map
├── notebook_01.py           # Module 01
│   ...
├── notebook_80.py           # Module 80
├── scientific_validator.py  # AST-based data extractor for audit pipeline
├── scientific_audit.py      # Cross-notebook consistency & biology checks
├── audit.json               # Machine-readable audit results
├── Scientific Validation Report.md  # Human-readable audit report
├── test_all_notebooks.py    # Regression runner for all modules
└── svg_output/              # Generated SVG artefacts
```

### Core Classes

#### `BaseConfig` (dataclass)
| Field | Type | Description |
|-------|------|-------------|
| `frames` | `int` | Number of simulation frames |
| `fps` | `int` | Frames per second (sets SVG animation duration) |
| `width` / `height` | `int` | Canvas size (default 1280 × 602) |
| `device` | `str` | `"cpu"` or `"cuda"` |
| + domain fields | — | Per-module ecological parameters |

#### `EcosystemBase`
Core simulation loop with hook methods for subclasses:

| Hook | Called when |
|------|-------------|
| `_extra_init()` | Once, after parent `__init__` |
| `extra_step(fi, am, sm)` | Every frame — domain state update |
| `extra_forces(fi, am, sm)` | Every frame — return force tensor |
| `extra_svg()` | After base SVG is built |
| `extra_svg_overlay()` | Annotation / status panels |

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

# 2. Install dependencies
pip install torch numpy

# 3. Run a single module (SVG saved to svg_output/)
python notebook_01.py

# 4. Run the full test suite
python test_all_notebooks.py

# 5. Run the scientific audit
python scientific_audit.py
```

Output SVGs are written to `svg_output/` and are also archived on Google Drive:  
<https://drive.google.com/drive/folders/1lKH13QLIMp1O_-yAwTTFD_tbcKVJX-2A?usp=sharing>

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Tensor math, physics engine, optional CUDA acceleration |
| `numpy` | Supplemental array operations |
| `python ≥ 3.10` | Dataclass `field` defaults, `match` statements |

No external plotting libraries are required — all visualizations are pure SVG.

---

## Scientific Validation

`scientific_audit.py` performs automated checks on the entire codebase:

- **DB bounds** — species parameters within plausible biological ranges
- **Weight / lifespan rank ordering** — scaled relative values are consistent with real-world species comparisons
- **Seasonal frame mapping** — fire, migration, and nuptial-flight frames match Cerrado phenology (Aug–Sep fire peak, Sep–Oct first rains, etc.)
- **Cross-notebook curve consistency** — shared ecological curves (rainfall, fire risk, termite swarm, etc.) agree across modules within defined tolerances
- **Pearson correlation** — phenological curves that should co-vary (e.g., nuptial-flight pulse vs. rainfall) are checked for expected directionality

Results are written to `audit.json` and summarised in `Scientific Validation Report.md`.

---

## Study Area

**RESEX Recanto das Araras de Terra Ronca**  
Três de Cima, Goiás, Brazil  
Bacia do Rio Lapa · Karst of Terra Ronca State Park  
Management instrument: PIGT RESEX Recanto das Araras 2024

---

## Selected References

- Melo, C. et al. (2009). Frugivory and seed dispersal by birds in Cerrado remnants. *Revista Brasileira de Botânica* 32(4).
- Oliveira, P.E. & Moreira, A.G. (1992). Anemochory, zoochory and phenology in cerrado. *Biotropica*.
- Pivello, V.R. (2011). The use of fire in the Cerrado and Amazonian rainforests of Brazil. *Fire Ecology* 7(1).
- Medici, E.P. (2012). Lowland Tapir (*Tapirus terrestris*) conservation in fragmented landscapes. *IUCN*.
- Christianini, A.V. & Oliveira, P.S. (2010). Birds and ants provide complementary seed dispersal in cerrado. *Biotropica*.
- Sick, H. (1997). *Ornitologia Brasileira*. Nova Fronteira, Rio de Janeiro.
- Emmons, L.H. (1997). *Neotropical Rainforest Mammals*. University of Chicago Press.

---

## Conference Presentation

This simulator is being presented at the **International Conference on Geoinformatics and Data Analysis (ICGDA 2026)**:

| Field | Detail |
|-------|--------|
| **Abstract ID** | AE2020-A |
| **Title** | Integrated Environmental Monitoring and Geoanalysis System of the Araras Reserve |
| **Author** | Hélio Craveiro Pessoa Júnior |
| **Event** | ICGDA 2026 (hybrid — online and in-person) |
| **Conference website** | <http://www.icgda.org/> |

The abstract was accepted for presentation by the conference review committee. ECO-SIM's computational approach — PyTorch-powered species-interaction modules, animated SVG phenological dashboards, and automated scientific auditing — is presented as a case study in integrated environmental monitoring and geoanalysis applied to the Recanto das Araras Extractive Reserve in the Brazilian Cerrado.
