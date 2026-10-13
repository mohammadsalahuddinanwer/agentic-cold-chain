"""
ga_seed_patch.py — make the A2 genetic algorithm reproducible.


WHY: run_a2_experiment.evolve_weights() seeds a LOCAL numpy generator, which
only controls the initial population. DEAP's cxBlend, mutGaussian and
selTournament all draw from Python's GLOBAL random module, which is never
seeded — so every run evolves differently and Sect. 5.6 cannot be reproduced.
This adds one seeding call so the whole search is deterministic.
"""
from pathlib import Path

p = Path("run_a2_experiment.py")
src = p.read_text()

if "random.seed(seed)" in src:
    print("already patched — nothing to do"); raise SystemExit

# 1. make sure `random` is imported
if "\nimport random\n" not in src:
    src = src.replace("import json\n", "import json\nimport random\n", 1)
    print("  added: import random")

# 2. seed Python's global RNG at the top of evolve_weights()
OLD = '    rng = np.random.default_rng(seed)\n    tb = base.Toolbox()'
NEW = ('    # Determinism: the local generator below only drives the initial\n'
       '    # population. DEAP\'s cxBlend, mutGaussian and selTournament all draw\n'
       '    # from Python\'s global RNG, so seed that too or the search is random.\n'
       '    random.seed(seed)\n'
       '    rng = np.random.default_rng(seed)\n'
       '    tb = base.Toolbox()')

if OLD not in src:
    print("ERROR: evolve_weights() body not as expected — NOT patched")
    raise SystemExit(1)

src = src.replace(OLD, NEW, 1)
p.write_text(src)
print("  added: random.seed(seed) in evolve_weights()")
print("\npatched run_a2_experiment.py — A2 is now reproducible.")
print("Next: re-run A2 twice and confirm you get identical numbers both times.")
