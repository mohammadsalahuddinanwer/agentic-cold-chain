"""
run_a2_experiment.py — does evolutionary weight optimisation improve decisions?
==============================================================================
Answers ablation condition A2, which earlier versions could not isolate because
the cost weights never influenced any decision.

DESIGN
---------------------------------------------
1. WEIGHTS MAKE A DECISION. For the DIAGNOSED anomaly type the Resolver scores
   every candidate action on four normalised criteria (expected spoilage, money,
   time, carbon) and picks the lowest weighted sum, exactly as
   resolver_agent.py does (`min(evaluated, key=...)`). Different weights
   therefore choose different actions.

2. THE OUTCOME IS JUDGED INDEPENDENTLY. The realised outcome is measured in
   physical units (fraction of cargo spoiled, money spent, minutes of delay,
   kg CO2) using the TRUE anomaly type, and converted to a single figure with an
   OPERATOR price vector that is separate from the weights under test. Results
   are also reported over a grid of operator prices, so no single choice of
   prices decides the winner.

3. THREE CONDITIONS, PAIRED AND SEED-SEPARATED
     ga         : weights evolved with DEAP on VALIDATION seeds (2000+) only
     researcher : the thesis weights (0.40, 0.30, 0.20, 0.10)
     random     : 20 Dirichlet draws — the null hypothesis
   All are tested ONCE on the test seeds (1000+), on identical cases with
   identical random draws.

MODELLING ASSUMPTION
---------------------------------------------
The action appropriate to the true anomaly type contains the excursion once it
completes; any other action leaves the excursion running until a human
intervenes (HUMAN_INTERVENTION_MIN). Cargo is fully lost SPOIL_FULL_MIN minutes
after the breach. Without an assumption of this kind, all actions are
interchangeable and no weighting can matter.


"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from deap import algorithms, base, creator, tools

from run_ablation_study import (BASE_SEED, Config, DEFAULT_POLICY, DIAG_ACCURACY,
                                apply_policy, build_cases, ci95, detect, load_monitor,
                                load_training_scaling, policy_tag)

VAL_BASE_SEED = 2000

# ── Physical properties of each action (from resolver_agent.py action_templates)
# `time` is taken from resolver_agent.py, where the plan reports it as
# `estimated_time_seconds`; --action_time_unit minutes reads the same numbers as
# minutes instead (sensitivity check).
ACTION_PHYSICAL = {
    "notify_driver":   {"money": 5,   "time": 10,  "carbon_kg": 0},
    "dispatch_backup": {"money": 50,  "time": 120, "carbon_kg": 15},
    "repair_on_site":  {"money": 30,  "time": 60,  "carbon_kg": 2},
    "reroute_fastest": {"money": 20,  "time": 40,  "carbon_kg": 5},
    "reroute_safest":  {"money": 15,  "time": 70,  "carbon_kg": 3},
}
TIME_DIVISOR = 60.0        # seconds -> minutes (set to 1.0 if the table is minutes)


def action_minutes(action):
    return ACTION_PHYSICAL[action]["time"] / TIME_DIVISOR
CANDIDATES = list(ACTION_PHYSICAL)

# Action that actually contains each failure mode (resolver_agent.py ACTION_MAP).
# gps_blackout has no thermal effect, so no action is needed to contain it.
APPROPRIATE = {
    "compressor_failure": "dispatch_backup", "door_left_open": "notify_driver",
    "power_outage": "dispatch_backup",       "driver_error": "notify_driver",
    "extreme_weather": "reroute_safest",     "gps_blackout": None,
}
TYPES = sorted(APPROPRIATE)

# ── Outcome assumptions
HUMAN_INTERVENTION_MIN = 180.0   # a wrong action is corrected by a human after this
SPOIL_FULL_MIN         = 60.0    # cargo fully lost this long after the breach
                                 # (both overridable from the command line)
SPOIL_EST_EFFECTIVE    = 0.20    # what the Resolver EXPECTS if the action fits
SPOIL_EST_INEFFECTIVE  = 0.80    # ...and if it does not

# ── Operator prices: how the outcome is valued (independent of the weights)
DEFAULT_PRICES = {"cargo_gbp": 10000.0, "money_gbp": 100.0,
                  "delay_gbp_per_min": 20.0, "carbon_gbp_per_kg": 0.05}
PRICE_GRID = [{"cargo_gbp": c, "money_gbp": 100.0, "delay_gbp_per_min": d,
               "carbon_gbp_per_kg": k}
              for c in (2000.0, 10000.0, 50000.0)
              for d in (5.0, 20.0)
              for k in (0.05, 1.00)]

RESEARCHER_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10])


# ════════════════════════════════════════════════════════════════════════
# DECISION: the weights choose an action for the DIAGNOSED type
# ════════════════════════════════════════════════════════════════════════
def choose_action(weights, diagnosed_type):
    money = np.array([ACTION_PHYSICAL[a]["money"] for a in CANDIDATES], float)
    time_ = np.array([action_minutes(a) for a in CANDIDATES], float)
    carb  = np.array([ACTION_PHYSICAL[a]["carbon_kg"] for a in CANDIDATES], float)
    fits  = APPROPRIATE.get(diagnosed_type)
    spoil = np.array([SPOIL_EST_EFFECTIVE if a == fits else SPOIL_EST_INEFFECTIVE
                      for a in CANDIDATES])
    norm = lambda v: v / v.max() if v.max() > 0 else v
    score = (weights[0] * spoil + weights[1] * norm(money)
             + weights[2] * norm(time_) + weights[3] * norm(carb))
    return CANDIDATES[int(np.argmin(score))]


# ════════════════════════════════════════════════════════════════════════
# OUTCOME: judged on the TRUE type, in physical units
# ════════════════════════════════════════════════════════════════════════
def realised_outcome(case, alarm, action):
    """Returns (spoiled_fraction, money, delay_min, carbon_kg)."""
    if case["breach"] is None:                       # no thermal risk
        spoil = 0.0
    else:
        if alarm is None:                            # never detected
            contained = case["breach"] + HUMAN_INTERVENTION_MIN
        else:
            fits = APPROPRIATE.get(case["anomaly_type"])
            contained = (alarm + action_minutes(action) if action == fits
                         else alarm + HUMAN_INTERVENTION_MIN)
        spoil = float(np.clip((contained - case["breach"]) / SPOIL_FULL_MIN, 0.0, 1.0))
    if alarm is None:
        return spoil, 0.0, 0.0, 0.0
    p = ACTION_PHYSICAL[action]
    return spoil, float(p["money"]), action_minutes(action), float(p["carbon_kg"])


def price(outcome, prices):
    spoil, money, delay, carbon = outcome
    return (prices["cargo_gbp"] * spoil + prices["money_gbp"] * money
            + prices["delay_gbp_per_min"] * delay + prices["carbon_gbp_per_kg"] * carbon)


def diagnosed_type(case, accuracy):
    """Same paired draw as the ablation: correct with probability `accuracy`,
    otherwise a deterministic wrong type."""
    if not case["has_anomaly"]:
        return None
    true = case["anomaly_type"]
    if case["u"] < accuracy:
        return true
    i = TYPES.index(true) if true in TYPES else 0
    return TYPES[(i + 1 + int(case["u"] * 5)) % len(TYPES)]


def run_weights(cases, weights, prices, accuracy):
    """Applies one weight vector to every case; returns per-case rows."""
    rows = []
    for c in cases:
        alarm = c["mon_alarm"]
        if not c["has_anomaly"]:                     # false alarms: same for all weights
            action = "notify_driver" if alarm is not None else None
        else:
            action = choose_action(weights, diagnosed_type(c, accuracy)) if alarm is not None else None
        out = realised_outcome(c, alarm, action or "notify_driver") if alarm is not None \
            else realised_outcome(c, None, "notify_driver")
        rows.append(dict(case_id=c["case_id"], has_anomaly=c["has_anomaly"],
                         anomaly_type=c["anomaly_type"], action=action,
                         spoiled=out[0], gbp=price(out, prices)))
    return rows


def mean_cost(cases, weights, prices, accuracy):
    return float(np.mean([r["gbp"] for r in run_weights(cases, weights, prices, accuracy)]))


# ════════════════════════════════════════════════════════════════════════
# GA: evolve the weights on VALIDATION cases only
# ════════════════════════════════════════════════════════════════════════
def evolve_weights(val_cases, accuracy, generations, population, seed=42, prices=None):
    """DEAP GA, same operators as resolver_agent/evolutionary_opt.py, but the
    fitness is the realised operator cost of the decisions the weights produce."""
    if "A2FitnessMin" not in dir(creator):
        creator.create("A2FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("A2Individual", list, fitness=creator.A2FitnessMin)
    # Determinism: the local generator below only drives the initial
    # population. DEAP's cxBlend, mutGaussian and selTournament all draw
    # from Python's global RNG, so seed that too or the search is random.
    random.seed(seed)
    rng = np.random.default_rng(seed)
    tb = base.Toolbox()
    tb.register("attr", lambda: float(rng.uniform(0.05, 1.0)))
    tb.register("individual", tools.initRepeat, creator.A2Individual, tb.attr, n=4)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    tb.register("select", tools.selTournament, tournsize=3)

    def fitness(ind):
        w = np.clip(np.array(ind, float), 0.001, None)
        w = w / w.sum()
        return (mean_cost(val_cases, w, prices or DEFAULT_PRICES, accuracy),)

    tb.register("evaluate", fitness)
    pop = tb.population(n=population)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, tb, cxpb=0.8, mutpb=0.1, ngen=generations,
                        halloffame=hof, verbose=False)
    w = np.clip(np.array(hof[0], float), 0.001, None)
    return w / w.sum()


# ════════════════════════════════════════════════════════════════════════
def main():
    global SPOIL_FULL_MIN, HUMAN_INTERVENTION_MIN, TIME_DIVISOR
    ap = argparse.ArgumentParser(description="A2: value of evolutionary weight optimisation")
    ap.add_argument("--seeds", type=int, default=10, help="test seeds (1000+)")
    ap.add_argument("--val_seeds", type=int, default=5, help="validation seeds (2000+)")
    ap.add_argument("--n_anomalous", type=int, default=100)
    ap.add_argument("--n_normal", type=int, default=100)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=60)
    ap.add_argument("--n_random", type=int, default=20, help="random weight vectors")
    ap.add_argument("--policy_file", type=str, default=None)
    ap.add_argument("--spoil_full_min", type=float, default=SPOIL_FULL_MIN,
                    help="minutes after the breach at which cargo is a total loss")
    ap.add_argument("--human_min", type=float, default=HUMAN_INTERVENTION_MIN,
                    help="minutes before a human corrects an unsuitable action")
    ap.add_argument("--action_time_unit", choices=["seconds", "minutes"], default="seconds",
                    help="unit of the `time` column in resolver_agent.py action_templates")
    ap.add_argument("--evolve_per_cargo", action="store_true",
                    help="evolve separate weights for each cargo value, to test whether "
                         "the GA adapts to the operator's economics")
    args = ap.parse_args()
    SPOIL_FULL_MIN, HUMAN_INTERVENTION_MIN = args.spoil_full_min, args.human_min
    TIME_DIVISOR = 60.0 if args.action_time_unit == "seconds" else 1.0

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ws = Config.MONITOR_WINDOW_SIZE
    policy = (json.loads(Path(args.policy_file).read_text()) if args.policy_file
              else DEFAULT_POLICY)
    accuracy = DIAG_ACCURACY["rag"]

    print("=" * 74)
    print("A2 EXPERIMENT: do evolved weights make better decisions?")
    print("=" * 74)
    scaler, y_mean, y_std, n_feat = load_training_scaling()
    model = load_monitor(n_feat, device)
    print(f"Monitor alarm policy: {policy_tag(policy)}, device={device}")

    def cases_for(seed):
        return apply_policy(detect(build_cases(seed, args.n_anomalous, args.n_normal, ws),
                                   model, scaler, ws, y_mean, y_std, device), policy)

    print(f"\nBuilding validation cases (seeds {VAL_BASE_SEED}-{VAL_BASE_SEED+args.val_seeds-1})...")
    val_cases = [c for s in range(VAL_BASE_SEED, VAL_BASE_SEED + args.val_seeds)
                 for c in cases_for(s)]
    print(f"  {len(val_cases)} cases")

    print(f"Evolving weights (population {args.population}, {args.generations} generations)...")
    ga_w = evolve_weights(val_cases, accuracy, args.generations, args.population)
    print(f"  GA weights (spoilage, money, time, carbon): "
          f"[{', '.join(f'{x:.3f}' for x in ga_w)}]")
    print(f"  researcher weights:                          "
          f"[{', '.join(f'{x:.3f}' for x in RESEARCHER_WEIGHTS)}]")
    (out / "a2_weights.json").write_text(json.dumps(
        {"ga": ga_w.tolist(), "researcher": RESEARCHER_WEIGHTS.tolist()}, indent=2))

    rng = np.random.default_rng(7)
    random_ws = rng.dirichlet(np.ones(4), size=args.n_random)

    print(f"\nTesting once on seeds {BASE_SEED}-{BASE_SEED+args.seeds-1}...")
    test_cases = {s_: cases_for(s_) for s_ in range(BASE_SEED, BASE_SEED + args.seeds)}
    rows, per_seed = [], []
    for seed, cases in test_cases.items():
        conds = {"researcher": [RESEARCHER_WEIGHTS], "ga": [ga_w],
                 "random": list(random_ws)}
        for name, wlist in conds.items():
            costs, spoils, actions = [], [], []
            for w in wlist:
                r = run_weights(cases, w, DEFAULT_PRICES, accuracy)
                costs.append(np.mean([x["gbp"] for x in r]))
                spoils.append(np.mean([x["spoiled"] for x in r if x["has_anomaly"]]))
                actions += [x["action"] for x in r if x["has_anomaly"] and x["action"]]
            per_seed.append(dict(seed=seed, condition=name, gbp=float(np.mean(costs)),
                                 spoiled=float(np.mean(spoils))))
            rows += [dict(seed=seed, condition=name, action=a) for a in actions]

    df = pd.DataFrame(per_seed)
    df.to_csv(out / "a2_per_seed.csv", index=False)
    base_ = df[df.condition == "researcher"].set_index("seed")["gbp"]
    print(f"\nRESULTS at default prices (cargo GBP {DEFAULT_PRICES['cargo_gbp']:,.0f}, "
          f"delay GBP {DEFAULT_PRICES['delay_gbp_per_min']:.0f}/min)")
    print(f"{'condition':<14}{'GBP per case':>26}{'vs researcher':>26}{'spoiled':>10}")
    print("-" * 76)
    summary = []
    for name in ["researcher", "ga", "random"]:
        s = df[df.condition == name]
        m, lo, hi = ci95(s.gbp)
        d = ci95([100 * (g - base_[sd]) / base_[sd] for g, sd in zip(s.gbp, s.seed)])
        rel = "baseline" if name == "researcher" else f"{d[0]:+6.1f}% [{d[1]:+.1f}, {d[2]:+.1f}]"
        print(f"{name:<14}{m:10.0f} [{lo:8.0f}, {hi:8.0f}]{rel:>26}{s.spoiled.mean():10.3f}")
        summary.append(dict(condition=name, gbp=m, ci_low=lo, ci_high=hi,
                            pct_vs_researcher=d[0], pct_ci_low=d[1], pct_ci_high=d[2],
                            spoiled=s.spoiled.mean()))
    pd.DataFrame(summary).to_csv(out / "a2_summary.csv", index=False)

    relevant = []
    for cases in test_cases.values():
        for c in cases:
            if not c["has_anomaly"] or c["mon_alarm"] is None:
                continue
            spoils = {realised_outcome(c, c["mon_alarm"], a)[0] for a in CANDIDATES}
            relevant.append(len(spoils) > 1)
    print(f"\nDECISION RELEVANCE: the action choice changes the outcome in "
          f"{np.mean(relevant):.1%} of alarmed anomaly events")
    if np.mean(relevant) < 0.10:
        print("  Almost no case is decision-relevant: with these action times and")
        print("  spoilage assumptions no weighting could matter. Re-run with, e.g.,")
        print("  --spoil_full_min 180 before concluding anything about the weights.")

    acts = pd.DataFrame(rows)
    print("\nACTIONS CHOSEN ON ANOMALY EVENTS (share)")
    tbl = acts.groupby(["condition", "action"]).size().unstack(fill_value=0)
    print((tbl.T / tbl.sum(axis=1)).T.round(3).to_string())
    if "ga" in tbl.index and "researcher" in tbl.index and tbl.loc["ga"].equals(tbl.loc["researcher"]):
        print("\nNOTE: evolved and researcher weights chose the SAME actions everywhere,")
        print("      so the weighting has no measurable effect in this action set.")

    print("\nROBUSTNESS: GA vs researcher across operator price vectors")
    print(f"  {'cargo':>8}{'delay/min':>11}{'carbon/kg':>11}{'GA vs researcher':>20}")
    grid = []
    for pr in PRICE_GRID:
        gcost = np.mean([mean_cost(cs, ga_w, pr, accuracy) for cs in test_cases.values()])
        rcost = np.mean([mean_cost(cs, RESEARCHER_WEIGHTS, pr, accuracy)
                         for cs in test_cases.values()])
        pct = 100 * (gcost - rcost) / rcost
        grid.append(dict(**pr, ga_gbp=gcost, researcher_gbp=rcost, pct=pct))
        print(f"  {pr['cargo_gbp']:>8,.0f}{pr['delay_gbp_per_min']:>11.0f}"
              f"{pr['carbon_gbp_per_kg']:>11.2f}{pct:>19.1f}%")
    pd.DataFrame(grid).to_csv(out / "a2_price_grid.csv", index=False)

    if args.evolve_per_cargo:
        print("\nPER-CARGO EVOLUTION: weights evolved separately for each cargo value")
        print(f"  {'cargo':>8}{'GA weights (spoil, money, time, carbon)':>42}"
              f"{'GA vs researcher':>20}")
        per_cargo = []
        for cargo in (2000.0, 10000.0, 50000.0):
            pr = dict(DEFAULT_PRICES, cargo_gbp=cargo)
            w = evolve_weights(val_cases, accuracy, args.generations, args.population,
                               seed=42, prices=pr)
            g = np.mean([mean_cost(cs, w, pr, accuracy) for cs in test_cases.values()])
            r = np.mean([mean_cost(cs, RESEARCHER_WEIGHTS, pr, accuracy)
                         for cs in test_cases.values()])
            pct = 100 * (g - r) / r
            per_cargo.append(dict(cargo_gbp=cargo, weights=w.tolist(), ga_gbp=g,
                                  researcher_gbp=r, pct=pct))
            print(f"  {cargo:>8,.0f}{'[' + ', '.join(f'{x:.3f}' for x in w) + ']':>42}"
                  f"{pct:>19.1f}%")
        pd.DataFrame(per_cargo).to_csv(out / "a2_per_cargo.csv", index=False)

    print("\nASSUMPTIONS (report with the results)")
    print("  the action appropriate to the TRUE type contains the excursion once it")
    print(f"  completes; any other action is corrected by a human after "
          f"{HUMAN_INTERVENTION_MIN:.0f} min")
    print(f"  cargo fully lost {SPOIL_FULL_MIN:.0f} min after the breach")
    print(f"  action times read as {args.action_time_unit} "
          f"(dispatch_backup = {action_minutes('dispatch_backup'):.1f} min)")
    print("  weights are selected on validation seeds and tested once on the test seeds")
    print("\nSaved: outputs/a2_weights.json, a2_per_seed.csv, a2_summary.csv, a2_price_grid.csv")


if __name__ == "__main__":
    main()
