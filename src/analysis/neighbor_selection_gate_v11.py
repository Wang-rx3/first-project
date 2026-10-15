"""Pure predeclared neighbor-value assessments, with seeds as replicates."""
from __future__ import annotations
import numpy as np
from .partial_gnss_validation_gate_v10 import METRICS, pooled


def correlation(x, y, rank=False):
    if len(x) < 3 or len(x) != len(y):
        return None
    a, b = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    def ranks(v):
        result = np.zeros(len(v))
        for value in np.unique(v):
            ids = np.flatnonzero(v == value)
            result[ids] = np.count_nonzero(v < value) + (len(ids)+1)/2
        return result
    if rank:
        a, b = ranks(a), ranks(b)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ranking_summary(rows):
    rr = [r for r in rows if r.get("dual_eligible") and r.get("score_difference") is not None]
    meaningful = [r for r in rr if r.get("meaningful")]
    x, y = [r["score_difference"] for r in meaningful], [r["utility_difference"] for r in meaningful]
    return {"dual_points": len(rr), "meaningful_points": len(meaningful),
            "ranking_accuracy": float(np.mean([r["ranking_correct"] for r in meaningful])) if meaningful else None,
            "pearson": correlation(x,y), "spearman": correlation(x,y,True),
            "preferred_peers": sorted({r["preferred_peer"] for r in meaningful}),
            "geometries": sorted({r["scene_id"] for r in meaningful})}


def assess(physical, cf, spec, integrity):
    confirmation = [r for r in physical if r["study_phase"] == "confirm"]
    bykey = {(r["scene_id"],r["seed"],r["policy"]):r for r in confirmation}
    failures, pergeometry, changes = [], {}, []
    cost_match = True
    for scene in spec["geometry_ids"]:
        rr = [r for r in confirmation if r["scene_id"]==scene and r["policy"]=="random_peer"]
        gg = [r for r in confirmation if r["scene_id"]==scene and r["policy"]=="greedy"]
        pergeometry[scene] = {}
        for metric in METRICS:
            baseline,after = pooled(rr,metric),pooled(gg,metric)
            gain = 1-after/baseline if baseline is not None and baseline>1e-12 and after is not None else None
            positive = 0
            for seed in spec["confirmation_seeds"]:
                r,g = bykey.get((scene,seed,"random_peer")),bykey.get((scene,seed,"greedy"))
                casegain = 1-g[metric]/r[metric] if r and g and r.get(metric) is not None and g.get(metric) is not None and r[metric]>1e-12 else None
                positive += int(casegain is not None and casegain>0)
                if casegain is None or casegain < -.05:
                    failures.append(f"case degradation {scene}:{seed}:{metric}")
                changes.append({"scene_id":scene,"seed":seed,"metric":metric,"improvement_fraction":casegain})
                if not r or not g or r.get("request_windows") != g.get("request_windows") or r.get("uwb_requested") != g.get("uwb_requested"):
                    cost_match = False
            pergeometry[scene][metric] = {"random":baseline,"greedy":after,"improvement_fraction":gain,"positive_seeds":positive}
            if gain is None or gain < .05 or positive < 3 or len(rr)!=4 or len(gg)!=4:
                failures.append("trajectory benefit "+scene+":"+metric)
    trajectory = not failures
    overall = ranking_summary(cf)
    held = [r for r in cf if r.get("study_phase")=="confirm"]
    confirm = ranking_summary(held)
    seeds = {str(seed):ranking_summary([r for r in held if r.get("seed")==seed]) for seed in spec["confirmation_seeds"]}
    represented = all(any(r.get("dual_eligible") and r.get("seed")==seed and r.get("scene_id")==scene for r in held) for seed in spec["confirmation_seeds"] for scene in spec["geometry_ids"])
    coverage = overall["dual_points"]>=48 and confirm["dual_points"]>=24 and represented
    differences = overall["meaningful_points"]>=36 and confirm["meaningful_points"]>=18 and len(confirm["preferred_peers"])==2 and len(confirm["geometries"])==2
    positive_seeds = sum((s["ranking_accuracy"] is not None and s["ranking_accuracy"]>.5 and s["spearman"] is not None and s["spearman"]>0) for s in seeds.values())
    ranking = confirm["ranking_accuracy"] is not None and confirm["ranking_accuracy"]>=.65 and confirm["spearman"] is not None and confirm["spearman"]>=.2 and positive_seeds>=3
    if not integrity or not coverage or not differences:
        status="not_passed"
    elif trajectory and ranking and cost_match:
        status="pass_limited_scope"
    else:
        status="conditional_pass"
    return {"status":status,"integrity_passed":integrity,"coverage_passed":coverage,
            "meaningful_difference_passed":differences,"trajectory_passed":trajectory,
            "ranking_passed":ranking,"cost_matched":cost_match,"trajectory_failures":sorted(set(failures)),
            "pergeometry":pergeometry,"paired_changes":changes,"counterfactual_all":overall,
            "counterfactual_confirmation":confirm,"counterfactual_seed_pools":seeds,
            "positive_rank_seed_pools":positive_seeds,"training_ready":False,
            "scope":"four independent confirmation seeds; windows/geometries are repeated blocks; no optimization"}
