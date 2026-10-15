"""Read-only supplement for the protocol's BOTH-metrics seed predicate."""
from .neighbor_selection_gate_v11 import assess


def joint_assessment(physical, cf, spec, integrity):
    result=assess(physical,cf,spec,integrity)
    rows=[r for r in physical if r["study_phase"]=="confirm"]
    bykey={(r["scene_id"],r["seed"],r["policy"]):r for r in rows}
    counts={}
    for scene in spec["geometry_ids"]:
        count=0
        for seed in spec["confirmation_seeds"]:
            random=bykey.get((scene,seed,"random_peer"));greedy=bykey.get((scene,seed,"greedy"))
            if random and greedy and all(random.get(k) is not None and greedy.get(k) is not None and greedy[k]<random[k] for k in ("uav2_absolute_rmse_m","cluster_relative_rmse_m")):
                count+=1
        counts[scene]=count
    passed=all(n>=3 for n in counts.values())
    result["joint_positive_seeds_per_geometry"]=counts
    result["joint_positive_seed_rule_passed"]=passed
    if not passed:
        result["trajectory_passed"]=False
        if result["status"]=="pass_limited_scope":result["status"]="conditional_pass"
    result["audit_scope"]="same predeclared BOTH-metrics rule; no new threshold, selection, physical or filter updates"
    return result


def main():
    import argparse
    from ..experiments import neighbor_selection_validation_v11 as run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id",default="research_001")
    batch=parser.parse_args().batch_id
    spec=run.verify();base=run.batch_path(batch)
    manifests={p:run.checked_manifest(base/p/"manifest.json") for p in ("develop","confirm","counterfactual")}
    physical=manifests["develop"]["summaries"]+manifests["confirm"]["summaries"]
    valid=all(manifests[p]["safety"]["passed"] for p in ("develop","confirm")) and not manifests["counterfactual"]["failures"]
    result=joint_assessment(physical,manifests["counterfactual"]["results"],spec,valid)
    target=base/"evidence/joint_scientific_assessment.json"
    if target.exists():raise FileExistsError(str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    run.old.write_json(target,result)
    print(run.old.encode(result))


if __name__=="__main__":main()
