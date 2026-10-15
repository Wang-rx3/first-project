"""Pure covariance-encoding redundancy audit of already logged V12 states."""
from __future__ import annotations
import argparse
import csv
import math

from ..experiments import neighbor_selection_validation_v12 as run


def number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def audit(batch):
    run.verify()
    source = run.LOGS / batch / "report/conditional_points.csv"
    csv.field_size_limit(20000000)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for phase in ("develop", "confirm"):
        rr = [r for r in rows if r["study_phase"] == phase and r.get("label_complete", "").lower() == "true"]
        peer_alias, self_alias = [], []
        ages = []
        for r in rr:
            for peer in ("uav_00", "uav_01"):
                projected = number(r.get("minimal_directional_peer_variance_"+peer+"_m2"))
                trace = number(r.get("minimal_"+peer+"_position_trace_m2"))
                own_projected = number(r.get("minimal_directional_self_variance_"+peer+"_m2"))
                own_trace = number(r.get("minimal_self_position_trace_m2"))
                if projected is not None and trace is not None:
                    peer_alias.append(abs(projected-trace/3))
                if own_projected is not None and own_trace is not None:
                    self_alias.append(abs(own_projected-own_trace/3))
            age = number(r.get("feature_aoi_difference_s"))
            if age is not None:
                ages.append(age)
        output.append({"study_phase": phase, "valid_points": len(rr), "peer_projection_samples": len(peer_alias),
                       "max_abs_peer_projection_minus_trace_over3_m2": max(peer_alias) if peer_alias else None,
                       "max_abs_self_projection_minus_trace_over3_m2": max(self_alias) if self_alias else None,
                       "peer_AoI_difference_min_s": min(ages) if ages else None,
                       "peer_AoI_difference_max_s": max(ages) if ages else None,
                       "interpretation": "covariance encoding redundancy, not proof of statistical state sufficiency"})
    folder = run.validate_batch(batch) / "state_audit"
    folder.mkdir(exist_ok=False)
    run.old.write_json(folder / "state_information.json", {"input_sha256": run.old.file_hash(source),
                       "source_sha256": run.old.file_hash(__file__), "rows": output,
                       "physical_steps": 0, "filter_calls": 0, "optimizer_updates": 0})
    print(run.old.encode(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default="research_001")
    audit(parser.parse_args().batch_id)
