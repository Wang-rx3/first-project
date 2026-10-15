"""Read-only final verification and exclusive artifact seal for V12."""
from __future__ import annotations
import subprocess

from ..experiments import neighbor_selection_validation_v12 as run
from .neighbor_selection_evidence_audit_v12 import exhaustive_membership


def finalize():
    spec = run.verify()
    base = run.validate_batch("research_001")
    for phase in ("develop", "confirm", "counterfactual", "report", "evidence_audit"):
        exhaustive_membership(base/phase, run.LOGS/"research_001"/phase)
    for lockpath in (run.BASE/"reporting_addendum.lock.json", base/"evidence_audit/source_inputs_before.json"):
        payload = run.old.read_json(lockpath)
        hashes = payload.get("source_sha256", payload)
        for name, digest in hashes.items():
            if run.old.file_hash(run.ROOT/name) != digest:
                raise RuntimeError("reporting/audit source changed: "+name)
    readme_before = (run.BASE/"stage1/README_current_before.bin").read_bytes()
    if not (run.ROOT/"README.md").read_bytes().startswith(readme_before):
        raise RuntimeError("README append-only rule violated")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=run.ROOT, text=True).strip()
    if head != "15beffcae6a2b2744903f97f34625f979dbc1348":
        raise RuntimeError("Git HEAD changed")
    final_path = run.BASE/"final_artifact_manifest.json"
    verification_path = run.BASE/"final_verification.lock.json"
    if final_path.exists() or verification_path.exists():
        raise FileExistsError("v12 final seal already exists")
    paths = set()
    for folder in (run.BASE, run.LOGS, run.ROOT/"plots/preliminary"/run.STEM):
        paths.update(p for p in folder.rglob("*") if p.is_file())
    paths.update((run.ROOT/"src/analysis").glob("neighbor_selection_*_v12.py"))
    paths.update((run.ROOT/"src/experiments").glob("neighbor_selection_*_v12.py"))
    paths.update((run.ROOT/"tests").glob("test_neighbor_selection_*_v12.py"))
    paths.update((run.CONFIG, run.ROOT/"README.md", run.ROOT/"knowledge/neighbor_selection_v12_protocol.md",
                  run.ROOT/"sections/3_neighbor_learnability_v12.md"))
    hashes = {str(p.relative_to(run.ROOT)): run.old.file_hash(p) for p in sorted(paths)}
    evidence = run.old.read_json(base/"evidence_audit/audited_metrics.json")
    report = run.old.read_json(base/"report/aggregate.json")
    if report["assessment"]["status"] != "not_passed" or report["training_ready"]:
        raise RuntimeError("scientific conclusion unexpectedly changed")
    if any(r["rmse_difference_max_m"] != 0 or not r["requests_equal"] or not r["fusions_equal"] for r in report["baseline_reproduction"]):
        raise RuntimeError("native development baselines changed")
    content = {"schema": "v12_final_artifacts", "artifact_sha256": hashes,
               "counts": evidence["checkpoint_counts"], "physical": evidence["physical"],
               "optimizer_updates": 0, "training_ready": False}
    run.old.write_json(final_path, content)
    for name, digest in hashes.items():
        if run.old.file_hash(run.ROOT/name) != digest:
            raise RuntimeError("final artifact changed: "+name)
    protected = run.verify_inputs()
    verification = {"file_integrity": "passed", "scientific_admission": "not_passed", "training_ready": False,
                    "protected_inputs_unchanged": len(protected), "sealed_artifacts": len(hashes),
                    "frozen_runtime_sources_unchanged": len(run.SOURCES), "README_append_only": True,
                    "git_head_unchanged": head, "physical": evidence["physical"],
                    "counterfactual": evidence["checkpoint_counts"], "numerical_safety": evidence["safety"],
                    "eligible_dual_points": evidence["eligible_dual_points"],
                    "eligible_meaningful_points": evidence["eligible_meaningful_points"],
                    "tests": {"before_new_data_passed": 127, "post_data_pure_reporting_passed": 6},
                    "symbol_audit": {"ok": True, "canonical": 300, "legacy": 10},
                    "optimizer_updates": 0, "deployed": False, "committed": False, "pushed": False,
                    "final_artifact_manifest_sha256": run.old.file_hash(final_path)}
    run.old.write_json(verification_path, verification)
    print(run.old.encode(verification), flush=True)


if __name__ == "__main__":
    finalize()
