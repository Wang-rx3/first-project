import pytest
from src.experiments import partial_gnss_revalidation_v9 as v9


def test_failed_frozen_gate_stops_before_reservation_environment_or_output(monkeypatch):
    fake = {"passed": False}
    monkeypatch.setattr(v9, "verify_design", lambda: {})
    monkeypatch.setattr(v9.v7.old, "read_json", lambda path: fake if str(path).endswith("gate.lock.json") else {"summaries": [], "artifact_sha256": {}})
    from src.analysis import partial_gnss_gate_v9 as rules
    monkeypatch.setattr(rules, "admission", lambda summaries, spec: fake)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not reserve, construct environment, or create output")
    for function in ("reserve_physical", "episode", "outputs"):
        monkeypatch.setattr(v9, function, forbidden)
    with pytest.raises(RuntimeError, match="candidate closed-loop count must remain zero"):
        v9.closed("closed_unit", "paired_unit")
