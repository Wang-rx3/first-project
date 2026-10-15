"""Pure checks for the supplementary record schema and transport audit."""
from src.analysis.neighbor_selection_evidence_v13 import audit_record


def empty_record():
    return {"case": {"scene_id": "g", "seed": 5}, "policy": "none", "quality_offset": 0,
            "frames": [], "v13_quality_log": [], "v13_metadata_events": [], "v11_decisions": []}


def test_uses_native_nested_case_schema():
    result = audit_record(empty_record(), {"dt_s": .2})
    assert result["scene_id"] == "g" and result["seed"] == 5 and result["passed"]


def test_metadata_must_be_real_enqueued_packet_and_prestep_time():
    r = empty_record()
    r["v13_metadata_events"] = [{"event": "enqueue", "packet_id": ["uav_00", "uav_02", .4, 4, 1],
                                "payload_bytes": 496, "quality_header_bytes": 16, "packet_size_bytes": 512,
                                "gnss_report_time_s": .6, "send_time_s": .4}]
    result = audit_record(r, {"dt_s": .2})
    assert not result["passed"]
    assert "GNSS header/payload temporal mismatch" in result["failures"]


def test_visible_arrival_cannot_invent_packet():
    r = empty_record()
    r["v13_metadata_events"] = [{"event": "visible_arrival", "packet_id": [1], "valid_quality": False}]
    result = audit_record(r, {"dt_s": .2})
    assert not result["passed"]
    assert result["missing_quality_arrivals"] == 1
