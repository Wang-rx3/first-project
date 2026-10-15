"""Pure quotas/path contracts; these tests never construct a simulation."""
from unittest.mock import patch

import pytest

from src.experiments import neighbor_selection_validation_v12 as run


@pytest.mark.parametrize("name", ("../escape", "a/b", "a\\b", "", "a"*65))
def test_batch_path_rejects_unsafe_names(name):
    with pytest.raises(ValueError):
        run.validate_batch(name)


def test_global_step_reservations_prevent_extra_physical_runs(tmp_path):
    spec = {"phase_caps": {"develop": {"physical": 3840}, "confirm": {"physical": 3840}},
            "physical_step_cap": 7680, "phase_wall_cap_s": 900}
    with patch.object(run, "BASE", tmp_path / "data"), patch.object(run, "LOGS", tmp_path / "logs"):
        run.start_phase("test", "develop", spec)
        run.start_phase("test", "confirm", spec)
        with pytest.raises(RuntimeError, match="global v12"):
            run.start_phase("another", "develop", spec)


def test_existing_phase_cannot_be_overwritten(tmp_path):
    spec = {"phase_caps": {"counterfactual": {"physical": 0}},
            "physical_step_cap": 7680, "phase_wall_cap_s": 900}
    with patch.object(run, "BASE", tmp_path / "data"), patch.object(run, "LOGS", tmp_path / "logs"):
        run.start_phase("test", "counterfactual", spec)
        with pytest.raises(FileExistsError):
            run.start_phase("test", "counterfactual", spec)
