from pathlib import Path

import pytest

from src.agents.orchestration.parameter_impact import (
    AspenSandboxRunner,
    FakeAspenSandboxRunner,
    ParameterImpactPlan,
    SensitivityAnalyzer,
    design_perturbations,
    make_simulation_plan,
    validate_parameter_impact_plan,
)


PATH = r"\Data\Blocks\X\Input\RR"


def plan(**kw):
    data = dict(name="RR", aspen_path=PATH, baseline=2.0, lower=1.0, upper=3.0, unit="-")
    data.update(kw)
    return ParameterImpactPlan("X", "S", "C", "mole_fraction", [data], max_cases=5, expected_outputs=["m"])


def test_plan_checks_bounds_and_integer_neighbor():
    assert validate_parameter_impact_plan(plan()) == []
    integer = plan(baseline=3, lower=2, upper=4, integer=True)
    values = [x["inputs"][PATH] for x in design_perturbations(integer)]
    assert set(values) >= {2, 3, 4}
    assert validate_parameter_impact_plan(plan(lower=3.0))


def test_unapproved_runner_does_not_execute(tmp_path: Path):
    model = tmp_path / "model.bkp"
    model.write_bytes(b"original")
    p = make_simulation_plan(plan(), base_model=str(model))
    with pytest.raises(PermissionError):
        FakeAspenSandboxRunner(lambda inputs: {"outputs": {"m": 1.0}}).run(p)
    assert model.read_bytes() == b"original"


def test_fake_results_are_analyzable_and_single_point_is_limited(tmp_path: Path):
    model = tmp_path / "model.bkp"
    model.write_bytes(b"original")
    p = make_simulation_plan(plan(), base_model=str(model)); p.approved = True
    runner = FakeAspenSandboxRunner(lambda inputs: {"outputs": {"m": inputs[PATH]}})
    results = runner.run(p)
    report = SensitivityAnalyzer().analyze(results, metric="m", parameters=[PATH])
    assert report["parameter_rank"][0]["pearson"] == pytest.approx(1.0)
    assert all(x.result_source == "fake_aspen" for x in results)
    assert model.read_bytes() == b"original"

