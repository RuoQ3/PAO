import json

from src.agents.tools import analyze_closed_loop_tool
from src.agents.tools.analyze_closed_loop import _impl_analyze_closed_loop
from src.database.simulation_db import SimulationDB
from src.models.process_case import (
    CaseStatus,
    ConstraintValue,
    ObjectiveValue,
    ProcessCase,
)
from src.models.simulation_result import RunStatus, SimulationResult


def _case(index: int, x: float) -> ProcessCase:
    sim = SimulationResult(
        status=RunStatus.SUCCESS,
        success=True,
        requested_inputs={"x": x},
        actual_inputs={"x": x},
    )
    return ProcessCase(
        case_id=f"case-{index}",
        iteration=index,
        status=CaseStatus.SUCCESS if x <= 0.8 else CaseStatus.INFEASIBLE,
        design_vars={"x": x},
        sim_result=sim,
        objectives=[
            ObjectiveValue("cost", 1.0 + x),
            ObjectiveValue("energy", 2.0 - x),
        ],
        constraints=[ConstraintValue("quality", x - 0.8)],
    )


def test_analyze_closed_loop_tool_returns_json_report(tmp_path):
    db_path = tmp_path / "simulation.db"
    with SimulationDB(db_path) as db:
        db.save_cases([_case(i, x).to_dict() for i, x in enumerate((0.4, 0.5, 0.9))])

    report_text = _impl_analyze_closed_loop(
        str(db_path), "cost,energy", sensitivity_method="spearman"
    )
    report = json.loads(report_text)
    assert report["data_quality"]["n_total"] == 3
    assert report["pareto"]["front_size"] >= 1
    assert report["constraints"]["quality"]["violation_rate"] > 0
    assert report["query"]["db_path"] == str(db_path)

    # The LangChain wrapper is registered and callable without Aspen COM.
    wrapped = analyze_closed_loop_tool.invoke({
        "db_path": str(db_path),
        "objective_names": "cost,energy",
    })
    assert json.loads(wrapped)["report_version"] == 1


def test_analyze_closed_loop_tool_reports_missing_database(tmp_path):
    result = _impl_analyze_closed_loop(
        str(tmp_path / "missing.db"), "cost,energy"
    )
    assert result.startswith("错误：")
