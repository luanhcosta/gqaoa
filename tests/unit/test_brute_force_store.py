import pytest

from gqaoa.domain.brute_force import BruteForceResult
from gqaoa.problem import brute_force_store
from gqaoa.problem.brute_force_store import load_brute_force_result, save_brute_force_result


@pytest.fixture(autouse=True)
def _isolated_problems_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(brute_force_store, "PROBLEMS_DIR", tmp_path / "problems")
    yield


def _result(bitstring="0110", energy=0.035, mode="full", n_evaluated=16, runtime=0.001):
    return BruteForceResult(
        optimal_bitstring=bitstring,
        optimal_energy=energy,
        search_mode=mode,
        n_candidates_evaluated=n_evaluated,
        runtime_seconds=runtime,
    )


def test_save_then_load_round_trip():
    result = _result()
    path = save_brute_force_result("problem-1", q=0.5, B=1, lamb=0.0, result=result)
    assert path.exists()

    payload = load_brute_force_result("problem-1")
    assert payload["problem_id"] == "problem-1"
    assert payload["search_mode"] == "full"
    assert payload["q"] == 0.5
    assert payload["B"] == 1.0
    assert payload["lamb"] == 0.0
    assert payload["optimal_bitstring"] == "0110"
    assert payload["optimal_energy"] == pytest.approx(0.035)
    assert payload["n_candidates_evaluated"] == 16
    assert payload["runtime_seconds"] == pytest.approx(0.001)
    assert payload["schema_version"] == 1
    assert "created_at" in payload


def test_load_unknown_problem_id_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        load_brute_force_result("does-not-exist")


def test_save_is_idempotent_noop_for_identical_content():
    result = _result()
    path1 = save_brute_force_result("problem-1", q=0.5, B=1, lamb=0.0, result=result)
    mtime1 = path1.stat().st_mtime_ns

    path2 = save_brute_force_result("problem-1", q=0.5, B=1, lamb=0.0, result=result)
    mtime2 = path2.stat().st_mtime_ns

    assert path1 == path2
    assert mtime1 == mtime2


def test_save_rejects_conflicting_params_without_overwrite():
    save_brute_force_result("problem-1", q=0.5, B=1, lamb=0.0, result=_result())

    with pytest.raises(FileExistsError):
        save_brute_force_result("problem-1", q=0.3, B=5, lamb=0.0, result=_result())


def test_save_rejects_conflicting_result_for_same_params_without_overwrite():
    save_brute_force_result("problem-1", q=0.5, B=1, lamb=0.0, result=_result())

    different_result = _result(bitstring="1001", energy=0.5)
    with pytest.raises(FileExistsError):
        save_brute_force_result("problem-1", q=0.5, B=1, lamb=0.0, result=different_result)


def test_save_overwrite_true_replaces_content():
    save_brute_force_result("problem-1", q=0.5, B=1, lamb=0.0, result=_result())

    replacement = _result(bitstring="1001", energy=0.5, mode="fixed-cardinality", n_evaluated=4)
    save_brute_force_result("problem-1", q=0.3, B=2, lamb=0.1, result=replacement, overwrite=True)

    payload = load_brute_force_result("problem-1")
    assert payload["optimal_bitstring"] == "1001"
    assert payload["q"] == 0.3
    assert payload["B"] == 2.0
    assert payload["lamb"] == pytest.approx(0.1)
    assert payload["search_mode"] == "fixed-cardinality"
