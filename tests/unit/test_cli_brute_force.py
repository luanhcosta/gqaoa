import json

import pytest

from gqaoa.cli import run_brute_force
from gqaoa.problem import store
from gqaoa.problem.synthetic import generate_synthetic_problem


def test_cli_generates_and_persists_synthetic_problem_then_runs_brute_force(_isolated_problems_dir, capsys):
    run_brute_force.main([
        "--n-assets", "4", "--seed", "7", "--n-trading-days", "50",
        "--q", "0.5", "--B", "1", "--lamb", "0", "--search-mode", "full",
    ])

    problem_dirs = list(_isolated_problems_dir.iterdir())
    assert len(problem_dirs) == 1
    problem_id = problem_dirs[0].name

    problem_json = json.loads((_isolated_problems_dir / problem_id / "problem.json").read_text())
    assert problem_json["n_assets"] == 4
    assert problem_json["source"] == "synthetic"

    brute_force_json = json.loads((_isolated_problems_dir / problem_id / "brute_force.json").read_text())
    assert brute_force_json["problem_id"] == problem_id
    assert brute_force_json["search_mode"] == "full"
    assert brute_force_json["q"] == 0.5
    assert brute_force_json["B"] == 1.0
    assert brute_force_json["lamb"] == 0.0
    assert len(brute_force_json["optimal_bitstring"]) == 4
    assert set(brute_force_json["optimal_bitstring"]) <= {"0", "1"}
    assert brute_force_json["n_candidates_evaluated"] == 16
    assert brute_force_json["schema_version"] == 1

    out = capsys.readouterr().out
    assert f"problem_id: {problem_id}" in out
    assert "optimal_bitstring" in out
    assert "optimal_energy" in out


def test_cli_resolves_by_existing_problem_id(_isolated_problems_dir, capsys):
    instance = generate_synthetic_problem(n_assets=4, seed=3, n_trading_days=50)
    store.save_problem(instance)

    run_brute_force.main([
        "--problem-id", instance.problem_id,
        "--q", "0.5", "--B", "2", "--lamb", "0", "--search-mode", "fixed-cardinality",
    ])

    brute_force_json = json.loads(
        (_isolated_problems_dir / instance.problem_id / "brute_force.json").read_text()
    )
    assert brute_force_json["problem_id"] == instance.problem_id
    assert brute_force_json["search_mode"] == "fixed-cardinality"
    assert brute_force_json["optimal_bitstring"].count("1") == 2

    out = capsys.readouterr().out
    assert f"problem_id: {instance.problem_id}" in out


def test_cli_rerunning_via_problem_id_with_identical_params_is_a_noop(_isolated_problems_dir):
    inline_args = ["--n-assets", "4", "--seed", "11", "--n-trading-days", "50", "--q", "0.5", "--B", "1", "--lamb", "0"]
    run_brute_force.main(inline_args)

    problem_id = next(_isolated_problems_dir.iterdir()).name
    brute_force_path = _isolated_problems_dir / problem_id / "brute_force.json"
    mtime1 = brute_force_path.stat().st_mtime_ns

    # Rerunning by --problem-id (no regeneration, so problem.json is untouched)
    # with the same q/B/lamb/search-mode sees an identical result and no-ops
    # instead of rewriting brute_force.json.
    run_brute_force.main(["--problem-id", problem_id, "--q", "0.5", "--B", "1", "--lamb", "0"])
    mtime2 = brute_force_path.stat().st_mtime_ns

    assert mtime1 == mtime2


def test_cli_regenerating_inline_without_overwrite_rejects_conflicting_problem(_isolated_problems_dir):
    inline_args = ["--n-assets", "4", "--seed", "11", "--n-trading-days", "50", "--q", "0.5", "--B", "1", "--lamb", "0"]
    run_brute_force.main(inline_args)

    # Regenerating from scratch produces a fresh created_at timestamp, so even
    # with identical parameters the serialized problem differs and save_problem
    # (unmodified, existing behavior) requires an explicit --overwrite.
    with pytest.raises(FileExistsError):
        run_brute_force.main(inline_args)

    # --overwrite makes the rerun succeed.
    run_brute_force.main([*inline_args, "--overwrite"])


def test_cli_requires_a_problem_source():
    with pytest.raises(SystemExit):
        run_brute_force.main([])


def test_cli_rejects_ambiguous_problem_source():
    with pytest.raises(SystemExit):
        run_brute_force.main(["--problem-id", "some-id", "--n-assets", "4"])


def test_cli_defaults_match_best_known_config():
    args = run_brute_force.build_arg_parser().parse_args(["--n-assets", "4"])
    assert args.q == pytest.approx(0.3)
    assert args.B == 5
    assert args.lamb == pytest.approx(0.0)
    assert args.search_mode == "fixed-cardinality"
    assert args.topology == "ring"
