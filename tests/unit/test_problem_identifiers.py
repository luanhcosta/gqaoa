import re

from gqaoa.problem.identifiers import compute_problem_id

_ID_RE = re.compile(r"^[a-z_]+-n\d+-[0-9a-f]{8}$")


def test_problem_id_matches_expected_format():
    problem_id = compute_problem_id("synthetic", 10, {"n_assets": 10, "seed": 1})
    assert _ID_RE.match(problem_id)
    assert problem_id.startswith("synthetic-n10-")


def test_problem_id_is_pure_and_deterministic():
    params = {"n_assets": 10, "seed": 1, "n_trading_days": 756, "topology": "ring"}
    a = compute_problem_id("synthetic", 10, params)
    b = compute_problem_id("synthetic", 10, dict(params))
    assert a == b


def test_problem_id_is_insensitive_to_key_order():
    a = compute_problem_id("synthetic", 10, {"seed": 1, "n_assets": 10})
    b = compute_problem_id("synthetic", 10, {"n_assets": 10, "seed": 1})
    assert a == b


def test_problem_id_changes_with_params():
    a = compute_problem_id("synthetic", 10, {"seed": 1})
    b = compute_problem_id("synthetic", 10, {"seed": 2})
    assert a != b
