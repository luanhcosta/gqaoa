import dataclasses

import pytest

from gqaoa.problem.default import default_problem_instance
from gqaoa.problem.store import load_problem, save_problem
from gqaoa.problem.synthetic import generate_synthetic_problem


def test_save_then_load_round_trip():
    instance = generate_synthetic_problem(n_assets=5, seed=99, n_trading_days=60)
    path = save_problem(instance)
    assert path.exists()

    loaded = load_problem(instance.problem_id)

    assert loaded.problem_id == instance.problem_id
    assert loaded.n_assets == instance.n_assets
    assert loaded.asset_names == instance.asset_names
    assert loaded.expected_value == pytest.approx(instance.expected_value)
    assert loaded.cov_matrix.to_numpy() == pytest.approx(instance.cov_matrix.to_numpy())
    assert loaded.edges_hc == instance.edges_hc
    assert loaded.edges_hb == instance.edges_hb
    assert loaded.source == instance.source
    assert loaded.provenance == instance.provenance
    assert loaded.created_at == instance.created_at
    assert loaded.schema_version == instance.schema_version


def test_load_problem_unknown_id_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        load_problem("does-not-exist")


def test_save_problem_is_idempotent_noop_for_identical_content():
    instance = default_problem_instance()
    path1 = save_problem(instance)
    mtime1 = path1.stat().st_mtime_ns

    path2 = save_problem(instance)
    mtime2 = path2.stat().st_mtime_ns

    assert path1 == path2
    assert mtime1 == mtime2  # file was not rewritten


def test_save_problem_rejects_conflicting_content_without_overwrite():
    instance = default_problem_instance()
    save_problem(instance)

    changed = dataclasses.replace(instance, expected_value=[0.0] * instance.n_assets)

    with pytest.raises(FileExistsError):
        save_problem(changed)


def test_save_problem_overwrite_true_replaces_content():
    instance = default_problem_instance()
    save_problem(instance)

    changed = dataclasses.replace(instance, expected_value=[0.0] * instance.n_assets)
    save_problem(changed, overwrite=True)

    reloaded = load_problem(instance.problem_id)
    assert reloaded.expected_value == [0.0] * instance.n_assets
