import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_mlflow_tracking(tmp_path, monkeypatch):
    # Every strategy's run_job() wraps its body in mlflow.start_run(); point
    # it at a throwaway per-test sqlite db so test runs don't create an
    # mlruns/ directory or pollute the real artifacts/mlflow.db.
    import mlflow

    test_uri = f"sqlite:///{tmp_path}/mlflow-test.db"
    mlflow.set_tracking_uri(test_uri)

    # gqaoa.tracking.mlflow_utils.init_mlflow() (used by the experiment
    # modules) hardcodes the real artifacts/mlflow.db path and would
    # override the line above, so redirect it too.
    monkeypatch.setattr("gqaoa.tracking.mlflow_utils.MLFLOW_TRACKING_URI", test_uri)
    monkeypatch.setattr("gqaoa.tracking.mlflow_utils.ARTIFACTS_DIR", tmp_path)

    # mlflow's fluent API caches the active experiment id at module scope, so
    # without this a test running after one that called set_experiment() in a
    # different tracking db would fail with "No Experiment with id=N exists".
    mlflow.set_experiment("pytest-default")
    yield


@pytest.fixture(autouse=True)
def _isolated_problems_dir(tmp_path, monkeypatch):
    # BEST_KNOWN_CONFIG (and every CLI's default config) now carries
    # problem_id=DEFAULT_PROBLEM_ID, so build_problem() touches
    # gqaoa.problem.store on every default run — bootstrapping
    # artifacts/problems/default-n10-fixed/problem.json on first use. Point
    # both problem-store modules at a throwaway per-test directory so test
    # runs don't write into the real (committed) artifacts/problems/.
    from gqaoa.problem import brute_force_store, store

    problems_dir = tmp_path / "problems"
    monkeypatch.setattr(store, "PROBLEMS_DIR", problems_dir)
    monkeypatch.setattr(brute_force_store, "PROBLEMS_DIR", problems_dir)
    yield problems_dir


@pytest.fixture
def device_name():
    return "default.qubit"


@pytest.fixture
def toy_ring_edges():
    # 4-node ring. QAOA.__init__ always sums cost_hamiltonian_wheight(q) for
    # q in range(4) regardless of n_assets (see domain/qaoa.py), so any toy
    # problem used to build a QAOA instance needs at least 4 assets.
    return [(0, 1), (1, 2), (2, 3), (3, 0)]


@pytest.fixture
def toy_expected_value():
    return [0.08, 0.05, 0.10, 0.03]


@pytest.fixture
def toy_cov_matrix():
    return pd.DataFrame([
        [1.00, 0.20, 0.10, 0.05],
        [0.20, 1.00, 0.15, 0.05],
        [0.10, 0.15, 1.00, 0.05],
        [0.05, 0.05, 0.05, 1.00],
    ])
