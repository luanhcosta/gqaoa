import pytest

pytestmark = pytest.mark.slow


def test_tiny_bracket_run_wires_together(monkeypatch, tmp_path):
    from gqaoa.experiments import bracket as bracket_module

    # Shrink the phase budgets so the real (unmocked) gqaoa_strategy pipeline
    # runs end-to-end quickly, to catch wiring regressions before a real
    # 1000-QPU-call bracket run.
    monkeypatch.setattr(bracket_module, "N_PHASE1", 2)
    monkeypatch.setattr(bracket_module, "QPU_PHASE1", 2)
    monkeypatch.setattr(bracket_module, "TOP_K", 1)
    monkeypatch.setattr(bracket_module, "QPU_PHASE2", 2)
    monkeypatch.setattr(bracket_module, "QPU_PHASE3", 2)
    monkeypatch.setattr(bracket_module, "CHECKPOINTS_DIR", str(tmp_path))

    import dataclasses
    from gqaoa.config import BEST_KNOWN_CONFIG, ModelConfig

    tiny_config = dataclasses.replace(
        BEST_KNOWN_CONFIG,
        model=ModelConfig(vocab_size=4, n_embd=32, n_layer=1, n_head=1),
    )

    finals = bracket_module.run_bracket(
        n_repetitions=1, cleanup_checkpoints=True,
        device_name="default.qubit", base_config=tiny_config,
    )

    assert len(finals) == 1
    assert isinstance(finals[0], float)
