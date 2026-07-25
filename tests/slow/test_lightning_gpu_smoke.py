import pytest

from gqaoa.config import ModelConfig, ProblemConfig, TrainingConfig
from gqaoa.strategies import gqaoa_strategy

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_gqaoa_strategy_on_real_lightning_gpu_device():
    try:
        import pennylane as qml
        qml.device("lightning.gpu", wires=2)
    except Exception as exc:
        pytest.skip(f"lightning.gpu not available: {exc}")

    problem = ProblemConfig(q=0.3, B=5, lamb=0, initial_state="dicke_state", mixture_layer="xy", sdp=False)
    training = TrainingConfig(depth=1, limit_epochs=2, limit_qpu_call=3, optimizer_lr=0.01, shots_probs_result=50)
    model = ModelConfig(vocab_size=4, n_embd=32, n_layer=1, n_head=1)

    result = gqaoa_strategy.run_job(
        problem, training, model, device_name="lightning.gpu", run_name="test-gqaoa-gpu",
    )

    assert result["probs"].sum() == pytest.approx(1.0, abs=1e-6)
