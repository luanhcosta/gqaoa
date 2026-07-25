import pennylane as qml


def get_device(device_name: str, wires: int, **kwargs) -> qml.devices.Device:
    """Single construction point for the quantum simulator backend.

    Production default across all strategies is "lightning.gpu" (requires an
    NVIDIA GPU + cuStateVec). Tests and CPU-only environments should pass
    "default.qubit" instead — every strategy's run_job() accepts device_name
    so the backend is injectable rather than hardcoded.
    """
    return qml.device(device_name, wires=wires, **kwargs)
