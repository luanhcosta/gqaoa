import pytest

from gqaoa.domain.device import get_device


def test_get_device_default_qubit_works():
    device = get_device("default.qubit", wires=3)
    assert device.wires.tolist() == [0, 1, 2]


def test_get_device_invalid_name_raises():
    with pytest.raises(Exception):
        get_device("not-a-real-device", wires=3)
