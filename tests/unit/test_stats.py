import math

import pytest

from gqaoa.reporting.stats import report_stats


def test_report_stats_exact_values(capsys):
    stats = report_stats([1, 2, 3, 4, 5], "toy")

    assert stats["min"] == pytest.approx(1.0)
    assert stats["p10"] == pytest.approx(1.4)
    assert stats["p25"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["p75"] == pytest.approx(4.0)
    assert stats["p90"] == pytest.approx(4.6)
    assert stats["max"] == pytest.approx(5.0)
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["std"] == pytest.approx(math.sqrt(2.0))

    captured = capsys.readouterr()
    assert "toy" in captured.out
