import json
import sys

import pytest

from ray.dashboard.modules.metrics.metrics_head import (
    DEFAULT_PROMETHEUS_HEADERS,
    parse_prom_headers,
)


def test_autoscaling_prefill_panel_maxes_across_handles():
    from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
        generate_serve_deployment_grafana_dashboard,
        generate_serve_llm_grafana_dashboard,
    )

    dashboard, _ = generate_serve_deployment_grafana_dashboard()
    assert all(p["title"] != "KV Router" for p in json.loads(dashboard)["panels"])
    dashboard, _ = generate_serve_llm_grafana_dashboard()
    row = next(p for p in json.loads(dashboard)["panels"] if p["title"] == "KV Router")
    assert row["type"] == "row"
    assert row["collapsed"] is True
    panel = next(
        p for p in row["panels"] if p["title"] == "KV Router: Uncached Prefill Tokens"
    )
    expr = panel["targets"][0]["expr"]
    assert expr.startswith(
        "max by (application, deployment) (ray_serve_autoscaling_router_metric{"
    )
    assert 'metric="outstanding_effective_prefill_tokens"' in expr
    assert 'deployment=~"$deployment"' in expr


@pytest.mark.parametrize(
    "headers, raises_error",
    [
        (DEFAULT_PROMETHEUS_HEADERS, False),
        ('{"H1": "V1", "H2": "V2"}', False),
        ('[["H1", "V1"], ["H2", "V2"], ["H2", "V3"]]', False),
        ('{"H1": "V1", "H2": ["V1", "V2"]}', True),
        ("not_json", True),
    ],
)
def test_parse_prom_headers(headers, raises_error):
    if raises_error:
        with pytest.raises(ValueError):
            parse_prom_headers(headers)
    else:
        result = parse_prom_headers(headers)
        assert result == json.loads(headers)


if __name__ == "__main__":
    sys.exit(pytest.main(["-vv", __file__]))
