import math
import sys

import pytest

from ray.serve._private.common import DeploymentID
from ray.serve.config import AutoscalingContext
from ray.serve.experimental.llm_token_autoscaling_policy import (
    OUTSTANDING_EFFECTIVE_PREFILL_TOKENS,
    REQUIRED_REPLICAS,
    LLMTokenAutoscalingPolicy,
)

_UNSET = object()


class TestLLMTokenAutoscalingPolicy:
    """Tests for LLMTokenAutoscalingPolicy, which scales on the uncached prefill
    tokens reported by KV routers."""

    def _make_ctx(
        self,
        *,
        prefill_by_handle=None,
        aggregated_router_metrics=_UNSET,
        target_num_replicas=2,
        total_num_requests=0,
    ):
        if aggregated_router_metrics is _UNSET:
            aggregated_router_metrics = {
                OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: prefill_by_handle or {},
            }
        return AutoscalingContext(
            deployment_id=DeploymentID(name="test", app_name="app"),
            deployment_name="test",
            app_name="app",
            current_num_replicas=target_num_replicas,
            target_num_replicas=target_num_replicas,
            running_replicas=[],
            total_num_requests=total_num_requests,
            total_queued_requests=0,
            aggregated_metrics=None,
            raw_metrics=None,
            capacity_adjusted_min_replicas=0,
            capacity_adjusted_max_replicas=100,
            policy_state={},
            last_scale_up_time=None,
            last_scale_down_time=None,
            current_time=None,
            config=None,
            aggregated_router_metrics=aggregated_router_metrics,
        )

    def test_scales_from_aggregated_tokens(self):
        """Requests ceil(tokens / target) replicas and returns the inputs as state."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(prefill_by_handle={"a": 350.0}, target_num_replicas=1)
        )

        assert replicas == 4
        assert state == {
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: 350.0,
            REQUIRED_REPLICAS: 4,
        }

    def test_takes_max_across_handles(self):
        """Uses the largest handle's tokens, since each router sees the global load."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": 200.0, "b": 300.0}, target_num_replicas=1
            )
        )

        assert replicas == 3
        assert state[OUTSTANDING_EFFECTIVE_PREFILL_TOKENS] == 300.0

    @pytest.mark.parametrize("outstanding_tokens", [0, 100])
    def test_scales_down(self, outstanding_tokens):
        """Scales below the current target when tokens drop, including to zero."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": float(outstanding_tokens)},
                target_num_replicas=7,
            )
        )

        assert replicas == math.ceil(outstanding_tokens / 100)
        assert state[REQUIRED_REPLICAS] == math.ceil(outstanding_tokens / 100)

    @pytest.mark.parametrize("aggregated_router_metrics", [None, {}, {"other": {}}])
    def test_holds_target_for_missing_metric(self, aggregated_router_metrics):
        """Keeps the current target when no router has reported the metric."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                aggregated_router_metrics=aggregated_router_metrics,
                target_num_replicas=7,
            )
        )

        assert replicas == 7
        assert state == {}

    @pytest.mark.parametrize("target_num_replicas", [0, 3])
    def test_scales_from_zero_for_requests_without_metric(self, target_num_replicas):
        """Starts at least one replica for requests before any router reports."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                aggregated_router_metrics={},
                target_num_replicas=target_num_replicas,
                total_num_requests=2,
            )
        )

        assert replicas == max(1, target_num_replicas)
        assert state == {}

    def test_keeps_one_replica_when_requests_are_in_flight(self):
        """Never scales to zero while requests are in flight."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(
                prefill_by_handle={"a": 0.0},
                target_num_replicas=0,
                total_num_requests=1,
            )
        )

        assert replicas == 1
        assert state[REQUIRED_REPLICAS] == 1

    def test_clamps_negative_tokens_to_zero(self):
        """Treats a negative token count as zero."""
        policy = LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=100)

        replicas, state = policy(
            self._make_ctx(prefill_by_handle={"a": -50.0}, target_num_replicas=0)
        )

        assert replicas == 0
        assert state[OUTSTANDING_EFFECTIVE_PREFILL_TOKENS] == 0

    @pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf")])
    def test_rejects_invalid_target(self, target):
        """Rejects a target that is not positive and finite."""
        with pytest.raises(ValueError, match="positive and finite"):
            LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=target)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
