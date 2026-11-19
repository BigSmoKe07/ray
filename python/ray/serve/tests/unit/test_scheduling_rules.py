import sys

import pytest

from ray.serve._private.common import DeploymentID
from ray.serve._private.scheduling_rules import (
    RAY_NODE_ID_LABEL,
    ExcludeNodesConstraint,
    MinTopologyDomainsConstraint,
    SchedulingConstraint,
    SchedulingContext,
    label_value,
    merge_selectors,
    rule_tiers,
)

DEPLOYMENT_ID = DeploymentID(name="d", app_name="app")


ZONE = "ray.io/availability-zone"

# a1 and a2 are in zone a, b1 in zone b, and u1 has no zone label.
NODE_LABELS = {
    "a1": {ZONE: "a"},
    "a2": {ZONE: "a"},
    "b1": {ZONE: "b"},
    "u1": {},
}


def make_ctx(
    node_labels=None, replicas_per_node=None, num_replicas=0, candidate_nodes=None
) -> SchedulingContext:
    return SchedulingContext(
        deployment_id=DEPLOYMENT_ID,
        node_labels=node_labels or {},
        replicas_per_node=replicas_per_node or {},
        num_replicas=num_replicas,
        candidate_nodes=candidate_nodes,
    )


def test_base_constraint_allows_every_node():
    assert SchedulingConstraint().allowed_nodes({"n1", "n2"}, make_ctx()) == {
        "n1",
        "n2",
    }
    assert SchedulingConstraint().describe() == "SchedulingConstraint"


def test_exclude_nodes_follows_the_current_set():
    excluded = set()
    rule = ExcludeNodesConstraint(lambda: excluded)
    ctx = make_ctx()

    assert rule.allowed_nodes({"n1", "n2"}, ctx) == {"n1", "n2"}

    excluded.add("n1")
    assert rule.allowed_nodes({"n1", "n2"}, ctx) == {"n2"}

    excluded.clear()
    assert rule.allowed_nodes({"n1", "n2"}, ctx) == {"n1", "n2"}


def test_label_value():
    assert label_value("a1", ZONE, NODE_LABELS) == "a"
    assert label_value("u1", ZONE, NODE_LABELS) is None
    assert label_value("unknown", ZONE, NODE_LABELS) is None
    # Every node is its own node id domain, labeled or not.
    assert label_value("u1", RAY_NODE_ID_LABEL, NODE_LABELS) == "u1"


def test_floor_allows_only_uncovered_domains_while_unmet():
    rule = MinTopologyDomainsConstraint(ZONE, 2)
    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1}, num_replicas=2)

    assert not rule.is_met(ctx)
    assert rule.covered_domains(ctx) == {"a"}
    # Zone a is covered, and u1 is in no zone, so only b1 adds a domain.
    assert rule.allowed_nodes(NODE_LABELS, ctx) == {"b1"}


def test_floor_allows_every_node_once_met():
    rule = MinTopologyDomainsConstraint(ZONE, 2)
    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1, "b1": 1}, num_replicas=3)

    assert rule.is_met(ctx)
    assert rule.allowed_nodes(NODE_LABELS, ctx) == set(NODE_LABELS)


def test_floor_is_capped_at_the_replica_count():
    rule = MinTopologyDomainsConstraint(ZONE, 3)

    ctx = make_ctx(NODE_LABELS, num_replicas=1)
    assert rule.effective_floor(ctx) == 1

    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1}, num_replicas=1)
    assert rule.is_met(ctx)


def test_floor_is_capped_at_the_domains_the_replica_can_use():
    rule = MinTopologyDomainsConstraint(ZONE, 3)

    # Two zones exist, so a floor of three acts as two.
    ctx = make_ctx(NODE_LABELS, num_replicas=6)
    assert rule.effective_floor(ctx) == 2

    # The replica fits only on zone a nodes, so the floor acts as one.
    ctx = make_ctx(
        NODE_LABELS,
        replicas_per_node={"a1": 1},
        num_replicas=6,
        candidate_nodes={"a1", "a2"},
    )
    assert rule.effective_floor(ctx) == 1
    assert rule.is_met(ctx)


def test_node_floor_counts_distinct_nodes():
    rule = MinTopologyDomainsConstraint(RAY_NODE_ID_LABEL, 3)
    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 2, "u1": 1}, num_replicas=4)

    # Every node is its own domain, unlabeled ones included.
    assert rule.covered_domains(ctx) == {"a1", "u1"}
    assert rule.allowed_nodes(NODE_LABELS, ctx) == {"a2", "b1"}


def test_floor_describe():
    assert (
        MinTopologyDomainsConstraint(ZONE, 2).describe()
        == "topology_spread[ray.io/availability-zone>=2]"
    )


def test_rule_tiers_drop_optional_rules_last_listed_first():
    first = MinTopologyDomainsConstraint(ZONE, 2)
    second = MinTopologyDomainsConstraint(RAY_NODE_ID_LABEL, 3)
    required = ExcludeNodesConstraint(set)

    assert rule_tiers([first, second, required]) == [
        (required, first, second),
        (required, first),
        (required,),
    ]
    assert rule_tiers([required]) == [(required,)]
    assert rule_tiers([]) == [()]


def test_floor_refuses_a_stop_that_drops_below_it():
    rule = MinTopologyDomainsConstraint(ZONE, 2)
    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 2, "b1": 1}, num_replicas=2)

    # b1 is the only replica in zone b.
    assert not rule.may_stop("b1", ctx)
    # Zone a keeps a replica after one on a1 stops.
    assert rule.may_stop("a1", ctx)
    # A node outside every domain never holds the floor up.
    assert rule.may_stop("u1", ctx)


def test_floor_allows_stops_down_to_the_new_replica_count():
    rule = MinTopologyDomainsConstraint(ZONE, 2)
    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1, "b1": 1}, num_replicas=1)

    # Scaling to one replica caps the floor at one zone.
    assert rule.may_stop("b1", ctx)


def test_floor_selector_avoids_covered_domains_while_unmet():
    rule = MinTopologyDomainsConstraint(ZONE, 2)

    ctx = make_ctx(NODE_LABELS, num_replicas=2)
    # Nothing is covered yet, so any node with a zone adds one.
    assert rule.selector(ctx) == {ZONE: "exists()"}

    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1}, num_replicas=2)
    assert rule.selector(ctx) == {ZONE: "exists(),!in(a)"}

    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1, "b1": 1}, num_replicas=2)
    assert rule.selector(ctx) == {}


def test_node_floor_selector_needs_no_exists():
    rule = MinTopologyDomainsConstraint(RAY_NODE_ID_LABEL, 2)
    assert rule.selector(make_ctx(NODE_LABELS, num_replicas=2)) == {}
    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1}, num_replicas=2)
    assert rule.selector(ctx) == {RAY_NODE_ID_LABEL: "!in(a1)"}


def test_exclude_nodes_selector():
    assert ExcludeNodesConstraint(set).selector(make_ctx()) == {}
    assert ExcludeNodesConstraint(lambda: {"n2", "n1"}).selector(make_ctx()) == {
        RAY_NODE_ID_LABEL: "!in(n1,n2)"
    }


def test_merge_selectors_joins_expressions_on_a_key():
    assert merge_selectors(
        [{ZONE: "in(a,b,c)"}, {ZONE: "!in(a)", "gpu": "a100"}, {ZONE: "!in(a)"}]
    ) == {ZONE: "in(a,b,c),!in(a)", "gpu": "a100"}
    assert merge_selectors([]) == {}


def test_required_floor_isnt_capped_at_the_domains_there_are():
    best_effort = MinTopologyDomainsConstraint(ZONE, 3)
    required = MinTopologyDomainsConstraint(ZONE, 3, required=True)
    ctx = make_ctx(NODE_LABELS, replicas_per_node={"a1": 1, "b1": 1}, num_replicas=6)

    # Only zones a and b exist.
    assert best_effort.is_met(ctx)
    assert not required.is_met(ctx)
    assert required.missing_domains(ctx) == 1
    # It's still capped at the replica count.
    assert required.effective_floor(make_ctx(NODE_LABELS, num_replicas=2)) == 2
    assert required.describe() == (
        "required topology_spread[ray.io/availability-zone>=3]"
    )


def test_required_rules_stay_in_every_tier():
    required = MinTopologyDomainsConstraint(ZONE, 2, required=True)
    optional = MinTopologyDomainsConstraint(RAY_NODE_ID_LABEL, 3)
    assert rule_tiers([required, optional]) == [(required, optional), (required,)]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-s", __file__]))
