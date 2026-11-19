"""Rules that decide which nodes a Serve replica may be placed on.

The deployment scheduler places a replica in three steps. It filters the
cluster's nodes down to those every rule allows, scores the rest to pick one,
and binds the replica to it. This module holds the rules used by the first
step.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ray.serve._private.common import DeploymentID

# Ray sets this label on every node, so each node is its own domain for it.
RAY_NODE_ID_LABEL = "ray.io/node-id"


def label_value(
    node_id: str, label_key: str, node_labels: Mapping[str, Mapping[str, str]]
) -> Optional[str]:
    """The node's value for `label_key`, or None if the node has no such label.

    A node is always its own `ray.io/node-id` domain, even before the label
    cache holds its labels.
    """
    value = node_labels.get(node_id, {}).get(label_key)
    if value is None and label_key == RAY_NODE_ID_LABEL:
        return node_id
    return value


@dataclass
class SchedulingContext:
    """What a rule may read about the cluster when judging one replica."""

    deployment_id: DeploymentID
    # Labels of every active node.
    node_labels: Dict[str, Dict[str, str]]
    # The deployment's replicas on each active node: running ones, and
    # launching ones that already have a target node.
    replicas_per_node: Mapping[str, int] = field(default_factory=dict)
    # The deployment's replica count, in every state.
    num_replicas: int = 0
    # The active nodes the replica could ever be placed on: those its label
    # selectors allow and whose total resources fit it. None when unknown.
    candidate_nodes: Optional[Set[str]] = None


class SchedulingConstraint:
    """A rule about which nodes a replica may be placed on.

    A required rule is never dropped. A rule that isn't required is dropped,
    last listed first, when no node with room passes every rule.
    """

    required: bool = False

    def describe(self) -> str:
        """A short name for the rule, used in logs."""
        return type(self).__name__

    def allowed_nodes(
        self, node_ids: Iterable[str], ctx: SchedulingContext
    ) -> Set[str]:
        """The nodes among `node_ids` that the replica may be placed on."""
        return set(node_ids)

    def may_stop(self, node_id: str, ctx: SchedulingContext) -> bool:
        """Whether one of the deployment's replicas on the node may stop.

        `ctx.replicas_per_node` is the layout before the stop, and
        `ctx.num_replicas` the replica count the deployment is scaling to.
        """
        return True


class ExcludeNodesConstraint(SchedulingConstraint):
    """Keeps replicas off a changing set of nodes, such as one being compacted.

    The set is read each time the rule is checked, so one instance follows the
    scheduler's state as it changes. It is required, so no replica is ever
    placed on such a node.
    """

    required = True

    def __init__(self, nodes: Callable[[], Set[str]]):
        self._nodes = nodes

    def allowed_nodes(
        self, node_ids: Iterable[str], ctx: SchedulingContext
    ) -> Set[str]:
        return set(node_ids) - self._nodes()


class MinTopologyDomainsConstraint(SchedulingConstraint):
    """Spreads a deployment over at least `min_domains` values of a node label.

    A domain is the set of nodes that share one value of `label_key`. While
    the deployment's replicas cover fewer domains than the floor, the rule
    allows only nodes in a domain they don't cover yet. A node without the
    label is in no domain, so it can never raise the count and isn't allowed
    either. Once the floor holds, the rule allows every node.

    The floor is capped at the replica count, and at the number of domains
    among the nodes the replica could ever use, so it never holds back a
    replica that the cluster has no domain for.
    """

    def __init__(self, label_key: str, min_domains: int):
        self._label_key = label_key
        self._min_domains = min_domains

    @property
    def label_key(self) -> str:
        return self._label_key

    @property
    def min_domains(self) -> int:
        return self._min_domains

    def describe(self) -> str:
        return f"topology_spread[{self._label_key}>={self._min_domains}]"

    def covered_domains(self, ctx: SchedulingContext) -> Set[str]:
        """The domains the deployment's replicas are on."""
        domains = set()
        for node_id, count in ctx.replicas_per_node.items():
            domain = label_value(node_id, self._label_key, ctx.node_labels)
            if count > 0 and domain is not None:
                domains.add(domain)
        return domains

    def effective_floor(self, ctx: SchedulingContext) -> int:
        """The floor capped at the replica count and at the domains there are."""
        nodes = (
            ctx.candidate_nodes if ctx.candidate_nodes is not None else ctx.node_labels
        )
        num_domains = len(
            {
                domain
                for node_id in nodes
                if (domain := label_value(node_id, self._label_key, ctx.node_labels))
                is not None
            }
        )
        return min(self._min_domains, ctx.num_replicas, num_domains)

    def is_met(self, ctx: SchedulingContext) -> bool:
        return len(self.covered_domains(ctx)) >= self.effective_floor(ctx)

    def may_stop(self, node_id: str, ctx: SchedulingContext) -> bool:
        """Refuses a stop that would leave fewer domains than the floor."""
        domain = label_value(node_id, self._label_key, ctx.node_labels)
        if domain is None:
            return True
        replicas_per_domain: Counter[str] = Counter()
        for other_node, count in ctx.replicas_per_node.items():
            other_domain = label_value(other_node, self._label_key, ctx.node_labels)
            if count > 0 and other_domain is not None:
                replicas_per_domain[other_domain] += count
        if replicas_per_domain[domain] > 1:
            return True
        return len(replicas_per_domain) - 1 >= self.effective_floor(ctx)

    def allowed_nodes(
        self, node_ids: Iterable[str], ctx: SchedulingContext
    ) -> Set[str]:
        if self.is_met(ctx):
            return set(node_ids)
        covered = self.covered_domains(ctx)
        return {
            node_id
            for node_id in node_ids
            if (domain := label_value(node_id, self._label_key, ctx.node_labels))
            is not None
            and domain not in covered
        }


def rule_tiers(
    rules: Sequence[SchedulingConstraint],
) -> List[Tuple[SchedulingConstraint, ...]]:
    """The sets of rules to try a replica against, strictest first.

    The first tier holds every rule. Each later tier drops the last listed rule
    that isn't required, so a rule listed earlier is kept longer. The last tier
    holds only the required rules.
    """
    required = tuple(rule for rule in rules if rule.required)
    optional = [rule for rule in rules if not rule.required]
    return [
        required + tuple(optional[:num_optional])
        for num_optional in range(len(optional), -1, -1)
    ]
