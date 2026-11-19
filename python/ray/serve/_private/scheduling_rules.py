"""Rules that decide which nodes a Serve replica may be placed on.

The deployment scheduler places a replica in three steps. It filters the
cluster's nodes down to those every rule allows, scores the rest to pick one,
and binds the replica to it. This module holds the rules used by the first
step.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Set

from ray.serve._private.common import DeploymentID


@dataclass
class SchedulingContext:
    """What a rule may read about the cluster when judging one replica."""

    deployment_id: DeploymentID
    # Labels of every active node.
    node_labels: Dict[str, Dict[str, str]]


class SchedulingConstraint:
    """A rule about which nodes a replica may be placed on."""

    def describe(self) -> str:
        """A short name for the rule, used in logs."""
        return type(self).__name__

    def allowed_nodes(
        self, node_ids: Iterable[str], ctx: SchedulingContext
    ) -> Set[str]:
        """The nodes among `node_ids` that the replica may be placed on."""
        return set(node_ids)


class ExcludeNodesConstraint(SchedulingConstraint):
    """Keeps replicas off a changing set of nodes, such as one being compacted.

    The set is read each time the rule is checked, so one instance follows the
    scheduler's state as it changes.
    """

    def __init__(self, nodes: Callable[[], Set[str]]):
        self._nodes = nodes

    def allowed_nodes(
        self, node_ids: Iterable[str], ctx: SchedulingContext
    ) -> Set[str]:
        return set(node_ids) - self._nodes()
