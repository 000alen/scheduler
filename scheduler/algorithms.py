"""TODO"""

from collections import defaultdict
from typing import Callable, Iterable, Optional, TypeVar

T = TypeVar("T")


def bfs(
    *,
    start: Optional[T] = None,
    stack: Optional[list[T]] = None,
    get_neighbors: Callable[[T], Iterable[T]] = lambda node: [],
    on_visit: Callable[[T], None | bool] = lambda parent: None,
):
    """Generic Breadth-First Search algorithm

    Note:
        * This assumes the underlying graph is a Directed Acyclic Graph
        * Allows for early-stopping the exploration of paths
    """

    if not start and not stack:
        raise ValueError("Either start or stack must be provided")

    if start and stack:
        raise ValueError("Either start or stack must be provided")

    if start:
        stack = [start]

    assert stack is not None

    while len(stack) > 0:
        node = stack.pop()

        should_continue = on_visit(node)

        # Allows for early-stopping
        if should_continue is not None and not should_continue:
            continue

        for neighbor in get_neighbors(node):
            stack.append(neighbor)


def dfs(
    *,
    start: Optional[T] = None,
    queue: Optional[list[T]] = None,
    get_neighbors: Callable[[T], Iterable[T]] = lambda node: [],
    on_visit: Callable[[T], None | bool] = lambda node: None,
):
    """Generic Depth-First Search algorithm

    Note:
        * This assumes the underlying graph is a Directed Acyclic Graph
        * Allows for early-stopping the exploration of paths
    """

    if not start and not queue:
        raise ValueError("Either start or stack must be provided")

    if start and queue:
        raise ValueError("Either start or stack must be provided")

    if start:
        queue = [start]

    assert queue is not None

    while len(queue) > 0:
        node = queue.pop(0)

        should_continue = on_visit(node)

        if should_continue is not None and not should_continue:
            continue

        for neighbor in get_neighbors(node):
            queue.append(neighbor)


def toposort(g: dict[T, set[T]]) -> list[T]:
    """Generic Topological Sort algorithm"""

    in_degrees: defaultdict[T, int] = defaultdict(lambda: 0)

    for neighbors in g.values():
        for neighbor in neighbors:
            in_degrees[neighbor] += 1

    queue = [node for node in g.keys() if in_degrees[node] == 0]

    sorted_nodes = []
    while len(queue) > 0:
        node = queue.pop(0)

        sorted_nodes.append(node)

        for neighbor in g[node]:
            in_degrees[neighbor] -= 1

            if in_degrees[neighbor] == 0:
                queue.append(neighbor)

    return sorted_nodes
