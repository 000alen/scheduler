"""Implements basic data structures for the scheduler utility."""

from typing import Callable, TypeVar, Generic


T = TypeVar("T")
Comparison = Callable[[T, T], bool]

_default_comparison: Comparison = lambda a, b: a > b
"""Comparison: Default comparison function. If used, the heap will be a max-heap."""


class Heap(Generic[T]):
    """Generic heap implementation."""

    values: list[T]
    """list[T]: List of values in the heap."""

    comparison: Comparison
    """Comparison: Comparison function used to determine the heap property."""

    @staticmethod
    def left(i: int) -> int:
        """Returns the index of the left child of the node at index i.

        Args:
            i (int): Index of the node.

        Returns:
            int: Index of the left child.
        """

        return 2 * i + 1

    @staticmethod
    def right(i: int) -> int:
        """Returns the index of the right child of the node at index i.

        Args:
            i (int): Index of the node.

        Returns:
            int: Index of the right child.
        """

        return 2 * i + 2

    @staticmethod
    def parent(i: int) -> int:
        """Returns the index of the parent of the node at index i.

        Args:
            i (int): Index of the node.

        Returns:
            int: Index of the parent.
        """

        return (i - 1) // 2

    def __init__(self, *, comparison: Comparison = _default_comparison) -> None:
        self.values = []
        self.comparison = comparison

    @property
    def size(self) -> int:
        """Returns the number of elements in the heap."""

        return len(self.values)

    def _swap(self, i: int, j: int) -> None:
        """Swaps the values at indices i and j.

        Args:
            i (int): Index of the first value.
            j (int): Index of the second value.
        """

        self.values[i], self.values[j] = self.values[j], self.values[i]

    @classmethod
    def from_list(cls, values: list[T], **kwargs) -> "Heap":
        """Creates a heap from a list of values.

        Args:
            values (list[T]): List of values to be inserted into the heap.
            **kwargs: Keyword arguments to be passed to the constructor.

        Returns:
            Heap: A heap containing the values.
        """

        heap = cls(**kwargs)
        heap.values = values
        heap.build()
        return heap

    def build(self) -> None:
        """Builds a heap from the values by applying heapify from the bottom up."""

        for i in range(len(self.values) // 2, -1, -1):
            self.heapify(i)

    def heapify(self, i: int) -> None:
        """Restores the heap property at index i.

        Args:
            i (int): Index of the node.

        Note:
            * Assumes that the subtrees of i are heaps.
        """

        left = self.left(i)
        right = self.right(i)

        largest = i

        if left < len(self.values) and self.comparison(
            self.values[left], self.values[largest]
        ):
            largest = left

        if right < len(self.values) and self.comparison(
            self.values[right], self.values[largest]
        ):
            largest = right

        if largest != i:
            self._swap(i, largest)
            self.heapify(largest)

    def push(self, value: T) -> None:
        """Pushes a value onto the heap.

        Args:
            value (T): Value to be pushed onto the heap.
        """

        self.values.append(value)

        i = len(self.values) - 1
        while i > 0 and not self.comparison(
            self.values[self.parent(i)], self.values[i]
        ):
            self._swap(i, self.parent(i))
            i = self.parent(i)

    def _pop(self) -> T:
        """Pops a single value from the heap. Does not check if the heap is empty.

        Returns:
            T: Value with the highest (or lowest, per the indicated comparison) priority.
        """

        self._swap(0, len(self.values) - 1)
        value = self.values.pop()
        self.heapify(0)
        return value

    def pop(self) -> T:
        """Pops a single value from the heap.

        Returns:
            T: Value with the highest (or lowest, per the indicated comparison) priority.
        """

        assert not self.empty()

        return self._pop()

    def pop_k(self, k) -> list[T]:
        """Pops k values from the heap.

        Args:
            k (int): Number of values to pop.

        Returns:
            list[T]: List of values with the highest (or lowest, per the
                indicated comparison) priority.
        """

        assert k > 0
        assert k <= len(self.values)

        values = []
        for _ in range(k):
            values.append(self._pop())

        return values

    def empty(self) -> bool:
        """Returns True if the heap is empty."""

        return len(self.values) == 0

    def peek(self) -> T:
        """Returns the value with the highest priority."""

        assert len(self.values) > 0

        return self.values[0]


class PriorityQueue(Heap[tuple[float, T]]):
    """Generic priority queue implementation."""

    priorities: dict[T, float]
    """dict[T, float]: Dictionary mapping values to their priorities."""

    def __init__(self, *, comparison: Comparison = _default_comparison) -> None:
        super().__init__(comparison=comparison)
        self.priorities = {}

    def push(self, value: tuple[float, T]) -> None:
        """Pushes a value onto the priority queue.

        Args:
            value (tuple[float, T]): Value to be pushed onto the priority queue.
        """

        _priority, _value = value

        if _value in self.priorities:
            raise ValueError(f"Value {_value} already in the priority queue")

        self.priorities[_value] = _priority

        return super().push(value)

    def _pop(self) -> tuple[float, T]:
        """
        Pops a single value from the priority queue. Does not check if the
        priority queue is empty.

        Returns:
            tuple[float, T]: Value with the highest (or lowest, per the
                indicated comparison) priority.
        """

        value = super()._pop()

        _, _value = value
        self.priorities.pop(_value)

        return value
