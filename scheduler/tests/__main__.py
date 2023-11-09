"""Unit tests for the scheduler, and the underlying Heap and PriorityQueue data structures."""

import unittest
import random

from tqdm import tqdm  # type: ignore

from scheduler.structures import Heap, PriorityQueue

random.seed(42)


class TestHeap(unittest.TestCase):
    """Unit tests for the Heap data structure."""

    def test_build(self):
        """Tests the construction of a max-heap from a list of values."""

        for n in tqdm(range(0, 100)):
            values = list(range(n))
            random.shuffle(values)

            heap = Heap.from_list(values)

            stack = [0]

            while stack:
                i = stack.pop()

                left = heap.left(i)
                right = heap.right(i)

                if left < len(heap.values):
                    self.assertGreaterEqual(heap.values[i], heap.values[left])
                    stack.append(left)

                if right < len(heap.values):
                    self.assertGreaterEqual(heap.values[i], heap.values[right])
                    stack.append(right)

    def test_heapify(self):
        """Tests the heapify operation to restore the heap property after a value is changed."""

        for n in tqdm(range(0, 100)):
            values = list(range(n))
            random.shuffle(values)

            heap = Heap.from_list(values)

            # pylint: disable=consider-using-enumerate
            for i in range(len(heap.values)):
                heap.heapify(i)

                left = heap.left(i)
                right = heap.right(i)

                if left < len(heap.values):
                    self.assertGreaterEqual(heap.values[i], heap.values[left])

                if right < len(heap.values):
                    self.assertGreaterEqual(heap.values[i], heap.values[right])

    def test_push(self):
        """Tests the push operation to add a value to the heap."""

        heap = Heap()

        for n in tqdm(range(0, 100)):
            values = list(range(n))
            random.shuffle(values)

            for value in values:
                heap.push(value)

            # pylint: disable=consider-using-enumerate
            for i in range(len(heap.values)):
                left = heap.left(i)
                right = heap.right(i)

                if left < len(heap.values):
                    self.assertGreaterEqual(heap.values[i], heap.values[left])

                if right < len(heap.values):
                    self.assertGreaterEqual(heap.values[i], heap.values[right])

    def test_pop(self):
        """Tests the pop operation to remove the maximum value from the heap."""

        for n in tqdm(range(0, 100)):
            values = list(range(n))
            random.shuffle(values)

            heap = Heap.from_list(values)

            for i in range(n - 1, -1, -1):
                self.assertEqual(heap.pop(), i)


class TestPriorityQueue(unittest.TestCase):
    """Unit tests for the PriorityQueue data structure."""

    def test_peek(self):
        """Tests the peek operation to return the maximum value from the priority queue."""

        queue = PriorityQueue()

        for n in tqdm(range(0, 100)):
            values = list(range(n))
            random.shuffle(values)

            for value in values:
                queue.push((value, value))

            for i in range(n - 1, -1, -1):
                _, value = queue.peek()
                self.assertEqual(value, i)
                queue.pop()


if __name__ == "__main__":
    unittest.main()
