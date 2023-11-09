"""A task scheduler utility."""

from collections import defaultdict
from typing import Callable, Iterable, Optional

import networkx as nx  # type: ignore

from scheduler.models import Task, TaskStatus
from scheduler.structures import PriorityQueue
from scheduler.algorithms import dfs

TaskPriorityPolicy = Callable[["Scheduler", Task], float]


def basic_priority_policy(
    scheduler: "Scheduler",
    task: Task,
    *,
    shorter_tasks_affinity: float = 1,
    dependents_affinity: float = 1,
    self_reported_priority_affinity: float = 1,
) -> float:
    """A basic task priority policy.

    This policy assigns a priority to a task based on the following criteria:
    - Shorter tasks have higher priority.
    - Tasks with more dependents have higher priority.
    - Tasks with higher self-reported priority have higher priority.

    Args:
        scheduler (Scheduler): The scheduler.
        task (Task): The task.
        shorter_tasks_affinity (float, optional): The shorter tasks affinity. Defaults to 1.
        dependents_affinity (float, optional): The dependents affinity. Defaults to 1.
        self_reported_priority_affinity (float, optional): The self-reported priority affinity. Defaults to 1.

    Returns:
        float: The priority.
    """

    priority = 0.0

    # Shorter tasks have higher priority
    priority += shorter_tasks_affinity / task.duration

    # Tasks with more dependencies have higher priority
    priority += len(scheduler.get_dependents(task)) * dependents_affinity

    # Tasks with higher self-reported priority have higher priority
    priority += task.priority * self_reported_priority_affinity

    return priority


_multipliers = {
    "class": 3,
    "friends": 5,
    "cs11": 3,
    "breakfast": 6,
    "lunch": 6,
}


def keyword_based_priority_policy(
    _: "Scheduler",
    task: Task,
) -> float:
    """A keyword-based task priority policy.

    Args:
        scheduler (Scheduler): The scheduler.
        task (Task): The task.

    Returns:
        float: The priority.
    """

    normalized_name = task.name.lower()

    priority = float(task.priority)

    for keyword, multiplier in _multipliers.items():
        if keyword in normalized_name:
            priority *= multiplier

    return priority


def pretty_print(schedule: list[tuple[float, Task]]) -> None:
    """Pretty prints a schedule.

    Args:
        schedule (list[tuple[float, Task]]): The schedule.
    """

    for time, task in schedule:
        print(f"🕰 t={time:.2f}")
        print(f"\tstarted '{task.name}' for {task.duration:.2f} hours")
        print(f"\t✅ t={time+task.duration:.2f} task completed!")


class Scheduler:
    """Task scheduler utility class.

    This class is responsible for generating a schedule for a given set of tasks.
    The schedule is a list of tuples, where each tuple contains the time at which
    the task should start and the task itself.

    The scheduler is able to schedule tasks with dependencies, fixed-time tasks
    and non-fixed-time tasks. The scheduler is also able to schedule tasks with
    priorities, the priority of a task is a float number that represents the
    importance of the task, the higher the number the more important the task is.

    Example:
        >>> from scheduler import Scheduler, Task
        >>> tasks = [
        ...     Task(id=1, name="A", duration=1, fixed=True, start=0),
        ...     Task(id=2, name="B", duration=1, fixed=True, start=1),
        ...     Task(id=3, name="C", duration=1, fixed=False),
        ...     Task(id=4, name="D", duration=1, fixed=False),
        ]
        >>> scheduler = Scheduler.from_tasks(tasks)
        >>> schedule = scheduler.run(start_time=0) # by default uses no priority
        >>> for time, task in schedule:
        ...     print(f"{task.name} @ {time}")
        A @ 0
        B @ 1
        C @ 2
        D @ 3

    Attributes:
        _priority_policy (TaskPriorityPolicy): The task priority policy.
        schedule (list[tuple[float, Task]]): The schedule.
        fixed_tasks (set[Task]): The set of fixed-time tasks.
        fixed_dependencies_tasks (set[Task]): The set of fixed-time tasks'
            dependencies.
        dynamic_tasks (set[Task]): The set of non-fixed-time tasks.
        fixed_pq (PriorityQueue[Task]): The fixed-time tasks priority queue. The
            priority of a task is its start time (increasing order priority).
        fixed_dependencies_pq (PriorityQueue[Task]): The fixed-time tasks'
            dependencies priority queue. The priority of a task is its start time
            (increasing order priority).
        dynamic_pq (PriorityQueue[Task]): The non-fixed-time tasks priority queue.
            The priority of a task is its priority (decreasing order priority).
        tasks_map (dict[int, Task]): A map from task id to task.
        tasks_dependents (defaultdict[int, set[Task]]): A map from task id to
            task dependents.

    Todo:
        * Use a data structure that efficiently supports look-ahead operations.
        * Use a data structure that efficiently supports querying ranges of time.
            For example, a segment tree.
        * Refactor this class so that it uses immutable data structures.
    """

    _priority_policy: TaskPriorityPolicy
    """TaskPriorityPolicy: The task priority policy."""

    schedule: list[tuple[float, Task]]
    """list[tuple[float, Task]]: The schedule. Only populated after running the scheduler."""

    fixed_tasks: set[Task]
    """set[Task]: The set of fixed-time tasks. A set is chosen for fast look-up."""

    fixed_dependencies_tasks: set[Task]
    """set[Task]: The set of fixed-time tasks' dependencies. A set is chosen for fast look-up."""

    dynamic_tasks: set[Task]
    """set[Task]: The set of non-fixed-time tasks. A set is chosen for fast look-up."""

    fixed_pq: PriorityQueue[Task]
    """
    PriorityQueue[Task]: The fixed-time tasks priority queue. The priority of a task is its 
    start time (increasing order priority).
    """

    fixed_dependencies_pq: PriorityQueue[Task]
    """
    PriorityQueue[Task]: The fixed-time tasks' dependencies priority queue. The priority of a task
    is its start time (increasing order priority).
    """

    dynamic_pq: PriorityQueue[Task]
    """
    PriorityQueue[Task]: The non-fixed-time tasks priority queue. The priority of a task is its 
    priority (decreasing order priority).
    """

    tasks_map: dict[int, Task]
    """dict[int, Task]: A map from task id to task. A map is chosen for fast look-up."""

    tasks_dependents: defaultdict[int, set[Task]]
    """
    defaultdict[int, set[Task]]: A map from task id to task dependents. 
    A map is chosen for fast look-up.
    """

    def __init__(
        self,
        *,
        priority_policy: TaskPriorityPolicy = basic_priority_policy,
    ) -> None:
        self._priority_policy = priority_policy

        self.schedule = []

        self.fixed_tasks = set()
        self.fixed_dependencies_tasks = set()
        self.dynamic_tasks = set()

        self.fixed_pq: PriorityQueue[Task] = PriorityQueue(
            comparison=lambda x, y: x < y
        )
        self.fixed_dependencies_pq: PriorityQueue[Task] = PriorityQueue(
            comparison=lambda x, y: x < y
        )
        self.dynamic_pq: PriorityQueue[Task] = PriorityQueue()

        self.tasks_map = {}
        self.tasks_dependents = defaultdict(set)

    @classmethod
    def from_tasks(
        cls,
        tasks: list[Task],
        **kwargs,
    ) -> "Scheduler":
        """Initializes a scheduler from a list of tasks

            Args:
                tasks (list[Task]): The list of tasks.
                **kwargs: The keyword arguments to pass to the scheduler constructor.

            Returns:
                Scheduler: The scheduler.

        Example:
            >>> from scheduler import Scheduler, Task
            >>> tasks = [
            ...     Task(id=1, name="A", duration=1, fixed=True, start=0),
            ...     Task(id=2, name="B", duration=1, fixed=True, start=1),
            ...     Task(id=3, name="C", duration=1, fixed=False),
            ...     Task(id=4, name="D", duration=1, fixed=False),
            ]
            >>> scheduler = Scheduler.from_tasks(tasks)
            >>> schedule = scheduler.run(start_time=0) # by default uses no priority
            >>> for time, task in schedule:
            ...     print(f"{task.name} @ {time}")
            A @ 0
            B @ 1
            C @ 2
            D @ 3
        """

        scheduler = cls(**kwargs)

        for task in tasks:
            scheduler.add_task(task)

        return scheduler

    @property
    def tasks(self) -> Iterable[Task]:
        """Iterator over all tasks."""

        for task in self.fixed_tasks:
            yield task

        for task in self.dynamic_tasks:
            yield task

    def _get_compound_priority(self, parent: Task, child: Task) -> float:
        """
        Utility function to compute the compound priority for tasks that are (sub)
        dependencies of fixed-time tasks.

        Args:
            parent (Task): The parent task. Can be a fixed-time task or a
                fixed-time task's (sub) dependency.
            child (Task): The child task.

        Returns:
            float: The compound priority.
        """

        # Case 1: The parent is a fixed-time task
        if parent.start is not None:
            # The compound priority is the the parent's start time minus the
            # child's priority (down-scaled by 100). This preserves the relative
            # priorities of the child tasks.
            return parent.start - child.priority / 100
        else:
            # Case 2: The parent is a fixed-time task's (sub) dependency
            # The compound priority is the the parent's priority (which at some
            # point while traversing the graph was computed using the above case)
            # minus the child's priority (down-scaled by 100). This preserves the
            # relative priorities of the child tasks.
            return self.fixed_dependencies_pq.priorities[parent] - child.priority / 100

    def _push_fixed_tasks(self) -> None:
        """Pushes all fixed tasks into a time priority queue.

        Note:
            * Has a time complexity of O(n).
        """

        for task in self.fixed_tasks:
            assert task.start is not None

            self.fixed_pq.push((task.start, task))
            task.status = TaskStatus.QUEUED

    def _push_fixed_tasks_dependencies(self) -> None:
        """
        Pushes all fixed tasks' (sub) dependencies into a compound (time + priority)
        priority queue, while preserving relative priorities.

        Note:
            * Has a time complexity of O(E + V), since it uses a DFS traversal.
        """

        def on_visit(node: tuple[Task, Task]):
            """Callback function to be called when a node is visited.

            Args:
                node (tuple[Task, Task]): The node
            """

            parent, child = node

            if child.status != TaskStatus.PENDING:
                return

            priority = self._get_compound_priority(parent, child)

            self.fixed_dependencies_pq.push((priority, child))
            child.status = TaskStatus.QUEUED

        # Initialize the queue with all fixed tasks' direct dependencies
        queue: list[tuple[Task, Task]] = []
        for fixed in self.fixed_tasks:
            for dependency in fixed.dependencies:
                queue.append((fixed, dependency))

        if queue:
            dfs(
                queue=queue,
                get_neighbors=lambda task: [
                    (task[1], dependency) for dependency in task[1].dependencies
                ],
                on_visit=on_visit,
            )

    def _push_unblocked_dynamic_tasks(self) -> None:
        """Pushes all unblocked dynamic tasks into a priority queue.

        Note:
            * Has a time complexity of O(n).
        """

        for task in self.dynamic_tasks:
            if task.status == TaskStatus.PENDING and not task.dependencies:
                priority = self.get_priority(task)
                self.dynamic_pq.push((priority, task))
                task.status = TaskStatus.QUEUED

    def _schedule_task(self, task: Task, time: float) -> float:
        """
        Adds a task to the schedule and removes it from other tasks'
        dependencies.

        Args:
            task (Task): The task to schedule.
            time (float): The current time.

        Returns:
            float: The new time.

        Raises:
            AssertionError: If the task has dependencies that are not finished.
        """

        assert len(task.dependencies) == 0, (
            f"Task with id {task.id} cannot be "
            + "scheduled because it has dependencies with ids: "
            + f'{", ".join(map(lambda dep: str(dep.id), task.dependencies))}.'
        )

        self.schedule.append((time, task))
        time += task.duration
        task.status = TaskStatus.FINISHED

        # Remove the task from other tasks' dependencies
        self.remove_dependency(task)
        return time

    def _wait_until(self, fixed: Task, time: float) -> float:
        """Waits until the fixed-time task can be executed.

        Args:
            fixed (Task): The fixed-time task.
            time (float): The current time.

        Returns:
            float: The new time.

        Raises:
            AssertionError: If the fixed-time task has no start time.
        """

        assert fixed.start is not None

        time = fixed.start

        return time

    def _schedule_dynamic_or_wait(self, fixed: Task, time: float) -> float:
        """
        Determines whether to schedule a dynamic task or wait until next
        fixed-time task.

        Args:
            fixed (Task): The fixed-time task.
            time (float): The current time.

        Returns:
            float: The new time.

        Raises:
            AssertionError: If the fixed-time task has no start time.
        """

        assert fixed.start is not None

        # If there are dynamic tasks that can be executed before the fixed-time
        # task, execute them
        if self.dynamic_pq.size > 0:
            _, dynamic = self.dynamic_pq.peek()

            # Check if there is enough time to execute the dynamic task
            while dynamic.duration <= fixed.start - time:
                _, task = self.dynamic_pq.pop()
                time = self._schedule_task(task, time)

                if self.dynamic_pq.size == 0:
                    break

                _, dynamic = self.dynamic_pq.peek()

        if time < fixed.start:
            time = self._wait_until(fixed, time)

        return time

    def _schedule_fixed_or_fixed_dependency(self, time: float) -> float:
        """
        Determines whether to schedule a fixed-time task or a fixed-time task's
        dependency.

        Args:
            time (float): The current time.

        Returns:
            float: The new time.

        Raises:
            AssertionError: If the fixed-time task has no start time.
        """

        _, fixed_dependency = self.fixed_dependencies_pq.peek()
        _, fixed = self.fixed_pq.peek()

        assert fixed.start is not None

        # Case 1: It's time to execute the fixed-time task
        if time == fixed.start:
            _, task = self.fixed_pq.pop()
            time = self._schedule_task(task, time)

        # Case 2: It's not yet time to execute the fixed-time task, but there
        # is also not enough time to execute the dependency
        elif time + fixed_dependency.duration > fixed.start:
            time = self._schedule_dynamic_or_wait(fixed, time)

            _, task = self.fixed_pq.pop()
            time = self._schedule_task(task, time)

        # Case 3: There is enough time to execute the dependency
        else:
            _, task = self.fixed_dependencies_pq.pop()
            time = self._schedule_task(task, time)
        return time

    def _schedule_fixed_or_dynamic(self, time: float) -> float:
        """
        Determines whether to schedule a fixed-time task or a non-fixed-time task.

        Args:
            time (float): The current time.

        Returns:
            float: The new time.

        Raises:
            AssertionError: If the fixed-time task has no start time.
        """

        _, dynamic = self.dynamic_pq.peek()
        _, fixed = self.fixed_pq.peek()

        assert fixed.start is not None

        # Case 1: It's time to execute the fixed-time task
        if time == fixed.start:
            _, task = self.fixed_pq.pop()
            time = self._schedule_task(task, time)

        # Case 2: It's not yet time to execute the fixed-time task, but there
        # is also not enough time to execute the non-fixed-time task
        elif time + dynamic.duration > fixed.start:
            _, task = self.fixed_pq.pop()
            time = self._wait_until(fixed, time)
            time = self._schedule_task(task, time)

        # Case 3: There is enough time to execute the non-fixed-time task
        else:
            _, task = self.dynamic_pq.pop()
            time = self._schedule_task(task, time)
        return time

    def set_priority_policy(self, priority_policy: TaskPriorityPolicy) -> None:
        """Sets the task priority policy.

        Args:
            priority_policy (TaskPriorityPolicy): The task priority policy.

        Example:
            >>> from scheduler import Scheduler, Task
            >>> tasks = [
            ...     Task(
            ...         id=1, name="A", duration=1,
            ...         priority=TaskPriority.LOW
            ...     ),
            ...     Task(
            ...         id=2, name="B", duration=1,
            ...         priority=TaskPriority.MEDIUM
            ...     ),
            ...     Task(
            ...         id=3, name="C", duration=1,
            ...         priority=TaskPriority.HIGH
            ...     ),
            ... ]
            >>> scheduler = Scheduler.from_tasks(tasks)
            >>> scheduler.set_priority_policy(lambda s, t: t.priority)
            >>> schedule = scheduler.run(start_time=0)
            >>> for time, task in schedule:
            ...     print(f"{task.name} @ {time}")
            C @ 0
            B @ 2
            A @ 1
        """

        self._priority_policy = priority_policy

    def add_task(self, task: Task) -> None:
        """Adds a task to the scheduler.

        Args:
            task (Task): The task to add.

        Example:
            >>> from scheduler import Scheduler, Task
            >>> scheduler = Scheduler()
            >>> scheduler.add_task(Task(id=1, name="A", duration=1))
            >>> scheduler.add_task(Task(id=2, name="B", duration=1))
            >>> scheduler.run(start_time=0)
            [(0, Task(id=1, name='A', duration=1, fixed=False, start=None, priority=0))]
        """

        if task.fixed:
            return self.add_fixed_task(task)

        # Auxiliary data structures for fast look-up
        self.dynamic_tasks.add(task)
        self.tasks_map[task.id] = task

        for dependency in task.dependencies:
            self.add_dependency(task, dependency)

    def add_fixed_task(self, task: Task) -> None:
        """Adds a fixed-time task to the scheduler.

        Args:
            task (Task): The task to add.

        Example:
            >>> from scheduler import Scheduler, Task
            >>> scheduler = Scheduler()
            >>> scheduler.add_task(Task(id=1, name="A", duration=1, fixed=True, start=0))
            >>> scheduler.add_task(Task(id=2, name="B", duration=1, fixed=True, start=1))
            >>> scheduler.run(start_time=0)
            [(0, Task(id=1, name='A', duration=1, fixed=False, start=None, priority=0))]
        """

        assert task.fixed
        assert task.start is not None

        # Auxiliary data structures for fast look-up
        self.fixed_tasks.add(task)
        self.tasks_map[task.id] = task

        for dependency in task.dependencies:
            self.add_dependency(task, dependency)

    def add_dependency(self, task: Task, dependency: Task) -> None:
        """Adds a dependency between two tasks.

        Args:
            task (Task): The task.
            dependency (Task): The dependency.

        Example:
            >>> from scheduler import Scheduler, Task
            >>> task1 = Task(id=1, name="A", duration=1, fixed=True, start=0)
            >>> task2 = Task(id=2, name="B", duration=1, fixed=True, start=1)
            >>> scheduler = Scheduler()
            >>> scheduler.add_task(task1)
            >>> scheduler.add_task(task2)
            >>> scheduler.add_dependency(task1, task2)
        """

        task.add_dependency(dependency)

        # Since the data structure used to store the tasks' dependencies is a
        # set (hash map), there is no need to check for duplicates
        self.tasks_dependents[dependency.id].add(task)

    def get_priority(self, task: Task) -> float:
        """Computes the priority based on the current priority policy.

        Args:
            task (Task): The task.

        Returns:
            float: The priority.
        """

        return self._priority_policy(self, task)

    def get_dependents(self, task: Task) -> set[Task]:
        """Gets the task dependents.

        Args:
            task (Task): The task.

        Returns:
            set[Task]: The task dependents.
        """

        return self.tasks_dependents[task.id]

    def remove_dependency(self, task: Task) -> None:
        """Removes a task from other tasks' dependencies.

        Args:
            task (Task): The task.
        """

        for dependent in self.tasks_dependents[task.id]:
            dependent.remove_dependency(task)

    def _has_unfinished_tasks(self) -> bool:
        """Checks whether there are unfinished tasks."""

        return any(task.status != TaskStatus.FINISHED for task in self.tasks)

    def plot(self, **kwargs):
        """
        Plots the tasks dependencies graph in a layout that showcases
        topological order.


        Note:
            * Retrieved from [networkx's documentation](
                    https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html
                ).
            * Requires `matplotlib` and `networkx` to be installed.
        """

        colors = []
        sizes = []
        labels = {}
        graph = nx.DiGraph()

        for task in self.tasks:
            graph.add_node(task.id)
            labels[task.id] = f"{task.id} {task.fixed}, {task.duration} {task.start}"
            colors.append("red" if task.fixed else "blue")
            sizes.append(task.duration * 100 + 100)

        for task in self.tasks:
            for dependency in task.dependencies:
                graph.add_edge(dependency.id, task.id)

        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                graph.nodes[node]["layer"] = layer

        positions = nx.multipartite_layout(graph, subset_key="layer")

        nx.draw_networkx(
            graph,
            pos=positions,
            node_color=colors,
            node_size=sizes,
            with_labels=False,
            **kwargs,
        )
        nx.draw_networkx_labels(graph, labels=labels, pos=positions, **kwargs)

    def run(
        self, start_time: float, *, priority_policy: Optional[TaskPriorityPolicy] = None
    ) -> list[tuple[float, Task]]:
        """Runs the scheduler. Returns the schedule.

        Args:
            start_time (float): The start time.
            priority_policy (Optional[TaskPriorityPolicy]): The task priority policy.

        Returns:
            list[tuple[float, Task]]: The schedule.

        Example:
            >>> from scheduler import Scheduler, Task
            >>> tasks = [
            ...     Task(id=1, name="A", duration=1, fixed=True, start=0),
            ...     Task(id=2, name="B", duration=1, fixed=True, start=1),
            ...     Task(id=3, name="C", duration=1, fixed=False),
            ...     Task(id=4, name="D", duration=1, fixed=False),
            ... ]
            >>> scheduler = Scheduler.from_tasks(tasks)
            >>> schedule = scheduler.run(start_time=0)
            >>> for time, task in schedule:
            ...     print(f"{task.name} @ {time}")
            A @ 0
            B @ 1
            C @ 2
            D @ 3

        Note:
            * Follows a strategy pattern, delegating the scheduling logic to
                specialized private methods.
        """

        # If a priority policy is provided, use it
        if priority_policy is not None:
            self.set_priority_policy(priority_policy)

        time = start_time

        # Push all tasks into their respective priority queues
        self._push_fixed_tasks()  # O(n)
        self._push_fixed_tasks_dependencies()  # O(E + V)
        self._push_unblocked_dynamic_tasks()  # O(n)

        # Schedule tasks
        while self._has_unfinished_tasks() and (
            self.fixed_dependencies_pq.size > 0
            or self.fixed_pq.size > 0
            or self.dynamic_pq.size > 0
        ):
            # Case 1: There are fixed-time tasks and fixed-time tasks' dependencies
            while self.fixed_pq.size > 0 and self.fixed_dependencies_pq.size > 0:
                time = self._schedule_fixed_or_fixed_dependency(time)

                # After scheduling a fixed-time task, there may be unblocked
                # non-fixed-time tasks
                self._push_unblocked_dynamic_tasks()

            # Case 2: There are fixed-time tasks and non-fixed-time tasks
            while self.fixed_pq.size > 0 and self.dynamic_pq.size > 0:
                time = self._schedule_fixed_or_dynamic(time)
                self._push_unblocked_dynamic_tasks()

            # Case 3: There are only fixed-time tasks
            while self.dynamic_pq.size > 0:
                _, task = self.dynamic_pq.pop()
                time = self._schedule_task(task, time)
                self._push_unblocked_dynamic_tasks()

            # Case 4: There are only fixed-time tasks' dependencies left
            while self.fixed_pq.size > 0:
                _, task = self.fixed_pq.pop()

                assert task.start is not None

                if time < task.start:
                    time = self._wait_until(task, time)

                time = self._schedule_task(task, time)

                self._push_unblocked_dynamic_tasks()

        return self.schedule
