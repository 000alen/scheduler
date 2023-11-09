"""Definitions of basic data models for the scheduler utility."""

from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum, Enum, auto


class TaskPriority(IntEnum):
    """An enum defining standard levels of self-reported priority."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ABSOLUTE = 4


class TaskStatus(Enum):
    """An enum defining the statuses for a task."""

    PENDING = auto()
    QUEUED = auto()
    IN_PROGRESS = auto()
    FINISHED = auto()


@dataclass(order=True)
class Task:
    """Dataclass for a Task"""

    id: int
    """int: Unique identifier for a task."""

    name: str
    """str: Name of the task or event."""

    duration: float = field(compare=True)
    """
    float: Duration of the task or event. Units are determined depending on the 
    overall usage of the Scheduler utility
    """

    description: Optional[str] = None
    """Optional[str]: Description of the task or event."""

    fixed: bool = field(default=False)
    """bool: Indicates whether or not this event happens at a fixed time."""

    start: Optional[float] = field(default=None)
    """Optional[float]: Indicates the starting time of the event if it is a fixed event."""

    dependencies: set["Task"] = field(default_factory=set)
    """set[Task]: Indicates the tasks that must be resolved before scheduling this task."""

    priority: TaskPriority = field(default=TaskPriority.MEDIUM)
    """TaskPriority: Self-reported priority for this task."""

    status: TaskStatus = field(default=TaskStatus.PENDING)
    """TaskStatus: Indicates the status in the scheduling process"""

    def __hash__(self) -> int:
        """
        Computes the hash (unique identifier) for this Task.
        Used for fast look-ups in hash maps.
        """

        return hash(self.id)

    def add_dependency(self, task: "Task") -> None:
        """Adds a dependency to this task."""

        self.dependencies.add(task)

    def remove_dependency(self, task: "Task") -> None:
        """Removes a dependency from this task."""

        self.dependencies.remove(task)
