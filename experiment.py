import bisect
from scheduler.models import Task


class Schedule:
    """TODO"""

    tasks: list[tuple[float, str, Task]]

    def __init__(self):
        self.tasks = []

    def insert_fixed_task(self, start: float, task: Task):
        """TODO"""

        end = start + task.duration

        bisect.insort(self.tasks, (start, "start", task))
        bisect.insort(self.tasks, (end, "end", task))

    def insert_dynamic_task(self, task: Task):
        """TODO"""

        duration = task.duration

        if not self.tasks or self.tasks[0][0] >= duration:
            self.insert_fixed_task(0, task)
            return 0
        for i in range(1, len(self.tasks)):
            if self.tasks[i][1] == "start" and self.tasks[i - 1][1] == "end":
                if self.tasks[i][0] - self.tasks[i - 1][0] >= duration:
                    start = self.tasks[i - 1][0]
                    self.insert_fixed_task(start, task)
                    return start
        start = self.tasks[-1][0]
        self.insert_fixed_task(start, task)
        return start
