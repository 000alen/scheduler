import copy
import random

import matplotlib.pyplot as plt  # type: ignore

from scheduler import Scheduler, basic_priority_policy, pretty_print
from scheduler.models import Task, TaskPriority


# breakfast
breakfast = Task(
    id=0,
    name="Breakfast",
    duration=0.5,
    priority=TaskPriority.ABSOLUTE,
)

# cs111 pcw
cs111_pre_class_work = Task(
    id=1,
    name="CS111 Pre-Class Work",
    duration=0.5,
    priority=TaskPriority.HIGH,
)

# cs110 pcw
cs110_pre_class_work = Task(
    id=2,
    name="CS110 Pre-Class Work",
    duration=0.5,
    priority=TaskPriority.MEDIUM,
)

# cs113 pcw
cs113_pre_class_work = Task(
    id=3,
    name="CS113 Pre-Class Work",
    duration=0.5,
    priority=TaskPriority.MEDIUM,
)

# cs111
cs111_class = Task(
    id=4,
    name="CS111",
    duration=1.5,
    fixed=True,
    start=8,
)
cs111_class.add_dependency(breakfast)
cs111_class.add_dependency(cs111_pre_class_work)

# cs110
cs110_class = Task(
    id=5,
    name="CS110",
    duration=1.5,
    fixed=True,
    start=10,
)
cs110_class.add_dependency(cs110_pre_class_work)

# cs113
cs113_class = Task(
    id=6,
    name="CS113",
    duration=1.5,
    fixed=True,
    start=12,
)
cs113_class.add_dependency(cs113_pre_class_work)

# lunch with friends
lunch_with_friends = Task(
    id=7,
    name="Lunch with Friends",
    duration=0.75,
    priority=TaskPriority.MEDIUM,
)

# korean class
korean_class = Task(
    id=8,
    name="Korean",
    duration=1.5,
    fixed=True,
    start=16,
)
korean_class.add_dependency(lunch_with_friends)

# cs111 skill builder
human_centered_ai = Task(
    id=9,
    name="Human Centered AI @ SWU",
    duration=2,
    fixed=True,
    start=17,
    priority=TaskPriority.MEDIUM,
)

# cs113 skill builder
meet_buddy_for_interview = Task(
    id=10,
    name="Meeting SWU buddy for interview",
    duration=1,
    priority=TaskPriority.HIGH,
)

meet_buddy_for_tour = Task(
    id=11,
    name="Meeting SWU buddy for tour",
    duration=1,
    priority=TaskPriority.MEDIUM,
)

tasks = [
    breakfast,
    cs111_pre_class_work,
    cs110_pre_class_work,
    cs113_pre_class_work,
    cs111_class,
    cs110_class,
    cs113_class,
    lunch_with_friends,
    korean_class,
    human_centered_ai,
    meet_buddy_for_interview,
    meet_buddy_for_tour,
]


scheduler = Scheduler.from_tasks(copy.deepcopy(tasks))
fig, ax = plt.subplots(figsize=(15, 10))
scheduler.plot(ax=ax)
plt.show()

schedule = scheduler.run(7, priority_policy=basic_priority_policy)
pretty_print(schedule)

_tasks = copy.deepcopy(tasks)
random.shuffle(_tasks)
_scheduler = Scheduler.from_tasks(_tasks)

assert _scheduler.run(7, priority_policy=basic_priority_policy) == schedule
