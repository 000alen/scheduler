# Scheduler

## Getting Started

### Setup

First, make sure you are **at least running Python 3.10**, then you can proceed to install the dependencies as follows:

```
pip install tqdm matplotlib networkx
```

Alternatively, if you use [conda](https://www.anaconda.com/), you can set up a conda environment (including the necessary Python version) by running

```
conda env create --file environment.yml
```

### Testing

If you would like to run the unit tests, you can do so by running

```
python -m scheduler.tests
```

### Usage

There are two main ways in which you can instance and run the scheduler. The recommended way is to instance all of your tasks (and the dependency relationships) before instancing the Scheduler. Here's an example of this:

```python
from scheduler import Scheduler
from scheduler.models import Task

task_1 = Task(...)
task_2 = Task(...)
task_2.add_dependency(task_1)

tasks = [task_1, task_2]

scheduler = Scheduler.from_tasks(tasks)
```

Alternatively, you could also set up the tasks and the relationships through the scheduler. Here's an example of this:

```python
from scheduler import Scheduler
from scheduler.models import Task

scheduler = Scheduler()

task_1 = Task(...)
scheduler.add_task(task_1)

task_2 = Task(...)
scheduler.add_task(task_2)
scheduler.add_dependency(task_2, task_1)
```

To generate a schedule, you can invoke the `run` method from a scheduler instance. This will return a schedule object (see the function's signature for the object shape). You can use a built-in helper function to visualize the schedule. Here's an example of this:

```python
from scheduler import Scheduler, pretty_print
from scheduler.models import Task

scheduler = Scheduler.from_tasks(...)

# This is another way of visualizing the tasks in a topological order
scheduler.plot()

schedule = scheduler.run()

# This will print the generated schedule
pretty_print(schedule)
```
