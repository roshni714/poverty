from heapq import heappush, heappop


class PriorityQueue:
    def __init__(self):
        """
        Initialize a priority queue.
        """
        self._pq = []  # list of entries arranged in a heap
        self._priority_finder = {}  # mapping of priority to entries

    def put(self, priority, task):
        """
        Add a new task to the priority queue.

        :param priority: The priority of the task.
        :type priority: int or float
        :param task: The task to add.
        :type task: Any
        """
        # any tasks with this priority?
        entry = self._priority_finder.get(priority)
        if entry:
            entry[1].append(task)
        else:
            entry = [priority, [task]]
            self._priority_finder[priority] = entry
            heappush(self._pq, entry)

    def get(self):
        """
        Remove and return the lowest priority tasks from the priority queue.
        Raise KeyError if the priority queue is empty.

        :return: A tuple containing the priority and tasks.
        :rtype: tuple(int or float, list)
        """
        if not self._pq:
            raise KeyError("pop from an empty priority queue")
        priority, tasks = heappop(self._pq)
        del self._priority_finder[priority]
        return priority, tasks

    def __bool__(self):
        """
        Return True if there are any tasks on the queue, False otherwise.
        """
        return True if self._pq else False
