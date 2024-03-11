from heapq import heappush, heappop


class PriorityQueue:
    def __init__(self):
        self._pq = []  # list of entries arranged in a heap
        self._priority_finder = {}  # mapping of priority to entries

    def put(self, priority, task):
        "Add a new task"
        # any tasks with this priority?
        entry = self._priority_finder.get(priority)
        if entry:
            entry[1].append(task)
        else:
            entry = [priority, [task]]
            self._priority_finder[priority] = entry
            heappush(self._pq, entry)

    def get(self):
        "Remove and return the lowest priority tasks. Raise KeyError if empty."
        if not self._pq:
            raise KeyError("pop from an empty priority queue")
        priority, tasks = heappop(self._pq)
        del self._priority_finder[priority]
        return priority, tasks

    def __bool__(self):
        "return True if any tasks on the queue"
        return True if self._pq else False
