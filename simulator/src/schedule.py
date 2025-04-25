from enum import Enum
import json
from collections import deque

class ScheduleItem:
    def __init__(self, items: list = []) -> None:
        self.items = items

class Schedule:
    def __init__(self, tasklist: list[ScheduleItem] = [], startTime=None, endTime=None) -> None:
        self.tasklist = tasklist
        self.start = startTime
        self.end = endTime

    def add_task(self, task: ScheduleItem):
        self.tasklist.append(task)
    
    def add_tasks(self, tasks: list[ScheduleItem]):
        self.tasklist.extend(tasks)
    
    def get_task(self, index: int) -> ScheduleItem:
        return self.tasklist[index]
    
    def get_tasks(self) -> list[ScheduleItem]:
        return self.tasklist
    
    def naive_serialize(self) -> str:
        return str(json.dumps(self.tasklist))
    
    def contains_anything(self):
        for task in self.tasklist:
            if len(task.items) > 0:
                return True
            
    def percentage_requiring_compute(self):
        total_tasks = len(self.tasklist)
        if total_tasks == 0:
            return 0
        compute_tasks = sum(len(task.items[0]) > 0 for task in self.tasklist)
        return compute_tasks / total_tasks
    
    def toQueue(self):
        return deque(self.tasklist)
    
    @classmethod
    def naive_deserialize(cls, data: str) -> 'Schedule':
        return Schedule(json.loads(data))
    


    
