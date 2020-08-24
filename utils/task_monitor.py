import time
import ee

ee.Initialize()

class GEETaskMonitor():
    
    def __init__(self, max_tasks=2500):
        self.task_dict = {}
        self.max_tasks = max_tasks
        self.failed_tasks = []
        return None

    def add_task(self, task_id, task):
        '''Add a task to the task dictionary'''
        self.task_dict[str(task_id)] = task
        return None
     
    def reset_monitor(self):
        '''Reset the task dictionary (removes all tasks)'''
        self.task_dict = {}
        return None 
    
    def get_failed_tasks(self):
        return self.failed_tasks
    
    def get_monitor_capacity(self):
        '''Computes the capacity (% of monitor's max capacity'''
        return len(self.task_dict) / self.max_tasks
        
    def monitor_tasks(self):
        '''
        Monitors the status of tasks being executed on the GEE server.
        Returns a dictionary of input key-value pairs which failed.
        '''
        while True:
            self.check_status()
            if len(self.task_dict) == 0:
                break
            time.sleep(5)
        return None

    def check_status(self):
        '''
        Remove tasks which completed -- successfully or insuccessfully
        '''
        del_keys = []           
        for key, value in self.task_dict.items():
            try:
                state = ee.data.getTaskStatus(value.id)[0]['state']
                if state in ['READY', 'RUNNING']:
                    pass
                else:
                    if state == "FAILED":
                        self.failed_tasks.append(key)
                    del_keys.append(key)
            except Exception:
                pass
        for i in del_keys:
            del self.task_dict[i]

        return None