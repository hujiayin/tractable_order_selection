import time

class Timer: 
    logs = {}  # static 

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        print(f"[{self.label}] Started")
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        if self.label not in Timer.logs:
            Timer.logs[self.label] = []
        Timer.logs[self.label].append(elapsed)
        
        # print(f"[{self.label}] Completed in {elapsed:.6f} seconds")

