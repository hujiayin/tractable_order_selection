# main.py
import time
from exp_timer import CONFIG, timer, time_block, time_all_methods, TimerConfig


CONFIG.__init__(log_file="exp_time.log", threshold_ms=2.0)

@timer(extra="standalone")
def heavy_fn(n):
    s = 0
    for i in range(n):
        s += i*i
    return s

@time_all_methods(threshold_ms=0.0)
class MyExp:
    def __init__(self, n):
        self.n = n 

    def step1(self):
        time.sleep(0.01)

    @staticmethod
    def s_step():
        time.sleep(0.003)

    @classmethod
    def c_step(cls):
        time.sleep(0.004)

    @timer()
    def run(self):
        with time_block("phase-A"):
            self.step1()
            self.s_step()
            self.c_step()
        with time_block("phase-B", extra=f"n={self.n}"):
            heavy_fn(self.n)

@time_all_methods(threshold_ms=0.0)
class Exp:
    @timer()          
    def step1(self):
        sum(i*i for i in range(10000))

    def step2(self):
        sum(i*i for i in range(10000))

if __name__ == "__main__":
    exp = MyExp(200_000)
    exp.run()


