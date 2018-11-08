import time

class Timer(object):
    def __init__(self):
        self.start_time = 0.0
        self.call_times = 0
        self.total_time = 0.0
        self.average_time = 0.0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.total_time = self.total_time + time.time() - self.start_time
        self.call_times += 1
        self.average_time = self.total_time / float(self.call_times)