import time




class Timer():
    def __init__(self, what=''):
        self.start = time.time()
        self.what = what
        self.period = 0
        self.running_avg = 0
    

    def reset(self):
        self.period = time.time() - self.start

        if self.running_avg == 0: 
            self.running_avg = self.period
        else:
            self.running_avg = (self.running_avg + self.period)/2
        
        self.start = time.time()


    def get_log_message(self):
        return f'{self.what} took {self.period/60.0:.2f} min ({self.period} sec)'