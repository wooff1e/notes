import time




class Logger():
    def __init__(self, log_path):
        self.log_path = str(log_path)
        self.timers = {}

        with open(log_path, "w") as log_file:
            log_file.write(f'{time.strftime("%c")}\n')


    def log(self, line):
        print(line)
        with open(self.log_path, "a") as log_file:
            log_file.write(f'{line}\n')


    def log_losses(self, losses):
        message = ''
        for a, b in losses.items():
            message += f'{a}: {b:.3f} '
        
        self.log(message)   