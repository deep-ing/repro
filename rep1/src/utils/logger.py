import logging 
import os 


from torch.utils.tensorboard import SummaryWriter

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class PlatformLogger():
    def __init__(self, result_path):
        self.iteration_logger = setup_logger('iteration', os.path.join(result_path, "log_iteration.log"))
        self.agent_logger = setup_logger('agent', os.path.join(result_path, "log_agent.log"))
        self.info_logger = setup_logger("info", os.path.join(result_path, "log_info.log"))
        self.iteration_log_count = 0 
        self.agent_log_count = 0
        self.writer = SummaryWriter(log_dir=os.path.join(result_path, 'runs'))          

    def log_iteration(self, dict):
        lst = []
        for  k,v in dict.items():
            if isinstance(v, float):
                lst.append(f"{k}:{v:.3f}")
                self.writer.add_scalar(f'Environment/{k}', v, self.iteration_log_count)
            else:
                lst.append(f"{k}:{v}")
            
        string =  f"{str(self.iteration_log_count)} | " +  "".join(" | ".join(lst))
        self.iteration_logger.info(string)
        self.iteration_log_count += 1 
        
    def log_info(self ,string):
        self.info_logger(string)

    def log_agent(self, dict):
        lst = []
        for  k,v in dict.items():
            if isinstance(v, float):
                lst.append(f"{k}:{v:.3f}")
                self.writer.add_scalar(f'Train/{k}', v,self.agent_log_count)
                
            else:
                lst.append(f"{k}:{v}")
        string = f"{str(self.agent_log_count)} | " + "".join(" | ".join(lst))
        self.agent_logger.info(string)
        self.agent_log_count += 1

