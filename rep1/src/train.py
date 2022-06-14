from agent.CRAR  import CRAR 
from omegaconf import OmegaConf

flags = OmegaConf.load("config.yml")

output_dir = "results/"

agent = CRAR(None, None, flags)

agent.act(None)

