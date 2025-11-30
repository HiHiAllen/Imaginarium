import os
from omegaconf import OmegaConf, DictConfig

class Config:
    """
    Centralized Configuration Management.
    集中式配置管理。
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.cfg = OmegaConf.load(config_path)
        self.root_dir = os.path.abspath(os.getcwd())
    
    @property
    def shared(self) -> DictConfig:
        return self.cfg.get('shared', {})
        
    def get(self, key: str, default=None):
        return self.cfg.get(key, default)

