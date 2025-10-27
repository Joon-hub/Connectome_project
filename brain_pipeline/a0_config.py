import yaml
from pathlib import Path
from typing import Dict, Any

# Load YAML file as python dictionary
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    '''Load configuration from YAML file'''
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Config:
    '''Configuration class for easy access to settings'''
    
    def __init__(self, config_path: str = "config.yaml"):
        # 
        self.config = load_config(config_path)
        self._setup_paths()
    
    # Create directories
    def _setup_paths(self):
        '''Create directories if they don't exist'''
        for key in ['raw_dir', 'processed_dir', 'results_dir']:
            path = Path(self.config['data'][key])
            path.mkdir(parents=True, exist_ok=True)
    
    # Get nested config value
    def get(self, *keys, fallback=None):
        '''Get nested config value with optional fallback'''
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return fallback
