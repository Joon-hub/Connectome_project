# import libraries 
import numpy as np
from typing import Dict 

# define utility functions

def print_section(title: str, width: int = 50):
    '''Print section header'''
    print(f"\\n" + "=" * width)
    print(title.center(width))
    print("=" * width)
    
def print_results(results: Dict):
    '''Print results dictionary in formatted way'''
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, list):
            print(f"{key}: {[f'{v:.4f}' for v in value]}")
        else:
            print(f"{key}: {value}")
            
            
            