"""
Configuration settings for the RIS Agentic System.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AppSettings:
    """
    Configuration settings for the RIS Agentic System.
    Contains API keys, model configurations, and simulation parameters.
    """
    
    def __init__(self):
        # Cerebras API Configuration
        self.CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
        self.LLM_MODEL_NAME = "llama3.1-8b"  # Default Cerebras model
        self.CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
        
        # Simulation Parameters
        self.SIMULATION_PARAMS = {
            "num_ris_elements": 64,          # Number of RIS elements
            "carrier_frequency": 28e9,       # 28 GHz for mmWave
            "noise_power_dbm": -90,          # Noise power in dBm
            "num_antennas_bs": 4,            # Base station antennas
            "num_antennas_user": 1,          # User antennas (single user focus)
            "distance_bs_ris": 50,           # BS-RIS distance in meters
            "distance_ris_user": 10,         # RIS-User distance in meters
            "rician_k_factor": 5.0,          # Rician K-factor
            "channel_coherence_time": 100,   # Channel coherence time in ms
        }
        
        # Performance thresholds
        self.PERFORMANCE_THRESHOLDS = {
            "min_snr_db": -10,
            "target_sum_rate_bps": 1e6,
        }
        
    def get_simulation_config(self) -> Dict[str, Any]:
        """Return simulation configuration dictionary"""
        return self.SIMULATION_PARAMS.copy()
