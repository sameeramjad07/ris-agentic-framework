"""Configuration settings for the simplified RIS Agentic System."""

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        # LLM Configuration
        self.CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
        self.LLM_MODEL_NAME = "llama3.1-8b"
        
        # RIS System Parameters
        self.TRANSMIT_POWER_DEFAULT = 1.0  # Watts
        self.NOISE_POWER_DEFAULT = 1e-12   # Watts
        
        # Dataset Generation Defaults
        self.DEFAULT_NUM_ELEMENTS = [8, 16, 64]
        self.DEFAULT_NUM_SCENARIOS = 10
        
        # Optimization Parameters
        self.MAX_ITERATIONS = 1000
        self.LEARNING_RATE = 0.01
        self.TOLERANCE = 1e-6
