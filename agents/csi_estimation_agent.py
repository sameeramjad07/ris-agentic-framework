import numpy as np
import logging
from typing import Dict, Any
from simulation.ris_environment import RISEnvironment

class CSIEstimationAgent:
    """
    Agent responsible for Channel State Information (CSI) estimation.
    Generates realistic CSI data based on scenario parameters.
    """
    
    def __init__(self, ris_environment: RISEnvironment):
        """
        Initialize CSI estimation agent.
        
        Args:
            ris_environment: RISEnvironment instance for channel generation
        """
        self.ris_environment = ris_environment
        self.logger = logging.getLogger(__name__)
        
    def estimate_csi(self, scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate Channel State Information for the given scenario.
        
        Args:
            scenario_params: Dictionary containing scenario configuration
            
        Returns:
            Dictionary containing CSI components and metadata
        """
        self.logger.info("Estimating CSI based on scenario parameters")
        
        try:
            # Generate CSI using the RIS environment
            csi_data = self.ris_environment.generate_csi(scenario_params)
            
            # Add estimation metadata
            csi_data.update({
                "estimation_method": self._select_estimation_method(scenario_params),
                "channel_sparsity": self._estimate_channel_sparsity(csi_data),
                "estimation_accuracy": self._estimate_accuracy(scenario_params),
                "timestamp": np.datetime64('now')
            })
            
            self.logger.info(f"CSI estimated with {csi_data['estimation_method']} method")
            self.logger.info(f"Channel sparsity: {csi_data['channel_sparsity']:.3f}")
            
            return csi_data
            
        except Exception as e:
            self.logger.error(f"CSI estimation failed: {e}")
            raise
    
    def _select_estimation_method(self, scenario_params: Dict[str, Any]) -> str:
        """
        Select appropriate CSI estimation method based on scenario characteristics.
        
        Args:
            scenario_params: Scenario configuration
            
        Returns:
            Name of the estimation method
        """
        channel_env = scenario_params.get("channel_environment", [])
        key_chars = scenario_params.get("key_characteristics", [])
        
        if "High-Mobility Scenarios" in key_chars:
            return "Kalman Filtering"
        elif "mmWave Channels" in channel_env or "THz Channels" in channel_env:
            return "Compressed Sensing"
        elif "Rayleigh Fading" in channel_env:
            return "MMSE Estimation"
        else:
            return "Least Squares"
    
    def _estimate_channel_sparsity(self, csi_data: Dict[str, Any]) -> float:
        """
        Estimate the sparsity level of the channel matrices.
        
        Args:
            csi_data: CSI data dictionary
            
        Returns:
            Sparsity ratio (0 to 1, where 1 is fully sparse)
        """
        try:
            H_br = csi_data.get("H_br")
            H_ru = csi_data.get("H_ru")
            
            if H_br is not None and H_ru is not None:
                # Calculate sparsity as ratio of small elements to total elements
                threshold = 0.1 * np.max([np.abs(H_br).max(), np.abs(H_ru).max()])
                
                sparse_br = np.sum(np.abs(H_br) < threshold) / H_br.size
                sparse_ru = np.sum(np.abs(H_ru) < threshold) / H_ru.size
                
                return (sparse_br + sparse_ru) / 2
            else:
                return 0.5  # Default moderate sparsity
                
        except Exception:
            return 0.5  # Default fallback
    
    def _estimate_accuracy(self, scenario_params: Dict[str, Any]) -> float:
        """
        Estimate the accuracy of CSI estimation based on scenario conditions.
        
        Args:
            scenario_params: Scenario configuration
            
        Returns:
            Accuracy score (0 to 1)
        """
        base_accuracy = 0.9
        
        # Reduce accuracy for challenging scenarios
        key_chars = scenario_params.get("key_characteristics", [])
        if "High-Mobility Scenarios" in key_chars:
            base_accuracy -= 0.2
        if "Low/Moderate SNR" in key_chars:
            base_accuracy -= 0.15
        if "channel aging" in str(key_chars).lower():
            base_accuracy -= 0.1
            
        return max(0.5, base_accuracy)  # Minimum 50% accuracy
