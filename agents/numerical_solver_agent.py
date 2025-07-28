import logging
from typing import Dict, Any
from utils.ris_algorithms import RISAlgorithms
from utils.performance_metrics import PerformanceMetrics

class NumericalSolverAgent:
    """
    Agent responsible for executing RIS algorithms and computing performance metrics.
    Acts as the interface between algorithm selection and numerical computation.
    """
    
    def __init__(self, ris_algorithms: RISAlgorithms, performance_metrics: PerformanceMetrics):
        """
        Initialize the numerical solver agent.
        
        Args:
            ris_algorithms: RISAlgorithms instance containing all algorithm implementations
            performance_metrics: PerformanceMetrics instance for computing system performance
        """
        self.ris_algorithms = ris_algorithms
        self.performance_metrics = performance_metrics
        self.logger = logging.getLogger(__name__)
        
    def execute_algorithm(self, algorithm_name: str, algorithm_params: Dict[str, Any], csi_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Execute the specified RIS algorithm and compute performance metrics.
        
        Args:
            algorithm_name: Name of the algorithm to execute
            algorithm_params: Parameters for the algorithm
            csi_data: Channel State Information data
            
        Returns:
            Dictionary containing performance metrics (SNR, sum rate, etc.)
        """
        self.logger.info(f"Executing algorithm: {algorithm_name}")
        
        try:
            # Get the algorithm function dynamically
            algorithm_function = self.ris_algorithms.get_algorithm(algorithm_name)
            
            if algorithm_function is None:
                raise ValueError(f"Algorithm '{algorithm_name}' not found")
            
            # Execute the algorithm to get optimized RIS phase shifts
            self.logger.debug(f"Running {algorithm_name} with params: {algorithm_params}")
            optimized_phases = algorithm_function(csi_data, algorithm_params)
            
            # Compute performance metrics
            snr = self.performance_metrics.calculate_snr(optimized_phases, csi_data)
            sum_rate = self.performance_metrics.calculate_sum_rate(optimized_phases, csi_data)
            
            # Additional metrics
            beamforming_gain = self.performance_metrics.calculate_beamforming_gain(optimized_phases, csi_data)
            energy_efficiency = self.performance_metrics.calculate_energy_efficiency(optimized_phases, csi_data)
            
            results = {
                "snr": float(snr),
                "sum_rate": float(sum_rate),
                "beamforming_gain": float(beamforming_gain),
                "energy_efficiency": float(energy_efficiency),
                "algorithm_name": algorithm_name,
                "execution_successful": True
            }
            
            self.logger.info(f"Algorithm executed successfully - SNR: {snr:.2f} dB, Sum Rate: {sum_rate:.2e} bps")
            return results
            
        except Exception as e:
            self.logger.error(f"Algorithm execution failed for {algorithm_name}: {e}")
            
            # Return degraded performance for failed algorithms
            return {
                "snr": -20.0,  # Poor SNR
                "sum_rate": 1e4,  # Low sum rate
                "beamforming_gain": 0.1,
                "energy_efficiency": 0.1,
                "algorithm_name": algorithm_name,
                "execution_successful": False,
                "error": str(e)
            }