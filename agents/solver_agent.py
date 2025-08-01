"""Agent responsible for executing RIS optimization algorithms."""

import numpy as np
import logging
from typing import Dict, Any, List
from utils.ris_algorithms import RISAlgorithms

class SolverAgent:
    def __init__(self):
        self.algorithms = RISAlgorithms()
        self.logger = logging.getLogger(__name__)
        
    def execute_algorithm(self, algorithm_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specified algorithm."""
        try:
            # Convert input to complex arrays
            h_d = complex(input_data['direct_channel_real'], input_data['direct_channel_imag'])
            
            H_br = np.array([
                complex(r, i) for r, i in zip(
                    input_data['bs_ris_channel_real'], 
                    input_data['bs_ris_channel_imag']
                )
            ]).reshape(1, -1)  # 1 x N for BS-RIS
            
            h_ru = np.array([
                complex(r, i) for r, i in zip(
                    input_data['ris_user_channel_real'], 
                    input_data['ris_user_channel_imag']
                )
            ])  # N x 1 for RIS-User
            
            N = input_data['num_ris_elements']
            transmit_power = 1.0  # Fixed transmit power
            
            # Execute algorithm
            if algorithm_name == 'Gradient Descent':
                phases, _, _, _ = self.algorithms.gradient_descent_adam(
                    h_d, H_br, h_ru, N, transmit_power
                )
            elif algorithm_name == 'Manifold Optimization':
                phases, _, _, _ = self.algorithms.manifold_optimization_adam(
                    h_d, H_br, h_ru, N, transmit_power
                )
            elif algorithm_name == 'Alternating Optimization':
                phases, _, _, _ = self.algorithms.alternating_optimization(
                    h_d, H_br, h_ru, N, transmit_power
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            # Calculate performance
            snr = self.calculate_snr(phases.tolist(), input_data)
            
            return {
                'phase_shifts': phases.tolist(),
                'snr': snr
            }
            
        except Exception as e:
            self.logger.error(f"Algorithm execution failed: {e}")
            # Return random phases as fallback
            N = input_data['num_ris_elements']
            random_phases = np.random.uniform(0, 2*np.pi, N).tolist()
            return {
                'phase_shifts': random_phases,
                'snr': self.calculate_snr(random_phases, input_data)
            }
    
    def calculate_snr(self, phase_shifts: List[float], input_data: Dict[str, Any]) -> float:
        """Calculate SNR for given phase shifts."""
        try:
            # Convert to complex arrays
            h_d = complex(input_data['direct_channel_real'], input_data['direct_channel_imag'])
            
            H_br = np.array([
                complex(r, i) for r, i in zip(
                    input_data['bs_ris_channel_real'], 
                    input_data['bs_ris_channel_imag']
                )
            ])
            
            h_ru = np.array([
                complex(r, i) for r, i in zip(
                    input_data['ris_user_channel_real'], 
                    input_data['ris_user_channel_imag']
                )
            ])
            
            # Create RIS reflection coefficients
            e = np.exp(1j * np.array(phase_shifts))
            
            # Calculate combined channel
            combined = h_d + np.dot(H_br, e * h_ru)
            
            # Calculate SNR
            signal_power = abs(combined)**2
            noise_power = 1e-12  # Fixed noise power
            snr_linear = signal_power / noise_power
            
            return float(10 * np.log10(snr_linear))
            
        except Exception as e:
            self.logger.error(f"SNR calculation failed: {e}")
            return -20.0  # Return poor SNR on error
