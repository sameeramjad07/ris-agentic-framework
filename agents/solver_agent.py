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
            # Convert input to complex arrays with correct dimensions
            h_d = complex(input_data['direct_channel_real'], input_data['direct_channel_imag'])
            
            # BS-RIS channel: should be 1 x N (1 BS antenna, N RIS elements)
            H_br = np.array([
                complex(r, i) for r, i in zip(
                    input_data['bs_ris_channel_real'], 
                    input_data['bs_ris_channel_imag']
                )
            ])  # Shape: (N,)
            
            # RIS-User channel: should be N x 1 (N RIS elements, 1 user)
            h_ru = np.array([
                complex(r, i) for r, i in zip(
                    input_data['ris_user_channel_real'], 
                    input_data['ris_user_channel_imag']
                )
            ])  # Shape: (N,)
            
            N = input_data['num_ris_elements']
            transmit_power = 1.0  # Fixed transmit power
            
            # Validate dimensions
            if len(H_br) != N or len(h_ru) != N:
                raise ValueError(f"Channel dimension mismatch: H_br={len(H_br)}, h_ru={len(h_ru)}, N={N}")
            
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
            
            # Ensure phases is a numpy array
            if not isinstance(phases, np.ndarray):
                phases = np.array(phases)
            
            # Calculate performance
            snr = self.calculate_snr(phases.tolist(), input_data)
            
            # Validate results
            if not np.isfinite(snr) or snr < -100:
                self.logger.warning(f"Invalid SNR {snr} for {algorithm_name}, using fallback")
                raise ValueError("Invalid SNR computed")
            
            return {
                'phase_shifts': phases.tolist(),
                'snr': snr
            }
            
        except Exception as e:
            self.logger.error(f"Algorithm execution failed for {algorithm_name}: {e}")
            # Return random phases as fallback
            N = input_data['num_ris_elements']
            random_phases = np.random.uniform(0, 2*np.pi, N)
            fallback_snr = self.calculate_snr(random_phases.tolist(), input_data)
            return {
                'phase_shifts': random_phases.tolist(),
                'snr': fallback_snr
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
            
            # Validate inputs
            if len(phase_shifts) != len(H_br) or len(phase_shifts) != len(h_ru):
                raise ValueError("Phase shifts dimension mismatch")
            
            # Create RIS reflection coefficients
            e = np.exp(1j * np.array(phase_shifts))
            
            # Calculate combined channel: h_d + H_br * diag(e) * h_ru
            # For single-user case: h_d + sum(H_br[i] * e[i] * h_ru[i])
            ris_contribution = np.sum(H_br * e * h_ru)
            combined = h_d + ris_contribution
            
            # Calculate SNR
            signal_power = abs(combined)**2
            noise_power = 1e-12  # Fixed noise power
            
            if signal_power <= 0:
                return -100.0  # Very poor SNR
                
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            
            # Sanity check
            if not np.isfinite(snr_db):
                return -100.0
                
            return float(snr_db)
            
        except Exception as e:
            self.logger.error(f"SNR calculation failed: {e}")
            return -100.0  # Return very poor SNR on error