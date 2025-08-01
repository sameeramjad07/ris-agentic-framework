"""Performance metrics calculation utilities."""

import numpy as np
from typing import List, Dict, Any

class PerformanceMetrics:
    """Calculate various performance metrics for RIS systems."""
    
    @staticmethod
    def calculate_snr(phase_shifts: List[float], input_data: Dict[str, Any], 
                     transmit_power: float = 1.0, noise_power: float = 1e-12) -> float:
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
            signal_power = transmit_power * abs(combined)**2
            snr_linear = signal_power / noise_power
            
            return float(10 * np.log10(snr_linear))
            
        except Exception:
            return -20.0  # Return poor SNR on error
    
    @staticmethod
    def calculate_sum_rate(phase_shifts: List[float], input_data: Dict[str, Any],
                          transmit_power: float = 1.0, bandwidth: float = 1e6) -> float:
        """Calculate sum rate (capacity) for given phase shifts."""
        snr_db = PerformanceMetrics.calculate_snr(phase_shifts, input_data, transmit_power)
        snr_linear = 10**(snr_db / 10)
        sum_rate = bandwidth * np.log2(1 + snr_linear)
        return float(sum_rate)
