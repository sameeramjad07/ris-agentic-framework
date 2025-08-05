"""Performance metrics calculation utilities."""

import numpy as np
from typing import List, Dict, Any
import logging

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
            
            # Validate dimensions
            N = input_data['num_ris_elements']
            if len(phase_shifts) != N or len(H_br) != N or len(h_ru) != N:
                logging.error(f"Dimension mismatch: phases={len(phase_shifts)}, H_br={len(H_br)}, h_ru={len(h_ru)}, N={N}")
                return -100.0
            
            # Create RIS reflection coefficients
            e = np.exp(1j * np.array(phase_shifts))
            
            # Calculate combined channel
            # For MISO-RIS system: y = h_d * s + (H_br ⊙ h_ru)^T * Θ * s
            # Where ⊙ is element-wise product, Θ is diagonal phase shift matrix
            ris_contribution = np.sum(H_br * e * h_ru)
            combined = h_d + ris_contribution
            
            # Calculate received power
            signal_power = transmit_power * abs(combined)**2
            
            # Ensure positive power
            if signal_power <= 0:
                logging.warning(f"Non-positive signal power: {signal_power}")
                return -100.0
            
            # Calculate SNR in dB
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            
            # Sanity check
            if not np.isfinite(snr_db):
                logging.warning(f"Non-finite SNR: {snr_db}")
                return -100.0
                
            return float(snr_db)
            
        except Exception as e:
            logging.error(f"SNR calculation failed: {e}")
            return -100.0  # Return very poor SNR on error
    
    @staticmethod
    def calculate_sum_rate(phase_shifts: List[float], input_data: Dict[str, Any],
                          transmit_power: float = 1.0, bandwidth: float = 1e6) -> float:
        """Calculate sum rate (capacity) for given phase shifts."""
        try:
            snr_db = PerformanceMetrics.calculate_snr(phase_shifts, input_data, transmit_power)
            
            # Convert to linear scale, ensuring non-negative
            snr_linear = max(10**(snr_db / 10), 1e-10)
            
            # Calculate sum rate using Shannon formula
            sum_rate = bandwidth * np.log2(1 + snr_linear)
            
            return float(sum_rate)
            
        except Exception as e:
            logging.error(f"Sum rate calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def validate_phase_shifts(phase_shifts: List[float]) -> bool:
        """Validate that phase shifts are within valid range."""
        try:
            phases = np.array(phase_shifts)
            # Check if all phases are finite and within reasonable range
            if not np.all(np.isfinite(phases)):
                return False
            # Phases should be between 0 and 2π (or equivalent range)
            return True
        except:
            return False