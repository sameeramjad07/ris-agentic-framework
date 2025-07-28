import numpy as np
import logging
from typing import Dict, Any

class PerformanceMetrics:
    """
    Performance metrics calculator for RIS communication systems.
    Computes SNR, sum rate, beamforming gain, and energy efficiency.
    """
    
    def __init__(self):
        """Initialize performance metrics calculator."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_snr(self, optimized_phases: np.ndarray, csi_data: Dict[str, Any]) -> float:
        """
        Calculate Signal-to-Noise Ratio for the RIS system.
        
        Args:
            optimized_phases: Complex-valued RIS phase shifts
            csi_data: Channel State Information dictionary
            
        Returns:
            SNR in dB
        """
        try:
            # Extract channel matrices and system parameters
            H_br = csi_data.get("H_br")  # BS-RIS channel
            H_ru = csi_data.get("H_ru")  # RIS-User channel
            H_direct = csi_data.get("H_direct")  # Direct BS-User channel
            noise_power = csi_data.get("noise_power", 1e-9)
            
            if H_br is None or H_ru is None:
                # Use simplified calculation if full CSI not available
                return self._calculate_simplified_snr(optimized_phases, csi_data)
            
            # Create RIS reflection matrix
            Phi = np.diag(optimized_phases)
            
            # Calculate effective channel
            H_effective = H_direct + H_br.T @ Phi @ H_ru
            
            # Calculate received signal power
            signal_power = np.abs(H_effective)**2
            
            # SNR calculation
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(np.real(snr_linear))
            
            self.logger.debug(f"Calculated SNR: {snr_db:.2f} dB")
            return float(snr_db)
            
        except Exception as e:
            self.logger.warning(f"SNR calculation failed: {e}, using simplified method")
            return self._calculate_simplified_snr(optimized_phases, csi_data)
    
    def calculate_sum_rate(self, optimized_phases: np.ndarray, csi_data: Dict[str, Any]) -> float:
        """
        Calculate sum rate (capacity) for the RIS system.
        
        Args:
            optimized_phases: Complex-valued RIS phase shifts
            csi_data: Channel State Information dictionary
            
        Returns:
            Sum rate in bits per second
        """
        try:
            # Calculate SNR first
            snr_db = self.calculate_snr(optimized_phases, csi_data)
            snr_linear = 10**(snr_db / 10)
            
            # Shannon capacity formula: C = log2(1 + SNR)
            bandwidth = csi_data.get("bandwidth", 20e6)  # Default 20 MHz
            sum_rate = bandwidth * np.log2(1 + snr_linear)
            
            self.logger.debug(f"Calculated sum rate: {sum_rate:.2e} bps")
            return float(sum_rate)
            
        except Exception as e:
            self.logger.warning(f"Sum rate calculation failed: {e}")
            return 1e5  # Fallback value
    
    def calculate_beamforming_gain(self, optimized_phases: np.ndarray, csi_data: Dict[str, Any]) -> float:
        """
        Calculate beamforming gain provided by the RIS.
        
        Args:
            optimized_phases: Complex-valued RIS phase shifts
            csi_data: Channel State Information dictionary
            
        Returns:
            Beamforming gain in dB
        """
        try:
            # Compare with random phases
            num_elements = len(optimized_phases)
            random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, num_elements))
            
            # Calculate power with optimized vs random phases
            snr_optimized = self.calculate_snr(optimized_phases, csi_data)
            snr_random = self.calculate_snr(random_phases, csi_data)
            
            beamforming_gain = snr_optimized - snr_random
            
            self.logger.debug(f"Calculated beamforming gain: {beamforming_gain:.2f} dB")
            return float(beamforming_gain)
            
        except Exception as e:
            self.logger.warning(f"Beamforming gain calculation failed: {e}")
            return 5.0  # Typical gain
    
    def calculate_energy_efficiency(self, optimized_phases: np.ndarray, csi_data: Dict[str, Any]) -> float:
        """
        Calculate energy efficiency of the RIS system.
        
        Args:
            optimized_phases: Complex-valued RIS phase shifts
            csi_data: Channel State Information dictionary
            
        Returns:
            Energy efficiency in bits/Joule
        """
        try:
            # Calculate sum rate
            sum_rate = self.calculate_sum_rate(optimized_phases, csi_data)
            
            # Estimate power consumption
            num_elements = len(optimized_phases)
            power_per_element = 1e-3  # 1 mW per element (typical)
            ris_power = num_elements * power_per_element
            
            # Add base station power
            bs_power = csi_data.get("bs_transmit_power", 1.0)  # 1W default
            
            total_power = ris_power + bs_power
            
            # Energy efficiency = throughput / power
            energy_efficiency = sum_rate / total_power
            
            self.logger.debug(f"Calculated energy efficiency: {energy_efficiency:.2e} bits/Joule")
            return float(energy_efficiency)
            
        except Exception as e:
            self.logger.warning(f"Energy efficiency calculation failed: {e}")
            return 1e8  # Fallback value
    
    def _calculate_simplified_snr(self, optimized_phases: np.ndarray, csi_data: Dict[str, Any]) -> float:
        """
        Simplified SNR calculation when full CSI is not available.
        
        Args:
            optimized_phases: Complex-valued RIS phase shifts
            csi_data: Channel State Information dictionary
            
        Returns:
            Simplified SNR in dB
        """
        try:
            # Use phase coherence as a proxy for channel alignment
            phase_coherence = np.abs(np.sum(optimized_phases)) / len(optimized_phases)
            
            # Map coherence to SNR (heuristic)
            base_snr = csi_data.get("base_snr_db", 0)  # Base SNR without RIS
            ris_gain = 10 * np.log10(phase_coherence * len(optimized_phases))
            
            snr_db = base_snr + ris_gain
            
            return float(snr_db)
            
        except Exception:
            return 0.0  # Worst case fallback
