"""
RIS Environment Simulator.
Generates realistic channel state information based on physical models.
"""

import numpy as np
from typing import Dict, Any

class RISEnvironment:
    """
    Simulates the RIS communication environment and generates realistic CSI.
    """
    
    def __init__(self):
        """Initialize the RIS environment."""
        self.c = 3e8  # Speed of light
    
    def generate_csi(self, scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate channel state information based on scenario parameters.
        
        Args:
            scenario_params: Dictionary containing scenario configuration
            
        Returns:
            Dictionary containing CSI components
        """
        try:
            # Extract parameters
            system_setup = scenario_params.get('system_setup', 'MISO')
            channel_environment = scenario_params.get('channel_environment', ['Rician Fading'])
            physical_params = scenario_params.get('physical_params', {})
            
            # Default physical parameters
            num_bs_antennas = physical_params.get('num_antennas_bs', 8)
            num_user_antennas = physical_params.get('num_antennas_user', 4)
            num_ris_elements = physical_params.get('num_ris_elements', 64)
            carrier_frequency = physical_params.get('carrier_frequency', 28e9)
            distance_bs_ris = physical_params.get('distance_bs_ris', 50)
            distance_ris_user = physical_params.get('distance_ris_user', 10)
            noise_power_dbm = physical_params.get('noise_power_dbm', -80)
            
            # Adjust antenna counts based on system setup
            if system_setup == 'MISO':
                num_user_antennas = 1
            
            # Calculate wavelength
            wavelength = self.c / carrier_frequency
            
            # Generate channels based on environment
            if 'Rician Fading' in channel_environment:
                H_br, H_ru = self._generate_rician_channels(
                    num_ris_elements, num_bs_antennas, num_user_antennas,
                    distance_bs_ris, distance_ris_user, wavelength, physical_params
                )
            elif 'Rayleigh Fading' in channel_environment:
                H_br, H_ru = self._generate_rayleigh_channels(
                    num_ris_elements, num_bs_antennas, num_user_antennas
                )
            elif any(env in ['mmWave Channels', 'THz Channels'] for env in channel_environment):
                H_br, H_ru = self._generate_sparse_channels(
                    num_ris_elements, num_bs_antennas, num_user_antennas,
                    wavelength, 'mmWave' if 'mmWave' in str(channel_environment) else 'THz'
                )
            else:
                # Default fallback
                H_br, H_ru = self._generate_rayleigh_channels(
                    num_ris_elements, num_bs_antennas, num_user_antennas
                )
            
            # Add high-mobility effects if specified
            if any('High-Mobility' in str(char) for char in scenario_params.get('key_characteristics', [])):
                H_br, H_ru = self._add_mobility_effects(H_br, H_ru, carrier_frequency)
            
            # Add channel aging if specified
            if 'Channel Aging' in str(channel_environment):
                H_br, H_ru = self._add_channel_aging(H_br, H_ru)
            
            # Calculate noise power
            noise_power = 10**(noise_power_dbm / 10) * 1e-3  # Convert dBm to Watts
            
            return {
                'H_br': H_br,  # RIS-to-BS channel
                'H_ru': H_ru,  # User-to-RIS channel
                'noise_power': noise_power,
                'carrier_frequency': carrier_frequency,
                'num_antennas_bs': num_bs_antennas,
                'num_antennas_user': num_user_antennas,
                'num_ris_elements': num_ris_elements,
                'wavelength': wavelength,
                'distance_bs_ris': distance_bs_ris,
                'distance_ris_user': distance_ris_user
            }
            
        except Exception as e:
            print(f"Error generating CSI: {e}")
            return self._generate_fallback_csi()
    
    def _generate_rician_channels(self, num_ris_elements: int, num_bs_antennas: int, 
                                 num_user_antennas: int, distance_bs_ris: float, 
                                 distance_ris_user: float, wavelength: float, 
                                 physical_params: Dict[str, Any]) -> tuple:
        """
        Generate Rician fading channels with LOS component.
        """
        rician_k_factor_db = physical_params.get('rician_k_factor', 5)  # dB
        rician_k_linear = 10**(rician_k_factor_db / 10)
        
        # BS-to-RIS channel (H_br)
        # LOS component
        H_br_los = self._generate_los_channel(num_ris_elements, num_bs_antennas, 
                                             distance_bs_ris, wavelength)
        
        # NLOS component (Rayleigh)
        H_br_nlos = (np.random.randn(num_ris_elements, num_bs_antennas) + 
                    1j * np.random.randn(num_ris_elements, num_bs_antennas)) / np.sqrt(2)
        
        # Combine LOS and NLOS with Rician K-factor
        H_br = np.sqrt(rician_k_linear / (rician_k_linear + 1)) * H_br_los + \
               np.sqrt(1 / (rician_k_linear + 1)) * H_br_nlos
        
        # User-to-RIS channel (H_ru)  
        # LOS component
        H_ru_los = self._generate_los_channel(num_user_antennas, num_ris_elements, 
                                             distance_ris_user, wavelength)
        
        # NLOS component
        H_ru_nlos = (np.random.randn(num_user_antennas, num_ris_elements) + 
                    1j * np.random.randn(num_user_antennas, num_ris_elements)) / np.sqrt(2)
        
        # Combine LOS and NLOS
        H_ru = np.sqrt(rician_k_linear / (rician_k_linear + 1)) * H_ru_los + \
               np.sqrt(1 / (rician_k_linear + 1)) * H_ru_nlos
        
        # Add path loss
        path_loss_br = self._calculate_path_loss(distance_bs_ris, wavelength)
        path_loss_ru = self._calculate_path_loss(distance_ris_user, wavelength)
        
        H_br *= np.sqrt(path_loss_br)
        H_ru *= np.sqrt(path_loss_ru)
        
        return H_br, H_ru
    
    def _generate_rayleigh_channels(self, num_ris_elements: int, num_bs_antennas: int, 
                                   num_user_antennas: int) -> tuple:
        """Generate Rayleigh fading channels (pure NLOS)."""
        # Complex Gaussian channels
        H_br = (np.random.randn(num_ris_elements, num_bs_antennas) + 
               1j * np.random.randn(num_ris_elements, num_bs_antennas)) / np.sqrt(2)
        
        H_ru = (np.random.randn(num_user_antennas, num_ris_elements) + 
               1j * np.random.randn(num_user_antennas, num_ris_elements)) / np.sqrt(2)
        
        return H_br, H_ru
    
    def _generate_sparse_channels(self, num_ris_elements: int, num_bs_antennas: int, 
                                 num_user_antennas: int, wavelength: float, 
                                 channel_type: str) -> tuple:
        """Generate sparse channels for mmWave/THz communications."""
        # Number of dominant paths (sparse)
        if channel_type == 'mmWave':
            num_paths_br = min(3, num_bs_antennas)  # Typically 1-3 paths
            num_paths_ru = min(2, num_user_antennas)
        else:  # THz
            num_paths_br = min(2, num_bs_antennas)  # Even sparser
            num_paths_ru = 1
        
        # Initialize channels
        H_br = np.zeros((num_ris_elements, num_bs_antennas), dtype=complex)
        H_ru = np.zeros((num_user_antennas, num_ris_elements), dtype=complex)
        
        # Generate sparse paths for H_br
        for path in range(num_paths_br):
            # Random angles of arrival/departure
            aoa = np.random.uniform(-np.pi/2, np.pi/2)  # Angle of arrival
            aod = np.random.uniform(-np.pi/2, np.pi/2)  # Angle of departure
            
            # Array response vectors (simplified uniform linear array)
            a_rx = np.exp(1j * 2 * np.pi * np.arange(num_ris_elements) * 0.5 * np.sin(aoa))
            a_tx = np.exp(1j * 2 * np.pi * np.arange(num_bs_antennas) * 0.5 * np.sin(aod))
            
            # Path gain (complex Gaussian)
            path_gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            
            # Add path contribution
            H_br += path_gain * np.outer(a_rx, a_tx.conj())
        
        # Generate sparse paths for H_ru
        for path in range(num_paths_ru):
            aoa = np.random.uniform(-np.pi/2, np.pi/2)
            aod = np.random.uniform(-np.pi/2, np.pi/2)
            
            a_rx = np.exp(1j * 2 * np.pi * np.arange(num_user_antennas) * 0.5 * np.sin(aoa))
            a_tx = np.exp(1j * 2 * np.pi * np.arange(num_ris_elements) * 0.5 * np.sin(aod))
            
            path_gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            
            H_ru += path_gain * np.outer(a_rx, a_tx.conj())
        
        return H_br, H_ru
    
    def _generate_los_channel(self, num_rx: int, num_tx: int, distance: float, wavelength: float) -> np.ndarray:
        """Generate line-of-sight channel matrix."""
        # Uniform linear array response
        k = 2 * np.pi / wavelength
        
        # Assume broadside transmission for simplicity
        phase_shift_rx = k * 0.5 * np.arange(num_rx)
        phase_shift_tx = k * 0.5 * np.arange(num_tx)
        
        # LOS channel matrix
        H_los = np.exp(1j * np.outer(phase_shift_rx, np.ones(num_tx))) * \
                np.exp(1j * np.outer(np.ones(num_rx), phase_shift_tx))
        
        return H_los
    
    def _calculate_path_loss(self, distance: float, wavelength: float) -> float:
        """Calculate free-space path loss."""
        return (wavelength / (4 * np.pi * distance))**2
    
    def _add_mobility_effects(self, H_br: np.ndarray, H_ru: np.ndarray, carrier_frequency: float) -> tuple:
        """Add Doppler effects for high-mobility scenarios."""
        # Simulate Doppler shift with random velocity
        velocity = np.random.uniform(5, 30)  # 5-30 m/s
        doppler_frequency = velocity * carrier_frequency / self.c
        
        # Add phase evolution due to Doppler
        time_evolution = np.exp(1j * 2 * np.pi * doppler_frequency * 0.001)  # 1ms evolution
        
        H_br_mobile = H_br * time_evolution
        H_ru_mobile = H_ru * time_evolution
        
        return H_br_mobile, H_ru_mobile
    
    def _add_channel_aging(self, H_br: np.ndarray, H_ru: np.ndarray) -> tuple:
        """Add channel aging effects."""
        # Add small random perturbations to simulate aging
        aging_factor = 0.05  # 5% aging
        
        H_br_aged = H_br + aging_factor * (np.random.randn(*H_br.shape) + 
                                          1j * np.random.randn(*H_br.shape)) / np.sqrt(2)
        H_ru_aged = H_ru + aging_factor * (np.random.randn(*H_ru.shape) + 
                                          1j * np.random.randn(*H_ru.shape)) / np.sqrt(2)
        
        return H_br_aged, H_ru_aged
    
    def _generate_fallback_csi(self) -> Dict[str, Any]:
        """Generate basic fallback CSI data."""
        num_bs_antennas = 4
        num_user_antennas = 2
        num_ris_elements = 32
        
        H_br = (np.random.randn(num_ris_elements, num_bs_antennas) + 
                1j * np.random.randn(num_ris_elements, num_bs_antennas)) / np.sqrt(2)
        H_ru = (np.random.randn(num_user_antennas, num_ris_elements) + 
                1j * np.random.randn(num_user_antennas, num_ris_elements)) / np.sqrt(2)
        
        return {
            'H_br': H_br,
            'H_ru': H_ru,
            'noise_power': 1e-10,
            'carrier_frequency': 28e9,
            'num_antennas_bs': num_bs_antennas,
            'num_antennas_user': num_user_antennas,
            'num_ris_elements': num_ris_elements,
            'wavelength': 0.0107,  # 28 GHz wavelength
            'distance_bs_ris': 50,
            'distance_ris_user': 10
        }
