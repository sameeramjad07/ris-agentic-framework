"""Central coordinator for the RIS optimization process."""

import logging
from typing import Dict, List, Any
from agents.optimizer_agent import OptimizerAgent
from agents.solver_agent import SolverAgent

class CoordinatorAgent:
    def __init__(self, optimizer_agent: OptimizerAgent, solver_agent: SolverAgent):
        self.optimizer_agent = optimizer_agent
        self.solver_agent = solver_agent
        self.logger = logging.getLogger(__name__)
        
    async def process_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single scenario through the agentic pipeline."""
        try:
            # Extract input features
            scenario_features = self._extract_features(scenario['input'])
            
            # Get algorithm recommendation from optimizer agent
            algorithm_name, reasoning = await self.optimizer_agent.select_algorithm(scenario_features)
            
            # Execute all algorithms for comparison
            results = {}
            
            # Execute agentic choice
            agentic_result = self.solver_agent.execute_algorithm(
                algorithm_name, scenario['input']
            )
            results['agentic'] = {
                'algorithm': algorithm_name,
                'reasoning': reasoning,
                **agentic_result
            }
            
            # Execute all algorithms for comparison
            algorithms = ['Gradient Descent', 'Manifold Optimization', 'Alternating Optimization']
            for alg in algorithms:
                if alg != algorithm_name:  # Don't repeat agentic choice
                    result = self.solver_agent.execute_algorithm(alg, scenario['input'])
                    results[alg.lower().replace(' ', '_')] = result
            
            # Add analytical solution (ground truth)
            results['analytical'] = {
                'phase_shifts': scenario['output']['optimized_phase_shifts'],
                'snr': self.solver_agent.calculate_snr(
                    scenario['output']['optimized_phase_shifts'], 
                    scenario['input']
                )
            }
            
            # Add random baseline
            import numpy as np
            num_elements = scenario['input']['num_ris_elements']
            random_phases = np.random.uniform(0, 2*np.pi, num_elements).tolist()
            results['random'] = {
                'phase_shifts': random_phases,
                'snr': self.solver_agent.calculate_snr(random_phases, scenario['input'])
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing scenario: {e}")
            raise
    
    def _extract_features(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract derived features from input data."""
        import numpy as np
        
        # Convert channel data to complex arrays
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
        
        # Calculate derived features
        features = {
            'direct_channel_norm': abs(h_d),
            'G_norm': np.linalg.norm(H_br),
            'hr_norm': np.linalg.norm(h_ru),
            'phase_alignment_score': self._calculate_phase_alignment(H_br, h_ru),
            'estimated_snr': self._estimate_snr(h_d, H_br, h_ru),
            'num_ris_elements': input_data['num_ris_elements']
        }
        
        return features
    
    def _calculate_phase_alignment(self, H_br: np.ndarray, h_ru: np.ndarray) -> float:
        """Calculate how well the phases align for constructive interference."""
        cascaded = H_br * h_ru
        alignment = abs(np.sum(cascaded)) / (np.linalg.norm(cascaded) * len(cascaded))
        return float(alignment)
    
    def _estimate_snr(self, h_d: complex, H_br: np.ndarray, h_ru: np.ndarray) -> float:
        """Estimate the system SNR in dB."""
        signal_power = abs(h_d + np.sum(H_br * h_ru))**2
        noise_power = 1e-12  # Default noise power
        snr_linear = signal_power / noise_power
        return float(10 * np.log10(snr_linear))
