import json
import asyncio
import logging
from typing import Dict, Any, Tuple, List
from utils.llm_interface import LLMInterface

class OptimizerAgent:
    """
    Agent responsible for intelligent algorithm selection using LLM reasoning.
    Analyzes scenario characteristics and selects optimal RIS algorithms.
    """
    
    def __init__(self, llm_interface: LLMInterface, algorithms_kb: List[Dict[str, Any]]):
        """
        Initialize the optimizer agent.
        
        Args:
            llm_interface: LLMInterface instance for LLM communication
            algorithms_kb: Knowledge base of RIS algorithms
        """
        self.llm_interface = llm_interface
        self.algorithms_kb = algorithms_kb
        self.logger = logging.getLogger(__name__)
        
    async def select_algorithm(self, csi_data: Dict[str, Any], scenario_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str, str]:
        """
        Select the optimal algorithm using LLM reasoning.
        
        Args:
            csi_data: Channel State Information data
            scenario_params: Scenario configuration parameters
            
        Returns:
            Tuple of (algorithm_name, algorithm_params, reasoning, complexity_note)
        """
        self.logger.info("Starting intelligent algorithm selection")
        
        try:
            # Prepare context for LLM
            scenario_context = self._format_scenario_context(csi_data, scenario_params)
            kb_context = json.dumps(self.algorithms_kb, indent=2)
            
            # Construct detailed prompt
            prompt = self._construct_selection_prompt(scenario_context, kb_context)
            
            # Define expected response schema
            response_schema = {
                "algorithm_name": "string",
                "algorithm_params": "object",
                "reasoning": "string", 
                "computational_complexity_note": "string"
            }
            
            # Query LLM
            self.logger.info("Querying LLM for algorithm selection")
            response = await self.llm_interface.query_llm(prompt, response_schema)
            
            # Validate and parse response
            return self._validate_and_parse_response(response)
            
        except Exception as e:
            self.logger.error(f"Algorithm selection failed: {e}")
            # Return fallback algorithm
            return self._get_fallback_algorithm()
    
    def _format_scenario_context(self, csi_data: Dict[str, Any], scenario_params: Dict[str, Any]) -> str:
        """
        Format scenario and CSI data into readable context for LLM.
        
        Args:
            csi_data: CSI data dictionary
            scenario_params: Scenario parameters
            
        Returns:
            Formatted context string
        """
        context = f"""
SCENARIO PARAMETERS:
- System Setup: {scenario_params.get('system_setup', 'Unknown')}
- Antenna Configuration: {scenario_params.get('antenna_configuration', 'Unknown')}
- Channel Environment: {', '.join(scenario_params.get('channel_environment', []))}
- Key Characteristics: {', '.join(scenario_params.get('key_characteristics', []))}

CSI CHARACTERISTICS:
- Number of RIS Elements: {csi_data.get('num_ris_elements', 'Unknown')}
- Number of BS Antennas: {csi_data.get('num_antennas_bs', 'Unknown')}
- Number of User Antennas: {csi_data.get('num_antennas_user', 'Unknown')}
- Channel Sparsity: {csi_data.get('channel_sparsity', 'Unknown')}
- Estimation Method: {csi_data.get('estimation_method', 'Unknown')}
- Estimation Accuracy: {csi_data.get('estimation_accuracy', 'Unknown')}
- Carrier Frequency: {csi_data.get('carrier_frequency', 'Unknown')} Hz
- SNR Environment: {'Low' if csi_data.get('noise_power_dbm', -80) > -85 else 'High'}
"""
        return context.strip()
    
    def _construct_selection_prompt(self, scenario_context: str, kb_context: str) -> str:
        """
        Construct detailed prompt for LLM algorithm selection.
        
        Args:
            scenario_context: Formatted scenario context
            kb_context: Knowledge base as JSON string
            
        Returns:
            Complete prompt string
        """
        prompt = f"""
You are an expert in Reconfigurable Intelligent Surface (RIS) communication systems, tasked with selecting the optimal algorithm for a given scenario.

CURRENT COMMUNICATION SCENARIO:
{scenario_context}

KNOWLEDGE BASE OF RIS ALGORITHMS:
{kb_context}

TASK:
Based on the scenario characteristics and the knowledge base, identify the single most suitable algorithm. Your response must be a valid JSON object containing:

- "algorithm_name": The exact name of the chosen algorithm from the recommended_algorithms in the knowledge base
- "algorithm_params": A dictionary of example parameters for this algorithm (e.g., {{"num_iterations": 100, "learning_rate": 0.01}})
- "reasoning": A detailed explanation of why this algorithm was chosen, referencing specific characteristics, benefits, and computational complexity
- "computational_complexity_note": The computational complexity note directly from the knowledge base entry

SELECTION CRITERIA:
1. Match scenario characteristics (system_setup, antenna_configuration, channel_environment) with knowledge base entries
2. Consider key_characteristics like mobility, SNR conditions, and channel sparsity
3. Prioritize algorithms that explicitly handle the scenario's challenges
4. Balance performance benefits with computational feasibility
5. Prefer algorithms with proven effectiveness for the specific conditions

Ensure your response is valid JSON and the algorithm_name exactly matches one from the knowledge base.
"""
        return prompt.strip()
    
    def _validate_and_parse_response(self, response: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str, str]:
        """
        Validate and parse LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed and validated response tuple
        """
        try:
            algorithm_name = response.get("algorithm_name", "").strip()
            algorithm_params = response.get("algorithm_params", {})
            reasoning = response.get("reasoning", "").strip()
            complexity_note = response.get("computational_complexity_note", "").strip()
            
            # Validate algorithm name exists in knowledge base
            all_algorithms = []
            for kb_entry in self.algorithms_kb:
                for alg in kb_entry.get("recommended_algorithms", []):
                    all_algorithms.append(alg.get("name", ""))
            
            if algorithm_name not in all_algorithms:
                self.logger.warning(f"Unknown algorithm '{algorithm_name}', using fallback")
                return self._get_fallback_algorithm()
            
            # Ensure algorithm_params is a dictionary
            if not isinstance(algorithm_params, dict):
                algorithm_params = {"num_iterations": 100, "learning_rate": 0.01}
            
            # Add default parameters if missing critical ones
            if not algorithm_params:
                algorithm_params = self._get_default_algorithm_params(algorithm_name)
            
            self.logger.info(f"Selected algorithm: {algorithm_name}")
            return algorithm_name, algorithm_params, reasoning, complexity_note
            
        except Exception as e:
            self.logger.error(f"Response validation failed: {e}")
            return self._get_fallback_algorithm()
    
    def _get_default_algorithm_params(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get default parameters for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of default parameters
        """
        if "Deep Learning" in algorithm_name or "Neural Network" in algorithm_name:
            return {
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32,
                "hidden_layers": [128, 64],
                "dropout": 0.1
            }
        elif "Compressed Sensing" in algorithm_name:
            return {
                "sparsity_level": 0.1,
                "num_iterations": 100,
                "tolerance": 1e-6,
                "measurement_ratio": 0.3
            }
        elif "Kalman Filter" in algorithm_name:
            return {
                "process_noise": 0.01,
                "measurement_noise": 0.1,
                "initial_estimate": 1.0
            }
        elif "PPO Algorithm" in algorithm_name:
            return {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "epsilon": 0.2,
                "num_epochs": 10
            }
        else:
            return {
                "num_iterations": 100,
                "learning_rate": 0.01,
                "tolerance": 1e-6
            }
    
    def _get_fallback_algorithm(self) -> Tuple[str, Dict[str, Any], str, str]:
        """
        Return a robust fallback algorithm when selection fails.
        
        Returns:
            Fallback algorithm tuple
        """
        return (
            "Compressed Sensing",
            {"sparsity_level": 0.1, "num_iterations": 100, "tolerance": 1e-6},
            "Fallback selection - Compressed Sensing is robust for most RIS scenarios",
            "O(log M) - Logarithmic complexity"
        )
