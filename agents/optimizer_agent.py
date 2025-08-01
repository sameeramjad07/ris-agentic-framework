"""Agent responsible for intelligent algorithm selection."""

import json
import logging
from typing import Dict, Any, Tuple
from utils.llm_interface import LLMInterface

class OptimizerAgent:
    def __init__(self, llm_interface: LLMInterface, knowledge_base: list):
        self.llm_interface = llm_interface
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(__name__)
        
    async def select_algorithm(self, scenario_features: Dict[str, float]) -> Tuple[str, str]:
        """Select optimal algorithm using LLM reasoning."""
        try:
            # Format scenario for LLM
            scenario_text = self._format_scenario(scenario_features)
            kb_text = json.dumps(self.knowledge_base, indent=2)
            
            prompt = f"""
You are an expert in RIS optimization. Given the scenario features and knowledge base, select the best algorithm.

SCENARIO FEATURES:
{scenario_text}

KNOWLEDGE BASE:
{kb_text}

Based on the scenario features, determine which knowledge base entry best matches and recommend the corresponding algorithm.

Respond with valid JSON only:
{{
    "selected_algorithm": "exact algorithm name from knowledge base",
    "reasoning": "detailed explanation of why this algorithm was chosen"
}}
"""
            
            response = await self.llm_interface.query_llm(prompt)
            
            algorithm = response.get('selected_algorithm', 'Alternating Optimization')
            reasoning = response.get('reasoning', 'Fallback selection')
            
            # Validate algorithm name
            valid_algorithms = ['Gradient Descent', 'Manifold Optimization', 'Alternating Optimization']
            if algorithm not in valid_algorithms:
                algorithm = 'Alternating Optimization'
                reasoning = f"Invalid selection, using fallback: {reasoning}"
            
            self.logger.info(f"Selected algorithm: {algorithm}")
            return algorithm, reasoning
            
        except Exception as e:
            self.logger.error(f"Algorithm selection failed: {e}")
            return 'Alternating Optimization', f'Error fallback: {str(e)}'
    
    def _format_scenario(self, features: Dict[str, float]) -> str:
        """Format scenario features for LLM."""
        return f"""
- Direct Channel Strength: {features['direct_channel_norm']:.4f}
- BS-RIS Channel Strength (G_norm): {features['G_norm']:.4f}  
- RIS-User Channel Strength (hr_norm): {features['hr_norm']:.4f}
- Phase Alignment Score: {features['phase_alignment_score']:.4f}
- Estimated SNR: {features['estimated_snr']:.2f} dB
- Number of RIS Elements: {int(features['num_ris_elements'])}
"""
