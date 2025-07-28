import asyncio
import logging
from typing import Dict, List, Any
from utils.ris_algorithms import RISAlgorithms
from utils.performance_metrics import PerformanceMetrics
from utils.data_visualizer import DataVisualizer

class BaseCoordinatorAgent:
    """
    Central coordinator that orchestrates the entire RIS optimization process.
    Manages CSI estimation, algorithm selection, execution, and performance comparison.
    """
    
    def __init__(self, csi_agent, optimizer_agent, solver_agent, data_visualizer):
        """
        Initialize the coordinator with all required agents.
        
        Args:
            csi_agent: CSIEstimationAgent instance
            optimizer_agent: OptimizerAgent instance  
            solver_agent: NumericalSolverAgent instance
            data_visualizer: DataVisualizer instance
        """
        self.csi_agent = csi_agent
        self.optimizer_agent = optimizer_agent
        self.solver_agent = solver_agent
        self.data_visualizer = data_visualizer
        self.ris_algorithms = RISAlgorithms()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def run_scenario(self, scenario_params: Dict[str, Any], scenario_id: str) -> List[Dict[str, Any]]:
        """
        Execute a complete RIS optimization scenario, comparing agentic vs brainless approaches.
        
        Args:
            scenario_params: Dictionary containing scenario configuration
            scenario_id: Unique identifier for this scenario
            
        Returns:
            List of results from all algorithm executions
        """
        self.logger.info(f"Starting scenario {scenario_id}")
        results = []
        
        try:
            # Step 1: Estimate CSI for the given scenario
            self.logger.info("Estimating CSI...")
            csi_data = self.csi_agent.estimate_csi(scenario_params)
            
            # Step 2: Use optimizer agent to select best algorithm (agentic approach)
            self.logger.info("Running agentic algorithm selection...")
            try:
                algorithm_name, algorithm_params, reasoning, complexity_note = await self.optimizer_agent.select_algorithm(
                    csi_data, scenario_params
                )
                
                # Execute the agent-selected algorithm
                agentic_result = self.solver_agent.execute_algorithm(
                    algorithm_name, algorithm_params, csi_data
                )
                
                # Store agentic result
                agentic_record = {
                    "scenario_id": scenario_id,
                    "system_type": "Agentic",
                    "algorithm_used": algorithm_name,
                    "snr_value": agentic_result["snr"],
                    "sum_rate_value": agentic_result["sum_rate"],
                    "chosen_by_agent": True,
                    "agent_reasoning": reasoning,
                    "computational_complexity": complexity_note
                }
                results.append(agentic_record)
                
                self.logger.info(f"Agentic system chose: {algorithm_name}")
                self.logger.info(f"Reasoning: {reasoning}")
                
            except Exception as e:
                self.logger.error(f"Agentic selection failed: {e}")
                # Fallback to a default algorithm
                fallback_result = self.solver_agent.execute_algorithm(
                    "Compressed Sensing", {"num_iterations": 100}, csi_data
                )
                agentic_record = {
                    "scenario_id": scenario_id,
                    "system_type": "Agentic (Fallback)",
                    "algorithm_used": "Compressed Sensing",
                    "snr_value": fallback_result["snr"],
                    "sum_rate_value": fallback_result["sum_rate"],
                    "chosen_by_agent": False,
                    "agent_reasoning": f"Fallback due to error: {str(e)}",
                    "computational_complexity": "O(log M)"
                }
                results.append(agentic_record)
            
            # Step 3: Run all available algorithms (brainless approach)
            self.logger.info("Running brainless algorithm comparison...")
            available_algorithms = self.ris_algorithms.get_all_algorithm_names()
            
            for alg_name in available_algorithms:
                try:
                    # Use generic default parameters for brainless execution
                    default_params = self._get_default_params(alg_name)
                    brainless_result = self.solver_agent.execute_algorithm(
                        alg_name, default_params, csi_data
                    )
                    
                    brainless_record = {
                        "scenario_id": scenario_id,
                        "system_type": "Brainless",
                        "algorithm_used": alg_name,
                        "snr_value": brainless_result["snr"],
                        "sum_rate_value": brainless_result["sum_rate"],
                        "chosen_by_agent": False,
                        "agent_reasoning": "Generic brainless execution",
                        "computational_complexity": "Not optimized"
                    }
                    results.append(brainless_record)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to execute {alg_name}: {e}")
            
            # Step 4: Visualize scenario results
            agent_reasoning = next((r["agent_reasoning"] for r in results if r["chosen_by_agent"]), "No reasoning available")
            self.data_visualizer.plot_scenario_comparison(scenario_id, results, agent_reasoning)
            self.data_visualizer.add_cumulative_data(results)
            
            self.logger.info(f"Completed scenario {scenario_id} with {len(results)} algorithm runs")
            
        except Exception as e:
            self.logger.error(f"Scenario {scenario_id} failed: {e}")
            raise
            
        return results
    
    def _get_default_params(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get generic default parameters for any algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary of default parameters
        """
        # Generic parameters that work for most algorithms
        default_params = {
            "num_iterations": 100,
            "learning_rate": 0.01,
            "tolerance": 1e-6,
            "max_iterations": 1000,
            "regularization": 0.01
        }
        
        # Algorithm-specific parameter adjustments
        if "neural" in algorithm_name.lower() or "deep" in algorithm_name.lower():
            default_params.update({
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001
            })
        elif "kalman" in algorithm_name.lower():
            default_params.update({
                "process_noise": 0.01,
                "measurement_noise": 0.1
            })
        elif "compressed" in algorithm_name.lower():
            default_params.update({
                "sparsity_level": 0.1,
                "measurement_ratio": 0.3
            })
            
        return default_params
