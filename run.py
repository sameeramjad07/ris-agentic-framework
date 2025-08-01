"""Main script to run RIS agentic system evaluation."""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

from config.settings import Settings
from agents.base_coordinator import CoordinatorAgent
from agents.optimizer_agent import OptimizerAgent
from agents.solver_agent import SolverAgent
from utils.llm_interface import LLMInterface
from utils.performance_metrics import PerformanceMetrics

import os
import uuid
from datetime import datetime

# Create a unique folder for plots on each run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
unique_folder = f"plots/run_{timestamp}_{uuid.uuid4().hex[:6]}"
os.makedirs(unique_folder, exist_ok=True)  # Create plots/run_<timestamp>_<uuid>/
plots_dir = unique_folder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(filename: str) -> List[Dict[str, Any]]:
    """Load dataset from file."""
    if not filename.startswith('data/'):
        filename = f'data/{filename}'
    
    required_input_keys = [
        'direct_channel_real', 'direct_channel_imag',
        'bs_ris_channel_real', 'bs_ris_channel_imag',
        'ris_user_channel_real', 'ris_user_channel_imag',
        'num_ris_elements', 'direct_channel_norm', 'G_norm',
        'hr_norm', 'phase_alignment_score', 'estimated_snr', 'objective'
    ]
    required_output_keys = ['optimized_phase_shifts']
    
    dataset = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                scenario = json.loads(line.strip())
                # Validate structure
                if 'input' not in scenario or 'output' not in scenario:
                    logger.warning(f"Skipping invalid scenario at line {i}: Missing input/output")
                    continue
                
                # Validate input keys
                missing_inputs = [k for k in required_input_keys if k not in scenario['input']]
                if missing_inputs:
                    logger.warning(f"Skipping scenario at line {i}: Missing input keys {missing_inputs}")
                    continue
                
                # Validate output keys
                missing_outputs = [k for k in required_output_keys if k not in scenario['output']]
                if missing_outputs:
                    logger.warning(f"Skipping scenario at line {i}: Missing output keys {missing_outputs}")
                    continue
                
                # Validate array lengths
                num_elements = scenario['input']['num_ris_elements']
                for key in ['bs_ris_channel_real', 'bs_ris_channel_imag', 
                           'ris_user_channel_real', 'ris_user_channel_imag']:
                    if len(scenario['input'][key]) != num_elements:
                        logger.warning(f"Skipping scenario at line {i}: Invalid length for {key}")
                        continue
                if len(scenario['output']['optimized_phase_shifts']) != num_elements:
                    logger.warning(f"Skipping scenario at line {i}: Invalid length for optimized_phase_shifts")
                    continue
                
                dataset.append(scenario)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {i}")
                continue
    
    if not dataset:
        raise ValueError(f"No valid scenarios found in {filename}")
    
    return dataset

def load_knowledge_base() -> List[Dict[str, Any]]:
    """Load knowledge base from file."""
    with open('config/knowledge_base.json', 'r') as f:
        return json.load(f)

async def evaluate_scenario(coordinator: CoordinatorAgent, scenario: Dict[str, Any], 
                          transmit_powers: List[float]) -> Dict[str, List[float]]:
    """Evaluate a single scenario across different transmit powers."""
    results = {
        'random': [],
        'analytical': [],
        'gradient_descent': [],
        'manifold_optimization': [],
        'alternating_optimization': [],
        'agentic': []
    }

    # Get objective from scenario input, default to 'snr'
    objective = scenario['input'].get('objective', 'maximize_snr').lower()
    if objective not in ['maximize_snr', 'maximize_sum_rate']:
        logger.warning(f"Unsupported objective '{objective}'; defaulting to SNR")
        objective = 'maximize_snr'

    try:
        # Process scenario once to get phase shifts
        scenario_results = await coordinator.process_scenario(scenario)
        
        # Calculate performance metric for each method across transmit powers
        for power in transmit_powers:
            for method in results.keys():
                if method in scenario_results:
                    phase_shifts = scenario_results[method]['phase_shifts']
                    if objective == 'maximize_snr':
                        metric = PerformanceMetrics.calculate_snr(
                            phase_shifts, scenario['input'], transmit_power=power
                        )
                    elif objective == 'maximize_sum_rate':
                        metric = PerformanceMetrics.calculate_sum_rate(
                            phase_shifts, scenario['input'], transmit_power=power
                        )
                    results[method].append(metric)
                else:
                    results[method].append(-50.0 if objective == 'maximize_snr' else 0.0)
                    
    except Exception as e:
        logger.error(f"Error evaluating scenario: {e}")
        for power in transmit_powers:
            for method in results.keys():
                results[method].append(-50.0 if objective == 'maximize_snr' else 0.0)
    
    return results, objective

def plot_individual_scenario(scenario_idx: int, results: Dict[str, List[float]], 
                           transmit_powers: List[float], num_elements: int, objective: str):
    """Plot results for an individual scenario."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'random': 'gray',
        'analytical': 'black',
        'gradient_descent': 'blue',
        'manifold_optimization': 'red',
        'alternating_optimization': 'green',
        'agentic': 'purple'
    }
    
    linestyles = {
        'random': '--',
        'analytical': '-',
        'gradient_descent': '-',
        'manifold_optimization': '-',
        'alternating_optimization': '-',
        'agentic': '-'
    }
    
    linewidths = {
        'random': 1,
        'analytical': 3,
        'gradient_descent': 2,
        'manifold_optimization': 2,
        'alternating_optimization': 2,
        'agentic': 3
    }
    
    labels = {
        'random': 'Random Baseline',
        'analytical': 'Optimal Analytical',
        'gradient_descent': 'Gradient Descent',
        'manifold_optimization': 'Manifold Optimization',
        'alternating_optimization': 'Alternating Optimization',
        'agentic': 'Agentic System'
    }
    
    transmit_powers_dbm = [10 * np.log10(p * 1000) for p in transmit_powers]  # Convert to dBm
    
    for method, snr_values in results.items():
        plt.plot(transmit_powers_dbm, snr_values, 
                color=colors[method], linestyle=linestyles[method], 
                linewidth=linewidths[method], label=labels[method], marker='o')
    
    plt.xlabel('Transmit Power (dBm)')
    plt.ylabel('SNR (dB)')
    plt.title(f'RIS Performance - Scenario {scenario_idx + 1} ({num_elements} elements, Objective: {objective.title()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, f'scenario_{scenario_idx + 1}_{num_elements}elements.png'), dpi=300)
    plt.close()

def plot_cumulative_results(all_results: List[Dict[str, List[float]]], 
                          transmit_powers: List[float], num_elements: int, objective: str):
    """Plot cumulative/average results across all scenarios."""
    plt.figure(figsize=(12, 8))
    
    # Calculate average performance across all scenarios
    avg_results = {}
    for method in all_results[0].keys():
        method_results = np.array([result[method] for result in all_results])
        avg_results[method] = np.mean(method_results, axis=0)
    
    colors = {
        'random': 'gray',
        'analytical': 'black', 
        'gradient_descent': 'blue',
        'manifold_optimization': 'red',
        'alternating_optimization': 'green',
        'agentic': 'purple'
    }
    
    linestyles = {
        'random': '--',
        'analytical': '-',
        'gradient_descent': '-', 
        'manifold_optimization': '-',
        'alternating_optimization': '-',
        'agentic': '-'
    }
    
    linewidths = {
        'random': 1,
        'analytical': 3,
        'gradient_descent': 2,
        'manifold_optimization': 2, 
        'alternating_optimization': 2,
        'agentic': 3
    }
    
    labels = {
        'random': 'Random Baseline',
        'analytical': 'Optimal Analytical',
        'gradient_descent': 'Gradient Descent',
        'manifold_optimization': 'Manifold Optimization',
        'alternating_optimization': 'Alternating Optimization', 
        'agentic': 'Agentic System'
    }
    
    transmit_powers_dbm = [10 * np.log10(p * 1000) for p in transmit_powers]
    
    for method, avg_snr in avg_results.items():
        plt.plot(transmit_powers_dbm, avg_snr,
                color=colors[method], linestyle=linestyles[method],
                linewidth=linewidths[method], label=labels[method], marker='o')
    
    plt.xlabel('Transmit Power (dBm)')
    plt.ylabel('Average SNR (dB)')
    plt.title(f'RIS Performance - Average Across All Scenarios ({num_elements} elements, Objective: {objective.title()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, f'cumulative_results_{num_elements}elements.png'), dpi=300)
    plt.close()
    
    return avg_results

def plot_multi_element_comparison(all_element_results: Dict[int, Dict[str, List[float]]], 
                                transmit_powers: List[float]):
    """Plot comparison across different numbers of RIS elements."""
    plt.figure(figsize=(14, 10))
    
    colors = {
        'random': 'gray',
        'analytical': 'black',
        'gradient_descent': 'blue', 
        'manifold_optimization': 'red',
        'alternating_optimization': 'green',
        'agentic': 'purple'
    }
    
    transmit_powers_dbm = [10 * np.log10(p * 1000) for p in transmit_powers]
    
    # Plot each method for each element count
    for method in ['random', 'analytical', 'gradient_descent', 'manifold_optimization', 
                   'alternating_optimization', 'agentic']:
        for num_elements, results in all_element_results.items():
            if method in results:
                label = f'{method.replace("_", " ").title()} ({num_elements} elements)'
                linestyle = '--' if method == 'random' else '-'
                linewidth = 3 if method in ['analytical', 'agentic'] else 2
                alpha = 0.7 if method == 'random' else 1.0
                
                plt.plot(transmit_powers_dbm, results[method], 
                        color=colors[method], linestyle=linestyle, 
                        linewidth=linewidth, label=label, alpha=alpha, marker='o')
    
    plt.xlabel('Transmit Power (dBm)')
    plt.ylabel('Average SNR (dB)')
    plt.title('RIS Performance Comparison - All Methods and Element Counts')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, 'multi_element_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

async def main():
    parser = argparse.ArgumentParser(description='Run RIS agentic system evaluation')
    parser.add_argument('--datasets', type=str, nargs='+', required=True,
                       help='Dataset files to evaluate')
    parser.add_argument('--max-scenarios', type=int, default=10,
                       help='Maximum number of scenarios to evaluate per dataset')
    parser.add_argument('--individual-plots', action='store_true',
                       help='Generate plots for individual scenarios')
    
    args = parser.parse_args()
    
    print("🚀 Starting RIS Agentic System Evaluation")
    print("=" * 60)
    
    # Initialize settings and components
    settings = Settings()
    
    if not settings.CEREBRAS_API_KEY:
        print("❌ Please set your CEREBRAS_API_KEY in your .env file")
        return
    
    # Load knowledge base
    knowledge_base = load_knowledge_base()
    
    # Initialize agents
    llm_interface = LLMInterface()
    optimizer_agent = OptimizerAgent(llm_interface, knowledge_base)
    solver_agent = SolverAgent()
    coordinator = CoordinatorAgent(optimizer_agent, solver_agent)
    
    # Define transmit power range
    transmit_powers = np.logspace(-2, 1, 10)  # 0.01W to 10W
    
    # Store results for multi-element comparison
    all_element_results = {}
    
    # Process each dataset
    for dataset_file in args.datasets:
        print(f"\n📊 Processing dataset: {dataset_file}")
        
        # Extract number of elements from filename
        num_elements = int(dataset_file.split('_')[1].replace('elements', ''))
        
        # Load dataset
        dataset = load_dataset(dataset_file)
        max_scenarios = min(args.max_scenarios, len(dataset))
        
        print(f"   - RIS Elements: {num_elements}")
        print(f"   - Scenarios to evaluate: {max_scenarios}")
        
        all_results = []
        scenario_objective = 'maximize_snr'  # Default
        
        # Evaluate scenarios
        for i, scenario in enumerate(dataset[:max_scenarios]):
            print(f"   - Processing scenario {i + 1}/{max_scenarios}...")
            
            try:
                results, objective = await evaluate_scenario(coordinator, scenario, transmit_powers) 
                scenario_objective = objective
                all_results.append(results)
                
                # Plot individual scenario if requested
                if args.individual_plots:
                    plot_individual_scenario(i, results, transmit_powers, num_elements, objective)
                    
            except Exception as e:
                logger.error(f"Failed to process scenario {i + 1}: {e}")
                continue
        
        # Plot cumulative results for this dataset
        if all_results:
            avg_results = plot_cumulative_results(all_results, transmit_powers, num_elements, scenario_objective)
            all_element_results[num_elements] = avg_results
            print(f"   ✅ Completed evaluation for {num_elements} elements")
        else:
            print(f"   ❌ No successful evaluations for {num_elements} elements")
    
    # Create multi-element comparison plot
    if len(all_element_results) > 1:
        print(f"\n📈 Creating multi-element comparison plot...")
        plot_multi_element_comparison(all_element_results, transmit_powers)
    
    # Print summary statistics
    print(f"\n📋 Performance Summary:")
    print("=" * 60)
    
    for num_elements, results in all_element_results.items():
        print(f"\nRIS Elements: {num_elements}")
        for method, snr_values in results.items():
            avg_snr = np.mean(snr_values)
            max_snr = np.max(snr_values)
            print(f"  {method.replace('_', ' ').title():.<25} Avg: {avg_snr:.1f} dB, Max: {max_snr:.1f} dB")
        
        # Calculate agentic improvement over baseline
        if 'agentic' in results and 'random' in results:
            agentic_avg = np.mean(results['agentic'])
            random_avg = np.mean(results['random'])
            improvement = agentic_avg - random_avg
            print(f"  {'Agentic Improvement':.<25} {improvement:.1f} dB over random baseline")
    
    print(f"\n🎉 Evaluation completed! Check the generated plots.")

if __name__ == "__main__":
    asyncio.run(main())
