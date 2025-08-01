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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(filename: str) -> List[Dict[str, Any]]:
    """Load dataset from file."""
    if not filename.startswith('data/'):
        filename = f'data/{filename}'
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            scenario = json.loads(line.strip())
            dataset.append(scenario)
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
    
    for power in transmit_powers:
        try:
            # Process scenario through agentic pipeline
            scenario_results = await coordinator.process_scenario(scenario)
            
            # Calculate SNR for each method at current transmit power
            for method in results.keys():
                if method in scenario_results:
                    phase_shifts = scenario_results[method]['phase_shifts']
                    snr = PerformanceMetrics.calculate_snr(
                        phase_shifts, scenario['input'], transmit_power=power
                    )
                    results[method].append(snr)
                else:
                    results[method].append(-50.0)  # Very poor performance
                    
        except Exception as e:
            logger.error(f"Error evaluating scenario: {e}")
            # Fill with poor performance values
            for method in results.keys():
                results[method].append(-50.0)
    
    return results

def plot_individual_scenario(scenario_idx: int, results: Dict[str, List[float]], 
                           transmit_powers: List[float], num_elements: int):
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
    plt.title(f'RIS Performance - Scenario {scenario_idx + 1} ({num_elements} elements)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'scenario_{scenario_idx + 1}_{num_elements}elements.png', dpi=300)
    plt.close()

def plot_cumulative_results(all_results: List[Dict[str, List[float]]], 
                          transmit_powers: List[float], num_elements: int):
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
    plt.title(f'RIS Performance - Average Across All Scenarios ({num_elements} elements)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'cumulative_results_{num_elements}elements.png', dpi=300)
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
    plt.savefig('multi_element_comparison.png', dpi=300, bbox_inches='tight')
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
    llm_interface = LLMInterface(settings.CEREBRAS_API_KEY, settings.LLM_MODEL_NAME)
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
        
        # Evaluate scenarios
        for i, scenario in enumerate(dataset[:max_scenarios]):
            print(f"   - Processing scenario {i + 1}/{max_scenarios}...")
            
            try:
                results = await evaluate_scenario(coordinator, scenario, transmit_powers) 
                all_results.append(results)
                
                # Plot individual scenario if requested
                if args.individual_plots:
                    plot_individual_scenario(i, results, transmit_powers, num_elements)
                    
            except Exception as e:
                logger.error(f"Failed to process scenario {i + 1}: {e}")
                continue
        
        # Plot cumulative results for this dataset
        if all_results:
            avg_results = plot_cumulative_results(all_results, transmit_powers, num_elements)
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
