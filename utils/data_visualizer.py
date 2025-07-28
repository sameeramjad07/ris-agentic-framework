import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Dict, Any
import logging

class DataVisualizer:
    """
    Data visualization utilities for RIS system performance analysis.
    Creates comparison plots and cumulative performance analysis.
    """
    
    def __init__(self, output_dir: str = "plots"):
        """
        Initialize data visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        self.cumulative_data = []
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_scenario_comparison(self, scenario_id: str, results: List[Dict[str, Any]], agent_reasoning: str):
        """
        Create comparison plot for a single scenario.
        
        Args:
            scenario_id: Unique scenario identifier
            results: List of algorithm performance results
            agent_reasoning: LLM reasoning for algorithm selection
        """
        try:
            self.logger.info(f"Creating comparison plot for scenario {scenario_id}")
            
            # Separate agentic and brainless results
            agentic_results = [r for r in results if r["chosen_by_agent"]]
            brainless_results = [r for r in results if not r["chosen_by_agent"]]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot SNR comparison
            self._plot_metric_comparison(ax1, agentic_results, brainless_results, 
                                       "snr_value", "SNR (dB)", scenario_id)
            
            # Plot sum rate comparison
            self._plot_metric_comparison(ax2, agentic_results, brainless_results,
                                       "sum_rate_value", "Sum Rate (bps)", scenario_id)
            
            # Add reasoning as text
            fig.text(0.5, 0.02, f"Agent Reasoning: {agent_reasoning[:200]}...", 
                    ha='center', fontsize=10, wrap=True)
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            
            # Save plot
            filename = os.path.join(self.output_dir, f"scenario_{scenario_id}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved scenario plot: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to create scenario plot for {scenario_id}: {e}")
    
    def _plot_metric_comparison(self, ax, agentic_results: List[Dict], brainless_results: List[Dict],
                              metric_key: str, metric_label: str, scenario_id: str):
        """
        Plot comparison for a specific metric.
        
        Args:
            ax: Matplotlib axis
            agentic_results: Agentic system results
            brainless_results: Brainless system results
            metric_key: Key for the metric in results dictionary
            metric_label: Label for the metric
            scenario_id: Scenario identifier
        """
        # Extract data
        algorithms = []
        values = []
        colors = []
        
        # Add agentic result
        if agentic_results:
            agentic = agentic_results[0]
            algorithms.append(f"Agentic:\n{agentic['algorithm_used']}")
            values.append(agentic[metric_key])
            colors.append('red')
        
        # Add brainless results (top 10 for readability)
        brainless_results_sorted = sorted(brainless_results, key=lambda x: x[metric_key], reverse=True)
        for result in brainless_results_sorted[:10]:
            algorithms.append(f"Brainless:\n{result['algorithm_used']}")
            values.append(result[metric_key])
            colors.append('lightblue')
        
        # Create bar plot
        bars = ax.bar(range(len(algorithms)), values, color=colors, alpha=0.7)
        
        # Highlight the agentic result
        if agentic_results:
            bars[0].set_color('red')
            bars[0].set_alpha(1.0)
            bars[0].set_edgecolor('black')
            bars[0].set_linewidth(2)
        
        # Customize plot
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} Comparison - Scenario {scenario_id}')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=8)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2e}' if value > 1000 else f'{value:.2f}',
                   ha='center', va='bottom', fontsize=8)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=1.0, label='Agentic System'),
                          Patch(facecolor='lightblue', alpha=0.7, label='Brainless Systems')]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def add_cumulative_data(self, results: List[Dict[str, Any]]):
        """
        Add results to cumulative performance tracking.
        
        Args:
            results: List of algorithm performance results
        """
        self.cumulative_data.extend(results)
        self.logger.debug(f"Added {len(results)} results to cumulative data")
    
    def plot_cumulative_performance(self):
        """
        Create cumulative performance analysis plot.
        Shows performance trends across all scenarios.
        """
        try:
            self.logger.info("Creating cumulative performance plot")
            
            if not self.cumulative_data:
                self.logger.warning("No cumulative data available for plotting")
                return
            
            # Separate data by system type
            agentic_data = [r for r in self.cumulative_data if r["chosen_by_agent"]]
            brainless_data = [r for r in self.cumulative_data if not r["chosen_by_agent"]]
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: SNR distribution
            self._plot_performance_distribution(ax1, agentic_data, brainless_data, 
                                              "snr_value", "SNR (dB)")
            
            # Plot 2: Sum rate distribution  
            self._plot_performance_distribution(ax2, agentic_data, brainless_data,
                                              "sum_rate_value", "Sum Rate (bps)")
            
            # Plot 3: Cumulative average SNR
            self._plot_cumulative_average(ax3, agentic_data, brainless_data,
                                        "snr_value", "Cumulative Average SNR (dB)")
            
            # Plot 4: Cumulative average sum rate
            self._plot_cumulative_average(ax4, agentic_data, brainless_data,
                                        "sum_rate_value", "Cumulative Average Sum Rate (bps)")
            
            plt.tight_layout()
            
            # Save plot
            filename = os.path.join(self.output_dir, "cumulative_performance.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved cumulative performance plot: {filename}")
            
            # Create summary statistics
            self._create_performance_summary()
            
        except Exception as e:
            self.logger.error(f"Failed to create cumulative performance plot: {e}")
    
    def _plot_performance_distribution(self, ax, agentic_data: List[Dict], brainless_data: List[Dict],
                                     metric_key: str, metric_label: str):
        """Plot performance distribution comparison."""
        agentic_values = [r[metric_key] for r in agentic_data]
        brainless_values = [r[metric_key] for r in brainless_data]
        
        # Create histograms
        ax.hist(brainless_values, bins=20, alpha=0.6, label='Brainless Systems', 
                color='lightblue', density=True)
        ax.hist(agentic_values, bins=10, alpha=0.8, label='Agentic System',
                color='red', density=True)
        
        ax.set_xlabel(metric_label)
        ax.set_ylabel('Density')
        ax.set_title(f'{metric_label} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_average(self, ax, agentic_data: List[Dict], brainless_data: List[Dict],
                               metric_key: str, metric_label: str):
        """Plot cumulative average performance."""
        # Calculate cumulative averages
        agentic_values = [r[metric_key] for r in agentic_data]
        brainless_values = [r[metric_key] for r in brainless_data]
        
        if agentic_values:
            agentic_cumavg = np.cumsum(agentic_values) / np.arange(1, len(agentic_values) + 1)
            ax.plot(range(1, len(agentic_cumavg) + 1), agentic_cumavg, 
                   'r-', linewidth=2, label='Agentic System', marker='o', markersize=4)
        
        if brainless_values:
            # Group brainless data by scenario for fair comparison
            scenarios = list(set(r["scenario_id"] for r in brainless_data))
            brainless_scenario_avgs = []
            
            for scenario in scenarios:
                scenario_values = [r[metric_key] for r in brainless_data if r["scenario_id"] == scenario]
                brainless_scenario_avgs.append(np.mean(scenario_values))
            
            if brainless_scenario_avgs:
                brainless_cumavg = np.cumsum(brainless_scenario_avgs) / np.arange(1, len(brainless_scenario_avgs) + 1)
                ax.plot(range(1, len(brainless_cumavg) + 1), brainless_cumavg,
                       'b--', linewidth=2, label='Brainless Average', marker='s', markersize=4)
        
        ax.set_xlabel('Scenario Number')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} - Cumulative Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_performance_summary(self):
        """Create and save performance summary statistics."""
        try:
            agentic_data = [r for r in self.cumulative_data if r["chosen_by_agent"]]
            brainless_data = [r for r in self.cumulative_data if not r["chosen_by_agent"]]
            
            summary = {
                "Total Scenarios": len(set(r["scenario_id"] for r in self.cumulative_data)),
                "Total Algorithm Runs": len(self.cumulative_data),
                "Agentic System Runs": len(agentic_data),
                "Brainless System Runs": len(brainless_data),
            }
            
            if agentic_data:
                summary.update({
                    "Agentic Average SNR (dB)": np.mean([r["snr_value"] for r in agentic_data]),
                    "Agentic Average Sum Rate (bps)": np.mean([r["sum_rate_value"] for r in agentic_data]),
                })
            
            if brainless_data:
                summary.update({
                    "Brainless Average SNR (dB)": np.mean([r["snr_value"] for r in brainless_data]),
                    "Brainless Average Sum Rate (bps)": np.mean([r["sum_rate_value"] for r in brainless_data]),
                })
            
            # Calculate performance gains
            if agentic_data and brainless_data:
                snr_gain = summary["Agentic Average SNR (dB)"] - summary["Brainless Average SNR (dB)"]
                rate_gain_ratio = summary["Agentic Average Sum Rate (bps)"] / summary["Brainless Average Sum Rate (bps)"]
                
                summary.update({
                    "SNR Gain (dB)": snr_gain,
                    "Sum Rate Gain Ratio": rate_gain_ratio,
                })
            
            # Save summary
            summary_file = os.path.join(self.output_dir, "performance_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("RIS Agentic System Performance Summary\n")
                f.write("=" * 50 + "\n\n")
                for key, value in summary.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Saved performance summary: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create performance summary: {e}")
