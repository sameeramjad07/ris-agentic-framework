"""
Main entry point for the RIS Agentic System.
Orchestrates the complete simulation and comparison.
"""

import asyncio
import json
import os
from config.settings import AppSettings
from agents.base_coordinator import BaseCoordinatorAgent
from agents.csi_estimation_agent import CSIEstimationAgent
from agents.optimizer_agent import OptimizerAgent
from agents.numerical_solver_agent import NumericalSolverAgent
from utils.ris_algorithms import RISAlgorithms
from utils.performance_metrics import PerformanceMetrics
from utils.data_visualizer import DataVisualizer
from utils.llm_interface import LLMInterface
from simulation.ris_environment import RISEnvironment
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def main():
    """Main function to run the RIS Agentic System."""
    print("🚀 Initializing RIS Agentic System")
    print("=" * 60)
    
    # Load configuration
    settings = AppSettings()
    
    # Load algorithms knowledge base
    try:
        with open('config/algorithms_kb.json', 'r') as f:
            algorithms_kb = json.load(f)
        print(f"✅ Loaded {len(algorithms_kb)} algorithm entries from knowledge base")
    except FileNotFoundError:
        print("❌ algorithms_kb.json not found. Please create it from the provided content.")
        return
    
    # Initialize components
    ris_environment = RISEnvironment()
    ris_algorithms = RISAlgorithms()
    performance_metrics = PerformanceMetrics()
    data_visualizer = DataVisualizer()
    
    # Initialize LLM interface
    if settings.CEREBRAS_API_KEY == os.environ.get("CEREBRAS_API_KEY"):
        print("❌ Please set your CEREBRAS_API_KEY in config/settings.py")
        return
    
    llm_interface = LLMInterface(settings.CEREBRAS_API_KEY, settings.LLM_MODEL_NAME)
    
    # Initialize agents
    csi_agent = CSIEstimationAgent(ris_environment)
    optimizer_agent = OptimizerAgent(llm_interface, algorithms_kb)
    solver_agent = NumericalSolverAgent(ris_algorithms, performance_metrics)
    coordinator = BaseCoordinatorAgent(csi_agent, optimizer_agent, solver_agent, data_visualizer)
    
    print("✅ All components initialized successfully")
    
    # Define diverse test scenarios
    scenarios = [
        {
            "scenario_id": "1",
            "system_setup": "MISO",
            "antenna_configuration": "Single-user, Narrowband",
            "channel_environment": ["Rician Fading"],
            "key_characteristics": ["Low/Moderate SNR", "mmWave channel sparsity"],
            "physical_params": {
                **settings.SIMULATION_PARAMS,
                "num_antennas_user": 1,
                "rician_k_factor": 3
            }
        },
        {
            "scenario_id": "2", 
            "system_setup": "MIMO",
            "antenna_configuration": "Multi-user, Narrowband",
            "channel_environment": ["Rayleigh Fading", "Cascaded Channels"],
            "key_characteristics": ["User-to-user channel estimation", "common RIS-BS channel"],
            "physical_params": {
                **settings.SIMULATION_PARAMS,
                "num_antennas_bs": 8,
                "num_antennas_user": 4
            }
        },
        {
            "scenario_id": "3",
            "system_setup": "MISO",
            "antenna_configuration": "Single-user, Broadband", 
            "channel_environment": ["mmWave Channels"],
            "key_characteristics": ["Reduces training complexity using a single neural network"],
            "physical_params": {
                **settings.SIMULATION_PARAMS,
                "num_antennas_user": 1,
                "carrier_frequency": 60e9  # 60 GHz
            }
        },
        {
            "scenario_id": "4",
            "system_setup": "MIMO",
            "antenna_configuration": "Single-user, Narrowband",
            "channel_environment": ["THz channels", "High-Mobility Scenarios"],
            "key_characteristics": ["Compressed sensing", "deep learning for THz channels", "high mobility"],
            "physical_params": {
                **settings.SIMULATION_PARAMS,
                "carrier_frequency": 300e9,  # 300 GHz (THz)
                "num_antennas_bs": 16,
                "num_antennas_user": 8
            }
        },
        {
            "scenario_id": "5",
            "system_setup": "MIMO",
            "antenna_configuration": "Multi-user, Broadband",
            "channel_environment": ["mmWave OTFS Systems", "High-Mobility Scenarios"],
            "key_characteristics": ["Joint channel estimation and data detection", "pilot sequences", "message passing"],
            "physical_params": {
                **settings.SIMULATION_PARAMS,
                "num_antennas_bs": 12,
                "num_antennas_user": 6,
                "carrier_frequency": 28e9
            }
        },
        {
            "scenario_id": "6",
            "system_setup": "MIMO",
            "antenna_configuration": "Single/Multi-user, Narrowband",
            "channel_environment": ["Real-world Imperfections (LTI/STI)"],
            "key_characteristics": ["Handles static and non-static imperfections", "tensor-based methods"],
            "physical_params": {
                **settings.SIMULATION_PARAMS,
                "num_antennas_bs": 10,
                "num_antennas_user": 5
            }
        },
        {
            "scenario_id": "7",
            "system_setup": "MIMO (Cell-Free)",
            "antenna_configuration": "Multi-user, Narrowband",
            "channel_environment": ["Quasi-static Block-Fading", "TDD Systems"],
            "key_characteristics": ["Multi-BS cooperation", "joint signal processing", "low-cost coverage enhancement"],
            "physical_params": {
                **settings.SIMULATION_PARAMS,
                "num_antennas_bs": 6,
                "num_antennas_user": 3,
                "num_ris_elements": 96
            }
        }
    ]
    
    print(f"🎯 Running {len(scenarios)} diverse scenarios...")
    print("=" * 60)
    
    # Run all scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🔄 Processing Scenario {scenario['scenario_id']} ({i}/{len(scenarios)})")
        try:
            await coordinator.run_scenario(scenario, scenario['scenario_id'])
            print(f"✅ Scenario {scenario['scenario_id']} completed successfully")
        except Exception as e:
            print(f"❌ Error in scenario {scenario['scenario_id']}: {e}")
            continue
    
    # Generate final cumulative analysis
    print(f"\n📊 Generating cumulative performance analysis...")
    data_visualizer.plot_cumulative_performance()
    
    print(f"\n🎉 RIS Agentic System simulation completed!")
    print(f"📁 Results saved in plots/ directory")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
