"""Generate RIS scenario datasets for evaluation."""

import json
import numpy as np
import argparse
from typing import List, Dict, Any
from utils.performance_metrics import PerformanceMetrics

def generate_channel_scenario(num_elements: int) -> Dict[str, Any]:
    """Generate a random RIS scenario."""
    # Generate random channels
    h_d = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    
    H_br = (np.random.randn(num_elements) + 1j * np.random.randn(num_elements)) / np.sqrt(2)
    h_ru = (np.random.randn(num_elements) + 1j * np.random.randn(num_elements)) / np.sqrt(2)
    
    # Calculate optimal phases analytically (for maximum SNR)
    cascaded = H_br * h_ru
    optimal_phases = -np.angle(cascaded) + np.angle(h_d)
    optimal_phases = np.mod(optimal_phases, 2 * np.pi)

    # Compute derived terms
    direct_channel_norm = float(abs(h_d))
    G_norm = float(np.linalg.norm(H_br))
    hr_norm = float(np.linalg.norm(h_ru))
    phase_alignment_score = float(abs(np.sum(cascaded)) / (np.linalg.norm(cascaded) * len(cascaded)))
    signal_power = abs(h_d + np.sum(cascaded))**2
    noise_power = 1e-12
    estimated_snr = float(10 * np.log10(signal_power / noise_power))
    objective = "maximize_snr"  # Default, can be parameterized later
    
    # Create input data
    input_data = {
        "direct_channel_real": float(h_d.real),
        "direct_channel_imag": float(h_d.imag),
        "bs_ris_channel_real": H_br.real.tolist(),
        "bs_ris_channel_imag": H_br.imag.tolist(),
        "ris_user_channel_real": h_ru.real.tolist(),
        "ris_user_channel_imag": h_ru.imag.tolist(),
        "num_ris_elements": num_elements,
        "direct_channel_norm": direct_channel_norm,
        "G_norm": G_norm,
        "hr_norm": hr_norm,
        "phase_alignment_score": phase_alignment_score,
        "estimated_snr": estimated_snr,
        "objective": objective
    }
    
    return {
        "input": input_data,
        "output": {"optimized_phase_shifts": optimal_phases.tolist()}
    }

def generate_dataset(num_elements: int, num_scenarios: int) -> List[Dict[str, Any]]:
    """Generate a complete dataset."""
    dataset = []
    print(f"Generating {num_scenarios} scenarios with {num_elements} RIS elements...")
    
    for i in range(num_scenarios):
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_scenarios} scenarios")
        scenario = generate_channel_scenario(num_elements)
        dataset.append(scenario)
    
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], num_elements: int, num_scenarios: int):
    """Save dataset to file."""
    filename = f"data/dataset_{num_elements}elements_{num_scenarios}scenarios.json"
    
    with open(filename, 'w') as f:
        for scenario in dataset:
            f.write(json.dumps(scenario) + '\n')
    
    print(f"Dataset saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate RIS scenario datasets')
    parser.add_argument('--elements', type=int, nargs='+', default=[8, 16, 32, 64],
                       help='Number of RIS elements for each dataset')
    parser.add_argument('--scenarios', type=int, default=100,
                       help='Number of scenarios per dataset')
    
    args = parser.parse_args()
    
    print("🚀 Starting RIS Dataset Generation")
    print("=" * 50)
    
    for num_elements in args.elements:
        dataset = generate_dataset(num_elements, args.scenarios)
        save_dataset(dataset, num_elements, args.scenarios)
        print(f"✅ Completed dataset for {num_elements} elements\n")
    
    print("🎉 All datasets generated successfully!")

if __name__ == "__main__":
    main()
