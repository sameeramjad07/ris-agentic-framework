"""Generate RIS scenario datasets for evaluation."""

import json
import numpy as np
import argparse
import os
from typing import List, Dict, Any
from utils.performance_metrics import PerformanceMetrics
from config.settings import Settings

def generate_channel_scenario(num_elements: int) -> Dict[str, Any]:
    """Generate a random RIS scenario with proper analytical solution."""
    
    # Generate random channels with proper scaling
    h_d = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
    
    # BS-RIS channel (1 x N): each element represents BS to RIS element i
    H_br = (np.random.randn(num_elements) + 1j * np.random.randn(num_elements)) / np.sqrt(2)
    
    # RIS-User channel (N x 1): each element represents RIS element i to User
    h_ru = (np.random.randn(num_elements) + 1j * np.random.randn(num_elements)) / np.sqrt(2)
    
    # Calculate optimal phases analytically
    # For maximum SNR: align all phases to maximize |h_d + sum(H_br[i] * e^(jθ_i) * h_ru[i])|
    # Optimal phase for element i: θ_i = -angle(H_br[i] * h_ru[i]) + angle(h_d)
    cascaded = H_br * h_ru  # Element-wise product for each RIS element
    
    # Method 1: Align each RIS path with direct path
    reference_phase = np.angle(h_d) if abs(h_d) > 1e-10 else 0
    optimal_phases = reference_phase - np.angle(cascaded)
    optimal_phases = np.mod(optimal_phases, 2 * np.pi)
    
    # Verify the solution by computing SNR
    e_opt = np.exp(1j * optimal_phases)
    combined_optimal = h_d + np.sum(cascaded * e_opt)
    optimal_snr = 10 * np.log10(abs(combined_optimal)**2 / 1e-12)
    
    # Compute derived features
    direct_channel_norm = float(abs(h_d))
    G_norm = float(np.linalg.norm(H_br))
    hr_norm = float(np.linalg.norm(h_ru))
    
    # Phase alignment score: how well phases naturally align
    if np.linalg.norm(cascaded) > 1e-10:
        phase_alignment_score = float(abs(np.sum(cascaded)) / (np.linalg.norm(cascaded) * len(cascaded)))
    else:
        phase_alignment_score = 0.0
    
    # Estimated SNR with optimal phases
    estimated_snr = float(optimal_snr)
    objective = "maximize_snr"
    
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
    
    # Verify our analytical solution
    computed_snr = PerformanceMetrics.calculate_snr(optimal_phases.tolist(), input_data)
    if abs(computed_snr - optimal_snr) > 1.0:  # Allow 1 dB tolerance
        print(f"Warning: SNR mismatch - Analytical: {optimal_snr:.2f}, Computed: {computed_snr:.2f}")
    
    return {
        "input": input_data,
        "output": {
            "optimized_phase_shifts": optimal_phases.tolist(),
            "optimal_snr": float(optimal_snr)
        }
    }

def generate_dataset(num_elements: int, num_scenarios: int) -> List[Dict[str, Any]]:
    """Generate a complete dataset."""
    dataset = []
    print(f"Generating {num_scenarios} scenarios with {num_elements} RIS elements...")
    
    for i in range(num_scenarios):
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_scenarios} scenarios")
        
        # Generate scenario and validate
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                scenario = generate_channel_scenario(num_elements)
                
                # Basic validation
                if len(scenario['input']['bs_ris_channel_real']) != num_elements:
                    raise ValueError("Channel dimension mismatch")
                if len(scenario['output']['optimized_phase_shifts']) != num_elements:
                    raise ValueError("Phase dimension mismatch")
                    
                # SNR sanity check
                snr = scenario['output']['optimal_snr']
                if not np.isfinite(snr) or snr < -50 or snr > 200:
                    raise ValueError(f"Invalid SNR: {snr}")
                
                dataset.append(scenario)
                break
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for scenario {i + 1}: {e}")
                if attempt == max_attempts - 1:
                    print(f"Failed to generate valid scenario {i + 1} after {max_attempts} attempts")
    
    print(f"Successfully generated {len(dataset)} valid scenarios")
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], num_elements: int, num_scenarios: int):
    """Save dataset to file."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    filename = f"data/dataset_{num_elements}elements_{num_scenarios}scenarios.json"
    
    with open(filename, 'w') as f:
        for scenario in dataset:
            f.write(json.dumps(scenario) + '\n')
    
    # Print dataset statistics
    snrs = [scenario['output']['optimal_snr'] for scenario in dataset]
    print(f"Dataset saved to {filename}")
    print(f"SNR statistics - Min: {min(snrs):.1f} dB, Max: {max(snrs):.1f} dB, Mean: {np.mean(snrs):.1f} dB")


def main():
    settings = Settings()

    parser = argparse.ArgumentParser(description='Generate RIS scenario datasets')
    parser.add_argument('--elements', type=int, nargs='+', default=settings.DEFAULT_NUM_ELEMENTS,
                       help='Number of RIS elements for each dataset')
    parser.add_argument('--scenarios', type=int, default=settings.DEFAULT_NUM_SCENARIOS,
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
