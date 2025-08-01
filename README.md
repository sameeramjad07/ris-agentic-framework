# RIS Agentic System: Intelligent Optimization for Reconfigurable Intelligent Surfaces

A focused and streamlined system demonstrating intelligent Reconfigurable Intelligent Surface (RIS) algorithm selection using Large Language Models (LLMs) for optimal signal enhancement.

## Overview

This project provides a robust framework for generating datasets, evaluating RIS optimization algorithms, and showcasing the performance benefits of an agentic, LLM-driven approach for selecting the best RIS optimization method. The system intelligently chooses from a simplified set of core algorithms to maximize performance metrics based on given communication scenario characteristics.

## Features

- **Intelligent Algorithm Selection**: Leverages LLMs to select optimal RIS optimization algorithms based on scenario-specific channel conditions.
- **Core Optimization Algorithms**: Implements three fundamental RIS optimization algorithms: **Gradient Descent (GD)**, **Manifold Optimization (MO)**, and **Alternating Optimization (AO)**.
- **Dataset Generation**: Generates synthetic datasets with various RIS element counts and scenarios, providing analytical optimal phase shifts as ground truth.
- **Streamlined Channel Modeling**: Focuses on the essential direct channel, Base Station (BS)-RIS channel, and RIS-User channel components.
- **Comprehensive Evaluation & Visualization**: Compares the agentic approach against all implemented algorithms, random phase shifts, and analytical optima, generating detailed performance plots.

---

## Directory Structure

```
ris_agentic_system/
├── config/ # Configuration and knowledge base
│ ├── init.py
│ ├── settings.py
│ └── knowledge_base.json
├── agents/ # Core agent implementations
│ ├── init.py
│ ├── coordinator_agent.py
│ ├── optimizer_agent.py
│ └── solver_agent.py
├── utils/ # Utilities (algorithms, metrics, LLM interface)
│ ├── init.py
│ ├── ris_algorithms.py
│ ├── performance_metrics.py
│ └── llm_interface.py
├── generate.py # Script for dataset generation
├── run.py # Script for evaluation and testing
├── .env # Environment variables (e.g., API keys)
└── requirements.txt # Project dependencies
```

---

## Setup

1.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Key**:

    - Obtain a Cerebras API key from [cloud.cerebras.ai](https://cloud.cerebras.ai).
    - Create a `.env` file in the project root and add your API key:
      ```bash
      echo "CEREBRAS_API_KEY=your_key_here" > .env
      ```

3.  **Knowledge Base**:
    - Ensure `config/knowledge_base.json` contains the simplified algorithm knowledge base content (as described in the "Simplified Knowledge Base" section below).

---

## Usage

The project workflow involves two main steps: dataset generation and evaluation.

### 1. Dataset Generation (`generate.py`)

This script creates synthetic datasets based on specified RIS element counts and the number of scenarios. Each dataset entry includes analytical optimal phase shifts as ground truth and derived features for LLM decision-making.

```bash
# Generate datasets with 8, 16, 32, and 64 RIS elements, each with 100 scenarios
python generate.py --elements 8 16 32 64 --scenarios 100
```

This will create files like `dataset_8elements_100scenarios.json`, `dataset_64elements_100scenarios.json`, etc., in the project root.

### 2. Evaluation & Testing (`run.py`)

This script evaluates the performance of the agentic system and other algorithms against the generated datasets.

```bash
# Run evaluation on two generated datasets, limiting to 10 scenarios per dataset
# and generating individual plots for each scenario
python run.py --datasets dataset_8elements_100scenarios.json dataset_64elements_100scenarios.json --max-scenarios 10 --individual-plots
```

## Key Components

### Agents

- **CoordinatorAgent**: Orchestrates the entire simulation and evaluation pipeline.
- **OptimizerAgent**: The core LLM-based agent responsible for intelligent algorithm selection based on the scenario's characteristics, leveraging the `knowledge_base.json`.
- **SolverAgent**: Executes the chosen RIS optimization algorithm and calculates performance metrics.

### Algorithms

The system implements three fundamental state-of-the-art RIS optimization algorithms:

- Gradient Descent (GD)
- Manifold Optimization (MO)
- Alternating Optimization (AO)

### Knowledge Base

The LLM in the `OptimizerAgent` utilizes a simplified knowledge base in `config/knowledge_base.json` to make informed decisions. It categorizes scenarios into five clear types, mapping them to the most suitable optimization algorithm:

- **Low SNR with Strong Direct Channel** → Alternating Optimization (AO)
- **High SNR with Weak Direct Channel** → Manifold Optimization (MO)
- **Medium SNR with Good Phase Alignment** → Gradient Descent (GD)
- **Poor Channel Conditions** → Alternating Optimization (AO)
- **Strong RIS Control** → Manifold Optimization (MO)

### Dataset Format

The generated datasets adhere to the following structure:

```json
{
  "input": {
    "direct_channel_real": -0.0,
    "direct_channel_imag": 0.0003,
    "bs_ris_channel_real": [-0.0191, -0.0096, -0.0065, 0.0288],
    "bs_ris_channel_imag": [0.015, 0.0139, 0.0151, 0.0002],
    "ris_user_channel_real": [-0.0241, 0.0277, 0.0312, 0.0346],
    "ris_user_channel_imag": [0.0324, 0.0194, -0.0012, 0.0125],
    "num_ris_elements": 4
  },
  "output": {
    "optimized_phase_shifts": [1.5708, 0.0, 5.7596, 2.0944]
  }
}
```

## Output

The `run.py` script generates comprehensive visualization and performance summaries:

- **Individual Scenario Plots**: For each scenario, plots show performance (e.g., SNR) versus transmit power, displaying 6 performance lines: Random, Analytical Optimal, Gradient Descent, Manifold Optimization, Alternating Optimization, and Agentic selection.
- **Cumulative Average Performance Plots**: Visualizes the average performance across all processed scenarios.
- **Multi-element Comparison Plots**: Compares all methods across different RIS element counts, showing 12 performance lines (6 methods × 2 element counts for the example usage).
- **Console Output:**: Detailed performance summary statistics, including the agentic system's improvement over baselines, will be printed to the console.

## Troubleshooting

1. **API Key Issues**: Ensure your Cerebras API key is valid and has sufficient credits
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **JSON Errors**: Ensure `algorithms_kb.json` contains valid JSON
4. **Plot Issues**: Check that the `plots/` directory can be created

## License

This project is for research and educational purposes.

## Citation

If you use this code in your research, please cite:

```
@software{ris_agentic_system,
  title={RIS Agentic System: Intelligent Algorithm Selection for Reconfigurable Intelligent Surfaces},
  year={2025},
  url={[https://github.com/sameeramjad07/ris-agentic-framework.git](https://github.com/sameeramjad07/ris-agentic-framework.git)}
}
```
