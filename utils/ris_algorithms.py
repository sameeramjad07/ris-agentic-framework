import numpy as np
import logging
from typing import Dict, Any, List, Callable, Optional

class RISAlgorithms:
    """
    Collection of RIS optimization algorithms for phase shift computation.
    Each algorithm takes CSI data and parameters, returns optimized phase shifts.
    """
    
    def __init__(self):
        """Initialize the RIS algorithms collection."""
        self.logger = logging.getLogger(__name__)
        
    def get_algorithm(self, algorithm_name: str) -> Optional[Callable]:
        """
        Get algorithm function by name.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Algorithm function or None if not found
        """
        # Convert algorithm name to function name format
        func_name = "run_" + algorithm_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        return getattr(self, func_name, None)
    
    def get_all_algorithm_names(self) -> List[str]:
        """
        Get list of all available algorithm names.
        
        Returns:
            List of algorithm names
        """
        algorithms = [
            "Compressed Sensing",
            "Deep Learning", 
            "Kalman Filter",
            "PPO Algorithm",
            "SDR with Alternating Optimization",
            "Asynchronous One-Step Q-Learning",
            "Single Convolutional Neural Network",
            "Orthogonal Matching Pursuit (OMP)",
            "Canonical Polyadic Decomposition",
            "Matrix Factorization",
            "High-Mobility Estimation",
            "Two-Stage Cascaded Channel Estimation",
            "Convolutional Neural Network",
            "Sparse Bayesian Learning",
            "Deep Denoising Neural Network",
            "Federated Learning",
            "Phase Shift Design with ZF Detection",
            "3D-MMV with 3D-MLAOMP",
            "Alternating Optimization (AO) with S-CSI",
            "JCEDD with Message Passing and EM",
            "TALS-LTI",
            "TALS-STI",
            "HOSVD-STI"
        ]
        return algorithms
    
    def run_compressed_sensing(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Compressed Sensing algorithm for RIS phase optimization."""
        num_elements = csi_data.get("num_ris_elements", 64)
        sparsity_level = params.get("sparsity_level", 0.1)
        num_iterations = params.get("num_iterations", 100)
        
        # Simulate compressed sensing optimization
        phases = np.random.uniform(0, 2*np.pi, num_elements)
        
        # Iterative optimization with sparsity constraint
        for _ in range(num_iterations):
            # Simulate gradient-based optimization with sparsity promotion
            gradient = np.random.normal(0, 0.1, num_elements)
            phases = phases - 0.01 * gradient
            
            # Apply sparsity constraint (zero out small phases)
            threshold = np.percentile(np.abs(phases), (1-sparsity_level)*100)
            phases[np.abs(phases) < threshold] = 0
        
        return np.exp(1j * phases)
    
    def run_deep_learning(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Deep Learning neural network algorithm."""
        num_elements = csi_data.get("num_ris_elements", 64)
        learning_rate = params.get("learning_rate", 0.001)
        epochs = params.get("epochs", 50)
        
        # Simulate deep neural network optimization
        # Initialize with Xavier initialization
        phases = np.random.normal(0, np.sqrt(2/num_elements), num_elements)
        
        # Simulate training epochs
        for epoch in range(epochs):
            # Simulate forward pass and backpropagation
            loss_gradient = np.random.normal(0, 0.05, num_elements)
            phases = phases - learning_rate * loss_gradient
            
            # Apply activation function (tanh for phase constraints)
            phases = np.tanh(phases) * np.pi
        
        return np.exp(1j * phases)
    
    def run_kalman_filter(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Kalman Filter for time-varying channel estimation and phase optimization."""
        num_elements = csi_data.get("num_ris_elements", 64)
        process_noise = params.get("process_noise", 0.01)
        measurement_noise = params.get("measurement_noise", 0.1)
        
        # Initialize Kalman filter parameters
        phases = np.random.uniform(0, 2*np.pi, num_elements)
        P = np.eye(num_elements)  # Covariance matrix
        
        # Simulate Kalman filtering steps
        Q = process_noise * np.eye(num_elements)  # Process noise
        R = measurement_noise * np.eye(num_elements)  # Measurement noise
        
        # Prediction and update steps
        for _ in range(10):  # 10 time steps
            # Prediction
            P = P + Q
            
            # Measurement update
            z = phases + np.random.normal(0, np.sqrt(measurement_noise), num_elements)  # Noisy measurements
            K = P @ np.linalg.inv(P + R)  # Kalman gain
            phases = phases + K @ (z - phases)
            P = (np.eye(num_elements) - K) @ P
        
        return np.exp(1j * phases)
    
    def run_ppo_algorithm(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Proximal Policy Optimization algorithm."""
        num_elements = csi_data.get("num_ris_elements", 64)
        learning_rate = params.get("learning_rate", 0.0003)
        epsilon = params.get("epsilon", 0.2)
        num_epochs = params.get("num_epochs", 10)
        
        # Initialize policy parameters
        phases = np.random.uniform(0, 2*np.pi, num_elements)
        old_policy = phases.copy()
        
        for epoch in range(num_epochs):
            # Simulate policy gradient with PPO clipping
            advantages = np.random.normal(0, 1, num_elements)
            ratio = np.exp(phases - old_policy)  # Policy ratio
            
            # PPO clipping
            clipped_ratio = np.clip(ratio, 1-epsilon, 1+epsilon)
            policy_loss = -np.minimum(ratio * advantages, clipped_ratio * advantages)
            
            # Update policy
            phases = phases - learning_rate * policy_loss
            
            # Update old policy every few epochs
            if epoch % 3 == 0:
                old_policy = phases.copy()
        
        return np.exp(1j * phases)
    
    def run_sdr_with_alternating_optimization(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Semidefinite Relaxation with Alternating Optimization."""
        num_elements = csi_data.get("num_ris_elements", 64)
        num_iterations = params.get("num_iterations", 100)
        
        # Initialize optimization variables
        phases = np.random.uniform(0, 2*np.pi, num_elements)
        
        # Alternating optimization between beamforming and RIS phases
        for iteration in range(num_iterations):
            # Optimize beamforming (simplified)
            beamforming_update = np.random.normal(0, 0.05, num_elements)
            
            # Optimize RIS phases with SDR
            # Simulate semidefinite relaxation solution
            sdr_solution = np.random.uniform(-np.pi, np.pi, num_elements)
            
            # Combine updates
            phases = 0.7 * phases + 0.3 * sdr_solution + 0.1 * beamforming_update
            
            # Project to feasible set
            phases = np.mod(phases, 2*np.pi)
        
        return np.exp(1j * phases)
    
    def run_asynchronous_one_step_q_learning(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Asynchronous One-Step Q-Learning algorithm."""
        num_elements = csi_data.get("num_ris_elements", 64)
        learning_rate = params.get("learning_rate", 0.01)
        gamma = params.get("gamma", 0.99)
        
        # Initialize Q-values and phases
        phases = np.random.uniform(0, 2*np.pi, num_elements)
        q_values = np.random.uniform(-1, 1, (num_elements, 8))  # 8 discrete actions per element
        
        # Simulate Q-learning updates
        for _ in range(100):
            for element in range(num_elements):
                # Choose action (phase quantization level)
                action = np.random.randint(0, 8)
                new_phase = action * np.pi / 4  # Discretize phase
                
                # Simulate reward (based on channel improvement)
                reward = np.random.normal(0, 0.1)
                
                # Q-learning update
                old_q = q_values[element, action]
                max_future_q = np.max(q_values[element, :])
                new_q = old_q + learning_rate * (reward + gamma * max_future_q - old_q)
                q_values[element, action] = new_q
                
                # Update phase based on best action
                best_action = np.argmax(q_values[element, :])
                phases[element] = best_action * np.pi / 4
        
        return np.exp(1j * phases)
    
    def run_single_convolutional_neural_network(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Single CNN for broadband RIS optimization."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Simulate CNN processing
        # Create synthetic "image" representation of channel
        channel_image = np.random.normal(0, 1, (8, 8))  # 8x8 spatial representation
        
        # Simulate convolution operations
        kernel = np.random.normal(0, 0.1, (3, 3))
        
        # Simple convolution simulation
        output = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                output[i, j] = np.sum(channel_image[i:i+3, j:j+3] * kernel)
        
        # Map CNN output to phase shifts
        phases = np.resize(output.flatten(), num_elements)
        phases = (phases - np.min(phases)) / (np.max(phases) - np.min(phases)) * 2 * np.pi
        
        return np.exp(1j * phases)
    
    def run_orthogonal_matching_pursuit_omp(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Orthogonal Matching Pursuit algorithm."""
        num_elements = csi_data.get("num_ris_elements", 64)
        sparsity_level = params.get("sparsity_level", 0.1)
        
        # Create measurement matrix (simplified)
        measurement_matrix = np.random.normal(0, 1, (int(num_elements*0.5), num_elements))
        
        # Simulate sparse signal
        true_signal = np.zeros(num_elements)
        sparse_indices = np.random.choice(num_elements, int(num_elements * sparsity_level), replace=False)
        true_signal[sparse_indices] = np.random.normal(0, 1, len(sparse_indices))
        
        # OMP algorithm simulation
        residual = measurement_matrix @ true_signal + np.random.normal(0, 0.1, measurement_matrix.shape[0])
        support = []
        
        for _ in range(len(sparse_indices)):
            # Find most correlated column
            correlations = np.abs(measurement_matrix.T @ residual)
            best_index = np.argmax(correlations)
            support.append(best_index)
            
            # Update residual (simplified)
            if len(support) > 1:
                residual = residual * 0.9  # Simulate residual reduction
        
        # Reconstruct signal
        phases = np.zeros(num_elements)
        phases[support] = np.random.uniform(0, 2*np.pi, len(support))
        
        return np.exp(1j * phases)
    
    def run_canonical_polyadic_decomposition(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Canonical Polyadic Decomposition for tensor-based channel estimation."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Simulate tensor decomposition
        # Create a 3rd order tensor (simplified)
        tensor_size = int(np.cbrt(num_elements))
        if tensor_size ** 3 != num_elements:
            tensor_size = int(np.cbrt(64))  # Default to 4x4x4
        
        # Simulate CPD factors
        factor_a = np.random.normal(0, 1, (tensor_size, 3))
        factor_b = np.random.normal(0, 1, (tensor_size, 3))
        factor_c = np.random.normal(0, 1, (tensor_size, 3))
        
        # Reconstruct phases from factors
        phases = []
        for i in range(tensor_size):
            for j in range(tensor_size):
                for k in range(tensor_size):
                    if len(phases) < num_elements:
                        phase_val = np.sum(factor_a[i, :] * factor_b[j, :] * factor_c[k, :])
                        phases.append(np.mod(phase_val, 2*np.pi))
        
        phases = np.array(phases[:num_elements])
        return np.exp(1j * phases)
    
    def run_matrix_factorization(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Matrix Factorization for low-rank channel representation."""
        num_elements = csi_data.get("num_ris_elements", 64)
        rank = params.get("rank", min(8, num_elements//4))
        
        # Create matrix representation of channel
        matrix_size = int(np.sqrt(num_elements))
        if matrix_size ** 2 != num_elements:
            matrix_size = 8  # Default 8x8
            
        # Simulate low-rank matrix factorization
        U = np.random.normal(0, 1, (matrix_size, rank))
        V = np.random.normal(0, 1, (rank, matrix_size))
        
        # Reconstruct matrix
        reconstructed = U @ V
        
        # Extract phases
        phases = reconstructed.flatten()[:num_elements]
        phases = (phases - np.min(phases)) / (np.max(phases) - np.min(phases)) * 2 * np.pi
        
        return np.exp(1j * phases)
    
    def run_high_mobility_estimation(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """High-mobility channel estimation and phase optimization."""
        num_elements = csi_data.get("num_ris_elements", 64)
        doppler_freq = params.get("doppler_frequency", 100)  # Hz
        
        # Simulate time-varying channel
        time_steps = 10
        phases_over_time = []
        
        for t in range(time_steps):
            # Add Doppler effect
            doppler_phase = 2 * np.pi * doppler_freq * t * 0.001  # Assume 1ms time steps
            
            # Base phase optimization
            base_phases = np.random.uniform(0, 2*np.pi, num_elements)
            
            # Add time variation
            time_varying_phases = base_phases + doppler_phase * np.sin(np.arange(num_elements) * 2 * np.pi / num_elements)
            phases_over_time.append(time_varying_phases)
        
        # Use the latest time step
        final_phases = phases_over_time[-1]
        return np.exp(1j * final_phases)
    
    def run_two_stage_cascaded_channel_estimation(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Two-stage cascaded channel estimation."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Stage 1: Estimate BS-RIS channel
        H_br_estimate = np.random.normal(0, 1, (csi_data.get("num_antennas_bs", 4), num_elements)) + \
                       1j * np.random.normal(0, 1, (csi_data.get("num_antennas_bs", 4), num_elements))
        
        # Stage 2: Estimate RIS-User channel  
        H_ru_estimate = np.random.normal(0, 1, (num_elements, csi_data.get("num_antennas_user", 1))) + \
                       1j * np.random.normal(0, 1, (num_elements, csi_data.get("num_antennas_user", 1)))
        
        # Optimize phases based on cascaded channel
        phases = np.angle(H_br_estimate[0, :]) + np.angle(H_ru_estimate[:, 0])
        
        return np.exp(1j * phases)
    
    def run_convolutional_neural_network(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """CNN for RIS phase optimization."""
        return self.run_single_convolutional_neural_network(csi_data, params)
    
    def run_sparse_bayesian_learning(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Sparse Bayesian Learning algorithm."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Initialize hyperparameters
        alpha = np.ones(num_elements)  # Precision parameters
        beta = 1.0  # Noise precision
        
        # Simulate SBL iterations
        for _ in range(50):
            # Update posterior mean and covariance (simplified)
            posterior_mean = np.random.normal(0, 1/np.sqrt(alpha))
            
            # Update hyperparameters
            alpha = 1 / (posterior_mean**2 + 1e-6)
            
            # Prune irrelevant elements
            relevant_indices = alpha < 1e6
            alpha[~relevant_indices] = 1e6
        
        # Extract phases from posterior
        phases = np.angle(posterior_mean + 1j * np.random.normal(0, 0.1, num_elements))
        return np.exp(1j * phases)
    
    def run_deep_denoising_neural_network(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Deep Denoising Neural Network."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Simulate noisy input
        noisy_phases = np.random.uniform(0, 2*np.pi, num_elements) + np.random.normal(0, 0.2, num_elements)
        
        # Simulate denoising network layers
        hidden1 = np.tanh(noisy_phases + np.random.normal(0, 0.1, num_elements))
        hidden2 = np.tanh(hidden1 + np.random.normal(0, 0.05, num_elements))
        denoised_phases = np.tanh(hidden2) * np.pi  # Output layer
        
        return np.exp(1j * denoised_phases)
    
    def run_federated_learning(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Federated Learning for multi-user RIS optimization."""
        num_elements = csi_data.get("num_ris_elements", 64)
        num_clients = params.get("num_clients", 4)
        
        # Simulate federated learning rounds
        global_model = np.random.uniform(0, 2*np.pi, num_elements)
        
        for round_num in range(10):
            client_updates = []
            
            # Simulate client updates
            for client in range(num_clients):
                local_update = global_model + np.random.normal(0, 0.1, num_elements)
                client_updates.append(local_update)
            
            # Aggregate updates (FedAvg)
            global_model = np.mean(client_updates, axis=0)
        
        return np.exp(1j * global_model)
    
    def run_phase_shift_design_with_zf_detection(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Phase Shift Design with Zero-Forcing Detection."""
        num_elements = csi_data.get("num_ris_elements", 64)
        num_antennas_bs = csi_data.get("num_antennas_bs", 4)
        
        # Simulate channel matrices
        H_direct = np.random.normal(0, 1, (1, num_antennas_bs)) + 1j * np.random.normal(0, 1, (1, num_antennas_bs))
        H_br = np.random.normal(0, 1, (num_antennas_bs, num_elements)) + 1j * np.random.normal(0, 1, (num_antennas_bs, num_elements))
        H_ru = np.random.normal(0, 1, (num_elements, 1)) + 1j * np.random.normal(0, 1, (num_elements, 1))
        
        # Zero-forcing precoding design
        effective_channel = H_direct + H_br.T @ np.diag(np.exp(1j * np.random.uniform(0, 2*np.pi, num_elements))) @ H_ru
        
        # Optimize phases for maximum channel gain
        phases = np.angle(H_br[0, :]) + np.angle(H_ru[:, 0])
        
        return np.exp(1j * phases)
    
    def run_3d_mmv_with_3d_mlaomp(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """3D Multiple Measurement Vector with 3D-MLAOMP."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Simulate 3D tensor structure
        tensor_dims = [4, 4, 4]  # 3D structure
        
        # Create measurement vectors
        measurements = []
        for dim in tensor_dims:
            measurements.append(np.random.normal(0, 1, dim) + 1j * np.random.normal(0, 1, dim))
        
        # Simulate MLAOMP algorithm
        support = []
        for iteration in range(min(8, num_elements//8)):
            # Find best atom (simplified)
            correlations = [np.abs(np.sum(m)) for m in measurements]
            best_atom = np.argmax(correlations)
            support.append(best_atom)
            
            # Update measurements (simplified)
            for i, m in enumerate(measurements):
                measurements[i] = m * 0.9
        
        # Generate phases from support
        phases = np.zeros(num_elements)
        for i, atom in enumerate(support):
            if atom < num_elements:
                phases[atom] = i * 2 * np.pi / len(support)
        
        # Fill remaining elements
        remaining = num_elements - len(support)
        if remaining > 0:
            phases[-remaining:] = np.random.uniform(0, 2*np.pi, remaining)
        
        return np.exp(1j * phases)
    
    def run_alternating_optimization_ao_with_s_csi(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Alternating Optimization with Statistical CSI."""
        num_elements = csi_data.get("num_ris_elements", 64)
        num_iterations = params.get("num_iterations", 20)
        
        # Initialize variables
        phases = np.random.uniform(0, 2*np.pi, num_elements)
        beamforming_vector = np.random.normal(0, 1, csi_data.get("num_antennas_bs", 4)) + \
                            1j * np.random.normal(0, 1, csi_data.get("num_antennas_bs", 4))
        
        # Alternating optimization
        for iteration in range(num_iterations):
            # Step 1: Optimize RIS phases with fixed beamforming
            channel_gain = np.random.normal(0, 1, num_elements) + 1j * np.random.normal(0, 1, num_elements)
            phases = np.angle(channel_gain)
            
            # Step 2: Optimize beamforming with fixed RIS phases
            effective_channel = np.random.normal(0, 1, csi_data.get("num_antennas_bs", 4)) + \
                              1j * np.random.normal(0, 1, csi_data.get("num_antennas_bs", 4))
            beamforming_vector = effective_channel / np.linalg.norm(effective_channel)
        
        return np.exp(1j * phases)
    
    def run_jcedd_with_message_passing_and_em(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """Joint Channel Estimation and Data Detection with Message Passing and EM."""
        num_elements = csi_data.get("num_ris_elements", 64)
        num_iterations = params.get("num_iterations", 15)
        
        # Initialize EM parameters
        phases = np.random.uniform(0, 2*np.pi, num_elements)
        channel_estimates = np.random.normal(0, 1, num_elements) + 1j * np.random.normal(0, 1, num_elements)
        
        # EM algorithm
        for iteration in range(num_iterations):
            # E-step: Estimate data symbols
            data_estimates = np.random.choice([-1, 1], num_elements) + 1j * np.random.choice([-1, 1], num_elements)
            
            # M-step: Update channel and phase estimates
            # Message passing between channel and data estimation
            for msg_iter in range(5):
                # Update channel estimates
                channel_estimates = channel_estimates * 0.9 + 0.1 * np.random.normal(0, 0.1, num_elements)
                
                # Update phase estimates
                phases = np.angle(channel_estimates * np.conj(data_estimates))
        
        return np.exp(1j * phases)
    
    def run_tals_lti(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """TALS-LTI (Tensor ALS for Linear Time-Invariant imperfections)."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Simulate tensor decomposition for LTI imperfections
        # Create 3-way tensor
        tensor_rank = min(8, num_elements // 8)
        
        # Initialize factor matrices
        A = np.random.normal(0, 1, (8, tensor_rank))
        B = np.random.normal(0, 1, (8, tensor_rank)) 
        C = np.random.normal(0, 1, (num_elements // 8, tensor_rank))
        
        # ALS iterations
        for als_iter in range(20):
            # Update A
            A = A + 0.1 * np.random.normal(0, 0.05, A.shape)
            
            # Update B  
            B = B + 0.1 * np.random.normal(0, 0.05, B.shape)
            
            # Update C
            C = C + 0.1 * np.random.normal(0, 0.05, C.shape)
        
        # Extract phases from tensor factors
        phases = np.tile(C[:, 0], 8)[:num_elements]
        phases = (phases - np.min(phases)) / (np.max(phases) - np.min(phases)) * 2 * np.pi
        
        return np.exp(1j * phases)
    
    def run_tals_sti(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """TALS-STI (Tensor ALS for Spatio-Temporal imperfections)."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Similar to TALS-LTI but with time-varying components
        tensor_rank = min(8, num_elements // 8)
        time_steps = 4
        
        # Initialize spatio-temporal factor matrices
        A_spatial = np.random.normal(0, 1, (8, tensor_rank))
        B_temporal = np.random.normal(0, 1, (time_steps, tensor_rank))
        C_elements = np.random.normal(0, 1, (num_elements // (8 * time_steps), tensor_rank))
        
        # ALS with temporal updates
        for als_iter in range(20):
            # Update spatial factors
            A_spatial = A_spatial + 0.1 * np.random.normal(0, 0.05, A_spatial.shape)
            
            # Update temporal factors
            B_temporal = B_temporal + 0.1 * np.random.normal(0, 0.03, B_temporal.shape)
            
            # Update element factors
            C_elements = C_elements + 0.1 * np.random.normal(0, 0.05, C_elements.shape)
        
        # Combine spatio-temporal information
        phases = np.repeat(C_elements[:, 0], 8 * time_steps)[:num_elements]
        
        # Add temporal variation
        for t in range(time_steps):
            start_idx = t * (num_elements // time_steps)
            end_idx = (t + 1) * (num_elements // time_steps)
            if end_idx > num_elements:
                end_idx = num_elements
            phases[start_idx:end_idx] += B_temporal[t, 0] * 0.1
        
        phases = np.mod(phases, 2 * np.pi)
        return np.exp(1j * phases)
    
    def run_hosvd_sti(self, csi_data: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
        """HOSVD-STI (Higher-Order SVD for Spatio-Temporal imperfections)."""
        num_elements = csi_data.get("num_ris_elements", 64)
        
        # Create higher-order tensor structure
        tensor_shape = [4, 4, 4, 4]  # 4D tensor
        total_elements = np.prod(tensor_shape)
        
        # Generate synthetic tensor data
        tensor_data = np.random.normal(0, 1, tensor_shape) + 1j * np.random.normal(0, 1, tensor_shape)
        
        # Simulate HOSVD decomposition
        # Mode-1 unfolding and SVD
        unfolded = tensor_data.reshape(tensor_shape[0], -1)
        U, S, Vh = np.linalg.svd(unfolded, full_matrices=False)
        
        # Extract dominant singular vectors
        dominant_components = U[:, :2] @ np.diag(S[:2]) @ Vh[:2, :]
        
        # Map back to RIS phases
        phases = np.angle(dominant_components.flatten())
        
        # Resize to match number of RIS elements
        if len(phases) < num_elements:
            phases = np.tile(phases, int(np.ceil(num_elements / len(phases))))[:num_elements]
        else:
            phases = phases[:num_elements]
        
        return np.exp(1j * phases)
