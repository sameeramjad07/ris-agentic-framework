"""RIS optimization algorithms implementation."""

import numpy as np
from typing import Tuple

class RISAlgorithms:
    """Implementation of the three core RIS optimization algorithms."""
    
    def gradient_descent_adam(self, h_d, H, h_1, N, transmit_power, 
                             learning_rate=0.01, max_iterations=1000, 
                             beta1=0.9, beta2=0.999, epsilon=1e-8, discrete_phases=None):
        """Gradient descent with Adam optimizer."""
        h_r = self.compute_cascaded_channel(H, h_1)
        theta = np.random.uniform(0, 2 * np.pi, N)
        e = np.exp(1j * theta)
        powers = [max(self.compute_received_power(h_d, h_r, e, transmit_power), 1e-20)]
        snrs = [self.compute_snr(powers[0], noise_power=1e-12)]
        iterations = [0]

        m = np.zeros(N)  # First moment
        v = np.zeros(N)  # Second moment
        t = 0  # Time step
        
        prev_power = powers[0]
        convergence_count = 0

        for iteration in range(max_iterations):
            t += 1
            grad = self.compute_gd_gradient(h_d, h_r, e, transmit_power)
            
            # Check for NaN or inf gradients
            if np.any(~np.isfinite(grad)):
                break
                
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update with clipping to prevent instability
            update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            update = np.clip(update, -np.pi/10, np.pi/10)  # Clip large updates
            theta -= update
            
            if discrete_phases is not None:
                theta = np.array([min(discrete_phases, key=lambda x: abs(x - t)) for t in theta])
            theta = np.mod(theta, 2 * np.pi)
            e = np.exp(1j * theta)
            power = max(self.compute_received_power(h_d, h_r, e, transmit_power), 1e-20)
            powers.append(power)
            snrs.append(self.compute_snr(power, noise_power=1e-12))
            iterations.append(iteration + 1)
            
            # Check convergence
            if abs(power - prev_power) < 1e-6 * prev_power:
                convergence_count += 1
                if convergence_count >= 5:
                    break
            else:
                convergence_count = 0
            prev_power = power

        return theta, powers, snrs, iterations
    
    def manifold_optimization_adam(h_d, H, h_1, N, transmit_power, learning_rate=0.01, max_iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8, discrete_phases=None):
        """
        Perform manifold optimization with Adam optimizer to optimize RIS phase shifts.

        Parameters:
        h_d : complex, direct channel
        H : ndarray, BS-to-RIS channel (1 x N)
        h_1 : ndarray, RIS-to-user channel (N,)
        N : int, number of RIS elements
        transmit_power : float, transmit power (Watts)
        learning_rate : float, step size
        max_iterations : int, number of iterations
        beta1, beta2, epsilon : Adam parameters
        discrete_phases : array, discrete phase shift values (radians), or None for continuous

        Returns:
        tuple, (optimized phase shifts, powers, SNRs, iterations)
        """
        h_r = compute_cascaded_channel(H, h_1)
        theta = np.random.uniform(0, 2 * np.pi, N)
        e = np.exp(1j * theta)
        powers = [compute_received_power(h_d, h_r, e, transmit_power)]
        snrs = [compute_snr(powers[0])]
        iterations = [0]

        # Adam variables
        m = np.zeros(N, dtype=complex)  # First moment
        v = np.zeros(N)  # Second moment
        t = 0  # Time step

        # Manifold variables
        c_l = np.zeros(N, dtype=complex) # Initialize c_l as a complex numpy array
        grad_riemannian_prev = None # To store the previous Riemannian gradient

        for iteration in range(max_iterations):
            t += 1
            grad_euclidean = compute_manifold_gradient(h_d, h_r, e, transmit_power)
            grad_riemannian, grad_k = project_to_tangent_space(e, grad_euclidean) # Get current Riemannian and Euclidean gradients

            # Adam update
            m = beta1 * m + (1 - beta1) * grad_riemannian
            v = beta2 * v + (1 - beta2) * (np.abs(grad_riemannian) ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Calculate ro using ro_calc if previous gradient exists
            current_ro = 0.0  # Default or initial ro
            if grad_riemannian_prev is not None:
                current_ro = ro_calc(grad_riemannian, grad_riemannian_prev) # Use current and previous Riemannian gradients

            # Manifold retraction
            step = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            e, c_l = retract_to_manifold(e, step, ro=current_ro, c_l=c_l, a_l=1.0)

            power = compute_received_power(h_d, h_r, e, transmit_power)
            powers.append(power)
            snrs.append(compute_snr(power))
            iterations.append(iteration + 1)

            # Store current Riemannian gradient for the next iteration
            grad_riemannian_prev = grad_riemannian

        theta = np.angle(e)
        return theta, powers, snrs, iterations
    
    def alternating_optimization(self, h_d, H, h_1, N, transmit_power, 
                               max_iterations=1000, tolerance=1e-6, discrete_phases=None):
        """Alternating optimization algorithm for single-user MISO RIS system."""
        h_r = self.compute_cascaded_channel(H, h_1)
        theta = np.random.uniform(0, 2 * np.pi, N)
        e = np.exp(1j * theta)
        powers = [max(self.compute_received_power(h_d, h_r, e, transmit_power), 1e-20)]
        snrs = [self.compute_snr(powers[0], noise_power=1e-12)]
        iterations = [0]

        for iteration in range(max_iterations):
            prev_e = e.copy()
            
            for n in range(N):
                # Compute combined channel excluding the nth element
                combined_without_n = h_d + np.sum(h_r * e) - h_r[n] * e[n]
                
                # Optimal phase for nth element
                if abs(h_r[n]) > 1e-12:  # Avoid division by zero
                    optimal_phase = -np.angle(h_r[n] * combined_without_n.conj())
                else:
                    optimal_phase = 0.0  # Keep current phase if channel is zero
                    
                if discrete_phases is not None:
                    optimal_phase = min(discrete_phases, key=lambda x: abs(x - optimal_phase))
                
                e[n] = np.exp(1j * optimal_phase)

            power = max(self.compute_received_power(h_d, h_r, e, transmit_power), 1e-20)
            powers.append(power)
            snrs.append(self.compute_snr(power, noise_power=1e-12))
            iterations.append(iteration + 1)

            # Check convergence based on phase change
            phase_change = np.linalg.norm(e - prev_e)
            if phase_change < tolerance:
                break

        theta = np.angle(e)
        return theta, powers, snrs, iterations
    
    # Helper methods
    def compute_cascaded_channel(self, H, h_1):
        """Compute cascaded channel."""
        H_flat = H.flatten() if H.ndim > 1 else H
        h_1_flat = h_1.flatten() if h_1.ndim > 1 else h_1
        return H_flat * h_1_flat
    
    def compute_received_power(self, h_d, h_r, e, transmit_power):
        """Compute received power."""
        combined = h_d + np.dot(h_r, e)
        return transmit_power * np.abs(combined)**2
    
    def compute_snr(self, power, noise_power=1e-12):
        """Compute SNR in dB."""
        return 10 * np.log10(max(power, 1e-20) / noise_power)
    
    def compute_gd_gradient(self, h_d, h_r, e, transmit_power):
        """Compute gradient for gradient descent."""
        N = len(h_r)
        combined = h_d + np.dot(h_r, e)
        grad = np.zeros(N)
        for m in range(N):
            # Gradient of |h_d + sum(h_r[i] * e[i])|^2 w.r.t. theta[m]
            term = h_r[m] * combined.conj()
            grad[m] = -2 * transmit_power * np.real(1j * e[m] * term)
        return grad
    
    def compute_manifold_gradient(h_d, h_r, e, transmit_power):
        """
        Compute the Euclidean gradient of the negative received power with respect to e.

        Parameters:
        h_d : complex, direct channel
        h_r : ndarray, cascaded channel (N,)
        e : ndarray, RIS coefficients (N,)
        transmit_power : float, transmit power (Watts)

        Returns:
        ndarray, Euclidean gradient with respect to e (N,)
        """
        combined = h_d + np.dot(h_r, e)
        grad = -transmit_power * h_r.conj() * combined
        return grad

    def project_to_tangent_space(e, grad):
        """
        Project the Euclidean gradient onto the tangent space of the complex circle manifold.

        Parameters:
        e : ndarray, RIS coefficients (N,)
        grad : ndarray, Euclidean gradient (N,)

        Returns:
        tuple, (ndarray, Riemannian gradient (N,), ndarray, original Euclidean gradient (N,))

        """
        gd_f= grad - np.real(np.multiply(e, grad)) * e
        return gd_f, grad # Return both projected and original gradients

    def retract_to_manifold(e, gd_f, ro=0.01, a_l=0.01, c_l=None):
        """
        Retract the updated coefficients to the complex circle manifold (|e_m| = 1).

        Parameters:
        e : ndarray, RIS coefficients (N,)
        gd_f : ndarray, Riemannian gradient (N,)
        ro : float, retraction parameter
        a_l : float, step size
        c_l : ndarray, previous step (for momentum, optional)

        Returns:
        tuple, (ndarray, retracted coefficients (N,), ndarray, updated c_l (N,))
        """
        if c_l is None:
            c_l = np.zeros_like(e)
        c_l = -gd_f + ro * (c_l - np.real(c_l * e.conj()) * e)
        return e + a_l * c_l * (1 / np.abs(e + a_l * c_l)), c_l

    def ro_calc(grad,grad_k):
        numerator = np.real(np.vdot(grad,grad - grad_k))  # Inner product: ⟨g_k, g_k - g_k-1⟩
        denominator = np.linalg.norm(grad_k)**2 + 1e-12   # Avoid division by zero
        ro=numerator/denominator
        return ro