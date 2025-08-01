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

        for iteration in range(max_iterations):
            t += 1
            grad = self.compute_gd_gradient(h_d, h_r, e, transmit_power)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            if discrete_phases is not None:
                theta = np.array([min(discrete_phases, key=lambda x: abs(x - t)) for t in theta])
            theta = np.mod(theta, 2 * np.pi)
            e = np.exp(1j * theta)
            power = max(self.compute_received_power(h_d, h_r, e, transmit_power), 1e-20)
            powers.append(power)
            snrs.append(self.compute_snr(power, noise_power=1e-12))
            iterations.append(iteration + 1)

        return theta, powers, snrs, iterations
    
    def manifold_optimization_adam(self, h_d, H, h_1, N, transmit_power, 
                                  learning_rate=0.01, max_iterations=1000, 
                                  beta1=0.9, beta2=0.999, epsilon=1e-8, discrete_phases=None):
        """Manifold optimization with Adam optimizer."""
        h_r = self.compute_cascaded_channel(H, h_1)
        theta = np.random.uniform(0, 2 * np.pi, N)
        e = np.exp(1j * theta)
        powers = [self.compute_received_power(h_d, h_r, e, transmit_power)]
        snrs = [self.compute_snr(powers[0])]
        iterations = [0]

        # Adam variables
        m = np.zeros(N, dtype=complex)  # First moment
        v = np.zeros(N)  # Second moment
        t = 0  # Time step

        # Manifold variables
        c_l = np.zeros(N, dtype=complex)
        grad_riemannian_prev = None

        for iteration in range(max_iterations):
            t += 1
            grad_euclidean = self.compute_manifold_gradient(h_d, h_r, e, transmit_power)
            grad_riemannian, grad_k = self.project_to_tangent_space(e, grad_euclidean)

            # Adam update
            m = beta1 * m + (1 - beta1) * grad_riemannian
            v = beta2 * v + (1 - beta2) * (np.abs(grad_riemannian) ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # Calculate ro using ro_calc if previous gradient exists
            current_ro = 0.0
            if grad_riemannian_prev is not None:
                current_ro = self.ro_calc(grad_riemannian, grad_riemannian_prev)

            # Manifold retraction
            step = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            e, c_l = self.retract_to_manifold(e, step, ro=current_ro, c_l=c_l, a_l=1.0)

            power = self.compute_received_power(h_d, h_r, e, transmit_power)
            powers.append(power)
            snrs.append(self.compute_snr(power))
            iterations.append(iteration + 1)

            grad_riemannian_prev = grad_riemannian

        theta = np.angle(e)
        return theta, powers, snrs, iterations
    
    def alternating_optimization(self, h_d, H, h_1, N, transmit_power, 
                               max_iterations=1000, tolerance=1e-6, discrete_phases=None):
        """Alternating optimization algorithm."""
        h_r = self.compute_cascaded_channel(H, h_1)
        theta = np.random.choice([np.pi, -np.pi], N)
        e = np.exp(1j * theta)
        powers = [max(self.compute_received_power(h_d, h_r, e, transmit_power), 1e-20)]
        snrs = [self.compute_snr(powers[0], noise_power=1e-12)]
        iterations = [0]

        for iteration in range(max_iterations):
            for n in range(N):
                combined_without_n = h_d + np.dot(h_r, e) - h_r[n] * e[n]
                optimal_phase = -np.angle(h_r[n] * combined_without_n.conj())
                if discrete_phases is not None:
                    optimal_phase = min(discrete_phases, key=lambda x: abs(x - optimal_phase))
                e[n] = np.exp(1j * optimal_phase)

            power = max(self.compute_received_power(h_d, h_r, e, transmit_power), 1e-20)
            powers.append(power)
            snrs.append(self.compute_snr(power, noise_power=1e-12))
            iterations.append(iteration + 1)

            if iteration > 0 and abs(powers[-1] - powers[-2]) < tolerance * powers[-2]:
                break

        theta = np.angle(e)
        return theta, powers, snrs, iterations
    
    # Helper methods
    def compute_cascaded_channel(self, H, h_1):
        """Compute cascaded channel."""
        return H.flatten() * h_1
    
    def compute_received_power(self, h_d, h_r, e, transmit_power):
        """Compute received power."""
        combined = h_d + np.dot(h_r, e)
        return transmit_power * np.abs(combined)**2
    
    def compute_snr(self, power, noise_power=1e-12):
        """Compute SNR in dB."""
        return 10 * np.log10(power / noise_power)
    
    def compute_gd_gradient(self, h_d, h_r, e, transmit_power):
        """Compute gradient for gradient descent."""
        N = len(h_r)
        combined = h_d + np.dot(h_r, e)
        grad = np.zeros(N)
        for m in range(N):
            term = h_r[m] * combined.conj()
            grad[m] = -2 * transmit_power * np.real(1j * e[m] * term)
        return grad
    
    def compute_manifold_gradient(self, h_d, h_r, e, transmit_power):
        """Compute Euclidean gradient for manifold optimization."""
        combined = h_d + np.dot(h_r, e)
        grad = -transmit_power * h_r.conj() * combined
        return grad
    
    def project_to_tangent_space(self, e, grad):
        """Project gradient to tangent space."""
        gd_f = grad - np.real(np.multiply(e, grad)) * e
        return gd_f, grad
    
    def retract_to_manifold(self, e, gd_f, ro=0.01, a_l=0.01, c_l=None):
        """Retract to manifold."""
        if c_l is None:
            c_l = np.zeros_like(e)
        c_l = -gd_f + ro * (c_l - np.real(c_l * e.conj()) * e)
        return e + a_l * c_l * (1 / np.abs(e + a_l * c_l)), c_l
    
    def ro_calc(self, grad, grad_k):
        """Calculate ro parameter."""
        numerator = np.real(np.vdot(grad, grad - grad_k))
        denominator = np.linalg.norm(grad_k)**2 + 1e-12
        return numerator / denominator
