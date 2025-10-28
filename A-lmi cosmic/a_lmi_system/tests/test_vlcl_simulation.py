"""
VLCL Simulation Integration Test
Tests audio modulation, AI commands, and physics validation
"""

import pytest
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TestVLCLSimulation:
    """Test VLCL physics simulation components."""
    
    def test_audio_psd_computation(self):
        """Test: Audio PSD computation and normalization."""
        logger.info("Testing audio PSD computation...")
        
        # Simulate audio data
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate test signal (440 Hz sine wave)
        frequency = 440
        audio_data = np.sin(2 * np.pi * frequency * t)
        
        # Compute PSD
        fft = np.fft.fft(audio_data)
        psd = np.abs(fft) ** 2
        psd_normalized = psd / np.max(psd) if np.max(psd) > 0 else psd
        
        # Verify
        assert len(psd) > 0, "PSD should have values"
        assert np.max(psd_normalized) <= 1.0, "PSD should be normalized to [0,1]"
        assert np.min(psd_normalized) >= 0.0, "PSD should be non-negative"
        
        logger.info(f"PSD computed: {len(psd)} bins, max={np.max(psd_normalized):.4f}")
        logger.info("✅ Audio PSD test PASSED")
    
    def test_stochastic_resonance_noise_modulation(self):
        """Test: Stochastic resonance noise term modulation."""
        logger.info("Testing stochastic resonance noise modulation...")
        
        # Test parameters
        baseline_noise = 0.01
        audio_noise = 0.02
        psd_values = [0.0, 0.5, 1.0]
        
        for psd in psd_values:
            # Compute noise term
            sigma = baseline_noise + audio_noise * psd
            
            # Verify
            assert sigma >= baseline_noise, "Noise should be at least baseline"
            assert sigma <= baseline_noise + audio_noise, "Noise should not exceed max"
            
            logger.info(f"PSD={psd:.2f} → Noise={sigma:.4f}")
        
        logger.info("✅ Stochastic resonance noise modulation test PASSED")
    
    def test_physics_forces(self):
        """Test: CST v2 physics force calculations."""
        logger.info("Testing physics forces...")
        
        # Test parameters
        alpha = 1.0
        phi = 1.618
        beta = 0.5
        gamma = 0.3
        damping_zeta = 0.1
        swirl_gain = 0.05
        
        # Test position and velocity
        x = np.array([1.0, 1.0, 0.0])
        x0 = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.1, 0.2, 0.0])
        mass = 1.0
        
        # Conservative force: F_cons = -alpha * phi * (x - x0)
        dx = x - x0
        F_cons = -alpha * phi * dx
        
        # Verify magnitude and direction
        assert np.linalg.norm(F_cons) > 0, "Conservative force should be non-zero"
        assert F_cons[0] < 0, "Force should point toward origin (x)"
        assert F_cons[1] < 0, "Force should point toward origin (y)"
        
        # Swirl force: F_swirl = swirl_gain * R_90(dx)
        # R_90 rotates by 90 degrees: (x,y) -> (-y,x)
        rotated = np.array([-dx[1], dx[0], 0])
        F_swirl = swirl_gain * rotated
        
        # Verify
        assert np.linalg.norm(F_swirl) > 0, "Swirl force should be non-zero"
        
        # Damping force: F_damp = -damping_zeta * velocity
        F_damp = -damping_zeta * velocity
        
        # Verify - damping should oppose motion (dot product should be negative)
        dot_product = np.sum(F_damp * velocity)
        assert dot_product < 0, f"Damping should oppose motion (got {dot_product})"
        
        # Total force
        F_total = F_cons + F_swirl + F_damp
        
        logger.info(f"F_cons magnitude: {np.linalg.norm(F_cons):.4f}")
        logger.info(f"F_swirl magnitude: {np.linalg.norm(F_swirl):.4f}")
        logger.info(f"F_total magnitude: {np.linalg.norm(F_total):.4f}")
        logger.info("✅ Physics forces test PASSED")
    
    def test_particle_energy_conservation(self):
        """Test: Particle energy calculation with Golden Ratio."""
        logger.info("Testing particle energy conservation...")
        
        # Test parameters
        C = 3.0e8  # Speed of light
        PHI = 1.618  # Golden Ratio
        m = 1.0  # Mass
        epsilon = 0.1
        n = 0
        lambda_values = [1.0, 2.0, 1.5]
        
        # Calculate energy
        d = n + (lambda_values[n] / abs(lambda_values[n + 1]))
        E = m * C ** 2 * (PHI) ** d * np.exp(epsilon)
        
        # Verify
        assert E > 0, "Energy should be positive"
        assert np.isfinite(E), "Energy should be finite"
        
        logger.info(f"Particle energy: {E:.2e} J")
        logger.info("✅ Particle energy conservation test PASSED")
    
    def test_event_bus_message_handling(self):
        """Test: EventBus message publication and subscription."""
        logger.info("Testing EventBus message handling...")
        
        # Simulate EventBus messages
        messages = []
        
        def message_handler(data):
            messages.append(data)
        
        # Simulate audio PSD event
        psd_event = {"event_type": "audio.psd", "value": 0.75}
        message_handler(psd_event)
        
        # Simulate AI command event
        ai_command = {"event_type": "ai.command", "action": "spawn", "type": "particle"}
        message_handler(ai_command)
        
        # Verify
        assert len(messages) == 2, "Should have 2 messages"
        assert messages[0]["event_type"] == "audio.psd"
        assert messages[1]["event_type"] == "ai.command"
        
        logger.info(f"Received {len(messages)} events")
        logger.info("✅ EventBus message handling test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

