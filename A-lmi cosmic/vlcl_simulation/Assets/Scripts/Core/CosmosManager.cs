using UnityEngine;
using VLCL.Core;

namespace VLCL.Core
{
    /// <summary>
    /// Cosmic Synapse Theory (CST) v2 Physics Engine
    /// Implements the refined potential and force calculations
    /// for the VLCL particle simulation.
    /// </summary>
    public class CosmosManager : MonoBehaviour
    {
        [Header("CST Physics Parameters")]
        public float alpha = 1.0f;        // Harmonic bowl scale
        public float beta = 0.5f;         // Gravity scale
        public float gamma = 0.3f;       // Connectivity scale
        public float phi = 1.618f;       // Golden Ratio
        public float dampingZeta = 0.1f;
        public float swirlGain = 0.05f;
        public float noiseAmplitude = 0.01f;

        [Header("Noise (Stochastic Resonance)")]
        public float baselineNoise = 0.01f;
        public float audioNoise = 0.0f;  // Updated by AudioManager

        private float psdNormalized = 0.0f;
        private Particle[] particles;

        private void Start()
        {
            // Subscribe to audio PSD updates
            EventBus.Instance.Subscribe("audio.psd", OnAudioPSD);
        }

        private void OnAudioPSD(object data)
        {
            if (data is float psd)
            {
                psdNormalized = psd;
            }
        }

        private void FixedUpdate()
        {
            // Update noise term with audio modulation (Stochastic Resonance)
            float sigma = baselineNoise + audioNoise * psdNormalized;

            // Apply forces to all particles
            if (particles != null)
            {
                foreach (var particle in particles)
                {
                    ApplyForces(particle, sigma);
                }
            }
        }

        private void ApplyForces(Particle particle, float sigma)
        {
            Vector3 pos = particle.transform.position;
            Vector3 center = Vector3.zero;

            // F_cons: Conservative force (gradient of potential)
            Vector3 F_cons = ComputeConservativeForce(pos, center, particle.mass);

            // F_swirl: Non-conservative orbital swirl
            Vector3 F_swirl = ComputeSwirlForce(pos, center);

            // F_damp: Damping force
            Vector3 F_damp = -dampingZeta * particle.velocity;

            // F_noise: Stochastic force (audio-driven)
            Vector3 F_noise = Random.insideUnitSphere * sigma * noiseAmplitude;

            // Total force
            Vector3 F_total = F_cons + F_swirl + F_damp + F_noise;

            // Update particle
            particle.ApplyForce(F_total);
        }

        private Vector3 ComputeConservativeForce(Vector3 x, Vector3 x0, float mass)
        {
            // Gradient of: alpha * phi * Ec * 0.5 * ||x - x0||^2
            Vector3 dx = x - x0;
            return -alpha * phi * dx;
        }

        private Vector3 ComputeSwirlForce(Vector3 x, Vector3 x0)
        {
            // F_swirl = k_Omega * R_90 (x - x0)
            // R_90 rotates by 90 degrees in XY plane
            Vector3 dx = x - x0;
            Vector3 rotated = new Vector3(-dx.y, dx.x, 0);
            return swirlGain * rotated;
        }

        public void SetParticles(Particle[] particles)
        {
            this.particles = particles;
        }
    }

    [System.Serializable]
    public class Particle
    {
        public Transform transform;
        public Vector3 velocity;
        public float mass;

        public void ApplyForce(Vector3 force)
        {
            velocity += force * Time.fixedDeltaTime;
            transform.position += velocity * Time.fixedDeltaTime;
        }
    }
}

