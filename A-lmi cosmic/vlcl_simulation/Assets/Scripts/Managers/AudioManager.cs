using UnityEngine;
using VLCL.Core;
using System;

namespace VLCL.Managers
{
    /// <summary>
    /// Audio Manager: Captures microphone input and computes PSD
    /// for audio-driven stochastic resonance.
    /// </summary>
    public class AudioManager : MonoBehaviour
    {
        [Header("Audio Capture")]
        public int sampleRate = 44100;
        public int bufferSize = 4096;
        public int numChannels = 1;

        private AudioClip audioClip;
        private float[] audioData;
        private float[] psdBuffer;

        private void Start()
        {
            // Start capturing audio
            try
            {
                audioClip = Microphone.Start(Microphone.devices[0], true, 1, sampleRate);
                audioData = new float[bufferSize];
                psdBuffer = new float[bufferSize / 2];
                InvokeRepeating(nameof(ProcessAudio), 0, 0.1f);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"Microphone not available: {e.Message}");
            }
        }

        private void ProcessAudio()
        {
            if (audioClip == null) return;

            // Get audio data
            int offset = Microphone.GetPosition(Microphone.devices[0]) - bufferSize;
            if (offset < 0) return;

            audioClip.GetData(audioData, offset);

            // Compute Power Spectral Density (PSD)
            float[] spectrum = ComputeFFT(audioData);

            // Compute PSD magnitude
            float psd = ComputePSD(spectrum);

            // Normalize and publish to EventBus
            float psd_normalized = NormalizePSD(psd);
            EventBus.Instance.Publish("audio.psd", psd_normalized);

            // Also publish raw audio data
            EventBus.Instance.Publish("audio.raw", audioData);
        }

        private float[] ComputeFFT(float[] input)
        {
            // Placeholder: Implement FFT or use Unity's built-in FFT
            // For now, return spectrum placeholder
            float[] spectrum = new float[bufferSize / 2];
            for (int i = 0; i < spectrum.Length; i++)
            {
                spectrum[i] = UnityEngine.Random.Range(0f, 1f);
            }
            return spectrum;
        }

        private float ComputePSD(float[] spectrum)
        {
            float sum = 0f;
            foreach (float s in spectrum)
            {
                sum += s * s;
            }
            return sum / spectrum.Length;
        }

        private float NormalizePSD(float psd)
        {
            // Normalize to [0, 1] range
            return Mathf.Clamp01(psd / 100f);
        }

        private void OnDestroy()
        {
            if (Microphone.IsRecording(Microphone.devices[0]))
            {
                Microphone.End(Microphone.devices[0]);
            }
        }
    }
}

