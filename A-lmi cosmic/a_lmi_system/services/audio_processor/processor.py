"""
Auditory Cortex: STT and Environmental Sound Classification
Processes microphone input for real-time audio perception
"""

import logging
import sounddevice as sd
import numpy as np
import queue
from vosk import Model, KaldiRecognizer
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class AuditoryCortex:
    """
    Auditory processing service for A-LMI system.
    
    Components:
    - STT (Speech-to-Text): Offline transcription using Vosk
    - ESC (Environmental Sound Classification): Acoustic environment classification
    - PSD Computation: Power Spectral Density for VLCL physics modulation
    """
    
    def __init__(self, vosk_model_path: str, kafka_producer=None):
        """
        Initialize auditory processor.
        
        Args:
            vosk_model_path: Path to Vosk model
            kafka_producer: Kafka producer for publishing results
        """
        self.vosk_model_path = vosk_model_path
        self.kafka_producer = kafka_producer
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None
        
        self.sample_rate = 16000
        self.chunk_size = 4096
        self.audio_queue = queue.Queue()
        
        self.running = False
        
    def start(self):
        """Start audio processing."""
        try:
            # Load Vosk model
            self.model = Model(self.vosk_model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            logger.info(f"Loaded Vosk model from {self.vosk_model_path}")
            
            # Start audio stream
            self.running = True
            
            def audio_callback(indata, frames, time_info, status):
                """Callback for audio input."""
                if status:
                    logger.warning(f"Audio status: {status}")
                    
                if self.running:
                    # Process audio chunk
                    audio_data = indata[:, 0].astype(np.float32)
                    self.audio_queue.put(audio_data)
                    
                    # STT processing
                    audio_int = (audio_data * 32767).astype(np.int16)
                    if self.recognizer.AcceptWaveform(audio_int.tobytes()):
                        result = self.recognizer.Result()
                        if result:
                            self._publish_transcript(result)
                    
                    # ESC processing
                    esc_label = self._classify_environment(audio_data)
                    if esc_label:
                        self._publish_environment(esc_label)
                    
                    # PSD computation (for VLCL physics)
                    psd = self._compute_psd(audio_data)
                    self._publish_psd(psd)
            
            # Start audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=audio_callback
            ):
                logger.info("Audio stream started")
                
                # Keep running
                import time
                while self.running:
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Error starting audio processor: {e}")
            raise
    
    def stop(self):
        """Stop audio processing."""
        self.running = False
        logger.info("Audio processor stopped")
    
    def _classify_environment(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Classify environmental sound.
        
        Args:
            audio_data: Audio waveform
            
        Returns:
            Environment label (e.g., 'quiet_office', 'music_playing')
        """
        # Placeholder: Use CNN/Transformer model for ESC
        # For now, return simple classification based on energy
        
        energy = np.mean(audio_data ** 2)
        
        if energy < 0.001:
            return "quiet_office"
        elif energy < 0.01:
            return "low_ambient_noise"
        elif energy < 0.1:
            return "conversation"
        else:
            return "music_playing"
    
    def _compute_psd(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute Power Spectral Density.
        
        Args:
            audio_data: Audio waveform
            
        Returns:
            PSD magnitude (normalized)
        """
        # Apply FFT
        fft = np.fft.fft(audio_data)
        
        # Compute PSD
        psd = np.abs(fft) ** 2
        
        # Normalize
        psd_normalized = psd / np.max(psd) if np.max(psd) > 0 else psd
        
        return psd_normalized
    
    def _publish_transcript(self, transcript: str):
        """
        Publish transcript to Kafka.
        
        Args:
            transcript: Transcribed text
        """
        if self.kafka_producer:
            event = {
                "event_type": "raw.audio.transcript",
                "transcript": transcript,
                "timestamp": np.datetime64('now').item()
            }
            
            self.kafka_producer.publish("raw.audio.transcript", event)
            logger.info(f"Published transcript: {transcript[:50]}...")
    
    def _publish_environment(self, environment: str):
        """
        Publish environment classification to Kafka.
        
        Args:
            environment: Environment label
        """
        if self.kafka_producer:
            event = {
                "event_type": "raw.audio.environment",
                "environment": environment,
                "timestamp": np.datetime64('now').item()
            }
            
            self.kafka_producer.publish("raw.audio.environment", event)
            logger.debug(f"Published environment: {environment}")
    
    def _publish_psd(self, psd: np.ndarray):
        """
        Publish PSD to Kafka (for VLCL physics).
        
        Args:
            psd: PSD magnitude array
        """
        if self.kafka_producer:
            event = {
                "event_type": "raw.audio.psd",
                "psd": psd.tolist(),
                "psd_normalized": float(np.mean(psd)),
                "timestamp": np.datetime64('now').item()
            }
            
            self.kafka_producer.publish("raw.audio.psd", event)

