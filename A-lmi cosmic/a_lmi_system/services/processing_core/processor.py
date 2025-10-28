"""
Processing Core: Generates complete LightTokens from raw data
Applies CLIP encoding, perceptual hashing, and FFT spectral signatures
"""

import logging
import numpy as np
import hashlib
from typing import Dict, Any, Optional
import imagehash
from PIL import Image
try:
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    torch = None

try:
    import clip
    CLIP_MODULE_AVAILABLE = True
except ImportError:
    CLIP_MODULE_AVAILABLE = False
    clip = None

from ...core.data_structures import LightToken
from ...core.event_bus import EventBus
from ...memory.vector_db_client import VectorDBClient

logger = logging.getLogger(__name__)


class ProcessingCore:
    """
    Central processing service for generating LightTokens.
    
    This service:
    1. Consumes raw data from Kafka
    2. Generates LightToken with all 3 layers:
       - Layer 1: CLIP embedding (1536D)
       - Layer 2: Perceptual hash
       - Layer 3: Spectral signature (FFT)
    3. Publishes complete tokens
    4. Stores in memory tiers
    """
    
    def __init__(self, event_bus: EventBus, vector_db: VectorDBClient):
        """
        Initialize processing core.
        
        Args:
            event_bus: Event bus for communication
            vector_db: Vector database client
        """
        self.event_bus = event_bus
        self.vector_db = vector_db
        self.clip_model, self.clip_preprocess = None, None
        
        self._load_clip_model()
        
        # Subscribe to raw data events
        self.event_bus.subscribe("raw.web.content", self.process_web_content)
        self.event_bus.subscribe("raw.audio.transcript", self.process_audio_transcript)
    
    def _load_clip_model(self):
        """Load CLIP model for joint embeddings."""
        if not CLIP_AVAILABLE or not CLIP_MODULE_AVAILABLE:
            logger.warning("CLIP not available - will use placeholder embeddings")
            self.clip_model = None
            self.clip_preprocess = None
            return
            
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            logger.info(f"Loaded CLIP model on {device}")
        except Exception as e:
            logger.warning(f"Could not load CLIP model: {e}")
            self.clip_model = None
            self.clip_preprocess = None
    
    def process_web_content(self, event: Dict[str, Any]):
        """
        Process web content and generate LightToken.
        
        Args:
            event: Web content event
        """
        url = event.get("url")
        text_content = event.get("text_content", "")
        images = event.get("images", [])
        
        logger.info(f"Processing web content: {url}")
        
        # Generate LightToken for text content
        if text_content:
            token = self._create_token(
                source_uri=url,
                modality="text",
                content=text_content
            )
            
            if token:
                self._store_and_publish(token)
        
        # Process images
        for image_url in images[:5]:  # Limit to 5 images
            try:
                token = self._create_token_from_image(
                    source_uri=image_url,
                    image_url=image_url
                )
                
                if token:
                    self._store_and_publish(token)
            except Exception as e:
                logger.error(f"Error processing image {image_url}: {e}")
    
    def process_audio_transcript(self, event: Dict[str, Any]):
        """
        Process audio transcript and generate LightToken.
        
        Args:
            event: Audio transcript event
        """
        transcript = event.get("transcript", "")
        environment = event.get("environment", "")
        
        logger.info(f"Processing audio transcript: {transcript[:50]}...")
        
        token = self._create_token(
            source_uri="microphone",
            modality="audio",
            content=transcript
        )
        
        if token:
            token.metadata["environment"] = environment
            self._store_and_publish(token)
    
    def _create_token(self, source_uri: str, modality: str, content: str) -> Optional[LightToken]:
        """
        Create LightToken from text content.
        
        Args:
            source_uri: Source URI
            modality: Data modality
            content: Text content
            
        Returns:
            LightToken instance
        """
        try:
            token = LightToken(
                source_uri=source_uri,
                modality=modality,
                raw_data_ref=source_uri,
                content_text=content
            )
            
            # Layer 1: CLIP embedding
            if self.clip_model and content:
                embedding = self._encode_text(content)
                if embedding is not None:
                    token.set_semantic_embedding(embedding)
            
            # Layer 2: Perceptual hash
            content_bytes = content.encode('utf-8')
            token.set_perceptual_hash(content_bytes)
            
            # Layer 3: Spectral signature
            token.compute_spectral_signature()
            
            return token
            
        except Exception as e:
            logger.error(f"Error creating token: {e}")
            return None
    
    def _create_token_from_image(self, source_uri: str, image_url: str) -> Optional[LightToken]:
        """
        Create LightToken from image.
        
        Args:
            source_uri: Source URI
            image_url: Image URL
            
        Returns:
            LightToken instance
        """
        try:
            # Download and process image
            import requests
            from io import BytesIO
            
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            
            token = LightToken(
                source_uri=image_url,
                modality="image",
                raw_data_ref=image_url
            )
            
            # Layer 1: CLIP embedding
            if self.clip_model:
                embedding = self._encode_image(image)
                if embedding is not None:
                    token.set_semantic_embedding(embedding)
            
            # Layer 2: Perceptual hash (pHash for images)
            image_bytes = response.content
            phash = imagehash.phash(image)
            token.set_perceptual_hash(image_bytes)
            token.metadata["phash_str"] = str(phash)
            
            # Layer 3: Spectral signature
            token.compute_spectral_signature()
            
            return token
            
        except Exception as e:
            logger.error(f"Error creating token from image: {e}")
            return None
    
    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """
        Encode text with CLIP.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector (1536D)
        """
        if not self.clip_model or not CLIP_AVAILABLE:
            # Return placeholder embedding if CLIP not available
            embedding = np.random.rand(1536).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding
        
        try:
            with torch.no_grad():
                text_tokens = clip.tokenize([text]).to(next(self.clip_model.parameters()).device)
                text_features = self.clip_model.encode_text(text_tokens)
                embedding = text_features.cpu().numpy()[0]
                return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def _encode_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Encode image with CLIP.
        
        Args:
            image: PIL Image
            
        Returns:
            Embedding vector (1536D)
        """
        if not self.clip_model or not CLIP_AVAILABLE:
            # Return placeholder embedding if CLIP not available
            embedding = np.random.rand(1536).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding
        
        try:
            with torch.no_grad():
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(next(self.clip_model.parameters()).device)
                image_features = self.clip_model.encode_image(image_tensor)
                embedding = image_features.cpu().numpy()[0]
                return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def _store_and_publish(self, token: LightToken):
        """
        Store token in memory tiers and publish to event bus.
        
        Args:
            token: LightToken instance
        """
        try:
            # Store in vector database
            if token.joint_embedding is not None:
                self.vector_db.insert_semantic_embedding(
                    token_id=str(token.token_id),
                    source_uri=token.source_uri,
                    modality=token.modality,
                    embedding=token.joint_embedding
                )
            
            if token.spectral_magnitude is not None:
                self.vector_db.insert_spectral_signature(
                    token_id=str(token.token_id),
                    source_uri=token.source_uri,
                    modality=token.modality,
                    spectral_magnitude=token.spectral_magnitude
                )
            
            # Publish to event bus
            event = {
                "event_type": "processed.lighttoken",
                "token": token.to_dict()
            }
            
            self.event_bus.publish("processed.lighttokens", event)
            logger.info(f"Processed and published token: {token.token_id}")
            
        except Exception as e:
            logger.error(f"Error storing/publishing token: {e}")

