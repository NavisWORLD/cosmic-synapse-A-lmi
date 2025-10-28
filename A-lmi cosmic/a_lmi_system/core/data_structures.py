"""
LightToken: Universal information representation with tripartite structure
Based on The Unified Theory of Vibrational Information Architecture
"""

import numpy as np
import hashlib
from uuid import uuid4, UUID
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LightToken:
    """
    Universal information representation with tripartite structure.
    
    Layer 1: Semantic Core (Joint Embedding) - 1536D semantic vector
    Layer 2: Perceptual Fingerprint - 16-byte perceptual hash
    Layer 3: Spectral Signature - FFT of semantic embedding (the innovation)
    
    This structure enables:
    - Semantic similarity (Layer 1)
    - Perceptual deduplication (Layer 2)
    - Spectral/frequency-based discovery (Layer 3)
    """
    
    def __init__(self, source_uri: str, modality: str, raw_data_ref: str, content_text: Optional[str] = None):
        """
        Initialize a LightToken.
        
        Args:
            source_uri: Original source of the data (URL, file path, etc.)
            modality: Type of data (text, image, audio, video)
            raw_data_ref: Reference to raw data in Tier 1 storage
            content_text: Optional text content for text data
        """
        self.token_id: UUID = uuid4()
        self.timestamp: datetime = datetime.now(timezone.utc)
        self.source_uri: str = source_uri
        self.modality: str = modality
        self.raw_data_ref: str = raw_data_ref
        self.content_text: Optional[str] = content_text
        
        # Layer 1: Semantic Core (1536D joint embedding from CLIP)
        self.joint_embedding: Optional[np.ndarray] = None
        
        # Layer 2: Perceptual Fingerprint
        self.perceptual_hash: Optional[str] = None
        
        # Layer 3: Spectral Signature (INNOVATION: FFT of embedding)
        self.spectral_signature: Optional[np.ndarray] = None  # Complex-valued
        self.spectral_magnitude: Optional[np.ndarray] = None  # For vector search
        
        self.metadata: Dict[str, Any] = {}
        
    def set_semantic_embedding(self, embedding: np.ndarray):
        """
        Layer 1: Set semantic vector from encoder (e.g., CLIP).
        
        Args:
            embedding: 1536-dimensional vector from joint embedding model
        """
        assert embedding.shape == (1536,), f"Expected 1536D embedding, got {embedding.shape}"
        self.joint_embedding = embedding.astype(np.float32)
        logger.debug(f"Set semantic embedding for token {self.token_id}")
        
    def set_perceptual_hash(self, data: bytes):
        """
        Layer 2: Generate perceptual hash for deduplication.
        
        Args:
            data: Raw data bytes to hash
        """
        # Using SHA256 as placeholder - actual implementation should use pHash/SimHash
        # For images: use imagehash (pHash)
        # For text: use SimHash
        hash_obj = hashlib.sha256(data)
        self.perceptual_hash = hash_obj.hexdigest()[:16]  # 16 bytes = 32 hex chars
        logger.debug(f"Set perceptual hash for token {self.token_id}")
        
    def compute_spectral_signature(self):
        """
        Layer 3: Apply FFT to embedding (THE INNOVATION).
        
        This reveals the "spectral signature" - the frequency-domain representation
        of the semantic content. This enables frequency-based clustering and discovery
        of abstract, cross-modal patterns that are invisible to standard cosine similarity.
        """
        assert self.joint_embedding is not None, "Must set semantic embedding first"
        
        # Apply Discrete Fourier Transform to semantic vector
        # This transforms from "temporal" (index) domain to frequency domain
        spectral = np.fft.fft(self.joint_embedding)
        self.spectral_signature = spectral.astype(np.complex128)
        
        # Store magnitude for vector search (vector databases typically don't support complex)
        self.spectral_magnitude = np.abs(spectral).astype(np.float32)
        
        logger.debug(f"Computed spectral signature for token {self.token_id}")
        
    def spectral_similarity(self, other: 'LightToken') -> float:
        """
        Compute similarity in frequency domain (CORE INNOVATION).
        
        This measures how similar two tokens are in their "spectral texture" -
        the abstract pattern of their meaning, independent of exact semantic content.
        
        Args:
            other: Another LightToken to compare against
            
        Returns:
            Spectral similarity score [0, 1]
        """
        assert self.spectral_signature is not None, "Must compute spectral signature first"
        assert other.spectral_signature is not None, "Other token must have spectral signature"
        
        # Use magnitude for cross-correlation
        mag1 = np.abs(self.spectral_signature)
        mag2 = np.abs(other.spectral_signature)
        
        # Normalized cross-correlation
        numerator = np.sum(mag1 * mag2)
        denominator = np.sqrt(np.sum(mag1**2) * np.sum(mag2**2))
        
        similarity = numerator / denominator if denominator > 0 else 0.0
        return float(similarity)
    
    def semantic_similarity(self, other: 'LightToken') -> float:
        """
        Compute standard cosine similarity on Layer 1.
        
        Args:
            other: Another LightToken to compare against
            
        Returns:
            Cosine similarity score [0, 1]
        """
        assert self.joint_embedding is not None, "Must set semantic embedding first"
        assert other.joint_embedding is not None, "Other token must have embedding"
        
        # Cosine similarity
        dot_product = np.dot(self.joint_embedding, other.joint_embedding)
        norm1 = np.linalg.norm(self.joint_embedding)
        norm2 = np.linalg.norm(other.joint_embedding)
        
        similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
        return float(similarity)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize LightToken to dictionary for storage.
        
        Returns:
            Dictionary representation of the token
        """
        token_dict = {
            "token_id": str(self.token_id),
            "timestamp": self.timestamp.isoformat(),
            "source_uri": self.source_uri,
            "modality": self.modality,
            "raw_data_ref": self.raw_data_ref,
            "content_text": self.content_text,
            "metadata": self.metadata,
            "perceptual_hash": self.perceptual_hash,
        }
        
        # Store embeddings as lists
        if self.joint_embedding is not None:
            token_dict["joint_embedding"] = self.joint_embedding.tolist()
            
        if self.spectral_magnitude is not None:
            token_dict["spectral_magnitude"] = self.spectral_magnitude.tolist()
        
        return token_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LightToken':
        """
        Deserialize LightToken from dictionary.
        
        Args:
            data: Dictionary representation of the token
            
        Returns:
            LightToken instance
        """
        token = cls(
            source_uri=data["source_uri"],
            modality=data["modality"],
            raw_data_ref=data["raw_data_ref"],
            content_text=data.get("content_text")
        )
        
        token.token_id = UUID(data["token_id"])
        token.timestamp = datetime.fromisoformat(data["timestamp"])
        token.metadata = data.get("metadata", {})
        token.perceptual_hash = data.get("perceptual_hash")
        
        # Restore embeddings
        if "joint_embedding" in data:
            token.joint_embedding = np.array(data["joint_embedding"], dtype=np.float32)
            
        if "spectral_magnitude" in data:
            token.spectral_magnitude = np.array(data["spectral_magnitude"], dtype=np.float32)
        
        return token
    
    def __repr__(self) -> str:
        return f"LightToken(id={self.token_id}, modality={self.modality}, source={self.source_uri})"

